"""
Forecasting pipeline used by the Streamlit app.

data (yfinance) -> features -> ARIMA (SARIMAX + volume) + TCN (log-returns) -> ensemble

Ported from the tcn_arima.ipynb notebook, with three changes needed for a live app:
  1. ARIMA uses *yesterday's* volume as exogenous input. The notebook used the
     same-day volume, which is not known until the day being predicted is over.
  2. ARIMA walk-forward uses fixed parameters (res.apply) instead of refitting
     578 times, so the backtest runs in seconds instead of many minutes.
  3. The TCN is fine-tuned on the full history before making the live forecast.
"""
from __future__ import annotations

import itertools
import os
import random
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

N_WINDOW = 60
TCN_EXOGS = ["Volume", "MA7", "MA21", "RSI", "Vol_Change"]
FEATURE_COLS = ["LogReturn"] + TCN_EXOGS
DEFAULT_ORDER = (2, 1, 2)   # best order found by the notebook's grid search on ETH
MIN_ROWS = 400


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_prices(symbol: str, start: str, drop_incomplete: bool = True) -> pd.DataFrame:
    """Download daily OHLCV from Yahoo Finance and return a clean, flat dataframe."""
    raw = yf.download(symbol, start=start, interval="1d", auto_adjust=True,
                      progress=False, threads=False)
    if raw is None or raw.empty:
        raise ValueError(f"Yahoo Finance returned no data for '{symbol}'. Check the ticker.")

    if isinstance(raw.columns, pd.MultiIndex):           # yfinance >= 0.2.48 style
        raw.columns = raw.columns.get_level_values(0)

    df = raw.reset_index()
    date_col = next((c for c in df.columns if "date" in str(c).lower()), None)
    if date_col is None:
        raise KeyError(f"No date column found. Columns: {list(df.columns)}")
    df = df.rename(columns={date_col: "Date"})

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if getattr(df["Date"].dt, "tz", None) is not None:
        df["Date"] = df["Date"].dt.tz_localize(None)
    df = (df.dropna(subset=["Date"])
            .drop_duplicates(subset="Date")
            .sort_values("Date")
            .reset_index(drop=True))

    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    # Crypto trades 24/7 and Yahoo's daily candle is UTC-based: today's row is
    # still moving, so leave it out and forecast today's close instead.
    if drop_incomplete:
        today_utc = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        df = df[df["Date"] < today_utc]

    return df.dropna(subset=["Close"]).reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same indicators as clean_and_fe() in the notebook (applied once)."""
    df = df.copy()
    df = df[df["Close"] > 0]

    df["Return"] = df["Close"].pct_change()
    df["LogReturn"] = np.log(df["Close"]).diff()
    df["MA7"] = df["Close"].rolling(7).mean()
    df["MA21"] = df["Close"].rolling(21).mean()

    delta = df["Close"].diff()
    roll_up = delta.clip(lower=0).ewm(span=14).mean()
    roll_down = (-delta.clip(upper=0)).ewm(span=14).mean()
    df["RSI"] = 100 - (100 / (1 + roll_up / (roll_down + 1e-9)))

    df["Vol_Change"] = df["Volume"].pct_change()
    df["Volume_lag1"] = df["Volume"].shift(1)       # ARIMA exog (known in advance)

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def score(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE %": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "Direction acc. %": float(np.nan),   # filled in by caller (needs previous close)
    }


def direction_accuracy(prev_close, y_true, y_pred) -> float:
    actual = np.sign(np.asarray(y_true) - np.asarray(prev_close))
    predicted = np.sign(np.asarray(y_pred) - np.asarray(prev_close))
    return float(np.mean(actual == predicted) * 100)


# --------------------------------------------------------------------------- #
# ARIMA
# --------------------------------------------------------------------------- #
def _sarimax(y, exog, order):
    return SARIMAX(y, exog=exog, order=order,
                   enforce_stationarity=False, enforce_invertibility=False)


def select_arima_order(y, exog) -> tuple:
    """AIC grid search over p,q in 0..2 and d in 0..1 (as in the notebook)."""
    best_aic, best_order = np.inf, DEFAULT_ORDER
    for order in itertools.product(range(3), [0, 1], range(3)):
        try:
            res = _sarimax(y, exog, order).fit(disp=False, maxiter=50)
            if res.aic < best_aic:
                best_aic, best_order = res.aic, order
        except Exception:
            continue
    return best_order


def run_arima(df: pd.DataFrame, split: int, auto_order: bool) -> dict:
    y = df["Close"].to_numpy()
    vol_lag = df["Volume_lag1"].to_numpy().reshape(-1, 1)

    scaler = StandardScaler().fit(vol_lag[:split])
    exog = scaler.transform(vol_lag)

    order = select_arima_order(y[:split], exog[:split]) if auto_order else DEFAULT_ORDER
    res_train = _sarimax(y[:split], exog[:split], order).fit(disp=False, maxiter=200)

    # One-step-ahead predictions over the test period with the trained parameters:
    # every prediction for day t only uses data up to t-1 (walk-forward, no refit).
    res_full = res_train.apply(y, exog=exog)
    test_pred = np.asarray(res_full.get_prediction(start=split, end=len(y) - 1).predicted_mean)

    # Live forecast: refit on all data, exog for tomorrow = today's volume.
    res_live = _sarimax(y, exog, order).fit(disp=False, maxiter=200,
                                             start_params=res_train.params)
    next_exog = scaler.transform(df["Volume"].to_numpy()[-1:].reshape(-1, 1))
    fc = res_live.get_forecast(steps=1, exog=next_exog)
    ci = np.asarray(fc.conf_int(alpha=0.05))[0]

    return {
        "order": tuple(int(v) for v in order),
        "aic": float(res_train.aic),
        "test_pred": test_pred,
        "next": float(np.asarray(fc.predicted_mean)[0]),
        "next_lo": float(ci[0]),
        "next_hi": float(ci[1]),
    }


# --------------------------------------------------------------------------- #
# TCN
# --------------------------------------------------------------------------- #
def _set_seeds(seed: int = 42):
    import tensorflow as tf
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_tcn(input_shape, filters: int = 64, kernel: int = 3,
              dilations=(1, 2, 4, 8), dropout: float = 0.1):
    """Residual dilated causal-conv network (same blocks as the notebook's model)."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inp = layers.Input(shape=input_shape)
    x = inp
    for d in dilations:
        prev = x
        x = layers.Conv1D(filters, kernel, padding="causal", dilation_rate=d)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv1D(filters, 1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        if prev.shape[-1] != filters:
            prev = layers.Conv1D(filters, 1)(prev)   # match channels for the skip
        x = layers.Add()([prev, x])

    x = layers.Cropping1D(cropping=(input_shape[0] - 1, 0))(x)   # keep last time step
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)

    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="huber")
    return model


def _windows(feats: np.ndarray, idx: np.ndarray, window: int) -> np.ndarray:
    return np.stack([feats[i - window:i] for i in idx]).astype("float32")


def run_tcn(df: pd.DataFrame, split: int, epochs: int, fine_tune_epochs: int,
            window: int = N_WINDOW) -> dict:
    import tensorflow as tf
    from tensorflow.keras import callbacks

    _set_seeds()
    n = len(df)

    # Scalers fit on the training period only
    scaler_x = StandardScaler().fit(df[FEATURE_COLS].to_numpy()[:split])
    scaler_y = StandardScaler().fit(df[["LogReturn"]].to_numpy()[:split])
    feats = scaler_x.transform(df[FEATURE_COLS].to_numpy())
    target = scaler_y.transform(df[["LogReturn"]].to_numpy()).ravel()

    # Sample i: features from rows i-window..i-1  ->  log-return of row i
    train_idx = np.arange(window, split)
    test_idx = np.arange(split, n)
    X_tr, y_tr = _windows(feats, train_idx, window), target[train_idx]
    X_te = _windows(feats, test_idx, window)

    val_start = int(len(X_tr) * 0.9)                 # last 10 % of train (chronological)
    model = build_tcn((window, len(FEATURE_COLS)))
    hist = model.fit(
        X_tr[:val_start], y_tr[:val_start],
        validation_data=(X_tr[val_start:], y_tr[val_start:]),
        epochs=epochs, batch_size=32, verbose=0,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
        ],
    )

    close = df["Close"].to_numpy()
    pred_ret = scaler_y.inverse_transform(model.predict(X_te, verbose=0)).ravel()
    test_pred = close[test_idx - 1] * np.exp(pred_ret)          # price_t = price_{t-1} * e^r

    # Live forecast: briefly fine-tune on everything we have, then predict tomorrow.
    if fine_tune_epochs > 0:
        all_idx = np.arange(window, n)
        model.fit(_windows(feats, all_idx, window), target[all_idx],
                  epochs=fine_tune_epochs, batch_size=32, verbose=0)
    x_next = feats[-window:][None].astype("float32")
    next_ret = float(scaler_y.inverse_transform(model.predict(x_next, verbose=0))[0, 0])

    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()
               if k in ("loss", "val_loss")}
    tf.keras.backend.clear_session()

    return {
        "test_pred": test_pred,
        "next": float(close[-1] * np.exp(next_ret)),
        "next_logret": next_ret,
        "history": history,
    }


# --------------------------------------------------------------------------- #
# Ensemble + full pipeline
# --------------------------------------------------------------------------- #
def pick_weight(truth, arima, tcn) -> float:
    grid = np.round(np.linspace(0, 1, 21), 2)
    errs = [np.sqrt(mean_squared_error(truth, w * arima + (1 - w) * tcn)) for w in grid]
    return float(grid[int(np.argmin(errs))])


def run_pipeline(df: pd.DataFrame, auto_order: bool = False, epochs: int = 60,
                 fine_tune_epochs: int = 5, arima_weight: float | None = None) -> dict:
    """
    df: output of add_features(). Returns plain arrays/numbers (cache-friendly).
    Test = last 20 %. First half of test picks the ensemble weight (validation),
    second half is the untouched evaluation period - same protocol as the notebook.
    """
    n = len(df)
    if n < MIN_ROWS:
        raise ValueError(f"Need at least {MIN_ROWS} days of data, got {n}. Use an earlier start date.")
    split = int(n * 0.8)

    arima = run_arima(df, split, auto_order)
    tcn = run_tcn(df, split, epochs, fine_tune_epochs)

    close = df["Close"].to_numpy()
    truth = close[split:]
    prev = close[split - 1:-1]                   # naive "tomorrow = today" baseline
    a, t = arima["test_pred"], tcn["test_pred"]

    half = len(truth) // 2
    weight = pick_weight(truth[:half], a[:half], t[:half]) if arima_weight is None else float(arima_weight)
    ens = weight * a + (1 - weight) * t

    rows = {}
    for name, pred in [("Ensemble", ens), ("ARIMA", a), ("TCN", t), ("Naive (yesterday's close)", prev)]:
        m = score(truth[half:], pred[half:])
        m["Direction acc. %"] = (direction_accuracy(prev[half:], truth[half:], pred[half:])
                                 if not name.startswith("Naive") else np.nan)
        rows[name] = m
    metrics = pd.DataFrame(rows).T

    last_date = pd.Timestamp(df["Date"].iloc[-1])
    return {
        "dates_test": df["Date"].iloc[split:].to_numpy(),
        "truth": truth, "arima_test": a, "tcn_test": t, "ens_test": ens, "naive_test": prev,
        "eval_start": half, "weight": weight, "metrics": metrics,
        "arima_order": arima["order"], "tcn_history": tcn["history"],
        "last_date": last_date, "last_close": float(close[-1]),
        "next_date": last_date + pd.Timedelta(days=1),
        "next_arima": arima["next"], "next_arima_lo": arima["next_lo"], "next_arima_hi": arima["next_hi"],
        "next_tcn": tcn["next"],
        "next_ensemble": weight * arima["next"] + (1 - weight) * tcn["next"],
    }
