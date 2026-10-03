"""
Fast prediction using models saved by train.py. No training happens here.

For a pre-trained coin: ~2-5 s (download recent prices + run saved models).
For any other coin: a quick ARIMA-only model is fitted on recent data (~5-10 s).
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

from model_utils import DEFAULT_ORDER, FEATURE_COLS, _sarimax, add_features, load_prices

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
HISTORY_DAYS = 90        # longest chart
DOWNLOAD_DAYS = 800      # enough for indicators, the 60-day window and ARIMA warm-up


def available_models() -> list:
    if not os.path.isdir(MODELS_DIR):
        return []
    return sorted(d for d in os.listdir(MODELS_DIR)
                  if os.path.exists(os.path.join(MODELS_DIR, d, "meta.json")))


def load_artifacts(ticker: str):
    """Load the saved TCN, scalers and ARIMA settings, or None if not trained."""
    folder = os.path.join(MODELS_DIR, ticker)
    if not os.path.exists(os.path.join(folder, "meta.json")):
        return None
    import tensorflow as tf
    model = tf.keras.models.load_model(os.path.join(folder, "tcn.keras"), compile=False)
    scalers = joblib.load(os.path.join(folder, "scalers.joblib"))
    with open(os.path.join(folder, "meta.json")) as f:
        meta = json.load(f)
    return model, scalers, meta


def fetch_recent(ticker: str) -> pd.DataFrame:
    start = (pd.Timestamp.today() - pd.Timedelta(days=DOWNLOAD_DAYS)).date().isoformat()
    return add_features(load_prices(ticker, start))


def live_quote(ticker: str) -> dict:
    """Current price and intraday stats from Yahoo (any field may be None)."""
    out = {"price": None, "day_high": None, "day_low": None, "market_cap": None}
    try:
        fi = yf.Ticker(ticker).fast_info
    except Exception:
        return out
    for key, attr in [("price", "last_price"), ("day_high", "day_high"),
                      ("day_low", "day_low"), ("market_cap", "market_cap")]:
        try:
            v = getattr(fi, attr)
            out[key] = float(v) if v else None
        except Exception:
            pass
    return out


def make_forecast(feat: pd.DataFrame, artifacts=None) -> dict:
    n = len(feat)
    close = feat["Close"].to_numpy()
    vol_lag = feat["Volume_lag1"].to_numpy().reshape(-1, 1)
    vol_today = feat["Volume"].to_numpy()[-1:].reshape(-1, 1)

    if artifacts is not None:
        model, sc, meta = artifacts
        window = int(meta["window"])
        k = min(HISTORY_DAYS, n - window - 1)
        idx = np.arange(n - k, n)

        # ARIMA with the saved parameters: just run the Kalman filter, no fitting
        order = tuple(meta["order"])
        exog = sc["vol"].transform(vol_lag)
        res = _sarimax(close, exog, order).filter(np.asarray(meta["arima_params"]))
        a_hist = np.asarray(res.get_prediction(start=n - k, end=n - 1).predicted_mean)
        fc = res.get_forecast(steps=1, exog=sc["vol"].transform(vol_today))
        a_next = float(np.asarray(fc.predicted_mean)[0])
        ci = np.asarray(fc.conf_int(alpha=0.05))[0]

        # TCN: one batch with the last k windows + tomorrow's window
        feats = sc["x"].transform(feat[FEATURE_COLS].to_numpy())
        X = np.stack([feats[i - window:i] for i in idx] + [feats[n - window:n]]).astype("float32")
        ret = sc["y"].inverse_transform(np.asarray(model(X, training=False))).ravel()
        tcn_all = np.append(close[idx - 1], close[-1]) * np.exp(ret)
        t_hist, t_next = tcn_all[:-1], float(tcn_all[-1])

        w = float(meta["weight"])
        pred_hist = w * a_hist + (1 - w) * t_hist
        next_pred = w * a_next + (1 - w) * t_next
        mode = "ensemble"
    else:
        # Quick fallback for coins without a saved model
        k = min(HISTORY_DAYS, n - 30)
        idx = np.arange(n - k, n)
        scaler = StandardScaler().fit(vol_lag)
        exog = scaler.transform(vol_lag)
        res = _sarimax(close, exog, DEFAULT_ORDER).fit(disp=False, maxiter=50)
        pred_hist = np.asarray(res.get_prediction(start=n - k, end=n - 1).predicted_mean)
        fc = res.get_forecast(steps=1, exog=scaler.transform(vol_today))
        a_next = next_pred = float(np.asarray(fc.predicted_mean)[0])
        ci = np.asarray(fc.conf_int(alpha=0.05))[0]
        t_next, w, meta = None, 1.0, None
        mode = "arima_quick"

    last_date = pd.Timestamp(feat["Date"].iloc[-1])
    return {
        "mode": mode,
        "dates": feat["Date"].iloc[idx].to_numpy(),
        "actual": close[idx],
        "predicted": pred_hist,
        "last_date": last_date,
        "last_close": float(close[-1]),
        "next_date": last_date + pd.Timedelta(days=1),
        "next_pred": float(next_pred),
        # 95% range: ARIMA interval width, centred on the final forecast
        "next_lo": float(next_pred - (ci[1] - ci[0]) / 2),
        "next_hi": float(next_pred + (ci[1] - ci[0]) / 2),
        "arima_next": a_next,
        "tcn_next": t_next,
        "weight": w,
        "meta": meta,
    }
