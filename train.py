"""
Train ARIMA + TCN once, offline, and save everything the website needs.

    python train.py                          # trains the default coins
    python train.py --coins bitcoin solana   # only these coins
    python train.py --auto-order             # also grid-search the ARIMA order (slower)

Output: models/<TICKER>/tcn.keras, scalers.joblib, meta.json
Re-run every week or two so the models see recent prices.
"""
import argparse
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from coins import DEFAULT_TRAIN, resolve
from model_utils import (DEFAULT_ORDER, FEATURE_COLS, N_WINDOW, _sarimax, _set_seeds,
                         _windows, add_features, build_tcn, direction_accuracy,
                         load_prices, pick_weight, score, select_arima_order)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def train_coin(ticker: str, start: str, epochs: int, fine_tune: int, auto_order: bool):
    from tensorflow.keras import callbacks

    t0 = time.time()
    df = add_features(load_prices(ticker, start))
    n = len(df)
    if n < 400:
        raise ValueError(f"only {n} days of data")
    split = int(n * 0.8)
    close = df["Close"].to_numpy()

    # ---------------- ARIMA ----------------
    vol = df["Volume_lag1"].to_numpy().reshape(-1, 1)
    vol_scaler = StandardScaler().fit(vol[:split])
    exog = vol_scaler.transform(vol)

    order = select_arima_order(close[:split], exog[:split]) if auto_order else DEFAULT_ORDER
    res_tr = _sarimax(close[:split], exog[:split], order).fit(disp=False, maxiter=200)
    arima_test = np.asarray(res_tr.apply(close, exog=exog)
                            .get_prediction(start=split, end=n - 1).predicted_mean)
    res_all = _sarimax(close, exog, order).fit(disp=False, maxiter=200,
                                               start_params=res_tr.params)

    # ---------------- TCN ----------------
    _set_seeds()
    scaler_x = StandardScaler().fit(df[FEATURE_COLS].to_numpy()[:split])
    scaler_y = StandardScaler().fit(df[["LogReturn"]].to_numpy()[:split])
    feats = scaler_x.transform(df[FEATURE_COLS].to_numpy())
    target = scaler_y.transform(df[["LogReturn"]].to_numpy()).ravel()

    tr_idx, te_idx = np.arange(N_WINDOW, split), np.arange(split, n)
    X_tr, y_tr = _windows(feats, tr_idx, N_WINDOW), target[tr_idx]
    v = int(len(X_tr) * 0.9)

    model = build_tcn((N_WINDOW, len(FEATURE_COLS)))
    model.fit(X_tr[:v], y_tr[:v], validation_data=(X_tr[v:], y_tr[v:]),
              epochs=epochs, batch_size=32, verbose=0,
              callbacks=[callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                         callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5)])
    ret = scaler_y.inverse_transform(model.predict(_windows(feats, te_idx, N_WINDOW), verbose=0)).ravel()
    tcn_test = close[te_idx - 1] * np.exp(ret)

    # ---------------- Ensemble weight + honest scores ----------------
    truth, prev = close[split:], close[split - 1:-1]
    half = len(truth) // 2
    w = pick_weight(truth[:half], arima_test[:half], tcn_test[:half])
    ens = w * arima_test + (1 - w) * tcn_test
    metrics = {}
    for name, p in [("Ensemble", ens), ("ARIMA", arima_test), ("TCN", tcn_test), ("Naive", prev)]:
        m = score(truth[half:], p[half:])
        m["Direction acc. %"] = None if name == "Naive" else direction_accuracy(prev[half:], truth[half:], p[half:])
        metrics[name] = m

    # ---------------- Final TCN: fine-tune on all data ----------------
    if fine_tune > 0:
        all_idx = np.arange(N_WINDOW, n)
        model.fit(_windows(feats, all_idx, N_WINDOW), target[all_idx],
                  epochs=fine_tune, batch_size=32, verbose=0)

    # ---------------- Save ----------------
    out = os.path.join(MODELS_DIR, ticker)
    os.makedirs(out, exist_ok=True)
    model.save(os.path.join(out, "tcn.keras"))
    joblib.dump({"x": scaler_x, "y": scaler_y, "vol": vol_scaler}, os.path.join(out, "scalers.joblib"))
    meta = {
        "ticker": ticker,
        "order": list(order),
        "arima_params": [float(p) for p in np.asarray(res_all.params)],
        "weight": w,
        "window": N_WINDOW,
        "trained_until": str(pd.Timestamp(df["Date"].iloc[-1]).date()),
        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "metrics": metrics,
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    import tensorflow as tf
    tf.keras.backend.clear_session()
    e = metrics["Ensemble"]
    print(f"  saved {out}  |  ARIMA{tuple(order)} weight={w:.2f}  "
          f"MAPE={e['MAPE %']:.2f}% (naive {metrics['Naive']['MAPE %']:.2f}%)  "
          f"[{time.time() - t0:.0f}s]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", nargs="*", help="names or symbols, e.g. bitcoin eth solana")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--fine-tune", type=int, default=5)
    ap.add_argument("--auto-order", action="store_true")
    args = ap.parse_args()

    tickers = [resolve(c) for c in args.coins] if args.coins else DEFAULT_TRAIN
    print(f"Training {len(tickers)} coin(s): {', '.join(tickers)}")
    for t in tickers:
        print(f"- {t}")
        try:
            train_coin(t, args.start, args.epochs, args.fine_tune, args.auto_order)
        except Exception as e:
            print(f"  skipped: {e}")


if __name__ == "__main__":
    main()
