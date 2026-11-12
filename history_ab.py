"""How much history does a forecast need?

    .venv/bin/python history_ab.py 1 "GROCERY I"

Fixes the test window to the final TEST_DAYS days of the series and
trains all five models (plus a seasonal-naive t-7 baseline) on
shrinking history prefixes. MAE is comparable across rows because the
test days never change; only the training history does.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from forecast.cleaning import load_series
from forecast.models import (
    build_gru,
    build_lstm,
    train_hgb,
    train_keras,
    train_rnn,
    train_sarimax,
)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "raw"
LOOKBACK = 28
TEST_DAYS = 90
HISTORIES = [45, 60, 90, 150, 300, 730, None]  # days before the test window


def make_fixed(series, lookback: int = LOOKBACK, test_days: int = TEST_DAYS) -> dict:
    """windows.make_dataset with the split pinned to the last test_days."""
    sales = series["sales"].to_numpy(dtype=np.float32)
    promo = series["onpromotion"].to_numpy(dtype=np.float32)
    wd = series["date"].dt.weekday.to_numpy()
    sin = np.sin(2 * np.pi * wd / 7).astype(np.float32)
    cos = np.cos(2 * np.pi * wd / 7).astype(np.float32)

    sales_scaler = MinMaxScaler(feature_range=(0, 1)).fit(sales.reshape(-1, 1))
    s = sales_scaler.transform(sales.reshape(-1, 1)).astype(np.float32).flatten()
    log_promo = np.log1p(promo).reshape(-1, 1)
    promo_scaler = MinMaxScaler(feature_range=(0, 1)).fit(log_promo)
    p = promo_scaler.transform(log_promo).astype(np.float32).flatten()

    hol = series["is_holiday"].to_numpy(dtype=np.float32)
    evt = series["is_event"].to_numpy(dtype=np.float32)
    hist = np.stack([s, p, hol, evt, sin, cos], axis=1)
    today = np.stack([p, hol, evt, sin, cos], axis=1)

    X_seq, X_today, y = [], [], []
    for i in range(lookback, len(s)):
        X_seq.append(hist[i - lookback : i])
        X_today.append(today[i])
        y.append(s[i])
    X_seq, X_today, y = np.array(X_seq), np.array(X_today), np.array(y)

    first_test = len(s) - test_days
    split = first_test - lookback
    return {
        "X_seq_train": X_seq[:split],
        "X_today_train": X_today[:split],
        "y_train": y[:split],
        "X_seq_test": X_seq[split:],
        "X_today_test": X_today[split:],
        "y_test": y[split:],
        "sales_scaler": sales_scaler,
        "raw_sales": sales,
        "first_test_index": first_test,
        "test_dates": series["date"].to_numpy()[first_test:],
    }


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: history_ab.py STORE FAMILY")
    store, family = int(sys.argv[1]), sys.argv[2]
    series = load_series(DATA, store, family)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for h in HISTORIES:
        sub = series if h is None else series.tail(h + TEST_DAYS).reset_index(drop=True)
        data = make_fixed(sub)
        n_train = len(data["y_train"])
        raw = data["raw_sales"]
        start = data["first_test_index"]
        actuals = raw[start:]
        naive7 = raw[start - 7 : len(raw) - 7]
        maes = {"naive7": float(np.mean(np.abs(naive7 - actuals)))}
        for name, fn in [
            ("RNN", lambda d: train_rnn(d, device)),
            ("GRU", lambda d: train_keras("GRU", build_gru, d)),
            ("LSTM", lambda d: train_keras("LSTM", build_lstm, d)),
            ("HGB", train_hgb),
            ("SARIMAX", train_sarimax),
        ]:
            try:
                preds = fn(data)
                maes[name] = float(np.mean(np.abs(preds - actuals)))
            except Exception as err:  # keep the sweep alive on one model failing
                print(f">> {name} failed at history {h}: {type(err).__name__}")
                maes[name] = float("nan")
        rows.append((h, n_train, maes))
        print(f">> history {h} done ({n_train} train windows)")

    cols = ["naive7", "RNN", "GRU", "LSTM", "HGB", "SARIMAX"]
    print(f">> store {store} / {family}: MAE on the same final {TEST_DAYS} days")
    print(
        ">> " + "hist".rjust(5) + "train".rjust(7) + "".join(c.rjust(9) for c in cols)
    )
    for h, n_train, maes in rows:
        cells = "".join(f"{maes[c]:>9.2f}" for c in cols)
        print(
            ">> "
            + (str(h) if h is not None else "full").rjust(5)
            + f"{n_train:>7}"
            + cells
        )


if __name__ == "__main__":
    main()
