"""Windowing: clean daily series -> scaled windows and an 80/20 split.

Scope: everything between a clean series and model-ready arrays —
channel building, scaling, sliding windows, chronological split. Every
model receives the same dataset dict so results stay comparable.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

LOOKBACK = 28

# Below ~120 train windows the models stop reliably beating a naive
# weekly (t-7) baseline (measured in A/B tests), so shorter series are
# refused rather than forecast on.
MIN_TRAIN = 120


class Dataset(TypedDict):
    """The make_dataset payload every trainer consumes."""

    X_seq_train: np.ndarray
    X_today_train: np.ndarray
    y_train: np.ndarray
    X_seq_test: np.ndarray
    X_today_test: np.ndarray
    y_test: np.ndarray
    sales_scaler: MinMaxScaler
    raw_sales: np.ndarray
    first_test_index: int
    test_dates: np.ndarray


def make_dataset(series: pd.DataFrame, lookback: int = LOOKBACK) -> Dataset:
    """Scale, window, split.

    Two input pieces per target day t:
      * a history sequence over rows t-lookback..t-1 with channels
        [sales, log-promo, is_holiday, is_event, dow_sin, dow_cos];
      * a "known today" vector from row t with the target day's own
        known-in-advance features [log-promo, is_holiday, is_event,
        dow_sin, dow_cos] (no sales — that is the label).
    """
    sales = series["sales"].to_numpy(dtype=np.float32)
    promo = series["onpromotion"].to_numpy(dtype=np.float32)
    is_holiday = series["is_holiday"].to_numpy(dtype=np.float32)
    is_event = series["is_event"].to_numpy(dtype=np.float32)

    weekday = series["date"].dt.weekday.to_numpy()
    dow_sin = np.sin(2 * np.pi * weekday / 7).astype(np.float32)
    dow_cos = np.cos(2 * np.pi * weekday / 7).astype(np.float32)

    # Raw-unit targets: log-space training cost ~2x MAE in absolute units
    # in an A/B test. Promotion counts have a huge skewed tail, so
    # log-compress them before scaling (their scale also shifts in 2017).
    sales_scaler = MinMaxScaler(feature_range=(0, 1)).fit(sales.reshape(-1, 1))
    s = sales_scaler.transform(sales.reshape(-1, 1)).astype(np.float32).flatten()
    log_promo = np.log1p(promo).reshape(-1, 1)
    promo_scaler = MinMaxScaler(feature_range=(0, 1)).fit(log_promo)
    p = promo_scaler.transform(log_promo).astype(np.float32).flatten()

    hist_channels = np.stack([s, p, is_holiday, is_event, dow_sin, dow_cos], axis=1)
    today_channels = np.stack([p, is_holiday, is_event, dow_sin, dow_cos], axis=1)

    X_seq, X_today, y = [], [], []
    for i in range(lookback, len(s)):
        X_seq.append(hist_channels[i - lookback : i])
        X_today.append(today_channels[i])
        y.append(s[i])
    X_seq, X_today, y = np.array(X_seq), np.array(X_today), np.array(y)

    # Split by raw-series length so train/test days match across all models.
    first_test = int(len(s) * 0.8)
    split_index = first_test - lookback
    if len(s) <= lookback:
        raise ValueError(
            f"series has {len(s)} days, fewer than the {lookback}-day lookback"
        )
    if split_index < MIN_TRAIN:
        raise ValueError(
            f"too little history to forecast: {len(s)} days give only "
            f"{split_index} train windows (minimum {MIN_TRAIN})"
        )
    print(
        f"Data loaded successfully! Inputs: {hist_channels.shape[1]} history "
        f"channels x {lookback} days + {today_channels.shape[1]} known-today "
        f"features. Train samples: {split_index}, Test samples: {len(y) - split_index}"
    )
    return {
        "X_seq_train": X_seq[:split_index],
        "X_today_train": X_today[:split_index],
        "y_train": y[:split_index],
        "X_seq_test": X_seq[split_index:],
        "X_today_test": X_today[split_index:],
        "y_test": y[split_index:],
        "sales_scaler": sales_scaler,
        "raw_sales": sales,
        "first_test_index": first_test,
        "test_dates": series["date"].to_numpy()[first_test:],
    }
