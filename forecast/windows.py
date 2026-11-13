"""Transform a clean daily series into scaled windows and an 80/20 split.

Build feature channels, scale inputs, construct sliding windows, and split arrays
chronologically. Every model receives the same dataset dictionary so results
stay comparable.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

LOOKBACK = 28

# Models trained on fewer than ~120 windows do not reliably beat a naive weekly
# t-7 baseline in A/B tests, so shorter series are rejected.
MIN_TRAIN = 120


class Dataset(TypedDict):
    """Dataset dictionary that make_dataset produces and every trainer consumes."""

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
    """Scale, window, and split the series into chronological training and test sets.

    Each target day t receives two inputs. The first is a history sequence over
    rows t-lookback to t-1 with channels [sales, log-promo, is_holiday,
    is_event, dow_sin, dow_cos]. The second is a known-today vector from row t
    holding the same-day features known in advance, [log-promo, is_holiday,
    is_event, dow_sin, dow_cos]. Sales is absent from the second vector because
    it is the label.
    """
    sales = series["sales"].to_numpy(dtype=np.float32)
    promo = series["onpromotion"].to_numpy(dtype=np.float32)
    is_holiday = series["is_holiday"].to_numpy(dtype=np.float32)
    is_event = series["is_event"].to_numpy(dtype=np.float32)

    total_days = len(sales)
    if total_days <= lookback:
        raise ValueError(
            f"series has {total_days} days, fewer than the {lookback}-day lookback"
        )

    # Split by raw-series length so train/test days match across all models.
    first_test = int(total_days * 0.8)
    split_index = first_test - lookback
    if split_index < MIN_TRAIN:
        raise ValueError(
            f"too little history to forecast: {total_days} days give only "
            f"{split_index} train windows (minimum {MIN_TRAIN})"
        )

    weekday = series["date"].dt.weekday.to_numpy()
    dow_sin = np.sin(2 * np.pi * weekday / 7).astype(np.float32)
    dow_cos = np.cos(2 * np.pi * weekday / 7).astype(np.float32)

    # Scale sales with MinMax instead of a log transform. An A/B test showed
    # log-space training roughly doubled MAE in absolute units.
    # Ref: https://scikit-learn.org/stable/modules/preprocessing.html
    sales_scaler = MinMaxScaler(feature_range=(0, 1)).fit(sales.reshape(-1, 1))
    scaled_sales = (
        sales_scaler.transform(sales.reshape(-1, 1)).astype(np.float32).flatten()
    )

    # Promotion counts have a skewed tail and their scale shifts in 2017, so
    # log-compress them before scaling. Use log1p rather than log because
    # promotion counts are often zero and log(0) is undefined.
    # Ref: https://numpy.org/doc/stable/reference/generated/numpy.log1p.html
    log_promo = np.log1p(promo).reshape(-1, 1)
    promo_scaler = MinMaxScaler(feature_range=(0, 1)).fit(log_promo)
    scaled_promo = promo_scaler.transform(log_promo).astype(np.float32).flatten()

    hist_channels = np.stack(
        [scaled_sales, scaled_promo, is_holiday, is_event, dow_sin, dow_cos], axis=1
    )
    today_channels = np.stack(
        [scaled_promo, is_holiday, is_event, dow_sin, dow_cos], axis=1
    )

    X_seq, X_today, y = [], [], []
    for i in range(lookback, total_days):
        X_seq.append(hist_channels[i - lookback : i])
        X_today.append(today_channels[i])
        y.append(scaled_sales[i])
    seq_windows = np.array(X_seq)
    today_windows = np.array(X_today)
    targets = np.array(y)

    print(
        f"Data loaded successfully! Inputs: {hist_channels.shape[1]} history "
        f"channels x {lookback} days + {today_channels.shape[1]} known-today "
        f"features. Train samples: {split_index}, "
        f"Test samples: {len(targets) - split_index}"
    )
    return {
        "X_seq_train": seq_windows[:split_index],
        "X_today_train": today_windows[:split_index],
        "y_train": targets[:split_index],
        "X_seq_test": seq_windows[split_index:],
        "X_today_test": today_windows[split_index:],
        "y_test": targets[split_index:],
        "sales_scaler": sales_scaler,
        "raw_sales": sales,
        "first_test_index": first_test,
        "test_dates": series["date"].to_numpy()[first_test:],
    }
