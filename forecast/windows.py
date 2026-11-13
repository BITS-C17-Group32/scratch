"""Transform a clean daily series into scaled windows and an 80/20 split.

scaled_channels builds the feature matrices, make_dataset constructs sliding
windows from them and splits arrays chronologically. Every model receives the
same dataset dictionary so results stay comparable.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from forecast.cleaning import CALENDAR_FLAG_COLS

LOOKBACK = 28

# Models trained on fewer than ~120 windows do not reliably beat a naive weekly
# t-7 baseline in A/B tests, so shorter series are rejected.
MIN_TRAIN = 120

# Rollouts compound each model's own error, so beyond about 90 days the
# numbers are directional at best.
MAX_HORIZON = 90

# scaled_channels stacks these calendar columns between promo and the
# day-of-week pair. The base pair is the always-on set; the full block adds
# the scoped national, regional, and local holiday channels and both days of
# leads for every flag.
BASE_FLAG_COLS = ("is_holiday", "is_event")
FULL15_FLAG_COLS = CALENDAR_FLAG_COLS

# The 15-channel block needs enough training windows to pay for its extra
# inputs. The calendar_ab.py A/B (one GROCERY I series per store rung, two
# seeds) put the boundary between 704 and 912: below it the block averages
# +16 MAE (worse; LSTM +108 at 575 windows), at 912+ it averages -25 and all
# three models win at 912 and 1321.
FULL15_MIN_TRAIN = 900


class Dataset(TypedDict):
    """Dataset dictionary that make_dataset produces and every trainer consumes."""

    X_seq_train: np.ndarray
    X_today_train: np.ndarray
    y_train: np.ndarray
    X_seq_test: np.ndarray
    X_today_test: np.ndarray
    y_test: np.ndarray
    sales_scaler: MinMaxScaler
    promo_scaler: MinMaxScaler
    raw_sales: np.ndarray
    first_test_index: int
    test_dates: np.ndarray
    use_full15: bool


def scaled_channels(
    series: pd.DataFrame,
    sales_scaler: MinMaxScaler | None = None,
    promo_scaler: MinMaxScaler | None = None,
    flag_cols: tuple[str, ...] = BASE_FLAG_COLS,
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """Build the scaled history and known-today channel matrices for a frame.

    Each history row carries the channels [sales, log-promo, *flag_cols,
    dow_sin, dow_cos], and the known-today matrix drops sales because it is
    the label. flag_cols selects the calendar channels: the base holiday/event
    pair, or the 15-column block that adds the scoped holiday flags and every
    flag's two-day leads. The frame must carry the named columns, which
    load_series and flags_frame provide. The two MinMax scalers are fitted
    when not supplied, so callers holding fitted scalers reuse the same
    transform. Pass both or neither, so a partial pair never fits one scaler
    on the frame being transformed.
    """
    if (sales_scaler is None) != (promo_scaler is None):
        raise ValueError("pass both scalers or neither")
    sales = series["sales"].to_numpy(dtype=np.float32)
    promo = series["onpromotion"].to_numpy(dtype=np.float32)
    flags = [series[c].to_numpy(dtype=np.float32) for c in flag_cols]

    weekday = series["date"].dt.weekday.to_numpy()
    dow_sin = np.sin(2 * np.pi * weekday / 7).astype(np.float32)
    dow_cos = np.cos(2 * np.pi * weekday / 7).astype(np.float32)

    # Scale sales with MinMax instead of a log transform. An A/B test showed
    # log-space training roughly doubled MAE in absolute units.
    # Ref: https://scikit-learn.org/stable/modules/preprocessing.html
    if sales_scaler is None:
        sales_scaler = MinMaxScaler(feature_range=(0, 1)).fit(sales.reshape(-1, 1))
    scaled_sales = (
        sales_scaler.transform(sales.reshape(-1, 1)).astype(np.float32).flatten()
    )

    # Promotion counts have a skewed tail and their scale shifts in 2017, so
    # log-compress them before scaling. Use log1p rather than log because
    # promotion counts are often zero and log(0) is undefined.
    # Ref: https://numpy.org/doc/stable/reference/generated/numpy.log1p.html
    log_promo = np.log1p(promo).reshape(-1, 1)
    if promo_scaler is None:
        promo_scaler = MinMaxScaler(feature_range=(0, 1)).fit(log_promo)
    scaled_promo = promo_scaler.transform(log_promo).astype(np.float32).flatten()

    hist_channels = np.stack(
        [scaled_sales, scaled_promo, *flags, dow_sin, dow_cos], axis=1
    )
    today_channels = np.stack([scaled_promo, *flags, dow_sin, dow_cos], axis=1)
    return hist_channels, today_channels, sales_scaler, promo_scaler


def make_dataset(
    series: pd.DataFrame, lookback: int = LOOKBACK, use_full15: bool | None = None
) -> Dataset:
    """Scale, window, and split the series into chronological training and test sets.

    Each target day t receives two inputs. The first is a history sequence over
    rows t-lookback to t-1 with channels [sales, log-promo, *flag_cols,
    dow_sin, dow_cos]. The second is a known-today vector from row t
    holding the same-day features known in advance, [log-promo, *flag_cols,
    dow_sin, dow_cos]. Sales is absent from the second vector because
    it is the label.

    The calendar block is conditional. With use_full15 unset, a series with
    at least FULL15_MIN_TRAIN training windows takes the 15-channel block and
    a shorter one keeps the base holiday/event pair — the split the
    calendar_ab.py A/B pinned. A frame built without the flag columns always
    gets the base pair.
    """
    sales = series["sales"].to_numpy(dtype=np.float32)

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

    if use_full15 is None:
        use_full15 = split_index >= FULL15_MIN_TRAIN and all(
            col in series.columns for col in FULL15_FLAG_COLS
        )
    elif use_full15 and any(col not in series.columns for col in FULL15_FLAG_COLS):
        raise ValueError(
            "use_full15 needs the calendar lead columns; build the frame with "
            "load_series"
        )
    flag_cols = FULL15_FLAG_COLS if use_full15 else BASE_FLAG_COLS

    hist_channels, today_channels, sales_scaler, promo_scaler = scaled_channels(
        series, flag_cols=flag_cols
    )

    X_seq, X_today, y = [], [], []
    for i in range(lookback, total_days):
        X_seq.append(hist_channels[i - lookback : i])
        X_today.append(today_channels[i])
        # The sales channel is the MinMax-scaled target the models train on.
        y.append(hist_channels[i, 0])
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
        "promo_scaler": promo_scaler,
        "raw_sales": sales,
        "first_test_index": first_test,
        "test_dates": series["date"].to_numpy()[first_test:],
        "use_full15": use_full15,
    }
