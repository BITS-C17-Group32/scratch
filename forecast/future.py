"""Extrapolate the five models past the last recorded day.

choose_future_dates in select.py collects the window from the user.
forecast_future then retrains every model on the full series and rolls each
one forward day by day. The window models fit every window with no
validation holdout, so the newest recorded days are trained on too. A
window model predicts the next day from the trailing lookback window plus
that day's known features, writes the prediction into the window's sales
channel, and slides ahead. SARIMAX propagates its Kalman state through
forecast() instead. Future promotions are unknowable, so every future day
assumes zero promotions, and the run notes each flag type whose calendar
does not cover the window's end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from forecast.cleaning import flags_frame, store_calendars
from forecast.models import (
    WindowPredictor,
    build_gru,
    build_lstm,
    fit_hgb,
    fit_keras,
    fit_rnn,
    fit_sarimax,
    hgb_window_predictor,
    invert_sales,
    keras_window_predictor,
    rnn_window_predictor,
)
from forecast.windows import (
    BASE_FLAG_COLS,
    FULL15_FLAG_COLS,
    Dataset,
    scaled_channels,
)


def _full_series_dataset(data: Dataset) -> Dataset:
    """Rebuild the dataset with every window assigned to training.

    make_dataset splits 80/20, so the train and test windows together cover
    the full series. Concatenating them gives the future fits all of it, and
    first_test_index at the end leaves SARIMAX fitted through the last day.
    """
    return {
        "X_seq_train": np.concatenate([data["X_seq_train"], data["X_seq_test"]]),
        "X_today_train": np.concatenate([data["X_today_train"], data["X_today_test"]]),
        "y_train": np.concatenate([data["y_train"], data["y_test"]]),
        "X_seq_test": data["X_seq_train"][:0],
        "X_today_test": data["X_today_train"][:0],
        "y_test": data["y_train"][:0],
        "sales_scaler": data["sales_scaler"],
        "promo_scaler": data["promo_scaler"],
        "raw_sales": data["raw_sales"],
        "first_test_index": len(data["raw_sales"]),
        "test_dates": data["test_dates"][:0],
        "use_full15": data["use_full15"],
    }


def forecast_future(
    series: pd.DataFrame,
    data: Dataset,
    device: torch.device,
    data_dir: Path,
    store: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    """Retrain every model on the full series and forecast the future window.

    The window models fit every window with no validation holdout, so the
    newest recorded days are trained on too. The rollout runs from the day
    after the last recorded date through end, so a later start is still
    covered, and each window model rolls its own predictions forward.
    The calendar channels match the training arm, which the dataset records.
    series must carry the calendar flag columns, which load_series attaches.
    Returns the rollout dates, one raw-unit prediction array per model, and
    the index of the first requested date so the caller can slice the
    requested span out of the full rollout.
    """
    last_date = series["date"].max()
    full_windows = _full_series_dataset(data)
    lookback = full_windows["X_seq_train"].shape[1]

    print("\nRetraining all five models on the full series for the future forecast...")
    print("Training RNN on the full series...")
    rnn_model = fit_rnn(full_windows, device, holdout=False)
    print("Training GRU on the full series...")
    gru_model = fit_keras(build_gru, full_windows, holdout=False)
    print("Training LSTM on the full series...")
    lstm_model = fit_keras(build_lstm, full_windows, holdout=False)
    print("Training HGB on the full series...")
    hgb_model = fit_hgb(full_windows)
    print("Training SARIMAX on the full series...")
    sarimax_fit = fit_sarimax(full_windows)
    predictors: dict[str, WindowPredictor] = {
        "RNN": rnn_window_predictor(rnn_model, device),
        "GRU": keras_window_predictor(gru_model),
        "LSTM": keras_window_predictor(lstm_model),
        "HGB": hgb_window_predictor(hgb_model),
    }

    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), end, freq="D")
    national, regional, local, events = store_calendars(data_dir, store)
    holidays = national | regional | local
    for flag, flagged in (
        ("store-locale holiday", holidays),
        ("global event", events),
    ):
        known_until = max(flagged, default=None)
        if known_until is not None and end > known_until:
            print(
                f"note: {flag} flags after {known_until:%Y-%m-%d} are treated as none."
            )
    # The calendar block matches the training arm, which the dataset records,
    # so the rollout consumes the channel layout the fits were built on.
    flag_cols = FULL15_FLAG_COLS if data["use_full15"] else BASE_FLAG_COLS
    future_flags = flags_frame(national, regional, local, events, future_dates)
    # The zero sales column exists for scaled_channels, which always builds
    # the history matrix. The rollout only consumes the known-today rows.
    future_frame = pd.DataFrame(
        {
            "date": future_dates,
            "sales": np.zeros(len(future_dates), dtype=np.float32),
            "onpromotion": np.zeros(len(future_dates), dtype=np.float32),
            **{col: future_flags[col].to_numpy() for col in flag_cols},
        }
    )

    # The scalers come from make_dataset's fit on the recorded series, so the
    # rollout transforms predictions with the exact scaling training used.
    _, future_today, _, _ = scaled_channels(
        future_frame, data["sales_scaler"], data["promo_scaler"], flag_cols
    )
    hist_channels = scaled_channels(
        series, data["sales_scaler"], data["promo_scaler"], flag_cols
    )[0]
    # The seed window width must match what the fitted models expect, so it
    # derives from the windows they trained on rather than the default.
    seed_window = hist_channels[-lookback:]

    rollouts: dict[str, np.ndarray] = {}
    for name, predictor in predictors.items():
        window = seed_window.copy()
        scaled_days: list[float] = []
        for today_row in future_today:
            scaled = predictor(window, today_row)
            scaled_days.append(scaled)
            new_row = np.empty(seed_window.shape[1], dtype=np.float32)
            new_row[0] = scaled
            new_row[1:] = today_row
            window = np.vstack([window[1:], new_row])
        rollouts[name] = invert_sales(
            np.asarray(scaled_days, dtype=np.float32), data["sales_scaler"]
        )
    rollouts["SARIMAX"] = np.asarray(
        sarimax_fit.forecast(steps=(end - last_date).days), dtype=np.float64
    )

    offset = (start - last_date).days - 1
    return future_dates.to_numpy(), rollouts, offset


def print_future_table(dates: np.ndarray, preds: dict[str, np.ndarray]) -> None:
    """Print one prediction column per model for the requested future span."""
    print(
        f"\n=== Future forecast ({pd.Timestamp(dates[0]):%Y-%m-%d} to "
        f"{pd.Timestamp(dates[-1]):%Y-%m-%d}), promotions assumed zero ==="
    )
    header = f"{'date':>10}" + "".join(f"{name:>9}" for name in preds)
    print(header)
    for i, day in enumerate(dates):
        cells = "".join(f"{preds[name][i]:9.1f}" for name in preds)
        print(f"{pd.Timestamp(day):%Y-%m-%d}" + cells)
