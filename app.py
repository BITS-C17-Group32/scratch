"""Interactive five-model demand forecasting application.

Run with 'uv run app.py'. The prompt takes a store number, then a family by
name, unique prefix, or list number. The app streams only that pair's rows
from train.csv, trains five forecasters (RNN, GRU, LSTM, HGB, SARIMAX) on an
80/20 chronological split, prints the error table, and opens two chart
windows with the test-period predictions and errors. Optionally, two future
dates retrain the models on the full series and extend the predictions
chart through the requested end date, including forecasted days before
the start.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from forecast.cleaning import load_series
from forecast.evaluate import evaluate, plot_errors, plot_predictions, print_summary
from forecast.future import forecast_future, print_future_table
from forecast.models import invert_sales, train_all
from forecast.select import choose_future_dates, choose_series
from forecast.windows import make_dataset

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "raw"


def main() -> None:
    print("=" * 60)
    print("        DAILY DEMAND FORECASTING - INTERACTIVE APP       ")
    print("=" * 60)

    if not (DATA_DIR / "train.csv").is_file():
        sys.exit(f"raw data not found at {DATA_DIR}")

    try:
        store, family = choose_series(DATA_DIR)
        series = load_series(DATA_DIR, store, family)
        print(
            f"\nLoaded {len(series):,} days for store {store} / {family} "
            f"({series['date'].min():%Y-%m-%d} to {series['date'].max():%Y-%m-%d})."
        )
        future_dates = choose_future_dates(series["date"].max())
        dataset = make_dataset(series)
    except ValueError as err:
        sys.exit(f"error: {err}")

    print("\nTraining five models (RNN, GRU, LSTM, HGB, SARIMAX)...")
    # Only the PyTorch RNN takes this device. CUDA is preferred when a GPU is
    # present, then Apple's MPS on Apple Silicon Macs, and CPU is the fallback.
    # AMD GPUs also use the CUDA branch. A ROCm build of PyTorch reports
    # torch.cuda.is_available() as True and selects the AMD GPU with
    # device("cuda"). The Keras models follow the TensorFlow backend, and HGB
    # and SARIMAX stay on CPU.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    predictions = train_all(dataset, device)
    actual_sales = invert_sales(dataset["y_test"], dataset["sales_scaler"])

    metrics = {
        model_name: evaluate(pred, actual_sales)
        for model_name, pred in predictions.items()
    }
    print_summary(metrics)

    future: tuple[np.ndarray, dict[str, np.ndarray]] | None = None
    if future_dates is not None:
        rollout_days, rollout_preds, first_requested = forecast_future(
            series, dataset, device, DATA_DIR, store, *future_dates
        )
        reported_days = rollout_days[first_requested:]
        reported_preds = {
            name: pred[first_requested:] for name, pred in rollout_preds.items()
        }
        print_future_table(reported_days, reported_preds)
        future = rollout_days, rollout_preds

    plot_predictions(
        dataset["test_dates"], actual_sales, predictions, store, family, future=future
    )
    plot_errors(metrics, store, family)

    # plot_predictions and plot_errors save each PNG before plt.show, so a
    # headless backend loses only the chart window.
    # Ref: https://matplotlib.org/stable/users/explain/figure/backends.html
    backend = os.environ.get("MPLBACKEND", "").lower()
    if backend in {"agg", "pdf", "svg", "ps"}:
        print(
            f"\nnote: MPLBACKEND={os.environ['MPLBACKEND']} cannot open chart "
            "windows; the plots are still saved as PNGs in figures/. "
            "Run `unset MPLBACKEND` and start again to see them."
        )

    print("\nDisplaying chart windows (close them to exit)...")
    plt.show()


if __name__ == "__main__":
    main()
