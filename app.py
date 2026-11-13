"""Interactive five-model demand forecaster.

    uv run app.py

Pick a store and a product family at the prompt (store number, then
family by name, unique prefix, or list number), and the real pipeline
runs: only that pair's rows are ever loaded from train.csv, all five
models (RNN, GRU, LSTM, HGB, SARIMAX) train on the 80/20 chronological
split, the summary table prints, and two chart windows open with the
test-period predictions and errors.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from forecast.cleaning import load_series
from forecast.evaluate import evaluate, plot_errors, plot_predictions, print_summary
from forecast.models import invert_sales, train_all
from forecast.select import choose_series
from forecast.windows import make_dataset

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "raw"


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
        data = make_dataset(series)
    except ValueError as err:
        sys.exit(f"error: {err}")

    print("\nTraining five models (RNN, GRU, LSTM, HGB, SARIMAX)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds = train_all(data, device)
    actuals = invert_sales(data["y_test"], data["sales_scaler"])

    metrics = {name: evaluate(name, pred, actuals) for name, pred in preds.items()}
    print_summary(metrics)

    plot_predictions(data["test_dates"], actuals, preds, store, family)
    plot_errors(metrics, store, family)

    if os.environ.get("MPLBACKEND", "").lower() in {"agg", "pdf", "svg", "ps"}:
        print(
            f"\nnote: MPLBACKEND={os.environ['MPLBACKEND']} cannot open chart "
            "windows; the plots are still saved as PNGs in figures/. "
            "Run `unset MPLBACKEND` and start again to see them."
        )

    print("\nDisplaying chart windows (close them to exit)...")
    plt.show()


if __name__ == "__main__":
    main()
