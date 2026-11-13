"""Evaluation: metrics, summary table, and the two comparison plots."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

COLORS = {
    "RNN": "blue",
    "GRU": "orange",
    "LSTM": "green",
    "HGB": "red",
    "SARIMAX": "purple",
}


def evaluate(name, predictions, actuals):
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    print(f"\n--- {name} Results ---")
    print(f"Test MAE:  {mae:.2f}")
    print(f"Test RMSE: {rmse:.2f}")
    return float(mae), float(rmse)


def print_summary(metrics):
    print("\n=== Summary (lower is better) ===")
    for name, (mae, rmse) in metrics.items():
        print(f"{name:>7}:  MAE {mae:8.2f}   RMSE {rmse:8.2f}")


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def plot_predictions(
    test_dates, actuals, preds, store, family, out_dir: Path = FIGURES_DIR
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"fork_actual_vs_predicted_s{store}_{slugify(family)}.png"
    plt.figure(figsize=(14, 6))
    plt.plot(
        test_dates,
        actuals,
        label="Actual Demand",
        color="black",
        linewidth=2,
        linestyle="--",
    )
    for name, pred in preds.items():
        plt.plot(
            test_dates, pred, label=f"{name} Predictions", color=COLORS[name], alpha=0.8
        )
    plt.title(f"Actual vs. Predicted Demand, store {store} / {family}")
    plt.xlabel("Date")
    plt.ylabel("Sales Quantity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(path, dpi=300)
    print(f"saved {path}")
    return path


def plot_errors(metrics, store, family, out_dir: Path = FIGURES_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"fork_error_comparison_s{store}_{slugify(family)}.png"
    models = list(metrics)
    mae_values = [metrics[m][0] for m in models]
    rmse_values = [metrics[m][1] for m in models]

    x = np.arange(len(models))
    width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, mae_values, width, label="MAE", color="skyblue")
    plt.bar(x + width / 2, rmse_values, width, label="RMSE", color="salmon")
    plt.title(f"Error Metrics, store {store} / {family}")
    plt.xlabel("Forecasting Models")
    plt.ylabel("Error Value (Lower is Better)")
    plt.xticks(x, models)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(path, dpi=300)
    print(f"saved {path}")
    return path
