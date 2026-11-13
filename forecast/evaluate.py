"""Evaluation metrics, summary table, and two plots."""

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


def evaluate(predictions: np.ndarray, actuals: np.ndarray) -> tuple[float, float]:
    """Compute the mean absolute error (MAE) and root mean squared error (RMSE).

    The app passes sales after invert_sales, so both numbers are in item
    units for the selected store and family. RMSE squares the residuals
    before averaging, so a large miss raises RMSE more than MAE. The final
    square root keeps RMSE in those units.
    """
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    return float(mae), float(rmse)


def print_summary(metrics: dict[str, tuple[float, float]]) -> None:
    print("\n=== Summary (lower is better) ===")
    for name, (mae, rmse) in metrics.items():
        print(f"{name:>7}:  MAE {mae:8.2f}   RMSE {rmse:8.2f}")


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def _save_figure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300)
    print(f"saved {path}")
    return path


def plot_predictions(
    test_dates: np.ndarray,
    actuals: np.ndarray,
    preds: dict[str, np.ndarray],
    store: int,
    family: str,
    out_dir: Path = FIGURES_DIR,
    *,
    future: tuple[np.ndarray, dict[str, np.ndarray]] | None = None,
) -> Path:
    """Plot test-period actuals against each model's predictions.

    When a future rollout is supplied, each model's dashed forecast
    continues from its own last test prediction through the end date, and
    one shared legend entry marks the forecast style. The dashed part
    comes from the separate full-series retrain, so the first dashed
    segment joins the test-period fit's last prediction to the retrained
    model's first forecast.
    """
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
    if future is not None:
        future_dates, future_preds = future
        # Each dashed forecast anchors at its model's own last test
        # prediction, so the first dashed segment joins the test-period
        # fit to the full-series retrain.
        for name, pred in future_preds.items():
            plt.plot(
                np.concatenate([test_dates[-1:], future_dates]),
                np.concatenate([preds[name][-1:], pred]),
                color=COLORS[name],
                alpha=0.8,
                linestyle="--",
                marker="o",
                markersize=3,
                label="_nolegend_",
            )
        plt.plot(
            [],
            [],
            color="gray",
            linestyle="--",
            label="Forecast (dashed, promotions assumed zero)",
        )
    suffix = " and Forecast" if future is not None else ""
    plt.title(f"Actual vs. Predicted{suffix} Demand, store {store} / {family}")
    plt.xlabel("Date")
    plt.ylabel("Sales Quantity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    return _save_figure(path)


def plot_errors(
    metrics: dict[str, tuple[float, float]],
    store: int,
    family: str,
    out_dir: Path = FIGURES_DIR,
) -> Path:
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
    return _save_figure(path)
