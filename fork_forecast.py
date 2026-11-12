"""Forecast one Favorita (store, family) daily series with five models
(RNN, GRU, LSTM, HistGradientBoosting, SARIMAX) on an 80/20 chronological
split, evaluated one-step-ahead on actual history.

    .venv/bin/python fork_forecast.py --store 1 --family "GROCERY I"

Store/family selection is interactive when the flags are omitted; --list
shows the valid choices. Only the selected pair's rows are ever loaded
from train.csv. Scopes live in forecast/: cleaning (raw slice -> clean
series), windows (series -> model arrays), models (training), evaluate
(metrics + plots).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from forecast.cleaning import available_series, load_series
from forecast.evaluate import evaluate, plot_errors, plot_predictions, print_summary
from forecast.models import invert_sales, train_all
from forecast.select import choose_family, choose_store, match_family, print_families
from forecast.windows import LOOKBACK, make_dataset

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "raw"


def pick_series(args: argparse.Namespace) -> tuple[int, str]:
    """Resolve (store, family) from flags or interactive prompts."""
    stores, families = available_series(DATA_DIR)

    if args.list:
        print(f"Stores ({len(stores)}): {', '.join(map(str, stores))}")
        print(f"Families ({len(families)}):")
        print_families(families)
        sys.exit(0)

    if args.store is not None:
        if args.store not in stores:
            sys.exit(
                f"store {args.store} is not valid; valid range is "
                f"{stores[0]}..{stores[-1]}"
            )
        store = args.store
    else:
        store = choose_store(stores)

    if args.family is not None:
        family = match_family(args.family, families)
        if family is None:
            sys.exit(
                f"family {args.family!r} did not match; run with --list to see choices"
            )
    else:
        family = choose_family(families)

    return store, family


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--store", type=int, default=None, help="store_nbr (omit to be prompted)"
    )
    parser.add_argument(
        "--family",
        type=str,
        default=None,
        help="family name, unique prefix, or its --list number (omit to be prompted)",
    )
    parser.add_argument(
        "--list", action="store_true", help="print valid stores and families"
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=LOOKBACK,
        help=f"window length in days (default {LOOKBACK})",
    )
    parser.add_argument(
        "--no-show", action="store_true", help="save plots without opening a window"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not (DATA_DIR / "train.csv").is_file():
        sys.exit(f"raw data not found at {DATA_DIR}")

    store, family = pick_series(args)
    try:
        series = load_series(DATA_DIR, store, family)
        print(
            f"\nLoaded {len(series):,} days for store {store} / {family} "
            f"({series['date'].min():%Y-%m-%d} to {series['date'].max():%Y-%m-%d})."
        )
        data = make_dataset(series, args.lookback)
    except ValueError as err:
        sys.exit(f"error: {err}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds = train_all(data, device)
    actuals = invert_sales(data["y_test"], data["sales_scaler"])

    metrics = {name: evaluate(name, pred, actuals) for name, pred in preds.items()}
    print_summary(metrics)

    plot_predictions(data["test_dates"], actuals, preds, store, family, HERE)
    plot_errors(metrics, store, family, HERE)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
