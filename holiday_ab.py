"""Boundary study: does the 15-flag calendar set beat the two binary flags,
and where does history length flip the answer?

    .venv/bin/python holiday_ab.py "GROCERY I" 21 20 53 36 1 3

Per series, two feature arms x three seeds (42, 7, 123) — same series,
split, and hyperparameters; only the calendar channels differ:
  current2  is_holiday, is_event                       (production)
  full15    national/regional/local/holiday/event x t, +1d, +2d
Prints mean±std MAE per model and the full15 - current2 delta. SARIMAX
uses no calendar features and is deterministic, so it runs once per
series as a control.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

import forecast.models as fm
from forecast.cleaning import load_series
from forecast.models import (
    build_gru,
    build_lstm,
    invert_sales,
    train_hgb,
    train_keras,
    train_rnn,
    train_sarimax,
)
from forecast.windows import LOOKBACK, make_dataset

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "raw"
SEEDS = [42, 7, 123]

ARMS = {
    "current2": [],
    "full15": [
        "nat",
        "nat_tomorrow",
        "nat_in_2days",
        "reg",
        "reg_tomorrow",
        "reg_in_2days",
        "loc",
        "loc_tomorrow",
        "loc_in_2days",
        "hol",
        "hol_tomorrow",
        "hol_in_2days",
        "evt",
        "evt_tomorrow",
        "evt_in_2days",
    ],
}


def flag_bank(series, store: int) -> dict[str, np.ndarray]:
    """All 15 calendar flag arrays, aligned to the series dates."""
    stores = pd.read_csv(DATA / "stores.csv")
    row = stores[stores["store_nbr"].eq(store)].iloc[0]
    city, state = row["city"], row["state"]

    h = pd.read_csv(DATA / "holidays_events.csv", parse_dates=["date"])
    h = h[~h["transferred"]]
    events = set(
        h.loc[
            h["type"].eq("Event")
            & ~h["description"].str.contains("Terremoto Manabi", case=False),
            "date",
        ]
    )
    h = h[h["type"].isin(["Holiday", "Transfer", "Additional", "Bridge"])]
    nat_d = set(h.loc[h["locale"].eq("National"), "date"])
    reg_p = set(
        zip(
            h.loc[h["locale"].eq("Regional"), "date"],
            h.loc[h["locale"].eq("Regional"), "locale_name"],
            strict=True,
        )
    )
    loc_p = set(
        zip(
            h.loc[h["locale"].eq("Local"), "date"],
            h.loc[h["locale"].eq("Local"), "locale_name"],
            strict=True,
        )
    )

    # Extend two days past the series so the lead flags stay real values.
    ext = pd.date_range(
        series["date"].min(), series["date"].max() + pd.Timedelta(days=2), freq="D"
    )
    nat = np.array([d in nat_d for d in ext], dtype=np.float32)
    reg = np.array([(d, state) in reg_p for d in ext], dtype=np.float32)
    loc = np.array([(d, city) in loc_p for d in ext], dtype=np.float32)
    hol = (nat + reg + loc > 0).astype(np.float32)
    evt = np.array([d in events for d in ext], dtype=np.float32)

    n = len(series)
    bank: dict[str, np.ndarray] = {}
    for name, arr in [
        ("nat", nat),
        ("reg", reg),
        ("loc", loc),
        ("hol", hol),
        ("evt", evt),
    ]:
        bank[name] = arr[:n]
        bank[name + "_tomorrow"] = arr[1 : n + 1]
        bank[name + "_in_2days"] = arr[2 : n + 2]
    return bank


def make_arm(series, flags: list[np.ndarray], lookback: int = LOOKBACK) -> dict:
    """windows.make_dataset with a configurable calendar-channel block."""
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

    hist = np.stack([s, p, *flags, sin, cos], axis=1)
    today = np.stack([p, *flags, sin, cos], axis=1)

    X_seq, X_today, y = [], [], []
    for i in range(lookback, len(s)):
        X_seq.append(hist[i - lookback : i])
        X_today.append(today[i])
        y.append(s[i])
    X_seq, X_today, y = np.array(X_seq), np.array(X_today), np.array(y)

    first_test = int(len(s) * 0.8)
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


def run_series(store: int, family: str, device) -> None:
    series = load_series(DATA, store, family)
    bank = flag_bank(series, store)
    first_test = int(len(series) * 0.8)
    print(
        f">> SERIES store={store} family={family} days={len(series)} "
        f"train={first_test - LOOKBACK} test={len(series) - first_test}"
    )

    maes: dict[str, dict[str, list[float]]] = {arm: {} for arm in ARMS}
    control = None
    for arm, keys in ARMS.items():
        data = (
            make_dataset(series, LOOKBACK)
            if arm == "current2"
            else make_arm(series, [bank[k] for k in keys])
        )
        actuals = invert_sales(data["y_test"], data["sales_scaler"])
        for seed in SEEDS:
            fm.SEED = seed
            preds = {
                "RNN": train_rnn(data, device),
                "GRU": train_keras("GRU", build_gru, data),
                "LSTM": train_keras("LSTM", build_lstm, data),
                "HGB": train_hgb(data),
            }
            if control is None:
                control = float(np.mean(np.abs(train_sarimax(data) - actuals)))
            for m, pr in preds.items():
                maes[arm].setdefault(m, []).append(float(np.mean(np.abs(pr - actuals))))

    for m in ["RNN", "GRU", "LSTM", "HGB"]:
        c, f = np.array(maes["current2"][m]), np.array(maes["full15"][m])
        print(
            f">> {m:<7} current2 {c.mean():7.2f}±{c.std():5.2f}   "
            f"full15 {f.mean():7.2f}±{f.std():5.2f}   "
            f"delta {f.mean() - c.mean():+8.2f}"
        )
    print(f">> SARIMAX {control:.2f}  (control: no calendar features)")


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: holiday_ab.py FAMILY STORE [STORE ...]")
    family = sys.argv[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for store in [int(s) for s in sys.argv[2:]]:
        run_series(store, family, device)


if __name__ == "__main__":
    main()
