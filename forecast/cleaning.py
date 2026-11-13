"""Cleaning: raw favorita CSVs -> one clean daily series.

Scope: locate the selected (store, family) rows in train.csv without ever
building the full frame, trim the series to its first real sale,
grid-fill the recorded closure days, and attach holiday/event flags.
Valid menu choices come from a light 2-column scan.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def available_series(data_dir: Path) -> tuple[list[int], list[str]]:
    """Valid store numbers and families, from a 2-column scan of train.csv."""
    heads = pd.read_csv(
        data_dir / "train.csv",
        usecols=["store_nbr", "family"],
        dtype={"store_nbr": np.int16, "family": "string"},
    )
    stores = sorted({int(s) for s in heads["store_nbr"].unique()})
    families = sorted({str(f) for f in heads["family"].unique()})
    return stores, families


def _calendars(data_dir: Path, city: str, state: str):
    """Per-store flag dates: this store's holiday dates, plus event dates.

    Each file row is scoped — national (whole country), regional (one
    state), local (one city) — so a store only observes the ones matching
    its own location. Transferred rows are dropped (the holiday moved to
    another date), Work Days are not holidays, and earthquake rows stay
    out of the event flag.
    """
    holidays = pd.read_csv(data_dir / "holidays_events.csv", parse_dates=["date"])
    holidays = holidays[~holidays["transferred"]]
    events = set(
        holidays[
            holidays["type"].eq("Event")
            & ~holidays["description"].str.contains("Terremoto Manabi", case=False)
        ]["date"]
    )
    holidays = holidays[
        holidays["type"].isin(["Holiday", "Transfer", "Additional", "Bridge"])
    ]
    national = holidays["locale"].eq("National")
    regional = holidays["locale"].eq("Regional") & holidays["locale_name"].eq(state)
    local = holidays["locale"].eq("Local") & holidays["locale_name"].eq(city)
    return set(holidays.loc[national | regional | local, "date"]), events


def load_series(data_dir: Path, store: int, family: str) -> pd.DataFrame:
    """Stream train.csv once, keeping only this (store, family) slice.

    The CSV is read in chunks and filtered per chunk, so peak memory is
    one chunk plus the matching rows — never the full frame. Rows before
    the series' first real sale are dropped as artifacts, then the span
    is grid-filled: missing days are the recorded Christmas closures
    (zero sales, no promos). Returns a contiguous daily frame with date,
    sales, onpromotion, is_holiday, is_event.
    """
    keep: list[pd.DataFrame] = []
    usecols = ["date", "store_nbr", "family", "sales", "onpromotion"]
    for chunk in pd.read_csv(
        data_dir / "train.csv",
        parse_dates=["date"],
        dtype={"store_nbr": np.int16, "sales": np.float32},
        usecols=usecols,
        chunksize=250_000,
    ):
        match = chunk[(chunk["store_nbr"] == store) & (chunk["family"] == family)]
        if len(match):
            keep.append(match.drop(columns=["store_nbr", "family"]))
    if not keep:
        raise ValueError(f"no rows in train.csv for store {store} / {family}")
    rows = pd.concat(keep, ignore_index=True)

    # Rows before the first real sale are artifacts: start the series there.
    first_sale = rows.loc[rows["sales"].ne(0.0), "date"].min()
    rows = rows[rows["date"] >= first_sale]
    if rows.empty:
        raise ValueError(f"store {store} / {family} records no sales in train.csv")

    store_row = pd.read_csv(data_dir / "stores.csv")
    store_row = store_row[store_row["store_nbr"].eq(store)].iloc[0]
    city, state = store_row["city"], store_row["state"]

    dates = pd.date_range(rows["date"].min(), rows["date"].max(), freq="D")
    filled = rows.set_index("date").reindex(dates)
    # Missing days are the recorded Christmas closures: zero sales, no promos.
    filled["sales"] = filled["sales"].fillna(0.0)
    filled["onpromotion"] = filled["onpromotion"].fillna(0.0)

    holidays, events = _calendars(data_dir, city, state)
    is_holiday = np.zeros(len(dates), dtype=np.float32)
    is_event = np.zeros(len(dates), dtype=np.float32)
    for i, day in enumerate(dates):
        is_holiday[i] = day in holidays
        is_event[i] = day in events

    return pd.DataFrame(
        {
            "date": dates,
            "sales": filled["sales"].to_numpy(dtype=np.float32),
            "onpromotion": filled["onpromotion"].to_numpy(dtype=np.float32),
            "is_holiday": is_holiday,
            "is_event": is_event,
        }
    )
