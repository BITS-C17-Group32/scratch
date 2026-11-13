"""Extract and clean raw Favorita CSV data into one daily series.

Locate the selected store and family rows in train.csv without building
the full DataFrame, trim the series to its first real sale, fill recorded
closure days, and attach holiday and event flags. Prompt setup reads only
the store_nbr and family columns, so the scan never builds the full DataFrame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def available_series(data_dir: Path) -> tuple[list[int], list[str]]:
    """Return valid store numbers and families from a two-column scan of train.csv."""
    records = pd.read_csv(
        data_dir / "train.csv",
        usecols=["store_nbr", "family"],
        dtype={"store_nbr": np.int16, "family": "string"},
    )
    stores = sorted({int(s) for s in records["store_nbr"].unique()})
    families = sorted({str(f) for f in records["family"].unique()})
    return stores, families


def _calendars(
    data_dir: Path, city: str, state: str
) -> tuple[set[pd.Timestamp], set[pd.Timestamp]]:
    """Return the holiday and event dates a store observes.

    The locale column scopes the holidays: National rows apply to every store,
    Regional rows to the store's state, and Local rows to its city. Event rows
    ignore locale, so every store sees the same events. This function drops
    transferred rows because each holiday appears on its new date. Rows typed
    Work Day are not holidays, and earthquake rows stay out of the event flag.
    """
    holidays_raw = pd.read_csv(data_dir / "holidays_events.csv", parse_dates=["date"])
    active_rows = holidays_raw[~holidays_raw["transferred"]]

    events = set(
        active_rows[
            active_rows["type"].eq("Event")
            & ~active_rows["description"].str.contains("Terremoto Manabi", case=False)
        ]["date"]
    )

    holiday_mask = active_rows["type"].isin(
        ["Holiday", "Transfer", "Additional", "Bridge"]
    )
    valid_holidays = active_rows[holiday_mask]

    locales = valid_holidays["locale"]
    locale_names = valid_holidays["locale_name"]
    is_national = locales.eq("National")
    is_regional = locales.eq("Regional") & locale_names.eq(state)
    is_local = locales.eq("Local") & locale_names.eq(city)

    store_holidays = set(
        valid_holidays.loc[is_national | is_regional | is_local, "date"]
    )
    return store_holidays, events


def store_calendars(
    data_dir: Path, store: int
) -> tuple[set[pd.Timestamp], set[pd.Timestamp]]:
    """Return the holiday and event dates a store observes.

    Reads the store's city and state from stores.csv and applies the locale
    scoping in _calendars. The sets are built from the full calendar file,
    not just the recorded train.csv span, and the holiday and event sets end
    on different dates, so each must be checked for coverage on its own.
    The holiday set is scoped to the store's locale while the event set is
    global.
    """
    stores_table = pd.read_csv(data_dir / "stores.csv")
    store_entry = stores_table[stores_table["store_nbr"].eq(store)].iloc[0]
    city, state = str(store_entry["city"]), str(store_entry["state"])
    return _calendars(data_dir, city, state)


def load_series(data_dir: Path, store: int, family: str) -> pd.DataFrame:
    """Stream train.csv once and keep only the selected store and family slice.

    Chunked reading caps peak memory at one chunk plus the matched rows, never
    the full frame. Rows before the first real sale are recording artifacts.
    Missing days are the recorded Christmas closures, so the grid-filled span
    carries zero sales and zero promotions. Returns a contiguous daily frame
    with date, sales, onpromotion, is_holiday, and is_event.
    """
    # The explicit element type tells basedpyright what to verify. Without it,
    # basedpyright infers an empty list as list[Unknown], so the checker cannot
    # confirm what pd.concat receives. The annotation makes it verify every
    # appended chunk is a DataFrame.
    # Ref: https://docs.python.org/3/library/typing.html
    keep: list[pd.DataFrame] = []
    columns_to_load = ["date", "store_nbr", "family", "sales", "onpromotion"]
    # chunksize makes read_csv yield 250,000-row chunks.
    # Ref: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
    for chunk in pd.read_csv(
        data_dir / "train.csv",
        parse_dates=["date"],
        dtype={"store_nbr": np.int16, "sales": np.float32},
        usecols=columns_to_load,
        chunksize=250_000,
    ):
        matched = chunk[(chunk["store_nbr"] == store) & (chunk["family"] == family)]
        if len(matched):
            keep.append(matched.drop(columns=["store_nbr", "family"]))
    if not keep:
        raise ValueError(f"no rows in train.csv for store {store} / {family}")
    series_rows = pd.concat(keep, ignore_index=True)

    # The first day with a non-zero sale marks the real start of the series.
    first_sale = series_rows.loc[series_rows["sales"].ne(0.0), "date"].min()
    series_rows = series_rows[series_rows["date"] >= first_sale]
    if series_rows.empty:
        raise ValueError(f"store {store} / {family} records no sales in train.csv")

    dates = pd.date_range(
        series_rows["date"].min(), series_rows["date"].max(), freq="D"
    )
    filled = series_rows.set_index("date").reindex(dates)
    # The source omits closure days instead of writing zero rows, so reindex
    # leaves NaN here. Fill both columns with zero.
    filled["sales"] = filled["sales"].fillna(0.0)
    filled["onpromotion"] = filled["onpromotion"].fillna(0.0)

    holidays, events = store_calendars(data_dir, store)
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
