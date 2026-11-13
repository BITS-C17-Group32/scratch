"""Extract and clean raw Favorita CSV data into one daily series.

Locate the selected store and family rows in train.csv without building
the full DataFrame, trim the series to its first real sale, fill recorded
closure days, and attach the 15 calendar flag columns. Prompt setup reads only
the store_nbr and family columns, so the scan never builds the full DataFrame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# The 15 calendar channels: five base flags (scoped national, regional, and
# local holidays, the merged holiday flag, and events) plus each one's
# one- and two-day leads.
CALENDAR_FLAG_COLS = (
    "is_national_holiday",
    "is_national_holiday_tomorrow",
    "is_national_holiday_in_2days",
    "is_regional_holiday",
    "is_regional_holiday_tomorrow",
    "is_regional_holiday_in_2days",
    "is_local_holiday",
    "is_local_holiday_tomorrow",
    "is_local_holiday_in_2days",
    "is_event",
    "is_event_tomorrow",
    "is_event_in_2days",
    "is_holiday",
    "is_holiday_tomorrow",
    "is_holiday_in_2days",
)


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
) -> tuple[
    set[pd.Timestamp],
    set[pd.Timestamp],
    set[pd.Timestamp],
    set[pd.Timestamp],
]:
    """Return the national, regional, local, and event dates a store observes.

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
    national = set(valid_holidays.loc[locales.eq("National"), "date"])
    regional = set(
        valid_holidays.loc[locales.eq("Regional") & locale_names.eq(state), "date"]
    )
    local = set(valid_holidays.loc[locales.eq("Local") & locale_names.eq(city), "date"])
    return national, regional, local, events


def store_calendars(
    data_dir: Path, store: int
) -> tuple[
    set[pd.Timestamp],
    set[pd.Timestamp],
    set[pd.Timestamp],
    set[pd.Timestamp],
]:
    """Return the national, regional, local, and event dates a store observes.

    Reads the store's city and state from stores.csv and applies the locale
    scoping in _calendars. The sets are built from the full calendar file,
    not just the recorded train.csv span. The scoped sets and the event set
    end on different dates, so each must be checked for coverage on its own.
    """
    stores_table = pd.read_csv(data_dir / "stores.csv")
    store_entry = stores_table[stores_table["store_nbr"].eq(store)].iloc[0]
    city, state = str(store_entry["city"]), str(store_entry["state"])
    return _calendars(data_dir, city, state)


def flags_frame(
    national: set[pd.Timestamp],
    regional: set[pd.Timestamp],
    local: set[pd.Timestamp],
    events: set[pd.Timestamp],
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build the CALENDAR_FLAG_COLS columns for dates, with two-day leads.

    Each base flag marks its calendar on the row's own day; the _tomorrow and
    _in_2days columns mark one and two days ahead. The lead columns need the
    two days after the span, so the base arrays extend two days past dates
    and the returned frame keeps len(dates) rows. Holiday calendars are
    published in advance, so those lead flags are genuinely known today.
    """
    # The last two span rows sit past dates, so every lead column stays real.
    last = dates[-1]
    span = dates.union(
        pd.DatetimeIndex([last + pd.Timedelta(days=1), last + pd.Timedelta(days=2)])
    )
    arrays = {
        "is_national_holiday": np.array(
            [day in national for day in span], dtype=np.float32
        ),
        "is_regional_holiday": np.array(
            [day in regional for day in span], dtype=np.float32
        ),
        "is_local_holiday": np.array([day in local for day in span], dtype=np.float32),
        "is_event": np.array([day in events for day in span], dtype=np.float32),
    }
    arrays["is_holiday"] = np.logical_or(
        arrays["is_national_holiday"],
        np.logical_or(arrays["is_regional_holiday"], arrays["is_local_holiday"]),
    ).astype(np.float32)

    columns: dict[str, np.ndarray] = {}
    for name in (
        "is_national_holiday",
        "is_regional_holiday",
        "is_local_holiday",
        "is_event",
        "is_holiday",
    ):
        arr = arrays[name]
        columns[name] = arr[: len(dates)]
        columns[name + "_tomorrow"] = arr[1 : len(dates) + 1]
        columns[name + "_in_2days"] = arr[2 : len(dates) + 2]
    return pd.DataFrame(columns)


def load_series(data_dir: Path, store: int, family: str) -> pd.DataFrame:
    """Stream train.csv once and keep only the selected store and family slice.

    Chunked reading caps peak memory at one chunk plus the matched rows, never
    the full frame. Rows before the first real sale are recording artifacts.
    Missing days are the recorded Christmas closures, so the grid-filled span
    carries zero sales and zero promotions. Returns a contiguous daily frame
    with date, sales, onpromotion, and the 15 calendar flag columns from
    flags_frame, so downstream arms can pick the base pair or the full block.
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

    national, regional, local, events = store_calendars(data_dir, store)
    flags = flags_frame(national, regional, local, events, dates)

    return pd.DataFrame(
        {
            "date": dates,
            "sales": filled["sales"].to_numpy(dtype=np.float32),
            "onpromotion": filled["onpromotion"].to_numpy(dtype=np.float32),
            **{col: flags[col].to_numpy() for col in CALENDAR_FLAG_COLS},
        }
    )
