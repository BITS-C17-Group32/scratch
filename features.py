from pathlib import Path

import numpy as np
import pandas as pd

from train import clean_train_data

REPO_ROOT = Path(__file__).resolve().parent
RAW = REPO_ROOT / "data" / "raw"
EARTHQUAKE_DATE = pd.to_datetime("2016-04-16")


def load_stores() -> pd.DataFrame:
    return pd.read_csv(
        RAW / "stores.csv",
        dtype={
            "store_nbr": np.int8,
            "city": "category",
            "state": "category",
            "type": "category",
            "cluster": np.int8,
        },
    ).rename(columns={"type": "store_type"})


def load_holidays() -> pd.DataFrame:
    return pd.read_csv(
        RAW / "holidays_events.csv",
        dtype={
            "type": "category",
            "locale": "category",
            "locale_name": "category",
            "description": "category",
            "transferred": "bool",
        },
        parse_dates=["date"],
    )


# "Lead" features. For each row, ask: does a special day land tomorrow,
# or the day after? One column per scope: national, regional, local,
# any-of-those, and events.
#
# Why ask the calendar and not the DataFrame?
# The earlier version used shift(-1) over rows, which means
# "what is the next observed row?". At the very end of the data
# there is no next row, so it produced NaN, which was filled with
# 0. That invents a fact ("no holiday ahead") that we never
# checked. Asking the calendar instead always has a correct
# answer, even in the tail of the frame.
#
# Is this leaking the future? No. Holidays and events are known
# years in advance (official calendars), so at train time and at
# forecast time these are legitimate, known-in-advance signals.
def calendar_flag(
    dates: pd.Series,
    keys: pd.Series,
    calendar: set[tuple[pd.Timestamp, str]],
) -> pd.Series:
    """Return 1 where the (date, key) pair is in the calendar, else 0."""
    pairs = pd.Series(list(zip(dates, keys, strict=True)), index=dates.index)
    return pairs.isin(calendar).astype(np.int8)


def build_features() -> pd.DataFrame:
    train = clean_train_data()

    stores = load_stores()
    holidays = load_holidays()

    # Transferred holidays are moved to a different date;
    # the original date is a normal workday.
    holidays = holidays[~holidays["transferred"]].copy()
    # or this
    # holidays = holidays[holidays["transferred"].eq(False)].copy()

    terremoto_in_desc = holidays["description"].str.contains(
        "Terremoto Manabi", case=False
    )

    events_dates = set(
        holidays[(~terremoto_in_desc) & (holidays["type"].eq("Event"))]["date"]
    )

    holidays_filter = holidays["type"].isin(
        ["Holiday", "Transfer", "Additional", "Bridge"]
    )
    holidays = holidays[holidays_filter]

    # sorting holidays
    national_holidays = holidays[holidays["locale"] == "National"].copy()
    regional_holidays = holidays[holidays["locale"] == "Regional"].copy()
    local_holidays = holidays[holidays["locale"] == "Local"].copy()

    # pandas 3 promotes categorical join keys with unequal category levels to
    # str. Share the levels between stores and each calendar side so city and
    # state stay category through the merges.
    state_levels = sorted(
        set(stores["state"].cat.categories)
        | set(regional_holidays["locale_name"].cat.categories)
    )
    stores["state"] = stores["state"].cat.set_categories(state_levels)
    regional_holidays["locale_name"] = regional_holidays[
        "locale_name"
    ].cat.set_categories(state_levels)

    city_levels = sorted(
        set(stores["city"].cat.categories)
        | set(local_holidays["locale_name"].cat.categories)
    )
    stores["city"] = stores["city"].cat.set_categories(city_levels)
    local_holidays["locale_name"] = local_holidays["locale_name"].cat.set_categories(
        city_levels
    )

    # Merge store data into train set
    df = train.merge(stores, on="store_nbr", how="left", validate="m:1")

    df["is_event"] = df["date"].isin(events_dates).astype(np.int8)

    # Map National Holidays
    national_dates = set(national_holidays["date"])
    df["is_national_holiday"] = df["date"].isin(national_dates).astype(np.int8)

    # Map Regional Holidays (matching state)
    df = df.merge(
        regional_holidays[["date", "locale_name"]].drop_duplicates(),
        left_on=["date", "state"],
        right_on=["date", "locale_name"],
        how="left",
        validate="m:1",
    )
    df["is_regional_holiday"] = df["locale_name"].notna().astype(np.int8)
    df = df.drop(columns=["locale_name"])

    #  Map Local Holidays (matching city)
    df = df.merge(
        local_holidays[["date", "locale_name"]].drop_duplicates(),
        left_on=["date", "city"],
        right_on=["date", "locale_name"],
        how="left",
        validate="m:1",
    )
    df["is_local_holiday"] = df["locale_name"].notna().astype(np.int8)
    df = df.drop(columns=["locale_name"])

    # Combine into a single binary feature
    df["is_holiday"] = (
        df["is_national_holiday"] | df["is_regional_holiday"] | df["is_local_holiday"]
    ).astype(np.int8)

    df = df.sort_values(
        by=["store_nbr", "family", "date"], kind="mergesort"
    ).reset_index(drop=True)

    tomorrow = df["date"] + pd.Timedelta(days=1)
    in_2days = df["date"] + pd.Timedelta(days=2)

    # Regional holidays are state-scoped, local holidays are city-scoped.
    # Key each calendar entry by (date, state) or (date, city).
    regional_calendar = set(
        zip(regional_holidays["date"], regional_holidays["locale_name"], strict=True)
    )
    local_calendar = set(
        zip(local_holidays["date"], local_holidays["locale_name"], strict=True)
    )

    # 1 if tomorrow (or the day after) is a holiday of that scope.
    df["is_national_holiday_tomorrow"] = tomorrow.isin(national_dates).astype(np.int8)
    df["is_national_holiday_in_2days"] = in_2days.isin(national_dates).astype(np.int8)
    df["is_regional_holiday_tomorrow"] = calendar_flag(
        tomorrow, df["state"], regional_calendar
    )
    df["is_regional_holiday_in_2days"] = calendar_flag(
        in_2days, df["state"], regional_calendar
    )
    df["is_local_holiday_tomorrow"] = calendar_flag(
        tomorrow, df["city"], local_calendar
    )
    df["is_local_holiday_in_2days"] = calendar_flag(
        in_2days, df["city"], local_calendar
    )

    # Same union shortcut as is_holiday: 1 if any scope flags the day.
    df["is_holiday_tomorrow"] = (
        df["is_national_holiday_tomorrow"]
        | df["is_regional_holiday_tomorrow"]
        | df["is_local_holiday_tomorrow"]
    ).astype(np.int8)
    df["is_holiday_in_2days"] = (
        df["is_national_holiday_in_2days"]
        | df["is_regional_holiday_in_2days"]
        | df["is_local_holiday_in_2days"]
    ).astype(np.int8)

    # Events (e.g. World Cup finals) get the same treatment. People stock
    # up ahead of these, so the 1-2 day lead genuinely carries signal.
    df["is_event_tomorrow"] = tomorrow.isin(events_dates).astype(np.int8)
    df["is_event_in_2days"] = in_2days.isin(events_dates).astype(np.int8)

    # EARTHQUAKE FEATURE

    earthquake_date = EARTHQUAKE_DATE

    # Post-quake relief-window decay.
    # Day 0 = 2016-04-16 (M 7.8, 27 km SSE of Muisne, Ecuador).
    # Linear from 1.0 on day 0 to 0.0 on day 30 (2016-05-16); identically 0 elsewhere.
    # Captures the "several weeks of elevated supermarket sales" after the quake
    # (Corporacion Favorita dataset description). Global across all stores, the
    # relief spike was a nationwide demand shock; a Manabi/Esmeraldas interaction
    # was considered and deferred.
    # This is an a-priori shape from the published post-disaster response
    # (Figueroa 2020 ICEDEG; Abe 2014 DP14-1), not an empirically fitted decay.
    # The competing binary window was a nested threshold of this one and was removed.
    # This feature is zero for the whole August 2017 forecast horizon by construction.
    days_since_eq = (df["date"] - earthquake_date).dt.days
    df["earthquake_decay"] = np.where(
        (days_since_eq >= 0) & (days_since_eq <= 30), 1 - (days_since_eq / 30), 0
    ).astype(np.float32)

    for col in [
        "is_national_holiday",
        "is_national_holiday_tomorrow",
        "is_national_holiday_in_2days",
        "is_regional_holiday",
        "is_regional_holiday_tomorrow",
        "is_regional_holiday_in_2days",
        "is_local_holiday",
        "is_local_holiday_tomorrow",
        "is_local_holiday_in_2days",
        "is_holiday",
        "is_holiday_tomorrow",
        "is_holiday_in_2days",
        "is_event",
        "is_event_tomorrow",
        "is_event_in_2days",
    ]:
        assert df[col].sum() > 0, f"{col} sum is unexpectedly zero!"
        assert df[col].isna().sum() == 0, f"{col} sum has NaN values!"
        assert df[col].isin([0, 1]).all(), f"{col} has non-binary values!"

    assert (
        df["is_holiday_tomorrow"]
        == (
            df["is_national_holiday_tomorrow"]
            | df["is_regional_holiday_tomorrow"]
            | df["is_local_holiday_tomorrow"]
        )
    ).all(), "is_holiday_tomorrow does not match the scope union!"
    assert (
        df["is_holiday_in_2days"]
        == (
            df["is_national_holiday_in_2days"]
            | df["is_regional_holiday_in_2days"]
            | df["is_local_holiday_in_2days"]
        )
    ).all(), "is_holiday_in_2days does not match the scope union!"

    assert (
        df["is_holiday"]
        == (
            df["is_national_holiday"]
            | df["is_regional_holiday"]
            | df["is_local_holiday"]
        )
    ).all(), "is_holiday does not match the scope union!"

    # Grid completeness: every (store, family) series spans every date.
    series = df.groupby(["store_nbr", "family"], observed=True).ngroups
    n_dates = df["date"].nunique()
    assert len(df) == series * n_dates, "grid is not the date x store x family product"
    assert (
        df.groupby(["store_nbr", "family"], observed=True)["date"]
        .nunique()
        .eq(n_dates)
        .all()
    ), "a (store, family) series is missing dates"

    # Calendar-shape invariants: flagged dates must match the scoped,
    # non-transferred calendar intersected with the frame's range.
    # (All regional names appear in stores.state and all local ones in stores.city;
    # a refresh breaking that fails loudly by design.)
    dmin, dmax = df["date"].min(), df["date"].max()

    def in_range(dates):
        return {d for d in dates if dmin <= d <= dmax}

    for col, dates in [
        ("is_national_holiday", set(national_holidays["date"])),
        ("is_regional_holiday", set(regional_holidays["date"])),
        ("is_local_holiday", set(local_holidays["date"])),
        ("is_event", events_dates),
    ]:
        assert set(df.loc[df[col] == 1, "date"].unique()) == in_range(dates), (
            f"{col} dates diverge from the scoped calendar"
        )

    # Scoped flags are date/store-level; they never vary within a family.
    for col in [
        "is_national_holiday",
        "is_regional_holiday",
        "is_local_holiday",
        "is_event",
    ]:
        assert (
            df.groupby(["date", "store_nbr"], observed=True)[col].nunique().max() == 1
        ), f"{col} varies within a (date, store) group"

    # Grid-fill rows are full-calendar closures; everything in them is a fill.
    missing = df["id"].isna()
    per_date = missing.groupby(df["date"]).sum()
    assert per_date.isin([0, series]).all(), (
        "grid gaps are not full-calendar closure days"
    )
    assert (
        df.loc[missing, "sales"].eq(0).all()
        and df.loc[missing, "onpromotion"].eq(0).all()
    ), "id=NA rows must be the sales/onpromotion fills"
    assert df["earthquake_decay"].max() == 1.0
    assert df["earthquake_decay"].min() == 0.0
    assert df["earthquake_decay"].between(0, 1).all()
    assert df.loc[df["date"] < EARTHQUAKE_DATE, "earthquake_decay"].eq(0).all()
    assert (
        df.loc[df["date"] > EARTHQUAKE_DATE + pd.Timedelta(days=30), "earthquake_decay"]
        .eq(0)
        .all()
    )
    assert df.loc[df["date"].eq(EARTHQUAKE_DATE), "earthquake_decay"].eq(1.0).all()
    assert set(df.loc[df["earthquake_decay"].gt(0), "date"].unique()) == {
        EARTHQUAKE_DATE + pd.Timedelta(days=i) for i in range(30)
    }, "earthquake_decay violates the day 0..29 positive design"

    # Dtype contract: category, uint16, float32. Regression guards.
    assert isinstance(df["city"].dtype, pd.CategoricalDtype), "city is not category"
    assert isinstance(df["state"].dtype, pd.CategoricalDtype), "state is not category"
    assert df["earthquake_decay"].dtype == np.float32, "decay is not float32"
    assert df["onpromotion"].dtype == np.uint16, "onpromotion is not uint16"

    return df


def main() -> None:
    # Single writer: main.py produces splits/frame_full.parquet, the split
    # partitions and the manifest. This entry only build-validates, so
    # `python3 features.py` stays a cheap smoke check against the data.
    df = build_features()
    feature_cols = [c for c in df.columns if c.startswith(("is_", "earthquake_"))]
    print(f"validated frame: {len(df)} rows x {len(df.columns)} columns")
    print("positive counts:", {c: int(df[c].sum()) for c in feature_cols})
    print("next: python3 main.py  (writes frame_full.parquet + splits + manifest)")


if __name__ == "__main__":
    main()
