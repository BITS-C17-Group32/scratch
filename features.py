from pathlib import Path

import numpy as np
import pandas as pd

from train import clean_train_data

REPO_ROOT = Path(__file__).resolve().parent
RAW = REPO_ROOT / "data" / "raw"
OUT_FRAME = REPO_ROOT / "splits" / "frame_features.parquet"


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

    earthquake_date = pd.to_datetime("2016-04-16")

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
    )

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
    assert df["id"].isna().sum() == 7128, (
        "expected exactly 7,128 fill rows from the four Christmas dates"
    )

    assert df["earthquake_decay"].max() == 1.0
    assert df["earthquake_decay"].min() == 0.0
    assert df["earthquake_decay"].between(0, 1).all()
    assert (
        df.loc[df["date"] < pd.Timestamp("2016-04-16"), "earthquake_decay"].eq(0).all()
    )
    assert (
        df.loc[df["date"] > pd.Timestamp("2016-05-16"), "earthquake_decay"].eq(0).all()
    )
    assert (
        df.loc[df["date"].eq(pd.Timestamp("2016-04-16")), "earthquake_decay"]
        .eq(1.0)
        .all()
    )
    active = df["earthquake_decay"].gt(0).groupby(df["date"]).any()
    assert int(active.sum()) == 30

    assert df["is_event"].sum() == 44550
    assert df["is_event"].ne(0).groupby(df["date"]).any().sum() == 25

    return df


def main() -> None:
    df = build_features()
    df.to_parquet(OUT_FRAME, index=False)


if __name__ == "__main__":
    main()
