import numpy as np
import pandas as pd

from train import clean_train_data

train = clean_train_data()
stores = pd.read_csv("data/raw/stores.csv")
holidays = pd.read_csv(
    "data/raw/holidays_events.csv",
    dtype={
        "type": "category",
        "locale": "category",
        "locale_name": "category",
        "description": "category",
        "transferred": np.bool,
    },
    parse_dates=["date"],
)

# print(holidays["type"].unique())

# Transferred holidays are moved to a different date; the original date is a normal workday.
holidays = holidays[~holidays["transferred"]].copy()
# or this
# holidays = holidays[holidays["transferred"].eq(False)].copy()

events_dates = set(holidays[holidays["type"].eq("Event")]["date"])

holidays_filter = holidays["type"].isin(["Holiday", "Transfer", "Additional", "Bridge"])
holidays = holidays[holidays_filter]

# sorting holidays
national_holidays = holidays[holidays["locale"] == "National"].copy()
regional_holidays = holidays[holidays["locale"] == "Regional"].copy()
local_holidays = holidays[holidays["locale"] == "Local"].copy()

# Merge store data into train set
df = train.merge(stores, on="store_nbr", how="left", validate="m:1")
print(df.columns)

df["is_event"] = df["date"].isin(events_dates).astype(int)

# Map National Holidays
national_dates = set(national_holidays["date"])
df["is_national_holiday"] = df["date"].isin(national_dates).astype(int)

# Map Regional Holidays (matching state)
df = df.merge(
    regional_holidays[["date", "locale_name"]].drop_duplicates(),
    left_on=["date", "state"],
    right_on=["date", "locale_name"],
    how="left",
    validate="m:1",
)
df["is_regional_holiday"] = df["locale_name"].notna().astype(int)
df = df.drop(columns=["locale_name"])

#  Map Local Holidays (matching city)
df = df.merge(
    local_holidays[["date", "locale_name"]].drop_duplicates(),
    left_on=["date", "city"],
    right_on=["date", "locale_name"],
    how="left",
    validate="m:1",
)
df["is_local_holiday"] = df["locale_name"].notna().astype(int)
df = df.drop(columns=["locale_name"])

# Combine into a single binary feature
df["is_holiday"] = (
    df["is_national_holiday"] | df["is_regional_holiday"] | df["is_local_holiday"]
).astype(int)

print(df.columns)

df = df.sort_values(by=["store_nbr", "family", "date"], kind="mergesort").reset_index(
    drop=True
)

# holiday lead feature (National + Regional + Local + (any) Holiday)
for col in ["national_holiday", "regional_holiday", "local_holiday", "holiday"]:
    df[f"is_{col}_tomorrow"] = (
        df.groupby(["store_nbr", "family"])[f"is_{col}"].shift(-1).fillna(0).astype(int)
    )
    df[f"is_{col}_in_2days"] = (
        df.groupby(["store_nbr", "family"])[f"is_{col}"].shift(-2).fillna(0).astype(int)
    )

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
# The competing binary window (F6) was a nested threshold of this one and was removed.
# This feature is zero for the whole August 2017 forecast horizon by construction.
days_since_eq = (df["date"] - earthquake_date).dt.days
df["earthquake_decay"] = np.where(
    (days_since_eq >= 0) & (days_since_eq <= 30), 1 - (days_since_eq / 30), 0
)

print(df.columns)

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
]:
    assert df[col].sum() > 0, f"{col} sum is unexpectedly zero!"
    assert df[col].isna().sum() == 0, f"{col} sum has NaN values!"
    assert df[col].isin([0, 1]).all(), f"{col} has non-binary values!"

assert (
    df["is_holiday"]
    == (df["is_national_holiday"] | df["is_regional_holiday"] | df["is_local_holiday"])
).all(), "is_holiday does not match the scope union!"

assert df["earthquake_decay"].max() == 1.0
assert df["earthquake_decay"].min() == 0.0
assert df["earthquake_decay"].between(0, 1).all()
assert df.loc[df["date"] < pd.Timestamp("2016-04-16"), "earthquake_decay"].eq(0).all()
assert df.loc[df["date"] > pd.Timestamp("2016-05-16"), "earthquake_decay"].eq(0).all()
assert (
    df.loc[df["date"].eq(pd.Timestamp("2016-04-16")), "earthquake_decay"].eq(1.0).all()
)
active = df["earthquake_decay"].gt(0).groupby(df["date"]).any()
assert int(active.sum()) == 30
