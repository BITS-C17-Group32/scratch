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
df = train.merge(stores, on="store_nbr", how="left")
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
)
df["is_regional_holiday"] = df["locale_name"].notna().astype(int)
df = df.drop(columns=["locale_name"])

#  Map Local Holidays (matching city)
df = df.merge(
    local_holidays[["date", "locale_name"]].drop_duplicates(),
    left_on=["date", "city"],
    right_on=["date", "locale_name"],
    how="left",
)
df["is_local_holiday"] = df["locale_name"].notna().astype(int)
df = df.drop(columns=["locale_name"])

# Combine into a single binary feature
df["is_holiday"] = (
    df["is_national_holiday"] | df["is_regional_holiday"] | df["is_local_holiday"]
).astype(int)

print(df.columns)

# holiday lead feature (National + Regional + Local)
for scope in ["national", "regional", "local"]:
    df[f"is_{scope}_holiday_tomorrow"] = (
        df.groupby(["store_nbr", "family"])[f"is_{scope}_holiday"].shift(-1).fillna(0).astype(int)
    )

for scope in ["national", "regional", "local"]:
    df[f"is_{scope}_holiday_in_2days"] = (
        df.groupby(["store_nbr", "family"])[f"is_{scope}_holiday"].shift(-2).fillna(0).astype(int)
    )

# EARTHQUAKE IMPACT FEATURES

earthquake_date = pd.to_datetime("2016-04-16")
# Binary indicator for the immediate high-impact relief window (4 weeks following)
df["is_earthquake_impact_window"] = (
    (df["date"] >= earthquake_date) & (df["date"] <= earthquake_date + pd.Timedelta(days=28))
).astype(int)

# Continuous decay feature (captures fading impact intensity)
# Value starts at 1 on the earthquake day and linearly decays to 0 over 30 days
days_since_eq = (df["date"] - earthquake_date).dt.days
df["earthquake_decay"] = np.where(
    (days_since_eq >= 0) & (days_since_eq <= 30), 1 - (days_since_eq / 30), 0
)

print(df.columns)
