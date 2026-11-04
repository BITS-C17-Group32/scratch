import numpy as np
import pandas as pd
from train import clean_train_data

train = clean_train_data()
stores = pd.read_csv("data/raw/stores.csv")
holidays = pd.read_csv("data/raw/holidays_events.csv", parse_dates=["date"])

# Transferred holidays are moved to a different date; the original date is a normal workday.
holidays = holidays[holidays["transferred"] == False].copy()



#sorting holidays 
national_holidays = holidays[holidays["type"] == "National"].copy()
regional_holidays = holidays[holidays["type"] == "Regional"].copy()
local_holidays = holidays[holidays["type"] == "Local"].copy()

# Merge store data into train set
df = train.merge(stores, on="store_nbr", how="left")
print(df.columns)

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




