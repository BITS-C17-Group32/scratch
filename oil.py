import numpy as np
import pandas as pd

oil_df = pd.read_csv("data/raw/oil.csv", parse_dates=["date"], dtype={"dcoilwtico": np.float32})

# 2013-01-01 00:00:00
# 2017-08-31 00:00:00
start_date = oil_df["date"].min()
end_date = oil_df["date"].max()

print(oil_df.head())
# Create a continuous date range covering the entire sequence
full_date_range = pd.date_range(start=start_date, end=end_date, freq="D")
oil_full = pd.DataFrame({"date": full_date_range})

# Merge and fill missing oil price values
oil_df = oil_full.merge(oil_df, on="date", how="left")

# check for consecutive gaps in the 'dcoilwtico' column.
"""
is_nan = oil_df['dcoilwtico'].isna()
group_id = (~is_nan).cumsum()[is_nan]
streak_counts = group_id.groupby(group_id).size()
large_streaks = streak_counts[streak_counts > 4]
print(f"Total gaps > {4} consecutive days: {len(large_streaks)}")"""
# with the above code, we can check for consecutive gaps in the 'dcoilwtico' column. The output will show the total number of gaps greater than [2,3,4] consecutive days.
"""
Total gaps > 2 consecutive days: 32
Total gaps > 3 consecutive days: 1
Total gaps > 4 consecutive days: 0
"""
# since we dont have any gaps greater than 4 days, we can use forward fill and backward fill to fill the missing values.
# Forward Fill (.ffill()) is actually preferred over linear interpolation for daily commodity prices (like crude oil, stocks, or foreign exchange rates)
oil_df["dcoilwtico"] = oil_df["dcoilwtico"].ffill().bfill()
print(oil_df.head())
