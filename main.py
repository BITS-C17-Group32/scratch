import numpy as np
import pandas as pd

# df_before = pd.read_csv("train.csv")
# print("No Types:")
# print(df_before.shape)
# print(df_before.info(memory_usage="deep"))
# print(df_before.memory_usage())

df = pd.read_csv(
    "train.csv",
    parse_dates=["date"],
    dtype={
        "store_nbr": np.int8,
        "family": "category",
        "sales": np.float32,
        "onpromotion": np.uint16,
    },
    usecols=[1, 2, 3, 4, 5],
)
# print("The dtyped One:")
# print(df.shape)
# print(df.info(memory_usage="deep"))
# print(df.memory_usage())
# print(df["family"].nunique())
# print(df["store_nbr"].nunique())
# print(df["date"].nunique())

start_date = pd.Timestamp(df["date"].min())
end_date = pd.Timestamp(df["date"].max())
span: pd.Timedelta = end_date - start_date
# print(span)

grouped_family = df.groupby(["family", "store_nbr"])
# print(grouped_family.size())
# print(grouped_family.size().describe())
# print(grouped_family.ngroup())
# print(grouped_family["date"].nunique().describe())

groupedAgain = df.groupby(["date", "store_nbr"])
# print(groupedAgain.size().describe())

dates_from_file = pd.DatetimeIndex(df["date"])
full_calendar_grid = pd.date_range(start_date, end_date)
# print(full_calendar_grid.difference(dates_from_file))
# pair_mins = grouped_family.min()
# print(pair_mins)
# print(pair_mins.describe())
# print(pair_mins.nunique())

# print(df["sales"].max())

agg_by_dates = df.groupby(["date"])
print(agg_by_dates["sales"].sum())
