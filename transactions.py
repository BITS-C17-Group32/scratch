import numpy as np
import pandas as pd

df = pd.read_csv(
    "transactions.csv",
    parse_dates=["date"],
    dtype={
        "store_nbr": np.int8,
        "transactions": np.float32,
    },
)

print(df.shape)
print(df.info(memory_usage="deep"))
print(df.memory_usage())
print(df["store_nbr"].nunique())
print(df["date"].nunique())

start_date = pd.Timestamp(df["date"].min())
end_date = pd.Timestamp(df["date"].max())
span: pd.Timedelta = end_date - start_date
print(span)

grouped_date_store = df.groupby(["date", "store_nbr"])
print(grouped_date_store.size().describe())

dates_from_file = pd.DatetimeIndex(df["date"])
full_calendar_grid = pd.date_range(start_date, end_date)
print(full_calendar_grid.difference(dates_from_file))
