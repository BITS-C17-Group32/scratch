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
summed = agg_by_dates["sales"].sum()
# print(summed.idxmax())
# print(summed.head())

print(summed.head(1347))
print(summed.tail(337))

mid_march_2017 = pd.Timestamp("2017-03-15")
mid_april_2017 = pd.Timestamp("2017-04-15")

# mask = (summed.index >= mid_march_2017) & (summed.index <= mid_april_2017)
# print(summed[mask])

onprom_sum = agg_by_dates["onpromotion"].sum()
# mask = onprom_sum.index == pd.Timestamp("2017-04-01")

# store = df["store_nbr"] == 33
# print(df[store].describe())


# print(df["sales"].describe(percentiles=[0.1, 0.5, 0.9, 0.95, 0.99]))

# max_sales = df["sales"].max()
# print(max_sales)

# mask = df["sales"] == max_sales
# print(df[mask])

# x = df["sales"].nsmallest(n=10)
# mask = df["sales"].isin(x)
# print(df[mask])

TRAIN_END = pd.Timestamp("2016-03-27")  # train     = [ ...            , TRAIN_END ]
VALID_END = pd.Timestamp("2016-09-11")  # valid     = ( TRAIN_END      , VALID_END ]
# local test = ( VALID_END     , ... ]

train_mask = df["date"] <= TRAIN_END
valid_mask = (df["date"] > TRAIN_END) & (df["date"] <= VALID_END)
local_test_mask = df["date"] > VALID_END

predicted_share = {"train": 0.70, "valid": 0.10, "local_test": 0.20}

for name, mask in [
    ("train", train_mask),
    ("valid", valid_mask),
    ("local_test", local_test_mask),
]:
    block = df.loc[mask]
    share = len(block) / len(df)
    print(
        f"{name:<10}",
        "rows:",
        len(block),
        "share:",
        f"{share:.1%}",
        "dates:",
        block["date"].min().date(),
        "..",
        block["date"].max().date(),
    )

assert (train_mask.astype(int) + valid_mask.astype(int) + local_test_mask.astype(int)).eq(1).all()
