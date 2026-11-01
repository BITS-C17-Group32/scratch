import pandas as pd

df = pd.read_csv(
    "holidays_events.csv",
    parse_dates=["date"],
    dtype={
        "type": "category",
    },
)

# dates = [
#     pd.Timestamp("2013-12-25"),
#     pd.Timestamp("2014-12-25"),
#     pd.Timestamp("2015-12-25"),
#     pd.Timestamp("2016-01-01"),
#     pd.Timestamp("2016-01-03"),
#     pd.Timestamp("2016-12-25"),
# ]

dates = [
    pd.Timestamp("2013-11-12"),
    pd.Timestamp("2016-10-07"),
    pd.Timestamp("2017-04-01"),
    pd.Timestamp("2016-04-16"),
]

dates_bool = df["date"].isin(dates)
mask = dates_bool.isin([bool("True")])
print(df[mask])
