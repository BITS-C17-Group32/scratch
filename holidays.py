import pandas as pd

df = pd.read_csv(
    "holidays_events.csv",
    parse_dates=["date"],
    dtype={
        "type": "category",
    },
)

dates = [
    pd.Timestamp("2013-12-25"),
    pd.Timestamp("2014-12-25"),
    pd.Timestamp("2015-12-25"),
    pd.Timestamp("2016-01-01"),
    pd.Timestamp("2016-01-03"),
    pd.Timestamp("2016-12-25"),
]


dates_bool = df["date"].isin(dates)
mask = dates_bool.isin([bool("True")])
print(df[mask])
