import pandas as pd

# Parse dates as timestamps. Storing repeated 'type' strings as categorical
# codes reduces memory usage.
# Ref: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
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

# Inspect selected holiday records, including the 2016-04-16 earthquake date.
dates = [
    pd.Timestamp("2013-11-12"),
    pd.Timestamp("2016-10-07"),
    pd.Timestamp("2017-04-01"),
    pd.Timestamp("2016-04-16"),
]

# Create a boolean Series marking rows whose date matches any in 'dates'.
# Ref: https://pandas.pydata.org/docs/reference/api/pandas.Series.isin.html
dates_bool = df["date"].isin(dates)

# Non-empty strings are truthy, so bool("True") evaluates to True; .isin([True])
# filters for rows where dates_bool is True.
mask = dates_bool.isin([bool("True")])

# Display holiday records matching the target dates mask.
print(df[mask])
