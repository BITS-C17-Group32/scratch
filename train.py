import numpy as np
import pandas as pd


def clean_train_data():

    train_df = pd.read_csv(
        "data/raw/train.csv",
        parse_dates=["date"],
        dtype={
            "store_nbr": np.int8,
            "family": "category",
            "sales": np.float32,
            "onpromotion": np.uint16,
        },
        usecols=["date", "store_nbr", "family", "sales", "onpromotion"],
    )

    # Identify unique stores, families, and complete date range
    unique_stores = train_df["store_nbr"].unique()
    unique_families = train_df["family"].unique()
    full_dates = pd.date_range(start=train_df["date"].min(), end=train_df["date"].max(), freq="D")

    # Create a complete Cartesian grid of (date x store_nbr x family)
    multi_index = pd.MultiIndex.from_product(
        [full_dates, unique_stores, unique_families], names=["date", "store_nbr", "family"]
    )
    grid = pd.DataFrame(index=multi_index).reset_index()

    # Merge original train data into the complete grid
    contiguous_train = grid.merge(train_df, on=["date", "store_nbr", "family"], how="left")

    # Fill missing values for closed days (e.g., Christmas Day)
    contiguous_train["sales"] = contiguous_train["sales"].fillna(0.0)
    contiguous_train["onpromotion"] = contiguous_train["onpromotion"].fillna(0).astype(int)

    # Sort chronologically to preserve time sequence integrity
    contiguous_train = contiguous_train.sort_values(["store_nbr", "family", "date"]).reset_index(
        drop=True
    )
    return contiguous_train
