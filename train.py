from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def clean_train_data():

    train_df = pd.read_csv(
        ROOT / "data" / "raw" / "train.csv",
        parse_dates=["date"],
        dtype={
            "id": "UInt32",
            "store_nbr": np.int8,
            "family": "category",
            "sales": np.float32,
            "onpromotion": np.uint32,
        },
    )

    # Identify unique stores, families, and complete date range
    unique_stores = train_df["store_nbr"].unique()
    unique_families = train_df["family"].unique()
    full_dates = pd.date_range(
        start=train_df["date"].min(), end=train_df["date"].max(), freq="D"
    )

    # Create a complete Cartesian grid of (date x store_nbr x family)
    multi_index = pd.MultiIndex.from_product(
        [full_dates, unique_stores, unique_families],
        names=["date", "store_nbr", "family"],
    )
    grid = pd.DataFrame(index=multi_index).reset_index()

    # Merge original train data into the complete grid
    contiguous_train = grid.merge(
        train_df, on=["date", "store_nbr", "family"], how="left", validate="m:1"
    )

    # These fill rows are the four Christmas dates the warehouse has no
    # rows for because Favorita closed nationally: 2013-12-25, 2014-12-25,
    # 2015-12-25, 2016-12-25. The zeros below are recorded closure, not
    # measured sales. They are kept as true labels. id is NaN on exactly
    # the 7,128 of these rows (1,782 store-family pairs x 4 dates), so
    # df["id"].isna() identifies them.
    contiguous_train["sales"] = contiguous_train["sales"].fillna(0.0)
    contiguous_train["onpromotion"] = (
        contiguous_train["onpromotion"].fillna(0).astype(np.uint16)
    )

    # Sort chronologically to preserve time sequence integrity
    contiguous_train = contiguous_train.sort_values(
        ["store_nbr", "family", "date"]
    ).reset_index(drop=True)
    return contiguous_train
