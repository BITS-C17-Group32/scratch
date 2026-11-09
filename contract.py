"""Dataset split loaders, truth filtering masks, and RMSLE evaluation scorer."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa  # pyright: ignore[reportMissingTypeStubs]

__all__ = [
    "PINNED_DTYPES",
    "ContractError",
    "closure_rows",
    "fit_rows",
    "load_frame",
    "load_split",
    "preopening_rows",
    "training_mask",
]

REPO_ROOT = Path(__file__).resolve().parent
SPLITS_DIR = REPO_ROOT / "splits"
SPLIT_NAMES = ("train", "valid", "local_test")

FRAME_ROWS = 2405700

PINNED_DTYPES: dict[str, str] = {
    "date": "datetime64[us]",
    "store_nbr": "int8",
    "family": "category",
    "id": "UInt32",
    "sales": "float32",
    "onpromotion": "uint16",
    "city": "category",
    "state": "category",
    "store_type": "category",
    "cluster": "int8",
    "is_event": "int8",
    "is_national_holiday": "int8",
    "is_regional_holiday": "int8",
    "is_local_holiday": "int8",
    "is_holiday": "int8",
    "is_national_holiday_tomorrow": "int8",
    "is_national_holiday_in_2days": "int8",
    "is_regional_holiday_tomorrow": "int8",
    "is_regional_holiday_in_2days": "int8",
    "is_local_holiday_tomorrow": "int8",
    "is_local_holiday_in_2days": "int8",
    "is_holiday_tomorrow": "int8",
    "is_holiday_in_2days": "int8",
    "is_event_tomorrow": "int8",
    "is_event_in_2days": "int8",
    "earthquake_decay": "float32",
}

MANIFEST_KEYS = (
    "bounds",
    "columns",
    "created",
    "naming_rule",
    "rounding_choice",
    "rows",
    "shares",
    "source",
    "total",
    "why",
)

STORE_52_OPEN = pd.Timestamp("2017-04-20")

DELAYS_OPEN_UNTIL: dict[int, pd.Timestamp] = {
    20: pd.Timestamp("2015-02-13"),
    21: pd.Timestamp("2015-07-24"),
    22: pd.Timestamp("2015-10-09"),
    29: pd.Timestamp("2015-03-20"),
    36: pd.Timestamp("2013-05-09"),
    42: pd.Timestamp("2015-08-21"),
    52: STORE_52_OPEN,
    53: pd.Timestamp("2014-05-29"),
}

CHRISTMAS_DATES = (
    pd.Timestamp("2013-12-25"),
    pd.Timestamp("2014-12-25"),
    pd.Timestamp("2015-12-25"),
    pd.Timestamp("2016-12-25"),
)

RUN_MAIN_POINTER = "run 'uv run python main.py' to rebuild splits/"


class ContractError(ValueError):
    """Raised when data or arguments violate contract invariants."""


def _load_manifest() -> dict[str, Any]:
    """Read and validate splits/split_manifest.json."""
    path = SPLITS_DIR / "split_manifest.json"
    if not path.exists():
        raise ContractError(f"splits/split_manifest.json missing; {RUN_MAIN_POINTER}")
    try:
        with path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as error:
        raise ContractError(
            f"splits/split_manifest.json unreadable: {error}; {RUN_MAIN_POINTER}"
        ) from error
    if not isinstance(manifest, dict):
        raise ContractError(
            f"splits/split_manifest.json is not a JSON object; {RUN_MAIN_POINTER}"
        )
    missing = [key for key in MANIFEST_KEYS if key not in manifest]
    if missing:
        raise ContractError(
            f"splits/split_manifest.json is missing keys {missing}; {RUN_MAIN_POINTER}"
        )
    for key in ("rows", "bounds", "why"):
        if not isinstance(manifest[key], dict):
            raise ContractError(
                f"manifest field {key!r} must be an object; {RUN_MAIN_POINTER}"
            )
    if not isinstance(manifest["columns"], list):
        raise ContractError(
            f"manifest field 'columns' must be a list; {RUN_MAIN_POINTER}"
        )
    return manifest


def load_split(split: str) -> pd.DataFrame:
    """Load and validate a split parquet file against the manifest schema."""
    if split not in SPLIT_NAMES:
        raise ContractError(f"unknown split {split!r}; expected one of {SPLIT_NAMES}")
    if split == "local_test" and os.environ.get("FREEZE_RUN") != "1":
        raise ContractError(
            "local_test is frozen: set FREEZE_RUN=1 and run through freeze path"
        )
    manifest = _load_manifest()
    path = SPLITS_DIR / f"{split}.parquet"
    if not path.exists():
        raise ContractError(f"{path.name} missing; {RUN_MAIN_POINTER}")
    try:
        data = pd.read_parquet(path)
    except (ValueError, OSError, pa.ArrowException) as error:
        raise ContractError(
            f"{path.name} unreadable: {error}; {RUN_MAIN_POINTER}"
        ) from error
    for col, expected_dtype in PINNED_DTYPES.items():
        if col in data.columns and str(data[col].dtype) != expected_dtype:
            raise ContractError(
                f"{split}: column {col!r} has dtype {data[col].dtype}, "
                f"expected {expected_dtype}; {RUN_MAIN_POINTER}"
            )
    if list(data.columns) != list(manifest["columns"]):
        raise ContractError(
            f"{split}: columns disagree with manifest; {RUN_MAIN_POINTER}"
        )
    try:
        expected_rows = int(manifest["rows"][split])
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError(
            f"manifest rows[{split!r}] invalid; {RUN_MAIN_POINTER}"
        ) from error
    if len(data) != expected_rows:
        raise ContractError(
            f"{split}: row count {len(data)} != manifest {expected_rows}; "
            f"{RUN_MAIN_POINTER}"
        )
    start_str, end_str = manifest["bounds"][split]
    start_ts = pd.Timestamp(start_str)
    end_ts = pd.Timestamp(end_str)
    actual_start = data["date"].min()
    actual_end = data["date"].max()
    if actual_start != start_ts or actual_end != end_ts:
        raise ContractError(
            f"{split}: date bounds ({actual_start.date()}, {actual_end.date()}) "
            f"!= manifest ({start_ts.date()}, {end_ts.date()}); {RUN_MAIN_POINTER}"
        )
    if data["sales"].isna().any():
        raise ContractError(f"{split}: sales contains NaN; {RUN_MAIN_POINTER}")
    return data


def load_frame(
    train: pd.DataFrame | None = None,
    valid: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Concatenate train and valid splits ordered by date."""
    if train is None:
        train = load_split("train")
    if valid is None:
        valid = load_split("valid")
    mismatched = {
        col: (str(train[col].dtype), str(valid[col].dtype))
        for col in train.columns
        if train[col].dtype != valid[col].dtype
    }
    if mismatched:
        raise ContractError(
            f"load_frame: train/valid dtype drift: {mismatched}; {RUN_MAIN_POINTER}"
        )
    data = pd.concat([train, valid], ignore_index=True)
    data = data.sort_values("date", kind="mergesort").reset_index(drop=True)
    if len(data) != FRAME_ROWS:
        raise ContractError(
            f"load_frame: {len(data)} rows instead of pinned {FRAME_ROWS}; "
            f"{RUN_MAIN_POINTER}"
        )
    if not data["date"].is_monotonic_increasing:
        raise ContractError(
            f"load_frame: concatenation is not date-ordered; {RUN_MAIN_POINTER}"
        )
    return data


def closure_rows(df: pd.DataFrame) -> pd.Series:
    """Return boolean mask for holiday closure fill rows (id is NaN)."""
    if "id" not in df.columns:
        raise ContractError("closure_rows: input dataframe missing 'id' column")
    return df["id"].isna()


def preopening_rows(df: pd.DataFrame) -> pd.Series:
    """Return boolean mask for dates before a store officially opened."""
    for col in ("store_nbr", "date"):
        if col not in df.columns:
            raise ContractError(f"preopening_rows: missing {col!r} column")
    mask = pd.Series(False, index=df.index)
    for store, open_date in DELAYS_OPEN_UNTIL.items():
        store_match = df["store_nbr"] == store
        before_open = df["date"] < open_date
        mask |= store_match & before_open
    return mask


def fit_rows(df: pd.DataFrame) -> pd.Series:
    """Return boolean mask of rows to include in model training."""
    is_preopen = preopening_rows(df)
    is_closure = closure_rows(df)
    return (~is_preopen) | is_closure


training_mask = fit_rows
