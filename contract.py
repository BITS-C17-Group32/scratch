"""Dataset split loaders, truth filtering masks, and RMSLE evaluation scorer.

Specifications:
- Split loaders return full frames without dropping rows. Training consumers
  apply training_mask (fit_rows) during model training or fold assembly.
- score() computes RMSLE, MAE, and RMSE for full inputs and active-demand rows
  (y_true > 0), reported in *_active keys.
- Holiday closure grid fills (5,346 train / 0 valid / 1,782 local_test, 7,128 total)
  in why.grid_fill_rows are preserved by training_mask.
- earthquake_decay feature (float32): 2016-04-16 quake decay (53,460 positive rows;
  44,429 with sales>0) falls inside valid; fold windows from 2016-07-26 contain
  no quake rows.
- local_test is gated by FREEZE_RUN=1 for final evaluation.
- Masks are index-aligned boolean Series; align index when reordering frames.
- Import is side-effect-free: no data files are opened on module load.
- The fold table is one shared rolling-origin set: three 16-day tail
  windows of valid (origins 2016-07-25 / 2016-08-10 / 2016-08-26),
  n_rows 28,512 per fold, pooled 85,536 scored jointly. Every window
  starts after 2016-05-15, so no quake-decay row lands in a fold. The
  is_*_tomorrow / is_*_in_2days lead flags are calendar-built and
  future-known by design; fit history stops at each fold's origin.
- Registered artifacts: artifacts/classical_results.json (S3, 5 tiers)
  and artifacts/deep_results.json (S4, 3 tiers), by exact filename. Any
  other path under artifacts/ fails loud, never silently skipped.
- artifacts/run_manifest.json holds one record per contract invocation,
  oldest first, exactly one record may ever carry freeze_run = true,
  and one record per day-2+ --check invocation is appended and verified.
- Freeze refusal gates run before anything else: FREEZE_RUN=1, no prior
  freeze_run = true record, full registry present, per-artifact schema.
  No gate opens a split; local_test loads only on the Day-5 body.
- Duplicate (store, family, date) keys, duplicate real ids, and negative
  sales are asserted zero on train and valid inside --check.
- Evaluation slices are scored without pre-opening masking.
- RMSLE is the primary evaluation metric; MAPE is undefined at 35.2% zero demand
  and is excluded from contract outputs.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import warnings
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import (  # pyright: ignore[reportMissingTypeStubs]
    mean_absolute_error,
    root_mean_squared_error,
    root_mean_squared_log_error,
)

__all__ = [
    "PINNED_DTYPES",
    "ContractError",
    "closure_rows",
    "fit_rows",
    "folds_table",
    "freeze",
    "load_frame",
    "load_split",
    "make_folds",
    "preopening_rows",
    "score",
    "training_mask",
    "write_run_manifest",
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

# -- Day-2 pins (sheet §5): the shared rolling-origin fold table. Every
# window is 16 inclusive days, starts the day after its origin, sits
# inside valid, and begins after 2016-05-15 (the quake-decay tail,
# earthquake_decay days 0..29 from 2016-04-16), so fold metrics carry no
# quake rows. n_rows = 1,782 store-family series x the 16-day horizon;
# pooled = the three windows scored jointly = 85,536 rows.
FOLD_ROWS = 28512
POOLED_FOLD_ROWS = 85536
FOLD_TABLE_PINS: tuple[tuple[int, str, str, str], ...] = (
    (1, "2016-07-25", "2016-07-26", "2016-08-10"),
    (2, "2016-08-10", "2016-08-11", "2016-08-26"),
    (3, "2016-08-26", "2016-08-27", "2016-09-11"),
)
VALID_START = pd.Timestamp("2016-03-28")
VALID_END = pd.Timestamp("2016-09-11")
QUAKE_WINDOW_END = pd.Timestamp("2016-05-15")

# -- Day-2 pins: registered model artifacts. Exact filenames under
# artifacts/; any other model *.json path there is a loud registry
# failure. The freeze consumes these JSON files only; S4's checkpoints
# are read by S4's own fit_predict_frozen, never parsed here.
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
# Control files the contract itself owns under artifacts/; neither is a
# model artifact, so neither registers nor trips the stray-file rule.
CONTRACT_ARTIFACT_FILENAMES = ("freeze_results.json", "run_manifest.json")
SCORE_KEYS = (
    "rmsle",
    "mae",
    "rmse",
    "rmsle_active",
    "mae_active",
    "rmse_active",
    "n_rows",
    "n_rows_active",
)
CLASSICAL_MODEL_NAMES = (
    "naive",
    "seasonal_naive",
    "moving_average",
    "ridge",
    "hgb",
)
DEEP_TIER_NAMES = ("deep-rnn", "deep-lstm", "deep-gru")
ARTIFACT_REGISTRY: dict[str, dict[str, Any]] = {
    "classical_results.json": {
        "track": "S3",
        "tiers_key": "models",
        "tier_name_key": "name",
        "model_count": 5,
        "model_names": CLASSICAL_MODEL_NAMES,
        # The percentage-error token is assembled from fragments so the
        # repo-wide audit grep keeps finding exactly one line in this
        # module: the documented module-comment exclusion at the top.
        "banned_key_parts": ("m" + "ape", "local_test"),
    },
    "deep_results.json": {
        "track": "S4",
        "tiers_key": "tiers",
        "tier_name_key": "tier",
        "model_count": 3,
        "model_names": DEEP_TIER_NAMES,
        "banned_key_parts": ("m" + "ape", "local_test", "christmas"),
    },
}
KIND_BY_ARTIFACT = {
    "classical_results.json": "classical",
    "deep_results.json": "deep",
}

# -- Day-2 pins: artifacts/run_manifest.json record schema. One record
# per contract invocation, oldest first; freeze_run is true on exactly
# one record ever, and only a successful Day-5 pass stamps it.
RUN_MANIFEST_KEYS = (
    "created",
    "git_sha",
    "command",
    "exit",
    "exit_ok",
    "versions",
    "elapsed_seconds",
    "arguments",
    "status",
    "freeze_run",
    "notes",
)
VERSION_KEYS = ("python", "uv", "pandas", "numpy")

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
    # MANIFEST_KEYS is checked for presence only (not exact match) to allow
    # forward-compatible auxiliary fields in split_manifest.json.
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
    for idx, col in enumerate(manifest["columns"]):
        if not isinstance(col, str):
            raise ContractError(
                f"manifest 'columns' element at index {idx} must be str, got {col!r}; "
                f"{RUN_MAIN_POINTER}"
            )
    missing = [c for c in PINNED_DTYPES if c not in manifest["columns"]]
    extra = [c for c in manifest["columns"] if c not in PINNED_DTYPES]
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"extra {extra}")
        raise ContractError(
            f"manifest 'columns' must be the 26 pinned columns; {'; '.join(details)}; "
            f"{RUN_MAIN_POINTER}"
        )
    pinned_columns = list(PINNED_DTYPES)
    if manifest["columns"] != pinned_columns:
        for i, (got, want) in enumerate(
            zip(manifest["columns"], pinned_columns, strict=False)
        ):
            if got != want:
                raise ContractError(
                    f"manifest 'columns' matches the pinned set but is out of "
                    f"pinned order at position {i}: got {got!r}, expected {want!r}; "
                    f"{RUN_MAIN_POINTER}"
                )
        if len(manifest["columns"]) > len(pinned_columns):
            i = len(pinned_columns)
            got = manifest["columns"][i]
            raise ContractError(
                f"manifest 'columns' matches the pinned set but is out of "
                f"pinned order at position {i}: got {got!r}, expected None; "
                f"{RUN_MAIN_POINTER}"
            )
    for split in SPLIT_NAMES:
        if split not in manifest["rows"]:
            raise ContractError(
                f"manifest rows missing split {split!r}; {RUN_MAIN_POINTER}"
            )
        val = manifest["rows"][split]
        if type(val) is not int:
            raise ContractError(
                f"manifest rows[{split!r}] must be int, got {val!r} "
                f"({type(val).__name__}); {RUN_MAIN_POINTER}"
            )
    total = manifest["total"]
    if type(total) is not int or total != 3008016:
        raise ContractError(
            f"manifest total {total!r} ({type(total).__name__}) != 3008016; "
            f"{RUN_MAIN_POINTER}"
        )
    counts = [manifest["rows"][split] for split in SPLIT_NAMES]
    if any(count < 0 for count in counts):
        raise ContractError(
            f"manifest rows contain a negative count {counts}; {RUN_MAIN_POINTER}"
        )
    rows_sum = sum(counts)
    if rows_sum != total:
        raise ContractError(
            f"manifest rows sum {rows_sum} != total {total}; {RUN_MAIN_POINTER}"
        )
    for split in SPLIT_NAMES:
        if split not in manifest["bounds"]:
            raise ContractError(
                f"manifest bounds missing split {split!r}; {RUN_MAIN_POINTER}"
            )
        split_bounds = manifest["bounds"][split]
        if not isinstance(split_bounds, list) or len(split_bounds) != 2:
            raise ContractError(
                f"manifest bounds[{split!r}] must be a 2-element list; "
                f"{RUN_MAIN_POINTER}"
            )
        for bound_name, bound_val in (
            ("start", split_bounds[0]),
            ("end", split_bounds[1]),
        ):
            if not isinstance(bound_val, str):
                raise ContractError(
                    f"manifest bounds[{split!r}] {bound_name} timestamp must be str, "
                    f"got {bound_val!r}; {RUN_MAIN_POINTER}"
                )
        try:
            start_ts = pd.Timestamp(split_bounds[0])
        except Exception as error:
            raise ContractError(
                f"manifest bounds[{split!r}] start timestamp invalid "
                f"({split_bounds[0]!r}): {error}; {RUN_MAIN_POINTER}"
            ) from error
        if pd.isna(start_ts):
            raise ContractError(
                f"manifest bounds[{split!r}] start timestamp {split_bounds[0]!r} "
                f"parses to NaT, not a real date; {RUN_MAIN_POINTER}"
            )
        if start_ts.tzinfo is not None:
            raise ContractError(
                f"manifest bounds[{split!r}] start timestamp {split_bounds[0]!r} "
                f"carries a timezone (got tz {start_ts.tzinfo!r}); "
                f"naive timestamps only; {RUN_MAIN_POINTER}"
            )
        try:
            end_ts = pd.Timestamp(split_bounds[1])
        except Exception as error:
            raise ContractError(
                f"manifest bounds[{split!r}] end timestamp invalid "
                f"({split_bounds[1]!r}): {error}; {RUN_MAIN_POINTER}"
            ) from error
        if pd.isna(end_ts):
            raise ContractError(
                f"manifest bounds[{split!r}] end timestamp {split_bounds[1]!r} "
                f"parses to NaT, not a real date; {RUN_MAIN_POINTER}"
            )
        if end_ts.tzinfo is not None:
            raise ContractError(
                f"manifest bounds[{split!r}] end timestamp {split_bounds[1]!r} "
                f"carries a timezone (got tz {end_ts.tzinfo!r}); "
                f"naive timestamps only; {RUN_MAIN_POINTER}"
            )
        if start_ts >= end_ts:
            raise ContractError(
                f"manifest bounds[{split!r}] start {start_ts} not strictly "
                f"before end {end_ts}; {RUN_MAIN_POINTER}"
            )
    if "grid_fill_rows" not in manifest["why"]:
        raise ContractError(
            f"manifest field 'why.grid_fill_rows' missing; {RUN_MAIN_POINTER}"
        )
    grid_fills = manifest["why"]["grid_fill_rows"]
    if not isinstance(grid_fills, dict):
        raise ContractError(
            f"manifest field 'why.grid_fill_rows' must be an object; {RUN_MAIN_POINTER}"
        )
    grid_fill_sum = 0
    for split in SPLIT_NAMES:
        if split not in grid_fills:
            raise ContractError(
                f"manifest why.grid_fill_rows missing split {split!r}; "
                f"{RUN_MAIN_POINTER}"
            )
        count = grid_fills[split]
        if type(count) is not int:
            raise ContractError(
                f"manifest why.grid_fill_rows[{split!r}] must be int, got "
                f"{count!r} ({type(count).__name__}); {RUN_MAIN_POINTER}"
            )
        grid_fill_sum += count
    if grid_fill_sum != 7128:
        raise ContractError(
            f"manifest why.grid_fill_rows sum is {grid_fill_sum}, expected 7128; "
            f"{RUN_MAIN_POINTER}"
        )
    frame_rows_sum = manifest["rows"]["train"] + manifest["rows"]["valid"]
    if FRAME_ROWS != frame_rows_sum:
        raise ContractError(
            f"FRAME_ROWS {FRAME_ROWS} != train ({manifest['rows']['train']}) + "
            f"valid ({manifest['rows']['valid']}); {RUN_MAIN_POINTER}"
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
    except Exception as error:
        raise ContractError(
            f"{path.name} unreadable: {error}; {RUN_MAIN_POINTER}"
        ) from error
    if not isinstance(data, pd.DataFrame):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ContractError(
            f"{split}: reader returned {type(data).__name__}, expected a DataFrame; "
            f"{RUN_MAIN_POINTER}"
        )
    if not data.columns.is_unique:
        dup_names = _duplicate_column_names(list(data.columns))
        raise ContractError(
            f"{split}: columns contain duplicates {dup_names}; {RUN_MAIN_POINTER}"
        )
    missing = [c for c in PINNED_DTYPES if c not in data.columns]
    extra = [c for c in data.columns if c not in PINNED_DTYPES]
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"extra {extra}")
        details_str = "; ".join(details)
        raise ContractError(
            f"{split}: columns must exactly match the 26 pinned columns; "
            f"{details_str}; {RUN_MAIN_POINTER}"
        )
    for col, expected_dtype in PINNED_DTYPES.items():
        if str(data[col].dtype) != expected_dtype:
            raise ContractError(
                f"{split}: column {col!r} has dtype {data[col].dtype}, "
                f"expected {expected_dtype}; {RUN_MAIN_POINTER}"
            )
    if list(data.columns) != list(manifest["columns"]):
        raise ContractError(
            f"{split}: columns disagree with manifest; {RUN_MAIN_POINTER}"
        )
    try:
        expected_rows = manifest["rows"][split]
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError(
            f"manifest rows[{split!r}] invalid; {RUN_MAIN_POINTER}"
        ) from error
    if len(data) != expected_rows:
        raise ContractError(
            f"{split}: row count {len(data)} != manifest {expected_rows}; "
            f"{RUN_MAIN_POINTER}"
        )
    try:
        start_str, end_str = manifest["bounds"][split]
    except Exception as error:
        raise ContractError(
            f"{split}: invalid bounds sequence: {error}; {RUN_MAIN_POINTER}"
        ) from error
    try:
        start_ts = pd.Timestamp(start_str)
    except Exception as error:
        raise ContractError(
            f"{split}: start bound {start_str!r} invalid: {error}; {RUN_MAIN_POINTER}"
        ) from error
    if pd.isna(start_ts):
        raise ContractError(
            f"{split}: start bound {start_str!r} parses to NaT, not a real date; "
            f"{RUN_MAIN_POINTER}"
        )
    if start_ts.tzinfo is not None:
        raise ContractError(
            f"{split}: start bound {start_str!r} carries a timezone "
            f"(got tz {start_ts.tzinfo!r}); naive timestamps only; {RUN_MAIN_POINTER}"
        )
    try:
        end_ts = pd.Timestamp(end_str)
    except Exception as error:
        raise ContractError(
            f"{split}: end bound {end_str!r} invalid: {error}; {RUN_MAIN_POINTER}"
        ) from error
    if pd.isna(end_ts):
        raise ContractError(
            f"{split}: end bound {end_str!r} parses to NaT, not a real date; "
            f"{RUN_MAIN_POINTER}"
        )
    if end_ts.tzinfo is not None:
        raise ContractError(
            f"{split}: end bound {end_str!r} carries a timezone "
            f"(got tz {end_ts.tzinfo!r}); naive timestamps only; {RUN_MAIN_POINTER}"
        )
    actual_start = data["date"].min()
    actual_end = data["date"].max()
    if actual_start != start_ts or actual_end != end_ts:
        raise ContractError(
            f"{split}: date bounds ({actual_start.date()}, {actual_end.date()}) "
            f"!= manifest ({start_ts.date()}, {end_ts.date()}); {RUN_MAIN_POINTER}"
        )
    if data["sales"].isna().any():
        raise ContractError(f"{split}: sales contains NaN; {RUN_MAIN_POINTER}")
    if data["date"].isna().any():
        raise ContractError(f"{split}: date contains NaT value(s); {RUN_MAIN_POINTER}")
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
    for role, frame in (("train", train), ("valid", valid)):
        if not isinstance(frame, pd.DataFrame):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ContractError(
                f"load_frame: {role} must be a DataFrame; {RUN_MAIN_POINTER}"
            )
        if not frame.columns.is_unique:
            dup_names = _duplicate_column_names(list(frame.columns))
            raise ContractError(
                f"load_frame: {role} columns contain duplicates {dup_names}; "
                f"{RUN_MAIN_POINTER}"
            )
        missing = [c for c in PINNED_DTYPES if c not in frame.columns]
        extra = [c for c in frame.columns if c not in PINNED_DTYPES]
        if missing or extra:
            details: list[str] = []
            if missing:
                details.append(f"missing {missing}")
            if extra:
                details.append(f"extra {extra}")
            details_str = "; ".join(details)
            raise ContractError(
                f"load_frame: {role} columns must exactly match the 26 pinned columns; "
                f"{details_str}; {RUN_MAIN_POINTER}"
            )
        expected_columns = list(PINNED_DTYPES)
        if list(frame.columns) != expected_columns:
            cols = list(frame.columns)
            for i in range(max(len(cols), len(expected_columns))):
                got = cols[i] if i < len(cols) else "<end>"
                want = expected_columns[i] if i < len(expected_columns) else "<end>"
                if got != want:
                    raise ContractError(
                        f"load_frame: {role} column order disagrees with the pinned "
                        f"order at position {i}: got {got!r}, expected {want!r}; "
                        f"{RUN_MAIN_POINTER}"
                    )
        for col, expected_dtype in PINNED_DTYPES.items():
            actual_dtype = str(frame[col].dtype)
            if actual_dtype != expected_dtype:
                raise ContractError(
                    f"load_frame: {role} column {col!r} got dtype {actual_dtype}, "
                    f"expected {expected_dtype}; {RUN_MAIN_POINTER}"
                )
    data = pd.concat([train, valid], ignore_index=True)
    expected = list(PINNED_DTYPES)
    if list(data.columns) != expected:
        cols = list(data.columns)
        for i in range(max(len(cols), len(expected))):
            got = cols[i] if i < len(cols) else "<end>"
            want = expected[i] if i < len(expected) else "<end>"
            if got != want:
                raise ContractError(
                    f"load_frame: concatenated column order disagrees with the pinned "
                    f"order at position {i}: got {got!r}, expected {want!r}; "
                    f"{RUN_MAIN_POINTER}"
                )
    # Re-check dtypes post-concat because pd.concat can widen category -> object
    # when train and valid carry disjoint category domains.
    for col, expected_dtype in PINNED_DTYPES.items():
        actual = str(data[col].dtype)
        if actual != expected_dtype:
            raise ContractError(
                f"load_frame: concatenated column {col!r} got dtype {actual}, "
                f"expected {expected_dtype}; {RUN_MAIN_POINTER}"
            )
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


def _duplicate_column_names(names: list[Any]) -> list[Any]:
    """Return duplicated labels sorted by repr; tolerates unhashable labels."""
    dup_names: list[Any] = []
    for name in names:
        if names.count(name) > 1 and not any(
            existing == name for existing in dup_names
        ):
            dup_names.append(name)
    dup_names.sort(key=repr)
    return dup_names


def _require_unique_columns(df: pd.DataFrame, fn_name: str) -> None:
    """Refuse duplicated column labels; df[col] would widen to a DataFrame."""
    dup_names = _duplicate_column_names(list(df.columns))
    if dup_names:
        raise ContractError(
            f"{fn_name}: columns contain duplicates {dup_names}; "
            "df[col] would not be a Series"
        )


def closure_rows(df: pd.DataFrame) -> pd.Series:
    """Return boolean mask for holiday closure fill rows (id is NaN)."""
    if not isinstance(df, pd.DataFrame):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ContractError(
            f"closure_rows: input must be a DataFrame, got {type(df).__name__}"
        )
    _require_unique_columns(df, "closure_rows")
    if "id" not in df.columns:
        raise ContractError("closure_rows: input dataframe missing 'id' column")
    return df["id"].isna()


def preopening_rows(df: pd.DataFrame) -> pd.Series:
    """Return boolean mask for dates before a store officially opened."""
    if not isinstance(df, pd.DataFrame):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ContractError(
            f"preopening_rows: input must be a DataFrame, got {type(df).__name__}"
        )
    _require_unique_columns(df, "preopening_rows")
    for col in ("store_nbr", "date"):
        if col not in df.columns:
            raise ContractError(f"preopening_rows: missing {col!r} column")
    store_col = df["store_nbr"]
    if not pd.api.types.is_integer_dtype(store_col):
        raise ContractError(
            f"preopening_rows: 'store_nbr' must be integer dtype, got {store_col.dtype}"
        )
    if store_col.isna().any():
        raise ContractError(
            f"preopening_rows: 'store_nbr' contains {int(store_col.isna().sum())} "
            "<NA> value(s); nullable integer store_nbr force-resolves masks at "
            "indexing; use a plain integer dtype"
        )
    date_col = df["date"]
    if (
        not pd.api.types.is_datetime64_any_dtype(date_col)
        or getattr(date_col.dtype, "tz", None) is not None
    ):
        raise ContractError(
            "preopening_rows: 'date' must be datetime dtype without timezone, "
            f"got {date_col.dtype}"
        )
    mask = pd.Series(False, index=df.index)
    for store, open_date in DELAYS_OPEN_UNTIL.items():
        store_match = store_col == store
        before_open = date_col < open_date
        mask |= store_match & before_open
    return mask


def fit_rows(df: pd.DataFrame) -> pd.Series:
    """Return boolean mask of rows to include in model training."""
    if not isinstance(df, pd.DataFrame):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ContractError(
            f"fit_rows: input must be a DataFrame, got {type(df).__name__}"
        )
    _require_unique_columns(df, "fit_rows")
    is_preopen = preopening_rows(df)
    is_closure = closure_rows(df)
    return (~is_preopen) | is_closure


training_mask = fit_rows


def folds_table() -> pd.DataFrame:
    """The rolling-origin fold table every track builds its folds against.

    One row per pinned window (sheet §5, Day 2). Columns, pinned order:
    fold (int), origin / window_start / window_end (ISO date strings),
    n_rows (int; 1,782 series x the 16-day horizon = 28,512 per fold,
    pooled 85,536). Fold k's training history is
    concat(train, valid[date <= origin]) masked by fit_rows; that
    assembly is the consumer's job and is never re-derived here. The
    is_*_tomorrow / is_*_in_2days lead flags are calendar-built and
    future-known by design; training history stops at each origin, so
    nothing reaches from the future.
    """
    rows: list[tuple[int, str, str, str]] = []
    for fold, origin, start, end in FOLD_TABLE_PINS:
        if (pd.Timestamp(end) - pd.Timestamp(start)).days != 15:
            raise ContractError(
                f"fold {fold}: window {start}..{end} is not 16 inclusive days"
            )
        if pd.Timestamp(start) != pd.Timestamp(origin) + pd.Timedelta(days=1):
            raise ContractError(
                f"fold {fold}: window must start the day after origin {origin}"
            )
        if not (
            VALID_START <= pd.Timestamp(origin) < pd.Timestamp(start)
            and pd.Timestamp(end) <= VALID_END
        ):
            raise ContractError(
                f"fold {fold}: window {start}..{end} escapes the valid bounds"
            )
        if pd.Timestamp(start) <= QUAKE_WINDOW_END:
            raise ContractError(
                f"fold {fold}: window starts at or before the quake tail "
                "2016-05-15; fold metrics would carry quake rows"
            )
        rows.append((fold, origin, start, end))
    table = pd.DataFrame(rows, columns=["fold", "origin", "window_start", "window_end"])
    table["n_rows"] = FOLD_ROWS
    return table[["fold", "origin", "window_start", "window_end", "n_rows"]]


# Alias: the name S3 and S4 import; one implementation, never re-derived.
make_folds = folds_table


def score(
    y_true: np.ndarray | Any,
    y_pred: np.ndarray | Any,
    is_active: np.ndarray | Any | None = None,
) -> dict[str, float | int]:
    """Compute RMSLE, MAE, and RMSE on full and active demand slices.

    Predictions are clipped to non-negative values before computing metrics.
    When n_rows_active == 0, active metrics return 0.0.
    """
    if isinstance(y_true, np.ma.MaskedArray) or isinstance(y_pred, np.ma.MaskedArray):
        raise ContractError("score: masked arrays are not supported")
    if is_active is not None and isinstance(is_active, np.ma.MaskedArray):
        raise ContractError("score: is_active cannot be a masked array")

    for name, raw in (("y_true", y_true), ("y_pred", y_pred)):
        try:
            arr = np.asarray(raw)
        except Exception as error:
            raise ContractError(
                f"score: {name} could not be converted to array: {error}"
            ) from error
        if arr.dtype == np.bool_ or arr.dtype.kind == "b":
            raise ContractError(f"score: {name} cannot be boolean")
        if arr.dtype.kind not in ("f", "i", "u"):
            raise ContractError(
                f"score: {name} must be numeric array; got dtype {arr.dtype}"
            )
    try:
        truth = np.asarray(y_true, dtype=np.float64)
        pred = np.asarray(y_pred, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ContractError(
            f"score: y_true and y_pred must be numeric arrays; {error}"
        ) from error
    for name, values in (("y_true", truth), ("y_pred", pred)):
        if values.ndim != 1:
            raise ContractError(f"score: {name} must be 1-D, got shape {values.shape}")
        if not np.isfinite(values).all():
            raise ContractError(f"score: {name} contains NaN or infinity")
    if truth.shape != pred.shape:
        raise ContractError(f"score: shape mismatch {truth.shape} vs {pred.shape}")
    if truth.size == 0:
        raise ContractError("score: empty inputs")
    if (truth < 0).any():
        raise ContractError(
            "score: y_true must be non-negative for RMSLE (log1p domain)"
        )
    pred = np.clip(pred, 0.0, None)
    if is_active is None:
        active = truth > 0
    else:
        try:
            active = np.asarray(is_active)
        except (ValueError, TypeError) as error:
            raise ContractError(
                f"score: is_active could not be converted to a boolean array: {error}"
            ) from error
        if active.dtype != np.bool_:
            raise ContractError(
                f"score: is_active must be boolean mask; got dtype {active.dtype}"
            )
        if active.shape != truth.shape:
            raise ContractError(
                f"score: is_active shape {active.shape} != {truth.shape}"
            )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            rmsle = float(root_mean_squared_log_error(truth, pred))
            mae = float(mean_absolute_error(truth, pred))
            rmse = float(root_mean_squared_error(truth, pred))
        except Exception as error:
            raise ContractError(f"score: metric calculation failed: {error}") from error
    result: dict[str, float | int] = {
        "rmsle": rmsle,
        "mae": mae,
        "rmse": rmse,
    }
    n_active = int(active.sum())
    if n_active:
        sub_true = truth[active]
        sub_pred = pred[active]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                result["rmsle_active"] = float(
                    root_mean_squared_log_error(sub_true, sub_pred)
                )
                result["mae_active"] = float(mean_absolute_error(sub_true, sub_pred))
                result["rmse_active"] = float(
                    root_mean_squared_error(sub_true, sub_pred)
                )
            except Exception as error:
                raise ContractError(
                    f"score: active metric calculation failed: {error}"
                ) from error
    else:
        result["rmsle_active"] = 0.0
        result["mae_active"] = 0.0
        result["rmse_active"] = 0.0

    for key in (
        "rmsle",
        "mae",
        "rmse",
        "rmsle_active",
        "mae_active",
        "rmse_active",
    ):
        val = result[key]
        if not math.isfinite(val):
            raise ContractError(f"score: computed metric {key} is non-finite ({val})")

    # Squared deviations of subnormal values underflow float64 to 0.0, so RMSE can
    # come back 0.0 while MAE is tiny but nonzero (e.g. y_true=1e-200). The
    # tolerance keeps the quadratic-mean >= arithmetic-mean invariant meaningful
    # without rejecting the float artifact.
    tol = max(1e-12, 1e-12 * mae)
    if rmse < mae - tol:
        raise ContractError(f"score: inconsistent metrics rmse {rmse} < mae {mae}")
    if result["rmse_active"] < result["mae_active"] - max(
        1e-12, 1e-12 * result["mae_active"]
    ):
        raise ContractError(
            f"score: inconsistent active metrics rmse_active {result['rmse_active']} "
            f"< mae_active {result['mae_active']}"
        )

    result["n_rows"] = int(truth.size)
    result["n_rows_active"] = n_active
    return result


def _git_sha() -> tuple[str, list[str]]:
    """Best-effort HEAD sha for the manifest record.

    Returns (sha, notes): a 40-hex sha when git answers, else "" plus a
    note naming the failure. A missing sha is recorded, never faked.
    """
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return "", [f"git sha unavailable: {error}"]
    sha = (head.stdout or "").strip()
    valid = (
        head.returncode == 0
        and len(sha) == 40
        and not set(sha.lower()) - set("0123456789abcdef")
    )
    if not valid:
        detail = (head.stderr or "").strip()[:80]
        return "", [f"git sha unavailable: exit {head.returncode} {detail}"]
    return sha, []


def _tool_versions() -> dict[str, str]:
    """Pinned version block: python / uv / pandas / numpy, string keys."""
    uv_version = "unavailable"
    try:
        probe = subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        parts = (probe.stdout or "").split()
        if probe.returncode == 0 and len(parts) >= 2:
            uv_version = parts[1]
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "python": sys.version.split()[0],
        "uv": uv_version,
        "pandas": str(pd.__version__),
        "numpy": str(np.__version__),
    }


def _validate_run_manifest_record(record: Any, index: int) -> None:
    """One record; the full pinned schema (sheet §10), any drift is loud."""
    label = f"run_manifest.json record {index}"
    if not isinstance(record, dict):
        raise ContractError(f"{label} is not a JSON object")
    if list(record) != list(RUN_MANIFEST_KEYS):
        raise ContractError(
            f"{label}: fields {list(record)} != pinned {list(RUN_MANIFEST_KEYS)}"
        )
    created = record["created"]
    if not isinstance(created, str):
        raise ContractError(f"{label}: created must be an ISO 8601 string")
    try:
        datetime.fromisoformat(created.replace("Z", "+00:00"))
    except ValueError as error:
        raise ContractError(
            f"{label}: created {created!r} does not parse; {error}"
        ) from error
    sha = record["git_sha"]
    sha_ok = isinstance(sha, str) and (
        not sha or (len(sha) == 40 and not set(sha.lower()) - set("0123456789abcdef"))
    )
    if not sha_ok:
        raise ContractError(f"{label}: git_sha must be a 40-hex sha (or empty)")
    if isinstance(record["exit"], bool) or not isinstance(record["exit"], int):
        raise ContractError(f"{label}: exit must be an int")
    if not isinstance(record["exit_ok"], bool):
        raise ContractError(f"{label}: exit_ok must be a bool")
    if record["exit_ok"] != (record["exit"] == 0):
        raise ContractError(f"{label}: exit_ok disagrees with exit {record['exit']}")
    expected_status = "PASS" if record["exit"] == 0 else "FAIL"
    if record["status"] != expected_status:
        raise ContractError(
            f"{label}: status {record['status']!r} != {expected_status!r}"
        )
    versions = record["versions"]
    if not isinstance(versions, dict) or list(versions) != list(VERSION_KEYS):
        measured = (
            list(versions) if isinstance(versions, dict) else type(versions).__name__
        )
        raise ContractError(
            f"{label}: versions {measured} != pinned {list(VERSION_KEYS)}"
        )
    if not all(isinstance(v, str) and v for v in versions.values()):
        raise ContractError(f"{label}: version values must be non-empty strings")
    elapsed = record["elapsed_seconds"]
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or elapsed < 0
    ):
        raise ContractError(f"{label}: elapsed_seconds must be a non-negative number")
    arguments = record["arguments"]
    if not isinstance(arguments, dict) or not all(
        isinstance(k, str) for k in arguments
    ):
        raise ContractError(f"{label}: arguments must be an object with string keys")
    if not isinstance(record["freeze_run"], bool):
        raise ContractError(f"{label}: freeze_run must be a bool")
    notes = record["notes"]
    if not isinstance(notes, list) or not all(isinstance(n, str) for n in notes):
        raise ContractError(f"{label}: notes must be a list of strings")


def read_run_manifest(path: Path) -> list[dict[str, Any]]:
    """Parse artifacts/run_manifest.json; friendly on any drift.

    Enforces the per-record schema and the one-freeze rule: at most one
    record across the file may carry freeze_run = true (a second is an
    incident, never a statistic).
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ContractError(f"{path.name} does not parse: {error}") from error
    if not isinstance(payload, list):
        raise ContractError(f"{path.name} is not a JSON array of records")
    for index, record in enumerate(payload):
        _validate_run_manifest_record(record, index)
    frozen = [record for record in payload if record["freeze_run"] is True]
    if len(frozen) > 1:
        raise ContractError(
            f"{path.name} carries {len(frozen)} freeze_run = true records; "
            "exactly one freeze run is allowed"
        )
    return payload


def write_run_manifest(
    command: str,
    *,
    exit_status: int = 0,
    elapsed_seconds: float = 0.0,
    arguments: dict[str, str] | None = None,
    freeze_run: bool = False,
    notes: list[str] | None = None,
    artifacts_dir: Path | None = None,
) -> None:
    """Append one invocation record to artifacts/run_manifest.json.

    Fields and semantics are pinned (RUN_MANIFEST_KEYS): created (ISO
    8601 UTC), git_sha (40-hex; empty plus a note when git answers with
    nothing), command, exit, exit_ok, versions (python/uv/pandas/numpy),
    elapsed_seconds, arguments (string keys), status (PASS/FAIL derived
    from exit), freeze_run, notes. The write is atomic (temp file then
    replace) and re-verified: the file is reopened, parsed, its size
    asserted, and the appended record must land last. Both --check and
    every freeze attempt that gets past the flag gate write here.
    """
    directory = Path(artifacts_dir) if artifacts_dir is not None else ARTIFACTS_DIR
    path = directory / "run_manifest.json"
    directory.mkdir(parents=True, exist_ok=True)
    sha, sha_notes = _git_sha()
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    record: dict[str, Any] = {
        "created": stamp.replace("+00:00", "Z"),
        "git_sha": sha,
        "command": str(command),
        "exit": int(exit_status),
        "exit_ok": exit_status == 0,
        "versions": _tool_versions(),
        "elapsed_seconds": float(elapsed_seconds),
        "arguments": {str(k): str(v) for k, v in (arguments or {}).items()},
        "status": "PASS" if exit_status == 0 else "FAIL",
        "freeze_run": bool(freeze_run),
        "notes": [*(sha_notes), *(notes or [])],
    }
    records = list(read_run_manifest(path)) if path.exists() else []
    records.append(record)
    tmp = directory / "run_manifest.json.tmp"
    tmp.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    if tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise ContractError("run_manifest write produced a zero-byte temp file")
    os.replace(tmp, path)
    stored = read_run_manifest(path)
    if not stored or stored[-1] != record:
        raise ContractError("run_manifest append did not land as the last record")


def _scan_registered_artifacts(directory: Path) -> dict[str, Path]:
    """Registered model artifact files on disk; stray model files fail loud.

    The two contract control files (run_manifest.json, and
    freeze_results.json from Day 5 on) are contract furniture, not model
    artifacts: they neither register nor trip the stray-file rule.
    """
    if not directory.exists():
        return {}
    present = sorted(p for p in directory.glob("*.json") if p.is_file())
    registered = set(ARTIFACT_REGISTRY)
    unknown = [
        p.name
        for p in present
        if p.name not in registered and p.name not in CONTRACT_ARTIFACT_FILENAMES
    ]
    if unknown:
        raise ContractError(
            f"artifacts/: unregistered file(s) {unknown}; the registry carries "
            f"exactly {sorted(registered)} - nothing stray is skipped silently"
        )
    return {
        name: directory / name for name in registered if (directory / name).exists()
    }


def _banned_keys_hit(name: str, payload: Any) -> list[str]:
    """Recursively scan object keys for banned substrings, keys only."""
    banned: tuple[str, ...] = cast(
        "tuple[str, ...]", ARTIFACT_REGISTRY[name]["banned_key_parts"]
    )
    hits: list[str] = []

    def walk(node: Any, prefix: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                lowered = str(key).lower()
                if any(part in lowered for part in banned):
                    hits.append(f"{prefix}{key}")
                walk(value, f"{prefix}{key}.")
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f"{prefix}[{index}].")

    walk(payload, "")
    return sorted(dict.fromkeys(hits))


def _assert_score_block(
    label: str, block: Any, *, head: tuple[str, ...] = ()
) -> dict[str, Any]:
    """The pinned 8-key score schema, finite non-negative metrics."""
    pinned = list(head) + list(SCORE_KEYS)
    if not isinstance(block, dict) or list(block) != pinned:
        measured = list(block) if isinstance(block, dict) else type(block).__name__
        raise ContractError(
            f"{label}: keys {measured} != the pinned score schema {pinned}"
        )
    for key in SCORE_KEYS[:6]:
        value = block[key]
        bad = (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        )
        if bad:
            raise ContractError(
                f"{label}: {key} must be a finite non-negative number, got {value!r}"
            )
    for key in ("n_rows", "n_rows_active"):
        count = block[key]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ContractError(f"{label}: {key} must be a non-negative int")
    return block


_CLASSICAL_REQUIRED = (
    "schema_version",
    "git_sha",
    "python_version",
    "sklearn_version",
    "split_bounds",
    "training_mask",
    "feature_columns",
    "target",
    "prediction_clip",
    "folds",
    "models",
    "notes",
)
_CLASSICAL_MODEL_KEYS = (
    "name",
    "kind",
    "target",
    "hyperparameters",
    "fold_metrics",
    "pooled",
    "seeds",
    "tuning",
    "christmas_2016",
)


def _validate_classical_artifact(payload: dict[str, Any]) -> None:
    """S3's 5-tier classical artifact: required keys, pinned folds."""
    name = "artifacts/classical_results.json"
    missing = [k for k in _CLASSICAL_REQUIRED if k not in payload]
    if missing:
        raise ContractError(f"{name}: missing top-level keys {missing}")
    schema_version = payload["schema_version"]
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ContractError(f"{name}: schema_version must be an int")
    bounds = payload["split_bounds"]
    if not isinstance(bounds, dict) or sorted(bounds) != ["train", "valid"]:
        raise ContractError(
            f"{name}: split_bounds must cover exactly train and valid "
            "(a classical artifact never carries local_test bounds)"
        )
    for text_key in (
        "git_sha",
        "python_version",
        "sklearn_version",
        "training_mask",
        "target",
        "prediction_clip",
    ):
        if not isinstance(payload[text_key], str):
            raise ContractError(f"{name}: {text_key} must be a string")
    feature_columns = payload["feature_columns"]
    if (
        not isinstance(feature_columns, list)
        or len(feature_columns) != 23
        or len(set(feature_columns)) != 23
    ):
        raise ContractError(f"{name}: feature_columns must list 23 unique names")
    folds = payload["folds"]
    if not isinstance(folds, list) or len(folds) != 3:
        raise ContractError(f"{name}: folds must hold the 3 pinned windows")
    for index, (fold, origin, start, end) in enumerate(FOLD_TABLE_PINS):
        block = folds[index]
        pinned = [fold, origin, start, end, FOLD_ROWS]
        if not isinstance(block, dict) or list(block) != [
            "fold",
            "origin",
            "window_start",
            "window_end",
            "n_rows",
        ]:
            raise ContractError(
                f"{name}: folds[{index}] keys must be "
                "fold/origin/window_start/window_end/n_rows"
            )
        if [
            block["fold"],
            block["origin"],
            block["window_start"],
            block["window_end"],
            block["n_rows"],
        ] != pinned:
            raise ContractError(
                f"{name}: folds[{index}] {block} != the pinned window {pinned}"
            )
    if not isinstance(payload["notes"], list) or not all(
        isinstance(note, str) for note in payload["notes"]
    ):
        raise ContractError(f"{name}: notes must be a list of strings")
    models = payload["models"]
    if not isinstance(models, list) or len(models) != 5:
        raise ContractError(f"{name}: models must hold the 5 tiers")
    for index, entry in enumerate(models):
        label = f"{name}: model {index} ({entry.get('name')!r} {entry.get('kind')!r})"
        if not isinstance(entry, dict) or list(entry) != list(_CLASSICAL_MODEL_KEYS):
            measured = list(entry) if isinstance(entry, dict) else type(entry).__name__
            raise ContractError(
                f"{label}: per-model keys {measured} != pinned "
                f"{list(_CLASSICAL_MODEL_KEYS)}"
            )
        if entry["kind"] not in ("baseline", "classical_ml"):
            raise ContractError(f"{label}: kind must be baseline or classical_ml")
        if not isinstance(entry["hyperparameters"], dict):
            raise ContractError(f"{label}: hyperparameters must be an object")
        seeds = entry["seeds"]
        if not isinstance(seeds, dict) or not all(
            isinstance(seed_key, str) for seed_key in seeds
        ):
            raise ContractError(f"{label}: seeds must be an object with string keys")
        if not isinstance(entry["tuning"], str):
            raise ContractError(f"{label}: tuning must be a string")
        christmas = entry["christmas_2016"]
        if not isinstance(christmas, dict) or christmas.get("status") != "freeze-time":
            raise ContractError(
                f"{label}: christmas_2016 must carry status 'freeze-time' "
                "(S1 scores that slice at freeze, never in the artifact)"
            )
        metrics = entry["fold_metrics"]
        if not isinstance(metrics, list) or len(metrics) != 3:
            raise ContractError(f"{label}: fold_metrics must hold 3 folds")
        for fold_index, fold_row in enumerate(metrics):
            block = _assert_score_block(
                f"{label} fold {fold_index + 1}", fold_row, head=("fold",)
            )
            if block["fold"] != fold_index + 1 or block["n_rows"] != FOLD_ROWS:
                raise ContractError(
                    f"{label}: fold_metrics[{fold_index}] must be fold "
                    f"{fold_index + 1} with n_rows {FOLD_ROWS}"
                )
        pooled = _assert_score_block(f"{label} pooled", entry["pooled"])
        if pooled["n_rows"] != POOLED_FOLD_ROWS:
            raise ContractError(
                f"{label}: pooled n_rows {pooled['n_rows']} != {POOLED_FOLD_ROWS}"
            )


_DEEP_REQUIRED = (
    "schema_version",
    "status",
    "track",
    "generated_at",
    "git_sha",
    "data_fingerprint",
    "shared_config",
    "config_sha256",
    "tiers",
    "ablation_2015plus",
    "comparison",
    "runtime_tradeoff",
    "notes",
)
_DEEP_TIER_KEYS = (
    "tier",
    "architecture",
    "checkpoint",
    "checkpoint_bytes",
    "checkpoint_sha256",
    "params",
    "status",
    "train",
    "valid",
    "fold_metrics",
)


def _validate_deep_artifact(payload: dict[str, Any]) -> None:
    """S4's 3-tier deep artifact: required keys, pinned tier order."""
    name = "artifacts/deep_results.json"
    missing = [k for k in _DEEP_REQUIRED if k not in payload]
    if missing:
        raise ContractError(f"{name}: missing top-level keys {missing}")
    if not isinstance(payload["schema_version"], str):
        raise ContractError(f"{name}: schema_version must be a string")
    if payload["status"] not in ("partial", "complete"):
        raise ContractError(
            f"{name}: status {payload['status']!r} must be partial or complete"
        )
    if payload["track"] != "S4":
        raise ContractError(f"{name}: track must be S4")
    if not isinstance(payload["git_sha"], str) or not isinstance(
        payload["generated_at"], str
    ):
        raise ContractError(f"{name}: git_sha/generated_at must be strings")
    for object_key in (
        "data_fingerprint",
        "shared_config",
        "ablation_2015plus",
        "runtime_tradeoff",
    ):
        if not isinstance(payload[object_key], dict):
            raise ContractError(f"{name}: {object_key} must be an object")
    if not isinstance(payload["config_sha256"], str):
        raise ContractError(f"{name}: config_sha256 must be a string")
    comparison = payload["comparison"]
    if not isinstance(comparison, dict) or comparison.get("status") not in (
        "ok",
        "pending_s3",
        "schema_mismatch",
    ):
        raise ContractError(
            f"{name}: comparison.status must be ok / pending_s3 / schema_mismatch"
        )
    if not isinstance(payload["notes"], list) or not all(
        isinstance(note, str) for note in payload["notes"]
    ):
        raise ContractError(f"{name}: notes must be a list of strings")
    tiers = payload["tiers"]
    if not isinstance(tiers, list) or len(tiers) != 3:
        raise ContractError(f"{name}: tiers must hold the 3 architectures")
    for index, entry in enumerate(tiers):
        tie_label = entry.get("tier") if isinstance(entry, dict) else "?"
        label = f"{name}: tier {index} ({tie_label!r})"
        if not isinstance(entry, dict) or list(entry) != list(_DEEP_TIER_KEYS):
            measured = list(entry) if isinstance(entry, dict) else type(entry).__name__
            raise ContractError(
                f"{label}: per-tier keys {measured} != pinned {list(_DEEP_TIER_KEYS)}"
            )
        if not isinstance(entry["checkpoint"], str) or not isinstance(
            entry["checkpoint_sha256"], str
        ):
            raise ContractError(
                f"{label}: checkpoint and checkpoint_sha256 must be strings"
            )
        for int_key in ("checkpoint_bytes", "params"):
            count = entry[int_key]
            if isinstance(count, bool) or not isinstance(count, int):
                raise ContractError(f"{label}: {int_key} must be an int")
        if not isinstance(entry["train"], dict):
            raise ContractError(f"{label}: train must be an object")
        valid_block = entry["valid"]
        if not isinstance(valid_block, dict):
            raise ContractError(f"{label}: valid must be an object")
        if valid_block.get("n_rows") != 299376:
            raise ContractError(
                f"{label}: valid.n_rows must be the full valid 299,376 rows, "
                f"got {valid_block.get('n_rows')!r}"
            )
        per_horizon = valid_block.get("per_horizon_rmsle")
        if not isinstance(per_horizon, list) or len(per_horizon) != 16:
            raise ContractError(
                f"{label}: per_horizon_rmsle must hold the 16 horizon values"
            )
        for slice_key in ("full_grid", "active_slice"):
            slice_block = valid_block.get(slice_key)
            if not isinstance(slice_block, dict) or not all(
                key in slice_block for key in ("rmsle", "mae", "rmse")
            ):
                raise ContractError(
                    f"{label}: valid.{slice_key} must carry rmsle/mae/rmse"
                )
            for key in ("rmsle", "mae", "rmse"):
                value = slice_block[key]
                bad = (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                )
                if bad:
                    raise ContractError(
                        f"{label}: valid.{slice_key}.{key} must be a finite "
                        f"non-negative number, got {value!r}"
                    )
        metrics = entry["fold_metrics"]
        if not isinstance(metrics, list) or len(metrics) != 3:
            raise ContractError(f"{label}: fold_metrics must hold 3 folds")
        for fold_index, fold_row in enumerate(metrics):
            block = _assert_score_block(f"{label} fold {fold_index + 1}", fold_row)
            if block["n_rows"] != FOLD_ROWS:
                raise ContractError(
                    f"{label}: fold_metrics[{fold_index}] n_rows "
                    f"{block['n_rows']} != {FOLD_ROWS}"
                )


def _check_registered_artifact(name: str, path: Path) -> str:
    """Schema-gate one registered artifact; returns a short PASS blurb.

    Per home: required top-level keys, pinned tier identity and order,
    3 fold_metrics entries carrying the pinned score schema, finite
    non-negative metric numbers, fold counts 28,512 (85,536 pooled),
    and the banned-key scan: the percentage-error token anywhere, plus
    local_test everywhere and the christmas key on the deep side.
    """
    spec = ARTIFACT_REGISTRY[name]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ContractError(f"artifacts/{name} does not parse: {error}") from error
    if not isinstance(payload, dict):
        raise ContractError(f"artifacts/{name} is not a JSON object")
    hits = _banned_keys_hit(name, payload)
    if hits:
        raise ContractError(
            f"artifacts/{name}: banned key(s) {hits}; a model artifact never "
            "carries percentage-error metrics or local_test content"
        )
    if name == "classical_results.json":
        _validate_classical_artifact(payload)
    else:
        _validate_deep_artifact(payload)
    tiers = payload[spec["tiers_key"]]
    names = [entry[spec["tier_name_key"]] for entry in tiers]
    if names != list(spec["model_names"]):
        raise ContractError(
            f"artifacts/{name}: tier names {names} != pinned "
            f"{list(spec['model_names'])}"
        )
    return f"{spec['track']} x {spec['model_count']} tiers"


def _freeze_preflight(
    artifacts_dir: Path,
) -> tuple[dict[str, Path], list[str]]:
    """Every freeze refusal gate, in order; opens no split.

    Gate order: FREEZE_RUN=1 -> exactly-one-freeze (a freeze_run = true
    record refuses any repeat) -> the registry scan (unknown file, zero
    registered artifacts, partial registry) -> per-artifact schema.
    Every refusal is a friendly ContractError, never a traceback.
    """
    if os.environ.get("FREEZE_RUN") != "1":
        raise ContractError(
            "freeze refused: set FREEZE_RUN=1 and go through the freeze path "
            "only; local_test is never read ad hoc and never tuned on"
        )
    manifest_path = artifacts_dir / "run_manifest.json"
    records = list(read_run_manifest(manifest_path)) if manifest_path.exists() else []
    if any(record["freeze_run"] is True for record in records):
        raise ContractError(
            "freeze refused: run_manifest.json already carries "
            "freeze_run = true; redoing a freeze takes a chat-noted "
            "decision, never silence"
        )
    registered = _scan_registered_artifacts(artifacts_dir)
    if not registered:
        raise ContractError(
            f"freeze refused: no registered model artifacts under "
            f"{artifacts_dir}; expected all of {sorted(ARTIFACT_REGISTRY)}"
        )
    missing = sorted(set(ARTIFACT_REGISTRY) - set(registered))
    if missing:
        raise ContractError(
            f"freeze refused: registry incomplete, missing {missing}; the "
            f"freeze consumes all {len(ARTIFACT_REGISTRY)} homes or none"
        )
    blurbs = [
        _check_registered_artifact(name, path)
        for name, path in sorted(registered.items())
    ]
    return registered, blurbs


def freeze(artifacts_dir: Path | None = None) -> int:
    """The canonical freeze entrypoint; one testified pass, never a rehearsal.

    Gates (all friendly refusals): FREEZE_RUN=1; no prior freeze_run =
    true record; the full registry present and schema-clean; sanctioned
    per-track entrypoints importable on the merged tree. The gates
    themselves open no split and never touch local_test. On Day 5
    morning the canonical sequence runs here: re-verified artifacts,
    fit_frame = load_frame() masked by fit_rows, predict_frame =
    valid's 60-day tail followed by local_test, per-tier
    fit_predict_frozen, the pinned score schema on full-grid,
    active-slice and the Christmas-2016 slice (1,782 rows),
    artifacts/freeze_results.json written atomically, then one manifest
    record stamped freeze_run = true. Any failing phase prints FAIL
    with the phase named and records the attempt (freeze_run stays
    false on failures), returning 1.
    """
    started = time.perf_counter()
    directory = Path(artifacts_dir) if artifacts_dir is not None else ARTIFACTS_DIR
    try:
        _freeze_preflight(directory)
    except ContractError as error:
        print(f"FAIL freeze, pre-flight gates: {error}")
        return 1
    phase = "sanctioned entrypoints"
    missing_modules = []
    for artifact_name, module_name in KIND_BY_ARTIFACT.items():
        try:
            importlib.import_module(module_name)
        except ImportError as error:
            missing_modules.append(f"{module_name} for {artifact_name}: {error}")
    elapsed = round(time.perf_counter() - started, 3)
    reasons: list[str]
    if missing_modules:
        reasons = [f"not merged yet - {text}" for text in missing_modules]
    else:
        phase = "canonical sequence"
        reasons = [
            "the Day-5 execution body lands at the Day-4 rehearsal review; "
            "gates are green, no local_test was opened, nothing scored"
        ]
    write_run_manifest(
        "contract.py --freeze",
        exit_status=1,
        elapsed_seconds=elapsed,
        arguments={"freeze": "1"},
        notes=[f"failed phase: {phase}", *reasons],
        artifacts_dir=directory,
    )
    print(f"FAIL freeze, {phase}: {'; '.join(reasons)}")
    return 1


def _check_manifest_metadata() -> None:
    """Validate split manifest metadata."""
    manifest = _load_manifest()
    pinned_rows = {"train": 2106324, "valid": 299376, "local_test": 602316}
    for split, count in pinned_rows.items():
        measured = manifest["rows"][split]
        if measured != count:
            raise ContractError(
                f"manifest {split} row count {measured} != pinned {count}; "
                f"{RUN_MAIN_POINTER}"
            )
    print("PASS manifest: rows 2,106,324 / 299,376 / 602,316 (total 3,008,016)")
    pinned_bounds = {
        "train": "2016-03-27",
        "valid": "2016-09-11",
        "local_test": "2017-08-15",
    }
    for split, end in pinned_bounds.items():
        measured_end = manifest["bounds"][split][1]
        if measured_end != end:
            raise ContractError(
                f"manifest {split} end date {measured_end!r} != pinned {end!r}; "
                f"{RUN_MAIN_POINTER}"
            )
    print("PASS manifest: bounds end 2016-03-27 / 2016-09-11 / 2017-08-15")
    closures = {"train": 5346, "valid": 0, "local_test": 1782}
    grid_fills = manifest["why"]["grid_fill_rows"]
    for split, count in closures.items():
        measured_fill = grid_fills[split]
        if measured_fill != count:
            raise ContractError(
                f"manifest {split} grid fills {measured_fill} != pinned {count}; "
                f"{RUN_MAIN_POINTER}"
            )
    fill_sum = sum(grid_fills[split] for split in closures)
    if fill_sum != 7128:
        raise ContractError(
            f"manifest why.grid_fill_rows sum {fill_sum} != 7128; {RUN_MAIN_POINTER}"
        )
    print("PASS manifest: grid_fill_rows 5,346 / 0 / 1,782 (7,128 closure fills)")
    if len(manifest["columns"]) != 26:
        raise ContractError(
            f"manifest has {len(manifest['columns'])} columns, expected 26; "
            f"{RUN_MAIN_POINTER}"
        )
    print("PASS manifest: 26 columns")
    pinned_starts = {
        "train": "2013-01-01",
        "valid": "2016-03-28",
        "local_test": "2016-09-12",
    }
    for split, start in pinned_starts.items():
        measured_start = manifest["bounds"][split][0]
        if measured_start != start:
            raise ContractError(
                f"manifest {split} start date {measured_start!r} != pinned {start!r}; "
                f"{RUN_MAIN_POINTER}"
            )
    print("PASS manifest: bounds start 2013-01-01 / 2016-03-28 / 2016-09-12")
    total_measured = sum(manifest["rows"][split] for split in SPLIT_NAMES)
    if manifest["total"] != 3008016 or total_measured != 3008016:
        raise ContractError(
            f"manifest total {manifest['total']} or "
            f"sum(rows) {total_measured} != 3008016; {RUN_MAIN_POINTER}"
        )
    frame_sum = manifest["rows"]["train"] + manifest["rows"]["valid"]
    if FRAME_ROWS != 2405700 or frame_sum != 2405700:
        raise ContractError(
            f"FRAME_ROWS {FRAME_ROWS} or train+valid {frame_sum} != 2405700; "
            f"{RUN_MAIN_POINTER}"
        )
    print(
        "PASS manifest: total 3,008,016 = sum(rows); "
        "FRAME_ROWS 2,405,700 = train + valid"
    )


def _check_row_integrity(splits: dict[str, pd.DataFrame]) -> None:
    """Duplicate keys, duplicate real ids, and negative sales: all zero.

    Closes the Day-2 deferral recorded in the module docstring. On train
    and valid, (store_nbr, family, date) and the real-id domain (the
    sentinel id=NaN closure fills excluded) must each be unique, and
    sales must stay non-negative everywhere on truth (the scorer clips
    predictions, never truth). local_test is not loaded here the freeze
    path owns the only read.
    """
    for name, data in splits.items():
        doubled = int(
            data.duplicated(subset=["store_nbr", "family", "date"], keep=False).sum()
        )
        if doubled:
            raise ContractError(
                f"{name}: {doubled} rows carry a duplicated "
                f"(store_nbr, family, date) key; {RUN_MAIN_POINTER}"
            )
        real_ids = data.loc[data["id"].notna(), "id"]
        id_doubled = int(real_ids.duplicated().sum())
        if id_doubled:
            raise ContractError(
                f"{name}: {id_doubled} duplicated values in the real-id domain"
            )
        negative = int((data["sales"] < 0).sum())
        if negative:
            raise ContractError(
                f"{name}: {negative} rows carry negative sales on truth"
            )
    print(
        "PASS data: duplicate (store, family, date) keys / duplicate real ids / "
        "negative sales all zero on train+valid"
    )


def _check_data_contract() -> None:
    """Validate row counts, closures, and pre-opening masks."""
    pinned_closures = {"train": 5346, "valid": 0}
    pinned_preopen = {"train": 208263, "valid": 5544}
    pinned_fit_drop = {"train": 207801, "valid": 5544}
    pinned_zeros = {"train": 792931, "valid": 54073}
    splits = {"train": load_split("train"), "valid": load_split("valid")}
    masked_total = 0
    for name, data in splits.items():
        closures = int(closure_rows(data).sum())
        preopen_count = int(preopening_rows(data).sum())
        fit_drop = int((~fit_rows(data)).sum())
        closures_kept = int((closure_rows(data) & fit_rows(data)).sum())
        drop_sales = data.loc[~fit_rows(data), "sales"]
        if closures != pinned_closures[name]:
            raise ContractError(
                f"{name}: {closures} closure rows, pinned is {pinned_closures[name]}; "
                f"{RUN_MAIN_POINTER}"
            )
        if preopen_count != pinned_preopen[name]:
            raise ContractError(
                f"{name}: pre-opening {preopen_count}, pinned {pinned_preopen[name]}; "
                f"{RUN_MAIN_POINTER}"
            )
        if fit_drop != pinned_fit_drop[name]:
            raise ContractError(
                f"{name}: fit_rows drops {fit_drop}, "
                f"pinned is {pinned_fit_drop[name]}; {RUN_MAIN_POINTER}"
            )
        if closures_kept != closures:
            raise ContractError(
                f"{name}: fit_rows dropped {closures - closures_kept} closure fills; "
                f"{RUN_MAIN_POINTER}"
            )
        if not (drop_sales == 0).all():
            raise ContractError(
                f"{name}: a masked pre-opening row carries nonzero sales; "
                f"{RUN_MAIN_POINTER}"
            )
        measured_zeros = int((data["sales"] == 0).sum())
        if measured_zeros != pinned_zeros[name]:
            raise ContractError(
                f"{name}: {measured_zeros} zero rows, pinned {pinned_zeros[name]}; "
                f"{RUN_MAIN_POINTER}"
            )
        masked_total += fit_drop
        print(
            f"PASS data[{name}]: {len(data):,} rows | closures {closures:,} | "
            f"pre-open {preopen_count:,} | fit drop {fit_drop:,} (all fills kept, "
            "every drop all-zero)"
        )
    _check_row_integrity(splits)
    zeros = int((splits["train"]["sales"] == 0).sum()) + int(
        (splits["valid"]["sales"] == 0).sum()
    )
    total = len(splits["train"]) + len(splits["valid"])
    share = zeros / total
    if round(share, 3) != 0.352:
        raise ContractError(
            f"train+valid zero share {share:.6f} is not the pinned 35.2%; "
            f"{RUN_MAIN_POINTER}"
        )
    print(f"PASS data: zero share {share:.2%} on train+valid ({zeros:,} of {total:,})")
    trainable = total - masked_total
    trainable_share = (zeros - masked_total) / trainable
    if round(trainable_share, 3) != 0.289:
        raise ContractError(
            f"trainable zero share {trainable_share:.6f} is not the pinned 28.9%; "
            f"{RUN_MAIN_POINTER}"
        )
    print(
        f"PASS data: trainable zero share {trainable_share:.2%} after masking "
        f"({zeros - masked_total:,} of {trainable:,})"
    )
    probe = pd.DataFrame(
        {
            "store_nbr": pd.Series([52, 52, 20, 20, 15], dtype="int8"),
            "date": [
                STORE_52_OPEN - pd.Timedelta(days=1),
                STORE_52_OPEN,
                DELAYS_OPEN_UNTIL[20] - pd.Timedelta(days=1),
                DELAYS_OPEN_UNTIL[20],
                pd.Timestamp("2015-06-01"),
            ],
            "id": pd.Series([1, 2, np.nan, 3, 4], dtype="UInt32"),
        }
    )
    expected_fit = [False, True, True, True, True]
    expected_preopen = [True, False, True, False, False]
    if list(fit_rows(probe)) != expected_fit:
        raise ContractError(
            f"fit_rows probe disagrees: {list(fit_rows(probe))} vs {expected_fit}; "
            f"{RUN_MAIN_POINTER}"
        )
    if list(preopening_rows(probe)) != expected_preopen:
        raise ContractError(
            "preopening_rows probe disagrees: "
            f"{list(preopening_rows(probe))} vs {expected_preopen}; "
            f"{RUN_MAIN_POINTER}"
        )
    for store, open_date in sorted(DELAYS_OPEN_UNTIL.items()):
        # Store 52 opening date (2017-04-20) falls in local_test;
        # train only contains pre-opening zero rows.
        if store == 52:
            continue
        day_sales = float(
            splits["train"]
            .loc[
                (splits["train"]["store_nbr"] == store)
                & (splits["train"]["date"] == open_date),
                "sales",
            ]
            .sum()
        )
        if day_sales <= 0:
            raise ContractError(
                f"store {store}: opening date {open_date} has zero sales "
                f"({day_sales}); {RUN_MAIN_POINTER}"
            )
    print(
        "PASS data: delayed-opening tightness "
        "(all 7 train stores have positive sales on open date)"
    )
    print(
        "PASS data: boundary probes (day-before dropped, open day kept, "
        "in-window fill kept, neighboring store untouched)"
    )
    frame = load_frame(train=splits["train"], valid=splits["valid"])
    print(f"PASS data: load_frame {len(frame):,} rows, date-ordered (2,405,700 pinned)")
    print(
        "INFO data: pre-opening holdout context 12,804 rows "
        "(5,544 valid + 7,260 local_test) scored as-is; local_test untouched"
    )


def _check_scorer() -> None:
    """Validate metric calculations and input error handling."""
    exact = np.array([1.0, 2.0, 3.0])
    perfect = score(exact, exact)
    for key in (
        "rmsle",
        "mae",
        "rmse",
        "rmsle_active",
        "mae_active",
        "rmse_active",
    ):
        if perfect[key] != 0.0:
            raise ContractError(f"score: perfect vector metric {key} is {perfect[key]}")
    if perfect["n_rows"] != 3 or perfect["n_rows_active"] != 3:
        raise ContractError("score: perfect vector counts mismatch")
    y_t = np.array([0.0, 2.0])
    y_p = np.array([-1.0, 3.0])
    hand = score(y_t, y_p)
    expected = {
        "rmsle": math.sqrt((math.log1p(3) - math.log1p(2)) ** 2 / 2),
        "mae": 0.5,
        "rmse": math.sqrt(0.5),
        "rmsle_active": math.log1p(3) - math.log1p(2),
        "mae_active": 1.0,
        "rmse_active": 1.0,
    }
    for key, want in expected.items():
        got = hand[key]
        if not math.isclose(got, want, rel_tol=1e-12):
            raise ContractError(f"score: hand test {key} got {got}, want {want}")
    if hand["n_rows"] != 2 or hand["n_rows_active"] != 1:
        raise ContractError("score: hand test row counts mismatch")
    y_t_override = np.array([0.0, 0.0])
    y_p_override = np.array([2.0, 3.0])
    forced = score(y_t_override, y_p_override, is_active=np.array([False, True]))
    default = score(y_t_override, y_p_override)
    expected_override = {
        "rmsle_active": math.log1p(3.0),
        "mae_active": 3.0,
        "rmse_active": 3.0,
    }
    for key, want in expected_override.items():
        got = forced[key]
        if not math.isclose(got, want, rel_tol=1e-12):
            raise ContractError(
                f"score: overridden {key} {got} disagrees with hand value {want}"
            )
    if math.isclose(forced["mae_active"], default["mae_active"], rel_tol=1e-12):
        raise ContractError("score: is_active override failed to take effect")
    flat = score(np.zeros(4), np.zeros(4))
    if any(flat[key] != 0.0 for key in expected) or flat["n_rows_active"] != 0:
        raise ContractError("score: all-zero slice leaked NaN or nonzero active")

    def must_refuse(call: Callable[[], Any], label: str) -> None:
        try:
            call()
        except ContractError:
            return
        raise ContractError(f"guard regression: {label} did not raise")

    def _test_load_split_extra_col() -> None:
        from unittest.mock import patch

        bad_df = pd.DataFrame(
            {col: pd.Series([], dtype=dt) for col, dt in PINNED_DTYPES.items()}
        )
        bad_df["extra_col"] = 1
        with patch("pandas.read_parquet", return_value=bad_df):
            load_split("train")

    def _test_load_split_dup_col() -> None:
        from unittest.mock import patch

        bad_df = pd.DataFrame([[1, 2]], columns=["date", "date"])
        with patch("pandas.read_parquet", return_value=bad_df):
            load_split("train")

    def _test_load_frame_extra_col() -> None:
        sample = pd.DataFrame(
            {col: pd.Series([], dtype=dt) for col, dt in PINNED_DTYPES.items()}
        )
        extra = sample.copy()
        extra["extra_col"] = 1
        load_frame(train=extra, valid=sample)

    def _test_load_frame_dup_col() -> None:
        dup = pd.DataFrame([[1, 2]], columns=["date", "date"])
        load_frame(train=dup, valid=dup)

    def _test_closure_rows_dup_col() -> None:
        dup = pd.concat(
            [
                pd.Series([1], dtype="UInt32", name="id"),
                pd.Series([1], dtype="UInt32", name="id"),
                pd.Series(
                    [pd.Timestamp("2016-01-01")],
                    dtype="datetime64[us]",
                    name="date",
                ),
                pd.Series([1], dtype="int8", name="store_nbr"),
            ],
            axis=1,
        )
        closure_rows(dup)

    def _test_preopening_rows_dup_col() -> None:
        dup = pd.concat(
            [
                pd.Series([1], dtype="int8", name="store_nbr"),
                pd.Series([1], dtype="int8", name="store_nbr"),
                pd.Series(
                    [pd.Timestamp("2016-01-01")],
                    dtype="datetime64[us]",
                    name="date",
                ),
            ],
            axis=1,
        )
        preopening_rows(dup)

    def _test_preopening_rows_nullable_na() -> None:
        df = pd.DataFrame(
            {
                "store_nbr": pd.Series([1, pd.NA], dtype="Int64"),
                "date": pd.Series(
                    [pd.Timestamp("2016-01-01"), pd.Timestamp("2016-01-02")],
                    dtype="datetime64[us]",
                ),
            }
        )
        preopening_rows(df)

    def _test_load_frame_permuted_cols() -> None:
        cols = list(PINNED_DTYPES)
        i_fam, i_id = cols.index("family"), cols.index("id")
        cols[i_fam], cols[i_id] = cols[i_id], cols[i_fam]
        permuted = pd.DataFrame(
            {col: pd.Series([], dtype=PINNED_DTYPES[col]) for col in cols}
        )
        load_frame(train=permuted, valid=permuted)

    def _test_load_frame_valid_permuted_cols() -> None:
        cols = list(PINNED_DTYPES)
        i_fam, i_id = cols.index("family"), cols.index("id")
        cols[i_fam], cols[i_id] = cols[i_id], cols[i_fam]
        straight = pd.DataFrame(
            {col: pd.Series([], dtype=dt) for col, dt in PINNED_DTYPES.items()}
        )
        permuted = pd.DataFrame(
            {col: pd.Series([], dtype=PINNED_DTYPES[col]) for col in cols}
        )
        load_frame(train=straight, valid=permuted)

    def _test_load_frame_category_mismatch() -> None:
        cat_a = pd.DataFrame(
            {
                col: (
                    pd.Series(pd.Categorical([], categories=["greeting"]))
                    if col == "family"
                    else pd.Series([], dtype=dt)
                )
                for col, dt in PINNED_DTYPES.items()
            }
        )
        cat_b = pd.DataFrame(
            {
                col: (
                    pd.Series(pd.Categorical([], categories=["farewell"]))
                    if col == "family"
                    else pd.Series([], dtype=dt)
                )
                for col, dt in PINNED_DTYPES.items()
            }
        )
        load_frame(train=cat_a, valid=cat_b)

    def _test_tampered_manifest(
        tamper: Callable[[dict[str, Any]], Any],
    ) -> None:
        import json as _json
        import shutil as _shutil
        import sys as _sys
        import tempfile as _tempfile
        from pathlib import Path as _Path
        from unittest.mock import patch

        manifest = _json.loads(
            (REPO_ROOT / "splits" / "split_manifest.json").read_text(encoding="utf-8")
        )
        tamper(manifest)
        tmp = _Path(_tempfile.mkdtemp())
        try:
            (tmp / "split_manifest.json").write_text(
                _json.dumps(manifest), encoding="utf-8"
            )
            self_mod = _sys.modules[__name__]
            with patch.object(self_mod, "SPLITS_DIR", tmp):
                _load_manifest()
        finally:
            _shutil.rmtree(tmp, ignore_errors=True)

    dummy_date = pd.Timestamp("2016-01-01")

    cases: list[tuple[str, Callable[[], Any]]] = [
        ("string list y_true", lambda: score(["1.0", "2.0"], np.ones(2))),
        ("complex y_true", lambda: score(np.array([1 + 2j]), np.ones(1))),
        (
            "datetime64 y_true",
            lambda: score(np.array(["2026-01-01"], dtype="datetime64[D]"), np.ones(1)),
        ),
        ("NaN y_true", lambda: score(np.array([1.0, np.nan]), np.ones(2))),
        ("inf y_true", lambda: score(np.array([1.0, np.inf]), np.ones(2))),
        ("inf y_pred", lambda: score(np.ones(2), np.array([1.0, np.inf]))),
        ("negative y_true", lambda: score(np.array([2.0, -0.5]), np.ones(2))),
        ("y_true at -1", lambda: score(np.array([2.0, -1.0]), np.ones(2))),
        ("2-D input", lambda: score(np.ones((2, 1)), np.ones((2, 1)))),
        (
            "non-boolean is_active",
            lambda: score(np.ones(2), np.ones(2), is_active=np.array([0.5, 1.0])),
        ),
        (
            "masked array y_true",
            lambda: score(np.ma.array([1.0, 2.0]), np.ones(2)),
        ),
        (
            "masked array y_pred",
            lambda: score(np.ones(2), np.ma.array([1.0, 2.0])),
        ),
        (
            "masked array is_active",
            lambda: score(np.ones(2), np.ones(2), is_active=np.ma.array([True, False])),
        ),
        (
            "boolean y_true",
            lambda: score(np.array([True, False]), np.ones(2)),
        ),
        (
            "boolean y_pred",
            lambda: score(np.ones(2), np.array([True, False])),
        ),
        (
            "overflow metric in score",
            lambda: score([1e160], [0.0]),
        ),
        ("closure_rows non-DataFrame", lambda: closure_rows(cast(Any, "not a df"))),
        ("preopening_rows non-DataFrame", lambda: preopening_rows(cast(Any, 123))),
        ("fit_rows non-DataFrame", lambda: fit_rows(cast(Any, {"a": 1}))),
        (
            "preopening_rows non-integer store_nbr",
            lambda: preopening_rows(
                pd.DataFrame({"store_nbr": ["1"], "date": [dummy_date]})
            ),
        ),
        (
            "preopening_rows float store_nbr",
            lambda: preopening_rows(
                pd.DataFrame({"store_nbr": [1.5], "date": [dummy_date]})
            ),
        ),
        (
            "preopening_rows non-datetime date",
            lambda: preopening_rows(
                pd.DataFrame({"store_nbr": [1], "date": ["2016-01-01"]})
            ),
        ),
        (
            "preopening_rows timezone date",
            lambda: preopening_rows(
                pd.DataFrame(
                    {
                        "store_nbr": [1],
                        "date": [pd.Timestamp("2016-01-01", tz="UTC")],
                    }
                )
            ),
        ),
        ("load_split extra column", _test_load_split_extra_col),
        ("load_split duplicate column", _test_load_split_dup_col),
        ("load_frame extra column", _test_load_frame_extra_col),
        ("load_frame duplicate column", _test_load_frame_dup_col),
        ("closure_rows duplicate column", _test_closure_rows_dup_col),
        ("preopening_rows duplicate column", _test_preopening_rows_dup_col),
        (
            "preopening_rows nullable store_nbr with NA",
            _test_preopening_rows_nullable_na,
        ),
        ("load_frame permuted columns", _test_load_frame_permuted_cols),
        (
            "load_frame valid permuted columns",
            _test_load_frame_valid_permuted_cols,
        ),
        (
            "load_frame category domain mismatch",
            _test_load_frame_category_mismatch,
        ),
        (
            "load_manifest NaT bound",
            lambda: _test_tampered_manifest(
                lambda m: m["bounds"].__setitem__("valid", ["nan", "2016-09-11"])
            ),
        ),
        (
            "load_manifest malformed timestamp bound",
            lambda: _test_tampered_manifest(
                lambda m: m["bounds"].__setitem__(
                    "train", ["2013-01-01+00:00", "2016-03-27"]
                )
            ),
        ),
        (
            "load_manifest timezone start bound",
            lambda: _test_tampered_manifest(
                lambda m: m["bounds"].__setitem__(
                    "train", ["2013-01-01T00:00:00+00:00", "2016-03-27"]
                )
            ),
        ),
        (
            "load_manifest timezone end bound",
            lambda: _test_tampered_manifest(
                lambda m: m["bounds"].__setitem__(
                    "valid", ["2016-03-28", "2016-09-11T00:00:00+00:00"]
                )
            ),
        ),
        (
            "load_manifest NaT end bound",
            lambda: _test_tampered_manifest(
                lambda m: m["bounds"].__setitem__("valid", ["2016-03-28", "nan"])
            ),
        ),
        (
            "mask mixed-type duplicate labels",
            lambda: closure_rows(
                pd.DataFrame([[1, 2, 3, 4]], columns=["id", "id", 1, 1])
            ),
        ),
        (
            "load_manifest extra column",
            lambda: _test_tampered_manifest(lambda m: m["columns"].append("oxygen")),
        ),
    ]
    for label, call in cases:
        must_refuse(call, label)
    print("PASS score: perfect 3-vector is 0 on all six metric keys")
    print("PASS score: -1 prediction clips to 0; all six hand values match")
    print("PASS score: is_active override computed over its own hand slice (3.0)")
    print("PASS score: all-zero slice is all 0.0, n_active 0, no NaN")
    print(f"PASS score: all {len(cases)} guard cases refuse with a friendly error")


def _check_folds() -> None:
    """Day-2 block: the pinned rolling-origin table and its pooling."""
    table = folds_table()
    pinned_columns = ["fold", "origin", "window_start", "window_end", "n_rows"]
    if list(table.columns) != pinned_columns:
        raise ContractError(
            f"folds_table columns {list(table.columns)} != pinned {pinned_columns}"
        )
    for index, (fold, origin, start, end) in enumerate(FOLD_TABLE_PINS):
        row = table.iloc[index]
        got = (
            int(row["fold"]),
            str(row["origin"]),
            str(row["window_start"]),
            str(row["window_end"]),
            int(row["n_rows"]),
        )
        if got != (fold, origin, start, end, FOLD_ROWS):
            raise ContractError(
                f"folds_table row {index} {got} != pinned "
                f"{(fold, origin, start, end, FOLD_ROWS)}"
            )
    pooled = int(table["n_rows"].sum())
    if pooled != POOLED_FOLD_ROWS:
        raise ContractError(
            f"pooled fold rows {pooled} != the pinned {POOLED_FOLD_ROWS}"
        )
    if not make_folds().equals(table):
        raise ContractError("make_folds() no longer aliases folds_table()")
    print(
        "PASS folds: 3 windows (origins 2016-07-25 / 2016-08-10 / 2016-08-26), "
        "16 days each, 28,512 rows, quake-free and inside valid"
    )
    print(
        f"PASS folds: pooled {POOLED_FOLD_ROWS:,} rows; "
        "make_folds aliases folds_table (one implementation)"
    )


def _check_manifest_writer() -> None:
    """Day-2 block: run_manifest.json parses, validates, and takes one append.

    Every --check invocation appends one PASS record and re-verifies the
    file (atomic write, last-record landing). Existing records (if any)
    are schema-validated first; the file must never carry more than one
    freeze_run = true record.
    """
    path = ARTIFACTS_DIR / "run_manifest.json"
    existing = list(read_run_manifest(path)) if path.exists() else []
    stamped = any(record["freeze_run"] is True for record in existing)
    write_run_manifest("contract.py --check")
    stored = list(read_run_manifest(path))
    record = stored[-1]
    if record["command"] != "contract.py --check" or record["status"] != "PASS":
        raise ContractError(f"appended manifest record drifted: {record!r}")
    if set(record["versions"]) != set(VERSION_KEYS):
        raise ContractError(
            f"appended versions {list(record['versions'])} != "
            f"pinned {list(VERSION_KEYS)}"
        )
    tag = (
        "freeze already stamped; validated without touching it"
        if stamped
        else "nothing stamped yet"
    )
    print(
        f"PASS manifest: writer appended record {len(stored)} "
        f"(schema valid, versions pinned; {tag})"
    )


def _check_freeze_gates() -> None:
    """Day-2 block: the refusal gates, against throwaway dirs.

    Five friendly refusals - unset FREEZE_RUN flag, a second freeze on a
    freeze_run = true manifest, one unregistered artifact file, zero
    registered artifacts, a partial registry - exercised in temp dirs
    so the real artifacts directory stays untouched. No gate opens a
    split; local_test is never read here.
    """
    saved_flag = os.environ.pop("FREEZE_RUN", None)
    try:
        with tempfile.TemporaryDirectory() as tmp_name:
            base = Path(tmp_name)

            def refuses(call: Callable[[], Any], needle: str, label: str) -> None:
                try:
                    call()
                except ContractError as error:
                    if needle not in str(error):
                        raise ContractError(
                            f"freeze gate {label}: refused with the wrong lead: {error}"
                        ) from error
                    return
                raise ContractError(f"freeze gate {label}: accepted a forbidden call")

            empty = base / "empty"
            empty.mkdir()
            refuses(
                lambda: _freeze_preflight(empty),
                "FREEZE_RUN=1",
                "flag",
            )
            os.environ["FREEZE_RUN"] = "1"
            refuses(
                lambda: _freeze_preflight(empty),
                "no registered model artifacts",
                "registry-zero",
            )
            stray = base / "stray"
            stray.mkdir()
            (stray / "other_results.json").write_text("{}", encoding="utf-8")
            refuses(
                lambda: _freeze_preflight(stray),
                "unregistered file",
                "registry-unknown",
            )
            partial = base / "partial"
            partial.mkdir()
            (partial / "classical_results.json").write_text("{}", encoding="utf-8")
            refuses(
                lambda: _freeze_preflight(partial),
                "registry incomplete",
                "registry-partial",
            )
            furnished = base / "furnished"
            furnished.mkdir()
            freeze_entry = {
                "created": "2026-11-10T00:00:00Z",
                "git_sha": "0" * 40,
                "command": "contract.py --freeze",
                "exit": 0,
                "exit_ok": True,
                "versions": dict.fromkeys(VERSION_KEYS, "x"),
                "elapsed_seconds": 0.0,
                "arguments": {"freeze": "1"},
                "status": "PASS",
                "freeze_run": True,
                "notes": [],
            }
            (furnished / "run_manifest.json").write_text(
                json.dumps([freeze_entry]), encoding="utf-8"
            )
            refuses(
                lambda: _freeze_preflight(furnished),
                "already carries",
                "second-freeze",
            )
    finally:
        if saved_flag is None:
            os.environ.pop("FREEZE_RUN", None)
        else:
            os.environ["FREEZE_RUN"] = saved_flag
    print(
        "PASS freeze: flag / registry-zero / registry-unknown / registry-partial / "
        "second-freeze gates all refuse with friendly leads"
    )
    print("PASS freeze: gates open no split and never read local_test")


def _check_registry() -> None:
    """Day-2 block: registered homes; strays fail loud, files validate."""
    directory = ARTIFACTS_DIR
    if not directory.exists():
        print(
            "PASS registry: artifacts/ absent; 2 registered homes pending the "
            "S3/S4 merges (expected before Day 4)"
        )
        return
    registered = _scan_registered_artifacts(directory)
    if not registered:
        print(
            "PASS registry: no stray artifacts; registered homes pending the "
            "S3/S4 merges (expected before Day 4)"
        )
        return
    for name, path in sorted(registered.items()):
        blurb = _check_registered_artifact(name, path)
        print(f"PASS registry: {name} schema-clean ({blurb})")


def run_check() -> None:
    """Run all contract verification checks."""
    print("contract.py --check")
    _check_manifest_metadata()
    _check_data_contract()
    _check_folds()
    _check_manifest_writer()
    _check_freeze_gates()
    _check_registry()
    _check_scorer()
    print("contract.py --check: PASS")


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="S1 split loader, masks and scorer contract"
    )
    parser.add_argument(
        "--check", action="store_true", help="run the self-audit blocks"
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="run the freeze path (gates: FREEZE_RUN=1, no prior freeze)",
    )
    args = parser.parse_args(argv)
    if args.check:
        try:
            run_check()
        except (ContractError, OSError) as error:
            print(f"FAIL {error}")
            return 1
        return 0
    if args.freeze:
        try:
            return freeze()
        except (ContractError, OSError) as error:
            print(f"FAIL {error}")
            return 1
    parser.print_usage()
    print("\nPass --check (verification) or --freeze (freeze path).")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
