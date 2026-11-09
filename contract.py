"""Dataset split loaders, truth filtering masks, and RMSLE evaluation scorer."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa  # pyright: ignore[reportMissingTypeStubs]
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
    "load_frame",
    "load_split",
    "preopening_rows",
    "score",
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
    for split in SPLIT_NAMES:
        if split not in manifest["bounds"]:
            raise ContractError(
                f"manifest bounds missing split {split!r}; {RUN_MAIN_POINTER}"
            )
        split_bounds = manifest["bounds"][split]
        if not isinstance(split_bounds, (list, tuple)) or len(split_bounds) != 2:
            raise ContractError(
                f"manifest bounds[{split!r}] must be a 2-element sequence; "
                f"{RUN_MAIN_POINTER}"
            )
        try:
            start_ts = pd.Timestamp(split_bounds[0])
        except Exception as error:
            raise ContractError(
                f"manifest bounds[{split!r}] start timestamp invalid "
                f"({split_bounds[0]!r}): {error}; {RUN_MAIN_POINTER}"
            ) from error
        try:
            end_ts = pd.Timestamp(split_bounds[1])
        except Exception as error:
            raise ContractError(
                f"manifest bounds[{split!r}] end timestamp invalid "
                f"({split_bounds[1]!r}): {error}; {RUN_MAIN_POINTER}"
            ) from error
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
    frame_rows_sum = int(manifest["rows"]["train"]) + int(manifest["rows"]["valid"])
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


def score(
    y_true: np.ndarray | Any,
    y_pred: np.ndarray | Any,
    is_active: np.ndarray | Any | None = None,
) -> dict[str, float | int]:
    """Compute RMSLE, MAE, and RMSE on full and active demand slices.

    Predictions are clipped to non-negative values before computing metrics.
    When n_rows_active == 0, active metrics return 0.0.
    """
    for name, raw in (("y_true", y_true), ("y_pred", y_pred)):
        try:
            arr = np.asarray(raw)
        except Exception as error:
            raise ContractError(
                f"score: {name} could not be converted to array: {error}"
            ) from error
        if arr.dtype.kind not in "fiub":
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
        active = np.asarray(is_active)
        if active.dtype != np.bool_:
            raise ContractError(
                f"score: is_active must be boolean mask; got dtype {active.dtype}"
            )
        if active.shape != truth.shape:
            raise ContractError(
                f"score: is_active shape {active.shape} != {truth.shape}"
            )
    result: dict[str, float | int] = {
        "rmsle": float(root_mean_squared_log_error(truth, pred)),
        "mae": float(mean_absolute_error(truth, pred)),
        "rmse": float(root_mean_squared_error(truth, pred)),
    }
    n_active = int(active.sum())
    if n_active:
        sub_true = truth[active]
        sub_pred = pred[active]
        result["rmsle_active"] = float(root_mean_squared_log_error(sub_true, sub_pred))
        result["mae_active"] = float(mean_absolute_error(sub_true, sub_pred))
        result["rmse_active"] = float(root_mean_squared_error(sub_true, sub_pred))
    else:
        result["rmsle_active"] = 0.0
        result["mae_active"] = 0.0
        result["rmse_active"] = 0.0
    result["n_rows"] = int(truth.size)
    result["n_rows_active"] = n_active
    return result


def _check_manifest_metadata() -> None:
    """Validate split manifest metadata."""
    manifest = _load_manifest()
    pinned_rows = {"train": 2106324, "valid": 299376, "local_test": 602316}
    for split, count in pinned_rows.items():
        measured = int(manifest["rows"][split])
        if measured != count:
            raise ContractError(
                f"manifest {split} row count {measured} != pinned {count}"
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
                f"manifest {split} end date {measured_end} != pinned {end}"
            )
    print("PASS manifest: bounds end 2016-03-27 / 2016-09-11 / 2017-08-15")
    closures = {"train": 5346, "valid": 0, "local_test": 1782}
    grid_fills = manifest["why"]["grid_fill_rows"]
    for split, count in closures.items():
        measured_fill = int(grid_fills[split])
        if measured_fill != count:
            raise ContractError(
                f"manifest {split} grid fills {measured_fill} != pinned {count}"
            )
    fill_sum = sum(int(grid_fills[split]) for split in closures)
    if fill_sum != 7128:
        raise ContractError(f"manifest why.grid_fill_rows sum {fill_sum} != 7128")
    print("PASS manifest: grid_fill_rows 5,346 / 0 / 1,782 (7,128 closure fills)")
    if len(manifest["columns"]) != 26:
        raise ContractError(
            f"manifest has {len(manifest['columns'])} columns, expected 26"
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
                f"manifest {split} start date {measured_start} != pinned {start}"
            )
    print("PASS manifest: bounds start 2013-01-01 / 2016-03-28 / 2016-09-12")
    total_measured = sum(int(manifest["rows"][split]) for split in SPLIT_NAMES)
    if int(manifest["total"]) != 3008016 or total_measured != 3008016:
        raise ContractError(
            f"manifest total {manifest['total']} or "
            f"sum(rows) {total_measured} != 3008016"
        )
    frame_sum = int(manifest["rows"]["train"]) + int(manifest["rows"]["valid"])
    if FRAME_ROWS != 2405700 or frame_sum != 2405700:
        raise ContractError(
            f"FRAME_ROWS {FRAME_ROWS} or train+valid {frame_sum} != 2405700"
        )
    print(
        "PASS manifest: total 3,008,016 = sum(rows); "
        "FRAME_ROWS 2,405,700 = train + valid"
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
                f"{name}: {closures} closure rows, pinned is {pinned_closures[name]}"
            )
        if preopen_count != pinned_preopen[name]:
            raise ContractError(
                f"{name}: pre-opening {preopen_count}, pinned {pinned_preopen[name]}"
            )
        if fit_drop != pinned_fit_drop[name]:
            raise ContractError(
                f"{name}: fit_rows drops {fit_drop}, pinned is {pinned_fit_drop[name]}"
            )
        if closures_kept != closures:
            raise ContractError(
                f"{name}: fit_rows dropped {closures - closures_kept} closure fills"
            )
        if not (drop_sales == 0).all():
            raise ContractError(
                f"{name}: a masked pre-opening row carries nonzero sales"
            )
        measured_zeros = int((data["sales"] == 0).sum())
        if measured_zeros != pinned_zeros[name]:
            raise ContractError(
                f"{name}: {measured_zeros} zero rows, pinned {pinned_zeros[name]}"
            )
        masked_total += fit_drop
        print(
            f"PASS data[{name}]: {len(data):,} rows | closures {closures:,} | "
            f"pre-open {preopen_count:,} | fit drop {fit_drop:,} (all fills kept, "
            "every drop all-zero)"
        )
    zeros = int((splits["train"]["sales"] == 0).sum()) + int(
        (splits["valid"]["sales"] == 0).sum()
    )
    total = len(splits["train"]) + len(splits["valid"])
    share = zeros / total
    if round(share, 3) != 0.352:
        raise ContractError(
            f"train+valid zero share {share:.6f} is not the pinned 35.2%"
        )
    print(f"PASS data: zero share {share:.2%} on train+valid ({zeros:,} of {total:,})")
    trainable = total - masked_total
    trainable_share = (zeros - masked_total) / trainable
    if round(trainable_share, 3) != 0.289:
        raise ContractError(
            f"trainable zero share {trainable_share:.6f} is not the pinned 28.9%"
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
            f"fit_rows probe disagrees: {list(fit_rows(probe))} vs {expected_fit}"
        )
    if list(preopening_rows(probe)) != expected_preopen:
        raise ContractError(
            "preopening_rows probe disagrees: "
            f"{list(preopening_rows(probe))} vs {expected_preopen}"
        )
    for store, open_date in sorted(DELAYS_OPEN_UNTIL.items()):
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
                f"store {store}: opening date {open_date} has zero sales ({day_sales})"
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
        raise ContractError(f"score guard regression: {label} did not raise")

    must_refuse(lambda: score(["1.0", "2.0"], np.ones(2)), "string list y_true")
    must_refuse(lambda: score(np.array([1 + 2j]), np.ones(1)), "complex y_true")
    must_refuse(
        lambda: score(np.array(["2026-01-01"], dtype="datetime64[D]"), np.ones(1)),
        "datetime64 y_true",
    )
    must_refuse(lambda: score(np.array([1.0, np.nan]), np.ones(2)), "NaN y_true")
    must_refuse(lambda: score(np.array([1.0, np.inf]), np.ones(2)), "inf y_true")
    must_refuse(lambda: score(np.ones(2), np.array([1.0, np.inf])), "inf y_pred")
    must_refuse(lambda: score(np.array([2.0, -0.5]), np.ones(2)), "negative y_true")
    must_refuse(lambda: score(np.array([2.0, -1.0]), np.ones(2)), "y_true at -1")
    must_refuse(lambda: score(np.ones((2, 1)), np.ones((2, 1))), "2-D input")
    must_refuse(
        lambda: score(np.ones(2), np.ones(2), is_active=np.array([0.5, 1.0])),
        "non-boolean is_active",
    )
    print("PASS score: perfect 3-vector is 0 on all six metric keys")
    print("PASS score: -1 prediction clips to 0; all six hand values match")
    print("PASS score: is_active override computed over its own hand slice (3.0)")
    print("PASS score: all-zero slice is all 0.0, n_active 0, no NaN")
    print("PASS score: all seven guard cases refuse with a friendly error")


def run_check() -> None:
    """Run all contract verification checks."""
    print("contract.py --check")
    _check_manifest_metadata()
    _check_data_contract()
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
    args = parser.parse_args(argv)
    if args.check:
        try:
            run_check()
        except (ContractError, OSError) as error:
            print(f"FAIL {error}")
            return 1
        return 0
    parser.print_usage()
    print("\nPass --check to run contract verification.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
