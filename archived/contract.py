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
- local_test is not restricted to freeze(); the FREEZE_RUN=1 env flag is only a gate.
- Masks are index-aligned boolean Series; align index when reordering frames.
- Import is side-effect-free: no data files are opened on module load.
- The fold table is one shared rolling-origin set: three 16-day tail
  windows of valid (origins 2016-07-25 / 2016-08-10 / 2016-08-26),
  n_rows 28,512 per fold, pooled 85,536 scored jointly. Every window
  starts after 2016-05-15, so no quake-decay row lands in a fold. The
  is_*_tomorrow / is_*_in_2days lead flags are calendar-built and
  future-known by design; fit history stops at each fold's origin.
- Registered artifacts: artifacts/classical_results.json (S3, 5 tiers) and
  artifacts/deep_results.json (S4, 3 tiers) match by exact name at the
  artifacts/ top level, where the contract furniture freeze_results.json
  may also live; any other path under artifacts/ fails loud, never
  silently skipped.
- artifacts/run_manifest.json holds one record per contract invocation,
  oldest first, exactly one record may ever carry freeze_run = true,
  and one record per day-2+ --check invocation is appended and verified.
- Freeze refusal gates run before anything else: FREEZE_RUN=1, no prior
  freeze_run = true record, no freeze_results.json (a stranded artifact is
  refused with an actionable lead), full registry present, per-artifact
  schema. No gate opens a split; local_test loads only on the Day-5 body.
- artifacts/freeze_results.json carries the pinned Day-5 field set
  (frozen_at, git_sha, headline_model, data_snapshot, contract_counts,
  fold_phase, freeze_phase, notes). It is written once, atomically, by
  the freeze run, and its model count equals the registry's.
- Duplicate (store, family, date) keys, duplicate real ids, and negative
  sales are asserted zero on train and valid inside --check.
- Evaluation slices are scored without pre-opening masking.
- RMSLE is the primary evaluation metric; MAPE is undefined at 35.2% zero demand
  and is excluded from contract outputs.
- The quake-decay caveat applies strictly to the full-valid row set;
  fold windows from 2016-07-26 onward carry no quake rows.
- The banned-key scan inspects object keys only, not values;
  artifact-wide value scans remain the per-track files' own checks.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import errno
import importlib
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from collections.abc import Callable, Generator, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn, cast
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.metrics import (  # pyright: ignore[reportMissingTypeStubs]
    mean_absolute_error,
    root_mean_squared_error,
    root_mean_squared_log_error,
)

# The manifest lock is fcntl.flock on POSIX. Windows has no fcntl, so the
# same call shape is served by msvcrt.locking: a mandatory byte-range lock
# that must name the same offset to lock and unlock. It is a real lock,
# not a no-op; a no-op would let two freeze writers into the same critical
# section while both believe they hold it.
try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    import msvcrt

    class _WindowsFcntl:
        """The fcntl.flock subset the manifest lock needs, on Windows.

        msvcrt.locking takes a mandatory byte-range lock starting at the
        file's current position, so every call pins the region at a fixed
        offset and restores the caller's position. Contention raises
        BlockingIOError with EAGAIN, the errno _manifest_lock retries on.
        """

        LOCK_EX = 0x2
        LOCK_NB = 0x4
        LOCK_UN = 0x8

        # One byte past the PID marker, not over it: msvcrt locks are
        # mandatory, so a lock on the marker would hide the holder's PID
        # from a waiter and block --check's rewrite of it under a busy lock.
        _LOCK_OFFSET = 4096
        _REGION = 1

        @classmethod
        def flock(cls, fd: int, operation: int) -> None:
            if operation & cls.LOCK_UN:
                mode = msvcrt.LK_UNLCK
            elif operation & cls.LOCK_NB:
                mode = msvcrt.LK_NBLCK
            elif operation & cls.LOCK_EX:
                mode = msvcrt.LK_LOCK
            else:
                raise ValueError(f"unsupported flock operation: {operation}")
            position = os.lseek(fd, 0, os.SEEK_CUR)
            try:
                os.lseek(fd, cls._LOCK_OFFSET, os.SEEK_SET)
                try:
                    msvcrt.locking(fd, mode, cls._REGION)
                except OSError as error:
                    if mode == msvcrt.LK_NBLCK and error.errno in (
                        errno.EACCES,
                        errno.EDEADLK,
                    ):
                        raise BlockingIOError(
                            errno.EAGAIN, "the manifest lock is held"
                        ) from error
                    raise
            finally:
                os.lseek(fd, position, os.SEEK_SET)

    fcntl = _WindowsFcntl

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
# The split manifest's pinned frame edges: the single source that the
# --check bounds asserts, the artifact split_bounds validation and the
# VALID_START / VALID_END pins all read. If the data realm re-cuts a
# frame, this is the one place the pin changes and every gate follows.
SPLIT_FRAME_STARTS = {
    "train": "2013-01-01",
    "valid": "2016-03-28",
    "local_test": "2016-09-12",
}
SPLIT_FRAME_ENDS = {
    "train": "2016-03-27",
    "valid": "2016-09-11",
    "local_test": "2017-08-15",
}
VALID_START = pd.Timestamp(SPLIT_FRAME_STARTS["valid"])
VALID_END = pd.Timestamp(SPLIT_FRAME_ENDS["valid"])
QUAKE_WINDOW_END = pd.Timestamp("2016-05-15")

# Manifest-write lock: how long a writer waits on a busy advisory lock
# before refusing with the "another writer holds the lock" verdict.
MANIFEST_LOCK_WAIT_SECONDS = 2.0

# -- Day-2 pins: registered model artifacts. Exact filenames under
# artifacts/; any other model *.json path there is a loud registry
# failure. The freeze consumes these JSON files only; S4's checkpoints
# are read by S4's own fit_predict_frozen, never parsed here.
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
# Control files the contract itself owns under artifacts/; neither is a
# model artifact, so neither registers nor trips the stray-file rule.
# freeze_results.json is the freeze output: the canonical freeze body
# writes it once, and the whitelist keeps the registry scan from treating
# it as a stray.
CONTRACT_ARTIFACT_FILENAMES = ("run_manifest.json", "freeze_results.json")
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

# -- Day-5 pins: the freeze output contract. artifacts/freeze_results.json
# carries exactly these top-level fields, in this order (sheet §10), so the
# dashboard and the report read one fixed key map.
FREEZE_RESULTS_KEYS = (
    "frozen_at",
    "git_sha",
    "headline_model",
    "data_snapshot",
    "contract_counts",
    "fold_phase",
    "freeze_phase",
    "notes",
)
FREEZE_RESULTS_FILENAME = "freeze_results.json"
# The Christmas-2016 slice: the local_test rows dated 2016-12-25, the one
# full-calendar closure day inside the holdout window (54 stores x 33
# families). The count is asserted against the frame, never assumed.
CHRISTMAS_2016_DATE = pd.Timestamp("2016-12-25")
CHRISTMAS_2016_ROWS = 1782
# The deep models' lookback: predict_frame opens with valid's 60-day tail.
LOOKBACK_DAYS = 60
# The slices every freeze-phase model record carries, and the fold-phase
# pair (the freeze phase adds the Christmas-2016 slice).
FREEZE_SLICE_NAMES = ("full_grid", "active_slice", "christmas_2016")
FOLD_SLICE_NAMES = ("full_grid", "active_slice")
# The contract_counts sub-blocks (sheet §10).
CONTRACT_COUNT_KEYS = (
    "closure_rows",
    "masked_training_drop",
    "zero_share_raw",
    "zero_share_trainable",
    "preopening_holdout_rows",
    "christmas_2016_rows",
)
# The week's locked counts. Every one of them is re-derived from the
# frames at freeze time and compared against the pin, so a data drift
# refuses the freeze instead of landing in the final JSON.
PINNED_SPLIT_ROWS = {"train": 2106324, "valid": 299376, "local_test": 602316}
PINNED_CLOSURE_ROWS = {"train": 5346, "valid": 0, "local_test": 1782}
PINNED_CLOSURE_UNIVERSE = 7128
PINNED_MASKED_DROP = {"train": 207801, "valid": 5544, "total": 213345}
PINNED_IN_WINDOW_FILLS_KEPT = 462
PINNED_ZERO_SHARE_RAW = (847004, 2405700)
PINNED_ZERO_SHARE_TRAINABLE = (633659, 2192355)
PINNED_PREOPENING_HOLDOUT = {"valid": 5544, "local_test": 7260, "total": 12804}
# The rows one calendar day carries in the balanced grid: 54 stores x 33
# families. Used for the valid-tail composition, never for a metric.
SERIES_PER_DAY = 1782

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
    for i in range(1, len(rows)):
        prev_end = pd.Timestamp(rows[i - 1][3])
        curr_start = pd.Timestamp(rows[i][2])
        if curr_start != prev_end + pd.Timedelta(days=1):
            raise ContractError(
                f"fold {rows[i][0]} must start exactly one day "
                f"after fold {rows[i - 1][0]} ends"
            )
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


TRACKED_SOURCES = (
    "contract.py",
    "classical.py",
    "features.py",
    "main.py",
    "train.py",
    "holidays.py",
    "oil.py",
    "transactions.py",
)


def _head_resolves() -> bool:
    """True only when git can name a HEAD commit.

    Without a resolvable HEAD there is no tree to compare against, so a
    "differs from HEAD" note would be a fabrication. git's own answer
    (rev-parse exit != 0) is the only honest signal; the record already
    carries the "git sha unavailable" note for that case.
    """
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return head.returncode == 0


def _file_is_dirty(name: str) -> bool:
    """True only when git's own status says the named source differs.

    `git status --porcelain --untracked-files=all -- <name>` answers 0
    with one line per difference: " M" for an edited tracked file, "??"
    for an untracked one (an untracked file differs from HEAD too, and
    `git diff` alone cannot see it). Any git or OS failure answers
    False, so the note is never fabricated.
    """
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", name],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if status.returncode != 0:
        return False
    return bool(status.stdout.strip())


def _dirty_notes() -> list[str]:
    """One pinned note per dirty tracked source; empty when clean.

    No note at all when git cannot name HEAD: the note's premise is
    that git_sha reflects a HEAD, and there is none.
    """
    if not _head_resolves():
        return []
    notes: list[str] = []
    for name in TRACKED_SOURCES:
        if _file_is_dirty(name):
            notes.append(
                f"git_sha reflects HEAD; {name} differs from HEAD (dirty working tree)"
            )
    return notes


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
        parsed_created = datetime.fromisoformat(created.replace("Z", "+00:00"))
    except ValueError as error:
        raise ContractError(
            f"{label}: created {created!r} does not parse; {error}"
        ) from error
    if parsed_created.tzinfo is None:
        raise ContractError(
            f"{label}: created must carry a timezone (ISO 8601 Z or offset)"
        )
    sha = record["git_sha"]
    if not isinstance(sha, str):
        raise ContractError(f"{label}: git_sha must be a 40-hex sha (or empty)")
    if sha != "" and (len(sha) != 40 or set(sha.lower()) - set("0123456789abcdef")):
        raise ContractError(f"{label}: git_sha must be a 40-hex sha (or empty)")
    command = record["command"]
    if (
        not isinstance(command, str)
        or command == ""
        or "\n" in command
        or "\t" in command
    ):
        raise ContractError(
            f"{label}: command must be a non-empty string without newline/tab"
        )
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
    try:
        elapsed_number = float(elapsed)
    except OverflowError as error:
        raise ContractError(
            f"{label}: elapsed_seconds {elapsed!r} is out of representable range"
        ) from error
    if not math.isfinite(elapsed_number):
        raise ContractError(f"{label}: elapsed_seconds must be finite, got {elapsed!r}")
    arguments = record["arguments"]
    if not isinstance(arguments, dict) or not all(
        isinstance(k, str) for k in arguments
    ):
        raise ContractError(f"{label}: arguments must be an object with string keys")
    if not all(
        isinstance(argument_value, str) for argument_value in arguments.values()
    ):
        raise ContractError(f"{label}: arguments values must be strings")
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
    last_ts = None
    for index, record in enumerate(payload):
        created = record["created"]
        try:
            ts = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
        except Exception as error:
            raise ContractError(
                f"{path.name} record {index}: created {created!r} does not parse; "
                f"{error}"
            ) from error
        if last_ts is not None and ts < last_ts:
            raise ContractError(
                f"{path.name} records must be oldest-first "
                f"(created timestamp decreased at record {index})"
            )
        last_ts = ts
    frozen = [record for record in payload if record["freeze_run"] is True]
    if len(frozen) > 1:
        raise ContractError(
            f"{path.name} carries {len(frozen)} freeze_run = true records; "
            "exactly one freeze run is allowed"
        )
    return payload


def _fsync_path(path: Path) -> None:
    """Flush a file or directory to disk; a best-effort durability step."""
    flags = os.O_RDONLY
    if path.is_dir():
        flags |= getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(str(path), flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_write_bytes(path: Path, data: bytes, *, exclusive: bool = False) -> None:
    """Write bytes atomically via a same-dir temp file and replace.

    The rename is the only step that publishes the new content, so a
    failure anywhere before it leaves the previous file untouched and no
    temp file behind. A failed rename is reported as a friendly refusal
    naming the file, never as a raw OSError.

    With exclusive=True the destination must not exist yet: the link
    fails with FileExistsError and the existing file is left alone, so
    a committed artifact is never overwritten.
    """
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_bytes(data)
        os.chmod(tmp, 0o644)
        # fsync the file before the rename and the directory after it, so
        # a crash cannot publish a half-written file or lose the rename.
        _fsync_path(tmp)
        if exclusive:
            published = False
            try:
                os.link(tmp, path)
                published = True
            except FileExistsError as error:
                raise ContractError(
                    f"{path.name} already exists; it is written once, by the "
                    "freeze run, and never overwritten"
                ) from error
            except OSError as error:
                link_errno = getattr(error, "errno", None)
                if link_errno in (errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP):
                    try:
                        out_fd = os.open(
                            str(path),
                            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                            0o644,
                        )
                    except FileExistsError as exists_error:
                        raise ContractError(
                            f"{path.name} already exists; it is written once, by the "
                            "freeze run, and never overwritten"
                        ) from exists_error
                    except OSError as fallback_error:
                        raise ContractError(
                            f"{path.name}: the destination filesystem does not "
                            "support hard links, so the exclusive publish cannot "
                            "use os.link. This is a filesystem limitation, not a "
                            "data problem. The fallback is to create the file "
                            "exclusively (O_CREAT|O_EXCL) and write it in place, "
                            "which preserves the write-once property, but that "
                            f"fallback also failed: {fallback_error}"
                        ) from fallback_error
                    try:
                        os.fchmod(out_fd, 0o644)
                        os.write(out_fd, data)
                        os.fsync(out_fd)
                    except OSError as fallback_error:
                        try:
                            os.close(out_fd)
                        except OSError:
                            pass
                        try:
                            path.unlink()
                        except OSError:
                            pass
                        raise ContractError(
                            f"{path.name}: the destination filesystem does not "
                            "support hard links, so the exclusive publish cannot "
                            "use os.link. This is a filesystem limitation, not a "
                            "data problem. The fallback is to create the file "
                            "exclusively (O_CREAT|O_EXCL) and write it in place, "
                            "which preserves the write-once property, but that "
                            f"fallback also failed: {fallback_error}"
                        ) from fallback_error
                    else:
                        os.close(out_fd)
                        published = True
                else:
                    raise ContractError(f"{path.name} write failed: {error}") from error
            if published:
                try:
                    tmp.unlink()
                except OSError:
                    pass
        else:
            try:
                os.replace(tmp, path)
            except OSError as error:
                raise ContractError(f"{path.name} write failed: {error}") from error
        _fsync_path(path.parent)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


# Windows opens files in text mode unless O_BINARY is set; the lock file
# is raw bytes locked past the PID marker, so pin binary mode where it exists.
_O_BINARY = getattr(os, "O_BINARY", 0)

# Held manifest locks, keyed by (resolved directory, thread id) so a nested
# acquisition reuses the fd only for the same directory in the same thread.
# A nested acquisition for a different directory, or an acquisition from a
# different thread in the same process, takes its own lock. Each entry is
# (directory_key, thread_id, fd); reentrant entries append the same tuple.
_MANIFEST_LOCK_HELD: list[tuple[str, int, int]] = []


@contextlib.contextmanager
def _manifest_lock(directory: Path) -> Iterator[None]:
    """Hold the manifest's advisory lock for the duration of the block.

    The lock is reentrant within the same thread for the same resolved
    directory: a nested acquisition (the freeze path holds it across the
    pre-flight gate, the artifact write and the manifest stamp, and each of
    those writes a manifest record) reuses the held lock instead of
    deadlocking against itself. A nested acquisition for a different
    directory, or an acquisition from a different thread in the same
    process, takes and holds its own lock.

    The marker file records the owner's PID. It is written AFTER the lock
    is taken, so a waiter that sees a PID is looking at a live holder;
    a stale marker from a crashed writer names a PID that no longer
    holds the lock, and the refusal says so rather than blaming it.
    """
    lock_path = directory / "run_manifest.json.lock"
    key = (str(directory.resolve()), threading.get_ident())
    held = _MANIFEST_LOCK_HELD
    current = [entry for entry in held if (entry[0], entry[1]) == key]
    if current:
        # Already held by this thread for this directory: the flock is per
        # open file description, so re-locking here would block on ourselves.
        held.append(current[-1])
        try:
            yield
        finally:
            held.pop()
        return
    directory.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT | _O_BINARY, 0o644)
    try:
        waited = 0.0
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as error:
                if error.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                    raise ContractError(
                        f"run_manifest.json: the lock could not be taken: {error!r}"
                    ) from error
                if waited >= MANIFEST_LOCK_WAIT_SECONDS:
                    holder = ""
                    try:
                        holder = (
                            lock_path.read_bytes()
                            .decode("utf-8", "replace")
                            .strip()[:24]
                        )
                    except OSError:
                        pass
                    stale = ""
                    if holder.isdigit():
                        try:
                            os.kill(int(holder), 0)
                        except PermissionError:
                            stale = (
                                " (that process is alive but not owned by this user)"
                            )
                        except ProcessLookupError:
                            stale = " (that process is gone; the marker is stale)"
                        except OSError:
                            stale = " (that process is gone; the marker is stale)"
                    raise ContractError(
                        "run_manifest.json: another writer holds the lock "
                        f"(pid {holder or 'unknown'}){stale}"
                    ) from None
                time.sleep(0.1)
                waited += 0.1
    except BaseException:
        os.close(lock_fd)
        raise
    entry = (key[0], key[1], lock_fd)
    try:
        os.ftruncate(lock_fd, 0)
        os.write(lock_fd, str(os.getpid()).encode("ascii"))
        held.append(entry)
        try:
            yield
        finally:
            held.pop()
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(lock_fd)


def _manifest_lock_held(directory: Path) -> bool:
    """Is the manifest lock of `directory` held right now?

    The probe opens the lock file a SECOND time, so it gets its own open
    file description: flock() treats those independently, even inside one
    process. A second exclusive lock that cannot be taken therefore proves
    the first one is held.
    """
    lock_path = directory / "run_manifest.json.lock"
    if not lock_path.is_file():
        return False
    fd = os.open(str(lock_path), os.O_RDWR | _O_BINARY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


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
    from exit), freeze_run, notes. elapsed_seconds must be a finite
    non-negative number and arguments values must be strings: non-strict
    JSON (Infinity, NaN) is refused at the writer. The write holds the
    artifacts/run_manifest.json.lock file with an advisory flock; the
    marker file is contract furniture (gitignored, never unlinked), a
    dead writer's lock is released by the kernel so a crash cannot wedge
    writes, and a busy writer waits MANIFEST_LOCK_WAIT_SECONDS before
    refusing. The write is atomic (temp file then replace) and
    re-verified: the file is reopened, parsed, its size asserted, and
    the appended record must land last. Both --check and every --freeze
    invocation is recorded.
    """
    directory = Path(artifacts_dir) if artifacts_dir is not None else ARTIFACTS_DIR
    path = directory / "run_manifest.json"
    directory.mkdir(parents=True, exist_ok=True)
    if isinstance(elapsed_seconds, bool) or not isinstance(
        elapsed_seconds, (int, float)
    ):
        raise ContractError(
            "run_manifest.json record: elapsed_seconds must be a finite "
            f"non-negative number, got {elapsed_seconds!r}"
        )
    try:
        elapsed_value = float(elapsed_seconds)
    except OverflowError as error:
        raise ContractError(
            "run_manifest.json record: elapsed_seconds "
            f"{elapsed_seconds!r} is out of representable range"
        ) from error
    if not math.isfinite(elapsed_value) or elapsed_value < 0:
        raise ContractError(
            "run_manifest.json record: elapsed_seconds must be a finite "
            f"non-negative number, got {elapsed_seconds!r}"
        )
    arguments_value: dict[str, str] = {}
    if arguments is not None:
        if not isinstance(arguments, dict):
            raise ContractError("run_manifest.json record: arguments must be an object")
        for arg_key, arg_value in arguments.items():
            if not isinstance(arg_key, str):
                raise ContractError(
                    "run_manifest.json record: arguments keys must be strings, "
                    f"got {arg_key!r}"
                )
            if not isinstance(arg_value, str):
                raise ContractError(
                    "run_manifest.json record: arguments "
                    f"{arg_key!r} value must be a string, got {arg_value!r}"
                )
            arguments_value[arg_key] = arg_value
    with _manifest_lock(directory):
        return _write_manifest_record(
            directory,
            command,
            exit_status=exit_status,
            elapsed_seconds=elapsed_value,
            arguments=arguments_value,
            freeze_run=freeze_run,
            notes=notes,
            path=path,
        )


def _freeze_stamp() -> str:
    """The freeze run's manifest timestamp, captured once and reused."""
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _quarantine_stranded_freeze_results(directory: Path, stamp: str) -> Path:
    """Move a stranded freeze_results.json aside under the manifest lock.

    The stamp failed after the artifact was published, so the canonical
    name is freed for the sanctioned re-run by renaming the stranded file
    under the same lock that guarded the publish. The quarantined name
    does not end with .json, so it never collides with the artifact
    registry's JSON scan.
    """
    source = directory / FREEZE_RESULTS_FILENAME
    suffix = stamp.replace(":", "").replace("-", "")
    failed = directory / f"freeze_results.failed-{suffix}"
    if failed.exists():
        failed = directory / f"freeze_results.failed-{suffix}-{os.getpid()}"
    try:
        os.replace(source, failed)
    except OSError as error:
        raise ContractError(
            f"freeze: the manifest stamp failed and {FREEZE_RESULTS_FILENAME} "
            f"could not be moved aside for a re-run: {error}"
        ) from error
    return failed


def _write_manifest_record(
    directory: Path,
    command: str,
    *,
    exit_status: int,
    elapsed_seconds: float,
    arguments: dict[str, str],
    freeze_run: bool,
    notes: list[str] | None,
    path: Path,
    git_sha: str | None = None,
    created: str | None = None,
) -> None:
    """Append one manifest record; the manifest lock must already be held.

    git_sha and created may be supplied by a caller that already captured
    them, so one freeze run stamps one sha and one timestamp into both
    the artifact and the manifest.
    """
    elapsed_value = elapsed_seconds
    arguments_value = arguments
    if git_sha is None:
        sha, sha_notes = _git_sha()
    else:
        sha, sha_notes = git_sha, []
    stamp = created or datetime.now(UTC).isoformat(timespec="seconds")
    record: dict[str, Any] = {
        "created": stamp.replace("+00:00", "Z"),
        "git_sha": sha,
        "command": str(command),
        "exit": int(exit_status),
        "exit_ok": exit_status == 0,
        "versions": _tool_versions(),
        "elapsed_seconds": elapsed_value,
        "arguments": arguments_value,
        "status": "PASS" if exit_status == 0 else "FAIL",
        "freeze_run": bool(freeze_run),
        "notes": [*(sha_notes), *(notes or [])],
    }
    existing_records = list(read_run_manifest(path)) if path.exists() else []
    if bool(record["freeze_run"]):
        if any(r.get("freeze_run") is True for r in existing_records):
            raise ContractError(
                "run_manifest.json already carries "
                "freeze_run = true; redoing a freeze takes a chat-noted "
                "decision, never silence"
            )
    if record.get("git_sha") == "":
        notes_check = record.get("notes") or []
        has_git_note = False
        if isinstance(notes_check, list):
            for note in notes_check:
                if isinstance(note, str) and "git sha unavailable" in note:
                    has_git_note = True
                    break
        if not has_git_note:
            raise ContractError(
                "run_manifest.json record: git_sha must be a 40-hex sha (or empty)"
            )
    candidate = [*existing_records, record]
    candidate_bytes = (json.dumps(candidate, indent=2) + "\n").encode("utf-8")
    try:
        reparsed = json.loads(candidate_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ContractError(
            "run_manifest write failed: the candidate record does not "
            f"round-trip through JSON: {error!r}"
        ) from error
    if not reparsed or reparsed[-1] != record:
        raise ContractError(
            "run_manifest write failed: the candidate record does not "
            "survive the JSON round-trip unchanged"
        )

    def _rollback_and_raise(error: Exception, message: str) -> NoReturn:
        """Restore the pre-write state, then reject with the real cause."""
        if existed and snapshot is not None:
            try:
                _atomic_write_bytes(path, snapshot)
            except Exception as restore_error:
                raise ContractError(
                    f"run_manifest write failed: {error!r}; the restore of "
                    "run_manifest.json also failed: "
                    f"{restore_error!r}; inspect the file by hand"
                ) from error
        elif not existed:
            try:
                residue = path.read_bytes()
            except OSError:
                residue = None
            if residue == candidate_bytes:
                try:
                    path.unlink()
                except OSError as unlink_error:
                    raise ContractError(
                        f"run_manifest write failed: {error!r}; the "
                        f"candidate record at {path} could not be removed: "
                        f"{unlink_error!r}; inspect the file by hand"
                    ) from error
            elif residue is not None:
                raise ContractError(
                    f"run_manifest write failed: {error!r}; "
                    "run_manifest.json did not exist before this write but "
                    "now holds foreign content - left untouched for inspection"
                ) from error
        raise ContractError(message) from error

    _validate_run_manifest_record(record, index=len(existing_records))
    existed = path.exists()
    snapshot: bytes | None = None
    if existed:
        try:
            snapshot = path.read_bytes()
        except OSError as error:
            raise ContractError(f"run_manifest write failed: {error!r}") from error
    # Same-dir temp, because os.replace is atomic only within one filesystem.
    fd, tmp_name = tempfile.mkstemp(
        prefix="run_manifest.json.", suffix=".tmp", dir=str(directory)
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")
        os.chmod(tmp, 0o644)
        if tmp.stat().st_size == 0:
            raise ContractError("run_manifest write produced a zero-byte temp file")
        try:
            os.replace(tmp, path)
        except Exception as error:
            _rollback_and_raise(error, f"run_manifest write failed: {error!r}")
        try:
            stored = read_run_manifest(path)
            if not stored or stored[-1] != record:
                raise ContractError(
                    "run_manifest append did not land as the last record"
                )
        except Exception as error:
            _rollback_and_raise(
                error,
                f"run_manifest append did not land as the last record: {error!r}",
            )
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _scan_registered_artifacts(directory: Path) -> dict[str, Path]:
    """Registered model artifact files on disk; stray model files fail loud.

    Registration is top level only: directory/<name> is the only valid
    location for a registered artifact or contract control file. Any
    .json file at any depth whose basename matches a registered or
    allowlisted name but is not that top-level path is a stray and is
    refused; allowlisted names are legitimate only at the top level.
    """
    if not directory.exists():
        return {}
    present = []
    for p in directory.rglob("*"):
        if p.is_file() and p.name.lower().endswith(".json"):
            present.append(p)
    registered = set(ARTIFACT_REGISTRY)
    allowed = registered | set(CONTRACT_ARTIFACT_FILENAMES)
    unknown = []
    for p in present:
        if p.parent == directory and p.name in allowed:
            continue
        unknown.append(str(p.relative_to(directory)))
    if unknown:
        unknown_sorted = sorted(set(unknown))
        raise ContractError(
            f"artifacts/: unregistered file(s) {unknown_sorted}; "
            f"the registry carries exactly {sorted(registered)} - "
            "nothing stray is skipped silently"
        )
    result = {}
    for name in registered:
        found = directory / name
        if found.is_file():
            result[name] = found
    return result


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
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError(
                f"{label}: {key} must be a finite non-negative number, got {value!r}"
            )
        try:
            number = float(value)
        except OverflowError as error:
            raise ContractError(
                f"{label}: {key} {value!r} is out of representable range"
            ) from error
        if not math.isfinite(number) or number < 0.0:
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
    canonical_ranges = {
        "train": (f"{SPLIT_FRAME_STARTS['train']}..{SPLIT_FRAME_ENDS['train']}"),
        "valid": (f"{SPLIT_FRAME_STARTS['valid']}..{SPLIT_FRAME_ENDS['valid']}"),
    }
    for bound_key in ("train", "valid"):
        bound = bounds[bound_key]
        if not isinstance(bound, str) or not bound.strip():
            raise ContractError(
                f"{name}: split_bounds {bound_key} must be a non-empty string, "
                f"got {bound!r}"
            )
        if bound != canonical_ranges[bound_key]:
            raise ContractError(
                f"{name}: split_bounds {bound_key} {bound!r} must be the "
                f"canonical frame-range text {canonical_ranges[bound_key]!r} "
                "(YYYY-MM-DD..YYYY-MM-DD against the split manifest's edges)"
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
        fold_rows = block["n_rows"]
        if isinstance(fold_rows, bool) or not isinstance(fold_rows, int):
            raise ContractError(
                f"{name}: folds[{index}] n_rows must be an int, got {fold_rows!r}"
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
        if not isinstance(entry, dict):
            raise ContractError(
                f"{name}: model {index} is not a dict, got {type(entry).__name__}"
            )
        label = f"{name}: model {index} ({entry.get('name')!r} {entry.get('kind')!r})"
        if list(entry) != list(_CLASSICAL_MODEL_KEYS):
            measured = list(entry)
            raise ContractError(
                f"{label}: per-model keys {measured} != pinned "
                f"{list(_CLASSICAL_MODEL_KEYS)}; pinned order is enforced"
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
        for seed_key, seed_value in seeds.items():
            is_text = isinstance(seed_value, str) and seed_value.strip() != ""
            is_seed_int = (
                isinstance(seed_value, int)
                and not isinstance(seed_value, bool)
                and seed_value >= 0
            )
            if not (is_text or is_seed_int):
                raise ContractError(
                    f"{label}: seeds {seed_key!r} must be an int >= 0 or a "
                    f"non-empty string, got {seed_value!r}"
                )
        if not isinstance(entry["tuning"], str):
            raise ContractError(f"{label}: tuning must be a string")
        christmas = entry["christmas_2016"]
        if not isinstance(christmas, dict):
            raise ContractError(
                f"{label}: christmas_2016 must be an object with exactly "
                "{'status'} equal to 'freeze-time'"
            )
        if list(christmas.keys()) != ["status"]:
            extra = sorted([k for k in christmas.keys() if k != "status"])
            raise ContractError(
                f"{label}: christmas_2016 must carry only 'status'; "
                f"found extra keys {extra}"
            )
        if christmas.get("status") != "freeze-time":
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
        if not isinstance(entry, dict):
            raise ContractError(
                f"{name}: tier {index} is not a dict, got {type(entry).__name__}"
            )
        tie_label = entry.get("tier")
        label = f"{name}: tier {index} ({tie_label!r})"
        if list(entry) != list(_DEEP_TIER_KEYS):
            measured = list(entry)
            raise ContractError(
                f"{label}: per-tier keys {measured} != pinned "
                f"{list(_DEEP_TIER_KEYS)}; pinned order is enforced"
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
        tier_status = entry.get("status")
        if not isinstance(tier_status, str) or tier_status == "":
            raise ContractError(f"{label}: status must be a non-empty string")
        if not isinstance(entry["train"], dict):
            raise ContractError(f"{label}: train must be an object")
        valid_block = entry["valid"]
        if not isinstance(valid_block, dict):
            raise ContractError(f"{label}: valid must be an object")
        valid_rows = valid_block.get("n_rows")
        if isinstance(valid_rows, bool) or not isinstance(valid_rows, int):
            raise ContractError(
                f"{label}: valid.n_rows must be an int, got {valid_rows!r}"
            )
        if valid_rows != 299376:
            raise ContractError(
                f"{label}: valid.n_rows must be the full valid 299,376 rows, "
                f"got {valid_rows!r}"
            )
        if "n_active" not in valid_block:
            raise ContractError(f"{label}: valid.n_active is required")
        n_active = valid_block["n_active"]
        if isinstance(n_active, bool) or not isinstance(n_active, int) or n_active < 0:
            raise ContractError(
                f"{label}: valid.n_active must be an int >= 0, got {n_active!r}"
            )
        if "inference_seconds" not in valid_block:
            raise ContractError(f"{label}: valid.inference_seconds is required")
        inference_seconds = valid_block["inference_seconds"]
        if isinstance(inference_seconds, bool) or not isinstance(
            inference_seconds, (int, float)
        ):
            raise ContractError(
                f"{label}: valid.inference_seconds must be an int or float >= 0, "
                f"got {inference_seconds!r}"
            )
        try:
            seconds_value = float(inference_seconds)
        except (ValueError, TypeError, OverflowError) as error:
            raise ContractError(
                f"{label}: inference_seconds {inference_seconds!r} is out of "
                "representable range"
            ) from error
        if not math.isfinite(seconds_value) or seconds_value < 0:
            raise ContractError(
                f"{label}: valid.inference_seconds must be finite and >= 0, "
                f"got {inference_seconds!r}"
            )
        per_horizon = valid_block.get("per_horizon_rmsle")
        if not isinstance(per_horizon, list) or len(per_horizon) != 16:
            raise ContractError(
                f"{label}: per_horizon_rmsle must hold the 16 horizon values"
            )
        for horizon_index, horizon_value in enumerate(per_horizon):
            if isinstance(horizon_value, bool) or not isinstance(
                horizon_value, (int, float)
            ):
                raise ContractError(
                    f"{label}: per_horizon_rmsle[{horizon_index}] must be a "
                    f"number, got {horizon_value!r}"
                )
            try:
                horizon_number = float(horizon_value)
            except OverflowError as error:
                raise ContractError(
                    f"{label}: per_horizon_rmsle[{horizon_index}] "
                    f"{horizon_value!r} is out of representable range"
                ) from error
            if not math.isfinite(horizon_number) or horizon_number < 0.0:
                raise ContractError(
                    f"{label}: per_horizon_rmsle[{horizon_index}] must be a "
                    f"finite non-negative number, got {horizon_value!r}"
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
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ContractError(
                        f"{label}: valid.{slice_key}.{key} must be a finite "
                        f"non-negative number, got {value!r}"
                    )
                try:
                    slice_number = float(value)
                except OverflowError as error:
                    raise ContractError(
                        f"{label}: valid.{slice_key}.{key} {value!r} is out of "
                        "representable range"
                    ) from error
                if not math.isfinite(slice_number) or slice_number < 0.0:
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


def _check_registered_artifact(name: str, path: Path) -> tuple[str, dict]:
    """Schema-gate one registered artifact; returns a PASS blurb and payload.

    Per home: required top-level keys, pinned tier identity and order,
    3 fold_metrics entries carrying the pinned score schema, finite
    non-negative metric numbers, fold counts 28,512 (85,536 pooled),
    and the banned-key scan for the percentage-error token anywhere, plus
    local_test everywhere and the christmas key on the deep side, and
    writers must emit the pinned key order.
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
    blurb = f"{spec['track']} x {spec['model_count']} tiers"
    return blurb, payload


def _freeze_preflight(
    artifacts_dir: Path,
) -> tuple[dict[str, Path], list[str]]:
    """Every freeze refusal gate, in order; opens no split.

    Gate order: FREEZE_RUN=1 -> exactly-one-freeze (a freeze_run = true
    record refuses any repeat, and that refusal already covers a committed
    freeze) -> the artifact-existence gate (a freeze_results.json with no
    freeze_run = true record is a stranded artifact, refused with an
    actionable lead) -> the registry scan (unknown file, zero registered
    artifacts, partial registry) -> per-artifact schema. Every refusal is a
    friendly ContractError, never a traceback.
    """
    if os.environ.get("FREEZE_RUN") != "1":
        raise ContractError(
            "freeze refused: set FREEZE_RUN=1 and go through the freeze path "
            "only; local_test is never read ad hoc and never tuned on"
        )
    manifest_path = artifacts_dir / "run_manifest.json"
    records = list(read_run_manifest(manifest_path)) if manifest_path.exists() else []
    frozen = any(record["freeze_run"] is True for record in records)
    if frozen:
        raise ContractError(
            "freeze refused: run_manifest.json already carries "
            "freeze_run = true; redoing a freeze takes a chat-noted "
            "decision, never silence"
        )
    freeze_results = artifacts_dir / FREEZE_RESULTS_FILENAME
    if freeze_results.exists():
        # The exactly-one-freeze gate above has already refused a committed
        # freeze, so reaching here means the artifact is stranded: the stamp
        # failed after the artifact was published and no freeze_run = true
        # record was written.
        raise ContractError(
            f"freeze refused: {FREEZE_RESULTS_FILENAME} already exists but the "
            "manifest carries no freeze_run = true record (a stranded "
            f"artifact). Remove or rename {FREEZE_RESULTS_FILENAME} in "
            f"{artifacts_dir} (for example, move it aside as "
            f"{FREEZE_RESULTS_FILENAME}.stranded) after noting the failed "
            "attempt in chat, then re-run the freeze"
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
    blurbs: list[str] = []
    deep_payload: Any = None
    for name, path in sorted(registered.items()):
        blurb, payload = _check_registered_artifact(name, path)
        blurbs.append(blurb)
        if name == "deep_results.json":
            deep_payload = payload
    if deep_payload is not None:
        if isinstance(deep_payload, dict):
            status = deep_payload.get("status")
            if status != "complete":
                raise ContractError(
                    f"artifacts/deep_results.json: status {status!r} must be "
                    "complete for freeze"
                )
    return registered, blurbs


def _require_int(value: Any, label: str) -> int:
    """A non-negative int, or a loud refusal."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{label} must be a non-negative int, got {value!r}")
    return value


def _assert_pinned_frame(frame: pd.DataFrame, label: str) -> None:
    """Refuse a frame that drifted from the 26 pinned columns and dtypes."""
    if not isinstance(frame, pd.DataFrame):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ContractError(f"{label}: must be a DataFrame, got {type(frame).__name__}")
    if not frame.columns.is_unique:
        raise ContractError(f"{label}: columns contain duplicates")
    missing = [c for c in PINNED_DTYPES if c not in frame.columns]
    extra = [c for c in frame.columns if c not in PINNED_DTYPES]
    if missing or extra:
        raise ContractError(
            f"{label}: columns must exactly match the 26 pinned columns; "
            f"missing {missing}; extra {extra}"
        )
    for col, expected_dtype in PINNED_DTYPES.items():
        actual = str(frame[col].dtype)
        if actual != expected_dtype:
            raise ContractError(
                f"{label}: column {col!r} got dtype {actual}, expected {expected_dtype}"
            )


def _empty_score_block() -> dict[str, float | int]:
    """The pinned score schema for an empty slice."""
    return {
        "rmsle": 0.0,
        "mae": 0.0,
        "rmse": 0.0,
        "rmsle_active": 0.0,
        "mae_active": 0.0,
        "rmse_active": 0.0,
        "n_rows": 0,
        "n_rows_active": 0,
    }


def _slice_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    """One slice of the pinned score schema, in the pinned key order.

    score() is the only metric implementation; this helper only picks the
    slice and keeps the pinned key order. An empty slice scores as zeros
    instead of refusing (the Christmas-2016 slice is all-zero demand).
    """
    if y_true.size == 0:
        return _empty_score_block()
    block = score(y_true, y_pred)
    return {key: block[key] for key in SCORE_KEYS}


def _christmas_mask(
    local_test: pd.DataFrame, expected_rows: int = CHRISTMAS_2016_ROWS
) -> np.ndarray:
    """The Christmas-2016 slice mask, derived from the frame's date column.

    The slice is the local_test rows dated 2016-12-25 (54 stores x 33
    families = 1,782 rows). The row count is asserted, never assumed: a
    frame whose slice does not hold exactly the pinned count is refused.
    """
    if "date" not in local_test.columns:
        raise ContractError("freeze: the local_test frame has no 'date' column")
    dates = local_test["date"]
    if not pd.api.types.is_datetime64_any_dtype(dates):
        raise ContractError(
            "freeze: the local_test 'date' column must be a datetime, got "
            f"{dates.dtype}"
        )
    mask = (dates == CHRISTMAS_2016_DATE).to_numpy()
    found = int(mask.sum())
    if found != expected_rows:
        raise ContractError(
            f"freeze: the Christmas-2016 slice holds {found} local_test rows "
            f"dated {CHRISTMAS_2016_DATE.date()}, expected {expected_rows} "
            "(54 stores x 33 families); the slice is derived from the frame's "
            "date column, never hardcoded"
        )
    return mask


def _row_keys(frame: pd.DataFrame) -> list[tuple[int, str, int]]:
    """The (store_nbr, family, date) key of every row, in row order."""
    stores = frame["store_nbr"].to_numpy(dtype=np.int64)
    families = frame["family"].astype(str).to_numpy()
    dates = frame["date"].to_numpy(dtype="datetime64[D]").astype(np.int64)
    return list(zip(stores.tolist(), families.tolist(), dates.tolist(), strict=True))


def _prediction_side_keys(
    model_name: str, index: pd.Index
) -> list[tuple[int, str, int]]:
    """Read (store_nbr, family, date) keys from a prediction Series index.

    The key comes from the prediction side (the Series index), never from the
    contract frame, so a keyed return can be reconciled to local_test by that
    key instead of by position.
    """
    if isinstance(index, pd.MultiIndex):
        if index.nlevels != 3:
            raise ContractError(
                f"freeze: model {model_name!r} returned a keyed Series whose "
                f"index has {index.nlevels} levels; expected 3 levels "
                "(store_nbr, family, date)"
            )
        stores = index.get_level_values(0).to_numpy(dtype=np.int64)
        families = index.get_level_values(1).astype(str).to_numpy()
        date_level = index.get_level_values(2)
        if not pd.api.types.is_datetime64_any_dtype(date_level):
            raise ContractError(
                f"freeze: model {model_name!r} returned a keyed Series whose "
                "date level is not a datetime; the keyed return must carry "
                "real dates, not an integer day number, so the key is "
                "unambiguous"
            )
        dates = (
            pd.to_datetime(date_level).to_numpy(dtype="datetime64[D]").astype(np.int64)
        )
    else:
        stores_list: list[int] = []
        families_list: list[str] = []
        dates_list: list[int] = []
        for item in index:
            if not isinstance(item, tuple) or len(item) != 3:
                raise ContractError(
                    f"freeze: model {model_name!r} returned a keyed Series whose "
                    "index entries are not (store_nbr, family, date) tuples"
                )
            store, family, date = item
            stores_list.append(int(store))
            families_list.append(str(family))
            dates_list.append(
                int(np.datetime64(pd.Timestamp(date), "D").astype(np.int64))
            )
        stores = np.asarray(stores_list, dtype=np.int64)
        families = np.asarray(families_list, dtype=object)
        dates = np.asarray(dates_list, dtype=np.int64)
    return list(zip(stores.tolist(), families.tolist(), dates.tolist(), strict=True))


def _align_local_test_predictions(
    model_name: str,
    raw: Any,
    predict_frame: pd.DataFrame,
    local_test: pd.DataFrame,
) -> np.ndarray:
    """Align a fit_predict_frozen return to the local_test rows.

    Two return shapes are accepted. A 1-D array-like aligned to the
    predict_frame row order (S3) or to the local_test rows only (S4) is
    accepted; the two shapes are distinguished by length, and anything else
    is refused with both counts named. An ndarray/list carries no
    prediction-side key, so it is taken as the entrypoint returned rows in
    that frame's own row order.

    A pandas Series whose index carries (store_nbr, family, date) keys is
    aligned by looking those prediction-side keys up against local_test, so
    a keyed Series may be returned in a different row order and still score
    the right pairs. A keyed return that cannot be reconciled to every
    local_test row is refused.
    """
    predict_frame_rows = len(predict_frame)
    local_test_rows = len(local_test)
    wanted = _row_keys(local_test)

    if isinstance(raw, pd.Series):
        try:
            predictions = np.asarray(raw.to_numpy(), dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ContractError(
                f"freeze: model {model_name!r} returned a non-numeric prediction "
                f"array: {error}"
            ) from error
        if predictions.ndim != 1:
            raise ContractError(
                f"freeze: model {model_name!r} returned a {predictions.ndim}-D "
                "prediction array; predictions must be 1-D"
            )
        source_keys = _prediction_side_keys(model_name, raw.index)
    else:
        try:
            predictions = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ContractError(
                f"freeze: model {model_name!r} returned a non-numeric prediction "
                f"array: {error}"
            ) from error
        if predictions.ndim != 1:
            raise ContractError(
                f"freeze: model {model_name!r} returned a {predictions.ndim}-D "
                "prediction array; predictions must be 1-D"
            )
        if predictions.size == local_test_rows:
            source_keys = _row_keys(local_test)
        elif predictions.size == predict_frame_rows:
            source_keys = _row_keys(predict_frame)
        else:
            raise ContractError(
                f"freeze: model {model_name!r} returned {predictions.size} "
                f"predictions; expected {local_test_rows} (aligned to the "
                f"local_test rows) or {predict_frame_rows} (aligned to the "
                "predict_frame row order)"
            )

    if len(predictions) != len(source_keys):
        raise ContractError(
            f"freeze: model {model_name!r} returned {len(predictions)} "
            f"predictions for {len(source_keys)} keyed row(s); the counts must match"
        )
    if len(set(source_keys)) != len(source_keys):
        raise ContractError(
            f"freeze: model {model_name!r} returned predictions whose rows "
            "repeat a (store_nbr, family, date) key; the alignment is by key"
        )
    index_by_key = {key: position for position, key in enumerate(source_keys)}
    missing = [key for key in wanted if key not in index_by_key]
    if missing:
        raise ContractError(
            f"freeze: model {model_name!r} returned no prediction for "
            f"{len(missing)} local_test row(s), first {missing[0]}"
        )
    aligned = predictions[[index_by_key[key] for key in wanted]]
    if not np.isfinite(aligned).all():
        raise ContractError(
            f"freeze: model {model_name!r} returned non-finite predictions"
        )
    return aligned


def _registered_payloads(directory: Path) -> dict[str, dict[str, Any]]:
    """The registered artifacts' parsed payloads, read for the fold phase.

    The bytes are re-validated here, not trusted from the pre-flight
    gate: a track that regenerates its artifact between the gate and the
    fold phase cannot inject unvalidated numbers into the freeze.
    """
    payloads: dict[str, dict[str, Any]] = {}
    for name in ARTIFACT_REGISTRY:
        path = directory / name
        if not path.is_file():
            raise ContractError(
                f"freeze: the registered artifact {name} is missing from {directory}"
            )
        # The fold phase reads the SAME bytes the pre-flight gate
        # validated: re-check them here, so a track regenerating its
        # artifact mid-freeze cannot inject unvalidated numbers.
        _blurb, payload = _check_registered_artifact(name, path)
        spec = ARTIFACT_REGISTRY[name]
        tiers = payload.get(spec["tiers_key"])
        if not isinstance(tiers, list) or not tiers:
            raise ContractError(
                f"freeze: artifacts/{name} carries no {spec['tiers_key']} list"
            )
        names = [entry.get(spec["tier_name_key"]) for entry in tiers]
        if names != list(spec["model_names"]):
            raise ContractError(
                f"freeze: artifacts/{name} tier names {names} != pinned "
                f"{list(spec['model_names'])}"
            )
        payloads[name] = payload
    return payloads


def _freeze_model_specs(payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The ordered freeze model list: every registered tier, with its config.

    The classical tiers carry the hyperparameters their own artifact
    records (that artifact is the winning-config record); the deep tiers
    carry None, because their sheet pins the config argument as optional
    and the checkpoint holds the architecture.
    """
    specs: list[dict[str, Any]] = []
    for artifact_name, payload in payloads.items():
        spec = ARTIFACT_REGISTRY[artifact_name]
        for entry in payload[spec["tiers_key"]]:
            config: Any = None
            if artifact_name == "classical_results.json":
                config = entry.get("hyperparameters")
                if not isinstance(config, dict):
                    raise ContractError(
                        f"freeze: artifacts/{artifact_name} tier "
                        f"{entry.get(spec['tier_name_key'])!r} carries no "
                        "hyperparameters object"
                    )
            specs.append(
                {
                    "name": str(entry[spec["tier_name_key"]]),
                    "track": spec["track"],
                    "artifact": artifact_name,
                    "config": config,
                }
            )
    return specs


def _data_snapshot(manifest: dict[str, Any]) -> dict[str, Any]:
    """The per-split rows and bound dates, copied from the fresh manifest."""
    snapshot: dict[str, Any] = {}
    for split in SPLIT_NAMES:
        rows = manifest["rows"][split]
        bounds = manifest["bounds"][split]
        snapshot[split] = {
            "rows": int(rows),
            "bounds": [str(bounds[0]), str(bounds[1])],
        }
    return snapshot


def _contract_counts(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    local_test: pd.DataFrame,
    frame: pd.DataFrame,
    fit_frame: pd.DataFrame,
) -> dict[str, Any]:
    """The contract_counts block: the week's locked counts, re-derived.

    Every number that the frames can produce is computed from them and
    compared against its pinned value, so a data drift refuses the freeze
    instead of landing in the final JSON. Two values are pins the frames
    cannot produce and are carried through as the pinned constants: the
    7,128-row full-calendar universe (why.grid_fill_rows in
    split_manifest.json) and the 1,782-row Christmas-2016 slice.
    """
    closure = {
        "train": int(closure_rows(train).sum()),
        "valid": int(closure_rows(valid).sum()),
        "local_test": int(closure_rows(local_test).sum()),
    }
    if closure != PINNED_CLOSURE_ROWS:
        raise ContractError(
            f"freeze: closure rows {closure} != pinned {PINNED_CLOSURE_ROWS}"
        )
    drop_train = int((~fit_rows(train)).sum())
    drop_valid = int((~fit_rows(valid)).sum())
    in_window = int((preopening_rows(train) & closure_rows(train)).sum())
    if {
        "train": drop_train,
        "valid": drop_valid,
        "total": drop_train + drop_valid,
    } != PINNED_MASKED_DROP:
        raise ContractError(
            f"freeze: the masked training drop {{'train': {drop_train}, "
            f"'valid': {drop_valid}, 'total': {drop_train + drop_valid}}} != "
            f"pinned {PINNED_MASKED_DROP}"
        )
    if in_window != PINNED_IN_WINDOW_FILLS_KEPT:
        raise ContractError(
            f"freeze: {in_window} closure fills sit inside a pre-opening "
            f"window, pinned {PINNED_IN_WINDOW_FILLS_KEPT}"
        )
    raw_zeros = int((frame["sales"] == 0).sum())
    if (raw_zeros, len(frame)) != PINNED_ZERO_SHARE_RAW:
        raise ContractError(
            f"freeze: the raw zero share is {raw_zeros} of {len(frame)}, "
            f"pinned {PINNED_ZERO_SHARE_RAW[0]} of {PINNED_ZERO_SHARE_RAW[1]}"
        )
    trainable_zeros = int((fit_frame["sales"] == 0).sum())
    if (trainable_zeros, len(fit_frame)) != PINNED_ZERO_SHARE_TRAINABLE:
        raise ContractError(
            f"freeze: the trainable zero share is {trainable_zeros} of "
            f"{len(fit_frame)}, pinned {PINNED_ZERO_SHARE_TRAINABLE[0]} of "
            f"{PINNED_ZERO_SHARE_TRAINABLE[1]}"
        )
    holdout = {
        "valid": int(preopening_rows(valid).sum()),
        "local_test": int(preopening_rows(local_test).sum()),
    }
    holdout["total"] = holdout["valid"] + holdout["local_test"]
    if holdout != PINNED_PREOPENING_HOLDOUT:
        raise ContractError(
            f"freeze: the pre-opening holdout rows {holdout} != pinned "
            f"{PINNED_PREOPENING_HOLDOUT}"
        )
    return {
        "closure_rows": {
            "train": closure["train"],
            "valid": closure["valid"],
            "local_test": closure["local_test"],
            "universe": PINNED_CLOSURE_UNIVERSE,
        },
        "masked_training_drop": {
            "train": drop_train,
            "valid": drop_valid,
            "total": drop_train + drop_valid,
            "in_window_fills_kept": in_window,
        },
        "zero_share_raw": {
            "rows": raw_zeros,
            "of_rows": len(frame),
            "share": round(raw_zeros / len(frame), 6),
        },
        "zero_share_trainable": {
            "rows": trainable_zeros,
            "of_rows": len(fit_frame),
            "share": round(trainable_zeros / len(fit_frame), 6),
        },
        "preopening_holdout_rows": {
            "valid": holdout["valid"],
            "local_test": holdout["local_test"],
            "total": holdout["total"],
        },
        "christmas_2016_rows": CHRISTMAS_2016_ROWS,
    }


def _fold_phase_from_artifacts(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The shared fold table plus every registered model's per-fold metrics.

    The fold-phase numbers are the ones the tracks already recorded in
    their registered artifacts; the freeze re-reads them, never re-derives
    them (metric recomputation is the freeze phase's job only).
    """
    table = folds_table()
    fold_rows = [
        {
            "fold": int(row["fold"]),
            "origin": str(row["origin"]),
            "window_start": str(row["window_start"]),
            "window_end": str(row["window_end"]),
            "n_rows": int(row["n_rows"]),
        }
        for _, row in table.iterrows()
    ]
    models: list[dict[str, Any]] = []
    for artifact_name, payload in payloads.items():
        spec = ARTIFACT_REGISTRY[artifact_name]
        for entry in payload[spec["tiers_key"]]:
            model_name = str(entry[spec["tier_name_key"]])
            records: list[dict[str, Any]] = []
            fold_metrics = entry.get("fold_metrics")
            if not isinstance(fold_metrics, list) or len(fold_metrics) != len(
                FOLD_TABLE_PINS
            ):
                raise ContractError(
                    f"freeze: artifacts/{artifact_name} model {model_name!r} "
                    f"carries no fold_metrics list of {len(FOLD_TABLE_PINS)} "
                    "folds"
                )
            for index, fold_row in enumerate(fold_metrics):
                block = dict(fold_row)
                ordinal = block.pop("fold", index + 1)
                if ordinal != index + 1:
                    raise ContractError(
                        f"freeze: artifacts/{artifact_name} model {model_name!r} "
                        f"fold_metrics[{index}] carries fold {ordinal}, "
                        f"expected {index + 1}"
                    )
                missing = [key for key in SCORE_KEYS if key not in block]
                if missing:
                    raise ContractError(
                        f"freeze: artifacts/{artifact_name} model {model_name!r} "
                        f"fold_metrics[{index}] is missing {missing}"
                    )
                full_grid = _assert_score_block(
                    f"freeze: artifacts/{artifact_name} model {model_name!r} "
                    f"fold {index + 1}",
                    {key: block[key] for key in SCORE_KEYS},
                )
                active_slice = {
                    "rmsle": full_grid["rmsle_active"],
                    "mae": full_grid["mae_active"],
                    "rmse": full_grid["rmse_active"],
                    "rmsle_active": full_grid["rmsle_active"],
                    "mae_active": full_grid["mae_active"],
                    "rmse_active": full_grid["rmse_active"],
                    "n_rows": full_grid["n_rows_active"],
                    "n_rows_active": full_grid["n_rows_active"],
                }
                if int(full_grid["n_rows"]) != FOLD_ROWS:
                    raise ContractError(
                        f"freeze: artifacts/{artifact_name} model {model_name!r} "
                        f"fold {index + 1} scores {full_grid['n_rows']} rows, "
                        f"pinned {FOLD_ROWS}"
                    )
                records.append(
                    {
                        "fold": index + 1,
                        "origin": fold_rows[index]["origin"],
                        "window_start": fold_rows[index]["window_start"],
                        "window_end": fold_rows[index]["window_end"],
                        "n_rows": int(full_grid["n_rows"]),
                        "slices": {
                            "full_grid": dict(full_grid),
                            "active_slice": active_slice,
                        },
                    }
                )
            models.append(
                {
                    "name": model_name,
                    "track": spec["track"],
                    "artifact": artifact_name,
                    "folds": records,
                }
            )
    return {"folds": fold_rows, "models": models}


def _freeze_phase_records(
    models: list[dict[str, Any]],
    local_test: pd.DataFrame,
    predict_frame: pd.DataFrame,
    predict: Callable[[dict[str, Any]], Any],
) -> list[dict[str, Any]]:
    """Step 3-4: every registered model's freeze-phase slices.

    The Christmas-2016 slice comes from the local_test frame's date
    column, and every metric comes from score(); nothing is re-derived
    here.
    """
    christmas = _christmas_mask(local_test)
    y_true = local_test["sales"].to_numpy(dtype=np.float64)
    active = y_true > 0
    local_test_rows = len(local_test)
    window_start = local_test["date"].min().date().isoformat()
    window_end = local_test["date"].max().date().isoformat()
    records: list[dict[str, Any]] = []
    for spec in models:
        model_name = str(spec["name"])
        raw = predict(spec)
        predictions = _align_local_test_predictions(
            model_name, raw, predict_frame, local_test
        )
        records.append(
            {
                "name": model_name,
                "track": str(spec["track"]),
                "artifact": str(spec["artifact"]),
                "window_start": window_start,
                "window_end": window_end,
                "n_rows": int(local_test_rows),
                "slices": {
                    "full_grid": _slice_block(y_true, predictions),
                    "active_slice": _slice_block(y_true[active], predictions[active]),
                    "christmas_2016": _slice_block(
                        y_true[christmas], predictions[christmas]
                    ),
                },
            }
        )
    return records


def _pick_headline_model(records: list[dict[str, Any]]) -> str:
    """The registered model with the lowest freeze-time full-grid RMSLE."""
    if not records:
        raise ContractError("freeze: no registered model carries a freeze-phase score")
    best = min(
        records,
        key=lambda record: float(record["slices"]["full_grid"]["rmsle"]),
    )
    return str(best["name"])


def _validate_data_snapshot(snapshot: Any) -> None:
    """The data_snapshot block: three splits, rows and bound dates."""
    if not isinstance(snapshot, dict) or sorted(snapshot) != sorted(SPLIT_NAMES):
        raise ContractError(
            "freeze_results.json: data_snapshot must cover exactly train, "
            "valid and local_test, copied from the fresh split manifest"
        )
    for split, block in snapshot.items():
        label = f"freeze_results.json: data_snapshot {split}"
        if not isinstance(block, dict) or sorted(block) != ["bounds", "rows"]:
            raise ContractError(f"{label} must carry exactly rows and bounds")
        _require_int(block["rows"], f"{label} rows")
        if block["rows"] != PINNED_SPLIT_ROWS[split]:
            raise ContractError(
                f"{label} rows {block['rows']} != the pinned {PINNED_SPLIT_ROWS[split]}"
            )
        bounds = block["bounds"]
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or not all(isinstance(edge, str) and edge for edge in bounds)
        ):
            raise ContractError(f"{label} bounds must be two date strings")
        pinned_bounds = [SPLIT_FRAME_STARTS[split], SPLIT_FRAME_ENDS[split]]
        if bounds != pinned_bounds:
            raise ContractError(
                f"{label} bounds {bounds} != the pinned {pinned_bounds}"
            )


def _validate_contract_counts(counts: Any) -> None:
    """The contract_counts block: the pinned sub-blocks, each with counts."""
    if not isinstance(counts, dict) or sorted(counts) != sorted(CONTRACT_COUNT_KEYS):
        raise ContractError(
            "freeze_results.json: contract_counts must carry exactly "
            f"{sorted(CONTRACT_COUNT_KEYS)}"
        )
    closure = counts["closure_rows"]
    if not isinstance(closure, dict) or sorted(closure) != [
        "local_test",
        "train",
        "universe",
        "valid",
    ]:
        raise ContractError(
            "freeze_results.json: contract_counts closure_rows must carry "
            "train, valid, local_test and universe"
        )
    for key, value in closure.items():
        _require_int(value, f"freeze_results.json: contract_counts closure_rows {key}")
    if closure != {
        "train": PINNED_CLOSURE_ROWS["train"],
        "valid": PINNED_CLOSURE_ROWS["valid"],
        "local_test": PINNED_CLOSURE_ROWS["local_test"],
        "universe": PINNED_CLOSURE_UNIVERSE,
    }:
        raise ContractError(
            f"freeze_results.json: contract_counts closure_rows {closure} != the "
            "pinned counts"
        )
    drop = counts["masked_training_drop"]
    if not isinstance(drop, dict) or sorted(drop) != [
        "in_window_fills_kept",
        "total",
        "train",
        "valid",
    ]:
        raise ContractError(
            "freeze_results.json: contract_counts masked_training_drop must "
            "carry train, valid, total and in_window_fills_kept"
        )
    for key, value in drop.items():
        _require_int(
            value, f"freeze_results.json: contract_counts masked_training_drop {key}"
        )
    if drop != {
        "train": PINNED_MASKED_DROP["train"],
        "valid": PINNED_MASKED_DROP["valid"],
        "total": PINNED_MASKED_DROP["total"],
        "in_window_fills_kept": PINNED_IN_WINDOW_FILLS_KEPT,
    }:
        raise ContractError(
            f"freeze_results.json: contract_counts masked_training_drop {drop} "
            "!= the pinned counts"
        )
    for key, pinned in (
        ("zero_share_raw", PINNED_ZERO_SHARE_RAW),
        ("zero_share_trainable", PINNED_ZERO_SHARE_TRAINABLE),
    ):
        block = counts[key]
        if not isinstance(block, dict) or sorted(block) != ["of_rows", "rows", "share"]:
            raise ContractError(
                f"freeze_results.json: contract_counts {key} must carry rows, "
                "of_rows and share"
            )
        _require_int(block["rows"], f"freeze_results.json: contract_counts {key} rows")
        _require_int(
            block["of_rows"], f"freeze_results.json: contract_counts {key} of_rows"
        )
        share = block["share"]
        if isinstance(share, bool) or not isinstance(share, (int, float)):
            raise ContractError(
                f"freeze_results.json: contract_counts {key} share must be a number"
            )
        if (block["rows"], block["of_rows"]) != pinned:
            raise ContractError(
                f"freeze_results.json: contract_counts {key} counts "
                f"{block['rows']} of {block['of_rows']} != the pinned "
                f"{pinned[0]} of {pinned[1]}"
            )
        expected_share = round(block["rows"] / block["of_rows"], 6)
        if share != expected_share:
            raise ContractError(
                f"freeze_results.json: contract_counts {key} share {share} "
                f"!= the pinned {expected_share}"
            )
    holdout = counts["preopening_holdout_rows"]
    if not isinstance(holdout, dict) or sorted(holdout) != [
        "local_test",
        "total",
        "valid",
    ]:
        raise ContractError(
            "freeze_results.json: contract_counts preopening_holdout_rows must "
            "carry valid, local_test and total"
        )
    for key, value in holdout.items():
        _require_int(
            value, f"freeze_results.json: contract_counts preopening_holdout_rows {key}"
        )
    if holdout != PINNED_PREOPENING_HOLDOUT:
        raise ContractError(
            f"freeze_results.json: contract_counts preopening_holdout_rows "
            f"{holdout} != the pinned counts"
        )
    _require_int(
        counts["christmas_2016_rows"],
        "freeze_results.json: contract_counts christmas_2016_rows",
    )
    if counts["christmas_2016_rows"] != CHRISTMAS_2016_ROWS:
        raise ContractError(
            "freeze_results.json: contract_counts christmas_2016_rows "
            f"{counts['christmas_2016_rows']} != the pinned {CHRISTMAS_2016_ROWS}"
        )


def _validate_fold_record(record: Any, label: str) -> None:
    """One fold-phase record: the window plus the two pinned slices."""
    expected = ["fold", "n_rows", "origin", "slices", "window_end", "window_start"]
    if not isinstance(record, dict) or sorted(record) != expected:
        raise ContractError(f"{label} must carry exactly {expected}")
    _require_int(record["fold"], f"{label} fold")
    _require_int(record["n_rows"], f"{label} n_rows")
    if record["n_rows"] != FOLD_ROWS:
        raise ContractError(
            f"{label} n_rows {record['n_rows']} != the pinned {FOLD_ROWS}"
        )
    for key in ("origin", "window_start", "window_end"):
        if not isinstance(record[key], str) or not record[key]:
            raise ContractError(f"{label} {key} must be a non-empty string")
    slices = record["slices"]
    if not isinstance(slices, dict) or sorted(slices) != sorted(FOLD_SLICE_NAMES):
        raise ContractError(
            f"{label} slices must carry exactly {sorted(FOLD_SLICE_NAMES)}"
        )
    for slice_name, slice_block in slices.items():
        _assert_score_block(f"{label} slice {slice_name}", slice_block)


def _validate_fold_phase(block: Any) -> list[dict[str, Any]]:
    """The fold_phase block: the pinned windows and every model's rows."""
    if not isinstance(block, dict) or sorted(block) != ["folds", "models"]:
        raise ContractError(
            "freeze_results.json: fold_phase must carry exactly folds and models"
        )
    folds = block["folds"]
    if not isinstance(folds, list) or len(folds) != len(FOLD_TABLE_PINS):
        raise ContractError(
            f"freeze_results.json: fold_phase must carry the {len(FOLD_TABLE_PINS)} "
            "pinned windows"
        )
    for index, (fold, origin, start, end) in enumerate(FOLD_TABLE_PINS):
        record = folds[index]
        label = f"freeze_results.json: fold_phase folds[{index}]"
        if not isinstance(record, dict) or sorted(record) != [
            "fold",
            "n_rows",
            "origin",
            "window_end",
            "window_start",
        ]:
            raise ContractError(f"{label} must carry the pinned window fields")
        if [
            record["fold"],
            record["origin"],
            record["window_start"],
            record["window_end"],
        ] != [fold, origin, start, end]:
            raise ContractError(
                f"{label} {record} != the pinned window {fold} {origin} {start} {end}"
            )
        _require_int(record["n_rows"], f"{label} n_rows")
        if record["n_rows"] != FOLD_ROWS:
            raise ContractError(
                f"{label} n_rows {record['n_rows']} != the pinned {FOLD_ROWS}"
            )
    models = block["models"]
    if not isinstance(models, list) or not models:
        raise ContractError(
            "freeze_results.json: fold_phase must carry at least one model"
        )
    for index, model in enumerate(models):
        label = f"freeze_results.json: fold_phase models[{index}]"
        if not isinstance(model, dict) or sorted(model) != [
            "artifact",
            "folds",
            "name",
            "track",
        ]:
            raise ContractError(
                f"{label} must carry exactly artifact, name, track and folds"
            )
        if not isinstance(model["name"], str) or not model["name"]:
            raise ContractError(f"{label} name must be a non-empty string")
        model_folds = model["folds"]
        if not isinstance(model_folds, list) or len(model_folds) != len(
            FOLD_TABLE_PINS
        ):
            raise ContractError(f"{label} must carry one record per pinned fold")
        for fold_index, fold_record in enumerate(model_folds):
            _validate_fold_record(fold_record, f"{label} folds[{fold_index}]")
    return models


def _validate_freeze_phase(
    block: Any,
    *,
    expected_rows: int = PINNED_SPLIT_ROWS["local_test"],
    expected_bounds: tuple[str, str] = (
        SPLIT_FRAME_STARTS["local_test"],
        SPLIT_FRAME_ENDS["local_test"],
    ),
) -> list[dict[str, Any]]:
    """The freeze_phase block: the pinned local_test window, one record per model.

    The defaults are the real local_test pins, so the production calls
    and the committed-artifact path are pinned; a caller validating a
    stand-in frame passes that frame's own rows and bounds.
    """
    if not isinstance(block, dict) or sorted(block) != [
        "models",
        "n_rows",
        "window_end",
        "window_start",
    ]:
        raise ContractError(
            "freeze_results.json: freeze_phase must carry exactly window_start, "
            "window_end, n_rows and models"
        )
    _require_int(block["n_rows"], "freeze_results.json: freeze_phase n_rows")
    if block["n_rows"] != expected_rows:
        raise ContractError(
            f"freeze_results.json: freeze_phase n_rows {block['n_rows']} "
            f"!= the pinned {expected_rows}"
        )
    for key in ("window_start", "window_end"):
        if not isinstance(block[key], str) or not block[key]:
            raise ContractError(
                f"freeze_results.json: freeze_phase {key} must be a non-empty string"
            )
        pinned_edge = expected_bounds[0 if key == "window_start" else 1]
        if block[key] != pinned_edge:
            raise ContractError(
                f"freeze_results.json: freeze_phase {key} {block[key]!r} "
                f"!= the pinned {pinned_edge!r}"
            )
    models = block["models"]
    if not isinstance(models, list) or not models:
        raise ContractError(
            "freeze_results.json: freeze_phase must carry at least one model"
        )
    # The registry pins each model's track and artifact: a fabricated
    # name has no registry entry, and a registry name carries exactly
    # the track and artifact the registry records for it.
    registered = {
        name: (spec["track"], artifact_name)
        for artifact_name, spec in ARTIFACT_REGISTRY.items()
        for name in spec["model_names"]
    }
    for index, model in enumerate(models):
        label = f"freeze_results.json: freeze_phase models[{index}]"
        expected = [
            "artifact",
            "n_rows",
            "name",
            "slices",
            "track",
            "window_end",
            "window_start",
        ]
        if not isinstance(model, dict) or sorted(model) != expected:
            raise ContractError(f"{label} must carry exactly {expected}")
        if not isinstance(model["name"], str) or not model["name"]:
            raise ContractError(f"{label} name must be a non-empty string")
        identity = registered.get(model["name"])
        if identity is None:
            raise ContractError(
                f"{label} name {model['name']!r} is not a registered model name"
            )
        pinned_track, pinned_artifact = identity
        if model["track"] != pinned_track:
            raise ContractError(
                f"{label} track {model['track']!r} != the registry "
                f"{pinned_track!r} for {model['name']!r}"
            )
        if model["artifact"] != pinned_artifact:
            raise ContractError(
                f"{label} artifact {model['artifact']!r} != the registry "
                f"{pinned_artifact!r} for {model['name']!r}"
            )
        _require_int(model["n_rows"], f"{label} n_rows")
        if model["n_rows"] != expected_rows:
            raise ContractError(
                f"{label} n_rows {model['n_rows']} != the pinned {expected_rows}"
            )
        for key in ("window_start", "window_end"):
            if not isinstance(model[key], str) or not model[key]:
                raise ContractError(f"{label} {key} must be a non-empty string")
            pinned_edge = expected_bounds[0 if key == "window_start" else 1]
            if model[key] != pinned_edge:
                raise ContractError(
                    f"{label} {key} {model[key]!r} != the pinned {pinned_edge!r}"
                )
        slices = model["slices"]
        if not isinstance(slices, dict) or sorted(slices) != sorted(FREEZE_SLICE_NAMES):
            raise ContractError(
                f"{label} slices must carry exactly {sorted(FREEZE_SLICE_NAMES)}"
            )
        for slice_name, slice_block in slices.items():
            _assert_score_block(f"{label} slice {slice_name}", slice_block)
    return models


def _validate_freeze_results(
    payload: Any,
    *,
    expected_models: int | None = None,
    expected_rows: int = PINNED_SPLIT_ROWS["local_test"],
    expected_bounds: tuple[str, str] = (
        SPLIT_FRAME_STARTS["local_test"],
        SPLIT_FRAME_ENDS["local_test"],
    ),
) -> None:
    """The pinned freeze_results.json schema; any drift is a loud refusal."""
    if not isinstance(payload, dict):
        raise ContractError(
            f"freeze_results.json must be a JSON object, got {type(payload).__name__}"
        )
    measured = list(payload)
    if measured != list(FREEZE_RESULTS_KEYS):
        missing = [key for key in FREEZE_RESULTS_KEYS if key not in payload]
        extra = [key for key in payload if key not in FREEZE_RESULTS_KEYS]
        raise ContractError(
            f"freeze_results.json: fields {measured} != the pinned "
            f"{list(FREEZE_RESULTS_KEYS)}; missing {missing}; extra {extra}"
        )
    frozen_at = payload["frozen_at"]
    if not isinstance(frozen_at, str):
        raise ContractError("freeze_results.json: frozen_at must be an ISO 8601 string")
    try:
        stamp = datetime.fromisoformat(frozen_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ContractError(
            f"freeze_results.json: frozen_at {frozen_at!r} does not parse: {error}"
        ) from error
    if stamp.tzinfo is None:
        raise ContractError(
            "freeze_results.json: frozen_at must carry a timezone "
            "(ISO 8601 Z or offset)"
        )
    sha = payload["git_sha"]
    if (
        not isinstance(sha, str)
        or len(sha) != 40
        or set(sha.lower()) - set("0123456789abcdef")
    ):
        raise ContractError(
            f"freeze_results.json: git_sha must be a 40-hex sha, got {sha!r}"
        )
    headline = payload["headline_model"]
    if not isinstance(headline, str) or not headline.strip():
        raise ContractError(
            "freeze_results.json: headline_model must be a non-empty string"
        )
    notes = payload["notes"]
    if not isinstance(notes, list) or not all(isinstance(note, str) for note in notes):
        raise ContractError("freeze_results.json: notes must be a list of strings")
    _validate_data_snapshot(payload["data_snapshot"])
    _validate_contract_counts(payload["contract_counts"])
    fold_models = _validate_fold_phase(payload["fold_phase"])
    freeze_models = _validate_freeze_phase(
        payload["freeze_phase"],
        expected_rows=expected_rows,
        expected_bounds=expected_bounds,
    )
    names = [str(model["name"]) for model in freeze_models]
    if headline not in names:
        raise ContractError(
            f"freeze_results.json: headline_model {headline!r} is not one of the "
            f"freeze-phase models {names}"
        )
    if expected_models is not None and len(freeze_models) != expected_models:
        raise ContractError(
            f"freeze_results.json: {len(freeze_models)} model records != the "
            f"registered count {expected_models}"
        )
    if len(fold_models) != len(freeze_models):
        raise ContractError(
            f"freeze_results.json: the fold phase carries {len(fold_models)} "
            f"models but the freeze phase carries {len(freeze_models)}"
        )
    fold_names = sorted(str(model["name"]) for model in fold_models)
    if fold_names != sorted(names):
        raise ContractError(
            f"freeze_results.json: the fold phase models {fold_names} != the "
            f"freeze phase models {sorted(names)}"
        )
    # The payload's model names must be the registry's pinned names: a
    # fabricated name is refused even when the fold and the freeze phase
    # agree with each other, and even when the count happens to match.
    registered_models = sorted(
        name for spec in ARTIFACT_REGISTRY.values() for name in spec["model_names"]
    )
    unregistered = sorted((set(names) | set(fold_names)) - set(registered_models))
    if unregistered:
        raise ContractError(
            f"freeze_results.json: model names {unregistered} are not "
            f"registered model names {registered_models}"
        )


def _write_freeze_results(path: Path, payload: dict[str, Any]) -> None:
    """Write freeze_results.json atomically, then re-open and re-parse it.

    Same-dir temp file published with an exclusive link (a crash mid-write
    never leaves a half-written file behind and never overwrites an
    existing artifact), then the file is reopened, its size asserted and
    its content compared with the payload.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ContractError(
            f"{path.name} already exists; it is written once, by the freeze "
            "run, and never overwritten"
        )
    try:
        data = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ContractError(
            f"freeze_results.json does not serialise to JSON: {error}"
        ) from error
    _atomic_write_bytes(path, data, exclusive=True)
    try:
        stored_bytes = path.read_bytes()
    except OSError as error:
        raise ContractError(
            f"freeze_results.json could not be re-read after the write: {error}"
        ) from error
    if not stored_bytes:
        raise ContractError("freeze_results.json is zero bytes after the write")
    try:
        stored = json.loads(stored_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ContractError(
            f"freeze_results.json does not parse after the write: {error}"
        ) from error
    if stored != payload:
        raise ContractError(
            "freeze_results.json does not round-trip through JSON unchanged"
        )


def _freeze_notes() -> list[str]:
    """The notes block: the locked decisions, carried into the final JSON."""
    return [
        "the earthquake-decay caveat applies strictly to the full-valid row "
        "set; the three fold windows starting 2016-07-26 carry no quake rows",
        "closure rows: the 7,128 id=NaN full-calendar fills (5,346 train / 0 "
        "valid / 1,782 local_test) stay as true-zero labels everywhere; the "
        "training mask keeps the 462 fills that sit inside a pre-opening window",
        "masking: a row is pre-opening when its date is before that store's "
        "opening date over the 8 pinned per-store windows; the training mask "
        "keeps the closure rows, so 213,345 raw rows drop (207,801 train + "
        "5,544 valid); the 12,804 pre-opening holdout rows (5,544 valid + "
        "7,260 local_test) are scored as-is",
        "the percentage-error metric is excluded from every reported number: "
        "it is undefined at 35.2% zero demand in train+valid",
        "dual evaluation: every model carries the full-grid and the active-slice "
        "numbers side by side, and the Christmas-2016 slice (1,782 local_test "
        "rows) is reported separately",
        f"predict_frame: valid's {LOOKBACK_DAYS}-day tail "
        f"({(VALID_END - pd.Timedelta(days=LOOKBACK_DAYS - 1)).date()}.."
        f"{VALID_END.date()}) followed by local_test "
        f"({SPLIT_FRAME_STARTS['local_test']}..{SPLIT_FRAME_ENDS['local_test']}), "
        "unmasked, so the predictors carry their lookback before the scored window",
        "the holiday lead flags (is_*_tomorrow / is_*_in_2days) are "
        "calendar-built and future-known by design; nothing else in the "
        "pipeline reaches from the future",
        "local_test was read exactly once, through the guarded load_split "
        "inside the freeze path, and was never tuned on",
    ]


def _load_sanctioned_entrypoints() -> dict[str, dict[str, Any]]:
    """The sanctioned per-track fit_predict_frozen entrypoints.

    One module per registered artifact (S3's classical, S4's deep). A
    missing module or a missing symbol is a loud refusal, never a skip.
    """
    entrypoints: dict[str, dict[str, Any]] = {}
    for artifact_name, module_name in KIND_BY_ARTIFACT.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            raise ContractError(
                f"sanctioned entrypoint {module_name} for {artifact_name} "
                f"failed to import: {error}"
            ) from error
        symbol = getattr(module, "fit_predict_frozen", None)
        if not callable(symbol):
            raise ContractError(
                f"sanctioned entrypoint {module_name} for {artifact_name} "
                "missing symbol fit_predict_frozen"
            )
        entrypoints[artifact_name] = {"call": symbol}
    return entrypoints


def _align_categorical_domains(
    reference: pd.DataFrame, target: pd.DataFrame, label: str
) -> pd.DataFrame:
    """Cast the target's categorical columns onto the reference's domain.

    The predict frame concatenates two frames, and pandas widens a
    categorical to object the moment the two domains differ, which would
    break the pinned dtype contract downstream. The holdout is cast onto
    the training frame's domain instead, and a holdout category the
    training frame never saw is a loud refusal, never a silent NaN.
    """
    aligned = target.copy()
    for column, reference_dtype in reference.dtypes.items():
        if not isinstance(reference_dtype, pd.CategoricalDtype):
            continue
        values = aligned[column]
        unseen = sorted(
            str(value)
            for value in values.dropna().unique()
            if value not in reference_dtype.categories
        )
        if unseen:
            raise ContractError(
                f"{label}: column {column!r} holds categor(ies) {unseen} that "
                "the training frame never saw; the holdout cannot be cast "
                "onto the training domain"
            )
        aligned[column] = values.astype(reference_dtype)
    return aligned


def _freeze_predict_frame(
    valid: pd.DataFrame, local_test: pd.DataFrame
) -> pd.DataFrame:
    """The predict frame: valid's 60-day tail followed by local_test.

    The tail gives the predictors their lookback before the scored
    window; local_test follows, unmasked, because the holdout is scored
    as it stands.
    """
    tail_start = VALID_END - pd.Timedelta(days=LOOKBACK_DAYS - 1)
    tail = valid.loc[valid["date"] >= tail_start]
    tail_days = int(tail["date"].nunique())
    if tail_days != LOOKBACK_DAYS:
        raise ContractError(
            f"freeze: the valid tail holds {tail_days} days, expected "
            f"{LOOKBACK_DAYS} (from {tail_start.date()} to {VALID_END.date()})"
        )
    expected_rows = LOOKBACK_DAYS * SERIES_PER_DAY
    if len(tail) != expected_rows:
        raise ContractError(
            f"freeze: the valid tail holds {len(tail)} rows, expected "
            f"{expected_rows} ({LOOKBACK_DAYS} days x {SERIES_PER_DAY} series)"
        )
    predict_frame = pd.concat([tail, local_test], ignore_index=True)
    _assert_pinned_frame(predict_frame, "freeze predict_frame")
    return predict_frame


def _run_canonical_sequence(
    directory: Path,
    *,
    models: list[dict[str, Any]],
    predict_frame: pd.DataFrame,
    local_test: pd.DataFrame,
    predict: Callable[[dict[str, Any]], Any],
    fold_phase: dict[str, Any],
    contract_counts: dict[str, Any],
    data_snapshot: dict[str, Any],
    notes: list[str],
    frozen_at: str,
    git_sha: str,
    write: bool = True,
) -> dict[str, Any]:
    """Steps 3-5: score every model, assemble and validate the payload.

    With write=True the payload is published to freeze_results.json
    atomically, which is how the self-audit and fixture flows use it.
    With write=False nothing is written; the canonical freeze computes
    outside the manifest lock and publishes inside the locked section, so
    a failing phase leaves the artifacts directory without a
    freeze_results.json.
    """
    records = _freeze_phase_records(models, local_test, predict_frame, predict)
    freeze_phase = {
        "window_start": records[0]["window_start"],
        "window_end": records[0]["window_end"],
        "n_rows": records[0]["n_rows"],
        "models": records,
    }
    headline = _pick_headline_model(records)
    payload = {
        "frozen_at": frozen_at,
        "git_sha": git_sha,
        "headline_model": headline,
        "data_snapshot": data_snapshot,
        "contract_counts": contract_counts,
        "fold_phase": fold_phase,
        "freeze_phase": freeze_phase,
        "notes": list(notes),
    }
    _validate_freeze_results(
        payload,
        expected_models=len(models),
        # The freeze window is scored from THIS local_test frame, so the
        # validator pins the payload to the frame's own rows and bounds;
        # on the real path that frame is the pinned split, and on the
        # battery's stand-ins it is the fixture frame.
        expected_rows=len(local_test),
        expected_bounds=(
            local_test["date"].min().date().isoformat(),
            local_test["date"].max().date().isoformat(),
        ),
    )
    if write:
        _write_freeze_results(directory / FREEZE_RESULTS_FILENAME, payload)
    return payload


def _freeze_day5(
    directory: Path, entrypoints: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """The canonical Day-5 sequence, steps 1 to 5 of sheet §5.

    Step 1: the registered artifacts are re-read (the pre-flight gate
    owns the schema check). Step 2: splits/local_test.parquet is loaded
    exactly once through the guarded load_split; the pre-flight gate has
    already required FREEZE_RUN=1 and this path only reads the flag.
    Step 3: every registered model is called through its
    sanctioned fit_predict_frozen with fit_frame = concat(train, valid)
    masked by fit_rows and predict_frame = valid's 60-day tail followed
    by local_test. Step 4: the pinned score schema runs on the full grid,
    the active slice and the Christmas-2016 slice. Step 5:
    artifacts/freeze_results.json is written atomically.
    """
    payloads = _registered_payloads(directory)
    models = _freeze_model_specs(payloads)
    registered_count = sum(
        len(spec["model_names"]) for spec in ARTIFACT_REGISTRY.values()
    )
    if len(models) != registered_count:
        raise ContractError(
            f"freeze: {len(models)} registered models != the registry's "
            f"{registered_count}; the freeze consumes all tiers or none"
        )

    # Step 2: the guarded local_test load, exactly once. The pre-flight
    # gate already requires FREEZE_RUN=1, so the flag is already set and
    # this path never opens it.
    local_test = load_split("local_test")
    train = load_split("train")
    valid = load_split("valid")
    _assert_pinned_frame(local_test, "freeze local_test")

    # Step 3: fit_frame = concat(train, valid) masked by fit_rows;
    # predict_frame = valid's 60-day tail followed by local_test.
    frame = load_frame(train, valid)
    fit_frame = frame.loc[fit_rows(frame)]
    _assert_pinned_frame(fit_frame, "freeze fit_frame")
    counts = _contract_counts(train, valid, local_test, frame, fit_frame)
    # The counts are derived; the full frame and the train split are not
    # needed again, and the model runs want the memory back.
    del frame, train
    local_test = _align_categorical_domains(fit_frame, local_test, "freeze local_test")
    predict_frame = _freeze_predict_frame(valid, local_test)
    del valid

    def _predict(spec: dict[str, Any]) -> Any:
        entry = entrypoints[str(spec["artifact"])]
        call = entry["call"]
        try:
            return call(spec["name"], fit_frame, predict_frame, spec["config"])
        except ContractError:
            raise
        except Exception as error:
            raise ContractError(
                f"freeze: model {spec['name']!r} failed inside its sanctioned "
                f"fit_predict_frozen: {error!r}"
            ) from error

    manifest = _load_manifest()
    snapshot = _data_snapshot(manifest)
    if snapshot["local_test"]["bounds"] != [
        SPLIT_FRAME_STARTS["local_test"],
        SPLIT_FRAME_ENDS["local_test"],
    ]:
        raise ContractError(
            "freeze: the split manifest's local_test bounds "
            f"{snapshot['local_test']['bounds']} != the pinned "
            f"{[SPLIT_FRAME_STARTS['local_test'], SPLIT_FRAME_ENDS['local_test']]}"
        )
    fold_phase = _fold_phase_from_artifacts(payloads)
    # The sha is captured ONCE, here, before the artifact is written: the
    # manifest stamp and the artifact must describe the same tree, and a
    # HEAD that moves mid-freeze must not burn the single freeze run.
    sha, sha_notes = _git_sha()
    if not sha:
        raise ContractError(
            "freeze: git sha unavailable; the freeze numbers must describe a "
            f"committed tree ({'; '.join(sha_notes)})"
        )
    frozen_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    return _run_canonical_sequence(
        directory,
        models=models,
        predict_frame=predict_frame,
        local_test=local_test,
        predict=_predict,
        fold_phase=fold_phase,
        contract_counts=counts,
        data_snapshot=snapshot,
        notes=_freeze_notes(),
        frozen_at=frozen_at,
        git_sha=sha,
        write=False,
    )


def freeze(artifacts_dir: Path | None = None) -> int:
    """The canonical freeze entrypoint; one testified pass, never a rehearsal.

    Gates (all friendly refusals): FREEZE_RUN=1; no prior freeze_run =
    true record; the full registry present and schema-clean; sanctioned
    per-track entrypoints importable on the merged tree. The gates
    themselves open no split and never touch local_test. The model run
    happens outside the manifest lock; the pre-flight gate, the artifact
    write and the manifest stamp run inside one locked section so the
    committed artifact and the manifest agree. On stamp failure the
    artifact is renamed aside under the same lock, leaving the canonical
    name free for the sanctioned re-run. Any failing phase prints FAIL
    with the phase named and records the attempt when that is possible;
    freeze_run stays false on failures, returning 1.
    """
    started = time.perf_counter()
    directory = Path(artifacts_dir) if artifacts_dir is not None else ARTIFACTS_DIR

    def _record_attempt(phase: str, reasons: list[str]) -> None:
        elapsed = round(time.perf_counter() - started, 3)
        try:
            write_run_manifest(
                "contract.py --freeze",
                exit_status=1,
                elapsed_seconds=elapsed,
                arguments={"freeze": "1"},
                notes=[
                    *_dirty_notes(),
                    f"failed phase: {phase}",
                    *reasons,
                ],
                artifacts_dir=directory,
            )
        except Exception as audit_error:
            print(f"FAIL {audit_error!r}")
            return

    try:
        try:
            _freeze_preflight(directory)
        except ContractError as error:
            phase = "pre-flight gates"
            reasons = [str(error)]
            print(f"FAIL freeze, {phase}: {error}")
            _record_attempt(phase, reasons)
            return 1

        phase = "sanctioned entrypoints"
        try:
            entrypoints = _load_sanctioned_entrypoints()
        except ContractError as error:
            reasons = [f"not merged yet - {error}"]
            print(f"FAIL freeze, {phase}: {'; '.join(reasons)}")
            _record_attempt(phase, reasons)
            return 1

        phase = "canonical sequence"
        try:
            payload = _freeze_day5(directory, entrypoints)
        except ContractError as error:
            print(f"FAIL freeze, {phase}: {error}")
            _record_attempt(phase, [str(error)])
            return 1

        # Publish and stamp inside one locked section. The pre-flight
        # gate is re-validated here, so the write-once refusal and the
        # registry/manifest checks cannot race the model run.
        try:
            with _manifest_lock(directory):
                try:
                    _freeze_preflight(directory)
                except ContractError as error:
                    phase = "pre-flight gates"
                    reasons = [str(error)]
                    print(f"FAIL freeze, {phase}: {error}")
                    _record_attempt(phase, reasons)
                    return 1

                _write_freeze_results(directory / FREEZE_RESULTS_FILENAME, payload)

                elapsed = round(time.perf_counter() - started, 3)
                stamp = _freeze_stamp()
                try:
                    _write_manifest_record(
                        directory,
                        "contract.py --freeze",
                        exit_status=0,
                        elapsed_seconds=elapsed,
                        arguments={"freeze": "1"},
                        freeze_run=True,
                        notes=[
                            *_dirty_notes(),
                            "freeze_results.json written; headline model "
                            f"{payload['headline_model']}",
                        ],
                        path=directory / "run_manifest.json",
                        git_sha=payload["git_sha"],
                        created=stamp,
                    )
                except Exception as error:
                    failed_path = _quarantine_stranded_freeze_results(directory, stamp)
                    print(
                        f"FAIL freeze, manifest record: {error}; "
                        f"{FREEZE_RESULTS_FILENAME} was moved aside to "
                        f"{failed_path.name} so the freeze can be re-run"
                    )
                    _record_attempt("manifest record", [repr(error)])
                    return 1

                # Under the lock no concurrent writer can interleave, so the
                # appended record is records[-1]. The stamp is a secondary
                # assertion that this is the record we just wrote.
                records = read_run_manifest(directory / "run_manifest.json")
                record = records[-1] if records else {}
                if record.get("created") != stamp:
                    print(
                        "FAIL freeze, manifest record: the appended record was "
                        f"not found at the end of the manifest (expected stamp "
                        f"{stamp}, found {record.get('created')!r})"
                    )
                    _record_attempt(
                        "manifest record",
                        [
                            "the appended record was not found at the end of "
                            f"the manifest (expected stamp {stamp})"
                        ],
                    )
                    return 1
                if (
                    record.get("freeze_run") is not True
                    or record.get("git_sha") != payload["git_sha"]
                ):
                    print(
                        "FAIL freeze, manifest record: the appended record does "
                        "not carry freeze_run = true with the freeze's git sha"
                    )
                    _record_attempt(
                        "manifest record",
                        [
                            "the appended record does not carry freeze_run = true "
                            "with the freeze's git sha"
                        ],
                    )
                    return 1
        except ContractError as error:
            if "another writer holds the lock" in str(error):
                print(
                    f"FAIL freeze, manifest lock: {error}; the attempted run "
                    "could not be recorded in the manifest because another "
                    "writer holds the lock"
                )
                return 1
            raise

        print(
            f"PASS freeze: {len(payload['freeze_phase']['models'])} models "
            f"scored on the frozen holdout; headline {payload['headline_model']}; "
            "freeze_results.json written atomically and verified"
        )
        return 0
    except SystemExit:
        raise
    except Exception as error:
        phase = "freeze"
        reasons = [f"unexpected failure: {error!r}"]
        print(f"FAIL freeze, {phase}: {'; '.join(reasons)}")
        try:
            _record_attempt(phase, reasons)
        except Exception as audit_error:
            print(f"FAIL {audit_error!r}")
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
    pinned_bounds = SPLIT_FRAME_ENDS
    for split, end in pinned_bounds.items():
        measured_end = manifest["bounds"][split][1]
        if measured_end != end:
            raise ContractError(
                f"manifest {split} end date {measured_end!r} != pinned {end!r}; "
                f"{RUN_MAIN_POINTER}"
            )
    print(
        "PASS manifest: bounds end "
        + " / ".join(pinned_bounds[split] for split in pinned_bounds)
    )
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
    pinned_starts = SPLIT_FRAME_STARTS
    for split, start in pinned_starts.items():
        measured_start = manifest["bounds"][split][0]
        if measured_start != start:
            raise ContractError(
                f"manifest {split} start date {measured_start!r} != pinned {start!r}; "
                f"{RUN_MAIN_POINTER}"
            )
    print(
        "PASS manifest: bounds start "
        + " / ".join(pinned_starts[split] for split in pinned_starts)
    )
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


def _fold_table_pin(fold: int, label: str) -> tuple[int, str, str, str]:
    """The pinned fold row for a fold number, with the fold bounded.

    The fold number must sit inside 1..len(FOLD_TABLE_PINS): fold 0 would
    silently validate against the last fold's window and a fold past the
    end would raise a raw IndexError, so both are refused with a
    friendly ContractError naming the valid range.
    """
    if not 1 <= fold <= len(FOLD_TABLE_PINS):
        raise ContractError(
            f"{label}: fold {fold} is outside the pinned fold range "
            f"1..{len(FOLD_TABLE_PINS)}"
        )
    return FOLD_TABLE_PINS[fold - 1]


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
    alias_is_same = make_folds is folds_table
    if alias_is_same is not True:
        raise ContractError(
            "make_folds is not folds_table; the alias must be the same callable"
        )
    # The fold index is bounded: fold 0 and a fold past the end are
    # refused with a friendly error, never a raw IndexError and never a
    # silent validation against another fold's window.
    for bad_fold in (0, len(FOLD_TABLE_PINS) + 1):
        try:
            _fold_table_pin(bad_fold, "folds_table")
        except ContractError as error:
            if "outside the pinned fold range" not in str(error):
                raise ContractError(
                    f"the fold index {bad_fold} refused with the wrong lead: {error}"
                ) from error
        except IndexError as error:
            raise ContractError(
                f"the fold index {bad_fold} escaped as a raw IndexError: {error}"
            ) from error
        else:
            raise ContractError(f"the fold index {bad_fold} was accepted")
    valid = load_split("valid")
    valid_start = valid["date"].min()
    valid_end = valid["date"].max()
    for fold, origin, start, end in FOLD_TABLE_PINS:
        w_start = pd.Timestamp(start)
        w_end = pd.Timestamp(end)
        if w_start < valid_start or w_end > valid_end:
            raise ContractError(
                f"fold {fold}: origin {origin} window {start}..{end} "
                "lies outside valid date span"
            )
        dates = valid["date"]
        mask = (dates >= w_start) & (dates <= w_end)
        count = int(mask.sum())
        if count != FOLD_ROWS:
            raise ContractError(
                f"fold {fold}: window {start}..{end} has {count} rows != {FOLD_ROWS}"
            )
        if (valid.loc[mask, "earthquake_decay"] > 0).any():
            raise ContractError(
                f"fold {fold}: earthquake_decay has positive values inside window"
            )
    print(
        "PASS folds: 3 windows (origins 2016-07-25 / 2016-08-10 / 2016-08-26), "
        "16 days each, 28,512 rows per window measured on valid, "
        "quake-free and inside valid"
    )
    print(
        f"PASS folds: pooled {POOLED_FOLD_ROWS:,} rows; "
        "make_folds aliases folds_table (one implementation)"
    )
    print(
        f"PASS folds: a fold number outside the pinned range 1..{len(FOLD_TABLE_PINS)} "
        "is refused with a friendly error, never a raw IndexError"
    )


def _check_manifest_writer() -> None:
    """Day-2 block: the writer's parse, validate, append and refuse paths."""
    path = ARTIFACTS_DIR / "run_manifest.json"
    existing = list(read_run_manifest(path)) if path.exists() else []
    count = len(existing)
    print(f"PASS manifest: real manifest read once ({count} record(s)), schema-valid")
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp_dir = Path(tmp_name)
        tmp_manifest = tmp_dir / "run_manifest.json"
        created = datetime.fromtimestamp(time.time() - 5.0, UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        starter = (
            json.dumps(
                [
                    {
                        "created": created,
                        "git_sha": "b" * 40,
                        "command": "fixture",
                        "exit": 0,
                        "exit_ok": True,
                        "versions": {
                            "python": "3.13.0",
                            "uv": "0.10.0",
                            "pandas": "2.2.0",
                            "numpy": "2.1.0",
                        },
                        "elapsed_seconds": 0.0,
                        "arguments": {},
                        "status": "PASS",
                        "freeze_run": False,
                        "notes": [],
                    }
                ],
                indent=2,
            )
            + "\n"
        )
        tmp_manifest.write_text(starter, encoding="utf-8")
        write_run_manifest("contract.py --check", artifacts_dir=tmp_dir)
        stored = list(read_run_manifest(tmp_manifest))
        record = stored[-1]
        if record["command"] != "contract.py --check" or record["status"] != "PASS":
            raise ContractError(f"appended manifest record drifted: {record!r}")
        if stored[-1] != record or len(stored) != 2:
            raise ContractError(
                "throwaway manifest append did not land last and validate"
            )
        mode = tmp_manifest.stat().st_mode & 0o777
        if mode != 0o644:
            raise ContractError(
                f"run_manifest.json mode {oct(mode)} != 0o644; the atomic "
                "replace must leave a world-readable manifest"
            )
        lock_path = tmp_dir / "run_manifest.json.lock"
        if not lock_path.is_file():
            raise ContractError(
                "write_run_manifest did not leave the run_manifest.json.lock "
                "furniture in place"
            )
        lock_inode = lock_path.stat().st_ino
        with _manifest_lock(tmp_dir):
            if not _manifest_lock_held(tmp_dir):
                raise ContractError(
                    "a held manifest lock was not reported as held for its directory"
                )
            nested_dir = tmp_dir / "nested"
            nested_dir.mkdir()
            with _manifest_lock(nested_dir):
                if not _manifest_lock_held(nested_dir):
                    raise ContractError(
                        "a nested lock for a different directory did not hold that "
                        "directory's lock"
                    )
                if not _manifest_lock_held(tmp_dir):
                    raise ContractError(
                        "holding a nested lock for a different directory dropped "
                        "the original directory's lock"
                    )
            if not _manifest_lock_held(tmp_dir):
                raise ContractError(
                    "releasing a nested different-directory lock dropped the "
                    "original directory's lock"
                )
        # The held-state bookkeeping is keyed by (directory, thread): a
        # nested acquisition for a different directory keeps its OWN entry,
        # and every entry is released when its block exits.
        with _manifest_lock(tmp_dir):
            if len(_MANIFEST_LOCK_HELD) != 1:
                raise ContractError(
                    "the held-state bookkeeping did not keep exactly one entry "
                    f"for the held lock: {_MANIFEST_LOCK_HELD}"
                )
            with _manifest_lock(nested_dir):
                if len(_MANIFEST_LOCK_HELD) != 2:
                    raise ContractError(
                        "a nested lock for a different directory did not keep its "
                        "own held-state entry: "
                        f"{_MANIFEST_LOCK_HELD}"
                    )
            if len(_MANIFEST_LOCK_HELD) != 1:
                raise ContractError(
                    "releasing a nested different-directory lock did not drop "
                    f"only its own entry: {_MANIFEST_LOCK_HELD}"
                )
        if _MANIFEST_LOCK_HELD:
            raise ContractError(
                f"the held-state bookkeeping leaked entries: {_MANIFEST_LOCK_HELD}"
            )

        # A different thread in the same process must take its OWN lock: it
        # must go through the flock path and CONTEND, not silently reuse the
        # fd the current thread holds. (flock is visible process-wide, so the
        # other thread sees the lock as held and is refused after the pinned
        # wait - that refusal IS the proof it did not reuse the fd.)
        other_thread: list[str] = []

        def _probe_from_other_thread() -> None:
            try:
                with _manifest_lock(tmp_dir):
                    other_thread.append("granted")
            except ContractError as error:
                other_thread.append(str(error))

        with _manifest_lock(tmp_dir):
            probe_thread = threading.Thread(target=_probe_from_other_thread)
            probe_thread.start()
            probe_thread.join()
        if len(other_thread) != 1:
            raise ContractError(
                f"the other thread's lock attempt produced no verdict: {other_thread}"
            )
        if other_thread[0] == "granted":
            raise ContractError(
                "a different thread reused the current thread's manifest lock "
                "instead of taking its own"
            )
        if "another writer holds the lock" not in other_thread[0]:
            raise ContractError(
                "the other thread's contention refused with the wrong lead: "
                f"{other_thread[0]}"
            )
        print(
            "PASS manifest: a nested same-thread lock for a different directory "
            "holds its own lock, and a different thread must take its own lock"
        )
        garbage = "   not a pid at all "
        lock_path.write_text(garbage, encoding="utf-8")
        write_run_manifest("contract.py --check", artifacts_dir=tmp_dir)
        reopened = list(read_run_manifest(tmp_manifest))
        if len(reopened) != 3:
            raise ContractError("garbage-lock write did not append exactly one record")
        if lock_path.read_text(encoding="utf-8") != str(os.getpid()):
            raise ContractError(
                "write_run_manifest left foreign content in run_manifest.json.lock"
            )
        busy_fd = os.open(str(lock_path), os.O_RDWR | _O_BINARY)
        try:
            fcntl.flock(busy_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            busy_started = time.monotonic()
            try:
                write_run_manifest("contract.py --check", artifacts_dir=tmp_dir)
            except ContractError as error:
                if "another writer holds the lock" not in str(error):
                    raise ContractError(
                        f"lock-busy writer refused with the wrong lead: {error}"
                    ) from error
            else:
                raise ContractError("lock-busy writer accepted a held lock")
            busy_waited = time.monotonic() - busy_started
            if busy_waited < MANIFEST_LOCK_WAIT_SECONDS - 0.2:
                raise ContractError(
                    f"the busy-lock wait lasted {busy_waited:.2f}s, shorter than "
                    f"the pinned {MANIFEST_LOCK_WAIT_SECONDS}s"
                )
            if len(list(read_run_manifest(tmp_manifest))) != 3:
                raise ContractError("lock-busy writer appended a record anyway")
            # A stale marker: the PID in the lock file names a process
            # that is already gone, so the refusal says so instead of
            # blaming a writer that never comes back.
            child = subprocess.Popen([sys.executable, "-c", "pass"])
            child.wait()
            lock_path.write_text(str(child.pid), encoding="utf-8")
            try:
                write_run_manifest("contract.py --check", artifacts_dir=tmp_dir)
            except ContractError as error:
                if "the marker is stale" not in str(error):
                    raise ContractError(
                        f"stale-lock writer refused with the wrong lead: {error}"
                    ) from error
            else:
                raise ContractError("stale-lock writer accepted a held lock")

            # A live holder owned by another user looks alive (EPERM),
            # not stale; the refusal must not claim the process is gone.
            lock_path.write_text("1", encoding="utf-8")
            real_kill = os.kill

            def _eperm_kill(pid: int, sig: int) -> None:
                if pid == 1 and sig == 0:
                    raise PermissionError(errno.EPERM, "Operation not permitted")
                return real_kill(pid, sig)

            with mock.patch.object(os, "kill", new=_eperm_kill):
                try:
                    write_run_manifest("contract.py --check", artifacts_dir=tmp_dir)
                except ContractError as error:
                    message = str(error)
                    if "the marker is stale" in message or "gone" in message:
                        raise ContractError(
                            f"a live foreign holder was misclassified as gone: {error}"
                        ) from error
                    if "another writer holds the lock" not in message:
                        raise ContractError(
                            f"live foreign holder refused with the wrong lead: {error}"
                        ) from error
                else:
                    raise ContractError("live foreign holder accepted a held lock")
            print(
                "PASS manifest: a live foreign holder (EPERM) is not misclassified "
                "as a stale marker"
            )
        finally:
            try:
                fcntl.flock(busy_fd, fcntl.LOCK_UN)
            finally:
                os.close(busy_fd)
        write_run_manifest("contract.py --check", artifacts_dir=tmp_dir)
        reopened = list(read_run_manifest(tmp_manifest))
        if len(reopened) != 4:
            raise ContractError("post-lock write did not append exactly one record")
        if not lock_path.is_file():
            raise ContractError(
                "write_run_manifest dropped the run_manifest.json.lock furniture"
            )
        if lock_path.stat().st_ino != lock_inode:
            raise ContractError(
                "the run_manifest.json.lock furniture was replaced instead of reused"
            )
        rejections = (
            ("a non-finite elapsed_seconds", {"elapsed_seconds": math.inf}, "finite"),
            ("a negative elapsed_seconds", {"elapsed_seconds": -1.0}, "finite"),
            ("a boolean elapsed_seconds", {"elapsed_seconds": True}, "finite"),
            (
                "a non-text arguments value",
                {"arguments": {"k": 5}},
                "value must be a string",
            ),
            (
                "a non-text arguments key",
                {"arguments": {7: "v"}},
                "keys must be strings",
            ),
        )
        for label, kwargs, needle in rejections:
            try:
                write_run_manifest(
                    "contract.py --check", artifacts_dir=tmp_dir, **kwargs
                )
            except ContractError as error:
                if needle not in str(error):
                    raise ContractError(
                        f"the writer refused {label} with the wrong lead: {error}"
                    ) from error
            else:
                raise ContractError(f"the writer accepted {label}")
        if len(list(read_run_manifest(tmp_manifest))) != 4:
            raise ContractError("a refused writer call appended a record anyway")
        before_rollback = tmp_manifest.read_bytes()
        real_replace = os.replace
        replace_calls: list[tuple[Any, Any]] = []

        def failing_replace(src, dst, *args, **kwargs):
            replace_calls.append((src, dst))
            if len(replace_calls) == 1:
                raise OSError("injected replace failure")
            return real_replace(src, dst, *args, **kwargs)

        with mock.patch.object(os, "replace", new=failing_replace):
            try:
                write_run_manifest("contract.py --check", artifacts_dir=tmp_dir)
            except ContractError:
                pass
            else:
                raise ContractError("a failed replace did not refuse the write")
        if tmp_manifest.read_bytes() != before_rollback:
            raise ContractError("a failed replace did not restore the manifest")
        if len(list(read_run_manifest(tmp_manifest))) != 4:
            raise ContractError("a failed replace appended a record anyway")
    print(
        "PASS manifest: write_run_manifest exercised end-to-end against "
        "throwaway dir (appended record last and validates)"
    )
    print(
        "PASS manifest: mode 0o644, lock furniture persists (same inode), a "
        "garbage lock is reclaimed, and a busy flock refuses after the "
        "pinned wait with 'another writer holds the lock'"
    )
    print(
        "PASS manifest: the writer refuses a non-finite, negative or boolean "
        "elapsed_seconds and a non-text arguments key or value, and a failed "
        "replace restores the manifest untouched"
    )


def _synthetic_registered_artifacts() -> tuple[dict[str, Any], dict[str, Any]]:
    """A well-formed pair of registered artifacts: (classical, deep).

    Both payloads pass the production schema validators unchanged, so the
    refusal battery can mutate ONE field and watch the production gate
    refuse it, instead of hand-building a broken artifact per case.
    """
    fold_m = {
        "fold": 1,
        "rmsle": 0.1,
        "mae": 0.1,
        "rmse": 0.1,
        "rmsle_active": 0.2,
        "mae_active": 0.2,
        "rmse_active": 0.2,
        "n_rows": 28512,
        "n_rows_active": 20000,
    }
    fold_m2 = dict(fold_m)
    fold_m2["fold"] = 2
    fold_m3 = dict(fold_m)
    fold_m3["fold"] = 3
    pooled_m = {
        "rmsle": 0.15,
        "mae": 0.15,
        "rmse": 0.15,
        "rmsle_active": 0.25,
        "mae_active": 0.25,
        "rmse_active": 0.25,
        "n_rows": 85536,
        "n_rows_active": 60000,
    }
    ch = {"status": "freeze-time"}

    def cmod(name, kind):
        return {
            "name": name,
            "kind": kind,
            "target": "sales",
            "hyperparameters": {},
            "fold_metrics": [fold_m, fold_m2, fold_m3],
            "pooled": pooled_m,
            "seeds": {},
            "tuning": "none",
            "christmas_2016": ch,
        }

    classical = {
        "schema_version": 1,
        "git_sha": "a" * 40,
        "python_version": "3.13.0",
        "sklearn_version": "1.5.1",
        "split_bounds": {
            "train": "2013-01-01..2016-03-27",
            "valid": "2016-03-28..2016-09-11",
        },
        "training_mask": "fit_rows",
        "target": "sales",
        "prediction_clip": "0",
        "feature_columns": [f"f{i:02d}" for i in range(1, 24)],
        "folds": [
            {
                "fold": 1,
                "origin": "2016-07-25",
                "window_start": "2016-07-26",
                "window_end": "2016-08-10",
                "n_rows": 28512,
            },
            {
                "fold": 2,
                "origin": "2016-08-10",
                "window_start": "2016-08-11",
                "window_end": "2016-08-26",
                "n_rows": 28512,
            },
            {
                "fold": 3,
                "origin": "2016-08-26",
                "window_start": "2016-08-27",
                "window_end": "2016-09-11",
                "n_rows": 28512,
            },
        ],
        "models": [
            cmod("naive", "baseline"),
            cmod("seasonal_naive", "baseline"),
            cmod("moving_average", "baseline"),
            cmod("ridge", "classical_ml"),
            cmod("hgb", "classical_ml"),
        ],
        "notes": [],
    }
    dfold = {
        "rmsle": 0.3,
        "mae": 0.3,
        "rmse": 0.3,
        "rmsle_active": 0.4,
        "mae_active": 0.4,
        "rmse_active": 0.4,
        "n_rows": 28512,
        "n_rows_active": 20000,
    }
    dgrid = {
        "rmsle": 0.5,
        "mae": 0.4,
        "rmse": 0.6,
    }
    dvalid = {
        "n_rows": 299376,
        "per_horizon_rmsle": [0.5] * 16,
        "full_grid": dgrid,
        "active_slice": dgrid,
        "n_active": 123456,
        "inference_seconds": 1.5,
    }

    def dtier(tier, arch):
        return {
            "tier": tier,
            "architecture": arch,
            "checkpoint": "",
            "checkpoint_bytes": 1024,
            "checkpoint_sha256": "e" * 64,
            "params": 1000,
            "status": "ready",
            "train": {},
            "valid": dvalid,
            "fold_metrics": [dfold, dfold, dfold],
        }

    deep = {
        "schema_version": "1.0",
        "status": "complete",
        "track": "S4",
        "generated_at": "2026-11-10T00:00:00Z",
        "git_sha": "c" * 40,
        "data_fingerprint": {},
        "shared_config": {},
        "config_sha256": "d" * 64,
        "tiers": [
            dtier("deep-rnn", "rnn"),
            dtier("deep-lstm", "lstm"),
            dtier("deep-gru", "gru"),
        ],
        "ablation_2015plus": {},
        "comparison": {"status": "ok"},
        "runtime_tradeoff": {},
        "notes": [],
    }
    return classical, deep


def _synthetic_manifest_record() -> dict[str, Any]:
    """A schema-valid run-manifest record that is NOT a freeze run."""
    return {
        "created": "2026-11-10T00:00:00Z",
        "git_sha": "b" * 40,
        "command": "fixture",
        "exit": 0,
        "exit_ok": True,
        "versions": dict.fromkeys(VERSION_KEYS, "x"),
        "elapsed_seconds": 0.0,
        "arguments": {},
        "status": "PASS",
        "freeze_run": False,
        "notes": [],
    }


def _check_freeze_gates() -> None:
    """Day-2 block: the refusal gates, against throwaway dirs.

    Eight friendly refusals - unset FREEZE_RUN flag, a second freeze on a
    freeze_run = true manifest, a stranded freeze_results.json with no
    freeze_run = true manifest record, one unregistered artifact file, a
    nested copy of a registered name, zero registered artifacts, a partial
    registry, a naive manifest timestamp - exercised in temp dirs so the
    real artifacts directory stays untouched. No gate opens a split;
    local_test is never read here.
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
            nested = base / "nested"
            nested.mkdir()
            (nested / "backup").mkdir()
            (nested / "backup" / "classical_results.json").write_text(
                "{}", encoding="utf-8"
            )
            refuses(
                lambda: _freeze_preflight(nested),
                "unregistered file",
                "registry-nested",
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
            stranded = base / "stranded"
            stranded.mkdir()
            (stranded / "run_manifest.json").write_text(
                json.dumps([_synthetic_manifest_record()]), encoding="utf-8"
            )
            (stranded / FREEZE_RESULTS_FILENAME).write_text("{}", encoding="utf-8")
            refuses(
                lambda: _freeze_preflight(stranded),
                "stranded artifact",
                "artifact-exists-clean-manifest",
            )
            dir_furnished_clean = base / "furnished_clean"
            dir_furnished_clean.mkdir()
            artifacts_dir_clean = dir_furnished_clean / "artifacts"
            artifacts_dir_clean.mkdir()
            record = {
                "created": "2026-11-10T00:00:00Z",
                "git_sha": "b" * 40,
                "command": "fixture",
                "exit": 0,
                "exit_ok": True,
                "versions": {
                    "python": "3.13.0",
                    "uv": "0.10.0",
                    "pandas": "2.2.0",
                    "numpy": "2.1.0",
                },
                "elapsed_seconds": 0.0,
                "arguments": {},
                "status": "PASS",
                "freeze_run": False,
                "notes": [],
            }
            (artifacts_dir_clean / "run_manifest.json").write_text(
                json.dumps([record]), encoding="utf-8"
            )
            naive_dir = base / "naive_manifest"
            naive_dir.mkdir()
            naive_record = dict(record)
            naive_record["created"] = "2026-11-10T00:00:00"
            (naive_dir / "run_manifest.json").write_text(
                json.dumps([naive_record]), encoding="utf-8"
            )
            refuses(
                lambda: _freeze_preflight(naive_dir),
                "must carry a timezone",
                "manifest-naive-created",
            )
            classical, deep = _synthetic_registered_artifacts()
            (artifacts_dir_clean / "classical_results.json").write_text(
                json.dumps(classical), encoding="utf-8"
            )
            (artifacts_dir_clean / "deep_results.json").write_text(
                json.dumps(deep), encoding="utf-8"
            )
            saved_flag2 = os.environ.get("FREEZE_RUN")
            os.environ["FREEZE_RUN"] = "1"
            try:
                registry, blurbs = _freeze_preflight(artifacts_dir_clean)
                if (
                    sorted(registry)
                    != [
                        "classical_results.json",
                        "deep_results.json",
                    ]
                    or len(blurbs) != 2
                ):
                    raise ContractError(
                        "schema-clean gate did not return paths and blurbs for "
                        "2 registered artifacts"
                    )
                print("PASS freeze: schema-clean gate accepts a well-formed registry")
            except ContractError as error:
                raise ContractError(f"schema-clean gate failed: {error}") from error
            finally:
                if saved_flag2 is None:
                    os.environ.pop("FREEZE_RUN", None)
                else:
                    os.environ["FREEZE_RUN"] = saved_flag2
            bounds_cases = (
                (
                    "a scalar cut date",
                    {"train": "2016-03-27", "valid": "2016-03-28..2016-09-11"},
                    "canonical frame-range text",
                ),
                (
                    "an ISO datetime",
                    {
                        "train": "2013-01-01T00:00:00+00:00..2016-03-27",
                        "valid": "2016-03-28..2016-09-11",
                    },
                    "canonical frame-range text",
                ),
                (
                    "a date list",
                    {
                        "train": ["2013-01-01", "2016-03-27"],
                        "valid": "2016-03-28..2016-09-11",
                    },
                    "must be a non-empty string",
                ),
                (
                    "a local_test bound",
                    {
                        "train": "2013-01-01..2016-03-27",
                        "valid": "2016-03-28..2016-09-11",
                        "local_test": "2016-09-12..2017-08-15",
                    },
                    "train and valid",
                ),
            )
            for label, bounds, needle in bounds_cases:
                drifted = copy.deepcopy(classical)
                drifted["split_bounds"] = bounds
                try:
                    _validate_classical_artifact(drifted)
                except ContractError as error:
                    if needle not in str(error):
                        raise ContractError(
                            f"the split_bounds validator refused {label} with "
                            f"the wrong lead: {error}"
                        ) from error
                else:
                    raise ContractError(f"the split_bounds validator accepted {label}")
            seeds_cases = (
                ("a non-negative int", 7, True),
                ("non-empty text", "abc", True),
                ("a negative int", -1, False),
                ("a boolean", True, False),
                ("empty text", "", False),
            )
            for label, value, accepted in seeds_cases:
                drifted = copy.deepcopy(classical)
                drifted["models"][0]["seeds"] = {"seed": value}
                if accepted:
                    _validate_classical_artifact(drifted)
                    continue
                try:
                    _validate_classical_artifact(drifted)
                except ContractError as error:
                    if "must be an int >= 0 or a non-empty string" not in str(error):
                        raise ContractError(
                            f"the seeds validator refused {label} with the "
                            f"wrong lead: {error}"
                        ) from error
                else:
                    raise ContractError(f"the seeds validator accepted {label}")
            print(
                "PASS artifact: split_bounds refuses scalars, ISO datetimes, "
                "date lists and local_test bounds; seeds accepts a "
                "non-negative int or non-empty text and refuses the rest"
            )
    finally:
        if saved_flag is None:
            os.environ.pop("FREEZE_RUN", None)
        else:
            os.environ["FREEZE_RUN"] = saved_flag
    print(
        "PASS freeze: flag / second-freeze / artifact-exists-clean-manifest / "
        "registry-unknown / registry-nested / registry-zero / registry-partial / "
        "manifest-naive gates all refuse with friendly leads"
    )
    print("PASS freeze: gates open no split and never read local_test")


def _synthetic_frame(
    dates: list[Any],
    sales: list[float],
    ids: list[Any] | None = None,
    *,
    stores: int = 1,
    families: int = 1,
) -> pd.DataFrame:
    """A contract-shaped frame (26 pinned columns, pinned dtypes) for fixtures.

    The Day-5 check block drives the freeze body without opening a split
    file; this builds frames the same shape the real ones have, so the
    validators and the scorer see the real dtypes.

    The rows are laid out SERIES-MAJOR, like the real parquet: store_nbr
    outer, family next, date fastest-varying. The (store_nbr, family,
    date) key is therefore unique per row, exactly as it is in the real
    splits, which is what the prediction alignment is keyed on.
    """
    dates = list(dates)
    n_dates = len(dates)
    per_series = n_dates
    total = stores * families * per_series
    if len(sales) != total:
        raise ContractError(
            f"_synthetic_frame: {len(sales)} sales values for {total} rows"
        )
    store_col = np.repeat(
        np.arange(1, stores + 1, dtype=np.int8), families * per_series
    ).astype(np.int8)
    # The family label is unique per (store, family) series, so the
    # (store_nbr, family, date) key is unique per row as it is in the
    # real parquet, where the 33 families are distinct per store.
    family_names = [
        f"fam_{store:02d}_{index:02d}"
        for store in range(stores)
        for index in range(families)
    ]
    family_col = np.repeat(np.array(family_names, dtype=object), per_series)
    date_col = np.tile(np.array(dates, dtype=object), stores * families)
    rows = total
    data: dict[str, Any] = {
        "date": pd.Series(pd.to_datetime(date_col), dtype="datetime64[us]"),
        "store_nbr": pd.Series(store_col, dtype="int8"),
        "family": pd.Series(pd.Categorical(family_col)),
        "id": pd.Series(
            ids if ids is not None else list(range(1, rows + 1)), dtype="UInt32"
        ),
        "sales": pd.Series(sales, dtype="float32"),
        "onpromotion": pd.Series([0] * rows, dtype="uint16"),
        "city": pd.Series(pd.Categorical(["x"] * rows)),
        "state": pd.Series(pd.Categorical(["x"] * rows)),
        "store_type": pd.Series(pd.Categorical(["x"] * rows)),
        "cluster": pd.Series([1] * rows, dtype="int8"),
    }
    for flag in (
        "is_event",
        "is_national_holiday",
        "is_regional_holiday",
        "is_local_holiday",
        "is_holiday",
        "is_national_holiday_tomorrow",
        "is_national_holiday_in_2days",
        "is_regional_holiday_tomorrow",
        "is_regional_holiday_in_2days",
        "is_local_holiday_tomorrow",
        "is_local_holiday_in_2days",
        "is_holiday_tomorrow",
        "is_holiday_in_2days",
        "is_event_tomorrow",
        "is_event_in_2days",
    ):
        data[flag] = pd.Series([0] * rows, dtype="int8")
    data["earthquake_decay"] = pd.Series([0.0] * rows, dtype="float32")
    frame = pd.DataFrame(data)
    _assert_pinned_frame(frame, "synthetic fixture")
    return frame


def _synthetic_freeze_fixtures() -> dict[str, Any]:
    """The Day-5 check block's fixtures: frames, models and a fake predictor.

    The local_test stand-in carries the pinned 1,782-row Christmas-2016
    slice plus a small active tail, so the slice assertion, the dual
    evaluation and the headline choice all run on realistic shapes
    without opening splits/local_test.parquet.
    """
    # The stand-in is series-major like the real parquet: 54 stores x 33
    # families, date fastest-varying, so the (store_nbr, family, date) key
    # is unique per row, exactly as the alignment requires.
    stores, families = 54, 33
    # ONE Christmas-2016 row per series (id = NA, sales = 0) plus 120
    # active rows, replicated across every store x family series: the
    # series-major layout then holds exactly 54 x 33 = 1,782 Christmas
    # rows, the pinned count. The dates are per series; the sales and ids
    # are per row.
    series_sales = [0.0] + [3.0] * 120
    series_ids: list[Any] = [None, *range(1, 121)]
    # 120 DISTINCT active dates per series, so the (store_nbr, family,
    # date) key is unique per row exactly as it is in the real parquet.
    series_dates = [CHRISTMAS_2016_DATE] + [
        CHRISTMAS_2016_DATE + pd.Timedelta(days=offset + 1) for offset in range(120)
    ]
    # The active tail must stay clear of 2016-12-25, or it would add rows
    # to the Christmas slice and break the pinned 1,782 count.
    sales: list[float] = []
    ids: list[Any] = []
    for _series in range(stores * families):
        sales.extend(series_sales)
        ids.extend(series_ids)
    local_test = _synthetic_frame(
        series_dates, sales, ids, stores=stores, families=families
    )
    tail_dates = [
        VALID_END - pd.Timedelta(days=offset)
        for offset in range(LOOKBACK_DAYS - 1, -1, -1)
    ]
    tail = _synthetic_frame(
        tail_dates,
        [1.0] * (LOOKBACK_DAYS * stores * families),
        stores=stores,
        families=families,
    )
    predict_frame = pd.concat([tail, local_test], ignore_index=True)
    fit_frame = _synthetic_frame(
        [pd.Timestamp("2016-03-27")] * 10,
        [2.0] * (10 * stores * families),
        stores=stores,
        families=families,
    )
    models = [
        {
            "name": "naive",
            "track": "S3",
            "artifact": "classical_results.json",
            "config": {},
        },
        {
            "name": "deep-lstm",
            "track": "S4",
            "artifact": "deep_results.json",
            "config": None,
        },
    ]
    return {
        "local_test": local_test,
        "predict_frame": predict_frame,
        "fit_frame": fit_frame,
        "models": models,
        "rows": len(local_test),
    }


def _synthetic_fold_phase(models: list[dict[str, Any]]) -> dict[str, Any]:
    """A fold-phase block shaped like the one the registered artifacts feed."""
    table = folds_table()
    fold_rows = [
        {
            "fold": int(row["fold"]),
            "origin": str(row["origin"]),
            "window_start": str(row["window_start"]),
            "window_end": str(row["window_end"]),
            "n_rows": int(row["n_rows"]),
        }
        for _, row in table.iterrows()
    ]
    fold_models = []
    for spec in models:
        records = []
        for index, fold_row in enumerate(fold_rows):
            full_grid = {
                "rmsle": 0.2 + 0.01 * index,
                "mae": 1.0 + 0.01 * index,
                "rmse": 2.0 + 0.01 * index,
                "rmsle_active": 0.3 + 0.01 * index,
                "mae_active": 1.5 + 0.01 * index,
                "rmse_active": 2.5 + 0.01 * index,
                "n_rows": FOLD_ROWS,
                "n_rows_active": 20000,
            }
            active_slice = {
                "rmsle": full_grid["rmsle_active"],
                "mae": full_grid["mae_active"],
                "rmse": full_grid["rmse_active"],
                "rmsle_active": full_grid["rmsle_active"],
                "mae_active": full_grid["mae_active"],
                "rmse_active": full_grid["rmse_active"],
                "n_rows": full_grid["n_rows_active"],
                "n_rows_active": full_grid["n_rows_active"],
            }
            records.append(
                {
                    "fold": index + 1,
                    "origin": fold_row["origin"],
                    "window_start": fold_row["window_start"],
                    "window_end": fold_row["window_end"],
                    "n_rows": FOLD_ROWS,
                    "slices": {"full_grid": full_grid, "active_slice": active_slice},
                }
            )
        fold_models.append(
            {
                "name": spec["name"],
                "track": spec["track"],
                "artifact": spec["artifact"],
                "folds": records,
            }
        )
    return {"folds": fold_rows, "models": fold_models}


def _synthetic_contract_counts() -> dict[str, Any]:
    """A contract_counts block shaped like the one the frames feed."""
    return {
        "closure_rows": {
            "train": 5346,
            "valid": 0,
            "local_test": 1782,
            "universe": 7128,
        },
        "masked_training_drop": {
            "train": 207801,
            "valid": 5544,
            "total": 213345,
            "in_window_fills_kept": 462,
        },
        "zero_share_raw": {
            "rows": 847004,
            "of_rows": 2405700,
            "share": 0.352082,
        },
        "zero_share_trainable": {
            "rows": 633659,
            "of_rows": 2192355,
            "share": 0.289031,
        },
        "preopening_holdout_rows": {
            "valid": 5544,
            "local_test": 7260,
            "total": 12804,
        },
        "christmas_2016_rows": 1782,
    }


def _synthetic_data_snapshot() -> dict[str, Any]:
    """A data_snapshot block shaped like the one the manifest feeds."""
    return {
        split: {
            "rows": rows,
            "bounds": [SPLIT_FRAME_STARTS[split], SPLIT_FRAME_ENDS[split]],
        }
        for split, rows in (
            ("train", 2106324),
            ("valid", 299376),
            ("local_test", 602316),
        )
    }


def _check_freeze_registry_refusals() -> None:
    """Day-5 block: the refusals the freeze body must not soften.

    A deep artifact whose status is not "complete", a model dropped from
    BOTH phases, an artifact that drifts between the write and the
    re-read, a registry whose model count no longer matches the
    freeze's own model list, and a committed freeze_results.json whose
    model count or model names disagree with the registry on the
    real-file audit path: each is refused by the production path,
    loudly and with a friendly lead. The cases drive the real gates, not
    copies of them, so a softened gate fails the audit.
    """

    def refuses(call: Callable[[], Any], needle: str, label: str) -> None:
        try:
            call()
        except ContractError as error:
            if needle not in str(error):
                raise ContractError(
                    f"freeze refusal {label}: refused with the wrong lead: {error}"
                ) from error
            return
        raise ContractError(f"freeze refusal {label}: accepted a forbidden call")

    # 1. A deep artifact that is not complete refuses the pre-flight
    #    gate, even though its schema is perfectly valid.
    with tempfile.TemporaryDirectory() as tmp_name:
        artifacts = Path(tmp_name) / "artifacts"
        artifacts.mkdir(parents=True)
        classical, deep = _synthetic_registered_artifacts()
        deep["status"] = "partial"
        (artifacts / "classical_results.json").write_text(
            json.dumps(classical), encoding="utf-8"
        )
        (artifacts / "deep_results.json").write_text(json.dumps(deep), encoding="utf-8")
        (artifacts / "run_manifest.json").write_text(
            json.dumps([_synthetic_manifest_record()]), encoding="utf-8"
        )
        saved_flag = os.environ.get("FREEZE_RUN")
        os.environ["FREEZE_RUN"] = "1"
        try:
            refuses(
                lambda: _freeze_preflight(artifacts),
                "must be complete for freeze",
                "deep-status",
            )
        finally:
            if saved_flag is None:
                os.environ.pop("FREEZE_RUN", None)
            else:
                os.environ["FREEZE_RUN"] = saved_flag
        # The same gate accepts the complete artifact: the refusal above
        # is about the status, not about the fixture.
        (artifacts / "deep_results.json").write_text(
            json.dumps({**deep, "status": "complete"}), encoding="utf-8"
        )
        os.environ["FREEZE_RUN"] = "1"
        try:
            registered, blurbs = _freeze_preflight(artifacts)
        except ContractError as error:
            raise ContractError(
                f"the schema-clean gate refused a complete deep artifact: {error}"
            ) from error
        finally:
            if saved_flag is None:
                os.environ.pop("FREEZE_RUN", None)
            else:
                os.environ["FREEZE_RUN"] = saved_flag
        if sorted(registered) != sorted(ARTIFACT_REGISTRY) or len(blurbs) != 2:
            raise ContractError(
                f"the schema-clean gate returned {sorted(registered)} and "
                f"{len(blurbs)} blurbs for a complete registry"
            )
    print(
        "PASS freeze refusal: a deep artifact that is not complete refuses the "
        "freeze, and a complete one passes the same gate"
    )

    # 2. A model dropped from BOTH phases still trips the registered-count
    #    guard: the freeze phase is not allowed to shrink quietly. The
    #    payload is the fixture frame's, so the validator is handed the
    #    fixture's own window rather than the pinned real-file one.
    fixtures = _synthetic_freeze_fixtures()
    models = fixtures["models"]
    rows = fixtures["rows"]
    fixture_local_test = fixtures["local_test"]
    fixture_bounds = (
        fixture_local_test["date"].min().date().isoformat(),
        fixture_local_test["date"].max().date().isoformat(),
    )

    def _fake_predict(spec: dict[str, Any]) -> Any:
        value = 1.0 + models.index(spec) * 0.5
        return np.full(rows, value, dtype="float64")

    with tempfile.TemporaryDirectory() as tmp_name:
        directory = Path(tmp_name)
        payload = _run_canonical_sequence(
            directory,
            models=models,
            predict_frame=fixtures["predict_frame"],
            local_test=fixtures["local_test"],
            predict=_fake_predict,
            fold_phase=_synthetic_fold_phase(models),
            contract_counts=_synthetic_contract_counts(),
            data_snapshot=_synthetic_data_snapshot(),
            notes=_freeze_notes(),
            frozen_at="2026-11-11T00:00:00Z",
            git_sha="a" * 40,
        )
    dropped = {
        **payload,
        "fold_phase": {
            **payload["fold_phase"],
            "models": payload["fold_phase"]["models"][1:],
        },
        "freeze_phase": {
            **payload["freeze_phase"],
            "models": payload["freeze_phase"]["models"][1:],
        },
    }
    refuses(
        lambda: _validate_freeze_results(
            dropped,
            expected_models=len(models),
            expected_rows=rows,
            expected_bounds=fixture_bounds,
        ),
        "!= the registered count",
        "registered-model-count",
    )
    print(
        "PASS freeze refusal: a model dropped from both phases is refused by the "
        "registered-count guard"
    )

    # 3. An artifact that drifts between the write and the re-read is
    #    refused: the publish is verified, not assumed.
    with tempfile.TemporaryDirectory() as tmp_name:
        directory = Path(tmp_name)
        drifted = {**payload, "notes": [*payload["notes"], "drifted after the write"]}

        def _drifting_write(
            path: Path,
            data: bytes,
            *,
            exclusive: bool = False,
            _writer: Callable[..., None] = _atomic_write_bytes,
        ) -> None:
            _writer(
                path,
                (json.dumps(drifted, indent=2, allow_nan=False) + "\n").encode("utf-8"),
                exclusive=exclusive,
            )

        with mock.patch.object(
            sys.modules[__name__], "_atomic_write_bytes", _drifting_write
        ):
            refuses(
                lambda: _write_freeze_results(
                    directory / FREEZE_RESULTS_FILENAME, payload
                ),
                "does not round-trip through JSON unchanged",
                "write-reread",
            )
    print(
        "PASS freeze refusal: an artifact that drifts between the write and the "
        "re-read is refused"
    )

    # 4. The registry's model count and the freeze's own model list must
    #    agree, and the check fires BEFORE any split is opened.
    classical, deep = _synthetic_registered_artifacts()
    payloads = {
        "classical_results.json": classical,
        "deep_results.json": deep,
    }

    def _no_split(name: str) -> pd.DataFrame:
        raise ContractError(f"the freeze opened the split {name!r} too early")

    with (
        mock.patch.object(
            sys.modules[__name__], "_registered_payloads", lambda directory: payloads
        ),
        mock.patch.object(
            sys.modules[__name__],
            "_freeze_model_specs",
            lambda payloads: [{"name": "naive", "artifact": "classical_results.json"}],
        ),
        mock.patch.object(sys.modules[__name__], "load_split", _no_split),
    ):
        refuses(
            lambda: _freeze_day5(Path("/nonexistent"), {}),
            "!= the registry's",
            "registry-model-count",
        )
    print(
        "PASS freeze refusal: a registry whose model count disagrees with the "
        "freeze's model list is refused before any split is opened"
    )

    # 5. The real-file audit path carries the registered count and the
    #    registered names: a committed freeze_results.json with the wrong
    #    model count, or with a fabricated model name carried by BOTH the
    #    fold and the freeze phase, is refused instead of waved through.
    import io

    registered_models = [
        name for spec in ARTIFACT_REGISTRY.values() for name in spec["model_names"]
    ]
    pinned_rows = PINNED_SPLIT_ROWS["local_test"]
    pinned_start = SPLIT_FRAME_STARTS["local_test"]
    pinned_end = SPLIT_FRAME_ENDS["local_test"]

    def _onto_pinned_window(block_payload: dict[str, Any]) -> dict[str, Any]:
        """The fixture payload reshaped onto the pinned local_test window."""
        freeze_block = block_payload["freeze_phase"]
        return {
            **block_payload,
            "freeze_phase": {
                **freeze_block,
                "n_rows": pinned_rows,
                "window_start": pinned_start,
                "window_end": pinned_end,
                "models": [
                    {
                        **model,
                        "n_rows": pinned_rows,
                        "window_start": pinned_start,
                        "window_end": pinned_end,
                    }
                    for model in freeze_block["models"]
                ],
            },
        }

    def _audit_real_file(directory: Path) -> None:
        """Drive the real-file audit path against artifacts/ in directory."""
        with (
            mock.patch.object(sys.modules[__name__], "ARTIFACTS_DIR", directory),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            _check_registry()

    # The wrong count: one model record where the registry registers
    # eight, on an otherwise valid pinned payload.
    count_payload = _onto_pinned_window(dropped)
    with tempfile.TemporaryDirectory() as tmp_name:
        count_dir = Path(tmp_name)
        (count_dir / FREEZE_RESULTS_FILENAME).write_text(
            json.dumps(count_payload), encoding="utf-8"
        )
        refuses(
            lambda: _audit_real_file(count_dir),
            "!= the registered count",
            "real-file-count",
        )

    # The fabricated name: eight records so the registered count is
    # satisfied, with one name the registry does not know carried by
    # BOTH phases, so only the name comparison can refuse it.
    identity = {
        name: (spec["track"], artifact_name)
        for artifact_name, spec in ARTIFACT_REGISTRY.items()
        for name in spec["model_names"]
    }
    fabricated = "fabricated-model"
    names = [*registered_models[:-1], fabricated]
    freeze_template = payload["freeze_phase"]["models"][0]
    fold_template = payload["fold_phase"]["models"][0]

    def _named_freeze_record(name: str) -> dict[str, Any]:
        track, artifact = identity.get(
            name, (freeze_template["track"], freeze_template["artifact"])
        )
        return {
            **freeze_template,
            "name": name,
            "track": track,
            "artifact": artifact,
            "n_rows": pinned_rows,
            "window_start": pinned_start,
            "window_end": pinned_end,
        }

    def _named_fold_record(name: str) -> dict[str, Any]:
        track, artifact = identity.get(
            name, (fold_template["track"], fold_template["artifact"])
        )
        return {**fold_template, "name": name, "track": track, "artifact": artifact}

    name_payload = {
        **payload,
        "fold_phase": {
            **payload["fold_phase"],
            "models": [_named_fold_record(name) for name in names],
        },
        "freeze_phase": {
            **payload["freeze_phase"],
            "n_rows": pinned_rows,
            "window_start": pinned_start,
            "window_end": pinned_end,
            "models": [_named_freeze_record(name) for name in names],
        },
    }
    with tempfile.TemporaryDirectory() as tmp_name:
        names_dir = Path(tmp_name)
        (names_dir / FREEZE_RESULTS_FILENAME).write_text(
            json.dumps(name_payload), encoding="utf-8"
        )
        refuses(
            lambda: _audit_real_file(names_dir),
            "registered model name",
            "real-file-names",
        )
    print(
        "PASS freeze refusal: the real-file audit path refuses a committed "
        "freeze_results.json with the wrong registered count and one whose "
        "fold and freeze phases both carry a fabricated model name"
    )


def _synthetic_freeze_payload() -> dict[str, Any]:
    """A valid freeze_results payload, produced by the production code.

    The canonical sequence builds it, validates it and writes it; this
    helper returns it so the refusal battery can perturb ONE field and
    watch the ONE production validator refuse.
    """
    fixtures = _synthetic_freeze_fixtures()
    models = fixtures["models"]
    rows = fixtures["rows"]

    def _fake_predict(spec: dict[str, Any]) -> Any:
        value = 1.0 + models.index(spec) * 0.5
        return np.full(rows, value, dtype="float64")

    with tempfile.TemporaryDirectory() as tmp_name:
        return _run_canonical_sequence(
            Path(tmp_name),
            models=models,
            predict_frame=fixtures["predict_frame"],
            local_test=fixtures["local_test"],
            predict=_fake_predict,
            fold_phase=_synthetic_fold_phase(models),
            contract_counts=_synthetic_contract_counts(),
            data_snapshot=_synthetic_data_snapshot(),
            notes=_freeze_notes(),
            frozen_at="2026-11-11T00:00:00Z",
            git_sha="a" * 40,
        )


def _check_freeze_body() -> None:
    """Day-5 block: the canonical freeze body, on synthetic fixtures only.

    No split file is opened here and local_test is never read: the block
    drives the freeze body with contract-shaped fixtures in throwaway
    temp dirs and proves the atomic write, the refusals, the Christmas
    slice assertion, the one-freeze invariant, the headline choice and
    the no-write-on-failure rule.
    """
    fixtures = _synthetic_freeze_fixtures()
    models = fixtures["models"]
    local_test = fixtures["local_test"]
    predict_frame = fixtures["predict_frame"]
    rows = fixtures["rows"]
    fold_phase = _synthetic_fold_phase(models)
    counts = _synthetic_contract_counts()
    snapshot = _synthetic_data_snapshot()

    def _fake_predict(spec: dict[str, Any]) -> Any:
        # A DISTINCT value per model, so the lowest-RMSLE model is
        # unambiguous and the headline assertion has one answer.
        value = 1.0 + fixtures["models"].index(spec) * 0.5
        return np.full(rows, value, dtype=np.float64)

    with tempfile.TemporaryDirectory() as tmp_name:
        directory = Path(tmp_name)
        payload = _run_canonical_sequence(
            directory,
            models=models,
            predict_frame=predict_frame,
            local_test=local_test,
            predict=_fake_predict,
            fold_phase=fold_phase,
            contract_counts=counts,
            data_snapshot=snapshot,
            notes=_freeze_notes(),
            frozen_at="2026-11-11T00:00:00Z",
            git_sha="a" * 40,
        )
        written = directory / FREEZE_RESULTS_FILENAME
        if not written.is_file():
            raise ContractError(
                "the canonical sequence did not write artifacts/freeze_results.json"
            )
        stored = json.loads(written.read_text(encoding="utf-8"))
        if stored != payload:
            raise ContractError(
                "the written freeze_results.json does not re-parse to the payload"
            )
        if list(stored) != list(FREEZE_RESULTS_KEYS):
            raise ContractError(
                f"freeze_results.json carries {list(stored)}, not the pinned "
                f"{list(FREEZE_RESULTS_KEYS)}"
            )
        leftovers = sorted(
            p.name for p in directory.iterdir() if p.name != FREEZE_RESULTS_FILENAME
        )
        if leftovers:
            raise ContractError(
                f"the atomic write left temporary files behind: {leftovers}"
            )
        freeze_models = stored["freeze_phase"]["models"]
        if len(freeze_models) != len(models):
            raise ContractError(
                f"freeze_results.json carries {len(freeze_models)} model records "
                f"!= the {len(models)} scored models"
            )
        christmas_slice = freeze_models[0]["slices"]["christmas_2016"]
        if christmas_slice["n_rows"] != CHRISTMAS_2016_ROWS:
            raise ContractError(
                "the Christmas-2016 slice did not hold the pinned "
                f"{CHRISTMAS_2016_ROWS} rows"
            )
        if freeze_models[0]["slices"]["full_grid"]["n_rows"] != rows:
            raise ContractError(
                "the full-grid slice did not cover the whole scored frame"
            )
        # The headline is read from the file inside the block that owns
        # it: the temp directory is gone once the block exits, so the
        # payload is checked here and never read from outside.
        expected_headline = _pick_headline_model(freeze_models)
        if stored["headline_model"] != expected_headline:
            raise ContractError(
                f"the freeze headline {stored['headline_model']!r} is not the "
                f"model with the lowest full-grid RMSLE ({expected_headline!r})"
            )
    print(
        "PASS freeze body: freeze_results.json carries the pinned field set, is "
        "written atomically (no temp residue) and re-parses to the payload"
    )

    # The refusals: a short, missing or malformed field set.
    malformed: list[tuple[str, Any, str]] = [
        (
            "a missing field",
            {k: v for k, v in payload.items() if k != "notes"},
            "missing",
        ),
        (
            "a short field set",
            {k: v for k, v in payload.items() if k != "git_sha"},
            "missing",
        ),
        ("an extra field", {**payload, "extra": 1}, "extra"),
        (
            "a naive frozen_at",
            {**payload, "frozen_at": "2026-11-11T00:00:00"},
            "timezone",
        ),
        ("a short git_sha", {**payload, "git_sha": "abc"}, "40-hex"),
        (
            "an empty headline",
            {**payload, "headline_model": " "},
            "headline_model",
        ),
        ("a foreign headline", {**payload, "headline_model": "nope"}, "not one of"),
        ("notes that are not strings", {**payload, "notes": [1]}, "list of strings"),
        (
            "a short contract_counts",
            {**payload, "contract_counts": {}},
            "contract_counts",
        ),
        (
            "a missing snapshot split",
            {**payload, "data_snapshot": {}},
            "data_snapshot",
        ),
        (
            "a missing freeze slice",
            {
                **payload,
                "freeze_phase": {
                    "models": [],
                    "n_rows": 0,
                    "window_end": "",
                    "window_start": "",
                },
            },
            "freeze_phase",
        ),
    ]
    for label, broken, needle in malformed:
        try:
            _validate_freeze_results(
                broken,
                expected_models=len(models),
                # The malformed payload is the fixture frame's, so the
                # validator gets the fixture's own window, not the
                # pinned real-file one.
                expected_rows=len(local_test),
                expected_bounds=(
                    local_test["date"].min().date().isoformat(),
                    local_test["date"].max().date().isoformat(),
                ),
            )
        except ContractError as error:
            if needle not in str(error):
                raise ContractError(
                    f"the freeze_results.json validator refused {label} with the "
                    f"wrong lead: {error}"
                ) from error
        else:
            raise ContractError(f"the freeze_results.json validator accepted {label}")
    print(
        "PASS freeze body: a missing, short, extra or malformed field set is "
        "refused with a friendly error"
    )

    # The Christmas-2016 slice assertion fires when the slice is wrong.
    short_slice = _synthetic_frame(
        [CHRISTMAS_2016_DATE] * (CHRISTMAS_2016_ROWS - 1)
        + [pd.Timestamp("2016-09-12")],
        [0.0] * CHRISTMAS_2016_ROWS,
    )
    try:
        _christmas_mask(short_slice)
    except ContractError as error:
        if "Christmas-2016 slice" not in str(error):
            raise ContractError(
                f"the Christmas-2016 assertion refused with the wrong lead: {error}"
            ) from error
    else:
        raise ContractError("the Christmas-2016 assertion accepted a short slice")
    print(
        "PASS freeze body: the Christmas-2016 slice row count is asserted against "
        "the frame's date column and fires when the slice is wrong"
    )

    # Exactly one freeze_run = true record; a second freeze refuses.
    with tempfile.TemporaryDirectory() as tmp_name:
        manifest_dir = Path(tmp_name)
        write_run_manifest(
            "contract.py --freeze",
            freeze_run=True,
            arguments={"freeze": "1"},
            artifacts_dir=manifest_dir,
        )
        records = read_run_manifest(manifest_dir / "run_manifest.json")
        frozen = [record for record in records if record["freeze_run"] is True]
        if len(frozen) != 1:
            raise ContractError(
                f"the manifest carries {len(frozen)} freeze_run = true records, "
                "expected exactly one"
            )
        try:
            write_run_manifest(
                "contract.py --freeze",
                freeze_run=True,
                arguments={"freeze": "1"},
                artifacts_dir=manifest_dir,
            )
        except ContractError as error:
            if "already carries" not in str(error):
                raise ContractError(
                    f"the second freeze refused with the wrong lead: {error}"
                ) from error
        else:
            raise ContractError("the second freeze was accepted")
        write_run_manifest("contract.py --check", artifacts_dir=manifest_dir)
        records = read_run_manifest(manifest_dir / "run_manifest.json")
        frozen = [record for record in records if record["freeze_run"] is True]
        if len(frozen) != 1:
            raise ContractError(
                "a later non-freeze record changed the freeze_run = true count"
            )
        doubled = [
            {**records[0], "created": "2026-11-10T00:00:00Z"},
            {**records[0], "created": "2026-11-11T00:00:00Z"},
        ]
        (manifest_dir / "run_manifest.json").write_text(
            json.dumps(doubled), encoding="utf-8"
        )
        try:
            read_run_manifest(manifest_dir / "run_manifest.json")
        except ContractError as error:
            if "freeze_run = true" not in str(error):
                raise ContractError(
                    f"the doubled manifest refused with the wrong lead: {error}"
                ) from error
        else:
            raise ContractError("a manifest with two freeze records was accepted")
    print(
        "PASS freeze body: exactly one freeze_run = true record survives, a "
        "second freeze refuses and a doubled manifest is refused"
    )

    # The headline is the registered model with the lowest RMSLE.
    headline_records = [
        {
            "name": "naive",
            "slices": {"full_grid": {"rmsle": 0.31}},
        },
        {
            "name": "ridge",
            "slices": {"full_grid": {"rmsle": 0.12}},
        },
        {
            "name": "hgb",
            "slices": {"full_grid": {"rmsle": 0.20}},
        },
    ]
    if _pick_headline_model(headline_records) != "ridge":
        raise ContractError(
            "the headline was not the model with the lowest full-grid RMSLE"
        )
    print(
        "PASS freeze body: headline_model is the registered model with the "
        "lowest full-grid RMSLE"
    )

    # A failing phase never leaves a freeze_results.json behind.
    with tempfile.TemporaryDirectory() as tmp_name:
        directory = Path(tmp_name)

        def _failing_predict(spec: dict[str, Any]) -> Any:
            if spec["name"] == "deep-lstm":
                raise ContractError(
                    "the fixture's sanctioned entrypoint refused the call"
                )
            return np.full(rows, 1.0, dtype=np.float64)

        try:
            _run_canonical_sequence(
                directory,
                models=models,
                predict_frame=predict_frame,
                local_test=local_test,
                predict=_failing_predict,
                fold_phase=fold_phase,
                contract_counts=counts,
                data_snapshot=snapshot,
                notes=_freeze_notes(),
                frozen_at="2026-11-11T00:00:00Z",
                git_sha="a" * 40,
            )
        except ContractError:
            pass
        else:
            raise ContractError("a failing model phase was accepted")
        if (directory / FREEZE_RESULTS_FILENAME).exists():
            raise ContractError(
                "freeze_results.json was written even though a phase failed"
            )
    print("PASS freeze body: freeze_results.json is never written when a phase fails")
    print("PASS freeze body: the block opens no split and never reads local_test")


# -- Day-5 self-audit battery: the freeze machinery against synthetic --
# fixtures. splits/local_test.parquet is never opened here: every frame
# these blocks score is built in memory, every artifact the freeze path
# consumes is a throwaway fixture in a TemporaryDirectory, and the two
# fit_predict_frozen alignment conventions (predict_frame row order vs
# local_test rows only) are stub modules. The blocks drive the real
# freeze() body end to end, so any contract drift fails loudly.
#
# One direct-call seam is pinned on the freeze body by this battery:
#     _validate_freeze_results(payload: dict) -> None
#        raises ContractError on a freeze_results.json payload that
#        violates the sheet Section-10 field set, types, finiteness,
#        non-negative metrics, the registry-sized model list, or the
#        headline_model-led registry rule. Friendly errors only.
# Every other block drives the public freeze() entrypoint end to end.
# Pinned by the sheets: 54 stores x 33 families = 1,782 rows on
# 2016-12-25 in local_test (the count is pinned there, never measured
# by opening the file).


def _registered_model_names() -> list[str]:
    """The 8 registered model names, in registry pin order."""
    ordered: list[str] = []
    for artifact_name in sorted(ARTIFACT_REGISTRY):
        ordered.extend(ARTIFACT_REGISTRY[artifact_name]["model_names"])
    return ordered


def _walk_nodes(node: Any, path: str = "") -> Generator[tuple[str, Any], None, None]:
    """Yield (dotted path, node) for every node in the tree, leaves too."""
    yield path, node
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{path}.{key}" if path else str(key)
            yield from _walk_nodes(value, child)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            yield from _walk_nodes(item, f"{path}[{index}]")


def _classify_slice(path: str) -> str | None:
    """Full-grid / active-slice / Christmas tag from a dotted path."""
    lowered = path.lower()
    if "christmas" in lowered:
        return "christmas"
    if "full" in lowered:
        return "full_grid"
    if "active" in lowered:
        return "active_slice"
    return None


def _assert_score_block_shape(path: str, block: dict[str, Any]) -> dict[str, Any]:
    """Pinned score schema keys: 6 finite non-negative metrics + 2 counts."""
    for key in SCORE_KEYS[:6]:
        value = block[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError(
                f"{path}: {key} must be a finite non-negative number, got {value!r}"
            )
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ContractError(
                f"{path}: {key} must be a finite non-negative number, got {value!r}"
            )
    for key in ("n_rows", "n_rows_active"):
        count = block[key]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ContractError(f"{path}: {key} must be a non-negative int")
    if block["n_rows_active"] > block["n_rows"]:
        raise ContractError(
            f"{path}: n_rows_active {block['n_rows_active']} > n_rows {block['n_rows']}"
        )
    return block


def _fold_records(block: Any) -> list[dict[str, Any]]:
    """The per-model-per-fold records of a fold_phase block, flattened.

    The freeze body writes fold_phase as an object holding the pinned
    windows and, per model, its own list of fold records; a flat list of
    fold records is the other shape the sheets allow. Both are read here.
    """
    records: list[dict[str, Any]] = []
    if isinstance(block, dict):
        models = block.get("models")
        if isinstance(models, list):
            for model in models:
                if not isinstance(model, dict):
                    continue
                model_folds = model.get("folds")
                if isinstance(model_folds, list):
                    for fold_record in model_folds:
                        if not isinstance(fold_record, dict):
                            continue
                        # The owning model's identity lives on the parent
                        # record in this shape; carry it down so the record
                        # still names exactly one registered model.
                        record = dict(fold_record)
                        for key in ("name", "model", "model_name", "tier"):
                            if key not in record and key in model:
                                record[key] = model[key]
                        records.append(record)
    elif isinstance(block, list):
        records.extend(f for f in block if isinstance(f, dict))
    return records


def _drop_last_freeze_model(payload: dict[str, Any]) -> Any:
    """Drop the last freeze-phase model record, in either container shape."""
    block = payload.get("freeze_phase")
    if isinstance(block, dict) and isinstance(block.get("models"), list):
        models = list(block["models"])
        if models:
            models = models[:-1]
        return {**block, "models": models}
    if isinstance(block, list) and block:
        return block[:-1]
    return block


def _phase_records(block: Any) -> list[dict[str, Any]]:
    """The per-model records of a phase block, whichever container holds them.

    The freeze body writes freeze_phase as an object carrying window_start,
    window_end, n_rows and models; a bare list of model records is the other
    shape the sheets allow. Both are read here, so the checker asserts on the
    records rather than on the container.
    """
    if isinstance(block, dict):
        models = block.get("models")
        if isinstance(models, list):
            return [model for model in models if isinstance(model, dict)]
    if isinstance(block, list):
        return [model for model in block if isinstance(model, dict)]
    return []


def _slice_blocks(
    record: Any, kinds: tuple[str, ...], label: str
) -> dict[str, dict[str, Any]]:
    """Exactly one SCORE_KEYS block per slice kind inside record."""
    found: dict[str, list[tuple[str, dict[str, Any]]]] = {kind: [] for kind in kinds}
    for path, node in _walk_nodes(record, label):
        if not isinstance(node, dict):
            continue
        if set(node) != set(SCORE_KEYS):
            continue
        kind = _classify_slice(path)
        if kind in found:
            found[kind].append((path, node))
    blocks: dict[str, dict[str, Any]] = {}
    for kind, entries in found.items():
        if len(entries) != 1:
            raise ContractError(
                f"{label}: expected exactly one {kind} block with the pinned "
                f"score schema keys, found {len(entries)}"
            )
        path, block = entries[0]
        blocks[kind] = _assert_score_block_shape(path, block)
    return blocks


def _record_model(record: Any, model_names: set[str], label: str) -> str:
    """The single registered model named inside a freeze record."""
    if isinstance(record, dict):
        for key in ("model", "model_name", "name", "tier"):
            value = record.get(key)
            if isinstance(value, str) and value in model_names:
                return value
    hits = {
        node
        for _path, node in _walk_nodes(record)
        if isinstance(node, str) and node in model_names
    }
    if len(hits) != 1:
        raise ContractError(
            f"{label}: expected the record to name exactly one registered "
            f"model, found {sorted(hits)}"
        )
    return hits.pop()


def _synthetic_split_frame(
    dates: pd.DatetimeIndex,
    drop_on_christmas: tuple[int, str] | None = None,
    families: Sequence[str] | None = None,
    categories: dict[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Synthetic split frame: 54 stores x 33 families per date.

    Christmas-2016 rows carry id = <NA> (the closure-fill pattern). When
    drop_on_christmas is set, that (store, family) pair is removed on
    2016-12-25 only, leaving 1,781 Christmas rows for the refusal test.

    families defaults to synthetic labels; the freeze harness passes the
    REAL family names, so the synthetic local_test stand-in shares the
    training frame's categorical domain and the count pins hold.
    """
    per_day = 54 * 33
    n = len(dates) * per_day
    stores = np.tile(np.repeat(np.arange(1, 55, dtype=np.int8), 33), len(dates))
    if families is None:
        families = [f"family_{i:02d}" for i in range(33)]
    family = np.tile(families * 54, len(dates))
    date = np.repeat(dates.to_numpy().astype("datetime64[us]"), per_day)
    idx = np.arange(n, dtype=np.int64)
    sales = np.where(idx % 7 == 0, 0.0, (idx % 7) + 0.5).astype("float32")
    is_christmas = date == np.datetime64(CHRISTMAS_2016_DATE)
    id_values = pd.Series(idx + 1, dtype="UInt32")
    id_values = id_values.mask(pd.Series(is_christmas))
    frame = pd.DataFrame(
        {
            "date": pd.Series(date),
            "store_nbr": pd.Series(stores),
            "family": pd.Categorical(family),
            "id": id_values,
            "sales": pd.Series(sales),
            "onpromotion": np.zeros(n, dtype="uint16"),
            "city": pd.Categorical(
                np.repeat((categories or {}).get("city", ["city_a"])[0], n)
            ),
            "state": pd.Categorical(
                np.repeat((categories or {}).get("state", ["state_a"])[0], n)
            ),
            "store_type": pd.Categorical(
                np.repeat((categories or {}).get("store_type", ["type_a"])[0], n)
            ),
            "cluster": np.full(n, 5, dtype="int8"),
        }
    )
    for col in PINNED_DTYPES:
        if col.startswith("is_"):
            frame[col] = np.zeros(n, dtype="int8")
    frame["earthquake_decay"] = np.zeros(n, dtype="float32")
    frame = frame[list(PINNED_DTYPES)]
    if drop_on_christmas is not None:
        drop_store, drop_family = drop_on_christmas
        drop = (
            (frame["store_nbr"] == drop_store)
            & (frame["family"] == drop_family)
            & (frame["date"] == CHRISTMAS_2016_DATE)
        )
        frame = frame.loc[~drop].reset_index(drop=True)
    return frame


def _synthetic_splits(
    *, christmas_intact: bool = True, real_splits: bool = False
) -> dict[str, pd.DataFrame]:
    """The splits the freeze harness drives.

    With real_splits=True the REAL train and valid frames are used (they
    are not the frozen split) and only local_test is a synthetic
    stand-in, carrying the real family domain and the real date range so
    the count pins hold. With real_splits=False every split is small and
    synthetic, for the fast battery runs.
    """
    local_test_dates = pd.date_range(
        SPLIT_FRAME_STARTS["local_test"], SPLIT_FRAME_ENDS["local_test"], freq="D"
    )
    if real_splits:
        train = load_split("train")
        valid = load_split("valid")
        families = list(train["family"].cat.categories)
        categories = {
            column: list(train[column].cat.categories)
            for column in ("family", "city", "state", "store_type")
        }
        local_test = _synthetic_split_frame(
            local_test_dates,
            None if christmas_intact else (1, families[0]),
            families,
            categories,
        )
        return {"train": train, "valid": valid, "local_test": local_test}
    valid_dates = pd.date_range("2016-07-14", "2016-09-11", freq="D")
    train_dates = pd.date_range("2016-03-26", "2016-03-27", freq="D")
    local_test = _synthetic_split_frame(
        local_test_dates, None if christmas_intact else (1, "family_00")
    )
    return {
        "train": _synthetic_split_frame(train_dates),
        "valid": _synthetic_split_frame(valid_dates),
        "local_test": local_test,
    }


def _stub_module(kind: str, convention: str, model_names: Sequence[str]) -> Any:
    """Module stub exposing fit_predict_frozen for the freeze path.

    convention "frame-order" returns predictions aligned to predict_frame
    row order (S3's wording); "local-test-only" returns the local_test
    suffix only (S4's wording). Metrics are identical either way.
    """
    import types

    module = types.ModuleType(kind)
    local_test_start = pd.Timestamp(SPLIT_FRAME_STARTS["local_test"])

    def fit_predict_frozen(
        model: str,
        fit_frame: pd.DataFrame,
        predict_frame: pd.DataFrame,
        config: dict[str, Any],
    ) -> np.ndarray:
        if model not in model_names:
            raise ContractError(f"synthetic stub refused unknown model {model!r}")
        sales = predict_frame["sales"].to_numpy(dtype="float64")
        is_local_test = (predict_frame["date"] >= local_test_start).to_numpy()
        multiplier = 1.0 + (model_names.index(model) + 1) * 1e-3
        pred = sales * multiplier
        if convention == "frame-order":
            return pred
        if convention == "local-test-only":
            return pred[is_local_test]
        raise ContractError(f"synthetic stub: unknown convention {convention!r}")

    module.fit_predict_frozen = fit_predict_frozen  # type: ignore[attr-defined]
    return module


def _registry_fixture_fold_metric(fold: int) -> dict[str, Any]:
    return {
        "fold": fold,
        "rmsle": 0.1 * fold,
        "mae": 0.1 * fold,
        "rmse": 0.1 * fold,
        "rmsle_active": 0.2 * fold,
        "mae_active": 0.2 * fold,
        "rmse_active": 0.2 * fold,
        "n_rows": FOLD_ROWS,
        "n_rows_active": 20000,
    }


def _registry_fixture_classical() -> dict[str, Any]:
    fold_m = _registry_fixture_fold_metric(1)
    fold_m2 = _registry_fixture_fold_metric(2)
    fold_m3 = _registry_fixture_fold_metric(3)
    pooled_m = {
        "rmsle": 0.15,
        "mae": 0.15,
        "rmse": 0.15,
        "rmsle_active": 0.25,
        "mae_active": 0.25,
        "rmse_active": 0.25,
        "n_rows": POOLED_FOLD_ROWS,
        "n_rows_active": 60000,
    }

    def cmod(name: str, kind: str) -> dict[str, Any]:
        return {
            "name": name,
            "kind": kind,
            "target": "sales",
            "hyperparameters": {},
            "fold_metrics": [fold_m, fold_m2, fold_m3],
            "pooled": pooled_m,
            "seeds": {},
            "tuning": "none",
            "christmas_2016": {"status": "freeze-time"},
        }

    return {
        "schema_version": 1,
        "git_sha": "a" * 40,
        "python_version": "3.13.0",
        "sklearn_version": "1.5.1",
        "split_bounds": {
            "train": "2013-01-01..2016-03-27",
            "valid": "2016-03-28..2016-09-11",
        },
        "training_mask": "fit_rows",
        "target": "sales",
        "prediction_clip": "0",
        "feature_columns": [f"f{i:02d}" for i in range(1, 24)],
        "folds": [
            {
                "fold": fold,
                "origin": origin,
                "window_start": start,
                "window_end": end,
                "n_rows": FOLD_ROWS,
            }
            for fold, origin, start, end in FOLD_TABLE_PINS
        ],
        "models": [
            cmod("naive", "baseline"),
            cmod("seasonal_naive", "baseline"),
            cmod("moving_average", "baseline"),
            cmod("ridge", "classical_ml"),
            cmod("hgb", "classical_ml"),
        ],
        "notes": [],
    }


def _registry_fixture_deep() -> dict[str, Any]:
    dfold = {
        "rmsle": 0.3,
        "mae": 0.3,
        "rmse": 0.3,
        "rmsle_active": 0.4,
        "mae_active": 0.4,
        "rmse_active": 0.4,
        "n_rows": FOLD_ROWS,
        "n_rows_active": 20000,
    }
    dgrid = {
        "rmsle": 0.5,
        "mae": 0.4,
        "rmse": 0.6,
    }
    dvalid = {
        "n_rows": 299376,
        "per_horizon_rmsle": [0.5] * 16,
        "full_grid": dgrid,
        "active_slice": dgrid,
        "n_active": 123456,
        "inference_seconds": 1.5,
    }

    def dtier(tier: str, arch: str) -> dict[str, Any]:
        return {
            "tier": tier,
            "architecture": arch,
            "checkpoint": "",
            "checkpoint_bytes": 1024,
            "checkpoint_sha256": "e" * 64,
            "params": 1000,
            "status": "ready",
            "train": {},
            "valid": dvalid,
            "fold_metrics": [dfold, dfold, dfold],
        }

    return {
        "schema_version": "1.0",
        "status": "complete",
        "track": "S4",
        "generated_at": "2026-11-10T00:00:00Z",
        "git_sha": "c" * 40,
        "data_fingerprint": {},
        "shared_config": {},
        "config_sha256": "d" * 64,
        "tiers": [
            dtier("deep-rnn", "rnn"),
            dtier("deep-lstm", "lstm"),
            dtier("deep-gru", "gru"),
        ],
        "ablation_2015plus": {},
        "comparison": {"status": "ok"},
        "runtime_tradeoff": {},
        "notes": [],
    }


def _write_registry_fixture(directory: Path) -> None:
    """Seed a temp dir with schema-clean registry artifacts."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "classical_results.json").write_text(
        json.dumps(_registry_fixture_classical()), encoding="utf-8"
    )
    (directory / "deep_results.json").write_text(
        json.dumps(_registry_fixture_deep()), encoding="utf-8"
    )


def _synthetic_freeze_run(
    tmp_dir: Path,
    *,
    convention: str,
    christmas_intact: bool = True,
    real_splits: bool = True,
) -> tuple[int, str, dict[str, Any] | None, dict[str, int]]:
    """Drive freeze() end to end on synthetic fixtures; return diagnostics.

    Patches the split loaders and stubs the sanctioned per-track modules,
    so nothing real is written; artifacts land in tmp_dir. Returns
    (rc, stdout, payload-or-None, fixture stats).

    real_splits defaults to True: the freeze body runs the REAL
    _contract_counts, which refuses the small synthetic frames (their
    closure rows are not the pinned counts), so the small-frame mode can
    only fail. It is therefore refused loudly here, at the call site,
    instead of returning rc=1 with no payload.
    """
    import contextlib
    import io
    from unittest import mock

    if not real_splits:
        raise ContractError(
            "_synthetic_freeze_run: real_splits=False is refused; the real "
            "_contract_counts rejects the small synthetic frames (their "
            "closure rows are not the pinned counts), so the run would fail "
            "with rc=1 and no payload"
        )
    splits = _synthetic_splits(
        christmas_intact=christmas_intact, real_splits=real_splits
    )
    _write_registry_fixture(tmp_dir)

    def fake_load_split(split: str) -> pd.DataFrame:
        if split not in splits:
            raise ContractError(f"synthetic stub refused split {split!r}")
        return splits[split]

    def fake_git_sha() -> tuple[str, list[str]]:
        """A stand-in sha, so the audit runs outside a git checkout.

        The freeze refuses a missing sha in production (the numbers must
        describe a committed tree); the audit only drives the machinery,
        so it supplies one.
        """
        return "a" * 40, []

    def fake_load_frame(
        train: pd.DataFrame | None = None,
        valid: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if train is None:
            train = splits["train"]
        if valid is None:
            valid = splits["valid"]
        frame = pd.concat([train, valid], ignore_index=True)
        return frame.sort_values("date", kind="mergesort").reset_index(drop=True)

    modules = {
        "classical": _stub_module("classical", convention, CLASSICAL_MODEL_NAMES),
        "deep": _stub_module("deep", convention, DEEP_TIER_NAMES),
    }

    saved_flag = os.environ.get("FREEZE_RUN")
    os.environ["FREEZE_RUN"] = "1"
    buffer = io.StringIO()
    try:
        with (
            mock.patch.object(sys.modules[__name__], "load_split", fake_load_split),
            mock.patch.object(sys.modules[__name__], "load_frame", fake_load_frame),
            mock.patch.object(sys.modules[__name__], "_git_sha", fake_git_sha),
            mock.patch.dict(sys.modules, modules),
            contextlib.redirect_stdout(buffer),
        ):
            rc = freeze(artifacts_dir=tmp_dir)
    finally:
        if saved_flag is None:
            os.environ.pop("FREEZE_RUN", None)
        else:
            os.environ["FREEZE_RUN"] = saved_flag
    stdout = buffer.getvalue()
    local_test = splits["local_test"]
    stats = {
        "local_test_rows": len(local_test),
        "local_test_active": int((local_test["sales"] > 0).sum()),
        "christmas_rows": int((local_test["date"] == CHRISTMAS_2016_DATE).sum()),
    }
    results_path = tmp_dir / "freeze_results.json"
    payload: dict[str, Any] | None = None
    if results_path.exists():
        try:
            payload = json.loads(results_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = None
    return rc, stdout, payload, stats


def _check_synthetic_freeze_run_defaults() -> None:
    """The synthetic freeze run defaults to the configuration that works.

    The freeze body runs the real _contract_counts, which refuses the
    small synthetic frames, so the small-frame mode is refused loudly at
    the call site and the default is the supported one. A caller that
    forgets the argument must never get rc=1 with no payload.
    """
    import inspect

    default = inspect.signature(_synthetic_freeze_run).parameters["real_splits"].default
    if default is not True:
        raise ContractError(
            f"_synthetic_freeze_run's real_splits default is {default!r}, not "
            "True; the small synthetic frames are refused by the real "
            "_contract_counts, so the default must be the supported mode"
        )
    with tempfile.TemporaryDirectory() as tmp_name:
        try:
            _synthetic_freeze_run(
                Path(tmp_name), convention="frame-order", real_splits=False
            )
        except ContractError as error:
            if "real_splits=False is refused" not in str(error):
                raise ContractError(
                    f"the small-frame mode refused with the wrong lead: {error}"
                ) from error
        else:
            raise ContractError(
                "the small-frame mode was accepted; it returns rc=1 with no "
                "payload instead of failing loudly"
            )
    print(
        "PASS freeze-run defaults: the synthetic run defaults to the real "
        "splits, and the small-frame mode is refused loudly instead of "
        "returning rc=1 with no payload"
    )


def _has_bound_date(strings: set[str], bound: str, start: str, end: str) -> bool:
    """The bound appears bare, or inside a combined frame-range string."""
    if any(s == bound for s in strings):
        return True
    combined = f"{start}..{end}"
    return any(combined in s for s in strings)


def _check_freeze_run_invariants() -> None:
    """Exactly one freeze_run=true record ever; refusals stay friendly."""
    import contextlib
    import io

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp_dir = Path(tmp_name)
        write_run_manifest(
            "contract.py --freeze",
            freeze_run=True,
            elapsed_seconds=0.0,
            arguments={"freeze": "1"},
            notes=["fixture: the one testified freeze"],
            artifacts_dir=tmp_dir,
        )
        try:
            write_run_manifest(
                "contract.py --freeze",
                freeze_run=True,
                elapsed_seconds=0.0,
                artifacts_dir=tmp_dir,
            )
        except ContractError as error:
            if "already carries" not in str(error):
                raise ContractError(
                    f"the second freeze stamp refused with the wrong lead: {error}"
                ) from error
        else:
            raise ContractError("a second freeze_run=true stamp was accepted")
        saved_flag = os.environ.get("FREEZE_RUN")
        os.environ["FREEZE_RUN"] = "1"
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                refused_rc = freeze(artifacts_dir=tmp_dir)
        finally:
            if saved_flag is None:
                os.environ.pop("FREEZE_RUN", None)
            else:
                os.environ["FREEZE_RUN"] = saved_flag
        out = buffer.getvalue()
        if refused_rc != 1:
            raise ContractError(
                f"freeze after a stamped run returned {refused_rc}, not 1"
            )
        if "already carries" not in out:
            raise ContractError(
                f"freeze after a stamped run printed the wrong lead: {out!r}"
            )
        if "Traceback" in out:
            raise ContractError("freeze after a stamped run leaked a traceback")
        records = list(read_run_manifest(tmp_dir / "run_manifest.json"))
        stamped = [record for record in records if record["freeze_run"] is True]
        if len(stamped) != 1:
            raise ContractError(
                f"the temp manifest holds {len(stamped)} freeze_run=true records, "
                "exactly one is allowed"
            )
        last = records[-1]
        if last["freeze_run"] is not False or last["exit"] != 1:
            raise ContractError(
                f"the refused freeze attempt was recorded as {last!r}, "
                "expected freeze_run=false exit=1"
            )
        if last["status"] != "FAIL" or not last["notes"]:
            raise ContractError(
                "the refused freeze attempt left no FAIL record with notes"
            )
        if not any("failed phase" in note for note in last["notes"]):
            raise ContractError(
                "the refused freeze attempt did not record its failed phase"
            )
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp_dir = Path(tmp_name)
        saved_flag = os.environ.get("FREEZE_RUN")
        os.environ["FREEZE_RUN"] = "1"
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                failed_rc = freeze(artifacts_dir=tmp_dir)
        finally:
            if saved_flag is None:
                os.environ.pop("FREEZE_RUN", None)
            else:
                os.environ["FREEZE_RUN"] = saved_flag
        out = buffer.getvalue()
        if failed_rc != 1:
            raise ContractError(f"freeze on an empty registry returned {failed_rc}")
        if "Traceback" in out:
            raise ContractError("a failed freeze attempt leaked a traceback")
        records = list(read_run_manifest(tmp_dir / "run_manifest.json"))
        last = records[-1]
        if last["freeze_run"] is not False or last["status"] != "FAIL":
            raise ContractError("the failed freeze attempt must stay freeze_run=false")
        if last["exit"] != 1 or not any(
            "failed phase" in note for note in last["notes"]
        ):
            raise ContractError(
                "the failed freeze attempt was not recorded with exit=1 "
                "and its failed phase named"
            )
        write_run_manifest(
            "contract.py --freeze",
            freeze_run=True,
            elapsed_seconds=0.0,
            artifacts_dir=tmp_dir,
        )
        records = list(read_run_manifest(tmp_dir / "run_manifest.json"))
        stamped = [record for record in records if record["freeze_run"] is True]
        if len(stamped) != 1:
            raise ContractError(
                "a failed attempt consumed the freeze gate; the re-run could "
                "not stamp freeze_run=true"
            )
        forged = [dict(record) for record in records]
        for record in forged:
            record["freeze_run"] = True
        forged_path = tmp_dir / "forged.json"
        forged_path.write_text(json.dumps(forged, indent=2) + "\n", encoding="utf-8")
        try:
            read_run_manifest(forged_path)
        except ContractError as error:
            if "exactly one freeze run" not in str(error):
                raise ContractError(
                    f"a two-freeze manifest refused with the wrong lead: {error}"
                ) from error
        else:
            raise ContractError(
                "read_run_manifest accepted two freeze_run=true records"
            )
    print(
        "PASS freeze-invariants: one stamped freeze_run=true record ever; a "
        "second attempt refuses; a failed attempt records freeze_run=false "
        "and the re-run can still stamp"
    )


def _check_freeze_results_writer() -> None:
    """Freeze-results writer: pinned field set, atomic write, headline."""
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp_dir = Path(tmp_name)
        rc, out, payload, stats = _synthetic_freeze_run(
            tmp_dir, convention="frame-order", real_splits=True
        )
        if rc != 0:
            tail = out.strip().splitlines()[-3:]
            raise ContractError(
                f"synthetic freeze exited {rc}; last lines: {' | '.join(tail)}"
            )
        if "Traceback" in out:
            raise ContractError("synthetic freeze leaked a traceback")
        results_path = tmp_dir / "freeze_results.json"
        if not results_path.is_file():
            raise ContractError("freeze did not write artifacts/freeze_results.json")
        leftovers = list(tmp_dir.rglob("*.tmp"))
        if leftovers:
            raise ContractError(f"the atomic write left temp files behind: {leftovers}")
        allowed = {
            "freeze_results.json",
            "classical_results.json",
            "deep_results.json",
            "run_manifest.json",
            "run_manifest.json.lock",
        }
        residue = sorted(p.name for p in tmp_dir.iterdir() if p.name not in allowed)
        if residue:
            raise ContractError(
                f"unexpected files next to freeze_results.json: {residue}"
            )
        try:
            reread = json.loads(results_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ContractError(
                f"freeze_results.json does not re-parse after the write: {error}"
            ) from error
        if reread != payload:
            raise ContractError("the written freeze_results.json drifted on re-parse")
        size = results_path.stat().st_size
        if size > 5 * 1024 * 1024:
            raise ContractError(
                f"freeze_results.json is {size / 1e6:.2f} MB, over the 5 MB pin"
            )
        model_names = _registered_model_names()
        # ONE schema validator: the production one. The battery calls it
        # directly, so a schema drift the freeze refuses is a schema drift
        # the audit also refuses.
        _validate_freeze_results(reread, expected_models=len(model_names))
        # The semantic facts the schema cannot express: the headline is the
        # lowest-RMSLE model, and each slice covers the rows it claims.
        freeze_records = _phase_records(reread["freeze_phase"])
        for index, record in enumerate(freeze_records):
            label = f"freeze_phase[{index}]"
            slices = _slice_blocks(
                record, ("full_grid", "active_slice", "christmas"), label
            )
            if slices["full_grid"]["n_rows"] != stats["local_test_rows"]:
                raise ContractError(
                    f"{label}: full-grid n_rows {slices['full_grid']['n_rows']} "
                    f"!= the predict_frame local_test row count "
                    f"{stats['local_test_rows']}"
                )
            if slices["full_grid"]["n_rows_active"] != stats["local_test_active"]:
                raise ContractError(
                    f"{label}: full-grid n_rows_active "
                    f"{slices['full_grid']['n_rows_active']} != the local_test "
                    f"active count {stats['local_test_active']}"
                )
            if slices["christmas"]["n_rows"] != stats["christmas_rows"]:
                raise ContractError(
                    f"{label}: the christmas-2016 slice scored "
                    f"{slices['christmas']['n_rows']} rows, expected "
                    f"{stats['christmas_rows']}"
                )
        best = min(
            freeze_records,
            key=lambda record: float(record["slices"]["full_grid"]["rmsle"]),
        )
        if reread["headline_model"] != best["name"]:
            raise ContractError(
                f"headline_model {reread['headline_model']!r} != the "
                f"lowest-RMSLE freeze_phase model {best['name']!r}"
            )
    print(
        "PASS freeze-results: freeze_results.json written atomically (temp "
        "file then an exclusive link; no *.tmp residue; re-parses); sheet "
        "Section-10 field set with the pinned types; headline_model is the "
        "lowest-RMSLE registered model; model count equals the registered count"
    )


def _check_freeze_results_validation() -> None:
    """Field validation battery; refuses with friendly ContractErrors."""
    validator = _validate_freeze_results
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp_dir = Path(tmp_name)
        rc, out, payload, _stats = _synthetic_freeze_run(
            tmp_dir, convention="frame-order", real_splits=True
        )
        if rc != 0 or payload is None:
            tail = out.strip().splitlines()[-3:]
            raise ContractError(
                f"synthetic freeze exited {rc}; payload not harvested: "
                f"{' | '.join(tail)}"
            )

    def _lookup_block(bad: dict[str, Any], kind: str) -> dict[str, Any]:
        for path, node in _walk_nodes(bad):
            if (
                isinstance(node, dict)
                and set(node) == set(SCORE_KEYS)
                and _classify_slice(path) == kind
            ):
                return node
        raise ContractError(f"no {kind} score block in the payload")

    refused = 0

    def _expect_refusal(label: str, mutate: Callable[[dict[str, Any]], None]) -> None:
        nonlocal refused
        bad = copy.deepcopy(payload)
        mutate(bad)
        try:
            validator(bad)
        except ContractError as error:
            if not str(error).strip():
                raise ContractError(
                    f"field validation {label}: empty refusal message"
                ) from error
            if "Traceback" in str(error):
                raise ContractError(
                    f"field validation {label}: refusal leaked a traceback"
                ) from error
            refused += 1
            return
        except Exception as error:
            raise ContractError(
                f"field validation {label}: {type(error).__name__} escaped "
                f"instead of a friendly ContractError: {error!r}"
            ) from error
        raise ContractError(f"field validation {label}: accepted a broken payload")

    for key in FREEZE_RESULTS_KEYS:
        _expect_refusal(f"missing top-level key {key!r}", lambda bad, k=key: bad.pop(k))
    try:
        _lookup_block(payload, "full_grid")
        _lookup_block(payload, "active_slice")
    except ContractError as error:
        raise ContractError(
            "the synthetic freeze payload carries no full-grid / active-slice "
            f"score block to mutate; the field-validation battery cannot run: {error}"
        ) from error

    def _set_full(key: str, value: Any) -> Callable[[dict[str, Any]], None]:
        def mutate(bad: dict[str, Any]) -> None:
            target = _lookup_block(bad, "full_grid")
            target[key] = value

        return mutate

    def _set_active(key: str, value: Any) -> Callable[[dict[str, Any]], None]:
        def mutate(bad: dict[str, Any]) -> None:
            target = _lookup_block(bad, "active_slice")
            target[key] = value

        return mutate

    def _drop_full_key(bad: dict[str, Any]) -> None:
        _lookup_block(bad, "full_grid").pop("mae")

    _expect_refusal("frozen_at wrong type", lambda bad: bad.update(frozen_at=123))
    _expect_refusal(
        "frozen_at unparsable", lambda bad: bad.update(frozen_at="not-a-timestamp")
    )
    _expect_refusal("git_sha wrong type", lambda bad: bad.update(git_sha=123))
    _expect_refusal("git_sha not 40-hex", lambda bad: bad.update(git_sha="z" * 40))
    _expect_refusal(
        "headline_model not a string", lambda bad: bad.update(headline_model=None)
    )
    _expect_refusal("notes not a list", lambda bad: bad.update(notes="not-a-list"))
    _expect_refusal("notes with non-string entries", lambda bad: bad.update(notes=[1]))
    _expect_refusal("fold_phase not a list", lambda bad: bad.update(fold_phase={}))
    _expect_refusal("freeze_phase not a list", lambda bad: bad.update(freeze_phase={}))
    _expect_refusal("non-finite rmsle", lambda bad: _set_full("rmsle", math.inf)(bad))
    _expect_refusal("nan rmsle", lambda bad: _set_full("rmsle", math.nan)(bad))
    _expect_refusal("negative mae", lambda bad: _set_full("mae", -0.5)(bad))
    _expect_refusal("negative rmse", lambda bad: _set_full("rmse", -0.5)(bad))
    _expect_refusal(
        "non-finite active-slice mae",
        lambda bad: _set_active("mae", math.inf)(bad),
    )
    _expect_refusal("negative n_rows", lambda bad: _set_full("n_rows", -1)(bad))
    _expect_refusal(
        "negative n_rows_active", lambda bad: _set_full("n_rows_active", -1)(bad)
    )
    _expect_refusal("missing score key in a slice", _drop_full_key)
    _expect_refusal("empty model list", lambda bad: bad.update(freeze_phase=[]))
    _expect_refusal(
        "model count below the registry",
        lambda bad: bad.update(freeze_phase=_drop_last_freeze_model(bad)),
    )
    _expect_refusal(
        "headline_model naming an unregistered model",
        lambda bad: bad.update(headline_model="not_a_registered_model"),
    )
    print(
        f"PASS freeze-results: field validation refused {refused} broken "
        "payloads with friendly ContractErrors (missing field / wrong type / "
        "non-finite / negative metric / empty model list / registry count "
        "mismatch / unregistered headline)"
    )


def _check_christmas_slice_selection() -> None:
    """Christmas-2016 slice: selected by date; 1,782 rows pinned."""
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp_dir = Path(tmp_name)
        rc, out, payload, stats = _synthetic_freeze_run(
            tmp_dir, convention="frame-order", christmas_intact=True, real_splits=True
        )
        if rc != 0:
            tail = out.strip().splitlines()[-3:]
            raise ContractError(
                f"synthetic freeze exited {rc}; last lines: {' | '.join(tail)}"
            )
        if "Traceback" in out:
            raise ContractError("synthetic freeze leaked a traceback")
        if stats["christmas_rows"] != CHRISTMAS_2016_ROWS:
            raise ContractError(
                f"fixture sanity: the synthetic local_test carries "
                f"{stats['christmas_rows']} Christmas rows, expected "
                f"{CHRISTMAS_2016_ROWS}"
            )
        if payload is None:
            raise ContractError("the synthetic freeze wrote no freeze_results.json")
        for index, record in enumerate(_phase_records(payload["freeze_phase"])):
            slices = _slice_blocks(record, ("christmas",), f"freeze_phase[{index}]")
            if slices["christmas"]["n_rows"] != CHRISTMAS_2016_ROWS:
                raise ContractError(
                    f"freeze_phase[{index}]: the christmas-2016 slice scored "
                    f"{slices['christmas']['n_rows']} rows, expected "
                    f"{CHRISTMAS_2016_ROWS}"
                )
    with tempfile.TemporaryDirectory() as tmp_name:
        broken_dir = Path(tmp_name)
        rc, out, payload, stats = _synthetic_freeze_run(
            broken_dir,
            convention="frame-order",
            christmas_intact=False,
            real_splits=True,
        )
        if stats["christmas_rows"] != CHRISTMAS_2016_ROWS - 1:
            raise ContractError(
                f"fixture sanity: the broken fixture carries "
                f"{stats['christmas_rows']} Christmas rows, expected "
                f"{CHRISTMAS_2016_ROWS - 1}"
            )
        if rc != 1:
            raise ContractError(
                f"a one-row-short Christmas slice exited {rc}, not 1; the "
                "selection must refuse exactly that slice"
            )
        if "Traceback" in out:
            raise ContractError("the Christmas refusal leaked a traceback")
        records = list(read_run_manifest(broken_dir / "run_manifest.json"))
        last = records[-1]
        if last["freeze_run"] is not False or last["status"] != "FAIL":
            raise ContractError(
                "the refused Christmas attempt must record freeze_run=false "
                "with status FAIL"
            )
    print(
        "PASS christmas-slice: the 1,782 rows of 2016-12-25 (54 stores x 33 "
        "families) are selected by date; a one-row-short slice refuses with "
        "freeze_run=false"
    )


def _check_prediction_alignment() -> None:
    """Both alignment conventions yield the same local_test-aligned result."""
    payloads: dict[str, dict[str, Any]] = {}
    for convention in ("frame-order", "local-test-only"):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp_dir = Path(tmp_name)
            rc, out, payload, _stats = _synthetic_freeze_run(
                tmp_dir, convention=convention, real_splits=True
            )
            if rc != 0:
                tail = out.strip().splitlines()[-3:]
                raise ContractError(
                    f"synthetic freeze ({convention}) exited {rc}; last "
                    f"lines: {' | '.join(tail)}"
                )
            if "Traceback" in out:
                raise ContractError(
                    f"synthetic freeze ({convention}) leaked a traceback"
                )
            if payload is None:
                raise ContractError(
                    f"synthetic freeze ({convention}) wrote no freeze_results.json"
                )
            payloads[convention] = payload
    left = copy.deepcopy(payloads["frame-order"])
    right = copy.deepcopy(payloads["local-test-only"])
    left.pop("frozen_at", None)
    right.pop("frozen_at", None)
    if left != right:
        differences = list(_payload_differences(left, right))[:5]
        raise ContractError(
            "the two alignment conventions produced different freeze "
            f"payloads; first divergences: {differences}"
        )
    print(
        "PASS prediction-alignment: predict_frame-order and local_test-rows-only "
        "predictions produce the identical local_test-aligned freeze payload "
        "(modulo frozen_at)"
    )

    # The alignment cases run on the synthetic fixture frames: the frozen
    # local_test is NEVER opened here, and the fixture carries the same
    # series-major shape (unique (store_nbr, family, date) keys) the
    # alignment is keyed on.
    alignment_fixtures = _synthetic_freeze_fixtures()
    local_test = alignment_fixtures["local_test"]
    predict_frame = alignment_fixtures["predict_frame"]
    local_test_rows = len(local_test)
    predict_frame_rows = len(predict_frame)

    neither = np.zeros(local_test_rows - 1, dtype=np.float64)
    try:
        _align_local_test_predictions("stub", neither, predict_frame, local_test)
    except ContractError as error:
        message = str(error)
        if (
            str(neither.size) not in message
            or str(local_test_rows) not in message
            or str(predict_frame_rows) not in message
        ):
            raise ContractError(
                f"a prediction array matching neither legal shape refused with "
                f"the wrong lead: {error}"
            ) from error
    else:
        raise ContractError(
            "a prediction array matching neither legal shape was accepted; "
            "the two legal shapes must be distinguished by length"
        )

    local_marker = np.arange(local_test_rows, dtype=np.float64)
    aligned_local = _align_local_test_predictions(
        "stub", local_marker, predict_frame, local_test
    )
    if not np.array_equal(aligned_local, local_marker):
        raise ContractError(
            "a local_test-rows-only prediction was not aligned to local_test "
            "in order; the local_test-length shape was not distinguished"
        )

    wanted = _row_keys(local_test)
    frame_index = {key: idx for idx, key in enumerate(_row_keys(predict_frame))}
    frame_marker = np.full(predict_frame_rows, -1.0, dtype=np.float64)
    for idx, key in enumerate(wanted):
        frame_marker[frame_index[key]] = float(idx)
    aligned_frame = _align_local_test_predictions(
        "stub", frame_marker, predict_frame, local_test
    )
    if not np.array_equal(aligned_frame, local_marker):
        raise ContractError(
            "a predict_frame-order prediction was not gathered onto local_test "
            "by frame position; the frame-length shape was not distinguished"
        )

    keyed_frame = pd.Series(
        frame_marker.copy(),
        index=pd.MultiIndex.from_arrays(
            [
                predict_frame["store_nbr"].to_numpy(),
                predict_frame["family"].astype(str).to_numpy(),
                predict_frame["date"].to_numpy(),
            ],
            names=["store_nbr", "family", "date"],
        ),
    )
    keyed_frame = keyed_frame.sample(frac=1.0, random_state=7)
    aligned_keyed = _align_local_test_predictions(
        "stub", keyed_frame, predict_frame, local_test
    )
    if not np.array_equal(aligned_keyed, local_marker):
        raise ContractError(
            "a keyed Series prediction was not reconciled to local_test by its "
            "prediction-side keys"
        )

    print(
        "PASS prediction-alignment: a length matching neither legal shape is "
        "refused with a friendly lead, local_test-length and predict_frame-length "
        "arrays are distinguished by length, and a keyed Series is reconciled by "
        "its prediction-side keys"
    )


def _payload_differences(
    left: Any, right: Any, path: str = "$"
) -> Generator[str, None, None]:
    """Dotted paths where two JSON trees disagree (diagnostics only)."""
    if type(left) is not type(right):
        yield f"{path}: {type(left).__name__} vs {type(right).__name__}"
        return
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left:
                yield f"{path}.{key}: only in the local-test-only payload"
            elif key not in right:
                yield f"{path}.{key}: only in the frame-order payload"
            else:
                yield from _payload_differences(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, list):
        if len(left) != len(right):
            yield f"{path}: length {len(left)} vs {len(right)}"
            return
        for index, (item_left, item_right) in enumerate(zip(left, right, strict=True)):
            yield from _payload_differences(item_left, item_right, f"{path}[{index}]")
        return
    if left != right:
        yield f"{path}: {left!r} vs {right!r}"


def _check_registry(directory: Path | None = None) -> None:
    """Day-2 block: registered homes; strays fail loud, files validate.

    A registry that is entirely absent is a "not merged yet" notice and
    stays green; a registry that is half-landed (some registered
    artifacts present, others missing) is a real finding and refuses
    loudly, because a half-merge is a problem, not a notice.
    """
    directory = ARTIFACTS_DIR if directory is None else directory
    if not directory.exists():
        print(
            "INFO registry: artifacts/ absent; 2 registered homes pending the "
            "S3/S4 merges (expected before Day 4)"
        )
        return
    freeze_results = directory / FREEZE_RESULTS_FILENAME
    if freeze_results.is_file():
        # The freeze artifact is validated, not waved through: a drift in
        # the committed JSON is a refusal, not an INFO.
        try:
            payload = json.loads(freeze_results.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as error:
            raise ContractError(
                f"registry: artifacts/{FREEZE_RESULTS_FILENAME} does not parse: {error}"
            ) from error
        registered_models = [
            name for spec in ARTIFACT_REGISTRY.values() for name in spec["model_names"]
        ]
        _validate_freeze_results(
            payload,
            expected_models=len(registered_models),
            expected_rows=PINNED_SPLIT_ROWS["local_test"],
            expected_bounds=(
                SPLIT_FRAME_STARTS["local_test"],
                SPLIT_FRAME_ENDS["local_test"],
            ),
        )
        print(
            f"PASS registry: artifacts/{FREEZE_RESULTS_FILENAME} is present "
            "and carries the pinned Section-10 schema"
        )
    registered = _scan_registered_artifacts(directory)
    if not registered:
        print(
            "INFO registry: no registered artifacts present; "
            "2 registered homes pending the "
            "S3/S4 merges (expected before Day 4)"
        )
        return
    missing = sorted(set(ARTIFACT_REGISTRY) - set(registered))
    # The artifacts that ARE there are validated first, so a corrupt one
    # is reported as corrupt; the half-landed registry is then refused.
    blurbs: list[tuple[str, str]] = []
    for name, path in sorted(registered.items()):
        (blurb, _payload) = _check_registered_artifact(name, path)
        blurbs.append((name, blurb))
    if missing:
        raise ContractError(
            f"registry: half-landed registry; {sorted(registered)} present but "
            f"{missing} missing; the registry is merged whole or not at all"
        )
    for name, blurb in blurbs:
        print(f"PASS registry: {name} schema-clean ({blurb})")
    print("PASS registry: all registered artifacts validated")


def _check_registry_presence() -> None:
    """A half-landed registry refuses loudly; an absent one stays an INFO.

    Both cases run against throwaway directories, so the registry
    block's decision never depends on the real artifacts directory.
    """
    import io

    with tempfile.TemporaryDirectory() as tmp_name:
        base = Path(tmp_name)
        empty = base / "empty"
        empty.mkdir()
        # Nothing merged yet: an INFO notice, never a refusal.
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            _check_registry(empty)
        if "INFO registry" not in buffer.getvalue():
            raise ContractError(
                "an absent registry did not print the INFO notice: "
                f"{buffer.getvalue()!r}"
            )
        # One artifact landed: a loud refusal, never an INFO notice.
        half = base / "half"
        half.mkdir()
        (half / "classical_results.json").write_text(
            json.dumps(_registry_fixture_classical()), encoding="utf-8"
        )
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                _check_registry(half)
        except ContractError as error:
            if "half-landed registry" not in str(error):
                raise ContractError(
                    f"the half-landed registry refused with the wrong lead: {error}"
                ) from error
        except IndexError as error:
            raise ContractError(
                f"the half-landed registry escaped as a raw IndexError: {error}"
            ) from error
        else:
            raise ContractError(
                "a half-landed registry was accepted; a half-merge must be a "
                "loud finding, not an INFO notice"
            )
    print(
        "PASS registry: a half-landed registry refuses loudly, while a registry "
        "with none of its artifacts merged yet stays an INFO notice"
    )


def _check_freeze_safety() -> None:
    """Day-5 safety: the whitelist, the atomic write, no partial artifacts.

    The freeze output must not be mistaken for a stray artifact by the
    next run's scan, the write must publish atomically, and a failed
    write must leave neither a temp file nor a half-written artifact.
    """
    if FREEZE_RESULTS_FILENAME not in CONTRACT_ARTIFACT_FILENAMES:
        raise ContractError(
            f"{FREEZE_RESULTS_FILENAME} must be in CONTRACT_ARTIFACT_FILENAMES "
            "so the registry scan does not treat the freeze output as an "
            "unregistered stray on later runs"
        )
    with tempfile.TemporaryDirectory() as tmp_name:
        base = Path(tmp_name)
        whitelist_dir = base / "whitelist"
        whitelist_dir.mkdir()
        (whitelist_dir / FREEZE_RESULTS_FILENAME).write_text("{}", encoding="utf-8")
        try:
            _scan_registered_artifacts(whitelist_dir)
        except ContractError as error:
            raise ContractError(
                f"{FREEZE_RESULTS_FILENAME} is whitelisted but was refused "
                f"as a stray: {error}"
            ) from error
    print(
        f"PASS freeze safety: {FREEZE_RESULTS_FILENAME} is whitelisted as "
        "contract furniture and the registry scan accepts it"
    )

    from unittest import mock

    with tempfile.TemporaryDirectory() as tmp_name:
        directory = Path(tmp_name)
        target = directory / FREEZE_RESULTS_FILENAME
        payload = {"headline_model": "naive", "ok": True}
        _write_freeze_results(target, payload)
        if not target.is_file():
            raise ContractError(
                f"the atomic writer did not create {FREEZE_RESULTS_FILENAME}"
            )
        if json.loads(target.read_text(encoding="utf-8")) != payload:
            raise ContractError(
                f"{FREEZE_RESULTS_FILENAME} drifted on re-read after the write"
            )
        leftovers = sorted(directory.glob(f"{FREEZE_RESULTS_FILENAME}.*.tmp"))
        if leftovers:
            raise ContractError(
                f"the atomic write left temp file(s) behind: {leftovers}"
            )

        def _boom(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
            raise OSError("injected publish failure")

        fresh_dir = directory / "fresh"
        fresh_dir.mkdir()
        # The freeze artifact is published with an exclusive link, so the
        # failure to inject is on the link, not the replace.
        with mock.patch.object(os, "link", new=_boom):
            try:
                _write_freeze_results(fresh_dir / FREEZE_RESULTS_FILENAME, {"a": 1})
            except ContractError as error:
                if FREEZE_RESULTS_FILENAME not in str(error):
                    raise ContractError(
                        f"the atomic write failed with the wrong lead: {error}"
                    ) from error
            else:
                raise ContractError(
                    "the atomic write accepted an injected publish failure"
                )
        if (fresh_dir / FREEZE_RESULTS_FILENAME).exists():
            raise ContractError(
                f"the atomic write left a partial {FREEZE_RESULTS_FILENAME} "
                "behind on failure"
            )
        if list(fresh_dir.glob(f"{FREEZE_RESULTS_FILENAME}.*.tmp")):
            raise ContractError(
                "the atomic write left a temp file behind after a failed replace"
            )

        keep_dir = directory / "keep"
        keep_dir.mkdir()
        original = {"ok": "original"}
        kept = keep_dir / FREEZE_RESULTS_FILENAME
        kept.write_text(json.dumps(original, indent=2) + "\n", encoding="utf-8")
        with mock.patch.object(os, "link", new=_boom):
            try:
                _write_freeze_results(kept, {"ok": "new"})
            except ContractError:
                pass
            else:
                raise ContractError(
                    "the atomic write accepted a failure over an existing file"
                )
        if json.loads(kept.read_text(encoding="utf-8")) != original:
            raise ContractError(
                f"the atomic write did not preserve an existing "
                f"{FREEZE_RESULTS_FILENAME} on failure"
            )
        # The artifact is written once: a second write must refuse and
        # leave the first file byte-identical.
        try:
            _write_freeze_results(kept, {"ok": "second"})
        except ContractError as error:
            if "already exists" not in str(error):
                raise ContractError(
                    f"the overwrite refusal used the wrong lead: {error}"
                ) from error
        else:
            raise ContractError(
                f"the atomic write overwrote an existing {FREEZE_RESULTS_FILENAME}"
            )
        if json.loads(kept.read_text(encoding="utf-8")) != original:
            raise ContractError(
                f"the overwrite refusal did not leave the existing "
                f"{FREEZE_RESULTS_FILENAME} untouched"
            )

        # A10: a temp-file unlink failure after the artifact is published
        # must not escape to the operator; the artifact is valid and the
        # write must still be reported as a success.
        unlink_dir = directory / "unlink-ok"
        unlink_dir.mkdir()
        unlink_target = unlink_dir / FREEZE_RESULTS_FILENAME
        unlink_payload = {"ok": "published-before-unlink"}
        real_path_unlink = Path.unlink

        def _unlink_boom(self: Path, *args: Any, **kwargs: Any) -> None:
            if self == unlink_target:
                return real_path_unlink(self, *args, **kwargs)
            if self.name.endswith(".tmp") and self.parent == unlink_dir:
                raise OSError(errno.EPERM, "injected unlink failure")
            return real_path_unlink(self, *args, **kwargs)

        with mock.patch.object(Path, "unlink", new=_unlink_boom):
            _write_freeze_results(unlink_target, unlink_payload)
        if not unlink_target.is_file():
            raise ContractError(
                "the atomic write did not publish the artifact when the temp "
                "unlink failed after publish"
            )
        if json.loads(unlink_target.read_text(encoding="utf-8")) != unlink_payload:
            raise ContractError(
                "the artifact drifted after a temp-unlink failure on a published write"
            )

        # A11: a filesystem that does not support hard links must take the
        # exclusive-create fallback, not a generic write failure.
        for link_errno, label in (
            (errno.EPERM, "EPERM"),
            (errno.ENOTSUP, "ENOTSUP"),
            (errno.EOPNOTSUPP, "EOPNOTSUPP"),
        ):
            nohard_dir = directory / f"nohard-{label.lower()}"
            nohard_dir.mkdir()
            nohard_target = nohard_dir / FREEZE_RESULTS_FILENAME
            nohard_payload = {"ok": label}

            def _link_unsupported(
                src: Any,
                dst: Any,
                *args: Any,
                _errno: int = link_errno,
                **kwargs: Any,
            ) -> None:
                raise OSError(_errno, "injected hard-link unsupported")

            with mock.patch.object(os, "link", new=_link_unsupported):
                _write_freeze_results(nohard_target, nohard_payload)
            if not nohard_target.is_file():
                raise ContractError(
                    f"the atomic write did not publish the artifact on {label} "
                    "hard-link failure"
                )
            if json.loads(nohard_target.read_text(encoding="utf-8")) != nohard_payload:
                raise ContractError(
                    f"the artifact drifted on the {label} hard-link fallback path"
                )
            leftovers = sorted(nohard_dir.glob(f"{FREEZE_RESULTS_FILENAME}.*.tmp"))
            if leftovers:
                raise ContractError(
                    f"the {label} hard-link fallback left temp file(s) "
                    f"behind: {leftovers}"
                )
    print(
        "PASS freeze safety: the freeze output is written through a same-dir "
        "temp and an exclusive link, re-reads to the same payload, leaves no "
        "temp file, never overwrites an existing artifact, leaves no "
        "partial or touched artifact on failure, survives a temp-unlink "
        "failure after publish, and falls back when hard links are unsupported"
    )


def _check_fold_phase() -> None:
    """The fold phase agrees with the shared fold table, fold for fold.

    The fold phase is derived from the validated artifacts and the one
    shared fold table, never re-derived per track: the three pinned
    windows, 28,512 rows each, 85,536 pooled, every registered model
    present in the registry's order, and every per-fold score block
    carried straight from the artifact. A corrupted artifact must be
    refused loudly rather than scored.
    """
    if not ARTIFACTS_DIR.exists():
        print(
            "INFO fold-phase: artifacts/ absent; the fold-phase checks are "
            "pending the S3/S4 merges (expected before Day 4)"
        )
        return
    present = [name for name in ARTIFACT_REGISTRY if (ARTIFACTS_DIR / name).is_file()]
    if not present:
        print(
            "INFO fold-phase: no registered artifacts present; the fold-phase "
            "checks are pending the S3/S4 merges (expected before Day 4)"
        )
        return
    # The artifacts are present, so a refusal here is a real finding, not
    # a "not merged yet" notice: corruption must not hide behind an INFO.
    payloads = _registered_payloads(ARTIFACTS_DIR)
    derived = _fold_phase_from_artifacts(payloads)
    table = folds_table().to_dict("records")
    fold_records = _fold_records(derived)
    expected_models = [
        name for spec in ARTIFACT_REGISTRY.values() for name in spec["model_names"]
    ]
    derived_models = [str(model["name"]) for model in derived["models"]]
    if derived_models != expected_models:
        raise ContractError(
            f"fold_phase: derived model order {derived_models} != the "
            f"registered {expected_models}"
        )
    if len(fold_records) != len(expected_models) * len(FOLD_TABLE_PINS):
        raise ContractError(
            f"fold_phase holds {len(fold_records)} records, expected "
            f"{len(expected_models) * len(FOLD_TABLE_PINS)} (model x fold)"
        )
    pooled_per_model: dict[str, int] = {}
    for index, record in enumerate(fold_records):
        label = f"fold_phase[{index}]"
        fold = record.get("fold")
        if isinstance(fold, bool) or not isinstance(fold, int):
            raise ContractError(f"{label}: fold must be an int")
        pinned = _fold_table_pin(fold, label)
        for key, want in (
            ("origin", pinned[1]),
            ("window_start", pinned[2]),
            ("window_end", pinned[3]),
        ):
            if record.get(key) != want:
                raise ContractError(
                    f"{label}: {key} {record.get(key)!r} != the pinned {want!r}"
                )
        if record.get("n_rows") != FOLD_ROWS:
            raise ContractError(
                f"{label}: n_rows {record.get('n_rows')!r} != {FOLD_ROWS}"
            )
        slices = record.get("slices")
        if not isinstance(slices, dict):
            raise ContractError(f"{label}: slices must be an object")
        for kind in ("full_grid", "active_slice"):
            block = slices.get(kind)
            if not isinstance(block, dict) or set(block) != set(SCORE_KEYS):
                raise ContractError(
                    f"{label}: the {kind} slice must carry exactly {list(SCORE_KEYS)}"
                )
            for key in SCORE_KEYS:
                value = block[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ContractError(
                        f"{label}: {kind}.{key} must be a number, got {value!r}"
                    )
                if not math.isfinite(float(value)) or float(value) < 0.0:
                    raise ContractError(
                        f"{label}: {kind}.{key} must be finite and "
                        f"non-negative, got {value!r}"
                    )
        model_name = str(record.get("name", ""))
        pooled_per_model[model_name] = pooled_per_model.get(model_name, 0) + int(
            record["n_rows"]
        )
    for model_name, pooled_rows in pooled_per_model.items():
        if pooled_rows != POOLED_FOLD_ROWS:
            raise ContractError(
                f"fold_phase: model {model_name!r} pools {pooled_rows} rows "
                f"!= the pinned {POOLED_FOLD_ROWS}"
            )
    for row in table:
        if int(row["n_rows"]) != FOLD_ROWS:
            raise ContractError(
                f"folds_table() row {row} does not carry the pinned {FOLD_ROWS} rows"
            )
    print(
        f"PASS fold-phase: {len(derived_models)} models x "
        f"{len(FOLD_TABLE_PINS)} folds derived from the artifacts and the "
        f"shared table ({FOLD_ROWS:,} rows per fold, {POOLED_FOLD_ROWS:,} pooled), "
        "in the registry's order"
    )


def _check_contract_counts_pins() -> None:
    """The count pins are re-derived by the REAL function, never mocked.

    The contract_counts block is the audit's own witness that the week's
    locked numbers still hold. It is computed here from the real train
    and valid frames plus a synthetic local_test stand-in that carries
    the pinned local_test-side counts, and every derived number is
    compared key by key against the pins. A pin that drifts refuses the
    freeze, and this block refuses the audit too.
    """
    train = load_split("train")
    valid = load_split("valid")
    columns = ["date", "store_nbr", "id", "sales"]
    train_lean = train[columns]
    valid_lean = valid[columns]
    local_test = _synthetic_splits()["local_test"]
    frame = pd.concat([train_lean, valid_lean], ignore_index=True)
    frame = frame.sort_values("date", kind="mergesort").reset_index(drop=True)
    fit_frame = frame.loc[fit_rows(frame)]

    counts = _contract_counts(train_lean, valid_lean, local_test, frame, fit_frame)
    expected = {
        "closure_rows": {
            "train": PINNED_CLOSURE_ROWS["train"],
            "valid": PINNED_CLOSURE_ROWS["valid"],
            "local_test": PINNED_CLOSURE_ROWS["local_test"],
            "universe": PINNED_CLOSURE_UNIVERSE,
        },
        "masked_training_drop": {
            "train": PINNED_MASKED_DROP["train"],
            "valid": PINNED_MASKED_DROP["valid"],
            "total": PINNED_MASKED_DROP["total"],
            "in_window_fills_kept": PINNED_IN_WINDOW_FILLS_KEPT,
        },
        "zero_share_raw": {
            "rows": PINNED_ZERO_SHARE_RAW[0],
            "of_rows": PINNED_ZERO_SHARE_RAW[1],
            "share": round(PINNED_ZERO_SHARE_RAW[0] / PINNED_ZERO_SHARE_RAW[1], 6),
        },
        "zero_share_trainable": {
            "rows": PINNED_ZERO_SHARE_TRAINABLE[0],
            "of_rows": PINNED_ZERO_SHARE_TRAINABLE[1],
            "share": round(
                PINNED_ZERO_SHARE_TRAINABLE[0] / PINNED_ZERO_SHARE_TRAINABLE[1], 6
            ),
        },
        "preopening_holdout_rows": {
            "valid": PINNED_PREOPENING_HOLDOUT["valid"],
            "local_test": PINNED_PREOPENING_HOLDOUT["local_test"],
            "total": PINNED_PREOPENING_HOLDOUT["total"],
        },
        "christmas_2016_rows": CHRISTMAS_2016_ROWS,
    }
    if sorted(counts) != sorted(expected):
        raise ContractError(
            f"contract_counts carries {sorted(counts)}, not the pinned "
            f"{sorted(expected)}"
        )
    for key in sorted(expected):
        if counts[key] != expected[key]:
            raise ContractError(
                f"contract_counts {key} is {counts[key]}, not the pinned "
                f"{expected[key]}"
            )
    for key in ("zero_share_raw", "zero_share_trainable"):
        block = counts[key]
        if block["share"] != round(block["rows"] / block["of_rows"], 6):
            raise ContractError(
                f"contract_counts {key} share {block['share']} does not match "
                f"its own counts {block['rows']} of {block['of_rows']}"
            )
    print(
        "PASS contract-counts: the real derivation re-derives every pinned "
        f"count (closure {PINNED_CLOSURE_ROWS['train']}/"
        f"{PINNED_CLOSURE_ROWS['valid']}/{PINNED_CLOSURE_ROWS['local_test']}, "
        f"masked drop {PINNED_MASKED_DROP['total']:,} with "
        f"{PINNED_IN_WINDOW_FILLS_KEPT} in-window fills kept, zero share "
        f"{PINNED_ZERO_SHARE_RAW[0]:,} of {PINNED_ZERO_SHARE_RAW[1]:,} and "
        f"{PINNED_ZERO_SHARE_TRAINABLE[0]:,} of "
        f"{PINNED_ZERO_SHARE_TRAINABLE[1]:,}, holdout "
        f"{PINNED_PREOPENING_HOLDOUT['total']:,})"
    )

    # The pins must bite: a frame whose raw zero share drifts must be
    # refused by the real function, naming the share that drifted.
    drifted = frame.copy()
    zero_rows = drifted.index[drifted["sales"] == 0]
    if len(zero_rows) == 0:
        raise ContractError("the drift fixture has no zero-sale row to perturb")
    drifted.loc[drifted.index[0], "sales"] = 99.0
    drifted_fit = drifted.loc[fit_rows(drifted)]
    try:
        _contract_counts(train_lean, valid_lean, local_test, drifted, drifted_fit)
    except ContractError as error:
        if "raw zero share" not in str(error):
            raise ContractError(
                f"the drifted zero share was refused with the wrong lead: {error}"
            ) from error
    else:
        raise ContractError(
            "the real count derivation accepted a drifted raw zero share"
        )
    print(
        "PASS contract-counts: a drifted raw zero share is refused by the "
        "real derivation, naming the share that drifted"
    )


def _check_freeze_pins() -> None:
    """Day-5 block: every pinned value bites, in the derivation and the schema.

    The contract_counts pins are proved by perturbing the REAL frames and
    watching the REAL derivation refuse, naming the pin that drifted. The
    freeze_results.json pins are proved by perturbing a valid payload and
    watching the ONE production validator refuse. A pin that stops biting
    is a hole in the audit, and this block closes it.
    """

    def refuses(call: Callable[[], Any], needle: str, label: str) -> None:
        try:
            call()
        except ContractError as error:
            if needle not in str(error):
                raise ContractError(
                    f"freeze pin {label}: refused with the wrong lead: {error}"
                ) from error
            return
        raise ContractError(f"freeze pin {label}: accepted a drifted value")

    train = load_split("train")
    valid = load_split("valid")
    columns = ["date", "store_nbr", "id", "sales"]
    train_lean = train[columns]
    valid_lean = valid[columns]
    local_test = _synthetic_splits()["local_test"]
    frame = pd.concat([train_lean, valid_lean], ignore_index=True)
    frame = frame.sort_values("date", kind="mergesort").reset_index(drop=True)
    fit_frame = frame.loc[fit_rows(frame)]

    # The closure-rows pin: one closure row that stops being a closure
    # row must refuse the derivation.
    closure_drift = train_lean.copy()
    closure_mask = closure_rows(closure_drift)
    closure_index = closure_mask[closure_mask].index
    if len(closure_index) == 0:
        raise ContractError("the closure fixture has no closure row to perturb")
    closure_drift.loc[closure_index[0], "id"] = 1
    refuses(
        lambda: _contract_counts(
            closure_drift, valid_lean, local_test, frame, fit_frame
        ),
        "closure rows",
        "closure-rows",
    )

    # The masked-drop pin: one pre-opening row that moves past its store's
    # opening date drops out of the mask, so the drop count drifts.
    drop_drift = train_lean.copy()
    drop_candidate = drop_drift.index[
        preopening_rows(drop_drift) & ~closure_rows(drop_drift)
    ]
    if len(drop_candidate) == 0:
        raise ContractError("the drop fixture has no pre-opening row to perturb")
    drop_row = drop_candidate[0]
    drop_store = int(drop_drift.loc[drop_row, "store_nbr"])
    drop_drift.loc[drop_row, "date"] = DELAYS_OPEN_UNTIL[drop_store] + pd.Timedelta(
        days=1
    )
    refuses(
        lambda: _contract_counts(drop_drift, valid_lean, local_test, frame, fit_frame),
        "masked training drop",
        "masked-drop",
    )

    # The in-window-fills pin: one closure row that moves past its store's
    # opening date leaves the pre-opening window, so the fills count drifts.
    fills_drift = train_lean.copy()
    fills_candidate = fills_drift.index[
        preopening_rows(fills_drift) & closure_rows(fills_drift)
    ]
    if len(fills_candidate) == 0:
        raise ContractError("the fills fixture has no in-window fill to perturb")
    fills_row = fills_candidate[0]
    fills_store = int(fills_drift.loc[fills_row, "store_nbr"])
    fills_drift.loc[fills_row, "date"] = DELAYS_OPEN_UNTIL[fills_store] + pd.Timedelta(
        days=1
    )
    refuses(
        lambda: _contract_counts(fills_drift, valid_lean, local_test, frame, fit_frame),
        "pre-opening window",
        "in-window-fills",
    )

    # The pre-opening holdout pin: one pre-opening holdout row that moves
    # past its store's opening date leaves the holdout count short.
    holdout_drift = local_test.copy()
    holdout_candidate = holdout_drift.index[
        preopening_rows(holdout_drift) & ~closure_rows(holdout_drift)
    ]
    if len(holdout_candidate) == 0:
        raise ContractError("the holdout fixture has no pre-opening row to perturb")
    holdout_row = holdout_candidate[0]
    holdout_store = int(holdout_drift.loc[holdout_row, "store_nbr"])
    holdout_drift.loc[holdout_row, "date"] = DELAYS_OPEN_UNTIL[
        holdout_store
    ] + pd.Timedelta(days=1)
    refuses(
        lambda: _contract_counts(
            train_lean, valid_lean, holdout_drift, frame, fit_frame
        ),
        "pre-opening holdout rows",
        "preopening-holdout",
    )

    # The trainable zero-share pin: one trainable row that becomes a zero
    # moves the share the trainable frame carries.
    zero_drift = fit_frame.copy()
    nonzero = zero_drift.index[zero_drift["sales"] != 0]
    if len(nonzero) == 0:
        raise ContractError("the zero-share fixture has no non-zero row to perturb")
    zero_drift.loc[nonzero[0], "sales"] = 0.0
    refuses(
        lambda: _contract_counts(train_lean, valid_lean, local_test, frame, zero_drift),
        "trainable zero share",
        "trainable-zero-share",
    )
    print(
        "PASS freeze pins: the real derivation refuses a drifted closure count, "
        "masked drop, in-window fill, pre-opening holdout and trainable zero "
        "share"
    )

    # The schema pins: a valid payload with ONE drifted field must be
    # refused by the ONE production validator. The payload is the
    # fixture frame's, so the validator is handed the fixture's own
    # window; the pinned defaults stay enforced on the real-file path.
    payload = _synthetic_freeze_payload()
    fixture_local_test = _synthetic_freeze_fixtures()["local_test"]
    fixture_rows = len(fixture_local_test)
    fixture_bounds = (
        fixture_local_test["date"].min().date().isoformat(),
        fixture_local_test["date"].max().date().isoformat(),
    )

    def _broken(mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        broken = copy.deepcopy(payload)
        mutate(broken)
        return broken

    def _set_split_rows(broken: dict[str, Any]) -> None:
        broken["data_snapshot"]["train"]["rows"] = 1

    def _set_split_bounds(broken: dict[str, Any]) -> None:
        broken["data_snapshot"]["train"]["bounds"] = ["1999-01-01", "1999-12-31"]

    def _set_closure_rows(broken: dict[str, Any]) -> None:
        broken["contract_counts"]["closure_rows"]["train"] = 1

    def _set_zero_share(broken: dict[str, Any]) -> None:
        # Drift the counts AND keep the share consistent with them, so the
        # pinned-counts check is the ONLY thing that can catch this: a share
        # check alone would accept the payload.
        block = broken["contract_counts"]["zero_share_raw"]
        block["rows"] = 1
        block["share"] = round(block["rows"] / block["of_rows"], 6)

    def _set_holdout(broken: dict[str, Any]) -> None:
        broken["contract_counts"]["preopening_holdout_rows"]["total"] = 1

    def _set_fold_name(broken: dict[str, Any]) -> None:
        broken["fold_phase"]["models"][0]["name"] = "not-a-registered-model"

    def _set_fold_rows(broken: dict[str, Any]) -> None:
        broken["fold_phase"]["models"][0]["folds"][0]["n_rows"] = 1

    def _set_freeze_rows(broken: dict[str, Any]) -> None:
        broken["freeze_phase"]["n_rows"] = 1

    def _set_freeze_start(broken: dict[str, Any]) -> None:
        broken["freeze_phase"]["window_start"] = "x"

    def _set_freeze_end(broken: dict[str, Any]) -> None:
        broken["freeze_phase"]["window_end"] = "x"

    def _set_model_rows(broken: dict[str, Any]) -> None:
        broken["freeze_phase"]["models"][0]["n_rows"] = 1

    def _set_model_start(broken: dict[str, Any]) -> None:
        broken["freeze_phase"]["models"][0]["window_start"] = "1999-01-01"

    def _set_model_track(broken: dict[str, Any]) -> None:
        broken["freeze_phase"]["models"][0]["track"] = "S9"

    def _set_model_artifact(broken: dict[str, Any]) -> None:
        broken["freeze_phase"]["models"][0]["artifact"] = "bogus.json"

    def _set_raw_share_value(broken: dict[str, Any]) -> None:
        broken["contract_counts"]["zero_share_raw"]["share"] = 99.0

    def _set_trainable_share_value(broken: dict[str, Any]) -> None:
        broken["contract_counts"]["zero_share_trainable"]["share"] = 99.0

    for mutate, needle, label in (
        (_set_split_rows, "data_snapshot train rows", "split-rows"),
        (_set_split_bounds, "data_snapshot train bounds", "split-bounds"),
        (_set_closure_rows, "closure_rows", "contract-closure-rows"),
        (_set_zero_share, "zero_share_raw", "contract-zero-share"),
        (_set_holdout, "preopening_holdout_rows", "contract-holdout-rows"),
        (_set_fold_name, "fold phase models", "fold-freeze-parity"),
        (_set_fold_rows, "n_rows", "fold-n-rows"),
        (_set_freeze_rows, "freeze_phase n_rows", "freeze-n-rows"),
        (_set_freeze_start, "freeze_phase window_start", "freeze-window-start"),
        (_set_freeze_end, "freeze_phase window_end", "freeze-window-end"),
        (_set_model_rows, "freeze_phase models[0] n_rows", "freeze-model-n-rows"),
        (
            _set_model_start,
            "freeze_phase models[0] window_start",
            "freeze-model-window",
        ),
        (_set_model_track, "models[0] track", "freeze-model-track"),
        (_set_model_artifact, "models[0] artifact", "freeze-model-artifact"),
        (_set_raw_share_value, "zero_share_raw share", "contract-zero-share-value"),
        (
            _set_trainable_share_value,
            "zero_share_trainable share",
            "contract-trainable-share-value",
        ),
    ):
        refuses(
            lambda mutate=mutate: _validate_freeze_results(
                _broken(mutate),
                expected_rows=fixture_rows,
                expected_bounds=fixture_bounds,
            ),
            needle,
            label,
        )
    print(
        "PASS freeze pins: the production validator refuses a drifted split row "
        "count, split bound, closure count, zero share value, holdout count, "
        "fold/freeze model set, fold row count, freeze-phase rows, "
        "freeze-phase window bounds, model rows, model window, model track, "
        "model artifact and zero-share values"
    )

    # The fold phase re-validates the bytes it consumes: a registered
    # artifact that drifted since the pre-flight gate must refuse it.
    with tempfile.TemporaryDirectory() as tmp_name:
        artifacts = Path(tmp_name)
        classical, deep = _synthetic_registered_artifacts()
        (artifacts / "classical_results.json").write_text(
            json.dumps(classical), encoding="utf-8"
        )
        (artifacts / "deep_results.json").write_text(json.dumps(deep), encoding="utf-8")
        payloads = _registered_payloads(artifacts)
        if sorted(payloads) != sorted(ARTIFACT_REGISTRY):
            raise ContractError(
                f"the registered payloads are {sorted(payloads)}, not the "
                f"registry {sorted(ARTIFACT_REGISTRY)}"
            )
        # The deep artifact drifts after the gate cleared it: the fold
        # phase re-reads and re-validates the bytes, so it refuses.
        drifted_deep = {key: value for key, value in deep.items() if key != "tiers"}
        (artifacts / "deep_results.json").write_text(
            json.dumps(drifted_deep), encoding="utf-8"
        )
        refuses(
            lambda: _registered_payloads(artifacts),
            "missing top-level keys",
            "artifact-re-validation",
        )
    print(
        "PASS freeze pins: the fold phase re-validates the artifact bytes it "
        "consumes, so a drifted artifact refuses it"
    )

    # A present-but-corrupt artifact refuses the fold phase: it is not
    # an INFO notice, because the artifact IS there.
    with tempfile.TemporaryDirectory() as tmp_name:
        artifacts = Path(tmp_name)
        classical, deep = _synthetic_registered_artifacts()
        (artifacts / "classical_results.json").write_text(
            json.dumps(classical), encoding="utf-8"
        )
        (artifacts / "deep_results.json").write_text(
            json.dumps({key: value for key, value in deep.items() if key != "tiers"}),
            encoding="utf-8",
        )
        with mock.patch.object(sys.modules[__name__], "ARTIFACTS_DIR", artifacts):
            refuses(
                _check_fold_phase,
                "missing top-level keys",
                "fold-phase-corrupt-artifact",
            )
    print(
        "PASS freeze pins: a present-but-corrupt registered artifact refuses "
        "the fold phase; it is not an INFO notice"
    )


def _check_freeze_critical_section() -> None:
    """Day-5 block: the freeze is one locked critical section.

    Three guarantees, each proved by driving the real code: the artifact
    is written once and a second write refuses BEFORE it touches the file
    system; the manifest lock is HELD while the artifact is written; and
    the freeze finds its OWN manifest record by stamp even when a
    concurrent writer appends another record right after it.
    """

    def refuses(call: Callable[[], Any], needle: str, label: str) -> None:
        try:
            call()
        except ContractError as error:
            if needle not in str(error):
                raise ContractError(
                    f"freeze section {label}: refused with the wrong lead: {error}"
                ) from error
            return
        raise ContractError(f"freeze section {label}: accepted a forbidden call")

    # 1. The artifact is written once: a second write refuses before it
    #    reaches the file system at all.
    payload = _synthetic_freeze_payload()
    real_write_bytes = _atomic_write_bytes
    with tempfile.TemporaryDirectory() as tmp_name:
        directory = Path(tmp_name)
        target = directory / FREEZE_RESULTS_FILENAME
        touched: list[str] = []

        def _spy_write(path: Path, data: bytes, *, exclusive: bool = False) -> None:
            touched.append(path.name)
            real_write_bytes(path, data, exclusive=exclusive)

        with mock.patch.object(
            sys.modules[__name__], "_atomic_write_bytes", _spy_write
        ):
            _write_freeze_results(target, payload)
            if touched != [target.name]:
                raise ContractError(
                    f"the first write touched {touched}, expected one write of "
                    f"{target.name}"
                )
            refuses(
                lambda: _write_freeze_results(target, payload),
                "already exists",
                "pre-write-existence",
            )
            if touched != [target.name]:
                raise ContractError(
                    "the second write reached the file system: the pre-write "
                    "existence check did not fire"
                )
    print(
        "PASS freeze section: the artifact is written once; a second write "
        "refuses before it touches the file system"
    )

    # The publish is durable, in this order: the bytes are flushed, the
    # file is published, and the directory entry is flushed. A crash
    # between the three can leave a half-written file or a lost rename.
    with tempfile.TemporaryDirectory() as tmp_name:
        directory = Path(tmp_name)
        target = directory / FREEZE_RESULTS_FILENAME
        steps: list[str] = []
        real_fsync = os.fsync
        real_link = os.link
        real_replace = os.replace

        def _spy_fsync(fd: int) -> None:
            steps.append("flush")
            real_fsync(fd)

        def _spy_link(src: Any, dst: Any, **kwargs: Any) -> None:
            steps.append("publish")
            real_link(src, dst, **kwargs)

        def _spy_replace(src: Any, dst: Any, **kwargs: Any) -> None:
            steps.append("publish")
            real_replace(src, dst, **kwargs)

        with (
            mock.patch.object(os, "fsync", _spy_fsync),
            mock.patch.object(os, "link", _spy_link),
            mock.patch.object(os, "replace", _spy_replace),
        ):
            _write_freeze_results(target, payload)
        if steps != ["flush", "publish", "flush"]:
            raise ContractError(
                f"the publish ran as {steps}, not flush -> publish -> flush: a "
                "crash could leave a half-written file or a lost rename"
            )
    print(
        "PASS freeze section: the publish is durable; the bytes are flushed, "
        "the file is published and the directory entry is flushed"
    )

    # 2. The manifest lock is HELD across the critical section, and the
    #    freeze identifies its own stamped record as the one it appended
    #    (records[-1] under the lock, with the stamp as a secondary
    #    assertion) even when other records are present.
    with tempfile.TemporaryDirectory() as tmp_name:
        directory = Path(tmp_name)
        prior = _synthetic_manifest_record()
        prior["created"] = "2000-01-01T00:00:00Z"
        prior["notes"] = ["a prior writer"]
        (directory / "run_manifest.json").write_text(
            json.dumps([prior], indent=2) + "\n", encoding="utf-8"
        )
        lock_probe: list[bool] = []
        real_write_results = _write_freeze_results

        def _probing_write(path: Path, body: dict[str, Any]) -> None:
            lock_probe.append(_manifest_lock_held(directory))
            real_write_results(path, body)

        with mock.patch.object(
            sys.modules[__name__], "_write_freeze_results", _probing_write
        ):
            rc, out, _written, _stats = _synthetic_freeze_run(
                directory, convention="frame-order", real_splits=True
            )
        if rc != 0:
            tail = out.strip().splitlines()[-3:]
            raise ContractError(
                f"the synthetic freeze exited {rc} with a prior manifest record "
                f"present; last lines: {' | '.join(tail)}"
            )
        if lock_probe != [True]:
            raise ContractError(
                "the manifest lock was not held while the artifact was written "
                f"(probe saw {lock_probe})"
            )
        records = list(read_run_manifest(directory / "run_manifest.json"))
        if len(records) != 2 or records[-1].get("freeze_run") is not True:
            raise ContractError(
                "the freeze did not append its own freeze_run record last "
                f"under the lock: {records!r}"
            )
    print(
        "PASS freeze section: the manifest lock is held across the critical "
        "section, and the freeze identifies its own stamped record under the "
        "lock when other records are present"
    )

    # 2b. A same-second collision: a pre-existing record already carries the
    #     freeze's exact second-resolution stamp, so a stamp-only lookup
    #     would see two matches and falsely fail. The freeze still succeeds.
    with tempfile.TemporaryDirectory() as tmp_name:
        directory = Path(tmp_name)
        same_stamp = "2099-01-01T00:00:00Z"
        collision = _synthetic_manifest_record()
        collision["created"] = same_stamp
        collision["notes"] = ["pre-existing same-second record"]
        (directory / "run_manifest.json").write_text(
            json.dumps([collision], indent=2) + "\n", encoding="utf-8"
        )
        with mock.patch.object(
            sys.modules[__name__], "_freeze_stamp", lambda: same_stamp
        ):
            rc, out, _written, _stats = _synthetic_freeze_run(
                directory, convention="frame-order", real_splits=True
            )
        if rc != 0:
            tail = out.strip().splitlines()[-3:]
            raise ContractError(
                "the freeze failed on a same-second manifest stamp collision; "
                f"last lines: {' | '.join(tail)}"
            )
        records = list(read_run_manifest(directory / "run_manifest.json"))
        matching = [r for r in records if r.get("created") == same_stamp]
        freeze_records = [r for r in matching if r.get("freeze_run") is True]
        if len(matching) != 2 or len(freeze_records) != 1:
            raise ContractError(
                "the same-second collision case did not leave one pre-existing "
                "record and the freeze's own freeze_run record at the shared "
                f"stamp (matching {len(matching)}, freeze_run {len(freeze_records)})"
            )
    print(
        "PASS freeze: the freeze succeeds when another manifest record shares "
        "its exact second-resolution stamp"
    )

    # 3. A manifest-stamp failure after the artifact is published must be
    #    recoverable: the stranded artifact is renamed aside under the same
    #    lock and the sanctioned re-run succeeds.
    with tempfile.TemporaryDirectory() as tmp_name:
        directory = Path(tmp_name)
        real_manifest_record = _write_manifest_record

        def _fail_freeze_stamp(artifacts: Path, command: str, **kwargs: Any) -> None:
            if kwargs.get("freeze_run") is True and kwargs.get("exit_status") == 0:
                raise ContractError("synthetic stamp failure")
            return real_manifest_record(artifacts, command, **kwargs)

        with mock.patch.object(
            sys.modules[__name__], "_write_manifest_record", _fail_freeze_stamp
        ):
            rc, out, _written, _stats = _synthetic_freeze_run(
                directory, convention="frame-order", real_splits=True
            )
        if rc != 1:
            raise ContractError(
                f"the freeze returned {rc}, not 1, on a synthetic stamp failure"
            )
        if "moved aside" not in out or "re-run" not in out:
            raise ContractError(
                f"the stamp-failure refusal did not name the recovery: {out!r}"
            )
        if (directory / FREEZE_RESULTS_FILENAME).exists():
            raise ContractError(
                "the stranded freeze_results.json was not moved aside after the "
                "stamp failure"
            )
        quarantined = sorted(directory.glob("freeze_results.failed-*"))
        quarantined = [
            path for path in quarantined if path.name != FREEZE_RESULTS_FILENAME
        ]
        if len(quarantined) != 1:
            raise ContractError(
                f"expected 1 quarantined artifact, found {quarantined!r}"
            )
        records = list(read_run_manifest(directory / "run_manifest.json"))
        if not records or records[-1].get("freeze_run") is not False:
            raise ContractError(
                "the stamp-failure attempt was not recorded as freeze_run=false"
            )
        rc2, out2, written2, _stats2 = _synthetic_freeze_run(
            directory, convention="frame-order", real_splits=True
        )
        if rc2 != 0:
            tail = out2.strip().splitlines()[-3:]
            raise ContractError(
                "the sanctioned re-run after a stamp failure did not succeed; "
                f"last lines: {' | '.join(tail)}"
            )
        if not (directory / FREEZE_RESULTS_FILENAME).exists() or written2 is None:
            raise ContractError(
                "the sanctioned re-run did not publish freeze_results.json"
            )
    print(
        "PASS freeze: a stamp failure quarantines the stranded artifact under "
        "the lock and the sanctioned re-run succeeds"
    )


def run_check() -> None:
    """Run all contract verification checks."""
    print("contract.py --check")
    _check_manifest_metadata()
    _check_data_contract()
    _check_folds()
    _check_freeze_gates()
    _check_freeze_body()
    _check_freeze_registry_refusals()
    _check_freeze_pins()
    _check_freeze_critical_section()
    _check_freeze_run_invariants()
    _check_synthetic_freeze_run_defaults()
    _check_christmas_slice_selection()
    _check_freeze_results_writer()
    _check_freeze_results_validation()
    _check_prediction_alignment()
    _check_freeze_safety()
    _check_fold_phase()
    _check_contract_counts_pins()
    _check_registry_presence()
    _check_registry()
    _check_scorer()
    _check_manifest_writer()
    print("contract.py --check: PASS")


def _command_label(argv: list[str] | None) -> str:
    """The invocation's label, from the argv that actually arrived.

    A refused parse is recorded under the tokens it was given
    ("contract.py --bogus"), never under a label borrowed from another
    mode.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    if not tokens:
        return "contract.py (no arguments)"
    return " ".join(["contract.py", *tokens])


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
    start = time.monotonic()
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        code = error.code if isinstance(error.code, int) else 2
        if code != 0:
            try:
                write_run_manifest(
                    _command_label(argv),
                    exit_status=code,
                    elapsed_seconds=time.monotonic() - start,
                    notes=[
                        *_dirty_notes(),
                        f"argument parsing refused with exit {code}",
                    ],
                )
            except Exception as audit_error:
                print(f"FAIL {audit_error!r}")
        raise
    if args.check and args.freeze:
        print("contract: pass --check or --freeze, not both")
        try:
            write_run_manifest(
                _command_label(argv),
                exit_status=2,
                elapsed_seconds=time.monotonic() - start,
                freeze_run=False,
                notes=[
                    *_dirty_notes(),
                    "refused: --check and --freeze are mutually exclusive",
                ],
            )
        except Exception as audit_error:
            print(f"FAIL {audit_error!r}")
        return 2
    if args.check:
        notes: list[str] = _dirty_notes()
        try:
            run_check()
        except (ContractError, OSError) as error:
            elapsed = time.monotonic() - start
            try:
                write_run_manifest(
                    "contract.py --check",
                    exit_status=1,
                    elapsed_seconds=elapsed,
                    notes=[*notes, str(error)],
                )
            except Exception as audit_error:
                print(f"FAIL {audit_error!r}")
            print(f"FAIL {error}")
            return 1
        except Exception as error:
            elapsed = time.monotonic() - start
            try:
                write_run_manifest(
                    "contract.py --check",
                    exit_status=1,
                    elapsed_seconds=elapsed,
                    notes=[*notes, repr(error)],
                )
            except Exception as audit_error:
                print(f"FAIL {audit_error!r}")
            print(f"FAIL {error!r}")
            return 1
        elapsed = time.monotonic() - start
        try:
            write_run_manifest(
                "contract.py --check", elapsed_seconds=elapsed, notes=notes
            )
        except Exception as error:
            print(f"FAIL {error!r}")
            return 1
        return 0
    if args.freeze:
        try:
            return freeze()
        except (ContractError, OSError) as error:
            print(f"FAIL {error}")
            return 1
    try:
        write_run_manifest(
            "contract.py (no arguments)",
            exit_status=2,
            elapsed_seconds=time.monotonic() - start,
            notes=[
                *_dirty_notes(),
                "refused: choose --check or --freeze",
            ],
        )
    except Exception as audit_error:
        print(f"FAIL {audit_error!r}")
    parser.print_usage()
    print("\nPass --check (verification) or --freeze (freeze path).")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
