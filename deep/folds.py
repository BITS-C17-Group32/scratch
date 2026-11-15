"""The shared rolling-origin fold protocol (S4 sheet section 10).

One implementation of the three tail windows of valid: each window is
predicted from a history that stops at its origin, so nothing reaches from
the future, and each window is scored against the valid rows inside it
(1,782 series x 16 days = 28,512 rows; 85,536 pooled over the three).

The row counts are asserted, not reported: a series that cannot be scored
is a loud failure here, never a smaller fold that nobody notices.
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from deep.artifact import FOLD_TABLE as ARTIFACT_FOLD_TABLE
from deep.data import NUM_NUMERIC, build_series_panels
from deep.metrics import compute_metrics
from deep.model import DeepForecaster

# fold, origin, window_start, window_end. The table lives in
# deep.artifact, next to the schema that pins the fold row counts, so the
# two cannot drift apart.
FOLD_TABLE = ARTIFACT_FOLD_TABLE
FOLD_ROWS = 28512
POOLED_FOLD_ROWS = 85536


class FoldError(RuntimeError):
    """The fold protocol could not be honoured."""


def resolve_device(pref: str = "auto") -> torch.device:
    """The device the fold re-prediction runs on."""
    if pref != "auto":
        if pref == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("cuda requested but unavailable")
        if pref == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("mps requested but unavailable")
        return torch.device(pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(ckpt_path: Path, device: torch.device) -> dict[str, Any]:
    """Rebuild the model from a checkpoint, on the device the caller names.

    The checkpoint is loaded from the CPU with weights_only=True, so no
    pickled object is ever executed; the model then moves to the device.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    cat_maps = ckpt.get("category_maps") or ckpt.get("cat_maps")
    if not cat_maps:
        raise FoldError(f"{ckpt_path} carries no category maps")
    scaler = ckpt.get("numeric_scaler") or ckpt.get("scaler")
    if not scaler:
        raise FoldError(f"{ckpt_path} carries no numeric scaler")
    cfg = ckpt.get("config") or {}
    arch = str(ckpt.get("arch", "lstm"))
    if arch.startswith("deep-"):
        arch = arch.replace("deep-", "")
    lookback = int(cfg.get("lookback_days", cfg.get("lookback", 60)))
    horizon = int(cfg.get("horizon_days", cfg.get("horizon", 16)))
    model = DeepForecaster(
        arch=arch,
        num_numeric=NUM_NUMERIC,
        cat_levels={column: len(levels) for column, levels in cat_maps.items()},
        embedding_dims=cfg.get("embedding_dims"),
        hidden_size=int(cfg.get("hidden_size", cfg.get("hidden_dim", 128))),
        num_layers=int(cfg.get("num_layers", 2)),
        dropout=float(cfg.get("dropout", 0.2)),
        horizon=horizon,
        lookback=lookback,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return {
        "model": model,
        "cat_maps": cat_maps,
        "scaler": scaler,
        "lookback": lookback,
        "horizon": horizon,
        "arch": arch,
        "config": cfg,
    }


def _window_truth(
    valid_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> dict[tuple[Any, str], np.ndarray]:
    """The valid rows inside one window, per series, in date order."""
    window = valid_df[
        (valid_df["date"] >= start) & (valid_df["date"] <= end)
    ].sort_values(["store_nbr", "family", "date"], kind="mergesort")
    truth: dict[tuple[Any, str], np.ndarray] = {}
    for (store, family), group in window.groupby(["store_nbr", "family"], sort=False):
        truth[(store, str(family))] = group["sales"].to_numpy(dtype=np.float64)
    return truth


def fold_metrics_for_checkpoint(
    ckpt_path: Path,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    *,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any], int, float]:
    """The three fold score blocks for one checkpoint.

    Returns (fold_metrics, pooled_metrics, pooled_rows, inference_seconds).
    The history is concat(train, valid[date <= origin]) -- the panel's
    last row IS the origin, which is asserted per series -- so the model
    input can never reach from the future. The truth is the valid rows
    inside the window, so each fold scores exactly the pinned 28,512 rows.
    """
    loaded = load_model(Path(ckpt_path), device)
    model = loaded["model"]
    lookback = loaded["lookback"]
    horizon = loaded["horizon"]
    cat_maps = loaded["cat_maps"]
    scaler = loaded["scaler"]

    fold_blocks: list[dict[str, Any]] = []
    pooled_true: list[np.ndarray] = []
    pooled_pred: list[np.ndarray] = []
    inference_seconds: list[float] = []

    for fold, origin_text, start_text, end_text in FOLD_TABLE:
        origin = pd.Timestamp(origin_text)
        start = pd.Timestamp(start_text)
        end = pd.Timestamp(end_text)
        window_days = (end - start).days + 1
        history = pd.concat(
            [train_df, valid_df[valid_df["date"] <= origin]],
            ignore_index=True,
        )
        panels = build_series_panels(history, cat_maps, scaler, lookback, horizon)
        truth = _window_truth(valid_df, start, end)

        features: list[np.ndarray] = []
        categories: list[np.ndarray] = []
        truths: list[np.ndarray] = []
        for panel in panels:
            dates = pd.to_datetime(panel["dates"])
            if len(dates) < lookback:
                raise FoldError(
                    f"fold {fold}: series {panel['store_nbr']}/"
                    f"{panel['family']} has {len(dates)} history rows, "
                    f"fewer than the {lookback}-day lookback"
                )
            if pd.Timestamp(dates[-1]) != origin:
                raise FoldError(
                    f"fold {fold}: series {panel['store_nbr']}/"
                    f"{panel['family']} ends at {pd.Timestamp(dates[-1]).date()}, "
                    f"not at the origin {origin.date()}"
                )
            key = (panel["store_nbr"], str(panel["family"]))
            if key not in truth:
                raise FoldError(
                    f"fold {fold}: series {key} has no valid rows inside the "
                    f"window {start.date()}..{end.date()}"
                )
            if len(truth[key]) != window_days:
                raise FoldError(
                    f"fold {fold}: series {key} has {len(truth[key])} valid rows "
                    f"inside the window, not the pinned {window_days}-day window"
                )
            features.append(panel["x"][-lookback:])
            categories.append(panel["cat"])
            truths.append(truth[key])

        if not panels:
            raise FoldError(f"fold {fold}: the history produced no series")
        if len(panels) != len(truth):
            raise FoldError(
                f"fold {fold}: {len(panels)} series have a history to predict "
                f"from but {len(truth)} series sit inside the window"
            )

        x_tensor = torch.from_numpy(np.ascontiguousarray(np.stack(features))).to(device)
        cat_tensor = torch.from_numpy(np.ascontiguousarray(np.stack(categories))).to(
            device
        )
        started = time.perf_counter()
        with torch.no_grad():
            predicted_log = model(x_tensor, cat_tensor).cpu().numpy()
        inference_seconds.append(time.perf_counter() - started)
        predicted = np.clip(np.expm1(predicted_log), 0, None)

        y_true = np.concatenate(truths)
        y_pred = predicted.reshape(-1)
        if len(y_true) != FOLD_ROWS:
            raise FoldError(
                f"fold {fold}: scored {len(y_true)} rows, not the pinned {FOLD_ROWS}"
            )
        if len(y_pred) != len(y_true):
            raise FoldError(
                f"fold {fold}: {len(y_pred)} predictions for {len(y_true)} rows"
            )
        metrics = compute_metrics(y_true, y_pred)
        block = {
            "rmsle": float(metrics["rmsle"]),
            "mae": float(metrics["mae"]),
            "rmse": float(metrics["rmse"]),
            "rmsle_active": float(metrics["rmsle_active"]),
            "mae_active": float(metrics["mae_active"]),
            "rmse_active": float(metrics["rmse_active"]),
            "n_rows": len(y_true),
            "n_rows_active": int(metrics["n_rows_active"]),
        }
        fold_blocks.append(block)
        pooled_true.append(y_true)
        pooled_pred.append(y_pred)

    pooled_rows = int(sum(len(chunk) for chunk in pooled_true))
    if pooled_rows != POOLED_FOLD_ROWS:
        raise FoldError(
            f"the three folds pooled {pooled_rows} rows, not the pinned "
            f"{POOLED_FOLD_ROWS}"
        )
    y_true = np.concatenate(pooled_true)
    y_pred = np.concatenate(pooled_pred)
    pooled = compute_metrics(y_true, y_pred)
    pooled_metrics = {
        "rmsle": float(pooled["rmsle"]),
        "mae": float(pooled["mae"]),
        "rmse": float(pooled["rmse"]),
        "rmsle_active": float(pooled["rmsle_active"]),
        "mae_active": float(pooled["mae_active"]),
        "rmse_active": float(pooled["rmse_active"]),
        "n_rows": pooled_rows,
        "n_rows_active": int(pooled["n_rows_active"]),
    }
    return (
        fold_blocks,
        pooled_metrics,
        pooled_rows,
        round(float(sum(inference_seconds)), 3),
    )
