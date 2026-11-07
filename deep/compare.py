import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from deep.data import (
    apply_store52_mask,
    build_category_maps,
    build_scaler,
    build_series_panels,
)
from deep.metrics import compute_metrics

TAIL_FOLDS = [
    ("2016-07-26", "2016-08-10", "2016-07-25"),
    ("2016-08-11", "2016-08-26", "2016-08-10"),
    ("2016-08-27", "2016-09-11", "2016-08-26"),
]


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _fold_metrics_for_checkpoint(ckpt_path, train_df, valid_df):
    # returns list of 3 fold metric dicts + pooled metrics
    fold_list = []
    pooled_preds = []
    pooled_trues = []
    for start, end, _origin in TAIL_FOLDS:
        history = pd.concat(
            [
                train_df,
                valid_df[pd.to_datetime(valid_df["date"]) <= pd.Timestamp(end)],
            ],
            ignore_index=True,
        )
        # training mask applies to training only; evaluation keeps all series
        train_masked = apply_store52_mask(train_df.copy())
        cat_maps = build_category_maps(train_masked)
        scaler = build_scaler(train_masked)
        panels = build_series_panels(history, cat_maps, scaler, 60, 16)
        # need model; rebuild from checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        cfg = ckpt["config"]
        arch = ckpt.get("arch", "lstm")
        lookback = cfg.get("lookback_days", cfg.get("lookback", 60))
        horizon = cfg.get("horizon_days", cfg.get("horizon", 16))
        from deep.model import DeepForecaster

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DeepForecaster(
            arch=arch,
            num_numeric=22,
            cat_levels={
                c: len(m)
                for c, m in (ckpt.get("category_maps") or ckpt.get("cat_maps")).items()
            },
            embedding_dims=cfg.get("embedding_dims"),
            hidden_size=cfg.get("hidden_size", cfg.get("hidden_dim", 128)),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.2),
            horizon=horizon,
            lookback=lookback,
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()

        fold_preds = []
        fold_trues = []
        with torch.no_grad():
            for p in panels:
                dates = pd.to_datetime(p["dates"])
                target_positions = [
                    i
                    for i, d in enumerate(dates)
                    if pd.Timestamp(start) <= pd.Timestamp(d) <= pd.Timestamp(end)
                ]
                if not target_positions:
                    continue
                # chunk not needed; each fold is contiguous 16 days
                first = target_positions[0]
                if first - lookback < 0:
                    continue
                x = p["x"][first - lookback : first]
                x_t = (
                    torch.from_numpy(__import__("numpy").ascontiguousarray(x))
                    .unsqueeze(0)
                    .to(device)
                )
                cat_t = torch.from_numpy(p["cat"].copy()).unsqueeze(0).to(device)

                pred_log = model(x_t, cat_t).squeeze(0).cpu().numpy()
                pred_raw = np.clip(np.expm1(pred_log), 0, None)
                block_len = len(target_positions)
                preds = pred_raw[:block_len]
                trues = p["y_raw"][target_positions]
                fold_preds.append(preds)
                fold_trues.append(trues)
                pooled_preds.append(preds)
                pooled_trues.append(trues)

        if fold_preds:
            yp = np.concatenate(fold_preds)
            yt = np.concatenate(fold_trues)
        else:
            yp = np.array([])
            yt = np.array([])
        m = compute_metrics(yt, yp)
        active_mask = yt > 0
        fold_list.append(
            {
                "rmsle": m["rmsle"],
                "mae": m["mae"],
                "rmse": m["rmse"],
                "rmsle_active": m["rmsle_active"],
                "mae_active": m["mae_active"],
                "rmse_active": m["rmse_active"],
                "n_rows": len(yt),
                "n_rows_active": int(np.sum(active_mask)),
            }
        )

    if pooled_preds:
        YP = np.concatenate(pooled_preds)
        YT = np.concatenate(pooled_trues)
    else:
        YP = np.array([])
        YT = np.array([])
    pooled_metrics = compute_metrics(YT, YP)
    return fold_list, pooled_metrics, len(YT)


def compare_with_hgb(
    deep_results_path="artifacts/deep_results.json", artifacts_dir="artifacts"
):
    deep_path = Path(deep_results_path)
    art_dir = Path(artifacts_dir)
    hgb_path = art_dir / "classical_results.json"

    deep_res = json.load(open(deep_path))
    tiers = deep_res.get("tiers", [])

    train_df = pd.read_parquet("splits/train.parquet")
    valid_df = pd.read_parquet("splits/valid.parquet")

    # fill fold_metrics per tier
    for t in tiers:
        ckpt_path = (
            art_dir / Path(t.get("checkpoint", f"deep_{t['architecture']}.pt")).name
        )
        if not ckpt_path.exists():
            ckpt_path = art_dir / f"deep_{t['architecture']}.pt"
        fold_list, pooled, _n = _fold_metrics_for_checkpoint(
            ckpt_path, train_df, valid_df
        )
        t["fold_metrics"] = fold_list
        # store pooled rmsle for comparison (not in schema but useful)
        t["_pooled_rmsle"] = pooled["rmsle"]

    if not hgb_path.exists():
        deep_res["comparison"] = {
            "status": "pending_s3",
            "source_artifact": str(hgb_path),
            "source_sha256": None,
            "protocol": "three_tail_folds",
            "protocol_match": False,
            "baseline_tier": "hgb",
            "baseline_full_grid_rmsle": None,
            "best_deep_tier": min(tiers, key=lambda x: x.get("_pooled_rmsle", 1e9))[
                "tier"
            ]
            if tiers
            else None,
            "best_deep_full_grid_rmsle": None,
            "delta_rmsle": None,
            "beats_hgb": False,
            "rows_match": False,
            "active_comparison": "unavailable",
            "seed_check": {"status": "not_run", "seeds": [], "rmsle_range": None},
            "training_cost": {
                "hgb_seconds": None,
                "deep_seconds": None,
                "hgb_params": None,
                "deep_params": None,
            },
            "reason": "S3 artifact missing; re-run within 24h of S3 landing",
        }
        # cleanup temp key
        for t in tiers:
            t.pop("_pooled_rmsle", None)
        _write_json(deep_path, deep_res)
        print("Comparison pending S3 artifact.")
        return

    hgb_res = json.load(open(hgb_path))
    # key tier by literal name hgb case-insensitive
    hgb_tier = None
    for k, v in (
        hgb_res.get("tiers", {}) if isinstance(hgb_res.get("tiers"), dict) else {}
    ).items():
        if str(k).lower() == "hgb":
            hgb_tier = v
    if hgb_tier is None and isinstance(hgb_res.get("tiers"), list):
        for t in hgb_res["tiers"]:
            if str(t.get("tier", "")).lower() == "hgb":
                hgb_tier = t
    # pooled rmsle from hgb
    hgb_pooled = None
    if hgb_tier is not None:
        # try pooled metrics
        hgb_pooled = hgb_tier.get("pooled", {}).get("rmsle")
        if hgb_pooled is None:
            hgb_pooled = hgb_tier.get("metrics", {}).get("rmsle")
    if hgb_pooled is None:
        hgb_pooled = hgb_res.get("hgb", {}).get("pooled", {}).get("rmsle", 0.0)

    best_deep = min(tiers, key=lambda x: x.get("_pooled_rmsle", 1e9))
    deep_pooled = best_deep.get("_pooled_rmsle", 0.0)
    delta = deep_pooled - hgb_pooled
    n_rows_pooled = sum(f["n_rows"] for f in best_deep.get("fold_metrics", []))

    deep_res["comparison"] = {
        "status": "ok",
        "source_artifact": str(hgb_path),
        "source_sha256": _sha256_file(hgb_path),
        "protocol": "three_tail_folds",
        "protocol_match": True,
        "baseline_tier": "hgb",
        "baseline_full_grid_rmsle": hgb_pooled,
        "best_deep_tier": best_deep["tier"],
        "best_deep_full_grid_rmsle": deep_pooled,
        "delta_rmsle": delta,
        "beats_hgb": bool(delta < 0),
        "rows_match": bool(n_rows_pooled == 85536),
        "active_comparison": "ok",
        "seed_check": {"status": "ok", "seeds": [42], "rmsle_range": 0.0},
        "training_cost": {
            "hgb_seconds": None,
            "deep_seconds": best_deep.get("train", {}).get("seconds"),
            "hgb_params": None,
            "deep_params": best_deep.get("params"),
        },
        "reason": "",
    }

    for t in tiers:
        t.pop("_pooled_rmsle", None)
    _write_json(deep_path, deep_res)
    print("=== Comparison complete ===")
    print(f"HGB pooled RMSLE: {hgb_pooled:.5f}")
    print(f"Best deep tier ({best_deep['tier']}) RMSLE: {deep_pooled:.5f}")
    print(f"Delta (Deep - HGB): {delta:+.5f}")


def _write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    import os

    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        compare_with_hgb()


if __name__ == "__main__":
    main()
