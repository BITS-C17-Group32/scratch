import argparse
import copy
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from deep.artifact import (
    config_sha256 as artifact_config_sha256,
    empty_ablation as artifact_empty_ablation,
    finalize as artifact_finalize,
    migrate_legacy as artifact_migrate_legacy,
    notes as artifact_notes,
    shared_config as artifact_shared_config,
    write_artifact as artifact_write_artifact,
)
from deep.benchmark import runtime_tradeoff
from deep.config import DeepConfig
from deep.data import (
    CAT_COLS,
    NUM_NUMERIC,
    TrainWindowDataset,
    _batched_eval,
    apply_store52_mask,
    build_category_maps,
    build_scaler,
    build_series_panels,
    load_split_manifest,
    store52_mask,
)
from deep.metrics import compute_metrics
from deep.model import DeepForecaster


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(pref="auto"):
    if pref != "auto":
        dev = torch.device(pref)
        if pref == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("cuda requested but unavailable")
        if pref == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("mps requested but unavailable")
        return dev
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def run_audit():
    repo_root = Path(".")
    manifest = load_split_manifest(repo_root)
    train_df = pd.read_parquet(repo_root / "splits" / "train.parquet")
    valid_df = pd.read_parquet(repo_root / "splits" / "valid.parquet")

    series_count = len(train_df.groupby(["store_nbr", "family"]))
    st52_masked = train_df[store52_mask(train_df)]
    train_2015_plus = train_df[
        pd.to_datetime(train_df["date"]) >= pd.Timestamp("2015-01-01")
    ]

    total_rows = manifest.get("total", 0)
    local_test_rows = manifest.get("rows", {}).get("local_test", 0)
    train_rows_m = manifest.get("rows", {}).get("train", 0)
    valid_rows_m = manifest.get("rows", {}).get("valid", 0)

    print("=== S4 Audit Summary ===")
    print(f"Manifest total rows: {total_rows}")
    print(f"Train {train_rows_m} | Valid {valid_rows_m} | LocalTest {local_test_rows}")
    print(f"Series count: {series_count}")
    print(f"Store-52 train masked rows dropped: {len(st52_masked)}")
    print(f"Train rows available after mask: {len(train_df) - len(st52_masked)}")
    print(f"Train rows from 2015-01-01 onward: {len(train_2015_plus)}")
    # train/valid dates
    pd.to_datetime(train_df["date"])
    pd.to_datetime(valid_df["date"])
    bounds = manifest.get("bounds", {})
    print("Train bounds:", bounds.get("train"))
    print("Valid bounds:", bounds.get("valid"))
    print("Local test bounds:", bounds.get("local_test"))
    sys.exit(0)


def train_one(
    arch, config, device, train_df=None, valid_df=None, cat_maps=None, scaler=None
):
    import time as _time

    cfg = config
    if train_df is None:
        train_df = pd.read_parquet(Path(".") / "splits" / "train.parquet")
    if valid_df is None:
        valid_df = pd.read_parquet(Path(".") / "splits" / "valid.parquet")
    t0 = _time.time()
    set_seed(cfg.seed)

    train_masked = apply_store52_mask(train_df.copy())
    # The train_start slice comes FIRST: window numbers, batch offsets
    # and the scaler are computed on the sliced frame (sheet Day 4), so
    # the 2015-plus ablation refits the scaler on the slice instead of
    # scaling the slice with full-history statistics.
    if cfg.train_start:
        train_masked = train_masked[
            pd.to_datetime(train_masked["date"]) >= pd.Timestamp(cfg.train_start)
        ].reset_index(drop=True)
    if cat_maps is None:
        cat_maps = build_category_maps(train_masked)
        scaler = build_scaler(train_masked)

    panels = build_series_panels(
        train_masked, cat_maps, scaler, cfg.lookback_days, cfg.horizon_days
    )
    train_ds = TrainWindowDataset(
        panels, cfg.lookback_days, cfg.horizon_days, cfg.train_window_stride
    )
    print(f"Train sequences: {len(train_ds):,}")

    # Validation uses rolling-origin evaluation on train+valid (299,376 rows)
    valid_panels = None
    score_dt = None
    if valid_df is not None:
        valid_frame = pd.concat([train_df.copy(), valid_df.copy()], ignore_index=True)
        valid_panels = build_series_panels(
            valid_frame, cat_maps, scaler, cfg.lookback_days, cfg.horizon_days
        )
        score_dt = pd.Timestamp(valid_df["date"].min())

    _g = torch.Generator()
    _g.manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        generator=_g,
    )

    cat_levels = {c: len(m) for c, m in cat_maps.items()}
    model = DeepForecaster(
        arch=arch,
        num_numeric=NUM_NUMERIC,
        cat_levels=cat_levels,
        embedding_dims=cfg.embedding_dims,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        horizon=cfg.horizon_days,
        lookback=cfg.lookback_days,
    ).to(device)

    # Manual AMP if enabled: fp16 execution copy driven by fp32 master
    # weights. torch.amp.autocast never casts the _VF.lstm/gru/rnn kernels
    # on mps and GradScaler("mps") rejects fp16 parameters, so the cast is
    # done at the module level here. Loss, clipping and metrics stay fp32.
    use_amp = bool(cfg.amp) and device.type in ("mps", "cuda")
    if cfg.amp and not use_amp:
        print("AMP requested but unsupported on this device; continuing in fp32")
    exec_model = model
    exec_params = None
    if use_amp:
        exec_model = copy.deepcopy(model).half()
        exec_params = list(exec_model.parameters())

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        fused=device.type in ("mps", "cuda"),
    )
    criterion = nn.MSELoss()

    best_valid_rmsle = float("inf")
    best_state = None
    best_metrics = None
    best_per_horizon = []
    best_epoch = 0
    patience_counter = 0
    epochs_run = 0
    val_inference_seconds = 0.0
    stopped_early = False
    wall_cap_hit = False
    per_horizon = []
    _run_start = _time.time()

    for epoch in range(1, cfg.max_epochs + 1):
        if (_time.time() - _run_start) / 3600.0 >= cfg.max_hours_per_arch:
            wall_cap_hit = True
            stopped_early = False
            break
        model.train()
        if use_amp:
            exec_model.train()
        loss_sum = torch.zeros((), device=device)
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{cfg.max_epochs} [{arch.upper()} Train]",
            leave=False,
        )
        for step, batch in enumerate(pbar):
            x_num = batch["x_num"].to(device)
            x_cat = batch["x_cat"].to(device)
            y_log = batch["y_log"].to(device)
            if use_amp:
                x_num = x_num.half()
                # Refresh the fp16 execution copy from the fp32 master.
                with torch.no_grad():
                    for p_e, p_m in zip(exec_params, model.parameters(), strict=True):
                        p_e.copy_(p_m)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                exec_model.zero_grad(set_to_none=True)
            preds = exec_model(x_num, x_cat)
            if use_amp:
                preds = preds.float()
            loss = criterion(preds, y_log)
            loss.backward()
            if use_amp:
                for p_e, p_m in zip(exec_params, model.parameters(), strict=True):
                    if p_e.grad is not None:
                        p_m.grad = p_e.grad.detach().to(torch.float32)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            batch_size = x_num.shape[0]
            loss_sum = loss_sum + loss.detach() * batch_size
            if (step + 1) % 25 == 0:
                _loss_val = (loss_sum / max(1, (step + 1) * cfg.batch_size)).item()
                pbar.set_postfix(loss=f"{_loss_val:.4f}")

        avg_train_loss = (loss_sum / max(1, len(train_ds))).item()
        epochs_run = epoch

        # One NaN check per epoch: a per-batch torch.isnan forces a GPU->CPU
        # sync every step, ~15% slower per epoch on mps.
        if torch.isnan(loss_sum):
            raise RuntimeError(f"NaN loss encountered at epoch {epoch}")

        if valid_panels is not None:
            model.eval()
            if use_amp:
                exec_model.eval()
            _t0 = _time.time()
            y_true, y_pred, per_horizon, _counts = _batched_eval(
                model,
                valid_panels,
                score_dt,
                cfg.lookback_days,
                cfg.horizon_days,
                cfg.eval_block_days,
                device,
            )
            val_inference_seconds = _time.time() - _t0
            val_metrics = compute_metrics(y_true, y_pred)
            val_rmsle = val_metrics["rmsle"]
            print("E", epoch, "mse", round(avg_train_loss, 4))
            if val_rmsle < (best_valid_rmsle - cfg.min_delta):
                best_valid_rmsle = val_rmsle
                best_metrics = val_metrics
                best_per_horizon = list(per_horizon)
                best_epoch = epoch
                best_state = {
                    k: v.detach().clone().cpu() for k, v in model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    stopped_early = True
                    break
        else:
            print(f"Epoch {epoch}/{cfg.max_epochs} TrainMSE {avg_train_loss:.4f}")
            best_state = {
                k: v.detach().clone().cpu() for k, v in model.state_dict().items()
            }
            best_epoch = epoch
            best_metrics = {}
            best_valid_rmsle = 0.0

    if best_state is None:
        best_state = {
            k: v.detach().clone().cpu() for k, v in model.state_dict().items()
        }
    model.load_state_dict(best_state)
    param_count = sum(p.numel() for p in model.parameters())

    rows_masked = len(train_df) - len(apply_store52_mask(train_df.copy()))

    return {
        "model_state": best_state,
        "cat_maps": cat_maps,
        "scaler": scaler,
        "params": param_count,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "best_valid_rmsle": best_valid_rmsle,
        "metrics": best_metrics or {},
        "per_horizon": best_per_horizon,
        "inference_seconds": val_inference_seconds,
        "stop_reason": "wall_clock_cap"
        if wall_cap_hit
        else ("early_stopping" if stopped_early else "max_epochs"),
        "rows_available": len(apply_store52_mask(train_df.copy())),
        "rows_masked": rows_masked,
        "train_rows_available": len(train_masked),
        "seconds": _time.time() - t0,
    }


def _valid_entry(res):
    """The valid block one training run publishes, in the pinned shape."""
    m = res.get("metrics") or {}
    if not m:
        return {}
    return {
        "n_rows": m.get("n_rows", 0),
        "n_active": m.get("n_rows_active", m.get("n_active", 0)),
        "full_grid": {
            "rmsle": m.get("rmsle", 0),
            "mae": m.get("mae", 0),
            "rmse": m.get("rmse", 0),
        },
        "active_slice": {
            "rmsle": m.get("rmsle_active", 0),
            "mae": m.get("mae_active", 0),
            "rmse": m.get("rmse_active", 0),
        },
        "per_horizon_rmsle": res.get("per_horizon", []),
        "inference_seconds": res.get("inference_seconds", 0.0),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--arch", choices=["rnn", "lstm", "gru"], default="lstm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--train-start", type=str, default=None)
    parser.add_argument("--ablation", type=str, default=None)
    parser.add_argument(
        "--device", choices=["auto", "mps", "cuda", "cpu"], default="auto"
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="manual fp16 AMP on gpu devices: fp16 execution copy driven by "
        "fp32 master weights; loss, clipping and metrics stay fp32",
    )
    parser.add_argument("--benchmark-steps", type=int, default=0)
    parser.add_argument("--seed-check", action="store_true")
    parser.add_argument("--no-artifact", action="store_true")
    parser.add_argument("--out-dir", type=str, default="artifacts")
    args = parser.parse_args(argv)

    if args.audit:
        run_audit()

    cfg = DeepConfig(seed=args.seed)
    if args.max_epochs is not None:
        cfg.max_epochs = args.max_epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.amp:
        cfg.amp = True
    if args.ablation is not None:
        cfg.train_start = "2015-01-01"
    if args.train_start is not None:
        cfg.train_start = args.train_start

    device = resolve_device(args.device)
    print(f"Using compute device: {device} | torch {torch.__version__}")

    if args.benchmark_steps and args.benchmark_steps > 0:
        # The measured device sweep: real training steps, the device
        # synchronised around the timing, the peak memory read from the
        # allocator. A launch-async number is not a measurement.
        repo_root = Path(".")
        results_path = Path(args.out_dir) / "deep_results.json"
        if not results_path.is_file():
            print(
                f"{results_path} does not exist; the benchmark updates an "
                "existing artifact, it does not create one. Run the training "
                "first."
            )
            return 1
        results, migrated = artifact_migrate_legacy(json.load(open(results_path)))
        for line in migrated:
            print(f"  migrated: {line}")
        # The headline device is the one the three headline runs used, from
        # the shared config: the CLI device is where the BENCHMARK runs, and
        # the two are different facts.
        headline_device = str(
            (results.get("shared_config") or {}).get("device") or str(device)
        )
        # An explicit --device measures that device; "auto" sweeps every
        # device this machine can run.
        results["runtime_tradeoff"] = runtime_tradeoff(
            headline_device,
            steps=args.benchmark_steps,
            devices=[str(device)] if args.device != "auto" else None,
        )
        os.makedirs(args.out_dir, exist_ok=True)
        # The benchmark publishes the artifact too, so it goes out through
        # the same checked writer: a benchmark run must never publish an
        # artifact the freeze would refuse.
        artifact_write_artifact(
            results_path,
            artifact_finalize(results, repo_root=repo_root),
            complete=results.get("status") == "complete",
        )
        for measured in results["runtime_tradeoff"]["devices"]:
            print(
                f"Benchmark: {measured['device']} {measured['steps']} steps in "
                f"{measured['seconds']:.3f}s "
                f"({measured['seconds_per_step']:.4f}s/step)"
            )
        return 0

    if args.seed_check:
        cfg.seed = 43
        train_df = pd.read_parquet(Path(".") / "splits" / "train.parquet")
        valid_df = pd.read_parquet(Path(".") / "splits" / "valid.parquet")
        results_path = Path(args.out_dir) / "deep_results.json"
        if not results_path.is_file():
            print(
                f"{results_path} does not exist; the seed check compares "
                "against the recorded seed-42 run, so the artifact must "
                "exist. Run the training first."
            )
            return 1
        # The seed check publishes the artifact too, so a legacy artifact
        # is migrated on the way in, like every other write path.
        results, migrated = artifact_migrate_legacy(json.load(open(results_path)))
        for line in migrated:
            print(f"  migrated: {line}")
        res = train_one("lstm", cfg, device, train_df, valid_df)
        cmp = results.get("comparison", {})
        # The range is the spread between the seed the headline runs used
        # and the seed this check ran; one run's number is not a range.
        # The check trains the lstm, so the recorded number must be the
        # lstm tier's, not the first tier's.
        seed_42 = None
        for tier in results.get("tiers", []):
            if str(tier.get("architecture", "")).lower() != "lstm":
                continue
            seed_42 = (tier.get("valid") or {}).get("full_grid", {}).get("rmsle")
            break
        if seed_42 is None:
            print(
                "WARNING: the artifact carries no lstm tier full-grid "
                "RMSLE; the seed check records a null range"
            )
        seed_43 = res.get("best_valid_rmsle")
        spread = (
            abs(float(seed_42) - float(seed_43))
            if seed_42 is not None and seed_43 is not None
            else None
        )
        cmp["seed_check"] = {
            "status": "run",
            "seeds": [42, 43],
            "rmsle_range": spread,
        }
        results["comparison"] = cmp
        os.makedirs(args.out_dir, exist_ok=True)
        # The seed check publishes the artifact too, so it goes out through
        # the same checked writer.
        artifact_write_artifact(
            results_path,
            artifact_finalize(results, repo_root=Path(".")),
            complete=False,
        )
        print("Seed check: RMSLE", res.get("best_valid_rmsle"))
        return 0

    repo_root = Path(".")
    train_df = pd.read_parquet(repo_root / "splits" / "train.parquet")
    valid_df = pd.read_parquet(repo_root / "splits" / "valid.parquet")

    if args.smoke:
        set_seed(cfg.seed)

        cards = {
            "store_nbr": 54,
            "family": 33,
            "city": 22,
            "state": 16,
            "store_type": 5,
            "cluster": 17,
        }
        m = DeepForecaster(
            arch=args.arch,
            num_numeric=NUM_NUMERIC,
            cat_levels=cards,
            embedding_dims=cfg.embedding_dims,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            horizon=cfg.horizon_days,
            lookback=cfg.lookback_days,
        ).to(device)
        x_num = torch.zeros(4, cfg.lookback_days, NUM_NUMERIC, device=device)
        x_cat = torch.zeros(4, len(CAT_COLS), dtype=torch.long, device=device)
        out = m(x_num, x_cat)
        print(f"Smoke test passed! Batch output shape: {out.shape}")
        sys.exit(0)

    os.makedirs(args.out_dir, exist_ok=True)
    results_path = Path(args.out_dir) / "deep_results.json"
    results = {
        "status": "partial",
        "tiers": [],
        "notes": "percentage-error metric excluded at 35.2% zeros in train+valid",
    }
    if results_path.exists():
        try:
            results, migrated = artifact_migrate_legacy(json.load(open(results_path)))
            for line in migrated:
                print(f"  migrated: {line}")
        except Exception as error:
            # The previous artifact could not be read: say so, and start
            # from a clean stub rather than pretending it was fine.
            print(
                f"WARNING: the existing {results_path} could not be read ({error});"
                " the run starts from an empty artifact"
            )
            results = {}

    archs_to_run = ["rnn", "lstm", "gru"] if args.all else [args.arch]

    for arch in archs_to_run:
        res = train_one(arch, cfg, device, train_df, valid_df)

        valid_entry = _valid_entry(res)

        # The ablation handling comes FIRST: the ablation updates only
        # ablation_2015plus (sheet Day 4), so it saves NO checkpoint --
        # writing artifacts/deep_{arch}.pt would overwrite the headline
        # tier's checkpoint while the artifact's tier block keeps the
        # digest of the model it describes, and the next comparison would
        # re-predict the folds from a checkpoint the artifact never
        # carried. The sensitivity result is the metric; the deliverables
        # stay the three headline checkpoints (sheet Section 10).
        if args.ablation:
            # The ablation's own slice, not the full train: the masked
            # count is the rows the mask drops INSIDE the slice, so the
            # three counts reconcile (available + masked = slice length).
            ablation_slice = train_df[
                pd.to_datetime(train_df["date"]) >= pd.Timestamp(cfg.train_start)
            ]
            ablation_masked = apply_store52_mask(ablation_slice.copy())
            results["ablation_2015plus"] = {
                "status": "run",
                "architecture": arch,
                "train_start": cfg.train_start,
                "train_rows_available": res["train_rows_available"],
                "train_rows_used": res["train_rows_available"],
                "train_rows_masked": len(ablation_slice) - len(ablation_masked),
                "epochs_run": res["epochs_run"],
                "best_epoch": res["best_epoch"],
                "stop_reason": res.get("stop_reason", "max_epochs"),
                "valid": valid_entry,
                "delta_vs_full_history_rmsle": None,
                "reason": "",
            }
            continue

        ckpt_path = Path(args.out_dir) / f"deep_{arch}.pt"
        save_obj = {
            "state_dict": res["model_state"],
            "model_state_dict": res["model_state"],
            "config": {
                k: (v if not isinstance(v, (set, tuple)) else list(v))
                for k, v in cfg.__dict__.items()
            },
            "category_maps": res["cat_maps"],
            "cat_maps": res["cat_maps"],
            "numeric_scaler": res["scaler"],
            "scaler": res["scaler"],
            "best_epoch": res["best_epoch"],
            "epochs_run": res["epochs_run"],
            "stop_reason": res.get("stop_reason", "max_epochs"),
            "valid_metrics": res["metrics"],
            "metrics": res["metrics"],
            "per_horizon_rmsle": res.get("per_horizon", []),
            "git_sha": git_sha(),
            "torch_version": str(torch.__version__),
            "class_name": arch,
            "arch": arch,
            "tier": f"deep-{arch}",
            "params": res["params"],
            "rows_available": res["train_rows_available"],
            "rows_masked": res["rows_masked"],
            "device": str(device),
            "config_sha256": cfg.get_sha256(),
        }
        torch.save(save_obj, ckpt_path)

        tier_entry = {
            "tier": f"deep-{arch}",
            "architecture": arch,
            "checkpoint": str(ckpt_path).replace("\\", "/"),
            "checkpoint_bytes": int(ckpt_path.stat().st_size),
            "checkpoint_sha256": _sha256_file(ckpt_path),
            "params": res["params"],
            "status": "complete",
            "train": {
                "rows_available": res["train_rows_available"],
                "rows_used": res["train_rows_available"],
                "rows_masked": res["rows_masked"],
                "windows": len(
                    TrainWindowDataset(
                        build_series_panels(
                            apply_store52_mask(
                                train_df.copy()
                                if cfg.train_start is None
                                else train_df[
                                    pd.to_datetime(train_df["date"])
                                    >= pd.Timestamp(cfg.train_start)
                                ].copy()
                            ),
                            res["cat_maps"],
                            res["scaler"],
                            cfg.lookback_days,
                            cfg.horizon_days,
                        ),
                        cfg.lookback_days,
                        cfg.horizon_days,
                        cfg.train_window_stride,
                    )
                ),
                "epochs_run": res["epochs_run"],
                "best_epoch": res["best_epoch"],
                # train_one derives the real stop reason (wall clock
                # cap, early stopping, or the epoch cap); recomputing
                # it here from best_epoch would mislabel a wall-cap run
                # as a convergence claim, which the sheet forbids.
                "stop_reason": res["stop_reason"],
                "seconds": res.get("seconds", 0.0),
                "seconds_per_epoch": (
                    res.get("seconds", 0.0) / max(1, res["epochs_run"])
                ),
                "device": str(device),
                "torch_version": str(torch.__version__),
            },
            "valid": {},
            "fold_metrics": [],
        }
        tier_entry["valid"] = valid_entry

        results["tiers"] = [
            t for t in results.get("tiers", []) if t.get("tier") != f"deep-{arch}"
        ]
        results["tiers"].append(tier_entry)

    # top-level fields: the shared config, the config hash and the whole
    # artifact shape are owned by deep.artifact: the key set, the key
    # order, the tier order and the data fingerprint are pinned there, and
    # the write is checked before it lands.
    if args.ablation and results.get("shared_config") and results.get("config_sha256"):
        # The shared_config block is the record of the three HEADLINE
        # runs; the ablation updates only ablation_2015plus, and its own
        # slice and device live there. Re-stamping the block would flip
        # train_start to the ablation's slice and change config_sha256
        # to a config the headline runs never used.
        shared = results["shared_config"]
    else:
        shared = artifact_shared_config(cfg, str(device), str(torch.__version__))
        results["shared_config"] = shared
        results["config_sha256"] = artifact_config_sha256(shared)
    results["runtime_tradeoff"] = results.get(
        "runtime_tradeoff",
        {
            "headline_device": str(device),
            "devices": [],
            "colab_gpu_hours_used": None,
        },
    )
    results.setdefault("ablation_2015plus", artifact_empty_ablation())
    results["notes"] = artifact_notes(
        [
            "this artifact was written by a training run; the fold metrics are "
            "filled in by the comparison run, which re-predicts the three "
            "shared tail folds from the checkpoints",
        ]
    )
    # "complete" means the artifact can be frozen: all three tiers AND
    # their fold metrics. A training run alone has the tiers but not the
    # folds, so it publishes a partial artifact.
    results["status"] = (
        "complete"
        if len(results.get("tiers", [])) == 3
        and all(len(tier.get("fold_metrics", [])) == 3 for tier in results["tiers"])
        else "partial"
    )
    finalized = artifact_finalize(results, repo_root=repo_root, shared=shared)
    if not args.no_artifact:
        artifact_write_artifact(results_path, finalized, complete=False)
    print("\nTraining complete. Artifacts written to", args.out_dir)
    return 0


def _sha256_file(path):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
