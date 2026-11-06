import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from deep.config import DeepConfig
from deep.data import WindowSequenceDataset, load_split_manifest, prepare_frame_data
from deep.metrics import compute_metrics
from deep.model import DeepForecaster


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def look_for_gpu_tool(name: str) -> bool:
    """True if a vendor GPU query CLI exists and answers (nvidia-smi, rocm-smi)."""
    path = shutil.which(name)
    if path is None:
        return False
    try:
        out = subprocess.run([path], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0


def resolve_device(preference: str = "auto") -> torch.device:
    """Resolve the compute device.

    auto: MPS (Apple Silicon) > CUDA (NVIDIA; ROCm builds surface HIP through
    the cuda API) > CPU. When a GPU is present but this torch build cannot use
    it, say why instead of silently falling back to CPU.
    """
    if preference != "auto":
        dev = torch.device(preference)
        if preference == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"--device cuda requested but torch.cuda is unavailable "
                f"(torch.version.cuda={torch.version.cuda}; this is a CPU-only "
                "wheel). Install a CUDA build per https://pytorch.org/get-started/locally/"
            )
        if preference == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError(
                "--device mps requested but MPS is unavailable (requires an "
                "Apple Silicon Mac with torch's arm64 build)."
            )
        return dev

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")

    # On macOS, torch wheels never ship CUDA/ROCm, so skip GPU-aware warnings
    # there (an Intel Mac legitimately trains on CPU).
    if sys.platform != "darwin" and look_for_gpu_tool("nvidia-smi"):
        print(
            "\nWarning: NVIDIA GPU detected via nvidia-smi, but this torch build "
            f"has no CUDA support (torch.version.cuda={torch.version.cuda}) — "
            "on Windows the default `pip install torch` ships CPU-only. Reinstall "
            "PyTorch with CUDA per https://pytorch.org/get-started/locally/ ; "
            "training will run on CPU meanwhile."
        )
    if sys.platform != "darwin" and look_for_gpu_tool("rocm-smi"):
        print(
            "\nWarning: AMD ROCm GPU detected via rocm-smi, but this torch build "
            "lacks ROCm/HIP support. Reinstall PyTorch from the ROCm wheel per "
            "https://pytorch.org/get-started/locally/ ; training will run on CPU "
            "meanwhile."
        )
    return torch.device("cpu")


def run_audit():
    repo_root = Path(".")
    manifest = load_split_manifest(repo_root)

    train_df = pd.read_parquet(repo_root / "splits" / "train.parquet")
    valid_df = pd.read_parquet(repo_root / "splits" / "valid.parquet")

    series_count = len(train_df.groupby(["store_nbr", "family"]))
    st52_masked = train_df[
        (train_df["store_nbr"] == 52)
        & (pd.to_datetime(train_df["date"]) < pd.to_datetime("2017-04-20"))
        & (train_df["id"].notna())
    ]
    train_2015_plus = train_df[
        pd.to_datetime(train_df["date"]) >= pd.to_datetime("2015-01-01")
    ]

    splits_dict = manifest.get("splits", manifest)

    total_rows = manifest.get(
        "total_rows",
        sum(s.get("num_rows", 0) for s in splits_dict.values() if isinstance(s, dict)),
    )
    local_test_rows = (
        splits_dict.get("local_test", {}).get("num_rows", "N/A")
        if isinstance(splits_dict.get("local_test"), dict)
        else "N/A"
    )

    print("=== S4 Audit Summary ===")
    print(f"Manifest total rows: {total_rows}")
    print(
        f"Train rows: {len(train_df)} | Valid rows: {len(valid_df)} | "
        f"Local test rows: {local_test_rows}"
    )
    print(f"Series count: {series_count}")
    print(f"Store-52 train masked rows dropped: {len(st52_masked)}")
    print(f"Train rows available after mask: {len(train_df) - len(st52_masked)}")
    print(f"Train rows from 2015-01-01 onward: {len(train_2015_plus)}")
    sys.exit(0)


def build_datasets(
    cfg: DeepConfig,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
):
    """Build window datasets once; they are shared across all architectures."""
    print("\n--- Preparing Dataset ---")
    train_prep, cat_maps, _ = prepare_frame_data(train_df, is_train=True)
    valid_prep, _, _ = prepare_frame_data(valid_df, is_train=False, cat_maps=cat_maps)

    train_ds = WindowSequenceDataset(
        train_prep,
        lookback=cfg.lookback,
        horizon=cfg.horizon,
        is_train=True,
        stride=8,
    )
    valid_ds = WindowSequenceDataset(
        valid_prep,
        lookback=cfg.lookback,
        horizon=cfg.horizon,
        is_train=False,
        stride=cfg.horizon,
    )
    return train_ds, valid_ds, cat_maps


def train_single_arch(
    arch: str,
    cfg: DeepConfig,
    train_ds: WindowSequenceDataset,
    valid_ds: WindowSequenceDataset,
    cat_maps: dict,
    device: torch.device,
) -> dict:
    set_seed(cfg.seed)

    print(f"\n--- Training [{arch.upper()}] ---")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False)

    print(f"Train sequences: {len(train_ds):,} | Valid sequences: {len(valid_ds):,}")

    cards = {c: len(m) for c, m in cat_maps.items()}
    model = DeepForecaster(
        cat_cardinalities=cards,
        arch=arch,
        lookback=cfg.lookback,
        horizon=cfg.horizon,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    criterion = nn.MSELoss()

    best_valid_rmsle = float("inf")
    best_state = None
    best_metrics = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        # Accumulate batch-mean losses on the device: calling loss.item()
        # every batch forces a GPU->CPU sync each step (~15% slower).
        loss_sum = torch.zeros((), device=device)
        n_batches = len(train_loader)

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{cfg.max_epochs} [{arch.upper()} Train]",
            leave=False,
        )
        for step, batch in enumerate(pbar):
            x_num = batch["x_num"].to(device)
            x_cat = batch["x_cat"].to(device)
            y_log = batch["y_log"].to(device)

            optimizer.zero_grad(set_to_none=True)
            preds = model(x_num, x_cat)
            loss = criterion(preds, y_log)

            if torch.isnan(loss):
                raise RuntimeError(f"NaN loss encountered at epoch {epoch}")

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_sum = loss_sum + loss.detach()
            if (step + 1) % 25 == 0:
                pbar.set_postfix({"loss": f"{(loss_sum / (step + 1)).item():.4f}"})

        avg_train_loss = (loss_sum / max(1, n_batches)).item()

        model.eval()
        all_preds_raw = []
        all_true_raw = []

        vbar = tqdm(
            valid_loader,
            desc=f"Epoch {epoch}/{cfg.max_epochs} [{arch.upper()} Valid]",
            leave=False,
        )
        with torch.no_grad():
            for batch in vbar:
                x_num = batch["x_num"].to(device)
                x_cat = batch["x_cat"].to(device)
                y_raw = batch["y_raw"].numpy()

                preds_log = model(x_num, x_cat).cpu().numpy()
                preds_raw = np.expm1(np.clip(preds_log, 0, None))

                all_preds_raw.append(preds_raw.reshape(-1))
                all_true_raw.append(y_raw.reshape(-1))

        y_pred = np.concatenate(all_preds_raw)
        y_true = np.concatenate(all_true_raw)

        val_metrics = compute_metrics(y_true, y_pred)
        val_rmsle = val_metrics["rmsle"]

        print(
            f"Epoch {epoch:2d}/{cfg.max_epochs:2d} | Train MSE: "
            f"{avg_train_loss:.4f} | Valid RMSLE: {val_rmsle:.4f}"
        )

        if val_rmsle < (best_valid_rmsle - 0.0001):
            best_valid_rmsle = val_rmsle
            best_metrics = val_metrics
            best_epoch = epoch
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    param_count = sum(p.numel() for p in model.parameters())

    return {
        "model_state": best_state,
        "cat_maps": cat_maps,
        "params": param_count,
        "best_epoch": best_epoch,
        "best_valid_rmsle": best_valid_rmsle,
        "metrics": best_metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--arch", choices=["rnn", "lstm", "gru"], default="lstm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--train-start", type=str, default=None)
    parser.add_argument("--ablation", type=str, default=None)
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
        help="compute device; auto picks MPS > CUDA > CPU",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="override DeepConfig.batch_size for quick throughput experiments",
    )
    args = parser.parse_args()

    if args.audit:
        run_audit()

    cfg = DeepConfig(seed=args.seed, max_epochs=args.max_epochs)
    if args.batch_size:
        cfg.batch_size = args.batch_size

    device = resolve_device(args.device)
    print(f"Using compute device: {device} | torch {torch.__version__}")

    repo_root = Path(".")
    train_df = pd.read_parquet(repo_root / "splits" / "train.parquet")
    valid_df = pd.read_parquet(repo_root / "splits" / "valid.parquet")

    if args.train_start:
        train_df = train_df[
            pd.to_datetime(train_df["date"]) >= pd.to_datetime(args.train_start)
        ].reset_index(drop=True)

    if args.smoke:
        set_seed(cfg.seed)
        t_prep, cm, _ = prepare_frame_data(train_df)
        ds = WindowSequenceDataset(t_prep, is_train=True)
        loader = DataLoader(ds, batch_size=32)
        batch = next(iter(loader))
        cards = {c: len(m) for c, m in cm.items()}
        m = DeepForecaster(cards, arch=args.arch).to(device)
        out = m(batch["x_num"].to(device), batch["x_cat"].to(device))
        print(f"Smoke test passed! Batch output shape: {out.shape}")
        sys.exit(0)

    os.makedirs("artifacts", exist_ok=True)
    results_path = Path("artifacts/deep_results.json")
    results = (
        json.load(open(results_path))
        if results_path.exists()
        else {
            "status": "complete",
            "tiers": [],
            "notes": ("percentage-error metric excluded at 35.2% zeros in train+valid"),
        }
    )

    archs_to_run = ["rnn", "lstm", "gru"] if args.all else [args.arch]

    train_ds, valid_ds, cat_maps = build_datasets(cfg, train_df, valid_df)

    for arch in archs_to_run:
        res = train_single_arch(arch, cfg, train_ds, valid_ds, cat_maps, device)

        ckpt_path = Path(f"artifacts/deep_{arch}.pt")
        save_obj = {
            "state_dict": res["model_state"],
            "cat_maps": res["cat_maps"],
            "config": cfg.__dict__,
            "arch": arch,
        }
        torch.save(save_obj, ckpt_path)

        tier_entry = {
            "tier": f"deep-{arch}",
            "architecture": arch,
            "params": res["params"],
            "best_epoch": res["best_epoch"],
            "config_sha256": cfg.get_sha256(),
            "device": str(device),
            "torch_version": torch.__version__,
            "metrics": res["metrics"],
        }

        results["tiers"] = [
            t for t in results.get("tiers", []) if t["tier"] != f"deep-{arch}"
        ]
        results["tiers"].append(tier_entry)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nTraining complete. Artifacts written to artifacts/")


if __name__ == "__main__":
    main()
