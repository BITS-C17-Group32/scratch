import argparse
import json
import os
from pathlib import Path
import random
import sys
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
        sum(
            s.get("num_rows", 0)
            for s in splits_dict.values()
            if isinstance(s, dict)
        ),
    )
    local_test_rows = (
        splits_dict.get("local_test", {}).get("num_rows", "N/A")
        if isinstance(splits_dict.get("local_test"), dict)
        else "N/A"
    )

    print("=== S4 Audit Summary ===")
    print(f"Manifest total rows: {total_rows}")
    print(
        f"Train rows: {len(train_df)} | Valid rows: {len(valid_df)} | Local test rows: {local_test_rows}"
    )
    print(f"Series count: {series_count}")
    print(f"Store-52 train masked rows dropped: {len(st52_masked)}")
    print(f"Train rows available after mask: {len(train_df) - len(st52_masked)}")
    print(f"Train rows from 2015-01-01 onward: {len(train_2015_plus)}")
    sys.exit(0)


def train_single_arch(
    arch: str,
    cfg: DeepConfig,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    device: torch.device,
) -> dict:
    set_seed(cfg.seed)

    print(f"\n--- Preparing Dataset for [{arch.upper()}] ---")
    train_prep, cat_maps, _ = prepare_frame_data(train_df, is_train=True)
    valid_prep, _, _ = prepare_frame_data(
        valid_df, is_train=False, cat_maps=cat_maps
    )

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

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=cfg.batch_size, shuffle=False
    )

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
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        train_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{cfg.max_epochs} [{arch.upper()} Train]",
            leave=False,
        )
        for batch in pbar:
            x_num = batch["x_num"].to(device)
            x_cat = batch["x_cat"].to(device)
            y_log = batch["y_log"].to(device)

            optimizer.zero_grad()
            preds = model(x_num, x_cat)
            loss = criterion(preds, y_log)

            if torch.isnan(loss):
                raise RuntimeError(f"NaN loss encountered at epoch {epoch}")

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / max(1, len(train_loader))

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
            f"Epoch {epoch:2d}/{cfg.max_epochs:2d} | Train MSE: {avg_train_loss:.4f} | Valid RMSLE: {val_rmsle:.4f}"
        )

        if val_rmsle < (best_valid_rmsle - 0.0001):
            best_valid_rmsle = val_rmsle
            best_epoch = epoch
            best_state = {
                k: v.cpu() for k, v in model.state_dict().items()
            }
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
        "metrics": val_metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--arch", choices=["rnn", "lstm", "gru"], default="lstm"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--train-start", type=str, default=None)
    parser.add_argument("--ablation", type=str, default=None)
    args = parser.parse_args()

    if args.audit:
        run_audit()

    cfg = DeepConfig(seed=args.seed, max_epochs=args.max_epochs)
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using compute device: {device}")

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
            "notes": (
                "percentage-error metric excluded at 35.2% zeros in train+valid"
            ),
        }
    )

    archs_to_run = ["rnn", "lstm", "gru"] if args.all else [args.arch]

    for arch in archs_to_run:
        res = train_single_arch(arch, cfg, train_df, valid_df, device)

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