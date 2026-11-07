import argparse
from pathlib import Path

import pandas as pd

from deep.data import (
    evaluate_checkpoint,
)


def verify_valid(checkpoint_path: str):
    repo_root = Path(".")
    train_df = pd.read_parquet(repo_root / "splits" / "train.parquet")
    valid_df = pd.read_parquet(repo_root / "splits" / "valid.parquet")
    frame = pd.concat([train_df, valid_df], ignore_index=True)
    res = evaluate_checkpoint(checkpoint_path, frame, "2016-03-28")
    print("=== Verify Valid Results ===")
    print(f"Evaluated rows: {res['n_rows']}")
    print(f"RMSLE: {res['full_grid']['rmsle']:.5f}")
    print(f"MAE:   {res['full_grid']['mae']:.5f}")
    print(f"RMSE:  {res['full_grid']['rmse']:.5f}")


def verify_checkpoint_tolerance(checkpoint_path: str, tol: float = 1e-6) -> bool:
    """Verify a checkpoint round-trips within tol (1e-6)."""
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # Re-save and reload to ensure byte-stable state_dict
    tmp = Path(checkpoint_path).with_suffix(".verify.pt")
    torch.save(ckpt, tmp)
    ckpt2 = torch.load(tmp, map_location="cpu", weights_only=True)
    tmp.unlink(missing_ok=True)

    maxdiff = 0.0
    for k in ckpt["state_dict"]:
        d = (ckpt["state_dict"][k] - ckpt2["state_dict"][k]).abs().max().item()
        if d > maxdiff:
            maxdiff = d
    ok = maxdiff <= tol
    print(f"{checkpoint_path}: maxdiff={maxdiff:.3e} tol={tol:.0e} ok={ok}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--verify-valid", action="store_true")
    args = parser.parse_args()
    if args.verify_valid:
        verify_valid(args.checkpoint)


if __name__ == "__main__":
    main()
