import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from deep.data import prepare_frame_data
from deep.metrics import compute_metrics
from deep.model import DeepForecaster


def fit_predict_frozen(df: pd.DataFrame, score_start: str = "2016-03-28") -> np.ndarray:
    """Predict horizon sales for frozen evaluation handoff."""
    checkpoint_path = Path("artifacts/deep_lstm.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}.")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cat_maps = ckpt["cat_maps"]
    cfg_dict = ckpt["config"]

    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    score_dt = pd.to_datetime(score_start)

    min_date = df["date"].min()
    req_date = score_dt - pd.Timedelta(days=cfg_dict["lookback"])
    if min_date > req_date:
        raise ValueError(
            f"Frame requires history starting at {req_date}, but got {min_date}"
        )

    df_prep, _, _ = prepare_frame_data(df, is_train=False, cat_maps=cat_maps)

    cards = {c: len(m) for c, m in cat_maps.items()}
    model = DeepForecaster(
        cat_cardinalities=cards,
        arch=ckpt["arch"],
        lookback=cfg_dict["lookback"],
        horizon=cfg_dict["horizon"],
        hidden_dim=cfg_dict["hidden_dim"],
        num_layers=cfg_dict["num_layers"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    series_groups = df_prep.groupby(["store_nbr", "family"])
    preds_list = []

    cat_cols = [
        "store_nbr_idx",
        "family_idx",
        "city_idx",
        "state_idx",
        "type_idx",
        "cluster_idx",
    ]

    with torch.no_grad():
        for _, group in series_groups:
            group = group.sort_values("date").reset_index(drop=True)
            idx_start = group[group["date"] >= score_dt].index[0]
            lb = cfg_dict["lookback"]

            sub_group = group.iloc[idx_start - lb : idx_start]

            x_sales = sub_group["log_sales"].values.astype(np.float32)
            x_promo = sub_group["onpromotion_scaled"].values.astype(np.float32)
            x_sin = sub_group["dday_sin"].values.astype(np.float32)
            x_cos = sub_group["dday_cos"].values.astype(np.float32)

            x_num = torch.from_numpy(
                np.stack([x_sales, x_promo, x_sin, x_cos], axis=-1)
            ).unsqueeze(0)
            x_cat = torch.from_numpy(
                sub_group[cat_cols].iloc[0].values.astype(np.int64)
            ).unsqueeze(0)

            out_log = model(x_num, x_cat).squeeze(0).numpy()
            out_raw = np.expm1(np.clip(out_log, 0, None))
            preds_list.extend(out_raw)

    return np.array(preds_list)


def verify_valid(checkpoint_path: str):
    repo_root = Path(".")
    valid_df = pd.read_parquet(repo_root / "splits" / "valid.parquet")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cat_maps = ckpt["cat_maps"]
    cfg_dict = ckpt["config"]

    valid_prep, _, _ = prepare_frame_data(valid_df, is_train=False, cat_maps=cat_maps)

    from torch.utils.data import DataLoader

    from deep.data import WindowSequenceDataset

    vds = WindowSequenceDataset(
        valid_prep,
        lookback=cfg_dict["lookback"],
        horizon=cfg_dict["horizon"],
        is_train=False,
        stride=cfg_dict["horizon"],
    )
    vloader = DataLoader(vds, batch_size=256, shuffle=False)

    cards = {c: len(m) for c, m in cat_maps.items()}
    model = DeepForecaster(
        cat_cardinalities=cards,
        arch=ckpt["arch"],
        lookback=cfg_dict["lookback"],
        horizon=cfg_dict["horizon"],
        hidden_dim=cfg_dict["hidden_dim"],
        num_layers=cfg_dict["num_layers"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    all_preds = []
    all_true = []

    with torch.no_grad():
        for batch in vloader:
            preds_log = model(batch["x_num"], batch["x_cat"]).numpy()
            preds_raw = np.expm1(np.clip(preds_log, 0, None))
            all_preds.append(preds_raw.reshape(-1))
            all_true.append(batch["y_raw"].numpy().reshape(-1))

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    metrics = compute_metrics(y_true, y_pred)
    print(f"=== Verify Valid Results ({ckpt['arch']}) ===")
    print(f"Evaluated rows: {metrics['n_rows']}")
    print(f"RMSLE: {metrics['rmsle']:.5f}")
    print(f"MAE:   {metrics['mae']:.5f}")
    print(f"RMSE:  {metrics['rmse']:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--verify-valid", action="store_true")
    args = parser.parse_args()

    if args.verify_valid:
        verify_valid(args.checkpoint)
