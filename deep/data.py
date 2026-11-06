import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_split_manifest(repo_root: Path) -> dict:
    manifest_path = repo_root / "splits" / "split_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_frame_data(
    df: pd.DataFrame,
    is_train: bool = True,
    cat_maps: Optional[Dict[str, Dict[Any, int]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[Any, int]], List[str]]:
    df = df.copy()

    # Determine available categorical columns present in df
    candidate_cat_cols = ["family", "city", "state", "type", "store_type"]
    cat_cols = [c for c in candidate_cat_cols if c in df.columns]

    if cat_maps is None:
        cat_maps = {}
        for col in cat_cols:
            unique_vals = sorted(df[col].dropna().unique().tolist())
            # Reserve 0 for unknown/unseen categories
            cat_maps[col] = {val: i + 1 for i, val in enumerate(unique_vals)}

    # Encode categorical columns
    for col in cat_cols:
        mapping = cat_maps.get(col, {})
        df[f"{col}_idx"] = df[col].map(mapping).fillna(0).astype(int)

    # Ensure required numerical features exist
    if "onpromotion" in df.columns:
        df["onpromotion"] = df["onpromotion"].fillna(0).astype(float)
    else:
        df["onpromotion"] = 0.0

    if "dcoilwtico" in df.columns:
        df["dcoilwtico"] = (
            df["dcoilwtico"].ffill().bfill().fillna(0.0).astype(float)
        )
    else:
        df["dcoilwtico"] = 0.0

    # Ensure log sales target column
    if "sales" in df.columns:
        df["sales_log"] = np.log1p(
            np.clip(df["sales"].fillna(0).values, 0, None)
        )
    else:
        df["sales_log"] = 0.0

    return df, cat_maps, cat_cols


class WindowSequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        lookback: int = 60,
        horizon: int = 16,
        is_train: bool = True,
        stride: int = 1,
    ):
        self.lookback = lookback
        self.horizon = horizon
        self.is_train = is_train

        cat_idx_cols = [c for c in df.columns if c.endswith("_idx")]

        self.samples = []
        grouped = df.groupby(["store_nbr", "family"])

        for _, group in grouped:
            group = group.sort_values("date").reset_index(drop=True)
            n_rows = len(group)

            sales_raw = group["sales"].values if "sales" in group.columns else np.zeros(n_rows)
            sales_log = group["sales_log"].values
            onpromo = group["onpromotion"].values
            oil = group["dcoilwtico"].values

            cat_vals = (
                group[cat_idx_cols].iloc[0].values
                if len(cat_idx_cols) > 0
                else np.array([], dtype=int)
            )

            num_features = np.stack([sales_log, onpromo, oil], axis=1)

            min_len = lookback + horizon
            if n_rows < min_len:
                continue

            for start_idx in range(0, n_rows - min_len + 1, stride):
                end_x = start_idx + lookback
                end_y = end_x + horizon

                self.samples.append(
                    {
                        "x_num": torch.tensor(
                            num_features[start_idx:end_x], dtype=torch.float32
                        ),
                        "x_cat": torch.tensor(cat_vals, dtype=torch.long),
                        "y_log": torch.tensor(
                            sales_log[end_x:end_y], dtype=torch.float32
                        ),
                        "y_raw": torch.tensor(
                            sales_raw[end_x:end_y], dtype=torch.float32
                        ),
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]