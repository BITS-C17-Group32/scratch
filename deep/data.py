import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CAT_COLS = ["store_nbr", "family", "city", "state", "store_type", "cluster"]
FLAG_COLS = [
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
]
SCALED_COLS = ["onpromotion", "sales"]
NUM_NUMERIC = 22  # 2 scaled + earthquake_decay + 15 flags + 4 calendar


def load_split_manifest(repo_root: Path) -> dict:
    manifest_path = repo_root / "splits" / "split_manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def store52_mask(df: pd.DataFrame) -> pd.Series:
    """Rows to DROP in training: store 52 pre-opening with a real id."""
    date = pd.to_datetime(df["date"])
    mask = (
        (df["store_nbr"] == 52)
        & (date < pd.Timestamp("2017-04-20"))
        & (df["id"].notna())
    )
    return mask


def apply_store52_mask(df: pd.DataFrame) -> pd.DataFrame:
    mask = store52_mask(df)
    return df.loc[~mask].reset_index(drop=True)


def build_category_maps(train_df: pd.DataFrame) -> dict[str, dict[Any, int]]:
    cat_maps = {}
    for col in CAT_COLS:
        vals = sorted(train_df[col].dropna().unique().tolist())
        cat_maps[col] = {v: i + 1 for i, v in enumerate(vals)}
    return cat_maps


def build_scaler(train_df: pd.DataFrame) -> dict:
    sales_log = np.log1p(
        np.clip(train_df["sales"].fillna(0).values.astype(float), 0, None)
    )
    promo_log = np.log1p(
        np.clip(train_df["onpromotion"].fillna(0).values.astype(float), 0, None)
    )
    mean_promo = float(np.mean(promo_log))
    std_promo = float(np.std(promo_log))
    if std_promo == 0:
        std_promo = 1.0
    mean_sales = float(np.mean(sales_log))
    std_sales = float(np.std(sales_log))
    if std_sales == 0:
        std_sales = 1.0
    return {"mean": [mean_promo, mean_sales], "std": [std_promo, std_sales]}


def _calendar_features(dates: pd.Series) -> dict[str, np.ndarray]:
    dt = pd.to_datetime(dates)
    dow = dt.dt.dayofweek.values.astype(float)
    doy = dt.dt.dayofyear.values.astype(float)
    return {
        "dow_sin": np.sin(2 * np.pi * dow / 7.0),
        "dow_cos": np.cos(2 * np.pi * dow / 7.0),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
    }


def prepare_frame(df: pd.DataFrame, cat_maps: dict, scaler: dict) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    for col in CAT_COLS:
        mapping = cat_maps.get(col, {})
        df[f"{col}_idx"] = df[col].map(mapping).fillna(0).astype(int)

    sales = (
        df["sales"].fillna(0).astype(float).values
        if "sales" in df.columns
        else np.zeros(len(df))
    )
    promo = (
        df["onpromotion"].fillna(0).astype(float).values
        if "onpromotion" in df.columns
        else np.zeros(len(df))
    )
    sales_log = np.log1p(np.clip(sales, 0, None))
    promo_log = np.log1p(np.clip(promo, 0, None))
    df["sales_log"] = sales_log
    df["onpromotion_scaled"] = (promo_log - scaler["mean"][0]) / scaler["std"][0]
    df["sales_log_scaled"] = (sales_log - scaler["mean"][1]) / scaler["std"][1]

    for c in FLAG_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(float)
        else:
            df[c] = 0.0

    if "earthquake_decay" in df.columns:
        df["earthquake_decay"] = df["earthquake_decay"].fillna(0).astype(float)
    else:
        df["earthquake_decay"] = 0.0

    cal = _calendar_features(df["date"])
    for k, v in cal.items():
        df[k] = v.astype(float)

    if "onpromotion" not in df.columns:
        df["onpromotion"] = 0.0
    df["onpromotion"] = df["onpromotion"].fillna(0).astype(float)

    return df


def _feature_row(df: pd.DataFrame) -> np.ndarray:
    cols = [
        "onpromotion_scaled",
        "sales_log_scaled",
        "earthquake_decay",
        *FLAG_COLS,
        "dow_sin",
        "dow_cos",
        "doy_sin",
        "doy_cos",
    ]
    mat = df[cols].values.astype(np.float32)
    return mat


def build_series_panels(
    df: pd.DataFrame, cat_maps: dict, scaler: dict, lookback: int, horizon: int
) -> list[dict]:
    df = prepare_frame(df, cat_maps, scaler)
    panels = []
    grouped = df.groupby(["store_nbr", "family"], sort=False)
    for (store, family), g in grouped:
        g = g.sort_values("date").reset_index(drop=True)
        n = len(g)
        if n < lookback + 1:
            continue
        cat_idx = np.array([g[f"{c}_idx"].iloc[0] for c in CAT_COLS], dtype=np.int64)
        x = _feature_row(g)
        y_log = g["sales_log"].values.astype(np.float32)
        y_raw = (
            g["sales"].values.astype(np.float32)
            if "sales" in g.columns
            else np.zeros(n, dtype=np.float32)
        )
        panels.append(
            {
                "store_nbr": store,
                "family": family,
                "dates": pd.to_datetime(g["date"]).values,
                "x": x,
                "cat": cat_idx,
                "y_log": y_log,
                "y_raw": y_raw,
            }
        )
    return panels


class TrainWindowDataset(Dataset):
    def __init__(self, panels, lookback, horizon, stride):
        self.panels = panels
        self.lookback = lookback
        self.horizon = horizon
        self.samples = []
        for pi, p in enumerate(panels):
            n = len(p["dates"])
            if n < lookback + horizon:
                continue
            for start in range(0, n - lookback - horizon + 1, stride):
                self.samples.append((pi, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pi, start = self.samples[idx]
        p = self.panels[pi]
        x = p["x"][start : start + self.lookback]
        y_log = p["y_log"][start + self.lookback : start + self.lookback + self.horizon]
        y_raw = p["y_raw"][start + self.lookback : start + self.lookback + self.horizon]
        return {
            "x_num": torch.from_numpy(np.ascontiguousarray(x)),
            "x_cat": torch.from_numpy(p["cat"].copy()),
            "y_log": torch.from_numpy(np.ascontiguousarray(y_log)),
            "y_raw": torch.from_numpy(np.ascontiguousarray(y_raw)),
        }


def _batched_eval(model, panels, score_dt, lookback, horizon, block_size, device):
    """Return (y_true, y_pred) arrays and per-horizon squared errors/counts."""
    model.eval()
    all_preds = []
    all_trues = []
    np.zeros(horizon, dtype=np.float64)
    counts = np.zeros(horizon, dtype=np.int64)
    log_sq = np.zeros(horizon, dtype=np.float64)

    # Determine number of chunks from max target length
    max_chunks = 0
    panel_targets = []
    for p in panels:
        dates = pd.to_datetime(p["dates"])
        target_positions = [
            i for i, d in enumerate(dates) if pd.Timestamp(d) >= score_dt
        ]
        panel_targets.append(target_positions)
        max_chunks = max(
            max_chunks, (len(target_positions) + block_size - 1) // block_size
        )

    with torch.no_grad():
        for ci in range(max_chunks):
            xs = []
            cats = []
            metas = []
            for pi, p in enumerate(panels):
                tps = panel_targets[pi]
                if not tps:
                    continue
                start_idx = ci * block_size
                chunk_pos = tps[start_idx : start_idx + block_size]
                if not chunk_pos:
                    continue
                first = chunk_pos[0]
                if first - lookback < 0:
                    continue
                x = p["x"][first - lookback : first]
                xs.append(x)
                cats.append(p["cat"])
                metas.append((p, chunk_pos))
            if not xs:
                continue
            x_t = torch.from_numpy(
                np.ascontiguousarray(np.stack(xs)).astype(np.float32)
            ).to(device)
            cat_t = torch.from_numpy(
                np.ascontiguousarray(np.stack(cats)).astype(np.int64)
            ).to(device)
            pred_log = model(x_t, cat_t).detach().cpu().numpy()  # (B, horizon)
            for i, (p, chunk_pos) in enumerate(metas):
                pred_raw = np.clip(np.expm1(pred_log[i]), 0, None)
                block_len = len(chunk_pos)
                preds = pred_raw[:block_len]
                trues = p["y_raw"][chunk_pos]
                all_preds.append(preds)
                all_trues.append(trues)
                for j in range(block_len):
                    hidx = j
                    if hidx >= horizon:
                        break
                    diff = np.log1p(preds[j]) - np.log1p(trues[j])
                    log_sq[hidx] += float(diff * diff)
                    counts[hidx] += 1

    if all_preds:
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_trues)
    else:
        y_pred = np.array([], dtype=np.float64)
        y_true = np.array([], dtype=np.float64)

    per_horizon = []
    for h in range(horizon):
        if counts[h] > 0:
            per_horizon.append(float(np.sqrt(log_sq[h] / counts[h])))
        else:
            per_horizon.append(0.0)

    return y_true, y_pred, per_horizon, counts


def evaluate_checkpoint(checkpoint_path, frame, score_start, device="cpu"):
    """Evaluate a checkpoint on a frame, scoring rows with date >= score_start."""
    import time

    import torch

    from deep.metrics import compute_metrics
    from deep.model import DeepForecaster

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cat_maps = ckpt["category_maps"] if "category_maps" in ckpt else ckpt["cat_maps"]
    scaler = ckpt.get("numeric_scaler", ckpt.get("scaler"))
    cfg = ckpt["config"]
    arch = ckpt.get("arch", "lstm")
    if arch.startswith("deep-"):
        arch = arch.replace("deep-", "")

    lookback = cfg.get("lookback_days", cfg.get("lookback", 60))
    horizon = cfg.get("horizon_days", cfg.get("horizon", 16))

    panels = build_series_panels(frame, cat_maps, scaler, lookback, horizon)

    score_dt = pd.Timestamp(score_start)

    dev = torch.device(device) if isinstance(device, str) else device
    # resolve device string
    if isinstance(device, str):
        if device == "auto":
            if torch.backends.mps.is_available():
                dev = torch.device("mps")
            elif torch.cuda.is_available():
                dev = torch.device("cuda")
            else:
                dev = torch.device("cpu")
        else:
            dev = torch.device(device)

    model = DeepForecaster(
        arch=arch,
        num_numeric=NUM_NUMERIC,
        cat_levels={c: len(m) for c, m in cat_maps.items()},
        embedding_dims=cfg.get("embedding_dims", None),
        hidden_size=cfg.get("hidden_size", cfg.get("hidden_dim", 128)),
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.2),
        horizon=horizon,
        lookback=lookback,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev)
    model.eval()

    t0 = time.time()
    y_true, y_pred, per_horizon, _counts = _batched_eval(
        model, panels, score_dt, lookback, horizon, cfg.get("eval_block_days", 16), dev
    )
    metrics = compute_metrics(y_true, y_pred)
    inference_seconds = time.time() - t0
    n_active = int(np.sum(y_true > 0))
    result = {
        "n_rows": len(y_true),
        "n_active": n_active,
        "full_grid": {
            "rmsle": metrics["rmsle"],
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
        },
        "active_slice": {
            "rmsle": metrics["rmsle_active"],
            "mae": metrics["mae_active"],
            "rmse": metrics["rmse_active"],
        },
        "per_horizon_rmsle": per_horizon,
        "inference_seconds": inference_seconds,
    }
    return result


def predict_frame(checkpoint_path, frame, score_start):
    import torch

    from deep.model import DeepForecaster

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cat_maps = ckpt["category_maps"] if "category_maps" in ckpt else ckpt["cat_maps"]
    scaler = ckpt.get("numeric_scaler", ckpt.get("scaler"))
    cfg = ckpt["config"]
    arch = ckpt.get("arch", "lstm")
    lookback = cfg.get("lookback_days", cfg.get("lookback", 60))
    horizon = cfg.get("horizon_days", cfg.get("horizon", 16))

    panels = build_series_panels(frame, cat_maps, scaler, lookback, horizon)
    score_dt = pd.Timestamp(score_start)

    model = DeepForecaster(
        arch=arch,
        num_numeric=NUM_NUMERIC,
        cat_levels={c: len(m) for c, m in cat_maps.items()},
        embedding_dims=cfg.get("embedding_dims"),
        hidden_size=cfg.get("hidden_size", cfg.get("hidden_dim", 128)),
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.2),
        horizon=horizon,
        lookback=lookback,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # collect per-row predictions using batched chunking
    rows = []
    # We'll replicate _batched_eval but record identifiers
    panel_targets = []
    for p in panels:
        dates = pd.to_datetime(p["dates"])
        tps = [i for i, d in enumerate(dates) if pd.Timestamp(d) >= score_dt]
        panel_targets.append(tps)

    max_chunks = 0
    for tps in panel_targets:
        max_chunks = max(max_chunks, (len(tps) + 15) // 16)

    block_size = 16
    with torch.no_grad():
        for ci in range(max_chunks):
            xs = []
            cats = []
            metas = []
            for pi, p in enumerate(panels):
                tps = panel_targets[pi]
                if not tps:
                    continue
                start_idx = ci * block_size
                chunk_pos = tps[start_idx : start_idx + block_size]
                if not chunk_pos:
                    continue
                first = chunk_pos[0]
                if first - lookback < 0:
                    continue
                x = p["x"][first - lookback : first]
                xs.append(x)
                cats.append(p["cat"])
                metas.append((p, chunk_pos))
            if not xs:
                continue
            x_t = torch.from_numpy(
                np.ascontiguousarray(np.stack(xs)).astype(np.float32)
            )
            cat_t = torch.from_numpy(
                np.ascontiguousarray(np.stack(cats)).astype(np.int64)
            )
            pred_log = model(x_t, cat_t).cpu().numpy()
            for i, (p, chunk_pos) in enumerate(metas):
                pred_raw = np.clip(np.expm1(pred_log[i]), 0, None)
                block_len = len(chunk_pos)
                preds = pred_raw[:block_len]
                dates = pd.to_datetime(p["dates"])
                for j in range(block_len):
                    rows.append(
                        {
                            "date": pd.Timestamp(
                                dates.iloc[chunk_pos[j]]
                                if hasattr(dates, "iloc")
                                else dates[chunk_pos[j]]
                            ),
                            "store_nbr": p["store_nbr"],
                            "family": p["family"],
                            "prediction": float(preds[j]),
                        }
                    )
    return pd.DataFrame(rows)


def fit_predict_frozen(*args, **kwargs):
    """S1 freeze entrypoint."""
    import torch

    from deep.config import DeepConfig
    from deep.model import DeepForecaster
    from deep.train import train_one

    # Normalize call styles: tier may be first or keyword; frames may be first.
    tier = kwargs.pop("tier", None)
    fit_frame = kwargs.pop("fit_frame", None)
    predict_frame = kwargs.pop("predict_frame", None)
    config = kwargs.pop("config", None)

    positional = list(args)
    if tier is None and positional and isinstance(positional[0], str):
        tier = positional.pop(0)
    # rotate if first positional is a DataFrame (fit_frame first)
    if tier is not None and isinstance(tier, (pd.DataFrame,)):
        # tier actually fit_frame; find string in remaining
        frames = [tier, *positional]
        # if predict_frame already? handle
        # find tier string
        str_candidates = [x for x in frames if isinstance(x, str)]
        if str_candidates:
            tier = str_candidates[0]
            frames = [x for x in frames if not (isinstance(x, str) and x == tier)]
        if fit_frame is None and frames:
            fit_frame = frames.pop(0)
        if predict_frame is None and frames:
            predict_frame = frames.pop(0)
    else:
        if fit_frame is None and positional:
            fit_frame = positional.pop(0)
        if predict_frame is None and positional:
            predict_frame = positional.pop(0)
        if config is None and positional:
            config = positional.pop(0)

    if tier is None:
        tier = kwargs.pop("tier", "lstm")
    if fit_frame is None:
        fit_frame = kwargs.pop("fit_frame", None)
    if predict_frame is None:
        predict_frame = kwargs.pop("predict_frame", None)
    if config is None:
        config = kwargs.pop("config", None)

    if isinstance(config, str):
        config = None

    cfg = DeepConfig()
    if config:
        for k, v in config.items() if hasattr(config, "items") else {}:
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    arch = (
        tier.replace("deep-", "")
        if isinstance(tier, str) and tier.startswith("deep-")
        else tier
    )
    if isinstance(arch, str):
        if "rnn" in arch.lower():
            arch = "rnn"
        elif "lstm" in arch.lower():
            arch = "lstm"
        elif "gru" in arch.lower():
            arch = "gru"
    else:
        arch = "lstm"

    fit_masked = apply_store52_mask(fit_frame.copy())
    cat_maps = build_category_maps(fit_masked)
    scaler = build_scaler(fit_masked)

    result = train_one(
        arch, cfg, torch.device("cpu"), fit_masked, None, cat_maps, scaler
    )
    # train_one expects train_df; passed fit_masked (mask no-op)

    ckpt_like_state = result["model_state"]

    # infer score_start
    fit_max = pd.to_datetime(fit_frame["date"]).max()
    score_candidates = pd.to_datetime(predict_frame["date"])
    future = score_candidates[score_candidates > fit_max]
    if len(future) == 0:
        score_start = score_candidates.min()
    else:
        score_start = future.min()

    panels = build_series_panels(
        predict_frame.copy(), cat_maps, scaler, cfg.lookback_days, cfg.horizon_days
    )
    model = DeepForecaster(
        arch=arch,
        num_numeric=NUM_NUMERIC,
        cat_levels={c: len(m) for c, m in cat_maps.items()},
        embedding_dims=cfg.embedding_dims,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        horizon=cfg.horizon_days,
        lookback=cfg.lookback_days,
    )
    model.load_state_dict(ckpt_like_state)
    model.eval()

    preds_out = []
    score_dt = pd.Timestamp(score_start)
    block_size = cfg.eval_block_days
    with torch.no_grad():
        for p in panels:
            dates = pd.to_datetime(p["dates"])
            target_positions = [
                i for i, d in enumerate(dates) if pd.Timestamp(d) >= score_dt
            ]
            if not target_positions:
                continue
            max_chunks = (len(target_positions) + block_size - 1) // block_size
            for ci in range(max_chunks):
                chunk_pos = target_positions[ci * block_size : (ci + 1) * block_size]
                if not chunk_pos:
                    continue
                first = chunk_pos[0]
                if first - cfg.lookback_days < 0:
                    continue
                x = p["x"][first - cfg.lookback_days : first]
                x_t = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)
                cat_t = torch.from_numpy(p["cat"].copy()).unsqueeze(0)
                pred_log = model(x_t, cat_t).squeeze(0).cpu().numpy()
                pred_raw = np.clip(np.expm1(pred_log), 0, None)
                block_len = len(chunk_pos)
                for j in range(block_len):
                    preds_out.append(float(pred_raw[j]))

    return np.array(preds_out, dtype=float)
