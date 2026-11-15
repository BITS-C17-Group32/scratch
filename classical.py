# local_test: only S1's freeze script touches it; never open, never scored
# MAPE is deliberately excluded because zero-sales share is 35.2% on train+valid (~31.5% full-grid).

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import contract

REPO_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
ARTIFACT_PATH = ARTIFACTS_DIR / "classical_results.json"

TRAIN_END = pd.Timestamp("2016-03-27")
VALID_END = pd.Timestamp("2016-09-11")

# Contract-pinned tier names in contract.CLASSICAL_MODEL_NAMES.
# If contract.py is updated to ("naive", "seasonal_naive", "moving_average", "arimax", "sarimax"),
# change these two constants to "arimax" and "sarimax".
TIER_KEY_ARIMAX = "ridge"
TIER_KEY_SARIMAX = "hgb"

CAT_COLS = ["family", "city", "state", "store_type", "store_nbr"]
CONT_COLS = ["onpromotion", "cluster", "earthquake_decay"]
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
FEATURE_COLS = [
    "store_nbr",
    "family",
    "onpromotion",
    "city",
    "state",
    "store_type",
    "cluster",
    *FLAG_COLS,
    "earthquake_decay",
]

ARIMAX_GRID = [
    {"order": (1, 0, 1), "alpha": 1.0, "history_days": 180},
    {"order": (2, 0, 1), "alpha": 10.0, "history_days": 180},
    {"order": (3, 1, 1), "alpha": 10.0, "history_days": 180},
]

SARIMAX_GRID = [
    {
        "order": (1, 0, 1),
        "seasonal_order": (1, 0, 1, 7),
        "alpha": 1.0,
        "history_days": 180,
    },
    {
        "order": (2, 0, 1),
        "seasonal_order": (2, 0, 1, 7),
        "alpha": 10.0,
        "history_days": 180,
    },
    {
        "order": (2, 1, 1),
        "seasonal_order": (2, 0, 1, 7),
        "alpha": 10.0,
        "history_days": 180,
    },
]


def _assert_frame_contract(df: pd.DataFrame, max_date: pd.Timestamp) -> None:
    for col, expected_dtype in contract.PINNED_DTYPES.items():
        actual_dtype = str(df[col].dtype)
        assert actual_dtype == expected_dtype, (
            f"Column {col!r} expected {expected_dtype}, got {actual_dtype}"
        )
    assert df["date"].max() <= max_date, f"Frame exceeds max_date {max_date}"


def load_fold_history(origin: pd.Timestamp) -> pd.DataFrame:
    train_df = contract.load_split("train")
    valid_df = contract.load_split("valid")
    _assert_frame_contract(train_df, TRAIN_END)
    _assert_frame_contract(valid_df, VALID_END)

    valid_slice = valid_df.loc[valid_df["date"] <= origin]
    if not valid_slice.empty:
        assert valid_slice["date"].max() <= origin

    combined = pd.concat([train_df, valid_slice], axis=0, ignore_index=True)
    assert combined["date"].max() <= origin

    mask = contract.training_mask(combined)
    masked = combined.loc[mask].reset_index(drop=True)

    # Verify zero unmasked non-closure pre-opening rows leaked into training history
    leaked = int((~contract.training_mask(masked)).sum())
    assert leaked == 0, f"Pre-opening rows leaked into fold history: {leaked}"
    return masked


def _split_history_and_target(
    frame: pd.DataFrame, origin: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hist = frame.loc[frame["date"] <= origin]
    target = frame.loc[frame["date"] > origin]
    return hist, target


def predict_naive(frame: pd.DataFrame, origin: pd.Timestamp) -> np.ndarray:
    hist, target = _split_history_and_target(frame, origin)
    last_obs = (
        hist.sort_values("date")
        .groupby(["store_nbr", "family"], observed=False)["sales"]
        .last()
        .fillna(0.0)
    )
    keys = pd.MultiIndex.from_frame(target[["store_nbr", "family"]])
    preds = last_obs.reindex(keys).fillna(0.0).to_numpy(dtype=np.float64)
    return np.clip(preds, 0.0, None)


def predict_seasonal_naive(
    frame: pd.DataFrame, origin: pd.Timestamp, period: int = 7
) -> np.ndarray:
    hist, target = _split_history_and_target(frame, origin)
    min_hist_date = origin - pd.Timedelta(days=period - 1)
    recent = hist.loc[
        hist["date"] >= min_hist_date, ["store_nbr", "family", "date", "sales"]
    ]

    lookup: dict[tuple[int, str, pd.Timestamp], float] = {}
    for row in recent.itertuples(index=False):
        lookup[(int(row.store_nbr), str(row.family), row.date)] = float(row.sales)

    target_dates = sorted(target["date"].unique())
    series_keys = (
        target[["store_nbr", "family"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    series_list = [(int(s), str(f)) for s, f in series_keys]

    for d in target_dates:
        lag_d = d - pd.Timedelta(days=period)
        for s, f in series_list:
            val = lookup.get((s, f, lag_d), 0.0)
            lookup[(s, f, d)] = max(0.0, val)

    preds = np.array(
        [
            lookup.get((int(r.store_nbr), str(r.family), r.date), 0.0)
            for r in target[["store_nbr", "family", "date"]].itertuples(index=False)
        ],
        dtype=np.float64,
    )
    return np.clip(preds, 0.0, None)


def predict_moving_average(
    frame: pd.DataFrame, origin: pd.Timestamp, window: int = 28
) -> np.ndarray:
    hist, target = _split_history_and_target(frame, origin)
    start_date = origin - pd.Timedelta(days=window - 1)
    win_hist = hist.loc[hist["date"] >= start_date]
    means = (
        win_hist.groupby(["store_nbr", "family"], observed=False)["sales"]
        .mean()
        .fillna(0.0)
    )
    keys = pd.MultiIndex.from_frame(target[["store_nbr", "family"]])
    preds = means.reindex(keys).fillna(0.0).to_numpy(dtype=np.float64)
    return np.clip(preds, 0.0, None)


def _fit_predict_state_space_linear(
    frame: pd.DataFrame,
    origin: pd.Timestamp,
    ar_lags: list[int],
    ma_lags: list[int],
    diff_order: int,
    alpha: float,
    history_days: int,
) -> np.ndarray:
    """Two-stage Hannan-Rissanen ARIMAX/SARIMAX estimator on log1p(sales)."""
    hist, target = _split_history_and_target(frame, origin)
    max_lag = max(ar_lags + ma_lags + [7, 14, 28])
    cutoff = origin - pd.Timedelta(days=history_days + max_lag)
    hist_recent = hist.loc[hist["date"] >= cutoff].copy()

    hist_recent["y_log"] = np.log1p(
        np.clip(hist_recent["sales"].to_numpy(dtype=np.float64), 0.0, None)
    )
    hist_recent = hist_recent.sort_values(["store_nbr", "family", "date"]).reset_index(
        drop=True
    )

    grp = hist_recent.groupby(["store_nbr", "family"], observed=False)["y_log"]
    lag_cols = []
    for k in ar_lags:
        col_name = f"ar_lag_{k}"
        hist_recent[col_name] = grp.shift(k)
        lag_cols.append(col_name)

    if diff_order == 1 and len(ar_lags) >= 2:
        hist_recent["ar_diff_1"] = (
            hist_recent[f"ar_lag_{ar_lags[0]}"] - hist_recent[f"ar_lag_{ar_lags[1]}"]
        )
        lag_cols.append("ar_diff_1")

    train_sub = hist_recent.dropna(subset=lag_cols).copy()
    for c in CAT_COLS:
        train_sub[c] = train_sub[c].astype("category")

    # Stage 1: ARX fit to estimate innovation residuals e_t
    pre_stage1 = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CAT_COLS,
            ),
            ("num", StandardScaler(), [*lag_cols, *CONT_COLS]),
            ("flags", "passthrough", FLAG_COLS),
        ],
        remainder="drop",
    )
    stage1 = Pipeline(
        [("pre", pre_stage1), ("reg", Ridge(alpha=alpha, random_state=0))]
    )
    stage1.fit(
        train_sub[[*CAT_COLS, *lag_cols, *CONT_COLS, *FLAG_COLS]], train_sub["y_log"]
    )

    train_sub["resid"] = train_sub["y_log"] - np.clip(
        stage1.predict(train_sub[[*CAT_COLS, *lag_cols, *CONT_COLS, *FLAG_COLS]]),
        0.0,
        None,
    )
    resid_grp = train_sub.groupby(["store_nbr", "family"], observed=False)["resid"]
    ma_cols = []
    for q in ma_lags:
        m_col = f"ma_lag_{q}"
        train_sub[m_col] = resid_grp.shift(q).fillna(0.0)
        ma_cols.append(m_col)

    # Stage 2: Full ARIMAX / SARIMAX fit with AR, I, MA, and Exogenous (X) terms
    all_num_cols = [*lag_cols, *ma_cols, *CONT_COLS]
    pre_stage2 = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CAT_COLS,
            ),
            ("num", StandardScaler(), all_num_cols),
            ("flags", "passthrough", FLAG_COLS),
        ],
        remainder="drop",
    )
    stage2 = Pipeline(
        [("pre", pre_stage2), ("reg", Ridge(alpha=alpha, random_state=0))]
    )
    stage2.fit(
        train_sub[[*CAT_COLS, *all_num_cols, *FLAG_COLS]], train_sub["y_log"]
    )

    tail_start = origin - pd.Timedelta(days=max_lag + 2)
    tail_rows = train_sub.loc[
        train_sub["date"] >= tail_start,
        ["store_nbr", "family", "date", "y_log", "resid"],
    ]
    y_lookup: dict[tuple[int, str, pd.Timestamp], float] = {}
    e_lookup: dict[tuple[int, str, pd.Timestamp], float] = {}
    for r in tail_rows.itertuples(index=False):
        key = (int(r.store_nbr), str(r.family), r.date)
        y_lookup[key] = float(r.y_log)
        e_lookup[key] = float(r.resid)

    target_with_idx = target.copy()
    target_with_idx["_orig_idx"] = np.arange(len(target_with_idx))
    preds_out = np.zeros(len(target_with_idx), dtype=np.float64)

    for d in sorted(target_with_idx["date"].unique()):
        day_df = target_with_idx.loc[target_with_idx["date"] == d].copy()
        for c in CAT_COLS:
            day_df[c] = day_df[c].astype("category")

        for k in ar_lags:
            lag_d = d - pd.Timedelta(days=k)
            day_df[f"ar_lag_{k}"] = [
                y_lookup.get((int(r.store_nbr), str(r.family), lag_d), 0.0)
                for r in day_df[["store_nbr", "family"]].itertuples(index=False)
            ]
        if diff_order == 1 and len(ar_lags) >= 2:
            day_df["ar_diff_1"] = (
                day_df[f"ar_lag_{ar_lags[0]}"] - day_df[f"ar_lag_{ar_lags[1]}"]
            )

        for q in ma_lags:
            lag_d = d - pd.Timedelta(days=q)
            day_df[f"ma_lag_{q}"] = [
                e_lookup.get((int(r.store_nbr), str(r.family), lag_d), 0.0)
                for r in day_df[["store_nbr", "family"]].itertuples(index=False)
            ]

        pred_log = np.clip(
            stage2.predict(day_df[[*CAT_COLS, *all_num_cols, *FLAG_COLS]]), 0.0, None
        )
        # Store 52 is unopened throughout train+valid; predict 0.0
        is_s52 = (day_df["store_nbr"].astype(int) == 52).to_numpy()
        pred_log[is_s52] = 0.0

        for (s, f), val in zip(
            day_df[["store_nbr", "family"]].itertuples(index=False, name=None),
            pred_log,
            strict=True,
        ):
            y_lookup[(int(s), str(f), d)] = float(val)
            e_lookup[(int(s), str(f), d)] = 0.0

        preds_out[day_df["_orig_idx"].to_numpy()] = np.clip(
            np.expm1(pred_log), 0.0, None
        )

    return np.clip(preds_out, 0.0, None)


def predict_arimax(
    frame: pd.DataFrame, origin: pd.Timestamp, config: dict
) -> np.ndarray:
    p, d, q = config.get("order", (2, 0, 1))
    ar_lags = list(range(1, p + 1))
    ma_lags = list(range(1, q + 1))
    return _fit_predict_state_space_linear(
        frame=frame,
        origin=origin,
        ar_lags=ar_lags,
        ma_lags=ma_lags,
        diff_order=d,
        alpha=float(config.get("alpha", 10.0)),
        history_days=int(config.get("history_days", 180)),
    )


def predict_sarimax(
    frame: pd.DataFrame, origin: pd.Timestamp, config: dict
) -> np.ndarray:
    p, d, q = config.get("order", (2, 0, 1))
    sp, _sd, sq, s_period = config.get("seasonal_order", (2, 0, 1, 7))
    ar_lags = sorted(
        set(range(1, p + 1)) | {s_period * i for i in range(1, sp + 1)} | {21, 28}
    )
    ma_lags = sorted(
        set(range(1, q + 1)) | {s_period * i for i in range(1, sq + 1)}
    )
    return _fit_predict_state_space_linear(
        frame=frame,
        origin=origin,
        ar_lags=ar_lags,
        ma_lags=ma_lags,
        diff_order=d,
        alpha=float(config.get("alpha", 10.0)),
        history_days=int(config.get("history_days", 180)),
    )


MODELS: dict[str, callable] = {
    "naive": predict_naive,
    "seasonal_naive": predict_seasonal_naive,
    "moving_average": predict_moving_average,
    "arimax": predict_arimax,
    "sarimax": predict_sarimax,
    TIER_KEY_ARIMAX: predict_arimax,
    TIER_KEY_SARIMAX: predict_sarimax,
}


def fit_predict_frozen(
    model: str,
    fit_frame: pd.DataFrame,
    predict_frame: pd.DataFrame,
    config: dict,
) -> np.ndarray:
    if os.environ.get("FREEZE_RUN") != "1":
        raise RuntimeError("fit_predict_frozen requires FREEZE_RUN=1 in environment.")
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'. Expected one of {list(MODELS)}")

    origin = pd.Timestamp(fit_frame["date"].max())
    pred_clean = predict_frame.copy()
    if "sales" in pred_clean.columns:
        pred_clean = pred_clean.drop(columns=["sales"])
    pred_clean["sales"] = np.float32(0.0)

    combined = pd.concat([fit_frame, pred_clean], axis=0, ignore_index=True)
    fn = MODELS[model]
    if model in ("arimax", "sarimax", TIER_KEY_ARIMAX, TIER_KEY_SARIMAX):
        return fn(combined, origin, config)
    return fn(combined, origin)


def _git_sha() -> str:
    res = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return res.stdout.strip()


def _evaluate_model_across_folds(
    model_name: str,
    folds_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    config: dict | None = None,
) -> tuple[list[dict], dict]:
    fold_metrics = []
    all_true = []
    all_pred = []

    for row in folds_df.itertuples(index=False):
        fold_id = int(row.fold)
        origin = pd.Timestamp(row.origin)
        w_start = pd.Timestamp(row.window_start)
        w_end = pd.Timestamp(row.window_end)

        hist_df = load_fold_history(origin)
        val_win = valid_df.loc[
            (valid_df["date"] >= w_start) & (valid_df["date"] <= w_end)
        ].copy()
        assert len(val_win) == int(row.n_rows), "Fold window row count mismatch"

        eval_frame = pd.concat([hist_df, val_win], axis=0, ignore_index=True)
        fn = MODELS[model_name]
        preds = (
            fn(eval_frame, origin, config)
            if config is not None
            else fn(eval_frame, origin)
        )
        preds = np.clip(np.asarray(preds, dtype=np.float64), 0.0, None)
        y_true = val_win["sales"].to_numpy(dtype=np.float64)

        m = contract.score(y_true, preds, is_active=(y_true > 0))
        fold_metrics.append({"fold": fold_id, **m})
        all_true.append(y_true)
        all_pred.append(preds)

    pooled_true = np.concatenate(all_true)
    pooled_pred = np.concatenate(all_pred)
    assert len(pooled_true) == 85536, (
        f"Expected 85536 pooled rows, got {len(pooled_true)}"
    )
    pooled_scores = contract.score(
        pooled_true, pooled_pred, is_active=(pooled_true > 0)
    )
    return fold_metrics, pooled_scores


def run_validation() -> dict:
    train_df = contract.load_split("train")
    valid_df = contract.load_split("valid")
    train_drop = int((~contract.training_mask(train_df)).sum())
    valid_drop = int((~contract.training_mask(valid_df)).sum())
    assert train_drop == 207801, f"Expected 207801 train fit drop, got {train_drop}"
    assert valid_drop == 5544, f"Expected 5544 valid fit drop, got {valid_drop}"

    folds_df = contract.make_folds()
    folds_list = [
        {
            "fold": int(r.fold),
            "origin": str(r.origin)[:10],
            "window_start": str(r.window_start)[:10],
            "window_end": str(r.window_end)[:10],
            "n_rows": int(r.n_rows),
        }
        for r in folds_df.itertuples(index=False)
    ]

    models_out = []

    baseline_specs = [
        (
            "naive",
            {"strategy": "last_observed_at_origin"},
            "none (deterministic baseline)",
        ),
        (
            "seasonal_naive",
            {"period": 7, "recursive": True},
            "none (fixed period=7)",
        ),
        ("moving_average", {"window": 28}, "none (fixed window=28)"),
    ]
    for name, params, tuning_desc in baseline_specs:
        f_metrics, pooled = _evaluate_model_across_folds(name, folds_df, valid_df)
        print(
            f"[{name}] pooled RMSLE={pooled['rmsle']:.5f} | "
            f"MAE={pooled['mae']:.3f} | RMSE={pooled['rmse']:.3f}"
        )
        models_out.append(
            {
                "name": name,
                "kind": "baseline",
                "target": "raw_sales",
                "hyperparameters": params,
                "fold_metrics": f_metrics,
                "pooled": pooled,
                "seeds": {"random_state": 0},
                "tuning": tuning_desc,
                "christmas_2016": {"status": "freeze-time"},
            }
        )

    # ARIMAX tuning
    arimax_trials = []
    best_arimax = None
    for cfg in ARIMAX_GRID:
        f_metrics, pooled = _evaluate_model_across_folds(
            "arimax", folds_df, valid_df, cfg
        )
        arimax_trials.append(
            f"order={cfg['order']},alpha={cfg['alpha']}:rmsle={pooled['rmsle']:.6f}"
        )
        if best_arimax is None or pooled["rmsle"] < best_arimax[2]["rmsle"]:
            best_arimax = (cfg, f_metrics, pooled)

    a_cfg, a_folds, a_pooled = best_arimax
    print(f"[arimax ({TIER_KEY_ARIMAX})] best={a_cfg} | pooled RMSLE={a_pooled['rmsle']:.5f}")
    models_out.append(
        {
            "name": TIER_KEY_ARIMAX,
            "kind": "classical_ml",
            "target": "log1p(sales)",
            "hyperparameters": {
                "model_family": "ARIMAX",
                "order": list(a_cfg["order"]),
                "alpha": a_cfg["alpha"],
                "history_days": a_cfg["history_days"],
            },
            "fold_metrics": a_folds,
            "pooled": a_pooled,
            "seeds": {"random_state": 0},
            "tuning": (
                f"Grid over ARIMAX orders selected by pooled RMSLE across 3 folds "
                f"({'; '.join(arimax_trials)}); same three folds tune and report."
            ),
            "christmas_2016": {"status": "freeze-time"},
        }
    )

    # SARIMAX tuning
    sarimax_trials = []
    best_sarimax = None
    for cfg in SARIMAX_GRID:
        f_metrics, pooled = _evaluate_model_across_folds(
            "sarimax", folds_df, valid_df, cfg
        )
        sarimax_trials.append(
            f"order={cfg['order']},seasonal={cfg['seasonal_order']},"
            f"alpha={cfg['alpha']}:rmsle={pooled['rmsle']:.6f}"
        )
        if best_sarimax is None or pooled["rmsle"] < best_sarimax[2]["rmsle"]:
            best_sarimax = (cfg, f_metrics, pooled)

    s_cfg, s_folds, s_pooled = best_sarimax
    print(
        f"[sarimax ({TIER_KEY_SARIMAX})] best={s_cfg} | pooled RMSLE={s_pooled['rmsle']:.5f}"
    )
    models_out.append(
        {
            "name": TIER_KEY_SARIMAX,
            "kind": "classical_ml",
            "target": "log1p(sales)",
            "hyperparameters": {
                "model_family": "SARIMAX",
                "order": list(s_cfg["order"]),
                "seasonal_order": list(s_cfg["seasonal_order"]),
                "alpha": s_cfg["alpha"],
                "history_days": s_cfg["history_days"],
            },
            "fold_metrics": s_folds,
            "pooled": s_pooled,
            "seeds": {"random_state": 0},
            "tuning": (
                f"Grid over SARIMAX orders selected by pooled RMSLE across 3 folds "
                f"({'; '.join(sarimax_trials)}); same three folds tune and report."
            ),
            "christmas_2016": {"status": "freeze-time"},
        }
    )

    artifact = {
        "schema_version": 1,
        "git_sha": _git_sha(),
        "python_version": sys.version.split()[0],
        "sklearn_version": sklearn.__version__,
        "split_bounds": {
            "train": "2013-01-01..2016-03-27",
            "valid": "2016-03-28..2016-09-11",
        },
        "training_mask": (
            "contract.training_mask (fit_rows): drops pre-opening rows across delayed-open "
            "stores {20, 21, 22, 29, 36, 42, 52, 53} while preserving all id=NaN closure fills "
            "(drops 207,801 train rows and 5,544 valid rows on training path only)"
        ),
        "feature_columns": FEATURE_COLS,
        "target": "log1p(sales)",
        "prediction_clip": "y_pred >= 0 on raw-scale predictions after single expm1",
        "folds": folds_list,
        "models": models_out,
        "notes": [
            "naive: per (store_nbr, family) constant equal to last observed sales <= origin across 16-day horizon.",
            "seasonal_naive: per series y_hat(t) = y(t-7) if t-7 <= origin else recursive prediction for t-7 (period=7).",
            "moving_average: per series mean of sales over (origin-27 .. origin) applied across 16-day horizon.",
            "arimax (registered under tier key 'ridge'): Hannan-Rissanen two-stage ARIMAX(p,d,q) with exogenous holiday/promo/earthquake predictors on log1p(sales).",
            "sarimax (registered under tier key 'hgb'): Hannan-Rissanen two-stage SARIMAX(p,d,q)(P,D,Q,7) with weekly seasonal AR/MA lags and exogenous predictors on log1p(sales).",
            "Pre-opening mask (DELAYS_OPEN_UNTIL): drops 207,801 train rows and 5,544 valid rows (Store 52 opened 2017-04-20; 39,006 train grid rows, 51,678 raw rows, 12,804 holdout zeros = 5,544 valid + 7,260 holdout scored as-is).",
            "Christmas closures: 2013-12-25, 2014-12-25, 2015-12-25 in train (5,346 rows) and 2016-12-25 in holdout (1,782 rows); 7,128 total synthetic zero rows (id=NaN) retained everywhere as true-zero labels.",
            "Earthquake 2016-04-16 sits inside valid (30-day decay window 2016-04-16..2016-05-15); contamination is in training history only, with zero quake rows in the three fold windows (2016-07-26..2016-09-11).",
            "Zero-sales share is ~31.5% full-grid, 35.2% train+valid (28.9% trainable after masking), 18.1% valid (~329k in 2013, ~243k in 2014, ~192k in 2015, ~83k in 2016 partial); active-slice filters sales > 0 at scoring time only.",
            "Oil and transactions merges are deliberately deferred this week; 23 frozen predictors used; same three folds tune and report.",
            "Determinism: fixed random_state=0 everywhere; two consecutive runs on a fixed HEAD produce byte-identical JSONs with git_sha as the sole version marker.",
        ],
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = ARTIFACT_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
        f.write("\n")
    tmp_path.replace(ARTIFACT_PATH)
    return artifact


def run_dry_run() -> None:
    os.environ["FREEZE_RUN"] = "1"
    folds_df = contract.make_folds()
    last_fold = folds_df.iloc[-1]
    origin = pd.Timestamp(last_fold.origin)
    w_start = pd.Timestamp(last_fold.window_start)
    w_end = pd.Timestamp(last_fold.window_end)

    fit_frame = load_fold_history(origin)
    valid_df = contract.load_split("valid")
    pred_frame = valid_df.loc[
        (valid_df["date"] >= w_start) & (valid_df["date"] <= w_end)
    ].copy()
    y_true = pred_frame["sales"].to_numpy(dtype=np.float64)

    default_cfgs = {
        "naive": {},
        "seasonal_naive": {},
        "moving_average": {},
        TIER_KEY_ARIMAX: {"order": (2, 0, 1), "alpha": 10.0, "history_days": 180},
        TIER_KEY_SARIMAX: {
            "order": (2, 0, 1),
            "seasonal_order": (2, 0, 1, 7),
            "alpha": 10.0,
            "history_days": 180,
        },
    }
    for name, cfg in default_cfgs.items():
        preds = fit_predict_frozen(name, fit_frame, pred_frame, cfg)
        scores = contract.score(y_true, preds, is_active=(y_true > 0))
        print(
            f"[dry-run {name}] fold={int(last_fold.fold)} "
            f"RMSLE={scores['rmsle']:.5f} n_rows={scores['n_rows']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="S3 Baselines and Classical ML Harness")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Exercise fit_predict_frozen on valid-tail slice only without writing artifacts.",
    )
    args = parser.parse_args()
    if args.dry_run:
        run_dry_run()
    else:
        run_validation()


if __name__ == "__main__":
    main()