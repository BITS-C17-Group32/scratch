import sys
from pathlib import Path
from typing import Any

import numpy as np

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute RMSLE, MAE, RMSE on raw sales units using float64 math.

    Predictions are assumed already clipped to >= 0 (clipping happens once
    at model output after expm1). No clipping is performed here.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    try:
        import contract  # type: ignore

        res = contract.score(y_true_arr, y_pred_arr)
        if isinstance(res, dict):
            # normalize keys
            out = {}
            mapping = {
                "rmsle": "rmsle",
                "mae": "mae",
                "rmse": "rmse",
                "rmsle_active": "rmsle_active",
                "mae_active": "mae_active",
                "rmse_active": "rmse_active",
                "n_rows": "n_rows",
                "n_rows_active": "n_rows_active",
                "n_active": "n_rows_active",
            }
            for k, v in mapping.items():
                if k in res:
                    out[v] = res[k]
            if "rmsle" in out:
                # ensure all keys
                out.setdefault("mae", 0.0)
                out.setdefault("rmse", 0.0)
                out.setdefault("rmsle_active", 0.0)
                out.setdefault("mae_active", 0.0)
                out.setdefault("rmse_active", 0.0)
                out.setdefault("n_rows", len(y_true_arr))
                out.setdefault("n_rows_active", int(np.sum(y_true_arr > 0)))
                return out
    except (ImportError, ModuleNotFoundError, Exception):
        pass

    err = y_pred_arr - y_true_arr
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    log_pred = np.log1p(np.clip(y_pred_arr, 0, None))
    log_true = np.log1p(np.clip(y_true_arr, 0, None))
    rmsle = float(np.sqrt(np.mean((log_pred - log_true) ** 2)))

    active_mask = y_true_arr > 0
    if np.any(active_mask):
        yt_act = y_true_arr[active_mask]
        yp_act = y_pred_arr[active_mask]
        err_act = yp_act - yt_act
        rmse_act = float(np.sqrt(np.mean(err_act**2)))
        mae_act = float(np.mean(np.abs(err_act)))
        rmsle_act = float(
            np.sqrt(
                np.mean(
                    (
                        np.log1p(np.clip(yp_act, 0, None))
                        - np.log1p(np.clip(yt_act, 0, None))
                    )
                    ** 2
                )
            )
        )
    else:
        rmse_act, mae_act, rmsle_act = 0.0, 0.0, 0.0

    return {
        "rmsle": rmsle,
        "mae": mae,
        "rmse": rmse,
        "rmsle_active": rmsle_act,
        "mae_active": mae_act,
        "rmse_active": rmse_act,
        "n_rows": len(y_true_arr),
        "n_rows_active": int(np.sum(active_mask)),
    }
