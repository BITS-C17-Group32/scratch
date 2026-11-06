import sys
from pathlib import Path
from typing import Any, Dict
import numpy as np

# Add repository root to path for top-level module resolution
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute RMSLE, MAE, and RMSE on raw sales units using float64 math."""
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, None)

    try:
        import contract # type: ignore

        return contract.score(y_true_arr, y_pred_arr)
    except (ImportError, ModuleNotFoundError):
        pass

    err = y_pred_arr - y_true_arr
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))

    log_pred = np.log1p(y_pred_arr)
    log_true = np.log1p(y_true_arr)
    rmsle = float(np.sqrt(np.mean((log_pred - log_true) ** 2)))

    active_mask = y_true_arr > 0
    if np.any(active_mask):
        yt_act = y_true_arr[active_mask]
        yp_act = y_pred_arr[active_mask]
        err_act = yp_act - yt_act
        rmse_act = float(np.sqrt(np.mean(err_act**2)))
        mae_act = float(np.mean(np.abs(err_act)))
        rmsle_act = float(
            np.sqrt(np.mean((np.log1p(yp_act) - np.log1p(yt_act)) ** 2))
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
        "n_rows": int(len(y_true_arr)),
        "n_rows_active": int(np.sum(active_mask)),
    }