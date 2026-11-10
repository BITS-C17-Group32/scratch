import sys
from pathlib import Path
from typing import Any

import numpy as np

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute metrics using contract.score only. No re-definition."""
    import contract  # type: ignore

    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    return contract.score(y_true_arr, y_pred_arr)
