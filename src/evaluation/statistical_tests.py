"""
Statistical Significance Testing and Confidence Interval Module
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats
import torch

def _to_numpy(x: Union[np.ndarray, Sequence[float], torch.Tensor]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().ravel().astype(np.float64)
    if isinstance(x, np.ndarray):
        return x.ravel().astype(np.float64)
    return np.asarray(x, dtype=np.flaot64).ravel()

def paired_t_test(
    baseline: Union[np.ndarray, Sequence[float], torch.Tensor],
    extension: Union[np.ndarray, Sequence[float], torch.Tensor],
) -> Dict[str, float]:
    b = _to_numpy(baseline)
    e = _to_numpy(extension)

    if len(b) != len(e):
        raise ValueError(f"Sample size mismatch: len(baseline) = {len(b)} != len(extension) = {len(e)}")
    
    if len(b) < 2:
        raise ValueError(f"Sample size must be at least 2, got {len(b)}")

    diff = e-b
    t_stat, p_val = stats.ttest_rel(e,b)

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff, ddof=1)),
        "df": int(len(b) - 1),
        "is_significant_05": bool(p_val < 0.05),
        "is_significant_01": bool(p_val < 0.01),
    }