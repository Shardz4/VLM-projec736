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

def wilcoxon_signed_rank_test(
    baseline: Union[np.ndarray, Sequence[float], torch.Tensor],
    extension: Union[np.ndarray, Sequence[float], torch.Tensor],
)-> Dict[str, float]:

    b = _to_numpy(baseline)
    e = _to_numpy(extension)
    if len(b) != len(e):
        raise ValueError("Sample sizes must match.")

    diff = e - b
    if np.allclose(diff, 0.0):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "median_diff": 0.0,
            "is_significant_05": False,
        }
    res = stats.wilcoxon(e, b, zero_method="wilcox", alternative="two-sized")
    return {
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "median_diff": float(np.median(diff)),
        "is_significant_05": bool(res.pvalue < 0.05),
    }

def cohens_d(
    baseline: Union[np.ndarray, Sequence[float], torch.Tensor],
    extension: Union[np.ndarray, Sequence[float], torch.Tensor],
) -> float:
    b = _to_numpy(baseline)
    e = _to_numpy(extension)
    diff = e - b
    s_diff = np.std(diff, ddof=1)
    if s_diff == 0.0:
        return 0.0
    return float(np.mean(diff) / s_diff)


def bootstrap_ci(
    data: Union[np.ndarray, Sequence[float], torch.Tensor],
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute empirical bootstrap confidence interval for the mean.
    Args:
        data: 1D array of sample values (e.g., test accuracies or predictions).
        n_resamples: Number of bootstrap resamples (default: 10,000).
        ci: Confidence level (default: 0.95 for 95% CI).
        seed: Random seed for deterministic reproducibility.
    Returns:
        Dict with mean, ci_lower, ci_upper, standard_error.
    """
    arr = _to_numpy(data)
    n = len(arr)
    if n == 0:
        raise ValueError("Cannot bootstrap empty array.")
    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n, size=(n_resamples, n))
    boot_means = np.mean(arr[boot_indices], axis=1)
    alpha = 1.0 - ci
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_lower = float(np.percentile(boot_means, lower_pct))
    ci_upper = float(np.percentile(boot_means, upper_pct))
    se = float(np.std(boot_means, ddof=1))
    return {
        "mean": float(np.mean(arr)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "standard_error": se,
        "confidence_level": ci,
    }

def bootstrap_difference_ci(
    baseline: Union[np.ndarray, Sequence[float], torch.Tensor],
    extension: Union[np.ndarray, Sequence[float], torch.Tensor],
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute bootstrap CI on paired difference (extension - baseline).
    If [ci_lower, ci_upper] strictly excludes 0, the improvement is statistically significant.
    """
    b = _to_numpy(baseline)
    e = _to_numpy(extension)
    diff = e - b
    res = bootstrap_ci(diff, n_resamples=n_resamples, ci=ci, seed=seed)
    res["excludes_zero"] = bool(res["ci_lower"] > 0.0 or res["ci_upper"] < 0.0)
    return res

def compute_statistical_report(
    baseline_scores: Union[np.ndarray, Sequence[float]],
    extension_scores: Union[np.ndarray, Sequence[float]],
    metric_name: str = "Accuracy",
) -> Dict:
    """Generate a complete statistical significance report comparing two models.
    Args:
        baseline_scores: Array of metric values across seeds.
        extension_scores: Array of metric values across seeds.
        metric_name: Name of metric being analyzed.
    Returns:
        Nested dict with descriptive stats, t-test, Wilcoxon, effect size, and bootstrap CIs.
    """
    b = _to_numpy(baseline_scores)
    e = _to_numpy(extension_scores)
    b_mean, b_std = float(np.mean(b)), float(np.std(b, ddof=1) if len(b) > 1 else 0.0)
    e_mean, e_std = float(np.mean(e)), float(np.std(e, ddof=1) if len(e) > 1 else 0.0)
    t_res = paired_t_test(b, e) if len(b) > 1 else {}
    w_res = wilcoxon_signed_rank_test(b, e) if len(b) > 1 else {}
    d_val = cohens_d(b, e) if len(b) > 1 else 0.0
    ci_res = bootstrap_difference_ci(b, e)
    return {
        "metric": metric_name,
        "n_seeds": len(b),
        "baseline_summary": f"{b_mean:.4f} ± {b_std:.4f}",
        "extension_summary": f"{e_mean:.4f} ± {e_std:.4f}",
        "delta_mean": float(e_mean - b_mean),
        "relative_gain_pct": float(((e_mean - b_mean) / b_mean) * 100.0) if b_mean != 0 else 0.0,
        "t_test": t_res,
        "wilcoxon_test": w_res,
        "cohens_d": d_val,
        "bootstrap_diff_ci": ci_res,
    }
