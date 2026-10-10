"""Unit tests and numerical verification for Step 13 Metric Comparison Study."""

from __future__ import annotations
import math
import sys
from pathlib import Path
from typing import Callable, List

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_metric_comparison import (
    METRICS_INFO,
    METRIC_NAMES,
    compute_correlations,
    compute_metrics_for_predictions,
    compute_ranks,
    find_ranking_inversions,
)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_results: List[tuple] = []


def test(name: str, fn: Callable) -> None:
    try:
        fn()
        _results.append((name, True, None))
        print(f"  [{PASS}]  {name}")
    except Exception as e:
        import traceback
        _results.append((name, False, traceback.format_exc()))
        print(f"  [{FAIL}]  {name}")
        print(f"           {e}")


print("\n" + "=" * 65)
print("STEP 13: Metric Comparison Study Verification Suite")
print("=" * 65)


def _test_compute_ranks_directionality():
    # Model 1 has high Acc and high MAE (bad MAE)
    # Model 2 has low Acc and low MAE (good MAE)
    scores_data = {
        "Accuracy": [90.0, 40.0],
        "MAE": [2.5, 0.5],
        "Macro_F1": [85.0, 35.0],
        "ECE": [0.15, 0.02],
        "Recall@1": [50.0, 10.0],
        "Recall@5": [80.0, 30.0],
        "Median_Rank": [1.0, 50.0],
        "Modality_Gap": [1.2, 0.3],
        "CMA": [0.8, 0.2],
    }
    df = pd.DataFrame(scores_data, index=["ModelA", "ModelB"])
    ranks = compute_ranks(df)

    # Accuracy: ModelA (90.0) should be Rank 1, ModelB (40.0) Rank 2
    assert ranks.loc["ModelA", "Accuracy"] == 1
    assert ranks.loc["ModelB", "Accuracy"] == 2

    # MAE (lower is better): ModelB (0.5) should be Rank 1, ModelA (2.5) Rank 2
    assert ranks.loc["ModelB", "MAE"] == 1
    assert ranks.loc["ModelA", "MAE"] == 2

    # ECE (lower is better): ModelB (0.02) should be Rank 1, ModelA (0.15) Rank 2
    assert ranks.loc["ModelB", "ECE"] == 1
    assert ranks.loc["ModelA", "ECE"] == 2


def _test_kendall_spearman_properties():
    # Create 5 synthetic models across 3 metrics
    ranks_data = {
        "Metric1": [1, 2, 3, 4, 5],
        "Metric2": [1, 2, 3, 4, 5],  # perfectly aligned with Metric1
        "Metric3": [5, 4, 3, 2, 1],  # perfectly inverted with Metric1
    }
    df = pd.DataFrame(ranks_data, index=[f"M{i}" for i in range(1, 6)])
    k_mat, s_mat = compute_correlations(df)

    # Diagonals should be 1.0
    for col in df.columns:
        assert math.isclose(k_mat.loc[col, col], 1.0, abs_tol=1e-5)
        assert math.isclose(s_mat.loc[col, col], 1.0, abs_tol=1e-5)

    # Metric1 vs Metric2: correlation = +1.0
    assert math.isclose(k_mat.loc["Metric1", "Metric2"], 1.0, abs_tol=1e-4)
    assert math.isclose(s_mat.loc["Metric1", "Metric2"], 1.0, abs_tol=1e-4)

    # Metric1 vs Metric3: correlation = -1.0
    assert math.isclose(k_mat.loc["Metric1", "Metric3"], -1.0, abs_tol=1e-4)
    assert math.isclose(s_mat.loc["Metric1", "Metric3"], -1.0, abs_tol=1e-4)


def _test_ranking_inversion_detection():
    # ModelA beats ModelB on Accuracy, but ModelB beats ModelA on ECE
    ranks_data = {
        "Accuracy": [1, 2],
        "ECE": [2, 1],
    }
    scores_data = {
        "Accuracy": [85.0, 75.0],
        "ECE": [0.20, 0.05],
    }
    ranks_df = pd.DataFrame(ranks_data, index=["ModelA", "ModelB"])
    scores_df = pd.DataFrame(scores_data, index=["ModelA", "ModelB"])

    inversions = find_ranking_inversions(ranks_df, scores_df)
    assert len(inversions) == 1
    inv = inversions[0]
    assert inv["metric_1"] == "Accuracy"
    assert inv["metric_2"] == "ECE"
    assert inv["model_A"] == "ModelA"  # winner in Metric 1
    assert inv["model_B"] == "ModelB"  # loser in Metric 1


def _test_metric_computation_synthetic():
    N = 100
    targets = np.random.randint(0, 8, size=N)
    # Perfect predictions
    probs = np.zeros((N, 8))
    for i, t in enumerate(targets):
        probs[i, t] = 1.0
    preds = targets.copy()

    img_reprs = np.random.randn(N, 64)
    txt_reprs = img_reprs.copy()  # perfect alignment

    metrics = compute_metrics_for_predictions(probs, preds, targets, img_reprs, txt_reprs)
    assert math.isclose(metrics["Accuracy"], 100.0, abs_tol=1e-4)
    assert math.isclose(metrics["MAE"], 0.0, abs_tol=1e-4)
    assert math.isclose(metrics["ECE"], 0.0, abs_tol=1e-4)
    assert math.isclose(metrics["Macro_F1"], 100.0, abs_tol=1e-4)
    assert math.isclose(metrics["CMA"], 1.0, abs_tol=1e-4)
    assert math.isclose(metrics["Modality_Gap"], 0.0, abs_tol=1e-4)


def main():
    test("compute_ranks respects higher-is-better vs lower-is-better", _test_compute_ranks_directionality)
    test("compute_correlations yields symmetric matrices with diagonal=1 and correct +/-1", _test_kendall_spearman_properties)
    test("find_ranking_inversions correctly detects inverted pairs", _test_ranking_inversion_detection)
    test("compute_metrics_for_predictions yields exact analytical values on synthetic data", _test_metric_computation_synthetic)

    n_pass = sum(1 for _, ok, _ in _results if ok)
    print("=" * 65)
    print(f"Results: {n_pass}/{len(_results)} tests passed.")
    print("=" * 65)
    if n_pass != len(_results):
        sys.exit(1)


if __name__ == "__main__":
    main()
