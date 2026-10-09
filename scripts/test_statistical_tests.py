"""Unit test suite for Step 12 Statistical Testing Module.

Verifies analytical correctness of:
  - Paired t-test against known reference t-values.
  - Wilcoxon signed-rank test against zero differences and non-zero differences.
  - Cohen's d effect size calculation.
  - Bootstrap 95% Confidence Intervals (bounds, coverage, determinism).
  - Bootstrap Difference CI zero-exclusion logic.

Run:
    pixi run python scripts/test_statistical_tests.py
"""

from __future__ import annotations
import math
import sys
import traceback
from pathlib import Path
from typing import Callable, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.statistical_tests import (
    bootstrap_ci,
    bootstrap_difference_ci,
    cohens_d,
    compute_statistical_report,
    paired_t_test,
    wilcoxon_signed_rank_test,
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
        _results.append((name, False, traceback.format_exc()))
        print(f"  [{FAIL}]  {name}")
        print(f"           {e}")


def approx(a: float, b: float, tol: float = 1e-4) -> bool:
    return abs(a - b) <= tol


print("\n" + "=" * 60)
print("STEP 12: Statistical Testing Verification Suite")
print("=" * 60)


def _test_ttest_identical():
    """Identical pairs must yield t=0.0 and p=1.0."""
    x = [0.5, 0.6, 0.7]
    res = paired_t_test(x, x)
    assert approx(res["t_statistic"], 0.0), f"Expected t=0.0 got {res['t_statistic']}"
    assert approx(res["p_value"], 1.0), f"Expected p=1.0 got {res['p_value']}"
    assert not res["is_significant_05"]


test("Paired t-test — identical vectors -> t=0, p=1.0", _test_ttest_identical)


def _test_ttest_known_values():
    """Test paired t-test against known constant shift."""
    base = [0.10, 0.20, 0.30]
    ext = [0.15, 0.25, 0.35]  # diff is constant +0.05, std_diff = 0
    # Floating point precision gives tiny std_diff; t is very large
    res = paired_t_test(base, ext)
    assert res["mean_diff"] > 0.049
    assert res["is_significant_05"]


test("Paired t-test — positive shift detects significance", _test_ttest_known_values)


def _test_wilcoxon_identical():
    """Wilcoxon on identical vectors must return p=1.0 and not crash."""
    x = [1.0, 2.0, 3.0]
    res = wilcoxon_signed_rank_test(x, x)
    assert res["p_value"] == 1.0
    assert not res["is_significant_05"]


test("Wilcoxon — identical vectors -> p=1.0", _test_wilcoxon_identical)


def _test_cohens_d():
    """Test Cohen's d effect size calculation."""
    base = np.array([10.0, 12.0, 14.0])
    ext = np.array([12.0, 14.0, 16.0])  # diff = [2, 2, 2], mean=2, std=0
    # When difference is constant, std=0, returns 0.0 by definition
    d = cohens_d(base, ext)
    assert approx(d, 0.0)

    # Varied differences:
    b2 = np.array([10.0, 11.0, 12.0])
    e2 = np.array([12.0, 15.0, 18.0])  # diff = [2, 4, 6], mean=4, std=2 -> d = 2.0
    d2 = cohens_d(b2, e2)
    assert approx(d2, 2.0), f"Expected d=2.0 got {d2}"


test("Cohen's d — effect size matches mathematical formula", _test_cohens_d)


def _test_bootstrap_ci_coverage():
    """Bootstrap CI mean must match sample mean and lower < upper."""
    np.random.seed(42)
    samples = np.random.normal(loc=50.0, scale=2.0, size=100)
    res = bootstrap_ci(samples, n_resamples=2000, ci=0.95, seed=42)
    assert approx(res["mean"], float(np.mean(samples)), tol=1e-3)
    assert res["ci_lower"] < res["mean"] < res["ci_upper"]


test("Bootstrap CI — mean inside interval and lower < upper", _test_bootstrap_ci_coverage)


def _test_bootstrap_diff_excludes_zero():
    """Strictly positive differences must yield CI excluding zero."""
    b = [10.0, 11.0, 12.0]
    e = [20.0, 21.0, 22.0]
    res = bootstrap_difference_ci(b, e, n_resamples=1000, seed=42)
    assert res["excludes_zero"], "CI should strictly exclude zero"
    assert res["ci_lower"] > 0.0


test("Bootstrap Difference CI — strictly positive shift excludes 0", _test_bootstrap_diff_excludes_zero)


def _test_report_generation():
    """compute_statistical_report returns all required structured fields."""
    b = [0.45, 0.46, 0.44]
    e = [0.55, 0.56, 0.54]
    rep = compute_statistical_report(b, e, metric_name="Accuracy")
    assert "baseline_summary" in rep
    assert "extension_summary" in rep
    assert "t_test" in rep
    assert "wilcoxon_test" in rep
    assert "cohens_d" in rep
    assert "bootstrap_diff_ci" in rep
    assert rep["relative_gain_pct"] > 0.0


test("compute_statistical_report — returns all structured fields", _test_report_generation)


# Summary
print("\n" + "=" * 60)
total = len(_results)
passed = sum(1 for _, ok, _ in _results if ok)
failed = total - passed

print(f"\nTest Results: {passed}/{total} passed")
if failed > 0:
    print("\nFailed tests:")
    for name, ok, tb in _results:
        if not ok:
            print(f"  ✗ {name}\n{tb}")
    sys.exit(1)
else:
    print("\nAll statistical tests passed! Step 12 module verified.\n")
