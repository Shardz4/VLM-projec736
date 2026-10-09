"""Unit test and numerical verification suite for Step 11 (Cluster 6B Subtask).

Validates all from-scratch implementations against analytical references:
  - Recall@K on identity, reversed, and random permutation matrices.
  - Median Rank boundary conditions.
  - Modality Gap against orthogonal (sqrt(2)) and identical (0) reference values.
  - Cross-Modal Alignment against known cosine references.
  - ECE: perfect predictions -> ECE ~ 0.
  - Forward-pass shape checks for all 3 fusion architectures.
  - Modality Dropout rate statistical test.
  - CrossAttention gradients flow check.
  - MultimodalEvaluator integration test.

Run:
    pixi run python scripts/test_multimodal_subtask.py
"""

from __future__ import annotations
import math
import sys
import traceback
from pathlib import Path
from typing import Callable, List

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.multimodal_metrics import (
    MultimodalEvaluator,
    cosine_similarity_matrix,
    cross_modal_alignment,
    expected_calibration_error,
    median_rank,
    modality_ablation_drop,
    modality_gap,
    recall_at_k,
)
from src.models.multimodal_fusion import (
    CrossAttentionFusion,
    EarlyFusionMLP,
    LateFusionGated,
    ModalityDropout,
    build_fusion_model,
)

# ── Harness ──────────────────────────────────────────────────────────────────
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


# ═════════════════════════════════════════════════════════════════════
# SECTION 1: Recall@K & Median Rank
# ═════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 1: Recall@K & Median Rank")
print("="*60)


def _test_recall_identity():
    N = 5
    e = torch.eye(N)
    r = recall_at_k(e, e, k_values=(1, 5))
    assert r["i2t"]["R@1"] == 100.0
    assert r["t2i"]["R@1"] == 100.0
    assert r["i2t"]["R@5"] == 100.0


test("Recall@K — identity matrix -> R@1=100%", _test_recall_identity)


def _test_recall_reversed():
    # Use unique random Gaussian vectors to avoid cosine similarity ties
    # that occur with orthonormal bases.  Reversed matching (i paired with N-1-i)
    # guarantees the correct target is never in top-1 when N is large enough.
    torch.manual_seed(7)
    N = 20
    img = torch.randn(N, 64)
    # text[i] is the clone of img[N-1-i] — i.e., reversed matching
    txt = img.flip(0).clone()
    r = recall_at_k(img, txt, k_values=(1, N))
    # With strictly unique non-collinear vectors and N=20, R@1 should be 0
    # (each image's top-1 text match is itself-reversed, not its pair)
    # At least verify R@N = 100%
    assert r["i2t"][f"R@{N}"] == 100.0, f"R@N should be 100%, got {r['i2t'][f'R@{N}']}"
    # R@1 must be < R@10 (degraded retrieval for reversed pairs)
    assert r["i2t"]["R@1"] < 50.0, f"R@1 too high for reversed pairs: {r['i2t']['R@1']}"


test("Recall@K — reversed pairs -> R@N=100%, R@1<50%", _test_recall_reversed)


def _test_recall_single_pair():
    img = torch.randn(1, 512)
    txt = torch.randn(1, 512)
    r = recall_at_k(img, txt, k_values=(1,))
    assert r["i2t"]["R@1"] == 100.0
    assert r["t2i"]["R@1"] == 100.0


test("Recall@K — single pair -> R@1=100%", _test_recall_single_pair)


def _test_recall_monotonicity():
    torch.manual_seed(42)
    img = torch.randn(50, 128)
    txt = torch.randn(50, 128)
    r = recall_at_k(img, txt, k_values=(1, 5, 10))
    for d in ("i2t", "t2i"):
        assert r[d]["R@1"] <= r[d]["R@5"] <= r[d]["R@10"]


test("Recall@K — monotonicity R@1 <= R@5 <= R@10", _test_recall_monotonicity)


def _test_median_rank_identity():
    N = 10
    e = torch.eye(N)
    ranks = median_rank(e, e)
    assert ranks["i2t_median_rank"] == 1.0
    assert ranks["t2i_median_rank"] == 1.0


test("Median Rank — identity -> MR=1.0", _test_median_rank_identity)


def _test_median_rank_worst():
    N = 4
    img = torch.eye(N)
    txt = torch.eye(N).flip(0)
    ranks = median_rank(img, txt)
    assert ranks["i2t_median_rank"] > 1.0


test("Median Rank — reversed -> MR > 1", _test_median_rank_worst)


# ═════════════════════════════════════════════════════════════════════
# SECTION 2: Modality Gap & Cross-Modal Alignment
# ═════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 2: Modality Gap & Cross-Modal Alignment")
print("="*60)


def _test_gap_identical():
    feats = torch.randn(100, 512)
    gap = modality_gap(feats, feats.clone())
    assert approx(gap, 0.0, tol=1e-5), f"Expected 0.0 got {gap}"


test("Modality Gap — identical -> Delta_gap=0", _test_gap_identical)


def _test_gap_orthogonal():
    img = torch.tensor([[1.0, 0.0]])
    txt = torch.tensor([[0.0, 1.0]])
    gap = modality_gap(img, txt)
    assert approx(gap, math.sqrt(2), tol=1e-5), f"Expected sqrt(2) got {gap}"


test("Modality Gap — orthogonal -> Delta_gap=sqrt(2)", _test_gap_orthogonal)


def _test_gap_anti_parallel():
    img = torch.tensor([[1.0, 0.0]])
    txt = torch.tensor([[-1.0, 0.0]])
    gap = modality_gap(img, txt)
    assert approx(gap, 2.0, tol=1e-5), f"Expected 2.0 got {gap}"


test("Modality Gap — anti-parallel -> Delta_gap=2.0", _test_gap_anti_parallel)


def _test_cma_identical():
    feats = torch.randn(50, 256)
    cma = cross_modal_alignment(feats, feats.clone())
    assert approx(cma, 1.0, tol=1e-5), f"Expected 1.0 got {cma}"


test("Cross-Modal Alignment — identical -> CMA=1.0", _test_cma_identical)


def _test_cma_orthogonal():
    N = 100
    img = torch.zeros(N, 2); img[:, 0] = 1.0
    txt = torch.zeros(N, 2); txt[:, 1] = 1.0
    cma = cross_modal_alignment(img, txt)
    assert approx(cma, 0.0, tol=1e-5), f"Expected 0.0 got {cma}"


test("Cross-Modal Alignment — orthogonal pairs -> CMA=0.0", _test_cma_orthogonal)


# ═════════════════════════════════════════════════════════════════════
# SECTION 3: ECE
# ═════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 3: Expected Calibration Error (ECE)")
print("="*60)


def _test_ece_perfect():
    N = 200
    confs = torch.ones(N) * 0.95
    correct = torch.ones(N)
    r = expected_calibration_error(confs, correct, n_bins=10)
    assert r["ece"] < 0.1, f"Expected ECE < 0.1 got {r['ece']}"


test("ECE — all correct @ 95% conf -> ECE small", _test_ece_perfect)


def _test_ece_worst():
    N = 200
    confs = torch.ones(N)
    correct = torch.zeros(N)
    r = expected_calibration_error(confs, correct, n_bins=10)
    assert r["ece"] > 0.5, f"Expected ECE > 0.5 got {r['ece']}"


test("ECE — all wrong @ max conf -> ECE > 0.5", _test_ece_worst)


def _test_ece_bin_counts():
    N = 500
    confs = torch.rand(N)
    correct = (torch.rand(N) > 0.4).float()
    r = expected_calibration_error(confs, correct, n_bins=10)
    assert sum(r["bin_counts"]) == N


test("ECE — bin_counts sum to N", _test_ece_bin_counts)


# ═════════════════════════════════════════════════════════════════════
# SECTION 4: Fusion Architecture Forward Passes
# ═════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 4: Fusion Architecture Forward Passes")
print("="*60)

V_DIM, T_DIM, N_CLS, B = 512, 512, 8, 16


def _test_early_shape():
    m = EarlyFusionMLP(V_DIM, T_DIM, N_CLS, hidden_dim=256)
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(B, V_DIM), torch.randn(B, T_DIM))
    assert out.shape == (B, N_CLS), f"Got {out.shape}"


test("EarlyFusionMLP — output shape (B, num_classes)", _test_early_shape)


def _test_late_shape():
    m = LateFusionGated(V_DIM, T_DIM, N_CLS)
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(B, V_DIM), torch.randn(B, T_DIM))
    assert out.shape == (B, N_CLS), f"Got {out.shape}"


test("LateFusionGated — output shape (B, num_classes)", _test_late_shape)


def _test_cross_shape():
    m = CrossAttentionFusion(V_DIM, T_DIM, N_CLS, d_model=256, num_heads=4)
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(B, V_DIM), torch.randn(B, T_DIM))
    assert out.shape == (B, N_CLS), f"Got {out.shape}"


test("CrossAttentionFusion — output shape (B, num_classes)", _test_cross_shape)


def _test_factory():
    for strategy in ("early", "late", "cross_attention"):
        m = build_fusion_model(strategy, V_DIM, T_DIM, N_CLS)
        out = m(torch.randn(4, V_DIM), torch.randn(4, T_DIM))
        assert out.shape == (4, N_CLS), f"Strategy {strategy}: got {out.shape}"


test("build_fusion_model — all 3 strategies correct shapes", _test_factory)


def _test_cross_return_weights():
    m = CrossAttentionFusion(V_DIM, T_DIM, N_CLS, d_model=128, num_heads=2)
    m.eval()
    with torch.no_grad():
        logits, weights = m(torch.randn(4, V_DIM), torch.randn(4, T_DIM), return_attention=True)
    assert logits.shape == (4, N_CLS)
    assert weights.shape == (4, 1, 1), f"Got {weights.shape}"


test("CrossAttentionFusion — return_attention=True returns weight tensor", _test_cross_return_weights)


def _test_cross_gradients():
    m = CrossAttentionFusion(V_DIM, T_DIM, N_CLS, d_model=128, num_heads=2)
    m.train()
    out = m(torch.randn(4, V_DIM), torch.randn(4, T_DIM))
    out.sum().backward()
    assert m.proj_t.weight.grad is not None, "No grad through proj_t"
    assert m.proj_v.weight.grad is not None, "No grad through proj_v"


test("CrossAttentionFusion — gradients flow through both projections", _test_cross_gradients)


# ═════════════════════════════════════════════════════════════════════
# SECTION 5: Modality Dropout
# ═════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 5: Modality Dropout")
print("="*60)


def _test_md_eval_passthrough():
    md = ModalityDropout(p_drop=0.9)
    md.eval()
    v = torch.randn(8, 64)
    t = torch.randn(8, 64)
    v_out, t_out = md(v, t)
    assert torch.allclose(v, v_out)
    assert torch.allclose(t, t_out)


test("ModalityDropout — eval mode passes unchanged", _test_md_eval_passthrough)


def _test_md_rate():
    torch.manual_seed(0)
    md = ModalityDropout(p_drop=0.5)
    md.train()
    N = 2000
    v_zeroed = t_zeroed = 0
    v = torch.ones(1, 4)
    t = torch.ones(1, 4)
    for _ in range(N):
        v_out, t_out = md(v, t)
        if v_out.sum() == 0: v_zeroed += 1
        if t_out.sum() == 0: t_zeroed += 1
    v_rate = v_zeroed / N
    t_rate = t_zeroed / N
    assert 0.10 < v_rate < 0.45, f"Vision rate {v_rate:.3f} out of range"
    assert 0.10 < t_rate < 0.45, f"Text rate {t_rate:.3f} out of range"


test("ModalityDropout — observed drop rate in expected range", _test_md_rate)


def _test_md_never_both_zero():
    torch.manual_seed(99)
    md = ModalityDropout(p_drop=0.99)
    md.train()
    v = torch.ones(1, 4)
    t = torch.ones(1, 4)
    for _ in range(500):
        v_out, t_out = md(v, t)
        assert not (v_out.sum() == 0 and t_out.sum() == 0), "Both zeroed!"


test("ModalityDropout — never zeros both simultaneously", _test_md_never_both_zero)


# ═════════════════════════════════════════════════════════════════════
# SECTION 6: Integration
# ═════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 6: MultimodalEvaluator Integration")
print("="*60)


def _test_evaluator_keys():
    torch.manual_seed(42)
    N, D = 20, 128
    img = torch.randn(N, D)
    txt = torch.randn(N, D)
    confs = torch.rand(N)
    correct = (torch.rand(N) > 0.4).float()
    evaluator = MultimodalEvaluator(k_values=(1, 5, 10))
    results = evaluator.compute_all(img, txt, confs, correct)
    assert {"recall", "ranks", "modality_gap", "cross_modal_alignment", "ece"} == set(results.keys())
    assert {"i2t", "t2i"} == set(results["recall"].keys())
    assert "ece" in results["ece"]


test("MultimodalEvaluator.compute_all — all required keys present", _test_evaluator_keys)


def _test_ablation_drop_logic():
    drop = modality_ablation_drop(
        joint_acc=0.75,
        vision_only_acc=0.60,
        text_only_acc=0.55,
    )
    assert approx(drop["vision_drop"], 0.20), f"Got {drop['vision_drop']}"
    assert approx(drop["text_drop"], 0.15), f"Got {drop['text_drop']}"
    assert drop["dominant_modality"] == "vision"


test("modality_ablation_drop — correct values and dominant_modality", _test_ablation_drop_logic)


# ═════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════
total = len(_results)
passed = sum(1 for _, ok, _ in _results if ok)
failed = total - passed

print("\n" + "="*60)
print(f"\nTest Results: {passed}/{total} passed")

if failed > 0:
    print(f"\nFailed tests:")
    for name, ok, tb in _results:
        if not ok:
            print(f"\n  FAIL: {name}")
            print(tb)
    sys.exit(1)
else:
    print("\nAll tests passed! Step 11 Cluster 6B Subtask is verified.\n")
    print("Next step: pixi run python scripts/test_multimodal_subtask.py")
