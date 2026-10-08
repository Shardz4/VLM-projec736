"""From-scratch Cluster 6B multimodal metric suite.

Implements:
  - Bidirectional Recall@K  (K in {1, 5, 10}), both I->T and T->I directions.
  - Median Rank             (median position of the correct match in the ranked list).
  - Modality Gap (Delta_gap) (L2 distance between L2-normalized visual and textual centroids).
  - Cross-Modal Alignment   (mean pairwise cosine similarity of matched pairs).
  - Expected Calibration Error (ECE)  (equal-mass 10-bin calibration error).
  - Modality Ablation Drop  (performance gap when one modality is zeroed out).

Reference:
  Liang et al. NeurIPS 2022. arXiv:2203.02053
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch


def _to_tensor(x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return x.float()


def _l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def cosine_similarity_matrix(
    a: Union[torch.Tensor, np.ndarray],
    b: Union[torch.Tensor, np.ndarray],
) -> torch.Tensor:
    a = _l2_normalize(_to_tensor(a))
    b = _l2_normalize(_to_tensor(b))
    return a @ b.T


def _ranks_from_similarity_matrix(sim_matrix: torch.Tensor) -> torch.Tensor:
    N = sim_matrix.size(0)
    sorted_indices = sim_matrix.argsort(dim=1, descending=True)
    diagonal_targets = torch.arange(N, dtype=torch.long)
    ranks = (sorted_indices == diagonal_targets.unsqueeze(1)).nonzero(as_tuple=False)
    ranks_sorted = ranks[ranks[:, 0].argsort()]
    return ranks_sorted[:, 1] + 1


def recall_at_k(
    image_features: Union[torch.Tensor, np.ndarray],
    text_features: Union[torch.Tensor, np.ndarray],
    k_values: Sequence[int] = (1, 5, 10),
) -> Dict[str, Dict[str, float]]:
    img = _to_tensor(image_features)
    txt = _to_tensor(text_features)
    sim_i2t = cosine_similarity_matrix(img, txt)
    ranks_i2t = _ranks_from_similarity_matrix(sim_i2t)
    sim_t2i = cosine_similarity_matrix(txt, img)
    ranks_t2i = _ranks_from_similarity_matrix(sim_t2i)
    results: Dict[str, Dict[str, float]] = {"i2t": {}, "t2i": {}}
    N = img.size(0)
    for k in k_values:
        key = f"R@{k}"
        results["i2t"][key] = float((ranks_i2t <= k).sum().item()) / N * 100.0
        results["t2i"][key] = float((ranks_t2i <= k).sum().item()) / N * 100.0
    return results


def median_rank(
    image_features: Union[torch.Tensor, np.ndarray],
    text_features: Union[torch.Tensor, np.ndarray],
) -> Dict[str, float]:
    img = _to_tensor(image_features)
    txt = _to_tensor(text_features)
    sim_i2t = cosine_similarity_matrix(img, txt)
    sim_t2i = cosine_similarity_matrix(txt, img)
    ranks_i2t = _ranks_from_similarity_matrix(sim_i2t).float()
    ranks_t2i = _ranks_from_similarity_matrix(sim_t2i).float()
    return {
        "i2t_median_rank": float(ranks_i2t.median().item()),
        "t2i_median_rank": float(ranks_t2i.median().item()),
        "i2t_mean_rank":   float(ranks_i2t.mean().item()),
        "t2i_mean_rank":   float(ranks_t2i.mean().item()),
    }


def modality_gap(
    image_features: Union[torch.Tensor, np.ndarray],
    text_features: Union[torch.Tensor, np.ndarray],
) -> float:
    r"""L2 distance between L2-normalized modality centroids on unit hypersphere.

    Delta_gap = ||mu_I - mu_T||_2
    where mu_I = mean(v_i / ||v_i||),  mu_T = mean(t_i / ||t_i||).
    Range: [0, 2].  sqrt(2) for orthogonal, 0 for identical centroids.
    """
    img_norm = _l2_normalize(_to_tensor(image_features))
    txt_norm = _l2_normalize(_to_tensor(text_features))
    mu_img = img_norm.mean(dim=0)
    mu_txt = txt_norm.mean(dim=0)
    return float((mu_img - mu_txt).norm(p=2).item())


def cross_modal_alignment(
    image_features: Union[torch.Tensor, np.ndarray],
    text_features: Union[torch.Tensor, np.ndarray],
) -> float:
    """Mean pairwise cosine similarity of matched (row_i, row_i) image-text pairs."""
    img_norm = _l2_normalize(_to_tensor(image_features))
    txt_norm = _l2_normalize(_to_tensor(text_features))
    pair_cosines = (img_norm * txt_norm).sum(dim=-1)
    return float(pair_cosines.mean().item())


def expected_calibration_error(
    confidences: Union[torch.Tensor, np.ndarray],
    correct: Union[torch.Tensor, np.ndarray],
    n_bins: int = 10,
) -> Dict:
    """ECE using equal-mass binning.  ECE=0 means perfect calibration."""
    conf = _to_tensor(confidences).numpy().ravel()
    corr = _to_tensor(correct).numpy().ravel().astype(float)
    N = len(conf)
    sort_idx = np.argsort(conf)
    conf_sorted = conf[sort_idx]
    corr_sorted = corr[sort_idx]
    bins = np.array_split(np.arange(N), n_bins)
    bin_accs: List[float] = []
    bin_confs: List[float] = []
    bin_counts: List[int] = []
    ece = 0.0
    for bin_idx in bins:
        if len(bin_idx) == 0:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            bin_counts.append(0)
            continue
        b_acc  = corr_sorted[bin_idx].mean()
        b_conf = conf_sorted[bin_idx].mean()
        b_n    = len(bin_idx)
        ece += (b_n / N) * abs(b_acc - b_conf)
        bin_accs.append(float(b_acc))
        bin_confs.append(float(b_conf))
        bin_counts.append(int(b_n))
    return {
        "ece":             float(ece),
        "bin_accuracies":  bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts":      bin_counts,
    }


def modality_ablation_drop(
    joint_acc: float,
    vision_only_acc: float,
    text_only_acc: float,
) -> Dict:
    """Compute modality dependence from three accuracy values.

    vision_drop = joint_acc - text_only_acc   (drop when text=0, vision only)
    text_drop   = joint_acc - vision_only_acc  (drop when image=0, text only)
    """
    return {
        "joint_accuracy":       float(joint_acc),
        "vision_only_accuracy": float(vision_only_acc),
        "text_only_accuracy":   float(text_only_acc),
        "vision_drop":          float(joint_acc - text_only_acc),
        "text_drop":            float(joint_acc - vision_only_acc),
        "dominant_modality":    "vision" if vision_only_acc >= text_only_acc else "text",
    }


class MultimodalEvaluator:
    """Aggregator: compute the full Cluster 6B metric suite in one call.

    Example:
        evaluator = MultimodalEvaluator()
        results = evaluator.compute_all(img_feats, txt_feats, confidences, correct)
        evaluator.print_report(results, prefix="CrossAttnFusion")
    """

    def __init__(self, k_values: Sequence[int] = (1, 5, 10), n_ece_bins: int = 10):
        self.k_values = k_values
        self.n_ece_bins = n_ece_bins

    def compute_all(
        self,
        image_features: Union[torch.Tensor, np.ndarray],
        text_features: Union[torch.Tensor, np.ndarray],
        confidences: Optional[Union[torch.Tensor, np.ndarray]] = None,
        correct: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> Dict:
        results: Dict = {}
        results["recall"] = recall_at_k(image_features, text_features, self.k_values)
        results["ranks"] = median_rank(image_features, text_features)
        results["modality_gap"] = modality_gap(image_features, text_features)
        results["cross_modal_alignment"] = cross_modal_alignment(image_features, text_features)
        if confidences is not None and correct is not None:
            results["ece"] = expected_calibration_error(
                confidences, correct, n_bins=self.n_ece_bins
            )
        return results

    def print_report(self, results: Dict, prefix: str = "") -> None:
        tag = f"[{prefix}] " if prefix else ""
        sep = "=" * 60
        print(f"\n{tag}{sep}")
        print(f"{tag}Cluster 6B Multimodal Metric Report")
        print(f"{tag}{sep}")
        rec = results.get("recall", {})
        print(f"\n{tag}  Bidirectional Recall@K:")
        for direction in ("i2t", "t2i"):
            label = "Image->Text" if direction == "i2t" else "Text->Image"
            vals = "  ".join(
                f"R@{k}: {rec.get(direction, {}).get(f'R@{k}', 0.0):.1f}%"
                for k in self.k_values
            )
            print(f"{tag}    {label:>12}: {vals}")
        rnk = results.get("ranks", {})
        print(f"\n{tag}  Median Rank:")
        print(f"{tag}    Image->Text: {rnk.get('i2t_median_rank', 'N/A'):.1f}")
        print(f"{tag}    Text->Image: {rnk.get('t2i_median_rank', 'N/A'):.1f}")
        print(
            f"\n{tag}  Modality Gap (Delta_gap): "
            f"{results.get('modality_gap', 'N/A'):.4f}  "
            f"(0=aligned, sqrt2~1.414=orthogonal)"
        )
        print(f"{tag}  Cross-Modal Alignment:    {results.get('cross_modal_alignment', 'N/A'):.4f}")
        ece = results.get("ece", {})
        if ece:
            print(f"\n{tag}  ECE: {ece.get('ece', 'N/A'):.4f}")
        print(f"{tag}{sep}\n")
