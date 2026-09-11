from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class CountingEvaluator:
    def __init__(
        self, clip_encoder, template: str = "A photo of {N} object(s)", count_range: Optional[List[int]] = None, device: Optional[str] = None):
        self.encoder = clip_encoder
        self.device = device or self.encoder.device
        self.template = template
        self.count_range = count_range or list(range(1, 21))
        self.num_classes = len(self.count_range)
        self.prompts = [self.template.replace("{N}", str(n)) for n in self.count_range]

        print(f"[Counting Evaluator] Pre-encoding {self.num_classes} candidate counting prompts...")
        self.text_features = self.encoder.encode_texts(self.prompts)

    @torch.no_grad()
    def evaluate_batch(self, images: torch.Tensor, ground_truth_counts: torch.Tensor) -> Dicr[str, Any]:
        images - images.to(self.device)
        image_feats = self.encoder.compute_similarity(image_feats, self.text_features, use_logit_scale=True)
        probs - logits.softmax(dim=-1)

        pred_indices = probs.argmax(dim=-1)
        confidence, _ = probs.max(dim=-1)

        counts_lookup = torch.tensor(self.count_range, device=self.device)
        pred_counts = counts_lookup[pred_indices]

        return {
            "predictions": pred_counts.cpu().numpy(),
            "confidence": confidence.cpu().numpy(),
            "probabilities": probs.cpu().numpy(),
            "ground_truth": ground_truth_counts.cpu().numpy(),
        }
    
    def evaluate_dataset(self,dataloader: DataLoader) -> Dict[str, Any]:
        all_preds: List[int] = []
        all_targets: List[int] = []
        all_confs: List[int] = []

        total_batches = len(dataloader)
        print(f"\n[CountingEvaluator] Evaluating {total_batches} batches...")

        for images, counts in tqdm(dataloader, desc="Counting Probe", unit="batch"):
            batch_res = self.evaluate_batch(images, counts)
            all_preds.extend(batch_res["predictions"].tolist())
            all_targets.extend(batch_res["ground_truth"].tolist())
            all_confs.extend(batch_res["confidence"].tolist())
        
        all_preds_arr = np.array(all_preds, dtype=int)
        all_targets_arr = np.array(all_targets, dtype=int)
        all_confs_arr = np.array(all_confs, dtype=float)

        total_samples = len(all_targets_arr)
        if total_samples == 0:
            raise RuntimeError("No samples evaluated in CountingProbe")
        

        #  Accuracy & MAE
        is_correct = (all_preds_arr == all_targets_arr)
        overall_acc = float(np.mean(is_correct))
        mae = float(np.mean(np.abs(all_preds_arr - all_targets_arr)))



        # 2. Per-count 
        per_count = {}
        for c in self.count_range:
            mask = (all_targets_arr == c)
            count_total = int(np.sum(mask))
            if count_total > 0:
                count_corr = int(np.sum(is_correct[mask]))
                count_acc = count_corr / count_total
                count_mae = float(np.mean(np.abs(all_preds_arr[mask] - c)))
                count_conf = float(np.mean(all_confs_arr[mask]))
            else:
                count_corr, count_acc, count_mae, count_conf = 0, 0.0, 0.0, 0.0
            per_count[c] = {
                "total": count_total,
                "correct": count_corr,
                "accuracy": float(count_acc),
                "mae": float(count_mae),
                "mean_confidence": float(count_conf),
            }


        # 3. K x K Confusion Matrix 
        K = self.num_classes
        conf_matrix = np.zeros((K, K), dtype=int)
        val_to_idx = {val: idx for idx, val in enumerate(self.count_range)}
        for t, p in zip(all_targets_arr, all_preds_arr):
            if t in val_to_idx and p in val_to_idx:
                conf_matrix[val_to_idx[t], val_to_idx[p]] += 1
                conf_stats = {
            "correct_mean": float(np.mean(all_confs_arr[is_correct])) if np.any(is_correct) else 0.0,
            "incorrect_mean": float(np.mean(all_confs_arr[~is_correct])) if np.any(~is_correct) else 0.0,
        }
        return {
            "overall_accuracy": overall_acc,
            "mean_absolute_error": mae,
            "total_samples": total_samples,
            "random_baseline": 1.0 / self.num_classes,
            "per_count": per_count,
            "confusion_matrix": conf_matrix.tolist(),
            "confidence_stats": conf_stats,
            "count_range": self.count_range,
        }

        