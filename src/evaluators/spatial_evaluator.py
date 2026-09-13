from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataloader
from tqdm import tqdm

def make_spatial_prompts(subject: str, relation: str, obj: str) -> Tuple[str, str]:
    correct = f"The {subject} is {relation} the {obj}."
    swapped = f"The {obj} is {relation} the {subject}."
    return correct, swapped

class SpatialEvaluator:
    def __init__(self, clip_encoder, device: Optional[str] = None):
        self.encoder = clip_encoder
        self.device = device or clip_encoder.device
    
    @torch.no_grad()
    def evaluate_batch(self, images: torch.Tensor, subjects: List[str], relations: List[str], objects: List[str], labels: Optional[torch.tensor] = None) -> Dict[str, Any]:
        batch_size = images.size(0)
        if batch_size == 0:
            return []
        images = images.to(self.device)
        images_feats = self.encoder.encode_images(images)

        flat_prompts: List[str] = []
        for i in range(batch_size):
            corr, swap = make_spatial_prompts(subjects[i], relations[i], objects[i])
            flat_prompts.extend([corr, swap])
        
        text_feats = self.encoder.encode_texts(flat_promps)
        text_feats = text_feats.view(batch_size, 2, -1)


        sims = torch.einsum("bd,bkd->bk", image_feats, text_feats)

        batch_results = []
        sims_cpu = sims.cpu()
        for i in range(batch_size):
            sim_corr = sims_cpu[i, 0].item()
            sim_swap = sims_cpu[i, 1].item()
            margin = sim_corr - sim_swap
            predicted_idx = 0 if sim_corr >= sim_swap else 1
            is_correct = (predicted_idx == 0)

            sample_res = {
                "subject": subjects[i],
                "relation": relations[i],
                "object": objects[i],
                "correct_prompt": flat_prompts[2 * i],
                "swapped_prompt": flat_prompts[2 * i + 1],
                "sim_correct": sim_corr,
                "sim_swapped": sim_swap,
                "margin": margin,
                "predicted_idx": predicted_idx,
                "is_correct": is_correct,
            }
            if labels is not None:
                sample_res["label"] = int(labels[i].item())
            batch_results.append(sample_res)

        return batch_results
    
    def evaluate_dataset(self, dataloader: DataLoader, only_positive_triplets: bool = True) -> Dict[str, Any]:
        all_samples: List[Dict[str, Any]] = []
        per_relation = default_dict(lambda: {"correct": 0, "total": 0, "margins": []})
        total_batches = len(dataloader)

    print(f"\n[Spatial Evaluator] Evaluating {total_batches} batches....")

    for batch in tqdm(dataloader, desc="Spatial Probe", unit="batch"):
        images, subjects, relations, objects, labels = batch
        batch_Size = images.size(0)

        if only_positive_triplets:
            valid_mask = [bool(labels[i].item() == 1) for i in range(batch_size)]
            if not any(valid_mask):
                continue
            
            images =images[valid_mask]
            subjects = [s for s, m in zip(subjects, valid_mask) if m]
            relations = [r for r, m in zip(relations, valid_mask) if m]
            objects = [o for o, m in zip(objects, valid_mask) if m]
            labels - labels[valid_mask]

        batch_es = self.evaluate_batch(images, subjects, relations, objects, labels)
        for item in batch_res:
            rel = item["relation"]
            per_relation[rel]["total"] += 1
            per_relation[rel]["correct"] += int(item["is_correct"])
            all_samples.append(item)
        
        if not all_samples:
            raise RuntimeError("No evaluation samples were processed")
        

        total_eval = len(all_samples)
        correct_total = sum(s["is_correct"] for s in all_samples)
        overall_acc = correct_total / total_eval
        margins = np.array([s["margin"] for s in all_samples])

        relation_summary = {}
        for rel, data in sorted(per_relation.items()):
            count = data["total"]
            corr = data["correct"]
            rel_acc = corr / count if count > 0 else 0.0
            rel_margins = np.array(data["margins"])
            relation_summary[rel] = {
                "total": count,
                "correct": corr,
                "accuracy": float(rel_acc),
                "mean_margin": float(np.mean(rel_margins)),
                "std_margin": float(np.std(rel_margins)),
            }
        sorted_by_worst_margin = sorted(all_samples, key=lambda x: x["margin"])
        top_failures = sorted_by_worst_margin[:20]
        return {
            "overall_accuracy": float(overall_acc),
            "total_samples": total_eval,
            "correct_predictions": correct_total,
            "random_baseline": 0.50,
            "margin_stats": {
                "mean": float(np.mean(margins)),
                "std": float(np.std(margins)),
                "median": float(np.median(margins)),
                "min": float(np.min(margins)),
                "max": float(np.max(margins)),
            },
            "per_relation": relation_summary,
            "top_failures": top_failures,
            "all_samples": all_samples,
        }