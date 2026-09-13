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