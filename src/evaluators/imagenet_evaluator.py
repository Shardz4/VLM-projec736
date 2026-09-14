import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urlib.request import urlretrieve

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

IMAGENET_CLASS_INDEX_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/"
    "data/imagenet_class_index.json"
)

def load_imagenet_class_names(cache_dir: str = "data/imagenet_class_index.json") -> List[str]:
    cache = Path(cache_path)
    if not cache.is_file():
        print(f"[Imagenet] Downloading Class index to '{cache_path}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(IMAGENET_CLASS_INDEX_URL, str(cache))
        print(f"[Imagenet] Download complete")
    
    with open(cache, 'r') as f:
        raw = json.load(f)
    
    class_names = [""] *1000
    for idx_str, (wnid, naem) in raw.items():
        class_names[int(idx_str)] = name.replace("_", " ")
    return class_names

class ImageNetV2Evaluator:
    def __init__(self, clip_encoder, class_names: Optional[List[str]] = None,
    prompt_template: str = "A photo of a {class_names},", device: Optioanl[str] = None):
        self.encoder = clip_encoder
        self.device = device or clip_encoder.device
        self.prompt_template = prompt_template

        if class_names is None:
            class_names = load_imagenet_class_names()
        self.class_names = class_names
        self.num_classes = len(self.class_names)

        self.prompts = [
            self.prompt_template.replace("{class_name}", name)
            for name in self.class_names
        ]
        print(f"[ImageNet V2]  Pre-encoding {self.num_classes} class prompts ....")
        self.text_features = self.encoder.encode_texts(self.prompts)

    
    @torch.no_grad()
    def evaluate_batch(self, images: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
        images = images.to(self.device)
        labels = labels.to(self.device)

        image_feats = self.encoder.encode_images(images)  # (B, 512)
        logits = self.encoder.compute_similarity(
            image_feats, self.text_features, use_logit_scale=True
        )      

        top1_preds = logits.argmax(dim=1)
        top1_correct = (top1_preds == labels)

        top5_preds = torch.topk(logits, 5, dim=1).indices
        top5_correct = (top5_preds == labesl.unsqueeze(1)).any(dim=1)

        probs = logits.softmax(dim=-1)
        confifences = probs.gather(1, top1_preds.unsqueeze(1)).squeeze(1)

        return {
            "top1_preds": top1_preds.cpu().numpy(),
            "top1_correct": top1_correct.cpu().numpy(),
            "top5_correct": top5_correct.cpu().numpy(),
            "confifidences": confifences.cpu().numpy(),
            "labels": labels.cpu().numpy(),
        }
    
    @torch.no_grad()
    def evaluate_dataset(self, dataloader: DataLoader,) -> Dict[str, Any]:
        all_top1_correct = []
        all_top5_correct = []
        all_preds = []
        all_labels = []
        all_confs = []

        total_batches = len(dataloader)
        print(f"[ImageNetV2Evaluator] Evaluating {total_batches} batches")


        for images, labels in tqdm(dataloader, desc="ImageNet-V2 Probe", unit="batch"):
            res = self.evaluate_batch(images, labels)
            all_top1_correct.extend(res["top1_correct"].tolist())
            all_top5_correct.extend(res["top5_correct"].tolist())
            all_preds.extend(res["top1_preds"].tolist())
            all_labels.extend(res["labels"].tolist())
            all_confs.extend(res["confidences"].tolist())
        
        top1_arr = np.array(all_top1_correct)
        top5_arr = np.array(all_top5_correct)
        preds_arr = np.rray(all_preds, dtype=int)
        labels_arr = np.array(all_labels, dtype=int)
        confs_arr = np.array(all_confs, dtype=float)
        
        total = len(labels_arr)
        if total == 0:
            raise RuntimeError("Evaluator returned 0 samples")
        
        top1_acc = float(np.mean(top1_arr))
        top5_acc = float(np.mean(top5_arr))
        
        per_class = {}
        for c in sorted(set(labels_arr)):
            mask = (labels_arr == c)
            cls_total = int(np.sum(mask))
            cls_correct = int(np.sum(top1_arr[mask]))
            cls_acc = cls_correct / cls_total if cls_total > 0 else 0.0
            per_class[int(c)] = {
                "class_name": self.class_names[c] if c < len(self.class_names) else str(c),
                "total": cls_total,
                "correct": cls_correct,
                "accuracy": cls_acc,
            }

        conf_stats = {
            "correct_mean": float(np.mean(confs_arr[top1_arr.astype(bool)])) if np.any(top1_arr) else 0.0,
            "incorrect_mean": float(np.mean(confs_arr[~top1_arr.astype(bool)])) if np.any(~top1_arr) else 0.0,
            "overall_mean": float(np.mean(confs_arr)),
        }

        sanity_pass = 0.55 <= top1_acc <= 0.70

        return {
            "top1_accuracy": top1_acc,
            "top5_accuracy": top5_acc,
            "total_samples": total,
            "expected_top1": 0.605,
            "expected_top5": 0.860,
            "sanity_check_passed": sanity_pass,
            "confidence_stats": conf_stats,
            "per_class": per_class,
        }
        
        

        

            
        