""" Clever counting dataset loader"""
import json
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class CLEVRCountingDataset(Dataset):
    def __init__(self, image_dir, annotations_path, clip_preprocess, max_objects=10):
        self.image_dir = Path(image_dir)
        self.clip_preprocess = clip_preprocess
        self.max_objects = max_objects

        with open(annotations_path, "r") as f:
            raw = json.load(f)

        if "scenes" in raw:
            mapping = {
                scene["image_filename"]: len(scene["objects"])
                for scene in raw["scenes"]
            } 
        else:
            mapping = {k: int(v) for k, v in raw.items()}

        self.samples = []
        for filename, count in mapping.items():
            if 1 <= count <= max_objects:
                img_path = self.image_dir / filename
                if img_path.exists():
                    self.samples.append((str(img_path), count))
        
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid samples found"
                f"against images in '{self.images_dir}' with count in [1, {max_objects}]."
            )
        self._count_distributions = {}
        for _, count in self.samples:
            self._count_distributions[count] = self._count_distributions.get(count,0) + 1
    

    def __getitem__(self, idx):
        image_path, count = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.clip_preprocess(image)
        return image, count
    
    def __len__(self):
        return len(self.samples)
    
    @property
    def count_distribution(self):
        return dict(sorted(self._count_distributions.items()))
    
    def __repr__(self):
        return (
            f"CLEVRCountingDataset(samples={len(self)})"
            f"max_objects = {self.max_objects}"
            f"distribution={self.count_distribution}"
        )
        