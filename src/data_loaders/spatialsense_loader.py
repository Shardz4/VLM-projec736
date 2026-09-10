"""SpatialSense spatial-relation dataset loader.
Loads images with annotated spatial-relation triplets
(subject, relation, object) and exposes them for contrastive
zero-shot evaluation with CLIP.
"""
import json
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class SpatialSenseDataset(Dataset):
    """PyTorch Dataset for SpatialSense spatial-reasoning evaluation.
    Supports both:
      1. Flat annotations list:
         [{"image": "...", "subject": "...", "relation": "...", "object": "...", "label": 1}]
      2. Official Princeton SpatialSense format:
         [{"url": "...", "annotations": [{"predicate": "...", "subject": {"name": "..."}, "object": {"name": "..."}, "label": true}]}]

    Expected directory layout:
        data/spatialsense/ (or data/images/)
        ├── images/ (or flickr/ and nyu/ subfolders)
        └── annotations.json
    Args:
        image_dir:        Path to directory containing images (or flickr/nyu subdirs).
        annotations_path: Path to annotations JSON file.
        clip_preprocess:  The ``preprocess`` transform returned by
                          ``clip.load()``. Applied to every image.
    """
    def __init__(self, image_dir, annotations_path, clip_preprocess):
        self.image_dir = Path(image_dir)
        self.clip_preprocess = clip_preprocess

        # Auto-resolve image_dir if pointed to data/spatialsense but images are in data/images
        if not (self.image_dir / "flickr").exists() and not (self.image_dir / "nyu").exists():
            candidates = [
                self.image_dir / "images",
                Path("data/images"),
                Path("data/spatialsense/images"),
            ]
            for cand in candidates:
                if (cand / "flickr").exists() or (cand / "nyu").exists() or cand.is_dir():
                    self.image_dir = cand
                    break

        ann_path = Path(annotations_path)
        if not ann_path.is_file():
            candidates = [
                ann_path,
                self.image_dir.parent / "annotations.json",
                self.image_dir / "annotations.json",
                Path("data/spatialsense/annotations.json"),
                Path("data/annotations.json"),
            ]
            for cand in candidates:
                if cand.is_file():
                    ann_path = cand
                    break

        if not ann_path.is_file():
            raise FileNotFoundError(
                f"SpatialSense annotations file not found at '{annotations_path}' "
                f"or any standard fallback locations."
            )

        with open(ann_path, "r") as f:
            raw_data = json.load(f)

        # Normalize raw annotations into a unified triplet list
        normalized_triplets = []
        if isinstance(raw_data, list) and len(raw_data) > 0 and "annotations" in raw_data[0]:
            # Official Princeton SpatialSense format
            for item in raw_data:
                # Resolve filename from url or image key
                url = item.get("url", "")
                filename = url.split("/")[-1] if url else item.get("image", "")
                for ann in item.get("annotations", []):
                    pred = ann.get("predicate", ann.get("relation", ""))
                    subj = ann.get("subject", {})
                    subj_name = subj.get("name", "") if isinstance(subj, dict) else str(subj)
                    obj = ann.get("object", {})
                    obj_name = obj.get("name", "") if isinstance(obj, dict) else str(obj)
                    label_val = ann.get("label", 1)
                    normalized_triplets.append({
                        "image": filename,
                        "subject": subj_name,
                        "relation": pred,
                        "object": obj_name,
                        "label": int(bool(label_val)),
                    })
        elif isinstance(raw_data, list):
            # Flat annotations format
            for ann in raw_data:
                normalized_triplets.append({
                    "image": ann.get("image", ""),
                    "subject": ann.get("subject", ""),
                    "relation": ann.get("relation", ann.get("predicate", "")),
                    "object": ann.get("object", ""),
                    "label": int(ann.get("label", 1)),
                })
        else:
            raise ValueError(f"Unexpected SpatialSense annotations JSON structure in '{ann_path}'.")

        # Validate and filter annotations to those with existing images on disk
        self.annotations = []
        skipped = 0
        for item in normalized_triplets:
            filename = item["image"]
            img_path = self._find_image(filename)
            if img_path is not None:
                self.annotations.append({
                    "image_path": str(img_path),
                    "subject": item["subject"],
                    "relation": item["relation"],
                    "object": item["object"],
                    "label": item["label"],
                })
            else:
                skipped += 1

        if len(self.annotations) == 0:
            raise RuntimeError(
                f"No valid samples found. Checked {len(normalized_triplets)} "
                f"annotations against images in '{self.image_dir}'."
            )

        if skipped > 0:
            print(
                f"[SpatialSenseDataset] Warning: skipped {skipped} annotations "
                f"with missing images."
            )

        # Catalogue available relations
        self._relations = sorted(set(a["relation"] for a in self.annotations))

    def _find_image(self, filename):
        """Locate image file in root or common subdirectories (flickr/, nyu/)."""
        candidates = [
            self.image_dir / filename,
            self.image_dir / "flickr" / filename,
            self.image_dir / "nyu" / filename,
        ]
        for p in candidates:
            if p.is_file():
                return p
        return None

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image = Image.open(ann["image_path"]).convert("RGB")
        image = self.clip_preprocess(image)
        return (
            image,
            ann["subject"],
            ann["relation"],
            ann["object"],
            ann["label"],
        )

    def __len__(self):
        return len(self.annotations)

    @property
    def relations(self):
        """Returns the sorted list of unique spatial relations in the dataset."""
        return list(self._relations)

    def __repr__(self):
        return (
            f"SpatialSenseDataset(samples={len(self)}, "
            f"relations={self._relations})"
        )