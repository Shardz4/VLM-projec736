"""SpatialSense spatial-relation dataset loader.
Loads images with annotated spatial-relation triplets
(subject, relation, object) and exposes them for contrastive
zero-shot evaluation with CLIP.
"""
import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class SpatialSenseDataset(Dataset):
    """PyTorch Dataset for SpatialSense spatial-reasoning evaluation.
    Expected directory layout:
        data/spatialsense/
        ├── images/           # Source images
        └── annotations.json  # List of annotation dicts
    Each annotation dict should contain:
        {
            "image":    "image_filename.jpg",
            "subject":  "cat",
            "relation": "on",
            "object":   "table",
            "label":    1          # 1 = correct relation, 0 = swapped/incorrect
        }
    Args:
        image_dir:        Path to the directory containing images.
        annotations_path: Path to the annotations JSON file.
        clip_preprocess:  The ``preprocess`` transform returned by
                         ``clip.load()``. Applied to every image.
    """
    def __init__(self, image_dir, annotations_path, clip_preprocess):
        self.image_dir = Path(image_dir)
        self.clip_preprocess = clip_preprocess

        with open(annotations_path, "r") as f:
            raw_annotations = json.load(f)

        # Validate and filter annotations to those with existing images
        self.annotations = []
        skipped = 0
        for ann in raw_annotations:
            img_path = self.image_dir / ann["image"]
            if img_path.exists():
                self.annotations.append({
                    "image_path": str(img_path),
                    "subject": ann["subject"],
                    "relation": ann["relation"],
                    "object": ann["object"],
                    "label": int(ann.get("label", 1)),
                })
            else:
                skipped += 1

        if len(self.annotations) == 0:
            raise RuntimeError(
                f"No valid samples found. Checked {len(raw_annotations)} "
                f"annotations against images in '{self.image_dir}'."
            )

        if skipped > 0:
            print(
                f"[SpatialSenseDataset] Warning: skipped {skipped} annotations "
                f"with missing images."
            )

        # Catalogue available relations
        self._relations = sorted(set(a["relation"] for a in self.annotations))

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