"""CLEVR Counting Dataset loader.
Loads CLEVR validation scenes and exposes (image, object_count) pairs.
The scene JSON (CLEVR_val_scenes.json) is parsed to extract per-image
ground-truth object counts, filtered to [1, max_objects].
"""
import json
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class CLEVRCountingDataset(Dataset):
    """PyTorch Dataset for CLEVR object-counting evaluation.
    Expected directory layout:
        data/clevr/ (or data/CLEVR_v1.0/)
        ├── images/          # CLEVR_v1.0 val images (or images/val/)
        └── annotations.json # Pre-parsed {filename: count} mapping
                             # OR scenes/CLEVR_val_scenes.json
    Args:
        image_dir:        Path to the directory containing CLEVR images.
        annotations_path: Path to annotations JSON — either a pre-parsed
                          {filename: count} dict, or the original CLEVR
                          scenes JSON (auto-detected).
        clip_preprocess:  The ``preprocess`` transform returned by
                          ``clip.load()``. Applied to every image.
        max_objects:      Maximum object count to include (default 10).
    """
    def __init__(self, image_dir, annotations_path, clip_preprocess, max_objects=10):
        self.image_dir = Path(image_dir)
        self.clip_preprocess = clip_preprocess
        self.max_objects = max_objects

        # Auto-resolve image_dir if pointed to dataset root
        if not (self.image_dir / "CLEVR_val_000000.png").exists():
            if (self.image_dir / "images" / "val").exists():
                self.image_dir = self.image_dir / "images" / "val"
            elif (self.image_dir / "val").exists():
                self.image_dir = self.image_dir / "val"
            elif (self.image_dir / "images").exists():
                self.image_dir = self.image_dir / "images"

        # Auto-resolve annotations_path if pointed to directory or alternative location
        ann_path = Path(annotations_path)
        if not ann_path.is_file():
            candidates = [
                ann_path,
                ann_path / "scenes" / "CLEVR_val_scenes.json",
                ann_path / "annotations.json",
                self.image_dir.parent / "scenes" / "CLEVR_val_scenes.json",
                self.image_dir.parent.parent / "scenes" / "CLEVR_val_scenes.json",
                Path("data/CLEVR_v1.0/scenes/CLEVR_val_scenes.json"),
                Path("data/clevr/annotations.json"),
            ]
            for cand in candidates:
                if cand.is_file():
                    ann_path = cand
                    break

        if not ann_path.is_file():
            raise FileNotFoundError(
                f"CLEVR annotations file not found at '{annotations_path}' "
                f"or any standard fallback locations."
            )

        # Load and parse annotations
        with open(ann_path, "r") as f:
            raw = json.load(f)

        # Support both the original CLEVR scenes JSON and a pre-parsed mapping
        if "scenes" in raw:
            # Original CLEVR format: {"scenes": [{"image_filename": ..., "objects": [...]}]}
            mapping = {
                scene["image_filename"]: len(scene["objects"])
                for scene in raw["scenes"]
            }
        else:
            # Pre-parsed format: {"CLEVR_val_000000.png": 5, ...}
            mapping = {k: int(v) for k, v in raw.items()}

        # Filter to [1, max_objects] and verify images exist on disk
        self.samples = []
        for filename, count in mapping.items():
            if 1 <= count <= max_objects:
                img_path = self.image_dir / filename
                if img_path.exists():
                    self.samples.append((str(img_path), count))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid samples found. Checked {len(mapping)} annotations "
                f"against images in '{self.image_dir}' with count in [1, {max_objects}]."
            )

        # Compute per-count distribution for diagnostics
        self._count_distribution = {}
        for _, count in self.samples:
            self._count_distribution[count] = self._count_distribution.get(count, 0) + 1

    def __getitem__(self, idx):
        img_path, count = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.clip_preprocess(image)
        return image, count

    def __len__(self):
        return len(self.samples)

    @property
    def count_distribution(self):
        """Returns a dict {object_count: num_images} for dataset diagnostics."""
        return dict(sorted(self._count_distribution.items()))

    def __repr__(self):
        return (
            f"CLEVRCountingDataset(samples={len(self)}, "
            f"max_objects={self.max_objects}, "
            f"distribution={self.count_distribution})"
        )