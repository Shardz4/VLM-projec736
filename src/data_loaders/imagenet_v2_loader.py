"""ImageNet-V2 dataset loader.
Used as a baseline sanity check: if CLIP achieves ~60% zero-shot top-1
accuracy on ImageNet-V2 (matched-frequency variant), the pipeline is
functioning correctly.
"""
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class ImageNetV2Dataset(Dataset):
    """PyTorch Dataset for ImageNet-V2 zero-shot classification baseline.
    Expected directory layout (matched-frequency variant):
        data/imagenet_v2/
        ├── 0/          # Class folder (class index)
        │   ├── 0.jpeg
        │   ├── 1.jpeg
        │   └── ...
        ├── 1/
        └── ...         # Up to 999/
    Args:
        root_dir:        Root directory of ImageNet-V2.
        clip_preprocess: The ``preprocess`` transform returned by
                        ``clip.load()``. Applied to every image.
        class_names:     Optional list of 1000 ImageNet class names
                        (index-aligned). If None, integer class indices
                        are used as labels.
    """
    # Standard ImageNet-V2 image extensions
    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG"}

    def __init__(self, root_dir, clip_preprocess, class_names=None):
        self.root_dir = Path(root_dir)
        self.clip_preprocess = clip_preprocess
        self.class_names = class_names

        # Discover all (image_path, class_index) pairs
        self.samples = []
        for class_folder in sorted(self.root_dir.iterdir()):
            if not class_folder.is_dir():
                continue
            try:
                class_idx = int(class_folder.name)
            except ValueError:
                continue  # Skip non-integer folder names

            for img_file in sorted(class_folder.iterdir()):
                if img_file.suffix in self.VALID_EXTENSIONS:
                    self.samples.append((str(img_file), class_idx))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid images found in '{self.root_dir}'. "
                f"Expected class-indexed subfolders (0/, 1/, ..., 999/)."
            )

        # Compute basic stats
        self._num_classes = len(set(idx for _, idx in self.samples))

    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.clip_preprocess(image)
        return image, class_idx

    def __len__(self):
        return len(self.samples)

    def get_class_name(self, class_idx):
        """Return the human-readable class name for a given index."""
        if self.class_names and 0 <= class_idx < len(self.class_names):
            return self.class_names[class_idx]
        return str(class_idx)

    @property
    def num_classes(self):
        """Number of unique classes discovered in the dataset."""
        return self._num_classes

    def __repr__(self):
        return (
            f"ImageNetV2Dataset(samples={len(self)}, "
            f"classes={self._num_classes})"
        )
