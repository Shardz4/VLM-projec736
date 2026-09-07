import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class ImageNetV2Dataset(Dataset):

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG"}

    def __init__(self, root_dir, clip_preprocess, class_names = None):
        self.root_dir = Path(root_dir)
        self.clip_preprocess = clip_preprocess
        self.class_names = class_names
        self.samples = []

        for class_folder in sorted(self.root_dir.iterdir()):
            if not class_folder.is_dir():
                continue
            try:
                class_idx = int(class_folder.name)
            except ValueError:
                continue
            
            
            for img_file in sorted(class_folder.iterdir()):
                if img_file.suffix in self.VALID_EXTENSIONS:
                    self.samples.append((str(img_file), class_idx))
        
        if len(self.samples) ==0:
            raise RuntimeError(
                f"No valid images found"
                f"Expected class_indexed subfolders (0/, 1/, ....., 999/)"
                )

        self._num_classes = len(set(idx for _, idx in self.samples))

    def __len__(self):
        return len(self.samples)

    def get_class_name(self, class_idx):
        if self.class_names and 0<= class_idx < len(self.class_names):
            return self.class_names[class_idx]
        return str(class_idx)
    
    @property
    def num_classes(Self):
        return self._num_classes
    
    def __repr__(self):
        return (
            f"ImageNetV2Dataset(samples={len(self)}"
            f"classes={self._num_classes})"
        )
