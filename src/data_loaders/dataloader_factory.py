""" Convinience based function to build Dataloders from experiment config"""

import os
import clip
from torch.utils.data import DataLoader

from.clevr_loader import CLEVRCountingDataset
from .spatialsense_loader import SpatialSenseDataset

from .imagenet_v2_loader import ImageNetV2Dataset

def build_dataloaders(config):

    device = config.get("device", "cuda")
    model_name = config.get("clip_model", "ViT-B/32")
    _, preprocess = clip.load(model_name, device=device)

    batch_size = config = config["inference"]["batch_size"]
    num_workers = config["inference"]["num_workers"]

    loaders = {}
    dataset_cfg = config.get("datasets", {})

    # CLEVR

    if "clevr" in datasets_cfg:
        clevr_cfg = datasets_cfg["clevr"]
        root = clevr_cfg["root"]
        clevr_dataset = CLEVRCountingDataset(
            image_dir=os.path.join(root, "images"),
            annotations_path=os.path.join(root, "annotations.json"),
            clip_preprocess=preprocess,
            max_objects=clevr_cfg.get("max_objects", 10),
        )
        loaders["clevr"] = DataLoader(
            clevr_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        pritn(f"[DataLoader] CLEVR: {clevr_dataset}")
    
    # SpatialSense
    if "spatialsense" in datasets_cfg:
        ss_cfg = datasets_cfg["spatialsense"]
        root = ss_cfg["root"]
        ss_dataset = SpatialSenseDataset(
            image_dir=os.path.join(root, "images"),
            annotations_path=os.path.join(root, "annotations.json"),
            clip_preprocess=preprocess,
        )
        loaders["spatialsense"] = DataLoader(
            ss_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        print(f"[DataLoader] SpatialSense: {ss_dataset}")

    # ImageNet-V2 
    if "imagenet_v2" in datasets_cfg:
        inv2_cfg = datasets_cfg["imagenet_v2"]
        root = inv2_cfg["root"]
        inv2_dataset = ImageNetV2Dataset(
            root_dir=root,
            clip_preprocess=preprocess,
        )
        loaders["imagenet_v2"] = DataLoader(
            inv2_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        print(f"[DataLoader] ImageNet-V2: {inv2_dataset}")
    return loaders

