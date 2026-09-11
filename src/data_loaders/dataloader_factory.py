"""Factory function to build DataLoaders from experiment config."""
import os
from pathlib import Path
import torch
import clip
from torch.utils.data import DataLoader
from .clevr_loader import CLEVRCountingDataset
from .spatialsense_loader import SpatialSenseDataset
from .imagenet_v2_loader import ImageNetV2Dataset


def build_dataloaders(config):
    """Build DataLoaders for datasets specified in the experiment config.
    Auto-detects dataset layouts (e.g. data/CLEVR_v1.0, data/images, etc.)
    and gracefully handles partially downloaded environments.

    Args:
        config: Parsed experiment config dict (from ``load_config()``).
    Returns:
        dict: Mapping of dataset name → DataLoader instance.
              Keys: "clevr", "spatialsense", "imagenet_v2"
              (only those present and found on disk are included).
    """
    # Load CLIP model to get the preprocess transform
    device = config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model_name = config.get("clip_model", "ViT-B/32")
    _, preprocess = clip.load(model_name, device=device)

    batch_size = config["inference"]["batch_size"]
    num_workers = config["inference"]["num_workers"]

    loaders = {}
    datasets_cfg = config.get("datasets", {})

    # --- CLEVR ---
    if "clevr" in datasets_cfg:
        clevr_cfg = datasets_cfg["clevr"]
        root = clevr_cfg.get("root", "data/clevr")
        split = clevr_cfg.get("split", "val")

        # Check candidate root directories (preferring those that actually contain images or scenes)
        candidate_roots = [
            Path(root),
            Path("data/CLEVR_v1.0"),
            Path("data/clevr"),
        ]
        clevr_root = None
        for cand in candidate_roots:
            if cand.is_dir() and ((cand / "images").is_dir() or (cand / "scenes").is_dir()):
                clevr_root = cand
                break
        if clevr_root is None:
            for cand in candidate_roots:
                if cand.is_dir():
                    clevr_root = cand
                    break

        if clevr_root is not None:
            try:
                # Resolve images directory
                img_dir = clevr_root / "images" / split
                if not img_dir.exists():
                    img_dir = clevr_root / "images"

                # Resolve annotations file
                ann_file = clevr_root / "scenes" / f"CLEVR_{split}_scenes.json"
                if not ann_file.exists():
                    ann_file = clevr_root / "annotations.json"

                clevr_dataset = CLEVRCountingDataset(
                    image_dir=str(img_dir),
                    annotations_path=str(ann_file),
                    clip_preprocess=preprocess,
                    max_objects=clevr_cfg.get("max_objects", 10),
                )
                loaders["clevr"] = DataLoader(
                    clevr_dataset,
                    batch_size=batch_size,
                    shuffle=False,       # Evaluation — deterministic order
                    num_workers=num_workers,
                    pin_memory=True,
                )
                print(f"[DataLoader] CLEVR: {clevr_dataset}")
            except Exception as e:
                print(f"[DataLoader] Notice: Could not initialize CLEVR ({e})")
        else:
            print(f"[DataLoader] Notice: CLEVR directory not found at '{root}' or fallbacks.")

    # --- SpatialSense ---
    if "spatialsense" in datasets_cfg:
        ss_cfg = datasets_cfg["spatialsense"]
        root = ss_cfg.get("root", "data/spatialsense")

        # Candidates for images and annotations
        img_candidates = [
            Path(root) / "images",
            Path("data/images"),
            Path(root),
        ]
        ann_candidates = [
            Path(root) / "annotations.json",
            Path("data/annotations.json"),
            Path("data/spatialsense/annotations.json"),
        ]

        img_dir = next((p for p in img_candidates if p.is_dir()), None)
        ann_file = next((p for p in ann_candidates if p.is_file()), None)

        if img_dir and ann_file:
            try:
                split = ss_cfg.get("split", "val")
                ss_dataset = SpatialSenseDataset(
                    image_dir=str(img_dir),
                    annotations_path=str(ann_file),
                    clip_preprocess=preprocess,
                    split=split,
                )
                loaders["spatialsense"] = DataLoader(
                    ss_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
                print(f"[DataLoader] SpatialSense: {ss_dataset}")
            except Exception as e:
                print(f"[DataLoader] Notice: Could not initialize SpatialSense ({e})")
        else:
            missing = []
            if not img_dir:
                missing.append(f"images dir ('{root}/images' or 'data/images')")
            if not ann_file:
                missing.append(f"annotations file ('{root}/annotations.json')")
            print(f"[DataLoader] Notice: SpatialSense skipped — missing {', '.join(missing)}.")

    # --- ImageNet-V2 ---
    if "imagenet_v2" in datasets_cfg:
        inv2_cfg = datasets_cfg["imagenet_v2"]
        root = inv2_cfg.get("root", "data/imagenet_v2")

        candidate_roots = [
            Path(root),
            Path("data/ImageNetV2-master/imagenetv2-matched-frequency-format-val"),
            Path("data/imagenetv2-matched-frequency-format-val"),
            Path("data/imagenetv2-matched-frequency"),
            Path("data/ImageNetV2-master/imagenetv2-matched-frequency"),
            Path("data/imagenet_v2"),
        ]
        inv2_root = next((p for p in candidate_roots if p.is_dir() and (p / "0").is_dir()), None)

        if inv2_root:
            try:
                inv2_dataset = ImageNetV2Dataset(
                    root_dir=str(inv2_root),
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
            except Exception as e:
                print(f"[DataLoader] Notice: Could not initialize ImageNet-V2 ({e})")
        else:
            print(
                f"[DataLoader] Notice: ImageNet-V2 skipped — class subfolders (0/, 1/, ..., 999/) "
                f"not found in '{root}' or candidate paths."
            )

    return loaders
