"""scripts/evaluate_retrieval_coco.py

Evaluates cross-modal retrieval (Recall@1, Recall@5, Recall@10, Median Rank)
and geometric alignment (Modality Gap, Cross-Modal Alignment) on the MS-COCO 2017
validation benchmark (5,000 images, 25,000 human captions).

Validates:
1. Exact 1-to-1 Retrieval (5,000 unique image-caption pairs) using from-scratch
   functions: recall_at_k, median_rank, modality_gap, cross_modal_alignment
   from src/evaluation/multimodal_metrics.py.
2. Standard Multi-Caption COCO Retrieval (5,000 images x 25,000 captions)
   matching published literature (Radford et al. 2021, Table 11).

Saves results to: results/tables/coco_retrieval_results.json
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.clip_encoder import CLIPEncoder
from src.evaluation.multimodal_metrics import (
    recall_at_k,
    median_rank,
    modality_gap,
    cross_modal_alignment,
)


class COCOValDataset(Dataset):
    """PyTorch Dataset for MS-COCO 2017 Validation Images."""

    def __init__(self, image_paths: List[Path], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                tensor = self.transform(img)
            else:
                tensor = img
        return tensor, idx


def load_coco_metadata(
    annotations_path: Path, images_dir: Path
) -> Tuple[List[dict], Dict[int, List[str]], List[Path]]:
    """Loads COCO annotations and matches available image files.

    Returns:
        images_info: list of valid image metadata dicts (id, file_name)
        img_to_captions: mapping from image_id -> list of 5 captions
        image_paths: list of Path objects for existing images
    """
    print(f"[1/4] Loading COCO annotations from {annotations_path}...")
    with open(annotations_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Map image_id -> captions
    img_to_captions: Dict[int, List[str]] = {}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        cap = ann["caption"].strip()
        img_to_captions.setdefault(img_id, []).append(cap)

    # Filter to images that actually exist on disk and have captions
    images_info = []
    image_paths = []
    for img_meta in data.get("images", []):
        img_id = img_meta["id"]
        file_name = img_meta["file_name"]
        path = images_dir / file_name
        if path.is_file() and img_id in img_to_captions:
            images_info.append(img_meta)
            image_paths.append(path)

    print(f"  Loaded {len(images_info)} images with valid files and captions.")
    return images_info, img_to_captions, image_paths


def encode_all_images(
    encoder: CLIPEncoder, image_paths: List[Path], batch_size: int = 128
) -> torch.Tensor:
    """Encodes all images using CLIPEncoder and returns normalized embeddings (N, 512)."""
    dataset = COCOValDataset(image_paths, transform=encoder.preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    all_embeds = []
    total_batches = len(dataloader)
    print(f"[2/4] Encoding {len(image_paths)} images with {encoder.model_name}...")

    t0 = time.time()
    for batch_idx, (batch_tensors, _) in enumerate(dataloader):
        with torch.no_grad():
            embeds = encoder.encode_images(batch_tensors)
            all_embeds.append(embeds.cpu())

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            elapsed = time.time() - t0
            sys.stdout.write(
                f"\r  Image Batch {batch_idx + 1}/{total_batches} ({len(all_embeds) * batch_size}/{len(image_paths)} images) [{elapsed:.1f}s]"
            )
            sys.stdout.flush()

    print(f"\n  [Done] Extracted image embeddings in {time.time() - t0:.2f}s")
    return torch.cat(all_embeds, dim=0)


def encode_all_texts(
    encoder: CLIPEncoder, captions: List[str], batch_size: int = 256
) -> torch.Tensor:
    """Encodes text list in batches using CLIPEncoder and returns normalized embeddings (M, 512)."""
    all_embeds = []
    total = len(captions)
    print(f"[3/4] Encoding {total} captions with {encoder.model_name}...")

    t0 = time.time()
    for i in range(0, total, batch_size):
        batch = captions[i : i + batch_size]
        with torch.no_grad():
            embeds = encoder.encode_texts(batch)
            all_embeds.append(embeds.cpu())

        if (i // batch_size + 1) % 20 == 0 or (i + batch_size) >= total:
            elapsed = time.time() - t0
            sys.stdout.write(
                f"\r  Caption Batch {min(i + batch_size, total)}/{total} captions [{elapsed:.1f}s]"
            )
            sys.stdout.flush()

    print(f"\n  [Done] Extracted caption embeddings in {time.time() - t0:.2f}s")
    return torch.cat(all_embeds, dim=0)


def evaluate_standard_coco_5k(
    img_feats: torch.Tensor,
    txt_feats: torch.Tensor,
    img_ids: List[int],
    txt_img_ids: List[int],
) -> Dict[str, Dict[str, float]]:
    """Evaluates standard 5-caption COCO retrieval benchmark (5k images x 25k captions).

    Matches OpenAI CLIP evaluation protocol (Radford et al. 2021 Table 11):
    - Text-to-Image (T2I): for each of 25,000 texts, ranks all 5,000 images.
      Success if the single ground-truth image is in top-K.
    - Image-to-Text (I2T): for each of 5,000 images, ranks all 25,000 captions.
      Success if ANY of the 5 ground-truth captions is in top-K.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_t = img_feats.to(device)
    txt_t = txt_feats.to(device)

    N_img = img_feats.size(0)
    N_txt = txt_feats.size(0)

    # 1. Image-to-Text (I2T): 5,000 x 25,000 similarity
    # We do in chunks if needed, but 5000 x 25000 = 125M floats = 500MB, fits easily in VRAM
    print("  Computing I2T similarity matrix (5,000 x 25,000)...")
    sim_i2t = img_t @ txt_t.T  # (5000, 25000)

    # Build image_id -> caption indices mapping
    txt_img_arr = np.array(txt_img_ids)
    i2t_ranks = []

    for i in range(N_img):
        target_img_id = img_ids[i]
        # Valid caption indices for this image
        gt_cap_indices = np.where(txt_img_arr == target_img_id)[0]
        # Rank all 25000 captions descending
        sorted_caps = torch.argsort(sim_i2t[i], descending=True).cpu().numpy()
        # Find highest rank among the 5 GT captions
        match_positions = [np.where(sorted_caps == cap_idx)[0][0] + 1 for cap_idx in gt_cap_indices]
        best_rank = min(match_positions)
        i2t_ranks.append(best_rank)

    i2t_ranks_t = torch.tensor(i2t_ranks, dtype=torch.float32)

    # 2. Text-to-Image (T2I): 25,000 x 5,000 similarity
    print("  Computing T2I similarity matrix (25,000 x 5,000)...")
    sim_t2i = sim_i2t.T  # (25000, 5000)
    img_id_arr = np.array(img_ids)
    t2i_ranks = []

    for j in range(N_txt):
        target_img_id = txt_img_ids[j]
        # The single target image index in 0..N_img-1
        gt_img_idx = np.where(img_id_arr == target_img_id)[0][0]
        # Rank all 5000 images descending
        sorted_imgs = torch.argsort(sim_t2i[j], descending=True).cpu().numpy()
        rank = np.where(sorted_imgs == gt_img_idx)[0][0] + 1
        t2i_ranks.append(rank)

    t2i_ranks_t = torch.tensor(t2i_ranks, dtype=torch.float32)

    return {
        "i2t": {
            "R@1": float((i2t_ranks_t <= 1).sum().item()) / N_img * 100.0,
            "R@5": float((i2t_ranks_t <= 5).sum().item()) / N_img * 100.0,
            "R@10": float((i2t_ranks_t <= 10).sum().item()) / N_img * 100.0,
            "median_rank": float(i2t_ranks_t.median().item()),
            "mean_rank": float(i2t_ranks_t.mean().item()),
        },
        "t2i": {
            "R@1": float((t2i_ranks_t <= 1).sum().item()) / N_txt * 100.0,
            "R@5": float((t2i_ranks_t <= 5).sum().item()) / N_txt * 100.0,
            "R@10": float((t2i_ranks_t <= 10).sum().item()) / N_txt * 100.0,
            "median_rank": float(t2i_ranks_t.median().item()),
            "mean_rank": float(t2i_ranks_t.mean().item()),
        },
    }


def main():
    print("=" * 75)
    print("CROSS-MODAL RETRIEVAL EVALUATION ON MS-COCO 2017 VALIDATION")
    print("=" * 75)

    coco_dir = Path("data/coco")
    images_dir = coco_dir / "val2017"
    captions_file = coco_dir / "annotations" / "captions_val2017.json"

    if not images_dir.exists() or not captions_file.exists():
        print(f"[Error] COCO dataset missing. Run download_coco_val.py first.")
        sys.exit(1)

    # 1. Load COCO metadata
    images_info, img_to_captions, image_paths = load_coco_metadata(captions_file, images_dir)

    # Prepare lists
    img_ids = [m["id"] for m in images_info]
    # 1-to-1: First caption for each image
    captions_1to1 = [img_to_captions[m["id"]][0] for m in images_info]

    # Full 5-captions list
    all_captions = []
    all_caption_img_ids = []
    for m in images_info:
        img_id = m["id"]
        for c in img_to_captions[img_id]:
            all_captions.append(c)
            all_caption_img_ids.append(img_id)

    print(f"  Prepared {len(captions_1to1)} 1-to-1 pairs and {len(all_captions)} full captions.")

    # 2. Initialize CLIP ViT-B/32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Initializing CLIP ViT-B/32 on device: {device}")
    encoder = CLIPEncoder(model_name="ViT-B/32", device=device)

    # 3. Encode Images (5,000 x 512)
    img_feats = encode_all_images(encoder, image_paths, batch_size=128)

    # 4. Encode 1-to-1 Text (5,000 x 512)
    txt_feats_1to1 = encode_all_texts(encoder, captions_1to1, batch_size=256)

    # 5. Encode All 25k Texts (25,000 x 512)
    txt_feats_all = encode_all_texts(encoder, all_captions, batch_size=256)

    print("\n[4/4] Computing Cluster 6B Retrieval Metrics...")

    # Evaluation A: Exact 1-to-1 Matching using from-scratch multimodal_metrics.py functions
    print("\n--- [A] 1-to-1 Instance Retrieval (5,000 Unique Pairs) ---")
    print("Using src/evaluation/multimodal_metrics.py (recall_at_k, median_rank):")
    r_1to1 = recall_at_k(img_feats, txt_feats_1to1, k_values=(1, 5, 10))
    ranks_1to1 = median_rank(img_feats, txt_feats_1to1)
    mg_1to1 = modality_gap(img_feats, txt_feats_1to1)
    cma_1to1 = cross_modal_alignment(img_feats, txt_feats_1to1)

    print(f"  Image->Text Recall@1  : {r_1to1['i2t']['R@1']:.2f}%")
    print(f"  Image->Text Recall@5  : {r_1to1['i2t']['R@5']:.2f}%")
    print(f"  Image->Text Recall@10 : {r_1to1['i2t']['R@10']:.2f}%")
    print(f"  Image->Text Median Rnk: {ranks_1to1['i2t_median_rank']:.1f}")
    print(f"  Text->Image Recall@1  : {r_1to1['t2i']['R@1']:.2f}%")
    print(f"  Text->Image Recall@5  : {r_1to1['t2i']['R@5']:.2f}%")
    print(f"  Text->Image Recall@10 : {r_1to1['t2i']['R@10']:.2f}%")
    print(f"  Text->Image Median Rnk: {ranks_1to1['t2i_median_rank']:.1f}")
    print(f"  Modality Gap (Delta)  : {mg_1to1:.4f}")
    print(f"  Cross-Modal Alignment : {cma_1to1:.4f}")

    # Evaluation B: Standard Multi-Caption COCO Benchmark (5,000 images x 25,000 captions)
    print("\n--- [B] Standard 5-Caption COCO 5k Benchmark (Radford et al. 2021 Table 11) ---")
    std_results = evaluate_standard_coco_5k(img_feats, txt_feats_all, img_ids, all_caption_img_ids)

    print(f"  Image->Text Recall@1  : {std_results['i2t']['R@1']:.2f}%  (Expected: ~58.4%)")
    print(f"  Image->Text Recall@5  : {std_results['i2t']['R@5']:.2f}%  (Expected: ~81.5%)")
    print(f"  Image->Text Recall@10 : {std_results['i2t']['R@10']:.2f}% (Expected: ~88.3%)")
    print(f"  Image->Text Median Rnk: {std_results['i2t']['median_rank']:.1f}  (Expected: 1.0)")
    print(f"  Text->Image Recall@1  : {std_results['t2i']['R@1']:.2f}%  (Expected: ~37.8%)")
    print(f"  Text->Image Recall@5  : {std_results['t2i']['R@5']:.2f}%  (Expected: ~62.4%)")
    print(f"  Text->Image Recall@10 : {std_results['t2i']['R@10']:.2f}% (Expected: ~72.2%)")
    print(f"  Text->Image Median Rnk: {std_results['t2i']['median_rank']:.1f}  (Expected: 2.0-3.0)")

    # 6. Save JSON results
    output_path = Path("results/tables/coco_retrieval_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "dataset": "MS-COCO 2017 Validation",
        "num_images": len(images_info),
        "num_captions": len(all_captions),
        "model": encoder.model_name,
        "one_to_one_5k_evaluation": {
            "recall": r_1to1,
            "ranks": ranks_1to1,
            "modality_gap": mg_1to1,
            "cross_modal_alignment": cma_1to1,
        },
        "standard_multi_caption_5k_benchmark": std_results,
        "expected_clip_vit_b32_literature": {
            "i2t_R@1": 58.4,
            "i2t_R@5": 81.5,
            "i2t_R@10": 88.3,
            "i2t_median_rank": 1.0,
            "t2i_R@1": 37.8,
            "t2i_R@5": 62.4,
            "t2i_R@10": 72.2,
            "t2i_median_rank": 2.0,
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"\n[OK] Results successfully saved to: {output_path.resolve()}")
    print("=" * 75)


if __name__ == "__main__":
    main()
