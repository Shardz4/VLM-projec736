"""Train & evaluate a linear probe on frozen CLIP ViT-B/32 visual embeddings for CLEVR counting.

Probes whether CLIP's vision backbone internally encodes object count/cardinality,
isolating whether counting failure is visual or a language-projection breakdown.
"""
import json
import os
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from src.config_loader import load_config
from src.data_loaders.clevr_loader import CLEVRCountingDataset
from src.models.clip_encoder import CLIPEncoder
from src.models.resnet_baseline import LinearProbeTrainer


def main():
    print("=" * 70)
    print("Linear Probe: CLIP ViT-B/32 Image Features on CLEVR Counting")
    print("=" * 70)

    config = load_config()
    device = config.get("device", "cuda")
    lp_cfg = config.get("linear_probe", {})
    clevr_cfg = config.get("datasets", {}).get("clevr", {})
    model_name = config.get("clip_model", "ViT-B/32")

    # 1. Initialize CLIP Encoder
    print(f"\n[1/4] Initializing CLIP model ({model_name})...")
    encoder = CLIPEncoder(model_name=model_name, device=device)

    # 2. Build CLEVR DataLoader with CLIP preprocessing
    print("\n[2/4] Building CLEVR DataLoader with CLIP preprocessing...")
    clevr_root = Path(clevr_cfg.get("root", "data/CLEVR_v1.0"))
    max_objects = clevr_cfg.get("max_objects", 10)

    dataset = CLEVRCountingDataset(
        image_dir=clevr_root,
        annotations_path=clevr_root,
        clip_preprocess=encoder.preprocess,
        max_objects=max_objects,
    )

    batch_size = config.get("inference", {}).get("batch_size", 32)
    num_workers = config.get("inference", {}).get("num_workers", 4)
    clevr_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    print(f"      CLEVR samples: {len(dataset)}")

    # 3. Extract & cache CLIP image features (512-d)
    print("\n[3/4] Extracting CLIP image features...")
    cache_path = Path("results/cache/clevr_clip_vit_features.pt")
    if cache_path.is_file():
        print(f"      Loading cached features from '{cache_path}'")
        data = torch.load(cache_path, weights_only=True)
        features, labels = data["features"], data["labels"]
    else:
        all_features = []
        all_labels = []
        for batch in tqdm(clevr_loader, desc="Extracting CLIP features", unit="batch"):
            images = batch[0].to(device)
            batch_labels = batch[1]
            with torch.no_grad():
                feats = encoder.encode_images(images)  # (B, 512) normalized
            all_features.append(feats.cpu())
            all_labels.append(batch_labels)
        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": features, "labels": labels}, cache_path)
        print(f"      Cached features to '{cache_path}'")

    print(f"      Features shape: {features.shape}")
    print(f"      Labels shape: {labels.shape}")

    # Contiguous mapping
    raw_counts = labels.long()
    unique_counts = sorted(set(raw_counts.tolist()))
    count_to_idx = {c: i for i, c in enumerate(unique_counts)}
    idx_to_count = {i: c for c, i in count_to_idx.items()}
    labels = torch.tensor([count_to_idx[c.item()] for c in raw_counts])
    num_classes = len(unique_counts)
    print(f"      Unique counts: {unique_counts} → {num_classes} classes")

    # Split into train (80%) and test (20%)
    n = len(features)
    perm = torch.randperm(n)
    split = int(0.8 * n)
    train_idx, test_idx = perm[:split], perm[split:]

    train_feats, train_labels = features[train_idx], labels[train_idx]
    test_feats, test_labels = features[test_idx], labels[test_idx]
    print(f"      Train: {len(train_feats)} | Test: {len(test_feats)}")

    # 4. Train linear probe
    print("\n[4/4] Training linear probe on CLIP vision features...")
    lr = lp_cfg.get("learning_rate", 1e-4)
    probe_lr = max(lr, 1e-3)
    epochs = lp_cfg.get("epochs", 50)
    probe_epochs = max(epochs, 100)

    trainer = LinearProbeTrainer(
        feature_dim=features.shape[1],  # 512
        num_classes=num_classes,
        lr=probe_lr,
        weight_decay=lp_cfg.get("weight_decay", 0.01),
        epochs=probe_epochs,
        device=device,
        early_stopping_patience=None,
        checkpoint_dir="results/checkpoints/clip_counting",
        use_wandb=True,
        wandb_project="vlm-linear-probe",
        wandb_run_name="clip-counting-probe",
        wandb_config={
            "task": "clevr_counting_clip_vit",
            "num_classes": num_classes,
            "counts": unique_counts,
            "feature_dim": features.shape[1],
            "lr": probe_lr,
            "epochs": probe_epochs,
        },
    )

    history = trainer.train(
        train_feats, train_labels,
        batch_size=64,
        val_features=test_feats,
        val_labels=test_labels,
    )

    print("\nFinal evaluation on test set...")
    results = trainer.evaluate(test_feats, test_labels)
    print("\n" + "=" * 70)
    print("CLIP VISION LINEAR PROBE COUNTING RESULTS:")
    print(f"  Overall Accuracy : {results['accuracy'] * 100:.2f}%")
    print(f"  MAE              : {results['mae']:.3f}")
    print(f"  Total Test       : {results['total_samples']}")
    print("=" * 70)

    print("\nPER-COUNT ACCURACY:")
    for cls_idx, stats in results["per_class"].items():
        count = idx_to_count.get(cls_idx, cls_idx)
        print(f"  Count {count:2d}: {stats['accuracy']*100:6.1f}% ({stats['correct']}/{stats['total']})")

    results_dir = Path(config.get("output", {}).get("results_dir", "results/logs"))
    results_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "task": "clevr_counting",
        "model": "clip_vit_b32_linear_probe",
        **results,
        "training_history": history,
        "config": lp_cfg,
        "idx_to_count": idx_to_count,
    }
    output_path = results_dir / "linear_probe_clip_counting.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
