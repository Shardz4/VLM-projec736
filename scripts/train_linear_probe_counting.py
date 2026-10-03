"""Step 10a: Train & evaluate a linear probe on CLEVR counting.

Uses ResNet-50 frozen features (2048-d) + nn.Linear(2048, K).
Builds its own DataLoader with ResNet's ImageNet preprocessing
(not CLIP's) to ensure feature quality.
"""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from src.config_loader import load_config
from src.data_loaders.clevr_loader import CLEVRCountingDataset
from src.models.resnet_baseline import ResNetFeatureExtractor, LinearProbeTrainer


def main():
    print("="*70)
    print("Linear Probe: CLEVR Object Counting")
    print("="*70)

    config = load_config()
    device = config.get("device", "cuda")
    lp_cfg = config.get("linear_probe", {})
    clevr_cfg = config.get("datasets", {}).get("clevr", {})

    # 1. Build ResNet extractor with SPATIAL features (mean+max+std of 7x7 map → 6144-d)
    #    avgpool destroys spatial info needed for counting
    print("\n[1/4] Initializing ResNet-50 feature extractor (spatial mode)...")
    extractor = ResNetFeatureExtractor(device=device, spatial=True)

    # 2. Build CLEVR DataLoader with ResNet's ImageNet preprocessing (NOT CLIP's)
    print("\n[2/4] Building CLEVR DataLoader with ResNet preprocessing...")
    clevr_root = clevr_cfg.get("root", "data/CLEVR_v1.0")
    max_objects = clevr_cfg.get("max_objects", 10)

    # Resolve image dir and annotations
    clevr_root_path = Path(clevr_root)
    image_dir = clevr_root_path
    annotations_path = clevr_root_path

    dataset = CLEVRCountingDataset(
        image_dir=image_dir,
        annotations_path=annotations_path,
        clip_preprocess=extractor.preprocess,  # Use ResNet's preprocessing!
        max_objects=max_objects,
    )

    batch_size = config.get("inference", {}).get("batch_size", 32)
    num_workers = config.get("inference", {}).get("num_workers", 4)
    clevr_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    total_samples = len(dataset)
    print(f"      CLEVR samples: {total_samples}")
    print(f"      Preprocessing: ResNet ImageNet (NOT CLIP)")

    # 3. Extract features
    print("\n[3/4] Extracting ResNet-50 features...")
    features, labels = extractor.extract_and_cache(
        clevr_loader, cache_path="results/cache/clevr_resnet_spatial_features.pt"
    )
    print(f"      Features shape: {features.shape}")
    print(f"      Labels shape: {labels.shape}")

    # Build contiguous class indices from actual counts in the dataset
    raw_counts = labels.long()
    unique_counts = sorted(set(raw_counts.tolist()))
    count_to_idx = {c: i for i, c in enumerate(unique_counts)}
    idx_to_count = {i: c for c, i in count_to_idx.items()}
    labels = torch.tensor([count_to_idx[c.item()] for c in raw_counts])
    num_classes = len(unique_counts)
    print(f"      Unique counts: {unique_counts} → {num_classes} classes")

    # Train/test split
    n = len(features)
    perm = torch.randperm(n)
    split = int(0.8 * n)
    train_idx, test_idx = perm[:split], perm[split:]

    train_feats, train_labels = features[train_idx], labels[train_idx]
    test_feats, test_labels = features[test_idx], labels[test_idx]
    print(f"      Train: {len(train_feats)} | Test: {len(test_feats)}")

    # 4. Train linear probe with higher LR (standard for linear probes)
    print("\n[4/4] Training Linear probe...")
    lr = lp_cfg.get("learning_rate", 1e-4)
    probe_lr = max(lr, 1e-3)  # Linear probes need higher LR than deep networks
    epochs = lp_cfg.get("epochs", 50)
    probe_epochs = max(epochs, 100)  # More epochs for convergence

    trainer = LinearProbeTrainer(
        feature_dim=extractor.feature_dim,
        num_classes=num_classes,
        lr=probe_lr,
        weight_decay=lp_cfg.get("weight_decay", 0.01),
        epochs=probe_epochs,
        device=device,
        checkpoint_dir="results/checkpoints/counting",
        use_wandb=True,
        wandb_project="vlm-linear-probe",
        wandb_run_name="counting-probe-spatial",
        wandb_config={
            "task": "clevr_counting",
            "num_classes": num_classes,
            "counts": unique_counts,
            "preprocess": "resnet_imagenet",
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
    print("LINEAR PROBE COUNTING RESULTS:")
    print(f"  Overall Accuracy : {results['accuracy'] * 100:.2f}%")
    print(f"  MAE              : {results['mae']:.3f}")
    print(f"  Total Test       : {results['total_samples']}")
    print("=" * 70)
    print("\nPER-COUNT ACCURACY:")
    for cls_idx, stats in results["per_class"].items():
        count = idx_to_count.get(cls_idx, cls_idx)
        print(f"  Count {count:2d}: {stats['accuracy']*100:6.1f}% ({stats['correct']}/{stats['total']})")

    # Save results
    results_dir = Path(config.get("output", {}).get("results_dir", "results/logs"))
    results_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "task": "clevr_counting",
        "model": "resnet50_linear_probe",
        "preprocess": "resnet_imagenet",
        **results,
        "training_history": history,
        "config": lp_cfg,
        "idx_to_count": idx_to_count,
    }
    output_path = results_dir / "linear_probe_counting.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to: {output_path}")
    
if __name__ == "__main__":
    main()