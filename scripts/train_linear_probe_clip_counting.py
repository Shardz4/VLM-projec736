"""Train & evaluate a linear probe on frozen CLIP ViT-B/32 visual embeddings for CLEVR counting.

Probes whether CLIP's vision backbone internally encodes object count/cardinality,
isolating whether counting failure is visual or a language-projection breakdown.
"""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from src.config_loader import load_config
from src.data_loaders.clevr_loader import CLEVRCountingDataset
from src.models.clip_encoder import CLIPEncoder
from src.models.resnet_baseline import LinearProbeTrainer
from src.utils import set_seed, seed_worker


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train & evaluate linear probe on CLIP visual embeddings for CLEVR counting"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay (default: 0.01)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--cache-dir", type=str, default="results/cache", help="Cache directory")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="results/checkpoints/clip_counting",
        help="Model checkpoint directory",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B experiment tracking")
    parser.add_argument("--force-extract", action="store_true", help="Force re-extraction of features")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 70)
    print("Linear Probe: CLIP ViT-B/32 Image Features on CLEVR Counting")
    print("=" * 70)

    config = load_config()
    device = config.get("device", "cuda")
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
    g = torch.Generator().manual_seed(args.seed)

    clevr_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    print(f"      CLEVR samples: {len(dataset)}")

    # 3. Extract & cache CLIP image features (512-d)
    print("\n[3/4] Extracting / Loading CLIP image features...")
    cache_path = Path(args.cache_dir) / "clevr_clip_vit_features.pt"

    if args.force_extract and cache_path.is_file():
        cache_path.unlink()

    features, labels = encoder.extract_and_cache(clevr_loader, cache_path=str(cache_path))
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

    # Deterministic split (80/20)
    n = len(features)
    perm_gen = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=perm_gen)
    split = int(0.8 * n)
    train_idx, test_idx = perm[:split], perm[split:]

    train_feats, train_labels = features[train_idx], labels[train_idx]
    test_feats, test_labels = features[test_idx], labels[test_idx]
    print(f"      Train: {len(train_feats)} | Test: {len(test_feats)}")

    # 4. Train linear probe on frozen CLIP embeddings
    print(f"\n[4/4] Training Linear probe ({args.epochs} epochs, LR={args.lr})...")
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = not args.no_wandb
    trainer = LinearProbeTrainer(
        feature_dim=encoder.embedding_dim,
        num_classes=num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        device=device,
        checkpoint_dir=str(ckpt_dir),
        use_wandb=use_wandb,
        wandb_project=config.get("linear_probe", {}).get("wandb_project", "vlm-linear-probe"),
        wandb_run_name="counting-probe-clip-vit",
        wandb_config={
            "task": "clevr_counting_clip_vit",
            "backbone": model_name,
            "feature_dim": encoder.embedding_dim,
            "num_classes": num_classes,
            "lr": args.lr,
            "epochs": args.epochs,
            "seed": args.seed,
        },
    )

    history = trainer.train(
        train_feats,
        train_labels,
        batch_size=args.batch_size,
        val_features=test_feats,
        val_labels=test_labels,
    )

    print("\nFinal evaluation on test set...")
    results = trainer.evaluate(test_feats, test_labels)

    print("\n" + "=" * 70)
    print("LINEAR PROBE ON CLIP ViT-B/32 FEATURES RESULTS:")
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
        "model": "clip_vit_b32_linear_probe",
        "feature_dim": encoder.embedding_dim,
        "seed": args.seed,
        "lr": args.lr,
        "epochs": args.epochs,
        **results,
        "training_history": history,
        "idx_to_count": idx_to_count,
    }
    output_path = results_dir / "linear_probe_clip_counting.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
