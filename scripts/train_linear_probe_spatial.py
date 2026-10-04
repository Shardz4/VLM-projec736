"""Step 10b: Train & evaluate a linear probe on SpatialSense (binary classification).

Supports two architectural paradigms:
1. Conditioned (Default): Combines visual features with the relation query via
   RelationConditionedLinearProbe, enabling the probe to evaluate whether the specific
   spatial relation holds.
2. Unconditioned Baseline: Feeds only image features (2048-d) into nn.Linear(2048, 2),
   demonstrating the ~50% random chance baseline when query context is withheld.
"""

import argparse
import json
import os
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from src.config_loader import load_config
from src.data_loaders.spatialsense_loader import SpatialSenseDataset
from src.models.resnet_baseline import (
    ResNetFeatureExtractor,
    LinearProbeTrainer,
    RelationProbeTrainer,
)
from src.utils import set_seed, seed_worker


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train & evaluate linear probe on SpatialSense binary classification"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["avgpool", "spatial"],
        default="avgpool",
        help="Feature pooling mode: 'avgpool' (2048-d) or 'spatial' (6144-d)",
    )
    parser.add_argument(
        "--unconditioned",
        action="store_true",
        help="Train unconditioned image-only probe (without relation query context)",
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
        default="results/checkpoints/spatial",
        help="Model checkpoint directory",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B experiment tracking")
    parser.add_argument("--force-extract", action="store_true", help="Force re-extraction of features")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    conditioned = not args.unconditioned
    cond_str = "CONDITIONED" if conditioned else "UNCONDITIONED (IMAGE-ONLY)"

    print("=" * 70)
    print(f"Linear Probe: SpatialSense Binary Classification [{cond_str}]")
    print("=" * 70)

    config = load_config()
    device = config.get("device", "cuda")
    spatial_cfg = config.get("datasets", {}).get("spatialsense", {})

    use_spatial = (args.mode == "spatial")
    mode_name = "spatial (6144-d)" if use_spatial else "avgpool (2048-d)"

    # 1. Initialize ResNet-50 extractor
    print(f"\n[1/4] Initializing ResNet-50 feature extractor in {mode_name} mode...")
    extractor = ResNetFeatureExtractor(device=device, spatial=use_spatial)

    # 2. Build SpatialSense DataLoader
    print("\n[2/4] Building SpatialSense DataLoader with ResNet preprocessing...")
    root = spatial_cfg.get("root", "data/spatialsense")
    ann_path = Path(root) / "annotations.json"

    dataset = SpatialSenseDataset(
        image_dir=root,
        annotations_path=ann_path,
        clip_preprocess=extractor.preprocess,
        split=spatial_cfg.get("split", "val"),
    )

    batch_size = config.get("inference", {}).get("batch_size", 32)
    num_workers = config.get("inference", {}).get("num_workers", 4)
    g = torch.Generator().manual_seed(args.seed)

    spatial_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    total_samples = len(dataset)
    print(f"      SpatialSense samples: {total_samples}")
    print(f"      Relations: {dataset.relations}")

    # 3. Extract features and relation indices
    print("\n[3/4] Extracting / Loading ResNet-50 features & relations...")
    cache_path = Path(args.cache_dir) / f"spatial_resnet_{args.mode}_features.pt"

    if args.force_extract and cache_path.is_file():
        cache_path.unlink()

    if cache_path.is_file():
        print(f"      Loading cached data from '{cache_path}'")
        data = torch.load(cache_path, weights_only=True)
        features = data["features"]
        labels = data["labels"]
        relation_indices = data.get("relation_indices", None)
    else:
        all_features = []
        all_labels = []
        for batch in tqdm(spatial_loader, desc=f"Extracting ResNet {args.mode} features", unit="batch"):
            images = batch[0].to(extractor.device)
            batch_labels = batch[4]

            with torch.no_grad():
                feat_map = extractor.feature_extractor(images)
                if use_spatial:
                    b, c, h, w = feat_map.shape
                    flat = feat_map.view(b, c, h * w)
                    f_mean = flat.mean(dim=2)
                    f_max = flat.max(dim=2).values
                    f_std = flat.std(dim=2)
                    feats = torch.cat([f_mean, f_max, f_std], dim=1)
                else:
                    feats = feat_map.squeeze(-1).squeeze(-1)

            all_features.append(feats.cpu())
            all_labels.append(batch_labels)

        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)

        # Extract relation indices from dataset annotations
        relation_indices = torch.tensor(
            [dataset.relation_to_idx[dataset.annotations[i]["relation"]] for i in range(len(dataset))],
            dtype=torch.long,
        )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"features": features, "labels": labels, "relation_indices": relation_indices},
            cache_path,
        )
        print(f"      Cached {len(features)} feature vectors to '{cache_path}'")

    if relation_indices is None:
        relation_indices = torch.tensor(
            [dataset.relation_to_idx[dataset.annotations[i]["relation"]] for i in range(len(dataset))],
            dtype=torch.long,
        )

    labels = labels.long()
    print(f"      Features shape: {features.shape}")
    print(f"      Labels shape: {labels.shape}")
    print(f"      Relation indices shape: {relation_indices.shape}")

    # Deterministic train/test split (80/20)
    n = len(features)
    perm_gen = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=perm_gen)
    split = int(0.8 * n)
    train_idx, test_idx = perm[:split], perm[split:]

    train_feats, train_labels = features[train_idx], labels[train_idx]
    test_feats, test_labels = features[test_idx], labels[test_idx]
    train_rels = relation_indices[train_idx]
    test_rels = relation_indices[test_idx]

    print(f"      Train: {len(train_feats)} | Test: {len(test_feats)}")

    # 4. Train Probe
    use_wandb = not args.no_wandb
    tag = "conditioned" if conditioned else "unconditioned"
    ckpt_dir = Path(args.checkpoint_dir) / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if conditioned:
        print(f"\n[4/4] Training Relation-Conditioned Probe ({args.epochs} epochs, LR={args.lr})...")
        trainer = RelationProbeTrainer(
            visual_dim=extractor.feature_dim,
            num_relations=len(dataset.relations),
            relation_embed_dim=128,
            num_classes=2,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            device=device,
            checkpoint_dir=str(ckpt_dir),
            use_wandb=use_wandb,
            wandb_project=config.get("linear_probe", {}).get("wandb_project", "vlm-linear-probe"),
            wandb_run_name=f"spatial-probe-{tag}",
            wandb_config={
                "task": "spatialsense_binary",
                "paradigm": "relation_conditioned",
                "mode": args.mode,
                "epochs": args.epochs,
                "lr": args.lr,
                "seed": args.seed,
            },
        )
        history = trainer.train(
            train_feats,
            train_rels,
            train_labels,
            batch_size=args.batch_size,
            val_features=test_feats,
            val_rel_indices=test_rels,
            val_labels=test_labels,
        )
        print("\nFinal evaluation on test set...")
        results = trainer.evaluate(
            test_feats,
            test_rels,
            test_labels,
            idx_to_relation=dataset.idx_to_relation,
        )
    else:
        print(f"\n[4/4] Training Unconditioned Linear Probe ({args.epochs} epochs, LR={args.lr})...")
        trainer = LinearProbeTrainer(
            feature_dim=extractor.feature_dim,
            num_classes=2,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            device=device,
            checkpoint_dir=str(ckpt_dir),
            use_wandb=use_wandb,
            wandb_project=config.get("linear_probe", {}).get("wandb_project", "vlm-linear-probe"),
            wandb_run_name=f"spatial-probe-{tag}",
            wandb_config={
                "task": "spatialsense_binary",
                "paradigm": "image_only_unconditioned",
                "mode": args.mode,
                "epochs": args.epochs,
                "lr": args.lr,
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
    print(f"LINEAR PROBE SPATIAL RESULTS [{cond_str}]:")
    print(f"  Overall Accuracy : {results['accuracy'] * 100:.2f}%  (Random chance: 50.00%)")
    print(f"  Total Test       : {results['total_samples']}")
    print("=" * 70)

    if "per_relation" in results:
        print("\nPER-RELATION ACCURACY:")
        for rel_name, stats in results["per_relation"].items():
            print(f"  {rel_name:<18}: {stats['accuracy']*100:6.1f}% ({stats['correct']}/{stats['total']})")

    # Save results
    results_dir = Path(config.get("output", {}).get("results_dir", "results/logs"))
    results_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "task": "spatialsense_binary",
        "model": f"resnet50_linear_probe_{tag}",
        "paradigm": tag,
        "mode": args.mode,
        "feature_dim": extractor.feature_dim,
        "seed": args.seed,
        "lr": args.lr,
        "epochs": args.epochs,
        **results,
        "training_history": history,
    }
    output_path = results_dir / f"linear_probe_spatial_{tag}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()