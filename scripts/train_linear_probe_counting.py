import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.config_loader import load_config
from src.data_loader.dataloader_factory import build_dataloaders
from src.models.resnet_baseline import ResNetFeatureExtractor, LinearProbeTrainer

def main():
    print("="*70)
    print("Linear Probe: CLEVR Object Counting")
    print("="*70)

    config = load_config()
    device = config.get("device", "cuda")
    lp_cfg = config.get("linear_probe", {})

    print("\n Building CLEVR Dataloader")
    loader = build_dataloaders(config)
    if "clevr" not in loaders:
        print("[Error] CLEVR Dataloader not available.")
        sys.exit(1)
    
    clevr_loader = loaders["clevr"]
    total_samples = len(clevr_loader.dataset)
    print(f" CLEVR samples: {total_samples}")

    print("\n Extracting ResNet-50 features...")
    extractor = ResNetFeatureExtractor(device=device)
    features, labels = extractor.extract_and_cache(
        clevr_loader, cache_path="results/cache/clevr_resnet_features.pt"
    )
    print(f" Features shape: {features.shape}")
    print(f" Labels shape: {labels.shape}")

    labels = labels.long() - 1
    num_classes = 10

    n = len(features)
    perm = torch.randperm(n)
    split = int(0.8*n)
    train_idx, test_idx = perm[:split], perm[split:]

    train_feats, train_labels = features[train_idx], labels[train_idx]
    test_feats, test_labels = features[test_idx], labels[test_idx]
    print(f" Train: {len(train_feats)} | Test: {len(test_feats)}")

    print("\n Training Linear probe...")
    trainer = LinearProbeTrainer(
        feature_dim=extractor.feature_dim,
        num_classes=num_classes,
        lr=lp_cfg.get("learning_rate", 1e-4),
        weight_decay = lp_cfg.get("weight_decay", 0.01),
        epochs=lp_cfg.get("epochs", 50),
        device=device,
    )
    history = trainer.train(
        train_feats, train_labels,
        batch_size=64,
        val_features=test_features,
        val_labels=test_labels,
    )

    print("\n[4/4] Final evaluation on test set...")
    results = trainer.evaluate(test_feats, test_labels)
    # Remap class indices back to counts for display
    print("\n" + "=" * 70)
    print("LINEAR PROBE COUNTING RESULTS:")
    print(f"  Overall Accuracy : {results['accuracy'] * 100:.2f}%")
    print(f"  MAE              : {results['mae']:.3f}")
    print(f"  Total Test       : {results['total_samples']}")
    print("=" * 70)
    print("\nPER-COUNT ACCURACY:")
    for cls_idx, stats in results["per_class"].items():
        count = cls_idx + 1
        print(f"  Count {count:2d}: {stats['accuracy']*100:6.1f}% ({stats['correct']}/{stats['total']})")
    # Save results
    results_dir = Path(config.get("output", {}).get("results_dir", "results/logs"))
    results_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "task": "clevr_counting",
        "model": "resnet50_linear_probe",
        **results,
        "training_history": history,
        "config": lp_cfg,
    }
    output_path = results_dir / "linear_probe_counting.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to: {output_path}")
    
if __name__ == "__main__":
    main()
   