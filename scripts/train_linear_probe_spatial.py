import json
import os
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.config_loader import load_config
from src.data_loaders.dataloader_factory import build_dataloaders
from src.models.resnet_baseline import ResNetFeatureExtractor, LinearProbeTrainer

def main():
    print("=" * 70)
    print("Linear Probe: SpatialSense Binary Classification")
    print("=" * 70)

    config = load_config()
    device = config.get("device", "cuda")
    lp_cfg = config.get("linear_probe", {})

    print("\n Building SpatialSense DataLoader....")
    loaders = build_dataloaders(config)
    if "spatialsense" not in loaders:
        print("[Error] SpatialSense Dataloader not available.")
        sys.exit(1)
    
    spatial_loader = loaders["spatialsense"]
    total_samples = len(spatial_loader.dataset)
    print(f" SpatialSense samples: {total_samples}")

    print("\n Extracting ResNet-50 features...")
    extractor = ResNetFeatureExtractor(device=device)

    cache_path = Path("results/cache/spatial_resnet_features.pt")
    if cache_path.is_file():
        print(f" Loading cached features from '{cache_path}'")
        data = torch.load(cache_path, weights_only=True)
        features, labels = data["features"], data["labels"]
    else:
        all_features = []
        all_labels = []
        for batch in tqdm(spatial_loader, desc="Extracting ResNet features", unit="batch"):
            images = batch[0].to(extractor.device)
            batch_labels = batch[4]

            with torch.no_grad():
                feats = extractor.feature_extractor(images).squeeze(-1).squeeze(-1)
            all_features.append(feats.cpu())
            all_labels.append(batch_labels)
        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0) 

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": features, "labels": labels}, cache_path)          
        print(f" Cached {len(features)} feature vectors")
    
    print(f" Features shape: {features.shape}")
    print(f" Labels shape: {labels.shape}")

    labels = labels.long()
    num_classes = 2

    n = len(features)
    perm = torch.randperm(n)
    split = int(0.8 * n)
    train_idx, test_idx = perm[:split], perm[split:]
    
    train_feats, train_labels = features[train_idx], labels[train_idx]
    test_feats, test_labels = features[test_idx], labels[test_idx]
    print(f" Train: {len(train_feats)} | Test: {len(test_feats)}")

    print("\n Training linear probe...")
    trainer = LinearProbeTrainer(
        feature_dim = extractor.feature_dim,
        num_classes = num_classes,
        lr = lp_cfg.get("learning_rate", 1e-4),
        weight_decay = lp_cfg.get("weight_decay", 0.01),
        epochs = lp_cfg.get("epochs", 50),
        device = device,
    )

    history = trainer.train(
        train_feats, train_labels,
        batch_size=64,
        val_features=test_feats,
        val_labels=test_labels,
    )
    
    print("\n[4/4] Final evaluation on test set...")
    results = trainer.evaluate(test_feats, test_labels)
    print("\n" + "=" * 70)
    print("LINEAR PROBE SPATIAL RESULTS:")
    print(f"  Overall Accuracy : {results['accuracy'] * 100:.2f}%  (Random baseline: 50%)")
    print(f"  Total Test       : {results['total_samples']}")
    print("=" * 70)
    print("\nPER-CLASS ACCURACY:")
    class_labels = {0: "Incorrect/Swapped", 1: "Correct Relation"}
    for cls_idx, stats in results["per_class"].items():
        label = class_labels.get(cls_idx, str(cls_idx))
        print(f"  {label}: {stats['accuracy']*100:6.1f}% ({stats['correct']}/{stats['total']})")
    # Save results
    results_dir = Path(config.get("output", {}).get("results_dir", "results/logs"))
    results_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "task": "spatialsense_binary",
        "model": "resnet50_linear_probe",
        **results,
        "training_history": history,
        "config": lp_cfg,
    }
    output_path = results_dir / "linear_probe_spatial.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to: {output_path}")
if __name__ == "__main__":
    main()