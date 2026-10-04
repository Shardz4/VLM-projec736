"""Verification script for ResNet-50 feature extractors and linear probes."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, TensorDataset
from src.config_loader import load_config
from src.models.resnet_baseline import (
    ResNetFeatureExtractor,
    LinearProbeTrainer,
    RelationConditionedLinearProbe,
    RelationProbeTrainer,
)
from src.utils import set_seed


def test_feature_extraction():
    print("\n[Test 1] ResNet-50 feature extraction shape (avgpool vs spatial)...")
    config = load_config()
    device = config.get("device", "cuda")

    # Avgpool
    extractor_avg = ResNetFeatureExtractor(device=device, spatial=False)
    dummy_images = torch.randn(4, 3, 224, 224)
    dummy_labels = torch.tensor([0, 1, 2, 3])
    dataset = TensorDataset(dummy_images, dummy_labels)
    loader = DataLoader(dataset, batch_size=2)

    features_avg, labels = extractor_avg.extract_features(loader)
    print(f"  Avgpool features shape: {features_avg.shape}")
    assert features_avg.shape == (4, 2048), f"Expected (4, 2048), got {features_avg.shape}"
    assert labels.shape == (4,), f"Expected (4,), got {labels.shape}"

    # Spatial
    extractor_sp = ResNetFeatureExtractor(device=device, spatial=True)
    features_sp, _ = extractor_sp.extract_features(loader)
    print(f"  Spatial features shape: {features_sp.shape}")
    assert features_sp.shape == (4, 6144), f"Expected (4, 6144), got {features_sp.shape}"
    print("  Feature extraction shape test PASSED")
    return features_avg, labels


def test_linear_probe_training(features, labels):
    print("\n[Test 2] Standard Linear probe training test...")
    config = load_config()
    device = config.get("device", "cuda")

    trainer = LinearProbeTrainer(
        feature_dim=2048,
        num_classes=4,
        lr=1e-3,
        epochs=5,
        device=device,
        checkpoint_dir=None,
    )
    history = trainer.train(features, labels, batch_size=2)
    assert len(history) == 5
    assert all("train_loss" in h and "train_accuracy" in h for h in history)
    print(f"  Final Train loss: {history[-1]['train_loss']:.4f}")
    print(f"  Final Train acc : {history[-1]['train_accuracy']*100:.1f}%")
    print("  Standard Linear Probe Training test PASSED")
    return trainer


def test_linear_probe_evaluation(trainer, features, labels):
    print("\n[Test 3] Linear Probe Evaluation...")
    results = trainer.evaluate(features, labels)
    print(f"  Accuracy: {results['accuracy']*100:.1f}%")
    print(f"  MAE     : {results['mae']:.3f}")
    assert 0.0 <= results["accuracy"] <= 1.0
    assert "per_class" in results
    assert "confusion_matrix" in results
    print("  Standard Linear Probe Evaluation test PASSED")


def test_relation_conditioned_probe():
    print("\n[Test 4] Relation-Conditioned Linear Probe test...")
    config = load_config()
    device = config.get("device", "cuda")

    dummy_visual = torch.randn(8, 2048)
    dummy_rels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    dummy_labels = torch.tensor([0, 1, 1, 0, 0, 1, 0, 1])

    trainer = RelationProbeTrainer(
        visual_dim=2048,
        num_relations=4,
        relation_embed_dim=64,
        num_classes=2,
        lr=1e-3,
        epochs=5,
        device=device,
        checkpoint_dir=None,
    )
    history = trainer.train(dummy_visual, dummy_rels, dummy_labels, batch_size=4)
    assert len(history) == 5
    print(f"  Final Conditioned Loss: {history[-1]['train_loss']:.4f}")

    results = trainer.evaluate(dummy_visual, dummy_rels, dummy_labels)
    assert 0.0 <= results["accuracy"] <= 1.0
    assert "per_relation" in results
    print(f"  Conditioned Accuracy: {results['accuracy']*100:.1f}%")
    print("  Relation-Conditioned Probe test PASSED")


def main():
    set_seed(42)
    print("=" * 60)
    print("Running ResNet Baseline Verification Suite")
    print("=" * 60)

    features, labels = test_feature_extraction()
    trainer = test_linear_probe_training(features, labels)
    test_linear_probe_evaluation(trainer, features, labels)
    test_relation_conditioned_probe()

    print("\n" + "=" * 60)
    print("All ResNet Baseline & Linear Probe Checks Passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()