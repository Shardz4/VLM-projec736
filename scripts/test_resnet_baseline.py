import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, TensorDataset
from src.config_loader import load_config
from src.models.resnet_baseline import ResNetFeatureExtractor, LinearProbeTrainer

def test_feature_extraction():
    print("\n Resnet-50 feature extraction shape...")
    config = load_config()
    device = config.get("device", "cuda")
    extractor = ResNetFeatureExtractor(device=device)

    dummy_images = torch.randn(4, 3, 224, 224)
    dummy_labels = torch.tensor([0, 1, 2, 3])
    dataset = TensorDataset(dummy_images, dummy_labels) 
    loader = DataLoader(dataset, batch_size=2)

    features, labels = extractor.extract_features(loader)
    print(f" Features shape: {features.shape}")
    print(f" Labels shape : {labels.shape}")
    assert features.shape == (4, 2048), f"Expected (4, 2048), got {features.shape}"
    assert labels.shape == (4,), f"Expected (4,), got {labels.shape}"
    print("Feature extraction shape test PASSED")
    return features, labels

def test_linear_probe_training(features, labels):
    print("\nLinear probe training test...")
    config = load_config()
    device = config.get("device", "cuda")
        
    trainer = LinearProbeTrainer(
        feature_dim=2048, num_classes=4,
        lr=1e-3, epochs=5, device=device,
    )
    history = trainer.train(features, labels, batch_size=2)
    assert len(history) == 5
    assert all("train_loss" in h and "train_accuracy" in h for h in history)
    print(f" Final Train loss: {history[-1]['train_loss']:.4f}")
    print(f" Final train acc: {history[-1]['train_accuracy']*100:.1f}%")
    print("Training Loop Passed")
    return trainer

def test_linear_probe_evaluation(trainer, features, labels):
    print("\n Linear Probe Evaluation...")
    results = trainer.evaluate(features, labels)
    print(f" Accuracy: {results['accuracy']*100:.1f}%")
    print(f" MAE: {results['mae']:.3f}")
    assert 0.0 <= results["accuracy"] <= 1.0
    assert "per_class" in results
    assert "confusion_matrix" in results
    print("Evaluation test PASSED")

def main():
    print("Running ResNet Baseline Verification Suite")

    features, labels = test_feature_extraction()
    trainer = test_linear_probe_training(features, labels)
    test_linear_probe_evaluation(trainer, features, labels)
    print("All Checks Passed")
if __name__ == "__main__":
    main()