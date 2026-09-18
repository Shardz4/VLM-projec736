import os
import sys

sys.path.insert(0, os.path.diname(os.path.dirname(os.path.abspath(__file__))))

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
    