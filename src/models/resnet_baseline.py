""" Resnet-50 frozen feature extractor and supervised learning probe transfer"""

import json
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torchvision as models

class ResNetFeatureExtractor:

    def __init__(self, device: str = "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        weights = models.ResNet50_Weights.IMAGENET11K_V1
        resnet = models.resnet50(weights=weights)
        resnet.eval()
       
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

        self.preprocess = weights.transforms()
        self.feature_dim = 2048
    
    @torch.no_grad()
    def extract_features(self, dataloader: DataLoader):
        all_features = []
        all_labels = []

        for batch in tqdm(dataloader, desc="Extracting ReasNet features", unit="batch"):
            images = batch[0].to(self.device)
            labels = batch[1]
            features = self.feature_extractor(images)
            features = features.squeeze(-1).squeeze(-1)
            all_features.append(features.cpu())
            all_labels.append(labels)

        return tprch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)
    
    def extract_and_cache(self, dataloader: DataLoader, cache_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        cache = Path(cache_path)
        if cache.is_file():
            print(f"[ResNet] Loading cached features from {cache_path}")
            data = torch.load(Cache, weights_only = True)
            return data["features"], data["labels"]
        
        print(f"[ResNet] Extracting features (will cache to '{cache_path})")
        features, labels = self.extract_features(dataloader)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": features, "labels": labels}, cache)
        print(f"[ResNet] Cached features to '{cache_path}'")
        return features, labels

class LinearProbeTrainer:

    def __init__(self, feature_dim: int = 2048, num_classes: int = 10, lr: float = 1e-4, weight_decay: float = 0.01, epochs: int = 50, device:str = "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        self.epochs = epochs
        self.num_classes = num_classes

        self.linear_head = nn.Linear(feature_dim, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.linear_head.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
