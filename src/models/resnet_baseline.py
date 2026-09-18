""" Resnet-50 frozen feature extractor and supervised learning probe transfer"""

import json
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torchvision.models as models

# Optional wandb import — gracefully degrade if not installed
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class ResNetFeatureExtractor:

    def __init__(self, device: str = "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        weights = models.ResNet50_Weights.IMAGENET1K_V1
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

        for batch in tqdm(dataloader, desc="Extracting ResNet features", unit="batch"):
            images = batch[0].to(self.device)
            labels = batch[1]
            features = self.feature_extractor(images)
            features = features.squeeze(-1).squeeze(-1)
            all_features.append(features.cpu())
            all_labels.append(labels)

        return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)
    
    def extract_and_cache(self, dataloader: DataLoader, cache_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        cache = Path(cache_path)
        if cache.is_file():
            print(f"[ResNet] Loading cached features from {cache_path}")
            data = torch.load(cache, weights_only=True)
            return data["features"], data["labels"]
        
        print(f"[ResNet] Extracting features (will cache to '{cache_path}')")
        features, labels = self.extract_features(dataloader)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": features, "labels": labels}, cache)
        print(f"[ResNet] Cached features to '{cache_path}'")
        return features, labels


class LinearProbeTrainer:

    def __init__(
        self,
        feature_dim: int = 2048,
        num_classes: int = 10,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        epochs: int = 50,
        device: str = "cuda",
        early_stopping_patience: int = 10,
        checkpoint_dir: Optional[str] = "results/checkpoints",
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
    ):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        self.epochs = epochs
        self.num_classes = num_classes
        self.early_stopping_patience = early_stopping_patience

        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.linear_head = nn.Linear(feature_dim, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.linear_head.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        # W&B setup
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if use_wandb and not WANDB_AVAILABLE:
            print("[Warning] wandb not installed. Install with: pip install wandb")
            print("          Continuing without W&B logging.")
        
        if self.use_wandb:
            wandb.init(
                project=wandb_project or "vlm-linear-probe",
                name=wandb_run_name,
                config={
                    "feature_dim": feature_dim,
                    "num_classes": num_classes,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "epochs": epochs,
                    "early_stopping_patience": early_stopping_patience,
                    **(wandb_config or {}),
                },
            )
            wandb.watch(self.linear_head, log="all", log_freq=10)
            print("[W&B] Logging initialized.")

    def _save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False):
        """Save a training checkpoint."""
        if self.checkpoint_dir is None:
            return

        state = {
            "epoch": epoch,
            "model_state_dict": self.linear_head.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_accuracy": val_acc,
        }

        # Always save latest
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        torch.save(state, latest_path)

        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(state, best_path)
            print(f"  [Checkpoint] New best model saved (val_acc={val_acc*100:.2f}%)")

    def load_checkpoint(self, path: str):
        """Load a saved checkpoint to resume training or for inference."""
        checkpoint = torch.load(path, weights_only=False, map_location=self.device)
        self.linear_head.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"[Checkpoint] Loaded from '{path}' (epoch {checkpoint['epoch']}, val_acc={checkpoint['val_accuracy']*100:.2f}%)")
        return checkpoint["epoch"], checkpoint["val_accuracy"]

    def train(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        batch_size: int = 64,
        val_features: Optional[torch.Tensor] = None,
        val_labels: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, float]]:
        train_dataset = TensorDataset(train_features, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        has_val = val_features is not None and val_labels is not None

        # Early stopping state
        best_val_acc = -1.0
        patience_counter = 0
        stopped_early = False

        history = []
        for epoch in range(self.epochs):
            self.linear_head.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for feats, labels in train_loader:
                feats = feats.to(self.device)
                labels = labels.to(self.device)

                logits = self.linear_head(feats)
                loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item() * len(labels)
                epoch_correct += (logits.argmax(dim=-1) == labels).sum().item()
                epoch_total += len(labels)
            
            self.scheduler.step()
            train_acc = epoch_correct / epoch_total
            avg_loss = epoch_loss / epoch_total
            current_lr = self.scheduler.get_last_lr()[0]

            metrics = {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "train_accuracy": train_acc,
                "learning_rate": current_lr,
            }

            # Validation
            val_acc = 0.0
            if has_val:
                val_acc = self.evaluate(val_features, val_labels)["accuracy"]
                metrics["val_accuracy"] = val_acc

            history.append(metrics)

            # W&B logging
            if self.use_wandb:
                wandb.log(metrics, step=epoch + 1)

            # Checkpointing & early stopping (only if we have validation)
            if has_val:
                is_best = val_acc > best_val_acc
                if is_best:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1

                self._save_checkpoint(epoch + 1, val_acc, is_best=is_best)

                # Early stopping check
                if patience_counter >= self.early_stopping_patience:
                    print(f"\n  [Early Stopping] No improvement for {self.early_stopping_patience} epochs. "
                          f"Best val_acc={best_val_acc*100:.2f}% at epoch {epoch + 1 - patience_counter}.")
                    stopped_early = True

                    # Restore best model weights
                    if self.checkpoint_dir and (self.checkpoint_dir / "best_model.pt").is_file():
                        self.load_checkpoint(str(self.checkpoint_dir / "best_model.pt"))
                    break
            
            # Console logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                log_str = (
                    f"  Epoch {epoch+1:3d}/{self.epochs} | "
                    f"Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.1f}%"
                )
                if has_val:
                    log_str += f" | Val Acc: {val_acc*100:.1f}%"
                log_str += f" | LR: {current_lr:.2e}"
                if has_val:
                    log_str += f" | Patience: {patience_counter}/{self.early_stopping_patience}"
                print(log_str)

        # Training complete summary
        if stopped_early:
            print(f"  Training stopped early at epoch {len(history)}/{self.epochs}")
        else:
            print(f"  Training completed all {self.epochs} epochs.")

        if has_val:
            print(f"  Best validation accuracy: {best_val_acc*100:.2f}%")

        # Finalize W&B
        if self.use_wandb:
            wandb.summary["best_val_accuracy"] = best_val_acc
            wandb.summary["stopped_early"] = stopped_early
            wandb.summary["total_epochs_trained"] = len(history)
            wandb.finish()
            print("[W&B] Run finished and synced.")

        return history

    @torch.no_grad()
    def evaluate(self, features: torch.Tensor, labels: torch.Tensor, batch_size: int = 256) -> Dict[str, Any]:
        self.linear_head.eval()
        dataset = TensorDataset(features, labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_preds = []
        all_labels = []

        for feats, lbls in loader:
            feats = feats.to(self.device)
            logits = self.linear_head(feats)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(lbls.tolist())
        
        preds_arr = np.array(all_preds, dtype=int)
        labels_arr = np.array(all_labels, dtype=int)

        is_correct = (preds_arr == labels_arr)
        overall_acc = float(np.mean(is_correct))
        mae = float(np.mean(np.abs(preds_arr - labels_arr)))

        per_class = {}
        for c in sorted(set(labels_arr)):
            mask = (labels_arr == c)
            cls_total = int(np.sum(mask))
            cls_correct = int(np.sum(is_correct[mask]))
            per_class[int(c)] = {
                "total": cls_total,
                "correct": cls_correct,
                "accuracy": cls_correct / cls_total if cls_total > 0 else 0.0,
            }
        
        classes = sorted(set(labels_arr) | set(preds_arr))
        cls_to_idx = {c: i for i, c in enumerate(classes)}
        K = len(classes)
        conf_matrix = np.zeros((K, K), dtype=int)
        for t, p in zip(labels_arr, preds_arr):
            conf_matrix[cls_to_idx[t], cls_to_idx[p]] += 1
        
        return {
            "accuracy": overall_acc,
            "mae": mae,
            "total_samples": len(labels_arr),
            "per_class": per_class,
            "confusion_matrix": conf_matrix.tolist(),
            "class_order": [int(c) for c in classes],
        }
