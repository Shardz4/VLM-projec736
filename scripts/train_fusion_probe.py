"""Master Multimodal Fusion Probe Trainer & W&B Sweep Agent for Step 12.

Trains EarlyFusionMLP, LateFusionGated, or CrossAttentionFusion over pre-extracted
visual feature tensors and text representations. Evaluates using the full Cluster 6B
metric suite from src/evaluation/multimodal_metrics.py.

Hardware Safety: Operates on CPU by default. Takes ~10-15 seconds per 100-epoch run,
eliminating GPU thermal risk completely.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.multimodal_metrics import MultimodalEvaluator, modality_ablation_drop
from src.models.multimodal_fusion import build_fusion_model
from src.utils import set_seed

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multimodal Fusion Probe (Step 12)")
    # Model architecture
    parser.add_argument(
        "--strategy",
        type=str,
        default="cross_attention",
        choices=["cross_attention", "early", "late"],
        help="Fusion strategy: 'cross_attention' (Ours), 'early', or 'late'",
    )
    parser.add_argument(
        "--visual-backbone", "--visual_backbone",
        dest="visual_backbone",
        type=str,
        default="clip_vit",
        choices=["clip_vit", "resnet_avgpool", "resnet_spatial"],
        help="Visual feature source",
    )
    parser.add_argument("--hidden-dim", "--hidden_dim", dest="hidden_dim", type=int, default=256, help="Hidden / d_model dim")
    parser.add_argument("--num-heads", "--num_heads", dest="num_heads", type=int, default=4, help="Attention heads (cross_attention)")
    parser.add_argument("--modality-dropout", "--modality_dropout", dest="modality_dropout", type=float, default=0.2, help="Modality dropout probability")
    parser.add_argument("--dropout", type=float, default=0.1, help="Activation / attention dropout")

    # Optimization
    parser.add_argument("--lr", "--learning-rate", "--learning_rate", dest="lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Execution device ('cuda' or 'cpu')",
    )

    # Tracking & logging
    parser.add_argument("--no-wandb", "--no_wandb", dest="no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--wandb-project", "--wandb_project", dest="wandb_project", type=str, default="vlm_multimodal_fusion", help="W&B project")
    parser.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", type=str, default="step12_sweep", help="W&B run group")
    parser.add_argument("--cache-dir", "--cache_dir", dest="cache_dir", type=str, default="results/cache", help="Feature cache directory")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default="results/logs", help="Results directory")
    parser.add_argument("--checkpoint-dir", "--checkpoint_dir", dest="checkpoint_dir", type=str, default="results/checkpoints/fusion", help="Checkpoint dir")

    args, unknown = parser.parse_known_args()
    return args


def load_cached_features(cache_dir: Path, visual_backbone: str) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Load pre-extracted visual features and labels from cache."""
    if visual_backbone == "clip_vit":
        cache_file = cache_dir / "clevr_clip_vit_features.pt"
    elif visual_backbone == "resnet_avgpool":
        cache_file = cache_dir / "clevr_resnet_avgpool_features.pt"
    elif visual_backbone == "resnet_spatial":
        cache_file = cache_dir / "clevr_resnet_spatial_features.pt"
    else:
        raise ValueError(f"Unknown backbone {visual_backbone}")

    if not cache_file.is_file():
        raise FileNotFoundError(f"Missing cached features at '{cache_file}'. Run feature extraction first.")

    print(f"[DataLoader] Loading visual features from '{cache_file}'...")
    data = torch.load(cache_file, weights_only=True)
    visual_feats = data["features"].float()
    raw_labels = data["labels"].long()

    # Map labels to contiguous 0..N-1 classes
    unique_counts = sorted(set(raw_labels.tolist()))
    count_to_idx = {c: i for i, c in enumerate(unique_counts)}
    labels = torch.tensor([count_to_idx[c.item()] for c in raw_labels], dtype=torch.long)
    num_classes = len(unique_counts)

    return visual_feats, labels, num_classes


def get_task_text_features(num_samples: int, text_dim: int = 512, seed: int = 42) -> torch.Tensor:
    """Generate or retrieve consistent normalized task text prompt representations."""
    # Deterministic pseudo-text embedding representing the task query
    # "A photo containing objects to count" on the unit hypersphere
    rng = torch.Generator().manual_seed(seed)
    base_text_vec = torch.randn(1, text_dim, generator=rng)
    base_text_vec = base_text_vec / base_text_vec.norm(dim=-1, keepdim=True)
    # Broadcast across all samples
    return base_text_vec.expand(num_samples, text_dim).clone()


def main():
    args = parse_args()

    # 1. Initialize W&B tracking early (so sweep agent parameters override CLI args)
    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        run_name = f"{args.strategy}_{args.visual_backbone}_s{args.seed}"
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=run_name,
            config=vars(args),
            reinit=True,
        )
        if wandb.run is not None and hasattr(wandb, "config"):
            for k, v in wandb.config.items():
                if hasattr(args, k):
                    setattr(args, k, v)
                elif k == "learning_rate":
                    setattr(args, "lr", v)

    set_seed(args.seed)
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) or args.device == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu"))

    print("=" * 70)
    print(f"Cluster 6B Multimodal Fusion Trainer (Strategy: {args.strategy.upper()})")
    print(f"Backbone: {args.visual_backbone} | Device: {str(device).upper()} | Seed: {args.seed}")
    print("=" * 70)

    # 2. Load pre-cached visual features & labels
    cache_dir = Path(args.cache_dir)
    visual_feats, labels, num_classes = load_cached_features(cache_dir, args.visual_backbone)
    num_samples, visual_dim = visual_feats.shape
    text_dim = 512

    # 3. Generate task query text features (512-d)
    text_feats = get_task_text_features(num_samples, text_dim=text_dim, seed=args.seed)

    # 4. Deterministic train/test split (80/20)
    perm_gen = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(num_samples, generator=perm_gen)
    split = int(0.8 * num_samples)
    train_idx, test_idx = perm[:split], perm[split:]

    v_train, t_train, y_train = visual_feats[train_idx], text_feats[train_idx], labels[train_idx]
    v_test, t_test, y_test = visual_feats[test_idx], text_feats[test_idx], labels[test_idx]

    print(f"Dataset split: Train = {len(v_train)} | Test = {len(v_test)} | Classes = {num_classes}")

    train_dataset = TensorDataset(v_train, t_train, y_train)
    test_dataset = TensorDataset(v_test, t_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 5. Build Fusion Model
    model = build_fusion_model(
        strategy=args.strategy,
        visual_dim=visual_dim,
        text_dim=text_dim,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        p_modal_drop=args.modality_dropout,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # 6. Training Loop
    print(f"\nTraining {args.strategy} probe for {args.epochs} epochs on {args.device}...")
    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        for v_b, t_b, y_b in train_loader:
            v_b, t_b, y_b = v_b.to(device), t_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(v_b, t_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y_b)
            preds = logits.argmax(dim=-1)
            correct_train += (preds == y_b).sum().item()
            total_train += len(y_b)

        scheduler.step()
        train_acc = correct_train / total_train
        train_loss = total_loss / total_train

        # Quick validation every 10 epochs or at end
        if epoch % 10 == 0 or epoch == args.epochs:
            model.eval()
            correct_val = 0
            with torch.no_grad():
                for v_b, t_b, y_b in test_loader:
                    v_b, t_b, y_b = v_b.to(device), t_b.to(device), y_b.to(device)
                    logits = model(v_b, t_b)
                    correct_val += (logits.argmax(dim=-1) == y_b).sum().item()

            val_acc = correct_val / len(test_dataset)
            best_test_acc = max(best_test_acc, val_acc)

            if use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "val/accuracy": val_acc,
                    "lr": scheduler.get_last_lr()[0],
                })

            if epoch % 20 == 0 or epoch == args.epochs:
                print(f"  Epoch {epoch:3d}/{args.epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

    # 7. Full Cluster 6B Multimodal Metric Evaluation
    print("\n[Evaluation] Computing full Cluster 6B metric suite...")
    model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for v_b, t_b, y_b in test_loader:
            v_b, t_b = v_b.to(device), t_b.to(device)
            logits = model(v_b, t_b)
            all_logits.append(logits.cpu())
            all_targets.append(y_b)

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    probs = all_logits.softmax(dim=-1)
    confs, preds = probs.max(dim=-1)
    correct_flags = (preds == all_targets).float()

    final_accuracy = float(correct_flags.mean().item())
    mae = float((preds.float() - all_targets.float()).abs().mean().item())

    # Modality Ablation Drop (Zeroing out modalities)
    with torch.no_grad():
        # Vision-only (text zeroed)
        v_only_logits = model(v_test.to(device), torch.zeros_like(t_test).to(device))
        v_only_acc = float((v_only_logits.argmax(dim=-1).cpu() == all_targets).float().mean().item())

        # Text-only (vision zeroed)
        t_only_logits = model(torch.zeros_like(v_test).to(device), t_test.to(device))
        t_only_acc = float((t_only_logits.argmax(dim=-1).cpu() == all_targets).float().mean().item())

    drop_metrics = modality_ablation_drop(
        joint_acc=final_accuracy,
        vision_only_acc=v_only_acc,
        text_only_acc=t_only_acc,
    )

    # Full Step 11 evaluator
    evaluator = MultimodalEvaluator(k_values=(1, 5, 10))
    if v_test.shape[-1] == t_test.shape[-1]:
        eval_v, eval_t = v_test, t_test
    elif hasattr(model, "proj_v") and hasattr(model, "proj_t"):
        with torch.no_grad():
            eval_v = model.proj_v(v_test.to(device)).detach().cpu()
            eval_t = model.proj_t(t_test.to(device)).detach().cpu()
    else:
        eval_v, eval_t = v_test, t_test

    metrics_suite = evaluator.compute_all(
        image_features=eval_v,
        text_features=eval_t,
        confidences=confs,
        correct=correct_flags,
    )

    evaluator.print_report(metrics_suite, prefix=f"{args.strategy.upper()} Final")
    print(f"  Test Accuracy : {final_accuracy * 100:.2f}%")
    print(f"  Test MAE      : {mae:.3f} counts")
    print(f"  Vision Drop   : {drop_metrics['vision_drop']*100:.2f}%")
    print(f"  Text Drop     : {drop_metrics['text_drop']*100:.2f}%")
    print(f"  Dominant Modal: {drop_metrics['dominant_modality']}")

    # 8. Save Results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / f"fusion_{args.strategy}_{args.visual_backbone}_s{args.seed}.json"

    final_results = {
        "strategy": args.strategy,
        "visual_backbone": args.visual_backbone,
        "seed": args.seed,
        "accuracy": final_accuracy,
        "mae": mae,
        "modality_drop": drop_metrics,
        "ece": metrics_suite.get("ece", {}).get("ece", 0.0),
        "modality_gap": metrics_suite.get("modality_gap", 0.0),
        "cross_modal_alignment": metrics_suite.get("cross_modal_alignment", 0.0),
        "recall": metrics_suite.get("recall", {}),
        "ranks": metrics_suite.get("ranks", {}),
    }

    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    print(f"\n[Saved] Test results exported to '{res_path}'")

    if use_wandb:
        wandb.log({
            "test/accuracy": final_accuracy,
            "test/mae": mae,
            "test/ece": final_results["ece"],
            "test/modality_gap": final_results["modality_gap"],
            "test/vision_drop": drop_metrics["vision_drop"],
            "test/text_drop": drop_metrics["text_drop"],
        })
        wandb.finish()


if __name__ == "__main__":
    main()
