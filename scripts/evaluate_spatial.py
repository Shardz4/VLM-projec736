"""Step 8: Zero-Shot Spatial Reasoning Evaluation on SpatialSense.

Evaluates CLIP ViT-B/32 on binary forced-choice spatial relation pairs:
"The {subject} is {relation} the {object}." vs. "The {object} is {relation} the {subject}."

Saves results to:
- results/logs/spatial_results.json
- results/logs/spatial_per_relation.csv
"""
import csv
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config_loader import load_config
from src.data_loaders.dataloader_factory import build_dataloaders
from src.evaluators.spatial_evaluator import SpatialEvaluator
from src.models.clip_encoder import CLIPEncoder


def main():
    print("=" * 70)
    print("Step 8 — Zero-Shot Spatial Reasoning Probe (SpatialSense)")
    print("=" * 70)

    config = load_config()
    device = config.get("device", "cuda")
    model_name = config.get("clip_model", "ViT-B/32")

    print(f"\n[1/4] Initializing CLIP model ({model_name}) on device '{device}'...")
    encoder = CLIPEncoder(model_name=model_name, device=device)

    print("\n[2/4] Initializing SpatialSense DataLoader...")
    loaders = build_dataloaders(config)
    if "spatialsense" not in loaders:
        print("\n[Error] SpatialSense DataLoader could not be initialized.")
        print("Please check that data/spatialsense and data/images exist.")
        sys.exit(1)

    spatial_loader = loaders["spatialsense"]
    print(f"      SpatialSense DataLoader ready: {len(spatial_loader.dataset)} samples.")

    print("\n[3/4] Running Spatial Reasoning Probe...")
    evaluator = SpatialEvaluator(clip_encoder=encoder, device=device)
    results = evaluator.evaluate_dataset(spatial_loader, only_positive_triplets=True)

    # Console Summary
    print("\n" + "=" * 70)
    print("SPATIAL PROBE RESULTS:")
    print(f"  Overall Accuracy : {results['overall_accuracy'] * 100:.2f}%  (Random baseline: 50.00%)")
    print(f"  Total Evaluated  : {results['total_samples']}")
    print(f"  Mean Margin      : {results['margin_stats']['mean']:.4f} (std: {results['margin_stats']['std']:.4f})")
    print("=" * 70)

    print("\nPER-RELATION ACCURACY BREAKDOWN:")
    print(f"  {'Relation':<18} | {'Accuracy':<10} | {'Correct/Total':<14} | {'Mean Margin'}")
    print("  " + "-" * 60)
    for rel, stats in results["per_relation"].items():
        print(
            f"  {rel:<18} | {stats['accuracy'] * 100:>8.1f}% | "
            f"{stats['correct']:>5}/{stats['total']:<8} | "
            f"{stats['mean_margin']:>10.4f}"
        )

    # Output artifact saving
    results_dir = Path(config.get("output", {}).get("results_dir", "results/logs"))
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "spatial_results.json"
    with open(summary_path, "w") as f:
        # Exclude large raw all_samples list if desired, or save full
        json.dump(results, f, indent=2)
    print(f"\n[4/4] Saved results summary to       : {summary_path}")

    csv_path = results_dir / "spatial_per_relation.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["relation", "accuracy", "correct", "total", "mean_margin", "std_margin"])
        for rel, s in results["per_relation"].items():
            writer.writerow([rel, f"{s['accuracy']:.4f}", s["correct"], s["total"], f"{s['mean_margin']:.4f}", f"{s['std_margin']:.4f}"])
    print(f"      Saved per-relation CSV to      : {csv_path}")
    print("\nStep 8 probe execution completed successfully!")


if __name__ == "__main__":
    main()
