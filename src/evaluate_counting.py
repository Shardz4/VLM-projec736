
import csv
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config_loader import load_config
from src.data_loaders.dataloader_factory import build_dataloaders
from src.evaluators.counting_evaluator import CountingEvaluator
from src.models.clip_encoder import CLIPEncoder


def main():
    print("=" * 70)
    print("Zero-Shot Object Counting Probe (CLEVR)")
    print("=" * 70)

    config = load_config()
    device = config.get("device", "cuda")
    model_name = config.get("clip_model", "ViT-B/32")

    print(f"\n[1/4] Initializing CLIP model ({model_name}) on device '{device}'...")
    encoder = CLIPEncoder(model_name=model_name, device=device)

    print("\n[2/4] Initializing DataLoaders...")
    loaders = build_dataloaders(config)
    if "clevr" not in loaders:
        print("\n[Error] CLEVR DataLoader could not be initialized.")
        print("Please check that data/CLEVR_v1.0 or data/clevr exists.")
        sys.exit(1)

    clevr_loader = loaders["clevr"]
    print(f"      CLEVR DataLoader ready: {len(clevr_loader.dataset)} samples.")

    print("\n[3/4] Running Counting Evaluation Probe...")
    prompt_cfg = config.get("prompts", {}).get("counting", {})
    template = prompt_cfg.get("template", "A photo of {N} objects.")
    count_range = prompt_cfg.get("range", list(range(1, 11)))

    evaluator = CountingEvaluator(
        clip_encoder=encoder,
        template=template,
        count_range=count_range,
    )
    results = evaluator.evaluate_dataset(clevr_loader)

    # Console Summary
    print("\n" + "=" * 70)
    print("COUNTING PROBE RESULTS:")
    print(f"  Overall Accuracy : {results['overall_accuracy'] * 100:.2f}% (Random chance: 10.00%)")
    print(f"  MAE              : {results['mean_absolute_error']:.3f} objects")
    print(f"  Total Evaluated  : {results['total_samples']}")
    print(f"  Mean Confidence  : Correct={results['confidence_stats']['correct_mean']:.3f} | Incorrect={results['confidence_stats']['incorrect_mean']:.3f}")
    print("=" * 70)

    print("\nPER-COUNT ACCURACY BREAKDOWN:")
    print(f"  {'Count':<8} | {'Accuracy':<10} | {'Correct/Total':<14} | {'MAE':<8} | {'Avg Conf'}")
    print("  " + "-" * 55)
    for c, stats in results["per_count"].items():
        print(
            f"  {c:<8} | {stats['accuracy'] * 100:>8.1f}% | "
            f"{stats['correct']:>5}/{stats['total']:<8} | "
            f"{stats['mae']:>6.2f} | {stats['mean_confidence']:.3f}"
        )

    # Output artifact saving
    results_dir = Path(config.get("output", {}).get("results_dir", "results/logs"))
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "counting_results.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[4/4] Saved results summary to       : {summary_path}")

    csv_path = results_dir / "counting_per_count.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["count", "accuracy", "correct", "total", "mae", "mean_confidence"])
        for c, s in results["per_count"].items():
            writer.writerow([c, f"{s['accuracy']:.4f}", s["correct"], s["total"], f"{s['mae']:.4f}", f"{s['mean_confidence']:.4f}"])
    print(f"      Saved per-count CSV to         : {csv_path}")

    matrix_path = results_dir / "counting_confusion_matrix.csv"
    with open(matrix_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["true_count"] + [f"pred_{c}" for c in count_range]
        writer.writerow(header)
        for c, row in zip(count_range, results["confusion_matrix"]):
            writer.writerow([c] + row)
    print(f"      Saved confusion matrix CSV to  : {matrix_path}")
    print("\n probe execution completed successfully!")


if __name__ == "__main__":
    main()
