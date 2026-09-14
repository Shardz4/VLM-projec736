import json
import os
from pathlib import Path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config_loader import load_config
from src.data_loaders.dataloader_factory import build_dataloaders
from src.evaluators.imagenet_evaluator import ImageNetEvalutaor
from src.models.clip_encoder import CLIPEncoder


def main():
    print("="*70)
    print("Imagenet-V2 Zero-Shot baseline sanity check")
    print("="*70)

    config = load_config()
    device = config.get("device", "cuda")

    model_name = config.get("clip_model", "ViT-B/32")
    encoder = CLIPEncoder(model_name=model_name, device=device)

    loaders = build_dataloaders(config)
    if "imagenet_v2" not in loaders:
        print("[Error] ImageNet-V2 DataLoader could not be initialized")
        pritn("Check that imagenet-v2 folders exist")
        sys.exit(1)

    inv2_loader = loaders["imagenet_v2"]
    print(f"ImageNet-V2 Dataloade ready: {len(inv2_loader.dataset)} ") 

    evaluator = ImageNetV2Evaluator(clip_encoder=encoder)
    results = evaluator.evaluate_dataset(inv2_loader)
    

    # Console Summary
    top1 = results["top1_accuracy"]
    top5 = results["top5_accuracy"]
    passed = results["sanity_check_passed"]
    print("\n" + "=" * 70)
    print("IMAGENET-V2 SANITY CHECK RESULTS:")
    print(f"  Top-1 Accuracy : {top1 * 100:.2f}%  (Expected: ~60.5%)")
    print(f"  Top-5 Accuracy : {top5 * 100:.2f}%  (Expected: ~86.0%)")
    print(f"  Total Evaluated: {results['total_samples']}")
    print(f"  Confidence     : Correct={results['confidence_stats']['correct_mean']:.3f} | Incorrect={results['confidence_stats']['incorrect_mean']:.3f}")
    print("=" * 70)
    if passed:
        print("\n✓ SANITY CHECK PASSED — Pipeline is correctly calibrated.")
        print("  You may proceed to compositional analysis steps.")
    else:
        print("\n✗ SANITY CHECK FAILED — Top-1 accuracy is outside [55%, 70%].")
        print("  Investigate: preprocessing transforms, class label ordering,")
        print("  image corruption, or CUDA precision. Do NOT proceed until fixed.")
    # Save results
    results_dir = Path(config.get("output", {}).get("results_dir", "results/logs"))
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "imagenet_v2_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[4/4] Saved results to: {output_path}")
    print("\nStep 9 execution completed.")
if __name__ == "__main__":
    main()
