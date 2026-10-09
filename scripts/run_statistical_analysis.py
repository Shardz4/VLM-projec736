"""Step 12 Multi-Seed Protocol Runner & Statistical Analysis Exporter.

Executes training across at least 3 random seeds (seeds = [42, 123, 999]) for:
  - Baseline (EarlyFusionMLP or Linear Probe)
  - Extension (CrossAttentionFusion with Modality Dropout)

Computes mean ± std, paired t-tests, Wilcoxon signed-rank tests, Cohen's d effect sizes,
and 10,000 bootstrap 95% Confidence Intervals. Generates LaTeX table for the IEEE paper.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from src.evaluation.statistical_tests import compute_statistical_report


SEEDS = [42, 123, 999]


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Seed Statistical Protocol (Step 12)")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs per run")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cpu/cuda)",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B tracking")
    parser.add_argument("--output-dir", type=str, default="results/tables", help="Output directory")
    return parser.parse_args()


def run_experiment(strategy: str, seed: int, epochs: int, device: str, no_wandb: bool) -> Dict:
    """Run train_fusion_probe.py and parse output JSON."""
    res_path = Path("results/logs") / f"fusion_{strategy}_clip_vit_s{seed}.json"

    cmd = [
        sys.executable,
        "scripts/train_fusion_probe.py",
        "--strategy", strategy,
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--device", device,
    ]
    if no_wandb:
        cmd.append("--no-wandb")

    print(f"\n[Multi-Seed Runner] Launching {strategy.upper()} on {device.upper()} (Seed: {seed})...")
    subprocess.run(cmd, check=True)

    with open(res_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_accs = []
    extension_accs = []
    baseline_maes = []
    extension_maes = []
    baseline_eces = []
    extension_eces = []

    print("=" * 75)
    print(f"STEP 12: MULTI-SEED STATISTICAL PROTOCOL (Seeds: 42, 123, 999 | Device: {args.device.upper()})")
    print("=" * 75)

    for seed in SEEDS:
        # Run Baseline (Early Fusion MLP)
        res_base = run_experiment("early", seed, args.epochs, args.device, args.no_wandb)
        baseline_accs.append(res_base["accuracy"])
        baseline_maes.append(res_base["mae"])
        baseline_eces.append(res_base["ece"])

        # Run Extension (Cross Attention Fusion)
        res_ext = run_experiment("cross_attention", seed, args.epochs, args.device, args.no_wandb)
        extension_accs.append(res_ext["accuracy"])
        extension_maes.append(res_ext["mae"])
        extension_eces.append(res_ext["ece"])

    # Compute Statistical Reports
    acc_report = compute_statistical_report(baseline_accs, extension_accs, metric_name="Accuracy")
    mae_report = compute_statistical_report(baseline_maes, extension_maes, metric_name="MAE")
    ece_report = compute_statistical_report(baseline_eces, extension_eces, metric_name="ECE")

    summary_results = {
        "seeds": SEEDS,
        "accuracy_report": acc_report,
        "mae_report": mae_report,
        "ece_report": ece_report,
    }

    # Save JSON report
    json_path = out_dir / "step12_statistical_significance.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    # Console Summary Table
    print("\n" + "=" * 75)
    print("MULTI-SEED STATISTICAL SIGNIFICANCE SUMMARY")
    print("=" * 75)
    print(f"{'Metric':<12} | {'Baseline (Early)':<18} | {'Extension (Cross-Attn)':<22} | {'p-value':<10} | {'95% CI excl 0'}")
    print("-" * 75)
    for rep in (acc_report, mae_report, ece_report):
        p_val_str = f"{rep['t_test'].get('p_value', 1.0):.4f}" if rep['t_test'] else "N/A"
        excl_zero = "YES" if rep["bootstrap_diff_ci"]["excludes_zero"] else "NO"
        print(f"{rep['metric']:<12} | {rep['baseline_summary']:<18} | {rep['extension_summary']:<22} | {p_val_str:<10} | {excl_zero}")
    print("=" * 75)

    # LaTeX Table Generation
    latex_path = out_dir / "step12_table_significance.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated Step 12 Multi-Seed Statistical Significance Table\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Headline Results (mean $\\pm$ std across 3 seeds) and Statistical Significance Tests}\n")
        f.write("\\begin{tabular}{lcccc}\n\\hline\n")
        f.write("Metric & Baseline (Early Fusion) & Extension (Cross-Attn) & $t$-test $p$ & 95\\% Bootstrap CI \\\\ \\hline\n")
        for rep in (acc_report, mae_report, ece_report):
            ci = rep["bootstrap_diff_ci"]
            ci_str = f"[{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]"
            p_val = rep["t_test"].get("p_value", 1.0)
            p_str = f"{p_val:.4f}" if p_val >= 0.001 else "$< 0.001$"
            f.write(f"{rep['metric']} & {rep['baseline_summary']} & \\textbf{{{rep['extension_summary']}}} & {p_str} & {ci_str} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\label{tab:step12_significance}\n\\end{table}\n")

    print(f"\n[Saved] Statistical analysis saved to '{json_path}'")
    print(f"[Saved] LaTeX paper table exported to  '{latex_path}'")


if __name__ == "__main__":
    main()
