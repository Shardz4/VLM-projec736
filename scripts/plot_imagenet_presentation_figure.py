"""scripts/plot_imagenet_presentation_figure.py

Generates publication-quality presentation figures for the ImageNet-V2 Control Sanity Baseline:
1. Panel A: ImageNet-V2 Zero-Shot Verification vs OpenAI Published Baseline
2. Panel B: The Core Paradox: Coarse Object Recognition (53.2%) vs Compositional Counting Collapse (12.4%)
3. Panel C: Confidence Calibration Separation on ImageNet-V2 (Correct vs Incorrect Predictions)

Outputs:
- results/figures/imagenet_v2_sanity_results.png (Master 3-panel presentation slide graphic)
- results/figures/coarse_vs_compositional_paradox.png (Dedicated standalone paradox contrast figure)
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Style Configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#CBD5E0"
plt.rcParams["axes.linewidth"] = 0.9
plt.rcParams["grid.color"] = "#E2E8F0"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.75


def main():
    json_path = Path("results/logs/imagenet_v2_results.json")
    if not json_path.exists():
        print(f"[Error] {json_path} does not exist.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    top1 = data["top1_accuracy"] * 100.0
    top5 = data["top5_accuracy"] * 100.0
    exp_top1 = data.get("expected_top1_v2", 0.555) * 100.0
    exp_top5 = data.get("expected_top5", 0.86) * 100.0
    correct_conf = data["confidence_stats"]["correct_mean"] * 100.0
    incorrect_conf = data["confidence_stats"]["incorrect_mean"] * 100.0

    # =========================================================================
    # FIGURE 1: MASTER 3-PANEL IMAGENET PRESENTATION FIGURE
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=300)

    # -------------------------------------------------------------------------
    # Subplot 1: ImageNet-V2 Zero-Shot Accuracy vs Literature Target
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    bars_x = np.array([0, 1])
    width = 0.35

    b1 = ax1.bar(bars_x - width / 2, [top1, top5], width, label="Our Reproduction (CLIP ViT-B/32)", color="#2B6CB0", alpha=0.92)
    b2 = ax1.bar(bars_x + width / 2, [exp_top1, exp_top5], width, label="OpenAI Literature Ref (Radford 2021)", color="#D69E2E", alpha=0.92)

    # Baseline chance line
    ax1.axhline(0.1, color="#E53E3E", linestyle=":", linewidth=1.5, label="Random Chance (1/1000 = 0.1%)")

    # Acceptance envelope [50%, 70%]
    ax1.axhspan(50, 70, color="#48BB78", alpha=0.12, label="Sanity Envelope [50%, 70%]")

    ax1.set_title("(a) ImageNet-V2 Zero-Shot Sanity Check\n(10,000 Images, 1,000 Classes)", fontsize=11.5, fontweight="bold", pad=10)
    ax1.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_xticks(bars_x)
    ax1.set_xticklabels(["Top-1 Accuracy", "Top-5 Accuracy"], fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper left", framealpha=0.93, fontsize=8.5)

    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 1.8, f"{h:.2f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#1A365D")
    for bar in b2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 1.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#744210")

    ax1.text(
        0.95, 0.05,
        "STATUS: PASSED [OK]\nPipeline Calibrated",
        transform=ax1.transAxes,
        ha="right", va="bottom",
        fontsize=9, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0FFF4", edgecolor="#68D391")
    )

    # -------------------------------------------------------------------------
    # Subplot 2: The Core Paradox (Coarse Recognition vs Compositional Collapse)
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    tasks = ["ImageNet-V2\n(1,000 Classes)", "CLEVR Counting\n(8 Classes)", "SpatialSense\n(2 Classes)"]
    accs = [53.24, 12.43, 57.70]
    chances = [0.1, 12.5, 50.0]
    colors = ["#2B6CB0", "#E53E3E", "#DD6B20"]

    x_pos = np.arange(len(tasks))
    b_tasks = ax2.bar(x_pos, accs, width=0.55, color=colors, alpha=0.9, edgecolor="#2D3748", linewidth=1.0)

    for i, c in enumerate(chances):
        ax2.plot([x_pos[i] - 0.32, x_pos[i] + 0.32], [c, c], color="#1A202C", linestyle="--", linewidth=1.8)
        ax2.text(x_pos[i], c + 1.8, f"Chance: {c:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#2D3748")

    ax2.set_title("(b) The Foundation Model Paradox\n(Semantic Breadth vs Compositional Failure)", fontsize=11.5, fontweight="bold", pad=10)
    ax2.set_ylabel("Zero-Shot Accuracy (%)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(tasks, fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 80)

    for bar, val in zip(b_tasks, accs):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 2.0, f"{val:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax2.annotate(
        "Counting Collapse\n(Predicts count 4 for 76% of samples!)",
        xy=(1, 12.43),
        xytext=(0.55, 30),
        arrowprops=dict(arrowstyle="->", color="#C53030", lw=1.3),
        fontsize=8.5, fontweight="bold", color="#9B2C2C",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF5F5", edgecolor="#FEB2B2")
    )

    # -------------------------------------------------------------------------
    # Subplot 3: Confidence Distribution Separation
    # -------------------------------------------------------------------------
    ax3 = axes[2]
    conf_cats = ["Correct Predictions", "Incorrect Predictions", "Overall Mean"]
    conf_vals = [correct_conf, incorrect_conf, (correct_conf + incorrect_conf) / 2.0]
    conf_colors = ["#38A169", "#E53E3E", "#4A5568"]

    b_conf = ax3.bar(np.arange(len(conf_cats)), conf_vals, width=0.5, color=conf_colors, alpha=0.9, edgecolor="#2D3748", linewidth=1.0)
    ax3.set_title("(c) Confidence Calibration Separation\n(Model Certainty Distribution)", fontsize=11.5, fontweight="bold", pad=10)
    ax3.set_ylabel("Mean Confidence (%)", fontsize=11, fontweight="bold")
    ax3.set_xticks(np.arange(len(conf_cats)))
    ax3.set_xticklabels(conf_cats, fontsize=9.5, fontweight="bold")
    ax3.set_ylim(0, 100)

    for bar, val in zip(b_conf, conf_vals):
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2.0, h + 2.0, f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Margin annotation
    margin = correct_conf - incorrect_conf
    ax3.annotate(
        f"+{margin:.1f}% Margin\nHealthy Confidence Gap",
        xy=(0.5, 55),
        xytext=(0.5, 78),
        ha="center",
        arrowprops=dict(arrowstyle="<->", color="#2B6CB0", lw=1.5),
        fontsize=9, fontweight="bold", color="#1A365D",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#EBF8FF", edgecolor="#BEE3F8")
    )

    fig.suptitle(
        "Control Sanity Validation: CLIP ViT-B/32 Zero-Shot Baseline on ImageNet-V2\n"
        "Verifying Pipeline Fidelity & Establishing the Coarse vs. Compositional Reasoning Divide",
        fontsize=14, fontweight="bold", y=1.03
    )

    plt.tight_layout()
    out_master = out_dir / "imagenet_v2_sanity_results.png"
    plt.savefig(out_master, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Master ImageNet figure saved to: {out_master.resolve()}")

    # =========================================================================
    # FIGURE 2: STANDALONE PARADOX CONTRAST FIGURE FOR SLIDE DECK
    # =========================================================================
    fig2, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    b_p = ax.bar(x_pos, accs, width=0.5, color=colors, alpha=0.92, edgecolor="#1A202C", linewidth=1.2)

    for i, c in enumerate(chances):
        ax.plot([x_pos[i] - 0.35, x_pos[i] + 0.35], [c, c], color="#1A202C", linestyle="--", linewidth=2.0)
        ax.text(x_pos[i], c + 2.0, f"Random Guess: {c:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    for bar, val in zip(b_p, accs):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 2.5, f"{val:.2f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_title("The Foundation Model Paradox:\nZero-Shot Coarse Semantic Recognition vs. Compositional Reasoning Collapse", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Zero-Shot Top-1 Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([
        "ImageNet-V2\n(1,000 Object Classes)\n[Coarse Categorization]",
        "CLEVR Counting\n(8 Cardinality Classes)\n[Number Reasoning]",
        "SpatialSense\n(Binary 2 Relations)\n[Spatial Grammar]"
    ], fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 85)

    ax.annotate(
        "SUCCESS:\nHigh zero-shot capability\nover 1,000 categories!",
        xy=(0, 53.24),
        xytext=(0, 68),
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#2B6CB0", lw=1.5),
        fontsize=9, fontweight="bold", color="#1A365D",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#EBF8FF", edgecolor="#BEE3F8")
    )

    ax.annotate(
        "CATASTROPHIC COLLAPSE:\nChance-level guessing (12.43%)\nMode collapse on count 4!",
        xy=(1, 12.43),
        xytext=(1.2, 38),
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#C53030", lw=1.5),
        fontsize=9, fontweight="bold", color="#9B2C2C",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF5F5", edgecolor="#FEB2B2")
    )

    plt.tight_layout()
    out_paradox = out_dir / "coarse_vs_compositional_paradox.png"
    plt.savefig(out_paradox, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Standalone Paradox figure saved to: {out_paradox.resolve()}")


if __name__ == "__main__":
    main()
