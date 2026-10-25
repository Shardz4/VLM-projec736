"""scripts/plot_minimalist_benchmarks.py

Generates ultra-minimalist, publication-grade presentation graphics:
1. results/figures/minimalist_zero_shot_audit.png:
   Clean, uncluttered bar chart displaying only:
   - ImageNet-V2: 53.2%
   - CLEVR Counting: 12.4%
   - SpatialSense: 57.7%
   with subtle dashed chance baselines and minimal labeling.

2. results/figures/minimalist_resolution.png:
   Clean contrast showing the resolution of the counting failure:
   - CLIP Zero-Shot: 12.4%
   - Cross-Attention (Ours): 69.8%
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set ultra-clean styling
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
plt.rcParams["axes.edgecolor"] = "#E2E8F0"
plt.rcParams["axes.linewidth"] = 0.8


def plot_minimalist_audit(output_path: Path):
    """Ultra-minimalist chart for the 3 core benchmarks."""
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    datasets = ["ImageNet-V2", "CLEVR Counting", "SpatialSense"]
    accuracies = [53.24, 12.43, 57.70]
    chances = [0.1, 12.5, 50.0]
    colors = ["#4F46E5", "#EF4444", "#F59E0B"]  # Indigo, Coral, Amber

    x = np.arange(len(datasets))
    bars = ax.bar(x, accuracies, width=0.48, color=colors, edgecolor="none", zorder=3)

    # Clean chance baseline dashed segments
    for i, c in enumerate(chances):
        ax.plot([i - 0.30, i + 0.30], [c, c], color="#94A3B8", linestyle="--", linewidth=1.5, zorder=4)
        ax.text(i, c + 1.8, f"Chance: {c:.1f}%", ha="center", va="bottom", fontsize=9, color="#64748B", fontweight="medium")

    # Value labels on top of bars
    for bar, acc in zip(bars, accuracies):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 2.2,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color="#0F172A",
        )

    # Minimal axes
    ax.set_title("Zero-Shot Accuracy Across Core Benchmarks (CLIP ViT-B/32)", fontsize=13, fontweight="bold", pad=16, color="#0F172A")
    ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="medium", color="#475569")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11.5, fontweight="bold", color="#1E293B")
    ax.set_ylim(0, 75)

    # Clean horizontal grid only
    ax.yaxis.grid(True, linestyle="-", color="#F1F5F9", linewidth=1.0, zorder=0)
    ax.xaxis.grid(False)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved minimalist audit chart: {output_path.resolve()}")


def plot_minimalist_resolution(output_path: Path):
    """Ultra-minimalist chart comparing Zero-Shot Counting vs Our Cross-Attention."""
    fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    models = ["CLIP Zero-Shot\n(Baseline)", "Cross-Attention\n(Ours)"]
    accuracies = [12.43, 69.83]
    colors = ["#94A3B8", "#10B981"]  # Slate Gray vs Emerald Green

    x = np.arange(len(models))
    bars = ax.bar(x, accuracies, width=0.45, color=colors, edgecolor="none", zorder=3)

    # Chance line for counting (12.5%)
    ax.axhline(12.5, color="#DC2626", linestyle="--", linewidth=1.3, zorder=2)
    ax.text(0.5, 13.8, "Random Chance (12.5%)", ha="center", va="bottom", fontsize=9, color="#DC2626", fontweight="bold")

    for bar, acc in zip(bars, accuracies):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 2.0,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color="#0F172A",
        )

    # Delta badge (+57.4%)
    ax.annotate(
        "+57.4% Gain",
        xy=(1, 69.83),
        xytext=(0.5, 52),
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#059669", lw=1.5),
        fontsize=10.5,
        fontweight="bold",
        color="#047857",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ECFDF5", edgecolor="#A7F3D0"),
    )

    ax.set_title("CLEVR Counting: Baseline vs. Proposed Solution", fontsize=12.5, fontweight="bold", pad=16, color="#0F172A")
    ax.set_ylabel("Accuracy (%)", fontsize=10.5, fontweight="medium", color="#475569")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight="bold", color="#1E293B")
    ax.set_ylim(0, 85)

    ax.yaxis.grid(True, linestyle="-", color="#F1F5F9", linewidth=1.0, zorder=0)
    ax.xaxis.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved minimalist resolution chart: {output_path.resolve()}")


def main():
    out_dir = Path("results/figures")
    plot_minimalist_audit(out_dir / "minimalist_zero_shot_audit.png")
    plot_minimalist_resolution(out_dir / "minimalist_counting_resolution.png")


if __name__ == "__main__":
    main()
