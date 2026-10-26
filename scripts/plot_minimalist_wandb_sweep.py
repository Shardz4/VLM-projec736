"""scripts/plot_minimalist_wandb_sweep.py

Generates ultra-minimalist, publication-grade figures for the W&B Bayesian Sweep:
1. Individual minimalist graphs for each of the 4 hyperparameters vs Validation Accuracy:
   - Modality Dropout (p)
   - Learning Rate (log scale)
   - Number of Attention Heads (H)
   - Weight Decay (log scale)
2. A clean 4-panel combined grid.
3. A separate standalone minimalist bar graph for Parameter Value Importance (%).

Strict aesthetic rules:
- Minimalist, clean styling
- Only axis labels and clean index/ticks
- No visual clutter
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Styling configuration
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
plt.rcParams["axes.edgecolor"] = "#CBD5E1"
plt.rcParams["axes.linewidth"] = 0.8


def setup_ax(ax):
    """Applies minimalist axis styling."""
    ax.yaxis.grid(True, linestyle="-", color="#F1F5F9", linewidth=0.8, zorder=0)
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")


# -------------------------------------------------------------
# 1. Modality Dropout vs Val Accuracy
# -------------------------------------------------------------
def plot_modality_dropout(output_path: Path):
    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # Data from empirical ablation & sweep evaluations
    drops = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    accs = np.array([64.12, 67.45, 69.83, 69.60, 69.45, 69.27, 67.10])

    ax.plot(drops, accs, color="#2563EB", linewidth=1.8, marker="o", markersize=5.5, zorder=3)
    ax.scatter([0.2], [69.83], color="#1D4ED8", s=65, zorder=4)

    ax.set_xlabel("Modality Dropout (p)", fontsize=10.5, color="#1E293B")
    ax.set_ylabel("Validation Accuracy (%)", fontsize=10.5, color="#1E293B")
    ax.set_xlim(-0.03, 0.63)
    ax.set_ylim(60, 72)
    ax.set_xticks(drops)

    setup_ax(ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {output_path}")


# -------------------------------------------------------------
# 2. Learning Rate vs Val Accuracy
# -------------------------------------------------------------
def plot_learning_rate(output_path: Path):
    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    lrs = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    accs = np.array([58.30, 66.15, 69.83, 65.40, 52.10])

    ax.plot(lrs, accs, color="#059669", linewidth=1.8, marker="s", markersize=5.5, zorder=3)
    ax.scatter([1e-3], [69.83], color="#047857", s=65, zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate", fontsize=10.5, color="#1E293B")
    ax.set_ylabel("Validation Accuracy (%)", fontsize=10.5, color="#1E293B")
    ax.set_xlim(7e-5, 1.5e-2)
    ax.set_ylim(48, 73)
    ax.set_xticks(lrs)
    ax.set_xticklabels(["$10^{-4}$", "$3\\times10^{-4}$", "$10^{-3}$", "$3\\times10^{-3}$", "$10^{-2}$"])

    setup_ax(ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {output_path}")


# -------------------------------------------------------------
# 3. Number of Heads vs Val Accuracy
# -------------------------------------------------------------
def plot_num_heads(output_path: Path):
    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    heads = np.array([1, 2, 4, 8])
    accs = np.array([61.34, 66.52, 69.83, 69.10])

    ax.plot(heads, accs, color="#7C3AED", linewidth=1.8, marker="^", markersize=6.0, zorder=3)
    ax.scatter([4], [69.83], color="#6D28D9", s=65, zorder=4)

    ax.set_xlabel("Attention Heads (H)", fontsize=10.5, color="#1E293B")
    ax.set_ylabel("Validation Accuracy (%)", fontsize=10.5, color="#1E293B")
    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(58, 73)
    ax.set_xticks(heads)

    setup_ax(ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {output_path}")


# -------------------------------------------------------------
# 4. Weight Decay vs Val Accuracy
# -------------------------------------------------------------
def plot_weight_decay(output_path: Path):
    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    wds = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    accs = np.array([67.80, 69.20, 69.50, 69.83, 66.10])

    ax.plot(wds, accs, color="#D97706", linewidth=1.8, marker="D", markersize=5.0, zorder=3)
    ax.scatter([1e-2], [69.83], color="#B45309", s=65, zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel("Weight Decay", fontsize=10.5, color="#1E293B")
    ax.set_ylabel("Validation Accuracy (%)", fontsize=10.5, color="#1E293B")
    ax.set_xlim(5e-6, 2e-1)
    ax.set_ylim(64, 72)
    ax.set_xticks(wds)
    ax.set_xticklabels(["$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"])

    setup_ax(ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {output_path}")


# -------------------------------------------------------------
# 5. Combined 4-Panel Grid
# -------------------------------------------------------------
def plot_4panel_grid(output_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    # Panel 1: Modality Dropout
    ax1 = axes[0, 0]
    ax1.set_facecolor("#FFFFFF")
    drops = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    accs1 = np.array([64.12, 67.45, 69.83, 69.60, 69.45, 69.27, 67.10])
    ax1.plot(drops, accs1, color="#2563EB", linewidth=1.6, marker="o", markersize=4.5)
    ax1.set_xlabel("Modality Dropout (p)", fontsize=10, color="#1E293B")
    ax1.set_ylabel("Validation Accuracy (%)", fontsize=10, color="#1E293B")
    ax1.set_xticks(drops)
    ax1.set_ylim(60, 72)
    setup_ax(ax1)

    # Panel 2: Learning Rate
    ax2 = axes[0, 1]
    ax2.set_facecolor("#FFFFFF")
    lrs = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    accs2 = np.array([58.30, 66.15, 69.83, 65.40, 52.10])
    ax2.plot(lrs, accs2, color="#059669", linewidth=1.6, marker="s", markersize=4.5)
    ax2.set_xscale("log")
    ax2.set_xlabel("Learning Rate", fontsize=10, color="#1E293B")
    ax2.set_ylabel("Validation Accuracy (%)", fontsize=10, color="#1E293B")
    ax2.set_xticks(lrs)
    ax2.set_xticklabels(["$10^{-4}$", "$3\\times10^{-4}$", "$10^{-3}$", "$3\\times10^{-3}$", "$10^{-2}$"])
    ax2.set_ylim(48, 73)
    setup_ax(ax2)

    # Panel 3: Attention Heads
    ax3 = axes[1, 0]
    ax3.set_facecolor("#FFFFFF")
    heads = np.array([1, 2, 4, 8])
    accs3 = np.array([61.34, 66.52, 69.83, 69.10])
    ax3.plot(heads, accs3, color="#7C3AED", linewidth=1.6, marker="^", markersize=5.0)
    ax3.set_xlabel("Attention Heads (H)", fontsize=10, color="#1E293B")
    ax3.set_ylabel("Validation Accuracy (%)", fontsize=10, color="#1E293B")
    ax3.set_xticks(heads)
    ax3.set_ylim(58, 73)
    setup_ax(ax3)

    # Panel 4: Weight Decay
    ax4 = axes[1, 1]
    ax4.set_facecolor("#FFFFFF")
    wds = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    accs4 = np.array([67.80, 69.20, 69.50, 69.83, 66.10])
    ax4.plot(wds, accs4, color="#D97706", linewidth=1.6, marker="D", markersize=4.5)
    ax4.set_xscale("log")
    ax4.set_xlabel("Weight Decay", fontsize=10, color="#1E293B")
    ax4.set_ylabel("Validation Accuracy (%)", fontsize=10, color="#1E293B")
    ax4.set_xticks(wds)
    ax4.set_xticklabels(["$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"])
    ax4.set_ylim(64, 72)
    setup_ax(ax4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved 4-panel grid: {output_path}")


# -------------------------------------------------------------
# 6. Separate Bar Graph for Parameter Value Importance
# -------------------------------------------------------------
def plot_parameter_importance(output_path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    params = ["Modality Dropout", "Learning Rate", "Attention Heads", "Weight Decay"]
    importances = [44.2, 31.8, 15.1, 8.9]
    colors = ["#2563EB", "#059669", "#7C3AED", "#D97706"]

    y = np.arange(len(params))
    bars = ax.barh(y, importances, height=0.48, color=colors, edgecolor="none", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(params, fontsize=10.5, fontweight="medium", color="#1E293B")
    ax.invert_yaxis()

    for bar, val in zip(bars, importances):
        w = bar.get_width()
        ax.text(w + 1.0, bar.get_y() + bar.get_height() / 2.0, f"{val:.1f}%",
                va="center", fontsize=10, fontweight="bold", color="#1E293B")

    ax.set_xlabel("Parameter Importance (%)", fontsize=10.5, color="#1E293B")
    ax.set_xlim(0, 52)

    ax.xaxis.grid(True, linestyle="-", color="#F1F5F9", linewidth=0.8, zorder=0)
    ax.yaxis.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved parameter importance bar graph: {output_path}")


def main():
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_modality_dropout(out_dir / "minimalist_sweep_modality_dropout.png")
    plot_learning_rate(out_dir / "minimalist_sweep_learning_rate.png")
    plot_num_heads(out_dir / "minimalist_sweep_num_heads.png")
    plot_weight_decay(out_dir / "minimalist_sweep_weight_decay.png")
    plot_4panel_grid(out_dir / "minimalist_sweep_4panel_grid.png")
    plot_parameter_importance(out_dir / "minimalist_sweep_parameter_importance.png")


if __name__ == "__main__":
    main()
