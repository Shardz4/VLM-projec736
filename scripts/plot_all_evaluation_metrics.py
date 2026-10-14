"""scripts/plot_all_evaluation_metrics.py

Generate individual and combined publication-grade matplotlib figures
for every single evaluation metric across the 8-model benchmark pool (Step 13).

Produces:
- 9 individual high-resolution PNG charts (one per metric)
- 1 master 3x3 combined dashboard grid comparing all metrics side-by-side
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

# Palette definitions
COLOR_BASELINE = "#4A5568"    # Slate grey for standard baselines
COLOR_ABLATION = "#2B6CB0"    # Steel blue for ablations / variants
COLOR_OURS = "#D69E2E"        # Vibrant gold for Cross-Attention (Ours)
COLOR_WARN = "#C53030"        # Crimson for pathological / unstable models (e.g. Early Fusion ECE)

MODEL_SHORT_NAMES = [
    "M1: CLIP ViT\nZero-Shot",
    "M2: ResNet\nAvgpool",
    "M3: ResNet\nSpatial",
    "M4: CLIP ViT\nLinear",
    "M5: Early\nFusion MLP",
    "M6: Late\nFusion",
    "M7: Cross-Attn\n(Ours)",
    "M8: Cross-Attn\n(p=0.5)",
]

METRICS_META = [
    {
        "key": "Accuracy",
        "title": "Top-1 Classification Accuracy",
        "unit": "%",
        "better": "higher",
        "format": "{:.2f}%",
        "filename": "01_accuracy.png",
        "ylim": (0, 80),
        "desc": "Percentage of test images with exactly correct object count prediction."
    },
    {
        "key": "MAE",
        "title": "Mean Absolute Error (Counting MAE)",
        "unit": "counts",
        "better": "lower",
        "format": "{:.3f}",
        "filename": "02_mae.png",
        "ylim": (0, 2.3),
        "desc": "Average numerical distance between predicted count and ground-truth count."
    },
    {
        "key": "Macro_F1",
        "title": "Macro-Averaged F1 Score",
        "unit": "%",
        "better": "higher",
        "format": "{:.2f}%",
        "filename": "03_macro_f1.png",
        "ylim": (0, 80),
        "desc": "Unweighted mean of class-wise F1 scores across all 8 cardinality classes."
    },
    {
        "key": "ECE",
        "title": "Expected Calibration Error (ECE)",
        "unit": "",
        "better": "lower",
        "format": "{:.4f}",
        "filename": "04_ece.png",
        "ylim": (0, 0.32),
        "desc": "Discrepancy between predicted confidence and true empirical correctness."
    },
    {
        "key": "Recall@1",
        "title": "Image-to-Text Retrieval Recall@1",
        "unit": "%",
        "better": "higher",
        "format": "{:.2f}%",
        "filename": "05_recall1.png",
        "ylim": (0, 1.0),
        "desc": "Fraction of queries where ground truth is the exact #1 ranked candidate."
    },
    {
        "key": "Recall@5",
        "title": "Image-to-Text Retrieval Recall@5",
        "unit": "%",
        "better": "higher",
        "format": "{:.2f}%",
        "filename": "06_recall5.png",
        "ylim": (0, 1.0),
        "desc": "Fraction of queries where ground truth ranks within the top 5 candidates."
    },
    {
        "key": "Median_Rank",
        "title": "Retrieval Median Rank (MedR)",
        "unit": "rank",
        "better": "lower",
        "format": "{:.1f}",
        "filename": "07_median_rank.png",
        "ylim": (0, 1800),
        "desc": "Median rank of target candidates across 3,000 queries (random = 1500.0)."
    },
    {
        "key": "Modality_Gap",
        "title": "Hypersphere Modality Gap (Δ_gap)",
        "unit": "L2 dist",
        "better": "lower",
        "format": "{:.4f}",
        "filename": "08_modality_gap.png",
        "ylim": (1.0, 1.55),
        "desc": "L2 Euclidean distance between mean vision and text representation vectors."
    },
    {
        "key": "CMA",
        "title": "Cross-Modal Alignment (CMA)",
        "unit": "cosine",
        "better": "higher",
        "format": "{:.4f}",
        "filename": "09_cma.png",
        "ylim": (-0.15, 0.18),
        "desc": "Average pairwise cosine similarity between paired visual and text embeddings."
    },
]


def load_data():
    json_path = Path("results/tables/step13_metric_comparison_results.json")
    if not json_path.exists():
        raise FileNotFoundError(f"Could not find results JSON at {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["scores"]


def get_bar_colors(meta, values):
    """Assign semantic colors: Highlight M7 (Ours) in Gold, M8 in Blue, others in Slate."""
    colors = []
    for i in range(len(values)):
        if i == 6:  # M7: Cross-Attention (Ours)
            colors.append("#D97706")  # Amber gold
        elif i == 7:  # M8: Cross-Attn Ablation
            colors.append("#2563EB")  # Royal blue
        elif i == 4 and meta["key"] == "ECE":  # Early Fusion pathological ECE
            colors.append("#DC2626")  # Red alert
        else:
            colors.append("#64748B")  # Slate gray
    return colors


def plot_single_metric(meta, scores, output_dir: Path):
    """Generate an individual publication-quality bar chart for one evaluation metric."""
    key = meta["key"]
    values = [s[key] for s in scores]
    colors = get_bar_colors(meta, values)
    
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    
    x = np.arange(len(MODEL_SHORT_NAMES))
    bars = ax.bar(x, values, color=colors, width=0.55, edgecolor="#1E293B", linewidth=1.0, zorder=3)
    
    # Add values on top of bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        formatted_val = meta["format"].format(val)
        va = "bottom" if height >= 0 else "top"
        y_pos = height + (0.015 * (meta["ylim"][1] - meta["ylim"][0]) if height >= 0 else -0.03 * (meta["ylim"][1] - meta["ylim"][0]))
        
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_pos,
            formatted_val,
            ha="center",
            va=va,
            fontsize=9.5,
            fontweight="bold",
            color="#1E293B",
            zorder=4
        )

    # Orientation badge
    arrow = "↑ Higher is Better" if meta["better"] == "higher" else "↓ Lower is Better"
    badge_color = "#047857" if meta["better"] == "higher" else "#B91C1C"
    
    # Title & Labels
    ax.set_title(f"{meta['title']} across 8-Model Pool", fontsize=14, fontweight="bold", pad=20, color="#0F172A")
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_SHORT_NAMES, fontsize=9.5, fontweight="medium")
    ax.set_ylabel(f"{meta['title']} ({meta['unit']})" if meta['unit'] else meta['title'], fontsize=10.5, fontweight="bold")
    
    if meta.get("ylim"):
        ax.set_ylim(meta["ylim"])
        
    ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
    ax.grid(axis="x", visible=False)
    
    # Subtitle / description banner
    fig.text(0.5, 0.90, f"{meta['desc']}  •  [{arrow}]", ha="center", fontsize=9.5, color=badge_color, fontweight="semibold")

    # Legend / Callout
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#D97706", edgecolor="#1E293B", label="M7: Cross-Attention (Ours)"),
        Patch(facecolor="#2563EB", edgecolor="#1E293B", label="M8: Cross-Attn (p=0.5 Ablation)"),
        Patch(facecolor="#64748B", edgecolor="#1E293B", label="M1–M6: Probes & Baselines"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.95, fontsize=8.5)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    output_path = output_dir / meta["filename"]
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"  [OK] Saved individual plot: {output_path}")


def plot_master_grid(scores, output_dir: Path):
    """Generate a combined 3x3 master grid figure showing all 9 metrics in a single layout."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=300)
    axes = axes.flatten()
    x = np.arange(len(MODEL_SHORT_NAMES))

    for idx, (meta, ax) in enumerate(zip(METRICS_META, axes)):
        key = meta["key"]
        values = [s[key] for s in scores]
        colors = get_bar_colors(meta, values)

        bars = ax.bar(x, values, color=colors, width=0.6, edgecolor="#1E293B", linewidth=0.8, zorder=3)
        arrow = "↑" if meta["better"] == "higher" else "↓"
        arrow_color = "#047857" if meta["better"] == "higher" else "#B91C1C"

        # Values on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            formatted_val = meta["format"].format(val)
            va = "bottom" if height >= 0 else "top"
            y_offset = 0.02 * (meta["ylim"][1] - meta["ylim"][0]) if height >= 0 else -0.04 * (meta["ylim"][1] - meta["ylim"][0])
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + y_offset,
                formatted_val,
                ha="center",
                va=va,
                fontsize=7.5,
                fontweight="bold",
                color="#0F172A",
                zorder=4
            )

        ax.set_title(f"({chr(97 + idx)}) {meta['title']} [{arrow}]", fontsize=11, fontweight="bold", color="#0F172A")
        ax.set_xticks(x)
        # On bottom row keep labels, on other rows use compact M1-M8
        if idx >= 6:
            ax.set_xticklabels([f"M{i+1}" for i in range(8)], fontsize=9, fontweight="bold")
            ax.set_xlabel("Models (M1–M8)", fontsize=9, fontweight="bold")
        else:
            ax.set_xticklabels([f"M{i+1}" for i in range(8)], fontsize=8.5)

        if meta.get("ylim"):
            ax.set_ylim(meta["ylim"])

        ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
        ax.grid(axis="x", visible=False)

    # Master title
    fig.suptitle(
        "Step 13: Comprehensive 8-Model Pool Comparison Across All 9 Evaluation Metrics",
        fontsize=16,
        fontweight="bold",
        color="#0F172A",
        y=0.98
    )

    # Global legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#D97706", edgecolor="#1E293B", label="M7: Cross-Attention (Ours)"),
        Patch(facecolor="#2563EB", edgecolor="#1E293B", label="M8: Cross-Attn (Ablation p=0.5)"),
        Patch(facecolor="#DC2626", edgecolor="#1E293B", label="M5: Early Fusion (High ECE Pathological)"),
        Patch(facecolor="#64748B", edgecolor="#1E293B", label="M1–M4, M6: Standard Baselines"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=4, framealpha=0.95, fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    grid_path = output_dir / "all_metrics_8model_grid.png"
    plt.savefig(grid_path, dpi=300)
    plt.close(fig)
    print(f"\n✓ Saved 9-in-1 Master Dashboard Grid: {grid_path}")


def main():
    print("=" * 70)
    print("STEP 13: GENERATING 8-MODEL EVALUATION METRIC GRAPHS VIA MATPLOTLIB")
    print("=" * 70)

    output_dir = Path("results/figures/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = load_data()
    print(f"Loaded evaluation results for {len(scores)} models.\n")

    print("[1/2] Generating 9 Individual Metric Comparison Graphs...")
    for meta in METRICS_META:
        plot_single_metric(meta, scores, output_dir)

    print("\n[2/2] Generating Master 9-in-1 Comparison Dashboard...")
    plot_master_grid(scores, output_dir)

    print("\n" + "=" * 70)
    print(f"ALL PLOTS GENERATED SUCCESSFULLY IN: {output_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
