"""scripts/plot_minimalist_coco_recall.py

Generates an ultra-minimalist, publication-grade bar graph for MS-COCO 2017 cross-modal retrieval:
- Evaluates Recall@1, Recall@5, Recall@10 for Image-to-Text and Text-to-Image
- Strict minimalist aesthetics: only axis labels, bar index (legend), clean flat bars
- Saves to: results/figures/minimalist_coco_recall_bar.png
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Ultra-clean styling
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
plt.rcParams["axes.edgecolor"] = "#CBD5E1"
plt.rcParams["axes.linewidth"] = 0.8


def plot_minimalist_recall_bar(json_path: Path, output_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    std = data["standard_multi_caption_5k_benchmark"]

    metrics = ["R@1", "R@5", "R@10"]
    i2t_vals = [std["i2t"]["R@1"], std["i2t"]["R@5"], std["i2t"]["R@10"]]
    t2i_vals = [std["t2i"]["R@1"], std["t2i"]["R@5"], std["t2i"]["R@10"]]

    x = np.arange(len(metrics))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # Clean flat bars without harsh borders
    b1 = ax.bar(x - width / 2, i2t_vals, width, label="Image-to-Text", color="#2563EB", edgecolor="none", zorder=3)
    b2 = ax.bar(x + width / 2, t2i_vals, width, label="Text-to-Image", color="#059669", edgecolor="none", zorder=3)

    # Clean value labels on top of bars
    for bars in [b1, b2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1.8,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#1E293B",
            )

    # Minimal axes
    ax.set_xlabel("Retrieval Metric", fontsize=11, fontweight="medium", color="#1E293B")
    ax.set_ylabel("Recall (%)", fontsize=11, fontweight="medium", color="#1E293B")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11, fontweight="bold", color="#1E293B")
    ax.set_ylim(0, 100)

    # Bar index (legend)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=10)

    # Subtle horizontal grid only
    ax.yaxis.grid(True, linestyle="-", color="#F1F5F9", linewidth=0.9, zorder=0)
    ax.xaxis.grid(False)

    # Clean borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved minimalist cross-modality recall bar graph to: {output_path.resolve()}")


def main():
    json_path = Path("results/tables/coco_retrieval_results.json")
    out_path = Path("results/figures/minimalist_coco_recall_bar.png")
    plot_minimalist_recall_bar(json_path, out_path)


if __name__ == "__main__":
    main()
