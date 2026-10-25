"""scripts/plot_coco_retrieval.py

Generates publication-quality figures for MS-COCO 2017 Validation cross-modal retrieval:
- Bidirectional Recall@K curves (Image->Text and Text->Image)
- Comparison between 1-to-1 instance retrieval, 5-caption benchmark, and OpenAI CLIP literature reference
- Median Rank comparison

Saves to: results/figures/coco_retrieval_benchmark.png
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Style setup
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def main():
    json_path = Path("results/tables/coco_retrieval_results.json")
    if not json_path.exists():
        print(f"[Error] {json_path} does not exist.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    o2o = data["one_to_one_5k_evaluation"]["recall"]
    std = data["standard_multi_caption_5k_benchmark"]
    lit = data["expected_clip_vit_b32_literature"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # ----------------------------------------------------
    # Subplot 1: Image-to-Text Retrieval
    # ----------------------------------------------------
    ax1 = axes[0]
    k_labels = ["R@1", "R@5", "R@10"]
    x = np.arange(len(k_labels))
    width = 0.25

    vals_1to1_i2t = [o2o["i2t"]["R@1"], o2o["i2t"]["R@5"], o2o["i2t"]["R@10"]]
    vals_std_i2t = [std["i2t"]["R@1"], std["i2t"]["R@5"], std["i2t"]["R@10"]]
    vals_lit_i2t = [lit["i2t_R@1"], lit["i2t_R@5"], lit["i2t_R@10"]]

    b1 = ax1.bar(x - width, vals_1to1_i2t, width, label="1-to-1 Instance (5k Pairs)", color="#3182CE", alpha=0.9)
    b2 = ax1.bar(x, vals_std_i2t, width, label="Standard 5-Caption (Ours)", color="#38A169", alpha=0.9)
    b3 = ax1.bar(x + width, vals_lit_i2t, width, label="Radford et al. 2021 (Ref)", color="#D69E2E", alpha=0.9)

    ax1.set_title("Image-to-Text (I2T) Cross-Modal Retrieval", fontsize=14, fontweight="bold", pad=12)
    ax1.set_ylabel("Recall (%)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(k_labels, fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper left", framealpha=0.95, fontsize=10)

    # Add value annotations
    for bars in [b1, b2, b3]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1.5,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # Annotate Median Rank
    medr_text = (
        f"Median Rank:\n"
        f"  1-to-1: {data['one_to_one_5k_evaluation']['ranks']['i2t_median_rank']:.1f}\n"
        f"  5-Caption: {std['i2t']['median_rank']:.1f}\n"
        f"  Literature Ref: {lit['i2t_median_rank']:.1f}"
    )
    ax1.text(
        0.96,
        0.05,
        medr_text,
        transform=ax1.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#CBD5E0", alpha=0.9),
    )

    # ----------------------------------------------------
    # Subplot 2: Text-to-Image Retrieval
    # ----------------------------------------------------
    ax2 = axes[1]
    vals_1to1_t2i = [o2o["t2i"]["R@1"], o2o["t2i"]["R@5"], o2o["t2i"]["R@10"]]
    vals_std_t2i = [std["t2i"]["R@1"], std["t2i"]["R@5"], std["t2i"]["R@10"]]
    vals_lit_t2i = [lit["t2i_R@1"], lit["t2i_R@5"], lit["t2i_R@10"]]

    b4 = ax2.bar(x - width, vals_1to1_t2i, width, label="1-to-1 Instance (5k Pairs)", color="#3182CE", alpha=0.9)
    b5 = ax2.bar(x, vals_std_t2i, width, label="Standard 5-Caption (Ours)", color="#38A169", alpha=0.9)
    b6 = ax2.bar(x + width, vals_lit_t2i, width, label="Radford et al. 2021 (Ref)", color="#D69E2E", alpha=0.9)

    ax2.set_title("Text-to-Image (T2I) Cross-Modal Retrieval", fontsize=14, fontweight="bold", pad=12)
    ax2.set_ylabel("Recall (%)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(k_labels, fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper left", framealpha=0.95, fontsize=10)

    for bars in [b4, b5, b6]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1.5,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    medr_text_t2i = (
        f"Median Rank:\n"
        f"  1-to-1: {data['one_to_one_5k_evaluation']['ranks']['t2i_median_rank']:.1f}\n"
        f"  5-Caption: {std['t2i']['median_rank']:.1f}\n"
        f"  Literature Ref: {lit['t2i_median_rank']:.1f}"
    )
    ax2.text(
        0.96,
        0.05,
        medr_text_t2i,
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#CBD5E0", alpha=0.9),
    )

    fig.suptitle(
        "MS-COCO 2017 Validation Cross-Modal Retrieval Performance (CLIP ViT-B/32)\n"
        "Validating Cluster 6B Metrics on Natural 1-to-1 Human Image-Caption Pairs",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "coco_retrieval_benchmark.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Figure generated and saved to: {out_file.resolve()}")


if __name__ == "__main__":
    main()
