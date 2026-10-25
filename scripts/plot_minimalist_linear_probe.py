"""scripts/plot_minimalist_linear_probe.py

Ultra-minimalist line graph comparing only:
1. CLIP Linear Probe (Validation Accuracy per Epoch)
2. CLIP Zero-Shot Baseline (12.43%)

Formatting:
- Thin, single lines (not bold)
- No ResNet-50
- No Random Chance baseline
- No shaded zones or verbose callout boxes
- Clean minimalist axes and legend

Saves to: results/figures/minimalist_linear_probe_accuracy.png
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt

# Clean minimalist style
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
plt.rcParams["axes.edgecolor"] = "#CBD5E1"
plt.rcParams["axes.linewidth"] = 0.8


def main():
    json_path = Path("results/logs/linear_probe_clip_counting.json")
    if not json_path.exists():
        print(f"[Error] {json_path} does not exist.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    history = data.get("training_history", [])
    epochs = [h["epoch"] for h in history]
    val_acc = [h["val_accuracy"] * 100.0 for h in history]

    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # 1. CLIP Linear Probe Validation Accuracy (Single, thin solid line)
    ax.plot(
        epochs,
        val_acc,
        color="#2563EB",
        linewidth=1.4,
        linestyle="-",
        label=f"CLIP Linear Probe ({val_acc[-1]:.1f}%)",
    )

    # 2. CLIP Zero-Shot Baseline (Single, thin dashed line)
    ax.axhline(
        12.43,
        color="#64748B",
        linewidth=1.1,
        linestyle="--",
        label="CLIP Zero-Shot (12.4%)",
    )

    # Clean simple labels
    ax.set_title("CLIP Linear Probe vs. Zero-Shot: Validation Accuracy per Epoch", fontsize=11.5, fontweight="bold", pad=12, color="#0F172A")
    ax.set_xlabel("Epoch", fontsize=10, color="#334155")
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=10, color="#334155")
    ax.set_xlim(1, len(epochs))
    ax.set_ylim(0, 55)

    # Minimalist legend
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9.5)

    # Subtle horizontal grid only
    ax.yaxis.grid(True, linestyle="-", color="#F1F5F9", linewidth=0.8)
    ax.xaxis.grid(False)

    # Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")

    plt.tight_layout()
    out_file = Path("results/figures/minimalist_linear_probe_accuracy.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved ultra-minimalist linear probe graph to: {out_file.resolve()}")


if __name__ == "__main__":
    main()
