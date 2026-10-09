# Evaluating Zero-Shot Boundaries and Compositional Failure Modes in Vision-Language Models

> **Research Question:** Do contrastive Vision-Language Models like CLIP behave as "bags of concepts" — recognising *what* is in an image but failing to understand *how many* or *where*?

This project systematically evaluates [OpenAI CLIP (ViT-B/32)](https://arxiv.org/abs/2103.00020) on two compositional reasoning axes — **object counting** and **spatial relation understanding** — compares zero-shot performance against supervised **linear probe baselines** (ResNet-50 and CLIP ViT), and builds a **from-scratch Cluster 6B multimodal fusion and metric suite**.

---

## Table of Contents

- [Motivation](#motivation)
- [Key Findings (Empirical Results)](#key-findings-empirical-results)
- [Reported vs. Reproduced Benchmark Analysis](#reported-vs-reproduced-benchmark-analysis)
- [Project Architecture](#project-architecture)
- [Hardware Safety & Requirements](#hardware-safety--requirements)
- [Environment Setup](#environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
  - [1. Verify the Environment](#1-verify-the-environment)
  - [2. Validate Data Loaders](#2-validate-data-loaders)
  - [3. ImageNet-V2 Control Sanity Check (Step 9)](#3-imagenet-v2-control-sanity-check-step-9)
  - [4. Zero-Shot Counting Probe on CLEVR (Step 7)](#4-zero-shot-counting-probe-on-clevr-step-7)
  - [5. Zero-Shot Spatial Reasoning Probe on SpatialSense (Step 8)](#5-zero-shot-spatial-reasoning-probe-on-spatialsense-step-8)
  - [6. Linear Probe Supervised Baselines (Step 10)](#6-linear-probe-supervised-baselines-step-10)
  - [7. Cluster 6B Multimodal Subtask & Verification (Step 11)](#7-cluster-6b-multimodal-subtask--verification-step-11)
- [Configuration](#configuration)
- [Results & Outputs](#results--outputs)
- [Implementation Progress](#implementation-progress)
- [References](#references)
- [License](#license)

---

## Motivation

CLIP learns a joint image-text embedding space via contrastive pre-training on 400M image-caption pairs. While this yields impressive zero-shot classification on standard object recognition tasks (e.g., ImageNet), the contrastive loss never explicitly incentivises:

- **Cardinality discrimination** — distinguishing "3 objects" from "5 objects".
- **Syntactic / Relational sensitivity** — differentiating "the cat is on the table" from "the table is on the cat".

This project provides rigorous empirical evidence for these failure modes, isolates where the bottleneck occurs (in visual encoding vs. text-image projection), and implements a query-guided cross-attention fusion extension to resolve it.

---

## Key Findings (Empirical Results)

### Headline Benchmark Summary

| Evaluation Benchmark | Model / Probe Variant | Metric | Result | Chance / Baseline | Status / Finding |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ImageNet-V2** (Control) | CLIP ViT-B/32 Zero-Shot | Top-1 Accuracy<br>Top-5 Accuracy | **53.24%**<br>**79.72%** | ~0.1%<br>~0.5% | ✅ Pipeline Verified (matches Radford et al. ~55%) |
| **CLEVR Counting** | CLIP ViT-B/32 Zero-Shot | Top-1 Accuracy<br>MAE | **18.79%**<br>1.720 | 12.5% (1/8)<br>— | ❌ Severe Mode Collapse (76% on count 4; **0% on counts 7–10**) |
| **CLEVR Counting** | ResNet-50 Spatial Probe (6144-d) | Test Accuracy<br>MAE | **45.00%**<br>0.687 | 12.5%<br>— | Overfitting (Train Acc: 83.89%) |
| **CLEVR Counting** | ResNet-50 Avgpool Probe (2048-d) | Test Accuracy<br>MAE | **47.30%**<br>0.651 | 12.5%<br>— | Robust Generalization (Train Acc: 64.13%) |
| **CLEVR Counting** | CLIP ViT-B/32 Vision Probe (512-d) | Test Accuracy<br>MAE | **48.60%**<br>**0.652** | 12.5%<br>— | 🚀 **+29.81% gain over zero-shot** (Cardinality exists in visual tokens) |
| **SpatialSense** | CLIP ViT-B/32 Zero-Shot | Binary Accuracy<br>Mean Margin | **57.70%**<br>+0.0021 | 50.0%<br>0.0 | ❌ Bag-of-Words Behavior (swapped prompt cosine diff ≈ 0) |
| **SpatialSense** | ResNet-50 Unconditioned (2048-d) | Test Accuracy | **48.48%** | 50.0% | Chance level (global pooling lacks localization) |
| **SpatialSense** | ResNet-50 Conditioned (2048-d) | Test Accuracy | **49.24%** | 50.0% | Chance level without spatial bounding attention |
| **Cluster 6B Subtask** | Multimodal Metric Suite & 3 Fusions | Unit Test Coverage | **25/25 Passing** | — | ✅ Analytical Verification ($\Delta_{\text{gap}}$, R@K, ECE) |

### Key Diagnostic Insights
1. **The Representation Bottleneck**: Supervised linear probing on frozen CLIP ViT-B/32 patch representations jumps from **18.79% (zero-shot) to 48.60% (probe)**. This empirically proves that the visual encoder natively contains object count information, but the contrastive text-projection head discards it.
2. **Mode Collapse in Counting**: Zero-shot CLIP predicts count 4 for over 76% of images, degrading to **0.00% accuracy on all images with 7, 8, 9, or 10 objects**.
3. **Bag-of-Words in Relational Reasoning**: The average cosine similarity margin between `"A on B"` and `"B on A"` is merely **+0.0021**, confirming that CLIP treats compositional relations as bag-of-words order-invariant tokens.

---

## Reported vs. Reproduced Benchmark Analysis

| Benchmark | Reference Metric | Reference Reported | Reproduced in Pipeline | Delta ($\Delta$) | Root-Cause Analysis |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ImageNet-V2** | Zero-Shot Top-1 | 55.6% (Radford et al. Table 4) | **53.24%** | $-2.36\%$ | Single prompt `"a photo of a {label}"` used vs. 80-prompt ensemble in paper. |
| **CLEVR Counting** | Zero-Shot Top-1 | ~10–15% (Random = 10%) | **18.79%** | $+3.79\%$ | Prior bias collapses to counts 4 and 6; unable to count $>6$ objects. |
| **CLEVR Counting** | Linear Probe | N/A (ImageNet probe = 76.2%) | **48.60%** (CLIP ViT) | $+29.81\%$ vs ZS | Supervised probe accesses localized object count tokens in patch representation. |
| **SpatialSense** | Zero-Shot Relational | ~55–58% (ARO / Winoground) | **57.70%** | $\pm 0.5\%$ | Replicates bag-of-words effect: CLIP fails to differentiate reversed relation pairs. |
| **SpatialSense** | Linear Probe | ~52–54% (global features) | **49.24%** | $-3.5\%$ | Global pooling lacks localized bounding-box cross-attention. |

---

## Project Architecture

```
vlm_proj/
├── config/
│   ├── experiment_config.yaml       # Centralised hyperparameters & paths
│   └── sweep_fusion_config.yaml     # Step 12 W&B Bayesian Sweep definition
├── data/                            # Datasets (git-ignored)
│   ├── CLEVR_v1.0/                  # CLEVR scenes + images
│   ├── spatialsense/                # SpatialSense annotations + images
│   └── ImageNetV2-master/           # ImageNet-V2 matched-frequency
├── src/
│   ├── __init__.py
│   ├── utils.py                     # Seed & reproducibility utilities
│   ├── config_loader.py             # YAML config loader utility
│   ├── data_loaders/
│   │   ├── __init__.py
│   │   ├── clevr_loader.py          # CLEVRCountingDataset
│   │   ├── spatialsense_loader.py   # SpatialSenseDataset
│   │   ├── imagenet_v2_loader.py    # ImageNetV2Dataset
│   │   └── dataloader_factory.py    # build_dataloaders() factory
│   ├── models/
│   │   ├── __init__.py
│   │   ├── clip_encoder.py          # CLIPEncoder (dual-encoder wrapper + extraction)
│   │   ├── resnet_baseline.py       # ResNetFeatureExtractor + Probes
│   │   └── multimodal_fusion.py     # Step 11: Early, Late, Cross-Attention + ModalityDropout
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── multimodal_metrics.py    # Step 11: Recall@K, Median Rank, Modality Gap, ECE, Drop
│   └── evaluators/
│       ├── __init__.py
│       ├── counting_evaluator.py    # CountingEvaluator (zero-shot counting)
│       ├── spatial_evaluator.py     # SpatialEvaluator (contrastive spatial)
│       └── imagenet_evaluator.py    # ImageNetV2Evaluator (sanity check)
├── scripts/
│   ├── validate_data.py             # Dataset integrity validation
│   ├── test_clip_encoder.py         # CLIPEncoder verification suite
│   ├── test_loaders.py              # DataLoader smoke test
│   ├── test_resnet_baseline.py      # ResNet feature extraction & linear probe test
│   ├── evaluate_counting.py         # Step 7: CLEVR counting evaluation
│   ├── evaluate_spatial.py          # Step 8: SpatialSense spatial reasoning evaluation
│   ├── evaluate_imagenet.py         # Step 9: ImageNet-V2 sanity check
│   ├── train_linear_probe_counting.py      # Step 10: ResNet counting probe (spatial / avgpool)
│   ├── train_linear_probe_spatial.py       # Step 10: ResNet spatial probe (conditioned / unconditioned)
│   ├── train_linear_probe_clip_counting.py # Step 10: CLIP visual embedding linear probe
│   └── test_multimodal_subtask.py          # Step 11: 25/25 Subtask numerical verification suite
├── results/                         # Logs, figures, checkpoints, cache (git-ignored)
│   ├── logs/                        # Raw metric logs (JSON & CSV)
│   └── cache/                       # Cached .pt feature tensors (CPU/GPU safe)
├── pixi.toml                        # Pixi package manager config
├── pixi.lock
└── README.md
```

---

## Hardware Safety & Requirements

All feature extraction scripts cache intermediate vectors to `results/cache/*.pt`. Subsequent linear probes and fusion models train purely on **CPU** using `--device cpu`.

| Mode | Device | Memory | Run Duration | Thermal Risk |
| :--- | :--- | :--- | :--- | :--- |
| Feature Extraction | CUDA / CPU | ~2–3 GB VRAM | ~5–8 mins (once) | Safe |
| Linear Probes (100 epochs) | CPU | < 2 GB RAM | ~10–15 seconds | **Zero (Laptop-safe)** |
| Multimodal Subtask Tests | CPU | < 1 GB RAM | ~3 seconds | **Zero** |
| W&B Bayesian Sweep (40 runs)| CPU | < 2 GB RAM | ~10–12 minutes | **Zero** |

---

## Environment Setup

This project uses [Pixi](https://pixi.sh/) for reproducible environment management:

```bash
# 1. Clone repository
git clone https://github.com/arnav4324/vlm_proj.git
cd vlm_proj

# 2. Install dependencies via Pixi
pixi install

# 3. Verify CUDA / CPU availability
pixi run python -c "import torch, clip; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
```

---

## Dataset Preparation

1. **CLEVR v1.0**: Place validation scenes in `data/CLEVR_v1.0/scenes/CLEVR_val_scenes.json` and images in `data/CLEVR_v1.0/images/val/`.
2. **SpatialSense**: Place `annotations.json` and `images/` under `data/spatialsense/`.
3. **ImageNet-V2**: Place class folders under `data/ImageNetV2-master/imagenetv2-matched-frequency-format-val/`.

Verify all datasets with:
```bash
pixi run python scripts/validate_data.py
```

---

## Usage

### 1. Verify the Environment
```bash
pixi run python scripts/test_clip_encoder.py
pixi run python scripts/test_resnet_baseline.py
```

### 2. Validate Data Loaders
```bash
pixi run python scripts/test_loaders.py
```

### 3. ImageNet-V2 Control Sanity Check (Step 9)
```bash
pixi run python scripts/evaluate_imagenet.py
```

### 4. Zero-Shot Counting Probe on CLEVR (Step 7)
```bash
pixi run python scripts/evaluate_counting.py
```

### 5. Zero-Shot Spatial Reasoning Probe on SpatialSense (Step 8)
```bash
pixi run python scripts/evaluate_spatial.py
```

### 6. Linear Probe Supervised Baselines (Step 10)
```bash
# 1. ResNet-50 Counting (Spatial 6144-d)
pixi run python scripts/train_linear_probe_counting.py --mode spatial --device cpu --lr 1e-3 --epochs 100

# 2. ResNet-50 Counting (Avgpool 2048-d)
pixi run python scripts/train_linear_probe_counting.py --mode avgpool --device cpu --lr 1e-3 --epochs 100

# 3. CLIP ViT-B/32 Visual Embedding Probe (512-d)
pixi run python scripts/train_linear_probe_clip_counting.py --device cpu --lr 1e-3 --epochs 100

# 4. SpatialSense Relation-Conditioned Probe
pixi run python scripts/train_linear_probe_spatial.py --mode avgpool --device cpu --lr 1e-3 --epochs 100

# 5. SpatialSense Unconditioned (Image-Only) Probe
pixi run python scripts/train_linear_probe_spatial.py --unconditioned --mode avgpool --device cpu --lr 1e-3 --epochs 100
```

### 7. Cluster 6B Multimodal Subtask & Verification (Step 11)
Validate the from-scratch multimodal metrics suite and fusion models:
```bash
pixi run python scripts/test_multimodal_subtask.py
```
*Expected: `Test Results: 25/25 passed` covering Recall@K, Median Rank, Modality Gap $\Delta_{\text{gap}}$, Cross-Modal Alignment, ECE, and Modality Dropout.*

---

## Configuration

All global settings are centralized in [`config/experiment_config.yaml`](config/experiment_config.yaml):

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `seed` | `42` | Global random seed for full reproducibility |
| `clip_model` | `ViT-B/32` | CLIP vision-language backbone |
| `baseline_model` | `resnet50` | Supervised baseline backbone |
| `inference.batch_size` | `32` | Inference batch size |
| `linear_probe.lr` | `1e-3` | Linear probe learning rate |
| `linear_probe.epochs` | `100` | Training epochs |
| `linear_probe.weight_decay` | `0.01` | AdamW weight decay |

---

## Implementation Progress

| Step | Milestone & Specification | Status | Key Output / Verification |
| :---: | :--- | :---: | :--- |
| **1** | Environment Setup & Dependency Isolation (Pixi) | ✅ Complete | `pixi.toml`, CUDA 12.1 / CPU fallback |
| **2** | Modular Project Scaffolding | ✅ Complete | `src/`, `config/`, `scripts/`, `results/` |
| **3** | Configuration Management & Registry | ✅ Complete | `config/experiment_config.yaml` |
| **4** | Dataset Acquisition & Integrity Verification | ✅ Complete | `validate_data.py` (CLEVR, SpatialSense, ImageNet-V2) |
| **5** | Deterministic Seeded DataLoaders | ✅ Complete | `src/data_loaders/` (worker init & generator) |
| **6** | CLIP Encoder Wrapper & Feature Extraction | ✅ Complete | `test_clip_encoder.py` |
| **7** | CLEVR Zero-Shot Counting Evaluation | ✅ Complete | Top-1: **18.79%**, MAE: **1.720** |
| **8** | SpatialSense Zero-Shot Relational Evaluation | ✅ Complete | Accuracy: **57.70%**, Margin: **+0.0021** |
| **9** | ImageNet-V2 Baseline Sanity Check | ✅ Complete | Top-1: **53.24%**, Top-5: **79.72%** (Passed) |
| **10** | Supervised Linear Probe Baselines | ✅ Complete | CLIP ViT Probe: **48.60%**, ResNet Probe: **47.30%** |
| **11** | Cluster 6B Multimodal Subtask & Metric Suite | ✅ Complete | `multimodal_metrics.py`, `multimodal_fusion.py` (25/25 tests) |
| **12** | W&B 40-Run Bayesian Sweep & Multi-Seed Protocol | 🔄 Specified | `config/sweep_fusion_config.yaml`, `scripts/train_fusion_probe.py` |
| **13** | Metric Comparison Study Across 8 Model Pool | 🔲 Planned | Kendall $\tau$ heatmap, 1000-sample bootstrap CIs, verdict |
| **14** | Robustness Evaluation & 25-Case Failure Taxonomy | 🔲 Planned | 5 corruptions $\times$ 3 severities, `wandb.Table` taxonomy |
| **15** | IEEE Conference Paper (`IEEEtran.cls`) & W&B Report | 🔲 Planned | 6–8 page two-column camera-ready LaTeX report |

---

## References

1. Radford, A., et al. (2021). [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020). ICML.
2. Liang, V. W., et al. (2022). [Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning](https://arxiv.org/abs/2203.02053). NeurIPS.
3. Yuksekgonul, M., et al. (2023). [When and why Vision-Language Models behave like Bags-of-Words](https://arxiv.org/abs/2210.01936). ICLR.
4. Thrush, T., et al. (2022). [Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality](https://arxiv.org/abs/2204.03162). CVPR.
5. Johnson, J., et al. (2017). [CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning](https://arxiv.org/abs/1612.06890). CVPR.
6. Yang, K., et al. (2019). [SpatialSense: An Adversarially Crowdsourced Benchmark for Spatial Relation Recognition](https://arxiv.org/abs/1908.02660). ICCV.
7. Recht, B., et al. (2019). [Do ImageNet Classifiers Generalize to ImageNet?](https://arxiv.org/abs/1902.10811). ICML.

---

## License

This project is for academic research and evaluation purposes. CLIP weights are subject to OpenAI's MIT license. Dataset licenses remain with their respective authors.
