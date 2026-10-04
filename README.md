# Evaluating Zero-Shot Boundaries and Compositional Failure Modes in Vision-Language Models

> **Research Question:** Do contrastive Vision-Language Models like CLIP behave as "bags of concepts" — recognising *what* is in an image but failing to understand *how many* or *where*?

This project systematically evaluates [OpenAI CLIP (ViT-B/32)](https://arxiv.org/abs/2103.00020) on two compositional reasoning axes — **object counting** and **spatial relation understanding** — and compares zero-shot performance against a supervised **ResNet-50 linear probe** baseline to quantify the gap.

---

## Table of Contents

- [Motivation](#motivation)
- [Key Findings (Expected)](#key-findings-expected)
- [Project Architecture](#project-architecture)
- [Hardware Requirements](#hardware-requirements)
- [Environment Setup](#environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
  - [1. Verify the Environment](#1-verify-the-environment)
  - [2. Validate Data Loaders](#2-validate-data-loaders)
  - [3. ImageNet-V2 Sanity Check (Step 9)](#3-imagenet-v2-sanity-check-step-9)
  - [4. Zero-Shot Counting Probe (Step 7)](#4-zero-shot-counting-probe-step-7)
  - [5. Zero-Shot Spatial Reasoning Probe (Step 8)](#5-zero-shot-spatial-reasoning-probe-step-8)
  - [6. ResNet-50 Linear Probe Baselines (Step 10)](#6-resnet-50-linear-probe-baselines-step-10)
- [Configuration](#configuration)
- [Results & Outputs](#results--outputs)
- [Implementation Progress](#implementation-progress)
- [References](#references)
- [License](#license)

---

## Motivation

CLIP learns a joint image-text embedding space via contrastive pre-training on 400M image-caption pairs. While this yields impressive zero-shot classification (∼63% Top-1 on ImageNet), the contrastive objective never explicitly incentivises:

- **Cardinality discrimination** — distinguishing "3 objects" from "5 objects"
- **Syntactic sensitivity** — differentiating "the cat is on the table" from "the table is on the cat"

This project provides empirical evidence for these failure modes using controlled benchmarks and compares against supervised baselines that *do* have access to task-specific labels.

---

## Key Findings (Expected)

| Task | CLIP ViT-B/32 (Zero-Shot) | ResNet-50 Linear Probe | Gap |
|------|---------------------------|------------------------|-----|
| CLEVR Counting (Overall) | ~20–35% | ~85–95% | ~50–60% |
| Spatial Reasoning (Binary) | ~50–55% (≈ chance) | ~65–75% | ~15–20% |
| ImageNet-V2 Top-1 (Sanity) | ~60–63% | — | — |

- **Counting**: Accuracy degrades sharply for counts > 3. CLIP exhibits a strong bias toward predicting mid-range counts regardless of ground truth.
- **Spatial Reasoning**: Near-chance accuracy confirms CLIP treats prompts as bags of tokens — both "cat on table" and "table on cat" produce nearly identical embeddings.

---

## Project Architecture

```
vlm_proj/
├── config/
│   └── experiment_config.yaml       # Centralised hyperparameters & paths
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
│   │   └── resnet_baseline.py       # ResNetFeatureExtractor + LinearProbeTrainer + RelationProbeTrainer
│   └── evaluators/
│       ├── __init__.py
│       ├── counting_evaluator.py    # CountingEvaluator (zero-shot counting)
│       ├── spatial_evaluator.py     # SpatialEvaluator (contrastive spatial)
│       └── imagenet_evaluator.py    # ImageNetV2Evaluator (sanity check)
├── scripts/
│   ├── validate_data.py             # Step 4: Dataset integrity validation
│   ├── test_clip_encoder.py         # CLIPEncoder verification suite
│   ├── test_loaders.py              # DataLoader smoke test
│   ├── test_resnet_baseline.py      # ResNet feature extraction & linear probe test
│   ├── evaluate_counting.py         # Step 7: CLEVR counting evaluation
│   ├── evaluate_spatial.py          # Step 8: SpatialSense spatial reasoning evaluation
│   ├── evaluate_imagenet.py         # Step 9: ImageNet-V2 sanity check
│   ├── train_linear_probe_counting.py      # Step 10: Counting linear probe (spatial / avgpool)
│   ├── train_linear_probe_spatial.py       # Step 10: Spatial linear probe (conditioned / unconditioned)
│   └── train_linear_probe_clip_counting.py # Step 10: CLIP visual embedding linear probe
├── notebooks/                       # Exploratory analysis
├── results/                         # Logs, figures, checkpoints, cache (git-ignored)
├── pixi.toml                        # Pixi package manager config
├── pixi.lock
├── implementation_plan.md           # Full 15-step research plan
└── README.md                        # ← You are here
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA with 4 GB VRAM | NVIDIA with 8+ GB VRAM |
| System RAM | 16 GB | 32 GB |
| Disk Space | ~25 GB (datasets + cache) | ~40 GB |
| CUDA Driver | ≥ 525.x (CUDA 12.x) | Latest stable |

---

## Environment Setup

This project uses [Pixi](https://pixi.sh/) for environment management with CUDA 12.1 PyTorch wheels.

```bash
# 1. Install Pixi (if not already installed)
# See: https://pixi.sh/latest/#installation

# 2. Clone the repository
git clone <repo-url> vlm_proj
cd vlm_proj

# 3. Install all dependencies
pixi install

# 4. Activate the environment
pixi shell

# 5. Verify the installation
pixi run test-cuda
```

**Expected output:**
```
PyTorch: 2.x.x | CUDA Available: True | Device: NVIDIA GeForce ... | CLIP models: ['RN50', 'RN101', 'ViT-B/32']
```

### Manual Setup (without Pixi)

```bash
conda create -n vlm_eval python=3.10 -y
conda activate vlm_eval

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/openai/CLIP.git
pip install transformers datasets pillow matplotlib seaborn scikit-learn tqdm pandas numpy scipy pyyaml statsmodels requests wandb
```

---

## Dataset Preparation

### 1. CLEVR (Counting Probe — Step 4)

Download [CLEVR v1.0](https://cs.stanford.edu/people/jcjohns/clevr/) and extract:

```
data/CLEVR_v1.0/
├── images/val/          # Validation images
└── scenes/
    └── CLEVR_val_scenes.json   # Ground-truth scene metadata
```

The loader auto-parses object counts from the scene JSON — no manual preprocessing needed.

### 2. SpatialSense (Spatial Reasoning Probe — Step 4)

Clone [SpatialSense](https://github.com/princeton-vl/SpatialSense) and set up:

```
data/spatialsense/
├── images/              # flickr/ and nyu/ subdirectories
└── annotations.json     # Spatial relation triplets
```

The loader supports both the official Princeton format and flat annotation lists.

### 3. ImageNet-V2 (Baseline Sanity Check — Step 4)

Download [ImageNet-V2 matched-frequency](https://github.com/modestyachts/ImageNetV2):

```
data/ImageNetV2-master/imagenetv2-matched-frequency-format-val/
├── 0/     # 10 images per class
├── 1/
└── ...    # Through 999/
```

---

## Usage

All scripts are run from the project root directory. The pipeline follows the implementation plan step order.

### 1. Verify the Environment

```bash
# Test CLIP encoder (shapes, normalization, similarity computation)
python scripts/test_clip_encoder.py

# Test ResNet-50 feature extraction & linear probe
python scripts/test_resnet_baseline.py
```

### 2. Validate Data Loaders

```bash
# Smoke-test all available dataset loaders
python scripts/test_loaders.py
```

Verifies: image tensor shapes `(B, 3, 224, 224)`, prompt tokenization `(N, 77)`, and correct annotation parsing.

### 3. ImageNet-V2 Sanity Check (Step 9)

```bash
python scripts/evaluate_imagenet.py
```

**Purpose**: Reproduces CLIP's known ~60–63% Top-1 accuracy on ImageNet-V2. If this fails (outside 55–70%), the pipeline has a bug. **Do not proceed until this passes.**

**Output**: `results/logs/imagenet_v2_results.json`

### 4. Zero-Shot Counting Probe (Step 7)

```bash
python scripts/evaluate_counting.py
```

Generates candidate prompts `"A photo of {N} objects."` for N ∈ [1, 10], encodes them once, then matches every CLEVR image against all 10 prompts via cosine similarity.

**Metrics**: Overall accuracy, per-count accuracy, MAE, confusion matrix, confidence statistics.

**Outputs**:
- `results/logs/counting_results.json`
- `results/logs/counting_per_count.csv`
- `results/logs/counting_confusion_matrix.csv`

### 5. Zero-Shot Spatial Reasoning Probe (Step 8)

```bash
python scripts/evaluate_spatial.py
```

For each image, generates a contrastive prompt pair:
- Correct: `"The {subject} is {relation} the {object}."`
- Swapped: `"The {object} is {relation} the {subject}."`

Binary forced-choice: CLIP must assign higher similarity to the correct prompt. Random baseline = 50%.

**Metrics**: Overall binary accuracy, per-relation accuracy, margin distribution, failure case logging.

**Outputs**:
- `results/logs/spatial_results.json`
- `results/logs/spatial_per_relation.csv`

### 6. Linear Probe Supervised Baselines (Step 10)

```bash
# 1. ResNet-50 Counting Probe (Spatial Pooling: 6144-d)
python scripts/train_linear_probe_counting.py --mode spatial --lr 1e-3 --epochs 100

# 2. ResNet-50 Counting Probe (Average Pooling: 2048-d)
python scripts/train_linear_probe_counting.py --mode avgpool --lr 1e-3 --epochs 100

# 3. CLIP ViT-B/32 Visual Embedding Counting Probe (512-d)
python scripts/train_linear_probe_clip_counting.py --lr 1e-3 --epochs 100

# 4. SpatialSense Relation-Conditioned Linear Probe
python scripts/train_linear_probe_spatial.py --mode avgpool --lr 1e-3 --epochs 100

# 5. SpatialSense Unconditioned (Image-Only) Baseline Probe
python scripts/train_linear_probe_spatial.py --unconditioned --mode avgpool --lr 1e-3 --epochs 100
```

Extracts frozen features, caches them to disk (`results/cache/`), and trains probe classifiers using AdamW with cosine annealing and cross-entropy loss.

**Features**: Configurable CLI args (`--mode`, `--lr`, `--epochs`, `--seed`, `--no-wandb`), deterministic seeding, and checkpointing (`best_model.pt` + `latest_checkpoint.pt`).

**Outputs**:
- `results/logs/linear_probe_counting_spatial.json`
- `results/logs/linear_probe_counting_avgpool.json`
- `results/logs/linear_probe_clip_counting.json`
- `results/logs/linear_probe_spatial_conditioned.json`
- `results/logs/linear_probe_spatial_unconditioned.json`
- `results/checkpoints/`

---

## Configuration

All hyperparameters are centralised in [`config/experiment_config.yaml`](config/experiment_config.yaml):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `seed` | `42` | Global random seed for full reproducibility |
| `clip_model` | `ViT-B/32` | CLIP backbone variant |
| `baseline_model` | `resnet50` | Supervised baseline |
| `inference.batch_size` | 32 | Evaluation batch size |
| `linear_probe.lr` | 1e-3 | Probe learning rate |
| `linear_probe.epochs` | 100 | Training epochs |
| `linear_probe.weight_decay` | 0.01 | AdamW weight decay |
| `prompts.counting.template` | `"A photo of {N} objects."` | Counting prompt template |
| `prompts.spatial.relations` | `[above, behind, in, ...]` | Evaluated spatial relations |

---

## Results & Outputs

All results are saved to `results/` (git-ignored):

```
results/
├── logs/
│   ├── counting_results.json                 # Step 7 full metrics
│   ├── counting_per_count.csv                # Per-count accuracy breakdown
│   ├── counting_confusion_matrix.csv         # 10×10 confusion matrix
│   ├── spatial_results.json                  # Step 8 full metrics & margins
│   ├── spatial_per_relation.csv              # Per-relation breakdown
│   ├── imagenet_v2_results.json              # Step 9 sanity check
│   ├── linear_probe_counting_spatial.json    # Step 10 spatial counting probe
│   ├── linear_probe_clip_counting.json       # Step 10 CLIP vision probe
│   └── linear_probe_spatial_conditioned.json # Step 10 conditioned spatial probe
├── cache/
│   ├── clevr_resnet_spatial_features.pt      # Cached ResNet spatial features
│   ├── clevr_clip_vit_features.pt            # Cached CLIP ViT features
│   └── spatial_resnet_avgpool_features.pt    # Cached SpatialSense features
├── checkpoints/
│   ├── counting/                             # Linear probe counting checkpoints
│   ├── clip_counting/                        # CLIP counting probe checkpoints
│   └── spatial/                              # Spatial probe checkpoints
└── figures/                                  # Generated visualisations (Step 12)
```

---

## Implementation Progress

| Step | Description | Status |
|------|-------------|--------|
| 1 | Environment Setup & Dependency Installation | ✅ Complete |
| 2 | Project Structure & Codebase Scaffolding | ✅ Complete |
| 3 | Configuration Management & Hyperparameter Registry | ✅ Complete |
| 4 | Dataset Acquisition & Preparation | ✅ Complete (`validate_data.py`) |
| 5 | Dataset Loaders & Pre-processing Pipeline | ✅ Complete (deterministic seeding) |
| 6 | CLIP Model Loading & Encoding Wrapper | ✅ Complete (`clip_encoder.py`) |
| 7 | Zero-Shot Counting Probe (CLEVR) | ✅ Complete (`evaluate_counting.py`) |
| 8 | Zero-Shot Spatial Reasoning Probe (SpatialSense) | ✅ Complete (`evaluate_spatial.py`) |
| 9 | ImageNet-V2 Baseline Sanity Check | ✅ Complete (`evaluate_imagenet.py`) |
| 10 | ResNet-50 Supervised Linear Probe Baseline | ✅ Complete (CLI suite + conditioned probe) |
| 11 | Metrics Computation & Statistical Analysis | 🔲 Pending |
| 12 | Visualisation & Figure Generation | 🔲 Pending |
| 13 | Ablation Studies & Extended Analysis | 🔲 Pending |
| 14 | Results Aggregation & Report Writing | 🔲 Pending |
| 15 | Reproducibility Packaging & Code Release | 🔲 Pending |

---

## References

1. Radford, A., et al. (2021). [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020). ICML.
2. Johnson, J., et al. (2017). [CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning](https://arxiv.org/abs/1612.06890). CVPR.
3. Yang, K., et al. (2019). [SpatialSense: An Adversarially Crowdsourced Benchmark for Spatial Relation Recognition](https://arxiv.org/abs/1908.02660).
4. Recht, B., et al. (2019). [Do ImageNet Classifiers Generalize to ImageNet?](https://arxiv.org/abs/1902.10811). ICML.
5. Thrush, T., et al. (2022). [Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality](https://arxiv.org/abs/2204.03162). CVPR.

---

## License

This project is for academic research purposes. CLIP weights are subject to [OpenAI's license](https://github.com/openai/CLIP/blob/main/LICENSE). Dataset licenses are governed by their respective providers.
