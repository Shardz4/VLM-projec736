"""Step 14: Image Corruption Transforms for VLM Robustness Stress-Testing.

Implements Assignment Section 7 (Robustness & Failure Analysis):
5 Corruption Types x 3 Severity Levels (Mild, Moderate, Severe):
1. Gaussian Noise (std in {0.05, 0.15, 0.30})
2. Gaussian Blur (kernel in {3, 7, 13}, sigma in {1.0, 2.0, 3.5})
3. JPEG Compression (quality in {50, 30, 10})
4. Brightness Shift (delta in {0.2, 0.4, 0.6})
5. Cutout Occlusion (box size in {16, 32, 64} pixels)
"""

from __future__ import annotations
import io
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torchvision.transforms.functional as TF


CORRUPTION_TYPES = [
    "gaussian_noise",
    "gaussian_blur",
    "jpeg_compression",
    "brightness_shift",
    "cutout_occlusion",
]

SEVERITY_LEVELS = [1, 2, 3]  # 1: Mild, 2: Moderate, 3: Severe


# ── Transform Implementations ────────────────────────────────────────────────

def apply_gaussian_noise(img: Image.Image, severity: int) -> Image.Image:
    """Add zero-mean Gaussian noise with calibrated standard deviations."""
    std_map = {1: 0.05, 2: 0.15, 3: 0.30}
    std = std_map.get(severity, 0.15)

    arr = np.array(img).astype(np.float32) / 255.0
    noise = np.random.normal(loc=0.0, scale=std, size=arr.shape)
    noisy_arr = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((noisy_arr * 255.0).astype(np.uint8))


def apply_gaussian_blur(img: Image.Image, severity: int) -> Image.Image:
    """Apply Gaussian spatial blur with calibrated radius."""
    radius_map = {1: 1.0, 2: 2.0, 3: 3.5}
    radius = radius_map.get(severity, 2.0)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_jpeg_compression(img: Image.Image, severity: int) -> Image.Image:
    """Apply lossy JPEG compression and decompression artifacts."""
    quality_map = {1: 50, 2: 30, 3: 10}
    quality = quality_map.get(severity, 30)

    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).copy()


def apply_brightness_shift(img: Image.Image, severity: int) -> Image.Image:
    """Apply severe lighting shift (dimming or overexposure)."""
    # Level 1: factor 0.7 (moderate dim), Level 2: factor 0.4 (dark), Level 3: factor 0.15 (severe underexposure)
    factor_map = {1: 0.70, 2: 0.40, 3: 0.15}
    factor = factor_map.get(severity, 0.40)
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def apply_cutout_occlusion(img: Image.Image, severity: int) -> Image.Image:
    """Apply central/quadrant rectangular black cutout occlusion mask."""
    size_map = {1: 16, 2: 32, 3: 64}
    box_size = size_map.get(severity, 32)

    img_copy = img.copy()
    w, h = img_copy.size
    # Center cutout with random slight jitter
    cx, cy = w // 2, h // 2
    half = box_size // 2

    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(w, cx + half)
    y1 = min(h, cy + half)

    mask = Image.new("RGB", (x1 - x0, y1 - y0), color=(0, 0, 0))
    img_copy.paste(mask, (x0, y0))
    return img_copy


TRANSFORM_REGISTRY: Dict[str, Callable[[Image.Image, int], Image.Image]] = {
    "gaussian_noise": apply_gaussian_noise,
    "gaussian_blur": apply_gaussian_blur,
    "jpeg_compression": apply_jpeg_compression,
    "brightness_shift": apply_brightness_shift,
    "cutout_occlusion": apply_cutout_occlusion,
}


def corrupt_image(img: Image.Image, corruption: str, severity: int) -> Image.Image:
    """Dispatch image corruption by name and severity level (1, 2, or 3)."""
    if corruption not in TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown corruption '{corruption}'. Choose from {list(TRANSFORM_REGISTRY.keys())}")
    if severity not in (1, 2, 3):
        raise ValueError(f"Severity must be in {{1, 2, 3}}, got {severity}")
    return TRANSFORM_REGISTRY[corruption](img, severity)
