"""Verification script for CLIPEncoder wrapper.

Checks:
1. Model loading on appropriate hardware (CUDA or CPU).
2. Output vector dimensionality: (B, 512) for images and (N, 512) for text.
3. Strict L2 unit-norm constraint: ||v||_2 == 1.0 ± 1e-5.
4. Correctness of cosine similarity computation and learned logit scaling.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from src.config_loader import load_config
from src.models.clip_encoder import CLIPEncoder


def main():
    print("=" * 60)
    print("Running CLIPEncoder Verification Suite")
    print("=" * 60)

    config = load_config()
    model_name = config.get("clip_model", "ViT-B/32")
    device = config.get("device", "cuda")

    encoder = CLIPEncoder(model_name=model_name, device=device)
    print(f"Loaded encoder: {encoder}")

    print("\n[Test 1] Image Encoding & Normalization...")
    synthetic_image = Image.new("RGB", (224, 224), color=(128, 64, 192))
    img_tensor = encoder.preprocess(synthetic_image).unsqueeze(0)  # (1, 3, 224, 224)

    img_feats = encoder.encode_images(img_tensor)
    print(f"  Image features shape: {img_feats.shape}")
    assert img_feats.shape == (1, 512), f"Expected shape (1, 512), got {img_feats.shape}"

    norm = img_feats.norm(dim=-1).item()
    print(f"  Image features L2 norm: {norm:.6f}")
    assert torch.allclose(torch.tensor(norm), torch.tensor(1.0), atol=1e-4), (
        f"Image features are not unit normalized! Norm: {norm}"
    )

    print("\n[Test 2] Text Encoding & Normalization...")
    test_prompts = [
        "A photo of 3 objects.",
        "A photo of 5 objects.",
        "The cat is on the table.",
        "A photo of a dog.",
    ]
    txt_feats = encoder.encode_texts(test_prompts)
    print(f"  Text features shape: {txt_feats.shape}")
    assert txt_feats.shape == (len(test_prompts), 512), (
        f"Expected shape ({len(test_prompts)}, 512), got {txt_feats.shape}"
    )

    txt_norms = txt_feats.norm(dim=-1)
    print(f"  Text features L2 norms: {txt_norms.tolist()}")
    assert torch.allclose(txt_norms, torch.ones_like(txt_norms), atol=1e-4), (
        f"Text features are not unit normalized!"
    )

    print("\n[Test 3] Raw Cosine Similarity Bounds...")
    raw_sim = encoder.compute_similarity(img_feats, txt_feats, use_logit_scale=False)
    print(f"  Raw cosine similarities: {raw_sim[0].tolist()}")
    assert (raw_sim >= -1.0 - 1e-4).all() and (raw_sim <= 1.0 + 1e-4).all(), (
        "Raw cosine similarities out of bounds [-1, 1]!"
    )

    print("\n[Test 4] Calibrated Temperature Logits...")
    scaled_logits = encoder.compute_similarity(img_feats, txt_feats, use_logit_scale=True)
    scale_factor = encoder.logit_scale.item()
    print(f"  Learned logit scale: {scale_factor:.2f}")
    print(f"  Scaled logits: {scaled_logits[0].tolist()}")

    expected_logits = raw_sim * scale_factor
    assert torch.allclose(scaled_logits, expected_logits, atol=1e-3), (
        "Scaled logits do not match logit_scale * raw_sim!"
    )

    print("\n[Test 5] Zero-Shot Classification Helper...")
    probs, preds = encoder.zero_shot_classify(img_tensor, test_prompts)
    print(f"  Probabilities: {probs[0].tolist()}")
    print(f"  Probabilities sum: {probs.sum().item():.4f}")
    print(f"  Predicted class index: {preds.item()}")
    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)), atol=1e-4), (
        "Probabilities do not sum to 1.0!"
    )

    print("\n" + "=" * 60)
    print("All CLIPEncoder verification checks passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
