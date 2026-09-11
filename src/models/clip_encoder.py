"""CLIP Model loading and feature encoding wrapper.

Provides a stateless, functional abstraction over OpenAI's CLIP dual-encoder,
enforcing L2-normalization, learned temperature scaling, and evaluation mode.
"""
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from PIL import Image
import clip


class CLIPEncoder:
    """Wrapper class for CLIP image and text dual-encoding.

    Args:
        model_name: Name of the CLIP model variant (default: "ViT-B/32").
        device: Target execution device ('cuda' or 'cpu'). Auto-falls back to 'cpu'
                if CUDA is unavailable.
    """

    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda"):
        # Safe device selection with hardware fallback
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.model_name = model_name

        # Load CLIP model and torchvision pre-processing pipeline
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()  # Mandatory: disable stochastic dropout

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the shared latent space (512 for ViT-B/32)."""
        return self.model.visual.output_dim

    @property
    def logit_scale(self) -> torch.Tensor:
        """Return the current learned temperature parameter exp(logit_scale)."""
        return self.model.logit_scale.exp()

    @torch.no_grad()
    def encode_images(self, images: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """Encode a batch of images into unit-normalized latent vectors.

        Args:
            images: Either a preprocessed torch.Tensor of shape (B, 3, 224, 224),
                    or a list of PIL Images (which will be transformed on-the-fly).

        Returns:
            torch.Tensor: L2-normalized feature vectors of shape (B, 512), dtype float32.
        """
        if isinstance(images, list):
            # Batch-preprocess list of PIL images
            images = torch.stack([self.preprocess(img) for img in images])

        images = images.to(self.device)
        image_features = self.model.encode_image(images)

        # Cast to float32 (CLIP sometimes uses float16 on CUDA)
        image_features = image_features.float()

        # L2 normalization to unit hypersphere
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    @torch.no_grad()
    def encode_texts(self, text_list: List[str]) -> torch.Tensor:
        """Encode a list of text strings into unit-normalized latent vectors.

        Args:
            text_list: List of N text prompt strings.

        Returns:
            torch.Tensor: L2-normalized feature vectors of shape (N, 512), dtype float32.
        """
        tokens = clip.tokenize(text_list).to(self.device)
        text_features = self.model.encode_text(tokens)

        # Cast to float32
        text_features = text_features.float()

        # L2 normalization to unit hypersphere
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    @torch.no_grad()
    def compute_similarity(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        use_logit_scale: bool = True,
    ) -> torch.Tensor:
        """Compute similarity matrix / logits between image and text embeddings.

        Args:
            image_features: L2-normalized image embeddings (B, 512).
            text_features:  L2-normalized text embeddings (N, 512).
            use_logit_scale: Whether to multiply by CLIP's learned temperature
                             (approx. 100.0). True gives calibrated classification
                             logits; False gives raw cosine similarities in [-1, 1].

        Returns:
            torch.Tensor: Logits/similarity matrix of shape (B, N).
        """
        # Cosine similarity matrix via matrix multiplication
        similarity = image_features @ text_features.T

        if use_logit_scale:
            logit_scale = self.model.logit_scale.exp().float()
            return logit_scale * similarity

        return similarity

    @torch.no_grad()
    def zero_shot_classify(
        self,
        images: torch.Tensor,
        prompt_templates: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience function: predict class probabilities for an image batch.

        Args:
            images: Tensor of preprocessed images (B, 3, 224, 224).
            prompt_templates: List of N candidate class prompts.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - probs: Softmax probability matrix of shape (B, N).
                - preds: Predicted class index tensor of shape (B,).
        """
        img_feats = self.encode_images(images)
        txt_feats = self.encode_texts(prompt_templates)
        logits = self.compute_similarity(img_feats, txt_feats, use_logit_scale=True)
        probs = logits.softmax(dim=-1)
        preds = probs.argmax(dim=-1)
        return probs, preds

    def __repr__(self) -> str:
        return (
            f"CLIPEncoder(model='{self.model_name}', "
            f"device='{self.device}', "
            f"embed_dim={self.embedding_dim}, "
            f"logit_scale={self.logit_scale.item():.2f})"
        )