"""Cluster 6B Multimodal Fusion Architectures.

Three fusion strategies over pre-extracted visual and textual feature vectors:

  1. EarlyFusionMLP       - concatenate [v; t] -> 3-layer MLP with BatchNorm.
  2. LateFusionGated      - independent classification heads + learned scalar gate.
  3. CrossAttentionFusion - text queries attend to visual keys/values via MHA.
  4. ModalityDropout      - randomly zero out an entire modality during training.

All architectures:
  - Operate on CPU by default (safe on 4 GB VRAM laptops).
  - Accept pre-extracted .pt feature tensors.
  - Expose a unified forward(v, t) -> logits API.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityDropout(nn.Module):
    """Randomly zero out an entire modality during training.

    With prob p_drop/2 the image embedding is zeroed, with prob p_drop/2 the
    text embedding is zeroed.  At least one modality is always preserved.

    Args:
        p_drop: Total probability of a dropout event per modality pair (default: 0.2).
    """

    def __init__(self, p_drop: float = 0.2):
        super().__init__()
        if not 0.0 <= p_drop < 1.0:
            raise ValueError(f"p_drop must be in [0, 1). Got {p_drop}")
        self.p_drop = p_drop

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.p_drop == 0.0:
            return v, t
        p_each = self.p_drop / 2.0
        drop_v = torch.rand(1).item() < p_each
        drop_t = torch.rand(1).item() < p_each
        if drop_v and drop_t:
            drop_t = False  # never zero both
        v_out = torch.zeros_like(v) if drop_v else v
        t_out = torch.zeros_like(t) if drop_t else t
        return v_out, t_out

    def extra_repr(self) -> str:
        return f"p_drop={self.p_drop}"


class EarlyFusionMLP(nn.Module):
    """Early Fusion: [v; t] -> 3-layer MLP with BatchNorm and Dropout.

    Args:
        visual_dim:   Dim of image feature vectors.
        text_dim:     Dim of text feature vectors.
        num_classes:  Number of output classes.
        hidden_dim:   Width of hidden layers (default: 256).
        dropout:      Activation dropout probability (default: 0.3).
        p_modal_drop: Modality dropout probability (default: 0.2).
    """

    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        p_modal_drop: float = 0.2,
    ):
        super().__init__()
        in_dim = visual_dim + text_dim
        self.modal_dropout = ModalityDropout(p_drop=p_modal_drop)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        v, t = self.modal_dropout(v, t)
        x = torch.cat([v, t], dim=-1)
        return self.net(x)


class LateFusionGated(nn.Module):
    """Late Fusion: independent classification heads gated by a learned scalar.

    alpha = sigmoid(w_g * [v; t])
    output = alpha * head_v(v) + (1 - alpha) * head_t(t)

    Args:
        visual_dim:   Dim of image feature vectors.
        text_dim:     Dim of text feature vectors.
        num_classes:  Number of output classes.
        p_modal_drop: Modality dropout probability (default: 0.2).
    """

    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        num_classes: int,
        p_modal_drop: float = 0.2,
    ):
        super().__init__()
        self.modal_dropout = ModalityDropout(p_drop=p_modal_drop)
        self.head_v = nn.Linear(visual_dim, num_classes)
        self.head_t = nn.Linear(text_dim, num_classes)
        self.gate = nn.Linear(visual_dim + text_dim, 1)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        v, t = self.modal_dropout(v, t)
        logits_v = self.head_v(v)
        logits_t = self.head_t(t)
        alpha = torch.sigmoid(self.gate(torch.cat([v, t], dim=-1)))
        return alpha * logits_v + (1.0 - alpha) * logits_t


class CrossAttentionFusion(nn.Module):
    """Cross-Attention Fusion: textual queries attend to visual keys/values.

    Architecture:
        v -> proj_v  (B, 1, d_model)
        t -> proj_t  (B, 1, d_model)
        h = LayerNorm(t + MHA(Q=proj_t, K=proj_v, V=proj_v))
        h = LayerNorm(h + FFN(h))
        logits = head(h.squeeze(1))

    Args:
        visual_dim:   Dim of image feature vectors.
        text_dim:     Dim of text feature vectors.
        num_classes:  Number of output classes.
        d_model:      Shared projection dim for attention (default: 256).
        num_heads:    Number of parallel attention heads (default: 4).
        ffn_dim:      Hidden size of FFN block (default: 512).
        dropout:      Attention/FFN dropout (default: 0.1).
        p_modal_drop: Modality dropout probability (default: 0.2).
    """

    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        num_classes: int,
        d_model: int = 256,
        num_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        p_modal_drop: float = 0.2,
    ):
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )
        self.modal_dropout = ModalityDropout(p_drop=p_modal_drop)
        self.proj_v = nn.Linear(visual_dim, d_model)
        self.proj_t = nn.Linear(text_dim, d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(p=dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(
        self,
        v: torch.Tensor,
        t: torch.Tensor,
        return_attention: bool = False,
    ):
        v, t = self.modal_dropout(v, t)
        v_proj = self.proj_v(v).unsqueeze(1)   # (B, 1, d_model)
        t_proj = self.proj_t(t).unsqueeze(1)   # (B, 1, d_model)
        attn_out, attn_weights = self.cross_attn(
            query=t_proj, key=v_proj, value=v_proj
        )
        h = self.norm1(t_proj + attn_out)
        h = self.norm2(h + self.ffn(h))
        logits = self.head(h.squeeze(1))
        if return_attention:
            return logits, attn_weights
        return logits


def build_fusion_model(
    strategy: str,
    visual_dim: int,
    text_dim: int,
    num_classes: int,
    hidden_dim: int = 256,
    num_heads: int = 4,
    p_modal_drop: float = 0.2,
    dropout: float = 0.1,
) -> nn.Module:
    """Build a fusion model by strategy name.

    Args:
        strategy: One of "early", "late", "cross_attention".
        visual_dim, text_dim, num_classes: Feature and class dimensions.
        hidden_dim: Hidden layer width / d_model.
        num_heads:  Attention heads (cross_attention only).
        p_modal_drop: Modality dropout probability.
        dropout:    Internal activation dropout.

    Returns:
        Instantiated nn.Module.
    """
    strategy = strategy.lower().strip()
    if strategy == "early":
        return EarlyFusionMLP(visual_dim, text_dim, num_classes, hidden_dim, dropout, p_modal_drop)
    elif strategy == "late":
        return LateFusionGated(visual_dim, text_dim, num_classes, p_modal_drop)
    elif strategy in ("cross_attention", "cross"):
        return CrossAttentionFusion(
            visual_dim, text_dim, num_classes,
            d_model=hidden_dim, num_heads=num_heads,
            dropout=dropout, p_modal_drop=p_modal_drop,
        )
    else:
        raise ValueError(
            f"Unknown fusion strategy '{strategy}'. "
            "Choose from: 'early', 'late', 'cross_attention'."
        )
