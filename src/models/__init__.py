"""Models package exposing the CLIP dual-encoder wrapper and ResNet baseline."""
from .clip_encoder import CLIPEncoder
from .resnet_baseline import (
    ResNetFeatureExtractor,
    LinearProbeTrainer,
    RelationConditionedLinearProbe,
    RelationProbeTrainer,
)

__all__ = [
    "CLIPEncoder",
    "ResNetFeatureExtractor",
    "LinearProbeTrainer",
    "RelationConditionedLinearProbe",
    "RelationProbeTrainer",
]
