"""Evaluation probes and benchmarking utilities for Vision-Language Models."""
from .counting_evaluator import CountingEvaluator
from .spatial_evaluator import SpatialEvaluator, make_spatial_prompts

__all__ = [
    "CountingEvaluator",
    "SpatialEvaluator",
    "make_spatial_prompts",
]
