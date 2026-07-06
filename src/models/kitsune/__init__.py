"""Kitsune (KitNET + AfterImage) adapter — an *independent* NIDS evaluator.

ADEPT scores purification efficacy against its own TCN (in-distribution → circular).
This package reuses **Kitsune**, the real anomaly-detection NIDS that TANTGen's attacks
were built to evade, as an INDEPENDENT judge of the Purifier. It is used strictly as an
evaluator — nothing about Kitsune enters the Reconstructor's training or the ensemble,
so the defense stays attack-agnostic.

See :mod:`src.models.kitsune.runner` for the API.
"""
from .runner import (  # noqa: F401
    FEATURE_DIM,
    KitsuneUnavailable,
    available,
    extract_features,
    load_model,
    score,
    train_model,
)

__all__ = [
    "FEATURE_DIM",
    "KitsuneUnavailable",
    "available",
    "extract_features",
    "load_model",
    "score",
    "train_model",
]
