"""Deployment baseline: the protected NIDS on clean, TANTGen and DEPT-defended
traffic. In the defended stream only the flows flagged by the meta-learner
are purified.
"""
from .evaluate import evaluate

__all__ = ["evaluate"]
