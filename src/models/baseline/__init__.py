"""DEPT deployment baseline — the protected-NIDS "before vs after DEPT" evaluation.

Scores three realized full-volume streams — clean / TANTGen-perturbed / DEPT-defended —
through the real NIDS each attack was built to evade, to show how DEPT, deployed in front
of a NIDS, restores detection that TANTGen's perturbations suppressed. The DEPT-defended
stream runs the actual gated pipeline (detector flags → purify only the flagged flows).

  * :mod:`.detector_infer` — per-flow meta-learner (detector) verdicts.
  * :mod:`.pipeline`       — purify + decode overlays + assemble the gated stream.
  * :mod:`.evaluate`       — orchestrate + score through :mod:`src.models.nids` + report.
"""
from .evaluate import evaluate

__all__ = ["evaluate"]
