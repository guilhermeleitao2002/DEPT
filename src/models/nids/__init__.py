"""Independent-NIDS evaluation core — shared by the baseline and the reconstructor eval.

DEPT's efficacy is measured by scoring traffic through the *real* NIDS each TANTGen
attack was built to evade — Kitsune (KitNET) for the Kitsune-native attacks, TANTGen's
CICFlowMeter ML-NIDS for the CIC attacks — never DEPT's own detector (which would be
circular). This package provides:

  * :class:`NIDSModel` / :class:`BranchScore` — the scoring interface (``base``).
  * :class:`KitsuneNIDS`, :class:`MLNIDS` — the two NIDS families.
  * :class:`NIDSRegistry` — attack → NIDS, from the shared ``nids`` config block.
  * detection-**restoration** metrics (``metrics``) — how much of the adv→clean signal
    gap the purifier / the full DEPT pipeline recovers.
"""
from .base import BranchScore, NIDSModel, NIDSUnavailable
from .registry import NIDSRegistry

__all__ = ["BranchScore", "NIDSModel", "NIDSUnavailable", "NIDSRegistry"]
