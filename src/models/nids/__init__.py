"""Independent NIDS used to evaluate DEPT (reconstructor eval and baseline).

Each attack is scored by the NIDS it was built to evade: Kitsune for the
Kitsune attacks and TANTGen's CICFlowMeter ML-NIDS for the CIC attacks.
"""
from .base import BranchScore, NIDSModel, NIDSUnavailable
from .registry import NIDSRegistry

__all__ = ["BranchScore", "NIDSModel", "NIDSUnavailable", "NIDSRegistry"]
