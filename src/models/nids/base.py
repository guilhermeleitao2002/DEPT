"""Common interface for the independent NIDS.

  * Kitsune (KitNET + AfterImage), for ``mirai`` and ``ssdp_flood``: per-packet
    RMSE, flagged when ``RMSE > φ``.
  * ML-NIDS (TANTGen's CICFlowMeter flow classifiers), for the CIC attacks:
    per-flow ``P(malicious)``, flagged by the classifier's own decision.

Both score a list of scapy packets into a :class:`BranchScore`. Models keep
no state between calls, so one model scores the adversarial, purified and
clean branches in turn.
"""
from __future__ import annotations

from dataclasses import dataclass


class NIDSUnavailable(RuntimeError):
    """A required NIDS dependency is missing (TANTGen tree, model file, or binary).

    The baseline aborts on it; the reconstructor eval skips that source.
    """


@dataclass
class BranchScore:
    """Per-flow detection summary for one scored packet set (one "branch").

    ``mean_score`` is the mean per-flow score (Kitsune RMSE or ML-NIDS
    ``P(malicious)``); ``detection_rate`` is the fraction of flows flagged.
    """
    n_flows: int
    n_detected: int
    detection_rate: float | None
    mean_score: float | None
    sum_score: float

    @classmethod
    def from_flow_scores(cls, flow_scores: dict, threshold: float | None,
                         predicted: dict | None = None) -> "BranchScore":
        """Build from per-flow scores.

        ``flow_scores`` maps flow-key → continuous anomaly score. Detection is either
        ``score > threshold`` (Kitsune) or, when ``predicted`` is given (ML-NIDS, whose
        classifier owns its own decision boundary), the classifier's 0/1 label.
        """
        import numpy as np

        vals = np.fromiter(flow_scores.values(), dtype=np.float64) if flow_scores \
            else np.zeros(0)
        n = int(vals.size)
        if predicted is not None:
            det = np.fromiter(predicted.values(), dtype=np.float64) if predicted \
                else np.zeros(0)
            n_detected = int(det.sum())
        elif threshold is not None:
            n_detected = int((vals > threshold).sum())
        else:
            n_detected = 0
        return cls(
            n_flows=n,
            n_detected=n_detected,
            detection_rate=(n_detected / n) if n else None,
            mean_score=float(vals.mean()) if n else None,
            sum_score=float(vals.sum()),
        )


class NIDSModel:
    """Base class for an independent NIDS scorer.

    Subclasses implement :meth:`score`. ``kind`` is ``"kitsune"`` or ``"ml_nids"``;
    ``name`` is a human-readable tag (e.g. ``"kitsune:mirai"``); ``threshold`` is the
    NIDS decision threshold (φ for Kitsune, ``None`` for ML-NIDS whose classifier owns
    its own boundary).
    """

    kind: str = "base"
    name: str = "nids"
    threshold: float | None = None

    def score(self, packets: list, keys: list) -> BranchScore:
        """Score an ordered packet list into a per-flow :class:`BranchScore`.

        ``keys[i]`` is the canonical 5-tuple string of ``packets[i]`` (used by
        stateful per-packet detectors such as Kitsune to aggregate per flow; ignored
        by flow-level detectors such as ML-NIDS, which re-derive flows themselves).
        """
        raise NotImplementedError


__all__ = ["NIDSUnavailable", "BranchScore", "NIDSModel"]
