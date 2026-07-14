"""Common interface for the *independent* NIDS DEPT is evaluated against.

DEPT is a defense that sits in front of a protected NIDS. To measure its efficacy
credibly we score traffic through the **actual** NIDS each TANTGen attack was built
to evade — never DEPT's own detector (that would be circular). Two NIDS families are
supported behind one interface:

  * **Kitsune** (KitNET autoencoder + AfterImage) — the anomaly detector for the
    Kitsune-native attacks (``mirai``, ``ssdp_flood``). Per-packet RMSE, flagged when
    ``RMSE > threshold φ``; higher score = more anomalous = more malicious.
  * **ML-NIDS** (TANTGen's CICFlowMeter-based flow classifiers) — the supervised
    detector for the CIC-IDS/CIC-DDoS attacks. Per-flow ``P(malicious)``; flagged when
    the classifier predicts the malicious class.

Both consume a realized packet stream (scapy packets) and return a
:class:`BranchScore` — the per-flow detection summary the metrics module turns into
restoration ratios. A :class:`NIDSModel` is *stateless across calls* at the scoring
level (any statefulness lives inside the feature extractor for a single call), so one
loaded model can score the adversarial / purified / clean branches in turn.
"""
from __future__ import annotations

from dataclasses import dataclass


class NIDSUnavailable(RuntimeError):
    """A required NIDS dependency is missing (TANTGen tree, model file, or binary).

    Raised as a hard error — the caller decides whether to abort (baseline: the NIDS
    *is* the evaluation) or skip-with-reason (reconstructor: the NIDS group is one of
    several metric groups and the run should still produce the others).
    """


@dataclass
class BranchScore:
    """Per-flow detection summary for one scored packet set (one "branch").

    ``mean_score`` is the mean per-flow anomaly score — Kitsune RMSE or ML-NIDS
    ``P(malicious)`` — used for the (scale-robust) restoration *ratio*.
    ``detection_rate`` is the fraction of flows the NIDS flags malicious, used for the
    (threshold-dependent) restoration *gain*. Higher of either = more detected =
    the attack is *less* successful.
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
