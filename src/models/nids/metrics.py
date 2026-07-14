"""NIDS-agnostic detection-restoration metrics.

Semantics — **detection restoration**. A NIDS flags a malicious flow; TANTGen *evades*
by suppressing that flag (Kitsune RMSE below φ, or the ML-NIDS classifier flipping to
benign); DEPT's purification recovers the original attack's fingerprint, so the NIDS
signal should *rise* from the evaded adversarial floor back toward the clean-original
oracle. These helpers quantify how much of that gap the purifier (or the full DEPT
pipeline) recovers, from :class:`~src.models.nids.base.BranchScore` summaries.

Moved here from ``reconstructor/evaluate.py`` so the reconstructor eval and the baseline
share one definition (the numbers must be comparable across both).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .base import BranchScore


def sub(a, b):
    """``a - b`` with ``None`` propagation (a metric is ``None`` if any input is)."""
    return (a - b) if (a is not None and b is not None) else None


def restoration_ratio(adv, pur, clean):
    """Fraction of the adv→clean signal gap the purifier recovers, in ~[0, 1].

    ``0`` = no restoration (purified ≈ adversarial), ``1`` = full restoration
    (purified ≈ clean original). Robust to a NIDS's absolute score scale (Kitsune's
    RMSE magnitude, or the classifier's probability calibration), unlike the raw
    detection rate. ``None`` when the gap is degenerate (clean ≈ adv — nothing to
    restore, e.g. a pure-timing attack an ML-NIDS never keyed on).
    """
    if adv is None or pur is None or clean is None:
        return None
    gap = clean - adv
    if abs(gap) < 1e-9:
        return None
    return (pur - adv) / gap


def flow_agg(rmse: np.ndarray, keys: list, detection: str = "max") -> dict:
    """Aggregate per-packet scores into a per-flow score (``max`` or ``mean``).

    ``rmse[i]`` and ``keys[i]`` are aligned (the NIDS consumed packets in list order);
    if feature extraction stopped early ``len(rmse) < len(keys)`` — zip the common
    prefix. Used by per-packet NIDS (Kitsune); flow-level NIDS score per flow directly.
    """
    per: dict = defaultdict(list)
    for i in range(min(len(rmse), len(keys))):
        per[keys[i]].append(float(rmse[i]))
    reduce = np.mean if detection == "mean" else np.max
    return {k: float(reduce(v)) for k, v in per.items()}


def finalize_restoration(adv: BranchScore, pur: BranchScore, clean: BranchScore,
                         *, threshold: float | None = None,
                         model_name: str | None = None) -> dict:
    """Restoration metrics from the three branch scores (adv / purified / clean).

    ``purified`` here is whatever occupies the "after DEPT" slot: the reconstruction
    (reconstructor eval) or the full gated-pipeline output (baseline). The primary
    metric is ``score_restoration_ratio`` (scale-robust); detection-rate ``gain`` /
    ``residual_gap`` are secondary and threshold-dependent.
    """
    return {
        # Primary (scale-robust): continuous anomaly score.
        "mean_score_adv": adv.mean_score,
        "mean_score_purified": pur.mean_score,
        "mean_score_clean": clean.mean_score,
        "score_restoration_ratio": restoration_ratio(
            adv.mean_score, pur.mean_score, clean.mean_score),
        # Secondary (threshold-dependent): detection rate.
        "detection_rate_adv": adv.detection_rate,
        "detection_rate_purified": pur.detection_rate,
        "detection_rate_clean": clean.detection_rate,
        "restoration_gain": sub(pur.detection_rate, adv.detection_rate),
        "residual_gap": sub(clean.detection_rate, pur.detection_rate),
        "threshold": threshold,
        "model": model_name,
        "n_flows": pur.n_flows,
    }


def accumulate(acc: dict, branch: str, score: BranchScore) -> None:
    """Fold one branch's counts into a running overall accumulator."""
    a = acc.setdefault(branch, {"sum_score": 0.0, "n_detected": 0, "n_flows": 0})
    a["sum_score"] += score.sum_score
    a["n_detected"] += score.n_detected
    a["n_flows"] += score.n_flows


def finalize_restoration_overall(acc: dict, *, branches=("adv", "purified", "clean")) -> dict:
    """Restoration metrics aggregated across every source from :func:`accumulate`."""
    def rate(br):
        n = acc.get(br, {}).get("n_flows", 0)
        return (acc[br]["n_detected"] / n) if n else None

    def mean(br):
        n = acc.get(br, {}).get("n_flows", 0)
        return (acc[br]["sum_score"] / n) if n else None

    adv_b, pur_b, cln_b = branches
    d_adv, d_pur, d_cln = rate(adv_b), rate(pur_b), rate(cln_b)
    r_adv, r_pur, r_cln = mean(adv_b), mean(pur_b), mean(cln_b)
    return {
        "mean_score_adv": r_adv,
        "mean_score_purified": r_pur,
        "mean_score_clean": r_cln,
        # Score-based ratio (RMSE / proba). Only meaningful within a single NIDS
        # family — mixing Kitsune RMSE with ML-NIDS probabilities averages different
        # scales, so a cross-family overall should read detection_restoration_ratio.
        "score_restoration_ratio": restoration_ratio(r_adv, r_pur, r_cln),
        "detection_rate_adv": d_adv,
        "detection_rate_purified": d_pur,
        "detection_rate_clean": d_cln,
        # Rate-based ratio — 0..1 for every NIDS, so it is safe to aggregate across
        # heterogeneous NIDS families (the cross-family overall headline).
        "detection_restoration_ratio": restoration_ratio(d_adv, d_pur, d_cln),
        "restoration_gain": sub(d_pur, d_adv),
        "residual_gap": sub(d_cln, d_pur),
        "n_flows": acc.get(pur_b, {}).get("n_flows", 0),
    }


__all__ = [
    "sub",
    "restoration_ratio",
    "flow_agg",
    "finalize_restoration",
    "accumulate",
    "finalize_restoration_overall",
]
