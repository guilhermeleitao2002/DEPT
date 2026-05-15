"""Streaming-native evaluation metrics for ADEPT's RF/TCN/ensemble.

Two metrics, computed from a per-flow stream of (packet_idx, p_adversarial)
predictions:

  1. ``AUROC@t`` for ``t ∈ T_GRID``  — Flow-level AUROC if the verdict had
     to be made at packet ``t``. For each flow we take its prediction at
     the largest packet index ≤ ``t`` (or skip the flow if none — flag
     reported alongside the score).

  2. ``Earliest-Detection-Packet (EDP)`` for adversarial flows  — Median +
     IQR of the packet index at which ``p_adversarial`` first crosses the
     ``edp_threshold``. Flows that never cross are counted as
     "not detected" and excluded from the median, but their fraction is
     reported as ``detection_rate``.

Both metrics are computed overall and per-adversarial-source under the
same source-vs-balanced-benign / source-vs-all-benign / source-vs-same-
base-benign slicings used by the existing per-source evaluator
(``src/models/{rf,tcn}/evaluate.py``).

The function is model-agnostic: ``per_flow_probs`` is a list of 1D arrays
with one prediction per emission point, plus ``per_flow_packet_idx``
giving the packet index (1-based) at each emission. TCN emits a dense
stream (one prediction per packet); RF emits a sparse stream (one per
``rf_checkpoint_stride`` packets). The same function handles both.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


ADV_MODE_SUFFIXES = ("_adv_both", "_adv_iat", "_adv_size")

DEFAULT_T_GRID = (1, 2, 5, 10, 20, 50, 100)  # plus "max" appended at runtime.


def _base_source(adv_source: str) -> str | None:
    for suffix in ADV_MODE_SUFFIXES:
        if adv_source.endswith(suffix):
            return adv_source[: -len(suffix)]
    return None


def _prob_at_or_before(
    packet_idx: np.ndarray, probs: np.ndarray, t: int,
) -> tuple[float | None, int | None]:
    """Return ``(prob, packet_idx)`` for the latest prediction with idx ≤ t.

    Returns ``(None, None)`` if no prediction exists at or before packet t.
    """
    if len(packet_idx) == 0:
        return None, None
    mask = packet_idx <= t
    if not mask.any():
        return None, None
    j = int(np.argmax(packet_idx[mask]))
    valid = np.where(mask)[0]
    chosen = int(valid[j])
    return float(probs[chosen]), int(packet_idx[chosen])


def _auroc_at_t(
    per_flow_probs: Sequence[np.ndarray],
    per_flow_packet_idx: Sequence[np.ndarray],
    labels: np.ndarray,
    t: int | str,
    per_flow_lengths: np.ndarray | None = None,
) -> dict:
    """Flow-level AUROC at packet ``t`` (or "max" — use each flow's last)."""
    probs_at_t: list[float] = []
    labs_at_t: list[int] = []
    n_excluded = 0
    for i, (pkt_idx, pr) in enumerate(zip(per_flow_packet_idx, per_flow_probs)):
        if t == "max":
            if len(pr) == 0:
                n_excluded += 1
                continue
            probs_at_t.append(float(pr[-1]))
            labs_at_t.append(int(labels[i]))
            continue
        t_int = int(t)
        p, _ = _prob_at_or_before(np.asarray(pkt_idx), np.asarray(pr), t_int)
        if p is None:
            n_excluded += 1
            continue
        probs_at_t.append(p)
        labs_at_t.append(int(labels[i]))
    probs_arr = np.asarray(probs_at_t, dtype=np.float64)
    labs_arr = np.asarray(labs_at_t, dtype=np.int64)
    out = {
        "n_used": int(len(probs_arr)),
        "n_excluded": int(n_excluded),
        "n_benign": int((labs_arr == 0).sum()),
        "n_adversarial": int((labs_arr == 1).sum()),
    }
    if out["n_benign"] > 0 and out["n_adversarial"] > 0:
        out["auroc"] = float(roc_auc_score(labs_arr, probs_arr))
    else:
        out["auroc"] = None
    return out


def _edp_summary(
    per_flow_probs: Sequence[np.ndarray],
    per_flow_packet_idx: Sequence[np.ndarray],
    labels: np.ndarray,
    threshold: float,
) -> dict:
    """Earliest-Detection-Packet summary over the adversarial subset."""
    edps: list[int] = []
    n_adv = 0
    n_detected = 0
    for i, (pkt_idx, pr) in enumerate(zip(per_flow_packet_idx, per_flow_probs)):
        if int(labels[i]) != 1:
            continue
        n_adv += 1
        pkt_idx = np.asarray(pkt_idx)
        pr = np.asarray(pr)
        crossed = pr >= threshold
        if not crossed.any():
            continue
        first_pkt = int(pkt_idx[np.argmax(crossed)])
        edps.append(first_pkt)
        n_detected += 1
    out = {
        "n_adversarial": int(n_adv),
        "n_detected": int(n_detected),
        "detection_rate": float(n_detected / n_adv) if n_adv else 0.0,
        "threshold": float(threshold),
    }
    if edps:
        edp_arr = np.asarray(edps, dtype=np.float64)
        out["edp_median"] = float(np.median(edp_arr))
        out["edp_iqr"] = [
            float(np.quantile(edp_arr, 0.25)),
            float(np.quantile(edp_arr, 0.75)),
        ]
        out["edp_min"] = int(edp_arr.min())
        out["edp_max"] = int(edp_arr.max())
    else:
        out["edp_median"] = None
        out["edp_iqr"] = None
        out["edp_min"] = None
        out["edp_max"] = None
    return out


@dataclass
class StreamingPredictions:
    """Per-flow streaming predictions in the canonical shape this module expects.

    ``probs[i]`` is a 1D array of P(adversarial) emissions for flow ``i``,
    aligned with ``packet_idx[i]`` (1-based packet index per emission).
    """
    probs: list[np.ndarray]
    packet_idx: list[np.ndarray]
    labels: np.ndarray
    sources: np.ndarray
    lengths: np.ndarray


def streaming_metrics(
    preds: StreamingPredictions,
    t_grid: Iterable[int] = DEFAULT_T_GRID,
    edp_threshold: float = 0.5,
    balance_k: int = 10,
    min_support: int = 50,
    seed: int = 42,
) -> dict:
    """Compute the full streaming metrics dictionary (overall + per source)."""
    t_grid_list: list = sorted({int(t) for t in t_grid}) + ["max"]

    overall: dict = {
        "n_flows": int(len(preds.labels)),
        "n_benign": int((preds.labels == 0).sum()),
        "n_adversarial": int((preds.labels == 1).sum()),
        "auroc_at": {},
        "edp": _edp_summary(preds.probs, preds.packet_idx,
                            preds.labels, edp_threshold),
    }
    for t in t_grid_list:
        overall["auroc_at"][str(t)] = _auroc_at_t(
            preds.probs, preds.packet_idx, preds.labels, t,
            per_flow_lengths=preds.lengths,
        )

    # Per-adversarial-source slicing.
    rng = np.random.default_rng(seed)
    benign_all_idx = np.where(preds.labels == 0)[0]
    adv_sources = sorted(set(preds.sources[preds.labels == 1].tolist()))
    per_source: dict = {}
    for src in adv_sources:
        adv_mask = (preds.sources == src) & (preds.labels == 1)
        n_adv = int(adv_mask.sum())
        base = _base_source(src)

        sel_all = adv_mask | (preds.labels == 0)
        if base is not None:
            sel_same = adv_mask | ((preds.labels == 0) & (preds.sources == base))
        else:
            sel_same = adv_mask

        target_bn = min(balance_k * n_adv, len(benign_all_idx))
        if n_adv > 0 and target_bn > 0:
            bn_pick = rng.choice(benign_all_idx, size=target_bn, replace=False)
            bal_mask = np.zeros_like(adv_mask)
            bal_mask[np.where(adv_mask)[0]] = True
            bal_mask[bn_pick] = True
        else:
            bal_mask = adv_mask

        def _slice(mask: np.ndarray) -> dict:
            idx = np.where(mask)[0]
            sub_probs = [preds.probs[i] for i in idx]
            sub_pkt = [preds.packet_idx[i] for i in idx]
            sub_labels = preds.labels[idx]
            sub_lengths = preds.lengths[idx]
            out: dict = {
                "n_flows": int(len(idx)),
                "n_benign": int((sub_labels == 0).sum()),
                "n_adversarial": int((sub_labels == 1).sum()),
                "auroc_at": {
                    str(t): _auroc_at_t(
                        sub_probs, sub_pkt, sub_labels, t,
                        per_flow_lengths=sub_lengths,
                    )
                    for t in t_grid_list
                },
                "edp": _edp_summary(
                    sub_probs, sub_pkt, sub_labels, edp_threshold,
                ),
            }
            return out

        per_source[src] = {
            "low_support": n_adv < min_support,
            "base_source": base,
            "n_adversarial_flows": n_adv,
            "vs_all_benign": _slice(sel_all),
            "vs_balanced_benign": _slice(bal_mask),
            "vs_same_base_benign": _slice(sel_same),
        }

    return {
        "overall": overall,
        "per_adversarial_source": per_source,
        "config": {
            "t_grid": t_grid_list,
            "edp_threshold": float(edp_threshold),
            "balance_k": int(balance_k),
            "min_support": int(min_support),
        },
    }


def plot_latency_accuracy(
    metrics: dict,
    output_path: Path,
    title: str = "Latency vs Accuracy",
) -> None:
    """Plot AUROC@t curves: overall + per-source vs_balanced_benign."""
    t_grid = metrics["config"]["t_grid"]
    finite_ts = [t for t in t_grid if isinstance(t, int)]

    def _series(auroc_at: dict) -> tuple[list[int], list[float]]:
        xs, ys = [], []
        for t in finite_ts:
            entry = auroc_at.get(str(t), {})
            v = entry.get("auroc")
            if v is None:
                continue
            xs.append(int(t))
            ys.append(float(v))
        return xs, ys

    fig, ax = plt.subplots(figsize=(7, 4.5))

    xs, ys = _series(metrics["overall"]["auroc_at"])
    if xs:
        ax.plot(xs, ys, marker="o", linewidth=2, label="overall", color="black")

    per_source = metrics.get("per_adversarial_source", {})
    for src in sorted(per_source.keys()):
        entry = per_source[src]
        if entry.get("low_support"):
            continue
        xs, ys = _series(entry["vs_balanced_benign"]["auroc_at"])
        if not xs:
            continue
        ax.plot(xs, ys, marker=".", linewidth=1, alpha=0.7, label=src)

    ax.set_xscale("log")
    ax.set_xlabel("Packet index t (log scale)")
    ax.set_ylabel("AUROC at packet t")
    ax.set_ylim(0.4, 1.02)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.6, label="chance")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_latency_accuracy_comparison(
    metrics_by_model: dict[str, dict],
    output_path: Path,
    title: str = "Latency vs Accuracy — model comparison",
) -> None:
    """Overlay AUROC@t curves for multiple models on one figure.

    ``metrics_by_model`` maps a label (e.g. ``"RF"``, ``"TCN"``,
    ``"Meta-Learner"``) to the model's full ``streaming_metrics()`` dict.
    Plots only the ``overall`` curves — per-source comparisons would be
    too dense to read.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, metrics in metrics_by_model.items():
        t_grid = metrics["config"]["t_grid"]
        finite_ts = [t for t in t_grid if isinstance(t, int)]
        xs, ys = [], []
        for t in finite_ts:
            entry = metrics["overall"]["auroc_at"].get(str(t), {})
            v = entry.get("auroc")
            if v is None:
                continue
            xs.append(int(t))
            ys.append(float(v))
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Packet index t (log scale)")
    ax.set_ylabel("AUROC at packet t (overall)")
    ax.set_ylim(0.4, 1.02)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.6, label="chance")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


__all__ = [
    "DEFAULT_T_GRID",
    "StreamingPredictions",
    "streaming_metrics",
    "plot_latency_accuracy",
    "plot_latency_accuracy_comparison",
]
