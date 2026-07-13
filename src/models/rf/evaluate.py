"""Evaluate a trained RF on the held-out test split (per-checkpoint).

Under the streaming-native schema each row of ``dataset.X`` is one
*multi-scale prefix view* of a flow (a checkpoint emitted every K
packets per flow). The metrics produced here are therefore per-checkpoint
— a fair view of the RF's behaviour across the full inference-time
distribution of prefix sizes. The richer per-flow streaming metrics
(AUROC@t, EDP) are also produced here via
``src/models/common/streaming_eval.py`` and written to
``streaming_metrics.json``.

Under DEPT's benign-vs-adversarial label convention, the positive class
is "adversarial" (TANTGen-perturbed traffic) and the negative class is
"benign" (all Kitsune + CIC-IDS2017 + CIC-DDoS-2019 checkpoints).
Per-source metrics are reported only for adversarial sources.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ..common.streaming_eval import (  # noqa: E402
    StreamingPredictions,
    plot_earliness_accuracy,
    streaming_metrics,
)


log = logging.getLogger(__name__)

ADV_MODE_SUFFIXES = ("_adv_both", "_adv_iat", "_adv_size")


def predict_streams(
    output_dir: Path,
    sample_indices: np.ndarray | None = None,
) -> dict:
    """Run the trained RF over ``sample_indices`` checkpoint rows.

    Public helper used by the meta-learner. Loads the trained classifier
    + persisted dataset, runs ``predict_proba`` on the requested rows, and
    groups the per-checkpoint outputs into per-flow streams.

    ``sample_indices`` is a 1D array of row indices into ``dataset.npz``
    (same index space as ``split.npz``'s ``train_idx``/``val_idx``/
    ``test_idx``). Defaults to ``test_idx`` if not given.

    Returns a dict with keys:
      * ``flow_keys``  — list of ``(source: str, flow_id: str)`` per flow
      * ``probs``      — list of [n_ckpt_i] float64 per-checkpoint P(adv)
      * ``logits``     — list of [n_ckpt_i] float64 per-checkpoint logit
      * ``packet_idx`` — list of [n_ckpt_i] int64 absolute packet indices
      * ``labels``     — [N] int64 per-flow label
      * ``sources``    — [N] object (str) per-flow source
      * ``lengths``    — [N] int64 = packet_idx[i][-1] (last covered packet)
    """
    clf = joblib.load(output_dir / "rf.joblib")
    split = np.load(output_dir / "split.npz")
    data = np.load(output_dir / "dataset.npz", allow_pickle=False)

    if sample_indices is None:
        sample_indices = split["test_idx"]
    sample_indices = np.asarray(sample_indices, dtype=np.int64)

    X = data["X"][sample_indices]
    y = data["y"][sample_indices]
    sources = data["sources"][sample_indices]
    flow_ids = data["flow_ids"][sample_indices]
    ckpt_idx = data["checkpoint_idx"][sample_indices]
    if "packet_idx" in data.files:
        pkt_idx = data["packet_idx"][sample_indices]
    else:
        # Legacy datasets without packet_idx assume uniform stride=10.
        pkt_idx = (ckpt_idx + 1) * 10

    probs = clf.predict_proba(X)[:, 1]
    p_clipped = np.clip(probs, 1e-6, 1 - 1e-6)
    logits = np.log(p_clipped / (1 - p_clipped))

    scoped = np.char.add(
        np.char.add(sources.astype(str), "|"), flow_ids.astype(str),
    )
    flow_keys_str, inv = np.unique(scoped, return_inverse=True)

    per_flow_probs: list[np.ndarray] = []
    per_flow_logits: list[np.ndarray] = []
    per_flow_pkt: list[np.ndarray] = []
    per_flow_labels: list[int] = []
    per_flow_sources: list[str] = []
    per_flow_lengths: list[int] = []
    flow_keys: list[tuple[str, str]] = []

    for fk_idx in range(len(flow_keys_str)):
        rows = np.where(inv == fk_idx)[0]
        order = np.argsort(ckpt_idx[rows])
        rows = rows[order]
        per_flow_probs.append(probs[rows].astype(np.float64))
        per_flow_logits.append(logits[rows].astype(np.float64))
        per_flow_pkt.append(pkt_idx[rows].astype(np.int64))
        per_flow_labels.append(int(y[rows][0]))
        per_flow_sources.append(str(sources[rows][0]))
        per_flow_lengths.append(int(per_flow_pkt[-1][-1]))
        src, fid = flow_keys_str[fk_idx].split("|", 1)
        flow_keys.append((src, fid))

    return {
        "flow_keys": flow_keys,
        "probs": per_flow_probs,
        "logits": per_flow_logits,
        "packet_idx": per_flow_pkt,
        "labels": np.asarray(per_flow_labels, dtype=np.int64),
        "sources": np.asarray(per_flow_sources, dtype=object),
        "lengths": np.asarray(per_flow_lengths, dtype=np.int64),
    }


def _base_source(adv_source: str) -> str | None:
    """Strip '_adv_<mode>' → base attack name (e.g. 'mirai_adv_both' → 'mirai')."""
    for suffix in ADV_MODE_SUFFIXES:
        if adv_source.endswith(suffix):
            return adv_source[: -len(suffix)]
    return None


def _classification_block(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    out = {
        "n": int(len(y_true)),
        "n_benign": int((y_true == 0).sum()),
        "n_adversarial": int((y_true == 1).sum()),
        "accuracy": float((y_pred == y_true).mean()) if len(y_true) else 0.0,
        "precision_adversarial": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_adversarial": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_adversarial": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_benign": float(
            precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_benign": float(
            recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_benign": float(
            f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if out["n_benign"] > 0 and out["n_adversarial"] > 0:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out


def evaluate(output_dir: Path) -> dict:
    clf = joblib.load(output_dir / "rf.joblib")
    split = np.load(output_dir / "split.npz")
    data = np.load(output_dir / "dataset.npz", allow_pickle=False)
    feature_names = json.loads((output_dir / "feature_names.json").read_text())

    test_idx = split["test_idx"]
    X_te = data["X"][test_idx]
    y_te = data["y"][test_idx]
    sources_te = data["sources"][test_idx]
    label_str_te = data["label_str"][test_idx]

    y_prob = clf.predict_proba(X_te)[:, 1]
    y_pred = clf.predict(X_te)

    metrics: dict = {"overall": _classification_block(y_te, y_pred, y_prob)}

    # Per adversarial source. For each TANTGen variant we report:
    #   * vs_all_benign       — this variant's positives + every benign
    #                           checkpoint. Realistic prevalence; precision is
    #                           dominated by total benign FP count.
    #   * vs_balanced_benign  — this variant's positives + K × n_adv benign
    #                           (subsample, capped at total benign). Makes
    #                           precision comparable across variants.
    #   * vs_same_base_benign — positives + benign from the same base-attack
    #                           capture (e.g. mirai_adv_both → benign from
    #                           mirai/). Lets us see within-capture separability
    #                           without cross-capture confounds.
    # low_support flags variants with fewer than MIN_SUPPORT adversarial
    # checkpoints.
    MIN_SUPPORT = 50
    BALANCE_K = 10
    rng = np.random.default_rng(42)
    benign_all_idx = np.where(y_te == 0)[0]

    adv_sources = sorted(set(sources_te[y_te == 1].tolist()))
    per_source: dict = {}
    for src in adv_sources:
        src_mask = (sources_te == src) & (y_te == 1)
        n_adv = int(src_mask.sum())
        sel_all = src_mask | (y_te == 0)

        base = _base_source(src)
        if base is not None:
            sel_same = src_mask | ((y_te == 0) & (sources_te == base))
        else:
            sel_same = src_mask  # no matching base → benign subset is empty

        target_bn = min(BALANCE_K * n_adv, len(benign_all_idx))
        if n_adv > 0 and target_bn > 0:
            bn_pick = rng.choice(benign_all_idx, size=target_bn, replace=False)
            bal_mask = np.zeros_like(src_mask)
            bal_mask[np.where(src_mask)[0]] = True
            bal_mask[bn_pick] = True
        else:
            bal_mask = src_mask

        per_source[src] = {
            "low_support": n_adv < MIN_SUPPORT,
            "base_source": base,
            "vs_all_benign": _classification_block(
                y_te[sel_all], y_pred[sel_all], y_prob[sel_all]),
            "vs_balanced_benign": _classification_block(
                y_te[bal_mask], y_pred[bal_mask], y_prob[bal_mask]),
            "vs_same_base_benign": _classification_block(
                y_te[sel_same], y_pred[sel_same], y_prob[sel_same]),
        }
    metrics["per_adversarial_source"] = per_source
    metrics["per_source_config"] = {"min_support": MIN_SUPPORT, "balance_k": BALANCE_K}

    # Per label_str (trivially just benign vs adversarial under this convention).
    per_label: dict[str, dict] = {}
    for lab in sorted(set(label_str_te.tolist())):
        mask = label_str_te == lab
        per_label[lab] = {
            "n": int(mask.sum()),
            "y_true_majority": int(round(float(y_te[mask].mean()))),
            "predicted_adversarial_rate": float((y_pred[mask] == 1).mean()),
            "mean_adversarial_prob": float(y_prob[mask].mean()),
        }
    metrics["per_label_str"] = per_label

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---------- Plots ----------
    cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["benign", "adversarial"]).plot(
        ax=ax, colorbar=False)
    ax.set_title("RF — Confusion Matrix (test, per-checkpoint)")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=140)
    plt.close(fig)

    if metrics["overall"]["roc_auc"] is not None:
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(fpr, tpr, label=f"AUC = {metrics['overall']['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("RF — ROC (test, per-checkpoint)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(output_dir / "roc_curve.png", dpi=140)
        plt.close(fig)

        prec, rec, _ = precision_recall_curve(y_te, y_prob)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(rec, prec, label=f"AP = {metrics['overall']['pr_auc']:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("RF — PR (test, per-checkpoint)")
        ax.legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(output_dir / "pr_curve.png", dpi=140)
        plt.close(fig)

    # ---------- Streaming metrics ----------
    # Group the per-checkpoint predictions by flow into a per-flow stream
    # ordered by ascending checkpoint_idx. The absolute packet index of
    # each checkpoint is captured during dataset assembly (cadence is per
    # source in metadata.json) and stored in ``packet_idx``. For datasets
    # built before that field existed we fall back to the legacy
    # ``(ckpt_idx + 1) * stride`` formula with stride=10.
    if "flow_ids" in data.files and "checkpoint_idx" in data.files:
        flow_ids_te = data["flow_ids"][test_idx]
        ckpt_idx_te = data["checkpoint_idx"][test_idx]
        if "packet_idx" in data.files:
            pkt_idx_te = data["packet_idx"][test_idx]
        else:
            pkt_idx_te = (ckpt_idx_te + 1) * 10
        scoped = np.char.add(
            np.char.add(sources_te.astype(str), "|"),
            flow_ids_te.astype(str),
        )
        flow_keys, inv = np.unique(scoped, return_inverse=True)
        per_flow_probs: list[np.ndarray] = []
        per_flow_pkt: list[np.ndarray] = []
        per_flow_label: list[int] = []
        per_flow_source: list[str] = []
        per_flow_length: list[int] = []
        for fk_idx in range(len(flow_keys)):
            rows = np.where(inv == fk_idx)[0]
            order = np.argsort(ckpt_idx_te[rows])
            rows = rows[order]
            per_flow_probs.append(y_prob[rows].astype(np.float64))
            per_flow_pkt.append(pkt_idx_te[rows].astype(np.int64))
            per_flow_label.append(int(y_te[rows][0]))
            per_flow_source.append(str(sources_te[rows][0]))
            per_flow_length.append(int(per_flow_pkt[-1][-1]))

        sp = StreamingPredictions(
            probs=per_flow_probs,
            packet_idx=per_flow_pkt,
            labels=np.asarray(per_flow_label, dtype=np.int64),
            sources=np.asarray(per_flow_source, dtype=object),
            lengths=np.asarray(per_flow_length, dtype=np.int64),
        )
        streaming = streaming_metrics(sp)
        (output_dir / "streaming_metrics.json").write_text(
            json.dumps(streaming, indent=2))
        plot_earliness_accuracy(
            streaming,
            output_dir / "earliness_vs_accuracy.png",
            title="RF — Earliness vs Accuracy (per-prefix)",
        )
        metrics["streaming"] = streaming["overall"]
        # Re-persist so the on-disk metrics.json includes the streaming
        # block (mirrors tcn/evaluate.py).
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * len(importances))))
    ax.barh(
        [feature_names[i] if i < len(feature_names) else f"f{i}" for i in order[::-1]],
        importances[order[::-1]],
    )
    ax.set_title("RF — Feature Importance")
    ax.set_xlabel("Importance (Gini)")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=140)
    plt.close(fig)

    log.info("evaluation written to %s", output_dir)
    return metrics
