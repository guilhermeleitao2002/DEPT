"""Evaluate a trained RF on the held-out test split."""
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


log = logging.getLogger(__name__)


def _classification_block(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    out = {
        "n": int(len(y_true)),
        "n_benign": int((y_true == 0).sum()),
        "n_attack": int((y_true == 1).sum()),
        "accuracy": float((y_pred == y_true).mean()),
        "precision_attack": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_attack": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_attack": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_benign": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_benign": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_benign": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    # AUC only well-defined when both classes present.
    if out["n_benign"] > 0 and out["n_attack"] > 0:
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
    attacks_te = data["attacks"][test_idx]
    label_str_te = data["label_str"][test_idx]

    y_prob = clf.predict_proba(X_te)[:, 1]
    y_pred = clf.predict(X_te)

    metrics: dict = {"overall": _classification_block(y_te, y_pred, y_prob)}

    # Per source_attack. We report three slices so the reader can see both the
    # raw numbers and a fair precision:
    #   * vs_all_benign     — this attack's positives + every benign window in test.
    #                         Realistic prevalence, but precision is dominated by
    #                         total benign FP count — tiny attacks are penalized
    #                         structurally even with perfect recall.
    #   * vs_balanced_benign — this attack's positives + a subsample of benign
    #                         of size min(K × n_attack, all_benign). Makes
    #                         precision comparable across attacks of different
    #                         sizes; use this as the primary per-attack metric.
    #   * vs_same_src_benign — benign from the same source dataset only.
    # A low_support flag is set when n_attack is below MIN_SUPPORT so the
    # printer can mark noisy rows without dropping them from the JSON.
    MIN_SUPPORT = 50
    BALANCE_K = 10
    rng = np.random.default_rng(42)
    benign_all_idx = np.where(y_te == 0)[0]

    per_attack = {}
    for attack in sorted(set(attacks_te.tolist())):
        attack_mask = attacks_te == attack
        n_atk = int(attack_mask.sum())
        sel = attack_mask | (y_te == 0)
        sel_same_src = attack_mask | ((y_te == 0) & (attacks_te == attack))

        # Balanced benign: subsample benign to K × n_atk (capped at total benign).
        target_bn = min(BALANCE_K * n_atk, len(benign_all_idx))
        if n_atk > 0 and target_bn > 0:
            bn_pick = rng.choice(benign_all_idx, size=target_bn, replace=False)
            bal_mask = np.zeros_like(attack_mask)
            bal_mask[np.where(attack_mask)[0]] = True
            bal_mask[bn_pick] = True
        else:
            bal_mask = attack_mask

        per_attack[attack] = {
            "low_support": n_atk < MIN_SUPPORT,
            "vs_all_benign": _classification_block(y_te[sel], y_pred[sel], y_prob[sel]),
            "vs_balanced_benign": _classification_block(
                y_te[bal_mask], y_pred[bal_mask], y_prob[bal_mask]
            ),
            "vs_same_src_benign": _classification_block(
                y_te[sel_same_src], y_pred[sel_same_src], y_prob[sel_same_src]
            ),
        }
    metrics["per_attack"] = per_attack
    metrics["per_attack_config"] = {"min_support": MIN_SUPPORT, "balance_k": BALANCE_K}

    # Per label_str breakdown (raw attack name).
    per_label: dict[str, dict] = {}
    for lab in sorted(set(label_str_te.tolist())):
        mask = label_str_te == lab
        per_label[lab] = {
            "n": int(mask.sum()),
            "y_true_majority": int(round(float(y_te[mask].mean()))),
            "predicted_attack_rate": float((y_pred[mask] == 1).mean()),
            "mean_attack_prob": float(y_prob[mask].mean()),
        }
    metrics["per_label_str"] = per_label

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---------- Plots ----------
    # Confusion matrix (binary, attack=1)
    cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["benign", "attack"]).plot(ax=ax, colorbar=False)
    ax.set_title("RF — Confusion Matrix (test)")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=140)
    plt.close(fig)

    # ROC curve
    if metrics["overall"]["roc_auc"] is not None:
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(fpr, tpr, label=f"AUC = {metrics['overall']['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("RF — ROC (test)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(output_dir / "roc_curve.png", dpi=140)
        plt.close(fig)

        prec, rec, _ = precision_recall_curve(y_te, y_prob)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(rec, prec, label=f"AP = {metrics['overall']['pr_auc']:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("RF — PR (test)")
        ax.legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(output_dir / "pr_curve.png", dpi=140)
        plt.close(fig)

    # Feature importance
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
