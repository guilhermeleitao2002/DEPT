"""Evaluate the trained Mini-Transformer meta-learner on the test split.

Writes ``metrics.json``, ``streaming_metrics.json``, the plots, and
``earliness_vs_accuracy_comparison.png`` with the RF, TCN and meta-learner
curves.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from torch.utils.data import DataLoader  # noqa: E402

from ..common.runtime import resolve_num_workers  # noqa: E402
from ..common.streaming_eval import (  # noqa: E402
    StreamingPredictions,
    plot_earliness_accuracy,
    plot_earliness_accuracy_comparison,
    streaming_metrics,
)
from .dataset import (  # noqa: E402
    MetaLearnerDataset,
    collate_padded,
    prepare_split_caches,
)
from .model import MetaLearner  # noqa: E402


log = logging.getLogger(__name__)


def _resolve_device(cfg: str) -> torch.device:
    if cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg)


def evaluate(output_dir: Path, config: dict) -> dict:
    cfg_path = output_dir / "metalearner_config.json"
    saved_cfg = json.loads(cfg_path.read_text())
    model_cfg = saved_cfg["model"]
    train_cfg_saved = saved_cfg["training"]

    device = _resolve_device(str(train_cfg_saved.get("device", "auto")))

    model = MetaLearner(
        d_model=int(model_cfg.get("d_model", 64)),
        n_layers=int(model_cfg.get("n_layers", 2)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        dim_feedforward=int(model_cfg.get("dim_feedforward", 128)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        num_classes=int(model_cfg.get("num_classes", 2)),
        max_seq_len=int(model_cfg.get("max_seq_len", 200)),
    ).to(device)
    state = torch.load(output_dir / "metalearner.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Reuse (or build) the test cache. prepare_split_caches loads the
    # existing cache when the base-learner cache key still matches and
    # rebuilds it otherwise (e.g. after retraining the RF or TCN).
    rf_dir = Path(config["rf_model_dir"])
    tcn_dir = Path(config["tcn_model_dir"])
    rf_split = np.load(rf_dir / "split.npz")
    tcn_idx = np.load(tcn_dir / "dataset_index.npz", allow_pickle=True)
    token_cfg = config.get("token", {})
    caches = prepare_split_caches(
        output_dir=output_dir,
        rf_dir=rf_dir, tcn_dir=tcn_dir,
        tensors_dir=Path(config["tensors_dir"]),
        adversarial_dir=Path(config["adversarial_dir"]),
        splits={"test": {
            "rf_sample_indices": rf_split["test_idx"],
            "tcn_sample_indices": tcn_idx["used_test_idx"],
        }},
        rf_age_clip=int(token_cfg.get("rf_age_clip", 30)),
        pkt_len_norm=float(token_cfg.get("pkt_len_norm", 1500.0)),
        max_length=int(train_cfg_saved.get("max_length", 200)),
    )
    cache = caches["test"]

    num_workers = resolve_num_workers(train_cfg_saved.get("num_workers", "auto"))
    ds = MetaLearnerDataset(cache)
    loader = DataLoader(
        ds,
        batch_size=int(train_cfg_saved.get("batch_size", 128)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=collate_padded,
    )

    per_flow_probs: list[np.ndarray] = []
    per_flow_pkt: list[np.ndarray] = []
    last_probs: list[float] = []
    with torch.inference_mode():
        for tokens, _labels, lengths, mask in loader:
            tokens = tokens.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            logits = model(tokens, key_padding_mask=mask)
            probs = torch.softmax(logits.float(), dim=2)[:, :, 1]
            probs_cpu = probs.detach().cpu().numpy()
            for b in range(probs_cpu.shape[0]):
                L = int(lengths[b].item())
                per_flow_probs.append(probs_cpu[b, :L].astype(np.float64))
                per_flow_pkt.append(np.arange(1, L + 1, dtype=np.int64))
                last_probs.append(float(probs_cpu[b, L - 1]))

    labels = cache.labels.astype(np.int64)
    sources = cache.sources.astype(object)
    lengths = cache.lengths.astype(np.int64)
    sp = StreamingPredictions(
        probs=per_flow_probs, packet_idx=per_flow_pkt,
        labels=labels, sources=sources, lengths=lengths,
    )
    streaming = streaming_metrics(sp)
    (output_dir / "streaming_metrics.json").write_text(
        json.dumps(streaming, indent=2)
    )
    plot_earliness_accuracy(
        streaming, output_dir / "earliness_vs_accuracy.png",
        title="Meta-Learner — Earliness vs Accuracy (per-packet)",
    )

    # Last-timestep verdict for the flow-level confusion matrix / ROC.
    y_pred = (np.asarray(last_probs) >= 0.5).astype(np.int64)
    y_prob = np.asarray(last_probs, dtype=np.float64)
    from sklearn.metrics import (
        average_precision_score, f1_score, precision_score, recall_score,
        roc_auc_score,
    )
    metrics = {
        "overall": {
            "n": int(len(labels)),
            "n_benign": int((labels == 0).sum()),
            "n_adversarial": int((labels == 1).sum()),
            "accuracy": float((y_pred == labels).mean()),
            "precision_adversarial": float(
                precision_score(labels, y_pred, pos_label=1, zero_division=0)),
            "recall_adversarial": float(
                recall_score(labels, y_pred, pos_label=1, zero_division=0)),
            "f1_adversarial": float(
                f1_score(labels, y_pred, pos_label=1, zero_division=0)),
            "macro_f1": float(
                f1_score(labels, y_pred, average="macro", zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, y_prob)) if labels.min() == 0 and labels.max() == 1 else None,
            "pr_auc": float(average_precision_score(labels, y_prob)) if labels.min() == 0 and labels.max() == 1 else None,
        },
        "verdict_at": "last_timestep",
        "streaming": streaming["overall"],
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Confusion + ROC + PR plots (last-timestep verdict).
    cm = confusion_matrix(labels, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["benign", "adversarial"]).plot(
        ax=ax, colorbar=False)
    ax.set_title("Meta-Learner — Confusion Matrix (last timestep)")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=140)
    plt.close(fig)

    if metrics["overall"]["roc_auc"] is not None:
        fpr, tpr, _ = roc_curve(labels, y_prob)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(fpr, tpr, label=f"AUC = {metrics['overall']['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("Meta-Learner — ROC (last timestep)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(output_dir / "roc_curve.png", dpi=140)
        plt.close(fig)
        prec, rec, _ = precision_recall_curve(labels, y_prob)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(rec, prec, label=f"AP = {metrics['overall']['pr_auc']:.3f}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title("Meta-Learner — PR (last timestep)")
        ax.legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(output_dir / "pr_curve.png", dpi=140)
        plt.close(fig)

    # Three-way comparison plot (overall AUROC@t for RF, TCN, Meta-Learner).
    cmp = {}
    for label, p in [("RF", rf_dir / "streaming_metrics.json"),
                     ("TCN", tcn_dir / "streaming_metrics.json")]:
        if p.exists():
            cmp[label] = json.loads(p.read_text())
    cmp["Meta-Learner"] = streaming
    plot_earliness_accuracy_comparison(
        cmp, output_dir / "earliness_vs_accuracy_comparison.png",
        title="DEPT base learners + Meta-Learner — overall AUROC@t (test)",
    )

    log.info("evaluation written to %s", output_dir)
    return metrics


__all__ = ["evaluate"]
