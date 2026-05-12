"""Evaluate a trained TCN on the held-out test split (flow-level).

Under the streaming-native schema the TCN emits per-packet logits, so the
flow-level test verdict here is taken at each flow's **final valid
timestep** — i.e., the model's prediction after seeing the whole flow.
This produces the same per-flow ``metrics.json`` schema RF used to write
(``overall``, ``per_adversarial_source`` with ``vs_*`` slicings,
``per_label_str``), so existing summary tooling keeps working.

The richer streaming metrics (AUROC@t, EDP) are computed in
``src/models/common/streaming_eval.py`` and will be folded into this
file in Phase 4 of the streaming retrofit.
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
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader  # noqa: E402

from .dataset import FlowSequenceDataset, collate_padded  # noqa: E402
from .model import TCN  # noqa: E402

from ..common.streaming_eval import (  # noqa: E402
    StreamingPredictions,
    plot_latency_accuracy,
    streaming_metrics,
)


log = logging.getLogger(__name__)

ADV_MODE_SUFFIXES = ("_adv_both", "_adv_iat", "_adv_size")


def _base_source(adv_source: str) -> str | None:
    """Strip '_adv_<mode>' → base attack name."""
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


def _resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def _load_dataset_index(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    keys = data["source_folder_keys"]
    vals = data["source_folder_values"]
    source_folders = {str(k): Path(str(v)) for k, v in zip(keys, vals)}
    src_vocab = data["source_vocab"].astype(str)
    lab_vocab = data["label_vocab"].astype(str)
    sources = src_vocab[data["source_codes"]]
    label_str = lab_vocab[data["label_codes"]]
    return {
        "sources": sources,
        "local_idx": data["local_idx"],
        "y": data["y"],
        "label_str": label_str,
        "feature_names": list(data["feature_names"]),
        "source_folders": source_folders,
        "used_test_idx": data["used_test_idx"],
    }


@torch.inference_mode()
def _collect_flow_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Forward through ``loader`` and return per-flow predictions.

    Returns ``(y_true, y_pred, y_prob, per_flow_streams)`` where:
      * ``y_true``/``y_pred``/``y_prob`` are the per-flow last-timestep
        verdict for the flow-level summary metrics.
      * ``per_flow_streams`` is a list of 1D ``np.float32`` arrays of
        length ``L_i`` — the TCN's per-packet ``P(adversarial)`` for
        each flow in dataloader order.
    """
    model.eval()
    y_true_chunks: list[np.ndarray] = []
    y_pred_chunks: list[np.ndarray] = []
    y_prob_chunks: list[np.ndarray] = []
    streams: list[np.ndarray] = []
    use_amp = device.type == "cuda"
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    for seq, labels, lengths, _mask in loader:
        seq = seq.to(device, non_blocking=True)
        lengths_dev = lengths.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
            logits = model(seq)              # [B, T, C]
        B, T, C = logits.shape
        # Per-flow last-timestep verdict.
        idx = (lengths_dev - 1).clamp(min=0).view(B, 1, 1).expand(B, 1, C)
        last_logits = logits.gather(dim=1, index=idx).squeeze(1).float()
        probs = torch.softmax(last_logits, dim=1)[:, 1]
        preds = last_logits.argmax(dim=1)
        y_true_chunks.append(labels.numpy().astype(np.int64))
        y_pred_chunks.append(preds.detach().cpu().numpy().astype(np.int64))
        y_prob_chunks.append(probs.detach().cpu().numpy().astype(np.float32))

        # Per-packet streams (clip to each flow's real length).
        per_pkt = torch.softmax(logits.float(), dim=2)[:, :, 1]  # [B, T]
        per_pkt_cpu = per_pkt.detach().cpu().numpy().astype(np.float32)
        for b in range(B):
            Lb = int(lengths[b].item())
            streams.append(per_pkt_cpu[b, :Lb].copy())
    if not y_true_chunks:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float32),
            [],
        )
    return (
        np.concatenate(y_true_chunks),
        np.concatenate(y_pred_chunks),
        np.concatenate(y_prob_chunks),
        streams,
    )


def _plot_training_curves(curves: list[dict], output_path: Path) -> None:
    if not curves:
        return
    epochs = [c["epoch"] for c in curves]
    train_loss = [c["train_loss"] for c in curves]
    val_loss = [c["val_loss"] for c in curves]
    val_f1 = [c["val_macro_f1"] for c in curves]

    fig, ax_loss = plt.subplots(figsize=(6, 4))
    ax_loss.plot(epochs, train_loss, label="train_loss", color="tab:blue")
    ax_loss.plot(epochs, val_loss, label="val_loss", color="tab:orange")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.3)

    ax_f1 = ax_loss.twinx()
    ax_f1.plot(epochs, val_f1, label="val_macro_f1", color="tab:green", linestyle="--")
    ax_f1.set_ylabel("Val Macro F1")
    ax_f1.set_ylim(0.0, 1.0)

    lines, labels = ax_loss.get_legend_handles_labels()
    lines2, labels2 = ax_f1.get_legend_handles_labels()
    ax_loss.legend(lines + lines2, labels + labels2, loc="best")

    fig.suptitle("TCN — Training Curves")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def evaluate(output_dir: Path) -> dict:
    """Load artifacts produced by ``train``, run flow-level inference on the
    test split, and write metrics + plots.
    """
    cfg = json.loads((output_dir / "tcn_config.json").read_text())
    train_info = json.loads((output_dir / "training_info.json").read_text())

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    device = _resolve_device(str(train_cfg.get("device", "auto")))

    model = TCN(
        num_features=int(model_cfg.get("num_features", 1090)),
        channels=tuple(model_cfg.get("channels", (128, 128, 128))),
        kernel_size=int(model_cfg.get("kernel_size", 3)),
        dropout=float(model_cfg.get("dropout", 0.15)),
        num_classes=int(model_cfg.get("num_classes", 2)),
    ).to(device)
    state = torch.load(output_dir / "tcn.pt", map_location=device)
    model.load_state_dict(state)

    idx = _load_dataset_index(output_dir / "dataset_index.npz")
    test_idx = idx["used_test_idx"]

    test_ds = FlowSequenceDataset(
        sources=idx["sources"],
        local_idx=idx["local_idx"],
        y=idx["y"],
        sample_indices=test_idx,
        source_folders=idx["source_folders"],
        max_length=train_cfg.get("max_length", None),
    )
    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", 4))
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=collate_padded,
    )

    y_te, y_pred, y_prob, per_flow_streams = _collect_flow_predictions(
        model, test_loader, device,
    )

    sources_te = idx["sources"][test_idx]
    label_str_te = idx["label_str"][test_idx]

    metrics: dict = {"overall": _classification_block(y_te, y_pred, y_prob)}

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
            sel_same = src_mask

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

    per_label: dict[str, dict] = {}
    for lab in sorted(set(label_str_te.tolist())):
        mask = label_str_te == lab
        per_label[str(lab)] = {
            "n": int(mask.sum()),
            "y_true_majority": int(round(float(y_te[mask].mean()))) if int(mask.sum()) else 0,
            "predicted_adversarial_rate": float((y_pred[mask] == 1).mean()) if int(mask.sum()) else 0.0,
            "mean_adversarial_prob": float(y_prob[mask].mean()) if int(mask.sum()) else 0.0,
        }
    metrics["per_label_str"] = per_label

    metrics["device"] = str(device)
    metrics["best_epoch"] = train_info.get("best_epoch")
    metrics["verdict_at"] = "last_valid_timestep"

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---------- Streaming metrics (per-packet AUROC@t + EDP) ----------
    if per_flow_streams:
        lengths = np.asarray([len(s) for s in per_flow_streams], dtype=np.int64)
        packet_idx_list = [np.arange(1, L + 1, dtype=np.int64) for L in lengths]
        sp = StreamingPredictions(
            probs=[s.astype(np.float64) for s in per_flow_streams],
            packet_idx=packet_idx_list,
            labels=y_te.astype(np.int64),
            sources=sources_te.astype(object),
            lengths=lengths,
        )
        streaming = streaming_metrics(sp)
        (output_dir / "streaming_metrics.json").write_text(
            json.dumps(streaming, indent=2))
        plot_latency_accuracy(
            streaming,
            output_dir / "latency_vs_accuracy.png",
            title="TCN — Latency vs Accuracy (per-packet)",
        )
        metrics["streaming"] = streaming["overall"]
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["benign", "adversarial"]).plot(
        ax=ax, colorbar=False)
    ax.set_title("TCN — Confusion Matrix (flow-level, end-of-flow)")
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
        ax.set_title("TCN — ROC (flow-level, end-of-flow)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(output_dir / "roc_curve.png", dpi=140)
        plt.close(fig)

        prec, rec, _ = precision_recall_curve(y_te, y_prob)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(rec, prec, label=f"AP = {metrics['overall']['pr_auc']:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("TCN — PR (flow-level, end-of-flow)")
        ax.legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(output_dir / "pr_curve.png", dpi=140)
        plt.close(fig)

    curves_path = output_dir / "training_curves.json"
    if curves_path.exists():
        curves = json.loads(curves_path.read_text())
        _plot_training_curves(curves, output_dir / "training_curves.png")

    log.info("evaluation written to %s", output_dir)
    return metrics
