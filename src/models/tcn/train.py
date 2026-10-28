"""Train the TCN base learner.

The loss is per-packet cross-entropy, averaged inside each flow (so long
flows don't dominate) and masked over padding. Model selection uses the
flow-level macro F1 of the prediction at each flow's last packet.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

# torch.amp (unified) was added in PyTorch 2.0; fall back to torch.cuda.amp.
# The old API doesn't accept device_type, so wrap it to swallow that kwarg.
try:
    _GradScaler = torch.amp.GradScaler
    _autocast = torch.amp.autocast
except AttributeError:
    import contextlib

    _GradScaler = torch.cuda.amp.GradScaler  # type: ignore[attr-defined]

    @contextlib.contextmanager
    def _autocast(device_type: str = "cuda", enabled: bool = True):  # type: ignore[misc]
        with torch.cuda.amp.autocast(enabled=enabled):  # type: ignore[attr-defined]
            yield

from ..common.dataset import (
    AssembledIndex,
    Split,
    assemble_index,
    flow_level_split,
    split_summary,
)
from ..common.runtime import resolve_num_workers
from .dataset import compute_class_weights, make_dataloaders
from .model import TCN


log = logging.getLogger(__name__)


def _resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def _masked_per_packet_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """Per-packet CE, averaged within flow, then averaged across flows.

    Args:
        logits: [B, T_max, C] per-timestep logits.
        labels: [B] flow-level labels (propagated to every timestep).
        mask: [B, T_max] bool — True at valid timesteps.
        class_weights: [C] per-class weights.
    """
    B, T_max, C = logits.shape
    target = labels.unsqueeze(1).expand(B, T_max)
    ce = F.cross_entropy(
        logits.reshape(B * T_max, C),
        target.reshape(B * T_max),
        weight=class_weights,
        reduction="none",
    ).view(B, T_max)
    ce = ce * mask.to(ce.dtype)
    valid = mask.to(ce.dtype).sum(dim=1).clamp(min=1)
    per_flow = ce.sum(dim=1) / valid
    return per_flow.mean()


def _last_timestep_logits(
    logits: torch.Tensor, lengths: torch.Tensor,
) -> torch.Tensor:
    """Gather logits at each flow's final valid timestep — shape [B, C]."""
    B, T_max, C = logits.shape
    idx = (lengths.to(logits.device) - 1).clamp(min=0)
    idx_exp = idx.view(B, 1, 1).expand(B, 1, C)
    return logits.gather(dim=1, index=idx_exp).squeeze(1)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: "torch.cuda.amp.GradScaler",
    class_weights: torch.Tensor,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    for seq, labels, lengths, mask in loader:
        seq = seq.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device_type=amp_device, enabled=use_amp):
            logits = model(seq)
            loss = _masked_per_packet_loss(logits, labels, mask, class_weights)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach().item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.inference_mode()
def _eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Flow-level evaluation using the logit at each flow's last valid timestep.

    Returns ``(val_loss_avg, val_macro_f1, y_true, y_pred, y_prob)``.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    y_true_chunks: list[np.ndarray] = []
    y_pred_chunks: list[np.ndarray] = []
    y_prob_chunks: list[np.ndarray] = []
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    for seq, labels, lengths, mask in loader:
        seq = seq.to(device, non_blocking=True)
        labels_dev = labels.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with _autocast(device_type=amp_device, enabled=use_amp):
            logits = model(seq)
            loss = _masked_per_packet_loss(logits, labels_dev, mask, class_weights)
        total_loss += float(loss.detach().item())
        n_batches += 1

        last_logits = _last_timestep_logits(logits, lengths).float()
        probs = torch.softmax(last_logits, dim=1)[:, 1]
        preds = last_logits.argmax(dim=1)
        y_true_chunks.append(labels.numpy().astype(np.int64))
        y_pred_chunks.append(preds.detach().cpu().numpy().astype(np.int64))
        y_prob_chunks.append(probs.detach().cpu().numpy().astype(np.float32))
    y_true = np.concatenate(y_true_chunks) if y_true_chunks else np.zeros(0, dtype=np.int64)
    y_pred = np.concatenate(y_pred_chunks) if y_pred_chunks else np.zeros(0, dtype=np.int64)
    y_prob = np.concatenate(y_prob_chunks) if y_prob_chunks else np.zeros(0, dtype=np.float32)
    macro_f1 = (
        float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        if len(y_true) > 0 else 0.0
    )
    return total_loss / max(n_batches, 1), macro_f1, y_true, y_pred, y_prob


def train(
    config: dict,
    tensors_root: Path,
    output_dir: Path,
) -> dict:
    """Train + persist artifacts. Returns a small status dict."""
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["split"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_cfg = dict(config.get("training", {}))
    model_cfg = dict(config.get("model", {}))

    device = _resolve_device(str(train_cfg.get("device", "auto")))
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"

    # ---- 1. Assemble per-flow index ------------------------------------------
    idx: AssembledIndex = assemble_index(
        tensors_dir=tensors_root,
        benign_sources=list(config["benign_sources"]),
        adversarial_dir=Path(config["adversarial_dir"]),
        benign_cap_ratio=float(config["benign_cap_ratio"]),
        seed=seed,
        adversarial_exclude=list(config.get("adversarial_exclude", [])),
    )
    log.info("assembled per-flow index: n_flows=%d sources=%d",
             len(idx.y), len(idx.source_folders))

    # ---- 2. Flow-level split -------------------------------------------------
    split: Split = flow_level_split(
        sources=idx.sources,
        flow_ids=idx.flow_ids,
        train_frac=float(config["split"]["train_frac"]),
        val_frac=float(config["split"]["val_frac"]),
        test_frac=float(config["split"]["test_frac"]),
        seed=seed,
    )
    summary = split_summary(idx.sources, idx.y, split)
    log.info("split summary:\n%s", json.dumps(summary, indent=2))

    train_idx = split.train_idx
    val_idx = split.val_idx
    test_idx = split.test_idx

    # ---- 3. Class weights (flow-level frequencies) ---------------------------
    class_weight_mode = str(train_cfg.get("class_weight", "balanced"))
    class_weights = compute_class_weights(idx.y[train_idx], mode=class_weight_mode)
    log.info("class weights (%s): %s", class_weight_mode, class_weights.tolist())

    # ---- 4. DataLoaders ------------------------------------------------------
    train_loader, val_loader, test_loader = make_dataloaders(
        sources=idx.sources,
        local_idx=idx.local_idx,
        y=idx.y,
        source_folders=idx.source_folders,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=int(train_cfg.get("batch_size", 32)),
        num_workers=resolve_num_workers(train_cfg.get("num_workers", "auto")),
        pin_memory=device.type == "cuda",
        max_length=train_cfg.get("max_length", None),
    )

    # ---- 5. Model + optimiser ------------------------------------------------
    model = TCN(
        num_features=int(model_cfg.get("num_features", 1090)),
        channels=tuple(model_cfg.get("channels", (128, 128, 128))),
        kernel_size=int(model_cfg.get("kernel_size", 3)),
        dropout=float(model_cfg.get("dropout", 0.15)),
        num_classes=int(model_cfg.get("num_classes", 2)),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(
        "model: %d params, receptive field=%d, variable-length sequences",
        n_params, model.receptive_field,
    )

    epochs = int(train_cfg.get("epochs", 20))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler_kind = str(train_cfg.get("scheduler", "cosine"))
    if scheduler_kind == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    elif scheduler_kind == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2)
    else:
        raise ValueError(f"unknown scheduler: {scheduler_kind}")

    class_weights_dev = class_weights.to(device)
    scaler = _GradScaler(enabled=use_amp)

    # ---- 6. Training loop ----------------------------------------------------
    early_stop_patience = int(train_cfg.get("early_stop_patience", 3))
    best = {"epoch": -1, "val_macro_f1": -math.inf, "state": None}
    patience_left = early_stop_patience
    curves: list[dict] = []
    t_train_start = time.time()
    for epoch in range(epochs):
        t0 = time.time()
        train_loss = _train_one_epoch(
            model, train_loader, optimizer, scaler, class_weights_dev,
            device, use_amp,
        )
        val_loss, val_macro_f1, _, _, _ = _eval_one_epoch(
            model, val_loader, class_weights_dev, device, use_amp,
        )
        if scheduler_kind == "plateau":
            scheduler.step(val_macro_f1)
        else:
            scheduler.step()
        epoch_t = time.time() - t0
        cur_lr = float(optimizer.param_groups[0]["lr"])
        curves.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_f1": val_macro_f1,
            "lr": cur_lr,
            "seconds": epoch_t,
        })
        improved = val_macro_f1 > best["val_macro_f1"]
        log.info(
            "epoch %02d  train_loss=%.4f  val_loss=%.4f  val_f1=%.4f  lr=%.2e  (%.1fs)%s",
            epoch, train_loss, val_loss, val_macro_f1, cur_lr, epoch_t,
            "  *" if improved else "")
        if improved:
            best = {
                "epoch": epoch,
                "val_macro_f1": val_macro_f1,
                "state": copy.deepcopy(model.state_dict()),
            }
            patience_left = early_stop_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                log.info("early stop at epoch %d (no val_f1 improvement for %d epochs)",
                         epoch, early_stop_patience)
                break

    train_seconds = time.time() - t_train_start

    if best["state"] is None:
        best = {
            "epoch": 0,
            "val_macro_f1": float("nan"),
            "state": copy.deepcopy(model.state_dict()),
        }

    # ---- 7. Persist artifacts ------------------------------------------------
    torch.save(best["state"], output_dir / "tcn.pt")
    np.savez(
        output_dir / "split.npz",
        train_idx=split.train_idx,
        val_idx=split.val_idx,
        test_idx=split.test_idx,
    )
    src_vocab, src_codes = np.unique(idx.sources.astype(str), return_inverse=True)
    lab_vocab, lab_codes = np.unique(idx.label_str.astype(str), return_inverse=True)
    np.savez(
        output_dir / "dataset_index.npz",
        source_codes=src_codes.astype(np.int32),
        source_vocab=src_vocab.astype(str),
        label_codes=lab_codes.astype(np.int8),
        label_vocab=lab_vocab.astype(str),
        local_idx=idx.local_idx,
        y=idx.y,
        feature_names=np.array(idx.feature_names or [], dtype=object),
        source_folder_keys=np.array(list(idx.source_folders.keys()), dtype=str),
        source_folder_values=np.array(
            [str(p) for p in idx.source_folders.values()], dtype=str),
        used_train_idx=train_idx,
        used_val_idx=val_idx,
        used_test_idx=test_idx,
    )
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "training_curves.json").write_text(json.dumps(curves, indent=2))
    (output_dir / "tcn_config.json").write_text(json.dumps(
        {"model": model_cfg, "training": train_cfg,
         "n_parameters": int(n_params),
         "receptive_field": int(model.receptive_field)},
        indent=2,
        default=str,
    ))
    (output_dir / "training_info.json").write_text(json.dumps(
        {
            "best_epoch": int(best["epoch"]),
            "best_val_macro_f1": float(best["val_macro_f1"]),
            "epochs_run": len(curves),
            "epochs_planned": epochs,
            "train_seconds": float(train_seconds),
            "device": str(device),
            "amp": bool(use_amp),
            "n_parameters": int(n_params),
            "split_summary": summary,
            "config": {k: v for k, v in config.items() if k not in {"training", "model"}},
            "training_config": train_cfg,
            "model_config": model_cfg,
        },
        indent=2,
        default=str,
    ))

    log.info("artifacts written to %s", output_dir)
    return {
        "best_epoch": int(best["epoch"]),
        "best_val_macro_f1": float(best["val_macro_f1"]),
        "epochs_run": len(curves),
        "split_summary": summary,
        "train_seconds": float(train_seconds),
        "device": str(device),
        "n_parameters": int(n_params),
    }
