"""Train the TCN base learner for ADEPT.

The dataset assembly and flow-level split mirror RF
(``src/models/rf/train.py``) — both share ``src/models/common/dataset.py``
and therefore see the same windows in the same train/val/test bins under
``seed=42``.
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
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from ..common.dataset import (
    AssembledIndex,
    Split,
    assemble_index,
    flow_level_split,
    split_summary,
)
from .dataset import compute_class_weights, make_dataloaders
from .model import TCN


log = logging.getLogger(__name__)


def _resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def _maybe_subsample(idx: np.ndarray, max_n: int | None, rng: np.random.Generator) -> np.ndarray:
    if max_n is None or max_n <= 0 or len(idx) <= max_n:
        return idx
    pick = rng.choice(len(idx), size=int(max_n), replace=False)
    return np.sort(idx[pick]).astype(np.int64)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: "torch.amp.GradScaler",
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
            logits = model(x)
            loss = loss_fn(logits, y)
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
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (val_loss_avg, val_macro_f1, y_true, y_pred, y_prob)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    y_true_chunks: list[np.ndarray] = []
    y_pred_chunks: list[np.ndarray] = []
    y_prob_chunks: list[np.ndarray] = []
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y_dev = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
            logits = model(x)
            loss = loss_fn(logits, y_dev)
        total_loss += float(loss.detach().item())
        n_batches += 1
        probs = torch.softmax(logits.float(), dim=1)[:, 1]
        preds = logits.argmax(dim=1)
        y_true_chunks.append(y.numpy().astype(np.int64))
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
    max_windows: int | None = None,
) -> dict:
    """Train + persist artifacts. Returns a small status dict."""
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["split"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    train_cfg = dict(config.get("training", {}))
    model_cfg = dict(config.get("model", {}))

    device = _resolve_device(str(train_cfg.get("device", "auto")))
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"

    # ---- 1. Assemble index (no payload loaded) ---------------------------------
    idx: AssembledIndex = assemble_index(
        tensors_dir=tensors_root,
        benign_sources=list(config["benign_sources"]),
        adversarial_dir=Path(config["adversarial_dir"]),
        benign_cap_ratio=float(config["benign_cap_ratio"]),
        seed=seed,
        adversarial_exclude=list(config.get("adversarial_exclude", [])),
    )
    log.info("assembled index: n_rows=%d sources=%d", len(idx.y), len(idx.source_folders))

    # ---- 2. Flow-level split ---------------------------------------------------
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

    # Optional smoke-test subsampling — applied AFTER split so each split
    # still has its own slice (and the subsampling preserves flow-atomicity
    # only loosely; this is a debug knob).
    train_idx = _maybe_subsample(split.train_idx, max_windows, rng)
    val_idx = _maybe_subsample(split.val_idx, max_windows, rng)
    test_idx = _maybe_subsample(split.test_idx, max_windows, rng)

    # ---- 3. Class weights ------------------------------------------------------
    class_weight_mode = str(train_cfg.get("class_weight", "balanced"))
    class_weights = compute_class_weights(idx.y[train_idx], mode=class_weight_mode)
    log.info("class weights (%s): %s", class_weight_mode, class_weights.tolist())

    # ---- 4. DataLoaders --------------------------------------------------------
    train_loader, val_loader, test_loader = make_dataloaders(
        sources=idx.sources,
        local_idx=idx.local_idx,
        y=idx.y,
        source_folders=idx.source_folders,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=int(train_cfg.get("batch_size", 256)),
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )

    # ---- 5. Model + optimiser --------------------------------------------------
    model = TCN(
        num_features=int(model_cfg.get("num_features", 1090)),
        channels=tuple(model_cfg.get("channels", (128, 128, 128))),
        kernel_size=int(model_cfg.get("kernel_size", 3)),
        dropout=float(model_cfg.get("dropout", 0.15)),
        num_classes=int(model_cfg.get("num_classes", 2)),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("model: %d params, receptive field=%d, T=10",
             n_params, model.receptive_field)

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

    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # ---- 6. Training loop ------------------------------------------------------
    early_stop_patience = int(train_cfg.get("early_stop_patience", 3))
    best = {"epoch": -1, "val_macro_f1": -math.inf, "state": None}
    patience_left = early_stop_patience
    curves: list[dict] = []
    t_train_start = time.time()
    for epoch in range(epochs):
        t0 = time.time()
        train_loss = _train_one_epoch(
            model, train_loader, optimizer, scaler, loss_fn, device, use_amp)
        val_loss, val_macro_f1, _, _, _ = _eval_one_epoch(
            model, val_loader, loss_fn, device, use_amp)
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

    # If training never improved (single epoch with NaN loss, etc.), persist
    # the last state so eval-mode still has something to load.
    if best["state"] is None:
        best = {
            "epoch": 0,
            "val_macro_f1": float("nan"),
            "state": copy.deepcopy(model.state_dict()),
        }

    # ---- 7. Persist artifacts --------------------------------------------------
    torch.save(best["state"], output_dir / "tcn.pt")
    np.savez(
        output_dir / "split.npz",
        train_idx=split.train_idx,
        val_idx=split.val_idx,
        test_idx=split.test_idx,
    )
    # Encode `sources` and `label_str` as int codes + a vocab so the saved
    # index stays small (object arrays serialise via pickle and balloon).
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
        # Subsampled split slices (when --max-windows used). Equal to the
        # full split when no subsampling. Note: flow_ids is not persisted —
        # it's only needed for the split, which is already materialised.
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
            "max_windows_per_split": int(max_windows) if max_windows else None,
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
