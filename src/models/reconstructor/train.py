"""Train the reconstructor (purification) transformer for DEPT.

Trains the causal residual denoiser on matched ``(adversarial → clean)`` flow
pairs. The loss is a per-element **smooth-L1** between the reconstruction and
the clean target, averaged within each flow (so long flows don't dominate) and
masked over right-side padding — the same flow-balancing used by the TCN.

The loss is **uniform** over all 1090 feature dimensions: the model is
supervised only to "reconstruct the clean flow" and is never told which
dimensions TANTGen perturbed.

Best-epoch tracking uses the validation reconstruction loss (lower is better).
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
import torch.nn.functional as F

# torch.amp (unified) was added in PyTorch 2.0; fall back to torch.cuda.amp.
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

from ..common.dataset import Split, flow_level_split
from ..common.runtime import resolve_num_workers
from .dataset import (
    build_pair_index,
    make_pair_dataloaders,
    pair_split_summary,
)
from .model import Reconstructor, num_parameters


log = logging.getLogger(__name__)


def _resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def _recon_loss(
    recon: torch.Tensor,
    clean: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Masked, flow-balanced smooth-L1 between reconstruction and clean target.

    Args:
        recon/clean: ``[B, T, F]`` reconstruction and target.
        mask: ``[B, T]`` bool — True at valid timesteps.

    The per-element smooth-L1 is averaged uniformly over all ``F`` dims, then
    over valid packets within each flow (so long flows don't dominate), then
    over flows.
    """
    sl1 = F.smooth_l1_loss(recon, clean, reduction="none")  # [B, T, F]
    per_packet = sl1.mean(dim=2)                     # [B, T]
    per_packet = per_packet * mask.to(per_packet.dtype)
    valid = mask.to(per_packet.dtype).sum(dim=1).clamp(min=1)
    per_flow = per_packet.sum(dim=1) / valid
    return per_flow.mean()


def _train_one_epoch(model, loader, optimizer, scaler, device, use_amp) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    for adv, clean, _lengths, mask, _rows in loader:
        adv = adv.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device_type=amp_device, enabled=use_amp):
            recon = model(adv, key_padding_mask=~mask)
            loss = _recon_loss(recon, clean, mask)
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
def _eval_one_epoch(model, loader, device, use_amp) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    for adv, clean, _lengths, mask, _rows in loader:
        adv = adv.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with _autocast(device_type=amp_device, enabled=use_amp):
            recon = model(adv, key_padding_mask=~mask)
            loss = _recon_loss(recon, clean, mask)
        total_loss += float(loss.detach().item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def train(
    config: dict,
    tensors_root: Path,
    output_dir: Path,
) -> dict:
    """Train + persist artifacts. Returns a small status dict."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["split"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_cfg = dict(config.get("training", {}))
    model_cfg = dict(config.get("model", {}))

    device = _resolve_device(str(train_cfg.get("device", "auto")))
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"

    # ---- 1. Build matched (adv → clean) pair index ---------------------------
    pairs = build_pair_index(
        tensors_dir=tensors_root,
        adversarial_dir=Path(config["adversarial_dir"]),
        adversarial_exclude=list(config.get("adversarial_exclude", [])),
        min_len=int(config.get("min_len", 2)),
    )
    log.info("pair index: %d pairs over %d adversarial source(s); skipped=%s",
             len(pairs), len(pairs.adv_folders), pairs.skipped)

    # ---- 2. Flow-level split (stratified by adversarial source) --------------
    split: Split = flow_level_split(
        sources=pairs.adv_source,
        flow_ids=pairs.flow_id,
        train_frac=float(config["split"]["train_frac"]),
        val_frac=float(config["split"]["val_frac"]),
        test_frac=float(config["split"]["test_frac"]),
        seed=seed,
    )
    summary = pair_split_summary(pairs, split.train_idx, split.val_idx, split.test_idx)

    train_idx = split.train_idx
    val_idx = split.val_idx
    test_idx = split.test_idx

    # ---- 3. DataLoaders ------------------------------------------------------
    train_loader, val_loader, _test_loader = make_pair_dataloaders(
        pairs=pairs,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=int(train_cfg.get("batch_size", 32)),
        num_workers=resolve_num_workers(train_cfg.get("num_workers", "auto")),
        pin_memory=device.type == "cuda",
        max_length=train_cfg.get("max_length", None),
    )

    # ---- 4. Model + optimiser ------------------------------------------------
    model = Reconstructor(
        num_features=int(model_cfg.get("num_features", 1090)),
        d_model=int(model_cfg.get("d_model", 128)),
        n_layers=int(model_cfg.get("n_layers", 3)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        dim_feedforward=int(model_cfg.get("dim_feedforward", 256)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        max_seq_len=int(model_cfg.get("max_seq_len", 200)),
    ).to(device)
    n_params = num_parameters(model)
    log.info("model: %d params, d_model=%d, layers=%d (causal residual denoiser)",
             n_params, model.d_model, int(model_cfg.get("n_layers", 3)))

    epochs = int(train_cfg.get("epochs", 20))
    lr = float(train_cfg.get("lr", 3e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler_kind = str(train_cfg.get("scheduler", "cosine"))
    if scheduler_kind == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    elif scheduler_kind == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2)
    else:
        raise ValueError(f"unknown scheduler: {scheduler_kind}")

    scaler = _GradScaler(enabled=use_amp)

    # ---- 5. Training loop ----------------------------------------------------
    early_stop_patience = int(train_cfg.get("early_stop_patience", 3))
    best = {"epoch": -1, "val_loss": math.inf, "state": None}
    patience_left = early_stop_patience
    curves: list[dict] = []
    t_train_start = time.time()
    for epoch in range(epochs):
        t0 = time.time()
        train_loss = _train_one_epoch(
            model, train_loader, optimizer, scaler, device, use_amp,
        )
        val_loss = _eval_one_epoch(model, val_loader, device, use_amp)
        if scheduler_kind == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()
        epoch_t = time.time() - t0
        cur_lr = float(optimizer.param_groups[0]["lr"])
        curves.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": cur_lr,
            "seconds": epoch_t,
        })
        improved = val_loss < best["val_loss"]
        log.info(
            "epoch %02d  train_loss=%.5f  val_loss=%.5f  lr=%.2e  (%.1fs)%s",
            epoch, train_loss, val_loss, cur_lr, epoch_t, "  *" if improved else "")
        if improved:
            best = {
                "epoch": epoch,
                "val_loss": val_loss,
                "state": copy.deepcopy(model.state_dict()),
            }
            patience_left = early_stop_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                log.info("early stop at epoch %d (no val_loss improvement for %d epochs)",
                         epoch, early_stop_patience)
                break

    train_seconds = time.time() - t_train_start
    if best["state"] is None:
        best = {"epoch": 0, "val_loss": float("nan"),
                "state": copy.deepcopy(model.state_dict())}

    # ---- 6. Persist artifacts ------------------------------------------------
    torch.save(best["state"], output_dir / "reconstructor.pt")
    # split.npz indexes into the deterministic pair table; evaluate.py rebuilds
    # the identical table from config (no need to persist the heavy index).
    np.savez(
        output_dir / "split.npz",
        train_idx=split.train_idx,
        val_idx=split.val_idx,
        test_idx=split.test_idx,
    )
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "training_curves.json").write_text(json.dumps(curves, indent=2))
    (output_dir / "reconstructor_config.json").write_text(json.dumps(
        {"model": model_cfg, "training": train_cfg,
         "n_parameters": int(n_params)},
        indent=2, default=str,
    ))
    (output_dir / "training_info.json").write_text(json.dumps(
        {
            "best_epoch": int(best["epoch"]),
            "best_val_loss": float(best["val_loss"]),
            "epochs_run": len(curves),
            "epochs_planned": epochs,
            "train_seconds": float(train_seconds),
            "device": str(device),
            "amp": bool(use_amp),
            "n_parameters": int(n_params),
            "n_pairs": int(len(pairs)),
            "pair_skipped": pairs.skipped,
            "split_summary": summary,
            "config": {k: v for k, v in config.items() if k not in {"training", "model"}},
            "training_config": train_cfg,
            "model_config": model_cfg,
        },
        indent=2, default=str,
    ))

    log.info("artifacts written to %s", output_dir)
    return {
        "best_epoch": int(best["epoch"]),
        "best_val_loss": float(best["val_loss"]),
        "epochs_run": len(curves),
        "n_pairs": int(len(pairs)),
        "split_summary": summary,
        "train_seconds": float(train_seconds),
        "device": str(device),
        "n_parameters": int(n_params),
    }
