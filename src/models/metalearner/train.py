"""Training loop for the Mini-Transformer meta-learner.

Trains on the seed=42 ``val`` split — out-of-fold for both RF and TCN by
construction — sub-split into a ``meta_train`` / ``meta_val`` partition
(seed=43 by default) so we have a held-out signal for early stopping.

The actual ``test`` split is never seen at training time; final
evaluation lives in ``evaluate.py``.

Streaming AUROC@10 on the meta-val partition is the early-stopping
signal (last-timestep macro-F1 saturates near 1.0 because RF's last
checkpoint is already near-perfect, so it would be uninformative).
"""
from __future__ import annotations

import json
import logging
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..common.streaming_eval import StreamingPredictions, streaming_metrics
from .dataset import (
    MetaLearnerDataset,
    TokenCache,
    collate_padded,
    prepare_split_caches,
)
from .model import MetaLearner, num_parameters


log = logging.getLogger(__name__)


def _resolve_device(cfg: str) -> torch.device:
    if cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg)


def _split_meta(
    n_flows: int, meta_val_frac: float, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_flows)
    n_val = int(round(n_flows * meta_val_frac))
    return perm[n_val:].astype(np.int64), perm[:n_val].astype(np.int64)


def _class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = Counter(int(y) for y in labels)
    n = sum(counts.values())
    n_classes = max(counts) + 1 if counts else 2
    weights = torch.ones(n_classes, dtype=torch.float32)
    for c in range(n_classes):
        cnt = counts.get(c, 0)
        weights[c] = n / (n_classes * cnt) if cnt else 1.0
    return weights


def _per_packet_loss(
    logits: torch.Tensor,        # [B, T, C]
    labels: torch.Tensor,        # [B]
    lengths: torch.Tensor,       # [B]
    class_weight: torch.Tensor,  # [C]
) -> torch.Tensor:
    """Per-packet cross-entropy averaged within each flow, then across flows.

    Padding positions (those past ``lengths[i]``) are excluded from the
    per-flow average so they don't pull the loss toward the padding
    label (always 0).
    """
    B, T, C = logits.shape
    device = logits.device
    arange = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    valid = arange < lengths.unsqueeze(1).to(device)  # [B, T] bool
    targets = labels.to(device).unsqueeze(1).expand(B, T).contiguous()
    loss_flat = nn.functional.cross_entropy(
        logits.reshape(B * T, C), targets.reshape(B * T),
        weight=class_weight.to(device), reduction="none",
    ).reshape(B, T)
    loss_flat = loss_flat * valid.float()
    per_flow = loss_flat.sum(dim=1) / lengths.to(device).clamp(min=1).float()
    return per_flow.mean()


@torch.inference_mode()
def _streaming_auroc_at(
    model: nn.Module,
    cache: TokenCache,
    sample_indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    t_for_early_stop: int = 10,
) -> tuple[float, dict]:
    """Compute streaming AUROC@``t_for_early_stop`` on ``sample_indices``.

    Returns ``(auroc, full_metrics_dict)``. The streaming metrics are
    computed via the canonical ``streaming_metrics()`` helper so the
    early-stopping signal is the same number that ends up in the final
    ``streaming_metrics.json``.
    """
    ds = MetaLearnerDataset(cache, sample_indices=sample_indices)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_padded,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    model.eval()
    per_flow_probs: list[np.ndarray] = []
    per_flow_pkt: list[np.ndarray] = []
    labels_list: list[int] = []
    lengths_list: list[int] = []
    sources_list: list[str] = []
    flow_pos = 0
    for tokens, labels, lengths, mask in loader:
        tokens = tokens.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits = model(tokens, key_padding_mask=mask)        # [B, T, 2]
        probs = torch.softmax(logits.float(), dim=2)[:, :, 1]
        probs_cpu = probs.detach().cpu().numpy()
        for b in range(probs_cpu.shape[0]):
            L = int(lengths[b].item())
            per_flow_probs.append(probs_cpu[b, :L].astype(np.float64))
            per_flow_pkt.append(np.arange(1, L + 1, dtype=np.int64))
            global_idx = int(sample_indices[flow_pos])
            labels_list.append(int(cache.labels[global_idx]))
            lengths_list.append(L)
            sources_list.append(str(cache.sources[global_idx]))
            flow_pos += 1
    sp = StreamingPredictions(
        probs=per_flow_probs,
        packet_idx=per_flow_pkt,
        labels=np.asarray(labels_list, dtype=np.int64),
        sources=np.asarray(sources_list, dtype=object),
        lengths=np.asarray(lengths_list, dtype=np.int64),
    )
    metrics = streaming_metrics(sp)
    auroc = metrics["overall"]["auroc_at"].get(str(t_for_early_stop), {}).get("auroc")
    return float(auroc) if auroc is not None else float("nan"), metrics


def train(output_dir: Path, config: dict) -> dict:
    """Top-level meta-learner training entry point."""
    output_dir.mkdir(parents=True, exist_ok=True)

    rf_dir = Path(config["rf_model_dir"])
    tcn_dir = Path(config["tcn_model_dir"])
    tensors_dir = Path(config["tensors_dir"])
    adversarial_dir = Path(config["adversarial_dir"])

    train_cfg = dict(config.get("training", {}))
    model_cfg = dict(config.get("model", {}))
    token_cfg = dict(config.get("token", {}))
    split_cfg = dict(config.get("meta_split", {}))

    device = _resolve_device(str(train_cfg.get("device", "auto")))
    log.info("device: %s", device)

    # ---- Materialise the (val, test) token caches once ----
    rf_split = np.load(rf_dir / "split.npz")
    tcn_idx = np.load(tcn_dir / "dataset_index.npz", allow_pickle=True)
    splits_arg = {
        "val": {
            "rf_sample_indices": rf_split["val_idx"],
            "tcn_sample_indices": tcn_idx["used_val_idx"],
        },
        "test": {
            "rf_sample_indices": rf_split["test_idx"],
            "tcn_sample_indices": tcn_idx["used_test_idx"],
        },
    }
    log.info("preparing token caches under %s/cache/", output_dir)
    caches = prepare_split_caches(
        output_dir=output_dir,
        rf_dir=rf_dir, tcn_dir=tcn_dir,
        tensors_dir=tensors_dir, adversarial_dir=adversarial_dir,
        splits=splits_arg,
        rf_age_clip=int(token_cfg.get("rf_age_clip", 30)),
        pkt_len_norm=float(token_cfg.get("pkt_len_norm", 1500.0)),
        max_length=int(train_cfg.get("max_length", 200)),
    )
    val_cache = caches["val"]

    # ---- meta-train / meta-val sub-split of val ----
    meta_train_idx, meta_val_idx = _split_meta(
        n_flows=len(val_cache),
        meta_val_frac=float(split_cfg.get("meta_val_frac", 0.15)),
        seed=int(split_cfg.get("seed", 43)),
    )
    log.info(
        "meta-train: %d flows, meta-val: %d flows",
        len(meta_train_idx), len(meta_val_idx),
    )
    # Persist split for reproducibility.
    np.savez(
        output_dir / "split.npz",
        meta_train_idx=meta_train_idx,
        meta_val_idx=meta_val_idx,
    )

    # ---- Class weights (from meta-train labels only) ----
    weight_mode = str(train_cfg.get("class_weight", "balanced"))
    if weight_mode == "balanced":
        class_weight = _class_weights(val_cache.labels[meta_train_idx])
    else:
        class_weight = torch.ones(2, dtype=torch.float32)
    log.info("class weights: %s", class_weight.tolist())

    # ---- Datasets / loaders ----
    train_ds = MetaLearnerDataset(val_cache, sample_indices=meta_train_idx)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 128)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        persistent_workers=int(train_cfg.get("num_workers", 4)) > 0,
        collate_fn=collate_padded,
    )

    # ---- Model ----
    model = MetaLearner(
        d_model=int(model_cfg.get("d_model", 64)),
        n_layers=int(model_cfg.get("n_layers", 2)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        dim_feedforward=int(model_cfg.get("dim_feedforward", 128)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        num_classes=int(model_cfg.get("num_classes", 2)),
        max_seq_len=int(model_cfg.get("max_seq_len", 200)),
    ).to(device)
    log.info("metalearner parameters: %d", num_parameters(model))

    # ---- Optimiser + scheduler ----
    lr = float(train_cfg.get("lr", 3e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    epochs = int(train_cfg.get("epochs", 15))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs),
    )

    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- Training loop with early stopping ----
    best_auroc = -math.inf
    best_state: dict[str, torch.Tensor] = {}
    best_epoch = -1
    patience = int(train_cfg.get("early_stop_patience", 3))
    no_improve = 0
    early_stop_t = int(train_cfg.get("early_stop_t", 10))
    curves: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        ep_n = 0
        for tokens, labels, lengths, mask in train_loader:
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, enabled=use_amp,
            ):
                logits = model(tokens, key_padding_mask=mask)
                loss = _per_packet_loss(logits, labels, lengths, class_weight)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            ep_loss += float(loss.detach().cpu()) * len(labels)
            ep_n += int(len(labels))
        scheduler.step()
        train_loss = ep_loss / max(1, ep_n)

        meta_val_auroc, meta_val_metrics = _streaming_auroc_at(
            model, val_cache, meta_val_idx, device,
            batch_size=int(train_cfg.get("batch_size", 128)),
            num_workers=int(train_cfg.get("num_workers", 4)),
            t_for_early_stop=early_stop_t,
        )
        edp = meta_val_metrics["overall"]["edp"].get("edp_median")
        log.info(
            "epoch %d: train_loss=%.4f  meta_val_auroc@%d=%.4f  "
            "meta_val_edp_median=%s  lr=%.2e",
            epoch, train_loss, early_stop_t, meta_val_auroc, edp,
            optimizer.param_groups[0]["lr"],
        )
        curves.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "meta_val_auroc_at": {str(early_stop_t): meta_val_auroc},
            "meta_val_edp_median": edp,
            "lr": optimizer.param_groups[0]["lr"],
        })

        if meta_val_auroc > best_auroc:
            best_auroc = meta_val_auroc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(
                    "early stopping at epoch %d (best=%d, AUROC@%d=%.4f)",
                    epoch, best_epoch, early_stop_t, best_auroc,
                )
                break

    # ---- Persist artefacts ----
    if best_state:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), output_dir / "metalearner.pt")
    (output_dir / "metalearner_config.json").write_text(
        json.dumps(
            {"model": model_cfg, "training": train_cfg,
             "token": token_cfg, "meta_split": split_cfg,
             "best_epoch": best_epoch, "best_meta_val_auroc_at": {
                 str(early_stop_t): best_auroc},
             "n_parameters": num_parameters(model)},
            indent=2, default=str,
        )
    )
    (output_dir / "training_curves.json").write_text(
        json.dumps(curves, indent=2)
    )
    log.info("artifacts written to %s", output_dir)
    return {
        "best_epoch": best_epoch,
        "best_meta_val_auroc": best_auroc,
        "n_parameters": num_parameters(model),
    }


__all__ = ["train"]
