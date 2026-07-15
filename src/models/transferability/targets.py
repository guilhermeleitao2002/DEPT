"""The three targets a single perturbation is scored against: RF, TCN, stack.

The comparison RQ1 asks for only means something if all three see the *same*
perturbation. So the attacker produces one ``(k bytes, d seconds)`` per packet —
crafted against the TCN alone — and this module replays it through each target's
own featurisation:

* **TCN**   — reads the perturbed ``[L, 1090]`` tensor directly.
* **RF**    — reads 24 prefix statistics recomputed from perturbed per-packet
  values (via the sidecar), because its features are not in the tensor.
* **Stack** — reads per-packet 13-dim tokens fusing both base learners' logits.

Nothing here is differentiable, and nothing needs to be: the attacker's gradient
path ends at the TCN. That asymmetry is the experiment.

Flow verdicts follow the convention the trained models already report: the
prediction after seeing the whole flow — ``probs[b, L-1]`` for the TCN and the
meta-learner (see ``tcn/evaluate.py``, ``metalearner/evaluate.py``), and the last
checkpoint for the RF.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..metalearner.base_inference import JoinedFlow
from ..metalearner.dataset import (
    DEFAULT_PKT_LEN_NORM,
    DEFAULT_RF_AGE_CLIP,
    TOKEN_DIM,
    build_flow_tokens,
)
from ..metalearner.model import MetaLearner
from ..tcn.evaluate import _resolve_device
from ..tcn.model import TCN

log = logging.getLogger(__name__)

TARGETS = ("rf", "tcn", "stack")


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LoadedModels:
    tcn: TCN
    meta: MetaLearner
    rf: object                    # sklearn RandomForestClassifier
    device: torch.device
    max_length: int
    rf_age_clip: int
    pkt_len_norm: float


def load_models(
    rf_dir: Path,
    tcn_dir: Path,
    meta_dir: Path,
    device_cfg: str = "auto",
) -> LoadedModels:
    """Load the three trained targets from their pipeline artifacts."""
    import joblib

    device = _resolve_device(device_cfg)

    tcn_cfg = json.loads((Path(tcn_dir) / "tcn_config.json").read_text())
    m = tcn_cfg.get("model", {})
    t = tcn_cfg.get("training", {})
    tcn = TCN(
        num_features=int(m.get("num_features", 1090)),
        channels=tuple(m.get("channels", (128, 128, 128))),
        kernel_size=int(m.get("kernel_size", 3)),
        dropout=float(m.get("dropout", 0.15)),
        num_classes=int(m.get("num_classes", 2)),
    ).to(device)
    tcn.load_state_dict(torch.load(Path(tcn_dir) / "tcn.pt", map_location=device))
    tcn.eval()

    meta_cfg = json.loads((Path(meta_dir) / "metalearner_config.json").read_text())
    mm = meta_cfg.get("model", {})
    tok = meta_cfg.get("token", {})
    meta = MetaLearner(
        d_model=int(mm.get("d_model", 64)),
        n_layers=int(mm.get("n_layers", 2)),
        n_heads=int(mm.get("n_heads", 4)),
        dim_feedforward=int(mm.get("dim_feedforward", 128)),
        dropout=float(mm.get("dropout", 0.1)),
        num_classes=int(mm.get("num_classes", 2)),
        max_seq_len=int(mm.get("max_seq_len", 200)),
    ).to(device)
    meta.load_state_dict(
        torch.load(Path(meta_dir) / "metalearner.pt", map_location=device))
    meta.eval()

    rf = joblib.load(Path(rf_dir) / "rf.joblib")

    return LoadedModels(
        tcn=tcn, meta=meta, rf=rf, device=device,
        max_length=int(t.get("max_length", 200)),
        rf_age_clip=int(tok.get("rf_age_clip", DEFAULT_RF_AGE_CLIP)),
        pkt_len_norm=float(tok.get("pkt_len_norm", DEFAULT_PKT_LEN_NORM)),
    )


# ──────────────────────────────────────────────────────────────────────────────
# RF
# ──────────────────────────────────────────────────────────────────────────────

def rf_predict_features(clf, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``X [n, 24] -> (probs, logits)`` for P(adversarial).

    ``rf/evaluate.predict_streams`` cannot serve here: it reads *precomputed*
    rows out of ``dataset.npz`` (``X = data["X"][sample_indices]``), so it can
    only score flows exactly as they were tensorised. Perturbed features have no
    row to look up. The logit clipping mirrors it so the two agree at delta=0.
    """
    if len(X) == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    probs = clf.predict_proba(np.asarray(X, dtype=np.float32))[:, 1]
    p = np.clip(probs, 1e-6, 1 - 1e-6)
    return probs.astype(np.float64), np.log(p / (1 - p))


# ──────────────────────────────────────────────────────────────────────────────
# TCN
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def score_tcn(
    model: TCN,
    seq: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """``seq [B, T, 1090] -> (per-flow probs, per-flow logits)``, valid steps only.

    ``logits`` is the binary margin ``logit_adv - logit_benign``, matching what
    ``tcn/evaluate.predict_streams`` hands the meta-learner.
    """
    out = model(seq)                                   # [B, T, 2]
    probs = torch.softmax(out.float(), dim=2)[:, :, 1]
    margin = out[..., 1] - out[..., 0]
    probs_np = probs.cpu().numpy()
    margin_np = margin.float().cpu().numpy()
    p_out, l_out = [], []
    for b in range(probs_np.shape[0]):
        L = int(lengths[b].item())
        p_out.append(probs_np[b, :L].astype(np.float64))
        l_out.append(margin_np[b, :L].astype(np.float32))
    return p_out, l_out


# ──────────────────────────────────────────────────────────────────────────────
# Stack (RF + TCN + meta-learner)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def score_stack(
    meta: MetaLearner,
    seq: torch.Tensor,
    lengths: torch.Tensor,
    tcn_logits: list[np.ndarray],
    rf_logits: list[np.ndarray],
    rf_packet_idx: list[np.ndarray],
    sources: list[str],
    flow_ids: list[str],
    labels: list[int],
    rf_age_clip: int = DEFAULT_RF_AGE_CLIP,
    pkt_len_norm: float = DEFAULT_PKT_LEN_NORM,
    seq_np: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Per-flow P(adversarial) streams from the fused stack.

    Tokens are built with ``metalearner.dataset.build_flow_tokens`` **verbatim**
    — the same code the meta-learner was trained through, fed the perturbed
    tensor instead of the on-disk one. ``join_streams`` is deliberately not used:
    it reads base-learner predictions off disk, which by construction are the
    unperturbed ones.

    ``seq_np`` lets a caller supply the host-side copy of ``seq``. The token
    builder is numpy, so scoring needs one; the ablations in
    :mod:`~.pipeline` score the *same* perturbed tensor three times, and without
    this each call would re-download an identical ``[B, T, 1090]`` block
    (~56 MB at the default batch) across the PCIe bus.
    """
    device = seq.device
    if seq_np is None:
        seq_np = seq.detach().cpu().numpy()

    token_list: list[np.ndarray] = []
    for b in range(seq_np.shape[0]):
        L = int(lengths[b].item())
        rf_pkt = np.asarray(rf_packet_idx[b], dtype=np.int64)
        rf_log = np.asarray(rf_logits[b], dtype=np.float64)
        # Drop checkpoints past the (possibly truncated) flow horizon, exactly as
        # join_streams does — tokens past L have no packet to attach to.
        if rf_pkt.size and rf_pkt[-1] > L:
            keep = rf_pkt <= L
            rf_pkt, rf_log = rf_pkt[keep], rf_log[keep]
        flow = JoinedFlow(
            source=str(sources[b]),
            flow_id=str(flow_ids[b]),
            label=int(labels[b]),
            length=L,
            tcn_packet_idx=np.arange(1, L + 1, dtype=np.int64),
            tcn_logit=np.asarray(tcn_logits[b], dtype=np.float32),
            rf_packet_idx=rf_pkt,
            rf_logit=rf_log,
        )
        token_list.append(
            build_flow_tokens(
                flow, seq_np[b], flow_start=0,
                rf_age_clip=rf_age_clip, pkt_len_norm=pkt_len_norm,
            )
        )

    B = len(token_list)
    T_max = max(int(t.shape[0]) for t in token_list)
    tokens = torch.zeros(B, T_max, TOKEN_DIM, dtype=torch.float32)
    key_padding = torch.ones(B, T_max, dtype=torch.bool)   # True = padded
    for b, t in enumerate(token_list):
        L = t.shape[0]
        tokens[b, :L] = torch.from_numpy(t)
        key_padding[b, :L] = False

    logits = meta(tokens.to(device), key_padding_mask=key_padding.to(device))
    probs = torch.softmax(logits.float(), dim=2)[:, :, 1].cpu().numpy()
    return [probs[b, : token_list[b].shape[0]].astype(np.float64)
            for b in range(B)]


# ──────────────────────────────────────────────────────────────────────────────
# Flow verdicts
# ──────────────────────────────────────────────────────────────────────────────

def flow_verdict(stream: np.ndarray) -> float:
    """The model's P(adversarial) after seeing the whole flow.

    The convention every trained target already reports: the TCN and meta-learner
    take ``probs[b, L-1]``; RF's stream ends at its last checkpoint.
    """
    if len(stream) == 0:
        return float("nan")
    return float(stream[-1])


def never_fires(stream: np.ndarray, threshold: float) -> bool:
    """True if the flow never crossed the threshold at any emission.

    The stricter, streaming-honest notion of evasion: a detector that fired at
    packet 5 and relaxed by packet 200 has still detected the flow. Reported
    alongside :func:`flow_verdict` rather than instead of it.
    """
    if len(stream) == 0:
        return True
    return bool(np.nanmax(stream) < threshold)


__all__ = [
    "TARGETS",
    "LoadedModels",
    "load_models",
    "rf_predict_features",
    "score_tcn",
    "score_stack",
    "flow_verdict",
    "never_fires",
]
