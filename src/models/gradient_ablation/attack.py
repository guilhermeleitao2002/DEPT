"""Projected gradient descent in physical units (bytes and seconds).

An L-inf ball on the raw tensor does not make sense here: the IPAT channel is
z-scored with std = 208.2 s while the nPrint channels are bits, so
``eps = 0.01`` would allow +/-2.08 s of delay and nothing in the bits. The
attacker's variables are instead ``k`` bytes of padding and ``d`` seconds of
delay per packet. The feasible set (increase-only, integer bytes, MTU cap, no
change on non-IPv4 packets) is a projection that ART's binary ``mask`` cannot
express, so the loop is written here; ART's AutoPGD is only used as a check
on the relaxed problem (:func:`art_autopgd_relaxed`).

The attack is increase-only because the packets are already sent, and because
DEPT's positive class is TANTGen traffic: removing the perturbation would
evade DEPT trivially but would also stop evading the protected NIDS.

Packet size reaches the TCN as the 16 IPv4 total-length bits, which have no
useful gradient. :class:`_EncodeTotalLength` uses the pseudoinverse of the
decode map in the backward pass, ``d bits_j / d tl = w_j / ||w||^2`` with
``w_j = 2^(15-j)``, i.e. the gradient with respect to the length value.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

log = logging.getLogger(__name__)

# nPrint absolute bit positions. Mirrors realize._IDX_IPV4_TL / _IDX_IPAT and
# metalearner.dataset._IDX_*, which _verify_nprint_layout() asserts at import.
IDX_IPV4_TL = (16, 32)
IDX_IPAT = 1088
TL_BITS = 16

# One float32 ULP of the z-scored IPAT channel, in seconds (2**-27 * 208.22 s,
# rounded up). Smaller delays round to zero in the tensor.
IPAT_ULP_SECONDS = 1.552e-06

# Ethernet MTU (captures contain larger, GRO-coalesced frames).
DEFAULT_TL_CAP = 1500


def _place_values(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """``[2^15, 2^14, ..., 2^0]`` — nPrint stores total length MSB-first."""
    return (2.0 ** torch.arange(TL_BITS - 1, -1, -1, device=device, dtype=dtype))


class _EncodeTotalLength(torch.autograd.Function):
    """``tl [B, T] -> bits [B, T, 16]``, differentiable via the decode pseudoinverse."""

    @staticmethod
    def forward(ctx, tl: torch.Tensor) -> torch.Tensor:
        w = _place_values(tl.device, torch.float32)
        ctx.save_for_backward(w)
        v = tl.detach().round().clamp_(0, 2 ** TL_BITS - 1).to(torch.int64)
        shifts = torch.arange(TL_BITS - 1, -1, -1, device=tl.device, dtype=torch.int64)
        bits = (v.unsqueeze(-1) >> shifts) & 1
        return bits.to(tl.dtype)

    @staticmethod
    def backward(ctx, grad_bits: torch.Tensor) -> torch.Tensor:
        (w,) = ctx.saved_tensors
        # Minimum-norm J with sum_j J_j * w_j = 1  =>  J = w / ||w||^2.
        j = w / (w * w).sum()
        return (grad_bits * j.to(grad_bits.dtype)).sum(dim=-1)


def encode_total_length(tl: torch.Tensor) -> torch.Tensor:
    return _EncodeTotalLength.apply(tl)


def decode_total_length(seq: torch.Tensor) -> torch.Tensor:
    """``seq [B, T, 1090] -> tl [B, T]``. nPrint ``-1`` ("layer absent") -> 0.

    Matches ``realize.decode_total_len_batch`` and
    ``metalearner.dataset._decode_pkt_len``.
    """
    bits = seq[..., IDX_IPV4_TL[0]:IDX_IPV4_TL[1]].clamp(min=0.0)
    return bits @ _place_values(seq.device, bits.dtype)


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    """Round forward, identity backward — on the **scalar**, not on the bits."""
    return x + (x.round() - x).detach()


@dataclass
class FlowBatch:
    """One padded batch of base (unperturbed) flows and its perturbable surface."""
    seq: torch.Tensor          # [B, T, 1090] float32, right-padded
    mask: torch.Tensor         # [B, T] bool, True at valid timesteps
    lengths: torch.Tensor      # [B] int64
    orig_tl: torch.Tensor      # [B, T] float32, decoded IPv4 total length
    ip_mask: torch.Tensor      # [B, T] bool, True where IPv4 is present

    @classmethod
    def from_padded(cls, seq: torch.Tensor, mask: torch.Tensor,
                    lengths: torch.Tensor) -> "FlowBatch":
        tl_bits = seq[..., IDX_IPV4_TL[0]:IDX_IPV4_TL[1]]
        # Packets without IPv4 (nPrint -1) cannot be padded.
        ip_mask = (tl_bits >= 0).all(dim=-1) & mask
        return cls(seq=seq, mask=mask, lengths=lengths,
                   orig_tl=decode_total_length(seq), ip_mask=ip_mask)


@dataclass
class Budget:
    """Per-packet attacker budget, in the units the attacker actually pays."""
    bytes_: float              # max padding per packet
    seconds: float             # max added delay per packet

    def is_null(self) -> bool:
        return self.bytes_ <= 0 and self.seconds <= 0


def project(
    k: torch.Tensor,
    d: torch.Tensor,
    batch: FlowBatch,
    budget: Budget,
    tl_cap: int = DEFAULT_TL_CAP,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project ``(k, d)`` onto the feasible set.

    * increase-only: ``k, d >= 0``
    * per-packet budget: ``k <= budget.bytes_``, ``d <= budget.seconds``
    * MTU: ``orig_tl + k <= tl_cap``
    * ``k = d = 0`` on padding and on non-IPv4 packets

    ``k`` stays continuous and is only rounded in the forward pass, so small
    steps can accumulate.
    """
    headroom = (tl_cap - batch.orig_tl).clamp(min=0.0)
    k = k.clamp(min=0.0).minimum(headroom.clamp(max=float(budget.bytes_)))
    d = d.clamp(min=0.0, max=float(budget.seconds))
    k = torch.where(batch.ip_mask, k, torch.zeros_like(k))
    d = torch.where(batch.mask, d, torch.zeros_like(d))
    return k, d


def apply_delta(
    batch: FlowBatch,
    k: torch.Tensor,
    d: torch.Tensor,
    ipat_std: float,
    tl_cap: int = DEFAULT_TL_CAP,
) -> torch.Tensor:
    """Write ``(k bytes, d seconds)`` into a copy of the base tensor.

    Only the IPv4 total-length bits and the IPAT channel change, which are the
    two values the realizer can write back. Adding ``d`` seconds shifts the
    z-scored IPAT by ``d / std``.
    """
    base = batch.seq          # not cloned: torch.cat below allocates the output

    tl = batch.orig_tl + _round_ste(k)
    tl = torch.where(batch.ip_mask, tl.clamp(max=float(tl_cap)), batch.orig_tl)
    bits = encode_total_length(tl)
    # Preserve nPrint's -1 "layer absent" marker on non-IPv4 packets.
    bits = torch.where(batch.ip_mask.unsqueeze(-1), bits,
                       base[..., IDX_IPV4_TL[0]:IDX_IPV4_TL[1]])
    ipat = base[..., IDX_IPAT] + d / float(ipat_std)

    # Splice both edited fields in a single concatenation.
    seq = torch.cat([
        base[..., :IDX_IPV4_TL[0]],
        bits,
        base[..., IDX_IPV4_TL[1]:IDX_IPAT],
        ipat.unsqueeze(-1),
        base[..., IDX_IPAT + 1:],
    ], dim=-1)

    return seq * batch.mask.unsqueeze(-1).to(seq.dtype)


# ──────────────────────────────────────────────────────────────────────────────
# PGD
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PGDConfig:
    steps: int = 40
    alpha_ratio: float = 2.5      # step = alpha_ratio * budget / steps (Madry et al.)
    random_init: bool = True
    restarts: int = 2
    seed: int = 42


def _pgd_once(
    objective,
    batch: FlowBatch,
    budget: Budget,
    ipat_std: float,
    cfg: PGDConfig,
    tl_cap: int,
    k: torch.Tensor,
    d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One PGD run from a given feasible start; returns the best iterate per flow."""
    k, d = project(k, d, batch, budget, tl_cap)
    with torch.no_grad():
        obj_best = objective(apply_delta(batch, k, d, ipat_std, tl_cap))
    k_best, d_best = k.clone(), d.clone()

    if budget.is_null() or cfg.steps <= 0:
        return k_best, d_best, obj_best

    alpha_k = cfg.alpha_ratio * float(budget.bytes_) / max(cfg.steps, 1)
    alpha_d = cfg.alpha_ratio * float(budget.seconds) / max(cfg.steps, 1)

    for _ in range(cfg.steps):
        k = k.detach().requires_grad_(True)
        d = d.detach().requires_grad_(True)
        obj = objective(apply_delta(batch, k, d, ipat_std, tl_cap))
        grad_k, grad_d = torch.autograd.grad(obj.sum(), [k, d], allow_unused=True)

        with torch.no_grad():
            # Sign steps: the update is then invariant to the decode
            # pseudoinverse's scale, which fixes direction but whose magnitude
            # (~1e-5) carries no information.
            if grad_k is not None:
                k = k - alpha_k * grad_k.sign()
            if grad_d is not None:
                d = d - alpha_d * grad_d.sign()
            k, d = project(k, d, batch, budget, tl_cap)

            obj_new = objective(apply_delta(batch, k, d, ipat_std, tl_cap))
            better = obj_new < obj_best
            obj_best = torch.where(better, obj_new, obj_best)
            k_best = torch.where(better.unsqueeze(-1), k, k_best)
            d_best = torch.where(better.unsqueeze(-1), d, d_best)

    return k_best, d_best, obj_best


def pgd(
    objective,
    batch: FlowBatch,
    budget: Budget,
    ipat_std: float,
    cfg: PGDConfig,
    tl_cap: int = DEFAULT_TL_CAP,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Minimise ``objective(seq_pert) -> [B]`` over the feasible ``(k, d)``.

    ``objective`` returns one value per flow (lower = more evasive) and is always
    computed on the TCN; the RF is not differentiable.

    The first restart always starts at delta=0 and the others at random points.
    With only random starts, a large budget can start far from anything PGD can
    recover from, and the success rate then drops as the budget grows. The best
    iterate is kept per flow over all restarts and steps.
    """
    device = batch.seq.device
    B, T = batch.mask.shape
    n_restarts = max(1, cfg.restarts if cfg.random_init else 1)

    k_best = d_best = obj_best = None
    for r in range(n_restarts):
        if r == 0 or not cfg.random_init or budget.is_null():
            k0 = torch.zeros((B, T), device=device, dtype=torch.float32)
            d0 = torch.zeros((B, T), device=device, dtype=torch.float32)
        else:
            g = torch.Generator(device="cpu").manual_seed(cfg.seed + r)
            k0 = torch.rand((B, T), generator=g).to(device) * float(budget.bytes_)
            d0 = torch.rand((B, T), generator=g).to(device) * float(budget.seconds)

        k_r, d_r, obj_r = _pgd_once(
            objective, batch, budget, ipat_std, cfg, tl_cap, k0, d0)
        if obj_best is None:
            k_best, d_best, obj_best = k_r, d_r, obj_r
        else:
            better = obj_r < obj_best
            obj_best = torch.where(better, obj_r, obj_best)
            k_best = torch.where(better.unsqueeze(-1), k_r, k_best)
            d_best = torch.where(better.unsqueeze(-1), d_r, d_best)

    return k_best, d_best, obj_best


# ──────────────────────────────────────────────────────────────────────────────
# Objectives
# ──────────────────────────────────────────────────────────────────────────────

def tcn_margin_objective(model, batch: FlowBatch, reduction: str = "final"):
    """Attacker's objective: the TCN margin ``logit_adv - logit_benign``.

    * ``final`` — margin at the last valid packet, the verdict DEPT reports.
    * ``max``   — largest margin over the stream ("never fires").
    * ``mean``  — mean margin over the stream.
    """
    idx = (batch.lengths - 1).clamp(min=0)

    def _obj(seq: torch.Tensor) -> torch.Tensor:
        logits = model(seq)                            # [B, T, 2]
        margin = logits[..., 1] - logits[..., 0]       # [B, T]
        if reduction == "final":
            return margin.gather(1, idx.view(-1, 1)).squeeze(1)
        if reduction == "max":
            neg_inf = torch.finfo(margin.dtype).min
            return margin.masked_fill(~batch.mask, neg_inf).max(dim=1).values
        if reduction == "mean":
            m = batch.mask.to(margin.dtype)
            return (margin * m).sum(1) / m.sum(1).clamp(min=1)
        raise ValueError(f"unknown reduction {reduction!r}")

    return _obj


# ──────────────────────────────────────────────────────────────────────────────
# ART bridge — relaxed problem only
# ──────────────────────────────────────────────────────────────────────────────

def _slice_batch(batch: FlowBatch, i: int) -> FlowBatch:
    """Single-flow view of ``batch``, trimmed to that flow's valid length."""
    L = int(batch.lengths[i].item())
    return FlowBatch(
        seq=batch.seq[i:i + 1, :L],
        mask=batch.mask[i:i + 1, :L],
        lengths=batch.lengths[i:i + 1],
        orig_tl=batch.orig_tl[i:i + 1, :L],
        ip_mask=batch.ip_mask[i:i + 1, :L],
    )


def art_autopgd_relaxed(
    model,
    batch: FlowBatch,
    eps_bytes: float,
    ipat_std: float,
    steps: int = 40,
    tl_cap: int = DEFAULT_TL_CAP,
    max_flows: int = 32,
) -> np.ndarray | None:
    """AutoPGD (Croce & Hein, 2020) on the relaxed problem, via IBM ART.

    A check that the hand-written search is not under-tuned. It is an upper bound
    only: the relaxation drops integrality and the physical budget, so its
    perturbation cannot be realized.

    Runs one flow at a time, so that ART's internal batching cannot pair one
    flow's base tensor with another flow's perturbation.

    Returns per-flow margins (lower = more evasive), or ``None`` without ART.
    """
    try:
        from art.attacks.evasion import AutoProjectedGradientDescent
        from art.estimators.classification import PyTorchClassifier
    except ImportError:
        log.info("ART not installed — skipping the AutoPGD relaxed bound")
        return None

    device = batch.seq.device
    n = min(int(batch.seq.shape[0]), max_flows)
    margins: list[float] = []

    # ART moves the model to torch.cuda.current_device(); keep it on our device.
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)

    for i in range(n):
        fb = _slice_batch(batch, i)
        T = int(fb.mask.shape[1])
        last = T - 1

        class _Wrap(torch.nn.Module):
            """k [1, T] -> flow logits [1, 2] at the final valid timestep."""

            def __init__(self, fb_: FlowBatch) -> None:
                super().__init__()
                self.inner = model
                self._fb = fb_

            def forward(self, k: torch.Tensor) -> torch.Tensor:
                k = k.to(device).float().view(1, -1)
                seq = apply_delta(self._fb, k, torch.zeros_like(k),
                                  ipat_std, tl_cap)
                return self.inner(seq)[:, last, :]

        clf = PyTorchClassifier(
            model=_Wrap(fb),
            loss=torch.nn.CrossEntropyLoss(),
            input_shape=(T,),
            nb_classes=2,
            clip_values=(0.0, float(eps_bytes)),
            device_type="gpu" if device.type == "cuda" else "cpu",
        )
        attack = AutoProjectedGradientDescent(
            estimator=clf, norm="inf", eps=float(eps_bytes),
            eps_step=float(eps_bytes) / 4.0, max_iter=steps,
            nb_random_init=1, targeted=True, batch_size=1, verbose=False,
        )
        try:
            k_adv = attack.generate(
                x=np.zeros((1, T), dtype=np.float32),
                y=np.zeros((1,), dtype=np.int64),   # target the benign class
            )
        except Exception as e:  # pragma: no cover - ART API drift
            log.warning("AutoPGD failed on flow %d: %s", i, e)
            return None

        with torch.no_grad():
            k_t = torch.from_numpy(k_adv).to(device).float()
            k_t, _ = project(k_t, torch.zeros_like(k_t), fb,
                             Budget(bytes_=eps_bytes, seconds=0.0), tl_cap)
            logits = model(apply_delta(fb, k_t, torch.zeros_like(k_t),
                                       ipat_std, tl_cap))
            margins.append(float(logits[0, last, 1] - logits[0, last, 0]))

    return np.asarray(margins, dtype=np.float64)


__all__ = [
    "IDX_IPV4_TL",
    "IDX_IPAT",
    "IPAT_ULP_SECONDS",
    "DEFAULT_TL_CAP",
    "FlowBatch",
    "Budget",
    "PGDConfig",
    "encode_total_length",
    "decode_total_length",
    "project",
    "apply_delta",
    "pgd",
    "tcn_margin_objective",
    "art_autopgd_relaxed",
]
