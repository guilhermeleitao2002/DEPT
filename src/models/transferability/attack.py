"""Projected gradient descent in **physical units** (bytes, seconds).

Why not an off-the-shelf epsilon-ball
-------------------------------------
The obvious move — hand the 1090-dim tensor to ART's PGD with an L-inf ball — is
physically meaningless here, and the reason is worth stating because it shapes
the whole module. The IPAT channel is z-scored with **std = 211047.5 seconds**
(``data/tensors/mean_std.json``), while the 1088 nPrint channels are bits in
``{-1, 0, 1}``. An ``eps = 0.01`` ball therefore buys the attacker **±2110
seconds** of delay and nothing at all in the bit channels. Any single epsilon
prices the two channels five orders of magnitude apart.

So the optimisation variable here is the perturbation the attacker actually pays
for: ``k`` bytes of padding and ``d`` seconds of delay, per packet. The budget is
a real deployment cost, and the feasible set — increase-only, integer bytes,
capped at the MTU, zero on non-IPv4 packets — is a projection operator, which is
exactly what ART's ``mask`` kwarg (a binary include/exclude) cannot express.
Hence a hand-rolled loop. :func:`art_autopgd_relaxed` keeps ART in the picture
for the one job it does well: a parameter-free upper bound on the *relaxed*
problem, as a check that this loop is not simply under-tuned.

Increase-only
-------------
The attacker may pad and delay, never un-send: the packets are already on the
wire. This is not a convenience — it is what stops the degenerate solution. DEPT's
positive class *is* "TANTGen-perturbed", so an attacker allowed to *reduce* the
perturbation would evade by simply not perturbing, at zero cost and at the price
of losing the evasion of the protected NIDS that motivated the perturbation. The
constraint is also empirically what TANTGen itself does: every negative delta
measured in its output is exactly one float32 ULP (-3.071e-06 s), i.e. round-off.

Gradients through a 16-bit integer
----------------------------------
The TCN reads packet size as the 16 IPv4 total-length bits at ``seq[:, :, 16:32]``,
so a byte-valued perturbation has to cross a binary encoding that is piecewise
constant — zero gradient almost everywhere. :class:`_EncodeTotalLength` bridges it
with the **pseudoinverse of the decode map**: the decode is
``tl = sum_j bits_j * 2^(15-j)``, so the minimum-norm Jacobian consistent with it
is ``d bits_j / d tl = w_j / ||w||^2`` with ``w_j = 2^(15-j)``. Backward projects
the bit-gradient onto the decode direction, which is precisely "how does the loss
move if the *value* moves".

This is deliberately not a straight-through estimator over the 16 bits
independently: identity-backward would hand bit 0 (place value 32768) and bit 15
(place value 1) the same gradient, and ``tl`` 255 -> 256 flips nine bits at once,
so an L-inf step in bit space corresponds to no physical quantity whatsoever.
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

# float32 ULP of the z-scored IPAT channel near its observed range, in seconds.
# Delays below this are unrepresentable in the tensor: they quantise to zero.
IPAT_ULP_SECONDS = 3.071e-06

# Ethernet MTU. A deployment choice, not a property of the data: Packet_Size_max
# reaches 24874 in data/models/rf/dataset.npz (LRO/GRO-coalesced capture frames).
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
        # nPrint marks an absent layer with -1 across the field. A packet with no
        # IPv4 header has no total length to pad, so it is outside the attacker's
        # reach and must stay untouched.
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
    """Project ``(k, d)`` onto the feasible set. The operator ART cannot express.

    * increase-only: ``k, d >= 0``
    * per-packet budget: ``k <= budget.bytes_``, ``d <= budget.seconds``
    * MTU: ``orig_tl + k <= tl_cap`` (and never below ``orig_tl``)
    * unreachable packets: ``k = d = 0`` at padding and on non-IPv4 packets

    ``k`` stays **continuous** here; rounding to whole bytes happens only in the
    forward pass via ``_round_ste``. Rounding inside the projection would discard
    the sub-byte accumulation that lets a small step size ever cross an integer.
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

    Only two of the 1090 channels move, and they are the only two the realizer
    can write back to the wire: the IPv4 total-length bits and the IPAT channel.

    The IPAT channel is z-scored, so adding ``d`` seconds is a shift of
    ``d / std`` — the mean cancels: ``(x + d - mu)/sigma = (x - mu)/sigma + d/sigma``.
    """
    base = batch.seq          # not cloned: torch.cat below allocates the output

    tl = batch.orig_tl + _round_ste(k)
    tl = torch.where(batch.ip_mask, tl.clamp(max=float(tl_cap)), batch.orig_tl)
    bits = encode_total_length(tl)
    # Preserve nPrint's -1 "layer absent" marker on non-IPv4 packets.
    bits = torch.where(batch.ip_mask.unsqueeze(-1), bits,
                       base[..., IDX_IPV4_TL[0]:IDX_IPV4_TL[1]])
    ipat = base[..., IDX_IPAT] + d / float(ipat_std)

    # One concatenation, not two: both edited fields are spliced in a single pass
    # over the 1090 channels. This runs on every PGD forward, so a second pass
    # would allocate another [B, T, 1090] for nothing.
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

    ``objective`` returns one scalar per flow (lower = more evasive). It is the
    attacker's view of the target, and for RQ1 it is always the **TCN**: the
    forest is not differentiable and never enters this loop.

    **Restarts, and why the first one is always zero.** The feasible sets are
    nested (a budget of 8 bytes is a subset of 1460), so a *perfect* attacker's
    success can only rise with the budget. A single randomly-initialised run does
    not inherit that: at a large budget the random start lands far out in a
    region PGD cannot walk back from, and measured ASR falls as the budget grows
    — which would read as "the defense got stronger" when it only means the
    attack got sloppier. Starting one restart at delta=0 anchors every budget to
    the same point and largely restores monotonicity; the rest explore.

    The best iterate is tracked **per flow** across all restarts and steps, not
    the last one: PGD's final step is not necessarily its best, and a real
    attacker keeps whatever worked.
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
    """Attacker's objective: the TCN's adversarial margin, lower = more evasive.

    Margin is ``logit_adversarial - logit_benign``; below 0 the flow reads benign.

    * ``final`` — margin at the last valid timestep. This is the operating point
      DEPT reports: both ``tcn/evaluate`` and ``metalearner/evaluate`` take the
      flow verdict at ``probs[b, L-1]``, and RF's is its last checkpoint.
    * ``max``   — worst margin over the stream, i.e. "never fires". Strictly
      harder, and the honest goal against a *streaming* detector.
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
    """AutoPGD (Croce & Hein 2020) on the **relaxed** problem, via IBM ART.

    A parameter-free, third-party check that this module's own search is not
    simply under-tuned — the standard answer to "did you just write a weak
    attack?". It is an upper bound and deliberately never a headline: the
    relaxation drops integrality and the physical budget, so its perturbation is
    not realizable on the wire.

    Runs **one flow at a time**. ART chunks ``x`` internally by its own
    ``batch_size``, but the base tensor here is frozen context that has to stay
    row-aligned with the incoming perturbation; a batched wrapper would silently
    pair flow *i*'s padding with flow *j*'s packets on any chunk ART chose to
    split. One flow per estimator removes the ambiguity entirely, at a cost that
    is irrelevant for a capped diagnostic.

    Returns per-flow adversarial margins (lower = more evasive), or ``None`` if
    ART is not installed.
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

    # ART's PyTorchClassifier resolves "gpu" via torch.cuda.current_device() and
    # then moves the model there. Without this it would drag a cuda:1 model onto
    # cuda:0 and fail mid-forward.
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
