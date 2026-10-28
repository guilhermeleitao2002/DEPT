"""The attack loop: craft on the TCN, score on RF, TCN and stack.

For each batch of flows and each budget, PGD runs against the TCN and the
resulting ``(k bytes, d seconds)`` is replayed through each target. Two
ablations replay the same tensor through the meta-learner with one base
learner's input reset to its clean value (see
:func:`~.evaluate.reliance_shift`).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .attack import (
    Budget,
    FlowBatch,
    PGDConfig,
    apply_delta,
    pgd,
    tcn_margin_objective,
)
from .features import Sidecar, rf_feats_for_delta
from .targets import (
    LoadedModels,
    flow_verdict,
    never_fires,
    rf_predict_features,
    score_stack,
    score_tcn,
)

log = logging.getLogger(__name__)


@dataclass
class BudgetSpec:
    """One point on the cost grid the ASR curves are drawn against."""
    id: str
    bytes_: float
    seconds: float

    def to_budget(self) -> Budget:
        return Budget(bytes_=self.bytes_, seconds=self.seconds)


def parse_budgets(specs: list) -> list[BudgetSpec]:
    """Config budget entries -> :class:`BudgetSpec`.

    Accepts ``{id, bytes, seconds}`` dicts. A 0/0 budget is the delta=0 control.
    """
    out = []
    for s in specs:
        if isinstance(s, dict):
            b = float(s.get("bytes", 0.0))
            sec = float(s.get("seconds", 0.0))
            sid = str(s.get("id") or f"b{b:g}_s{sec:g}")
        else:
            b, sec, sid = float(s), 0.0, f"b{float(s):g}"
        out.append(BudgetSpec(id=sid, bytes_=b, seconds=sec))
    return out


def _load_padded_batch(
    packets: np.ndarray,
    offsets: np.ndarray,
    local_ids: list[int],
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read flows off the memmap and right-pad them, as ``collate_padded`` does."""
    blocks = []
    for li in local_ids:
        start, length = offsets[int(li)]
        L = min(int(length), max_length)
        x = np.asarray(packets[int(start):int(start) + L], dtype=np.float32)
        blocks.append(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
    lengths = np.asarray([b.shape[0] for b in blocks], dtype=np.int64)
    T = int(lengths.max())
    B, F = len(blocks), int(blocks[0].shape[1])
    seq = np.zeros((B, T, F), dtype=np.float32)
    mask = np.zeros((B, T), dtype=bool)
    for i, b in enumerate(blocks):
        seq[i, : b.shape[0]] = b
        mask[i, : b.shape[0]] = True
    return (torch.from_numpy(seq).to(device),
            torch.from_numpy(mask).to(device),
            torch.from_numpy(lengths).to(device))


def _rf_streams(
    models: LoadedModels,
    sidecar: Sidecar,
    sc_rows: list[int],
    k: np.ndarray,
    d: np.ndarray,
    lengths: np.ndarray,
    first_checkpoint: int,
    stride: int,
    pad_byte_mode: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """RF per-checkpoint ``(probs, logits, packet_idx)`` under a perturbation.

    The whole batch is scored with a single ``predict_proba`` call.
    """
    feats_all, pos_all = [], []
    for i, row in enumerate(sc_rows):
        feats, hist = sidecar.flow(row)
        # Lengths should match (see verify_sidecar); min() guards against overflow.
        L = min(int(lengths[i]), feats.shape[0])
        X, pos = rf_feats_for_delta(
            feats[:L], hist[:L], k[i, :L], d[i, :L],
            first_checkpoint, stride, pad_byte_mode=pad_byte_mode,
        )
        feats_all.append(X)
        pos_all.append(pos)

    if not feats_all or all(len(p) == 0 for p in pos_all):
        n = len(sc_rows)
        empty = [np.zeros(0) for _ in range(n)]
        return empty, empty, [np.zeros(0, dtype=np.int64) for _ in range(n)]

    X = np.concatenate([f for f in feats_all if len(f)], axis=0)
    probs, logits = rf_predict_features(models.rf, X)

    p_out, l_out, i_out = [], [], []
    cursor = 0
    for pos in pos_all:
        n = len(pos)
        p_out.append(probs[cursor:cursor + n])
        l_out.append(logits[cursor:cursor + n])
        i_out.append(pos.astype(np.int64))
        cursor += n
    return p_out, l_out, i_out


def attack_source(
    models: LoadedModels,
    source: str,
    folder: Path,
    sidecar: Sidecar,
    select_flow_ids: list[str],
    budgets: list[BudgetSpec],
    ipat_std: float,
    first_checkpoint: int,
    stride: int,
    pgd_cfg: PGDConfig,
    batch_size: int = 32,
    threshold: float = 0.5,
    pad_byte_mode: str = "random",
    objective: str = "final",
    tl_cap: int = 1500,
    variants: tuple[str, ...] = ("pgd", "random"),
) -> list[dict]:
    """Attack one adversarial source across the budget grid.

    Returns one record per (flow, budget, variant) with each target's perturbed
    and unperturbed verdicts, the two meta-learner ablations, and the spend.
    """
    from ..common.dataset import load_source_meta

    packets = np.load(folder / "flow_packets.npy", mmap_mode="r")
    offsets = np.load(folder / "flow_offsets.npy")
    _y, fids, _ls, local_idx, _n = load_source_meta(folder, source)
    fid_to_local = {str(f): int(li) for f, li in zip(fids, local_idx)}

    # Keep only the selected flows (a cached sidecar may hold more).
    wanted = set(select_flow_ids)
    rows = [i for i in range(len(sidecar))
            if str(sidecar.flow_ids[i]) in wanted
            and str(sidecar.flow_ids[i]) in fid_to_local]
    records: list[dict] = []

    for s0 in range(0, len(rows), batch_size):
        sc_rows = rows[s0:s0 + batch_size]
        flow_ids = [str(sidecar.flow_ids[r]) for r in sc_rows]
        local_ids = [fid_to_local[f] for f in flow_ids]

        seq, mask, lengths = _load_padded_batch(
            packets, offsets, local_ids, models.max_length, models.device,
        )
        batch = FlowBatch.from_padded(seq, mask, lengths)
        L_np = lengths.cpu().numpy()
        B = len(sc_rows)
        labels = [1] * B          # every selected flow is adversarial by construction

        # Unperturbed streams: ablation baselines and the ASR denominator.
        zeros = np.zeros((B, int(mask.shape[1])), dtype=np.float32)
        with torch.no_grad():
            tcn_probs_clean, tcn_logit_clean = score_tcn(models.tcn, seq, lengths)
        rf_probs_clean, rf_logit_clean, rf_pos_clean = _rf_streams(
            models, sidecar, sc_rows, zeros, zeros, L_np,
            first_checkpoint, stride, pad_byte_mode,
        )
        stack_probs_clean = score_stack(
            models.meta, seq=seq, lengths=lengths, tcn_logits=tcn_logit_clean,
            rf_logits=rf_logit_clean, rf_packet_idx=rf_pos_clean,
            sources=[source] * B, flow_ids=flow_ids, labels=labels,
            rf_age_clip=models.rf_age_clip, pkt_len_norm=models.pkt_len_norm,
        )
        verdicts_clean = [
            {"rf": flow_verdict(rf_probs_clean[b]),
             "tcn": flow_verdict(tcn_probs_clean[b]),
             "stack": flow_verdict(stack_probs_clean[b])}
            for b in range(B)
        ]

        for spec in budgets:
            for variant in variants:
                # `random`: same feasible set, budget and number of forward
                # passes as PGD, without the gradient.
                vcfg = (
                    pgd_cfg if variant == "pgd"
                    else PGDConfig(
                        steps=0, random_init=True, seed=pgd_cfg.seed,
                        restarts=1 + pgd_cfg.restarts * pgd_cfg.steps,
                    )
                )
                budget = spec.to_budget()
                obj = tcn_margin_objective(models.tcn, batch, reduction=objective)
                k, d, _ = pgd(obj, batch, budget, ipat_std, vcfg, tl_cap)

                with torch.no_grad():
                    seq_p = apply_delta(batch, k, d, ipat_std, tl_cap)
                    tcn_probs, tcn_logit_p = score_tcn(models.tcn, seq_p, lengths)

                k_np = k.detach().cpu().numpy().round()
                d_np = d.detach().cpu().numpy()

                rf_probs, rf_logit_p, rf_pos_p = _rf_streams(
                    models, sidecar, sc_rows, k_np, d_np, L_np,
                    first_checkpoint, stride, pad_byte_mode,
                )

                # One host copy, shared by the stack score and both ablations —
                # they all read the same perturbed tensor.
                common = dict(
                    seq=seq_p, seq_np=seq_p.detach().cpu().numpy(),
                    lengths=lengths, sources=[source] * B,
                    flow_ids=flow_ids, labels=labels,
                    rf_age_clip=models.rf_age_clip,
                    pkt_len_norm=models.pkt_len_norm,
                )
                stack_probs = score_stack(
                    models.meta, tcn_logits=tcn_logit_p,
                    rf_logits=rf_logit_p, rf_packet_idx=rf_pos_p, **common,
                )
                # Ablations: same perturbed tensor, one base-learner slot handed
                # back its clean value — isolates which slot carries the verdict.
                abl_rf_clean = score_stack(
                    models.meta, tcn_logits=tcn_logit_p,
                    rf_logits=rf_logit_clean, rf_packet_idx=rf_pos_clean, **common,
                )
                abl_tcn_clean = score_stack(
                    models.meta, tcn_logits=tcn_logit_clean,
                    rf_logits=rf_logit_p, rf_packet_idx=rf_pos_p, **common,
                )

                for b in range(B):
                    Lb = int(L_np[b])
                    kb = k_np[b, :Lb]
                    db = d_np[b, :Lb]
                    records.append({
                        "source": source,
                        "flow_id": flow_ids[b],
                        "budget_id": spec.id,
                        "variant": variant,
                        "length": Lb,
                        "verdicts": {
                            "rf": flow_verdict(rf_probs[b]),
                            "tcn": flow_verdict(tcn_probs[b]),
                            "stack": flow_verdict(stack_probs[b]),
                        },
                        "verdicts_clean": verdicts_clean[b],
                        "never_fires": {
                            "rf": never_fires(rf_probs[b], threshold),
                            "tcn": never_fires(tcn_probs[b], threshold),
                            "stack": never_fires(stack_probs[b], threshold),
                        },
                        "ablations": {
                            "stack_rf_clean": flow_verdict(abl_rf_clean[b]),
                            "stack_tcn_clean": flow_verdict(abl_tcn_clean[b]),
                        },
                        "spend": {
                            "median_pad_bytes": float(np.median(kb)) if Lb else 0.0,
                            "median_delay_seconds": (float(np.median(db))
                                                     if Lb else 0.0),
                            "total_pad_bytes": float(kb.sum()),
                            "total_delay_seconds": float(db.sum()),
                        },
                    })

        log.info("%s: attacked %d/%d flows", source, min(s0 + batch_size, len(rows)),
                 len(rows))

    return records


def autopgd_check(
    models: LoadedModels,
    folder: Path,
    local_ids: list[int],
    eps_bytes: float,
    ipat_std: float,
    pgd_cfg: PGDConfig,
    tl_cap: int = 1500,
    steps: int = 20,
) -> dict:
    """Compare this search with ART's AutoPGD on one batch.

    Diagnostic only: AutoPGD runs on the relaxed problem (no integrality, no
    physical budget), so its result cannot be realized.
    """
    from .attack import art_autopgd_relaxed

    packets = np.load(folder / "flow_packets.npy", mmap_mode="r")
    offsets = np.load(folder / "flow_offsets.npy")
    seq, mask, lengths = _load_padded_batch(
        packets, offsets, local_ids, models.max_length, models.device)
    batch = FlowBatch.from_padded(seq, mask, lengths)
    budget = Budget(bytes_=eps_bytes, seconds=0.0)
    obj = tcn_margin_objective(models.tcn, batch, reduction="final")

    art_margins = art_autopgd_relaxed(
        models.tcn, batch, eps_bytes, ipat_std, steps=steps, tl_cap=tl_cap,
        max_flows=len(local_ids),
    )
    if art_margins is None:
        return {"available": False,
                "reason": "ART not installed or AutoPGD unavailable"}
    n = len(art_margins)

    _k, _d, ours = pgd(obj, batch, budget, ipat_std, pgd_cfg, tl_cap)
    rnd_cfg = PGDConfig(steps=0, random_init=True, seed=pgd_cfg.seed,
                        restarts=1 + pgd_cfg.restarts * pgd_cfg.steps)
    _k2, _d2, rnd = pgd(obj, batch, budget, ipat_std, rnd_cfg, tl_cap)
    ours_np = ours.detach().cpu().numpy()[:n]
    rnd_np = rnd.detach().cpu().numpy()[:n]

    return {
        "available": True,
        "note": ("relaxed problem (no integrality / physical budget) — an "
                 "upper bound on attack strength, not a realizable attack"),
        "eps_bytes": float(eps_bytes),
        "n_flows": int(n),
        "frac_margin_below_zero": {
            "art_autopgd": float((art_margins < 0).mean()),
            "ours_pgd": float((ours_np < 0).mean()),
            "ours_random": float((rnd_np < 0).mean()),
        },
        "median_margin": {
            "art_autopgd": float(np.median(art_margins)),
            "ours_pgd": float(np.median(ours_np)),
            "ours_random": float(np.median(rnd_np)),
        },
    }


def score_benign_reference(
    models: LoadedModels,
    tensors_dir: Path,
    benign_sources: list[str],
    flows_per_source: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Unperturbed benign flow verdicts per target, for the matched-FPR thresholds.

    The RF checkpoints are read from disk, since benign flows are not perturbed.
    """
    import json

    from ...feature_extraction.tensor_builder import checkpoint_positions
    from ..common.dataset import load_source_meta

    rng = np.random.default_rng(seed)
    out: dict[str, list[float]] = {"rf": [], "tcn": [], "stack": []}

    for src in benign_sources:
        folder = Path(tensors_dir) / src
        if not folder.exists():
            log.warning("benign source missing: %s", folder)
            continue
        packets = np.load(folder / "flow_packets.npy", mmap_mode="r")
        offsets = np.load(folder / "flow_offsets.npy")
        ckpts = np.load(folder / "rf_checkpoints.npy", mmap_mode="r")
        ck_off = np.load(folder / "rf_checkpoint_offsets.npy")
        _y, fids, _ls, local_idx, _n = load_source_meta(folder, src)
        # The checkpoint cadence is stored per source.
        meta = json.loads((folder / "metadata.json").read_text())
        first_ck = int(meta.get("rf_first_checkpoint", 2))
        stride = int(meta.get("rf_checkpoint_stride", 10))

        eligible = [int(li) for li in local_idx if int(offsets[int(li)][1]) >= 2]
        if not eligible:
            continue
        pick = (rng.choice(eligible, size=min(flows_per_source, len(eligible)),
                           replace=False)
                if len(eligible) > flows_per_source else np.asarray(eligible))

        for s0 in range(0, len(pick), 32):
            chunk = [int(x) for x in pick[s0:s0 + 32]]
            seq, mask, lengths = _load_padded_batch(
                packets, offsets, chunk, models.max_length, models.device,
            )
            with torch.no_grad():
                tcn_probs, tcn_logits = score_tcn(models.tcn, seq, lengths)

            rf_p, rf_l, rf_i = [], [], []
            for li in chunk:
                start, n_ck = ck_off[int(li)]
                X = np.asarray(ckpts[int(start):int(start) + int(n_ck)])
                p, lg = rf_predict_features(models.rf, X)
                rf_p.append(p)
                rf_l.append(lg)
                # rf_checkpoints.npy stores values, not positions; rebuild them
                # from the cadence this source was tensorised with.
                rf_i.append(
                    checkpoint_positions(int(offsets[int(li)][1]), first_ck, stride))

            stack_probs = score_stack(
                models.meta, seq=seq, lengths=lengths, tcn_logits=tcn_logits,
                rf_logits=rf_l, rf_packet_idx=rf_i,
                sources=[src] * len(chunk), flow_ids=[str(c) for c in chunk],
                labels=[0] * len(chunk),
                rf_age_clip=models.rf_age_clip, pkt_len_norm=models.pkt_len_norm,
            )
            for b in range(len(chunk)):
                out["rf"].append(flow_verdict(rf_p[b]))
                out["tcn"].append(flow_verdict(tcn_probs[b]))
                out["stack"].append(flow_verdict(stack_probs[b]))
        log.info("%s: scored %d benign reference flows", src, len(pick))

    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


__all__ = [
    "BudgetSpec",
    "parse_budgets",
    "attack_source",
    "score_benign_reference",
]
