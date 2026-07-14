"""The DEPT gated deployment pipeline: detector → purify-if-flagged-else-pass.

This is the runtime dataflow the README's block diagram promises but the codebase never
actually wired (each stage was a standalone train+eval program). Here we assemble it for
the baseline: for an attack's flows we

  1. **purify** every flow with the trained reconstructor and decode its size/timing
     overlay (plus the adversarial and clean-target overlays, for the 3-way realization);
  2. get the **detector** (meta-learner) verdict per flow (:mod:`.detector_infer`);
  3. build the **defended** packet stream = the *purified* overlay for flows the detector
     flags adversarial, and the *adversarial* overlay (unchanged — "non-adversarial passes
     through immediately") for the rest.

The defended stream is then scored through the independent NIDS exactly like the clean and
adversarial streams, so the three regimes are directly comparable.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..common.runtime import resolve_num_workers
from ..reconstructor.dataset import PairedFlowDataset, collate_pairs
from ..reconstructor.evaluate import _load_reconstructor, _round_recon
from ..reconstructor.realize import decode_total_len_batch

log = logging.getLogger(__name__)

_IPAT_COL = 1088


def load_reconstructor(recon_dir: Path, device: torch.device):
    return _load_reconstructor(Path(recon_dir), device)


def purify_source(pairs, source_rows: np.ndarray, model, device: torch.device, *,
                  batch_size: int = 32, num_workers="auto",
                  max_length: int | None = None) -> dict[str, dict]:
    """Purify every flow in ``source_rows`` → ``{fid: {"adv"/"recon"/"clean": (tl, ipat)}}``.

    Mirrors the reconstructor eval's per-flow decode, but over an arbitrary row set (the
    baseline uses *all* of a source's flows, for full-stream volume). ``(tl, ipat)`` is
    the decoded IPv4 total length + z-scored IPAT per packet — the compact overlay
    :func:`realize.realize_triplet` maps back onto the adversarial packets.
    """
    ds = PairedFlowDataset(pairs, sample_indices=source_rows, max_length=max_length)
    nw = resolve_num_workers(num_workers)
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=nw,
                        pin_memory=device.type == "cuda", persistent_workers=nw > 0,
                        collate_fn=collate_pairs)
    out: dict[str, dict] = {}
    for adv, clean, lengths, mask, rows in loader:
        adv_d = adv.to(device, non_blocking=True)
        mask_d = mask.to(device, non_blocking=True)
        with torch.inference_mode():
            recon_d = model(adv_d, key_padding_mask=~mask_d)
        rr = _round_recon(recon_d).detach().cpu().numpy()
        adv_np = adv.numpy()
        clean_np = clean.numpy()
        len_np = lengths.numpy()
        rows_np = rows.numpy()
        for b in range(adv_np.shape[0]):
            L = int(len_np[b])
            fid = str(pairs.flow_id[int(rows_np[b])])
            a = adv_np[b, :L]
            c = clean_np[b, :L]
            r = rr[b, :L]
            out[fid] = {
                "adv": (decode_total_len_batch(a), a[:, _IPAT_COL].astype(np.float32)),
                "recon": (decode_total_len_batch(r), r[:, _IPAT_COL].astype(np.float32)),
                "clean": (decode_total_len_batch(c), c[:, _IPAT_COL].astype(np.float32)),
            }
    return out


def assemble_defended(branches: dict, verdicts: dict, source: str,
                      threshold: float) -> tuple[list, list, dict]:
    """Gate the realized packets: purified branch if flagged, adversarial branch if not.

    ``branches`` is :func:`realize.realize_triplet`'s output (per-packet ``packets`` +
    ``keys``); ``verdicts`` maps ``(source, flow_id)`` → detector ``P(adversarial)``.
    Returns ``(packets, keys, stats)`` — the defended stream, re-sorted by (branch-native)
    timestamp, ready for the NIDS.
    """
    adv_by: dict[str, list] = defaultdict(list)
    pur_by: dict[str, list] = defaultdict(list)
    for pkt, k in zip(branches["adv"]["packets"], branches["adv"]["keys"]):
        adv_by[k].append(pkt)
    for pkt, k in zip(branches["purified"]["packets"], branches["purified"]["keys"]):
        pur_by[k].append(pkt)

    chosen: list[tuple[float, object, str]] = []
    n_flagged = 0
    for k in adv_by:
        flagged = float(verdicts.get((source, k), 0.0)) >= threshold
        if flagged:
            n_flagged += 1
            src_pkts = pur_by.get(k, adv_by[k])   # fall back to adv if purified dropped
        else:
            src_pkts = adv_by[k]
        for pkt in src_pkts:
            chosen.append((float(pkt.time), pkt, k))
    chosen.sort(key=lambda t: t[0])
    packets = [t[1] for t in chosen]
    keys = [t[2] for t in chosen]
    n = len(adv_by)
    stats = {"n_flows": n, "n_flagged_adversarial": n_flagged,
             "flag_rate": (n_flagged / n) if n else None}
    return packets, keys, stats


def subset_by_flows(packets: list, keys: list, wanted: set) -> tuple[list, list]:
    """Restrict a scored stream to a set of flow keys (the held-out test subset)."""
    out_p, out_k = [], []
    for pkt, k in zip(packets, keys):
        if k in wanted:
            out_p.append(pkt)
            out_k.append(k)
    return out_p, out_k


__all__ = ["load_reconstructor", "purify_source", "assemble_defended", "subset_by_flows"]
