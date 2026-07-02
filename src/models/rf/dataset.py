"""Random Forest dataset assembly — streaming-native edition.

Under the new schema each source folder stores ``rf_checkpoints.npy``
([C_total, 24]) holding *multi-scale prefix* features computed at the
optional ``rf_first_checkpoint`` packet index (default 2, set to 0 to
disable) plus every ``rf_checkpoint_stride`` packets per flow. The k-th
checkpoint of a flow contains aggregated stats over packets
``[0:packet_idx_k]`` where ``packet_idx_k`` is captured per checkpoint
on the dataset (see ``RFDataset.packet_idx``).

This module expands the per-flow assembled index into a **per-checkpoint**
training matrix: each row of ``RFDataset.X`` is one prefix view of one
flow, labelled with the flow's class. Training the RF on this multi-scale
pool ensures the model is calibrated for the full inference-time
distribution of prefix sizes (not just N=10 windows).

The split is computed in **per-flow** space first (using the shared
``flow_level_split``) and then expanded to per-checkpoint rows so all
prefix views of the same flow land in the same split — no leakage.

Public API: ``RFDataset``, ``Split``, ``assemble``, ``flow_level_split``,
``split_summary``, ``discover_adversarial_sources``, ``cap_benign``, plus
the label string constants.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.dataset import (
    ADVERSARIAL_LABEL_STR,
    BENIGN_LABEL_STR,
    UNKNOWN_LABEL_STR,
    AssembledIndex,
    Split,
    assemble_index,
    cap_benign,
    discover_adversarial_sources,
    flow_level_split as _common_flow_level_split,
)


__all__ = [
    "ADVERSARIAL_LABEL_STR",
    "BENIGN_LABEL_STR",
    "UNKNOWN_LABEL_STR",
    "RFDataset",
    "Split",
    "assemble",
    "cap_benign",
    "discover_adversarial_sources",
    "flow_level_split",
    "split_summary",
]


@dataclass
class RFDataset:
    X: np.ndarray               # [N, F] float32  — N = total checkpoints
    y: np.ndarray               # [N]    int8     — flow label propagated
    flow_ids: np.ndarray        # [N]    str      — duplicated per checkpoint
    sources: np.ndarray         # [N]    str      — duplicated per checkpoint
    label_str: np.ndarray       # [N]    str
    checkpoint_idx: np.ndarray  # [N]    int32    — 0..n_ckpt-1 within flow
    packet_idx: np.ndarray = field(  # [N] int32  — absolute 1-based packet
        default_factory=lambda: np.zeros(0, dtype=np.int32),  # index of each
    )                                                          # checkpoint
    feature_names: list[str] = field(default_factory=list)


def _flow_codes(sources: np.ndarray, flow_ids: np.ndarray) -> np.ndarray:
    """Integer code per row identifying its ``(source, flow_id)`` flow.

    ``codes[i] == codes[j]`` iff rows *i* and *j* belong to the same flow,
    with codes assigned ``0, 1, 2, …`` in **first-appearance order** — the
    same grouping and ordering as the previous
    ``np.unique(("src|flow").astype(str))`` path, but computed with
    hash-based ``pd.factorize`` (O(N), no object-dtype sort). This is the
    fix for the multi-day single-threaded ``object``-string bottleneck.
    """
    if len(sources) == 0:
        return np.zeros(0, dtype=np.int64)
    # Factorize source and flow_id separately (hash), then combine into a
    # single integer key that is unique per (source, flow_id) pair, then
    # factorize that key to compact first-appearance codes.
    s_codes = pd.factorize(np.asarray(sources), sort=False)[0]
    f_codes = pd.factorize(np.asarray(flow_ids), sort=False)[0]
    pair = s_codes.astype(np.int64) * (np.int64(f_codes.max()) + 1) + f_codes
    codes = pd.factorize(pair, sort=False)[0]
    return codes.astype(np.int64, copy=False)


def _expand_ranges(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Vectorized concat of ``arange(starts[k], starts[k]+counts[k])``.

    Replaces a per-flow Python loop: builds the full gather index in one
    cumulative-sum pass. ``counts`` must all be > 0 and ``len(starts) > 0``.
    """
    total = int(counts.sum())
    idx = np.ones(total, dtype=np.int64)
    idx[0] = starts[0]
    if len(starts) > 1:
        group_offsets = np.cumsum(counts)[:-1]
        idx[group_offsets] = starts[1:] - (starts[:-1] + counts[:-1] - 1)
    return np.cumsum(idx)


def _expand_source_checkpoints(
    folder: Path,
    flow_local_idx: np.ndarray,
    y_per_flow: np.ndarray,
    flow_ids_per_flow: np.ndarray,
    label_str_per_flow: np.ndarray,
    source_name: str,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    """Expand the selected flows of one source into per-checkpoint rows.

    Reads ``rf_checkpoints.npy`` (small, eagerly loaded) and
    ``rf_checkpoint_offsets.npy`` (small) and slices each flow's
    contiguous checkpoint block. Flows whose length produces zero
    checkpoints under the source's cadence are silently skipped.

    The source's ``metadata.json`` is consulted to capture the actual
    packet index of each checkpoint (cadence is per-source by
    construction and stays constant within a source folder).
    """
    rf_ckpts = np.load(folder / "rf_checkpoints.npy").astype(np.float32, copy=False)
    rf_offsets = np.load(folder / "rf_checkpoint_offsets.npy")

    # Per-source cadence from metadata.json (default: legacy stride-only).
    meta_path = folder / "metadata.json"
    try:
        meta = json.loads(meta_path.read_text())
    except FileNotFoundError:
        meta = {}
    rf_stride = int(meta.get("rf_checkpoint_stride", 10))
    rf_first = int(meta.get("rf_first_checkpoint", 0))

    n_cols = rf_ckpts.shape[1] if rf_ckpts.ndim == 2 else 24
    empty = (
        np.zeros((0, n_cols), dtype=np.float32),
        np.zeros(0, dtype=np.int8),
        np.zeros(0, dtype=object),
        np.zeros(0, dtype=object),
        np.zeros(0, dtype=object),
        np.zeros(0, dtype=np.int32),
        np.zeros(0, dtype=np.int32),
    )

    fi = np.asarray(flow_local_idx, dtype=np.int64)
    if fi.size == 0:
        return empty
    starts = rf_offsets[fi, 0].astype(np.int64)
    counts = rf_offsets[fi, 1].astype(np.int64)

    keep = counts > 0
    if not keep.any():
        return empty
    starts = starts[keep]
    counts = counts[keep]
    y_keep = np.asarray(y_per_flow)[keep]
    fid_keep = np.asarray(flow_ids_per_flow, dtype=object)[keep]
    lab_keep = np.asarray(label_str_per_flow, dtype=object)[keep]

    # X: gather every checkpoint block in one shot — rows stay in flow order,
    # checkpoints ascending within a flow (identical to the old slice loop).
    gather = _expand_ranges(starts, counts)
    total = len(gather)
    X = rf_ckpts[gather]

    # Per-checkpoint y / flow-id / source / label via run-length repeat.
    y = np.repeat(y_keep.astype(np.int8, copy=False), counts)
    flows = np.repeat(fid_keep, counts)
    srcs = np.full(total, source_name, dtype=object)
    labs = np.repeat(lab_keep, counts)

    # checkpoint_idx: per-flow 0..n_ckpt-1, reset to 0 at each block start.
    cidx = np.ones(total, dtype=np.int64)
    cidx[0] = 0
    if len(counts) > 1:
        group_offsets = np.cumsum(counts)[:-1]
        cidx[group_offsets] = -(counts[:-1] - 1)
    cidx = np.cumsum(cidx).astype(np.int32)

    # packet_idx: closed form of checkpoint_packet_indices() applied to cidx.
    if rf_first > 0 and (rf_first % rf_stride) != 0:
        pidx = np.where(cidx == 0, rf_first, cidx.astype(np.int64) * rf_stride)
    else:
        pidx = (cidx.astype(np.int64) + 1) * rf_stride
    pidx = pidx.astype(np.int32)

    return X, y, flows, srcs, labs, cidx, pidx


def assemble(
    tensors_dir: Path,
    benign_sources: list[str],
    adversarial_dir: Path,
    benign_cap_ratio: float,
    seed: int,
    adversarial_exclude: list[str] = (),
) -> RFDataset:
    """Merge benign + adversarial sources, expand to per-checkpoint rows.

    Row order: grouped by source in the iteration order of the shared
    ``assemble_index``, then by flow (per-flow row order), then by
    checkpoint index (ascending prefix length).
    """
    idx: AssembledIndex = assemble_index(
        tensors_dir=tensors_dir,
        benign_sources=benign_sources,
        adversarial_dir=adversarial_dir,
        benign_cap_ratio=benign_cap_ratio,
        seed=seed,
        adversarial_exclude=list(adversarial_exclude),
    )

    feature_names = list(idx.feature_names)
    seen_order: list[str] = []
    for src in idx.sources:
        if not seen_order or seen_order[-1] != src:
            seen_order.append(src)

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    flow_parts: list[np.ndarray] = []
    src_parts: list[np.ndarray] = []
    lab_parts: list[np.ndarray] = []
    ckpt_parts: list[np.ndarray] = []
    pkt_parts: list[np.ndarray] = []

    for src in seen_order:
        folder = idx.source_folders[src]
        mask = idx.sources == src
        X_s, y_s, flow_s, src_s, lab_s, cidx_s, pidx_s = _expand_source_checkpoints(
            folder=folder,
            flow_local_idx=idx.local_idx[mask],
            y_per_flow=idx.y[mask],
            flow_ids_per_flow=idx.flow_ids[mask],
            label_str_per_flow=idx.label_str[mask],
            source_name=src,
        )
        X_parts.append(X_s)
        y_parts.append(y_s)
        flow_parts.append(flow_s)
        src_parts.append(src_s)
        lab_parts.append(lab_s)
        ckpt_parts.append(cidx_s)
        pkt_parts.append(pidx_s)
        if not feature_names:
            meta = json.loads((folder / "metadata.json").read_text())
            feature_names = list(meta.get("rf_stat_names", []))

    X = np.concatenate(X_parts, axis=0) if X_parts else np.zeros((0, 24), dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.concatenate(y_parts, axis=0) if y_parts else np.zeros(0, dtype=np.int8)
    flow_ids = (
        np.concatenate(flow_parts, axis=0) if flow_parts else np.zeros(0, dtype=object)
    )
    sources = (
        np.concatenate(src_parts, axis=0) if src_parts else np.zeros(0, dtype=object)
    )
    label_str = (
        np.concatenate(lab_parts, axis=0) if lab_parts else np.zeros(0, dtype=object)
    )
    checkpoint_idx = (
        np.concatenate(ckpt_parts, axis=0) if ckpt_parts else np.zeros(0, dtype=np.int32)
    )
    packet_idx = (
        np.concatenate(pkt_parts, axis=0) if pkt_parts else np.zeros(0, dtype=np.int32)
    )

    return RFDataset(
        X=X,
        y=y,
        flow_ids=flow_ids,
        sources=sources,
        label_str=label_str,
        checkpoint_idx=checkpoint_idx,
        packet_idx=packet_idx,
        feature_names=feature_names or [f"f{i}" for i in range(X.shape[1])],
    )


def flow_level_split(
    dataset: RFDataset,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Split:
    """Flow-atomic split, expanded to per-checkpoint row indices.

    The decision is made on the deduplicated per-flow view, then mapped
    back to every checkpoint row of that flow, so a single flow's
    prefixes never split across train/val/test.
    """
    if len(dataset.sources) == 0:
        return Split(
            train_idx=np.zeros(0, dtype=np.int64),
            val_idx=np.zeros(0, dtype=np.int64),
            test_idx=np.zeros(0, dtype=np.int64),
        )

    # Integer flow code per checkpoint row (hash-based; replaces the old
    # object-dtype np.unique/np.isin that ran single-threaded for days).
    codes = _flow_codes(dataset.sources, dataset.flow_ids)
    # First-occurrence row of each code, in code (== first-appearance) order —
    # identical to the previous np.unique(...return_index)+np.sort ordering, so
    # _common_flow_level_split receives byte-identical per-flow inputs.
    _, first_pos = np.unique(codes, return_index=True)
    per_flow_sources = dataset.sources[first_pos]
    per_flow_flow_ids = dataset.flow_ids[first_pos]

    flow_split = _common_flow_level_split(
        sources=per_flow_sources,
        flow_ids=per_flow_flow_ids,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )

    # Map the per-flow split decision back to every checkpoint row via codes
    # (per-flow index k corresponds to flow code k).
    split_of_flow = np.full(len(first_pos), -1, dtype=np.int8)
    split_of_flow[flow_split.train_idx] = 0
    split_of_flow[flow_split.val_idx] = 1
    split_of_flow[flow_split.test_idx] = 2
    row_split = split_of_flow[codes]

    return Split(
        train_idx=np.where(row_split == 0)[0].astype(np.int64),
        val_idx=np.where(row_split == 1)[0].astype(np.int64),
        test_idx=np.where(row_split == 2)[0].astype(np.int64),
    )


def split_summary(dataset: RFDataset, split: Split) -> dict:
    """Per-split checkpoint + unique-flow counts."""
    # Shared integer flow codes — counting distinct codes per split replaces
    # the old per-split object-string rebuild + set().
    codes = _flow_codes(dataset.sources, dataset.flow_ids)

    def _stats(idx: np.ndarray) -> dict:
        if len(idx) == 0:
            return {
                "n_checkpoints": 0, "n_flows": 0,
                "n_benign": 0, "n_adversarial": 0,
                "per_source": {},
            }
        ys = dataset.y[idx]
        srcs = dataset.sources[idx]
        return {
            "n_checkpoints": int(len(idx)),
            "n_flows": int(np.unique(codes[idx]).size),
            "n_benign": int((ys == 0).sum()),
            "n_adversarial": int((ys == 1).sum()),
            "per_source": {
                s: int((srcs == s).sum())
                for s in sorted(pd.unique(srcs).tolist())
            },
        }
    return {
        "train": _stats(split.train_idx),
        "val": _stats(split.val_idx),
        "test": _stats(split.test_idx),
    }
