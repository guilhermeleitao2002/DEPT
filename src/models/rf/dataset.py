"""Random Forest dataset assembly — RF-specific wrapper around
``src/models/common/dataset.py``.

Loads the 24-dim aggregate-statistics view (``rf_stats.npy``) for every
window selected by the shared ``assemble_index``, so RF and TCN see the
same flow-level split.

Public API preserved: ``RFDataset``, ``Split``, ``assemble``,
``flow_level_split``, ``split_summary``, ``load_source``,
``discover_adversarial_sources``, ``cap_benign``, plus the label string
constants.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

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
    load_source_meta,
    split_summary as _common_split_summary,
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
    "load_source",
    "split_summary",
]


@dataclass
class RFDataset:
    X: np.ndarray               # [N, F] float32
    y: np.ndarray               # [N]    int8   (0 benign, 1 adversarial)
    flow_ids: np.ndarray        # [N]    str    (stringified 5-tuple)
    sources: np.ndarray         # [N]    str    (source folder name)
    label_str: np.ndarray       # [N]    str    ("benign" or "adversarial")
    feature_names: list[str] = field(default_factory=list)


def load_source(folder: Path, source_name: str) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]
]:
    """Load one source folder. Returns (X, y, flow_ids, label_str, feature_names).

    Kept for backwards compatibility — internally this loads the heavy
    ``rf_stats.npy`` and slices to the rows that survived label
    normalisation. Prefer the shared ``assemble_index`` for new code.
    """
    y, flow_ids, label_str, local_idx, feature_names = load_source_meta(folder, source_name)
    X_full = np.load(folder / "rf_stats.npy").astype(np.float32, copy=False)
    if X_full.shape[0] < (int(local_idx.max()) + 1 if len(local_idx) else 0):
        raise ValueError(f"{source_name}: rf_stats rows mismatch")
    X = X_full[local_idx]
    return X, y, flow_ids, label_str, feature_names


def assemble(
    tensors_dir: Path,
    benign_sources: list[str],
    adversarial_dir: Path,
    benign_cap_ratio: float,
    seed: int,
    adversarial_exclude: list[str] = (),
) -> RFDataset:
    """Merge benign sources + auto-discovered adversarial sources into an
    ``RFDataset`` whose row order matches the shared ``assemble_index``.
    """
    idx: AssembledIndex = assemble_index(
        tensors_dir=tensors_dir,
        benign_sources=benign_sources,
        adversarial_dir=adversarial_dir,
        benign_cap_ratio=benign_cap_ratio,
        seed=seed,
        adversarial_exclude=list(adversarial_exclude),
    )

    # Gather rf_stats rows per source (small files, ~1MB each — fine in RAM).
    # The index's row order groups rows by source in iteration order, so we
    # can slice once per source and concatenate.
    parts: list[np.ndarray] = []
    feature_names = list(idx.feature_names)
    seen_order: list[str] = []
    for src in idx.sources:
        if not seen_order or seen_order[-1] != src:
            seen_order.append(src)
    for src in seen_order:
        folder = idx.source_folders[src]
        X_full = np.load(folder / "rf_stats.npy").astype(np.float32, copy=False)
        mask = idx.sources == src
        parts.append(X_full[idx.local_idx[mask]])
        if not feature_names:
            # Fall back to source-specific names if assemble_index didn't pick any up.
            import json
            meta = json.loads((folder / "metadata.json").read_text())
            feature_names = list(meta.get("rf_stat_names", []))

    X = np.concatenate(parts, axis=0)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return RFDataset(
        X=X,
        y=idx.y,
        flow_ids=idx.flow_ids,
        sources=idx.sources,
        label_str=idx.label_str,
        feature_names=feature_names or [f"f{i}" for i in range(X.shape[1])],
    )


def flow_level_split(
    dataset: RFDataset,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Split:
    """Stratified, flow-atomic split — RF-flavoured wrapper accepting an
    ``RFDataset``."""
    return _common_flow_level_split(
        sources=dataset.sources,
        flow_ids=dataset.flow_ids,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )


def split_summary(dataset: RFDataset, split: Split) -> dict:
    """Per-split window/source counts."""
    return _common_split_summary(
        sources=dataset.sources,
        y=dataset.y,
        split=split,
    )
