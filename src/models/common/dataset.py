"""Model-agnostic dataset primitives for ADEPT base learners.

This module owns the parts of the data pipeline that don't care which
tensor view a model consumes:

  * source discovery (benign + auto-discovered adversarial folders)
  * label normalisation under the benign-vs-adversarial convention
  * flow-atomic, source-stratified train/val/test splitting
  * benign capping at a configurable ratio of adversarial windows

The Random Forest base learner consumes ``rf_stats.npy`` (small, eagerly
concatenated) and the TCN base learner consumes ``sequence.npy`` (multi-GB
per source, read lazily through memmaps). Both reuse ``assemble_index``
below to land on the **same** flow-level split, so a per-window
RF/TCN/ensemble comparison is apples-to-apples.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


BENIGN_LABEL_STR = "benign"
ADVERSARIAL_LABEL_STR = "adversarial"
UNKNOWN_LABEL_STR = "unknown"


@dataclass
class Split:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


@dataclass
class AssembledIndex:
    """Per-window index into the on-disk tensors, with no payload loaded.

    Each row identifies one window by ``(source, local_idx)`` — the source
    folder name and the row number within that source's ``*.npy`` files.
    Consumers materialise the heavy payload themselves (RF concatenates
    ``rf_stats.npy``; TCN memmaps ``sequence.npy``).
    """
    sources: np.ndarray         # [N] str    source folder name
    local_idx: np.ndarray       # [N] int64  row index within that source
    y: np.ndarray               # [N] int8   0 benign / 1 adversarial
    flow_ids: np.ndarray        # [N] str    stringified canonical 5-tuple
    label_str: np.ndarray       # [N] str    "benign" or "adversarial"
    feature_names: list[str] = field(default_factory=list)
    source_folders: dict[str, Path] = field(default_factory=dict)


def _flow_id_to_str(fid_row: np.ndarray) -> str:
    # object array of 5 items: src_ip, dst_ip, sport, dport, proto
    return "|".join(str(x) for x in fid_row)


def _normalize_label(raw: str) -> tuple[int, bool]:
    """Return (label, keep). 0 benign / 1 adversarial; keep=False for unknown."""
    s = str(raw).strip().lower()
    if s == UNKNOWN_LABEL_STR or s == "" or s == "nan":
        return 0, False
    if s == BENIGN_LABEL_STR:
        return 0, True
    return 1, True


def discover_adversarial_sources(adversarial_dir: Path) -> list[str]:
    """Return all subdir names of adversarial_dir/ that contain metadata.json."""
    if not adversarial_dir.exists():
        return []
    return sorted(
        d.name for d in adversarial_dir.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
    )


def load_source_meta(folder: Path, source_name: str) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, int, list[str]
]:
    """Read labels/flow_ids/metadata for one source without loading any payload.

    Returns ``(y, flow_ids, label_str, n_rows, feature_names)`` after
    dropping rows whose label_str is empty/unknown.
    """
    flow_ids_raw = np.load(folder / "flow_ids.npy", allow_pickle=True)
    labels_df = pd.read_csv(folder / "window_labels.csv")
    meta = json.loads((folder / "metadata.json").read_text())
    n_meta_rows = int(meta.get("num_windows", len(labels_df)))
    feature_names = list(meta.get("rf_stat_names", []))

    if len(labels_df) != n_meta_rows:
        raise ValueError(
            f"{source_name}: rows mismatch ({len(labels_df)} labels vs {n_meta_rows} in metadata)")
    if len(flow_ids_raw) != n_meta_rows:
        raise ValueError(f"{source_name}: flow_ids rows mismatch")

    y_parts, keep_mask = [], []
    for raw in labels_df["label_str"].values:
        lab, keep = _normalize_label(raw)
        y_parts.append(lab)
        keep_mask.append(keep)
    y_full = np.array(y_parts, dtype=np.int8)
    keep = np.array(keep_mask, dtype=bool)

    y = y_full[keep]
    label_str = labels_df["label_str"].values[keep].astype(str)
    flow_ids = np.array(
        [_flow_id_to_str(flow_ids_raw[i]) for i in np.where(keep)[0]], dtype=object)
    # local_idx (within this source) is the row index *before* dropping unknowns
    # so consumers can slice the underlying .npy files correctly.
    local_idx = np.where(keep)[0].astype(np.int64)
    return y, flow_ids, label_str, local_idx, feature_names


def cap_benign(
    y: np.ndarray,
    scoped_flow_ids: np.ndarray,
    ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Globally cap benign windows at ratio × adversarial windows, flow-atomic.

    scoped_flow_ids are expected to already encode the source (e.g.
    ``"mirai|1.2.3.4|..."``) so benign flows from different folders don't
    collide.
    """
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return np.ones(len(y), dtype=bool)
    cap = int(ratio * n_pos)
    benign_mask = y == 0
    n_benign = int(benign_mask.sum())
    if n_benign <= cap:
        return np.ones(len(y), dtype=bool)

    benign_flows = np.unique(scoped_flow_ids[benign_mask])
    rng.shuffle(benign_flows)
    flow_to_count = pd.Series(scoped_flow_ids[benign_mask]).value_counts().to_dict()

    keep_flows: set[str] = set()
    kept = 0
    for f in benign_flows:
        if kept >= cap:
            break
        keep_flows.add(f)
        kept += flow_to_count.get(f, 0)

    mask = np.ones(len(y), dtype=bool)
    benign_drop = benign_mask & ~np.isin(scoped_flow_ids, list(keep_flows))
    mask[benign_drop] = False
    return mask


def assemble_index(
    tensors_dir: Path,
    benign_sources: list[str],
    adversarial_dir: Path,
    benign_cap_ratio: float,
    seed: int,
    adversarial_exclude: list[str] = (),
) -> AssembledIndex:
    """Build a per-window index over benign + adversarial sources.

    No heavy payload (``rf_stats.npy``, ``sequence.npy``) is touched here —
    only metadata, labels, and flow ids. Flow-atomic capping at
    ``benign_cap_ratio × n_adversarial`` is applied globally across all
    benign sources.
    """
    rng = np.random.default_rng(seed)
    src_arrs: list[np.ndarray] = []
    local_arrs: list[np.ndarray] = []
    y_arrs: list[np.ndarray] = []
    fid_arrs: list[np.ndarray] = []
    lstr_arrs: list[np.ndarray] = []
    folders: dict[str, Path] = {}
    feature_names: list[str] | None = None

    # Benign pool (Kitsune + CIC folders).
    for src in benign_sources:
        folder = tensors_dir / src
        if not folder.exists():
            raise FileNotFoundError(f"benign source not found: {folder}")
        y, flow_ids, label_str, local_idx, fnames = load_source_meta(folder, src)
        feature_names = feature_names or fnames
        folders[src] = folder
        src_arrs.append(np.full(len(y), src, dtype=object))
        local_arrs.append(local_idx)
        y_arrs.append(y)
        fid_arrs.append(flow_ids)
        lstr_arrs.append(label_str)

    # Adversarial pool (auto-discovered TANTGen outputs).
    adv_names = discover_adversarial_sources(adversarial_dir)
    if not adv_names:
        raise FileNotFoundError(
            f"no adversarial sources under {adversarial_dir}. "
            f"Run scripts/packet_to_tensor/process_all_attacks.py --tantgen first.")
    exclude = set(adversarial_exclude)
    for src in adv_names:
        if src in exclude:
            continue
        folder = adversarial_dir / src
        y, flow_ids, label_str, local_idx, fnames = load_source_meta(folder, src)
        feature_names = feature_names or fnames
        folders[src] = folder
        src_arrs.append(np.full(len(y), src, dtype=object))
        local_arrs.append(local_idx)
        y_arrs.append(y)
        fid_arrs.append(flow_ids)
        lstr_arrs.append(label_str)

    sources = np.concatenate(src_arrs, axis=0)
    local_idx_all = np.concatenate(local_arrs, axis=0)
    y_all = np.concatenate(y_arrs, axis=0)
    flow_ids_all = np.concatenate(fid_arrs, axis=0)
    label_str_all = np.concatenate(lstr_arrs, axis=0)

    scoped_flow = np.array(
        [f"{s}|{f}" for s, f in zip(sources, flow_ids_all)], dtype=object)
    keep = cap_benign(y_all, scoped_flow, benign_cap_ratio, rng)

    return AssembledIndex(
        sources=sources[keep],
        local_idx=local_idx_all[keep],
        y=y_all[keep],
        flow_ids=flow_ids_all[keep],
        label_str=label_str_all[keep],
        feature_names=feature_names or [],
        source_folders=folders,
    )


def flow_level_split(
    sources: np.ndarray,
    flow_ids: np.ndarray,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Split:
    """Stratified, flow-atomic split.

    Each source contributes a single class (benign xor adversarial) under
    ADEPT's label convention, so the stratum is just the source folder.
    Flows are bin-packed largest-first into the split with the greatest
    window-count deficit so every split lands near its target window
    fraction.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    rng = np.random.default_rng(seed)

    scoped_flow = np.array(
        [f"{s}|{f}" for s, f in zip(sources, flow_ids)], dtype=object)

    flow_df = pd.DataFrame({"flow": scoped_flow, "source": sources})
    flow_agg = (
        flow_df.groupby("flow")
        .agg(source=("source", "first"), n=("flow", "size"))
        .reset_index()
    )

    flow_to_split: dict[str, str] = {}
    fracs = {"train": train_frac, "val": val_frac, "test": test_frac}

    for key in sorted(flow_agg["source"].unique()):
        stratum = flow_agg.loc[flow_agg["source"] == key, ["flow", "n"]].reset_index(drop=True)
        perm = rng.permutation(len(stratum))
        stratum = stratum.iloc[perm].reset_index(drop=True)
        stratum = stratum.sort_values("n", ascending=False, kind="stable").reset_index(drop=True)

        total = int(stratum["n"].sum())
        targets = {s: total * f for s, f in fracs.items()}
        current = {"train": 0, "val": 0, "test": 0}
        for _, row in stratum.iterrows():
            flow, n = row["flow"], int(row["n"])
            pick = max(current.keys(), key=lambda s: targets[s] - current[s])
            flow_to_split[flow] = pick
            current[pick] += n

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for i, f in enumerate(scoped_flow):
        s = flow_to_split.get(f)
        if s == "train":
            train_idx.append(i)
        elif s == "val":
            val_idx.append(i)
        elif s == "test":
            test_idx.append(i)

    split = Split(
        train_idx=np.array(sorted(train_idx), dtype=np.int64),
        val_idx=np.array(sorted(val_idx), dtype=np.int64),
        test_idx=np.array(sorted(test_idx), dtype=np.int64),
    )

    tr_f = set(scoped_flow[split.train_idx].tolist())
    va_f = set(scoped_flow[split.val_idx].tolist())
    te_f = set(scoped_flow[split.test_idx].tolist())
    assert not (tr_f & va_f), "leakage: train∩val"
    assert not (tr_f & te_f), "leakage: train∩test"
    assert not (va_f & te_f), "leakage: val∩test"
    return split


def split_summary(
    sources: np.ndarray,
    y: np.ndarray,
    split: Split,
) -> dict:
    def _stats(idx: np.ndarray) -> dict:
        ys = y[idx]
        return {
            "n_windows": int(len(idx)),
            "n_benign": int((ys == 0).sum()),
            "n_adversarial": int((ys == 1).sum()),
            "per_source": {
                s: int((sources[idx] == s).sum())
                for s in sorted(set(sources.tolist()))
            },
        }

    return {
        "train": _stats(split.train_idx),
        "val": _stats(split.val_idx),
        "test": _stats(split.test_idx),
    }
