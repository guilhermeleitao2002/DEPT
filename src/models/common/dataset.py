"""Model-agnostic dataset primitives for ADEPT's streaming base learners.

This module owns the parts of the data pipeline that don't care which
tensor view a model consumes:

  * source discovery (benign + auto-discovered adversarial folders)
  * label normalisation under the benign-vs-adversarial convention
  * source-stratified train/val/test splitting (one row = one flow)
  * benign capping at a configurable ratio of adversarial flows

Under the streaming-native schema, each *row* of the assembled index
represents one **flow**, not one window. The flow's underlying tensor
data is the variable-length slice
``flow_packets.npy[start_p:start_p + length]`` (for TCN) or the
multi-scale prefix rows
``rf_checkpoints.npy[start_c:start_c + n_checkpoints]`` (for RF), keyed
off ``flow_offsets.npy`` / ``rf_checkpoint_offsets.npy`` respectively.
Both base learners reuse ``assemble_index`` to land on the **same**
flow-level split.
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
    """Per-flow index into the on-disk ragged tensors, with no payload loaded.

    Each row identifies one flow by ``(source, local_idx)`` — the source
    folder name and the flow's index within that source's
    ``flow_offsets.npy`` / ``rf_checkpoint_offsets.npy`` tables. Heavy
    payload (``flow_packets.npy``, ``rf_checkpoints.npy``) is loaded by
    the consumer.
    """
    sources: np.ndarray         # [N] str    source folder name
    local_idx: np.ndarray       # [N] int64  flow index within that source
    y: np.ndarray               # [N] int8   0 benign / 1 adversarial
    flow_ids: np.ndarray        # [N] str    stringified canonical 5-tuple
    label_str: np.ndarray       # [N] str    "benign" or "adversarial"
    feature_names: list[str] = field(default_factory=list)
    source_folders: dict[str, Path] = field(default_factory=dict)


def _flow_id_to_str(fid_row) -> str:
    """Stringify a flow-id row (5-tuple object array or already-stringified)."""
    if isinstance(fid_row, (tuple, list, np.ndarray)):
        return "|".join(str(x) for x in fid_row)
    return str(fid_row)


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
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]
]:
    """Read labels/flow_ids/metadata for one source without loading any payload.

    Returns ``(y, flow_ids, label_str, local_idx, rf_stat_names)`` after
    dropping rows whose label_str is empty/unknown. ``local_idx`` is the
    flow's row in this source's ``flow_offsets.npy`` table.
    """
    flow_ids_raw = np.load(folder / "flow_ids.npy", allow_pickle=True)
    labels_df = pd.read_csv(folder / "flow_labels.csv")
    meta = json.loads((folder / "metadata.json").read_text())
    n_flows_meta = int(meta.get("num_flows", len(labels_df)))
    rf_stat_names = list(meta.get("rf_stat_names", []))

    if len(labels_df) != n_flows_meta:
        raise ValueError(
            f"{source_name}: rows mismatch ({len(labels_df)} labels "
            f"vs {n_flows_meta} in metadata)")
    if len(flow_ids_raw) != n_flows_meta:
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
        [_flow_id_to_str(flow_ids_raw[i]) for i in np.where(keep)[0]],
        dtype=object,
    )
    local_idx = np.where(keep)[0].astype(np.int64)
    return y, flow_ids, label_str, local_idx, rf_stat_names


def cap_benign(
    y: np.ndarray,
    scoped_flow_ids: np.ndarray,
    ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Globally cap benign flows at ratio × adversarial flows.

    Under the per-flow schema every row is already a unique flow, so this
    function selects a random subset of benign flows. ``scoped_flow_ids``
    is expected to encode the source (e.g. ``"mirai|1.2.3.4|..."``) so
    benign flows from different folders don't collide.
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
    keep_flows = set(benign_flows[:cap].tolist())

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
    """Build a per-flow index over benign + adversarial sources.

    No heavy payload (``flow_packets.npy``, ``rf_checkpoints.npy``) is
    touched here — only metadata, labels, and flow ids. Flow-level
    capping at ``benign_cap_ratio × n_adversarial_flows`` is applied
    globally across all benign sources.
    """
    rng = np.random.default_rng(seed)
    src_arrs: list[np.ndarray] = []
    local_arrs: list[np.ndarray] = []
    y_arrs: list[np.ndarray] = []
    fid_arrs: list[np.ndarray] = []
    lstr_arrs: list[np.ndarray] = []
    folders: dict[str, Path] = {}
    feature_names: list[str] | None = None

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
    """Stratified split. One row per flow, one source per stratum.

    Each source contributes a single class (benign xor adversarial) under
    ADEPT's label convention, so the stratum is just the source folder.
    Flows are bin-packed into the split with the greatest deficit so every
    split lands near its target flow fraction.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    rng = np.random.default_rng(seed)

    scoped_flow = np.array(
        [f"{s}|{f}" for s, f in zip(sources, flow_ids)], dtype=object)

    flow_to_split: dict[str, str] = {}
    fracs = {"train": train_frac, "val": val_frac, "test": test_frac}

    for src_key in sorted(np.unique(sources).tolist()):
        mask = sources == src_key
        stratum_flows = scoped_flow[mask]
        perm = rng.permutation(len(stratum_flows))
        stratum_flows = stratum_flows[perm]
        total = len(stratum_flows)
        targets = {s: total * f for s, f in fracs.items()}
        current = {"train": 0, "val": 0, "test": 0}
        for f in stratum_flows:
            pick = max(current.keys(), key=lambda s: targets[s] - current[s])
            flow_to_split[str(f)] = pick
            current[pick] += 1

    train_idx, val_idx, test_idx = [], [], []
    for i, f in enumerate(scoped_flow):
        s = flow_to_split.get(str(f))
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
            "n_flows": int(len(idx)),
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
