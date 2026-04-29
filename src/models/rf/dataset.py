"""Load, merge, and flow-level split rf_stats tensors under ADEPT's
benign-vs-adversarial label convention.

Benign pool: Kitsune + CIC-IDS2017 folders listed in rf.yaml's
`benign_sources`. Every window in these folders has label_str="benign".

Adversarial pool: auto-discovered subfolders of `adversarial_dir` (one per
TANTGen variant, e.g. `mirai_adv_both`). Every window has
label_str="adversarial".

The split is stratified by source folder — a flow belongs atomically to
one source and therefore to one class, so the label-transition handling
from the previous per-attack setup is no longer needed.
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
class RFDataset:
    X: np.ndarray               # [N, F] float32
    y: np.ndarray               # [N]    int8   (0 benign, 1 adversarial)
    flow_ids: np.ndarray        # [N]    str    (stringified 5-tuple)
    sources: np.ndarray         # [N]    str    (source folder name)
    label_str: np.ndarray       # [N]    str    ("benign" or "adversarial")
    feature_names: list[str] = field(default_factory=list)


@dataclass
class Split:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


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


def load_source(folder: Path, source_name: str) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]
]:
    """Load one source folder. Returns (X, y, flow_ids, label_str, feature_names)."""
    X = np.load(folder / "rf_stats.npy").astype(np.float32, copy=False)
    flow_ids_raw = np.load(folder / "flow_ids.npy", allow_pickle=True)
    labels_df = pd.read_csv(folder / "window_labels.csv")
    meta = json.loads((folder / "metadata.json").read_text())
    feature_names = list(meta.get("rf_stat_names", [f"f{i}" for i in range(X.shape[1])]))

    if len(labels_df) != X.shape[0]:
        raise ValueError(
            f"{source_name}: rows mismatch ({len(labels_df)} labels vs {X.shape[0]} stats)")
    if len(flow_ids_raw) != X.shape[0]:
        raise ValueError(f"{source_name}: flow_ids rows mismatch")

    y_parts, keep_mask = [], []
    for raw in labels_df["label_str"].values:
        lab, keep = _normalize_label(raw)
        y_parts.append(lab)
        keep_mask.append(keep)
    y = np.array(y_parts, dtype=np.int8)
    keep = np.array(keep_mask, dtype=bool)

    X = X[keep]
    y = y[keep]
    label_str = labels_df["label_str"].values[keep].astype(str)
    flow_ids = np.array(
        [_flow_id_to_str(flow_ids_raw[i]) for i in np.where(keep)[0]], dtype=object)
    return X, y, flow_ids, label_str, feature_names


def discover_adversarial_sources(adversarial_dir: Path) -> list[str]:
    """Return all subdir names of adversarial_dir/ that contain metadata.json."""
    if not adversarial_dir.exists():
        return []
    return sorted(
        d.name for d in adversarial_dir.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
    )


def cap_benign(
    y: np.ndarray,
    scoped_flow_ids: np.ndarray,
    ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Globally cap benign windows at ratio × adversarial windows, flow-atomic.

    scoped_flow_ids are expected to already encode the source (e.g.
    "mirai|1.2.3.4|...") so benign flows from different folders don't
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


def assemble(
    tensors_dir: Path,
    benign_sources: list[str],
    adversarial_dir: Path,
    benign_cap_ratio: float,
    seed: int,
    adversarial_exclude: list[str] = (),
) -> RFDataset:
    """Merge benign sources + auto-discovered adversarial sources."""
    rng = np.random.default_rng(seed)
    Xs, ys, fids, srcs, lstrs = [], [], [], [], []
    feature_names: list[str] | None = None

    # Benign pool (Kitsune + CIC folders).
    for src in benign_sources:
        folder = tensors_dir / src
        if not folder.exists():
            raise FileNotFoundError(f"benign source not found: {folder}")
        X, y, flow_ids, label_str, fnames = load_source(folder, src)
        feature_names = feature_names or fnames
        Xs.append(X)
        ys.append(y)
        fids.append(flow_ids)
        lstrs.append(label_str)
        srcs.append(np.full(len(y), src, dtype=object))

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
        X, y, flow_ids, label_str, fnames = load_source(folder, src)
        feature_names = feature_names or fnames
        Xs.append(X)
        ys.append(y)
        fids.append(flow_ids)
        lstrs.append(label_str)
        srcs.append(np.full(len(y), src, dtype=object))

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    flow_ids = np.concatenate(fids, axis=0)
    sources = np.concatenate(srcs, axis=0)
    label_str = np.concatenate(lstrs, axis=0)

    # Scope flow id by source so benign flows from different folders don't merge.
    scoped_flow = np.array(
        [f"{s}|{f}" for s, f in zip(sources, flow_ids)], dtype=object)
    keep = cap_benign(y, scoped_flow, benign_cap_ratio, rng)

    X = np.nan_to_num(X[keep], nan=0.0, posinf=0.0, neginf=0.0)
    return RFDataset(
        X=X,
        y=y[keep],
        flow_ids=flow_ids[keep],
        sources=sources[keep],
        label_str=label_str[keep],
        feature_names=feature_names or [],
    )


def flow_level_split(
    dataset: RFDataset,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Split:
    """Stratified, flow-atomic split.

    Each source contributes a single class (benign xor adversarial) under
    ADEPT's label convention, so the stratum is just the source folder —
    no label-transition handling is required. Flows are bin-packed
    largest-first into the split with the greatest window-count deficit
    so every split lands near its target window fraction.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    rng = np.random.default_rng(seed)

    scoped_flow = np.array(
        [f"{s}|{f}" for s, f in zip(dataset.sources, dataset.flow_ids)], dtype=object)

    flow_df = pd.DataFrame({"flow": scoped_flow, "source": dataset.sources})
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


def split_summary(dataset: RFDataset, split: Split) -> dict:
    def _stats(idx: np.ndarray) -> dict:
        y = dataset.y[idx]
        return {
            "n_windows": int(len(idx)),
            "n_benign": int((y == 0).sum()),
            "n_adversarial": int((y == 1).sum()),
            "per_source": {
                s: int((dataset.sources[idx] == s).sum())
                for s in sorted(set(dataset.sources.tolist()))
            },
        }

    return {
        "train": _stats(split.train_idx),
        "val": _stats(split.val_idx),
        "test": _stats(split.test_idx),
    }
