"""Load, merge, and flow-level split rf_stats tensors across attacks."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


BENIGN_LABEL_STR = "benign"
UNKNOWN_LABEL_STR = "unknown"


@dataclass
class RFDataset:
    X: np.ndarray               # [N, F] float32
    y: np.ndarray               # [N]    int8   (0 benign, 1 attack)
    flow_ids: np.ndarray        # [N]    str    (stringified 5-tuple)
    attacks: np.ndarray         # [N]    str    (source attack folder)
    label_str: np.ndarray       # [N]    str    (raw label_str)
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
    """Return (label, keep). label is 0 benign / 1 attack; keep=False for unknown."""
    s = str(raw).strip().lower()
    if s == UNKNOWN_LABEL_STR or s == "" or s == "nan":
        return 0, False
    if s == BENIGN_LABEL_STR:
        return 0, True
    return 1, True


def load_attack(tensors_dir: Path, attack: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, flow_ids, label_str, feature_names) for a single attack folder, with 'unknown' dropped."""
    folder = tensors_dir / attack
    X = np.load(folder / "rf_stats.npy").astype(np.float32, copy=False)
    flow_ids_raw = np.load(folder / "flow_ids.npy", allow_pickle=True)
    labels_df = pd.read_csv(folder / "window_labels.csv")
    meta = json.loads((folder / "metadata.json").read_text())
    feature_names = list(meta.get("rf_stat_names", [f"f{i}" for i in range(X.shape[1])]))

    if len(labels_df) != X.shape[0]:
        raise ValueError(f"{attack}: rows mismatch ({len(labels_df)} labels vs {X.shape[0]} stats)")
    if len(flow_ids_raw) != X.shape[0]:
        raise ValueError(f"{attack}: flow_ids rows mismatch")

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
    flow_ids = np.array([_flow_id_to_str(flow_ids_raw[i]) for i in np.where(keep)[0]], dtype=object)
    return X, y, flow_ids, label_str, feature_names


def cap_benign(y: np.ndarray, flow_ids: np.ndarray, ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Cap benign *windows* at ratio × attack windows, but drop whole flows (not individual windows)."""
    n_attack = int((y == 1).sum())
    if n_attack == 0:
        return np.ones(len(y), dtype=bool)
    cap = int(ratio * n_attack)
    benign_mask = y == 0
    n_benign = int(benign_mask.sum())
    if n_benign <= cap:
        return np.ones(len(y), dtype=bool)

    benign_flows = np.unique(flow_ids[benign_mask])
    rng.shuffle(benign_flows)
    keep_flows: set[str] = set()
    kept = 0
    flow_to_count = pd.Series(flow_ids[benign_mask]).value_counts().to_dict()
    for f in benign_flows:
        if kept >= cap:
            break
        keep_flows.add(f)
        kept += flow_to_count.get(f, 0)

    mask = np.ones(len(y), dtype=bool)
    benign_drop = benign_mask & ~np.isin(flow_ids, list(keep_flows))
    mask[benign_drop] = False
    return mask


def assemble(
    tensors_dir: Path,
    attacks: list[str],
    benign_cap_ratio: float,
    seed: int,
) -> RFDataset:
    rng = np.random.default_rng(seed)
    Xs, ys, fids, atks, lstrs = [], [], [], [], []
    feature_names: list[str] | None = None

    for attack in attacks:
        X, y, flow_ids, label_str, fnames = load_attack(tensors_dir, attack)
        feature_names = feature_names or fnames
        mask = cap_benign(y, flow_ids, benign_cap_ratio, rng)
        Xs.append(X[mask])
        ys.append(y[mask])
        fids.append(flow_ids[mask])
        lstrs.append(label_str[mask])
        atks.append(np.full(int(mask.sum()), attack, dtype=object))

    X = np.concatenate(Xs, axis=0)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return RFDataset(
        X=X,
        y=np.concatenate(ys, axis=0),
        flow_ids=np.concatenate(fids, axis=0),
        attacks=np.concatenate(atks, axis=0),
        label_str=np.concatenate(lstrs, axis=0),
        feature_names=feature_names or [],
    )


def flow_level_split(
    dataset: RFDataset,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Split:
    """Split flows (not windows) within each (attack, label) stratum. Windows of a flow stay together."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    rng = np.random.default_rng(seed)

    # Scope flow id by attack so the same 5-tuple reused by two attacks doesn't merge.
    scoped_flow = np.array(
        [f"{a}|{f}" for a, f in zip(dataset.attacks, dataset.flow_ids)], dtype=object
    )

    # A single flow can contain both benign and attack windows (CIC label transitions mid-flow).
    # Assign each scoped_flow one stratum via its majority label, so flows remain atomic.
    flow_df = pd.DataFrame({"flow": scoped_flow, "attack": dataset.attacks, "y": dataset.y})
    flow_majority = (
        flow_df.groupby("flow")
        .agg(attack=("attack", "first"), y_mean=("y", "mean"))
        .reset_index()
    )
    flow_majority["y_dom"] = (flow_majority["y_mean"] >= 0.5).astype(int)
    flow_majority["stratum"] = (
        flow_majority["attack"].astype(str) + "|" + flow_majority["y_dom"].astype(str)
    )
    flow_to_split: dict[str, str] = {}

    for key in sorted(flow_majority["stratum"].unique()):
        flows_in_stratum = flow_majority.loc[flow_majority["stratum"] == key, "flow"].values
        flows_in_stratum = np.array(flows_in_stratum, dtype=object)
        rng.shuffle(flows_in_stratum)
        n = len(flows_in_stratum)
        n_tr = int(round(n * train_frac))
        n_va = int(round(n * val_frac))
        for f in flows_in_stratum[:n_tr]:
            flow_to_split[f] = "train"
        for f in flows_in_stratum[n_tr : n_tr + n_va]:
            flow_to_split[f] = "val"
        for f in flows_in_stratum[n_tr + n_va :]:
            flow_to_split[f] = "test"

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

    # Leakage assertion: no scoped_flow appears in more than one split.
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
            "n_attack": int((y == 1).sum()),
            "per_attack": {
                a: int((dataset.attacks[idx] == a).sum())
                for a in sorted(set(dataset.attacks.tolist()))
            },
        }

    return {
        "train": _stats(split.train_idx),
        "val": _stats(split.val_idx),
        "test": _stats(split.test_idx),
    }
