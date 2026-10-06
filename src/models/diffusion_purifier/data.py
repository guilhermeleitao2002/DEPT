"""Pairs, split, evaluation subsets and flow extraction for phase 9.

Everything here is anchored on the reconstructor's own pair table and split, so the
diffusion purifier trains on exactly the flows DEPT's purifier trained on and is tested
on exactly the flows DEPT's purifier never saw:

  * :func:`load_pairs_and_split` rebuilds the deterministic pair table (as phase 6's
    ``evaluate.py`` does), loads ``split.npz`` and checks it against
    ``split_summary.json``.
  * :func:`strict_test_keys` is the leakage-free subset: the three perturbation modes are
    split independently, so most test pairs have a sibling pair (same original flow,
    another mode) in training, whose clean side both purifiers learned from. The strict
    subset keeps only test pairs whose original flow never appears in training.
  * :func:`read_clean_overlays` reads only the size / IPAT / direction columns of the
    clean flows (the diffusion model's training data).
  * :func:`extract_flows` reads full pairs and, optionally, runs DEPT's reconstructor on
    them — mirroring :func:`src.models.baseline.pipeline.purify_source`, so DEPT's
    overlays here are the ones phase 8 realized.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..common.dataset import flow_level_split
from ..common.runtime import resolve_num_workers
from ..reconstructor.dataset import (
    PairedFlowDataset,
    PairIndex,
    build_pair_index,
    collate_pairs,
    pair_split_summary,
)
from ..reconstructor.evaluate import _round_recon
from ..reconstructor.realize import decode_total_len_batch

log = logging.getLogger(__name__)

_IPAT_COL = 1088
_DIR_COL = 1089


@dataclass
class PairSplit:
    pairs: PairIndex
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    src_names: np.ndarray      # [P] "<base>_adv_<mode>", the phase-8 source name
    origin: str                # where the split came from


def load_pairs_and_split(cfg: dict) -> PairSplit:
    """The reconstructor's pair table + split, verified against its split summary."""
    pairs = build_pair_index(
        tensors_dir=Path(cfg["tensors_dir"]),
        adversarial_dir=Path(cfg["adversarial_dir"]),
        adversarial_exclude=list(cfg.get("adversarial_exclude") or []),
        min_len=int(cfg.get("min_len", 2)),
    )
    recon_dir = Path(cfg["reconstructor_model_dir"])
    split_file = recon_dir / "split.npz"
    if split_file.exists():
        z = np.load(split_file)
        train_idx, val_idx, test_idx = (z["train_idx"].astype(np.int64),
                                        z["val_idx"].astype(np.int64),
                                        z["test_idx"].astype(np.int64))
        origin = str(split_file)
    else:
        sp = cfg["split"]
        split = flow_level_split(
            sources=pairs.adv_source, flow_ids=pairs.flow_id,
            train_frac=float(sp["train_frac"]), val_frac=float(sp["val_frac"]),
            test_frac=float(sp["test_frac"]), seed=int(sp["seed"]))
        train_idx, val_idx, test_idx = split.train_idx, split.val_idx, split.test_idx
        origin = f"recomputed (seed={sp['seed']})"
    n = len(pairs)
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        if len(idx) and int(idx.max()) >= n:
            raise RuntimeError(
                f"{split_file}: {name}_idx points past the pair table ({int(idx.max())} >= {n})"
                " — the tensors changed since phase 6 was trained")
    _check_against_summary(pairs, train_idx, val_idx, test_idx,
                           recon_dir / "split_summary.json")
    src_names = np.array([f"{b}_adv_{m}" for b, m in zip(pairs.base, pairs.mode)],
                         dtype=object)
    log.info("pairs=%d  train=%d val=%d test=%d  (split from %s)",
             n, len(train_idx), len(val_idx), len(test_idx), origin)
    return PairSplit(pairs, train_idx, val_idx, test_idx, src_names, origin)


def _check_against_summary(pairs, train_idx, val_idx, test_idx, summary_path: Path) -> None:
    if not summary_path.exists():
        log.warning("no %s — cannot check that phase 9 uses the reconstructor's split",
                    summary_path)
        return
    want = json.loads(summary_path.read_text())
    got = pair_split_summary(pairs, train_idx, val_idx, test_idx)
    for name in ("train", "val", "test"):
        w = want.get(name, {}).get("per_source", {})
        g = got[name]["per_source"]
        if w != g:
            diff = [(s, w.get(s), g.get(s)) for s in sorted(set(w) | set(g))
                    if w.get(s) != g.get(s)]
            raise RuntimeError(
                f"phase-9 {name} split differs from the reconstructor's ({len(diff)} "
                f"sources, e.g. {diff[:3]} as (source, expected, got)). The tensors or "
                f"the reconstructor config changed since phase 6 was trained.")


def rows_by_source(ps: PairSplit, idx: np.ndarray) -> dict[str, np.ndarray]:
    idx = np.asarray(idx, dtype=np.int64)
    out: dict[str, list] = {}
    for r in idx:
        out.setdefault(str(ps.src_names[r]), []).append(int(r))
    return {s: np.array(v, dtype=np.int64) for s, v in out.items()}


def subsample_per_source(ps: PairSplit, idx: np.ndarray, max_per_source: int | None,
                         seed: int, sources: set[str] | None = None) -> np.ndarray:
    """At most ``max_per_source`` rows of each source (seeded), sorted."""
    rng = np.random.default_rng(seed)
    keep = []
    for s, rows in sorted(rows_by_source(ps, idx).items()):
        if sources is not None and s not in sources:
            continue
        if max_per_source is not None and len(rows) > max_per_source:
            rows = rng.choice(rows, size=int(max_per_source), replace=False)
        keep.append(np.asarray(rows, dtype=np.int64))
    return np.sort(np.concatenate(keep)) if keep else np.zeros(0, dtype=np.int64)


def unique_clean_rows(ps: PairSplit, idx: np.ndarray) -> np.ndarray:
    """First row of each distinct clean flow ``(base, flow_id)`` among ``idx``."""
    seen: set[tuple[str, str]] = set()
    out = []
    for r in np.asarray(idx, dtype=np.int64):
        key = (str(ps.pairs.base[r]), str(ps.pairs.flow_id[r]))
        if key not in seen:
            seen.add(key)
            out.append(int(r))
    return np.array(out, dtype=np.int64)


def train_flow_keys(ps: PairSplit) -> set[tuple[str, str]]:
    """``(base, flow_id)`` of every original flow that appears in a training pair."""
    return {(str(ps.pairs.base[r]), str(ps.pairs.flow_id[r])) for r in ps.train_idx}


def strict_test_keys(ps: PairSplit) -> set[tuple[str, str]]:
    """``(source, flow_id)`` of test pairs whose original flow is in no training pair."""
    seen = train_flow_keys(ps)
    return {(str(ps.src_names[r]), str(ps.pairs.flow_id[r])) for r in ps.test_idx
            if (str(ps.pairs.base[r]), str(ps.pairs.flow_id[r])) not in seen}


# ──────────────────────────────────────────────────────────────────────────────
# Compact reads (training data): size / IPAT / direction columns only
# ──────────────────────────────────────────────────────────────────────────────

class _ColumnReader:
    """Lazy per-source memmaps that read only the columns the purifier uses."""

    def __init__(self, folders: dict[str, Path]) -> None:
        self._folders = {s: Path(p) for s, p in folders.items()}
        self._open: dict[str, tuple] = {}

    def _ensure(self, source: str):
        if source not in self._open:
            folder = self._folders[source]
            self._open[source] = (np.load(folder / "flow_packets.npy", mmap_mode="r"),
                                  np.load(folder / "flow_offsets.npy"))
        return self._open[source]

    def read(self, source: str, local: int, length: int):
        mm, offsets = self._ensure(source)
        start, n = (int(v) for v in offsets[int(local)])
        block = mm[start:start + min(int(length), n)]
        head = np.nan_to_num(np.asarray(block[:, :32], dtype=np.float32))
        ipat = np.nan_to_num(np.asarray(block[:, _IPAT_COL], dtype=np.float32),
                             nan=0.0, posinf=0.0, neginf=0.0)
        direction = np.nan_to_num(np.asarray(block[:, _DIR_COL], dtype=np.float32))
        tl = decode_total_len_batch(head)          # reads columns 16:32 only
        return tl, ipat, np.clip(np.round(direction), 0, 1).astype(np.int8)


class _CleanOverlayDataset(Dataset):
    def __init__(self, pairs: PairIndex, rows: np.ndarray, max_length: int) -> None:
        self._p = pairs
        self._rows = np.asarray(rows, dtype=np.int64)
        self._T = int(max_length)
        self._reader = _ColumnReader(pairs.clean_folders)

    def __len__(self) -> int:
        return int(len(self._rows))

    def __getitem__(self, i: int):
        r = int(self._rows[i])
        L = min(int(self._p.length[r]), self._T)
        tl, ipat, d = self._reader.read(str(self._p.clean_source[r]),
                                        int(self._p.clean_local[r]), L)
        return tl, ipat, d, i


class _FixedCollate:
    """Pad a batch of compact flows to ``T`` packets (a class, so workers can pickle it)."""

    def __init__(self, T: int) -> None:
        self.T = int(T)

    def __call__(self, batch):
        B, T = len(batch), self.T
        tl = np.zeros((B, T), dtype=np.uint16)
        ipat = np.zeros((B, T), dtype=np.float32)
        d = np.zeros((B, T), dtype=np.int8)
        length = np.zeros(B, dtype=np.int16)
        pos = np.zeros(B, dtype=np.int64)
        for b, (t_, i_, d_, p_) in enumerate(batch):
            L = len(t_)
            tl[b, :L] = np.clip(t_, 0, 65535)
            ipat[b, :L] = i_
            d[b, :L] = d_
            length[b] = L
            pos[b] = p_
        return tl, ipat, d, length, pos


def read_clean_overlays(pairs: PairIndex, rows: np.ndarray, max_length: int, *,
                        batch_size: int = 256, num_workers="auto") -> dict:
    """Padded ``[N, max_length]`` size / IPAT / direction arrays of the clean flows."""
    T = int(max_length)
    N = int(len(rows))
    out = {"tl": np.zeros((N, T), dtype=np.uint16),
           "ipat_z": np.zeros((N, T), dtype=np.float32),
           "dir": np.zeros((N, T), dtype=np.int8),
           "length": np.zeros(N, dtype=np.int16)}
    if N == 0:
        return out
    nw = resolve_num_workers(num_workers)
    loader = DataLoader(_CleanOverlayDataset(pairs, rows, T), batch_size=int(batch_size),
                        shuffle=False, num_workers=nw, collate_fn=_FixedCollate(T),
                        persistent_workers=False)
    done, next_log = 0, N // 10
    for tl, ipat, d, length, pos in loader:
        out["tl"][pos] = tl
        out["ipat_z"][pos] = ipat
        out["dir"][pos] = d
        out["length"][pos] = length
        done += len(pos)
        if done >= next_log:
            log.info("  read %d / %d clean flows", done, N)
            next_log += max(N // 10, 1)
    return out


def file_id(path: Path) -> str | None:
    """Short content hash of a file (stages use it to notice a changed upstream)."""
    import hashlib

    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def save_arrays(dirpath: Path, arrays: dict, meta: dict | None = None) -> None:
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    for k, v in arrays.items():
        np.save(dirpath / f"{k}.npy", v)
    (dirpath / "manifest.json").write_text(json.dumps(
        {"arrays": {k: list(v.shape) for k, v in arrays.items()}, **(meta or {})},
        indent=2, default=str))


def load_arrays(dirpath: Path) -> dict:
    dirpath = Path(dirpath)
    man = json.loads((dirpath / "manifest.json").read_text())
    return {k: np.load(dirpath / f"{k}.npy", allow_pickle=True) for k in man["arrays"]}


# ──────────────────────────────────────────────────────────────────────────────
# Full reads (evaluation): adversarial + clean overlays, optionally DEPT's output
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FlowOverlay:
    source: str
    fid: str
    length: int
    adv: tuple            # (total_len int64 [L], ipat_z float32 [L])
    clean: tuple
    direction: np.ndarray  # int8 [L], adversarial side
    dept: tuple | None = None


def extract_flows(ps: PairSplit, rows: np.ndarray, *, recon_model=None,
                  device: torch.device | None = None, batch_size: int = 64,
                  num_workers="auto", max_length: int | None = 200) -> list[FlowOverlay]:
    """Per-flow overlays for ``rows``; ``dept`` is filled when a reconstructor is given."""
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) == 0:
        return []
    ds = PairedFlowDataset(ps.pairs, sample_indices=rows, max_length=max_length)
    nw = resolve_num_workers(num_workers)
    device = device or torch.device("cpu")
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=nw,
                        pin_memory=device.type == "cuda", persistent_workers=False,
                        collate_fn=collate_pairs)
    out: list[FlowOverlay] = []
    for adv, clean, lengths, mask, rows_b in loader:
        rr = None
        if recon_model is not None:
            with torch.inference_mode():
                recon = recon_model(adv.to(device, non_blocking=True),
                                    key_padding_mask=~mask.to(device, non_blocking=True))
            rr = _round_recon(recon).detach().cpu().numpy()
        adv_np, clean_np = adv.numpy(), clean.numpy()
        for b in range(adv_np.shape[0]):
            L = int(lengths[b])
            row = int(rows_b[b])
            a, c = adv_np[b, :L], clean_np[b, :L]
            dept = None
            if rr is not None:
                r = rr[b, :L]
                dept = (decode_total_len_batch(r), r[:, _IPAT_COL].astype(np.float32))
            out.append(FlowOverlay(
                source=str(ps.src_names[row]), fid=str(ps.pairs.flow_id[row]), length=L,
                adv=(decode_total_len_batch(a), a[:, _IPAT_COL].astype(np.float32)),
                clean=(decode_total_len_batch(c), c[:, _IPAT_COL].astype(np.float32)),
                direction=np.clip(np.round(a[:, _DIR_COL]), 0, 1).astype(np.int8),
                dept=dept))
    return out


__all__ = [
    "PairSplit", "FlowOverlay", "load_pairs_and_split", "rows_by_source",
    "subsample_per_source", "unique_clean_rows", "train_flow_keys", "strict_test_keys",
    "read_clean_overlays", "save_arrays", "load_arrays", "extract_flows", "file_id",
]
