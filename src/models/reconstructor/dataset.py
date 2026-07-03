"""Paired (adversarial → clean) per-flow dataset for the reconstructor.

The reconstructor learns to map an adversarially-perturbed flow back onto its
clean counterpart. Training examples are therefore **pairs** of flows — the
``<base>_adv_<mode>`` flow and the matching ``<base>`` clean flow — aligned
**by 5-tuple** (never by row position, since TANTGen can split/merge/drop
flows so the two folders are not row-aligned).

For each adversarial source we:
  1. resolve its clean base folder (strip ``_adv_<mode>``; skip sources whose
     clean base is absent — e.g. ``ddos_kitsune`` has no clean counterpart);
  2. match flows by their stringified canonical 5-tuple (``flow_ids.npy``);
  3. keep matched pairs, truncating both to ``min(len_adv, len_clean)`` packets
     (the rare ~0.2% of pairs whose packet counts diverge) and dropping pairs
     shorter than ``min_len``.

Both sides are read lazily from the per-source memmaps via the TCN's
``FlowMemmapCache``. Batching right-pads both sequences to the batch max and
returns a shared validity mask, mirroring ``tcn/dataset.collate_padded``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..common.dataset import discover_adversarial_sources, load_source_meta
from ..tcn.dataset import FlowMemmapCache


ADV_MODE_SUFFIXES = ("_adv_both", "_adv_iat", "_adv_size")


def _base_source(adv_source: str) -> str | None:
    """Strip '_adv_<mode>' → base attack name (or None if not an adv source)."""
    for suffix in ADV_MODE_SUFFIXES:
        if adv_source.endswith(suffix):
            return adv_source[: -len(suffix)]
    return None


def _mode(adv_source: str) -> str:
    for suffix in ADV_MODE_SUFFIXES:
        if adv_source.endswith(suffix):
            return suffix[len("_adv_"):]
    return ""


def _load_lengths(folder: Path) -> np.ndarray:
    """Per-flow packet lengths from ``flow_offsets.npy`` (column 1)."""
    return np.load(folder / "flow_offsets.npy")[:, 1].astype(np.int64)


@dataclass
class PairIndex:
    """Parallel arrays describing matched (adversarial, clean) flow pairs."""
    adv_source: np.ndarray      # [P] str    adversarial source folder name
    adv_local: np.ndarray       # [P] int64  flow idx in adv flow_offsets.npy
    clean_source: np.ndarray    # [P] str    clean base source folder name
    clean_local: np.ndarray     # [P] int64  flow idx in clean flow_offsets.npy
    length: np.ndarray          # [P] int64  matched min(len_adv, len_clean)
    base: np.ndarray            # [P] str    base attack name
    mode: np.ndarray            # [P] str    both | iat | size
    flow_id: np.ndarray         # [P] str    stringified canonical 5-tuple
    adv_folders: dict = field(default_factory=dict)    # adv_source -> Path
    clean_folders: dict = field(default_factory=dict)  # clean_source -> Path
    skipped: dict = field(default_factory=dict)        # diagnostics

    def __len__(self) -> int:
        return int(len(self.adv_source))


def build_pair_index(
    tensors_dir: Path,
    adversarial_dir: Path,
    adversarial_exclude: Sequence[str] = (),
    min_len: int = 2,
) -> PairIndex:
    """Discover and 5-tuple-match every (adversarial, clean) flow pair."""
    tensors_dir = Path(tensors_dir)
    adversarial_dir = Path(adversarial_dir)
    exclude = set(adversarial_exclude)

    adv_names = discover_adversarial_sources(adversarial_dir)
    if not adv_names:
        raise FileNotFoundError(
            f"no adversarial sources under {adversarial_dir}. "
            f"Run scripts/packet_to_tensor/process_all_attacks.py first.")

    a_src, a_loc, c_src, c_loc = [], [], [], []
    lens, bases, modes, fids = [], [], [], []
    adv_folders: dict[str, Path] = {}
    clean_folders: dict[str, Path] = {}
    skipped = {"no_clean_base": [], "excluded": [], "len_mismatch": 0,
               "too_short": 0, "unmatched_adv": 0}

    for adv_src in adv_names:
        if adv_src in exclude:
            skipped["excluded"].append(adv_src)
            continue
        base = _base_source(adv_src)
        if base is None:
            skipped["excluded"].append(adv_src)
            continue
        clean_folder = tensors_dir / base
        if not (clean_folder / "flow_ids.npy").exists():
            skipped["no_clean_base"].append(adv_src)
            continue
        adv_folder = adversarial_dir / adv_src

        _ya, fid_a, _la, local_a, _ = load_source_meta(adv_folder, adv_src)
        _yc, fid_c, _lc, local_c, _ = load_source_meta(clean_folder, base)
        len_a = _load_lengths(adv_folder)
        len_c = _load_lengths(clean_folder)

        clean_map = {str(f): int(l) for f, l in zip(fid_c, local_c)}
        mode = _mode(adv_src)
        for f, la in zip(fid_a, local_a):
            lc = clean_map.get(str(f))
            if lc is None:
                skipped["unmatched_adv"] += 1
                continue
            length = int(min(len_a[int(la)], len_c[int(lc)]))
            if len_a[int(la)] != len_c[int(lc)]:
                skipped["len_mismatch"] += 1
            if length < min_len:
                skipped["too_short"] += 1
                continue
            a_src.append(adv_src)
            a_loc.append(int(la))
            c_src.append(base)
            c_loc.append(int(lc))
            lens.append(length)
            bases.append(base)
            modes.append(mode)
            fids.append(str(f))

        adv_folders[adv_src] = adv_folder
        clean_folders[base] = clean_folder

    if not a_src:
        raise RuntimeError(
            "build_pair_index produced 0 pairs — check that clean base folders "
            f"exist under {tensors_dir} for the adversarial sources.")

    return PairIndex(
        adv_source=np.array(a_src, dtype=object),
        adv_local=np.array(a_loc, dtype=np.int64),
        clean_source=np.array(c_src, dtype=object),
        clean_local=np.array(c_loc, dtype=np.int64),
        length=np.array(lens, dtype=np.int64),
        base=np.array(bases, dtype=object),
        mode=np.array(modes, dtype=object),
        flow_id=np.array(fids, dtype=object),
        adv_folders=adv_folders,
        clean_folders=clean_folders,
        skipped=skipped,
    )


class PairedFlowDataset(Dataset):
    """Lazy reader of matched ``(adv[L, F], clean[L, F])`` flow pairs."""

    def __init__(
        self,
        pairs: PairIndex,
        sample_indices: np.ndarray,
        max_length: int | None = None,
    ) -> None:
        super().__init__()
        self._p = pairs
        self._idx = np.asarray(sample_indices, dtype=np.int64)
        self._adv_cache = FlowMemmapCache(pairs.adv_folders)
        self._clean_cache = FlowMemmapCache(pairs.clean_folders)
        self._max_length = int(max_length) if max_length is not None else None

    def __len__(self) -> int:
        return int(len(self._idx))

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        row = int(self._idx[i])
        L = int(self._p.length[row])
        if self._max_length is not None:
            L = min(L, self._max_length)
        adv = self._adv_cache.slice(str(self._p.adv_source[row]),
                                    int(self._p.adv_local[row]))[:L]
        clean = self._clean_cache.slice(str(self._p.clean_source[row]),
                                        int(self._p.clean_local[row]))[:L]
        adv = np.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)
        clean = np.nan_to_num(clean, nan=0.0, posinf=0.0, neginf=0.0)
        return (torch.from_numpy(adv), torch.from_numpy(clean), int(L), row)


def collate_pairs(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor, int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad both sides to the batch max length.

    Returns ``(adv[B, T, F], clean[B, T, F], lengths[B], mask[B, T], rows[B])``
    where ``mask`` is True at valid timesteps and ``rows`` indexes back into
    the ``PairIndex`` (used by the evaluator for per-source slicing).
    """
    B = len(batch)
    F = int(batch[0][0].shape[1])
    lengths = torch.tensor([int(item[2]) for item in batch], dtype=torch.int64)
    rows = torch.tensor([int(item[3]) for item in batch], dtype=torch.int64)
    T_max = int(lengths.max().item())

    adv = torch.zeros((B, T_max, F), dtype=torch.float32)
    clean = torch.zeros((B, T_max, F), dtype=torch.float32)
    mask = torch.zeros((B, T_max), dtype=torch.bool)
    for b, (a, c, L, _row) in enumerate(batch):
        adv[b, :L] = a
        clean[b, :L] = c
        mask[b, :L] = True
    return adv, clean, lengths, mask, rows


def make_pair_dataloaders(
    pairs: PairIndex,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    max_length: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    common = dict(pairs=pairs, max_length=max_length)
    train_ds = PairedFlowDataset(sample_indices=train_idx, **common)
    val_ds = PairedFlowDataset(sample_indices=val_idx, **common)
    test_ds = PairedFlowDataset(sample_indices=test_idx, **common)

    loader_kwargs = dict(
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=int(num_workers) > 0,
        collate_fn=collate_pairs,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def pair_split_summary(pairs: PairIndex, train_idx, val_idx, test_idx) -> dict:
    """Per-split pair counts + per-source breakdown (source = base+mode)."""
    src = np.array([f"{b}_adv_{m}" for b, m in zip(pairs.base, pairs.mode)],
                   dtype=object)
    all_srcs = sorted(set(src.tolist()))

    def _stats(idx: np.ndarray) -> dict:
        idx = np.asarray(idx, dtype=np.int64)
        return {
            "n_pairs": int(len(idx)),
            "per_source": {s: int((src[idx] == s).sum()) for s in all_srcs},
        }

    return {
        "train": _stats(train_idx),
        "val": _stats(val_idx),
        "test": _stats(test_idx),
    }
