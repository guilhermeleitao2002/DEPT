"""TCN dataset — lazy memmap reader over per-source ragged flow tensors.

Each source folder stores ``flow_packets.npy`` ([P_total, 1090]) and a
sibling ``flow_offsets.npy`` ([N_flows, 2]) giving each flow's
``(start_idx, length)`` slice. ``FlowSequenceDataset.__getitem__`` reads
one variable-length ``[L, F]`` slice directly from the right memmap and
returns it alongside the flow's label and length.

Batching uses a custom ``collate_fn`` that right-pads the per-flow
sequences to the batch max length, yielding ``(seq[B, T_max, F],
labels[B], lengths[B], mask[B, T_max])``. The padding mask is used by
the per-packet CE loss in :mod:`src.models.tcn.train` to ignore padded
timesteps.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class _SourceMmaps:
    """Per-source memory-maps + offset table.

    ``packets`` is a memmap of ``flow_packets.npy``; ``offsets`` is the
    loaded ``flow_offsets.npy`` (small — [N_flows, 2]). Memmaps open
    lazily so the dataset is cheap to pickle to DataLoader workers.
    """
    folder: Path
    packets: np.memmap | None = None
    offsets: np.ndarray | None = None


class FlowMemmapCache:
    """Lazy per-source memmap pool, worker-safe under DataLoader fan-out."""

    def __init__(self, source_folders: dict[str, Path]) -> None:
        self._folders = {s: Path(p) for s, p in source_folders.items()}
        self._open: dict[str, _SourceMmaps] = {}

    def _ensure(self, source: str) -> _SourceMmaps:
        cached = self._open.get(source)
        if cached is not None and cached.packets is not None:
            return cached
        folder = self._folders[source]
        cached = _SourceMmaps(
            folder=folder,
            packets=np.load(folder / "flow_packets.npy", mmap_mode="r"),
            offsets=np.load(folder / "flow_offsets.npy"),
        )
        self._open[source] = cached
        return cached

    def slice(self, source: str, local_idx: int) -> np.ndarray:
        mm = self._ensure(source)
        start, length = mm.offsets[local_idx]
        return np.asarray(mm.packets[int(start):int(start) + int(length)],
                          dtype=np.float32)


class FlowSequenceDataset(Dataset):
    """Lazy per-flow ``[L, F]`` reader.

    Args:
        sources: per-row source folder name, length ``N``.
        local_idx: per-row flow index into that source's
            ``flow_offsets.npy``, length ``N``.
        y: per-row flow labels (0/1), length ``N``.
        sample_indices: which rows of ``sources`` / ``local_idx`` / ``y``
            this split owns.
        source_folders: ``source_name -> Path`` lookup for memmaps.
        max_length: optional truncation cap (None = no cap).
    """

    def __init__(
        self,
        sources: np.ndarray,
        local_idx: np.ndarray,
        y: np.ndarray,
        sample_indices: np.ndarray,
        source_folders: dict[str, Path],
        max_length: int | None = None,
    ) -> None:
        super().__init__()
        self._sources = sources
        self._local_idx = local_idx.astype(np.int64, copy=False)
        self._y = y.astype(np.int64, copy=False)
        self._idx = sample_indices.astype(np.int64, copy=False)
        self._cache = FlowMemmapCache(source_folders)
        self._max_length = int(max_length) if max_length is not None else None

    def __len__(self) -> int:
        return int(len(self._idx))

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, int]:
        row = int(self._idx[i])
        src = str(self._sources[row])
        local = int(self._local_idx[row])
        x = self._cache.slice(src, local)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if self._max_length is not None and x.shape[0] > self._max_length:
            x = x[: self._max_length]
        return torch.from_numpy(x), int(self._y[row]), int(x.shape[0])


def collate_padded(
    batch: Sequence[tuple[torch.Tensor, int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad variable-length sequences to the batch's max length.

    Returns
    -------
    seq : torch.Tensor, shape [B, T_max, F]
        Padded float32 sequences.
    labels : torch.Tensor, shape [B]
        Per-flow int64 labels.
    lengths : torch.Tensor, shape [B]
        Original (pre-padding) flow lengths in int64.
    mask : torch.Tensor, shape [B, T_max]
        ``True`` at valid timesteps, ``False`` at padding.
    """
    B = len(batch)
    F = int(batch[0][0].shape[1])
    lengths = torch.tensor([int(item[2]) for item in batch], dtype=torch.int64)
    T_max = int(lengths.max().item())

    seq = torch.zeros((B, T_max, F), dtype=torch.float32)
    labels = torch.zeros(B, dtype=torch.int64)
    mask = torch.zeros((B, T_max), dtype=torch.bool)
    for b, (x, y, L) in enumerate(batch):
        seq[b, :L] = x
        labels[b] = int(y)
        mask[b, :L] = True
    return seq, labels, lengths, mask


def make_dataloaders(
    sources: np.ndarray,
    local_idx: np.ndarray,
    y: np.ndarray,
    source_folders: dict[str, Path],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    max_length: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    common_kwargs = dict(
        sources=sources,
        local_idx=local_idx,
        y=y,
        source_folders=source_folders,
        max_length=max_length,
    )
    train_ds = FlowSequenceDataset(sample_indices=train_idx, **common_kwargs)
    val_ds = FlowSequenceDataset(sample_indices=val_idx, **common_kwargs)
    test_ds = FlowSequenceDataset(sample_indices=test_idx, **common_kwargs)

    loader_kwargs = dict(
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=int(num_workers) > 0,
        collate_fn=collate_padded,
    )

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def compute_class_weights(y_train: np.ndarray, mode: str = "balanced") -> torch.Tensor:
    """Inverse-frequency weights for ``CrossEntropyLoss``.

    Mirrors sklearn's ``class_weight='balanced'``: ``w_c = N / (C · n_c)``.
    These are flow-level weights; the per-packet loss broadcasts them
    along the timestep axis.
    """
    if mode == "none":
        return torch.ones(2, dtype=torch.float32)
    if mode != "balanced":
        raise ValueError(f"unknown class_weight mode: {mode}")
    n_total = int(len(y_train))
    n0 = int((y_train == 0).sum())
    n1 = int((y_train == 1).sum())
    n_classes = 2
    w0 = n_total / (n_classes * max(n0, 1))
    w1 = n_total / (n_classes * max(n1, 1))
    return torch.tensor([w0, w1], dtype=torch.float32)
