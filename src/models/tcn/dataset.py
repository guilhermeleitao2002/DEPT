"""TCN dataset — lazy memmap reader over per-source ``sequence.npy`` files.

A single source's ``sequence.npy`` can be ~17 GB, so we never concatenate
them into RAM. Instead the shared ``assemble_index`` builds a per-window
index ``(source, local_idx)`` and ``TCNWindowDataset.__getitem__`` reads
one ``[T, F]`` slice at a time directly from the right memmap.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class _SourceMmap:
    path: Path
    array: np.memmap


class WindowMemmapCache:
    """Lazy per-source memmap pool.

    DataLoader workers each construct their own cache via
    ``Dataset.__init__`` -> first ``__getitem__``, so file handles are
    not shared across processes. The cache itself is just a path lookup
    table; memmaps materialise on first access.
    """

    def __init__(self, source_folders: dict[str, Path]) -> None:
        # Keep paths only at construction time so the dataset is cheap to
        # pickle to DataLoader workers. memmaps open lazily.
        self._folders = {s: Path(p) for s, p in source_folders.items()}
        self._open: dict[str, _SourceMmap] = {}

    def get(self, source: str) -> np.memmap:
        cached = self._open.get(source)
        if cached is not None:
            return cached.array
        path = self._folders[source] / "sequence.npy"
        arr = np.load(path, mmap_mode="r")
        self._open[source] = _SourceMmap(path=path, array=arr)
        return arr


class TCNWindowDataset(Dataset):
    """Lazy ``[T, F]`` window reader.

    Args:
        sources: per-row source folder name, length ``N``.
        local_idx: per-row index into that source's ``sequence.npy``,
            length ``N``.
        y: per-row labels (0/1), length ``N``.
        sample_indices: which rows of ``sources``/``local_idx``/``y`` this
            split owns (i.e. the train/val/test row ids).
        source_folders: ``source_name -> Path`` lookup for memmaps.
    """

    def __init__(
        self,
        sources: np.ndarray,
        local_idx: np.ndarray,
        y: np.ndarray,
        sample_indices: np.ndarray,
        source_folders: dict[str, Path],
    ) -> None:
        super().__init__()
        self._sources = sources
        self._local_idx = local_idx.astype(np.int64, copy=False)
        self._y = y.astype(np.int64, copy=False)
        self._idx = sample_indices.astype(np.int64, copy=False)
        self._cache = WindowMemmapCache(source_folders)

    def __len__(self) -> int:
        return int(len(self._idx))

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        row = int(self._idx[i])
        src = str(self._sources[row])
        local = int(self._local_idx[row])
        mm = self._cache.get(src)
        x = np.asarray(mm[local], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(x), int(self._y[row])


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
) -> tuple[DataLoader, DataLoader, DataLoader]:
    common_kwargs = dict(
        sources=sources,
        local_idx=local_idx,
        y=y,
        source_folders=source_folders,
    )
    train_ds = TCNWindowDataset(sample_indices=train_idx, **common_kwargs)
    val_ds = TCNWindowDataset(sample_indices=val_idx, **common_kwargs)
    test_ds = TCNWindowDataset(sample_indices=test_idx, **common_kwargs)

    loader_kwargs = dict(
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=int(num_workers) > 0,
    )

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def compute_class_weights(y_train: np.ndarray, mode: str = "balanced") -> torch.Tensor:
    """Inverse-frequency weights for ``CrossEntropyLoss``.

    Mirrors sklearn's ``class_weight='balanced'``: ``w_c = N / (C · n_c)``.
    """
    if mode == "none":
        return torch.ones(2, dtype=torch.float32)
    if mode != "balanced":
        raise ValueError(f"unknown class_weight mode: {mode}")
    n_total = int(len(y_train))
    n0 = int((y_train == 0).sum())
    n1 = int((y_train == 1).sum())
    n_classes = 2
    eps = 1e-9
    w0 = n_total / (n_classes * max(n0, 1))
    w1 = n_total / (n_classes * max(n1, 1))
    return torch.tensor([w0, w1], dtype=torch.float32)
