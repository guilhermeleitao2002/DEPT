"""
Build the dual-output tensors for ADEPT's *streaming-native* ensemble.

Each source folder under ``data/tensors/<source>/`` now stores **ragged
per-flow** data instead of fixed N-packet windows. Two memory-mapped views
support the new ensemble:

    1. flow_packets.npy             [P_total, F=1090] float32
       Per-packet nPrint bits + IPAT (z-score) + Direction, concatenated
       across all flows in flow-arrival order.
    2. flow_offsets.npy             [N_flows, 2]      int64
       (start_idx, length) into ``flow_packets.npy`` for each flow.
    3. rf_checkpoints.npy           [C_total, S=24]   float32
       Aggregated statistics over the **growing prefix** of each flow,
       emitted every ``rf_checkpoint_stride`` packets (default 10). For a
       flow of length L this gives ``floor(L / stride)`` checkpoints,
       computed over packets [0:stride], [0:2*stride], …, [0:L_floor].
    4. rf_checkpoint_offsets.npy    [N_flows, 2]      int64
       (start_idx, n_checkpoints) into ``rf_checkpoints.npy``.

The streaming-native architecture means **TCN reads full per-flow sequences**
(variable length) and emits a per-packet prediction stream, while **RF reads
multi-scale prefix checkpoints** and emits one prediction per checkpoint.
The meta-learner combines both streams.

Implementation notes
--------------------
- All RF prefix statistics are computed via Welford's online algorithm so a
  flow of length L is processed in O(L) time and O(1) extra memory per
  flow. The emitted ``[mean, std, min, max]`` per base feature matches the
  semantics of the old ``_compute_rf_stats`` but over the growing prefix
  instead of the latest N-packet window.
- Output tensors are pre-allocated as memory-mapped ``.npy`` files using the
  same pattern as the previous windowed builder, so peak RAM stays bounded
  even when a source's ``flow_packets.npy`` exceeds available memory.
"""

import json
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .pcap_parser import NPRINT_COLUMNS

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None


# ──────────────────────────────────────────────────────────────────────────────
# Standardization
# ──────────────────────────────────────────────────────────────────────────────

def _load_or_compute_stats(
    df: pd.DataFrame,
    columns: List[str],
    mean_std_path: str,
) -> Dict[str, Tuple[float, float]]:
    """Load existing mean/std from JSON, or compute & save them."""
    if os.path.exists(mean_std_path):
        print(f"[i] Loading mean/std from {mean_std_path}")
        with open(mean_std_path) as f:
            stats = json.load(f)
        return stats

    stats = {}
    for col in columns:
        if col not in df.columns:
            continue
        mean = float(df[col].mean())
        std = float(df[col].std())
        if std == 0:
            std = 1.0
        stats[col] = {"mean": mean, "std": std}

    os.makedirs(os.path.dirname(mean_std_path) or ".", exist_ok=True)
    with open(mean_std_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[+] Saved mean/std to {mean_std_path}")
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Streaming RF statistics
# ──────────────────────────────────────────────────────────────────────────────

# Aggregations supported in the emitted RF checkpoint vector. The order
# ``[mean, std, min, max]`` is mirrored from the previous windowed builder.
_SUPPORTED_AGGS = ("mean", "std", "min", "max")


class StreamingStats:
    """Per-flow running (Welford) mean/std/min/max for the RF base features.

    Maintains ``(count, mean, M2, min, max)`` per feature vector entry. The
    ``emit`` method returns the flattened ``[feat0_mean, feat0_std, feat0_min,
    feat0_max, feat1_mean, …]`` vector that the RF base learner consumes,
    matching the configured ``rf_base_features × rf_agg_functions`` order.
    """

    __slots__ = ("n", "mean", "_M2", "min", "max", "_n_feat")

    def __init__(self, n_features: int) -> None:
        self._n_feat = int(n_features)
        self.n = 0
        self.mean = np.zeros(self._n_feat, dtype=np.float64)
        self._M2 = np.zeros(self._n_feat, dtype=np.float64)
        self.min = np.full(self._n_feat, np.inf, dtype=np.float64)
        self.max = np.full(self._n_feat, -np.inf, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        """Fold one packet's [n_features] vector into the accumulator."""
        x = np.asarray(x, dtype=np.float64)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self._M2 += delta * delta2
        np.minimum(self.min, x, out=self.min)
        np.maximum(self.max, x, out=self.max)

    def emit(self, agg_functions: List[str]) -> np.ndarray:
        """Return ``[feat × agg]`` flattened vector for the current prefix.

        Uses sample std (ddof=1, matching pandas ``.std()`` and the previous
        windowed ``np.std`` semantics with ``ddof=0`` — see note). For
        single-packet prefixes std is 0.

        Note on std semantics: the previous batch builder used ``np.std``
        with default ``ddof=0`` (population std). We preserve that here for
        consistency: ``std = sqrt(M2 / n)``. Switch to ``ddof=1`` only if
        ``rf_agg_functions`` is later versioned.
        """
        if self.n == 0:
            return np.zeros(self._n_feat * len(agg_functions), dtype=np.float32)
        std = np.sqrt(self._M2 / self.n)  # ddof=0 (matches old np.std default)
        out = np.empty(self._n_feat * len(agg_functions), dtype=np.float32)
        for i in range(self._n_feat):
            for j, agg in enumerate(agg_functions):
                k = i * len(agg_functions) + j
                if agg == "mean":
                    out[k] = self.mean[i]
                elif agg == "std":
                    out[k] = std[i]
                elif agg == "min":
                    out[k] = self.min[i]
                elif agg == "max":
                    out[k] = self.max[i]
                else:
                    raise ValueError(
                        f"Unsupported RF aggregation {agg!r}. Supported: "
                        f"{_SUPPORTED_AGGS}"
                    )
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def build_tensors(
    nprint_bits: np.ndarray,
    meta_df: pd.DataFrame,
    flow_indices: OrderedDict,
    output_dir: str,
    rf_checkpoint_stride: int = 10,
    rf_base_features: Optional[List[str]] = None,
    rf_agg_functions: Optional[List[str]] = None,
    standardize_columns: Optional[List[str]] = None,
    mean_std_path: str = "mean_std.json",
) -> Dict:
    """
    Convert per-flow data into ADEPT's ragged streaming tensors.

    Writes four memory-mapped artifacts (``flow_packets.npy``,
    ``flow_offsets.npy``, ``rf_checkpoints.npy``,
    ``rf_checkpoint_offsets.npy``) plus ``flow_ids.npy`` directly to
    ``output_dir``. Output never fully resides in RAM.

    Parameters
    ----------
    nprint_bits : np.ndarray, shape (P, 1088), dtype int8
        Compact nPrint bit matrix (sorted same as meta_df).
    meta_df : pd.DataFrame
        Metadata with IPAT and Direction columns (sorted by Timestamp).
    flow_indices : OrderedDict[tuple, np.ndarray]
        Flow key → array of row indices into nprint_bits / meta_df. Keys are
        iterated in insertion order, which defines the per-flow ordering in
        the output ``flow_packets.npy``.
    output_dir : str
        Directory to write tensors into.
    rf_checkpoint_stride : int
        Emit a multi-scale RF feature row every ``stride`` packets. The
        first checkpoint covers packets ``[0:stride]``, the next
        ``[0:2*stride]``, etc.
    rf_base_features : list[str]
        Metadata columns to aggregate for the RF input.
    rf_agg_functions : list[str]
        Aggregation function names in ``{mean, std, min, max}``.
    standardize_columns : list[str]
        Columns to z-score normalize (typically just IPAT).
    mean_std_path : str
        Path to load/save normalization statistics.

    Returns
    -------
    dict with keys:
        "flow_packets_path"           : str   — path to flow_packets.npy
        "flow_offsets_path"           : str   — path to flow_offsets.npy
        "rf_checkpoints_path"         : str   — path to rf_checkpoints.npy
        "rf_checkpoint_offsets_path"  : str   — path to rf_checkpoint_offsets.npy
        "flow_ids"                    : list  — flow key per flow
        "stats"                       : dict  — normalization statistics
        "feature_names"               : list  — sequence column order
        "rf_stat_names"               : list  — RF stat column names
        "num_flows"                   : int
        "num_packets"                 : int   — total packets across flows
        "num_checkpoints"             : int   — total RF checkpoint rows
        "rf_checkpoint_stride"        : int
    """
    # Defaults
    if rf_base_features is None:
        rf_base_features = [
            "IPAT", "Packet_Size", "Payload_Length",
            "Payload_Entropy", "TCP_Flags", "Direction",
        ]
    if rf_agg_functions is None:
        rf_agg_functions = list(_SUPPORTED_AGGS)
    if standardize_columns is None:
        standardize_columns = ["IPAT"]
    if rf_checkpoint_stride < 1:
        raise ValueError(
            f"rf_checkpoint_stride must be >=1, got {rf_checkpoint_stride}"
        )

    unknown = [a for a in rf_agg_functions if a not in _SUPPORTED_AGGS]
    if unknown:
        raise ValueError(
            f"Unsupported RF aggregations {unknown}. Supported: {_SUPPORTED_AGGS}"
        )

    # --- Step 1: Global statistics for standardization -----------------------
    stats = _load_or_compute_stats(meta_df, standardize_columns, mean_std_path)

    sequence_features = list(NPRINT_COLUMNS) + ["IPAT", "Direction"]
    n_seq_features = len(sequence_features)   # 1090

    rf_meta_cols = [c for c in rf_base_features if c in meta_df.columns]
    rf_stat_names = [
        f"{feat}_{agg}"
        for feat in rf_meta_cols
        for agg in rf_agg_functions
    ]
    n_rf_stats = len(rf_stat_names)
    n_base = len(rf_meta_cols)

    # Pre-extract metadata as numpy arrays for fast per-packet access.
    rf_meta_values = meta_df[rf_meta_cols].values.astype(np.float32)
    ipat_values = meta_df["IPAT"].values.astype(np.float64)
    direction_values = meta_df["Direction"].values.astype(np.float32)
    ipat_mean = stats.get("IPAT", {}).get("mean", 0.0)
    ipat_std = stats.get("IPAT", {}).get("std", 1.0)

    # --- Step 2: Pre-count packets and checkpoints, build offset tables -----
    flow_keys = list(flow_indices.keys())
    n_flows = len(flow_keys)

    flow_packet_offsets = np.zeros((n_flows, 2), dtype=np.int64)
    rf_checkpoint_offsets = np.zeros((n_flows, 2), dtype=np.int64)

    pkt_cursor = 0
    ckpt_cursor = 0
    for i, fk in enumerate(flow_keys):
        L = int(len(flow_indices[fk]))
        flow_packet_offsets[i, 0] = pkt_cursor
        flow_packet_offsets[i, 1] = L
        n_ckpt = L // rf_checkpoint_stride
        rf_checkpoint_offsets[i, 0] = ckpt_cursor
        rf_checkpoint_offsets[i, 1] = n_ckpt
        pkt_cursor += L
        ckpt_cursor += n_ckpt

    total_packets = int(pkt_cursor)
    total_checkpoints = int(ckpt_cursor)

    if total_packets == 0:
        print("[!] No flows / no packets to write.")
        return {
            "flow_packets_path": None,
            "flow_offsets_path": None,
            "rf_checkpoints_path": None,
            "rf_checkpoint_offsets_path": None,
            "flow_ids": [],
            "stats": stats,
            "feature_names": sequence_features,
            "rf_stat_names": rf_stat_names,
            "num_flows": 0,
            "num_packets": 0,
            "num_checkpoints": 0,
            "rf_checkpoint_stride": rf_checkpoint_stride,
        }

    os.makedirs(output_dir, exist_ok=True)
    flow_packets_path = os.path.join(output_dir, "flow_packets.npy")
    flow_offsets_path = os.path.join(output_dir, "flow_offsets.npy")
    rf_checkpoints_path = os.path.join(output_dir, "rf_checkpoints.npy")
    rf_checkpoint_offsets_path = os.path.join(
        output_dir, "rf_checkpoint_offsets.npy",
    )

    # Memory-mapped output: scattered per-flow writes are fine — each flow's
    # packets occupy a contiguous slice of flow_packets.npy.
    pkt_mmap = np.lib.format.open_memmap(
        flow_packets_path, mode="w+", dtype=np.float32,
        shape=(total_packets, n_seq_features),
    )
    if total_checkpoints > 0:
        ckpt_mmap = np.lib.format.open_memmap(
            rf_checkpoints_path, mode="w+", dtype=np.float32,
            shape=(total_checkpoints, n_rf_stats),
        )
    else:
        # Some sources may have only sub-stride flows; still write a 0-row file
        # so downstream loaders can stat it consistently.
        ckpt_mmap = np.lib.format.open_memmap(
            rf_checkpoints_path, mode="w+", dtype=np.float32,
            shape=(0, n_rf_stats),
        )

    # --- Step 3: Per-flow streaming write -----------------------------------
    flow_iter = enumerate(flow_keys)
    if _tqdm is not None:
        pbar = _tqdm(
            flow_iter, total=n_flows,
            desc="      Building per-flow tensors", unit=" flow",
            dynamic_ncols=True,
        )
    else:
        pbar = flow_iter

    for i, fk in pbar:
        indices = flow_indices[fk]
        L = int(len(indices))
        if L == 0:
            continue

        flow_start = int(flow_packet_offsets[i, 0])
        ckpt_start = int(rf_checkpoint_offsets[i, 0])

        # Write the per-flow [L, 1090] block directly into the memmap.
        pkt_mmap[flow_start:flow_start + L, :1088] = (
            nprint_bits[indices].astype(np.float32)
        )
        pkt_mmap[flow_start:flow_start + L, 1088] = (
            (ipat_values[indices] - ipat_mean) / ipat_std
        ).astype(np.float32)
        pkt_mmap[flow_start:flow_start + L, 1089] = direction_values[indices]

        # Streaming aggregation over the growing prefix → multi-scale RF rows.
        rf_meta_flow = rf_meta_values[indices]
        stat = StreamingStats(n_base)
        ckpt_local = 0
        for t in range(L):
            stat.update(rf_meta_flow[t])
            if (t + 1) % rf_checkpoint_stride == 0:
                if ckpt_local < int(rf_checkpoint_offsets[i, 1]):
                    ckpt_mmap[ckpt_start + ckpt_local] = stat.emit(rf_agg_functions)
                ckpt_local += 1

        if _tqdm is not None and i % 1000 == 0:
            pbar.set_postfix(
                flows=i + 1, packets=int(flow_packet_offsets[i, 0]) + L,
            )

    if _tqdm is not None:
        pbar.close()

    # Persist offsets and flush memmaps.
    np.save(flow_offsets_path, flow_packet_offsets)
    np.save(rf_checkpoint_offsets_path, rf_checkpoint_offsets)
    pkt_mmap.flush()
    ckpt_mmap.flush()

    print(f"[+] Built ragged tensors:")
    print(f"    flow_packets    : {pkt_mmap.shape}   "
          f"(P, F={n_seq_features})")
    print(f"    flow_offsets    : {flow_packet_offsets.shape}  "
          f"(N_flows, 2)")
    print(f"    rf_checkpoints  : {ckpt_mmap.shape}  "
          f"(C, S={n_rf_stats})  stride={rf_checkpoint_stride}")
    print(f"    rf_ckpt_offsets : {rf_checkpoint_offsets.shape}  "
          f"(N_flows, 2)")

    return {
        "flow_packets_path": flow_packets_path,
        "flow_offsets_path": flow_offsets_path,
        "rf_checkpoints_path": rf_checkpoints_path,
        "rf_checkpoint_offsets_path": rf_checkpoint_offsets_path,
        "flow_ids": flow_keys,
        "stats": stats,
        "feature_names": sequence_features,
        "rf_stat_names": rf_stat_names,
        "num_flows": n_flows,
        "num_packets": total_packets,
        "num_checkpoints": total_checkpoints,
        "rf_checkpoint_stride": rf_checkpoint_stride,
    }
