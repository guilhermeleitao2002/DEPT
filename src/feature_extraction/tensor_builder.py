"""
Build the dual-output tensors for DEPT's *streaming-native* ensemble.

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
       emitted at an optional ``rf_first_checkpoint`` packet index (default
       2, set to 0 to disable) and then every ``rf_checkpoint_stride``
       packets (default 10). With the defaults a flow of length L gets
       checkpoints at packets {2, 10, 20, …} — i.e. one early one for
       short-flow coverage (PortScan) plus the regular cadence after.
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
  flow, emitting ``[mean, std, min, max]`` per base feature over the growing
  prefix.
- Output tensors are pre-allocated as memory-mapped ``.npy`` files, so peak
  RAM stays bounded even when a source's ``flow_packets.npy`` exceeds
  available memory.
"""

import json
import os
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple

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

# Aggregations supported in the emitted RF checkpoint vector, in emit order.
_SUPPORTED_AGGS = ("mean", "std", "min", "max")

# Output kinds that can be selectively built. "tcn" governs the per-packet
# ``flow_packets.npy``/``flow_offsets.npy`` pair (Transformer + TCN input);
# "rf" governs ``rf_checkpoints.npy``/``rf_checkpoint_offsets.npy``.
_SUPPORTED_OUTPUTS = ("rf", "tcn")


def count_rf_checkpoints(flow_length: int, first: int, stride: int) -> int:
    """Number of RF checkpoints a flow of ``flow_length`` packets produces.

    ``first`` is the absolute packet index of an extra early checkpoint
    (0 disables it). ``stride`` is the regular cadence after that. If
    ``first`` is a positive multiple of ``stride`` it coincides with the
    first regular checkpoint and contributes nothing extra.
    """
    n = flow_length // stride
    if first > 0 and (first % stride) != 0 and flow_length >= first:
        n += 1
    return n


def is_rf_checkpoint_packet(packet_index: int, first: int, stride: int) -> bool:
    """True if a checkpoint should be emitted at 1-based ``packet_index``."""
    if packet_index <= 0:
        return False
    if first > 0 and (first % stride) != 0 and packet_index == first:
        return True
    return packet_index % stride == 0


def checkpoint_packet_indices(
    n_checkpoints: int, first: int, stride: int,
) -> np.ndarray:
    """Absolute 1-based packet indices for the first ``n_checkpoints``.

    Under the default cadence (first=2, stride=10) the sequence is
    ``[2, 10, 20, 30, …]``. When ``first`` is 0 or coincides with
    ``stride`` (mod-zero), the sequence is the regular ``[stride, 2*stride,
    3*stride, …]``.
    """
    if n_checkpoints <= 0:
        return np.empty(0, dtype=np.int64)
    out = np.empty(n_checkpoints, dtype=np.int64)
    has_early = first > 0 and (first % stride) != 0
    if has_early:
        out[0] = first
        for k in range(1, n_checkpoints):
            out[k] = k * stride
    else:
        for k in range(n_checkpoints):
            out[k] = (k + 1) * stride
    return out


def _normalize_outputs(outputs: Optional[Iterable[str]]) -> Tuple[str, ...]:
    """Validate and canonicalise the ``outputs`` selector."""
    if outputs is None:
        return tuple(_SUPPORTED_OUTPUTS)
    norm = tuple(o.lower() for o in outputs)
    bad = [o for o in norm if o not in _SUPPORTED_OUTPUTS]
    if bad:
        raise ValueError(
            f"Unknown outputs {bad}. Supported: {list(_SUPPORTED_OUTPUTS)}"
        )
    if not norm:
        raise ValueError(
            f"outputs must contain at least one of {list(_SUPPORTED_OUTPUTS)}"
        )
    # Preserve insertion order but de-dupe.
    seen: set = set()
    return tuple(o for o in norm if not (o in seen or seen.add(o)))


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

        Uses population std (``ddof=0``: ``std = sqrt(M2 / n)``). For
        single-packet prefixes std is 0.
        """
        if self.n == 0:
            return np.zeros(self._n_feat * len(agg_functions), dtype=np.float32)
        std = np.sqrt(self._M2 / self.n)  # ddof=0 (population std)
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
    rf_first_checkpoint: int = 2,
    rf_base_features: Optional[List[str]] = None,
    rf_agg_functions: Optional[List[str]] = None,
    standardize_columns: Optional[List[str]] = None,
    mean_std_path: str = "mean_std.json",
    outputs: Optional[Iterable[str]] = None,
) -> Dict:
    """
    Convert per-flow data into DEPT's ragged streaming tensors.

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
        first regular checkpoint covers packets ``[0:stride]``, the next
        ``[0:2*stride]``, etc.
    rf_first_checkpoint : int
        Optional extra checkpoint emitted once at this absolute (1-based)
        packet index, for short-flow coverage. Default 2 → flows of
        length ≥ 2 get a checkpoint at packet 2 in addition to the regular
        every-``stride`` cadence. Set to 0 to disable. Automatically
        deduplicated when it coincides with a multiple of ``stride``.
    rf_base_features : list[str]
        Metadata columns to aggregate for the RF input.
    rf_agg_functions : list[str]
        Aggregation function names in ``{mean, std, min, max}``.
    standardize_columns : list[str]
        Columns to z-score normalize (typically just IPAT).
    mean_std_path : str
        Path to load/save normalization statistics.
    outputs : iterable[str], optional
        Subset of ``{"rf", "tcn"}`` selecting which on-disk tensor groups
        to materialise. ``"tcn"`` controls ``flow_packets.npy`` +
        ``flow_offsets.npy``; ``"rf"`` controls ``rf_checkpoints.npy`` +
        ``rf_checkpoint_offsets.npy``. Defaults to both. The returned
        ``num_packets`` / ``num_checkpoints`` always reflect the data
        counts, but the matching ``*_path`` entries are ``None`` for
        groups that were not written.

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
        "rf_first_checkpoint"         : int   — 0 if no early checkpoint
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
    if rf_first_checkpoint < 0:
        raise ValueError(
            f"rf_first_checkpoint must be >=0, got {rf_first_checkpoint}"
        )

    unknown = [a for a in rf_agg_functions if a not in _SUPPORTED_AGGS]
    if unknown:
        raise ValueError(
            f"Unsupported RF aggregations {unknown}. Supported: {_SUPPORTED_AGGS}"
        )

    outputs_norm = _normalize_outputs(outputs)
    build_rf = "rf" in outputs_norm
    build_tcn = "tcn" in outputs_norm

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
    # Only materialise the views the selected outputs actually need.
    rf_meta_values = (
        meta_df[rf_meta_cols].values.astype(np.float32) if build_rf else None
    )
    if build_tcn:
        ipat_values = meta_df["IPAT"].values.astype(np.float64)
        direction_values = meta_df["Direction"].values.astype(np.float32)
        ipat_mean = stats.get("IPAT", {}).get("mean", 0.0)
        ipat_std = stats.get("IPAT", {}).get("std", 1.0)
    else:
        ipat_values = direction_values = None
        ipat_mean = ipat_std = None

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
        n_ckpt = count_rf_checkpoints(L, rf_first_checkpoint, rf_checkpoint_stride)
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
            "rf_first_checkpoint": rf_first_checkpoint,
            "outputs": list(outputs_norm),
        }

    os.makedirs(output_dir, exist_ok=True)
    flow_packets_path = os.path.join(output_dir, "flow_packets.npy")
    flow_offsets_path = os.path.join(output_dir, "flow_offsets.npy")
    rf_checkpoints_path = os.path.join(output_dir, "rf_checkpoints.npy")
    rf_checkpoint_offsets_path = os.path.join(
        output_dir, "rf_checkpoint_offsets.npy",
    )

    # Memory-mapped output: scattered per-flow writes are fine — each flow's
    # packets occupy a contiguous slice of flow_packets.npy. Only allocate
    # the views the selected outputs need.
    if build_tcn:
        pkt_mmap = np.lib.format.open_memmap(
            flow_packets_path, mode="w+", dtype=np.float32,
            shape=(total_packets, n_seq_features),
        )
    else:
        pkt_mmap = None

    if build_rf:
        # Some sources may have only sub-stride flows; still write a 0-row file
        # so downstream loaders can stat it consistently.
        ckpt_mmap = np.lib.format.open_memmap(
            rf_checkpoints_path, mode="w+", dtype=np.float32,
            shape=(max(total_checkpoints, 0), n_rf_stats),
        )
    else:
        ckpt_mmap = None

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

        if build_tcn:
            # Write the per-flow [L, 1090] block directly into the memmap.
            pkt_mmap[flow_start:flow_start + L, :1088] = (
                nprint_bits[indices].astype(np.float32)
            )
            pkt_mmap[flow_start:flow_start + L, 1088] = (
                (ipat_values[indices] - ipat_mean) / ipat_std
            ).astype(np.float32)
            pkt_mmap[flow_start:flow_start + L, 1089] = direction_values[indices]

        if build_rf:
            # Streaming aggregation over the growing prefix → multi-scale RF rows.
            rf_meta_flow = rf_meta_values[indices]
            stat = StreamingStats(n_base)
            ckpt_local = 0
            for t in range(L):
                stat.update(rf_meta_flow[t])
                if is_rf_checkpoint_packet(
                    t + 1, rf_first_checkpoint, rf_checkpoint_stride
                ):
                    if ckpt_local < int(rf_checkpoint_offsets[i, 1]):
                        ckpt_mmap[ckpt_start + ckpt_local] = stat.emit(rf_agg_functions)
                    ckpt_local += 1

        if _tqdm is not None and i % 1000 == 0:
            pbar.set_postfix(
                flows=i + 1, packets=int(flow_packet_offsets[i, 0]) + L,
            )

    if _tqdm is not None:
        pbar.close()

    # Persist offsets and flush memmaps — only for the groups we built.
    if build_tcn:
        np.save(flow_offsets_path, flow_packet_offsets)
        pkt_mmap.flush()
    if build_rf:
        np.save(rf_checkpoint_offsets_path, rf_checkpoint_offsets)
        ckpt_mmap.flush()

    print("[+] Built ragged tensors:")
    if build_tcn:
        print(f"    flow_packets    : {pkt_mmap.shape}   "
              f"(P, F={n_seq_features})")
        print(f"    flow_offsets    : {flow_packet_offsets.shape}  "
              f"(N_flows, 2)")
    if build_rf:
        cadence = (
            f"stride={rf_checkpoint_stride}, first={rf_first_checkpoint}"
            if rf_first_checkpoint > 0
            else f"stride={rf_checkpoint_stride}"
        )
        print(f"    rf_checkpoints  : {ckpt_mmap.shape}  "
              f"(C, S={n_rf_stats})  {cadence}")
        print(f"    rf_ckpt_offsets : {rf_checkpoint_offsets.shape}  "
              f"(N_flows, 2)")

    return {
        "flow_packets_path": flow_packets_path if build_tcn else None,
        "flow_offsets_path": flow_offsets_path if build_tcn else None,
        "rf_checkpoints_path": rf_checkpoints_path if build_rf else None,
        "rf_checkpoint_offsets_path": (
            rf_checkpoint_offsets_path if build_rf else None
        ),
        "flow_ids": flow_keys,
        "stats": stats,
        "feature_names": sequence_features,
        "rf_stat_names": rf_stat_names,
        "num_flows": n_flows,
        "num_packets": total_packets,
        "num_checkpoints": total_checkpoints,
        "rf_checkpoint_stride": rf_checkpoint_stride,
        "rf_first_checkpoint": rf_first_checkpoint,
        "outputs": list(outputs_norm),
    }
