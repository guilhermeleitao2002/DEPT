"""
Build the dual-output tensors for ADEPT's ensemble.

For each N-packet window in a flow, produce:
    1. Sequence tensor  [N, F]  — nPrint bit-level features + IPAT + Direction
                                  for TCN & Purification Transformer
    2. Statistics vector [S]    — aggregated stats for Random Forest

Both outputs are synchronized at the Nth packet boundary ("N-Packet Snapshot").

The sequence tensor uses the full nPrint bit columns (1088) plus IPAT and
Direction, giving F ≈ 1090. The RF statistics are computed from the
Scapy-derived metadata columns (Packet_Size, Payload_Length, etc.) to
preserve ensemble diversity — aggregating binary bits would be meaningless.

Only IPAT is z-score standardized; nPrint bits are already bounded {-1, 0, 1}
and Direction is binary.

Memory strategy
---------------
Large pcaps (2–4 M packets) produce per-flow float32 arrays that exceed
available RAM when accumulated in Python lists before stacking. To avoid
this, build_tensors:

    1. Pre-counts the total number of windows so output arrays can be
       pre-allocated exactly once.
    2. Writes windows directly into memory-mapped .npy files on disk,
       so the output never fully resides in RAM.
    3. Converts nPrint bits to float32 *per window* (10 packets at a time)
       rather than per flow, capping the working-set overhead at ~43 KB/window.
"""

import json
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .pcap_parser import NPRINT_COLUMNS


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
# Aggregate statistics for Random Forest
# ──────────────────────────────────────────────────────────────────────────────

_AGG_FUNCS = {
    "mean": np.mean,
    "std": np.std,
    "min": np.min,
    "max": np.max,
}


def _compute_rf_stats(
    window_meta: np.ndarray,
    meta_feature_names: List[str],
    base_features: List[str],
    agg_functions: List[str],
) -> np.ndarray:
    """
    Compute aggregated statistics over one N-packet window.

    Uses the Scapy-derived metadata columns (not nPrint bits).
    """
    col_idx = {name: i for i, name in enumerate(meta_feature_names)}
    stats = []
    for feat in base_features:
        if feat not in col_idx:
            continue
        col = window_meta[:, col_idx[feat]]
        for agg_name in agg_functions:
            stats.append(_AGG_FUNCS[agg_name](col))
    return np.array(stats, dtype=np.float32)


def _count_windows(
    flow_indices: OrderedDict,
    window_size: int,
    stride: int,
) -> int:
    """Count total windows across all qualifying flows."""
    total = 0
    for indices in flow_indices.values():
        n = len(indices)
        if n >= window_size:
            total += (n - window_size) // stride + 1
    return total


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def build_tensors(
    nprint_bits: np.ndarray,
    meta_df: pd.DataFrame,
    flow_indices: OrderedDict,
    output_dir: str,
    window_size: int = 10,
    stride: int = 10,
    rf_base_features: Optional[List[str]] = None,
    rf_agg_functions: Optional[List[str]] = None,
    standardize_columns: Optional[List[str]] = None,
    mean_std_path: str = "mean_std.json",
) -> Dict:
    """
    Main entry-point: convert per-flow data into ADEPT tensors.

    Writes sequence.npy and rf_stats.npy as memory-mapped files directly
    to output_dir, avoiding the need to hold all windows in RAM at once.

    Parameters
    ----------
    nprint_bits : np.ndarray, shape (P, 1088), dtype int8
        Compact nPrint bit matrix (sorted same as meta_df).
    meta_df : pd.DataFrame
        Metadata with IPAT and Direction columns (sorted by Timestamp).
    flow_indices : OrderedDict[tuple, np.ndarray]
        Flow key → array of row indices into nprint_bits / meta_df.
    output_dir : str
        Directory to write sequence.npy and rf_stats.npy.
    window_size : int
        N in the "N-Packet Snapshot".
    stride : int
        Window stride (typically == window_size for non-overlapping).
    rf_base_features : list[str]
        Metadata columns to aggregate for the RF input.
    rf_agg_functions : list[str]
        Aggregation function names.
    standardize_columns : list[str]
        Columns to z-score normalize (typically just IPAT).
    mean_std_path : str
        Path to load/save normalization statistics.

    Returns
    -------
    dict with keys:
        "sequence_path"  : str            — path to sequence.npy
        "rf_stats_path"  : str            — path to rf_stats.npy
        "flow_ids"       : list[tuple]    — flow key for each window
        "stats"          : dict           — normalization statistics
        "feature_names"  : list[str]      — sequence feature column order
        "rf_stat_names"  : list[str]      — RF stat column names
        "num_windows"    : int
    """
    # Defaults
    if rf_base_features is None:
        rf_base_features = [
            "IPAT", "Packet_Size", "Payload_Length",
            "Payload_Entropy", "TCP_Flags", "Direction",
        ]
    if rf_agg_functions is None:
        rf_agg_functions = ["mean", "std", "min", "max"]
    if standardize_columns is None:
        standardize_columns = ["IPAT"]

    # --- Step 1: Global statistics for standardization -----------------------
    stats = _load_or_compute_stats(meta_df, standardize_columns, mean_std_path)

    # Sequence features: all nPrint columns + IPAT + Direction
    sequence_features = list(NPRINT_COLUMNS) + ["IPAT", "Direction"]
    n_seq_features = len(sequence_features)   # 1090

    # RF metadata columns
    rf_meta_cols = [c for c in rf_base_features if c in meta_df.columns]
    rf_stat_names = [
        f"{feat}_{agg}"
        for feat in rf_base_features
        for agg in rf_agg_functions
    ]
    n_rf_stats = len(rf_stat_names)

    # Pre-extract metadata as numpy arrays for fast indexing
    rf_meta_values = meta_df[rf_meta_cols].values.astype(np.float32)
    ipat_values = meta_df["IPAT"].values.astype(np.float64)
    direction_values = meta_df["Direction"].values.astype(np.float32)
    ipat_mean = stats.get("IPAT", {}).get("mean", 0.0)
    ipat_std = stats.get("IPAT", {}).get("std", 1.0)

    # --- Step 2: Pre-count windows and open memmaps --------------------------
    n_windows = _count_windows(flow_indices, window_size, stride)

    if n_windows == 0:
        print("[!] No windows produced (all flows shorter than window_size)")
        return {
            "sequence_path": None,
            "rf_stats_path": None,
            "flow_ids": [],
            "stats": stats,
            "feature_names": sequence_features,
            "rf_stat_names": rf_stat_names,
            "num_windows": 0,
        }

    os.makedirs(output_dir, exist_ok=True)
    seq_path = os.path.join(output_dir, "sequence.npy")
    rf_path = os.path.join(output_dir, "rf_stats.npy")

    # Memory-mapped output: windows are written directly to disk, not
    # accumulated in RAM. Peak working set = one window (~43 KB at N=10).
    seq_mmap = np.lib.format.open_memmap(
        seq_path, mode="w+", dtype=np.float32,
        shape=(n_windows, window_size, n_seq_features),
    )
    rf_mmap = np.lib.format.open_memmap(
        rf_path, mode="w+", dtype=np.float32,
        shape=(n_windows, n_rf_stats),
    )

    # --- Step 3: Per-flow, per-window fill -----------------------------------
    window_flow_ids: List[Tuple] = []
    w = 0

    for flow_key, indices in flow_indices.items():
        n_pkts = len(indices)
        if n_pkts < window_size:
            continue

        for start in range(0, n_pkts - window_size + 1, stride):
            end = start + window_size
            win_idx = indices[start:end]

            # nPrint bits: convert only this window to float32 (~43 KB at N=10)
            seq_mmap[w, :, :1088] = nprint_bits[win_idx].astype(np.float32)
            seq_mmap[w, :, 1088] = (
                (ipat_values[win_idx] - ipat_mean) / ipat_std
            ).astype(np.float32)
            seq_mmap[w, :, 1089] = direction_values[win_idx]

            rf_mmap[w] = _compute_rf_stats(
                rf_meta_values[win_idx], rf_meta_cols,
                rf_base_features, rf_agg_functions,
            )
            window_flow_ids.append(flow_key)
            w += 1

    # Flush memmaps to disk
    seq_mmap.flush()
    rf_mmap.flush()

    print(f"[+] Built {n_windows} windows:")
    print(f"    Sequence tensor : {seq_mmap.shape}  "
          f"(W, N={window_size}, F={n_seq_features})")
    print(f"    RF stats tensor : {rf_mmap.shape}  "
          f"(W, S={n_rf_stats})")

    return {
        "sequence_path": seq_path,
        "rf_stats_path": rf_path,
        "flow_ids": window_flow_ids,
        "stats": stats,
        "feature_names": sequence_features,
        "rf_stat_names": rf_stat_names,
        "num_windows": n_windows,
    }
