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
"""

import json
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .packet_parser import NPRINT_COLUMNS


# ──────────────────────────────────────────────────────────────────────────────
# Windowing (adapted from TANTGen utils.make_windows_features)
# ──────────────────────────────────────────────────────────────────────────────

def _make_windows(array: np.ndarray, window_size: int, stride: int):
    """Yield non-overlapping (or strided) windows from a 2-D array.

    Incomplete trailing windows are dropped — every window must have
    exactly N packets to avoid polluting RF statistics with padding.
    """
    n = len(array)
    for start in range(0, n - window_size + 1, stride):
        yield array[start : start + window_size]


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


def _apply_standardization(
    df: pd.DataFrame,
    stats: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Apply z-score normalization in-place using pre-computed stats."""
    df = df.copy()
    for col, s in stats.items():
        if col in df.columns:
            df[col] = (df[col] - s["mean"]) / s["std"]
    return df


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


# ──────────────────────────────────────────────────────────────────────────────
# Sequence feature detection
# ──────────────────────────────────────────────────────────────────────────────

def _get_sequence_columns(df: pd.DataFrame) -> List[str]:
    """
    Auto-detect the columns for the sequence tensor:
    all nPrint bit columns present in the DataFrame + IPAT + Direction.
    """
    nprint_cols = [c for c in NPRINT_COLUMNS if c in df.columns]
    extra = []
    if "IPAT" in df.columns:
        extra.append("IPAT")
    if "Direction" in df.columns:
        extra.append("Direction")
    return nprint_cols + extra


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def build_tensors(
    flows: OrderedDict,
    window_size: int = 10,
    stride: int = 10,
    rf_base_features: Optional[List[str]] = None,
    rf_agg_functions: Optional[List[str]] = None,
    standardize_columns: Optional[List[str]] = None,
    mean_std_path: str = "mean_std.json",
) -> Dict:
    """
    Main entry-point: convert per-flow DataFrames into ADEPT tensors.

    Parameters
    ----------
    flows : OrderedDict[tuple, pd.DataFrame]
        Output of flow_grouper.group_by_flow().
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
        "sequence"       : np.ndarray [W, N, F]  — TCN / Transformer input
        "rf_stats"       : np.ndarray [W, S]     — Random Forest input
        "flow_ids"       : list[tuple]           — flow key for each window
        "stats"          : dict                  — normalization statistics
        "feature_names"  : list[str]             — sequence feature column order
        "rf_stat_names"  : list[str]             — RF stat column names
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

    # --- Step 1: Compute global statistics for standardization ----------------
    all_packets = pd.concat(flows.values(), ignore_index=True)
    stats = _load_or_compute_stats(all_packets, standardize_columns, mean_std_path)

    # Detect sequence columns from actual DataFrame
    sequence_features = _get_sequence_columns(all_packets)

    # RF metadata columns needed for aggregation
    rf_meta_cols = [c for c in rf_base_features if c in all_packets.columns]

    # Build RF stat names
    rf_stat_names = []
    for feat in rf_base_features:
        for agg in rf_agg_functions:
            rf_stat_names.append(f"{feat}_{agg}")

    # --- Step 2: Per-flow windowing -------------------------------------------
    seq_windows: List[np.ndarray] = []
    rf_windows: List[np.ndarray] = []
    window_flow_ids: List[Tuple] = []

    for flow_key, flow_df in flows.items():
        if len(flow_df) < window_size:
            continue

        # Standardize (only IPAT typically)
        flow_std = _apply_standardization(flow_df, stats)

        # Extract sequence columns (nPrint bits + IPAT + Direction)
        available_seq = [c for c in sequence_features if c in flow_std.columns]
        seq_array = flow_std[available_seq].values.astype(np.float32)

        # Extract RF metadata columns
        available_rf = [c for c in rf_meta_cols if c in flow_std.columns]
        rf_array = flow_std[available_rf].values.astype(np.float32)

        # Window both arrays in sync
        seq_iter = list(_make_windows(seq_array, window_size, stride))
        rf_iter = list(_make_windows(rf_array, window_size, stride))

        for seq_win, rf_win in zip(seq_iter, rf_iter):
            seq_windows.append(seq_win)
            rf_stat = _compute_rf_stats(rf_win, available_rf, rf_base_features, rf_agg_functions)
            rf_windows.append(rf_stat)
            window_flow_ids.append(flow_key)

    # --- Step 3: Stack into arrays -------------------------------------------
    if not seq_windows:
        print("[!] No windows produced (all flows shorter than window_size)")
        return {
            "sequence": np.empty((0, window_size, len(sequence_features)), dtype=np.float32),
            "rf_stats": np.empty((0, len(rf_stat_names)), dtype=np.float32),
            "flow_ids": [],
            "stats": stats,
            "feature_names": sequence_features,
            "rf_stat_names": rf_stat_names,
        }

    sequence_tensor = np.stack(seq_windows)     # [W, N, F]
    rf_tensor = np.stack(rf_windows)            # [W, S]

    print(f"[+] Built {len(seq_windows)} windows:")
    print(f"    Sequence tensor : {sequence_tensor.shape}  "
          f"(W, N={window_size}, F={sequence_tensor.shape[2]})")
    print(f"    RF stats tensor : {rf_tensor.shape}  "
          f"(W, S={rf_tensor.shape[1]})")

    return {
        "sequence": sequence_tensor,
        "rf_stats": rf_tensor,
        "flow_ids": window_flow_ids,
        "stats": stats,
        "feature_names": sequence_features,
        "rf_stat_names": rf_stat_names,
    }
