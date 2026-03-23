"""
Build the dual-output tensors for ADEPT's ensemble.

For each N-packet window in a flow, produce:
    1. Sequence tensor  [N, F]  — raw per-packet features for TCN & Transformer
    2. Statistics vector [S]    — aggregated stats for Random Forest

Both outputs are synchronized at the Nth packet boundary ("N-Packet Snapshot").

Standardization follows the same pattern as TANTGen's generate_sequences.py:
z-score normalization with mean/std saved to (or loaded from) a JSON file.
"""

import json
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Windowing (adapted from TANTGen utils.make_windows_features)
# ──────────────────────────────────────────────────────────────────────────────

def _make_windows(array: np.ndarray, window_size: int, stride: int):
    """Yield non-overlapping (or strided) windows from a 2-D array.

    Incomplete trailing windows are dropped (unlike TANTGen which pads them).
    For ADEPT we want every window to have exactly N packets — padding would
    pollute the RF statistics with artificial zeros.
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
        mean = float(df[col].mean())
        std = float(df[col].std())
        # Guard against zero std (constant column)
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
    window: np.ndarray,
    feature_names: List[str],
    base_features: List[str],
    agg_functions: List[str],
) -> np.ndarray:
    """
    Compute aggregated statistics over one N-packet window.

    Parameters
    ----------
    window : np.ndarray, shape [N, F]
        Raw (standardized) sequence window.
    feature_names : list[str]
        Column names matching the F dimension.
    base_features : list[str]
        Subset of feature_names to aggregate for RF.
    agg_functions : list[str]
        Names of aggregation functions ("mean", "std", "min", "max").

    Returns
    -------
    np.ndarray, shape [len(base_features) * len(agg_functions)]
    """
    col_idx = {name: i for i, name in enumerate(feature_names)}
    stats = []
    for feat in base_features:
        if feat not in col_idx:
            continue
        col = window[:, col_idx[feat]]
        for agg_name in agg_functions:
            stats.append(_AGG_FUNCS[agg_name](col))
    return np.array(stats, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def build_tensors(
    flows: OrderedDict,
    window_size: int = 10,
    stride: int = 10,
    sequence_features: Optional[List[str]] = None,
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
    sequence_features : list[str]
        Columns to include in the TCN/Transformer sequence tensor.
    rf_base_features : list[str]
        Columns to aggregate for the RF input.
    rf_agg_functions : list[str]
        Aggregation function names.
    standardize_columns : list[str]
        Columns to z-score normalize.
    mean_std_path : str
        Path to load/save normalization statistics.

    Returns
    -------
    dict with keys:
        "sequence"   : np.ndarray [W, N, F]  — TCN / Transformer input
        "rf_stats"   : np.ndarray [W, S]     — Random Forest input
        "labels"     : np.ndarray [W]        — per-window label (if available)
        "flow_ids"   : list[tuple]           — flow key for each window
        "stats"      : dict                  — normalization statistics
        "feature_names" : list[str]          — sequence feature column order
        "rf_stat_names" : list[str]          — RF stat column names
    """
    # Defaults
    if sequence_features is None:
        sequence_features = [
            "IPAT", "Packet_Size", "Protocol", "TCP_Flags",
            "Payload_Length", "Payload_Entropy", "Direction",
        ]
    if rf_base_features is None:
        rf_base_features = [
            "IPAT", "Packet_Size", "Payload_Length",
            "Payload_Entropy", "TCP_Flags", "Direction",
        ]
    if rf_agg_functions is None:
        rf_agg_functions = ["mean", "std", "min", "max"]
    if standardize_columns is None:
        standardize_columns = ["IPAT", "Packet_Size", "Payload_Length", "Payload_Entropy"]

    # --- Step 1: Concatenate all flows to compute global statistics ----------
    all_packets = pd.concat(flows.values(), ignore_index=True)
    stats = _load_or_compute_stats(all_packets, standardize_columns, mean_std_path)

    # --- Step 2: Per-flow windowing -------------------------------------------
    seq_windows: List[np.ndarray] = []
    rf_windows: List[np.ndarray] = []
    window_flow_ids: List[Tuple] = []
    window_labels: List[int] = []

    rf_stat_names = [
        f"{feat}_{agg}" for feat in rf_base_features
        if feat in [c for c in sequence_features]
        for agg in rf_agg_functions
    ]
    # Recompute properly — RF features might not all be in sequence_features
    rf_stat_names = []
    for feat in rf_base_features:
        for agg in rf_agg_functions:
            rf_stat_names.append(f"{feat}_{agg}")

    for flow_key, flow_df in flows.items():
        if len(flow_df) < window_size:
            continue  # skip flows shorter than one window

        # Standardize
        flow_std = _apply_standardization(flow_df, stats)

        # Select sequence columns
        available = [c for c in sequence_features if c in flow_std.columns]
        seq_array = flow_std[available].values.astype(np.float32)

        # Window the sequence
        for window in _make_windows(seq_array, window_size, stride):
            seq_windows.append(window)
            rf_stat = _compute_rf_stats(window, available, rf_base_features, rf_agg_functions)
            rf_windows.append(rf_stat)
            window_flow_ids.append(flow_key)

            # Label: if the flow_df has a "Label" column, take majority vote
            # (optional — may not exist in raw PCAPs)
            if "Label" in flow_df.columns:
                start = len(seq_windows) - 1
                # Not meaningful here; labels handled by the pipeline script
                window_labels.append(-1)

    # --- Step 3: Stack into arrays -------------------------------------------
    if not seq_windows:
        print("[!] No windows produced (all flows shorter than window_size)")
        return {
            "sequence": np.empty((0, window_size, len(sequence_features)), dtype=np.float32),
            "rf_stats": np.empty((0, len(rf_stat_names)), dtype=np.float32),
            "labels": np.empty(0, dtype=np.int64),
            "flow_ids": [],
            "stats": stats,
            "feature_names": sequence_features,
            "rf_stat_names": rf_stat_names,
        }

    sequence_tensor = np.stack(seq_windows)     # [W, N, F]
    rf_tensor = np.stack(rf_windows)            # [W, S]

    print(f"[+] Built {len(seq_windows)} windows:")
    print(f"    Sequence tensor : {sequence_tensor.shape}  (W, N={window_size}, F={sequence_tensor.shape[2]})")
    print(f"    RF stats tensor : {rf_tensor.shape}  (W, S={rf_tensor.shape[1]})")

    return {
        "sequence": sequence_tensor,
        "rf_stats": rf_tensor,
        "labels": np.array(window_labels, dtype=np.int64) if window_labels else np.empty(0, dtype=np.int64),
        "flow_ids": window_flow_ids,
        "stats": stats,
        "feature_names": sequence_features,
        "rf_stat_names": rf_stat_names,
    }
