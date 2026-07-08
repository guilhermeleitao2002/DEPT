"""
Group packets by 5-tuple flow and compute per-flow IPAT.

This is the key difference from TANTGen: TANTGen computes a global IAT
across all packets; DEPT needs *per-flow* Inter-Packet Arrival Time (IPAT)
because each flow window is processed independently by the ensemble.

A flow is identified by the canonical 5-tuple:
    (min(src,dst)_IP, max(src,dst)_IP, min_port, max_port, protocol)
so that forward and reverse packets belong to the same flow.

Direction is assigned relative to the first packet seen in each flow:
    1 = same direction as flow initiator (forward)
    0 = reverse direction

Performance note: uses vectorized operations on the 5-tuple columns only,
avoiding per-row iteration over the full (potentially ~1100-column) DataFrame.
"""

from collections import OrderedDict
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _compute_canonical_keys(df: pd.DataFrame) -> pd.Series:
    """
    Build canonical (order-independent) 5-tuple keys vectorized.

    Returns a Series of tuples: (ip_lo, ip_hi, port_lo, port_hi, protocol).
    """
    src_ip = df["Source_IP"].astype(str)
    dst_ip = df["Destination_IP"].astype(str)
    src_port = df["Source_Port"].astype(int)
    dst_port = df["Destination_Port"].astype(int)
    proto = df["Protocol"].astype(int)

    # Determine canonical ordering: (src, src_port) <= (dst, dst_port)
    src_key = src_ip + ":" + src_port.astype(str)
    dst_key = dst_ip + ":" + dst_port.astype(str)
    swap = src_key > dst_key

    ip_lo = src_ip.where(~swap, dst_ip)
    ip_hi = dst_ip.where(~swap, src_ip)
    port_lo = src_port.where(~swap, dst_port)
    port_hi = dst_port.where(~swap, src_port)

    # Build tuple keys
    keys = list(zip(ip_lo, ip_hi, port_lo, port_hi, proto))
    return pd.Series(keys, index=df.index)


def group_by_flow(
    meta_df: pd.DataFrame,
) -> tuple:
    """
    Group packets by 5-tuple flow using metadata only.

    Returns lightweight index arrays instead of per-flow DataFrames
    to avoid duplicating the full (potentially 1088-column) data.

    Parameters
    ----------
    meta_df : pd.DataFrame
        Metadata from pcap_parser (Timestamp, Source_IP, …, TCP_Flags).
        Must **not** contain nPrint bit columns (those stay in numpy to avoid using up too much memory).

    Returns
    -------
    (meta_df, flow_indices, sort_order) : tuple
        meta_df       : sorted by Timestamp, with IPAT and Direction added
        flow_indices  : OrderedDict[tuple, np.ndarray] — flow key → row
                        indices into the sorted arrays
        sort_order    : np.ndarray — permutation that was applied to sort
                        by Timestamp (caller must apply to nprint_bits too)
    """
    # Stable sort so packets with identical timestamps keep pcap order
    sort_order = meta_df["Timestamp"].values.argsort(kind="mergesort")
    df = meta_df.iloc[sort_order].reset_index(drop=True)

    # Compute canonical flow keys (vectorized on 5-tuple columns only)
    flow_keys = _compute_canonical_keys(df)

    # Compute IPAT and Direction per flow
    initiator: Dict[Tuple, Tuple] = {}  # flow_key → (src_ip, src_port)
    last_ts: Dict[Tuple, float] = {}

    ipats = np.zeros(len(df), dtype=np.float64)
    directions = np.zeros(len(df), dtype=np.int8)

    src_ips = df["Source_IP"].astype(str).values
    src_ports = df["Source_Port"].astype(int).values
    timestamps = df["Timestamp"].values

    for i in range(len(df)):
        fk = flow_keys.iloc[i]

        # IPAT
        if fk in last_ts:
            ipats[i] = timestamps[i] - last_ts[fk]
        # else: 0.0 (first packet in flow)
        last_ts[fk] = timestamps[i]

        # Direction
        if fk not in initiator:
            initiator[fk] = (src_ips[i], src_ports[i])
            directions[i] = 1
        else:
            init_ip, init_port = initiator[fk]
            directions[i] = 1 if (src_ips[i], src_ports[i]) == (init_ip, init_port) else 0

    df["IPAT"] = ipats
    df["Direction"] = directions

    # Build flow → index mapping (no per-flow DataFrame copies)
    flow_key_arr = flow_keys.values  # numpy array of tuples
    flow_indices = OrderedDict()
    # Use a dict of lists for fast appending, then convert to arrays
    _tmp: Dict[Tuple, list] = {}
    for i, fk in enumerate(flow_key_arr):
        if fk not in _tmp:
            _tmp[fk] = []
        _tmp[fk].append(i)
    for fk, idx_list in _tmp.items():
        flow_indices[fk] = np.array(idx_list, dtype=np.int64)

    print(f"[+] Grouped {len(df)} packets into {len(flow_indices)} flows")
    return df, flow_indices, sort_order
