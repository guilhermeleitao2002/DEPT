"""
Group packets by 5-tuple flow and compute per-flow IPAT.

This is the key difference from TANTGen: TANTGen computes a global IAT
across all packets; ADEPT needs *per-flow* Inter-Packet Arrival Time (IPAT)
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


def group_by_flow(df: pd.DataFrame) -> Dict[Tuple, pd.DataFrame]:
    """
    Group a packet DataFrame by 5-tuple flow.

    For each flow the following columns are added:
        - IPAT      : inter-packet arrival time *within the flow*
        - Direction : 1 if packet matches the first-seen (src→dst), else 0

    Parameters
    ----------
    df : pd.DataFrame
        Output of packet_parser.extract_packet_features(), must contain at
        least: Timestamp, Source_IP, Destination_IP, Source_Port,
        Destination_Port, Protocol.

    Returns
    -------
    flows : OrderedDict[tuple, pd.DataFrame]
        Mapping from canonical 5-tuple → DataFrame of packets (sorted by
        Timestamp), with IPAT and Direction columns added.
    """
    # Ensure packets are sorted globally by timestamp
    df = df.sort_values("Timestamp").reset_index(drop=True)

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

    df = df.copy()
    df["IPAT"] = ipats
    df["Direction"] = directions
    df["_flow_key"] = flow_keys

    # Split into per-flow DataFrames
    flows = OrderedDict()
    for fk, group in df.groupby("_flow_key", sort=False):
        flows[fk] = group.drop(columns=["_flow_key"]).reset_index(drop=True)

    print(f"[+] Grouped {len(df)} packets into {len(flows)} flows")
    return flows
