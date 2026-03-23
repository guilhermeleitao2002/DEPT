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
"""

from collections import OrderedDict
from typing import Dict, Tuple

import pandas as pd


def _canonical_flow_key(row: pd.Series) -> Tuple:
    """
    Build a canonical (order-independent) 5-tuple from a packet row.
    Returns (ip_lo, ip_hi, port_lo, port_hi, protocol).
    """
    src_ip = str(row["Source_IP"])
    dst_ip = str(row["Destination_IP"])
    src_port = int(row["Source_Port"])
    dst_port = int(row["Destination_Port"])
    proto = int(row["Protocol"])

    if (src_ip, src_port) <= (dst_ip, dst_port):
        return (src_ip, dst_ip, src_port, dst_port, proto)
    else:
        return (dst_ip, src_ip, dst_port, src_port, proto)


def group_by_flow(df: pd.DataFrame) -> Dict[Tuple, pd.DataFrame]:
    """
    Group a packet DataFrame by 5-tuple flow.

    For each flow the following columns are added:
        - Flow_Key  : the canonical 5-tuple (as string for readability)
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

    # Assign canonical flow key to every packet
    flow_keys = df.apply(_canonical_flow_key, axis=1)

    # Track the "initiator" direction for each flow (first packet seen)
    initiator: Dict[Tuple, Tuple] = {}  # flow_key → (src_ip, src_port)

    ipats = []
    directions = []
    last_ts: Dict[Tuple, float] = {}

    for idx, row in df.iterrows():
        fk = flow_keys[idx]

        # --- IPAT ---
        if fk in last_ts:
            ipats.append(row["Timestamp"] - last_ts[fk])
        else:
            ipats.append(0.0)  # first packet in flow
        last_ts[fk] = row["Timestamp"]

        # --- Direction ---
        src_ip = str(row["Source_IP"])
        src_port = int(row["Source_Port"])
        if fk not in initiator:
            initiator[fk] = (src_ip, src_port)
            directions.append(1)
        else:
            init_ip, init_port = initiator[fk]
            directions.append(1 if (src_ip, src_port) == (init_ip, init_port) else 0)

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
