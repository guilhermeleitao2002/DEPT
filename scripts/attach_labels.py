#!/usr/bin/env python3
"""
ADEPT — Window Label Attachment
================================

Produces <tensor-dir>/window_labels.csv from ground-truth label files.

Two dataset-specific alignment problems:

  kitsune   Labels are per-packet in PCAP order; the pipeline reorders packets
            by timestamp in group_by_flow(). A lightweight PCAP re-read
            (5-tuple + timestamp only, no nPrint encoding — ~5-10x faster)
            reconstructs the same sort order. Labels are then applied to sorted
            positions and majority-voted per canonical flow, then joined with
            flow_ids.npy.

  cic       Labels are per-flow in CICFlowMeter TrafficLabelling CSVs, keyed by
            unidirectional 5-tuple (Flow ID column). The script canonicalises
            those 5-tuples using the same bidirectional key logic as
            flow_grouper._compute_canonical_keys, then joins with flow_ids.npy.
            No PCAP re-read is needed.

Output columns: window_idx, label (0/1, -1=unknown), label_str.

Usage
-----
    # Kitsune (PCAP required to reconstruct timestamp sort order):
    python scripts/attach_labels.py \\
        --dataset kitsune \\
        --tensor-dir data/tensors/mirai/ \\
        --labels-csv data/Kitsune/csv/mirai_labels.csv \\
        --pcap data/Kitsune/pcap/Mirai_pcap.pcap

    # CIC-IDS2017 (flow-level join, no PCAP needed):
    python scripts/attach_labels.py \\
        --dataset cic \\
        --tensor-dir data/tensors/ddos/ \\
        --labels-csv "data/CIC-IDS-2017/csv/TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.feature_extraction.flow_grouper import _compute_canonical_keys

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

OUTPUT_FILENAME = "window_labels.csv"


# ---------------------------------------------------------------------------
# Kitsune
# ---------------------------------------------------------------------------

def _read_pcap_headers(pcap_path: Path) -> pd.DataFrame:
    """Lightweight PCAP pass: 5-tuple + timestamp only. No bit encoding."""
    from scapy.all import IP, TCP, UDP, PcapReader
    records = []
    with PcapReader(str(pcap_path)) as reader:
        it = _tqdm(reader, desc="  PCAP headers", unit=" pkt", unit_scale=True,
                   dynamic_ncols=True) if _tqdm else reader
        for pkt in it:
            ts = float(pkt.time)
            if pkt.haslayer(IP):
                ip = pkt[IP]
                src_ip, dst_ip, proto = ip.src, ip.dst, int(ip.proto)
                if pkt.haslayer(TCP):
                    src_port, dst_port = int(pkt[TCP].sport), int(pkt[TCP].dport)
                elif pkt.haslayer(UDP):
                    src_port, dst_port = int(pkt[UDP].sport), int(pkt[UDP].dport)
                else:
                    src_port = dst_port = -1
            else:
                src_ip = dst_ip = "-1"
                src_port = dst_port = proto = -1
            records.append((ts, src_ip, dst_ip, src_port, dst_port, proto))
    return pd.DataFrame(records, columns=[
        "Timestamp", "Source_IP", "Destination_IP",
        "Source_Port", "Destination_Port", "Protocol",
    ])


def label_kitsune(tensor_dir: Path, labels_csv: Path, pcap_path: Path) -> pd.DataFrame:
    # Per-packet labels in PCAP order (CSV: index col + "x" col, values 0/1)
    labels_pcap = pd.read_csv(labels_csv, index_col=0).iloc[:, 0].to_numpy(dtype=np.int8)

    # Lightweight PCAP re-read to reconstruct the timestamp sort order
    meta_df = _read_pcap_headers(pcap_path)
    if len(meta_df) != len(labels_pcap):
        raise ValueError(
            f"Packet count mismatch: PCAP {len(meta_df):,} vs labels {len(labels_pcap):,}")

    # Reconstruct sort_order — identical to group_by_flow()'s argsort
    sort_order = meta_df["Timestamp"].values.argsort(kind="mergesort")
    meta_sorted = meta_df.iloc[sort_order].reset_index(drop=True)
    labels_sorted = labels_pcap[sort_order]  # labels now aligned to sorted packet positions

    # Canonical flow key per sorted packet (same logic as the pipeline)
    flow_keys = _compute_canonical_keys(meta_sorted)

    # Majority-vote label per canonical flow (all packets in one flow should agree)
    attack_name = tensor_dir.name
    votes: dict = defaultdict(list)
    for fk, lbl in zip(flow_keys, labels_sorted):
        votes[fk].append(int(lbl))
    flow_label = {
        fk: (int(sum(v) > len(v) / 2), attack_name if sum(v) > len(v) / 2 else "benign")
        for fk, v in votes.items()
    }

    return _join_flow_ids(tensor_dir, flow_label)


# ---------------------------------------------------------------------------
# CIC-IDS2017
# ---------------------------------------------------------------------------

def _canonicalize_cic_flow_id(flow_id: str) -> tuple:
    """
    Parse a CICFlowMeter Flow ID and canonicalize to the same bidirectional
    key used by flow_grouper._compute_canonical_keys.

    Flow ID format: "src_ip-dst_ip-src_port-dst_port-proto"
    """
    parts = flow_id.strip().split("-")
    if len(parts) != 5:
        raise ValueError(f"Unexpected Flow ID: {flow_id!r}")
    src_ip, dst_ip = parts[0], parts[1]
    src_port, dst_port, proto = int(parts[2]), int(parts[3]), int(parts[4])
    # Same swap condition as _compute_canonical_keys
    if src_ip + ":" + str(src_port) > dst_ip + ":" + str(dst_port):
        return (dst_ip, src_ip, dst_port, src_port, proto)
    return (src_ip, dst_ip, src_port, dst_port, proto)


def label_cic(tensor_dir: Path, labels_csv: Path) -> pd.DataFrame:
    cic_df = pd.read_csv(labels_csv, skipinitialspace=True)
    cic_df.columns = cic_df.columns.str.strip()
    flow_id_col = cic_df["Flow ID"].astype(str).str.strip()
    label_col = cic_df["Label"].astype(str).str.strip()

    def _safe_canon(fid):
        try:
            return _canonicalize_cic_flow_id(fid)
        except (ValueError, IndexError):
            return None

    canonical_keys = flow_id_col.apply(_safe_canon)
    valid = canonical_keys.notna()
    labels_binary = np.where(label_col.str.upper() == "BENIGN", 0, 1)

    # Last row wins if the same flow appears twice (CIC duplicates are rare)
    flow_label = {
        ck: (int(lb), ls)
        for ck, lb, ls in zip(canonical_keys[valid], labels_binary[valid], label_col[valid])
    }
    return _join_flow_ids(tensor_dir, flow_label)


# ---------------------------------------------------------------------------
# Shared: join flow_label dict with flow_ids.npy
# ---------------------------------------------------------------------------

def _join_flow_ids(tensor_dir: Path, flow_label: dict) -> pd.DataFrame:
    flow_ids = np.load(tensor_dir / "flow_ids.npy", allow_pickle=True)
    rows, n_unknown = [], 0
    for w in range(len(flow_ids)):
        fk = tuple(flow_ids[w])
        if fk in flow_label:
            label, label_str = flow_label[fk]
        else:
            label, label_str = -1, "unknown"
            n_unknown += 1
        rows.append({"window_idx": w, "label": label, "label_str": label_str})
    if n_unknown:
        pct = 100 * n_unknown / len(flow_ids)
        print(f"  Warning: {n_unknown:,} windows ({pct:.1f}%) unmatched — "
              "flow key absent from label source")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Attach per-window ground-truth labels to a tensor directory.")
    parser.add_argument("-d", "--dataset", required=True, choices=["kitsune", "cic"])
    parser.add_argument("-t", "--tensor-dir", required=True)
    parser.add_argument("-l", "--labels-csv", required=True)
    parser.add_argument("-i", "--pcap", default=None,
                        help="Source PCAP (required for --dataset kitsune)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing window_labels.csv")
    args = parser.parse_args()

    tensor_dir = Path(args.tensor_dir)
    labels_csv = Path(args.labels_csv)
    out_path = tensor_dir / OUTPUT_FILENAME

    if not (tensor_dir / "flow_ids.npy").exists():
        print(f"ERROR: flow_ids.npy not found in {tensor_dir}. Run the pipeline first.")
        sys.exit(1)
    if not labels_csv.exists():
        print(f"ERROR: labels CSV not found: {labels_csv}")
        sys.exit(1)
    if out_path.exists() and not args.force:
        print(f"Already labelled ({out_path}). Use --force to overwrite.")
        sys.exit(0)
    if args.dataset == "kitsune" and args.pcap is None:
        parser.error("--pcap is required for --dataset kitsune")

    if args.dataset == "kitsune":
        df = label_kitsune(tensor_dir, labels_csv, Path(args.pcap))
    else:
        df = label_cic(tensor_dir, labels_csv)

    df.to_csv(out_path, index=False)
    n = len(df)
    n_atk = (df["label"] == 1).sum()
    n_ben = (df["label"] == 0).sum()
    n_unk = (df["label"] == -1).sum()
    print(f"  Labelled {n:,} windows — attack: {n_atk:,}, benign: {n_ben:,}"
          + (f", unknown: {n_unk:,}" if n_unk else ""))
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
