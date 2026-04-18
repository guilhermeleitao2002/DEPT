#!/usr/bin/env python3
"""
ADEPT — Window Label Attachment
================================

Produces <tensor-dir>/window_labels.csv from ground-truth label files.

Two dataset-specific alignment problems:

  kitsune   Labels are per-packet in PCAP order; the pipeline reorders packets
            by timestamp in group_by_flow(). A lightweight PCAP re-read
            (5-tuple + timestamp only, no nPrint encoding — ~5-10x faster)
            reconstructs the same sort order and flow grouping.

            Labels are applied per-window: for each N-packet window (using
            the same window_size and stride as the pipeline, read from
            metadata.json), the window is labeled attack if ANY of its N
            packets carries an attack label. Per-flow majority vote is NOT
            used because Kitsune flows often span the benign→attack transition
            (e.g., ARP MitM puts all ARP traffic in one flow; Video Injection
            runs the camera stream throughout), so a flow-level vote assigns
            the wrong label to every window in those flows.

  cic       Labels are per-flow in CICFlowMeter TrafficLabelling CSVs, keyed by
            unidirectional 5-tuple (Flow ID column). The script canonicalises
            those 5-tuples using the same bidirectional key logic as
            flow_grouper._compute_canonical_keys, then joins with flow_ids.npy.
            No PCAP re-read is needed.

Output columns: window_idx, label (0/1, -1=unknown), label_str.

Usage
-----
    # Kitsune (PCAP required to reconstruct timestamp sort order):
    python scripts/packet_to_tensor/attach_labels.py \\
        --dataset kitsune \\
        --tensor-dir data/tensors/mirai/ \\
        --labels-csv data/Kitsune/csv/mirai_labels.csv \\
        --pcap data/Kitsune/pcap/Mirai_pcap.pcap

    # CIC-IDS2017 (flow-level join, no PCAP needed):
    python scripts/packet_to_tensor/attach_labels.py \\
        --dataset cic \\
        --tensor-dir data/tensors/ddos/ \\
        --labels-csv "data/CIC-IDS-2017/csv/TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
"""

import argparse
import json
import os
import sys
from collections import defaultdict, OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
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


def _read_kitsune_labels(labels_csv: Path) -> np.ndarray:
    """
    Read per-packet label values from a Kitsune labels CSV.

    Handles two formats found in the dataset:
      - Two-column with header: col 0 = row-index, col 1 = label  (e.g. SYN_DoS_labels.csv)
        File starts: `"","x"` then rows `"1",0  "2",0  ...`
      - One-column, no header: bare 0/1 values only               (e.g. mirai_labels.csv)
    """
    raw = pd.read_csv(labels_csv, header=None, low_memory=False)
    if raw.shape[1] == 1:
        # Single-column file: values are in column 0
        return raw.iloc[:, 0].to_numpy(dtype=np.int8)
    else:
        # Two-column file: row 0 is the header ("","x"), data starts at row 1
        return raw.iloc[1:, 1].to_numpy(dtype=np.int8)


def label_kitsune(tensor_dir: Path, labels_csv: Path, pcap_path: Path) -> pd.DataFrame:
    """
    Attach per-window labels for Kitsune attacks.

    Labels are per-packet in PCAP order. We reconstruct the exact flow
    grouping and windowing the pipeline used, then label each window based
    on whether ANY of its N packets is an attack packet (any-attack rule).

    Per-flow majority vote is intentionally NOT used: Kitsune flows often
    span the benign→attack boundary (e.g. ARP MitM aggregates all ARP into
    one flow; Video Injection streams throughout the capture), so majority
    vote assigns the wrong label to large swaths of windows.
    """
    # --- Read pipeline config from metadata.json ----------------------------
    meta_path = tensor_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {tensor_dir}. "
                                "Run the pipeline first.")
    with open(meta_path) as f:
        meta = json.load(f)
    window_size = meta["window_size"]
    stride = meta["stride"]

    # --- Load per-packet labels (PCAP order) --------------------------------
    labels_pcap = _read_kitsune_labels(labels_csv)

    # --- Lightweight PCAP re-read to reconstruct timestamp sort order -------
    meta_df = _read_pcap_headers(pcap_path)
    if len(meta_df) != len(labels_pcap):
        raise ValueError(
            f"Packet count mismatch: PCAP {len(meta_df):,} vs labels {len(labels_pcap):,}")

    # Reconstruct sort_order — identical to group_by_flow()'s argsort
    sort_order = meta_df["Timestamp"].values.argsort(kind="mergesort")
    meta_sorted = meta_df.iloc[sort_order].reset_index(drop=True)
    labels_sorted = labels_pcap[sort_order]   # labels aligned to timestamp-sorted positions

    # --- Reconstruct flow grouping (same logic as flow_grouper) -------------
    flow_keys = _compute_canonical_keys(meta_sorted)
    flow_indices: dict = OrderedDict()
    for i, fk in enumerate(flow_keys):
        if fk not in flow_indices:
            flow_indices[fk] = []
        flow_indices[fk].append(i)

    # --- Per-window labeling ------------------------------------------------
    # flow_ids.npy contains the flow key for each window in pipeline order.
    flow_ids = np.load(tensor_dir / "flow_ids.npy", allow_pickle=True)
    attack_name = tensor_dir.name

    # Build a map: flow_key → list of per-window labels (in window order)
    flow_window_labels: dict = {}
    for fk, idx_list in flow_indices.items():
        idx_arr = np.array(idx_list, dtype=np.int64)
        n_pkts = len(idx_arr)
        if n_pkts < window_size:
            continue
        win_labels = []
        for start in range(0, n_pkts - window_size + 1, stride):
            win_pkt_labels = labels_sorted[idx_arr[start:start + window_size]]
            # Any-attack rule: if at least one packet in the window is attack → attack
            is_attack = int(win_pkt_labels.any())
            win_labels.append(is_attack)
        flow_window_labels[fk] = win_labels

    # Align with flow_ids.npy (which was written in the same flow/window order)
    rows = []
    # Track how many windows we've consumed per flow key
    flow_window_cursor: dict = defaultdict(int)
    n_unknown = 0
    for w in range(len(flow_ids)):
        fk = tuple(flow_ids[w])
        if fk in flow_window_labels:
            cursor = flow_window_cursor[fk]
            win_labels_list = flow_window_labels[fk]
            if cursor < len(win_labels_list):
                lbl = win_labels_list[cursor]
                flow_window_cursor[fk] += 1
                label_str = attack_name if lbl == 1 else "benign"
                rows.append({"window_idx": w, "label": lbl, "label_str": label_str})
            else:
                # Cursor overrun — shouldn't happen if pipeline and re-read agree
                rows.append({"window_idx": w, "label": -1, "label_str": "unknown"})
                n_unknown += 1
        else:
            rows.append({"window_idx": w, "label": -1, "label_str": "unknown"})
            n_unknown += 1

    if n_unknown:
        pct = 100 * n_unknown / len(flow_ids)
        print(f"  Warning: {n_unknown:,} windows ({pct:.1f}%) unmatched — "
              "flow key absent from label source")
    return pd.DataFrame(rows)


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
    cic_df = pd.read_csv(labels_csv, skipinitialspace=True, encoding="latin-1", low_memory=False)
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
