"""
Incremental (online) packet-to-tensor pipeline.

Eliminates the global timestamp sort required by the batch pipeline
(pcap_parser → flow_grouper → tensor_builder).

Key idea
--------
Per-flow IPAT and Direction are computed as each packet arrives:

    IPAT      = pkt.timestamp − flow.last_timestamp
    Direction = 1 if src matches flow initiator, else 0

A window is emitted the moment a flow's buffer fills to N packets —
no need to buffer the entire PCAP first.

Passes
------
Pass 1 — lightweight prescan (5-tuple + timestamp, no nPrint encoding):
    • Counts packets per canonical flow → total windows for memmap pre-allocation.
    • Computes IPAT mean/std via Welford's online algorithm (used for z-score
      normalization of the sequence tensor, same role as mean_std.json in batch).

Pass 2 — main encoding pass:
    • Full nPrint bit encoding per packet.
    • Per-flow state update: last_ts, buffer of encoded packets.
    • Window emitted (written directly to memory-mapped .npy) as soon as
      len(flow.buf) == window_size.

Memory profile
--------------
Peak RAM is O(window_size × active_flows × features_per_packet), compared to
O(total_packets × features_per_packet) for the batch pipeline.  For large
captures (e.g., Brute Force FTP, 6.5 M packets) this is the difference between
a few hundred MB and several GB — allowing datasets that OOM the batch pipeline
to complete on memory-constrained machines.
"""

import json
import os

import numpy as np
from scapy.all import IP, TCP, UDP, PcapReader

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

from .pcap_parser import (
    _encode_packet,
    NPRINT_COLUMNS,
    DEFAULT_MASK_PREFIXES,
    TOTAL_BITS,
)
from .tensor_builder import _compute_rf_stats


# ──────────────────────────────────────────────────────────────────────────────
# Per-flow mutable state
# ──────────────────────────────────────────────────────────────────────────────

class _FlowState:
    """
    Minimal per-flow bookkeeping for incremental processing.

    buf holds the last (window_size − 1) already-encoded packets so that the
    next incoming packet can complete a full window.  Each element is a dict:

        {
          "bits":            np.ndarray (1088,) int8   — nPrint bits
          "ipat":            float                     — raw seconds
          "direction":       int                       — 1=fwd, 0=rev
          "Packet_Size":     float
          "Payload_Length":  float
          "Payload_Entropy": float
          "TCP_Flags":       float
        }
    """
    __slots__ = ("init_ip", "init_port", "last_ts", "buf")

    def __init__(self, src_ip: str, src_port: int, timestamp: float):
        self.init_ip   = src_ip
        self.init_port = src_port
        self.last_ts   = timestamp
        self.buf: list = []


# ──────────────────────────────────────────────────────────────────────────────
# Canonical 5-tuple key  (must match flow_grouper._compute_canonical_keys)
# ──────────────────────────────────────────────────────────────────────────────

def _canonical_key(src_ip, dst_ip, src_port, dst_port, proto) -> tuple:
    """
    Order-independent 5-tuple flow key.

    Comparison is done on "ip:port" strings, identical to the vectorized
    logic in flow_grouper._compute_canonical_keys, so keys produced here
    will match keys in flow_ids.npy from the batch pipeline.
    """
    src_str = f"{src_ip}:{int(src_port)}"
    dst_str = f"{dst_ip}:{int(dst_port)}"
    if src_str <= dst_str:
        return (str(src_ip), str(dst_ip), int(src_port), int(dst_port), int(proto))
    else:
        return (str(dst_ip), str(src_ip), int(dst_port), int(src_port), int(proto))


# ──────────────────────────────────────────────────────────────────────────────
# Pass 1 — lightweight prescan
# ──────────────────────────────────────────────────────────────────────────────

def _prescan(pcap_path: str, window_size: int, stride: int) -> tuple:
    """
    Lightweight first pass: count total windows and compute IPAT statistics.

    Reads only the 5-tuple and timestamp per packet — no nPrint bit encoding.
    Mirrors the packet handling of the main pass so that window counts match
    exactly (non-IP packets are included as a single degenerate flow).

    Parameters
    ----------
    pcap_path : str
    window_size, stride : int

    Returns
    -------
    (n_windows, ipat_mean, ipat_std) : (int, float, float)
        n_windows  — total windows across all qualifying flows
        ipat_mean  — mean IPAT (seconds) computed via Welford's algorithm
        ipat_std   — std  IPAT (seconds) computed via Welford's algorithm
    """
    flow_counts  = {}   # flow_key → packet count
    flow_last_ts = {}   # flow_key → last timestamp

    # Welford's online mean/variance for IPAT (O(1) memory)
    welf_n    = 0
    welf_mean = 0.0
    welf_M2   = 0.0

    reader = PcapReader(pcap_path)
    pbar = (
        _tqdm(reader, desc="      Prescan", unit=" pkt",
              unit_scale=True, dynamic_ncols=True)
        if _tqdm is not None else reader
    )

    for pkt in pbar:
        ts = float(pkt.time)

        # Extract 5-tuple (same defaults as _encode_packet for non-IP packets)
        src_ip = dst_ip = "-1"
        src_port = dst_port = proto = -1

        if pkt.haslayer(IP):
            ip       = pkt[IP]
            src_ip   = ip.src
            dst_ip   = ip.dst
            proto    = ip.proto
            if pkt.haslayer(TCP):
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
            elif pkt.haslayer("UDP"):
                src_port = pkt["UDP"].sport
                dst_port = pkt["UDP"].dport

        fk = _canonical_key(src_ip, dst_ip, src_port, dst_port, proto)

        # Incremental IPAT → feed Welford's online algorithm.
        # Include ALL packets (first packet has ipat=0) so that the resulting
        # mean/std matches what the batch pipeline computes from meta_df["IPAT"]
        # (which also contains the per-flow first-packet zeros).
        ipat = (ts - flow_last_ts[fk]) if fk in flow_last_ts else 0.0

        # Welford's update (sample statistics, ddof=1, matching pandas .std())
        welf_n    += 1
        delta      = ipat - welf_mean
        welf_mean += delta / welf_n
        welf_M2   += delta * (ipat - welf_mean)

        flow_last_ts[fk] = ts
        flow_counts[fk]  = flow_counts.get(fk, 0) + 1

    if _tqdm is not None:
        pbar.close()
    reader.close()

    # Finalise Welford — use sample std (ddof=1) to match pandas .std()
    ipat_mean = welf_mean
    ipat_std  = (welf_M2 / (welf_n - 1)) ** 0.5 if welf_n > 1 else 1.0
    if ipat_std == 0.0:
        ipat_std = 1.0

    n_windows = sum(
        (n - window_size) // stride + 1
        for n in flow_counts.values()
        if n >= window_size
    )
    return n_windows, ipat_mean, ipat_std


# ──────────────────────────────────────────────────────────────────────────────
# Pass 2 — incremental main pass
# ──────────────────────────────────────────────────────────────────────────────

def build_tensors_incremental(
    pcap_path: str,
    output_dir: str,
    window_size: int = 10,
    stride: int = 10,
    rf_base_features=None,
    rf_agg_functions=None,
    mean_std_path: str = None,
    mask_prefixes=None,
) -> dict:
    """
    Two-pass incremental packet-to-tensor pipeline.

    Replaces the three-stage batch pipeline (pcap_parser → flow_grouper →
    tensor_builder) with a two-pass approach that eliminates the global
    timestamp sort and emits windows as soon as N packets arrive per flow.

    Pass 1 (prescan): counts windows + computes IPAT mean/std for z-score
                      normalization.  Skipped if mean_std_path already exists.
    Pass 2 (main):    nPrint encoding per packet, per-flow state update,
                      window emission directly into memory-mapped output files.

    Parameters
    ----------
    pcap_path : str
    output_dir : str
    window_size : int
        N packets per window.
    stride : int
        Advance stride (== window_size → non-overlapping).
    rf_base_features : list[str] or None
    rf_agg_functions : list[str] or None
    mean_std_path : str or None
        Path to load/save IPAT normalization stats.  Defaults to
        <output_dir>/mean_std.json.  If the file already exists its stats
        are used and the prescan still runs (to count windows).
    mask_prefixes : list[str] or None
        nPrint column prefixes to zero out after encoding.

    Returns
    -------
    dict with the same keys as tensor_builder.build_tensors:
        sequence_path, rf_stats_path, flow_ids, stats,
        feature_names, rf_stat_names, num_windows
    """
    if rf_base_features is None:
        rf_base_features = [
            "IPAT", "Packet_Size", "Payload_Length",
            "Payload_Entropy", "TCP_Flags", "Direction",
        ]
    if rf_agg_functions is None:
        rf_agg_functions = ["mean", "std", "min", "max"]
    if mask_prefixes is None:
        mask_prefixes = DEFAULT_MASK_PREFIXES
    if mean_std_path is None:
        mean_std_path = os.path.join(output_dir, "mean_std.json")

    sequence_features = list(NPRINT_COLUMNS) + ["IPAT", "Direction"]
    n_seq_features    = len(sequence_features)   # 1090
    rf_stat_names     = [
        f"{feat}_{agg}"
        for feat in rf_base_features
        for agg in rf_agg_functions
    ]
    n_rf_stats = len(rf_stat_names)

    # ── Pass 1: prescan ────────────────────────────────────────────────────
    print("      Prescan: counting windows + computing IPAT statistics")
    n_windows, ipat_mean_computed, ipat_std_computed = _prescan(
        pcap_path, window_size, stride
    )

    # Load or save normalisation stats
    if os.path.exists(mean_std_path):
        print(f"[i] Loading mean/std from {mean_std_path}")
        with open(mean_std_path) as f:
            stats = json.load(f)
        ipat_mean = stats.get("IPAT", {}).get("mean", 0.0)
        ipat_std  = stats.get("IPAT", {}).get("std",  1.0)
    else:
        ipat_mean = ipat_mean_computed
        ipat_std  = ipat_std_computed
        stats = {"IPAT": {"mean": ipat_mean, "std": ipat_std}}
        os.makedirs(output_dir, exist_ok=True)
        with open(mean_std_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[+] Saved mean/std to {mean_std_path}")

    if n_windows == 0:
        print("[!] No windows produced (all flows shorter than window_size)")
        return {
            "sequence_path": None, "rf_stats_path": None,
            "flow_ids": [], "stats": stats,
            "feature_names": sequence_features,
            "rf_stat_names": rf_stat_names,
            "num_windows": 0,
        }

    # ── Allocate memory-mapped output files ───────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    seq_path = os.path.join(output_dir, "sequence.npy")
    rf_path  = os.path.join(output_dir, "rf_stats.npy")

    seq_mmap = np.lib.format.open_memmap(
        seq_path, mode="w+", dtype=np.float32,
        shape=(n_windows, window_size, n_seq_features),
    )
    rf_mmap = np.lib.format.open_memmap(
        rf_path, mode="w+", dtype=np.float32,
        shape=(n_windows, n_rf_stats),
    )

    # ── Pre-compute masking column indices (done once, not per packet) ─────
    cols_to_mask = []
    if mask_prefixes:
        for i, col in enumerate(NPRINT_COLUMNS):
            for prefix in mask_prefixes:
                if col.startswith(prefix + "_") and col[len(prefix) + 1:].isdigit():
                    cols_to_mask.append(i)
                    break

    # ── RF column name → index mapping (for _compute_rf_stats) ────────────
    rf_col_names = [
        "IPAT", "Packet_Size", "Payload_Length",
        "Payload_Entropy", "TCP_Flags", "Direction",
    ]

    # ── Pass 2: incremental encoding + window emission ─────────────────────
    flow_states: dict = {}    # flow_key → _FlowState
    window_flow_ids:  list = []
    w = 0                     # window counter

    reader = PcapReader(pcap_path)
    pbar = (
        _tqdm(reader, desc="      Encoding packets", unit=" pkt",
              unit_scale=True, dynamic_ncols=True)
        if _tqdm is not None else reader
    )

    for pkt in pbar:
        # ── nPrint encoding + metadata extraction ─────────────────────────
        bits_list, meta = _encode_packet(pkt)
        bits = np.array(bits_list, dtype=np.int8)
        if cols_to_mask:
            bits[cols_to_mask] = 0

        # ── Canonical flow key ────────────────────────────────────────────
        fk = _canonical_key(
            meta["Source_IP"],       meta["Destination_IP"],
            meta["Source_Port"],     meta["Destination_Port"],
            meta["Protocol"],
        )
        ts = meta["Timestamp"]

        # ── Incremental IPAT + Direction ──────────────────────────────────
        if fk not in flow_states:
            # First packet for this flow: IPAT = 0, always forward
            flow_states[fk] = _FlowState(meta["Source_IP"], meta["Source_Port"], ts)
            ipat      = 0.0
            direction = 1
        else:
            state     = flow_states[fk]
            ipat      = ts - state.last_ts   # per-flow, not global
            state.last_ts = ts
            direction = (
                1 if (meta["Source_IP"], meta["Source_Port"])
                     == (state.init_ip, state.init_port)
                else 0
            )

        # ── Append to flow buffer ─────────────────────────────────────────
        flow_states[fk].buf.append({
            "bits":            bits,
            "ipat":            ipat,
            "direction":       direction,
            "Packet_Size":     meta["Packet_Size"],
            "Payload_Length":  meta["Payload_Length"],
            "Payload_Entropy": meta["Payload_Entropy"],
            "TCP_Flags":       meta["TCP_Flags"],
        })

        # ── Emit window when buffer is full ───────────────────────────────
        buf = flow_states[fk].buf
        if len(buf) >= window_size:
            win = buf[-window_size:]   # last N packets form the window

            # Sequence tensor: nPrint bits + normalised IPAT + Direction
            for i, p in enumerate(win):
                seq_mmap[w, i, :1088] = p["bits"].astype(np.float32)
                seq_mmap[w, i, 1088]  = (p["ipat"] - ipat_mean) / ipat_std
                seq_mmap[w, i, 1089]  = float(p["direction"])

            # RF statistics: aggregated metadata (raw, un-normalised IPAT)
            rf_meta = np.array(
                [[p["ipat"], p["Packet_Size"], p["Payload_Length"],
                  p["Payload_Entropy"], p["TCP_Flags"], float(p["direction"])]
                 for p in win],
                dtype=np.float32,
            )
            rf_mmap[w] = _compute_rf_stats(
                rf_meta, rf_col_names, rf_base_features, rf_agg_functions,
            )

            window_flow_ids.append(fk)
            w += 1

            if _tqdm is not None:
                pbar.set_postfix(windows=w)

            # Slide buffer: drop the stride-many oldest packets
            flow_states[fk].buf = buf[stride:]

    if _tqdm is not None:
        pbar.close()
    reader.close()

    # ── Sanity-check window count ──────────────────────────────────────────
    if w != n_windows:
        # Prescan and main pass disagreed (e.g., truncated/corrupt PCAP).
        # Reload the written portion and re-save so the files have correct shape.
        print(f"[!] Window count mismatch: prescan={n_windows}, written={w}. "
              "Re-saving output to correct shapes.")
        final_seq = np.array(seq_mmap[:w], dtype=np.float32)
        final_rf  = np.array(rf_mmap[:w],  dtype=np.float32)
        del seq_mmap, rf_mmap
        np.save(seq_path, final_seq)
        np.save(rf_path,  final_rf)
    else:
        seq_mmap.flush()
        rf_mmap.flush()

    print(f"[+] Built {w} windows (incremental):")
    print(f"    Sequence tensor : ({w}, {window_size}, {n_seq_features})  "
          f"(W, N={window_size}, F={n_seq_features})")
    print(f"    RF stats tensor : ({w}, {n_rf_stats})  "
          f"(W, S={n_rf_stats})")

    return {
        "sequence_path":   seq_path,
        "rf_stats_path":   rf_path,
        "flow_ids":        window_flow_ids,
        "stats":           stats,
        "feature_names":   sequence_features,
        "rf_stat_names":   rf_stat_names,
        "num_windows":     w,
        "num_flows_total": len(flow_states),   # all flows seen in main pass
    }
