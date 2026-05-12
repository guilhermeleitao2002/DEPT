"""
Incremental (online) packet-to-tensor pipeline (streaming-native edition).

Eliminates the global timestamp sort required by the batch pipeline
(pcap_parser → flow_grouper → tensor_builder) and produces the same ragged
per-flow tensor layout used by the streaming RF/TCN ensemble.

Passes
------
Pass 1 — lightweight prescan (5-tuple + timestamp, no nPrint encoding):
    • Counts packets per canonical flow → allocates ``flow_packets.npy``
      and ``rf_checkpoints.npy`` exactly once.
    • Records each flow's first-occurrence order so each flow gets a
      contiguous slot in the output, written to in arrival order at
      Pass 2 time.
    • Computes IPAT mean/std via Welford's online algorithm (same role as
      the previous batch pipeline's mean_std.json).

Pass 2 — main encoding pass:
    • Full nPrint bit encoding per packet.
    • Per-flow state: encoded packet counter, ``StreamingStats`` for RF
      features.
    • Per-packet write directly into the memory-mapped ``flow_packets.npy``
      at the flow's pre-assigned slot.
    • Every ``rf_checkpoint_stride`` packets per flow, emit a multi-scale
      RF row into ``rf_checkpoints.npy``.

Memory profile
--------------
Peak RAM is O(active_flows × per_flow_state). Per-flow state is the
``StreamingStats`` accumulator + an integer counter — small constants.
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
from scapy.all import IP, TCP, PcapReader

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

from .pcap_parser import (
    _encode_packet,
    NPRINT_COLUMNS,
    DEFAULT_MASK_PREFIXES,
)
from .tensor_builder import StreamingStats, _SUPPORTED_AGGS


# ──────────────────────────────────────────────────────────────────────────────
# Per-flow mutable state
# ──────────────────────────────────────────────────────────────────────────────

class _FlowState:
    """Per-flow bookkeeping during Pass 2 (incremental encoding).

    Holds the running Welford accumulator for the RF base-feature aggregates
    plus the per-flow packet counter that drives both packet-slot writes
    into ``flow_packets.npy`` and the every-N-packet checkpoint emission
    into ``rf_checkpoints.npy``.
    """
    __slots__ = ("init_ip", "init_port", "last_ts", "pkt_idx",
                 "ckpt_idx", "stats")

    def __init__(self, src_ip: str, src_port: int, timestamp: float,
                 n_base: int):
        self.init_ip = src_ip
        self.init_port = src_port
        self.last_ts = timestamp
        self.pkt_idx = 0
        self.ckpt_idx = 0
        self.stats = StreamingStats(n_base)


# ──────────────────────────────────────────────────────────────────────────────
# Canonical 5-tuple key  (must match flow_grouper._compute_canonical_keys)
# ──────────────────────────────────────────────────────────────────────────────

def _canonical_key(src_ip, dst_ip, src_port, dst_port, proto) -> tuple:
    """Order-independent 5-tuple flow key, matching the batch pipeline."""
    src_str = f"{src_ip}:{int(src_port)}"
    dst_str = f"{dst_ip}:{int(dst_port)}"
    if src_str <= dst_str:
        return (str(src_ip), str(dst_ip), int(src_port),
                int(dst_port), int(proto))
    return (str(dst_ip), str(src_ip), int(dst_port),
            int(src_port), int(proto))


# ──────────────────────────────────────────────────────────────────────────────
# Pass 1 — lightweight prescan
# ──────────────────────────────────────────────────────────────────────────────

def _prescan(pcap_path: str) -> tuple:
    """Count packets per flow, capture arrival order, compute IPAT stats.

    Returns
    -------
    (flow_arrival_order, flow_counts, ipat_mean, ipat_std)
        flow_arrival_order : list[tuple]
            Flow keys in first-occurrence order.
        flow_counts : dict[tuple, int]
            Packet count per flow.
        ipat_mean, ipat_std : float
            Welford sample statistics (matches pandas .std()).
    """
    flow_arrival_order: list = []
    flow_counts: dict = {}
    flow_last_ts: dict = {}

    welf_n = 0
    welf_mean = 0.0
    welf_M2 = 0.0

    reader = PcapReader(pcap_path)
    pbar = (
        _tqdm(reader, desc="      Prescan", unit=" pkt",
              unit_scale=True, dynamic_ncols=True)
        if _tqdm is not None else reader
    )

    for pkt in pbar:
        ts = float(pkt.time)

        src_ip = dst_ip = "-1"
        src_port = dst_port = proto = -1
        if pkt.haslayer(IP):
            ip = pkt[IP]
            src_ip = ip.src
            dst_ip = ip.dst
            proto = ip.proto
            if pkt.haslayer(TCP):
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
            elif pkt.haslayer("UDP"):
                src_port = pkt["UDP"].sport
                dst_port = pkt["UDP"].dport

        fk = _canonical_key(src_ip, dst_ip, src_port, dst_port, proto)

        ipat = (ts - flow_last_ts[fk]) if fk in flow_last_ts else 0.0
        welf_n += 1
        delta = ipat - welf_mean
        welf_mean += delta / welf_n
        welf_M2 += delta * (ipat - welf_mean)

        if fk not in flow_counts:
            flow_arrival_order.append(fk)
            flow_counts[fk] = 0
        flow_counts[fk] += 1
        flow_last_ts[fk] = ts

    if _tqdm is not None:
        pbar.close()
    reader.close()

    ipat_mean = welf_mean
    ipat_std = (welf_M2 / (welf_n - 1)) ** 0.5 if welf_n > 1 else 1.0
    if ipat_std == 0.0:
        ipat_std = 1.0
    return flow_arrival_order, flow_counts, ipat_mean, ipat_std


# ──────────────────────────────────────────────────────────────────────────────
# Pass 2 — incremental main pass
# ──────────────────────────────────────────────────────────────────────────────

def build_tensors_incremental(
    pcap_path: str,
    output_dir: str,
    rf_checkpoint_stride: int = 10,
    rf_base_features: Optional[List[str]] = None,
    rf_agg_functions: Optional[List[str]] = None,
    mean_std_path: Optional[str] = None,
    mask_prefixes: Optional[List[str]] = None,
) -> Dict:
    """Two-pass incremental packet-to-tensor pipeline (ragged output).

    Pass 1 (prescan): counts packets per flow, captures arrival order, and
    computes IPAT mean/std for z-score normalisation. Pass 2 (main): nPrint
    encoding per packet, per-flow state update, direct writes into the
    pre-allocated memory-mapped ``flow_packets.npy``; every
    ``rf_checkpoint_stride`` packets per flow a multi-scale RF row is
    appended to ``rf_checkpoints.npy``.

    Returns a dict with the same keys as ``tensor_builder.build_tensors``.
    """
    if rf_base_features is None:
        rf_base_features = [
            "IPAT", "Packet_Size", "Payload_Length",
            "Payload_Entropy", "TCP_Flags", "Direction",
        ]
    if rf_agg_functions is None:
        rf_agg_functions = list(_SUPPORTED_AGGS)
    if mask_prefixes is None:
        mask_prefixes = DEFAULT_MASK_PREFIXES
    if mean_std_path is None:
        mean_std_path = os.path.join(output_dir, "mean_std.json")
    if rf_checkpoint_stride < 1:
        raise ValueError(
            f"rf_checkpoint_stride must be >=1, got {rf_checkpoint_stride}"
        )
    unknown = [a for a in rf_agg_functions if a not in _SUPPORTED_AGGS]
    if unknown:
        raise ValueError(
            f"Unsupported RF aggregations {unknown}. Supported: {_SUPPORTED_AGGS}"
        )

    sequence_features = list(NPRINT_COLUMNS) + ["IPAT", "Direction"]
    n_seq_features = len(sequence_features)   # 1090

    # Pass-1 collects flow lengths and IPAT stats.
    print("      Prescan: counting packets + computing IPAT statistics")
    flow_arrival_order, flow_counts, ipat_mean_p1, ipat_std_p1 = _prescan(pcap_path)

    if os.path.exists(mean_std_path):
        print(f"[i] Loading mean/std from {mean_std_path}")
        with open(mean_std_path) as f:
            stats = json.load(f)
        ipat_mean = stats.get("IPAT", {}).get("mean", 0.0)
        ipat_std = stats.get("IPAT", {}).get("std", 1.0)
    else:
        ipat_mean = ipat_mean_p1
        ipat_std = ipat_std_p1
        stats = {"IPAT": {"mean": ipat_mean, "std": ipat_std}}
        os.makedirs(output_dir, exist_ok=True)
        with open(mean_std_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[+] Saved mean/std to {mean_std_path}")

    # Filter out anything that's strictly empty (shouldn't happen but defensive).
    flow_arrival_order = [fk for fk in flow_arrival_order if flow_counts.get(fk, 0) > 0]
    n_flows = len(flow_arrival_order)

    # Effective RF base columns (filtered against what _encode_packet exposes
    # — see Pass 2 below). Sticking with the full list keeps the on-disk
    # rf_stat_names order stable across batch and incremental modes.
    rf_meta_cols = list(rf_base_features)
    rf_stat_names = [
        f"{feat}_{agg}"
        for feat in rf_meta_cols
        for agg in rf_agg_functions
    ]
    n_rf_stats = len(rf_stat_names)
    n_base = len(rf_meta_cols)

    # Pre-compute offsets.
    flow_packet_offsets = np.zeros((n_flows, 2), dtype=np.int64)
    rf_checkpoint_offsets = np.zeros((n_flows, 2), dtype=np.int64)
    pkt_cursor = 0
    ckpt_cursor = 0
    flow_slot: dict = {}
    for i, fk in enumerate(flow_arrival_order):
        L = int(flow_counts[fk])
        flow_packet_offsets[i, 0] = pkt_cursor
        flow_packet_offsets[i, 1] = L
        n_ckpt = L // rf_checkpoint_stride
        rf_checkpoint_offsets[i, 0] = ckpt_cursor
        rf_checkpoint_offsets[i, 1] = n_ckpt
        flow_slot[fk] = i
        pkt_cursor += L
        ckpt_cursor += n_ckpt

    total_packets = int(pkt_cursor)
    total_checkpoints = int(ckpt_cursor)

    if total_packets == 0:
        print("[!] No packets to write.")
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
        }

    os.makedirs(output_dir, exist_ok=True)
    flow_packets_path = os.path.join(output_dir, "flow_packets.npy")
    flow_offsets_path = os.path.join(output_dir, "flow_offsets.npy")
    rf_checkpoints_path = os.path.join(output_dir, "rf_checkpoints.npy")
    rf_checkpoint_offsets_path = os.path.join(
        output_dir, "rf_checkpoint_offsets.npy",
    )

    pkt_mmap = np.lib.format.open_memmap(
        flow_packets_path, mode="w+", dtype=np.float32,
        shape=(total_packets, n_seq_features),
    )
    ckpt_shape = (max(total_checkpoints, 0), n_rf_stats)
    ckpt_mmap = np.lib.format.open_memmap(
        rf_checkpoints_path, mode="w+", dtype=np.float32,
        shape=ckpt_shape,
    )

    # Pre-compute masking column indices (done once, not per packet).
    cols_to_mask: list = []
    if mask_prefixes:
        for i, col in enumerate(NPRINT_COLUMNS):
            for prefix in mask_prefixes:
                if col.startswith(prefix + "_") and col[len(prefix) + 1:].isdigit():
                    cols_to_mask.append(i)
                    break

    # Pass 2: encode packets and write into pre-allocated slots.
    flow_states: dict = {}
    reader = PcapReader(pcap_path)
    pbar = (
        _tqdm(reader, desc="      Encoding packets", unit=" pkt",
              unit_scale=True, dynamic_ncols=True)
        if _tqdm is not None else reader
    )

    pkts_seen = 0
    ckpts_written = 0
    for pkt in pbar:
        bits_list, meta = _encode_packet(pkt)
        bits = np.array(bits_list, dtype=np.int8)
        if cols_to_mask:
            bits[cols_to_mask] = 0

        fk = _canonical_key(
            meta["Source_IP"], meta["Destination_IP"],
            meta["Source_Port"], meta["Destination_Port"],
            meta["Protocol"],
        )
        ts = meta["Timestamp"]

        if fk not in flow_states:
            flow_states[fk] = _FlowState(
                meta["Source_IP"], meta["Source_Port"], ts, n_base,
            )
            ipat = 0.0
            direction = 1
        else:
            state = flow_states[fk]
            ipat = ts - state.last_ts
            state.last_ts = ts
            direction = (
                1 if (meta["Source_IP"], meta["Source_Port"])
                     == (state.init_ip, state.init_port)
                else 0
            )

        state = flow_states[fk]
        slot = flow_slot.get(fk)
        if slot is None:
            # Should not happen — prescan saw every flow that Pass 2 sees.
            continue
        flow_start = int(flow_packet_offsets[slot, 0])
        flow_len = int(flow_packet_offsets[slot, 1])
        per_flow_idx = state.pkt_idx
        if per_flow_idx >= flow_len:
            # Defensive: Pass 2 yielded more packets than prescan counted
            # (truncated/corrupt PCAP); skip rather than corrupt the memmap.
            continue

        row = flow_start + per_flow_idx
        pkt_mmap[row, :1088] = bits.astype(np.float32)
        pkt_mmap[row, 1088] = (ipat - ipat_mean) / ipat_std
        pkt_mmap[row, 1089] = float(direction)

        # Streaming RF stats over the growing flow prefix.
        # Order matches rf_meta_cols == rf_base_features.
        feat_vec = np.array(
            [ipat, meta["Packet_Size"], meta["Payload_Length"],
             meta["Payload_Entropy"], meta["TCP_Flags"], float(direction)],
            dtype=np.float64,
        )
        state.stats.update(feat_vec)
        state.pkt_idx += 1
        pkts_seen += 1

        if state.pkt_idx % rf_checkpoint_stride == 0:
            ckpt_slot = int(rf_checkpoint_offsets[slot, 0]) + state.ckpt_idx
            if state.ckpt_idx < int(rf_checkpoint_offsets[slot, 1]):
                ckpt_mmap[ckpt_slot] = state.stats.emit(rf_agg_functions)
                state.ckpt_idx += 1
                ckpts_written += 1

        if _tqdm is not None and pkts_seen % 50000 == 0:
            pbar.set_postfix(packets=pkts_seen, checkpoints=ckpts_written)

    if _tqdm is not None:
        pbar.close()
    reader.close()

    # Persist offsets and flush memmaps.
    np.save(flow_offsets_path, flow_packet_offsets)
    np.save(rf_checkpoint_offsets_path, rf_checkpoint_offsets)
    pkt_mmap.flush()
    ckpt_mmap.flush()

    if pkts_seen != total_packets:
        print(f"[!] Packet-count mismatch: prescan={total_packets}, "
              f"main={pkts_seen}. Output left as allocated; downstream "
              f"loaders should rely on flow_offsets.npy for valid ranges.")

    print(f"[+] Built ragged tensors (incremental):")
    print(f"    flow_packets    : {pkt_mmap.shape}   (P, F={n_seq_features})")
    print(f"    flow_offsets    : {flow_packet_offsets.shape}  (N_flows, 2)")
    print(f"    rf_checkpoints  : {ckpt_mmap.shape}  "
          f"(C, S={n_rf_stats})  stride={rf_checkpoint_stride}")
    print(f"    rf_ckpt_offsets : {rf_checkpoint_offsets.shape}  (N_flows, 2)")

    return {
        "flow_packets_path": flow_packets_path,
        "flow_offsets_path": flow_offsets_path,
        "rf_checkpoints_path": rf_checkpoints_path,
        "rf_checkpoint_offsets_path": rf_checkpoint_offsets_path,
        "flow_ids": flow_arrival_order,
        "stats": stats,
        "feature_names": sequence_features,
        "rf_stat_names": rf_stat_names,
        "num_flows": n_flows,
        "num_packets": total_packets,
        "num_checkpoints": total_checkpoints,
        "rf_checkpoint_stride": rf_checkpoint_stride,
    }
