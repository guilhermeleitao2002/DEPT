"""
nPrint-compatible bit-level packet header extraction.

Reimplements the nPrint encoding (Holland et al., CCS 2021) in pure Python
using Scapy, producing the same fixed-width bit-level header matrix:

    IPv4 (480 bits) + TCP (480 bits) + UDP (64 bits) + ICMP (64 bits) = 1088 bits

Each bit is encoded as:
    1  = bit is set
    0  = bit is unset
   -1  = protocol layer not present in this packet

Additionally extracts lightweight per-packet metadata needed downstream:
    Timestamp, Source_IP, Destination_IP, Source_Port, Destination_Port,
    Protocol, Packet_Size, Payload_Length, Payload_Entropy, TCP_Flags

Inspired by FlowPaint (SIGCOMM'26) §3.3, selective masking zeros out
high-entropy identity fields (IPs, ports, IP ID) to prevent overfitting.
"""

import math
import os
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional

import numpy as np
import pandas as pd
from scapy.all import TCP, UDP, IP, ICMP, PcapReader

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None


# ──────────────────────────────────────────────────────────────────────────────
# nPrint header layout (matches nprint C++ source exactly)
# ──────────────────────────────────────────────────────────────────────────────

# (field_name, bit_width) — order must match nPrint's get_bitstring_header()
IPV4_FIELDS = [
    ("ipv4_ver", 4), ("ipv4_hl", 4), ("ipv4_tos", 8), ("ipv4_tl", 16),
    ("ipv4_id", 16), ("ipv4_rbit", 1), ("ipv4_dfbit", 1), ("ipv4_mfbit", 1),
    ("ipv4_foff", 13), ("ipv4_ttl", 8), ("ipv4_proto", 8), ("ipv4_cksum", 16),
    ("ipv4_src", 32), ("ipv4_dst", 32), ("ipv4_opt", 320),
]

TCP_FIELDS = [
    ("tcp_sprt", 16), ("tcp_dprt", 16), ("tcp_seq", 32), ("tcp_ackn", 32),
    ("tcp_doff", 4), ("tcp_res", 3), ("tcp_ns", 1),
    ("tcp_cwr", 1), ("tcp_ece", 1), ("tcp_urg", 1), ("tcp_ackf", 1),
    ("tcp_psh", 1), ("tcp_rst", 1), ("tcp_syn", 1), ("tcp_fin", 1),
    ("tcp_wsize", 16), ("tcp_cksum", 16), ("tcp_urp", 16), ("tcp_opt", 320),
]

UDP_FIELDS = [
    ("udp_sport", 16), ("udp_dport", 16), ("udp_len", 16), ("udp_cksum", 16),
]

ICMP_FIELDS = [
    ("icmp_type", 8), ("icmp_code", 8), ("icmp_cksum", 16), ("icmp_roh", 32),
]

# Max byte sizes for variable-length headers (nPrint pads to these)
_IPV4_MAX_BYTES = 60   # 20 base + 40 options
_TCP_MAX_BYTES = 60    # 20 base + 40 options
_UDP_BYTES = 8
_ICMP_BYTES = 8

# Total bits per layer
IPV4_BITS = sum(w for _, w in IPV4_FIELDS)   # 480
TCP_BITS = sum(w for _, w in TCP_FIELDS)      # 480
UDP_BITS = sum(w for _, w in UDP_FIELDS)      # 64
ICMP_BITS = sum(w for _, w in ICMP_FIELDS)    # 64
TOTAL_BITS = IPV4_BITS + TCP_BITS + UDP_BITS + ICMP_BITS  # 1088


def _build_column_names() -> List[str]:
    """Build the full list of nPrint column names (field_name + '_' + bit_idx)."""
    names = []
    for fields in [IPV4_FIELDS, TCP_FIELDS, UDP_FIELDS, ICMP_FIELDS]:
        for field_name, width in fields:
            for i in range(width):
                names.append(f"{field_name}_{i}")
    return names


NPRINT_COLUMNS = _build_column_names()
assert len(NPRINT_COLUMNS) == TOTAL_BITS


# ──────────────────────────────────────────────────────────────────────────────
# Bit-level encoding (mirrors nPrint's make_bitstring)
# ──────────────────────────────────────────────────────────────────────────────

def _bytes_to_bits(raw_bytes: bytes, max_bytes: int, fill: int = -1) -> List[int]:
    """
    Expand raw header bytes into a bit vector, padding to max_bytes.

    If raw_bytes is None or empty, the entire vector is filled with `fill` (-1).
    Actual bytes are expanded MSB-first (matching nPrint's make_bitstring).
    Remaining bytes up to max_bytes are zero-padded (matching nPrint behavior
    for variable-length headers like IPv4/TCP with options).
    """
    total_bits = max_bytes * 8
    if raw_bytes is None:
        return [fill] * total_bits

    bits = []
    for byte in raw_bytes:
        for j in range(7, -1, -1):
            bits.append((byte >> j) & 1)

    # Pad remaining space with zeros (header exists but options space is unused)
    remaining = total_bits - len(bits)
    if remaining > 0:
        bits.extend([0] * remaining)

    return bits[:total_bits]


def _shannon_entropy(data: bytes) -> float:
    """Shannon entropy (bits) of a byte sequence."""
    if not data:
        return 0.0
    occur = {}
    for byte in data:
        occur[byte] = occur.get(byte, 0) + 1
    length = len(data)
    return -sum((c / length) * math.log2(c / length) for c in occur.values())


# ──────────────────────────────────────────────────────────────────────────────
# Two-phase helpers for parallel nPrint encoding
# ──────────────────────────────────────────────────────────────────────────────

def _scan_raw_packet(pkt) -> tuple:
    """
    Phase 1 (main process): extract raw header bytes and metadata from a packet.

    Defers the bit expansion — the expensive _bytes_to_bits inner loops are
    moved to _encode_raw_chunk (worker processes).

    Returns
    -------
    (raw_ip, raw_tcp, raw_udp, raw_icmp, meta_dict)
    raw_* : bytes if that layer is present, None otherwise.
    """
    timestamp = float(pkt.time)
    framelen  = len(pkt)
    src_ip = "-1";  dst_ip   = "-1"
    src_port = -1;  dst_port = -1
    protocol = -1;  tcp_flags = 0
    payload_len = 0; payload_entropy = 0.0
    raw_ip = raw_tcp = raw_udp = raw_icmp = None

    if pkt.haslayer(IP):
        ip_layer        = pkt[IP]
        protocol        = ip_layer.proto
        src_ip          = ip_layer.src
        dst_ip          = ip_layer.dst
        raw_ip          = bytes(ip_layer)[: ip_layer.ihl * 4]
        payload_bytes   = bytes(ip_layer.payload)
        payload_len     = len(payload_bytes)
        payload_entropy = _shannon_entropy(payload_bytes)

    if pkt.haslayer(TCP):
        tcp_layer = pkt[TCP]
        src_port  = tcp_layer.sport
        dst_port  = tcp_layer.dport
        tcp_flags = int(tcp_layer.flags)
        raw_tcp   = bytes(tcp_layer)[: tcp_layer.dataofs * 4]

    if pkt.haslayer(UDP):
        udp_layer = pkt[UDP]
        src_port  = udp_layer.sport
        dst_port  = udp_layer.dport
        raw_udp   = bytes(udp_layer)[: _UDP_BYTES]

    if pkt.haslayer(ICMP):
        raw_icmp = bytes(pkt[ICMP])[: _ICMP_BYTES]

    meta = {
        "Timestamp":        timestamp,
        "Source_IP":        src_ip,
        "Destination_IP":   dst_ip,
        "Source_Port":      src_port,
        "Destination_Port": dst_port,
        "Protocol":         protocol,
        "Packet_Size":      framelen,
        "Payload_Length":   payload_len,
        "Payload_Entropy":  payload_entropy,
        "TCP_Flags":        tcp_flags,
    }
    return raw_ip, raw_tcp, raw_udp, raw_icmp, meta


def _encode_raw_chunk(raw_list: list) -> np.ndarray:
    """
    Phase 2 (worker process): expand raw header bytes into nPrint bit vectors.

    Called via ProcessPoolExecutor — no Scapy dependency here.  With the
    default fork start method on Linux the module constants (TOTAL_BITS, etc.)
    are inherited from the parent; with spawn they are re-imported.

    Parameters
    ----------
    raw_list : list of (raw_ip, raw_tcp, raw_udp, raw_icmp)
        Each element is bytes | None, as returned by _scan_raw_packet.

    Returns
    -------
    np.ndarray of shape (len(raw_list), TOTAL_BITS) int8
    """
    n = len(raw_list)
    result = np.empty((n, TOTAL_BITS), dtype=np.int8)
    for i, (raw_ip, raw_tcp, raw_udp, raw_icmp) in enumerate(raw_list):
        bits = (
            _bytes_to_bits(raw_ip,  _IPV4_MAX_BYTES)
            + _bytes_to_bits(raw_tcp, _TCP_MAX_BYTES)
            + _bytes_to_bits(raw_udp,  _UDP_BYTES)
            + _bytes_to_bits(raw_icmp, _ICMP_BYTES)
        )
        result[i] = bits
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Selective masking (FlowPaint §3.3.2)
# ──────────────────────────────────────────────────────────────────────────────

# Default prefixes to mask — high-entropy identity fields
DEFAULT_MASK_PREFIXES = [
    "ipv4_src", "ipv4_dst",    # IP addresses
    "ipv4_id",                  # IP identification (host-specific)
    "tcp_sprt", "tcp_dprt",    # TCP ports
    "udp_sport", "udp_dport",  # UDP ports
]


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def extract_packet_features(
    pcap_path: str,
    mask_prefixes: Optional[List[str]] = None,
    workers: Optional[int] = None,
) -> tuple:
    """
    Read a PCAP and return nPrint bit-level features as a compact numpy
    array plus a lightweight metadata DataFrame.

    Uses a two-phase approach to parallelize the nPrint bit expansion:

    Phase 1 (main process, sequential):
        Read packets with Scapy, extract raw header bytes + metadata.
        Submits each CHUNK_SIZE-packet batch to the worker pool immediately.

    Phase 2 (worker pool, parallel):
        Each worker expands raw bytes into the 1088-bit nPrint vector.
        Workers have no Scapy dependency — pure Python + numpy.

    Parameters
    ----------
    pcap_path : str
        Path to input PCAP file.
    mask_prefixes : list[str] or None
        nPrint column prefixes to zero out (identity masking).
        Defaults to DEFAULT_MASK_PREFIXES. Pass [] to disable masking.
    workers : int or None
        Number of parallel encoding worker processes.
        Defaults to os.cpu_count() (all available cores).

    Returns
    -------
    (nprint_bits, meta_df) : tuple[np.ndarray, pd.DataFrame]
        nprint_bits : int8 array of shape (P, 1088)
        meta_df     : DataFrame with columns Timestamp, Source_IP, …, TCP_Flags
    """
    n_workers  = workers or (os.cpu_count() or 1)
    CHUNK_SIZE = 50_000

    reader = PcapReader(pcap_path)
    pbar = (
        _tqdm(reader, desc="      Parsing packets", unit=" pkt",
              unit_scale=True, dynamic_ncols=True)
        if _tqdm is not None else reader
    )

    all_meta   = []   # metadata rows — accumulated in main process
    raw_chunk  = []   # (raw_ip, raw_tcp, raw_udp, raw_icmp) for current chunk
    pending    = []   # Future[np.ndarray] in submission order

    # Phase 1 + dispatch: read sequentially, hand off encoding to workers.
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for pkt in pbar:
            raw_ip, raw_tcp, raw_udp, raw_icmp, meta = _scan_raw_packet(pkt)
            all_meta.append(meta)
            raw_chunk.append((raw_ip, raw_tcp, raw_udp, raw_icmp))

            if len(raw_chunk) >= CHUNK_SIZE:
                pending.append(executor.submit(_encode_raw_chunk, raw_chunk))
                raw_chunk = []   # free the Python bytes objects

        if raw_chunk:
            pending.append(executor.submit(_encode_raw_chunk, raw_chunk))
            raw_chunk = []

        # Phase 2 collect: gather results in submission order (preserves packet
        # order) while workers may still be running earlier chunks.
        bit_chunks = [fut.result() for fut in pending]

    if _tqdm is not None:
        pbar.close()
    reader.close()

    n_packets = sum(c.shape[0] for c in bit_chunks)
    if n_packets == 0:
        raise ValueError(
            f"No packets found in {pcap_path}. "
            "If this is a time-filtered PCAP, the time window may be incorrect."
        )

    # Build compact nPrint array (stays as numpy — never enters pandas)
    nprint_bits = np.vstack(bit_chunks) if len(bit_chunks) > 1 else bit_chunks[0]
    del bit_chunks

    # Build lightweight metadata DataFrame (~10 columns only)
    meta_df = pd.DataFrame(all_meta)
    del all_meta

    # Apply selective masking directly on the numpy array
    if mask_prefixes is None:
        mask_prefixes = DEFAULT_MASK_PREFIXES
    if mask_prefixes:
        print(f"      Masking {len(mask_prefixes)} field prefixes: {mask_prefixes}")
        cols_to_mask = []
        for i, col in enumerate(NPRINT_COLUMNS):
            for prefix in mask_prefixes:
                if col.startswith(prefix + "_") and col[len(prefix) + 1:].isdigit():
                    cols_to_mask.append(i)
                    break
        if cols_to_mask:
            nprint_bits[:, cols_to_mask] = 0
        print(f"      {len(cols_to_mask)} columns masked to zero")

    print(f"      Final shape: ({n_packets}, {nprint_bits.shape[1]}) "
          f"({TOTAL_BITS} nPrint bits + {len(meta_df.columns)} metadata columns)")

    return nprint_bits, meta_df


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract nPrint bit-level features from a PCAP file."
    )
    parser.add_argument("-i", "--input", required=True, help="Input PCAP path")
    parser.add_argument("-o", "--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--no-mask", action="store_true",
        help="Disable identity masking (keep IPs, ports, etc.)"
    )
    args = parser.parse_args()

    mask = [] if args.no_mask else None  # None = use defaults
    nprint_bits, meta_df = extract_packet_features(args.input, mask_prefixes=mask)
    # Recombine for CSV output (CLI-only; pipeline keeps them separate)
    nprint_df = pd.DataFrame(nprint_bits, columns=NPRINT_COLUMNS)
    df = pd.concat([meta_df, nprint_df], axis=1)
    df.to_csv(args.output, index=False)
    print(f"[+] Saved to {args.output}")
