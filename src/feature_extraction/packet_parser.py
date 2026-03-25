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
from typing import List, Optional

import numpy as np
import pandas as pd
from scapy.all import TCP, UDP, IP, IPv6, ICMP, PcapReader


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


def _encode_packet(pkt) -> tuple:
    """
    Encode a single Scapy packet into:
    1. nPrint bit vector (1088 ints)
    2. Metadata dict (Timestamp, IPs, ports, etc.)

    Returns (bits_list, metadata_dict).
    """
    timestamp = float(pkt.time)
    framelen = len(pkt)

    # --- Defaults ---
    src_ip = "-1"
    dst_ip = "-1"
    src_port = -1
    dst_port = -1
    protocol = -1
    tcp_flags = 0
    payload_len = 0
    payload_entropy = 0.0

    # --- IPv4 layer ---
    if pkt.haslayer(IP):
        ip_layer = pkt[IP]
        protocol = ip_layer.proto
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst

        # Extract raw IPv4 header bytes (variable length: ip_hl * 4)
        ip_hdr_len = ip_layer.ihl * 4
        ip_raw = bytes(ip_layer)[:ip_hdr_len]
        ipv4_bits = _bytes_to_bits(ip_raw, _IPV4_MAX_BYTES)

        # Payload
        payload_bytes = bytes(ip_layer.payload)
        payload_len = len(payload_bytes)
        payload_entropy = _shannon_entropy(payload_bytes)
    else:
        ipv4_bits = _bytes_to_bits(None, _IPV4_MAX_BYTES)

    # --- TCP layer ---
    if pkt.haslayer(TCP):
        tcp_layer = pkt[TCP]
        src_port = tcp_layer.sport
        dst_port = tcp_layer.dport
        tcp_flags = int(tcp_layer.flags)

        tcp_hdr_len = tcp_layer.dataofs * 4
        # Extract raw TCP header bytes from the TCP layer
        tcp_raw = bytes(tcp_layer)[:tcp_hdr_len]
        tcp_bits = _bytes_to_bits(tcp_raw, _TCP_MAX_BYTES)
    else:
        tcp_bits = _bytes_to_bits(None, _TCP_MAX_BYTES)

    # --- UDP layer ---
    if pkt.haslayer(UDP):
        udp_layer = pkt[UDP]
        src_port = udp_layer.sport
        dst_port = udp_layer.dport

        udp_raw = bytes(udp_layer)[:_UDP_BYTES]
        udp_bits = _bytes_to_bits(udp_raw, _UDP_BYTES)
    else:
        udp_bits = _bytes_to_bits(None, _UDP_BYTES)

    # --- ICMP layer ---
    if pkt.haslayer(ICMP):
        icmp_raw = bytes(pkt[ICMP])[:_ICMP_BYTES]
        icmp_bits = _bytes_to_bits(icmp_raw, _ICMP_BYTES)
    else:
        icmp_bits = _bytes_to_bits(None, _ICMP_BYTES)

    # Combine all layers
    bits = ipv4_bits + tcp_bits + udp_bits + icmp_bits

    metadata = {
        "Timestamp": timestamp,
        "Source_IP": src_ip,
        "Destination_IP": dst_ip,
        "Source_Port": src_port,
        "Destination_Port": dst_port,
        "Protocol": protocol,
        "Packet_Size": framelen,
        "Payload_Length": payload_len,
        "Payload_Entropy": payload_entropy,
        "TCP_Flags": tcp_flags,
    }

    return bits, metadata


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
# Selective masking (FlowPaint §3.3.2)
# ──────────────────────────────────────────────────────────────────────────────

# Default prefixes to mask — high-entropy identity fields
DEFAULT_MASK_PREFIXES = [
    "ipv4_src", "ipv4_dst",    # IP addresses
    "ipv4_id",                  # IP identification (host-specific)
    "tcp_sprt", "tcp_dprt",    # TCP ports
    "udp_sport", "udp_dport",  # UDP ports
]


def apply_masking(
    df: pd.DataFrame,
    mask_prefixes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Zero out nPrint columns matching the given prefixes.

    This prevents the model from learning identity shortcuts
    (e.g., correlating evasion with a specific source IP).
    """
    if mask_prefixes is None:
        mask_prefixes = DEFAULT_MASK_PREFIXES

    cols_to_mask = []
    for col in df.columns:
        for prefix in mask_prefixes:
            # Match "ipv4_src_0", "ipv4_src_1", ... but not "ipv4_srcXYZ"
            # nPrint columns are: prefix + "_" + digit
            if col.startswith(prefix + "_") and col[len(prefix) + 1:].isdigit():
                cols_to_mask.append(col)
                break

    if cols_to_mask:
        df = df.copy()
        df[cols_to_mask] = 0

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def extract_packet_features(
    pcap_path: str,
    mask_prefixes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Read a PCAP and return a DataFrame with nPrint bit-level encoding
    plus metadata columns for downstream processing.

    Each row is one packet. Columns include:
        - 1088 nPrint bit columns (ipv4_ver_0, ipv4_ver_1, ..., icmp_roh_31)
        - Metadata: Timestamp, Source_IP, Destination_IP, Source_Port,
          Destination_Port, Protocol, Packet_Size, Payload_Length,
          Payload_Entropy, TCP_Flags

    Parameters
    ----------
    pcap_path : str
        Path to input PCAP file.
    mask_prefixes : list[str] or None
        nPrint column prefixes to zero out (identity masking).
        Defaults to DEFAULT_MASK_PREFIXES. Pass [] to disable masking.

    Returns
    -------
    pd.DataFrame
        Combined nPrint + metadata DataFrame, one row per packet.
    """
    print(f"[1/3] Extracting nPrint bit-level features from: {pcap_path}")
    reader = PcapReader(pcap_path)

    all_bits = []
    all_meta = []

    for pkt in reader:
        bits, meta = _encode_packet(pkt)
        all_bits.append(bits)
        all_meta.append(meta)

    reader.close()
    n_packets = len(all_bits)
    print(f"      {n_packets} packets extracted")

    # Build nPrint DataFrame
    nprint_array = np.array(all_bits, dtype=np.int8)
    nprint_df = pd.DataFrame(nprint_array, columns=NPRINT_COLUMNS)

    # Build metadata DataFrame
    meta_df = pd.DataFrame(all_meta)

    # Apply selective masking
    if mask_prefixes is None:
        mask_prefixes = DEFAULT_MASK_PREFIXES
    if mask_prefixes:
        print(f"      Masking {len(mask_prefixes)} field prefixes: {mask_prefixes}")
        nprint_df = apply_masking(nprint_df, mask_prefixes)

    masked_count = sum(
        1 for c in nprint_df.columns
        if (nprint_df[c] == 0).all() and any(
            c.startswith(p + "_") for p in (mask_prefixes or [])
        )
    )
    print(f"      {masked_count} columns masked to zero")

    # Combine: metadata columns first, then nPrint bit columns
    df = pd.concat([meta_df, nprint_df], axis=1)
    print(f"      Final shape: {df.shape} ({len(NPRINT_COLUMNS)} nPrint bits + "
          f"{len(meta_df.columns)} metadata columns)")

    return df


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
    df = extract_packet_features(args.input, mask_prefixes=mask)
    df.to_csv(args.output, index=False)
    print(f"[+] Saved to {args.output}")
