"""
Packet-level feature extraction from PCAP files.

Adapted from TANTGen's dataloader.py — extended with Direction field
and per-flow IPAT computation (delegated to flow_grouper).

At this stage we extract *per-packet* features from the raw PCAP.
Flow grouping and IPAT (which is flow-relative) happen downstream.
"""

import math
import argparse

import pandas as pd
from scapy.all import TCP, UDP, IP, IPv6, ARP, ICMP, PcapReader


def _shannon_entropy(data: bytes) -> float:
    """Shannon entropy (bits) of a byte sequence."""
    if not data:
        return 0.0
    occur = {}
    for byte in data:
        occur[byte] = occur.get(byte, 0) + 1
    length = len(data)
    return -sum((c / length) * math.log2(c / length) for c in occur.values())


def extract_packet_features(pcap_path: str) -> pd.DataFrame:
    """
    Read a PCAP and return a DataFrame with one row per packet.

    Columns produced
    ----------------
    Timestamp, Packet_Size, Protocol, Source_IP, Destination_IP,
    Source_Port, Destination_Port, TCP_Flags, Payload_Length,
    Payload_Entropy, IP_Type

    Notes
    -----
    - IAT is NOT computed here (it is global in the PCAP, not per-flow).
      Per-flow IPAT is computed later in flow_grouper.
    - Direction is also assigned in flow_grouper (needs to know which
      endpoint is the "source" of the flow).
    """
    print(f"[i] Extracting packet features from: {pcap_path}")
    reader = PcapReader(pcap_path)
    rows = []

    for pkt in reader:
        timestamp = float(pkt.time)
        framelen = len(pkt)

        # --- IP layer ---
        ip_type = -1
        protocol = -1
        src_ip = "-1"
        dst_ip = "-1"

        if pkt.haslayer(IP):
            ip_type = 0
            protocol = pkt[IP].proto
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
        elif pkt.haslayer(IPv6):
            ip_type = 1
            src_ip = pkt[IPv6].src
            dst_ip = pkt[IPv6].dst

        # --- Transport layer ---
        src_port = -1
        dst_port = -1
        if pkt.haslayer(TCP):
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
        elif pkt.haslayer(UDP):
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport

        # Handle L2 protocols (ARP / ICMP fallback — same logic as TANTGen)
        if src_port == -1:
            if pkt.haslayer(ARP):
                src_ip = pkt[ARP].psrc
                dst_ip = pkt[ARP].pdst
                ip_type = 0
            elif pkt.haslayer(ICMP):
                ip_type = 0

        # --- Payload ---
        payload_len = len(pkt[IP].payload) if pkt.haslayer(IP) else 0
        payload_bytes = bytes(pkt[IP].payload) if pkt.haslayer(IP) else b""
        payload_entropy = _shannon_entropy(payload_bytes)

        # --- TCP flags ---
        tcp_flags = int(pkt[TCP].flags) if pkt.haslayer(TCP) else 0

        rows.append({
            "Timestamp": timestamp,
            "Packet_Size": framelen,
            "Protocol": protocol,
            "Source_IP": src_ip,
            "Destination_IP": dst_ip,
            "Source_Port": src_port,
            "Destination_Port": dst_port,
            "TCP_Flags": tcp_flags,
            "Payload_Length": payload_len,
            "Payload_Entropy": payload_entropy,
            "IP_Type": ip_type,
        })

    reader.close()
    print(f"[+] Extracted {len(rows)} packets")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI entry-point (mirrors TANTGen's dataloader.py __main__)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract per-packet features from a PCAP file."
    )
    parser.add_argument("-i", "--input", required=True, help="Input PCAP path")
    parser.add_argument("-o", "--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    df = extract_packet_features(args.input)
    df.to_csv(args.output, index=False)
    print(f"[+] Saved to {args.output}")
