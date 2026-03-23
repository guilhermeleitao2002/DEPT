#!/usr/bin/env python3
"""
Generate a synthetic test PCAP for validating the ADEPT Packet-to-Tensor pipeline.

Creates realistic-looking network traffic with multiple flows (TCP, UDP, ICMP),
varying packet sizes, payloads, and timing patterns — enough to exercise every
stage of the pipeline (parsing, flow grouping, windowing, standardisation).

Usage
-----
    python scripts/generate_test_pcap.py                       # defaults
    python scripts/generate_test_pcap.py -o data/test/test.pcap -n 500
    python scripts/generate_test_pcap.py --flows 20 --window-size 10
"""

import argparse
import os
import random
import time

from scapy.all import (
    IP, TCP, UDP, ICMP, Ether, Raw, wrpcap,
)


def _random_payload(min_len=0, max_len=200):
    length = random.randint(min_len, max_len)
    return Raw(load=bytes(random.getrandbits(8) for _ in range(length)))


def generate_tcp_flow(src_ip, dst_ip, sport, dport, num_packets, base_ts):
    """Generate a bidirectional TCP flow with SYN, data, FIN."""
    pkts = []
    ts = base_ts
    seq_c, seq_s = 1000, 5000

    # SYN
    pkts.append((ts, Ether() / IP(src=src_ip, dst=dst_ip) /
                 TCP(sport=sport, dport=dport, flags="S", seq=seq_c)))
    ts += random.uniform(0.001, 0.05)

    # SYN-ACK
    pkts.append((ts, Ether() / IP(src=dst_ip, dst=src_ip) /
                 TCP(sport=dport, dport=sport, flags="SA", seq=seq_s, ack=seq_c + 1)))
    ts += random.uniform(0.001, 0.05)

    # ACK
    seq_c += 1
    pkts.append((ts, Ether() / IP(src=src_ip, dst=dst_ip) /
                 TCP(sport=sport, dport=dport, flags="A", seq=seq_c, ack=seq_s + 1)))
    ts += random.uniform(0.001, 0.05)

    # Data packets (bidirectional)
    for _ in range(num_packets - 5):  # reserve 2 for FIN/ACK
        if random.random() < 0.6:  # 60% client→server
            pkt = (Ether() / IP(src=src_ip, dst=dst_ip) /
                   TCP(sport=sport, dport=dport, flags="A", seq=seq_c, ack=seq_s + 1) /
                   _random_payload(10, 1400))
            seq_c += len(pkt[Raw].load)
        else:
            pkt = (Ether() / IP(src=dst_ip, dst=src_ip) /
                   TCP(sport=dport, dport=sport, flags="A", seq=seq_s, ack=seq_c) /
                   _random_payload(10, 1400))
            seq_s += len(pkt[Raw].load)
        ts += random.uniform(0.0005, 0.1)
        pkts.append((ts, pkt))

    # FIN
    pkts.append((ts + 0.01, Ether() / IP(src=src_ip, dst=dst_ip) /
                 TCP(sport=sport, dport=dport, flags="FA", seq=seq_c, ack=seq_s + 1)))
    pkts.append((ts + 0.02, Ether() / IP(src=dst_ip, dst=src_ip) /
                 TCP(sport=dport, dport=sport, flags="FA", seq=seq_s, ack=seq_c + 1)))

    return pkts


def generate_udp_flow(src_ip, dst_ip, sport, dport, num_packets, base_ts):
    """Generate a UDP flow (e.g. DNS-like or streaming)."""
    pkts = []
    ts = base_ts
    for i in range(num_packets):
        if random.random() < 0.5:
            pkt = (Ether() / IP(src=src_ip, dst=dst_ip) /
                   UDP(sport=sport, dport=dport) / _random_payload(20, 512))
        else:
            pkt = (Ether() / IP(src=dst_ip, dst=src_ip) /
                   UDP(sport=dport, dport=sport) / _random_payload(20, 512))
        ts += random.uniform(0.001, 0.2)
        pkts.append((ts, pkt))
    return pkts


def generate_icmp_flow(src_ip, dst_ip, num_packets, base_ts):
    """Generate ICMP echo request/reply pairs."""
    pkts = []
    ts = base_ts
    for i in range(num_packets // 2):
        pkts.append((ts, Ether() / IP(src=src_ip, dst=dst_ip) /
                     ICMP(type=8, id=random.randint(1, 65535), seq=i) /
                     _random_payload(32, 64)))
        ts += random.uniform(0.001, 0.01)
        pkts.append((ts, Ether() / IP(src=dst_ip, dst=src_ip) /
                     ICMP(type=0, id=random.randint(1, 65535), seq=i) /
                     _random_payload(32, 64)))
        ts += random.uniform(0.5, 1.5)
    return pkts


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic test PCAP")
    parser.add_argument("-o", "--output", default="data/test/test.pcap",
                        help="Output PCAP path (default: data/test/test.pcap)")
    parser.add_argument("-n", "--packets", type=int, default=500,
                        help="Approximate total number of packets (default: 500)")
    parser.add_argument("--flows", type=int, default=15,
                        help="Number of flows to generate (default: 15)")
    parser.add_argument("--window-size", type=int, default=10,
                        help="Target pipeline window size — ensures flows have enough packets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    subnets = ["10.0.0", "192.168.1", "172.16.0"]
    server_ips = ["93.184.216.34", "8.8.8.8", "1.1.1.1", "151.101.1.69"]
    common_ports = [80, 443, 8080, 22, 53, 8443, 3306]

    all_pkts = []
    base_ts = time.time() - 3600  # start 1 hour ago
    pkts_per_flow = max(args.window_size + 5, args.packets // args.flows)

    for i in range(args.flows):
        src_ip = f"{random.choice(subnets)}.{random.randint(2, 254)}"
        dst_ip = random.choice(server_ips)
        sport = random.randint(49152, 65535)
        dport = random.choice(common_ports)

        flow_type = random.choices(["tcp", "udp", "icmp"], weights=[0.6, 0.3, 0.1])[0]
        flow_ts = base_ts + random.uniform(0, 60)

        if flow_type == "tcp":
            all_pkts.extend(generate_tcp_flow(src_ip, dst_ip, sport, dport,
                                              pkts_per_flow, flow_ts))
        elif flow_type == "udp":
            all_pkts.extend(generate_udp_flow(src_ip, dst_ip, sport, dport,
                                              pkts_per_flow, flow_ts))
        else:
            all_pkts.extend(generate_icmp_flow(src_ip, dst_ip, pkts_per_flow, flow_ts))

    # Sort by timestamp (interleave flows realistically)
    all_pkts.sort(key=lambda x: x[0])

    # Write PCAP with per-packet timestamps
    packets = []
    for ts, pkt in all_pkts:
        pkt.time = ts
        packets.append(pkt)

    wrpcap(args.output, packets)
    print(f"Written {len(packets)} packets across ~{args.flows} flows → {args.output}")


if __name__ == "__main__":
    main()
