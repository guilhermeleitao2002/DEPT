"""Traffic-space realization: map a purified nPrint flow back onto real packets.

This is ADEPT's "parallel pipeline". The reconstructor denoises in feature
space (nPrint tensors); to honor the thesis's realizability claim (Table 1:
scope = Ψ traffic-space) we map the purified length/timing back onto the
**original adversarial PCAP packets** and validate the result with ``tshark``.

Why overlay onto the original packet rather than rebuild from nPrint? nPrint
covers only headers (not payload content) and the pipeline zeroes identity
fields (IPs/ports). Keeping the original packet and writing back only the two
perturbed quantities sidesteps both gaps:

  * **size**  → the reconstructed IPv4 total-length sets how much application
    payload to trim/pad (TANTGen's size attack pads; purification un-pads);
  * **iat**   → the reconstructed (de-normalised) IPAT sets the packet's
    capture timestamp.

Identity + payload content are preserved verbatim; scapy recomputes IP/L4
checksums and lengths on rebuild.

Alignment: the tensorizer groups by canonical 5-tuple and stable-sorts by
timestamp (:mod:`src.feature_extraction.flow_grouper`). We reproduce exactly
that here, giving a 1:1 ``tensor-row ↔ packet`` correspondence; a flow whose
packet count doesn't match its tensor length is skipped (counted as a miss).
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# nPrint IPv4 header bit offsets (MSB-first), matching pcap_parser's layout.
_IDX_IPV4_TL = (16, 32)
_IDX_IPAT = 1088
_WEIGHTS16 = (1 << np.arange(15, -1, -1)).astype(np.float64)


def _round_tristate(x: np.ndarray) -> np.ndarray:
    """Round continuous reconstruction values back to nPrint {-1, 0, 1}."""
    return np.clip(np.round(x), -1.0, 1.0)


def _decode_bits(bits: np.ndarray, weights: np.ndarray) -> float:
    """Decode an MSB-first bit slice to an unsigned int (−1 clamped to 0)."""
    cleaned = np.maximum(_round_tristate(bits), 0.0)
    return float(cleaned @ weights)


def decode_total_len(seq_row: np.ndarray) -> int:
    """Reconstructed IPv4 total length (bytes) from one purified packet row."""
    return int(_decode_bits(seq_row[_IDX_IPV4_TL[0]:_IDX_IPV4_TL[1]], _WEIGHTS16))


def decode_total_len_batch(seq: np.ndarray) -> np.ndarray:
    """Vectorised :func:`decode_total_len` over a ``[L, F]`` flow → ``[L]`` int."""
    bits = np.maximum(_round_tristate(seq[:, _IDX_IPV4_TL[0]:_IDX_IPV4_TL[1]]), 0.0)
    return (bits.astype(np.float64) @ _WEIGHTS16).astype(np.int64)


# ──────────────────────────────────────────────────────────────────────────────
# Packet realization
# ──────────────────────────────────────────────────────────────────────────────

def _canonical_key(pkt) -> str:
    """Reproduce flow_grouper's canonical 5-tuple key as a '|'-joined string."""
    from scapy.all import IP, TCP, UDP

    src_ip, dst_ip = "-1", "-1"
    src_port, dst_port, proto = -1, -1, -1
    if pkt.haslayer(IP):
        ip = pkt[IP]
        proto = int(ip.proto)
        src_ip, dst_ip = ip.src, ip.dst
    if pkt.haslayer(TCP):
        src_port, dst_port = int(pkt[TCP].sport), int(pkt[TCP].dport)
    elif pkt.haslayer(UDP):
        src_port, dst_port = int(pkt[UDP].sport), int(pkt[UDP].dport)
    swap = f"{src_ip}:{src_port}" > f"{dst_ip}:{dst_port}"
    ip_lo, ip_hi = (dst_ip, src_ip) if swap else (src_ip, dst_ip)
    p_lo, p_hi = (dst_port, src_port) if swap else (src_port, dst_port)
    return f"{ip_lo}|{ip_hi}|{p_lo}|{p_hi}|{proto}"


def _apply_size(pkt, target_total: int):
    """Trim/pad application payload so the IPv4 total length equals target."""
    from scapy.all import IP, TCP, UDP, Raw

    if not pkt.haslayer(IP):
        return pkt
    ip = pkt[IP]
    ihl = (int(ip.ihl) * 4) if ip.ihl else 20
    if not (ihl <= target_total <= 65535):
        return pkt
    if pkt.haslayer(TCP):
        l4 = pkt[TCP]
    elif pkt.haslayer(UDP):
        l4 = pkt[UDP]
    else:
        return pkt
    app = bytes(l4.payload)
    header_total = len(bytes(ip)) - len(app)
    target_app = target_total - header_total
    if target_app < 0 or target_app == len(app):
        return pkt
    new_app = (app[:target_app] if target_app < len(app)
               else app + b"\x00" * (target_app - len(app)))
    l4.remove_payload()
    if new_app:
        l4.add_payload(Raw(load=new_app))
    # Drop computed fields so scapy recomputes them on re-serialisation.
    for layer, attr in ((ip, "len"), (ip, "chksum"), (l4, "chksum")):
        if hasattr(layer, attr):
            try:
                delattr(layer, attr)
            except Exception:
                setattr(layer, attr, None)
    if pkt.haslayer(UDP):
        try:
            del pkt[UDP].len
        except Exception:
            pass
    return pkt.__class__(bytes(pkt))


def realize_source(
    adv_pcap: Path,
    flow_recon: dict[str, tuple[np.ndarray, np.ndarray]],
    ipat_mean: float,
    ipat_std: float,
    max_pcap_packets: int = 0,
) -> tuple[list, dict]:
    """Realize purified packets for the flows in ``flow_recon``.

    ``flow_recon`` maps a flow's stringified 5-tuple to a tuple
    ``(total_len[L] int, ipat_z[L] float)`` — the reconstructed IPv4 total
    length and z-scored IPAT per packet (kept compact to bound memory).
    Returns ``(realized_packets, stats)``; packets are sorted by their
    (re-timed) timestamp, ready for ``wrpcap``.

    The source PCAP is streamed in a single pass, retaining only packets whose
    canonical 5-tuple belongs to a reconstructed flow — so resident memory
    scales with the realized flows, not with the (potentially multi-GB) source
    capture. ``max_pcap_packets`` bounds how many packets are *scanned* from the
    file (a time guard); ``<= 0`` means unlimited — scan the whole file, the
    right setting for full per-source coverage.
    """
    from scapy.all import PcapReader

    stats = {"target_flows": len(flow_recon), "realized_flows": 0,
             "miss_len": 0, "unmatched": 0, "packets": 0,
             "pcap_packets_read": 0, "pcap_packets_kept": 0, "capped": False}
    if not Path(adv_pcap).exists():
        stats["error"] = f"pcap not found: {adv_pcap}"
        return [], stats

    # Single streaming pass: scan the PCAP but keep only packets belonging to a
    # reconstructed flow (filter-on-read), so memory is bounded by the realized
    # flows rather than the whole capture.
    target_keys = set(flow_recon)
    rows: list[tuple[float, str, object]] = []
    n_scanned = 0
    try:
        with PcapReader(str(adv_pcap)) as reader:
            for pkt in reader:
                n_scanned += 1
                key = _canonical_key(pkt)
                if key in target_keys:
                    rows.append((float(pkt.time), key, pkt))
                if 0 < max_pcap_packets <= n_scanned:
                    stats["capped"] = True
                    break
    except Exception as e:  # pragma: no cover - defensive
        stats["error"] = f"pcap read failed: {e}"
        return [], stats
    stats["pcap_packets_read"] = n_scanned
    stats["pcap_packets_kept"] = len(rows)

    # Stable sort by timestamp (mergesort-equivalent: Python sort is stable).
    order = sorted(range(len(rows)), key=lambda i: rows[i][0])
    groups: dict[str, list[int]] = {}
    for pos in order:
        groups.setdefault(rows[pos][1], []).append(pos)

    realized: list = []
    for fid, (total_len, ipat_z) in flow_recon.items():
        positions = groups.get(fid)
        if positions is None:
            stats["unmatched"] += 1
            continue
        L = int(len(total_len))
        if len(positions) != L:
            stats["miss_len"] += 1
            continue
        # Re-timed timestamps from the reconstructed (de-normalised) IPAT.
        ipat = np.asarray(ipat_z, dtype=np.float64) * float(ipat_std) + float(ipat_mean)
        ipat = np.clip(ipat, 0.0, None)
        t0 = rows[positions[0]][0]
        new_ts = t0 + np.cumsum(np.concatenate([[0.0], ipat[1:]]))
        for i, pos in enumerate(positions):
            pkt = rows[pos][2]
            try:
                pkt = _apply_size(pkt, int(total_len[i]))
                pkt.time = float(new_ts[i])
            except Exception:
                continue
            realized.append(pkt)
        stats["realized_flows"] += 1
    stats["packets"] = len(realized)
    realized.sort(key=lambda p: float(p.time))
    return realized, stats


def write_pcap(packets: list, out_path: Path) -> bool:
    """Write realized packets to ``out_path`` (best-effort)."""
    if not packets:
        return False
    from scapy.all import wrpcap

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        wrpcap(str(out_path), packets)
        return True
    except Exception as e:  # pragma: no cover - defensive
        log.warning("wrpcap failed for %s: %s", out_path, e)
        return False


def tshark_available() -> bool:
    return shutil.which("tshark") is not None


def tshark_compliance(pcap_path: Path) -> dict:
    """Run tshark over a realized PCAP; count malformed / bad-checksum packets.

    Returns ``{n_packets, n_bad, compliance_rate}`` (or ``error``). Checksum
    validation is enabled explicitly (off by default in tshark).
    """
    pcap_path = Path(pcap_path)
    if not tshark_available():
        return {"error": "tshark not found"}
    if not pcap_path.exists():
        return {"error": f"pcap not found: {pcap_path}"}

    def _count(display_filter: str | None) -> int | None:
        cmd = ["tshark", "-r", str(pcap_path),
               "-o", "ip.check_checksum:TRUE",
               "-o", "tcp.check_checksum:TRUE",
               "-o", "udp.check_checksum:TRUE",
               "-T", "fields", "-e", "frame.number"]
        if display_filter:
            cmd += ["-Y", display_filter]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except Exception as e:  # pragma: no cover - defensive
            log.warning("tshark failed: %s", e)
            return None
        if out.returncode != 0:
            log.warning("tshark returned %d: %s", out.returncode, out.stderr[:200])
            return None
        return sum(1 for line in out.stdout.splitlines() if line.strip())

    bad_filter = ('_ws.malformed || ip.checksum.status=="Bad" || '
                  'tcp.checksum.status=="Bad" || udp.checksum.status=="Bad"')
    n_total = _count(None)
    n_bad = _count(bad_filter)
    if n_total is None or n_bad is None:
        return {"error": "tshark execution failed"}
    return {
        "n_packets": int(n_total),
        "n_bad": int(n_bad),
        "compliance_rate": float((n_total - n_bad) / n_total) if n_total else None,
    }


__all__ = [
    "decode_total_len",
    "decode_total_len_batch",
    "realize_source",
    "write_pcap",
    "tshark_available",
    "tshark_compliance",
]
