"""Traffic-space realization: map a purified nPrint flow back onto real packets.

nPrint has no payload and the identity fields are masked, so packets cannot
be rebuilt from the tensor. Instead, the purified values are overlaid on the
original adversarial packets:

  * size: the IPv4 total length sets how much payload to trim or pad;
  * iat: the de-normalised IPAT sets the packet timestamp.

Identity and payload are kept; scapy recomputes lengths and checksums.
Packets are grouped and sorted as in :mod:`src.feature_extraction.flow_grouper`,
so tensor rows and packets match 1:1. A flow whose packet count differs from
its tensor length is skipped.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
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


def decode_total_len_batch(seq: np.ndarray) -> np.ndarray:
    """Reconstructed IPv4 total length (bytes) per packet of a ``[L, F]`` flow → ``[L]`` int."""
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


# Application-layer text protocols whose grammar a raw byte resize corrupts.
_L7_TEXT_PORTS = (1900,)  # SSDP (M-SEARCH/NOTIFY over UDP)
_L7_TEXT_PREFIXES = (b"M-SEARCH", b"NOTIFY ", b"HTTP/", b"GET ", b"POST ",
                     b"HEAD ", b"OPTIONS ", b"PUT ", b"DELETE ")


def _is_l7_text(l4, app: bytes) -> bool:
    """True if the payload is an application-layer text protocol (SSDP/HTTP).

    Resizing these (e.g. ``M-SEARCH * HTTP/1.1``) breaks their grammar and
    tshark reports them as malformed. Detected by SSDP port or an HTTP-style
    start line.
    """
    try:
        ports = (int(getattr(l4, "dport", -1)), int(getattr(l4, "sport", -1)))
    except Exception:
        ports = ()
    if any(p in _L7_TEXT_PORTS for p in ports):
        return True
    return app[:10].upper().startswith(_L7_TEXT_PREFIXES)


# Minimum Ethernet frame (802.3), excluding FCS. Frames shorter than this are
# zero-padded by the sender; that trailer is framing, not datagram.
_ETH_MIN_FRAME = 60


def _split_payload(pkt, l4) -> tuple[bytes, bytes]:
    """Split ``l4``'s payload into ``(application bytes, Ethernet trailer)``.

    ``bytes(l4.payload)`` includes the Ethernet padding (up to the 60-byte
    minimum frame), which is not part of the IP datagram. Counting it as
    payload would trim real bytes on a resize that should be a no-op.
    """
    from scapy.all import Padding

    app = bytes(l4.payload)
    pad_layer = pkt.getlayer(Padding)
    if pad_layer is None:
        return app, b""
    pad = bytes(pad_layer)
    # Padding is always trailing; guard the slice anyway rather than trust it.
    if pad and app.endswith(pad):
        return app[: len(app) - len(pad)], pad
    return app, b""


def _apply_size(pkt, target_total: int):
    """Trim/pad application payload so the IPv4 total length equals target.

    Returns ``(pkt, edited, unrealizable)``:

    * ``edited``        — the payload length actually changed.
    * ``unrealizable``  — an edit was needed but the payload is an
      application-layer text protocol (:func:`_is_l7_text`); the packet is
      dropped instead of being counted as a purifier failure.
    """
    from scapy.all import IP, TCP, UDP, Padding, Raw

    if not pkt.haslayer(IP):
        return pkt, False, False
    ip = pkt[IP]
    ihl = (int(ip.ihl) * 4) if ip.ihl else 20
    if not (ihl <= target_total <= 65535):
        return pkt, False, False
    if pkt.haslayer(TCP):
        l4 = pkt[TCP]
    elif pkt.haslayer(UDP):
        l4 = pkt[UDP]
    else:
        return pkt, False, False

    app, _pad = _split_payload(pkt, l4)
    # Use ip.len, not len(bytes(ip)), which includes the Ethernet trailer.
    header_total = int(ip.len) - len(app)
    target_app = target_total - header_total
    if target_app < 0 or target_app == len(app):
        return pkt, False, False
    if _is_l7_text(l4, app):
        return pkt, True, True
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
    out = pkt.__class__(bytes(pkt))
    # Re-pad to Ethernet's 60-byte minimum if the new frame is shorter.
    short = _ETH_MIN_FRAME - len(bytes(out))
    if short > 0:
        out = out / Padding(load=b"\x00" * short)
        out = out.__class__(bytes(out))
    return out, True, False


def realize_source(
    adv_pcap: Path,
    flow_recon: dict[str, tuple[np.ndarray, np.ndarray]],
    ipat_mean: float,
    ipat_std: float,
) -> tuple[list, dict]:
    """Realize purified packets for the flows in ``flow_recon``.

    ``flow_recon`` maps a flow's stringified 5-tuple to a tuple
    ``(total_len[L] int, ipat_z[L] float)`` — the reconstructed IPv4 total
    length and z-scored IPAT per packet (kept compact to bound memory).
    Returns ``(realized_packets, stats)``; packets are sorted by their
    (re-timed) timestamp, ready for ``wrpcap``.

    The PCAP is read in one pass, keeping only packets of reconstructed
    flows.
    """
    from scapy.all import PcapReader

    stats = {"target_flows": len(flow_recon), "realized_flows": 0,
             "miss_len": 0, "unmatched": 0, "packets": 0,
             "pcap_packets_read": 0, "pcap_packets_kept": 0,
             "unrealizable_l7": 0}
    if not Path(adv_pcap).exists():
        stats["error"] = f"pcap not found: {adv_pcap}"
        return [], stats

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
                pkt, edited, unrealizable = _apply_size(pkt, int(total_len[i]))
                # Size edit on SSDP/HTTP text: drop it and count it separately.
                if edited and unrealizable:
                    stats["unrealizable_l7"] += 1
                    continue
                pkt.time = float(new_ts[i])
            except Exception:
                continue
            realized.append(pkt)
        stats["realized_flows"] += 1
    stats["packets"] = len(realized)
    realized.sort(key=lambda p: float(p.time))
    return realized, stats


def realize_triplet(
    adv_pcap: Path,
    flow_vectors: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    ipat_mean: float,
    ipat_std: float,
) -> tuple[dict[str, dict[str, list]], dict]:
    """Realize three aligned packet sets (adv / purified / clean) for the Kitsune metric.

    ``flow_vectors`` maps a flow's stringified 5-tuple to a dict with keys ``"adv"``,
    ``"recon"`` and ``"clean"`` — each a tuple ``(total_len[L] int, ipat_z[L] float)``
    decoded from the adversarial / reconstructed / clean-target tensor of that flow.

    All three overlays are applied to the same adversarial packets and share one
    keep-mask: a packet is dropped from all three if any of the three size
    targets would break an application-layer text protocol
    (:func:`_is_l7_text`). The three branches then differ only in size and
    timing.

    Returns ``({branch: {"packets": [...], "keys": [...]}}, stats)`` — each branch's
    packets are sorted by their own re-timed timestamps and ``keys[i]`` is the canonical
    5-tuple of ``packets[i]`` (so a caller can aggregate per-flow after scoring).
    """
    from scapy.all import PcapReader

    branches = ("adv", "purified", "clean")
    out: dict[str, dict[str, list]] = {b: {"packets": [], "keys": []} for b in branches}
    stats = {"target_flows": len(flow_vectors), "realized_flows": 0,
             "miss_len": 0, "unmatched": 0, "unrealizable_l7": 0,
             "pcap_packets_read": 0, "pcap_packets_kept": 0, "packets_per_branch": 0}
    if not Path(adv_pcap).exists():
        stats["error"] = f"pcap not found: {adv_pcap}"
        return out, stats

    target_keys = set(flow_vectors)
    rows: list[tuple[float, str, object]] = []
    n_scanned = 0
    try:
        with PcapReader(str(adv_pcap)) as reader:
            for pkt in reader:
                n_scanned += 1
                key = _canonical_key(pkt)
                if key in target_keys:
                    rows.append((float(pkt.time), key, pkt))
    except Exception as e:  # pragma: no cover - defensive
        stats["error"] = f"pcap read failed: {e}"
        return out, stats
    stats["pcap_packets_read"] = n_scanned
    stats["pcap_packets_kept"] = len(rows)

    order = sorted(range(len(rows)), key=lambda i: rows[i][0])
    groups: dict[str, list[int]] = {}
    for pos in order:
        groups.setdefault(rows[pos][1], []).append(pos)

    def _retime(ipat_z, t0):
        ipat = np.asarray(ipat_z, dtype=np.float64) * float(ipat_std) + float(ipat_mean)
        ipat = np.clip(ipat, 0.0, None)
        return t0 + np.cumsum(np.concatenate([[0.0], ipat[1:]]))

    acc: dict[str, list[tuple[float, str, object]]] = {b: [] for b in branches}
    for fid, vecs in flow_vectors.items():
        positions = groups.get(fid)
        if positions is None:
            stats["unmatched"] += 1
            continue
        (tl_a, ip_a) = vecs["adv"]
        (tl_r, ip_r) = vecs["recon"]
        (tl_c, ip_c) = vecs["clean"]
        L = int(len(tl_r))
        if not (len(positions) == L == len(tl_a) == len(tl_c)):
            stats["miss_len"] += 1
            continue
        t0 = rows[positions[0]][0]
        ts = {"adv": _retime(ip_a, t0), "purified": _retime(ip_r, t0),
              "clean": _retime(ip_c, t0)}
        flow_pkts: dict[str, list] = {b: [] for b in branches}
        for i, pos in enumerate(positions):
            base = rows[pos][2]
            # Overlay each size target on an independent copy (_apply_size mutates).
            p_adv, _, un_a = _apply_size(base.copy(), int(tl_a[i]))
            p_pur, _, un_r = _apply_size(base.copy(), int(tl_r[i]))
            p_cln, _, un_c = _apply_size(base.copy(), int(tl_c[i]))
            if un_a or un_r or un_c:      # shared keep-mask: exclude from all three
                stats["unrealizable_l7"] += 1
                continue
            for b, pkt in (("adv", p_adv), ("purified", p_pur), ("clean", p_cln)):
                pkt.time = float(ts[b][i])
                flow_pkts[b].append((pkt.time, fid, pkt))
        for b in branches:
            acc[b].extend(flow_pkts[b])
        stats["realized_flows"] += 1

    for b in branches:
        acc[b].sort(key=lambda r: r[0])
        out[b]["packets"] = [r[2] for r in acc[b]]
        out[b]["keys"] = [r[1] for r in acc[b]]
    stats["packets_per_branch"] = len(out["adv"]["packets"])
    return out, stats


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
    "decode_total_len_batch",
    "realize_source",
    "realize_triplet",
    "write_pcap",
    "tshark_available",
    "tshark_compliance",
]
