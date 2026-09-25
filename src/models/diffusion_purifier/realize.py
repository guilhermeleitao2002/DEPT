"""Traffic-space realization for any number of overlays, one branch at a time.

Phase 8's :func:`~src.models.reconstructor.realize.realize_triplet` realizes three
overlays (adv / purified / clean) onto the same adversarial packets with one shared
keep-mask. Phase 9 needs more branches (adversarial, clean, DEPT, diffusion, and the
two gated mixtures), and holding every realized copy of a large stream at once would
multiply memory. :class:`BranchRealizer` keeps phase 8's semantics — the same packets,
timestamps re-derived per branch from its IPAT, sizes overlaid by
:func:`~src.models.reconstructor.realize._apply_size`, and one keep-mask shared by all
branches — but computes the keep-mask up front and then builds each branch on demand,
so only the original packets plus one realized branch are in memory at a time.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..reconstructor.realize import (
    _apply_size,
    _canonical_key,
    _is_l7_text,
    _split_payload,
)

log = logging.getLogger(__name__)


def _size_probe(pkt):
    """What :func:`_apply_size` needs to decide realizability, computed once per packet.

    Returns ``None`` when no size edit can ever be unrealizable for this packet (not
    IPv4, or no TCP/UDP), else ``(ihl, header_total, app_len, is_l7_text)``.
    """
    from scapy.all import IP, TCP, UDP

    if not pkt.haslayer(IP):
        return None
    ip = pkt[IP]
    if pkt.haslayer(TCP):
        l4 = pkt[TCP]
    elif pkt.haslayer(UDP):
        l4 = pkt[UDP]
    else:
        return None
    app, _pad = _split_payload(pkt, l4)
    ihl = (int(ip.ihl) * 4) if ip.ihl else 20
    return ihl, int(ip.len) - len(app), len(app), _is_l7_text(l4, app)


def _unrealizable(probe, target_total: int) -> bool:
    """Mirror of :func:`_apply_size`'s ``unrealizable`` flag, without rebuilding."""
    if probe is None:
        return False
    ihl, header_total, app_len, is_text = probe
    if not (ihl <= target_total <= 65535):
        return False
    target_app = target_total - header_total
    if target_app < 0 or target_app == app_len:
        return False
    return bool(is_text)


class BranchRealizer:
    """Realize ``flow_vectors[fid][branch] = (total_len[L], ipat_z[L])`` onto a PCAP.

    ``mask_branches`` are the overlays whose size edits decide the shared keep-mask
    (a packet is dropped from every branch if resizing it under any of them would
    corrupt an application-layer text protocol). Other branches may be realized too,
    as long as their overlays only mix values of the mask branches (the gated streams).
    """

    def __init__(self, adv_pcap: Path, flow_vectors: dict, ipat_mean: float,
                 ipat_std: float, mask_branches: Sequence[str]) -> None:
        from scapy.all import PcapReader

        self._fv = flow_vectors
        self._mean = float(ipat_mean)
        self._std = float(ipat_std)
        self.stats = {"target_flows": len(flow_vectors), "realized_flows": 0,
                      "miss_len": 0, "unmatched": 0, "unrealizable_l7": 0,
                      "pcap_packets_read": 0, "pcap_packets_kept": 0,
                      "packets_per_branch": 0}
        self._flows: dict[str, tuple] = {}      # fid -> (t0, [packets], keep[bool])
        if not Path(adv_pcap).exists():
            self.stats["error"] = f"pcap not found: {adv_pcap}"
            return

        target = set(flow_vectors)
        rows: list[tuple[float, str, object]] = []
        n = 0
        try:
            with PcapReader(str(adv_pcap)) as reader:
                for pkt in reader:
                    n += 1
                    key = _canonical_key(pkt)
                    if key in target:
                        rows.append((float(pkt.time), key, pkt))
        except Exception as e:  # pragma: no cover - defensive
            self.stats["error"] = f"pcap read failed: {e}"
            return
        self.stats["pcap_packets_read"] = n
        self.stats["pcap_packets_kept"] = len(rows)

        # Same grouping as realize_triplet: stable time order within each 5-tuple.
        order = sorted(range(len(rows)), key=lambda i: rows[i][0])
        groups: dict[str, list[int]] = {}
        for pos in order:
            groups.setdefault(rows[pos][1], []).append(pos)

        for fid, vecs in flow_vectors.items():
            positions = groups.get(fid)
            if positions is None:
                self.stats["unmatched"] += 1
                continue
            L = len(vecs["adv"][0])
            if not all(len(vecs[b][0]) == L for b in mask_branches) or len(positions) != L:
                self.stats["miss_len"] += 1
                continue
            pkts = [rows[p][2] for p in positions]
            keep = np.ones(L, dtype=bool)
            for i, pkt in enumerate(pkts):
                probe = _size_probe(pkt)
                if any(_unrealizable(probe, int(vecs[b][0][i])) for b in mask_branches):
                    keep[i] = False
            self.stats["unrealizable_l7"] += int((~keep).sum())
            self._flows[fid] = (rows[positions[0]][0], pkts, keep)
            self.stats["realized_flows"] += 1
        self.stats["packets_per_branch"] = int(sum(int(k.sum())
                                                   for _t, _p, k in self._flows.values()))

    @property
    def realized_fids(self) -> set[str]:
        return set(self._flows)

    def _retime(self, ipat_z, t0: float) -> np.ndarray:
        ipat = np.asarray(ipat_z, dtype=np.float64) * self._std + self._mean
        ipat = np.clip(ipat, 0.0, None)
        return t0 + np.cumsum(np.concatenate([[0.0], ipat[1:]]))

    def packets(self, branch: str, flows: Iterable[str] | None = None
                ) -> tuple[list, list]:
        """The realized stream of ``branch`` (optionally only ``flows``), time-sorted."""
        wanted = self._flows if flows is None else {
            f: self._flows[f] for f in flows if f in self._flows}
        acc: list[tuple[float, str, object]] = []
        for fid, (t0, pkts, keep) in wanted.items():
            tl, ipat_z = self._fv[fid][branch]
            ts = self._retime(ipat_z, t0)
            for i, base in enumerate(pkts):
                if not keep[i]:
                    continue
                p, _edited, _un = _apply_size(base.copy(), int(tl[i]))
                p.time = float(ts[i])
                acc.append((p.time, fid, p))
        acc.sort(key=lambda r: r[0])
        return [r[2] for r in acc], [r[1] for r in acc]


__all__ = ["BranchRealizer"]
