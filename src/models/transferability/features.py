"""Flow selection and the per-packet **sidecar** the RF branch needs.

Why a sidecar exists at all
---------------------------
Scoring the RF on a *perturbed* flow means recomputing its 24 prefix statistics
from perturbed per-packet values. Those per-packet values are not on disk. The
tensorizer computes them (``tensor_builder.py`` builds ``rf_meta_values``), feeds
them straight to :func:`~src.feature_extraction.tensor_builder._flow_checkpoint_stats`,
and persists **only the aggregate** — ``rf_checkpoints.npy`` holds prefix
mean/std/min/max at packets {2, 12, 22, …}, roughly a 10:1 reduction. The
aggregate cannot be inverted: four numbers per ten packets, and prefix min/max
are non-invertible in principle.

The sequence tensor does not rescue us either. ``flow_packets.npy`` is
``NPRINT_COLUMNS`` (1088 header bits) + IPAT + Direction; nPrint covers headers
only, so ``Payload_Length`` and ``Payload_Entropy`` — together **45.7% of the
trained RF's decision mass** — are absent at any price.

So one scapy pass per source re-derives the per-packet RF base features and
caches them. It is the only I/O-heavy step in this experiment, and it is
cached, so it runs once.

Byte histograms
---------------
We store the full 256-bin histogram of each packet's IP payload rather than a
compact ``(len, zeros, entropy)`` triple. The triple gives a closed form for
zero-padding only; the histogram makes **both** zero- and random-byte padding
exact for the same ~0.5 KB/packet.

This matters more than it looks. The attacker in this experiment differentiates
the TCN, and the TCN's input contains no payload bytes — so the attacker is
*blind* to what its padding does to ``Payload_Entropy``, the RF's single largest
feature group. The padding-content choice is therefore not something a
TCN-guided gradient can optimise, and we report both bounds instead of silently
picking one. See :func:`entropy_after_pad`.

Exactness check
---------------
:func:`verify_sidecar` recomputes the checkpoint stats at δ=0 and compares them
against the on-disk ``rf_checkpoints.npy``. If the sidecar is faithful they
agree to float32 round-off; if they don't, every RF number downstream is wrong.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...feature_extraction.tensor_builder import (
    _flow_checkpoint_stats,
    checkpoint_positions,
)
from ..common.dataset import load_source_meta

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Layout — must mirror tensor_builder's defaults exactly.
# ──────────────────────────────────────────────────────────────────────────────

# Order matters: it fixes the column order of the emitted 24-dim RF vector, and
# must match `rf_base_features` in tensor_builder.build_tensors (and therefore
# `rf_stat_names` in each source's metadata.json).
RF_BASE_FEATURES: tuple[str, ...] = (
    "IPAT", "Packet_Size", "Payload_Length",
    "Payload_Entropy", "TCP_Flags", "Direction",
)
RF_AGG_FUNCTIONS: tuple[str, ...] = ("mean", "std", "min", "max")

# Sidecar column indices into the [P, 6] float32 feature block.
SC_IPAT = 0
SC_PACKET_SIZE = 1
SC_PAYLOAD_LENGTH = 2
SC_PAYLOAD_ENTROPY = 3
SC_TCP_FLAGS = 4
SC_DIRECTION = 5

HIST_BINS = 256
SIDECAR_SCHEMA = "transferability-sidecar-v1"


@dataclass
class Sidecar:
    """Per-packet RF base features for one source's selected flows.

    Ragged, mirroring the repo's ``ragged-per-flow-v1`` schema: ``feats`` is
    flow-major and ``offsets[i] = (start, length)`` slices flow ``i`` out of it.
    """
    source: str
    flow_ids: np.ndarray            # [N] object   canonical 5-tuple string
    offsets: np.ndarray             # [N, 2] int64 (start, length) into feats
    feats: np.ndarray               # [P, 6] float32, RF_BASE_FEATURES order
    hist: np.ndarray                # [P, 256] uint16, IP-payload byte histogram

    def __len__(self) -> int:
        return int(len(self.flow_ids))

    def flow(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(feats[L, 6], hist[L, 256])`` for flow ``i``."""
        start, length = self.offsets[i]
        sl = slice(int(start), int(start) + int(length))
        return self.feats[sl], self.hist[sl]


# ──────────────────────────────────────────────────────────────────────────────
# Payload entropy under padding
# ──────────────────────────────────────────────────────────────────────────────

def _entropy_from_hist(hist: np.ndarray) -> np.ndarray:
    """Shannon entropy (bits) per row of a ``[L, 256]`` histogram.

    Matches :func:`~src.feature_extraction.pcap_parser._shannon_entropy`:
    ``H = -Σ (c/N) log2(c/N)``, and ``H = 0`` for an empty payload.
    """
    h = np.asarray(hist, dtype=np.float64)
    n = h.sum(axis=1)
    out = np.zeros(len(h), dtype=np.float64)
    nz = n > 0
    if not nz.any():
        return out
    p = h[nz] / n[nz][:, None]
    # 0·log0 = 0; np.where alone would still evaluate log2(0) and warn.
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log2(p, where=p > 0), 0.0)
    out[nz] = -(p * logp).sum(axis=1)
    return out


def entropy_after_pad(
    hist: np.ndarray,
    k: np.ndarray,
    mode: str = "random",
) -> np.ndarray:
    """Payload entropy after appending ``k[i]`` bytes to each packet's payload.

    ``hist`` is ``[L, 256]`` (byte histogram of the original IP payload), ``k``
    is ``[L]`` non-negative integer pad counts. Exact for both modes:

    * ``"zero"``   — ``realize.py``'s overlay pads with ``b"\\x00"``, so all *k*
      bytes land in bin 0. Entropy **falls** toward 0.
    * ``"random"`` — TANTGen's ``apply_TANTGen_to_pcap`` pads with random bytes,
      spreading ``k/256`` per bin in expectation. Entropy **rises** toward 8.

    The two bracket the achievable range, and the attacker cannot tell them
    apart from the TCN's gradient (the TCN sees no payload bytes) — hence both
    are reported rather than one assumed.
    """
    h = np.asarray(hist, dtype=np.float64).copy()
    k = np.asarray(k, dtype=np.float64)
    if mode == "zero":
        h[:, 0] += k
    elif mode == "random":
        h += (k / float(HIST_BINS))[:, None]
    else:
        raise ValueError(f"pad_byte_mode must be 'zero' or 'random', got {mode!r}")
    return _entropy_from_hist(h)


# ──────────────────────────────────────────────────────────────────────────────
# Perturbed RF features
# ──────────────────────────────────────────────────────────────────────────────

def rf_feats_for_delta(
    feats: np.ndarray,
    hist: np.ndarray,
    delta_size: np.ndarray,
    delta_iat: np.ndarray,
    first_checkpoint: int,
    stride: int,
    pad_byte_mode: str = "random",
) -> tuple[np.ndarray, np.ndarray]:
    """RF's ``[n_ckpt, 24]`` prefix stats for one flow under a perturbation.

    ``delta_size`` is per-packet padding in **bytes** (≥ 0) and ``delta_iat``
    per-packet added delay in **seconds** (≥ 0). Both are length ``L``.

    Padding *k* bytes moves three of the six base features by exactly *k*
    (``Packet_Size`` is the frame length, ``Payload_Length`` the bytes after the
    IP header) and moves ``Payload_Entropy`` non-linearly; ``TCP_Flags`` and
    ``Direction`` are invariant. Delay moves ``IPAT`` only.

    Returns ``(stats[n_ckpt, 24], positions[n_ckpt])`` with 1-based positions.
    Aggregation is delegated to the pipeline's own ``_flow_checkpoint_stats`` —
    the same function, not a reimplementation — so δ=0 reproduces the on-disk
    checkpoints bit-for-bit modulo float32 round-off.
    """
    L = int(feats.shape[0])
    pos = checkpoint_positions(L, first_checkpoint, stride)
    if pos.size == 0:
        return np.zeros((0, len(RF_BASE_FEATURES) * len(RF_AGG_FUNCTIONS)),
                        dtype=np.float32), pos

    x = feats.copy()
    x[:, SC_IPAT] = x[:, SC_IPAT] + delta_iat
    x[:, SC_PACKET_SIZE] = x[:, SC_PACKET_SIZE] + delta_size
    x[:, SC_PAYLOAD_LENGTH] = x[:, SC_PAYLOAD_LENGTH] + delta_size
    if np.any(delta_size > 0):
        x[:, SC_PAYLOAD_ENTROPY] = entropy_after_pad(
            hist, delta_size, mode=pad_byte_mode,
        ).astype(np.float32)

    # float32 in, exactly as the tensorizer feeds it (rf_meta_values is float32).
    stats = _flow_checkpoint_stats(
        x[: int(pos[-1])].astype(np.float32, copy=False),
        pos,
        list(RF_AGG_FUNCTIONS),
    )
    return stats, pos


# ──────────────────────────────────────────────────────────────────────────────
# Flow selection
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SelectedFlows:
    """Adversarial test-split flows chosen for the attack, grouped by source."""
    sources: np.ndarray            # [N] object
    local_idx: np.ndarray          # [N] int64  row in that source's flow_offsets
    flow_ids: np.ndarray           # [N] object
    source_folders: dict[str, Path]

    def by_source(self) -> dict[str, np.ndarray]:
        """``source -> row indices into this selection``."""
        return {
            s: np.where(self.sources == s)[0]
            for s in sorted(set(self.sources.tolist()))
        }


def select_flows(
    adversarial_dir: Path,
    flows_per_source: int,
    seed: int,
    test_flow_ids: dict[str, set[str]],
    adversarial_exclude: tuple[str, ...] = (),
    min_len: int = 2,
) -> SelectedFlows:
    """Sample up to ``flows_per_source`` adversarial flows per source.

    Restricted to flows the trained models held out (``test_flow_ids``, derived
    from the seed=42 split the base learners share) so ASR is a generalization
    number rather than a training-set artifact.

    Per-source stratification is not cosmetic: source support spans three orders
    of magnitude (``syn_ddos_adv_*`` ~935k train flows vs ``dns_ddos_adv_*``
    ~4.7k), so a global sample would be almost entirely ``syn_ddos``.
    """
    from ..common.dataset import discover_adversarial_sources

    rng = np.random.default_rng(seed)
    exclude = set(adversarial_exclude)
    names = [s for s in discover_adversarial_sources(Path(adversarial_dir))
             if s not in exclude]
    if not names:
        raise FileNotFoundError(
            f"no adversarial sources under {adversarial_dir} "
            f"(after excluding {sorted(exclude)})")

    src_out: list[np.ndarray] = []
    loc_out: list[np.ndarray] = []
    fid_out: list[np.ndarray] = []
    folders: dict[str, Path] = {}

    for src in names:
        folder = Path(adversarial_dir) / src
        _y, flow_ids, _lstr, local_idx, _names = load_source_meta(folder, src)
        offsets = np.load(folder / "flow_offsets.npy")

        # A source absent from the base learners' test index must be skipped, not
        # silently unfiltered: falling back to "keep everything" would draw the
        # sample from flows the models were TRAINED on and report the resulting
        # ASR as a generalization number.
        keep_ids = test_flow_ids.get(src)
        if keep_ids is None:
            log.warning("%s: not present in the base learners' test split — "
                        "skipped (attacking it would sample training flows)", src)
            continue

        eligible = []
        for row, (fid, li) in enumerate(zip(flow_ids, local_idx)):
            if str(fid) not in keep_ids:
                continue
            if int(offsets[int(li)][1]) < min_len:
                continue
            eligible.append(row)
        if not eligible:
            log.warning("%s: no eligible test flows (min_len=%d)", src, min_len)
            continue

        eligible = np.asarray(eligible, dtype=np.int64)
        if len(eligible) > flows_per_source:
            eligible = rng.choice(eligible, size=flows_per_source, replace=False)
            eligible.sort()

        folders[src] = folder
        src_out.append(np.full(len(eligible), src, dtype=object))
        loc_out.append(local_idx[eligible].astype(np.int64))
        fid_out.append(np.asarray([str(flow_ids[r]) for r in eligible], dtype=object))
        log.info("%s: selected %d flows", src, len(eligible))

    if not src_out:
        raise RuntimeError("no flows selected from any source")

    return SelectedFlows(
        sources=np.concatenate(src_out),
        local_idx=np.concatenate(loc_out),
        flow_ids=np.concatenate(fid_out),
        source_folders=folders,
    )


def test_flow_ids_from_split(
    tcn_dir: Path,
    repo_root: Path,
) -> dict[str, set[str]]:
    """``source -> {flow_id_str}`` for the TCN's held-out test split.

    Read from ``data/models/tcn/dataset_index.npz`` so the selection lands on
    exactly the flows RF/TCN/meta held out (all three share the seed=42
    flow-level split from ``common.dataset.flow_level_split``).
    """
    from ..tcn.evaluate import _load_dataset_index, _reanchor_folder

    idx = _load_dataset_index(Path(tcn_dir) / "dataset_index.npz")
    test_idx = np.asarray(idx["used_test_idx"], dtype=np.int64)
    sources = idx["sources"][test_idx]
    local = idx["local_idx"][test_idx].astype(np.int64)

    out: dict[str, set[str]] = {}
    for src in sorted(set(sources.tolist())):
        folder = _reanchor_folder(idx["source_folders"][src], Path(repo_root))
        _y, fids, _ls, locals_meta, _ = load_source_meta(folder, src)
        lookup = {int(li): str(fid) for li, fid in zip(locals_meta, fids)}
        want = local[sources == src]
        out[str(src)] = {lookup[int(li)] for li in want if int(li) in lookup}
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Sidecar construction
# ──────────────────────────────────────────────────────────────────────────────

def _packet_row(pkt) -> tuple[float, int, int, int, np.ndarray, str, int]:
    """Per-packet scan mirroring ``pcap_parser._scan_raw_packet`` exactly.

    Returns ``(timestamp, framelen, payload_len, tcp_flags, hist, src_ip,
    src_port)``. ``hist`` is the 256-bin byte histogram of the IP payload —
    ``pcap_parser`` derives both ``Payload_Length`` and ``Payload_Entropy`` from
    the very same ``bytes(ip_layer.payload)``, so this reproduces them.
    """
    from scapy.all import ICMP, IP, TCP, UDP  # noqa: F401

    hist = np.zeros(HIST_BINS, dtype=np.uint16)
    payload_len = 0
    tcp_flags = 0
    src_ip, src_port = "-1", -1

    if pkt.haslayer(IP):
        ip_layer = pkt[IP]
        src_ip = ip_layer.src
        payload_bytes = bytes(ip_layer.payload)
        payload_len = len(payload_bytes)
        if payload_len:
            counts = np.bincount(
                np.frombuffer(payload_bytes, dtype=np.uint8), minlength=HIST_BINS,
            )
            hist[:] = counts.astype(np.uint16)
    if pkt.haslayer(TCP):
        src_port = int(pkt[TCP].sport)
        tcp_flags = int(pkt[TCP].flags)
    elif pkt.haslayer(UDP):
        src_port = int(pkt[UDP].sport)

    return (float(pkt.time), len(pkt), payload_len, tcp_flags,
            hist, src_ip, src_port)


def build_sidecar(
    source: str,
    adv_pcap: Path,
    flow_ids: list[str],
    max_length: int | None = 200,
) -> Sidecar:
    """One streaming scapy pass over ``adv_pcap``, keeping only ``flow_ids``.

    Reproduces the tensorizer's flow assembly exactly — canonical 5-tuple
    grouping (:func:`~src.models.reconstructor.realize._canonical_key`) plus a
    stable sort by timestamp — so sidecar row *i* of a flow is the same packet
    as tensor row *i*. :func:`verify_sidecar` checks that claim rather than
    trusting it.

    Filter-on-read: only packets belonging to a selected flow are retained, so
    resident memory scales with the selection (~9k flows), not with the
    multi-GB capture.
    """
    from scapy.all import PcapReader

    target = set(flow_ids)
    rows: list[tuple[float, str, tuple]] = []
    n_scanned = 0
    with PcapReader(str(adv_pcap)) as reader:
        for pkt in reader:
            n_scanned += 1
            key = _canonical_key_str(pkt)
            if key in target:
                rows.append((float(pkt.time), key, _packet_row(pkt)))

    # Stable sort by timestamp, then group — mirrors flow_grouper.group_by_flow
    # (mergesort) and realize.realize_source.
    order = sorted(range(len(rows)), key=lambda i: rows[i][0])
    groups: dict[str, list[int]] = {}
    for p in order:
        groups.setdefault(rows[p][1], []).append(p)

    kept_ids: list[str] = []
    feats_parts: list[np.ndarray] = []
    hist_parts: list[np.ndarray] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0

    for fid in flow_ids:
        positions = groups.get(fid)
        if not positions:
            continue
        if max_length is not None:
            positions = positions[:max_length]
        L = len(positions)

        f = np.zeros((L, len(RF_BASE_FEATURES)), dtype=np.float32)
        h = np.zeros((L, HIST_BINS), dtype=np.uint16)
        init: tuple[str, int] | None = None
        prev_ts: float | None = None
        for i, p in enumerate(positions):
            ts, framelen, payload_len, tcp_flags, hist, src_ip, src_port = rows[p][2]
            f[i, SC_IPAT] = 0.0 if prev_ts is None else (ts - prev_ts)
            prev_ts = ts
            f[i, SC_PACKET_SIZE] = framelen
            f[i, SC_PAYLOAD_LENGTH] = payload_len
            f[i, SC_TCP_FLAGS] = tcp_flags
            if init is None:
                init = (src_ip, src_port)
                f[i, SC_DIRECTION] = 1.0
            else:
                f[i, SC_DIRECTION] = 1.0 if (src_ip, src_port) == init else 0.0
            h[i] = hist
        f[:, SC_PAYLOAD_ENTROPY] = _entropy_from_hist(h).astype(np.float32)

        kept_ids.append(fid)
        feats_parts.append(f)
        hist_parts.append(h)
        offsets.append((cursor, L))
        cursor += L

    if not kept_ids:
        raise RuntimeError(
            f"{source}: none of the {len(flow_ids)} selected flows matched a "
            f"packet in {adv_pcap} ({n_scanned} packets scanned)")

    log.info("%s: sidecar %d/%d flows, %d packets (%d scanned)",
             source, len(kept_ids), len(flow_ids), cursor, n_scanned)
    return Sidecar(
        source=source,
        flow_ids=np.asarray(kept_ids, dtype=object),
        offsets=np.asarray(offsets, dtype=np.int64),
        feats=np.concatenate(feats_parts, axis=0),
        hist=np.concatenate(hist_parts, axis=0),
    )


def _canonical_key_str(pkt) -> str:
    """Canonical 5-tuple string, matching ``flow_ids.npy``'s stringification.

    ``realize._canonical_key`` joins with ``'|'`` and
    ``common.dataset._flow_id_to_str`` joins the 5-tuple object array the same
    way, so the two key spaces coincide.
    """
    from ..reconstructor.realize import _canonical_key

    return _canonical_key(pkt)


def save_sidecar(sc: Sidecar, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema=np.asarray(SIDECAR_SCHEMA),
        source=np.asarray(sc.source),
        flow_ids=sc.flow_ids,
        offsets=sc.offsets,
        feats=sc.feats,
        hist=sc.hist,
    )


def load_sidecar(path: Path) -> Sidecar:
    d = np.load(Path(path), allow_pickle=True)
    schema = str(d["schema"])
    if schema != SIDECAR_SCHEMA:
        raise ValueError(
            f"{path}: sidecar schema {schema!r}, expected {SIDECAR_SCHEMA!r}. "
            f"Delete the cache and rebuild.")
    return Sidecar(
        source=str(d["source"]),
        flow_ids=d["flow_ids"],
        offsets=d["offsets"],
        feats=d["feats"],
        hist=d["hist"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Exactness check
# ──────────────────────────────────────────────────────────────────────────────

def verify_sidecar(
    sc: Sidecar,
    folder: Path,
    first_checkpoint: int,
    stride: int,
    max_flows: int = 50,
    atol: float = 1e-2,
) -> dict:
    """Compare δ=0 sidecar checkpoints against the source's ``rf_checkpoints.npy``.

    This is the experiment's ground-truth gate. The sidecar re-derives, from
    packets, values the pipeline derived independently; if the two agree the
    scapy pass reproduced the tensorizer's flow assembly, per-packet features
    and ordering. If they don't, every RF and stack number downstream is
    measuring the sidecar's bugs rather than the attack.

    Prefix stats at packet *p* depend only on packets 1..*p*, so truncating a
    flow at ``max_length`` leaves the checkpoints at positions ≤ ``max_length``
    unchanged — the comparison is exact on the positions both sides emit.
    """
    folder = Path(folder)
    ckpts = np.load(folder / "rf_checkpoints.npy", mmap_mode="r")
    ck_off = np.load(folder / "rf_checkpoint_offsets.npy")
    _y, fids, _ls, local_idx, _n = load_source_meta(folder, sc.source)
    fid_to_local = {str(f): int(li) for f, li in zip(fids, local_idx)}

    n_checked = 0
    n_rows = 0
    max_abs = 0.0
    mismatches: list[str] = []

    for i in range(min(len(sc), max_flows)):
        fid = str(sc.flow_ids[i])
        li = fid_to_local.get(fid)
        if li is None:
            continue
        feats, hist = sc.flow(i)
        L = feats.shape[0]
        got, pos = rf_feats_for_delta(
            feats, hist,
            np.zeros(L, dtype=np.float64), np.zeros(L, dtype=np.float64),
            first_checkpoint, stride,
        )
        if pos.size == 0:
            continue
        start, n_ck = ck_off[li]
        want = np.asarray(ckpts[int(start):int(start) + int(n_ck)])
        m = min(len(pos), len(want))
        if m == 0:
            continue
        diff = np.abs(got[:m] - want[:m])
        max_abs = max(max_abs, float(diff.max()))
        if diff.max() > atol:
            worst = int(np.unravel_index(diff.argmax(), diff.shape)[1])
            mismatches.append(f"{fid}: col {worst} Δ={diff.max():.4g}")
        n_checked += 1
        n_rows += m

    return {
        "flows_checked": n_checked,
        "checkpoint_rows": n_rows,
        "max_abs_diff": max_abs,
        "atol": atol,
        "ok": bool(n_checked > 0 and max_abs <= atol),
        "mismatches": mismatches[:10],
    }


__all__ = [
    "RF_BASE_FEATURES",
    "RF_AGG_FUNCTIONS",
    "Sidecar",
    "SelectedFlows",
    "entropy_after_pad",
    "rf_feats_for_delta",
    "select_flows",
    "test_flow_ids_from_split",
    "build_sidecar",
    "save_sidecar",
    "load_sidecar",
    "verify_sidecar",
]
