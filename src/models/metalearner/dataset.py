"""Token-stream dataset for the Mini-Transformer meta-learner.

For each flow we build a per-packet ``[L, 13]`` token matrix combining:

  * The two base-learner predictions (carry-forward RF logit + dense TCN
    logit), with a recency feature ``rf_age`` and a binary
    ``is_rf_checkpoint`` flag.
  * A 9-dim packet summary (IPAT, direction, normalised packet length,
    six TCP flag bits) instead of the full nPrint vector.

Token layout (column index → meaning):

    0  p_RF_logit          (NaN before the first RF checkpoint emits)
    1  p_TCN_logit
    2  rf_age              raw packets since last RF emission, clipped at 30
    3  is_rf_checkpoint    1 iff a fresh RF prediction emits at this packet
    4  ipat                already z-scored upstream
    5  direction
    6  pkt_len_norm        IPv4 total length / 1500
    7  urg                 \\
    8  ack                  \\
    9  psh                   \\  TCP flag bits, clamped {-1, 0, 1} → {0, 0, 1}
   10  rst                   /
   11  syn                  /
   12  fin                 /

The dataset materialises tokens **once** per (rf, tcn, source) triple
and caches them under ``data/models/metalearner/cache/<split>/``. The
cache is invalidated by a key composed of the base-learner checkpoint
file mtimes + sizes; mismatch triggers a rebuild.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .base_inference import JoinedFlow, JoinedSplit, join_streams


# ──────────────────────────────────────────────────────────────────────────────
# Token layout — must match the model's expectations.
# ──────────────────────────────────────────────────────────────────────────────

TOKEN_DIM = 13
SLOT_RF_LOGIT = 0
SLOT_TCN_LOGIT = 1
SLOT_RF_AGE = 2
SLOT_IS_RF_CKPT = 3
SLOT_IPAT = 4
SLOT_DIRECTION = 5
SLOT_PKT_LEN = 6
SLOT_TCP_URG = 7
SLOT_TCP_ACK = 8
SLOT_TCP_PSH = 9
SLOT_TCP_RST = 10
SLOT_TCP_SYN = 11
SLOT_TCP_FIN = 12

# nPrint absolute bit positions (verified at import — see _verify_nprint_layout).
_IDX_IPV4_TL = (16, 32)         # half-open slice [16, 32) → 16 bits MSB-first
_IDX_IPAT = 1088
_IDX_DIRECTION = 1089
_IDX_TCP_URG = 586
_IDX_TCP_ACK = 587
_IDX_TCP_PSH = 588
_IDX_TCP_RST = 589
_IDX_TCP_SYN = 590
_IDX_TCP_FIN = 591

DEFAULT_RF_AGE_CLIP = 30
DEFAULT_PKT_LEN_NORM = 1500.0
CACHE_SCHEMA_VERSION = "metalearner-tokens-v1"


def _verify_nprint_layout() -> None:
    """Fail fast at import time if nPrint column indices have shifted."""
    from ...feature_extraction.pcap_parser import _build_column_names

    cols = _build_column_names()
    expected = {
        _IDX_TCP_URG: "tcp_urg_0",
        _IDX_TCP_ACK: "tcp_ackf_0",
        _IDX_TCP_PSH: "tcp_psh_0",
        _IDX_TCP_RST: "tcp_rst_0",
        _IDX_TCP_SYN: "tcp_syn_0",
        _IDX_TCP_FIN: "tcp_fin_0",
        16: "ipv4_tl_0",
        31: "ipv4_tl_15",
    }
    for idx, want in expected.items():
        got = cols[idx]
        if got != want:
            raise RuntimeError(
                f"nPrint column index drift: position {idx} is {got!r}, "
                f"expected {want!r}. Update _IDX_* constants in "
                f"src/models/metalearner/dataset.py."
            )


_verify_nprint_layout()


# ──────────────────────────────────────────────────────────────────────────────
# Per-flow token builder
# ──────────────────────────────────────────────────────────────────────────────

def _decode_pkt_len(bits_2d: np.ndarray) -> np.ndarray:
    """Decode IPv4 total length from nPrint bit columns.

    ``bits_2d`` is ``[L, 16]`` float32 with values in ``{-1, 0, 1}`` —
    nPrint convention. ``-1`` means "field not present" (non-IPv4
    packet); we clamp those to ``0`` so the resulting length is 0 for
    such packets. nPrint stores bits MSB-first.
    """
    cleaned = np.maximum(bits_2d, 0.0)
    weights = (1 << np.arange(15, -1, -1)).astype(np.float32)
    return cleaned @ weights  # [L]


def _packet_features(
    flow_packets: np.ndarray,
    flow_start: int,
    flow_length: int,
    pkt_len_norm: float = DEFAULT_PKT_LEN_NORM,
) -> dict[str, np.ndarray]:
    """Slice the per-packet feature subset the meta-learner needs.

    ``flow_packets`` is the full ``[P_total, 1090]`` per-source memmap.
    Returns a dict of 1D arrays of length ``flow_length`` ready to be
    written into token columns.
    """
    sl = slice(flow_start, flow_start + flow_length)
    block = flow_packets[sl]
    return {
        "ipat": block[:, _IDX_IPAT].astype(np.float32, copy=False),
        "direction": block[:, _IDX_DIRECTION].astype(np.float32, copy=False),
        "pkt_len_norm": (
            _decode_pkt_len(block[:, _IDX_IPV4_TL[0]:_IDX_IPV4_TL[1]])
            / np.float32(pkt_len_norm)
        ).astype(np.float32, copy=False),
        "urg": np.maximum(block[:, _IDX_TCP_URG], 0.0).astype(np.float32, copy=False),
        "ack": np.maximum(block[:, _IDX_TCP_ACK], 0.0).astype(np.float32, copy=False),
        "psh": np.maximum(block[:, _IDX_TCP_PSH], 0.0).astype(np.float32, copy=False),
        "rst": np.maximum(block[:, _IDX_TCP_RST], 0.0).astype(np.float32, copy=False),
        "syn": np.maximum(block[:, _IDX_TCP_SYN], 0.0).astype(np.float32, copy=False),
        "fin": np.maximum(block[:, _IDX_TCP_FIN], 0.0).astype(np.float32, copy=False),
    }


def build_flow_tokens(
    flow: JoinedFlow,
    flow_packets: np.ndarray,
    flow_start: int,
    rf_age_clip: int = DEFAULT_RF_AGE_CLIP,
    pkt_len_norm: float = DEFAULT_PKT_LEN_NORM,
) -> np.ndarray:
    """Build the ``[L, 13]`` token matrix for a single joined flow.

    The RF logit is carried forward between checkpoints. Packets before
    the first RF checkpoint hold ``NaN`` in the RF slot; the model
    substitutes its learnable ``rf_unknown_logit`` parameter there.
    """
    L = flow.length
    tokens = np.empty((L, TOKEN_DIM), dtype=np.float32)

    # Slot 0/2/3: RF logit carry-forward + age + checkpoint flag.
    tokens[:, SLOT_RF_LOGIT] = np.nan
    tokens[:, SLOT_RF_AGE] = float(rf_age_clip)  # max age before first ckpt
    tokens[:, SLOT_IS_RF_CKPT] = 0.0
    if flow.rf_packet_idx.size:
        # Walk through rf checkpoints; carry forward into [pkt, next_pkt).
        rf_pkts = flow.rf_packet_idx
        rf_logits = flow.rf_logit.astype(np.float32, copy=False)
        for k, p in enumerate(rf_pkts):
            start = int(p) - 1  # 0-based
            end = int(rf_pkts[k + 1]) - 1 if k + 1 < len(rf_pkts) else L
            if start >= L:
                break
            end = min(end, L)
            tokens[start:end, SLOT_RF_LOGIT] = rf_logits[k]
            tokens[start, SLOT_IS_RF_CKPT] = 1.0
        # rf_age: raw packets since last RF emission, clipped. The
        # pre-first-checkpoint window keeps the clip-max sentinel; from each
        # checkpoint onward age increments after every packet.
        ages = np.full(L, rf_age_clip, dtype=np.float32)
        for k, p in enumerate(rf_pkts):
            start = int(p) - 1
            end = int(rf_pkts[k + 1]) - 1 if k + 1 < len(rf_pkts) else L
            if start >= L:
                break
            end = min(end, L)
            local = np.arange(end - start, dtype=np.float32)
            ages[start:end] = np.minimum(local, float(rf_age_clip))
        tokens[:, SLOT_RF_AGE] = ages

    # Slot 1: TCN logit, dense per packet.
    tcn_logit = flow.tcn_logit.astype(np.float32, copy=False)
    if tcn_logit.shape[0] != L:
        raise ValueError(
            f"flow length mismatch: TCN logit has {tcn_logit.shape[0]} rows, "
            f"expected {L}"
        )
    tokens[:, SLOT_TCN_LOGIT] = tcn_logit

    # Slots 4..12: packet-feature subset.
    feats = _packet_features(flow_packets, flow_start, L, pkt_len_norm=pkt_len_norm)
    tokens[:, SLOT_IPAT] = feats["ipat"]
    tokens[:, SLOT_DIRECTION] = feats["direction"]
    tokens[:, SLOT_PKT_LEN] = feats["pkt_len_norm"]
    tokens[:, SLOT_TCP_URG] = feats["urg"]
    tokens[:, SLOT_TCP_ACK] = feats["ack"]
    tokens[:, SLOT_TCP_PSH] = feats["psh"]
    tokens[:, SLOT_TCP_RST] = feats["rst"]
    tokens[:, SLOT_TCP_SYN] = feats["syn"]
    tokens[:, SLOT_TCP_FIN] = feats["fin"]

    return tokens


# ──────────────────────────────────────────────────────────────────────────────
# On-disk cache
# ──────────────────────────────────────────────────────────────────────────────

def _cache_key(rf_dir: Path, tcn_dir: Path) -> str:
    """Deterministic key from base-learner checkpoint mtime+size."""
    parts = [CACHE_SCHEMA_VERSION]
    for f in (rf_dir / "rf.joblib", tcn_dir / "tcn.pt"):
        st = f.stat()
        parts.append(f"{f.name}:{int(st.st_mtime)}:{st.st_size}")
    return "|".join(parts)


@dataclass
class TokenCache:
    """Memmapped per-split token store."""
    cache_dir: Path
    tokens: np.ndarray            # [P_total, 13] float32
    offsets: np.ndarray           # [N, 2]        int64  (start, length)
    labels: np.ndarray            # [N]           int8
    sources: np.ndarray           # [N]           object (str)
    lengths: np.ndarray           # [N]           int64

    def __len__(self) -> int:
        return int(self.offsets.shape[0])


def build_token_cache(
    output_dir: Path,
    split_name: str,
    joined: JoinedSplit,
    tensors_dir: Path,
    adversarial_dir: Path,
    rf_age_clip: int = DEFAULT_RF_AGE_CLIP,
    pkt_len_norm: float = DEFAULT_PKT_LEN_NORM,
    max_length: int | None = None,
) -> TokenCache:
    """Materialise tokens for one split and persist them under ``output_dir``.

    ``output_dir`` is the meta-learner output dir (e.g. ``data/models/
    metalearner/cache``). One sub-directory per split.
    """
    cache_dir = output_dir / split_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    n_flows = len(joined.flows)
    flow_offsets = np.zeros((n_flows, 2), dtype=np.int64)
    pkt_cursor = 0
    for i, fl in enumerate(joined.flows):
        L = min(fl.length, max_length) if max_length is not None else fl.length
        flow_offsets[i, 0] = pkt_cursor
        flow_offsets[i, 1] = L
        pkt_cursor += L
    total_packets = int(pkt_cursor)

    tokens = np.lib.format.open_memmap(
        cache_dir / "tokens.npy", mode="w+",
        dtype=np.float32, shape=(total_packets, TOKEN_DIM),
    )

    # Cache per-source flow_packets memmaps so we open each file once.
    pkt_memmaps: dict[str, np.ndarray] = {}
    pkt_offsets_per_src: dict[str, np.ndarray] = {}

    def _resolve_source(src: str) -> tuple[np.ndarray, np.ndarray]:
        if src in pkt_memmaps:
            return pkt_memmaps[src], pkt_offsets_per_src[src]
        # Source folder lives under tensors_dir/<src>/ for benign sources
        # and under adversarial_dir/<src>/ for adversarial sources.
        candidate = tensors_dir / src
        if not candidate.exists():
            candidate = adversarial_dir / src
        if not candidate.exists():
            raise FileNotFoundError(
                f"could not locate source folder for {src!r} under "
                f"{tensors_dir} or {adversarial_dir}"
            )
        fp = np.load(candidate / "flow_packets.npy", mmap_mode="r")
        fo = np.load(candidate / "flow_offsets.npy")
        pkt_memmaps[src] = fp
        pkt_offsets_per_src[src] = fo
        return fp, fo

    # Map flow_id_str -> per-source local_idx via flow_ids.npy on demand.
    src_to_flow_lookup: dict[str, dict[str, int]] = {}

    def _local_idx(src: str, flow_id_str: str) -> int:
        if src not in src_to_flow_lookup:
            from ..common.dataset import _flow_id_to_str
            candidate = tensors_dir / src
            if not candidate.exists():
                candidate = adversarial_dir / src
            raw = np.load(candidate / "flow_ids.npy", allow_pickle=True)
            src_to_flow_lookup[src] = {
                _flow_id_to_str(raw[i]): i for i in range(len(raw))
            }
        return src_to_flow_lookup[src][flow_id_str]

    labels = np.zeros(n_flows, dtype=np.int8)
    sources_arr = np.empty(n_flows, dtype=object)
    lengths_arr = np.zeros(n_flows, dtype=np.int64)
    for i, fl in enumerate(joined.flows):
        fp, fo = _resolve_source(fl.source)
        li = _local_idx(fl.source, fl.flow_id)
        flow_start_in_src = int(fo[li, 0])
        L = int(flow_offsets[i, 1])
        # Build over the truncated length (so packet features stay aligned
        # with the truncated TCN/RF arrays).
        if L < fl.length:
            tcn_logit_t = fl.tcn_logit[:L]
            rf_keep = fl.rf_packet_idx <= L
            fl_eff = JoinedFlow(
                source=fl.source, flow_id=fl.flow_id, label=fl.label, length=L,
                tcn_packet_idx=fl.tcn_packet_idx[:L], tcn_logit=tcn_logit_t,
                rf_packet_idx=fl.rf_packet_idx[rf_keep],
                rf_logit=fl.rf_logit[rf_keep],
            )
        else:
            fl_eff = fl
        toks = build_flow_tokens(
            fl_eff, fp, flow_start_in_src,
            rf_age_clip=rf_age_clip, pkt_len_norm=pkt_len_norm,
        )
        start = int(flow_offsets[i, 0])
        tokens[start:start + L] = toks
        labels[i] = fl.label
        sources_arr[i] = fl.source
        lengths_arr[i] = L

    tokens.flush()
    np.save(cache_dir / "offsets.npy", flow_offsets)
    np.save(cache_dir / "labels.npy", labels)
    np.save(cache_dir / "sources.npy", sources_arr, allow_pickle=True)
    np.save(cache_dir / "lengths.npy", lengths_arr)
    (cache_dir / "flow_keys.json").write_text(
        json.dumps([(f.source, f.flow_id) for f in joined.flows])
    )

    return TokenCache(
        cache_dir=cache_dir,
        tokens=np.load(cache_dir / "tokens.npy", mmap_mode="r"),
        offsets=flow_offsets,
        labels=labels,
        sources=sources_arr,
        lengths=lengths_arr,
    )


def load_token_cache(cache_dir: Path) -> TokenCache:
    return TokenCache(
        cache_dir=cache_dir,
        tokens=np.load(cache_dir / "tokens.npy", mmap_mode="r"),
        offsets=np.load(cache_dir / "offsets.npy"),
        labels=np.load(cache_dir / "labels.npy"),
        sources=np.load(cache_dir / "sources.npy", allow_pickle=True),
        lengths=np.load(cache_dir / "lengths.npy"),
    )


def prepare_split_caches(
    output_dir: Path,
    rf_dir: Path,
    tcn_dir: Path,
    tensors_dir: Path,
    adversarial_dir: Path,
    splits: dict[str, dict],
    rf_age_clip: int = DEFAULT_RF_AGE_CLIP,
    pkt_len_norm: float = DEFAULT_PKT_LEN_NORM,
    max_length: int | None = None,
) -> dict[str, TokenCache]:
    """Build (or reuse) the per-split token caches.

    ``splits`` maps a split name (e.g. ``"val"``) to a dict with keys
    ``rf_sample_indices`` and ``tcn_sample_indices`` (either may be
    ``None`` to use the respective default test split). Re-uses existing
    caches when the cache key matches; otherwise every existing split
    cache is dropped (not only the requested ones — a stale sibling must
    not survive the key bump) and the requested splits are rebuilt.
    """
    cache_root = output_dir / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    key = _cache_key(rf_dir, tcn_dir)
    key_path = cache_root / "cache_key.txt"
    stale = not key_path.exists() or key_path.read_text().strip() != key
    if stale:
        import shutil
        for d in cache_root.iterdir():
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)

    out: dict[str, TokenCache] = {}
    for split_name, args in splits.items():
        split_dir = cache_root / split_name
        ready = (
            split_dir.exists()
            and (split_dir / "tokens.npy").exists()
            and (split_dir / "offsets.npy").exists()
        )
        if ready and not stale:
            out[split_name] = load_token_cache(split_dir)
            continue
        joined = join_streams(
            rf_dir=rf_dir, tcn_dir=tcn_dir,
            rf_sample_indices=args.get("rf_sample_indices"),
            tcn_sample_indices=args.get("tcn_sample_indices"),
        )
        out[split_name] = build_token_cache(
            output_dir=cache_root,
            split_name=split_name,
            joined=joined,
            tensors_dir=tensors_dir,
            adversarial_dir=adversarial_dir,
            rf_age_clip=rf_age_clip,
            pkt_len_norm=pkt_len_norm,
            max_length=max_length,
        )
    key_path.write_text(key)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset + collation
# ──────────────────────────────────────────────────────────────────────────────

class MetaLearnerDataset(Dataset):
    """Per-flow token slices into a memmapped cache."""

    def __init__(
        self,
        cache: TokenCache,
        sample_indices: Iterable[int] | None = None,
    ) -> None:
        self.cache = cache
        if sample_indices is None:
            self.sample_indices = np.arange(len(cache), dtype=np.int64)
        else:
            self.sample_indices = np.asarray(list(sample_indices), dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.sample_indices))

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, int]:
        idx = int(self.sample_indices[i])
        start, length = (int(x) for x in self.cache.offsets[idx])
        # Copy the read-only memmap slice (avoids PyTorch's warning).
        toks = np.array(
            self.cache.tokens[start:start + length], dtype=np.float32, copy=True,
        )
        return torch.from_numpy(toks), int(self.cache.labels[idx]), int(length)


def collate_padded(
    batch: list[tuple[torch.Tensor, int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad variable-length flow tokens to a common batch length.

    Returns ``(tokens, labels, lengths, key_padding_mask)`` where:
      * ``tokens``            — ``[B, T_max, 13]``  float32, NaNs preserved
      * ``labels``            — ``[B]``             int64
      * ``lengths``           — ``[B]``             int64
      * ``key_padding_mask``  — ``[B, T_max]``      bool, True at padding
    """
    lengths = torch.tensor([item[2] for item in batch], dtype=torch.int64)
    T_max = int(lengths.max().item())
    B = len(batch)
    tokens = torch.zeros(B, T_max, TOKEN_DIM, dtype=torch.float32)
    mask = torch.ones(B, T_max, dtype=torch.bool)  # True = padded
    labels = torch.zeros(B, dtype=torch.int64)
    for i, (t, y, L) in enumerate(batch):
        tokens[i, :L] = t
        mask[i, :L] = False
        labels[i] = y
    return tokens, labels, lengths, mask


__all__ = [
    "TOKEN_DIM",
    "build_flow_tokens",
    "build_token_cache",
    "load_token_cache",
    "prepare_split_caches",
    "MetaLearnerDataset",
    "TokenCache",
    "collate_padded",
]
