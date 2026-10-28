"""Thin adapter over TANTGen's Kitsune (KitNET autoencoder + AfterImage extractor).

Kitsune lives in the TANTGen repo. Its two source dirs are added to ``sys.path``
and ``KitNET`` / ``FEKitsune`` are called directly (TANTGen's ``KitNET/model.py``
CLI reads ``config-folders.yaml`` at import time). The compiled
``AfterImage*.so`` is required (built for CPython 3.10). If anything is missing,
:class:`KitsuneUnavailable` is raised.

A model pickle holds four objects ``[feature_map, ensembleLayer, outputLayer,
AD_threshold]`` (TANTGen's ``model.py -M train`` format).
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# AfterImage vector length in TANTGen's (modified) Kitsune.
FEATURE_DIM = 100

# TANTGen Kitsune source dirs, relative to ``tantgen_root``.
_FE_SUBDIR = "src/Kitsune/Feature_Extraction"   # netStat, AfterImage(.so), FeatureExtractor, FEKitsune, KitsuneTools
_KN_SUBDIR = "src/Kitsune/KitNET"               # KitNET, dA, corClust, utils

_STOP_FLAG = 999999999.0                          # FEKitsune's end-of-stream sentinel


class KitsuneUnavailable(RuntimeError):
    """Kitsune cannot be used (missing TANTGen tree, compiled ``.so``, or model)."""


def _ensure_paths(tantgen_root) -> None:
    """Put TANTGen's Kitsune dirs on ``sys.path`` (idempotent) and check the ``.so``."""
    import sys

    tantgen_root = Path(tantgen_root)
    fe_dir = tantgen_root / _FE_SUBDIR
    kn_dir = tantgen_root / _KN_SUBDIR
    if not fe_dir.is_dir() or not kn_dir.is_dir():
        raise KitsuneUnavailable(
            f"TANTGen Kitsune dirs not found under {tantgen_root} "
            f"(need {_FE_SUBDIR} and {_KN_SUBDIR})")
    if not list(fe_dir.glob("AfterImage*.so")):
        raise KitsuneUnavailable(
            f"compiled AfterImage .so not found in {fe_dir} "
            f"(build it: cd '{fe_dir}' && python setup.py build_ext --inplace)")
    for d in (str(fe_dir), str(kn_dir)):
        if d not in sys.path:
            sys.path.insert(0, d)


def available(tantgen_root) -> tuple[bool, str]:
    """Best-effort availability probe → ``(ok, reason)``. Never raises."""
    try:
        _ensure_paths(tantgen_root)
        import FEKitsune  # noqa: F401
        import KitNET     # noqa: F401
        return True, ""
    except Exception as e:  # KitsuneUnavailable, ImportError, ABI mismatch, …
        return False, str(e)


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction (AfterImage) — stateful, temporal order
# ──────────────────────────────────────────────────────────────────────────────

def extract_features(packets, tantgen_root) -> np.ndarray:
    """AfterImage 100-dim features for an ordered scapy packet list.

    ``packets`` must be indexable with ``len`` (a scapy ``PacketList`` or a plain
    ``list``); AfterImage is stateful, so packets must already be in temporal order.
    Returns ``[P, 100]`` (``P`` may be < ``len(packets)`` if extraction stops early on a
    malformed packet — TANTGen's ``RunFE`` breaks on the first bad packet).
    """
    _ensure_paths(tantgen_root)
    import FEKitsune
    from KitsuneTools import RunFE

    if len(packets) == 0:
        return np.empty((0, FEATURE_DIM), dtype=np.float64)
    fe = FEKitsune.Kitsune(packets, np.inf)
    feats, _ = RunFE(fe)
    if not feats:
        return np.empty((0, FEATURE_DIM), dtype=np.float64)
    arr = np.asarray(feats, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != FEATURE_DIM:
        raise KitsuneUnavailable(
            f"AfterImage returned features of shape {arr.shape}, expected [*, {FEATURE_DIM}]")
    if arr.shape[0] < len(packets):
        log.warning("AfterImage stopped early: %d/%d packets extracted",
                    arr.shape[0], len(packets))
    return arr


def read_pcap_head(pcap_path, limit) -> list:
    """First ``limit`` packets of a PCAP via a streaming reader (memory-bounded)."""
    from scapy.all import PcapReader

    pkts: list = []
    with PcapReader(str(pcap_path)) as reader:
        for pkt in reader:
            pkts.append(pkt)
            if len(pkts) >= limit:
                break
    return pkts


# ──────────────────────────────────────────────────────────────────────────────
# KitNET — load (execute-mode) / score / train (calibrate)
# ──────────────────────────────────────────────────────────────────────────────

def _load_threshold(model_path) -> float:
    """Read the 4th pickled object (``AD_threshold``) from a Kitsune model pickle."""
    with open(model_path, "rb") as f:
        pickle.load(f)  # feature_map (v)
        pickle.load(f)  # ensembleLayer
        pickle.load(f)  # outputLayer
        return float(pickle.load(f))  # AD_threshold


def load_model(model_path, tantgen_root, n: int = FEATURE_DIM):
    """Load a calibrated Kitsune in execute-mode → ``(kitnet, threshold)``.

    ``kitnet.process(x)`` returns the per-packet RMSE. Execute is per-packet stateless
    (only AfterImage is stateful), so one loaded ``kitnet`` can score several branches.
    """
    _ensure_paths(tantgen_root)
    import KitNET

    model_path = Path(model_path)
    if not model_path.exists():
        raise KitsuneUnavailable(f"Kitsune model not found: {model_path}")
    # AD_grace_period=-1 → execute-mode: loads [feature_map, ensembleLayer, outputLayer].
    kitnet = KitNET.KitNET(str(model_path), n, max_autoencoder_size=10,
                           FM_grace_period=None, AD_grace_period=-1)
    return kitnet, _load_threshold(model_path)


def score(kitnet, feats: np.ndarray) -> np.ndarray:
    """Per-packet RMSE for a ``[P, 100]`` feature matrix from an execute-mode KitNET."""
    rmse = np.empty(len(feats), dtype=np.float64)
    for i in range(len(feats)):
        rmse[i] = float(kitnet.process(np.asarray(feats[i], dtype=np.float64)))
    return rmse


def train_model(benign_pcap, out_path, tantgen_root, *,
                fm: int = 5000, ad: int = 50000, cap: int = 55000,
                n: int = FEATURE_DIM, seed: int = 42) -> dict:
    """Calibrate a fresh Kitsune on a benign PCAP (TANTGen protocol) and persist it.

    Feeds the first ``fm+ad`` AfterImage vectors: ``fm`` for the feature map, ``ad`` for
    the autoencoders; the threshold is ``max(rmse[fm:])`` (max benign RMSE after FM-grace).
    Persists ``[feature_map, ensembleLayer, outputLayer, AD_threshold]`` to ``out_path``.
    If the benign stream is shorter than ``fm+ad`` the grace split is scaled down to the
    available count (keeping the ~1:10 ratio) so the model is always fully written.
    """
    _ensure_paths(tantgen_root)
    import KitNET

    benign_pcap = Path(benign_pcap)
    if not benign_pcap.exists():
        raise KitsuneUnavailable(f"benign PCAP not found: {benign_pcap}")

    np.random.seed(seed)  # dA weight init draws from np.random → reproducible threshold
    cap = max(int(cap), fm + ad)
    pkts = read_pcap_head(benign_pcap, cap)
    feats = extract_features(pkts, tantgen_root)
    total = int(feats.shape[0])
    if total < 2:
        raise KitsuneUnavailable(
            f"benign feature extraction yielded {total} vectors from {benign_pcap}")

    # KitNET only persists once fm+ad packets have been fed. Adapt if benign is smaller.
    if total < fm + ad:
        fm = max(1, total // 11)
        ad = total - fm
    n_feed = fm + ad

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kitnet = KitNET.KitNET(str(out_path), n, max_autoencoder_size=10,
                           FM_grace_period=fm, AD_grace_period=ad)
    rmse = np.empty(n_feed, dtype=np.float64)
    for i in range(n_feed):
        rmse[i] = float(kitnet.process(np.asarray(feats[i], dtype=np.float64)))
    threshold = float(np.max(rmse[fm:]))
    with open(out_path, "ab") as f:
        pickle.dump(threshold, f)
    log.info("trained Kitsune → %s (fm=%d ad=%d threshold=%.6g)",
             out_path, fm, ad, threshold)
    return {"threshold": threshold, "fm": fm, "ad": ad, "n_features": total}
