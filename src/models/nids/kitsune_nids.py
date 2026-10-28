"""Kitsune (KitNET + AfterImage) as a :class:`NIDSModel`.

Wrapper over ``src/models/kitsune/runner.py`` for ``mirai`` and ``ssdp_flood``.
TANTGen's own KitNET pickles are reused; a new model is calibrated on benign
traffic only if a pickle is missing. The score is the per-packet RMSE,
aggregated per flow (``max`` by default) and compared to the threshold φ.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..kitsune import runner as krun
from .base import BranchScore, NIDSModel, NIDSUnavailable
from .metrics import flow_agg

log = logging.getLogger(__name__)


class KitsuneNIDS(NIDSModel):
    kind = "kitsune"

    def __init__(self, kitnet, threshold: float, tantgen_root: Path,
                 base: str, detection: str = "max", source: str = "pretrained") -> None:
        self._kitnet = kitnet
        self.threshold = float(threshold)
        self._tantgen_root = Path(tantgen_root)
        self._detection = detection
        self.name = f"kitsune:{base}"
        self.model_source = source  # pretrained | trained

    def score(self, packets: list, keys: list) -> BranchScore:
        if not packets:
            return BranchScore(0, 0, None, None, 0.0)
        feats = krun.extract_features(packets, self._tantgen_root)
        rmse = krun.score(self._kitnet, feats)
        per_flow = flow_agg(rmse, keys, self._detection)
        return BranchScore.from_flow_scores(per_flow, self.threshold)

    def score_flows(self, packets: list, keys: list) -> list[tuple[str, float, int]]:
        """Per-flow variant of :meth:`score` (phase 9): ``(flow_key, RMSE, detected)``.

        The whole stream is still fed to AfterImage (its statistics are stateful across
        flows); only the per-flow results are returned, so a caller can restrict the
        metric to a subset of flows without changing what Kitsune saw.
        """
        if not packets:
            return []
        feats = krun.extract_features(packets, self._tantgen_root)
        rmse = krun.score(self._kitnet, feats)
        per_flow = flow_agg(rmse, keys, self._detection)
        return [(k, float(s), int(s > self.threshold)) for k, s in per_flow.items()]


def build(base: str, cfg: dict, repo_root: Path) -> KitsuneNIDS:
    """Locate (or calibrate) the Kitsune model for ``base`` → :class:`KitsuneNIDS`.

    Priority: TANTGen's calibrated pretrained pickle → a previously DEPT-trained pickle
    → train a fresh one on the benign split. ``cfg`` is the resolved ``nids`` config
    block; ``repo_root`` anchors any relative DEPT paths.
    """
    kcfg = dict(cfg.get("kitsune", {}))
    tantgen_root = _abs(cfg.get("tantgen_root", ""), repo_root)
    ok, why = krun.available(tantgen_root)
    if not ok:
        raise NIDSUnavailable(f"kitsune unavailable: {why}")

    detection = str(kcfg.get("detection", "max"))
    entry = dict(cfg.get("attacks", {}).get(base, {}))

    # 1) explicit pretrained pickle (relative to tantgen_root).
    rel = entry.get("model")
    if rel:
        p = Path(rel)
        if not p.is_absolute():
            p = tantgen_root / p
        if p.exists():
            kn, thr = krun.load_model(p, tantgen_root)
            return KitsuneNIDS(kn, thr, tantgen_root, base, detection, "pretrained")

    # 2) a previously DEPT-calibrated pickle under model_dir.
    model_dir = _abs(kcfg.get("model_dir", "data/models/kitsune"), repo_root)
    local = model_dir / f"{base}_model.pkl"
    if local.exists():
        kn, thr = krun.load_model(local, tantgen_root)
        return KitsuneNIDS(kn, thr, tantgen_root, base, detection, "trained")

    # 3) calibrate fresh on the benign split (TANTGen protocol).
    base_to_tg = kcfg.get("base_to_tg", {}) or {}
    tg = base_to_tg.get(base, entry.get("tg_attack"))
    if not tg:
        raise NIDSUnavailable(
            f"no pretrained/trained Kitsune and no base_to_tg entry for '{base}'")
    benign_dir = _abs(kcfg.get("benign_pcap_dir", "data/original/Kitsune/Benign"),
                      tantgen_root)
    benign_pcap = benign_dir / f"{tg}_benign.pcap"
    if not benign_pcap.exists():
        raise NIDSUnavailable(f"benign PCAP not found for calibration: {benign_pcap}")
    log.info("calibrating Kitsune for %s from %s", base, benign_pcap)
    krun.train_model(
        benign_pcap, local, tantgen_root,
        fm=int(kcfg.get("fm_grace", 5000)), ad=int(kcfg.get("ad_grace", 50000)),
        cap=int(kcfg.get("train_cap_packets", 55000)), seed=int(kcfg.get("seed", 42)))
    kn, thr = krun.load_model(local, tantgen_root)
    return KitsuneNIDS(kn, thr, tantgen_root, base, detection, "trained")


def _abs(p, anchor: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else anchor / p


__all__ = ["KitsuneNIDS", "build"]
