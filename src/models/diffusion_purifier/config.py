"""Resolve phase 9's configuration.

Phase 9 has to see exactly what phases 6 and 8 saw, so it does not repeat their
settings. The split, excludes and tensor paths come from ``reconstructor.yaml`` (the
diffusion purifier trains on the reconstructor's training pairs), and the NIDS routing,
gate and model paths come from ``baseline.yaml`` (the evaluation is the phase-8 harness
with the purifier swapped). ``diffusion_purifier.yaml`` only holds what is new.
"""
from __future__ import annotations

from pathlib import Path

import yaml

# Settings phase 9 must share with the reconstructor (phase 6).
_FROM_RECONSTRUCTOR = ("tensors_dir", "adversarial_dir", "adversarial_exclude",
                       "min_len", "split")
# Settings phase 9 must share with the deployment baseline (phase 8).
_FROM_BASELINE = ("adversarial_pcap_dir", "rf_model_dir", "tcn_model_dir",
                  "metalearner_model_dir", "reconstructor_model_dir",
                  "detector_threshold", "token", "nids", "num_workers")
_PATH_KEYS = ("tensors_dir", "adversarial_dir", "adversarial_pcap_dir", "rf_model_dir",
              "tcn_model_dir", "metalearner_model_dir", "reconstructor_model_dir",
              "output_dir", "detector_cache_dir")


def _abs(p, anchor: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else anchor / p


def load_config(cfg_path: Path, repo_root: Path) -> dict:
    """Phase-9 YAML merged with the shared phase-6/8 settings, paths made absolute."""
    repo_root = Path(repo_root)
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    recon = yaml.safe_load(_abs(cfg["reconstructor_config"], repo_root).read_text())
    base = yaml.safe_load(_abs(cfg["baseline_config"], repo_root).read_text())

    for k in _FROM_RECONSTRUCTOR:
        cfg[k] = recon.get(k)
    for k in _FROM_BASELINE:
        cfg[k] = base.get(k)
    # Reuse phase 8's meta-learner token cache (same source set → same cache tag).
    cfg["detector_cache_dir"] = str(Path(base.get("output_dir", "data/models/baseline"))
                                    / "detector_cache")
    for k in _PATH_KEYS:
        if cfg.get(k) is not None:
            cfg[k] = str(_abs(cfg[k], repo_root))
    cfg["_repo_root"] = str(repo_root)
    return cfg


__all__ = ["load_config"]
