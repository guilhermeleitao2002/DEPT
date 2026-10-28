"""Phase 9: DEPT's purifier vs the diffusion purifier of Merzouk et al. (DBSec 2025).

The diffusion purifier (github.com/mamerzouk/adversarial-purification) replaces
DEPT's reconstructor in the phase-8 setup: same flows, same realization into
packets and same independent NIDS.

  * :mod:`.config`   — phase-9 YAML + the settings shared with phases 6 and 8
  * :mod:`.data`     — the reconstructor's pairs/split, evaluation subsets, flow reads
  * :mod:`.overlay`  — the (size, log-IPAT) flow vector the purifier works in
  * :mod:`.model`    — the vendored diffusion model and purification loop
  * :mod:`.train`    — stages ``overlays`` (training data) and ``train``
  * :mod:`.sweep`    — stage ``sweep``: choose the diffusion step t* on validation flows
  * :mod:`.realize`  — N-branch realization with phase 8's shared keep-mask
  * :mod:`.evaluate` — stage ``evaluate``: scoring through the NIDS, subsets, latency
"""
from .config import load_config
from .evaluate import evaluate
from .sweep import run_sweep
from .train import build_caches, train

__all__ = ["load_config", "build_caches", "train", "run_sweep", "evaluate"]
