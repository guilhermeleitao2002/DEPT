"""Resolve which independent NIDS scores each attack, from the shared ``nids`` config.

One ``nids:`` block (in ``config/baseline.yaml`` and ``config/reconstructor.yaml``) maps
every base attack to the NIDS it was actually built to evade:

  * Kitsune-native attacks (``mirai``, ``ssdp_flood``) → :class:`KitsuneNIDS`.
  * CIC-IDS / CIC-DDoS attacks → :class:`MLNIDS`.

so a caller just asks ``registry.get("<base>")`` and gets the right, lazily-loaded model.
Loaded models are cached per base attack (loading a KitNET pickle or a scaler+classifier
is not free, and the same model scores the adv / purified / clean branches in turn).
"""
from __future__ import annotations

import logging
from pathlib import Path

from . import kitsune_nids, ml_nids
from .base import NIDSModel, NIDSUnavailable

log = logging.getLogger(__name__)


class NIDSRegistry:
    def __init__(self, cfg: dict, repo_root: Path) -> None:
        self._cfg = dict(cfg or {})
        self._repo_root = Path(repo_root)
        self._attacks = dict(self._cfg.get("attacks", {}))
        self._cache: dict[str, NIDSModel] = {}

    def kind_for(self, base: str) -> str | None:
        entry = self._attacks.get(base)
        return str(entry.get("kind")) if entry else None

    def known(self, base: str) -> bool:
        return base in self._attacks

    def get(self, base: str) -> NIDSModel:
        """Return the (cached) :class:`NIDSModel` for base attack ``base``.

        Raises :class:`NIDSUnavailable` if the attack is unmapped or its NIDS/deps are
        missing — the caller decides whether that aborts the run (baseline) or skips
        one source with a reason (reconstructor).
        """
        if base in self._cache:
            return self._cache[base]
        entry = self._attacks.get(base)
        if entry is None:
            raise NIDSUnavailable(f"no NIDS mapping for attack '{base}' (add it under nids.attacks)")
        kind = str(entry.get("kind", "")).lower()
        if kind == "kitsune":
            model = kitsune_nids.build(base, self._cfg, self._repo_root)
        elif kind == "ml_nids":
            model = ml_nids.build(base, self._cfg, self._repo_root)
        else:
            raise NIDSUnavailable(f"unknown NIDS kind {kind!r} for attack '{base}'")
        self._cache[base] = model
        log.info("resolved NIDS for %s → %s (source=%s)", base, model.name,
                 getattr(model, "model_source", "?"))
        return model


__all__ = ["NIDSRegistry"]
