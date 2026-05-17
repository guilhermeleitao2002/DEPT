"""Runtime helpers shared across base learners and the meta-learner.

Currently exposes ``resolve_num_workers``, which lets the YAML configs
ship ``num_workers: auto`` so the same files run unchanged on a 4-core
laptop and a 52-core server. The cap keeps memmap workers from
oversubscribing big NUMA hosts when the model is GPU-bound.
"""
from __future__ import annotations

import os


def resolve_num_workers(value, *, default_auto_cap: int = 16) -> int:
    """Resolve a ``num_workers`` config value to a concrete int.

    Accepts ``"auto"`` / ``None`` (pick ``min(default_auto_cap, cpu_count // 2)``,
    floored at 2), any int (returned as-is), or any string parseable as int.
    The auto cap exists so a 52-core server doesn't spawn 26 DataLoader
    workers for a model that's GPU-bound after the first few prefetches.
    """
    if value is None or (isinstance(value, str) and value.strip().lower() == "auto"):
        cores = os.cpu_count() or 1
        return max(2, min(default_auto_cap, cores // 2))
    return int(value)
