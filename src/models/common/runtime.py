"""Runtime helpers shared across the models."""
from __future__ import annotations

import os


def resolve_num_workers(value, *, default_auto_cap: int = 16) -> int:
    """Resolve a ``num_workers`` config value to a concrete int.

    Accepts ``"auto"`` / ``None`` (pick ``min(default_auto_cap, cpu_count // 2)``,
    floored at 2), any int (returned as-is), or any string parseable as int.
    """
    if value is None or (isinstance(value, str) and value.strip().lower() == "auto"):
        cores = os.cpu_count() or 1
        return max(2, min(default_auto_cap, cores // 2))
    return int(value)
