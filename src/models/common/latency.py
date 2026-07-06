"""Wall-clock timing utilities for the ADEPT latency/throughput benchmark.

Shared helpers used by ``scripts/benchmark/run_latency.py`` to time model
forward passes with a warmup phase and CUDA synchronisation, and to summarise
per-iteration samples into a compact percentile block.

Timing convention: we use ``time.perf_counter`` bracketed by
``torch.cuda.synchronize`` so the measured interval covers the full device
compute (GPU kernels are launched asynchronously; without the sync the host
clock would stop before the kernels finish). On CPU the sync is a no-op.
"""
from __future__ import annotations

import time
from statistics import median, pstdev
from typing import Callable

import torch


def sync(device: torch.device) -> None:
    """Block until all queued work on ``device`` has finished (no-op on CPU)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_calls(
    fn: Callable[[], object],
    device: torch.device,
    *,
    warmup: int = 10,
    iters: int = 50,
) -> list[float]:
    """Time a no-arg callable ``fn`` ``iters`` times; return per-call ms.

    ``fn`` should run exactly the work under test (e.g. one forward pass) and
    is called under ``torch.inference_mode``. The first ``warmup`` calls are
    discarded so lazy allocation, cuDNN autotuning, and CUDA-graph capture do
    not pollute the measurement.
    """
    samples: list[float] = []
    with torch.inference_mode():
        for _ in range(max(0, warmup)):
            fn()
        sync(device)
        for _ in range(max(1, iters)):
            sync(device)
            t0 = time.perf_counter()
            fn()
            sync(device)
            samples.append((time.perf_counter() - t0) * 1e3)
    return samples


def time_sweep(
    fn: Callable[[], object],
    device: torch.device,
    *,
    warmup: int = 1,
    repeats: int = 3,
) -> list[float]:
    """Time a whole-pool sweep ``fn`` ``repeats`` times; return per-sweep ms.

    Used for throughput: ``fn`` runs one full pass over a preloaded batch pool.
    """
    return time_calls(fn, device, warmup=warmup, iters=repeats)


def summarize(samples_ms: list[float]) -> dict:
    """Reduce per-call ms samples to ``{mean,median,p95,p99,std,min,max,n}``."""
    if not samples_ms:
        return {"mean": None, "median": None, "p95": None, "p99": None,
                "std": None, "min": None, "max": None, "n": 0}
    s = sorted(samples_ms)
    n = len(s)

    def pct(p: float) -> float:
        if n == 1:
            return s[0]
        k = int(round((p / 100.0) * (n - 1)))
        return s[min(n - 1, max(0, k))]

    return {
        "mean": sum(s) / n,
        "median": float(median(s)),
        "p95": pct(95),
        "p99": pct(99),
        "std": float(pstdev(s)) if n > 1 else 0.0,
        "min": s[0],
        "max": s[-1],
        "n": n,
    }
