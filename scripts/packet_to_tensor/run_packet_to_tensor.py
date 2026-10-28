#!/usr/bin/env python3
"""
DEPT — Packet-to-Tensor engine
===============================

Called by ``process_all_attacks.py``; ``cfg`` is the parsed ``config/pipeline.yaml``.

    parse_pcap(input_pcap, cfg)
        Parse and flow-group one PCAP (int8 nPrint bits, numeric metadata,
        flow index). About 1.1 GB of RAM per million packets.

    fit_mean_std(pcap_paths, out_path, cfg, parsed_cache=None)
        Fit the shared IPAT standardizer (Welford/Chan merge) and write it to
        ``out_path``. With ``parsed_cache``, each parse is kept for ``run``.

    run(input_pcap, output_dir, mean_std_path, cfg, parsed=None)
        Write one PCAP's ragged per-flow tensors:

            flow_packets.npy            [P_total, 1090] float32
            flow_offsets.npy            [N_flows, 2]    int64 (start, length)
            rf_checkpoints.npy          [C_total, 24]   float32
            rf_checkpoint_offsets.npy   [N_flows, 2]    int64 (start, n_checkpoints)

        plus flow_ids.npy and metadata.json.
"""

import json
import os
import sys
import time

import numpy as np
import yaml

# Allow importing src.* when driven from the project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.feature_extraction.pcap_parser import extract_packet_features
from src.feature_extraction.flow_grouper import group_by_flow
from src.feature_extraction.tensor_builder import build_tensors


# ---------------------------------------------------------------------------
# Terminal colours (ANSI — disabled when stdout is not a tty)
# ---------------------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def bold(t):    return _c("1", t)
def dim(t):     return _c("2", t)
def green(t):   return _c("32", t)
def cyan(t):    return _c("36", t)


def _step(n: int, total: int, msg: str) -> None:
    print(f"\n{dim(f'[{n}/{total}]')} {bold(msg)}")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# One-shot parse: Scapy read + nPrint encode + flow grouping
# ---------------------------------------------------------------------------
def parse_pcap(input_pcap: str, cfg: dict) -> dict:
    """Parse one PCAP into the in-memory form the engine uses.

    Returns a dict with:

        nprint_bits   : int8 [P, 1088], time-sorted
        meta_df       : DataFrame, time-sorted, numeric columns only
        flow_indices  : OrderedDict[flow key → row indices]
    """
    nprint_bits, meta_df = extract_packet_features(
        input_pcap, mask_prefixes=cfg.get("mask_prefixes", []),
    )
    meta_df, flow_indices, sort_order = group_by_flow(meta_df)
    nprint_bits = nprint_bits[sort_order]
    del sort_order

    keep = [
        c for c in meta_df.columns
        if c in set(cfg["standardize_columns"])
        | set(cfg["rf_base_features"])
        | {"IPAT", "Direction"}
    ]
    meta_df = meta_df[keep]

    return {
        "nprint_bits": nprint_bits,
        "meta_df": meta_df,
        "flow_indices": flow_indices,
        "num_packets": len(meta_df),
    }


# ---------------------------------------------------------------------------
# Streaming mean/std fitter (Welford / Chan parallel update)
# ---------------------------------------------------------------------------
def _welford_merge(state: dict, vals: "np.ndarray") -> dict:
    """Fold a numpy batch of observations into a (count, mean, M2) accumulator."""
    n_b = len(vals)
    if n_b == 0:
        return state
    n_a = state["count"]
    mean_b = float(vals.mean())
    M2_b = float(((vals - mean_b) ** 2).sum())
    if n_a == 0:
        return {"count": n_b, "mean": mean_b, "M2": M2_b}
    n = n_a + n_b
    delta = mean_b - state["mean"]
    mean = state["mean"] + delta * n_b / n
    M2 = state["M2"] + M2_b + delta * delta * n_a * n_b / n
    return {"count": n, "mean": mean, "M2": M2}


def fit_mean_std(
    pcap_paths, out_path: str, cfg: dict, parsed_cache: dict = None,
) -> dict:
    """Fit the per-column mean/std across PCAPs and write it to JSON.

    The ``cfg["standardize_columns"]`` are merged with Welford/Chan. When
    ``parsed_cache`` is a dict, each :func:`parse_pcap` result is stored in it by
    path so :func:`run` can skip the parse; the caller should ``pop`` entries as
    it uses them.
    """
    columns = cfg["standardize_columns"]
    accumulators = {c: {"count": 0, "mean": 0.0, "M2": 0.0} for c in columns}
    for i, pcap in enumerate(pcap_paths, 1):
        print(f"\n  {dim(f'[{i}/{len(pcap_paths)}]')} {os.path.basename(pcap)}")
        parsed = parse_pcap(pcap, cfg)
        meta_df = parsed["meta_df"]
        for c in columns:
            if c in meta_df.columns:
                accumulators[c] = _welford_merge(
                    accumulators[c], meta_df[c].values.astype(np.float64)
                )
        if parsed_cache is not None:
            parsed_cache[pcap] = parsed
        del parsed, meta_df

    stats = {}
    for c, acc in accumulators.items():
        n = acc["count"]
        if n < 2:
            std = 1.0
        else:
            std = (acc["M2"] / (n - 1)) ** 0.5
            if std == 0:
                std = 1.0
        stats[c] = {"mean": acc["mean"], "std": std}

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


# ---------------------------------------------------------------------------
# Per-PCAP tensor build (batch / global-sort path)
# ---------------------------------------------------------------------------
def run(
    input_pcap: str, output_dir: str, mean_std_path: str, cfg: dict,
    parsed: dict = None,
) -> dict:
    """Convert one PCAP into ragged per-flow tensors.

    ``mean_std_path`` is the shared standardizer from :func:`fit_mean_std`.
    ``parsed`` is an optional :func:`parse_pcap` result for ``input_pcap``.
    """
    rf_checkpoint_stride = int(cfg.get("rf_checkpoint_stride", 10))
    rf_first_checkpoint = int(cfg.get("rf_first_checkpoint", 2))
    rf_base = cfg["rf_base_features"]
    rf_agg = cfg["rf_agg_functions"]
    std_cols = cfg["standardize_columns"]
    mask_prefixes = cfg.get("mask_prefixes", [])
    selected_outputs = ("rf", "tcn")

    os.makedirs(output_dir, exist_ok=True)

    fname = os.path.basename(input_pcap)
    t0 = time.time()

    if parsed is None:
        _step(1, 2, f"Extracting nPrint features  {dim(fname)}")
        parsed = parse_pcap(input_pcap, cfg)
    else:
        _step(1, 2, f"Reusing cached parse  {dim(fname)}")
    nprint_bits = parsed["nprint_bits"]
    meta_df = parsed["meta_df"]
    flow_indices = parsed["flow_indices"]
    t2 = time.time()
    print(f"      {green('✓')} {len(meta_df):,} packets, "
          f"{len(flow_indices):,} flows  {dim(f'({t2 - t0:.1f}s)')}")

    sizes = [len(idx) for idx in flow_indices.values()]
    if sizes:
        min_ckpt_len = (
            min(rf_first_checkpoint, rf_checkpoint_stride)
            if rf_first_checkpoint > 0
            else rf_checkpoint_stride
        )
        usable = sum(1 for s in sizes if s >= min_ckpt_len)
        print(f"      Flow sizes: min={min(sizes)}, "
              f"median={np.median(sizes):.0f}, "
              f"max={max(sizes)} — "
              f"{cyan(str(usable))}/{len(sizes)} have ≥{min_ckpt_len} packets")

    cadence_str = (
        f"first={rf_first_checkpoint}, stride={rf_checkpoint_stride}"
        if rf_first_checkpoint > 0
        else f"stride={rf_checkpoint_stride}"
    )
    _step(2, 2, f"Building ragged tensors  {dim(f'(rf cadence: {cadence_str})')}")
    result = build_tensors(
        nprint_bits=nprint_bits,
        meta_df=meta_df,
        flow_indices=flow_indices,
        output_dir=output_dir,
        rf_checkpoint_stride=rf_checkpoint_stride,
        rf_first_checkpoint=rf_first_checkpoint,
        rf_base_features=rf_base,
        rf_agg_functions=rf_agg,
        standardize_columns=std_cols,
        mean_std_path=mean_std_path,
        outputs=selected_outputs,
    )
    del nprint_bits, meta_df
    t3 = time.time()
    print(f"      {green('✓')} {result['num_flows']:,} flows, "
          f"{result['num_packets']:,} packets, "
          f"{result['num_checkpoints']:,} RF checkpoints  "
          f"{dim(f'({t3 - t2:.1f}s)')}")

    # ── Write flow_ids.npy and metadata.json ──────────────────────────────────
    np.save(os.path.join(output_dir, "flow_ids.npy"),
            np.array(result["flow_ids"], dtype=object), allow_pickle=True)

    n_flows = result["num_flows"]
    n_packets = result["num_packets"]
    n_ckpts = result["num_checkpoints"]
    metadata = {
        "input":                    input_pcap,
        "encoding":                 "nprint",
        "pipeline_mode":            "batch",
        "schema":                   "ragged-per-flow-v1",
        "outputs":                  ["rf", "tcn"],
        "nprint_bits":              1088,
        "identity_masking":         True,
        "mask_prefixes":            mask_prefixes,
        "rf_checkpoint_stride":     rf_checkpoint_stride,
        "rf_first_checkpoint":      rf_first_checkpoint,
        "num_flows":                n_flows,
        "num_packets":              n_packets,
        "num_checkpoints":          n_ckpts,
        "sequence_feature_count":   len(result["feature_names"]),
        "rf_stat_names":            result["rf_stat_names"],
        "normalization_stats":      result["stats"],
        "flow_packets_shape":       [n_packets, len(result["feature_names"])],
        "rf_checkpoints_shape":     [n_ckpts, len(result["rf_stat_names"])],
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    total = t3 - t0
    W = 62
    print(f"\n{bold('═' * W)}")
    print(f" {bold('DEPT Packet-to-Tensor (streaming)')}  {dim(f'Done in {total:.1f}s')}")
    print(f"{'─' * W}")
    print(f" {cyan('flow_packets')}     [{n_packets}, {len(result['feature_names'])}]  "
          f"{dim('TCN / Transformer (per-flow ragged)')}")
    print(f" {cyan('rf_checkpoints')}   [{n_ckpts}, {len(result['rf_stat_names'])}]  "
          f"{dim(f'Random Forest ({cadence_str})')}")
    print(f" {cyan('Identity masking')} {green('ON')}")
    print(f" {cyan('Output')}           {output_dir}")
    print(f"{bold('═' * W)}\n")

    return result
