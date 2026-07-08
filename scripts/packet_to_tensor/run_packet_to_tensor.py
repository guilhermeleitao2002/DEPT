#!/usr/bin/env python3
"""
DEPT — Packet-to-Tensor engine (streaming-native, nPrint encoding)
===================================================================

Importable engine module driven in-process by ``process_all_attacks.py``.
There is no command-line interface: the DEPT core pipeline has a single,
fixed configuration, so the two public functions below are called directly
with the per-source paths and the parsed ``config/pipeline.yaml``.

    fit_mean_std(pcap_paths, out_path, cfg)
        Fit the shared IPAT standardizer over one or more PCAPs using a
        streaming Welford accumulator and write it to ``out_path`` (JSON).
        Always overwrites.

    run(input_pcap, output_dir, mean_std_path, cfg)
        Convert one PCAP into DEPT's ragged per-flow tensors:

            flow_packets.npy            [P_total, F=1090] float32
              Per-packet nPrint bits + IPAT (z-score) + Direction,
              concatenated across all flows in flow-arrival order.
            flow_offsets.npy            [N_flows, 2]      int64
              (start_idx, length) into flow_packets.npy per flow.
            rf_checkpoints.npy          [C_total, S=24]   float32
              Aggregated statistics over the growing prefix of each flow,
              emitted at ``rf_first_checkpoint`` (config default 2) and then
              every ``rf_checkpoint_stride`` packets (config default 10).
            rf_checkpoint_offsets.npy   [N_flows, 2]      int64
              (start_idx, n_checkpoints) into rf_checkpoints.npy per flow.

        Plus flow_ids.npy, metadata.json, and a local mean_std.json copy.

All tunables come from ``cfg`` (config/pipeline.yaml). The pipeline is fixed
to the DEPT core configuration: the batch (global-sort) path, identity
masking on, both the RF and TCN tensor groups, and the config RF checkpoint
cadence.
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
def yellow(t):  return _c("33", t)
def cyan(t):    return _c("36", t)


def _step(n: int, total: int, msg: str) -> None:
    print(f"\n{dim(f'[{n}/{total}]')} {bold(msg)}")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


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


def fit_mean_std(pcap_paths, out_path: str, cfg: dict) -> dict:
    """Fit the shared per-column mean/std across PCAPs and write it to JSON.

    Streams each PCAP once (nPrint bits discarded — only the flow metadata
    columns listed in ``cfg["standardize_columns"]`` are accumulated) with a
    numerically-stable Welford/Chan merge. Always overwrites ``out_path``.
    """
    columns = cfg["standardize_columns"]
    mask_prefixes = cfg.get("mask_prefixes", [])
    accumulators = {c: {"count": 0, "mean": 0.0, "M2": 0.0} for c in columns}
    for i, pcap in enumerate(pcap_paths, 1):
        print(f"\n  {dim(f'[{i}/{len(pcap_paths)}]')} {os.path.basename(pcap)}")
        nprint_bits, meta_df = extract_packet_features(
            pcap, mask_prefixes=mask_prefixes,
        )
        del nprint_bits
        meta_df, _, _ = group_by_flow(meta_df)
        for c in columns:
            if c in meta_df.columns:
                accumulators[c] = _welford_merge(
                    accumulators[c], meta_df[c].values.astype(np.float64)
                )
        del meta_df

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
def run(input_pcap: str, output_dir: str, mean_std_path: str, cfg: dict) -> dict:
    """Convert one PCAP into DEPT's ragged per-flow tensors.

    Fixed to the DEPT core configuration: batch pipeline, identity masking
    on, both RF and TCN tensor groups, RF checkpoint cadence from ``cfg``.
    ``mean_std_path`` is the shared standardizer produced by
    :func:`fit_mean_std`; it is loaded for IPAT z-scoring and copied into
    ``output_dir`` as the source's local ``mean_std.json``.
    """
    rf_checkpoint_stride = int(cfg.get("rf_checkpoint_stride", 10))
    rf_first_checkpoint = int(cfg.get("rf_first_checkpoint", 2))
    rf_base = cfg["rf_base_features"]
    rf_agg = cfg["rf_agg_functions"]
    std_cols = cfg["standardize_columns"]
    mask_prefixes = cfg.get("mask_prefixes", [])
    selected_outputs = ("rf", "tcn")

    os.makedirs(output_dir, exist_ok=True)

    # Keep a local copy of the shared standardizer alongside the tensors.
    local_mean_std = os.path.join(output_dir, cfg["mean_std_filename"])
    if os.path.abspath(mean_std_path) != os.path.abspath(local_mean_std):
        with open(mean_std_path) as src, open(local_mean_std, "w") as dst:
            dst.write(src.read())

    fname = os.path.basename(input_pcap)
    t0 = time.time()

    _step(1, 3, f"Extracting nPrint features  {dim(fname)}")
    nprint_bits, meta_df = extract_packet_features(
        input_pcap, mask_prefixes=mask_prefixes,
    )
    t1 = time.time()
    print(f"      {green('✓')} {len(meta_df):,} packets  {dim(f'({t1 - t0:.1f}s)')}")

    _step(2, 3, "Grouping by 5-tuple flow")
    meta_df, flow_indices, sort_order = group_by_flow(meta_df)
    nprint_bits = nprint_bits[sort_order]
    del sort_order
    t2 = time.time()
    print(f"      {green('✓')} {len(flow_indices):,} flows  {dim(f'({t2 - t1:.1f}s)')}")

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
    _step(3, 3, f"Building ragged tensors  {dim(f'(rf cadence: {cadence_str})')}")
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
