#!/usr/bin/env python3
"""
ADEPT — Packet-to-Tensor Pipeline (streaming-native edition)
=============================================================

End-to-end script that converts a raw PCAP into the ragged per-flow tensor
layout consumed by the streaming-native ADEPT ensemble:

    1. flow_packets.npy            [P_total, F=1090] float32
       Per-packet nPrint bits + IPAT (z-score) + Direction, concatenated
       across all flows in flow-arrival order.
    2. flow_offsets.npy            [N_flows, 2]      int64
       (start_idx, length) into flow_packets.npy per flow.
    3. rf_checkpoints.npy          [C_total, S=24]   float32
       Aggregated statistics over the growing prefix of each flow, emitted
       every `rf_checkpoint_stride` packets (default 10).
    4. rf_checkpoint_offsets.npy   [N_flows, 2]      int64
       (start_idx, n_checkpoints) into rf_checkpoints.npy per flow.

Plus:
    flow_ids.npy      — per-flow 5-tuple keys (object array)
    metadata.json     — config, shapes, normalisation stats
    mean_std.json     — IPAT normalisation parameters

Usage
-----
    python scripts/packet_to_tensor/run_packet_to_tensor.py \\
        --input data/CIC-IDS-2017/pcap/Monday-WorkingHours.pcap \\
        --output-dir data/tensors/monday/

    # With custom RF checkpoint stride (e.g., every 20 packets):
    python scripts/packet_to_tensor/run_packet_to_tensor.py \\
        --input data/Kitsune/ddos.pcap \\
        --output-dir data/tensors/ddos_s20/ \\
        --rf-checkpoint-stride 20

    # Disable identity masking:
    python scripts/packet_to_tensor/run_packet_to_tensor.py \\
        --input data/CIC-IDS-2017/pcap/Monday-WorkingHours.pcap \\
        --output-dir data/tensors/monday_unmasked/ \\
        --no-mask
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

# Allow running from project root: `python scripts/packet_to_tensor/run_packet_to_tensor.py`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.feature_extraction.pcap_parser import extract_packet_features
from src.feature_extraction.flow_grouper import group_by_flow
from src.feature_extraction.tensor_builder import build_tensors
from src.feature_extraction.incremental_pipeline import build_tensors_incremental


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


def _fit_mean_std_streaming(pcap_paths, columns, output_path, mask_prefixes,
                            workers=None):
    """Fit per-column mean/std across multiple PCAPs and write JSON."""
    accumulators = {c: {"count": 0, "mean": 0.0, "M2": 0.0} for c in columns}
    for i, pcap in enumerate(pcap_paths, 1):
        print(f"\n  {dim(f'[{i}/{len(pcap_paths)}]')} {os.path.basename(pcap)}")
        nprint_bits, meta_df = extract_packet_features(
            pcap, mask_prefixes=mask_prefixes, workers=workers,
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

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


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


def main():
    parser = argparse.ArgumentParser(
        description="ADEPT Packet-to-Tensor pipeline (streaming-native, nPrint encoding)"
    )
    parser.add_argument(
        "-i", "--input", nargs="+", required=True,
        help="Input PCAP file(s). With --fit-mean-std, accepts multiple paths.",
    )
    parser.add_argument(
        "--fit-mean-std", action="store_true",
        help="Fit-only mode: stream one or more PCAPs (-i ...), accumulate "
             "mean/std for the columns listed in pipeline.yaml, and write "
             "the result to -o (treated as a JSON file path). No tensors "
             "are produced.",
    )
    parser.add_argument(
        "-o", "--output-dir", required=True,
        help="Output directory for tensors and metadata"
    )
    parser.add_argument(
        "-c", "--config",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "config", "pipeline.yaml"),
        help="Pipeline config YAML (default: config/pipeline.yaml)"
    )
    parser.add_argument(
        "--rf-checkpoint-stride", type=int, default=None,
        help="Emit an RF feature row every K packets per flow (override config)"
    )
    parser.add_argument(
        "--mean-std", default=None,
        help="Path to existing mean_std.json (for test sets — use training stats)"
    )
    parser.add_argument(
        "--no-mask", action="store_true",
        help="Disable identity masking (keep IPs, ports, IP ID in nPrint bits)"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Parallel encoding workers (default: all CPU cores)"
    )
    parser.add_argument(
        "--online", action="store_true",
        help=(
            "Incremental (online) mode: no global timestamp sort. "
            "Two lightweight PCAP passes; lower peak RAM than batch mode."
        )
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    rf_checkpoint_stride = (
        args.rf_checkpoint_stride
        if args.rf_checkpoint_stride is not None
        else int(cfg.get("rf_checkpoint_stride", 10))
    )
    rf_base = cfg["rf_base_features"]
    rf_agg = cfg["rf_agg_functions"]
    std_cols = cfg["standardize_columns"]
    mask_prefixes = [] if args.no_mask else cfg.get("mask_prefixes", [])

    # ── Fit-only mode (multi-PCAP shared standardizer) ──────────────────────
    if args.fit_mean_std:
        print(f"\n{bold('Fitting shared mean/std')}  {dim(f'over {len(args.input)} PCAP(s)')}")
        print(f"  Columns: {std_cols}")
        print(f"  Output:  {args.output_dir}")
        t0 = time.time()
        stats = _fit_mean_std_streaming(
            pcap_paths=args.input,
            columns=std_cols,
            output_path=args.output_dir,
            mask_prefixes=mask_prefixes,
            workers=args.workers,
        )
        print(f"\n{green('✓')} Wrote {args.output_dir}  {dim(f'({time.time() - t0:.1f}s)')}")
        for c, s in stats.items():
            print(f"  {cyan(c):<20}  mean={s['mean']:.6f}  std={s['std']:.6f}")
        return

    if len(args.input) != 1:
        parser.error("In tensor-building mode, -i takes exactly one PCAP. "
                     "Use --fit-mean-std for multi-PCAP standardizer fitting.")
    input_pcap = args.input[0]

    os.makedirs(args.output_dir, exist_ok=True)

    mean_std_path = args.mean_std or os.path.join(
        args.output_dir, cfg["mean_std_filename"]
    )

    if args.mean_std and os.path.abspath(args.mean_std) != os.path.abspath(
        os.path.join(args.output_dir, cfg["mean_std_filename"])
    ):
        local_copy = os.path.join(args.output_dir, cfg["mean_std_filename"])
        with open(args.mean_std) as src, open(local_copy, "w") as dst:
            dst.write(src.read())

    fname = os.path.basename(input_pcap)
    t0 = time.time()

    if not args.online:
        _step(1, 3, f"Extracting nPrint features  {dim(fname)}")
        nprint_bits, meta_df = extract_packet_features(
            input_pcap, mask_prefixes=mask_prefixes, workers=args.workers
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
            usable = sum(1 for s in sizes if s >= rf_checkpoint_stride)
            print(f"      Flow sizes: min={min(sizes)}, "
                  f"median={np.median(sizes):.0f}, "
                  f"max={max(sizes)} — "
                  f"{cyan(str(usable))}/{len(sizes)} have ≥{rf_checkpoint_stride} packets")

        _step(3, 3, f"Building ragged tensors  "
                    f"{dim(f'(rf_checkpoint_stride={rf_checkpoint_stride})')}")
        result = build_tensors(
            nprint_bits=nprint_bits,
            meta_df=meta_df,
            flow_indices=flow_indices,
            output_dir=args.output_dir,
            rf_checkpoint_stride=rf_checkpoint_stride,
            rf_base_features=rf_base,
            rf_agg_functions=rf_agg,
            standardize_columns=std_cols,
            mean_std_path=mean_std_path,
        )
        del nprint_bits, meta_df
        t3 = time.time()
        print(f"      {green('✓')} {result['num_flows']:,} flows, "
              f"{result['num_packets']:,} packets, "
              f"{result['num_checkpoints']:,} RF checkpoints  "
              f"{dim(f'({t3 - t2:.1f}s)')}")

    else:
        _step(1, 2, f"Prescan  {dim(fname)}")
        _step(2, 2, f"Encoding + building ragged tensors  "
                    f"{dim(f'(rf_checkpoint_stride={rf_checkpoint_stride})')}")
        result = build_tensors_incremental(
            pcap_path=input_pcap,
            output_dir=args.output_dir,
            rf_checkpoint_stride=rf_checkpoint_stride,
            rf_base_features=rf_base,
            rf_agg_functions=rf_agg,
            mean_std_path=mean_std_path,
            mask_prefixes=mask_prefixes,
        )
        t3 = time.time()
        print(f"      {green('✓')} {result['num_flows']:,} flows, "
              f"{result['num_packets']:,} packets, "
              f"{result['num_checkpoints']:,} RF checkpoints  "
              f"{dim(f'({t3 - t0:.1f}s)')}")

    # ── Common: write flow_ids.npy and metadata.json ──────────────────────────
    np.save(os.path.join(args.output_dir, "flow_ids.npy"),
            np.array(result["flow_ids"], dtype=object), allow_pickle=True)

    n_flows = result["num_flows"]
    n_packets = result["num_packets"]
    n_ckpts = result["num_checkpoints"]
    metadata = {
        "input":                    input_pcap,
        "encoding":                 "nprint",
        "pipeline_mode":            "online" if args.online else "batch",
        "schema":                   "ragged-per-flow-v1",
        "nprint_bits":              1088,
        "identity_masking":         not args.no_mask,
        "mask_prefixes":            mask_prefixes,
        "rf_checkpoint_stride":     rf_checkpoint_stride,
        "num_flows":                n_flows,
        "num_packets":              n_packets,
        "num_checkpoints":          n_ckpts,
        "flow_packets_shape":       [n_packets, len(result["feature_names"])],
        "rf_checkpoints_shape":     [n_ckpts, len(result["rf_stat_names"])],
        "sequence_feature_count":   len(result["feature_names"]),
        "rf_stat_names":            result["rf_stat_names"],
        "normalization_stats":      result["stats"],
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    total = t3 - t0
    W = 62
    masking_str = green("ON") if not args.no_mask else yellow("OFF")
    print(f"\n{bold('═' * W)}")
    print(f" {bold('ADEPT Packet-to-Tensor (streaming)')}  {dim(f'Done in {total:.1f}s')}")
    print(f"{'─' * W}")
    print(f" {cyan('flow_packets')}     [{n_packets}, {len(result['feature_names'])}]  "
          f"{dim('TCN / Transformer (per-flow ragged)')}")
    print(f" {cyan('rf_checkpoints')}   [{n_ckpts}, {len(result['rf_stat_names'])}]  "
          f"{dim(f'Random Forest (stride={rf_checkpoint_stride})')}")
    print(f" {cyan('Identity masking')} {masking_str}")
    print(f" {cyan('Output')}           {args.output_dir}")
    print(f"{bold('═' * W)}\n")


if __name__ == "__main__":
    main()
