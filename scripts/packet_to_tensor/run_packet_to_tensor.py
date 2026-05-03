#!/usr/bin/env python3
"""
ADEPT — Packet-to-Tensor Pipeline (nPrint Edition)
====================================================

End-to-end script that converts a raw PCAP into the two synchronized tensor
outputs consumed by the ADEPT ensemble:

    1. Sequence tensor  [W, N, F≈1090]  → TCN & Purification Transformer
       (nPrint bit-level header encoding + IPAT + Direction)
    2. RF statistics    [W, S=24]       → Random Forest
       (aggregated from Scapy-derived metadata)

Uses nPrint-compatible bit-level encoding (Holland et al., CCS 2021),
with selective identity masking inspired by FlowPaint (SIGCOMM'26).

Usage
-----
    python scripts/packet_to_tensor/run_packet_to_tensor.py \\
        --input data/CIC-IDS-2017/pcap/Monday-WorkingHours.pcap \\
        --output-dir data/processed/monday/

    # With custom window size (e.g., N=20):
    python scripts/packet_to_tensor/run_packet_to_tensor.py \\
        --input data/CIC-IDS-2017/pcap/Monday-WorkingHours.pcap \\
        --output-dir data/processed/monday_n20/ \\
        --window-size 20

    # Disable identity masking:
    python scripts/packet_to_tensor/run_packet_to_tensor.py \\
        --input data/CIC-IDS-2017/pcap/Monday-WorkingHours.pcap \\
        --output-dir data/processed/monday_unmasked/ \\
        --no-mask

Outputs
-------
    <output-dir>/
        sequence.npy       — [W, N, F≈1090] float32
        rf_stats.npy       — [W, S=24]      float32
        flow_ids.npy       — [W]            object (5-tuple per window)
        mean_std.json      — normalization statistics
        metadata.json      — pipeline config & tensor shapes
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
    """Fold a numpy batch of observations into a (count, mean, M2) accumulator.

    Uses Chan's parallel-Welford formula so per-PCAP batches are merged into
    one global accumulator without holding all values in memory at once.
    """
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
    """Fit per-column mean/std across multiple PCAPs and write JSON.

    Streams one PCAP at a time: extracts metadata + IPAT via the existing
    extract_packet_features → group_by_flow path, accumulates with Welford,
    discards arrays before moving to the next PCAP. nprint_bits is dropped
    immediately to bound peak RAM.
    """
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
    """Print a coloured step header."""
    print(f"\n{dim(f'[{n}/{total}]')} {bold(msg)}")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="ADEPT Packet-to-Tensor pipeline (nPrint encoding)"
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
             "are produced. Used to derive a shared standardizer over "
             "benign + adversarial inputs before training.",
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
        "-N", "--window-size", type=int, default=None,
        help="Override N-Packet window size from config"
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
            "Emits each window the moment N packets arrive for a flow. "
            "Lower peak RAM than batch mode; uses two lightweight PCAP passes."
        )
    )
    args = parser.parse_args()

    # --- Load config ---------------------------------------------------------
    cfg = load_config(args.config)
    window_size = args.window_size or cfg["window_size"]
    stride = cfg.get("window_stride", window_size)
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
            output_path=args.output_dir,  # treated as JSON file path
            mask_prefixes=mask_prefixes,
            workers=args.workers,
        )
        print(f"\n{green('✓')} Wrote {args.output_dir}  {dim(f'({time.time() - t0:.1f}s)')}")
        for c, s in stats.items():
            print(f"  {cyan(c):<20}  mean={s['mean']:.6f}  std={s['std']:.6f}")
        return

    # ── Tensor-building modes require exactly one input PCAP ────────────────
    if len(args.input) != 1:
        parser.error("In tensor-building mode, -i takes exactly one PCAP. "
                     "Use --fit-mean-std for multi-PCAP standardizer fitting.")
    input_pcap = args.input[0]

    os.makedirs(args.output_dir, exist_ok=True)

    mean_std_path = args.mean_std or os.path.join(
        args.output_dir, cfg["mean_std_filename"]
    )

    # When a shared standardizer is supplied, mirror it into the local
    # output dir so per-attack consumers can find mean_std.json next to
    # their tensors (downstream code that looks up the per-output copy
    # keeps working unchanged).
    if args.mean_std and os.path.abspath(args.mean_std) != os.path.abspath(
        os.path.join(args.output_dir, cfg["mean_std_filename"])
    ):
        local_copy = os.path.join(args.output_dir, cfg["mean_std_filename"])
        with open(args.mean_std) as src, open(local_copy, "w") as dst:
            dst.write(src.read())

    fname = os.path.basename(input_pcap)
    t0 = time.time()

    # ── Pipeline branch: batch (default) or online (incremental) ─────────────

    if not args.online:
        # ── Batch: 3-stage pipeline ───────────────────────────────────────
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
        usable = sum(1 for s in sizes if s >= window_size)
        print(f"      Flow sizes: min={min(sizes)}, "
              f"median={np.median(sizes):.0f}, "
              f"max={max(sizes)} — "
              f"{cyan(str(usable))}/{len(sizes)} usable (≥N={window_size})")

        _step(3, 3, f"Building tensors  {dim(f'(N={window_size}, stride={stride})')}")
        result = build_tensors(
            nprint_bits=nprint_bits,
            meta_df=meta_df,
            flow_indices=flow_indices,
            output_dir=args.output_dir,
            window_size=window_size,
            stride=stride,
            rf_base_features=rf_base,
            rf_agg_functions=rf_agg,
            standardize_columns=std_cols,
            mean_std_path=mean_std_path,
        )
        del nprint_bits, meta_df
        t3 = time.time()
        print(f"      {green('✓')} {result['num_windows']:,} windows  {dim(f'({t3 - t2:.1f}s)')}")

        num_flows_total = len(flow_indices)
        num_flows_used  = len(set(result["flow_ids"]))

    else:
        # ── Online: incremental 2-pass pipeline ───────────────────────────
        _step(1, 2, f"Prescan  {dim(fname)}")
        _step(2, 2, f"Encoding + building tensors  "
                    f"{dim(f'(N={window_size}, stride={stride})')}")
        result = build_tensors_incremental(
            pcap_path=input_pcap,
            output_dir=args.output_dir,
            window_size=window_size,
            stride=stride,
            rf_base_features=rf_base,
            rf_agg_functions=rf_agg,
            mean_std_path=mean_std_path,
            mask_prefixes=mask_prefixes,
        )
        t3 = time.time()
        print(f"      {green('✓')} {result['num_windows']:,} windows  "
              f"{dim(f'({t3 - t0:.1f}s)')}")

        num_flows_total = result.get("num_flows_total", len(set(result["flow_ids"])))
        num_flows_used  = len(set(result["flow_ids"]))

    # ── Common: write flow_ids.npy and metadata.json ──────────────────────────
    n_windows = result["num_windows"]
    np.save(os.path.join(args.output_dir, "flow_ids.npy"),
            np.array(result["flow_ids"], dtype=object), allow_pickle=True)

    seq_shape = (n_windows, window_size, len(result["feature_names"]))
    rf_shape  = (n_windows, len(result["rf_stat_names"]))
    metadata  = {
        "input":                    input_pcap,
        "encoding":                 "nprint",
        "pipeline_mode":            "online" if args.online else "batch",
        "nprint_bits":              1088,
        "identity_masking":         not args.no_mask,
        "mask_prefixes":            mask_prefixes,
        "window_size":              window_size,
        "stride":                   stride,
        "num_windows":              n_windows,
        "sequence_shape":           list(seq_shape),
        "rf_stats_shape":           list(rf_shape),
        "sequence_feature_count":   len(result["feature_names"]),
        "rf_stat_names":            result["rf_stat_names"],
        "normalization_stats":      result["stats"],
        "num_flows_total":          num_flows_total,
        "num_flows_used":           num_flows_used,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # --- Summary -------------------------------------------------------------
    total = t3 - t0
    W = 62
    masking_str = green("ON") if not args.no_mask else yellow("OFF")
    print(f"\n{bold('═' * W)}")
    print(f" {bold('ADEPT Packet-to-Tensor')}  {dim(f'Done in {total:.1f}s')}")
    print(f"{'─' * W}")
    print(f" {cyan('Sequence tensor')}  {str(seq_shape):<28} {dim('TCN / Transformer')}")
    print(f" {cyan('RF stats tensor')}  {str(rf_shape):<28} {dim('Random Forest')}")
    print(f" {cyan('Identity masking')} {masking_str}")
    print(f" {cyan('Output')}           {args.output_dir}")
    print(f"{bold('═' * W)}\n")


if __name__ == "__main__":
    main()
