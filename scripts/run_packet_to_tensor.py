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
    python scripts/run_packet_to_tensor.py \\
        --input data/CIC-IDS-2017/pcap/Monday-WorkingHours.pcap \\
        --output-dir data/processed/monday/

    # With custom window size (e.g., N=20):
    python scripts/run_packet_to_tensor.py \\
        --input data/CIC-IDS-2017/pcap/Monday-WorkingHours.pcap \\
        --output-dir data/processed/monday_n20/ \\
        --window-size 20

    # Disable identity masking:
    python scripts/run_packet_to_tensor.py \\
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

# Allow running from project root: `python scripts/run_packet_to_tensor.py`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.feature_extraction.packet_parser import extract_packet_features
from src.feature_extraction.flow_grouper import group_by_flow
from src.feature_extraction.tensor_builder import build_tensors


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="ADEPT Packet-to-Tensor pipeline (nPrint encoding)"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Input PCAP file"
    )
    parser.add_argument(
        "-o", "--output-dir", required=True,
        help="Output directory for tensors and metadata"
    )
    parser.add_argument(
        "-c", "--config",
        default=os.path.join(os.path.dirname(__file__), "..", "config", "pipeline.yaml"),
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
    args = parser.parse_args()

    # --- Load config ---------------------------------------------------------
    cfg = load_config(args.config)
    window_size = args.window_size or cfg["window_size"]
    stride = cfg.get("window_stride", window_size)
    rf_base = cfg["rf_base_features"]
    rf_agg = cfg["rf_agg_functions"]
    std_cols = cfg["standardize_columns"]
    mask_prefixes = [] if args.no_mask else cfg.get("mask_prefixes", [])

    os.makedirs(args.output_dir, exist_ok=True)

    mean_std_path = args.mean_std or os.path.join(
        args.output_dir, cfg["mean_std_filename"]
    )

    # --- Step 1: nPrint extraction -------------------------------------------
    t0 = time.time()

    nprint_bits, meta_df = extract_packet_features(
        args.input, mask_prefixes=mask_prefixes
    )

    t1 = time.time()
    print(f"      {len(meta_df)} packets in {t1 - t0:.1f}s")

    # --- Step 2: Flow grouping -----------------------------------------------
    print(f"[2/3] Grouping packets by 5-tuple flow...")
    meta_df, flow_indices, sort_order = group_by_flow(meta_df)
    # Apply the same timestamp sort to the nPrint array
    nprint_bits = nprint_bits[sort_order]
    del sort_order

    t2 = time.time()
    print(f"      {len(flow_indices)} flows in {t2 - t1:.1f}s")

    # Print flow size distribution
    sizes = [len(idx) for idx in flow_indices.values()]
    print(f"      Flow sizes: min={min(sizes)}, median={np.median(sizes):.0f}, "
          f"max={max(sizes)}, >=N({window_size}): "
          f"{sum(1 for s in sizes if s >= window_size)}/{len(sizes)}")

    # --- Step 3: Build tensors -----------------------------------------------
    print(f"[3/3] Building tensors (N={window_size}, stride={stride})...")
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
    print(f"      Tensor build in {t3 - t2:.1f}s")

    # sequence.npy and rf_stats.npy are already on disk (memory-mapped).
    # Only flow_ids and metadata.json still need to be written.
    n_windows = result["num_windows"]
    np.save(os.path.join(args.output_dir, "flow_ids.npy"),
            np.array(result["flow_ids"], dtype=object), allow_pickle=True)

    seq_shape = (n_windows, window_size, len(result["feature_names"]))
    rf_shape = (n_windows, len(result["rf_stat_names"]))
    metadata = {
        "input": args.input,
        "encoding": "nprint",
        "nprint_bits": 1088,
        "identity_masking": not args.no_mask,
        "mask_prefixes": mask_prefixes,
        "window_size": window_size,
        "stride": stride,
        "num_windows": n_windows,
        "sequence_shape": list(seq_shape),
        "rf_stats_shape": list(rf_shape),
        "sequence_feature_count": len(result["feature_names"]),
        "rf_stat_names": result["rf_stat_names"],
        "normalization_stats": result["stats"],
        "num_flows_total": len(flow_indices),
        "num_flows_used": len(set(result["flow_ids"])),
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # --- Summary -------------------------------------------------------------
    total = t3 - t0
    print(f"\n{'='*60}")
    print(f" ADEPT Packet-to-Tensor (nPrint) — Done ({total:.1f}s)")
    print(f"{'='*60}")
    print(f" Sequence tensor : {seq_shape}  → TCN / Transformer")
    print(f" RF stats tensor : {rf_shape}  → Random Forest")
    print(f" Identity masking: {'ON' if not args.no_mask else 'OFF'}")
    print(f" Output dir      : {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
