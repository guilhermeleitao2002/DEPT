#!/usr/bin/env python3
"""
ADEPT — Packet-to-Tensor Pipeline
==================================

End-to-end script that converts a raw PCAP (or pre-extracted CSV) into the
two synchronized tensor outputs consumed by the ADEPT ensemble:

    1. Sequence tensor  [W, N, F]  → TCN & Purification Transformer
    2. RF statistics    [W, S]     → Random Forest

Usage
-----
    # From PCAP:
    python scripts/run_packet_to_tensor.py \\
        --input data/raw/Wednesday-workingHours.pcap \\
        --output-dir data/processed/wednesday/

    # From pre-extracted CSV (skip PCAP parsing):
    python scripts/run_packet_to_tensor.py \\
        --input data/features/wednesday_packets.csv \\
        --output-dir data/processed/wednesday/ \\
        --from-csv

    # With custom window size (e.g., N=20):
    python scripts/run_packet_to_tensor.py \\
        --input data/raw/capture.pcap \\
        --output-dir data/processed/capture/ \\
        --window-size 20

Outputs
-------
    <output-dir>/
        sequence.npy       — [W, N, F] float32
        rf_stats.npy       — [W, S]    float32
        flow_ids.npy       — [W]       object (5-tuple per window)
        mean_std.json      — normalization statistics
        packets.csv        — intermediate per-packet features (if from PCAP)
        metadata.json      — pipeline config & tensor shapes
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
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
        description="ADEPT Packet-to-Tensor pipeline"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Input PCAP file or CSV (if --from-csv)"
    )
    parser.add_argument(
        "-o", "--output-dir", required=True,
        help="Output directory for tensors and metadata"
    )
    parser.add_argument(
        "--from-csv", action="store_true",
        help="Input is a pre-extracted CSV (skip PCAP parsing)"
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
    args = parser.parse_args()

    # --- Load config ---------------------------------------------------------
    cfg = load_config(args.config)
    window_size = args.window_size or cfg["window_size"]
    stride = cfg.get("window_stride", window_size)
    seq_features = cfg["sequence_features"]
    rf_base = cfg["rf_base_features"]
    rf_agg = cfg["rf_agg_functions"]
    std_cols = cfg["standardize_columns"]

    os.makedirs(args.output_dir, exist_ok=True)

    mean_std_path = args.mean_std or os.path.join(
        args.output_dir, cfg["mean_std_filename"]
    )

    # --- Step 1: Packet features ---------------------------------------------
    t0 = time.time()

    if args.from_csv:
        print(f"[1/3] Loading pre-extracted CSV: {args.input}")
        df = pd.read_csv(args.input)
    else:
        print(f"[1/3] Extracting packet features from PCAP: {args.input}")
        df = extract_packet_features(args.input)
        csv_path = os.path.join(args.output_dir, "packets.csv")
        df.to_csv(csv_path, index=False)
        print(f"      Saved intermediate CSV: {csv_path}")

    t1 = time.time()
    print(f"      {len(df)} packets in {t1 - t0:.1f}s")

    # --- Step 2: Flow grouping -----------------------------------------------
    print(f"[2/3] Grouping packets by 5-tuple flow...")
    flows = group_by_flow(df)
    t2 = time.time()
    print(f"      {len(flows)} flows in {t2 - t1:.1f}s")

    # Print flow size distribution
    sizes = [len(f) for f in flows.values()]
    print(f"      Flow sizes: min={min(sizes)}, median={np.median(sizes):.0f}, "
          f"max={max(sizes)}, >=N({window_size}): "
          f"{sum(1 for s in sizes if s >= window_size)}/{len(sizes)}")

    # --- Step 3: Build tensors -----------------------------------------------
    print(f"[3/3] Building tensors (N={window_size}, stride={stride})...")
    result = build_tensors(
        flows=flows,
        window_size=window_size,
        stride=stride,
        sequence_features=seq_features,
        rf_base_features=rf_base,
        rf_agg_functions=rf_agg,
        standardize_columns=std_cols,
        mean_std_path=mean_std_path,
    )
    t3 = time.time()
    print(f"      Tensor build in {t3 - t2:.1f}s")

    # --- Save outputs --------------------------------------------------------
    np.save(os.path.join(args.output_dir, "sequence.npy"), result["sequence"])
    np.save(os.path.join(args.output_dir, "rf_stats.npy"), result["rf_stats"])
    np.save(os.path.join(args.output_dir, "flow_ids.npy"),
            np.array(result["flow_ids"], dtype=object), allow_pickle=True)

    metadata = {
        "input": args.input,
        "window_size": window_size,
        "stride": stride,
        "num_windows": int(result["sequence"].shape[0]),
        "sequence_shape": list(result["sequence"].shape),
        "rf_stats_shape": list(result["rf_stats"].shape),
        "sequence_features": result["feature_names"],
        "rf_stat_names": result["rf_stat_names"],
        "normalization_stats": result["stats"],
        "num_flows_total": len(flows),
        "num_flows_used": len(set(result["flow_ids"])),
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # --- Summary -------------------------------------------------------------
    total = t3 - t0
    print(f"\n{'='*60}")
    print(f" ADEPT Packet-to-Tensor — Done ({total:.1f}s)")
    print(f"{'='*60}")
    print(f" Sequence tensor : {result['sequence'].shape}  → TCN / Transformer")
    print(f" RF stats tensor : {result['rf_stats'].shape}  → Random Forest")
    print(f" Output dir      : {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
