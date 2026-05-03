#!/usr/bin/env python3
"""
ADEPT — Window Label Attachment
================================

Produces <tensor-dir>/window_labels.csv.

ADEPT is a defense against adversarially-perturbed traffic: the purifier
and ensemble are trained to distinguish the benign manifold from TANTGen-
generated perturbations. Under this framing, the "attack" traffic that
ships with CIC-IDS-2017 and Kitsune is already part of the benign manifold
— the NIDS upstream is expected to catch it — and the only novel threat
is the adversarial variant produced by TANTGen.

Labels are therefore driven purely by the source dataset:

  kitsune      → label=0, label_str="benign"
  cic          → label=0, label_str="benign"
  adversarial  → label=1, label_str="adversarial"

No ground-truth CSV or PCAP re-read is needed; `num_windows` is read from
metadata.json produced by the pipeline.

Output columns: window_idx, label (0=benign, 1=adversarial), label_str.

Usage
-----
    # Kitsune / CIC-IDS2017 tensor directory → all benign
    python scripts/packet_to_tensor/attach_labels.py \\
        --dataset kitsune \\
        --tensor-dir data/tensors/mirai/

    python scripts/packet_to_tensor/attach_labels.py \\
        --dataset cic \\
        --tensor-dir data/tensors/ddos/

    # TANTGen adversarial tensor directory → all adversarial
    python scripts/packet_to_tensor/attach_labels.py \\
        --dataset adversarial \\
        --tensor-dir data/tensors/adversarial/mirai_adv_both/
"""

import argparse
import json
import sys
from pathlib import Path

OUTPUT_FILENAME = "window_labels.csv"

# Dataset → (label, label_str).
LABEL_FOR_DATASET = {
    "kitsune": (0, "benign"),
    "cic": (0, "benign"),
    "adversarial": (1, "adversarial"),
}


def write_window_labels(tensor_dir: Path, label: int, label_str: str) -> int:
    """Write window_labels.csv with `num_windows` uniform rows. Returns count."""
    meta_path = tensor_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found in {tensor_dir}. Run the pipeline first.")
    with open(meta_path) as f:
        meta = json.load(f)
    n = int(meta["num_windows"])
    out = tensor_dir / OUTPUT_FILENAME
    with open(out, "w") as f:
        f.write("window_idx,label,label_str\n")
        for i in range(n):
            f.write(f"{i},{label},{label_str}\n")
    return n


def main():
    parser = argparse.ArgumentParser(
        description="Write window_labels.csv using ADEPT's dataset-driven "
                    "label convention (benign vs adversarial).")
    parser.add_argument("-d", "--dataset", required=True,
                        choices=sorted(LABEL_FOR_DATASET))
    parser.add_argument("-t", "--tensor-dir", required=True)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing window_labels.csv")
    args = parser.parse_args()

    tensor_dir = Path(args.tensor_dir)
    out_path = tensor_dir / OUTPUT_FILENAME

    if not (tensor_dir / "metadata.json").exists():
        print(f"ERROR: metadata.json not found in {tensor_dir}. "
              f"Run the pipeline first.")
        sys.exit(1)
    if out_path.exists() and not args.force:
        print(f"Already labelled ({out_path}). Use --force to overwrite.")
        sys.exit(0)

    label, label_str = LABEL_FOR_DATASET[args.dataset]
    n = write_window_labels(tensor_dir, label, label_str)
    print(f"  Labelled {n:,} windows — {label_str}")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
