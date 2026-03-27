#!/usr/bin/env python3
"""
ADEPT — Batch Processing for Selected Attack Datasets
======================================================

Automates the Packet-to-Tensor pipeline for all 7 attacks selected for
ADEPT evaluation, plus the Monday benign baseline.

Datasets and rationale (see plan for full justification):

  Tier 1 — Kitsune (TANTGen adversarial results available):
    1. Mirai          — Botnet C&C / IoT DDoS          [already processed]
    2. SYN DoS        — Volumetric DoS (TCP)
    3. Fuzzing        — Stealthy protocol exploitation
    4. SSDP Flood     — Volumetric amplification (UDP)

  Tier 2 — CIC-IDS2017 (cross-dataset generalization):
    5. DDoS           — Distributed DoS (Friday 19:30-21:05)
    6. DoS            — Low-and-slow + volumetric (Wednesday 09:20–12:00)
    7. Port Scan      — Reconnaissance (Friday 17:00-19:30)

  Benign baseline:
    8. Monday benign   — Enterprise benign manifold (sampled)

CIC-IDS2017 PCAPs are time-filtered with editcap before processing to
reduce size and isolate attack windows.

Usage
-----
    # Process all attacks (skips Mirai by default since it's already done):
    python scripts/process_all_attacks.py

    # Process a specific attack:
    python scripts/process_all_attacks.py --only syn_dos

    # Re-process Mirai (force overwrite):
    python scripts/process_all_attacks.py --only mirai --force

    # Dry run — show what would be processed without running:
    python scripts/process_all_attacks.py --dry-run

    # Custom window size:
    python scripts/process_all_attacks.py --window-size 20

    # Process with benign baseline:
    python scripts/process_all_attacks.py --include-benign
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root = ADEPT/
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PIPELINE_SCRIPT = SCRIPT_DIR / "run_packet_to_tensor.py"
PIPELINE_CONFIG = PROJECT_ROOT / "config" / "pipeline.yaml"

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
KITSUNE_PCAP = DATA_DIR / "Kitsune" / "pcap"
CIC_PCAP = DATA_DIR / "CIC-IDS-2017" / "pcap"
TENSOR_DIR = DATA_DIR / "tensors"


# ---------------------------------------------------------------------------
# Attack definitions
# ---------------------------------------------------------------------------
@dataclass
class Attack:
    """Defines one attack to process."""
    key: str                        # Short identifier (used with --only)
    name: str                       # Human-readable name
    dataset: str                    # "kitsune" or "cic-ids2017"
    pcap_path: Path                 # Path to raw PCAP
    output_dir: Path                # Tensor output directory
    # CIC-IDS2017 time-filtering (editcap -A / -B)
    time_start: str = ""            # e.g. "2017-07-05 09:20:00"
    time_end: str = ""              # e.g. "2017-07-05 12:00:00"
    extra_args: list = field(default_factory=list)


ATTACKS = [
    # --- Tier 1: Kitsune ---------------------------------------------------
    Attack(
        key="mirai",
        name="Mirai Botnet",
        dataset="kitsune",
        pcap_path=KITSUNE_PCAP / "Mirai_pcap.pcap",
        output_dir=TENSOR_DIR / "mirai",
    ),
    Attack(
        key="syn_dos",
        name="SYN DoS",
        dataset="kitsune",
        pcap_path=KITSUNE_PCAP / "SYN_DoS_pcap.pcap",
        output_dir=TENSOR_DIR / "syn_dos",
    ),
    Attack(
        key="fuzzing",
        name="Fuzzing",
        dataset="kitsune",
        pcap_path=KITSUNE_PCAP / "Fuzzing_pcap.pcapng",
        output_dir=TENSOR_DIR / "fuzzing",
    ),
    Attack(
        key="ssdp_flood",
        name="SSDP Flood",
        dataset="kitsune",
        pcap_path=KITSUNE_PCAP / "SSDP_Flood_pcap.pcap",
        output_dir=TENSOR_DIR / "ssdp_flood",
    ),
    # --- Tier 2: CIC-IDS2017 -----------------------------------------------
    # Time windows from CIC-IDS2017 documentation and TANTGen run scripts.
    Attack(
        key="ddos",
        name="DDoS (CIC-IDS2017 Friday)",
        dataset="cic-ids2017",
        pcap_path=CIC_PCAP / "Friday-WorkingHours.pcap",
        output_dir=TENSOR_DIR / "ddos",
        time_start="2017-07-07 19:30:00",
        time_end="2017-07-07 21:05:00",
    ),
    Attack(
        key="dos_slow",
        name="DoS Slowloris+SlowHTTPTest+Hulk+GoldenEye (CIC-IDS2017 Wednesday)",
        dataset="cic-ids2017",
        pcap_path=CIC_PCAP / "Wednesday-workingHours.pcap",
        output_dir=TENSOR_DIR / "dos_slow",
        # Covers all 4 DoS sub-attacks:
        #   Slowloris     09:47–10:10
        #   SlowHTTPTest  10:14–10:35
        #   Hulk          10:43–11:00
        #   GoldenEye     11:10–11:23
        time_start="2017-07-05 09:20:00",
        time_end="2017-07-05 12:00:00",
    ),
    Attack(
        key="portscan",
        name="Port Scan (CIC-IDS2017 Friday)",
        dataset="cic-ids2017",
        pcap_path=CIC_PCAP / "Friday-WorkingHours.pcap",
        output_dir=TENSOR_DIR / "portscan",
        time_start="2017-07-07 17:00:00",
        time_end="2017-07-07 19:30:00",
    ),
    # --- Benign baseline ---------------------------------------------------
    Attack(
        key="monday_benign",
        name="Monday Benign (CIC-IDS2017)",
        dataset="cic-ids2017",
        pcap_path=CIC_PCAP / "Monday-WorkingHours.pcap",
        output_dir=TENSOR_DIR / "monday_benign",
        # Monday traffic runs roughly 09:00–17:00. Sample a 2-hour window
        # to keep tensor output manageable (~10K flows target).
        time_start="2017-07-03 09:00:00",
        time_end="2017-07-03 11:00:00",
    ),
]

ATTACK_KEYS = [a.key for a in ATTACKS]
# Keys that are not included by default (need --include-benign)
BENIGN_KEYS = {"monday_benign"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_editcap() -> str:
    """Return path to editcap or raise."""
    path = shutil.which("editcap")
    if path is None:
        print("ERROR: editcap not found. Install Wireshark CLI tools:")
        print("       sudo apt install wireshark-common")
        sys.exit(1)
    return path


def time_filter_pcap(editcap: str, src: Path, dst: Path,
                     start: str, end: str) -> None:
    """Use editcap to extract a time window from a PCAP."""
    cmd = [editcap, "-A", start, "-B", end, str(src), str(dst)]
    print(f"  editcap: {start} → {end}")
    print(f"  src: {src.name} → dst: {dst.name}")
    subprocess.run(cmd, check=True)
    size_mb = dst.stat().st_size / (1024 * 1024)
    print(f"  Filtered PCAP: {size_mb:.1f} MB")


def run_pipeline(pcap: Path, output_dir: Path, window_size: int | None,
                 extra_args: list) -> subprocess.CompletedProcess:
    """Run run_packet_to_tensor.py on a single PCAP."""
    cmd = [
        sys.executable, str(PIPELINE_SCRIPT),
        "-i", str(pcap),
        "-o", str(output_dir),
        "-c", str(PIPELINE_CONFIG),
    ]
    if window_size is not None:
        cmd += ["-N", str(window_size)]
    cmd += extra_args
    return subprocess.run(cmd, check=True)


def is_already_processed(output_dir: Path) -> bool:
    """Check if tensor outputs already exist."""
    return (output_dir / "metadata.json").exists()


def load_metadata(output_dir: Path) -> dict | None:
    """Load metadata.json from a processed attack."""
    meta_path = output_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return None


def print_summary(results: list[tuple[str, str, dict | None]]) -> None:
    """Print a summary table of all processed attacks."""
    print(f"\n{'=' * 72}")
    print(f" ADEPT Batch Processing — Summary")
    print(f"{'=' * 72}")
    print(f" {'Attack':<35} {'Status':<12} {'Windows':>10} {'Flows':>12}")
    print(f" {'-' * 35} {'-' * 12} {'-' * 10} {'-' * 12}")

    for name, status, meta in results:
        if meta:
            windows = f"{meta['num_windows']:,}"
            flows = f"{meta['num_flows_used']:,}/{meta['num_flows_total']:,}"
        else:
            windows = "—"
            flows = "—"
        print(f" {name:<35} {status:<12} {windows:>10} {flows:>12}")

    print(f"{'=' * 72}")
    print(f" Output directory: {TENSOR_DIR}")
    print(f"{'=' * 72}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Batch-process all selected attacks through the "
                    "ADEPT Packet-to-Tensor pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available attack keys: {', '.join(ATTACK_KEYS)}",
    )
    parser.add_argument(
        "--only", nargs="+", metavar="KEY",
        help=f"Process only these attacks (choices: {', '.join(ATTACK_KEYS)})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process attacks even if output already exists",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be processed without actually running",
    )
    parser.add_argument(
        "-N", "--window-size", type=int, default=None,
        help="Override N-Packet window size (default: from pipeline.yaml)",
    )
    parser.add_argument(
        "--include-benign", action="store_true",
        help="Also process the Monday benign baseline",
    )
    args = parser.parse_args()

    # Validate --only keys
    if args.only:
        for key in args.only:
            if key not in ATTACK_KEYS:
                parser.error(f"Unknown attack key: '{key}'. "
                             f"Choose from: {', '.join(ATTACK_KEYS)}")

    # Build list of attacks to process
    selected = []
    for attack in ATTACKS:
        # Skip benign unless explicitly requested
        if attack.key in BENIGN_KEYS and not args.include_benign:
            if not (args.only and attack.key in args.only):
                continue
        # Filter by --only
        if args.only and attack.key not in args.only:
            continue
        selected.append(attack)

    if not selected:
        print("Nothing to process. Check --only keys or add --include-benign.")
        sys.exit(0)

    print(f"\nADEPT Batch Processing — {len(selected)} attack(s) selected\n")
    for i, a in enumerate(selected, 1):
        tag = " [time-filtered]" if a.time_start else ""
        print(f"  {i}. {a.name} ({a.dataset}{tag})")
    print()

    if args.dry_run:
        print("Dry run — nothing will be processed.\n")
        for a in selected:
            exists = is_already_processed(a.output_dir)
            status = "EXISTS (skip)" if exists and not args.force else "WILL PROCESS"
            if exists and args.force:
                status = "EXISTS (force re-process)"
            print(f"  [{status}] {a.name}")
            print(f"    PCAP:   {a.pcap_path}")
            print(f"    Output: {a.output_dir}")
            if a.time_start:
                print(f"    Filter: {a.time_start} → {a.time_end}")
            print()
        sys.exit(0)

    # Check editcap availability if any CIC attacks are selected
    editcap = None
    if any(a.time_start for a in selected):
        editcap = find_editcap()

    # Process each attack
    results: list[tuple[str, str, dict | None]] = []
    total_start = time.time()

    for i, attack in enumerate(selected, 1):
        print(f"\n{'─' * 72}")
        print(f" [{i}/{len(selected)}] {attack.name}")
        print(f"{'─' * 72}")

        # Skip if already processed
        if is_already_processed(attack.output_dir) and not args.force:
            meta = load_metadata(attack.output_dir)
            print(f"  Already processed ({meta['num_windows']:,} windows). "
                  f"Use --force to re-process.")
            results.append((attack.name, "skipped", meta))
            continue

        # Verify source PCAP exists
        if not attack.pcap_path.exists():
            print(f"  ERROR: PCAP not found: {attack.pcap_path}")
            results.append((attack.name, "MISSING", None))
            continue

        t0 = time.time()
        pcap_to_process = attack.pcap_path

        try:
            # Time-filter CIC-IDS2017 PCAPs
            if attack.time_start:
                filtered_dir = TENSOR_DIR / ".filtered"
                filtered_dir.mkdir(parents=True, exist_ok=True)
                filtered_pcap = filtered_dir / f"{attack.key}_filtered.pcap"

                print(f"  Step 0: Time-filtering PCAP...")
                time_filter_pcap(
                    editcap, attack.pcap_path, filtered_pcap,
                    attack.time_start, attack.time_end,
                )
                pcap_to_process = filtered_pcap

            # Run the pipeline
            print(f"  Running packet-to-tensor pipeline...")
            run_pipeline(
                pcap=pcap_to_process,
                output_dir=attack.output_dir,
                window_size=args.window_size,
                extra_args=attack.extra_args,
            )

            meta = load_metadata(attack.output_dir)
            elapsed = time.time() - t0
            print(f"  Completed in {elapsed:.1f}s")
            results.append((attack.name, "OK", meta))

        except subprocess.CalledProcessError as e:
            elapsed = time.time() - t0
            print(f"  FAILED after {elapsed:.1f}s (exit code {e.returncode})")
            results.append((attack.name, "FAILED", None))

        finally:
            # Clean up filtered PCAP to save disk space
            if attack.time_start:
                filtered_pcap = TENSOR_DIR / ".filtered" / f"{attack.key}_filtered.pcap"
                if filtered_pcap.exists():
                    filtered_pcap.unlink()
                    print(f"  Cleaned up filtered PCAP")

    total_elapsed = time.time() - total_start

    # Summary
    print_summary(results)
    print(f"Total time: {total_elapsed:.1f}s "
          f"({total_elapsed / 60:.1f} min)\n")


if __name__ == "__main__":
    main()
