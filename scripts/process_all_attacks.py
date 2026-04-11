#!/usr/bin/env python3
"""
ADEPT — Batch Processing for All Attack Datasets
=================================================

Automates the Packet-to-Tensor pipeline for all available attack PCAPs,
plus the Monday benign baseline.

Datasets:

  Kitsune:
     1. Mirai             — Botnet C&C / IoT DDoS
     2. SYN DoS           — Volumetric DoS (TCP)
     3. Fuzzing           — Stealthy protocol exploitation
     4. SSDP Flood        — Volumetric amplification (UDP)
     5. ARP MitM          — Layer-2 ARP poisoning / MitM
     6. Active Wiretap    — Passive tap (Linux-bridge based)
     7. OS Scan           — OS fingerprinting / reconnaissance
     8. SSL Renegotiation — TLS renegotiation abuse
     9. Video Injection   — Adversarial video stream injection

  CIC-IDS2017:
    10. DDoS              — Distributed DoS (Friday 19:30-21:05 UTC)
    11. DoS Slow          — Slowloris+SlowHTTPTest+Hulk+GoldenEye (Wednesday 13:40-15:30 UTC)
    12. Port Scan         — Reconnaissance (Friday 17:00-19:30 UTC)
    13. Brute Force FTP   — FTP-Patator credential stuffing (Tuesday 13:15-14:25 UTC)
    14. Brute Force SSH   — SSH-Patator credential stuffing (Tuesday 18:00-19:05 UTC)
    15. Web Attacks       — HTTP Brute Force + XSS + SQL Injection (Thursday 13:15-14:45 UTC)
    16. Infiltration      — Malware download + C&C beaconing (Thursday 18:15-18:40 UTC)

  Benign baseline:
    17. Monday benign     — Enterprise benign manifold (sampled, CIC-IDS2017)

CIC-IDS2017 PCAPs are time-filtered with editcap before processing to
reduce size and isolate attack windows.

Usage
-----
    # Process all attacks (skips already-processed ones):
    python scripts/process_all_attacks.py

    # Process a specific attack:
    python scripts/process_all_attacks.py --only syn_dos

    # Re-process an attack even if output already exists:
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
# Terminal colours (ANSI — disabled when stdout is not a tty)
# ---------------------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def bold(t):    return _c("1", t)
def dim(t):     return _c("2", t)
def green(t):   return _c("32", t)
def yellow(t):  return _c("33", t)
def red(t):     return _c("31", t)
def cyan(t):    return _c("36", t)
def magenta(t): return _c("35", t)

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
    # --- Kitsune -----------------------------------------------------------
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
    Attack(
        key="arp_mitm",
        name="ARP MitM",
        dataset="kitsune",
        pcap_path=KITSUNE_PCAP / "ARP_MitM_pcap.pcapng",
        output_dir=TENSOR_DIR / "arp_mitm",
    ),
    Attack(
        key="active_wiretap",
        name="Active Wiretap",
        dataset="kitsune",
        pcap_path=KITSUNE_PCAP / "Active_Wiretap_pcap.pcapng",
        output_dir=TENSOR_DIR / "active_wiretap",
    ),
    Attack(
        key="os_scan",
        name="OS Scan",
        dataset="kitsune",
        pcap_path=KITSUNE_PCAP / "OS_Scan_pcap.pcapng",
        output_dir=TENSOR_DIR / "os_scan",
    ),
    Attack(
        key="ssl_renegotiation",
        name="SSL Renegotiation",
        dataset="kitsune",
        pcap_path=KITSUNE_PCAP / "SSL_Renegotiation_pcap.pcap",
        output_dir=TENSOR_DIR / "ssl_renegotiation",
    ),
    Attack(
        key="video_injection",
        name="Video Injection",
        dataset="kitsune",
        pcap_path=KITSUNE_PCAP / "Video_Injection_pcap.pcapng",
        output_dir=TENSOR_DIR / "video_injection",
    ),
    # --- CIC-IDS2017 -------------------------------------------------------
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
        # Covers all 4 DoS sub-attacks (times are UTC; CIC-IDS2017 uses EDT=UTC-4):
        #   Slowloris     09:47-10:10 EDT = 13:47-14:10 UTC
        #   SlowHTTPTest  10:14-10:35 EDT = 14:14-14:35 UTC
        #   Hulk          10:43-11:00 EDT = 14:43-15:00 UTC
        #   GoldenEye     11:10-11:23 EDT = 15:10-15:23 UTC
        # Window includes ~7 min of benign before first attack.
        time_start="2017-07-05 13:40:00",
        time_end="2017-07-05 15:30:00",
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
    Attack(
        key="brute_force_ftp",
        name="Brute Force FTP (CIC-IDS2017 Tuesday)",
        dataset="cic-ids2017",
        pcap_path=CIC_PCAP / "Tuesday-WorkingHours.pcap",
        output_dir=TENSOR_DIR / "brute_force_ftp",
        # FTP-Patator: 09:20-10:20 EDT = 13:20-14:20 UTC
        time_start="2017-07-04 13:15:00",
        time_end="2017-07-04 14:25:00",
    ),
    Attack(
        key="brute_force_ssh",
        name="Brute Force SSH (CIC-IDS2017 Tuesday)",
        dataset="cic-ids2017",
        pcap_path=CIC_PCAP / "Tuesday-WorkingHours.pcap",
        output_dir=TENSOR_DIR / "brute_force_ssh",
        # SSH-Patator: 14:05-15:00 EDT = 18:05-19:00 UTC
        time_start="2017-07-04 18:00:00",
        time_end="2017-07-04 19:05:00",
    ),
    Attack(
        key="web_attacks",
        name="Web Attacks (CIC-IDS2017 Thursday)",
        dataset="cic-ids2017",
        pcap_path=CIC_PCAP / "Thursday-WorkingHours.pcap",
        output_dir=TENSOR_DIR / "web_attacks",
        # Brute Force HTTP: 09:20-10:00 EDT = 13:20-14:00 UTC
        # XSS:             10:15-10:35 EDT = 14:15-14:35 UTC
        # SQL Injection:   10:40-10:42 EDT = 14:40-14:42 UTC
        # Single window covers all three sub-attacks.
        time_start="2017-07-06 13:15:00",
        time_end="2017-07-06 14:45:00",
    ),
    Attack(
        key="infiltration",
        name="Infiltration (CIC-IDS2017 Thursday)",
        dataset="cic-ids2017",
        pcap_path=CIC_PCAP / "Thursday-WorkingHours.pcap",
        output_dir=TENSOR_DIR / "infiltration",
        # Malware download: 14:19-14:21 EDT = 18:19-18:21 UTC
        # C&C beaconing:   14:33-14:35 EDT = 18:33-18:35 UTC
        time_start="2017-07-06 18:15:00",
        time_end="2017-07-06 18:40:00",
    ),
    # --- Benign baseline ---------------------------------------------------
    Attack(
        key="monday_benign",
        name="Monday Benign (CIC-IDS2017)",
        dataset="cic-ids2017",
        pcap_path=CIC_PCAP / "Monday-WorkingHours.pcap",
        output_dir=TENSOR_DIR / "monday_benign",
        # Monday traffic runs roughly 09:00-17:00. Sample a 2-hour window
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
        print(f"{red(bold('ERROR:'))} editcap not found. Install Wireshark CLI tools:")
        print(f"       {bold('sudo apt install wireshark-common')}")
        sys.exit(1)
    return path


def time_filter_pcap(editcap: str, src: Path, dst: Path,
                     start: str, end: str) -> None:
    """Use editcap to extract a time window from a PCAP."""
    cmd = [editcap, "-A", start, "-B", end, str(src), str(dst)]
    print(f"  {dim('editcap:')} {start} → {end}")
    print(f"  {dim('src:')} {src.name} → {dim('dst:')} {dst.name}")
    subprocess.run(cmd, check=True)
    size_mb = dst.stat().st_size / (1024 * 1024)
    color = yellow if size_mb < 1 else dim
    print(f"  {dim('Filtered PCAP:')} {color(f'{size_mb:.1f} MB')}"
          + (f"  {yellow('⚠  empty — check time window')}" if size_mb < 1 else ""))


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


def _status_colored(status: str) -> str:
    if status == "OK":
        return green(bold(f"{'OK':<12}"))
    if status == "skipped":
        return yellow(f"{'skipped':<12}")
    if status == "MISSING":
        return red(bold(f"{'MISSING':<12}"))
    return red(bold(f"{status:<12}"))


def print_summary(results: list[tuple[str, str, dict | None]]) -> None:
    """Print a summary table of all processed attacks."""
    W = 72
    print(f"\n{bold('=' * W)}")
    print(f" {bold('ADEPT Batch Processing — Summary')}")
    print(f"{bold('=' * W)}")
    h_attack  = cyan(f"{'Attack':<35}")
    h_status  = cyan(f"{'Status':<12}")
    h_windows = cyan(f"{'Windows':>10}")
    h_flows   = cyan(f"{'Flows':>12}")
    print(f" {h_attack} {h_status} {h_windows} {h_flows}")
    print(f" {'-' * 35} {'-' * 12} {'-' * 10} {'-' * 12}")

    for name, status, meta in results:
        if meta:
            windows = green(f"{meta['num_windows']:>10,}")
            flows = f"{meta['num_flows_used']:,}/{meta['num_flows_total']:,}"
        else:
            windows = dim(f"{'—':>10}")
            flows = dim(f"{'—':>12}")
        status_str = _status_colored(status)
        print(f" {name:<35} {status_str} {windows} {flows:>12}")

    print(f"{bold('=' * W)}")
    print(f" Output directory: {cyan(str(TENSOR_DIR))}")
    print(f"{bold('=' * W)}\n")


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
    parser.add_argument(
        "--keep-filtered", action="store_true",
        help="Keep time-filtered PCAPs in data/tensors/.filtered/ after processing "
             "(by default they are deleted to save disk space)",
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

    print(f"\n{bold('ADEPT Batch Processing')} — {cyan(str(len(selected)))} attack(s) selected\n")
    for i, a in enumerate(selected, 1):
        tag = dim(" [time-filtered]") if a.time_start else ""
        print(f"  {dim(str(i) + '.')} {a.name} {dim('(' + a.dataset + ')')}{tag}")
    print()

    if args.dry_run:
        print("Dry run — nothing will be processed.\n")
        for a in selected:
            exists = is_already_processed(a.output_dir)
            if exists and not args.force:
                status = yellow("EXISTS — skip")
            elif exists and args.force:
                status = magenta("EXISTS — force re-process")
            else:
                status = green("WILL PROCESS")
            cached_info = ""
            if a.time_start:
                cached_pcap = TENSOR_DIR / ".filtered" / f"{a.key}_filtered.pcap"
                if cached_pcap.exists() and cached_pcap.stat().st_size > 0:
                    size_mb = cached_pcap.stat().st_size / (1024 * 1024)
                    cached_info = f"  {dim(f'(filtered PCAP cached: {size_mb:.1f} MB)')}"
            print(f"  [{status}] {bold(a.name)}")
            print(f"    PCAP:   {a.pcap_path}")
            print(f"    Output: {a.output_dir}")
            if a.time_start:
                print(f"    Filter: {a.time_start} → {a.time_end}{cached_info}")
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
        print(f"\n{dim('─' * 72)}")
        print(f" {bold(cyan(f'[ {i} / {len(selected)} ]'))}  {bold(attack.name)}  "
              f"{dim('(' + attack.dataset + ')')}")
        print(f"{dim('─' * 72)}")

        # Skip if already processed
        if is_already_processed(attack.output_dir) and not args.force:
            meta = load_metadata(attack.output_dir)
            print(f"  {yellow('Already processed')} ({meta['num_windows']:,} windows). "
                  f"Use {bold('--force')} to re-process.")
            results.append((attack.name, "skipped", meta))
            continue

        # Verify source PCAP exists
        if not attack.pcap_path.exists():
            print(f"  {red(bold('ERROR:'))} PCAP not found: {attack.pcap_path}")
            results.append((attack.name, "MISSING", None))
            continue

        t0 = time.time()
        pcap_to_process = attack.pcap_path

        # Pre-compute filtered PCAP path (needed by both try and finally)
        filtered_pcap = None
        if attack.time_start:
            filtered_dir = TENSOR_DIR / ".filtered"
            filtered_dir.mkdir(parents=True, exist_ok=True)
            filtered_pcap = filtered_dir / f"{attack.key}_filtered.pcap"

        try:
            # Time-filter CIC-IDS2017 PCAPs (skip if a cached copy already exists)
            if attack.time_start:
                if filtered_pcap.exists() and filtered_pcap.stat().st_size > 0:
                    size_mb = filtered_pcap.stat().st_size / (1024 * 1024)
                    print(f"  {dim('Step 0:')} Using cached filtered PCAP "
                          f"{dim(f'({size_mb:.1f} MB)')}")
                else:
                    print(f"  {dim('Step 0:')} Time-filtering PCAP...")
                    time_filter_pcap(
                        editcap, attack.pcap_path, filtered_pcap,
                        attack.time_start, attack.time_end,
                    )
                pcap_to_process = filtered_pcap

            # Run the pipeline
            print(f"  {dim('Running packet-to-tensor pipeline...')}")
            run_pipeline(
                pcap=pcap_to_process,
                output_dir=attack.output_dir,
                window_size=args.window_size,
                extra_args=attack.extra_args,
            )

            meta = load_metadata(attack.output_dir)
            elapsed = time.time() - t0
            print(f"  {green('✓')} Completed in {elapsed:.1f}s")
            results.append((attack.name, "OK", meta))

        except subprocess.CalledProcessError as e:
            elapsed = time.time() - t0
            code = e.returncode
            hint = " (OOM — out of memory)" if code == -9 else ""
            print(f"  {red(bold('✗ FAILED'))} after {elapsed:.1f}s "
                  f"(exit code {red(str(code))}{hint})")
            results.append((attack.name, "FAILED", None))

        finally:
            # Clean up filtered PCAP unless --keep-filtered was set
            if (filtered_pcap is not None
                    and filtered_pcap.exists()
                    and not args.keep_filtered):
                filtered_pcap.unlink()
                print(f"  {dim('Cleaned up filtered PCAP')}")

    total_elapsed = time.time() - total_start

    # Summary
    print_summary(results)
    print(f"{dim('Total time:')} {total_elapsed:.1f}s "
          f"{dim('(' + f'{total_elapsed / 60:.1f} min' + ')')}\n")


if __name__ == "__main__":
    main()
