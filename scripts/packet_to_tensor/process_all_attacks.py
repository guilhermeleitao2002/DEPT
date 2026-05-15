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
     3. SSDP Flood        — Volumetric amplification (UDP)
     4. ARP MitM          — Layer-2 ARP poisoning / MitM
     5. Active Wiretap    — Passive tap (Linux-bridge based)
     6. OS Scan           — OS fingerprinting / reconnaissance
     7. SSL Renegotiation — TLS renegotiation abuse
     8. Video Injection   — Adversarial video stream injection

  CIC-IDS2017:
     9. DDoS              — Distributed DoS (Friday 19:30-21:05 UTC)
    10. DoS Slow          — Slowloris+SlowHTTPTest+Hulk+GoldenEye (Wednesday 13:40-15:30 UTC)
    11. Port Scan         — Reconnaissance (Friday 17:00-19:30 UTC)
    12. Brute Force FTP   — FTP-Patator credential stuffing (Tuesday 13:15-14:25 UTC)
    13. Brute Force SSH   — SSH-Patator credential stuffing (Tuesday 18:00-19:05 UTC)
    14. Brute Force       — FTP-Patator + SSH-Patator combined (Tuesday 13:15-19:05 UTC)
    15. Web Attacks       — HTTP Brute Force + XSS + SQL Injection (Thursday 13:15-14:45 UTC)
    16. Infiltration      — Malware download + C&C beaconing (Thursday 18:15-18:40 UTC)

  Benign baseline:
    17. Monday benign     — Enterprise benign manifold (sampled, CIC-IDS2017)

The --tantgen flag restricts processing to the 5 raw-dataset benign attacks
for which TANTGen-generated adversarial PCAPs exist on disk (mirai,
ssdp_flood, ddos, portscan, brute_force), and additionally processes the
matching adversarial PCAPs (plus the adversarial-only ddos_kitsune variant)
found under
data/adversarial/<attack>/<attack>_adversarial_<mode>.pcap
(modes: both / iat / size; output: data/tensors/adversarial/<attack>_adv_<mode>/).
Adversarial directories not listed in TANTGEN_KEYS (e.g. brute_force_kitsune,
portscan_kitsune sitting on disk for future use) are skipped here.

Labels follow ADEPT's dataset-driven convention: every Kitsune/CIC window
is written as "benign" and every adversarial window as "adversarial" — the
upstream NIDS is expected to catch the raw-dataset attacks, so the only
novel threat ADEPT learns to flag is the TANTGen-generated perturbation.

CIC-IDS2017 PCAPs are time-filtered with editcap before processing to
reduce size and isolate attack windows.

Usage
-----
    # Process all attacks (skips already-processed ones):
    python3 scripts/packet_to_tensor/process_all_attacks.py

    # Process a specific attack:
    python3 scripts/packet_to_tensor/process_all_attacks.py --only syn_dos

    # Re-process an attack even if output already exists:
    python3 scripts/packet_to_tensor/process_all_attacks.py --only mirai --force

    # Process all attacks except some:
    python3 scripts/packet_to_tensor/process_all_attacks.py --exclude mirai syn_dos

    # Dry run — show what would be processed without running:
    python3 scripts/packet_to_tensor/process_all_attacks.py --dry-run

    # Custom window size:
    python3 scripts/packet_to_tensor/process_all_attacks.py --rf-checkpoint-stride 20

    # Process with benign baseline:
    python3 scripts/packet_to_tensor/process_all_attacks.py --include-benign

    # Restrict to the 5 attacks whose TANTGen adversarial PCAPs exist on disk
    # and also tensorise those matching adversarial PCAPs under data/adversarial/:
    python3 scripts/packet_to_tensor/process_all_attacks.py --tantgen

    # Build only the RF input tensors (rf_checkpoints.npy + offsets):
    python3 scripts/packet_to_tensor/process_all_attacks.py --rf

    # Build only the TCN/Transformer input tensors (flow_packets.npy + offsets):
    python3 scripts/packet_to_tensor/process_all_attacks.py --tcn
"""

import argparse
import json
import shutil
import subprocess
import sys
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
PROJECT_ROOT = SCRIPT_DIR.parents[1]
PIPELINE_SCRIPT = SCRIPT_DIR / "run_packet_to_tensor.py"
PIPELINE_CONFIG = PROJECT_ROOT / "config" / "pipeline.yaml"

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
KITSUNE_PCAP = DATA_DIR / "Kitsune" / "pcap"
KITSUNE_CSV = DATA_DIR / "Kitsune" / "csv"
CIC_PCAP = DATA_DIR / "CIC-IDS-2017" / "pcap"
CIC_CSV = DATA_DIR / "CIC-IDS-2017" / "csv" / "TrafficLabelling"
TENSOR_DIR = DATA_DIR / "tensors"
ADVERSARIAL_DIR = DATA_DIR / "adversarial"
ADVERSARIAL_MODES = ("both", "iat", "size")
# Shared IPAT standardizer fitted over benign + adversarial inputs together
# (see fit_shared_mean_std()). Per-attack pipeline runs reuse this file via
# run_packet_to_tensor.py's --mean-std flag instead of refitting locally.
SHARED_MEAN_STD = TENSOR_DIR / "mean_std.json"


# ---------------------------------------------------------------------------
# Attack definitions
# ---------------------------------------------------------------------------
@dataclass
class Attack:
    """Defines one attack to process."""
    key: str                        # Short identifier (used with --only)
    name: str                       # Human-readable name
    dataset: str                    # "kitsune", "cic-ids2017", or "adversarial"
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
        # SSH-Patator: 14:00-15:00 EDT = 18:00-19:00 UTC
        time_start="2017-07-04 18:00:00",
        time_end="2017-07-04 19:05:00",
    ),
    Attack(
        key="brute_force",
        name="Brute Force FTP+SSH (CIC-IDS2017 Tuesday)",
        dataset="cic-ids2017",
        pcap_path=CIC_PCAP / "Tuesday-WorkingHours.pcap",
        output_dir=TENSOR_DIR / "brute_force",
        # Combined window: FTP-Patator (13:15-14:25 UTC) + SSH-Patator (18:00-19:05 UTC)
        # plus the benign gap in between. Matches the single TANTGen
        # BruteForce_adversarial_*.pcap output and avoids splitting the
        # adversarial PCAPs across two benign sources.
        time_start="2017-07-04 13:15:00",
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
        # MetaExploit Vista: 14:19-14:21, 14:33-14:35 EDT = 18:19-18:21, 18:33-18:35 UTC
        # Cool disk MAC:    14:53-15:00 EDT = 18:53-19:00 UTC
        # Dropbox + Portscan (Vista): 15:04-15:45 EDT = 19:04-19:45 UTC
        time_start="2017-07-06 18:15:00",
        time_end="2017-07-06 19:50:00",
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
# Attack keys consumed under --tantgen. Most entries are benign/adversarial
# pairs: the same key names a benign Attack in ATTACKS and a subdirectory
# of data/adversarial/. ddos_kitsune is adversarial-only — its TANTGen
# model was trained against a Kitsune capture but the adversarial PCAPs
# perturb the same CIC DDoS benign manifold, so they augment the
# adversarial pool without introducing a new benign source.
# Additional kitsune variants (brute_force_kitsune, portscan_kitsune) live
# on disk under data/adversarial/ but are intentionally left out of this
# set for now; the discovery in build_adversarial_attacks() filters them.
TANTGEN_KEYS = {
    "mirai", "ssdp_flood",                  # Kitsune (benign + adversarial)
    "ddos", "portscan", "brute_force",      # CIC-IDS-2017 (benign + adversarial)
    "ddos_kitsune",                          # Adversarial-only (paired with ddos benign)
}


def build_adversarial_attacks() -> list[Attack]:
    """Discover TANTGen-generated adversarial PCAPs under data/adversarial/.

    Expected layout (written by scripts/adversarial_generation/generate_adversarials.py):
        data/adversarial/<attack>/<attack>_adversarial_<mode>.pcap
    where <mode> ∈ {both, iat, size}. Produces one Attack per (attack, mode)
    pair, with output under data/tensors/adversarial/<attack>_adv_<mode>/.

    Only directories whose name appears in TANTGEN_KEYS are tensorised;
    other directories on disk (e.g. brute_force_kitsune / portscan_kitsune)
    are intentionally skipped here so they never reach
    data/tensors/adversarial/ until they are explicitly opted in.
    """
    attacks: list[Attack] = []
    if not ADVERSARIAL_DIR.exists():
        return attacks
    for attack_dir in sorted(ADVERSARIAL_DIR.iterdir()):
        if not attack_dir.is_dir():
            continue
        atk = attack_dir.name
        if atk not in TANTGEN_KEYS:
            continue
        for mode in ADVERSARIAL_MODES:
            pcap = attack_dir / f"{atk}_adversarial_{mode}.pcap"
            if not pcap.exists() or pcap.stat().st_size == 0:
                continue
            key = f"{atk}_adv_{mode}"
            pretty = atk.replace("_", " ").title()
            attacks.append(Attack(
                key=key,
                name=f"{pretty} (adversarial / {mode})",
                dataset="adversarial",
                pcap_path=pcap,
                output_dir=TENSOR_DIR / "adversarial" / key,
            ))
    return attacks


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


def run_pipeline(pcap: Path, output_dir: Path,
                 rf_checkpoint_stride: int | None,
                 rf_first_checkpoint: int | None,
                 extra_args: list, mean_std: Path | None = None,
                 outputs: tuple[str, ...] = ("rf", "tcn"),
                 ) -> subprocess.CompletedProcess:
    """Run run_packet_to_tensor.py on a single PCAP."""
    cmd = [
        sys.executable, str(PIPELINE_SCRIPT),
        "-i", str(pcap),
        "-o", str(output_dir),
        "-c", str(PIPELINE_CONFIG),
    ]
    if rf_checkpoint_stride is not None:
        cmd += ["--rf-checkpoint-stride", str(rf_checkpoint_stride)]
    if rf_first_checkpoint is not None:
        cmd += ["--rf-first-checkpoint", str(rf_first_checkpoint)]
    if mean_std is not None:
        cmd += ["--mean-std", str(mean_std)]
    # outputs == ("rf", "tcn") is the pipeline's default — emit a flag only
    # when the user has restricted to a single tensor group.
    if outputs == ("rf",):
        cmd += ["--rf-only"]
    elif outputs == ("tcn",):
        cmd += ["--tcn-only"]
    cmd += extra_args
    return subprocess.run(cmd, check=True)


def collect_shared_fit_pcaps(selected: list[Attack]) -> list[Path]:
    """Inputs to fit the shared IPAT standardizer over.

    Restricted to the PCAPs ADEPT will actually tensorise on this run (i.e.
    the user's --tantgen / --only / --exclude selection, including the
    adversarial Attack entries appended by build_adversarial_attacks()).
    Fitting on the wider registry was a bug: it polluted the z-score with
    attacks the train/val/test set never sees.

    Adversarial variants are intentionally folded in alongside the raw
    attacks so the standardizer sees the heavier IPAT tails TANTGen
    introduces (e.g. PortScan retains its long inter-scan gaps; SYN_DoS
    adversarial IATs are ~24× the originals). Without this, training-time
    z-scoring on adversarial windows lands far in the tails of a
    benign-only fit.

    For CIC attacks we prefer the cached time-filtered PCAP (under
    TENSOR_DIR/.filtered/) when present and fall back to the full day
    capture otherwise. Paths are deduplicated so two selected attacks that
    share a source PCAP (e.g. ddos and portscan both on Friday) only
    contribute once.
    """
    paths: list[Path] = []
    seen: set[Path] = set()
    for atk in selected:
        if atk.time_start:
            cached = TENSOR_DIR / ".filtered" / f"{atk.key}_filtered.pcap"
            if cached.exists() and cached.stat().st_size > 0:
                p = cached
            elif atk.pcap_path.exists():
                p = atk.pcap_path
            else:
                continue
        elif atk.pcap_path.exists():
            p = atk.pcap_path
        else:
            continue
        if p not in seen:
            paths.append(p)
            seen.add(p)
    return paths


def fit_shared_mean_std(selected: list[Attack], force: bool = False) -> Path:
    """Fit (once) the shared mean_std.json over the selected attacks' PCAPs.

    Idempotent: skips if SHARED_MEAN_STD already exists unless force=True.
    Returns the path to the JSON file regardless.
    """
    SHARED_MEAN_STD.parent.mkdir(parents=True, exist_ok=True)
    if SHARED_MEAN_STD.exists() and not force:
        print(f"  {dim('Shared mean/std already present:')} {SHARED_MEAN_STD}")
        return SHARED_MEAN_STD
    inputs = collect_shared_fit_pcaps(selected)
    if not inputs:
        print(f"  {yellow('No PCAPs found for shared fit; falling back to per-PCAP fits.')}")
        return SHARED_MEAN_STD  # caller can check for existence
    print(f"  Fitting shared mean/std over {len(inputs)} PCAP(s) → {SHARED_MEAN_STD}")
    cmd = [
        sys.executable, str(PIPELINE_SCRIPT),
        "--fit-mean-std",
        "-i", *[str(p) for p in inputs],
        "-o", str(SHARED_MEAN_STD),
        "-c", str(PIPELINE_CONFIG),
    ]
    subprocess.run(cmd, check=True)
    return SHARED_MEAN_STD


def write_flow_labels(output_dir: Path, label: int, label_str: str) -> bool:
    """Write flow_labels.csv with `num_flows` uniform rows."""
    meta = load_metadata(output_dir)
    if meta is None:
        print(f"  {yellow('⚠  metadata.json missing, cannot write labels')}")
        return False
    n = int(meta["num_flows"])
    out = output_dir / "flow_labels.csv"
    with open(out, "w") as f:
        f.write("flow_idx,label,label_str\n")
        for i in range(n):
            f.write(f"{i},{label},{label_str}\n")
    print(f"  {dim('Wrote labels:')} {n:,} flows → {label_str}")
    return True


def run_label_attach(attack: Attack) -> bool:
    """Write flow_labels.csv for an attack.

    ADEPT treats all raw-dataset traffic (Kitsune + CIC-IDS2017) as benign
    — the upstream NIDS is expected to catch those attacks — and only
    TANTGen-generated adversarial PCAPs as malicious. The label is driven
    entirely by the source dataset; no ground-truth CSV is consulted.
    """
    if attack.dataset == "adversarial":
        return write_flow_labels(attack.output_dir, 1, "adversarial")
    return write_flow_labels(attack.output_dir, 0, "benign")


def is_already_processed(output_dir: Path,
                         requested_outputs: tuple[str, ...] = ("rf", "tcn"),
                         ) -> bool:
    """Check if the requested tensor outputs already exist on disk.

    A previous run that only built one tensor group still leaves a
    metadata.json behind, so we additionally require the requested
    outputs to be a subset of what that run actually produced. Falls
    back to the pre-feature default (both groups) when metadata predates
    the per-output ``outputs`` field.
    """
    meta = load_metadata(output_dir)
    if meta is None:
        return False
    existing = set(meta.get("outputs") or ("rf", "tcn"))
    return set(requested_outputs).issubset(existing)


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
    W = 80
    print(f"\n{bold('=' * W)}")
    print(f" {bold('ADEPT Batch Processing — Summary')}")
    print(f"{bold('=' * W)}")
    h_attack  = cyan(f"{'Attack':<35}")
    h_status  = cyan(f"{'Status':<12}")
    h_flows   = cyan(f"{'Flows':>10}")
    h_pkts    = cyan(f"{'Packets':>12}")
    print(f" {h_attack} {h_status} {h_flows} {h_pkts}")
    print(f" {'-' * 35} {'-' * 12} {'-' * 10} {'-' * 12}")

    for name, status, meta in results:
        if meta:
            flows_str = green(f"{meta.get('num_flows', 0):>10,}")
            pkts_str  = f"{meta.get('num_packets', 0):>12,}"
        else:
            flows_str = dim(f"{'—':>10}")
            pkts_str  = dim(f"{'—':>12}")
        status_str = _status_colored(status)
        print(f" {name:<35} {status_str} {flows_str} {pkts_str}")

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
    only_exclude = parser.add_mutually_exclusive_group()
    only_exclude.add_argument(
        "--only", nargs="+", metavar="KEY",
        help=f"Process only these attacks (choices: {', '.join(ATTACK_KEYS)})",
    )
    only_exclude.add_argument(
        "--exclude", nargs="+", metavar="KEY",
        help=f"Exclude these attacks from processing (choices: {', '.join(ATTACK_KEYS)})",
    )
    only_exclude.add_argument(
        "--tantgen", action="store_true",
        help=f"Restrict to the 5 raw-dataset benign attacks for which "
             f"TANTGen adversarial PCAPs exist on disk (mirai, ssdp_flood, "
             f"ddos, portscan, brute_force) and tensorise the matching "
             f"adversarial PCAPs plus the adversarial-only ddos_kitsune "
             f"variant under "
             f"data/adversarial/<attack>/<attack>_adversarial_<mode>.pcap "
             f"(output: data/tensors/adversarial/<attack>_adv_<mode>/). "
             f"Other on-disk kitsune variants (brute_force_kitsune, "
             f"portscan_kitsune) are filtered out by build_adversarial_attacks().",
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
        "--rf-checkpoint-stride", type=int, default=None,
        help="Override RF checkpoint stride (default: from pipeline.yaml)",
    )
    parser.add_argument(
        "--rf-first-checkpoint", type=int, default=None,
        help="Override RF early-checkpoint packet index "
             "(default: from pipeline.yaml; 0 disables)",
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
    parser.add_argument(
        "--no-shared-mean-std", action="store_true",
        help="Disable the shared mean_std.json step and let each per-PCAP "
             "run refit its own standardizer. Default behaviour fits one "
             "shared mean_std.json over benign + adversarial PCAPs and "
             "passes it to every per-attack run via --mean-std.",
    )
    parser.add_argument(
        "--refit-shared-mean-std", action="store_true",
        help="Force the shared mean_std.json fit even if data/tensors/mean_std.json "
             "already exists.",
    )
    outputs_group = parser.add_mutually_exclusive_group()
    outputs_group.add_argument(
        "--rf", action="store_true",
        help="Only build the RF input tensors (rf_checkpoints.npy + "
             "rf_checkpoint_offsets.npy) for each selected attack. Skips "
             "the TCN per-packet view. Default builds both.",
    )
    outputs_group.add_argument(
        "--tcn", action="store_true",
        help="Only build the TCN/Transformer input tensors "
             "(flow_packets.npy + flow_offsets.npy) for each selected "
             "attack. Skips the RF checkpoint view. Default builds both.",
    )
    args = parser.parse_args()

    if args.rf:
        selected_outputs: tuple[str, ...] = ("rf",)
    elif args.tcn:
        selected_outputs = ("tcn",)
    else:
        selected_outputs = ("rf", "tcn")

    # Validate --only / --exclude keys
    if args.only:
        for key in args.only:
            if key not in ATTACK_KEYS:
                parser.error(f"Unknown attack key: '{key}'. "
                             f"Choose from: {', '.join(ATTACK_KEYS)}")
    if args.exclude:
        for key in args.exclude:
            if key not in ATTACK_KEYS:
                parser.error(f"Unknown attack key: '{key}'. "
                             f"Choose from: {', '.join(ATTACK_KEYS)}")

    # Build list of attacks to process from the standard registry.
    excluded = set(args.exclude) if args.exclude else set()
    # --tantgen acts like a preset --only restricted to the TANTGen subset.
    only_set = set(args.only) if args.only else (TANTGEN_KEYS if args.tantgen else None)
    selected = []
    for attack in ATTACKS:
        # Skip benign unless explicitly requested
        if attack.key in BENIGN_KEYS and not args.include_benign:
            if not (only_set and attack.key in only_set):
                continue
        # Filter by --only / --tantgen / --exclude
        if only_set is not None and attack.key not in only_set:
            continue
        if attack.key in excluded:
            continue
        selected.append(attack)

    # --tantgen automatically also processes the matching TANTGen-generated
    # adversarial PCAPs (the two are directly correlated): standard benign
    # attacks first, then their adversarial variants.
    if args.tantgen:
        adversarial_attacks = build_adversarial_attacks()
        if not adversarial_attacks:
            print(f"{yellow('WARNING:')} no adversarial PCAPs found under "
                  f"{ADVERSARIAL_DIR}. Run "
                  f"scripts/adversarial_generation/generate_adversarials.py first "
                  f"to produce them.\n")
        else:
            selected.extend(adversarial_attacks)

    if not selected:
        print("Nothing to process. Check --only keys or add --include-benign.")
        sys.exit(0)

    print(f"\n{bold('ADEPT Batch Processing')} — {cyan(str(len(selected)))} attack(s) selected"
          f"  {dim(f'(outputs={list(selected_outputs)})')}\n")
    for i, a in enumerate(selected, 1):
        tag = dim(" [time-filtered]") if a.time_start else ""
        print(f"  {dim(str(i) + '.')} {a.name} {dim('(' + a.dataset + ')')}{tag}")
    print()

    if args.dry_run:
        print("Dry run — nothing will be processed.\n")
        for a in selected:
            exists = is_already_processed(a.output_dir, selected_outputs)
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

    # Fit shared IPAT standardizer once over the SELECTED attacks' PCAPs
    # (benign + adversarial) so every per-attack run uses the same z-score
    # reference and the standardizer reflects exactly the data training
    # will see.
    shared_mean_std: Path | None = None
    if not args.no_shared_mean_std:
        print(f"\n{bold('Shared standardizer')}")
        shared_mean_std = fit_shared_mean_std(
            selected, force=args.refit_shared_mean_std,
        )
        if not shared_mean_std.exists():
            shared_mean_std = None  # fallback to per-PCAP fits

    # Process each attack
    results: list[tuple[str, str, dict | None]] = []
    total_start = time.time()

    try:
        for i, attack in enumerate(selected, 1):
            print(f"\n{dim('─' * 72)}")
            print(f" {bold(cyan(f'[ {i} / {len(selected)} ]'))}  {bold(attack.name)}  "
                  f"{dim('(' + attack.dataset + ')')}")
            print(f"{dim('─' * 72)}")

            # Skip if already processed for the requested output set
            if (is_already_processed(attack.output_dir, selected_outputs)
                    and not args.force):
                meta = load_metadata(attack.output_dir)
                print(f"  {yellow('Already processed')} ({meta.get('num_flows', '?')} flows, "
                      f"{meta.get('num_packets', '?')} packets, "
                      f"outputs={meta.get('outputs', ['rf', 'tcn'])}). "
                      f"Use {bold('--force')} to re-process.")
                # Still attach labels if flow_labels.csv is missing
                if not (attack.output_dir / "flow_labels.csv").exists():
                    print(f"  {dim('flow_labels.csv missing — attaching labels...')}")
                    run_label_attach(attack)
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
                    rf_checkpoint_stride=args.rf_checkpoint_stride,
                    rf_first_checkpoint=args.rf_first_checkpoint,
                    extra_args=attack.extra_args,
                    mean_std=shared_mean_std,
                    outputs=selected_outputs,
                )

                meta = load_metadata(attack.output_dir)
                elapsed = time.time() - t0
                print(f"  {green('✓')} Completed in {elapsed:.1f}s")
                results.append((attack.name, "OK", meta))

                # Attach labels immediately after a successful pipeline run
                print(f"  {dim('Attaching labels...')}")
                run_label_attach(attack)

            except KeyboardInterrupt:
                elapsed = time.time() - t0
                print(f"\n  {yellow(bold('⚠ Interrupted'))} after {elapsed:.1f}s")
                results.append((attack.name, "aborted", None))
                raise  # propagate to outer handler

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

    except KeyboardInterrupt:
        print(f"\n{yellow(bold('Interrupted — stopping early.'))}\n")

    total_elapsed = time.time() - total_start

    # Summary (partial if interrupted)
    print_summary(results)
    print(f"{dim('Total time:')} {total_elapsed:.1f}s "
          f"{dim('(' + f'{total_elapsed / 60:.1f} min' + ')')}\n")


if __name__ == "__main__":
    main()
