#!/usr/bin/env python3
"""
DEPT — Packet-to-Tensor batch build (DEPT core, no flags)
===========================================================

Single entry point that materialises every tensor source DEPT trains on.
There are no command-line options: the core pipeline has one fixed
configuration, so running the script always does the whole job and always
overwrites its outputs.

    python3 scripts/packet_to_tensor/process_all_attacks.py

What it builds
--------------
The ten TANTGen-paired benign sources and their adversarial counterparts:

  Kitsune:        mirai, ssdp_flood
  CIC-IDS2017:    ddos, portscan, brute_force        (time-filtered windows)
  CIC-DDoS-2019:  ntp_ddos, dns_ddos, ssdp_ddos,     (first 1,000,000 packets;
                  syn_ddos, ldap_ddos                  these floods have no clean
                                                       time window)

For every source above, the matching TANTGen adversarial PCAPs under
``data/adversarial/<key>/<key>_adversarial_<mode>.pcap`` (modes: both / iat /
size) are also tensorised, plus the adversarial-only ``ddos_kitsune`` variant
(it perturbs the CIC DDoS benign manifold). Output layout:
``data/tensors/<key>/`` for benign, ``data/tensors/adversarial/<key>_adv_<mode>/``
for adversarial.

Labels follow DEPT's dataset-driven convention: every Kitsune/CIC window is
"benign" and every adversarial window is "adversarial" — the upstream NIDS is
expected to catch the raw-dataset attacks, so the only novel threat DEPT
learns to flag is the TANTGen-generated perturbation.

Pipeline
--------
1. Slice the CIC sources (editcap time-filter for CIC-IDS2017; tcpdump/editcap
   packet-cap for CIC-DDoS-2019) into ``data/tensors/.filtered/``.
2. Force-fit one shared IPAT standardizer ``data/tensors/mean_std.json`` over
   all sliced/benign/adversarial PCAPs (so benign and adversarial z-scores are
   comparable and the fit reflects exactly the data being tensorised).
3. Rebuild every source into ragged per-flow tensors (via the in-process
   ``run_packet_to_tensor`` engine), overwriting any previous output, and write
   ``flow_labels.csv``.
4. Remove the sliced PCAPs.
"""

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
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

# ---------------------------------------------------------------------------
# Project layout + the packet-to-tensor engine (imported, driven in-process)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
PIPELINE_CONFIG = PROJECT_ROOT / "config" / "pipeline.yaml"

sys.path.insert(0, str(SCRIPT_DIR))
from run_packet_to_tensor import run, fit_mean_std, load_config

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
KITSUNE_PCAP = DATA_DIR / "Kitsune" / "pcap"
CIC_PCAP = DATA_DIR / "CIC-IDS-2017" / "pcap"
CIC_DDOS_PCAP = DATA_DIR / "CIC-DDoS-2019" / "PCAPs"
TENSOR_DIR = DATA_DIR / "tensors"
ADVERSARIAL_DIR = DATA_DIR / "adversarial"
ADVERSARIAL_MODES = ("both", "iat", "size")
# Shared IPAT standardizer, fitted over benign + adversarial inputs together
# and reused (via run()'s mean_std_path) by every per-source tensor build.
SHARED_MEAN_STD = TENSOR_DIR / "mean_std.json"


# ---------------------------------------------------------------------------
# Attack definitions
# ---------------------------------------------------------------------------
@dataclass
class Attack:
    """Defines one source to process."""
    key: str                        # Short identifier / output-dir name
    name: str                       # Human-readable name
    dataset: str                    # "kitsune", "cic-ids2017", "cic-ddos2019", "adversarial"
    pcap_path: Path                 # Path to raw PCAP
    output_dir: Path                # Tensor output directory
    # CIC-IDS2017 time-filtering (editcap -A / -B). NOTE: editcap interprets
    # these in the *local* timezone of the machine running it; the windows
    # below are tuned for Europe/Lisbon (WEST, UTC+1 in July).
    time_start: str = ""            # e.g. "2017-07-05 09:20:00"
    time_end: str = ""              # e.g. "2017-07-05 12:00:00"
    # CIC-DDoS-2019 packet-count capping (tcpdump -c / editcap 1-N). Used when
    # there is no clean attack time window: the source PCAP is sliced to its
    # first `max_packets` packets before processing. 0 = no cap. Mutually
    # exclusive with time_start.
    max_packets: int = 0


# The ten TANTGen-paired benign sources, in the canonical processing order.
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
        key="ssdp_flood",
        name="SSDP Flood",
        dataset="kitsune",
        pcap_path=KITSUNE_PCAP / "SSDP_Flood_pcap.pcap",
        output_dir=TENSOR_DIR / "ssdp_flood",
    ),
    # --- CIC-IDS2017 (time-filtered attack windows) ------------------------
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
        key="portscan",
        name="Port Scan (CIC-IDS2017 Friday)",
        dataset="cic-ids2017",
        pcap_path=CIC_PCAP / "Friday-WorkingHours.pcap",
        output_dir=TENSOR_DIR / "portscan",
        time_start="2017-07-07 17:00:00",
        time_end="2017-07-07 19:30:00",
    ),
    Attack(
        key="brute_force",
        name="Brute Force FTP+SSH (CIC-IDS2017 Tuesday)",
        dataset="cic-ids2017",
        pcap_path=CIC_PCAP / "Tuesday-WorkingHours.pcap",
        output_dir=TENSOR_DIR / "brute_force",
        # Combined window (local WEST time, see Attack.time_start note):
        # FTP-Patator (13:15-14:25) + SSH-Patator (18:00-19:05) plus the
        # benign gap in between. Matches the single TANTGen
        # BruteForce_adversarial_*.pcap output.
        time_start="2017-07-04 13:15:00",
        time_end="2017-07-04 19:05:00",
    ),
    # --- CIC-DDoS-2019 (packet-capped reflection/amplification floods) -----
    # Attack-dominated from the first seconds and tens of GB in size, so each
    # is sliced to its first `max_packets` packets — the same scale the
    # adversarial generator caps its malicious input to.
    Attack(
        key="ntp_ddos",
        name="NTP DDoS (CIC-DDoS-2019)",
        dataset="cic-ddos2019",
        pcap_path=CIC_DDOS_PCAP / "NTP.pcap",
        output_dir=TENSOR_DIR / "ntp_ddos",
        max_packets=1_000_000,
    ),
    Attack(
        key="dns_ddos",
        name="DNS DDoS (CIC-DDoS-2019)",
        dataset="cic-ddos2019",
        pcap_path=CIC_DDOS_PCAP / "DNS.pcap",
        output_dir=TENSOR_DIR / "dns_ddos",
        max_packets=1_000_000,
    ),
    Attack(
        key="ssdp_ddos",
        name="SSDP DDoS (CIC-DDoS-2019)",
        dataset="cic-ddos2019",
        pcap_path=CIC_DDOS_PCAP / "SSDP.pcap",
        output_dir=TENSOR_DIR / "ssdp_ddos",
        max_packets=1_000_000,
    ),
    Attack(
        key="syn_ddos",
        name="SYN DDoS (CIC-DDoS-2019)",
        dataset="cic-ddos2019",
        pcap_path=CIC_DDOS_PCAP / "Syn.pcap",
        output_dir=TENSOR_DIR / "syn_ddos",
        max_packets=1_000_000,
    ),
    Attack(
        key="ldap_ddos",
        name="LDAP DDoS (CIC-DDoS-2019)",
        dataset="cic-ddos2019",
        pcap_path=CIC_DDOS_PCAP / "LDAP.pcap",
        output_dir=TENSOR_DIR / "ldap_ddos",
        max_packets=1_000_000,
    ),
]

# Adversarial subdirectories under data/adversarial/ that are tensorised.
# Most name a benign Attack above; ddos_kitsune is adversarial-only (its
# TANTGen model was trained against a Kitsune capture but its PCAPs perturb the
# CIC DDoS benign manifold, augmenting the adversarial pool without a new benign
# source). Other on-disk variants (e.g. brute_force_kitsune, portscan_kitsune)
# are intentionally excluded.
TANTGEN_KEYS = {
    "mirai", "ssdp_flood",                  # Kitsune (benign + adversarial)
    "ddos", "portscan", "brute_force",      # CIC-IDS-2017 (benign + adversarial)
    "ddos_kitsune",                          # Adversarial-only (paired with ddos benign)
    "ntp_ddos", "dns_ddos", "ssdp_ddos",    # CIC-DDoS-2019 (benign + adversarial)
    "syn_ddos", "ldap_ddos",                 # CIC-DDoS-2019 (benign + adversarial)
}


def build_adversarial_attacks() -> list:
    """Discover TANTGen-generated adversarial PCAPs under data/adversarial/.

    Expected layout (written by scripts/adversarial_generation/generate_adversarials.py):
        data/adversarial/<attack>/<attack>_adversarial_<mode>.pcap
    where <mode> in {both, iat, size}. Produces one Attack per (attack, mode)
    pair, with output under data/tensors/adversarial/<attack>_adv_<mode>/.
    Only directories whose name is in TANTGEN_KEYS are tensorised.
    """
    attacks: list = []
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
# PCAP slicing helpers
# ---------------------------------------------------------------------------
def find_editcap() -> str:
    """Return path to editcap or exit."""
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


def cap_pcap_packets(src: Path, dst: Path, n: int) -> None:
    """Slice the first `n` packets of a PCAP into `dst`.

    Used for the CIC-DDoS-2019 captures, which have no clean attack time window
    and run to tens of GB. ``tcpdump -c N`` stops reading after N packets, so
    this is fast even on a 37 GB capture; ``editcap 1-N`` is the fallback.
    """
    tcpdump, editcap = shutil.which("tcpdump"), shutil.which("editcap")
    print(f"  {dim('cap:')} first {n:,} packets")
    print(f"  {dim('src:')} {src.name} → {dim('dst:')} {dst.name}")
    if tcpdump:
        subprocess.run([tcpdump, "-r", str(src), "-w", str(dst), "-c", str(n)],
                       check=True)
    elif editcap:
        subprocess.run([editcap, "-r", str(src), str(dst), f"1-{n}"], check=True)
    else:
        print(f"{red(bold('ERROR:'))} need tcpdump or editcap to packet-cap "
              f"{src.name}. Install with:")
        print(f"       {bold('sudo apt install tcpdump wireshark-common')}")
        sys.exit(1)
    size_mb = dst.stat().st_size / (1024 * 1024)
    print(f"  {dim('Capped PCAP:')} {dim(f'{size_mb:.1f} MB')}")


def needs_prefilter(attack: "Attack") -> bool:
    """True if the attack's PCAP must be sliced (time-filtered or packet-
    capped) before the packet-to-tensor engine runs on it."""
    return bool(attack.time_start) or attack.max_packets > 0


# ---------------------------------------------------------------------------
# Labels + metadata
# ---------------------------------------------------------------------------
def load_metadata(output_dir: Path) -> dict:
    """Load metadata.json from a processed source (or None)."""
    meta_path = output_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return None


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
    """Write flow_labels.csv for a source using the dataset-driven convention.

    All raw-dataset traffic (Kitsune + CIC) is "benign"; only TANTGen
    adversarial PCAPs are "adversarial". No ground-truth CSV is consulted.
    """
    if attack.dataset == "adversarial":
        return write_flow_labels(attack.output_dir, 1, "adversarial")
    return write_flow_labels(attack.output_dir, 0, "benign")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _status_colored(status: str) -> str:
    if status == "OK":
        return green(bold(f"{'OK':<12}"))
    if status == "MISSING":
        return red(bold(f"{'MISSING':<12}"))
    return red(bold(f"{status:<12}"))


def print_summary(results: list) -> None:
    """Print a summary table of all processed sources."""
    W = 80
    print(f"\n{bold('=' * W)}")
    print(f" {bold('DEPT Batch Processing — Summary')}")
    print(f"{bold('=' * W)}")
    h_attack  = cyan(f"{'Source':<35}")
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
    cfg = load_config(str(PIPELINE_CONFIG))
    adversarial_attacks = build_adversarial_attacks()
    selected = list(ATTACKS) + adversarial_attacks

    if not adversarial_attacks:
        print(f"{yellow('WARNING:')} no adversarial PCAPs found under "
              f"{ADVERSARIAL_DIR}. Run "
              f"scripts/adversarial_generation/generate_adversarials.py first "
              f"to produce them.\n")

    print(f"\n{bold('DEPT Batch Processing')} — {cyan(str(len(selected)))} source(s)\n")
    for i, a in enumerate(selected, 1):
        if a.time_start:
            tag = dim(" [time-filtered]")
        elif a.max_packets:
            tag = dim(f" [first {a.max_packets:,} pkts]")
        else:
            tag = ""
        print(f"  {dim(str(i) + '.')} {a.name} {dim('(' + a.dataset + ')')}{tag}")
    print()

    # Check slicing-tool availability up front.
    editcap = None
    if any(a.time_start for a in selected):
        editcap = find_editcap()
    if any(a.max_packets for a in selected) and not (
            shutil.which("tcpdump") or shutil.which("editcap")):
        print(f"{red(bold('ERROR:'))} packet-capped sources selected but neither "
              f"tcpdump nor editcap is available. Install with:")
        print(f"       {bold('sudo apt install tcpdump wireshark-common')}")
        sys.exit(1)

    filtered_dir = TENSOR_DIR / ".filtered"
    proc_pcap: dict = {}          # attack.key -> PCAP to tensorise (sliced or full)
    sliced_files: list = []       # sliced PCAPs to remove at the end
    results: list = []            # (name, status, meta) for the summary
    total_start = time.time()

    try:
        # ── Step 1: slice every CIC source that needs prefiltering ──────────
        print(f"{bold('Step 1')} — slicing CIC sources")
        for attack in selected:
            if not attack.pcap_path.exists():
                print(f"  {red(bold('MISSING:'))} {attack.name} — {attack.pcap_path}")
                results.append((attack.name, "MISSING", None))
                continue
            if needs_prefilter(attack):
                filtered_dir.mkdir(parents=True, exist_ok=True)
                dst = filtered_dir / f"{attack.key}_filtered.pcap"
                print(f"\n  {bold(attack.name)}")
                if attack.time_start:
                    time_filter_pcap(editcap, attack.pcap_path, dst,
                                     attack.time_start, attack.time_end)
                else:
                    cap_pcap_packets(attack.pcap_path, dst, attack.max_packets)
                proc_pcap[attack.key] = dst
                sliced_files.append(dst)
            else:
                proc_pcap[attack.key] = attack.pcap_path

        # Runnable = sources whose (sliced or full) PCAP exists.
        runnable = [a for a in selected if a.key in proc_pcap]

        # ── Step 2: force-fit the shared IPAT standardizer ──────────────────
        # Fit over exactly the PCAPs being tensorised (sliced for CIC, full
        # otherwise), in processing order, so benign and adversarial z-scores
        # are comparable and the standardizer reflects the tensorised data.
        fit_inputs: list = []
        seen: set = set()
        for a in runnable:
            p = proc_pcap[a.key]
            if p not in seen:
                fit_inputs.append(str(p))
                seen.add(p)
        print(f"\n{bold('Step 2')} — fitting shared standardizer over "
              f"{len(fit_inputs)} PCAP(s) → {SHARED_MEAN_STD}")
        SHARED_MEAN_STD.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        stats = fit_mean_std(fit_inputs, str(SHARED_MEAN_STD), cfg)
        print(f"\n{green('✓')} Wrote {SHARED_MEAN_STD}  {dim(f'({time.time() - t0:.1f}s)')}")
        for c, s in stats.items():
            print(f"  {cyan(c):<20}  mean={s['mean']:.6f}  std={s['std']:.6f}")

        # ── Step 3: tensorise every source (overwriting) + attach labels ────
        print(f"\n{bold('Step 3')} — building tensors")
        for i, attack in enumerate(runnable, 1):
            print(f"\n{dim('─' * 72)}")
            print(f" {bold(cyan(f'[ {i} / {len(runnable)} ]'))}  {bold(attack.name)}  "
                  f"{dim('(' + attack.dataset + ')')}")
            print(f"{dim('─' * 72)}")
            t0 = time.time()
            try:
                run(str(proc_pcap[attack.key]), str(attack.output_dir),
                    str(SHARED_MEAN_STD), cfg)
                meta = load_metadata(attack.output_dir)
                print(f"  {green('✓')} Completed in {time.time() - t0:.1f}s")
                results.append((attack.name, "OK", meta))
                print(f"  {dim('Attaching labels...')}")
                run_label_attach(attack)
            except KeyboardInterrupt:
                print(f"\n  {yellow(bold('⚠ Interrupted'))} after {time.time() - t0:.1f}s")
                results.append((attack.name, "aborted", None))
                raise
            except Exception as e:  # noqa: BLE001 — report and keep going
                print(f"  {red(bold('✗ FAILED'))} after {time.time() - t0:.1f}s "
                      f"({type(e).__name__}: {e})")
                results.append((attack.name, "FAILED", None))

    except KeyboardInterrupt:
        print(f"\n{yellow(bold('Interrupted — stopping early.'))}\n")
    finally:
        # ── Step 4: remove the sliced PCAPs ─────────────────────────────────
        for f in sliced_files:
            if f.exists():
                f.unlink()
        if filtered_dir.exists() and not any(filtered_dir.iterdir()):
            filtered_dir.rmdir()

    print_summary(results)
    total_elapsed = time.time() - total_start
    print(f"{dim('Total time:')} {total_elapsed:.1f}s "
          f"{dim('(' + f'{total_elapsed / 60:.1f} min' + ')')}\n")


if __name__ == "__main__":
    main()
