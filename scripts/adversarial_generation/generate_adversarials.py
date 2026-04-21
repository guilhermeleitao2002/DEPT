#!/usr/bin/env python3
"""
ADEPT — TANTGen Adversarial Generation Driver
=============================================

Drives the TANTGen repository (sibling to ADEPT) to produce adversarial
versions of attack packets for the 7 selected attacks (4 Kitsune + 3 CIC-IDS2017),
across three perturbation modes (both, iat-only, size-only).

Design notes
------------
- TANTGen scripts hard-code relative paths (config-folders.yaml,
  src/TANTGen/configs/*.yaml, src/TANTGen/*_logs/). They are invoked with
  cwd=TANTGEN_ROOT so those relative paths resolve.
- All input PCAPs and label CSVs live under ADEPT/data/. We pass absolute
  paths to TANTGen CLIs via their -i / -o / --dataset_path flags.
- Final adversarial PCAPs are written to ADEPT/data/adversarial/<key>/.
- Intermediate artifacts (per-attack benign/malicious splits, feature CSVs,
  TANTGen checkpoints) stay inside the TANTGen repo under its own data/ and
  src/TANTGen/*_logs/ trees, because several TANTGen modules load them from
  paths baked into config-folders.yaml.

Usage
-----
    # One-time: build a benign-pretrain CSV (uses first available Kitsune
    # attack's benign split) then pretrain the transformer base.
    python scripts/adversarial_generation/generate_adversarials.py --pretrain

    # Generate adversarials for all 7 attacks across all 3 modes:
    python scripts/adversarial_generation/generate_adversarials.py

    # Only one attack, only one mode:
    python scripts/adversarial_generation/generate_adversarials.py \\
        --only mirai --modes both

    # Skip target-NIDS-related steps; only run what's needed for apply:
    (default — target NIDS is never trained here, ADEPT doesn't need it)
"""

import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ADEPT_ROOT = SCRIPT_DIR.parents[1]
TANTGEN_ROOT = ADEPT_ROOT.parent / "Transformer-based_Adversarial_Network_Traffic_Generator"

ADEPT_DATA = ADEPT_ROOT / "data"
KITSUNE_PCAP = ADEPT_DATA / "Kitsune" / "pcap"
KITSUNE_CSV = ADEPT_DATA / "Kitsune" / "csv"
CIC_PCAP = ADEPT_DATA / "CIC-IDS-2017" / "pcap"
CIC_CSV = ADEPT_DATA / "CIC-IDS-2017" / "csv" / "TrafficLabelling"
FILTERED_CACHE = ADEPT_DATA / "tensors" / ".filtered"
OUT_ROOT = ADEPT_DATA / "adversarial"

# TANTGen-side paths (match config-folders.yaml conventions)
TG_ORIG_KITSUNE_MAL = TANTGEN_ROOT / "data/original/Kitsune/Malicious"
TG_ORIG_KITSUNE_BEN = TANTGEN_ROOT / "data/original/Kitsune/Benign"
TG_ORIG_KITSUNE_LAB = TANTGEN_ROOT / "data/original/Kitsune/Labels"
TG_TARGET_TRAIN = TANTGEN_ROOT / "data/preprocessed/target_train"
TG_ADV_TRAIN = TANTGEN_ROOT / "data/preprocessed/adversarial_train"
TG_SURR_TRAIN = TG_ADV_TRAIN / "surrogate_train"
TG_ADV_TEST = TANTGEN_ROOT / "data/preprocessed/adversarial_test"
TG_PRETRAIN_DIR = TG_ADV_TRAIN / "pretrain"

# TANTGen uses sliding_window_size=64 by default; mirror the naming pattern
# TANTGen uses for its finetuned checkpoint.
SLIDING_WINDOW_SIZE = 64
TG_FINETUNE_CKPT = (
    TANTGEN_ROOT
    / "src/TANTGen/TANTGen_logs"
    / f"1#_final_TANTGen_window{SLIDING_WINDOW_SIZE}.ckpt"
)

MODES = ("both", "iat", "size")


# -----------------------------------------------------------------------------
# Terminal colour (disabled when not a tty)
# -----------------------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty()
def _c(code, s): return f"\033[{code}m{s}\033[0m" if _USE_COLOR else s
def bold(s):   return _c("1", s)
def dim(s):    return _c("2", s)
def green(s):  return _c("32", s)
def yellow(s): return _c("33", s)
def red(s):    return _c("31", s)
def cyan(s):   return _c("36", s)


# -----------------------------------------------------------------------------
# Attack definitions — 7 attacks selected for ADEPT
# -----------------------------------------------------------------------------
@dataclass
class Attack:
    key: str               # short identifier
    dataset: str           # "kitsune" or "cic"
    tg_name: str           # attack_name passed to TANTGen scripts
    pcap: Path             # raw input PCAP (ADEPT-side)
    labels_csv: Path       # per-packet (Kitsune) or per-flow (CIC) labels
    time_start: str = ""   # CIC only: editcap -A
    time_end: str = ""     # CIC only: editcap -B
    # Split ratios (same as TANTGen's run_*_kitsune.py defaults)
    train_nids_ratio: float = 0.6
    train_surrogate_b_ratio: float = 0.2
    train_adversarial_ratio: float = 0.6
    train_surrogate_m_ratio: float = 0.1


ATTACKS = [
    # Kitsune ------------------------------------------------------------------
    Attack("mirai", "kitsune", "Mirai",
           KITSUNE_PCAP / "Mirai_pcap.pcap",
           KITSUNE_CSV / "mirai_labels.csv"),
    Attack("syn_dos", "kitsune", "SYN_DoS",
           KITSUNE_PCAP / "SYN_DoS_pcap.pcap",
           KITSUNE_CSV / "SYN_DoS_labels.csv"),
    Attack("fuzzing", "kitsune", "Fuzzing",
           KITSUNE_PCAP / "Fuzzing_pcap.pcapng",
           KITSUNE_CSV / "Fuzzing_labels.csv"),
    Attack("ssdp_flood", "kitsune", "SSDP_Flood",
           KITSUNE_PCAP / "SSDP_Flood_pcap.pcap",
           KITSUNE_CSV / "SSDP_Flood_labels.csv"),
    # CIC-IDS2017 --------------------------------------------------------------
    # For CIC, we reuse ADEPT's time-filtering convention to carve per-attack
    # PCAPs out of the merged day captures (same windows as ADEPT's
    # process_all_attacks.py). TANTGen sees a per-attack PCAP + the day's
    # TrafficLabelling CSV.
    Attack("ddos", "cic", "DDoS",
           CIC_PCAP / "Friday-WorkingHours.pcap",
           CIC_CSV / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
           time_start="2017-07-07 19:30:00", time_end="2017-07-07 21:05:00"),
    Attack("dos_slow", "cic", "DoS_Slow",
           CIC_PCAP / "Wednesday-workingHours.pcap",
           CIC_CSV / "Wednesday-workingHours.pcap_ISCX.csv",
           time_start="2017-07-05 13:40:00", time_end="2017-07-05 15:30:00"),
    Attack("portscan", "cic", "PortScan",
           CIC_PCAP / "Friday-WorkingHours.pcap",
           CIC_CSV / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
           time_start="2017-07-07 17:00:00", time_end="2017-07-07 19:30:00"),
]
ATTACK_KEYS = [a.key for a in ATTACKS]


# -----------------------------------------------------------------------------
# Subprocess helpers
# -----------------------------------------------------------------------------
def run(cmd, cwd=None, check=True):
    """Run a command, streaming stdout/stderr. cwd defaults to TANTGEN_ROOT."""
    cwd = cwd or TANTGEN_ROOT
    pretty = " ".join(str(c) for c in cmd)
    print(f"  {dim('$')} {pretty}  {dim(f'(cwd={cwd.name})')}")
    return subprocess.run(cmd, cwd=cwd, check=check)


def which_or_die(name):
    p = shutil.which(name)
    if p is None:
        sys.exit(f"{red('ERROR')}: '{name}' not found on PATH.")
    return p


# -----------------------------------------------------------------------------
# Stage: prepare per-attack input PCAP
# -----------------------------------------------------------------------------
def prepare_input_pcap(attack: Attack) -> Path:
    """Return the path to a PCAP ready to feed into TANTGen's divide_BM.

    Kitsune attacks use the raw PCAP unchanged. CIC attacks are time-filtered
    via editcap; filtered PCAPs are cached in ADEPT's .filtered/ cache
    (reuses entries already produced by ADEPT's packet-to-tensor pipeline).
    """
    if attack.dataset == "kitsune":
        if not attack.pcap.exists():
            sys.exit(f"{red('ERROR')}: missing PCAP {attack.pcap}")
        return attack.pcap

    # CIC: time-filter, reusing ADEPT's cache if present
    FILTERED_CACHE.mkdir(parents=True, exist_ok=True)
    filtered = FILTERED_CACHE / f"{attack.key}_filtered.pcap"
    if filtered.exists() and filtered.stat().st_size > 0:
        print(f"  {dim('Using cached filtered PCAP:')} {filtered.name}")
        return filtered

    editcap = which_or_die("editcap")
    if not attack.pcap.exists():
        sys.exit(f"{red('ERROR')}: missing PCAP {attack.pcap}")
    print(f"  {dim('editcap:')} {attack.time_start} → {attack.time_end}")
    run([editcap, "-A", attack.time_start, "-B", attack.time_end,
         str(attack.pcap), str(filtered)], cwd=ADEPT_ROOT)
    return filtered


# -----------------------------------------------------------------------------
# Stage: normalize Kitsune label CSV to the format TANTGen expects
# -----------------------------------------------------------------------------
def normalize_kitsune_labels(labels_csv: Path, attack_key: str) -> Path:
    """TANTGen's Kitsune_divide_BM.py reads `pd.read_csv(labels_file)['x']`.

    Most Kitsune label CSVs already use an R-style `"","x"` header, but some
    (e.g. Mirai) are headerless streams of 0/1. Produce a normalized copy
    with an `x` column when needed and return its path.
    """
    with open(labels_csv, "r") as f:
        first = f.readline().strip()
    # Headered already → use as-is. Columns may be quoted ("x") or bare (x).
    cols = [c.strip().strip('"').strip("'") for c in first.split(",")]
    if "x" in cols:
        return labels_csv

    # Headerless → emit a normalized copy next to the TANTGen labels folder
    TG_ORIG_KITSUNE_LAB.mkdir(parents=True, exist_ok=True)
    out = TG_ORIG_KITSUNE_LAB / f"{attack_key}_labels_normalized.csv"
    if out.exists():
        return out
    print(f"  {dim('Normalizing labels CSV (adding x header):')} {labels_csv.name} → {out.name}")
    with open(labels_csv, "r") as src, open(out, "w") as dst:
        dst.write('"","x"\n')
        for i, line in enumerate(src, start=1):
            val = line.strip()
            if not val:
                continue
            dst.write(f'"{i}",{val}\n')
    return out


# -----------------------------------------------------------------------------
# Native divide_BM for Kitsune PCAPs (linktype-preserving)
# -----------------------------------------------------------------------------
def kitsune_divide_bm_native(pcap_file: Path, labels_csv: Path,
                             malicious_pcap: Path, benign_pcap: Path) -> None:
    """Split pcap/pcapng into benign/malicious PCAPs using a per-packet
    label CSV. Forces output to Ethernet linktype so the result is a plain
    .pcap that TANTGen's tshark-based split_dataset and its scapy readers
    can both consume uniformly.
    """
    import pandas as pd
    from scapy.all import PcapReader, PcapWriter, Ether, IP, IPv6

    labels = pd.read_csv(labels_csv)["x"].tolist()

    writer_mal = PcapWriter(str(malicious_pcap), linktype=1, sync=True)  # Ethernet
    writer_ben = PcapWriter(str(benign_pcap),    linktype=1, sync=True)

    mal_count = ben_count = total = 0
    with PcapReader(str(pcap_file)) as reader:
        for i, pkt in enumerate(reader):
            total = i + 1
            if i >= len(labels):
                break
            # Force Ethernet framing: wrap bare IP packets so the writer
            # (linktype=1) can serialize them consistently.
            if not pkt.haslayer(Ether):
                if pkt.haslayer(IP) or pkt.haslayer(IPv6):
                    pkt = Ether() / pkt
                else:
                    # Unhandled link-layer — skip quietly (rare in Kitsune)
                    continue
            if labels[i] == 1:
                writer_mal.write(pkt)
                mal_count += 1
            else:
                writer_ben.write(pkt)
                ben_count += 1

    writer_mal.close()
    writer_ben.close()
    if total != len(labels):
        print(f"  {yellow('WARNING')}: pcap has {total} packets, labels has {len(labels)} rows.")
    print(f"  Split: {mal_count:,} malicious / {ben_count:,} benign "
          f"→ {malicious_pcap.name}, {benign_pcap.name}")


# -----------------------------------------------------------------------------
# Stage: divide benign / malicious
# -----------------------------------------------------------------------------
def divide_bm(attack: Attack, input_pcap: Path):
    """Split the input PCAP into benign and malicious PCAPs + (CIC only)
    a per-packet labels CSV. Outputs go to TANTGen's data/original/ tree."""
    TG_ORIG_KITSUNE_MAL.mkdir(parents=True, exist_ok=True)
    TG_ORIG_KITSUNE_BEN.mkdir(parents=True, exist_ok=True)
    TG_ORIG_KITSUNE_LAB.mkdir(parents=True, exist_ok=True)

    malicious = TG_ORIG_KITSUNE_MAL / f"{attack.tg_name}_malicious.pcap"
    benign = TG_ORIG_KITSUNE_BEN / f"{attack.tg_name}_benign.pcap"

    if malicious.exists() and benign.exists():
        print(f"  {dim('Benign/malicious split already present; skipping divide_BM.')}")
        return benign, malicious, None

    if attack.dataset == "kitsune":
        labels = normalize_kitsune_labels(attack.labels_csv, attack.key)
        # TANTGen's Kitsune_divide_BM.py uses scapy PcapWriter, which breaks on
        # PCAPs with unusual/mixed link-layer types (e.g. Mirai's Raw linktype).
        # Do the split natively with RawPcapReader/Writer to pass packets
        # byte-for-byte and preserve the source linktype.
        kitsune_divide_bm_native(input_pcap, labels, malicious, benign)
        return benign, malicious, None

    # CIC path additionally emits a per-packet labels CSV
    out_labels = TG_ORIG_KITSUNE_LAB / f"{attack.tg_name}_labels.csv"
    run([sys.executable, "src/Feature_Extraction/Split/CIC_IDS_divide_BM.py",
         "-i", str(input_pcap),
         "-l", str(attack.labels_csv),
         "-m", str(malicious),
         "-b", str(benign),
         "-ol", str(out_labels)])
    return benign, malicious, out_labels


# -----------------------------------------------------------------------------
# Stage: subset splits
# -----------------------------------------------------------------------------
def split_subsets(attack: Attack, benign: Path, malicious: Path, cic_labels: Path | None):
    """Partition benign+malicious into target_train / adversarial_train /
    surrogate_train / adversarial_test subsets."""
    # Expected-produced files (used for skip-logic)
    surrogate_all = TG_SURR_TRAIN / f"{_effective_tg_name(attack)}_all.pcap"
    adv_train_mal = TG_ADV_TRAIN / f"{_effective_tg_name(attack)}_malicious.pcap"
    if surrogate_all.exists() and adv_train_mal.exists():
        print(f"  {dim('Subset splits already present; skipping split_dataset.')}")
        return

    # TANTGen's split scripts write to these hardcoded relative dirs
    for d in (TG_TARGET_TRAIN, TG_ADV_TRAIN, TG_SURR_TRAIN, TG_ADV_TEST):
        d.mkdir(parents=True, exist_ok=True)

    if attack.dataset == "kitsune":
        labels = normalize_kitsune_labels(attack.labels_csv, attack.key)
        run([sys.executable, "src/Feature_Extraction/Split/Kitsune_split_dataset.py",
             "-a", attack.tg_name,
             "-b", str(benign),
             "-m", str(malicious),
             "-l", str(labels),
             "--train_nids_ratio", str(attack.train_nids_ratio),
             "--train_surrogate_b_ratio", str(attack.train_surrogate_b_ratio),
             "--train_adversarial_ratio", str(attack.train_adversarial_ratio),
             "--train_surrogate_m_ratio", str(attack.train_surrogate_m_ratio)])
    else:
        # CIC uses "_kitsune" suffix in TANTGen's run_kitsune_on_CICIDS_pipeline
        run([sys.executable, "src/Feature_Extraction/Split/CIC_IDS_split_4Kitsune.py",
             "-a", f"{attack.tg_name}_kitsune",
             "-b", str(benign),
             "-m", str(malicious),
             "-l", str(cic_labels),
             "--train_nids_b_ratio", str(attack.train_nids_ratio),
             "--train_surrogate_b_ratio", str(attack.train_surrogate_b_ratio),
             "--train_nids_m_ratio", "0",
             "--train_adversarial_ratio", str(attack.train_adversarial_ratio),
             "--train_surrogate_m_ratio", str(attack.train_surrogate_m_ratio)])


def _effective_tg_name(attack: Attack) -> str:
    """Name used in TANTGen's subset folders (CIC gets a '_kitsune' suffix)."""
    return attack.tg_name if attack.dataset == "kitsune" else f"{attack.tg_name}_kitsune"


# -----------------------------------------------------------------------------
# Stage: feature extraction (for surrogate + adversarial training)
# -----------------------------------------------------------------------------
def extract_features(pcap: Path, out_csv: Path):
    if out_csv.exists():
        print(f"  {dim('Features already extracted:')} {out_csv.name}")
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "src/Feature_Extraction/dataloader.py",
         "-i", str(pcap), "-o", str(out_csv)])


# -----------------------------------------------------------------------------
# Stage: surrogate NIDS training
# -----------------------------------------------------------------------------
def train_surrogate(features_csv: Path, labels_csv: Path):
    run([sys.executable, "-m", "src.TANTGen.TANTGen_NIDS",
         "--dataset_path", str(features_csv),
         "--dataset_labels_path", str(labels_csv)])


# -----------------------------------------------------------------------------
# Stage: TANTGen finetune (per-mode) + apply to full malicious PCAP
# -----------------------------------------------------------------------------
def train_tantgen(features_csv: Path, mode: str):
    cmd = [sys.executable, "-m", "src.TANTGen.TANTGen",
           "--dataset_path", str(features_csv)]
    if mode == "iat":
        cmd.append("-iat")
    elif mode == "size":
        cmd.append("-size")
    # mode == "both": no flag, default behavior
    run(cmd)


def apply_tantgen(malicious_pcap: Path, out_pcap: Path):
    out_pcap.parent.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "src.TANTGen.apply_TANTGen_to_pcap",
         "-i", str(malicious_pcap),
         "-o", str(out_pcap)])


# -----------------------------------------------------------------------------
# Stage: global pretrain (one-time, shared across attacks)
# -----------------------------------------------------------------------------
def build_pretrain_benign_csv():
    """Populate data/preprocessed/adversarial_train/pretrain/benign.csv by
    extracting features from the first available per-attack benign PCAP.

    TANTGen pretrains on benign traffic; any sufficiently large benign PCAP
    works. We prefer Kitsune benign because no time-filtering is involved.
    """
    TG_PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = TG_PRETRAIN_DIR / "benign.csv"
    if out_csv.exists():
        print(f"  {dim('Pretrain benign CSV already exists:')} {out_csv}")
        return out_csv

    # Use the first Kitsune attack whose benign split is on disk.
    candidates = sorted(TG_ORIG_KITSUNE_BEN.glob("*_benign.pcap"))
    if not candidates:
        sys.exit(f"{red('ERROR')}: no benign PCAP found under {TG_ORIG_KITSUNE_BEN}. "
                 f"Run the driver on at least one attack first (to produce a "
                 f"benign split), then rerun with --pretrain.")
    source = candidates[0]
    print(f"  {dim('Extracting benign features for pretrain from:')} {source.name}")
    run([sys.executable, "src/Feature_Extraction/dataloader.py",
         "-i", str(source), "-o", str(out_csv)])
    return out_csv


def pretrain_tantgen():
    build_pretrain_benign_csv()
    run([sys.executable, "-m", "src.TANTGen.TANTGen_pretrain"])


# -----------------------------------------------------------------------------
# Per-attack orchestration
# -----------------------------------------------------------------------------
def process_attack(attack: Attack, modes: tuple[str, ...], force: bool):
    print(f"\n{dim('─' * 76)}")
    print(f" {bold(cyan(f'[{attack.key}]'))}  {bold(attack.tg_name)}  {dim('(' + attack.dataset + ')')}")
    print(f"{dim('─' * 76)}")

    # Short-circuit: if all requested modes' outputs exist and --force not set
    out_dir = OUT_ROOT / attack.key
    out_paths = {m: out_dir / f"{attack.key}_adversarial_{m}.pcap" for m in modes}
    if not force and all(p.exists() and p.stat().st_size > 0 for p in out_paths.values()):
        print(f"  {yellow('All requested adversarial PCAPs already exist; skipping.')} "
              f"Use --force to rebuild.")
        return

    # 1. Input PCAP
    input_pcap = prepare_input_pcap(attack)

    # 2. Divide BM
    benign, malicious, cic_labels = divide_bm(attack, input_pcap)

    # 3. Subset splits
    split_subsets(attack, benign, malicious, cic_labels)

    # 4. Feature extraction (surrogate + adversarial train)
    name = _effective_tg_name(attack)
    surrogate_pcap = TG_SURR_TRAIN / f"{name}_all.pcap"
    surrogate_feats = TG_SURR_TRAIN / f"{name}_all_features.csv"
    surrogate_labels = TG_SURR_TRAIN / f"{name}_labels.csv"
    adv_train_mal = TG_ADV_TRAIN / f"{name}_malicious.pcap"
    adv_train_feats = TG_ADV_TRAIN / f"{name}_malicious_features.csv"

    extract_features(surrogate_pcap, surrogate_feats)
    extract_features(adv_train_mal, adv_train_feats)

    # 5. Surrogate NIDS
    train_surrogate(surrogate_feats, surrogate_labels)

    # 6. For each mode: train TANTGen, apply to FULL malicious PCAP
    #    "Full malicious" = the divide_BM output (all attack packets), NOT the
    #    smaller adversarial_test split that stock TANTGen uses.
    for mode in modes:
        out_pcap = out_paths[mode]
        if not force and out_pcap.exists() and out_pcap.stat().st_size > 0:
            print(f"\n  {yellow(f'[{mode}] already produced:')} {out_pcap.name}  (skipping)")
            continue
        print(f"\n  {bold(cyan(f'[mode={mode}]'))} train + apply")
        train_tantgen(adv_train_feats, mode)
        apply_tantgen(malicious, out_pcap)
        print(f"  {green('✓')} {out_pcap.relative_to(ADEPT_ROOT)}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate TANTGen adversarial PCAPs for ADEPT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Attack keys: {', '.join(ATTACK_KEYS)}  |  Modes: {', '.join(MODES)}",
    )
    parser.add_argument("--only", nargs="+", metavar="KEY",
                        help="Restrict to these attack keys.")
    parser.add_argument("--exclude", nargs="+", metavar="KEY",
                        help="Skip these attack keys.")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES),
                        help="Perturbation modes to run (default: all three).")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild adversarial PCAPs even if present.")
    parser.add_argument("--pretrain", action="store_true",
                        help="Run (or re-run) the one-time global pretrain "
                             "step and exit. Requires at least one attack's "
                             "benign split to already exist under "
                             "TANTGen/data/original/Kitsune/Benign/.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List selection and exit.")
    args = parser.parse_args()

    if not TANTGEN_ROOT.exists():
        sys.exit(f"{red('ERROR')}: TANTGen repo not found at {TANTGEN_ROOT}")

    # Ensure all directories TANTGen writes into exist. Several TANTGen
    # modules use pandas.to_json / torch.save with relative paths and crash
    # if a parent directory is missing.
    for d in (
        TG_ORIG_KITSUNE_MAL, TG_ORIG_KITSUNE_BEN, TG_ORIG_KITSUNE_LAB,
        TG_TARGET_TRAIN, TG_ADV_TRAIN, TG_SURR_TRAIN, TG_ADV_TEST,
        TG_PRETRAIN_DIR,
        TANTGEN_ROOT / "src/TANTGen/models",
        TANTGEN_ROOT / "src/TANTGen/transformer_pretrained_logs",
        TANTGEN_ROOT / "src/TANTGen/transformer_nids_logs",
        TANTGEN_ROOT / "src/TANTGen/TANTGen_logs",
        TANTGEN_ROOT / "src/TANTGen/plots",
        OUT_ROOT,
    ):
        d.mkdir(parents=True, exist_ok=True)

    if args.pretrain:
        print(f"{bold('Running TANTGen global pretrain')}  {dim('(one-time)')}")
        pretrain_tantgen()
        return

    # Attack selection
    if args.only:
        unknown = set(args.only) - set(ATTACK_KEYS)
        if unknown:
            parser.error(f"Unknown attack keys: {unknown}")
    if args.exclude:
        unknown = set(args.exclude) - set(ATTACK_KEYS)
        if unknown:
            parser.error(f"Unknown attack keys: {unknown}")

    selected = [
        a for a in ATTACKS
        if (not args.only or a.key in args.only)
        and a.key not in (set(args.exclude) if args.exclude else set())
    ]
    if not selected:
        sys.exit("Nothing to do. Check --only / --exclude.")

    print(f"\n{bold('ADEPT × TANTGen — Adversarial Generation')}")
    print(f"  TANTGen:  {TANTGEN_ROOT}")
    print(f"  Output:   {OUT_ROOT}")
    print(f"  Attacks:  {', '.join(a.key for a in selected)}")
    print(f"  Modes:    {', '.join(args.modes)}\n")

    if args.dry_run:
        for a in selected:
            print(f"  • {a.key:<12} {a.dataset:<8} "
                  f"→ {OUT_ROOT / a.key}/{a.key}_adversarial_<mode>.pcap")
        return

    # Guard: pretrain checkpoint must exist before per-attack processing.
    pretrain_ckpts = list((TANTGEN_ROOT / "src/TANTGen/transformer_pretrained_logs").glob("**/*.ckpt"))
    if not pretrain_ckpts:
        print(f"{yellow('WARNING')}: no pretrain checkpoint found under "
              f"src/TANTGen/transformer_pretrained_logs/. You likely need to "
              f"run `--pretrain` first (after at least one attack has produced "
              f"a benign split). Continuing anyway — TANTGen.py will fail if "
              f"it can't load a base model.\n")

    total_start = time.time()
    for attack in selected:
        try:
            process_attack(attack, tuple(args.modes), args.force)
        except subprocess.CalledProcessError as e:
            print(f"  {red(bold('✗ FAILED'))} on {attack.key} (exit {e.returncode})")
        except KeyboardInterrupt:
            print(f"\n{yellow(bold('Interrupted.'))}")
            break

    elapsed = time.time() - total_start
    print(f"\n{bold('Done.')}  Total: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Outputs under: {cyan(str(OUT_ROOT))}")


if __name__ == "__main__":
    main()
