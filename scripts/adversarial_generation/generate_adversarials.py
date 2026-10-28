#!/usr/bin/env python3
"""
DEPT — TANTGen Adversarial Generation Driver
=============================================

Runs the TANTGen repository (next to DEPT) to produce adversarial versions of
the 10 attacks in three perturbation modes (both, iat-only, size-only):

  - kitsune     : Mirai, SSDP_Flood
  - cic-ids2017 : DDoS, PortScan, BruteForce
  - cic-ddos2019: NTP_DDoS, DNS_DDoS, SSDP_DDoS, SYN_DDoS, LDAP_DDoS

Steps:
  0. One-time TANTGen pretrain (only if the checkpoint is missing).
  For each attack × mode:
  1. divide benign/malicious   2. subset split   3. feature extraction
  4. surrogate NIDS training   5. per-attack evasion margin
  6. TANTGen finetune          7. apply to the malicious PCAP

Input data is read from the TANTGen tree (data/original/...). The adversarial
PCAPs are written to DEPT/data/adversarial/<key>/<key>_adversarial_<mode>.pcap.
TANTGen scripts use relative paths, so they run with cwd=TANTGEN_ROOT and their
intermediate files stay in the TANTGen tree. The evasion margin is written into
src/TANTGen/configs/config-TANTGen-finetune.yaml (CIC = -4.0, Kitsune = 0.0).

Usage
-----
No options: the script runs the pretrain if needed and then all 10 attacks × 3
modes, overwriting the PCAPs under data/adversarial/<key>/.

    python scripts/adversarial_generation/generate_adversarials.py
"""

import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# -----------------------------------------------------------------------------
# Paths — DEPT and TANTGen are siblings under the same parent directory
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEPT_ROOT = SCRIPT_DIR.parents[1]
TANTGEN_ROOT = DEPT_ROOT.parent / "Transformer-based_Adversarial_Network_Traffic_Generator"

# INPUT side (TANTGen tree) -----------------------------------------------------
TG_ORIGINAL = TANTGEN_ROOT / "data" / "original"
# divide_BM / split scripts write into these hard-coded (config-folders.yaml) dirs
TG_ORIG_KITSUNE_MAL = TANTGEN_ROOT / "data/original/Kitsune/Malicious"
TG_ORIG_KITSUNE_BEN = TANTGEN_ROOT / "data/original/Kitsune/Benign"
TG_ORIG_KITSUNE_LAB = TANTGEN_ROOT / "data/original/Kitsune/Labels"
TG_TARGET_TRAIN = TANTGEN_ROOT / "data/preprocessed/target_train"
TG_ADV_TRAIN = TANTGEN_ROOT / "data/preprocessed/adversarial_train"
TG_SURR_TRAIN = TG_ADV_TRAIN / "surrogate_train"
TG_ADV_TEST = TANTGEN_ROOT / "data/preprocessed/adversarial_test"
TG_PRETRAIN_DIR = TG_ADV_TRAIN / "pretrain"

FINETUNE_CONFIG = TANTGEN_ROOT / "src/TANTGen/configs/config-TANTGen-finetune.yaml"

# TANTGen default sliding window; mirror its checkpoint naming pattern.
SLIDING_WINDOW_SIZE = 64
TG_PRETRAIN_CKPT = (
    TANTGEN_ROOT / "src/TANTGen/transformer_pretrained_logs"
    / f"TANTGen_pretrain_window{SLIDING_WINDOW_SIZE}.ckpt"
)
TG_FINETUNE_CKPT = (
    TANTGEN_ROOT / "src/TANTGen/TANTGen_logs"
    / f"1#_final_TANTGen_window{SLIDING_WINDOW_SIZE}.ckpt"
)

# OUTPUT side (DEPT tree) ------------------------------------------------------
OUT_ROOT = DEPT_ROOT / "data" / "adversarial"

MODES = ("both", "iat", "size")

# Cap each malicious PCAP to its first N packets before applying TANTGen
# (only the large CIC-DDoS captures are affected).
MAX_MALICIOUS_PACKETS = 1_000_000


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
# Attack registry — all 10 attacks (mirrors TANTGen scripts/attacks.yaml).
# Input paths resolve under TANTGEN_ROOT/data/original. Split ratios feed the
# Kitsune-style split (TANTGen surrogate is per-packet regardless of family);
# margin feeds the finetune evasion objective.
# -----------------------------------------------------------------------------
@dataclass
class Attack:
    key: str                # DEPT-side identifier + output dir / tensor source name
    dataset: str            # "kitsune" or "cic"
    tg_name: str            # attack_name passed to TANTGen scripts (== attacks.yaml id)
    pcap: Path              # raw input PCAP (TANTGen tree)
    labels_csv: Path        # per-packet (Kitsune) or per-flow (CIC) labels
    margin: float           # finetune evasion margin (CIC -4.0, Kitsune 0.0)
    # Kitsune-style split ratios
    train_nids_ratio: float = 0.6
    train_surrogate_b_ratio: float = 0.2
    train_adversarial_ratio: float = 0.6
    train_surrogate_m_ratio: float = 0.1


_K = TG_ORIGINAL / "Kitsune"
_C17 = TG_ORIGINAL / "CIC-IDS-2017"
_C19 = TG_ORIGINAL / "CIC-DDoS-2019"

ATTACKS = [
    # --- Kitsune --------------------------------------------------------------
    Attack("mirai", "kitsune", "Mirai",
           _K / "Mirai Botnet/Mirai.pcap",
           _K / "Mirai Botnet/mirai_labels.csv",
           margin=0.0,
           train_nids_ratio=0.6, train_surrogate_b_ratio=0.2,
           train_adversarial_ratio=0.6, train_surrogate_m_ratio=0.1),
    Attack("ssdp_flood", "kitsune", "SSDP_Flood",
           _K / "SSDP Flood/SSDP_Flood_pcap.pcapng",
           _K / "SSDP Flood/SSDP_Flood_labels.csv",
           margin=0.0,
           train_nids_ratio=0.8, train_surrogate_b_ratio=0.1,
           train_adversarial_ratio=0.6, train_surrogate_m_ratio=0.2),
    # --- CIC-IDS-2017 ---------------------------------------------------------
    Attack("ddos", "cic", "DDoS",
           _C17 / "PCAPs/Friday-WorkingHours-Afternoon-DDoS.pcap",
           _C17 / "CSVs/TrafficLabelling/Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv",
           margin=-4.0),
    Attack("portscan", "cic", "PortScan",
           _C17 / "PCAPs/Friday-WorkingHours-Afternoon-PortScan.pcap",
           _C17 / "CSVs/TrafficLabelling/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
           margin=-4.0),
    Attack("brute_force", "cic", "BruteForce",
           _C17 / "PCAPs/Tuesday-WorkingHours.pcap",
           _C17 / "CSVs/TrafficLabelling/Tuesday-WorkingHours.pcap_ISCX.csv",
           margin=-4.0),
    # --- CIC-DDoS-2019 --------------------------------------------------------
    Attack("ntp_ddos", "cic", "NTP_DDoS",
           _C19 / "PCAPs/NTP.pcap", _C19 / "CSVs/NTP.csv", margin=-4.0),
    Attack("dns_ddos", "cic", "DNS_DDoS",
           _C19 / "PCAPs/DNS.pcap", _C19 / "CSVs/DNS.csv", margin=-4.0),
    Attack("ssdp_ddos", "cic", "SSDP_DDoS",
           _C19 / "PCAPs/SSDP.pcap", _C19 / "CSVs/SSDP.csv", margin=-4.0),
    Attack("syn_ddos", "cic", "SYN_DDoS",
           _C19 / "PCAPs/Syn.pcap", _C19 / "CSVs/Syn.csv", margin=-4.0),
    Attack("ldap_ddos", "cic", "LDAP_DDoS",
           _C19 / "PCAPs/LDAP.pcap", _C19 / "CSVs/LDAP.csv", margin=-4.0),
]


# -----------------------------------------------------------------------------
# Subprocess helper
# -----------------------------------------------------------------------------
def run(cmd, cwd=None, check=True):
    """Run a command, streaming stdout/stderr. cwd defaults to TANTGEN_ROOT."""
    cwd = cwd or TANTGEN_ROOT
    pretty = " ".join(str(c) for c in cmd)
    print(f"  {dim('$')} {pretty}  {dim(f'(cwd={cwd.name})')}")
    return subprocess.run(cmd, cwd=cwd, check=check)


# -----------------------------------------------------------------------------
# Per-attack evasion margin → config-TANTGen-finetune.yaml (in place)
# -----------------------------------------------------------------------------
def template_finetune_margin(margin: float) -> None:
    """Rewrite the `margin:` line in config-TANTGen-finetune.yaml in place.

    Same as TANTGen's scripts/run_attack.py:template_finetune_margin.
    """
    if margin is None:
        return
    with open(FINETUNE_CONFIG) as f:
        lines = f.readlines()
    new_lines, found = [], False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("margin:") or stripped.startswith("margin :"):
            indent = line[: len(line) - len(stripped)]
            new_lines.append(f"{indent}margin: {margin}\n")
            found = True
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(f"margin: {margin}\n")
    with open(FINETUNE_CONFIG, "w") as f:
        f.writelines(new_lines)
    print(f"  {dim('Set finetune margin →')} {margin}")


# -----------------------------------------------------------------------------
# Normalize Kitsune label CSV to the format TANTGen expects
# -----------------------------------------------------------------------------
def normalize_kitsune_labels(labels_csv: Path, attack_key: str) -> Path:
    """TANTGen's Kitsune split reads `pd.read_csv(labels_file)['x']`.

    Most Kitsune label CSVs already use an R-style `"","x"` header; some (e.g.
    Mirai) are headerless 0/1 streams. Produce a normalized copy with an `x`
    column when needed and return its path.
    """
    with open(labels_csv, "r") as f:
        first = f.readline().strip()
    cols = [c.strip().strip('"').strip("'") for c in first.split(",")]
    if "x" in cols:
        return labels_csv

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
    """Split pcap/pcapng into benign/malicious PCAPs using a per-packet label
    CSV. Forces Ethernet linktype so the result is a plain .pcap that TANTGen's
    tshark-based split and its scapy readers can both consume uniformly.
    (TANTGen's Kitsune_divide_BM.py breaks on Mirai's Raw linktype.)
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
            if not pkt.haslayer(Ether):
                if pkt.haslayer(IP) or pkt.haslayer(IPv6):
                    pkt = Ether() / pkt
                else:
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
# Stage: validate input PCAP (no time-filtering — divide-by-label isolates the
# per-attack traffic, so the raw per-attack/day capture is used directly)
# -----------------------------------------------------------------------------
def prepare_input_pcap(attack: Attack) -> Path:
    if not attack.pcap.exists():
        sys.exit(f"{red('ERROR')}: missing input PCAP for '{attack.key}': {attack.pcap}")
    if not attack.labels_csv.exists():
        sys.exit(f"{red('ERROR')}: missing labels for '{attack.key}': {attack.labels_csv}")
    return attack.pcap


def _effective_tg_name(attack: Attack) -> str:
    """Name used in TANTGen's subset folders (CIC gets a '_kitsune' suffix)."""
    return attack.tg_name if attack.dataset == "kitsune" else f"{attack.tg_name}_kitsune"


# -----------------------------------------------------------------------------
# Stage: divide benign / malicious  → TANTGen data/original/Kitsune/{Mal,Ben}
# -----------------------------------------------------------------------------
def divide_bm(attack: Attack, input_pcap: Path):
    TG_ORIG_KITSUNE_MAL.mkdir(parents=True, exist_ok=True)
    TG_ORIG_KITSUNE_BEN.mkdir(parents=True, exist_ok=True)
    TG_ORIG_KITSUNE_LAB.mkdir(parents=True, exist_ok=True)

    malicious = TG_ORIG_KITSUNE_MAL / f"{attack.tg_name}_malicious.pcap"
    benign = TG_ORIG_KITSUNE_BEN / f"{attack.tg_name}_benign.pcap"

    if attack.dataset == "kitsune":
        if malicious.exists() and benign.exists():
            print(f"  {dim('Benign/malicious split already present; skipping divide_BM.')}")
            return benign, malicious, None
        labels = normalize_kitsune_labels(attack.labels_csv, attack.key)
        kitsune_divide_bm_native(input_pcap, labels, malicious, benign)
        return benign, malicious, None

    # CIC_IDS_divide_BM.py writes the labels CSV last; if it is missing, the
    # previous divide was interrupted, so divide again.
    out_labels = TG_ORIG_KITSUNE_LAB / f"{attack.tg_name}_labels.csv"
    if malicious.exists() and benign.exists() and out_labels.exists():
        print(f"  {dim('Benign/malicious split + labels already present; skipping divide_BM.')}")
        return benign, malicious, out_labels
    if malicious.exists() or benign.exists():
        print(f"  {yellow('Incomplete prior divide (labels CSV missing) — re-dividing.')}")
    run([sys.executable, "src/Feature_Extraction/Split/CIC_IDS_divide_BM.py",
         "-i", str(input_pcap),
         "-l", str(attack.labels_csv),
         "-m", str(malicious),
         "-b", str(benign),
         "-ol", str(out_labels)])
    return benign, malicious, out_labels


# -----------------------------------------------------------------------------
# Stage: subset splits (target_train / adversarial_train / surrogate_train / test)
# -----------------------------------------------------------------------------
def split_subsets(attack: Attack, benign: Path, malicious: Path, cic_labels):
    name = _effective_tg_name(attack)
    surrogate_all = TG_SURR_TRAIN / f"{name}_all.pcap"
    adv_train_mal = TG_ADV_TRAIN / f"{name}_malicious.pcap"
    if surrogate_all.exists() and adv_train_mal.exists():
        print(f"  {dim('Subset splits already present; skipping split_dataset.')}")
        return

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
        if cic_labels is None:
            sys.exit(f"{red('ERROR')}: CIC split needs a per-packet labels CSV "
                     f"for '{attack.key}' but none was produced by divide_BM.")
        run([sys.executable, "src/Feature_Extraction/Split/CIC_IDS_split_4Kitsune.py",
             "-a", name,
             "-b", str(benign),
             "-m", str(malicious),
             "-l", str(cic_labels),
             "--train_nids_b_ratio", str(attack.train_nids_ratio),
             "--train_surrogate_b_ratio", str(attack.train_surrogate_b_ratio),
             "--train_nids_m_ratio", "0",
             "--train_adversarial_ratio", str(attack.train_adversarial_ratio),
             "--train_surrogate_m_ratio", str(attack.train_surrogate_m_ratio)])


# -----------------------------------------------------------------------------
# Stage: feature extraction / surrogate / finetune / apply
# -----------------------------------------------------------------------------
def extract_features(pcap: Path, out_csv: Path):
    if out_csv.exists():
        print(f"  {dim('Features already extracted:')} {out_csv.name}")
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "src/Feature_Extraction/dataloader.py",
         "-i", str(pcap), "-o", str(out_csv)])


def train_surrogate(features_csv: Path, labels_csv: Path):
    run([sys.executable, "-m", "src.TANTGen.TANTGen_NIDS",
         "--dataset_path", str(features_csv),
         "--dataset_labels_path", str(labels_csv)])


def train_tantgen(features_csv: Path, mode: str):
    cmd = [sys.executable, "-m", "src.TANTGen.TANTGen",
           "--dataset_path", str(features_csv)]
    if mode == "iat":
        cmd.append("-iat")
    elif mode == "size":
        cmd.append("-size")
    # mode == "both": no flag, default (perturb both size and IAT)
    run(cmd)


def apply_tantgen(malicious_pcap: Path, out_pcap: Path):
    out_pcap.parent.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "src.TANTGen.apply_TANTGen_to_pcap",
         "-i", str(malicious_pcap),
         "-o", str(out_pcap)])


def cap_malicious(malicious: Path) -> Path:
    """Return a malicious PCAP capped to the first MAX_MALICIOUS_PACKETS packets.

    apply_TANTGen_to_pcap rewrites every packet with scapy, which takes hours on
    the CIC-DDoS captures (NTP's malicious split is ~75M packets). Smaller
    captures (Kitsune, CIC-IDS) are returned unchanged.
    """
    # Small captures cannot exceed the cap.
    SIZE_THRESHOLD = 400 * 1024 * 1024  # 400 MB
    if malicious.stat().st_size < SIZE_THRESHOLD:
        return malicious
    capped = malicious.with_name(f"{malicious.stem}_first{MAX_MALICIOUS_PACKETS}.pcap")
    if capped.exists() and capped.stat().st_size > 0:
        print(f"  {dim('Using cached capped malicious PCAP:')} {capped.name}")
        return capped
    tcpdump, editcap = shutil.which("tcpdump"), shutil.which("editcap")
    print(f"  {dim(f'Capping {malicious.name} -> first {MAX_MALICIOUS_PACKETS:,} packets')} ({capped.name})")
    if tcpdump:  # -c N stops after N packets (fast even on a 38 GB capture)
        run([tcpdump, "-r", str(malicious), "-w", str(capped), "-c", str(MAX_MALICIOUS_PACKETS)],
            cwd=DEPT_ROOT)
    elif editcap:
        run([editcap, "-r", str(malicious), str(capped), f"1-{MAX_MALICIOUS_PACKETS}"], cwd=DEPT_ROOT)
    else:
        sys.exit(f"{red('ERROR')}: need tcpdump or editcap to cap {malicious.name} "
                 f"(sudo apt install tcpdump wireshark-common).")
    return capped


# -----------------------------------------------------------------------------
# Stage: global pretrain (one-time, shared across attacks)
# -----------------------------------------------------------------------------
def build_pretrain_benign_csv() -> Path:
    """Populate data/preprocessed/adversarial_train/pretrain/benign.csv by
    extracting features from an existing benign split. Prefers the largest
    Kitsune benign PCAP on disk (more benign traffic = a better base manifold).
    """
    TG_PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = TG_PRETRAIN_DIR / "benign.csv"
    if out_csv.exists():
        print(f"  {dim('Pretrain benign CSV already exists:')} {out_csv}")
        return out_csv

    candidates = sorted(TG_ORIG_KITSUNE_BEN.glob("*_benign.pcap"),
                        key=lambda p: p.stat().st_size, reverse=True)
    if not candidates:
        sys.exit(f"{red('ERROR')}: no benign PCAP under {TG_ORIG_KITSUNE_BEN}. "
                 f"ensure_pretrain() should have produced one — check the first "
                 f"Kitsune attack's divide step.")
    source = candidates[0]
    print(f"  {dim('Extracting benign features for pretrain from:')} {source.name}")
    run([sys.executable, "src/Feature_Extraction/dataloader.py",
         "-i", str(source), "-o", str(out_csv)])
    return out_csv


def ensure_pretrain() -> None:
    """Make sure a TANTGen pretrain checkpoint exists before the per-attack work.

    TANTGen's own pipeline never builds benign.csv and ignores the pretrain exit
    code, so finetuning would silently start from random weights. This builds a
    benign split and benign.csv if needed and runs the pretrain with check=True.
    """
    print("=" * 79)
    print(bold(" Global TANTGen pretrain (one-time, shared across all attacks)"))
    print("=" * 79)

    if TG_PRETRAIN_CKPT.exists():
        print(f"  {green('✓')} Pretrain checkpoint present: "
              f"{dim(str(TG_PRETRAIN_CKPT.relative_to(TANTGEN_ROOT)))} — skipping.")
        return

    # (a) need at least one benign split to extract benign.csv from
    if not any(TG_ORIG_KITSUNE_BEN.glob("*_benign.pcap")):
        seed = next((a for a in ATTACKS if a.dataset == "kitsune"), None)
        if seed is None:
            sys.exit(f"{red('ERROR')}: no Kitsune attack available to seed the "
                     f"benign pretrain corpus.")
        print(f"  {dim('No benign split on disk; dividing')} {bold(seed.key)} "
              f"{dim('to seed the pretrain corpus...')}")
        divide_bm(seed, prepare_input_pcap(seed))

    # (b) build benign.csv  (c) pretrain (loud on failure)
    build_pretrain_benign_csv()
    run([sys.executable, "-m", "src.TANTGen.TANTGen_pretrain"], check=True)

    if not TG_PRETRAIN_CKPT.exists():
        sys.exit(f"{red('ERROR')}: pretrain finished but no checkpoint at "
                 f"{TG_PRETRAIN_CKPT}. Check src/TANTGen/configs/"
                 f"config-TANTGen-general.yaml (save_path_pretrain / "
                 f"model_name_pretrain / sliding_window_size).")
    print(f"  {green('✓')} Pretrain checkpoint written.")


# -----------------------------------------------------------------------------
# Per-attack orchestration
# -----------------------------------------------------------------------------
def process_attack(attack: Attack):
    print(f"\n{dim('─' * 79)}")
    print(f" {bold(cyan(f'[{attack.key}]'))}  {bold(attack.tg_name)}  "
          f"{dim('(' + attack.dataset + f', margin={attack.margin})')}")
    print(f"{dim('─' * 79)}")

    out_dir = OUT_ROOT / attack.key
    out_paths = {m: out_dir / f"{attack.key}_adversarial_{m}.pcap" for m in MODES}

    # 1. validate input → 2. divide → 3. subset split
    input_pcap = prepare_input_pcap(attack)
    benign, malicious, cic_labels = divide_bm(attack, input_pcap)
    split_subsets(attack, benign, malicious, cic_labels)

    # 4. features (surrogate + adversarial-train malicious)
    name = _effective_tg_name(attack)
    surrogate_feats = TG_SURR_TRAIN / f"{name}_all_features.csv"
    surrogate_labels = TG_SURR_TRAIN / f"{name}_labels.csv"
    adv_train_mal = TG_ADV_TRAIN / f"{name}_malicious.pcap"
    adv_train_feats = TG_ADV_TRAIN / f"{name}_malicious_features.csv"

    extract_features(TG_SURR_TRAIN / f"{name}_all.pcap", surrogate_feats)
    extract_features(adv_train_mal, adv_train_feats)

    # 5. surrogate NIDS
    train_surrogate(surrogate_feats, surrogate_labels)

    # 6. per-attack evasion margin (persists across this attack's mode loop)
    template_finetune_margin(attack.margin)

    # Cap the malicious set fed to apply (see cap_malicious).
    apply_src = cap_malicious(malicious)

    # 7. per mode: finetune TANTGen, then apply to the (capped) malicious PCAP.
    for mode in MODES:
        out_pcap = out_paths[mode]
        print(f"\n  {bold(cyan(f'[mode={mode}]'))} finetune + apply")
        train_tantgen(adv_train_feats, mode)
        apply_tantgen(apply_src, out_pcap)
        print(f"  {green('✓')} {out_pcap.relative_to(DEPT_ROOT)}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    if not TANTGEN_ROOT.exists():
        sys.exit(f"{red('ERROR')}: TANTGen repo not found at {TANTGEN_ROOT}")

    print(f"\n{bold('DEPT × TANTGen — Adversarial Generation')}")
    print(f"  TANTGen:  {TANTGEN_ROOT}")
    print(f"  Output:   {OUT_ROOT}")
    print(f"  Attacks:  {', '.join(a.key for a in ATTACKS)}")
    print(f"  Modes:    {', '.join(MODES)}\n")

    # Ensure every directory TANTGen writes into exists (several modules use
    # relative paths and crash on a missing parent dir).
    for d in (
        TG_ORIG_KITSUNE_MAL, TG_ORIG_KITSUNE_BEN, TG_ORIG_KITSUNE_LAB,
        TG_TARGET_TRAIN, TG_ADV_TRAIN, TG_SURR_TRAIN, TG_ADV_TEST, TG_PRETRAIN_DIR,
        TANTGEN_ROOT / "src/TANTGen/models",
        TANTGEN_ROOT / "src/TANTGen/transformer_pretrained_logs",
        TANTGEN_ROOT / "src/TANTGen/transformer_nids_logs",
        TANTGEN_ROOT / "src/TANTGen/TANTGen_logs",
        TANTGEN_ROOT / "src/TANTGen/plots",
        OUT_ROOT,
    ):
        d.mkdir(parents=True, exist_ok=True)

    # Global pretrain (one-time) before any per-attack finetuning.
    ensure_pretrain()

    total_start = time.time()
    failures = []
    for attack in ATTACKS:
        t0 = time.time()
        try:
            process_attack(attack)
            print(f"  {dim(f'[{attack.key}] done in {time.time() - t0:.1f}s')}")
        except subprocess.CalledProcessError as e:
            print(f"  {red(bold('✗ FAILED'))} on {attack.key} (exit {e.returncode})")
            failures.append(attack.key)
        except KeyboardInterrupt:
            print(f"\n{yellow(bold('Interrupted.'))}")
            break

    elapsed = time.time() - total_start
    print(f"\n{bold('Done.')}  Total: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Outputs under: {cyan(str(OUT_ROOT))}")
    if failures:
        print(f"{red('Failed attacks:')} {', '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
