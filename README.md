# ADEPT — Adaptive Defense via Ensemble and Purification Transformer

Research implementation for my Master Thesis at IST/INESC-ID (Universidade de Lisboa).

ADEPT is a defense framework against adversarial attacks on Machine Learning-based Network Intrusion Detection Systems (NIDS). It combines heterogeneous base learners with a meta-learner Transformer and a purification module to detect and denoise adversarial network traffic.

## Architecture overview

```
                        Raw PCAP / Dataset
                              │
                    ┌─────────┴─────────┐
                    │  Packet-to-Tensor  │  ── Phase 1 ✅
                    │     Pipeline       │
                    └────┬─────────┬─────┘
                         │         │
              ┌──────────┘         └──────────┐
              ▼                               ▼
    Sequence Tensor [W,N,F]          RF Statistics [W,S]
              │                               │
              ▼                               ▼
     ┌────────────────┐                ┌────────────────┐
     │      TCN       │ ── Phase 3 ✅ │  Random Forest  │ ── Phase 2 ✅
     │ (temporal base │                │ (statistical   │
     │    learner)    │                │  base learner) │
     └───────┬────────┘                └───────┬────────┘
             │                                 │
             └───────────┬─────────────────────┘
                         ▼
              ┌──────────────────┐
              │  Meta-Learner    │
              │ (Mini-Transformer│            ── Phase 4 ⏳
              │  + self-attn)    │
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │   Purification   │
              │   Transformer    │            ── Phase 5 ⏳
              │ (Decoder-only    │
              │  Seq2Seq)        │
              └──────────────────┘
```

ADEPT is paired with **TANTGen**, a Transformer-based black-box adversarial network traffic generator (sibling repo). TANTGen produces the adversarial PCAPs that ADEPT learns to detect and denoise — the two systems form a complete attack/defense loop.

## Implementation phases

| # | Phase | Status | Code | Documentation |
|---|---|---|---|---|
| 1 | Packet-to-Tensor pipeline | ✅ done | [`scripts/packet_to_tensor/`](scripts/packet_to_tensor/) | [Phase 1 README](scripts/packet_to_tensor/README.md) |
| 2 | Random Forest base learner | ✅ done | [`scripts/rf/`](scripts/rf/) | [Phase 2 README](scripts/rf/README.md) |
| — | TANTGen adversarial generation | ✅ done | [`scripts/adversarial_generation/`](scripts/adversarial_generation/) | [TANTGen integration README](scripts/adversarial_generation/README.md) |
| 3 | TCN base learner | ✅ done | [`scripts/tcn/`](scripts/tcn/) | [Phase 3 README](scripts/tcn/README.md) |
| 4 | Mini-Transformer meta-learner | ⏳ planned | — | — |
| 5 | Purification Transformer | ⏳ planned | — | — |
| 6 | RL Adaptation Agent | ⏳ planned | — | — |

## Core concepts

These are the cross-cutting ideas that show up in every stage.

### N-Packet Snapshot

The atomic unit of analysis in ADEPT is a fixed-size window of **N packets from the same flow** (default N=10). Both the sequence and statistics views are produced **synchronized at the N-th packet boundary** — every window is exactly N packets, and the two tensors describe the same N-packet slice from two different angles. Incomplete tail windows are dropped (no zero padding) so the RF aggregates stay clean.

### Two synchronized tensor views per window

| Tensor | Shape | Consumer | Representation |
|---|---|---|---|
| `sequence.npy` | `[W, N, F=1090]` float32 | TCN, Purification Transformer | Raw per-packet bit-level header sequence (nPrint) + IPAT + Direction |
| `rf_stats.npy` | `[W, S=24]` float32 | Random Forest | mean/std/min/max aggregated over the same N packets, on 6 base features |

Keeping these representations **disjoint** is deliberate — it forces ensemble diversity between the temporal and statistical base learners.

### nPrint bit-level packet encoding

Each packet's header is encoded as a fixed-length **1088-bit tri-state vector** (`-1` = layer absent, `0` = bit unset, `1` = bit set). This is the [nPrint](https://nprint.github.io) representation (Holland et al., CCS 2021), inspired by FlowPaint (SIGCOMM '26). High-entropy identity fields (IPs, ports, IP ID) are zeroed out so the model learns behavioral patterns instead of memorising endpoints. See the [Phase 1 README](scripts/packet_to_tensor/README.md) for the full bit layout.

### Dataset-driven label convention

ADEPT defends specifically against **TANTGen-style adversarial perturbations** — not the raw-dataset attacks the upstream NIDS already catches. Labels therefore come straight from the source dataset, with no ground-truth CSV needed:

| Dataset | `label` | `label_str` |
|---|---|---|
| Kitsune (any attack capture) | `0` | `benign` |
| CIC-IDS2017 (any attack capture) | `0` | `benign` |
| TANTGen-generated adversarial PCAP | `1` | `adversarial` |

This convention is enforced by [`scripts/packet_to_tensor/attach_labels.py`](scripts/packet_to_tensor/attach_labels.py) and propagates through every downstream model.

### Flow-level discipline

Whenever data is split or capped, **flows are atomic** — every window from the same canonical 5-tuple stays together. This prevents the trivial leak where two windows of the same flow end up on opposite sides of a train/test split.

## Repository layout

```
ADEPT/
├── README.md                                ← you are here
├── requirements.txt
├── config/
│   ├── pipeline.yaml                        # Phase 1 — Packet-to-Tensor parameters
│   ├── rf.yaml                              # Phase 2 — Random Forest
│   └── tcn.yaml                             # Phase 3 — TCN
├── data/                                    # Raw PCAPs/CSVs + produced tensors & models
│   ├── Kitsune/                             #   raw Kitsune captures + per-packet labels
│   ├── CIC-IDS-2017/                        #   raw CIC-IDS2017 day captures + per-flow labels
│   ├── adversarial/<attack>/<*_adversarial_{both,iat,size}.pcap>
│   │                                        #   Phase-A output (TANTGen)
│   ├── tensors/                             #   Phase-1 output (one subdir per attack)
│   │   ├── <attack>/                        #     Kitsune / CIC tensors (label=benign)
│   │   ├── adversarial/<attack>_adv_<mode>/ #     Adversarial tensors (label=adversarial)
│   │   ├── .filtered/                       #     editcap'd CIC sub-PCAPs (cache)
│   │   └── mean_std.json                    #     shared IPAT z-score reference
│   └── models/
│       ├── rf/                              #   Phase-2 output (rf.joblib + plots + metrics)
│       └── tcn/                             #   Phase-3 output (tcn.pt + plots + metrics)
├── scripts/
│   ├── packet_to_tensor/                    # Phase 1 CLIs
│   │   ├── README.md                        #   ← Phase 1 documentation
│   │   ├── run_packet_to_tensor.py          #   Single-PCAP pipeline (batch + online modes)
│   │   ├── attach_labels.py                 #   Window label attachment
│   │   └── process_all_attacks.py           #   Batch runner for all attacks
│   ├── adversarial_generation/              # TANTGen driver
│   │   ├── README.md                        #   ← TANTGen integration documentation
│   │   ├── generate_adversarials.py
│   │   └── __init__.py
│   ├── rf/                                  # Phase 2 CLI
│   │   ├── README.md                        #   ← Phase 2 documentation
│   │   └── run_rf.py                        #   Train + evaluate Random Forest
│   └── tcn/                                 # Phase 3 CLI
│       ├── README.md                        #   ← Phase 3 documentation
│       └── run_tcn.py                       #   Train + evaluate TCN
└── src/
    ├── feature_extraction/                  # Phase 1 internals
    │   ├── pcap_parser.py
    │   ├── flow_grouper.py
    │   ├── tensor_builder.py
    │   └── incremental_pipeline.py
    └── models/                              # Phase 2+ internals
        ├── common/
        │   └── dataset.py                   #   Shared flow-level split (RF + TCN)
        ├── rf/
        │   ├── dataset.py
        │   ├── train.py
        │   └── evaluate.py
        └── tcn/
            ├── model.py                     #   Causal TCN architecture
            ├── dataset.py
            ├── train.py
            └── evaluate.py
```

## Requirements

### Python

```bash
pip3 install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `scapy`, `pyyaml`, `tqdm`, `scikit-learn`, `joblib`, `matplotlib`, `torch`.

### System

```bash
# editcap (used to time-filter CIC-IDS2017 day captures)
sudo apt install wireshark-common -y

# tshark (only needed when driving TANTGen)
sudo apt install tshark -y
```

For TANTGen integration, additional setup is needed (sibling repo install, Cython feature-extraction build); see the [TANTGen integration README](scripts/adversarial_generation/README.md).

## End-to-end pipeline

A complete ADEPT training run currently goes through four stages. All commands are run from `ADEPT/`.

```bash
# 1. Generate TANTGen adversarial PCAPs (6 attacks × 3 modes)
python scripts/adversarial_generation/generate_adversarials.py

# 2. Build tensors for the 6 TANTGen-supported attacks AND their
#    matching adversarial PCAPs in one go
python scripts/packet_to_tensor/process_all_attacks.py --tantgen

# 3. Train + evaluate the Random Forest base learner
python scripts/rf/run_rf.py

# 4. Train + evaluate the TCN base learner
python scripts/tcn/run_tcn.py
```

Steps 3 and 4 share the same `data/tensors/` artifacts and the same flow-level split (via `src/models/common/dataset.py`), so RF and TCN results are directly comparable.

To process the full Kitsune + CIC-IDS2017 attack set (16 captures) instead, drop `--tantgen`. Use `--include-benign` to add the Monday benign baseline.

Each step is fully documented in its own README:

- **Step 1 — Packet-to-Tensor:** [scripts/packet_to_tensor/README.md](scripts/packet_to_tensor/README.md)
- **Step 2 — Adversarial generation:** [scripts/adversarial_generation/README.md](scripts/adversarial_generation/README.md)
- **Step 3 — Random Forest:** [scripts/rf/README.md](scripts/rf/README.md)
- **Step 4 — TCN:** [scripts/tcn/README.md](scripts/tcn/README.md)

Subsequent phases (meta-Transformer, Purification Transformer, RL Adaptation Agent) will consume the RF and TCN logits produced by steps 3 and 4.

## Reference documents

The following reference PDFs accompany this repository (kept in the parent `Master Thesis/` directory):

- [PIC2___Guilherme_Leitão.pdf](PIC2___Guilherme_Leitão.pdf) — ADEPT framework specification (this repo).
- `99236-Ines-Pissarra.pdf` — TANTGen specification (sibling adversarial generator).
