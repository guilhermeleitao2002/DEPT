# ADEPT — Adaptive Defense via Ensemble and Purification Transformer

Research implementation for a Master Thesis at IST/INESC-ID (Universidade de Lisboa).

ADEPT is a defense framework against adversarial attacks on Machine Learning-based Network Intrusion Detection Systems (NIDS). It combines heterogeneous base learners with a meta-learner Transformer and a purification module to detect and denoise adversarial network traffic.

## Architecture Overview

```
                        Raw PCAP / Dataset
                              │
                    ┌─────────┴─────────┐
                    │  Packet-to-Tensor  │  ← this repo (current stage)
                    │     Pipeline       │
                    └────┬─────────┬─────┘
                         │         │
              ┌──────────┘         └──────────┐
              ▼                               ▼
    Sequence Tensor [W,N,F]          RF Statistics [W,S]
              │                               │
              ▼                               ▼
     ┌────────────────┐              ┌────────────────┐
     │      TCN       │              │  Random Forest  │
     │ (temporal base │              │ (statistical    │
     │    learner)    │              │  base learner)  │
     └───────┬────────┘              └───────┬────────┘
             │                               │
             └───────────┬───────────────────┘
                         ▼
              ┌──────────────────┐
              │  Meta-Learner    │
              │ (Mini-Transformer│
              │  + self-attn)    │
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │   Purification   │
              │   Transformer    │
              │ (Decoder-only    │
              │  Seq2Seq)        │
              └──────────────────┘
```

**Current status**: only the Packet-to-Tensor pipeline (top box) is implemented.

## Packet-to-Tensor Pipeline

Converts raw PCAP captures (or datasets like CIC-IDS2018) into the two synchronized tensor inputs consumed by the ADEPT ensemble:

1. **Sequence tensor** `[W, N, F]` — per-packet nPrint bit vectors for the TCN and Purification Transformer
2. **RF statistics** `[W, S]` — aggregated window statistics for the Random Forest

| Dimension | Meaning | Value |
|---|---|---|
| `W` | Number of windows (samples) | varies with dataset |
| `N` | Window size — packets per window | 10 (default) |
| `F` | Features per packet | 1090 (1088 nPrint bits + IPAT + Direction) |
| `S` | RF statistic vector length | 24 (6 features × 4 aggregations) |

Both outputs are synchronized at the N-th packet boundary ("N-Packet Snapshot"): every window contains exactly N packets from the same flow.

### Packet encoding — nPrint

Packet headers are encoded using the [nPrint](https://nprint.github.io) bit-level representation (Holland et al., CCS 2021). Every bit of every header field is mapped to a tri-state value:

| Value | Meaning |
|---|---|
| `1` | Bit is set |
| `0` | Bit is unset |
| `-1` | Protocol layer absent (e.g. no TCP header in a UDP packet) |

This produces a **fixed-length 1088-bit vector per packet**, regardless of which protocols are present:

| Layer | Bits |
|---|---|
| IPv4 | 480 (60 bytes max, incl. options) |
| TCP | 480 (60 bytes max, incl. options) |
| UDP | 64 (8 bytes fixed) |
| ICMP | 64 (8 bytes fixed) |

High-entropy identity fields (source/destination IP, IP ID, source/destination port) are **zeroed out** before training — this prevents the model from overfitting to specific addresses and forces it to learn behavioral patterns instead.

### Processing stages

```
PCAP file
  │
  ├─ [1] packet_parser    Encode each packet as a 1088-bit nPrint vector (Scapy)
  │       → 1088 header bits (IPv4 + TCP/UDP/ICMP)
  │       → Identity masking: zero out IPs, ports, IP ID
  │       → Metadata: Timestamp, Packet_Size, Protocol, TCP_Flags,
  │                   Payload_Length, Payload_Entropy
  │
  ├─ [2] flow_grouper     Group packets by canonical 5-tuple
  │       → Per-flow IPAT (Inter-Packet Arrival Time)
  │       → Direction (forward / reverse)
  │
  └─ [3] tensor_builder   Window into N-packet snapshots
          → Sequence tensor [W, N, 1090]   (1088 nPrint bits + IPAT + Direction)
          → RF stats tensor [W, 24]        (mean/std/min/max of 6 metadata features)
```

### Key design decisions

| Decision | Rationale |
|---|---|
| **Per-flow IPAT** (not global IAT) | Each flow has its own timing; global IAT mixes unrelated flows |
| **Canonical 5-tuple** for flow ID | Order-independent — forward and reverse packets belong to the same flow |
| **Drop incomplete windows** (no padding) | Padding with zeros would pollute RF aggregated statistics |
| **Separate RF and sequence outputs** | RF uses statistical summaries; TCN/Transformer uses raw sequences — different input representations for ensemble diversity |

### Sequence features (F=1090)

| # | Feature | Description |
|---|---|---|
| 0-1087 | nPrint bits | Tri-state {-1, 0, 1} encoding of all IPv4/TCP/UDP/ICMP header bits |
| 1088 | `IPAT` | Inter-Packet Arrival Time within the flow (z-score normalized) |
| 1089 | `Direction` | 1 = forward (flow initiator), 0 = reverse |

### RF statistics (S=24)

For each of the 6 base features (IPAT, Packet_Size, Payload_Length, Payload_Entropy, TCP_Flags, Direction), four aggregations are computed over the N-packet window: **mean, std, min, max**.

## Usage

Run commands from the `ADEPT/` directory.

### From a PCAP file

```bash
python scripts/run_packet_to_tensor.py \
    -i data/raw/capture.pcap \
    -o data/processed/capture/
```

### Custom window size (N=20)

```bash
python scripts/run_packet_to_tensor.py \
    -i data/raw/capture.pcap \
    -o data/processed/output/ \
    -N 20
```

### Apply training statistics to a test set

```bash
python scripts/run_packet_to_tensor.py \
    -i data/raw/test.pcap \
    -o data/processed/test/ \
    --mean-std data/processed/train/mean_std.json
```

### Batch process the thesis attack set

`scripts/process_all_attacks.py` runs the packet-to-tensor pipeline across the predefined attack PCAPs under `data/Kitsune/pcap/` and `data/CIC-IDS-2017/pcap/`, writing outputs to `data/tensors/`.

Install Python dependencies first:

```bash
pip install -r requirements.txt
```

If you want to process any CIC-IDS2017 attack, you also need `editcap` from the Wireshark CLI tools because the script time-filters those PCAPs before tensorization:

```bash
sudo apt install wireshark-common
```

Process all predefined attacks (the Monday benign baseline is not included unless requested):

```bash
python scripts/process_all_attacks.py
```

Preview what would run without processing anything:

```bash
python scripts/process_all_attacks.py --dry-run
```

Process only one attack:

```bash
python scripts/process_all_attacks.py --only syn_dos
```

Re-process an attack even if `data/tensors/<attack>/metadata.json` already exists:

```bash
python scripts/process_all_attacks.py --only mirai --force
```

Override the window size from `config/pipeline.yaml`:

```bash
python scripts/process_all_attacks.py -N 20
```

Include the Monday benign baseline:

```bash
python scripts/process_all_attacks.py --include-benign
```

Available `--only` keys:

```text
mirai
syn_dos
fuzzing
ssdp_flood
ddos
dos_slow
portscan
monday_benign
```

### Output files

| File | Downstream consumer | Description |
|---|---|---|
| `sequence.npy` | TCN, Purification Transformer | Sequence tensor `[W, N, F]` (float32). Each window is a raw per-packet sequence — the TCN reads it for classification, the Purification Transformer reads it to denoise adversarial perturbations. |
| `rf_stats.npy` | Random Forest | RF statistics `[W, S]` (float32). Each window summarized as mean/std/min/max of 6 metadata features — intentionally a different representation from `sequence.npy` for ensemble diversity. |
| `flow_ids.npy` | Evaluation, RL Agent | Flow 5-tuple per window. Maps each window back to its originating flow for per-flow metrics, RL reward correlation, and debugging. |
| `mean_std.json` | Inference pipeline | Z-score normalization statistics (IPAT mean/std). Must be passed via `--mean-std` when processing test sets or live traffic to ensure consistent scaling. |
| `packets.csv` | — | Intermediate per-packet DataFrame (1088 nPrint bits + metadata columns). Useful for inspection/debugging. |
| `metadata.json` | — | Full pipeline provenance: input file, encoding config, masking, tensor shapes, flow counts. For reproducibility. |

## Configuration

All pipeline parameters are in `config/pipeline.yaml`: window size, stride, feature lists, aggregation functions, and standardization columns.

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `scapy`, `pyyaml`.

## Project structure

```
ADEPT/
├── config/
│   └── pipeline.yaml                  # Pipeline parameters
├── scripts/
│   └── run_packet_to_tensor.py        # CLI entry-point
├── src/
│   └── feature_extraction/
│       ├── packet_parser.py           # PCAP → per-packet DataFrame
│       ├── flow_grouper.py            # 5-tuple grouping + IPAT + Direction
│       └── tensor_builder.py          # Windowing + standardization + dual output
└── requirements.txt
```

## Acknowledgements

The packet parsing logic is adapted from [TANTGen](https://github.com/inespissarra/Transformer-based_Adversarial_Network_Traffic_Generator) (Inês Pissarra, IST/INESC-ID), a Transformer-based adversarial network traffic generator that ADEPT is designed to defend against.
