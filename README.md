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
| 0–1087 | nPrint bits | Tri-state {-1, 0, 1} encoding of all IPv4/TCP/UDP/ICMP header bits |
| 1088 | `IPAT` | Inter-Packet Arrival Time within the flow (z-score normalized) |
| 1089 | `Direction` | 1 = forward (flow initiator), 0 = reverse |

### RF statistics (S=24)

For each of the 6 base features (IPAT, Packet_Size, Payload_Length, Payload_Entropy, TCP_Flags, Direction), four aggregations are computed over the N-packet window: **mean, std, min, max**.

## Usage

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

### Output files

| File | Description |
|---|---|
| `sequence.npy` | Sequence tensor `[W, N, F]` — float32 |
| `rf_stats.npy` | RF statistics `[W, S]` — float32 |
| `flow_ids.npy` | Flow 5-tuple per window (pickled) |
| `mean_std.json` | Normalization statistics (reuse for test sets) |
| `packets.csv` | Intermediate per-packet DataFrame (1088 nPrint bits + metadata columns) |
| `metadata.json` | Pipeline config, tensor shapes, feature names |

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

The packet parsing logic is adapted from [TANTGen](https://github.com/inesPisswormo/TANTGen) (Inês Pissarra, IST/INESC-ID), a Transformer-based adversarial network traffic generator that ADEPT is designed to defend against.
