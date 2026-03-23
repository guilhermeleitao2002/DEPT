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

1. **Sequence tensor** `[W, N, F]` — per-packet header features for the TCN and Purification Transformer
2. **RF statistics** `[W, S]` — aggregated window statistics for the Random Forest

Both outputs are synchronized at the N-th packet boundary ("N-Packet Snapshot"): every window contains exactly N packets from the same flow.

### Processing stages

```
PCAP file
  │
  ├─ [1] packet_parser    Extract per-packet features (Scapy)
  │       → Timestamp, Packet_Size, Protocol, Ports, TCP_Flags,
  │         Payload_Length, Payload_Entropy
  │
  ├─ [2] flow_grouper     Group packets by canonical 5-tuple
  │       → Per-flow IPAT (Inter-Packet Arrival Time)
  │       → Direction (forward / reverse)
  │
  └─ [3] tensor_builder   Window into N-packet snapshots
          → Sequence tensor [W, N, 7]   (z-score standardized)
          → RF stats tensor [W, 24]     (mean/std/min/max of 6 features)
```

### Key design decisions

| Decision | Rationale |
|---|---|
| **Per-flow IPAT** (not global IAT) | Each flow has its own timing; global IAT mixes unrelated flows |
| **Canonical 5-tuple** for flow ID | Order-independent — forward and reverse packets belong to the same flow |
| **Drop incomplete windows** (no padding) | Padding with zeros would pollute RF aggregated statistics |
| **Separate RF and sequence outputs** | RF uses statistical summaries; TCN/Transformer uses raw sequences — different input representations for ensemble diversity |

### Sequence features (F=7)

| # | Feature | Description |
|---|---|---|
| 0 | `IPAT` | Inter-Packet Arrival Time within the flow |
| 1 | `Packet_Size` | Frame length (bytes) |
| 2 | `Protocol` | IP protocol number (TCP=6, UDP=17, ...) |
| 3 | `TCP_Flags` | TCP flags as integer (0 if not TCP) |
| 4 | `Payload_Length` | IP payload length (bytes) |
| 5 | `Payload_Entropy` | Shannon entropy of payload bytes |
| 6 | `Direction` | 1 = forward (flow initiator), 0 = reverse |

### RF statistics (S=24)

For each of the 6 base features (IPAT, Packet_Size, Payload_Length, Payload_Entropy, TCP_Flags, Direction), four aggregations are computed over the N-packet window: **mean, std, min, max**.

## Usage

### From a PCAP file

```bash
python scripts/run_packet_to_tensor.py \
    -i data/raw/capture.pcap \
    -o data/processed/capture/
```

### From a pre-extracted CSV

```bash
python scripts/run_packet_to_tensor.py \
    -i data/features/packets.csv \
    -o data/processed/output/ \
    --from-csv
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
| `packets.csv` | Intermediate per-packet features (from PCAP only) |
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
