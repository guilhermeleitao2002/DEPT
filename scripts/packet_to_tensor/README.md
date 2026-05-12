# Packet-to-Tensor Pipeline

> Part of [ADEPT](../../README.md). Converts raw PCAP captures into the ragged per-flow tensor format consumed by all downstream models.

## Output schema

For each processed source, the pipeline writes a self-contained directory under `data/tensors/<key>/`:

| File | Shape | Description |
|---|---|---|
| `flow_packets.npy` | `[P_total, 1090]` float32 (memmap) | All packets of all flows in arrival order. Each row is one packet. |
| `flow_offsets.npy` | `[N_flows, 2]` int64 | `(start_idx, length)` into `flow_packets.npy` per flow. |
| `rf_checkpoints.npy` | `[C_total, 24]` float32 | Multi-scale prefix statistics: one row per 10-packet prefix milestone (t=10, 20, 30, …) per flow. |
| `rf_checkpoint_offsets.npy` | `[N_flows, 2]` int64 | `(start_idx, n_checkpoints)` into `rf_checkpoints.npy` per flow. |
| `flow_labels.csv` | `[N_flows]` rows | `flow_idx, label, label_str` — one row per flow. |
| `metadata.json` | JSON | Source metadata: schema version, `n_flows`, `total_packets`, `num_checkpoints`, encoding config. |
| `mean_std.json` | JSON | IPAT z-score reference (shared across all sources in one training run). |

This ragged layout avoids any fixed-window constraint: a 12-packet Mirai flow and a 12,000-packet SSDP flow coexist without padding or truncation.

## Per-packet features (F=1090)

| # | Feature | Description |
|---|---|---|
| 0–1087 | nPrint bits | Tri-state `{-1, 0, 1}` encoding of all IPv4/TCP/UDP/ICMP header bits. |
| 1088 | IPAT | Inter-Packet Arrival Time within the flow (z-score normalized). |
| 1089 | Direction | `1` = forward (flow initiator), `0` = reverse. |

nPrint maps every bit of every header field to a tri-state value: `1` (set), `0` (unset), `-1` (layer absent). This produces a fixed-length 1088-bit vector per packet regardless of which protocols are present:

| Layer | Bits |
|---|---|
| IPv4 | 480 (60 bytes max, incl. options) |
| TCP | 480 (60 bytes max, incl. options) |
| UDP | 64 (8 bytes fixed) |
| ICMP | 64 (8 bytes fixed) |

High-entropy identity fields (source/destination IP, IP ID, source/destination port) are zeroed before any model sees them, forcing the model to learn behavioral patterns rather than memorise endpoints.

## RF prefix statistics (S=24)

At every 10-packet prefix milestone, four aggregations (mean, std, min, max) are computed for each of six base features: `IPAT`, `Packet_Size`, `Payload_Length`, `Payload_Entropy`, `TCP_Flags`, `Direction`. These are computed incrementally via Welford's online algorithm — O(1) memory per flow regardless of flow length.

## Label convention

Labels are derived entirely from the source dataset — no ground-truth CSV is needed.

| Source | `label` | `label_str` |
|---|---|---|
| Kitsune (any attack capture) | `0` | `benign` |
| CIC-IDS2017 (any attack capture) | `0` | `benign` |
| TANTGen adversarial PCAP | `1` | `adversarial` |

Under ADEPT's defense framing, the upstream NIDS is responsible for the raw-dataset attacks. ADEPT's job is specifically to catch TANTGen-style perturbations that evade the upstream model.

## Dataset statistics

Results from `process_all_attacks.py --tantgen` on WSL2. Identity masking on; `rf_checkpoint_stride=10`.

| Source | Flows | Packets |
|---|---:|---:|
| Mirai Botnet (Kitsune) | 32,921 | 764,137 |
| SSDP Flood (Kitsune) | 65,738 | 4,077,266 |
| DDoS (CIC-IDS2017) | 86,272 | 2,151,810 |
| Port Scan (CIC-IDS2017) | 230,156 | 2,026,245 |
| Brute Force FTP+SSH (CIC-IDS2017) | 158,115 | 9,830,521 |
| Brute Force adversarial (×3 modes) | 80 each | 2,500 each |
| DDoS adversarial (×3 modes) | 1,011 each | 9,741 each |
| DDoS Kitsune adversarial (×3 modes) | 15,819 each | 255,827 each |
| Mirai adversarial (×3 modes) | 3,830 each | 192,755 each |
| PortScan adversarial (×3 modes) | 15,839 each | 31,885 each |
| SSDP Flood adversarial (×3 modes) | 64,722 each | 287,921 each |

Total processing time: ~100 min on a desktop Linux machine (WSL2).

## Design decisions

| Decision | Rationale |
|---|---|
| Ragged per-flow storage (offsets + flat array) | Avoids a T_max cap; no padding waste. Works for both 2-packet PortScan flows and 10k-packet SSDP flows. |
| Welford's incremental stats for RF checkpoints | Computes mean/std/min/max for each prefix in O(1) memory per flow, no re-scan required. |
| Shared `mean_std.json` across all sources | Without a shared fit, IPAT z-scores are incomparable across the benign/adversarial boundary. The shared fit covers all PCAPs ADEPT will see during training. |
| Per-flow IPAT (not global IAT) | Each flow has its own timing context; global IAT mixes unrelated flows. |
| Canonical 5-tuple flow ID | Order-independent — forward and reverse packets belong to the same flow. |
| Identity masking (IPs, ports, IP ID zeroed) | Prevents models from memorising source/destination addresses instead of behavioral patterns. |
