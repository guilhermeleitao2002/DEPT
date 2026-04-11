# ADEPT — Adaptive Defense via Ensemble and Purification Transformer

Research implementation for my Master Thesis at IST/INESC-ID (Universidade de Lisboa).

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

## Module Reference

### `src/feature_extraction/pcap_parser.py` — PCAP → nPrint bits + metadata

Reads a PCAP with Scapy and encodes each packet as a 1088-bit nPrint vector plus a lightweight metadata row. Kept separate from pandas to avoid the ~10× memory overhead of storing 1088 columns in a DataFrame.

| Function | Inputs | Outputs | Notes |
|---|---|---|---|
| `extract_packet_features(pcap_path, mask_prefixes)` | PCAP path; list of field prefixes to zero | `(nprint_bits, meta_df)` — `int8 (P, 1088)` + `DataFrame (P, 10)` | Main entry point. Chunks 50k packets at a time to keep peak RAM low. |
| `_encode_packet(pkt)` | Scapy packet | `(list[int 1088], dict)` | Encodes one packet; detects layers, extracts raw bytes, calls `_bytes_to_bits()`. |
| `_bytes_to_bits(raw_bytes, max_bytes, fill=-1)` | Raw bytes or None; target size; fill value | `list[int]` of `max_bytes × 8` values `{-1, 0, 1}` | MSB-first expansion; missing layer → fill with `-1`; unused option bytes → `0`. |
| `_shannon_entropy(data)` | `bytes` | `float` ∈ [0, 8] | H = −Σ(p·log₂p). Used for `Payload_Entropy` metadata column. |
| `apply_masking(df, mask_prefixes)` | nPrint DataFrame; prefix list | Masked DataFrame | DataFrame-level API (deprecated — pipeline uses in-place numpy masking). |

**`meta_df` columns**: `Timestamp, Source_IP, Destination_IP, Source_Port, Destination_Port, Protocol, Packet_Size, Payload_Length, Payload_Entropy, TCP_Flags`

**Identity masking**: columns matching a prefix (e.g., `ipv4_src_0` … `ipv4_src_31`) are zeroed in-place on the numpy array. Controlled by `mask_prefixes` in `config/pipeline.yaml`.

---

### `src/feature_extraction/flow_grouper.py` — 5-tuple grouping + IPAT + Direction

Groups packets into bidirectional flows and annotates each packet with per-flow IPAT and Direction. IPAT is per-flow (not global) because each flow window is processed independently by the ensemble.

| Function | Inputs | Outputs | Notes |
|---|---|---|---|
| `group_by_flow(meta_df)` | `meta_df` (P, 10) from `pcap_parser` | `(meta_df_updated, flow_indices, sort_order)` | Main entry point. |
| `_compute_canonical_keys(df)` | `meta_df` | `pd.Series` of 5-tuples | — |

**Canonical 5-tuple**: `(min_ip, max_ip, min_port, max_port, protocol)` — order-independent, so forward and reverse packets map to the same flow.

**`group_by_flow` return values**:
- `meta_df` — updated with `IPAT (float64, seconds)` and `Direction (int8: 1=fwd, 0=rev)`, sorted by Timestamp (stable mergesort)
- `flow_indices` — `OrderedDict[5-tuple → np.ndarray[int64]]`: each flow's sorted row indices
- `sort_order` — `np.ndarray[int64]`: the permutation applied; **caller must apply this to `nprint_bits`** (`nprint_bits = nprint_bits[sort_order]`) to keep the arrays in sync

---

### `src/feature_extraction/tensor_builder.py` — windowing → sequence.npy + rf_stats.npy

Slides a non-overlapping N-packet window over each flow and writes the two output tensors directly to memory-mapped `.npy` files, avoiding OOM on large datasets and allowing for pipeline to be run locally.

| Function | Inputs | Outputs | Notes |
|---|---|---|---|
| `build_tensors(nprint_bits, meta_df, flow_indices, output_dir, ...)` | `int8 (P, 1088)`, `DataFrame`, `OrderedDict`, config params | `dict` with paths + metadata | Main entry point. |
| `_load_or_compute_stats(df, columns, mean_std_path)` | DataFrame; column list; JSON path | `{"IPAT": {"mean": float, "std": float}}` | Loads from JSON if it exists, otherwise computes and saves. |
| `_compute_rf_stats(window_meta, feature_names, base_features, agg_functions)` | `float32 (N, M)` window; name lists | `float32 (S,)` | Applies mean/std/min/max per base feature over the N-packet window. |
| `_count_windows(flow_indices, window_size, stride)` | flow map; N; stride | `int` | Pre-counts total windows to pre-allocate memmaps. Formula: Σ `(n − N) // stride + 1` for each flow with `n ≥ N`. |

**`build_tensors` return dict keys**: `sequence_path`, `rf_stats_path`, `flow_ids`, `stats`, `feature_names` (1090 names), `rf_stat_names` (24 names), `num_windows`

**Memory strategy**: pre-count → allocate `np.lib.format.open_memmap` → write one window at a time (~43 KB working set at N=10).

---

### `scripts/run_packet_to_tensor.py` — single-PCAP CLI

Orchestrates the three modules for a single PCAP file.

```
[1/3] extract_packet_features()  →  nprint_bits (P,1088) int8  +  meta_df (P,10)
[2/3] group_by_flow()            →  meta_df+IPAT+Direction, flow_indices, sort_order
      nprint_bits = nprint_bits[sort_order]          ← critical sync step
[3/3] build_tensors()            →  sequence.npy, rf_stats.npy, flow_ids.npy
      write metadata.json
```

**CLI flags**: `-i/--input`, `-o/--output-dir`, `-c/--config`, `-N/--window-size`, `--mean-std` (pass training stats to a test set), `--no-mask`

**Output files**:

| File | Shape | Consumer |
|---|---|---|
| `sequence.npy` | `(W, N, 1090)` float32 | TCN, Transformer |
| `rf_stats.npy` | `(W, 24)` float32 | Random Forest |
| `flow_ids.npy` | `(W,)` object | Evaluation, RL Agent |
| `mean_std.json` | — | Normalization stats; pass via `--mean-std` at test time |
| `window_labels.csv` | Training, Evaluation | Per-window ground-truth labels (`label` 0/1, `label_str`). Produced by `attach_labels.py`. |
| `metadata.json` | — | Full pipeline information |

---

### `scripts/attach_labels.py` — window label attachment

Produces `window_labels.csv` inside a tensor directory by joining ground-truth label files with `flow_ids.npy`. Handles two alignment problems:

- **Kitsune** — labels are per-packet in PCAP order. The pipeline reorders packets by timestamp inside `flow_grouper`; this script does a lightweight PCAP re-read (5-tuple + timestamp only, no nPrint encoding — ~5–10× faster) to reconstruct the same sort order, majority-votes a label per canonical flow, then joins with `flow_ids.npy`.
- **CIC-IDS2017** — labels are per-flow in CICFlowMeter TrafficLabelling CSVs, keyed by a unidirectional 5-tuple (`Flow ID` column). The script canonicalises those 5-tuples using the same bidirectional key logic as `flow_grouper._compute_canonical_keys`, then joins directly with `flow_ids.npy`. No PCAP re-read needed.

**Output** — `<tensor-dir>/window_labels.csv`:

| Column | Description |
|---|---|
| `window_idx` | Row index into `sequence.npy` / `rf_stats.npy` |
| `label` | `1` = attack, `0` = benign, `-1` = unmatched flow |
| `label_str` | Original label string (e.g. `"DDoS"`, `"BENIGN"`, `"mirai"`) |

**CLI flags**: `-d/--dataset {kitsune,cic}`, `-t/--tensor-dir`, `-l/--labels-csv`, `-i/--pcap` (required for kitsune), `--force`

`process_all_attacks.py` calls this automatically after each successful pipeline run, and also retroactively for any tensor directory where `window_labels.csv` is missing.

---

### `scripts/process_all_attacks.py` — batch processor for all attacks

Automates batch processing across all available attack PCAPs. For CIC-IDS2017 attacks, uses `editcap` to extract the attack time window before processing.

**Attack registry** (pass keys to `--only`):

| Key | Name | Dataset | Time window (UTC) |
|---|---|---|---|
| `mirai` | Mirai Botnet | Kitsune | — |
| `syn_dos` | SYN DoS | Kitsune | — |
| `fuzzing` | Fuzzing | Kitsune | — |
| `ssdp_flood` | SSDP Flood | Kitsune | — |
| `arp_mitm` | ARP MitM | Kitsune | — |
| `active_wiretap` | Active Wiretap | Kitsune | — |
| `os_scan` | OS Scan | Kitsune | — |
| `ssl_renegotiation` | SSL Renegotiation | Kitsune | — |
| `video_injection` | Video Injection | Kitsune | — |
| `ddos` | DDoS | CIC-IDS2017 | 2017-07-07 19:30–21:05 |
| `dos_slow` | DoS Slow (Slowloris + Hulk + …) | CIC-IDS2017 | 2017-07-05 13:40–15:30 |
| `portscan` | Port Scan | CIC-IDS2017 | 2017-07-07 17:00–19:30 |
| `brute_force_ftp` | Brute Force FTP | CIC-IDS2017 | 2017-07-04 13:15–14:25 |
| `brute_force_ssh` | Brute Force SSH | CIC-IDS2017 | 2017-07-04 18:00–19:05 |
| `web_attacks` | Web Attacks (Brute Force + XSS + SQLi) | CIC-IDS2017 | 2017-07-06 13:15–14:45 |
| `infiltration` | Infiltration | CIC-IDS2017 | 2017-07-06 18:15–18:40 |
| `monday_benign` | Monday Benign | CIC-IDS2017 | 2017-07-03 09:00–11:00 |

**CLI flags**: `--only KEY [KEY...]`, `--force`, `--dry-run`, `-N/--window-size`, `--include-benign`, `--keep-filtered`

**Per-attack flow**:
1. Skip if `metadata.json` exists (unless `--force`)
2. If CIC-IDS2017: check for a cached filtered PCAP in `data/tensors/.filtered/`; run `editcap` only if not cached
3. Run `run_packet_to_tensor.py` as a subprocess
4. Delete the filtered PCAP (unless `--keep-filtered`)

## Usage

Run commands from the `ADEPT/` directory.

### From a PCAP file

```bash
python scripts/run_packet_to_tensor.py \
    -i data/Kitsune/pcap/Mirai_pcap.pcap \
    -o data/tensors/mirai/
```

### Custom window size (N=20)

```bash
python scripts/run_packet_to_tensor.py \
    -i data/Kitsune/pcap/Mirai_pcap.pcap \
    -o data/tensors/mirai/ \
    -N 20
```

### Apply training statistics to a test set

```bash
python scripts/run_packet_to_tensor.py \
    -i data/Kitsune/pcap/Mirai_pcap.pcap \
    -o data/tensors/mirai/ \
    --mean-std data/tensors/mirai/mean_std.json
```

### Batch process the thesis attack set

`scripts/process_all_attacks.py` runs the packet-to-tensor pipeline across all available attack PCAPs under `data/Kitsune/pcap/` and `data/CIC-IDS-2017/pcap/`, writing outputs to `data/tensors/`.

Install Python dependencies first:

```bash
pip install -r requirements.txt
```

If you want to process any CIC-IDS2017 attack, you also need `editcap` from the Wireshark CLI tools because the script time-filters those PCAPs before tensorization, in order to ignore benign traffic by default:

```bash
sudo apt install wireshark-common
```

Process all predefined attacks (the Monday benign baseline is not included unless requested):

```bash
python3 scripts/process_all_attacks.py
```

Preview what would run without processing anything:

```bash
python3 scripts/process_all_attacks.py --dry-run
```

Process only one attack:

```bash
python3 scripts/process_all_attacks.py --only syn_dos
```

Re-process an attack even if `data/tensors/<attack>/metadata.json` already exists:

```bash
python3 scripts/process_all_attacks.py --only mirai --force
```

Override the window size from `config/pipeline.yaml`:

```bash
python3 scripts/process_all_attacks.py -N 20
```

Include the Monday benign baseline:

```bash
python3 scripts/process_all_attacks.py --include-benign
```

Keep the time-filtered CIC-IDS2017 PCAPs after processing (re-runs reuse them automatically):

```bash
python3 scripts/process_all_attacks.py --keep-filtered
```

Available `--only` keys:

```text
mirai
syn_dos
fuzzing
ssdp_flood
arp_mitm
active_wiretap
os_scan
ssl_renegotiation
video_injection
ddos
dos_slow
portscan
brute_force_ftp
brute_force_ssh
web_attacks
infiltration
monday_benign
```

### Output files

| File | Downstream user | Description |
|---|---|---|
| `sequence.npy` | TCN, Purification Transformer | Sequence tensor `[W, N, F]` (float32). Each window is a raw per-packet sequence — the TCN reads it for classification, the Purification Transformer reads it to denoise adversarial perturbations. |
| `rf_stats.npy` | Random Forest | RF statistics `[W, S]` (float32). Each window summarized as mean/std/min/max of 6 metadata features — intentionally a different representation from `sequence.npy` for ensemble diversity. |
| `flow_ids.npy` | Evaluation, RL Agent | Flow 5-tuple per window. Maps each window back to its originating flow for per-flow metrics, eventual RL reward correlation, and debugging. |
| `mean_std.json` | Inference pipeline | Z-score normalization statistics (IPAT mean/std). Must be passed via `--mean-std` when processing test sets or live traffic to ensure consistent scaling. |
| `packets.csv` | — | Intermediate per-packet DataFrame (1088 nPrint bits + metadata columns). Useful for inspection/debugging. Only produced when `run_packet_to_tensor` is called through CLI, to help avoid disk over-usage |
| `metadata.json` | — | Full pipeline provenance: input file, encoding config, masking, tensor shapes, flow counts. For reproducibility and debugging. |

## Processed dataset statistics

Results from running `process_all_attacks.py` on a desktop Linux machine (WSL2). Window size N=10, non-overlapping stride, identity masking ON.

| Attack | Dataset | Packets | Flows (total / usable) | Windows | nPrint Parse time | Total time |
|---|---|---:|---:|---:|---:|---:|
| Mirai Botnet | Kitsune | 764,137 | 32,921 / 164 | 69,340 | 2m 38s | 2m 53s |
| SYN DoS | Kitsune | 2,771,276 | 753 / 33 | 276,871 | 11m 23s | 12m 23s |
| Fuzzing | Kitsune | 2,244,139 | 1,803 / 19 | 224,226 | 10m 01s | 10m 54s |
| SSDP Flood | Kitsune | 4,077,266 | 65,738 / 65,470 | 378,331 | 16m 51s | 18m 32s |
| ARP MitM | Kitsune | 2,504,267 | 109 / 30 | 250,394 | 11m 27s | 12m 27s |
| Active Wiretap | Kitsune | 2,278,689 | 89 / 65 | 227,834 | 10m 21s | 11m 14s |
| OS Scan | Kitsune | 1,697,851 | 65,588 / 7 | 156,664 | 7m 47s | 8m 25s |
| SSL Renegotiation | Kitsune | 2,207,571 | 779 / 79 | 220,333 | 9m 03s | 9m 54s |
| Video Injection | Kitsune | 2,472,401 | 60 / 24 | 247,222 | 11m 16s | 12m 16s |
| DDoS | CIC-IDS2017 | 2,151,810 | 86,272 / 60,941 | 181,651 | 10m 57s | 11m 52s |
| DoS Slow | CIC-IDS2017 | 4,405,006 | 75,513 / 40,436 | 407,178 | 21m 42s | 23m 24s |
| Port Scan | CIC-IDS2017 | 2,026,245 | 230,156 / 30,668 | 139,679 | 10m 29s | 11m 10s |
| Brute Force FTP | CIC-IDS2017 | — | — | — | OOM | — |
| Brute Force SSH | CIC-IDS2017 | 1,151,286 | 42,280 / 19,627 | 96,573 | 6m 11s | 6m 36s |
| Web Attacks | CIC-IDS2017 | 2,811,263 | 46,845 / 19,148 | 260,533 | 15m 33s | 16m 38s |
| Infiltration | CIC-IDS2017 | 418,560 | 15,755 / 4,942 | 35,744 | 2m 14s | 2m 23s |
| **Total** | | **33,981,767** | | **3,172,573** | | **~171m** |

**Usable flows** = flows with ≥ N=10 packets. The low usable ratio for Mirai (164/32,921), SYN DoS (33/753), OS Scan (7/65,588), and Video Injection (24/60) reflects the highly skewed flow size distributions in those captures — a small number of flows contain virtually all packets.

**Brute Force FTP** (CIC-IDS2017 Tuesday 13:15–14:25 UTC) OOM'd during nPrint parsing (6.53M packets, 7.5 GB filtered PCAP) on the local desktop. Needs to be re-run on the DPSS server with its larger RAM.

**Parsing throughput**: ~3,000–4,900 packets/s (Scapy `PcapReader`). The bottleneck is packet parsing; flow grouping and tensor building are fast by comparison (seconds to ~1 min).

## Configuration

All pipeline parameters are in `config/pipeline.yaml`: window size, stride, feature lists, aggregation functions, and standardization columns.

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `scapy`, `pyyaml`, `tqdm`.

## Project structure

```
ADEPT/
├── config/
│   └── pipeline.yaml                  # Pipeline parameters
├── data/                              # Raw PCAPs and CSVs, and output tensors
├── scripts/
│   ├── run_packet_to_tensor.py        # CLI entry-point (single PCAP)
│   ├── attach_labels.py               # Window label attachment (Kitsune + CIC)
│   └── process_all_attacks.py         # Batch processor for all attacks + benign
├── src/
│   └── feature_extraction/
│       ├── pcap_parser.py             # PCAP → nPrint bits + metadata
│       ├── flow_grouper.py            # 5-tuple grouping + IPAT + Direction
│       └── tensor_builder.py          # Windowing + standardization + dual output
└── requirements.txt
```
