# Phase 1 — Packet-to-Tensor Pipeline

> Part of [ADEPT](../../README.md). This stage converts raw PCAP captures (or labelled datasets like Kitsune / CIC-IDS2017) into the two synchronized tensor inputs consumed by every downstream ADEPT model.

## What this stage produces

Per processed PCAP, the pipeline writes a self-contained tensor directory under `data/tensors/<key>/`:

| File | Shape / format | Consumer |
|---|---|---|
| `sequence.npy` | `(W, N, 1090)` float32 | TCN, Purification Transformer |
| `rf_stats.npy` | `(W, 24)` float32 | Random Forest |
| `flow_ids.npy` | `(W,)` object — 5-tuple per window | Evaluation, RL Agent, debugging |
| `mean_std.json` | JSON — IPAT z-score stats | Inference / test-set normalization |
| `metadata.json` | JSON — full pipeline provenance (input, encoding cfg, masking, shapes, flow counts) | Reproducibility |
| `window_labels.csv` | CSV — `window_idx, label, label_str` | All downstream models |
| `packets.csv` | CSV — intermediate per-packet DataFrame (only when `run_packet_to_tensor.py` is called directly) | Inspection / debugging |

`W` (number of windows), `N` (window size — default 10), and `F=1090` are documented below.

## Tensor specification

The pipeline emits two synchronized views of every N-packet window:

| Dim | Meaning | Value |
|---|---|---|
| `W` | Number of windows (samples) — varies with capture | — |
| `N` | Window size — packets per window | 10 (default; configurable) |
| `F` | Features per packet | 1090 (1088 nPrint bits + IPAT + Direction) |
| `S` | RF statistic vector length | 24 (6 base features × 4 aggregations) |

Both tensors are aligned at the **N-Packet Snapshot** boundary: row `w` of `sequence.npy[w]` and `rf_stats.npy[w]` describe exactly the same N packets from the same flow.

### Sequence features (F=1090)

| # | Feature | Description |
|---|---|---|
| 0–1087 | nPrint bits | Tri-state `{-1, 0, 1}` encoding of all IPv4/TCP/UDP/ICMP header bits |
| 1088 | `IPAT` | Inter-Packet Arrival Time within the flow (z-score normalized) |
| 1089 | `Direction` | `1` = forward (flow initiator), `0` = reverse |

### RF statistics (S=24)

For each of the 6 base features (`IPAT`, `Packet_Size`, `Payload_Length`, `Payload_Entropy`, `TCP_Flags`, `Direction`), four aggregations are computed over the N-packet window: **mean, std, min, max**.

## Packet encoding — nPrint

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

High-entropy identity fields (source/destination IP, IP ID, source/destination port) are **zeroed out** before training — this prevents the model from overfitting to specific addresses and forces it to learn behavioral patterns instead. Masking is controlled by `mask_prefixes` in `config/pipeline.yaml` and can be disabled with `--no-mask`.

## Processing stages (batch mode)

```
PCAP file
  │
  ├─ [1] pcap_parser     Encode each packet as a 1088-bit nPrint vector (Scapy)
  │       → 1088 header bits (IPv4 + TCP/UDP/ICMP)
  │       → Identity masking: zero out IPs, ports, IP ID
  │       → Metadata: Timestamp, Packet_Size, Protocol, TCP_Flags,
  │                   Payload_Length, Payload_Entropy
  │
  ├─ [2] flow_grouper    Group packets by canonical 5-tuple
  │       → Per-flow IPAT (Inter-Packet Arrival Time)
  │       → Direction (forward / reverse)
  │
  └─ [3] tensor_builder  Window into N-packet snapshots
          → Sequence tensor [W, N, 1090]   (1088 nPrint bits + IPAT + Direction)
          → RF stats tensor [W, 24]        (mean/std/min/max of 6 metadata features)
```

A second, lower-RAM **online (incremental)** mode is also available — see [Online (incremental) mode](#online-incremental-mode) below.

## Key design decisions

| Decision | Rationale |
|---|---|
| **Per-flow IPAT** (not global IAT) | Each flow has its own timing; global IAT mixes unrelated flows |
| **Canonical 5-tuple** for flow ID | Order-independent — forward and reverse packets belong to the same flow |
| **Drop incomplete windows** (no padding) | Padding with zeros would pollute RF aggregated statistics |
| **Separate RF and sequence outputs** | RF uses statistical summaries; TCN/Transformer uses raw sequences — different input representations enforce ensemble diversity |
| **Tri-state `-1` for absent layers** | Distinguishes "bit unset" from "layer not present"; preserves header-presence signal even with masking on |

## Module reference

### `src/feature_extraction/pcap_parser.py` — PCAP → nPrint bits + metadata

Reads a PCAP with Scapy and encodes each packet as a 1088-bit nPrint vector plus a lightweight metadata row. The bit array is kept in `int8` numpy form rather than a pandas DataFrame to avoid the ~10× memory overhead of holding 1088 columns in pandas.

| Function | Inputs | Outputs | Notes |
|---|---|---|---|
| `extract_packet_features(pcap_path, mask_prefixes)` | PCAP path; list of field prefixes to zero | `(nprint_bits, meta_df)` — `int8 (P, 1088)` + `DataFrame (P, 10)` | Main entry point. Chunks 50k packets at a time to keep peak RAM low. |
| `_encode_packet(pkt)` | Scapy packet | `(list[int 1088], dict)` | Encodes one packet; detects layers, extracts raw bytes, calls `_bytes_to_bits()`. |
| `_bytes_to_bits(raw_bytes, max_bytes, fill=-1)` | Raw bytes or None; target size; fill value | `list[int]` of `max_bytes × 8` values `{-1, 0, 1}` | MSB-first expansion; missing layer → fill with `-1`; unused option bytes → `0`. |
| `_shannon_entropy(data)` | `bytes` | `float` ∈ [0, 8] | H = −Σ(p·log₂p). Used for `Payload_Entropy` metadata column. |
| `apply_masking(df, mask_prefixes)` | nPrint DataFrame; prefix list | Masked DataFrame | DataFrame-level API (deprecated — pipeline now uses in-place numpy masking). |

**`meta_df` columns:** `Timestamp, Source_IP, Destination_IP, Source_Port, Destination_Port, Protocol, Packet_Size, Payload_Length, Payload_Entropy, TCP_Flags`.

**Identity masking:** columns matching a prefix (e.g., `ipv4_src_0` … `ipv4_src_31`) are zeroed in-place on the numpy array. Controlled by `mask_prefixes` in `config/pipeline.yaml`.

### `src/feature_extraction/flow_grouper.py` — 5-tuple grouping + IPAT + Direction

Groups packets into bidirectional flows and annotates each packet with per-flow IPAT and Direction. IPAT is per-flow (not global) because each flow window is processed independently by the ensemble.

| Function | Inputs | Outputs | Notes |
|---|---|---|---|
| `group_by_flow(meta_df)` | `meta_df` (P, 10) from `pcap_parser` | `(meta_df_updated, flow_indices, sort_order)` | Main entry point. |
| `_compute_canonical_keys(df)` | `meta_df` | `pd.Series` of 5-tuples | — |

**Canonical 5-tuple:** `(min_ip, max_ip, min_port, max_port, protocol)` — order-independent, so forward and reverse packets map to the same flow.

**`group_by_flow` return values:**
- `meta_df` — updated with `IPAT (float64, seconds)` and `Direction (int8: 1=fwd, 0=rev)`, sorted by Timestamp (stable mergesort).
- `flow_indices` — `OrderedDict[5-tuple → np.ndarray[int64]]`: each flow's sorted row indices.
- `sort_order` — `np.ndarray[int64]`: the permutation applied; **caller must apply this to `nprint_bits`** (`nprint_bits = nprint_bits[sort_order]`) to keep the arrays in sync.

### `src/feature_extraction/tensor_builder.py` — windowing → `sequence.npy` + `rf_stats.npy`

Slides a non-overlapping N-packet window over each flow and writes the two output tensors directly to memory-mapped `.npy` files, avoiding OOM on large datasets and allowing the pipeline to run locally.

| Function | Inputs | Outputs | Notes |
|---|---|---|---|
| `build_tensors(nprint_bits, meta_df, flow_indices, output_dir, ...)` | `int8 (P, 1088)`, `DataFrame`, `OrderedDict`, config params | `dict` with paths + metadata | Main entry point. |
| `_load_or_compute_stats(df, columns, mean_std_path)` | DataFrame; column list; JSON path | `{"IPAT": {"mean": float, "std": float}}` | Loads from JSON if it exists, otherwise computes and saves. |
| `_compute_rf_stats(window_meta, feature_names, base_features, agg_functions)` | `float32 (N, M)` window; name lists | `float32 (S,)` | Applies mean/std/min/max per base feature over the N-packet window. |
| `_count_windows(flow_indices, window_size, stride)` | flow map; N; stride | `int` | Pre-counts total windows to pre-allocate memmaps. Formula: Σ `(n − N) // stride + 1` for each flow with `n ≥ N`. |

**`build_tensors` return dict keys:** `sequence_path`, `rf_stats_path`, `flow_ids`, `stats`, `feature_names` (1090 names), `rf_stat_names` (24 names), `num_windows`.

**Memory strategy:** pre-count → allocate `np.lib.format.open_memmap` → write one window at a time (~43 KB working set at N=10).

### `src/feature_extraction/incremental_pipeline.py` — online / streaming mode

Two-pass pipeline that eliminates the global timestamp sort required by the batch pipeline. Used when `run_packet_to_tensor.py` is invoked with `--online`.

- **Pass 1 (prescan):** reads only the 5-tuple and timestamp; counts packets per canonical flow (so memmaps can be pre-allocated) and computes IPAT mean/std via Welford's online algorithm.
- **Pass 2 (encoding):** full nPrint encoding, per-flow state update (`last_ts`, packet buffer), windows emitted directly to memmap the moment a flow's buffer reaches N packets.

**Memory profile:** O(window_size × active_flows × features) instead of O(total_packets × features). For large captures (e.g. Brute Force FTP, 6.5M packets) this is the difference between a few hundred MB and several GB.

The canonical 5-tuple key in `_canonical_key()` matches `flow_grouper._compute_canonical_keys()` byte-for-byte, so flow IDs are interchangeable between batch and online modes.

## CLI — `scripts/packet_to_tensor/run_packet_to_tensor.py`

Single-PCAP driver. Orchestrates the three modules (or the incremental pipeline) end-to-end.

```
[1/3] extract_packet_features()  →  nprint_bits (P,1088) int8  +  meta_df (P,10)
[2/3] group_by_flow()            →  meta_df+IPAT+Direction, flow_indices, sort_order
      nprint_bits = nprint_bits[sort_order]          ← critical sync step
[3/3] build_tensors()            →  sequence.npy, rf_stats.npy, flow_ids.npy
      write metadata.json
```

### Flags

| Flag | Default | Purpose |
|---|---|---|
| `-i / --input` | required | Input PCAP file. With `--fit-mean-std`, accepts multiple paths. |
| `-o / --output-dir` | required | Output directory for tensors and metadata. |
| `-c / --config` | `config/pipeline.yaml` | Pipeline config YAML. |
| `-N / --window-size` | from config | Override the N-Packet window size. |
| `--mean-std` | — | Path to existing `mean_std.json` (use training stats on a test set). |
| `--no-mask` | off | Disable identity masking (keep IPs, ports, IP ID in nPrint bits). |
| `--workers` | all CPU cores | Parallel encoding workers. |
| `--online` | off | Use the incremental two-pass pipeline (lower peak RAM, no global sort). |
| `--fit-mean-std` | off | Fit-only mode: stream multiple PCAPs and write a shared `mean_std.json`; emits no tensors. |

### Examples

```bash
# 1. Plain batch run on a Kitsune PCAP
python3 scripts/packet_to_tensor/run_packet_to_tensor.py \
    -i data/Kitsune/pcap/Mirai_pcap.pcap \
    -o data/tensors/mirai/

# 2. Custom window size (N=20)
python3 scripts/packet_to_tensor/run_packet_to_tensor.py \
    -i data/Kitsune/pcap/Mirai_pcap.pcap \
    -o data/tensors/mirai_n20/ \
    -N 20

# 3. Apply training statistics to a test set
python3 scripts/packet_to_tensor/run_packet_to_tensor.py \
    -i data/Kitsune/pcap/Mirai_pcap.pcap \
    -o data/tensors/mirai/ \
    --mean-std data/tensors/mean_std.json

# 4. Online (incremental) mode for memory-constrained machines
python3 scripts/packet_to_tensor/run_packet_to_tensor.py \
    -i data/CIC-IDS-2017/pcap/Tuesday-WorkingHours.pcap \
    -o data/tensors/brute_force_ftp/ \
    --online

# 5. Fit a shared mean_std.json over multiple PCAPs (no tensors emitted)
python3 scripts/packet_to_tensor/run_packet_to_tensor.py \
    --fit-mean-std \
    -i data/tensors/.filtered/ddos_filtered.pcap \
       data/tensors/.filtered/portscan_filtered.pcap \
       data/Kitsune/pcap/Mirai_pcap.pcap \
    -o data/tensors/mean_std.json
```

### Welford / Chan parallel update for shared standardizer

`run_packet_to_tensor.py --fit-mean-std` streams one PCAP at a time, encodes packets into the metadata pipeline, and folds each batch into a global `(count, mean, M2)` accumulator using **Chan's parallel-Welford** formula. Per-PCAP arrays are discarded between batches, so peak RAM stays bounded by a single PCAP's metadata. The result is a `mean_std.json` reflecting **every input PCAP's IPAT distribution combined**, which is essential when training has to span benign and adversarial captures with very different timing profiles. (See the `--no-shared-mean-std` / `--refit-shared-mean-std` discussion in [`process_all_attacks.py`](#cli--scriptspacket_to_tensorprocess_all_attackspy) below.)

## CLI — `scripts/packet_to_tensor/attach_labels.py`

Produces `<tensor-dir>/window_labels.csv` under ADEPT's dataset-driven label convention. Because ADEPT defends against TANTGen-style perturbations (not the raw-dataset attacks the upstream NIDS already catches), the binary label is determined entirely by the source dataset:

| Dataset | `label` | `label_str` |
|---|---|---|
| `kitsune` | `0` | `benign` |
| `cic` | `0` | `benign` |
| `adversarial` | `1` | `adversarial` |

No ground-truth CSV or PCAP re-read is needed; `num_windows` is read directly from the `metadata.json` written by the pipeline.

### Output columns

| Column | Description |
|---|---|
| `window_idx` | Row index into `sequence.npy` / `rf_stats.npy` |
| `label` | `0` = benign (Kitsune / CIC), `1` = adversarial (TANTGen) |
| `label_str` | `"benign"` or `"adversarial"` |

### Flags

| Flag | Description |
|---|---|
| `-d / --dataset {kitsune,cic,adversarial}` | Which label convention to apply. |
| `-t / --tensor-dir` | Tensor directory (must already contain `metadata.json`). |
| `--force` | Overwrite an existing `window_labels.csv`. |

`process_all_attacks.py` calls the same logic in-process after each successful pipeline run, and retroactively for any tensor directory where `window_labels.csv` is missing.

## CLI — `scripts/packet_to_tensor/process_all_attacks.py`

Batch driver that runs `run_packet_to_tensor.py` across every PCAP in the thesis attack registry. For CIC-IDS2017 attacks it uses `editcap` to slice out the relevant attack time window before processing.

### Attack registry (pass keys to `--only`)

| Key | Name | Dataset | Time window (UTC) |
|---|---|---|---|
| `mirai` | Mirai Botnet | Kitsune | — |
| `syn_dos` | SYN DoS | Kitsune | — |
| `ssdp_flood` | SSDP Flood | Kitsune | — |
| `arp_mitm` | ARP MitM | Kitsune | — |
| `active_wiretap` | Active Wiretap | Kitsune | — |
| `os_scan` | OS Scan | Kitsune | — |
| `ssl_renegotiation` | SSL Renegotiation | Kitsune | — |
| `video_injection` | Video Injection | Kitsune | — |
| `ddos` | DDoS | CIC-IDS2017 | 2017-07-07 19:30–21:05 |
| `dos_slow` | DoS Slow (Slowloris + SlowHTTPTest + Hulk + GoldenEye) | CIC-IDS2017 | 2017-07-05 13:40–15:30 |
| `portscan` | Port Scan | CIC-IDS2017 | 2017-07-07 17:00–19:30 |
| `brute_force_ftp` | Brute Force FTP | CIC-IDS2017 | 2017-07-04 13:15–14:25 |
| `brute_force_ssh` | Brute Force SSH | CIC-IDS2017 | 2017-07-04 18:00–19:05 |
| `brute_force` | Brute Force FTP+SSH combined (paired with `BruteForce_adversarial_*.pcap`) | CIC-IDS2017 | 2017-07-04 13:15–19:05 |
| `web_attacks` | Web Attacks (Brute Force HTTP + XSS + SQLi) | CIC-IDS2017 | 2017-07-06 13:15–14:45 |
| `infiltration` | Infiltration | CIC-IDS2017 | 2017-07-06 18:15–19:50 |
| `monday_benign` | Monday Benign baseline | CIC-IDS2017 | 2017-07-03 09:00–11:00 |

The Fuzzing Kitsune capture was deliberately excluded — see the [TANTGen integration README](../adversarial_generation/README.md#why-fuzzing-was-excluded) for details.

### Flags

| Flag | Purpose |
|---|---|
| `--only KEY [KEY...]` | Process only these attack keys. |
| `--exclude KEY [KEY...]` | Skip these attack keys. |
| `--tantgen` | Restrict to the 5-attack subset for which TANTGen adversarial PCAPs exist on disk (`mirai`, `ssdp_flood`, `ddos`, `portscan`, `brute_force`) **and** also tensorise those matching adversarial PCAPs from `data/adversarial/<attack>/<attack>_adversarial_<mode>.pcap` (output: `data/tensors/adversarial/<attack>_adv_<mode>/`, labels written as `label=1, label_str="adversarial"`). Mutually exclusive with `--only` / `--exclude`. |
| `--force` | Re-process attacks even if `metadata.json` already exists. |
| `--dry-run` | Show what would be processed without running. |
| `-N / --window-size` | Override the N-Packet window size from `config/pipeline.yaml`. |
| `--include-benign` | Include `monday_benign` in the default selection. |
| `--keep-filtered` | Keep the time-filtered CIC PCAPs in `data/tensors/.filtered/` after processing (cached re-runs reuse them). |
| `--no-shared-mean-std` | Disable the shared `mean_std.json` step; let each per-PCAP run refit its own standardizer. |
| `--refit-shared-mean-std` | Force re-fitting the shared `mean_std.json` even if it already exists. |

### Per-attack flow

1. Skip if `metadata.json` already exists (unless `--force`).
2. If CIC-IDS2017: check `data/tensors/.filtered/` for a cached `editcap`-filtered PCAP; run `editcap` only if not cached.
3. (Once, before per-attack processing) fit the shared `mean_std.json` over benign + adversarial PCAPs (unless disabled).
4. Run `run_packet_to_tensor.py` as a subprocess, passing `--mean-std` so every per-attack run uses the same z-score reference.
5. Attach labels with `attach_labels.py`'s in-process logic.
6. Delete the filtered PCAP (unless `--keep-filtered`).

### Why a shared `mean_std.json`?

Without a shared fit, every per-attack run computes its own IPAT mean/std. The benign-only fit is dominated by Kitsune's tight inter-packet timing (mean ≈ 0.04 s, std ≈ 3.24 s on SYN_DoS), which lands adversarial windows far in the tails when TANTGen widens IATs (mean ≈ 9.85 s, std ≈ 31.08 s on the joint fit). Sharing a single standardizer over **all PCAPs ADEPT will see during training** makes the z-scores comparable across the benign / adversarial boundary.

### Examples

```bash
# Default — process the 15 raw-dataset attacks (Monday benign excluded)
python3 scripts/packet_to_tensor/process_all_attacks.py

# Preview without processing
python3 scripts/packet_to_tensor/process_all_attacks.py --dry-run

# Single attack
python3 scripts/packet_to_tensor/process_all_attacks.py --only syn_dos

# Force re-run of one attack
python3 scripts/packet_to_tensor/process_all_attacks.py --only mirai --force

# Override window size from CLI
python3 scripts/packet_to_tensor/process_all_attacks.py -N 20

# Include the Monday benign baseline
python3 scripts/packet_to_tensor/process_all_attacks.py --include-benign

# Keep filtered CIC PCAPs (re-runs reuse them automatically)
python3 scripts/packet_to_tensor/process_all_attacks.py --keep-filtered

# Restrict to the 5 attacks whose adversarial PCAPs exist on disk AND
# tensorise their adversarial variants
# (5 raw attacks + 15 adversarial PCAPs = 20 tensor dirs in one run)
python3 scripts/packet_to_tensor/process_all_attacks.py --tantgen
```

## Configuration — `config/pipeline.yaml`

| Key | Purpose |
|---|---|
| `window_size` | N — packets per window (default 10). |
| `window_stride` | Stride between windows (default = window_size, i.e. non-overlapping). |
| `mask_prefixes` | List of nPrint column-name prefixes zeroed before training (`ipv4_src`, `ipv4_dst`, `ipv4_id`, `tcp_sprt`, `tcp_dprt`, `udp_sport`, `udp_dport`). |
| `rf_base_features` | The 6 metadata columns aggregated into RF stats (`IPAT`, `Packet_Size`, `Payload_Length`, `Payload_Entropy`, `TCP_Flags`, `Direction`). |
| `rf_agg_functions` | The 4 aggregations applied per base feature (`mean`, `std`, `min`, `max`). |
| `standardize_columns` | Which metadata columns are z-score normalized (default: `IPAT` only — nPrint bits are bounded `{-1, 0, 1}` and Direction is binary). |
| `mean_std_filename` | Output filename for the per-tensor-dir copy of the standardizer (default `mean_std.json`). |
| `default_output_dir` | Fallback when no `-o` is given. |

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

**Brute Force FTP** (CIC-IDS2017 Tuesday 13:15–14:25 UTC) OOM'd during nPrint parsing in batch mode (6.53M packets, 7.5 GB filtered PCAP) on the local desktop. The `--online` flag eliminates this OOM by switching to the streaming pipeline; it is also the recommended mode for any capture that exceeds local RAM.

**Parsing throughput:** ~3,000–4,900 packets/s (Scapy `PcapReader`). The bottleneck is packet parsing; flow grouping and tensor building are fast by comparison (seconds to ~1 min).

## Production performance analysis

Derived from the 15 successful runs above (33.98 M packets, 3.17 M windows total).

### Per-stage throughput

| Stage | Throughput | Per-packet latency | Wall-time share |
|---|---:|---:|---:|
| [1/3] nPrint parsing (Scapy) | **3,655 pkt/s** | **274 µs** | **92.2 %** |
| [2/3] Flow grouping | 188,893 pkt/s | 5.3 µs | 1.8 % |
| [3/3] Tensor building | 52,223 pkt-equiv/s (5,222 windows/s) | 19 µs (191 µs/window) | 6.0 % |

Stage 1 (Scapy nPrint encoding) is the **sole bottleneck**: it accounts for >92% of all wall time and is 50× slower than flow grouping and 14× slower than tensor building. Stages 2 and 3 together add less than 25 µs per packet.

### End-to-end tensor latency ("online mode", N=10)

| Component | Latency |
|---|---:|
| Parse 10 packets (Scapy) | 2.74 ms |
| Flow grouping + IPAT (10 pkts) | 0.05 ms |
| Build 1 tensor window | 0.19 ms |
| **Total** | **~3 ms** |

### Production gap

| Link rate | Required pkt/s | Gap vs. current Scapy pipeline |
|---|---:|---:|
| 100 Mbps (small pkts) | ~100 k | ~27× |
| 1 Gbps (small pkts) | ~1 M | ~274× |
| 10 Gbps | ~10 M | ~2,740× |

The current pipeline runs comfortably for **offline post-hoc analysis** and **research datasets**. For real-time deployment the Scapy parser must/should be replaced:

- **Drop Scapy:** use DPDK, AF_XDP, or a C/Rust extension for kernel-bypass header parsing; target >500 k pkt/s per core.
- **Parallelise per-flow:** flows are independent — shard the nPrint encoding across CPU cores (each core owns a subset of flow keys).
- **Incremental IPAT:** per-flow state is already local (see `incremental_pipeline.py`); an online pipeline can emit a window the moment the N-th packet arrives without any global sort pass.

Stages 2 and 3 are already fast enough for 1 Gbps line rates; only Stage 1 requires re-implementation.
