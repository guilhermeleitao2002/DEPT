# Packet-to-Tensor Pipeline

> Part of [DEPT](../../README.md). Converts raw PCAP captures into the ragged per-flow tensor format consumed by all downstream models.

## Output schema

For each processed source, the pipeline writes a directory under `data/tensors/<key>/`:

| File | Shape | Description |
|---|---|---|
| `flow_packets.npy` | `[P_total, 1090]` float32 (memmap) | All packets of all flows in arrival order. Each row is one packet. |
| `flow_offsets.npy` | `[N_flows, 2]` int64 | `(start_idx, length)` into `flow_packets.npy` per flow. |
| `rf_checkpoints.npy` | `[C_total, 24]` float32 | Multi-scale prefix statistics: one row per prefix milestone per flow. Default cadence (`rf_first_checkpoint=2`, `rf_checkpoint_stride=10`) emits at packets 2, 10, 20, 30, … — the early packet-2 row gives short flows (e.g. PortScan) an RF verdict. |
| `rf_checkpoint_offsets.npy` | `[N_flows, 2]` int64 | `(start_idx, n_checkpoints)` into `rf_checkpoints.npy` per flow. |
| `flow_labels.csv` | `[N_flows]` rows | `flow_idx, label, label_str` — one row per flow. |
| `metadata.json` | JSON | Source metadata: schema version, `n_flows`, `total_packets`, `num_checkpoints`, encoding config. |

IPAT z-scoring uses one shared `data/tensors/mean_std.json` standardizer for every source (see below).

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

At every prefix milestone (packets 2, 10, 20, 30, … under the default cadence), four aggregations (mean, std, min, max) are computed for each of six base features: `IPAT`, `Packet_Size`, `Payload_Length`, `Payload_Entropy`, `TCP_Flags`, `Direction`. The batch build computes them with vectorised per-flow cumulative sums / running extrema; `StreamingStats` (Welford, O(1) memory per flow) remains in `tensor_builder.py` as the reference form a deployed streaming ingress would run per live flow — the two agree to float32 round-off.

## Label convention

Labels are derived entirely from the source dataset — no ground-truth CSV is needed.

| Source | `label` | `label_str` |
|---|---|---|
| Kitsune (any attack capture) | `0` | `benign` |
| CIC-IDS2017 (any attack capture) | `0` | `benign` |
| CIC-DDoS-2019 (any reflection-flood capture) | `0` | `benign` |
| TANTGen adversarial PCAP | `1` | `adversarial` |

Under DEPT's defense framing, the upstream NIDS is responsible for the raw-dataset attacks. DEPT's job is specifically to catch TANTGen-style perturbations that evade the upstream model.

## Dataset statistics

Results from `process_all_attacks.py` on WSL2, **before the CIC-DDoS-2019 expansion**. Identity masking on; `rf_checkpoint_stride=10`.

| Source | Flows | Packets |
|---|---:|---:|
| Mirai Botnet (Kitsune) | 32,921 | 764,137 |
| SSDP Flood (Kitsune) | 65,738 | 4,077,266 |
| DDoS (CIC-IDS2017) | 86,272 | 2,151,810 |
| Port Scan (CIC-IDS2017) | 230,156 | 2,026,245 |
| Brute Force FTP+SSH (CIC-IDS2017) | 158,115 | 9,830,521 |
| Brute Force adversarial (×3 modes) | 80 each | 2,500 each |
| DDoS adversarial (×3 modes) | 1,011 each | 9,741 each |
| Mirai adversarial (×3 modes) | 3,830 each | 192,755 each |
| PortScan adversarial (×3 modes) | 15,839 each | 31,885 each |
| SSDP Flood adversarial (×3 modes) | 64,722 each | 287,921 each |

Total processing time: ~100 min on a desktop Linux machine (WSL2, 32 GB RAM, parse-twice pipeline). The single-parse + vectorised build (July 2026) removes one full Scapy pass and the per-flow build loops; on the 160 GB server one 275k-packet source went 258 s → 134 s end-to-end (build stage 126 s → 3 s).

The CIC-DDoS-2019 expansion adds five benign + adversarial (×3 modes) pairs — `ntp_ddos`, `dns_ddos`, `ssdp_ddos`, `syn_ddos`, `ldap_ddos`. Each benign source is the first 1,000,000 packets of the raw capture (`data/CIC-DDoS-2019/PCAPs/{NTP,DNS,SSDP,Syn,LDAP}.pcap`), packet-capped because these floods have no clean attack time window. Counts for these sources are produced by the next `process_all_attacks.py` run.

## Design decisions

| Decision | Rationale |
|---|---|
| Ragged per-flow storage (offsets + flat array) | Avoids a T_max cap; no padding waste. Works for both 2-packet PortScan flows and 10k-packet SSDP flows. |
| Vectorised batch build (permutation gather + cumulative prefix stats) | Flows are contiguous in the output, so `flow_packets` is one numpy gather and the RF checkpoints are per-flow cumsums — replaces the per-flow/per-packet Python loops the 32 GB build host used to require (the Mirai-scale build stage drops from minutes to seconds). Assumes the capture's int8 bits fit in RAM (~1.1 GB per million packets). |
| Single Scapy parse per PCAP (`parsed_cache`) | The standardizer fit and the tensor build used to each parse every PCAP; Scapy parsing dominates the pipeline, so caching the parse between the two passes roughly halves total wall time. Costs ~1.1 GB RAM per million cached packets — sized for the 160 GB build host; calling `fit_mean_std`/`run` without a cache restores the low-RAM parse-twice behaviour. |
| Welford's incremental stats retained as reference (`StreamingStats`) | The vectorised checkpoints reproduce it to float32 round-off; the class stays as the O(1)-memory form a deployed streaming ingress runs per live flow. |
| Shared `mean_std.json` across all sources | Without a shared fit, IPAT z-scores are incomparable across the benign/adversarial boundary. The shared fit covers all PCAPs DEPT will see during training. |
| Per-flow IPAT (not global IAT) | Each flow has its own timing context; global IAT mixes unrelated flows. |
| Canonical 5-tuple flow ID | Order-independent — forward and reverse packets belong to the same flow. |
| Identity masking (IPs, ports, IP ID zeroed) | Prevents models from memorising source/destination addresses instead of behavioral patterns. |
| CIC-IDS2017 time-filtered, CIC-DDoS-2019 packet-capped | CIC-IDS2017 attacks sit in a known window of a workday capture, so `editcap -A/-B` isolates them. The CIC-DDoS-2019 floods are attack-dominated from the first seconds and run to tens of GB with no clean boundary, so the first 1,000,000 packets (`tcpdump -c N`, fast even on a 37 GB file) are sliced instead — the same scale the generator caps its malicious input to, keeping the benign/adversarial pools balanced. |
