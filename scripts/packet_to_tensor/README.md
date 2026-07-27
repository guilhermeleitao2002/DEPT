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
| `flow_ids.npy` | `[N_flows]` object | Canonical 5-tuple key per flow — used downstream to pair adversarial flows with their clean counterparts (reconstructor, baseline, gradient ablation). |
| `flow_labels.csv` | `[N_flows]` rows | `flow_idx, label, label_str` — one row per flow. |
| `metadata.json` | JSON | Source metadata: `schema` version, `num_flows`, `num_packets`, `num_checkpoints`, encoding config. |

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

Final run (`process_all_attacks.py`, 40,340 s = 11.2 h on the 160 GB / 104-core build host). Identity masking on; `rf_first_checkpoint=2`, `rf_checkpoint_stride=10`. **40 sources, 8,624,958 flows, 61,516,406 packets, 12,479,802 RF checkpoints.**

### Benign sources

| Source | Dataset | Flows | Packets | Checkpoints |
|---|---|---:|---:|---:|
| `brute_force` | CIC-IDS2017 | 158,115 | 9,830,521 | 1,069,702 |
| `ddos` | CIC-IDS2017 | 86,272 | 2,151,810 | 267,923 |
| `portscan` | CIC-IDS2017 | 230,156 | 2,026,245 | 369,437 |
| `mirai` | Kitsune | 32,921 | 764,137 | 83,318 |
| `ssdp_flood` | Kitsune | 65,738 | 4,077,266 | 444,066 |
| `dns_ddos` | CIC-DDoS-2019 | 7,164 | 1,000,000 | 105,892 |
| `ldap_ddos` | CIC-DDoS-2019 | 220,641 | 1,000,000 | 276,396 |
| `ntp_ddos` | CIC-DDoS-2019 | 14,072 | 1,000,000 | 108,865 |
| `ssdp_ddos` | CIC-DDoS-2019 | 269,503 | 1,000,000 | 280,888 |
| `syn_ddos` | CIC-DDoS-2019 | 323,104 | 1,000,000 | 331,737 |

### TANTGen adversarial sources

Counts are per mode; `both` and `iat` are always identical in shape (timing edits do not change packet counts), `size` differs only where padding merged or dropped packets.

| Source | Flows (per mode) | Packets (per mode) |
|---|---:|---:|
| `brute_force_adv_*` | 6,970 | 274,882 |
| `ddos_adv_both` / `_iat` | 38,544 | 1,000,000 |
| `ddos_adv_size` | 32,284 | 740,859 |
| `dns_ddos_adv_*` | 6,796 | 1,000,000 |
| `ldap_ddos_adv_*` | 498,065 | 1,000,000 |
| `mirai_adv_*` | 13,046 | 642,516 |
| `ntp_ddos_adv_*` | 14,020 | 1,000,000 |
| `portscan_adv_*` | 158,671 | 321,444 |
| `ssdp_ddos_adv_*` | 269,641 | 1,000,000 |
| `ssdp_flood_adv_*` | 65,538 | 1,439,604 |
| `syn_ddos_adv_*` | 1,336,553 | 4,963,410 |

`syn_ddos` and `ldap_ddos` dominate the adversarial flow pool (1.34 M and 0.50 M flows per mode against ≤ 0.27 M for everything else) — this is why the downstream evaluation splits are ≈ 84% adversarial and why base-rate-dependent metrics must be read with that prior.

The single-parse + vectorised build (July 2026) removes one full Scapy pass and the per-flow build loops; on the 160 GB server one 275k-packet source went 258 s → 134 s end-to-end (build stage 126 s → 3 s). The earlier parse-twice pipeline took ~100 min for the five original sources on a 32 GB desktop.

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
