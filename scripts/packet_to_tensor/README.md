# Packet-to-Tensor Pipeline

> Part of [DEPT](../../README.md) — **phase 2**. Converts raw PCAP captures into the ragged per-flow tensors every downstream model consumes.

```bash
python3 scripts/packet_to_tensor/process_all_attacks.py
```

## What it produces

Per source, under `data/tensors/<key>/`:

| File | Shape | Content |
|---|---|---|
| `flow_packets.npy` | `[P_total, 1090]` | One row per packet: 1088 nPrint header bits + IPAT (z-scored) + direction |
| `flow_offsets.npy` | `[N_flows, 2]` | `(start, length)` per flow |
| `rf_checkpoints.npy` | `[C_total, 24]` | Prefix statistics (mean/std/min/max × 6 base features) at packets 2, 10, 20, … |
| `rf_checkpoint_offsets.npy` | `[N_flows, 2]` | `(start, n_checkpoints)` per flow |
| `flow_ids.npy`, `flow_labels.csv`, `metadata.json` | — | Canonical 5-tuple keys, labels, source metadata |

Ragged storage (offsets into a flat array) means no fixed window and no padding: a 2-packet PortScan flow and a 12,000-packet SSDP flow coexist. Identity fields (IPs, ports, IP ID) are zeroed so models learn behaviour, not endpoints. A single shared `mean_std.json` z-scores IPAT across all sources, keeping benign and adversarial comparable.

Labels come from the source dataset alone: every Kitsune / CIC capture is `benign` (the upstream NIDS owns those attacks), every TANTGen PCAP is `adversarial`.

## Results

Final run: **40 sources → 8,624,958 flows, 61,516,406 packets, 12,479,802 checkpoints in 11.2 h** (160 GB / 104-core host).

`syn_ddos` (1.34 M flows/mode) and `ldap_ddos` (0.50 M) dominate the adversarial pool against ≤ 0.27 M for everything else — this is why the downstream splits are ≈ 84% adversarial and base-rate-dependent metrics must be read with that prior.

A single Scapy parse plus a vectorised build (permutation gather + cumulative prefix stats) replaced the earlier parse-twice, per-flow-loop version: one 275 k-packet source went 258 s → 134 s end-to-end, its build stage 126 s → 3 s. Welford's `StreamingStats` is retained in `tensor_builder.py` as the O(1)-memory reference a live ingress would run; it agrees with the vectorised path to float32 round-off.
