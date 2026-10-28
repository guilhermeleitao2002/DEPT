# Packet-to-Tensor Pipeline

> Part of [DEPT](../../README.md), **phase 2**. Converts the PCAP captures into the ragged per-flow tensors used by all models.

```bash
python3 scripts/packet_to_tensor/process_all_attacks.py
```

## Output

Per source, under `data/tensors/<key>/`:

| File | Shape | Content |
|---|---|---|
| `flow_packets.npy` | `[P_total, 1090]` | One row per packet: 1088 nPrint header bits + IPAT (z-scored) + direction |
| `flow_offsets.npy` | `[N_flows, 2]` | `(start, length)` per flow |
| `rf_checkpoints.npy` | `[C_total, 24]` | Prefix statistics (mean/std/min/max × 6 base features) at packets 2, 10, 20, … |
| `rf_checkpoint_offsets.npy` | `[N_flows, 2]` | `(start, n_checkpoints)` per flow |
| `flow_ids.npy`, `flow_labels.csv`, `metadata.json` | — | Canonical 5-tuple keys, labels, source metadata |

Flows are stored as offsets into one flat array, so there is no fixed window and no padding. Identity fields (IPs, ports, IP ID) are set to zero, so the models cannot rely on endpoints. One shared `mean_std.json` z-scores IPAT for all sources.

Labels depend only on the source: every Kitsune / CIC capture is `benign` and every TANTGen PCAP is `adversarial`.

## Results

Final run: **40 sources → 8,624,958 flows, 61,516,406 packets, 12,479,802 checkpoints in 11.2 h** (160 GB RAM, 104 cores).

`syn_ddos` (1.34 M flows per mode) and `ldap_ddos` (0.50 M) are much larger than the other sources (≤ 0.27 M each), which is why the splits are about 84% adversarial.

The build parses each PCAP once with Scapy and computes the prefix statistics with vectorised cumulative sums. `StreamingStats` in `tensor_builder.py` is the per-flow (Welford) version a live deployment would use; both give the same values up to float32 round-off.
