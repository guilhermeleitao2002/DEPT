# TCN Base Learner

> Part of [DEPT](../../README.md) — **phase 4**. The **dense, early-warning** half of the streaming ensemble.

```bash
python3 scripts/tcn/run_tcn.py
```

## What it does

A causal Temporal Convolutional Network over `flow_packets.npy` — variable-length sequences of 1090-dim per-packet vectors (1088 nPrint bits + IPAT + direction) — emitting `P(adversarial)` after **every** packet, starting at packet 1. It trades precision for earliness: noisier than the RF, but a verdict exists before the RF's first checkpoint.

Three stacked dilated `TemporalBlock`s (channels 128, k=3, dilations 1/2/4, weight norm, dropout 0.15, 1×1 residual) plus a per-timestep 1×1 head → `[B, T, 2]` logits. **805,762 parameters, receptive field 29 packets.** `Chomp1d` trims the right side after each dilated conv, so output `t` depends only on inputs ≤ `t` — shifting the input by a packet shifts the output by a packet, with no leakage.

Trained with per-packet cross-entropy (class-balanced, averaged within each flow so long flows don't dominate the gradient), AdamW + cosine schedule, mixed precision, flows truncated at 200 packets per batch. Split is flow-atomic and source-stratified with the same `seed=42` as the RF, so both are evaluated on exactly the same held-out flows. Best epoch 6 of 10 (early-stopped, patience 3), 24.7 h on one GPU over 6.04 M training flows.

## Results (test: 1.29 M flows, ten sources)

| Last timestep | | Streaming | |
|---|---|---|---|
| Macro F1 | 0.6608 | AUROC@1 | **0.8191** |
| ROC AUC | 0.8386 | AUROC@5 / @max | 0.8375 / 0.8386 |
| PR AUC | 0.9592 | EDP median | **1 packet** |
| Precision (adv) | 0.9220 | Detection rate (p>0.5) | 78.40% |
| Recall (adv) / (benign) | 0.7820 / 0.6606 | | |

**Packet-1 earliness is the TCN's contribution** — 78% of adversarial flows are flagged from the first packet, and AUROC is essentially flat after packet 5, so most of its discriminative signal arrives immediately. Everything else is weaker than the RF:

- **The four reflection floods are its blind spot** (same-base streaming AUROC ≈ 0.50 on `dns/ldap/ntp/ssdp_ddos` in *every* mode): one- or two-packet spoofed UDP flows give a per-packet sequence nothing to grip.
- **Timing-only edits are the hardest mode on TCP sources** — `portscan_iat` 0.327 (below chance), `syn_ddos_iat` 0.602, `ddos_iat` 0.544, against ≈ 1.0 for `both`/`size` on the same attacks. With few packets there is essentially one inter-packet interval, indistinguishable from natural variability.
- **`brute_force` and `mirai` fail at verdict level** (balanced ROC 0.40–0.52) despite usable streaming AUROC — the threshold sits at the wrong operating point, another reason the ensemble defers to the RF there.

The RF's aggregate prefix statistics beat the raw per-packet sequence on exactly the floods and timing edits the TCN misses — the complementarity the meta-learner is built to exploit.

Artifacts under `data/models/tcn/`: `tcn.pt`, `tcn_config.json`, `dataset_index.npz`, `split.npz` (element-wise compatible with the RF's), `metrics.json`, `streaming_metrics.json`, `training_curves.json`, plus ROC / PR / confusion / earliness / training-curve plots.
