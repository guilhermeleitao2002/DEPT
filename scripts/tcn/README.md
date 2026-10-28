# TCN Base Learner

> Part of [DEPT](../../README.md), **phase 4**. The dense, early stream of the ensemble.

```bash
python3 scripts/tcn/run_tcn.py
```

## Method

A causal Temporal Convolutional Network over `flow_packets.npy` (variable-length sequences of 1090-dim packet vectors: 1088 nPrint bits + IPAT + direction). It outputs `P(adversarial)` after every packet, starting at packet 1, so it is less precise than the RF but already has a verdict before the RF's first checkpoint.

Three dilated `TemporalBlock`s (128 channels, k=3, dilations 1/2/4, weight norm, dropout 0.15, 1×1 residual) and a 1×1 head per time step give `[B, T, 2]` logits. **805,762 parameters, receptive field 29 packets.** `Chomp1d` removes the right padding of each convolution, so the output at packet `t` only depends on packets ≤ `t`.

Training uses per-packet cross-entropy (class-balanced, averaged within each flow), AdamW with a cosine schedule, mixed precision, and flows truncated at 200 packets. The split is the same as the RF's (flow-level, `seed=42`). Training stopped early after 10 of 20 epochs (best at the 7th), 24.7 h on one GPU over 6.04 M training flows.

## Results (test: 1.29 M flows)

| Last packet | | Streaming | |
|---|---|---|---|
| Macro F1 | 0.6608 | AUROC@1 | **0.8191** |
| ROC AUC | 0.8386 | AUROC@5 / @max | 0.8375 / 0.8386 |
| PR AUC | 0.9592 | EDP median | **1 packet** |
| Precision (adv) | 0.9220 | Detection rate (p>0.5) | 78.40% |
| Recall (adv) / (benign) | 0.7820 / 0.6606 | | |

78% of adversarial flows are flagged at the first packet, and AUROC hardly changes after packet 5. Otherwise the TCN is weaker than the RF:

- **It cannot separate the four reflection floods** (same-capture streaming AUROC ≈ 0.50 on `dns/ldap/ntp/ssdp_ddos` in every mode). These flows have one or two spoofed UDP packets.
- **Timing-only changes are the hardest mode on TCP sources**: `portscan_iat` 0.327, `syn_ddos_iat` 0.602 and `ddos_iat` 0.544, against ≈ 1.0 for `both`/`size`. With so few packets there is almost only one inter-packet interval.
- **`brute_force` and `mirai` fail at the 0.5 threshold** (balanced ROC 0.40–0.52) even though their streaming AUROC is usable.

These are the cases where the RF's prefix statistics work better, which is what the meta-learner relies on.

Artifacts in `data/models/tcn/`: `tcn.pt`, `tcn_config.json`, `dataset_index.npz`, `split.npz`, `metrics.json`, `streaming_metrics.json`, `training_curves.json`, and the ROC / PR / confusion / earliness / training-curve plots.
