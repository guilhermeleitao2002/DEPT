# Random Forest Base Learner

> Part of [DEPT](../../README.md) — **phase 3**. The **sparse, high-precision** half of the streaming ensemble.

```bash
python3 scripts/rf/run_rf.py
```

## What it does

Consumes `rf_checkpoints.npy` — a 24-dim prefix-statistics vector (mean/std/min/max over IPAT, packet size, payload length, payload entropy, TCP flags, direction) computed at packets 2, 10, 20, … of each flow — and emits one verdict per checkpoint. The early packet-2 checkpoint exists so short flows (PortScan averages ~2 packets) still get an RF verdict; training on the same checkpoint distribution seen at inference avoids fixed-window distribution shift.

Splits are flow-atomic and source-stratified (`seed=42`, shared with the TCN), benign checkpoints are capped at 3× adversarial, and `class_weight=balanced_subsample` handles the residual imbalance. Grid search over `n_estimators` / `max_depth` / `min_samples_leaf` selected **200 trees, depth 20, min_samples_leaf 5** (val macro-F1 0.8925); the final model refits on train+val.

Against the TCN's dense-but-noisy per-packet stream, the RF gives the meta-learner a choice at every packet: act now, or wait for the next checkpoint.

## Results (test: 1.84 M checkpoints / 1.29 M flows, ten sources)

| Per-checkpoint | | Streaming (flow-level) | |
|---|---|---|---|
| ROC AUC | 0.9852 | AUROC@2 | **0.9805** |
| PR AUC | 0.9948 | AUROC@10 / @max | **0.9813** |
| Macro F1 | 0.8935 | EDP median | 2 packets |
| Precision (adv) | **0.9939** | Detection rate (p>0.5) | **88.99%** |
| Recall (adv) | 0.8832 | | |

The pool is adversarial-dominated (≈74% of test checkpoints), so read precision/accuracy against that prior; the flow-level column is the cross-model-comparable one. The profile is precision-first — 7,406 of 483,114 benign checkpoints flagged (FPR 1.5%).

**On the flow-level axis the RF is DEPT's strongest single stream**, essentially flat from packet 2 onward. Two caveats:

- **The reflection-flood `_size` variants are its one systematic blind spot.** Recall at threshold 0.5 collapses to 0.000–0.021 on `dns/ntp/ssdp/ldap_ddos_size`, and against same-base benign their streaming AUROC is *below chance* (0.21–0.48): the size edit makes a flood flow look more like generic benign traffic than that flood's own benign flows do. The `both`/`iat` variants of the same floods are solved (ROC ≥ 0.998, recall 1.000).
- **Timing-only edits are easy for the RF** (≥ 0.98 everywhere except `mirai_iat` 0.967) — exactly where the per-packet TCN fails, which is the complementarity the meta-learner exploits. **Mirai is the weakest source** (0.83–0.93 streaming): short bursty IoT flows give prefix statistics little to work with.

Artifacts under `data/models/rf/`: `rf.joblib`, `dataset.npz`, `split.npz`, `metrics.json`, `streaming_metrics.json`, `training_info.json`, plus ROC / PR / confusion / earliness / feature-importance plots.
