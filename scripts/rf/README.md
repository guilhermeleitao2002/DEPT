# Random Forest Base Learner

> Part of [DEPT](../../README.md), **phase 3**. The sparse, high-precision stream of the ensemble.

```bash
python3 scripts/rf/run_rf.py
```

## Method

The RF reads `rf_checkpoints.npy`: a 24-dim vector of prefix statistics (mean/std/min/max of IPAT, packet size, payload length, payload entropy, TCP flags and direction) computed at packets 2, 10, 20, … of each flow, and gives one verdict per checkpoint. The checkpoint at packet 2 exists so that short flows (PortScan flows have about 2 packets) also get a verdict. Training uses the same checkpoints that are seen at inference.

The split is flow-level and stratified by source (`seed=42`, the same as the TCN), and `class_weight=balanced_subsample` handles the class imbalance. The grid search over `n_estimators` / `max_depth` / `min_samples_leaf` selected **200 trees, depth 20, min_samples_leaf 5** (val macro-F1 0.8925); the final model is refit on train+val.

## Results (test: 1.84 M checkpoints / 1.29 M flows)

| Per checkpoint | | Streaming (per flow) | |
|---|---|---|---|
| ROC AUC | 0.9852 | AUROC@2 | **0.9805** |
| PR AUC | 0.9948 | AUROC@10 / @max | **0.9813** |
| Macro F1 | 0.8935 | EDP median | 2 packets |
| Precision (adv) | **0.9939** | Detection rate (p>0.5) | **88.99%** |
| Recall (adv) | 0.8832 | | |

About 74% of the test checkpoints are adversarial, so precision and accuracy depend on that prior; the per-flow streaming metrics are the ones to compare between models. 7,406 of 483,114 benign checkpoints are flagged (FPR 1.5%).

Per flow, the RF is the strongest single model and is almost flat from packet 2 on. Notes per source:

- **The `_size` variants of the reflection floods are not detected.** Recall at 0.5 is 0.000–0.021 on `dns/ntp/ssdp/ldap_ddos_size`, and against benign flows of the same capture the streaming AUROC is below chance (0.21–0.48): after the size change, these flows look more like generic benign traffic than the capture's own benign flows. The `both`/`iat` variants of the same floods are detected (ROC ≥ 0.998, recall 1.000).
- **Timing-only changes are easy for the RF** (≥ 0.98 everywhere except `mirai_iat`, 0.967), while they are hard for the TCN.
- **Mirai is the hardest source** (0.83–0.93 streaming): its flows are short and bursty.

Artifacts in `data/models/rf/`: `rf.joblib`, `dataset.npz`, `split.npz`, `metrics.json`, `streaming_metrics.json`, `training_info.json`, and the ROC / PR / confusion / earliness / feature-importance plots.
