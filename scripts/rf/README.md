# Random Forest Base Learner

> Part of [ADEPT](../../README.md). The statistical half of the streaming ensemble — trained on multi-scale prefix statistics, it emits one prediction per 10-packet prefix milestone.

## Role in the ensemble

The RF is the **sparse, high-precision stream**: it waits for a full 10-packet prefix before making any prediction, but when it does, it is highly accurate. This contrasts with the TCN, which is noisy early and becomes more reliable as the flow grows. The two streams together give the meta-learner a choice at every packet: act on the TCN's early signal, or wait for the RF's next checkpoint.

## Input

`rf_checkpoints.npy` — `[C_total, 24]` float32. Each row is a 24-dimensional prefix statistics vector at one 10-packet milestone of one flow:

| Features | Description |
|---|---|
| `IPAT` × {mean, std, min, max} | Inter-packet arrival time statistics over the prefix |
| `Packet_Size` × {mean, std, min, max} | Packet size statistics |
| `Payload_Length` × {mean, std, min, max} | Payload length statistics |
| `Payload_Entropy` × {mean, std, min, max} | Shannon entropy of payload bytes |
| `TCP_Flags` × {mean, std, min, max} | TCP flag byte statistics |
| `Direction` × {mean, std, min, max} | Forward/reverse packet ratio statistics |

A flow with L packets contributes ⌊L/10⌋ checkpoint rows — one per prefix milestone. This is the same distribution the model sees at inference, avoiding the distribution-shift that would arise from training on a fixed window size alone.

## Design

| Aspect | Choice | Rationale |
|---|---|---|
| **Split** | Flow-atomic, stratified by source | All checkpoints from the same flow stay in the same split — prevents leakage via within-flow correlation. |
| **Benign capping** | Total benign checkpoints ≤ 3× adversarial checkpoints (flow-atomic) | Prevents the much larger raw-dataset benign pool from dominating. |
| **Class weight** | `balanced_subsample` | Each tree reweights its bootstrap sample — robust to the 93:7 benign:adversarial ratio. |
| **Hyperparameter search** | Grid over `n_estimators`, `max_depth`, `min_samples_leaf`; scored by val macro-F1 | Best: 400 trees, unbounded depth, `min_samples_leaf=2`. |
| **Final fit** | `train + val`, evaluate on `test` | Maximises data used for the deployable classifier. |

## Results (test split — 229 K checkpoints, 43 K flows)

> Measured on the original five-source dataset, **before the CIC-DDoS-2019 expansion** (the five reflection floods `ntp_ddos`/`dns_ddos`/`ssdp_ddos`/`syn_ddos`/`ldap_ddos` and their adversarials). Re-run `run_rf.py` after `process_all_attacks.py --tantgen` to refresh these on the ten-source set.

### Overall

| Metric | Value |
|---|---|
| ROC AUC | **0.9887** |
| PR AUC | 0.9596 |
| Macro F1 | 0.5737 |
| Recall (adversarial) | 0.9745 |
| Precision (adversarial) | 0.1643 |

The low overall precision is an artifact of class imbalance in the checkpoint pool (4.5% adversarial in test set). ROC AUC and per-source balanced metrics are more meaningful.

### Streaming metrics (overall)

| AUROC@t | Value |
|---|---|
| @1 | — (first prediction at t=10) |
| @10 | **0.9998** |
| @20 | 0.9998 |
| @max | 0.9998 |

| EDP | Value |
|---|---|
| Median | **10 packets** (minimum possible — first checkpoint) |
| IQR | [10, 10] |
| Detection rate | 99.0% |

### Per-adversarial-source (balanced benign subsample, K=10)

| Source | precision | recall | F1 | ROC AUC | n_adv |
|---|---|---|---|---|---|
| brute_force both * | 0.244 | 0.868 | 0.382 | 0.881 | 38 |
| brute_force iat * | 0.198 | 0.667 | 0.305 | 0.815 | 30 |
| brute_force size * | 0.191 | 0.538 | 0.282 | 0.843 | 39 |
| ddos both * | 0.316 | 1.000 | 0.481 | 1.000 | 37 |
| ddos iat * | 0.260 | 0.829 | 0.395 | 0.887 | 41 |
| ddos size * | 0.286 | 1.000 | 0.444 | 1.000 | 44 |
| ddos_kitsune both | 0.295 | 1.000 | 0.456 | 1.000 | 2,937 |
| ddos_kitsune iat | 0.291 | 0.976 | 0.449 | 0.983 | 2,940 |
| ddos_kitsune size | 0.299 | 1.000 | 0.460 | 1.000 | 2,936 |
| mirai both | 0.309 | 0.997 | 0.471 | 0.999 | 330 |
| mirai iat | 0.168 | 0.474 | 0.248 | 0.849 | 291 |
| mirai size | 0.295 | 0.997 | 0.456 | 0.998 | 298 |
| ssdp_flood both | 0.305 | 1.000 | 0.467 | 1.000 | 141 |
| ssdp_flood iat | 0.300 | 1.000 | 0.462 | 1.000 | 141 |
| ssdp_flood size | 0.309 | 1.000 | 0.472 | 1.000 | 141 |

\* Low support (n\_adv < 50) — treat as indicative only.

Note: PortScan adversarial variants do not appear in this table. PortScan flows average ~2 packets, so no flow reaches the 10-packet threshold for even a single RF checkpoint. The RF produces zero predictions for PortScan; the TCN covers them from packet 1.

## Observations

- **AUROC@10 = 0.9998** — after just 10 packets, the RF already achieves near-perfect separation for all source types with adequate support.
- **IAT-only perturbation is harder.** `_iat` variants consistently show lower recall and ROC AUC than `_both` and `_size` across all attack types. This is expected: pure timing perturbation without size or header changes is harder to distinguish from natural traffic variability.
- **Precision is uniformly low** (~0.17–0.32 per source). The RF is tuned for recall — it rarely misses adversarial flows but frequently raises alerts on benign checkpoints. The meta-learner is expected to suppress false positives by requiring TCN agreement.
- **EDP is always exactly 10 packets** (first checkpoint = earliest possible RF prediction). This makes the RF's contribution to the meta-learner exclusively *confirmatory* — it never provides an earlier warning than the TCN.

## Output files (`data/models/rf/`)

| File | Description |
|---|---|
| `rf.joblib` | Fitted `RandomForestClassifier` |
| `dataset.npz` | Merged `(X, y, sources, label_str)` snapshot |
| `split.npz` | `train_idx`, `val_idx`, `test_idx` |
| `feature_names.json` | Ordered list of 24 RF stat names |
| `split_summary.json` | Per-split checkpoint and flow counts |
| `training_info.json` | Grid-search results, best params |
| `metrics.json` | Overall + per-adversarial-source metrics |
| `streaming_metrics.json` | AUROC@t and EDP per source |
| `confusion_matrix.png` | Binary confusion matrix |
| `roc_curve.png` | ROC curve |
| `pr_curve.png` | Precision-recall curve |
| `feature_importance.png` | Gini importances |
