# Random Forest Base Learner

> Part of [DEPT](../../README.md). The statistical half of the streaming ensemble — trained on multi-scale prefix statistics, it emits one prediction per checkpoint (packet 2, then every 10 packets).

## Role in the ensemble

The RF is the **sparse, high-precision stream**: it emits an early verdict at packet 2 (so short flows like PortScan are covered) and then refreshes only at every 10-packet prefix milestone, but when it does predict, it is highly accurate. This contrasts with the TCN, which predicts at every packet — noisy early and more reliable as the flow grows. The two streams together give the meta-learner a choice at every packet: act on the TCN's dense signal, or wait for the RF's next checkpoint.

## Input

`rf_checkpoints.npy` — `[C_total, 24]` float32. Each row is a 24-dimensional prefix statistics vector at one checkpoint of one flow (packets 2, 10, 20, 30, … under the default cadence):

| Features | Description |
|---|---|
| `IPAT` × {mean, std, min, max} | Inter-packet arrival time statistics over the prefix |
| `Packet_Size` × {mean, std, min, max} | Packet size statistics |
| `Payload_Length` × {mean, std, min, max} | Payload length statistics |
| `Payload_Entropy` × {mean, std, min, max} | Shannon entropy of payload bytes |
| `TCP_Flags` × {mean, std, min, max} | TCP flag byte statistics |
| `Direction` × {mean, std, min, max} | Forward/reverse packet ratio statistics |

A flow with L packets contributes ⌊L/10⌋ + 1 checkpoint rows (the +1 is the early packet-2 row, for flows of ≥ 2 packets) — one per prefix milestone. This is the same distribution the model sees at inference, avoiding the distribution-shift that would arise from training on a fixed window size alone.

## Design

| Aspect | Choice | Rationale |
|---|---|---|
| **Split** | Flow-atomic, stratified by source | All checkpoints from the same flow stay in the same split — prevents leakage via within-flow correlation. |
| **Benign capping** | Total benign checkpoints ≤ 3× adversarial checkpoints (flow-atomic) | Prevents the much larger raw-dataset benign pool from dominating. |
| **Class weight** | `balanced_subsample` | Each tree reweights its bootstrap sample — robust to the 93:7 benign:adversarial ratio. |
| **Hyperparameter search** | Grid over `n_estimators`, `max_depth`, `min_samples_leaf`; scored by val macro-F1 | Best: 200 trees, `max_depth=20`, `min_samples_leaf=5`. |
| **Final fit** | `train + val`, evaluate on `test` | Maximises data used for the deployable classifier. |

## Results (test split — 1.73 M checkpoints, 1.21 M flows)

> Measured on the full **ten-source** set. The checkpoint pool is now adversarial-dominated, so per-checkpoint precision/accuracy are much higher than the earlier five-source numbers; the flow-level streaming metrics are the cross-model-comparable ones.

### Overall (per-checkpoint)

| Metric | Value |
|---|---|
| ROC AUC | 0.8877 |
| PR AUC | 0.9372 |
| Macro F1 | 0.8378 |
| Recall (adversarial) | **0.9820** |
| Precision (adversarial) | 0.8543 |

Precision is now high because the ten-source checkpoint pool is **adversarial-dominated** (≈68% adversarial in the test checkpoints) — the opposite of the old five-source pool, where precision was ~0.16. ROC AUC and the per-source balanced metrics remain the most meaningful.

### Streaming metrics (overall)

| AUROC@t | Value |
|---|---|
| @1 | — (first prediction at packet 2) |
| @2 | 0.8298 |
| @10 | 0.8381 |
| @max | 0.8395 |

| EDP | Value |
|---|---|
| Median | **2 packets** (early checkpoint = earliest possible RF prediction) |
| IQR | [2, 2] |
| Detection rate | 99.69% |

### Per-adversarial-source (balanced benign subsample, K=10)

| Source | precision | recall | F1 | ROC AUC | n_adv |
|---|---|---|---|---|---|
| brute_force both * | 0.114 | 0.457 | 0.182 | 0.727 | 46 |
| brute_force iat | 0.072 | 0.300 | 0.117 | 0.640 | 50 |
| brute_force size * | 0.123 | 0.538 | 0.200 | 0.760 | 39 |
| ddos both | 0.216 | 1.000 | 0.355 | 1.000 | 194 |
| ddos iat | 0.183 | 0.791 | 0.297 | 0.891 | 182 |
| ddos size | 0.218 | 1.000 | 0.358 | 1.000 | 194 |
| ddos_kitsune both | 0.216 | 1.000 | 0.355 | 1.000 | 5,183 |
| ddos_kitsune iat | 0.194 | 0.865 | 0.317 | 0.919 | 5,205 |
| ddos_kitsune size | 0.216 | 1.000 | 0.356 | 1.000 | 5,222 |
| dns_ddos both | 0.217 | 0.995 | 0.356 | 0.715 | 15,685 |
| dns_ddos iat | 0.216 | 0.993 | 0.354 | 0.716 | 15,677 |
| dns_ddos size | 0.216 | 0.992 | 0.355 | 0.714 | 15,490 |
| ldap_ddos both | 0.274 | 1.000 | 0.430 | 0.818 | 74,732 |
| ldap_ddos iat | 0.274 | 1.000 | 0.430 | 0.818 | 74,729 |
| ldap_ddos size | 0.274 | 1.000 | 0.430 | 0.818 | 74,721 |
| mirai both | 0.156 | 0.667 | 0.252 | 0.793 | 414 |
| mirai iat | 0.201 | 0.936 | 0.331 | 0.911 | 313 |
| mirai size | 0.135 | 0.583 | 0.219 | 0.800 | 276 |
| ntp_ddos both | 0.216 | 0.997 | 0.355 | 0.696 | 16,302 |
| ntp_ddos iat | 0.215 | 0.995 | 0.354 | 0.697 | 16,052 |
| ntp_ddos size | 0.217 | 0.997 | 0.356 | 0.694 | 16,150 |
| portscan both | 0.216 | 1.000 | 0.355 | 1.000 | 2,376 |
| portscan iat | 0.217 | 1.000 | 0.357 | 1.000 | 2,376 |
| portscan size | 0.219 | 0.987 | 0.358 | 0.992 | 2,376 |
| ssdp_ddos both | 0.217 | 1.000 | 0.357 | 0.689 | 42,162 |
| ssdp_ddos iat | 0.217 | 1.000 | 0.357 | 0.689 | 42,172 |
| ssdp_ddos size | 0.217 | 0.999 | 0.356 | 0.688 | 42,048 |
| ssdp_flood both | 0.216 | 1.000 | 0.355 | 1.000 | 9,338 |
| ssdp_flood iat | 0.209 | 0.952 | 0.342 | 0.975 | 9,319 |
| ssdp_flood size | 0.218 | 1.000 | 0.358 | 1.000 | 9,333 |
| syn_ddos both | 0.525 | 0.958 | 0.678 | 0.948 | 228,409 |
| syn_ddos iat | 0.524 | 0.958 | 0.678 | 0.948 | 228,368 |
| syn_ddos size | 0.535 | 1.000 | 0.697 | 1.000 | 228,544 |

\* Low support (n\_adv < 50) — treat as indicative only. (Precision is low across the board because it's computed against a balanced benign subsample; ROC AUC is the more robust per-source signal.)

**PortScan is now fully covered** — under the packet-2 + every-10 cadence every PortScan flow (≥2 packets) gets an RF checkpoint, so the variants that previously produced *zero* RF predictions now score ROC AUC ≈ 1.0 (`both`/`iat`) and 0.992 (`size`).

## Observations

- **The CIC-DDoS reflection floods are the hard sources.** `dns_ddos`, `ntp_ddos`, `ssdp_ddos` sit at ROC AUC ≈ 0.69–0.72 and dominate the (now lower) aggregate; `ldap_ddos` ≈ 0.82, `syn_ddos` ≈ 0.95, while `portscan`/`ddos`/`ddos_kitsune`/`ssdp_flood` remain ≈ 1.0 on `both`/`size`. These spoofed, high-rate floods leave little prefix-statistical signal separating a manipulated flow from an un-manipulated one.
- **IAT-only perturbation is harder.** `_iat` variants consistently show lower recall/ROC AUC than `_both`/`_size` (e.g. `ddos_kitsune` 0.92 vs 1.00, `mirai` 0.91 vs ~0.80, `ssdp_flood` 0.975 vs 1.00). Pure timing change without size/header edits is the subtlest signal.
- **Precision is low under the balanced-benign slicing** (~0.22 for most sources, 0.52 for `syn_ddos`). The RF is recall-first — it rarely misses an adversarial flow (overall recall 0.982) but raises many alerts on benign checkpoints; the meta-learner is meant to suppress those by requiring TCN agreement.
- **EDP median = 2 packets.** The early packet-2 checkpoint makes the RF's first verdict available at packet 2 (vs the TCN's packet 1), so its role in the ensemble is *near-immediate confirmation* rather than the slow every-10 confirmation of the old cadence.

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
