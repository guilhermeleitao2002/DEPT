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
| **Class weight** | `balanced_subsample` | Each tree reweights its bootstrap sample — robust to the residual benign:adversarial imbalance left after capping. |
| **Hyperparameter search** | Grid over `n_estimators`, `max_depth`, `min_samples_leaf`; scored by val macro-F1 | Best: 200 trees, `max_depth=20`, `min_samples_leaf=5`. |
| **Final fit** | `train + val`, evaluate on `test` | Maximises data used for the deployable classifier. |

## Results (test split — 1.84 M checkpoints, 1.29 M flows)

> Measured on the full **ten-source** set. The checkpoint pool is adversarial-dominated (≈74% of test checkpoints), so per-checkpoint precision/accuracy must be read against that prior; the flow-level streaming metrics are the cross-model-comparable ones.

### Overall (per-checkpoint)

| Metric | Value |
|---|---|
| ROC AUC | 0.9852 |
| PR AUC | 0.9948 |
| Macro F1 | 0.8935 |
| Accuracy | 0.9099 |
| Precision (adversarial) | **0.9939** |
| Recall (adversarial) | 0.8832 |
| Recall (benign) | 0.9847 |

The confusion profile is precision-first: only 7,406 of 483,114 benign test checkpoints are flagged (FPR 1.5%), while the 0.8832 adversarial recall is dragged by the `_size` variants of the reflection floods (see below).

### Streaming metrics (overall)

| AUROC@t | Value |
|---|---|
| @1 | — (first prediction at packet 2) |
| @2 | 0.9805 |
| @10 | 0.9813 |
| @max | 0.9813 |

| EDP | Value |
|---|---|
| Median | **2 packets** (early checkpoint = earliest possible RF prediction) |
| IQR | [2, 2] |
| Detection rate | 88.99% |

### Per-adversarial-source (verdict-level, balanced benign subsample K=10)

| Source | precision | recall | F1 | ROC AUC | n_adv |
|---|---|---|---|---|---|
| brute_force both | 0.791 | 0.601 | 0.683 | 0.982 | 4,465 |
| brute_force iat | 0.808 | 0.620 | 0.702 | 0.983 | 4,516 |
| brute_force size | 0.846 | 0.866 | 0.856 | 0.990 | 4,531 |
| ddos both | 0.867 | 1.000 | 0.929 | 1.000 | 18,498 |
| ddos iat | 0.854 | 0.886 | 0.870 | 0.983 | 18,441 |
| ddos size | 0.868 | 1.000 | 0.929 | 1.000 | 13,886 |
| dns_ddos both | 0.867 | 1.000 | 0.929 | 0.998 | 16,238 |
| dns_ddos iat | 0.868 | 1.000 | 0.929 | 0.998 | 15,995 |
| dns_ddos size | 0.054 | 0.009 | 0.015 | 0.819 | 15,803 |
| ldap_ddos both | 0.910 | 1.000 | 0.953 | 1.000 | 74,727 |
| ldap_ddos iat | 0.910 | 1.000 | 0.953 | 1.000 | 74,724 |
| ldap_ddos size | 0.174 | 0.021 | 0.037 | 0.947 | 74,729 |
| mirai both | 0.857 | 0.878 | 0.867 | 0.979 | 2,123 |
| mirai iat | 0.852 | 0.798 | 0.824 | 0.967 | 1,472 |
| mirai size | 0.827 | 0.700 | 0.758 | 0.946 | 1,853 |
| ntp_ddos both | 0.866 | 1.000 | 0.928 | 0.998 | 16,512 |
| ntp_ddos iat | 0.865 | 1.000 | 0.928 | 0.999 | 16,478 |
| ntp_ddos size | 0.025 | 0.004 | 0.007 | 0.798 | 16,945 |
| portscan both | 0.868 | 1.000 | 0.929 | 1.000 | 23,829 |
| portscan iat | 0.867 | 1.000 | 0.928 | 0.999 | 23,817 |
| portscan size | 0.866 | 0.987 | 0.923 | 0.996 | 23,824 |
| ssdp_ddos both | 0.867 | 1.000 | 0.929 | 1.000 | 42,192 |
| ssdp_ddos iat | 0.867 | 1.000 | 0.929 | 1.000 | 42,164 |
| ssdp_ddos size | 0.001 | 0.000 | 0.000 | 0.805 | 42,094 |
| ssdp_flood both | 0.866 | 1.000 | 0.928 | 1.000 | 27,016 |
| ssdp_flood iat | 0.854 | 0.895 | 0.874 | 0.976 | 27,007 |
| ssdp_flood size | 0.865 | 1.000 | 0.927 | 1.000 | 27,066 |
| syn_ddos both | 0.969 | 1.000 | 0.984 | 1.000 | 228,198 |
| syn_ddos iat | 0.969 | 1.000 | 0.984 | 1.000 | 228,483 |
| syn_ddos size | 0.969 | 1.000 | 0.984 | 1.000 | 228,058 |

(The per-checkpoint ROC AUC column stays high even for the `_size` flood variants because, against the K=10 balanced benign pool, some threshold still separates them; the *recall at 0.5* collapses to ≈0, and on the flow-level streaming axis against **same-base** benign these variants score *below chance* — RF AUROC@max: `dns_size` 0.258, `ntp_size` 0.214, `ssdp_ddos_size` 0.409, `ldap_size` 0.476.)

## Observations

- **The reflection floods' `both`/`iat` variants are now solved** (ROC AUC ≥ 0.998, recall 1.000): TANTGen's timing perturbation on these near-constant-IPAT floods shifts the IPAT statistics enough for the forest to key on. The aggregate recall cost is concentrated in the four `_size` variants.
- **The four reflection-flood `_size` variants invert.** At threshold 0.5 the RF recalls almost none of them (0.000–0.021), and against same-base benign flows their streaming AUROC is *below 0.5*: the size edit makes a flood flow look more like generic benign traffic than the flood's own benign flows do. This is the RF's one systematic blind spot.
- **IAT-only perturbation is no longer hard for the RF** (`ddos_iat` 0.983, `ssdp_flood_iat` 0.976, everything else ≥ 0.98 except `mirai_iat` 0.967): the prefix IPAT statistics capture pure-timing edits that the per-packet TCN misses.
- **Mirai is the weakest source family** (0.946–0.979 verdict-level; 0.83–0.93 streaming): short bursty IoT flows give the prefix statistics the least to work with.
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
| `earliness_vs_accuracy.png` | Per-source AUROC@t curves |
| `confusion_matrix.png` | Binary confusion matrix |
| `roc_curve.png` | ROC curve |
| `pr_curve.png` | Precision-recall curve |
| `feature_importance.png` | Gini importances |
