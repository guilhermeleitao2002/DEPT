# TCN Base Learner

> Part of [ADEPT](../../README.md). The temporal half of the streaming ensemble — a causal Temporal Convolutional Network that emits a prediction after every new packet.

## Role in the ensemble

The TCN is the **dense, early-warning stream**: it produces a prediction the moment the first packet of a flow arrives and updates it with every subsequent packet. It trades precision (it is noisier than the RF early on) for latency — it can detect adversarial flows orders of magnitude earlier than the RF's first 10-packet checkpoint. This is the production motivation: detection latency is bounded by TCN forward-pass time, not by flow length.

## Input

`flow_packets.npy` — `[P_total, 1090]` float32 memmap. Each row is one packet, encoded as:

- **1088 nPrint bits** — tri-state `{-1, 0, 1}` encoding of all IPv4/TCP/UDP/ICMP header bits (identity fields zeroed).
- **IPAT** — Inter-Packet Arrival Time within the flow (z-score normalized).
- **Direction** — `1` = forward, `0` = reverse.

Flows are loaded as variable-length sequences via `(start_idx, length)` offsets. No fixed window size, no padding at storage time — padding is applied within each training batch to the batch-maximum length.

## Architecture

```
Input [B, T, 1090]
    │
    ├─ TemporalBlock(1090 → 128, k=3, d=1, chomp)  receptive field: 1+2=3
    │       Causal dilated Conv1d × 2, weight norm, ReLU, Dropout(0.15)
    │       Residual via 1×1 Conv1d
    │
    ├─ TemporalBlock(128 → 128, k=3, d=2, chomp)   receptive field: +4=7
    │
    ├─ TemporalBlock(128 → 128, k=3, d=4, chomp)   receptive field: +8=15 (cumulative: 29)
    │
    └─ Conv1d(128 → 2, k=1)   per-timestep head
       → [B, T, 2] logits
```

Total receptive field: **29 packets**. Every output at position `t` depends only on inputs at positions ≤ t — causality is enforced by the Chomp1d layer (right-side trim after each dilated convolution). This makes the model streaming-compatible: shifting the input by one packet shifts the output by one packet with no leakage.

805,762 parameters. Trained with AdamW + CosineAnnealingLR, mixed precision on CUDA.

## Training

- **Loss**: per-packet cross-entropy with class weights (`balanced`), averaged within each flow before averaging across flows. This prevents long flows from dominating the gradient.
- **Batching**: variable-length flows padded to the batch-maximum length; a padding mask suppresses gradient at padded positions. Flows truncated at 200 packets before batching to bound the padded tensor size.
- **Split**: flow-atomic, source-stratified — same `seed=42` and same split logic as the RF, so the two models are evaluated on exactly the same held-out flows.

Best epoch: 17/20, val macro-F1 = 0.9611. Training time: 121 min on RTX 4060 Ti.

## Results (test split — 131 K flows)

> Measured on the original five-source dataset, **before the CIC-DDoS-2019 expansion** (the five reflection floods `ntp_ddos`/`dns_ddos`/`ssdp_ddos`/`syn_ddos`/`ldap_ddos` and their adversarials). Re-run `run_tcn.py` after `process_all_attacks.py --tantgen` to refresh these on the ten-source set.

### Overall

| Metric | Value |
|---|---|
| Macro F1 | **0.9608** |
| ROC AUC | 0.9857 |
| PR AUC | 0.9807 |
| Accuracy | 0.9652 |
| Precision (adversarial) | 0.9893 |
| Recall (adversarial) | 0.9094 |

### Streaming metrics (overall)

| AUROC@t | Value |
|---|---|
| @1 | **0.9493** |
| @5 | 0.9815 |
| @10 | 0.9817 |
| @20 | 0.9851 |
| @max | 0.9857 |

| EDP | Value |
|---|---|
| Median | **1 packet** |
| IQR | [1, 1] |
| Detection rate | 91.7% |

### Per-adversarial-source (balanced benign subsample, K=10)

| Source | precision | recall | F1 | ROC AUC | n_adv |
|---|---|---|---|---|---|
| brute_force both * | 0.889 | 0.667 | 0.762 | 0.903 | 12 |
| brute_force iat * | 0.857 | 0.500 | 0.632 | 0.869 | 12 |
| brute_force size * | 1.000 | 0.417 | 0.588 | 0.862 | 12 |
| ddos both | 0.962 | 1.000 | 0.981 | 1.000 | 151 |
| ddos iat | 0.947 | 0.828 | 0.883 | 0.977 | 151 |
| ddos size | 0.956 | 1.000 | 0.977 | 1.000 | 151 |
| ddos_kitsune both | 0.951 | 1.000 | 0.975 | 1.000 | 2,373 |
| ddos_kitsune iat | 0.934 | 0.719 | 0.812 | 0.971 | 2,373 |
| ddos_kitsune size | 0.951 | 1.000 | 0.975 | 1.000 | 2,373 |
| mirai both | 0.900 | 0.486 | 0.631 | 0.972 | 574 |
| mirai iat | 0.876 | 0.396 | 0.545 | 0.965 | 574 |
| mirai size | 0.910 | 0.509 | 0.653 | 0.975 | 574 |
| portscan both | 0.946 | 0.988 | 0.967 | 0.998 | 2,376 |
| portscan iat | 0.024 | 0.001 | 0.002 | 0.789 | 2,376 |
| portscan size | 0.946 | 0.988 | 0.966 | 0.998 | 2,376 |
| ssdp_flood both | 0.956 | 1.000 | 0.977 | 1.000 | 9,708 |
| ssdp_flood iat | 0.956 | 0.994 | 0.974 | 0.999 | 9,708 |
| ssdp_flood size | 0.956 | 1.000 | 0.977 | 1.000 | 9,708 |

\* Low support (n\_adv < 50) — treat as indicative only.

## Observations

- **EDP median = 1 packet.** The TCN flags 91.7% of adversarial flows before the RF even makes its first prediction. This is the core production advantage of the streaming design.
- **IAT-only perturbation is consistently harder.** All `_iat` variants show lower recall and ROC AUC than the corresponding `_both` and `_size` variants. Timing perturbation without structural changes produces less visible signal in the per-packet feature space.
- **portscan\_iat is a near-total failure** (F1=0.002, AUROC@1=0.789). PortScan flows average ~2 packets — there is effectively one inter-packet interval. TANTGen's IAT perturbation on a 2-packet flow changes a single timing value, which is indistinguishable from natural timing variability at the per-packet level. The `_both` and `_size` variants, which also perturb packet sizes and headers, remain detectable (F1≈0.97).
- **Mirai adversarial recall is low** (0.40–0.51 across all modes). The RF reaches AUROC 0.85–0.999 on Mirai variants, suggesting that the aggregated statistical view is more informative than the raw per-packet sequence for this attack type. This makes Mirai a candidate where RF confirmation is particularly valuable.
- **AUROC is stable after packet 10.** The jump from AUROC@1 (0.949) to AUROC@10 (0.982) is large; subsequent gains (→0.986 at max) are marginal. Most of the TCN's discriminative information is present within the first 10 packets of adversarial flows.

## Output files (`data/models/tcn/`)

| File | Description |
|---|---|
| `tcn.pt` | Best PyTorch `state_dict` (best val macro-F1 epoch) |
| `tcn_config.json` | Resolved model + training config |
| `dataset_index.npz` | Per-flow index: source, local_idx, y, flow_ids, label_str |
| `split.npz` | `train_idx`, `val_idx`, `test_idx` (element-wise compatible with RF's split.npz) |
| `split_summary.json` | Per-split flow counts (benign / adversarial / per-source) |
| `training_info.json` | Best epoch, device, train time, effective config |
| `training_curves.json` | Per-epoch train_loss, val_loss, val_macro_f1, lr |
| `metrics.json` | Overall + per-adversarial-source metrics |
| `streaming_metrics.json` | AUROC@t and EDP per source |
| `confusion_matrix.png` | Binary confusion matrix |
| `roc_curve.png` | ROC curve |
| `pr_curve.png` | Precision-recall curve |
| `training_curves.png` | Loss + macro-F1 over epochs |
