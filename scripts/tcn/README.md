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

Best epoch: 5/9 (early-stopped). Val macro-F1 = 0.7264. Training time: ~17.6 h on one GPU (5.66 M training flows).

## Results (test split — 1.21 M flows)

> Measured on the full **ten-source** set (adversarial-dominated; the flow-level streaming metrics are the cross-model-comparable axis).

### Overall (last timestep)

| Metric | Value |
|---|---|
| Macro F1 | 0.7261 |
| ROC AUC | 0.8359 |
| PR AUC | 0.9426 |
| Accuracy | 0.8225 |
| Precision (adversarial) | 0.9225 |
| Recall (adversarial) | 0.8570 |

### Streaming metrics (overall)

| AUROC@t | Value |
|---|---|
| @1 | **0.8009** |
| @5 | 0.8248 |
| @10 | 0.8330 |
| @20 | 0.8359 |
| @max | 0.8359 |

| EDP | Value |
|---|---|
| Median | **1 packet** |
| IQR | [1, 1] |
| Detection rate | 86.3% |

### Per-adversarial-source (balanced benign subsample, K=10)

| Source | precision | recall | F1 | ROC AUC | n_adv |
|---|---|---|---|---|---|
| brute_force both * | 0.127 | 0.583 | 0.209 | 0.701 | 12 |
| brute_force iat * | 0.043 | 0.167 | 0.069 | 0.429 | 12 |
| brute_force size * | 0.062 | 0.250 | 0.100 | 0.470 | 12 |
| ddos both | 0.226 | 1.000 | 0.368 | 1.000 | 151 |
| ddos iat | 0.181 | 0.775 | 0.294 | 0.863 | 151 |
| ddos size | 0.219 | 1.000 | 0.360 | 1.000 | 151 |
| ddos_kitsune both | 0.229 | 1.000 | 0.373 | 1.000 | 2,373 |
| ddos_kitsune iat | 0.044 | 0.159 | 0.069 | 0.483 | 2,373 |
| ddos_kitsune size | 0.227 | 0.999 | 0.370 | 0.999 | 2,373 |
| dns_ddos both | 0.046 | 0.164 | 0.072 | 0.669 | 1,019 |
| dns_ddos iat | 0.046 | 0.162 | 0.072 | 0.668 | 1,019 |
| dns_ddos size | 0.045 | 0.161 | 0.070 | 0.669 | 1,019 |
| ldap_ddos both | 0.508 | 0.997 | 0.673 | 0.837 | 74,710 |
| ldap_ddos iat | 0.508 | 0.997 | 0.673 | 0.837 | 74,710 |
| ldap_ddos size | 0.508 | 0.997 | 0.673 | 0.837 | 74,710 |
| mirai both | 0.084 | 0.328 | 0.134 | 0.558 | 574 |
| mirai iat | 0.000 | 0.000 | 0.000 | 0.360 | 574 |
| mirai size | 0.095 | 0.364 | 0.151 | 0.582 | 574 |
| ntp_ddos both | 0.000 | 0.001 | 0.000 | 0.614 | 2,103 |
| ntp_ddos iat | 0.000 | 0.000 | 0.000 | 0.616 | 2,103 |
| ntp_ddos size | 0.001 | 0.002 | 0.001 | 0.614 | 2,103 |
| portscan both | 0.223 | 0.988 | 0.363 | 0.992 | 2,376 |
| portscan iat | 0.000 | 0.000 | 0.000 | 0.320 | 2,376 |
| portscan size | 0.224 | 0.984 | 0.365 | 0.990 | 2,376 |
| ssdp_ddos both | 0.000 | 0.000 | 0.000 | 0.550 | 40,446 |
| ssdp_ddos iat | 0.000 | 0.000 | 0.000 | 0.551 | 40,446 |
| ssdp_ddos size | 0.000 | 0.000 | 0.000 | 0.551 | 40,446 |
| ssdp_flood both | 0.227 | 1.000 | 0.369 | 1.000 | 9,708 |
| ssdp_flood iat | 0.203 | 0.871 | 0.329 | 0.946 | 9,708 |
| ssdp_flood size | 0.225 | 1.000 | 0.368 | 1.000 | 9,708 |
| syn_ddos both | 0.733 | 0.987 | 0.841 | 0.836 | 200,483 |
| syn_ddos iat | 0.733 | 0.987 | 0.841 | 0.836 | 200,483 |
| syn_ddos size | 0.735 | 1.000 | 0.847 | 1.000 | 200,483 |

\* Low support (n\_adv < 50) — treat as indicative only. Precision is computed against a balanced benign subsample, so absolute values are low; ROC AUC is the robust per-source signal.

## Observations

- **EDP median = 1 packet.** The TCN flags 86% of adversarial flows from the very first packet — before the RF's packet-2 checkpoint. This packet-1 latency is the core advantage the meta-learner inherits.
- **IAT-only perturbation is consistently the hardest mode**, and on short flows it collapses entirely: `portscan_iat` (ROC AUC 0.320, F1 0.000), `mirai_iat` (0.360, 0.000), `ddos_kitsune_iat` (0.483). With ~2 packets there is effectively one inter-packet interval, so a pure timing change is indistinguishable from natural variability — while the `_both`/`_size` variants of the same attacks stay detectable (ROC AUC ≈ 0.99).
- **The CIC-DDoS reflection floods are the new hard sources.** `dns_ddos`/`ntp_ddos`/`ssdp_ddos` sit at ROC AUC ≈ 0.55–0.67 across all modes; `ldap_ddos` (0.837) and `syn_ddos` (0.836–1.000) fare better. These spoofed, near-constant-timing floods are where the per-packet view has least to grip.
- **Mirai/ntp/ssdp adversarial recall is low for the TCN.** The RF's aggregated prefix statistics are more informative than the raw per-packet sequence for these floods, which is exactly the complementarity the meta-learner is designed to exploit.
- **AUROC is roughly flat after packet 5** (0.825 → 0.836 at max): most of the TCN's discriminative signal is present within the first few packets of a flow.

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
