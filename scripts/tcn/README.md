# TCN Base Learner

> Part of [DEPT](../../README.md). The temporal half of the streaming ensemble — a causal Temporal Convolutional Network that emits a prediction after every new packet.

## Role in the ensemble

The TCN is the **dense, early-warning stream**: it produces a prediction the moment the first packet of a flow arrives and updates it with every subsequent packet. It trades precision (it is noisier than the RF early on) for earliness — it flags a flow from packet 1, before the RF's first checkpoint at packet 2 and far ahead of the RF's every-10-packet refresh cadence. This is the production motivation: a verdict is available from the very first packet, regardless of flow length.

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
    ├─ TemporalBlock(1090 → 128, k=3, d=1, chomp)  receptive field: 1+4=5
    │       Causal dilated Conv1d × 2, weight norm, ReLU, Dropout(0.15)
    │       Residual via 1×1 Conv1d
    │
    ├─ TemporalBlock(128 → 128, k=3, d=2, chomp)   receptive field: +8=13
    │
    ├─ TemporalBlock(128 → 128, k=3, d=4, chomp)   receptive field: +16=29
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

Best epoch: 6 of 10 run (early-stopped, patience 3; 20 planned). Val macro-F1 = 0.6611. Training time: 24.7 h on one GPU (6.04 M training flows).

## Results (test split — 1.29 M flows)

> Measured on the full **ten-source** set (adversarial-dominated; the flow-level streaming metrics are the cross-model-comparable axis).

### Overall (last timestep)

| Metric | Value |
|---|---|
| Macro F1 | 0.6608 |
| ROC AUC | 0.8386 |
| PR AUC | 0.9592 |
| Accuracy | 0.7621 |
| Precision (adversarial) | 0.9220 |
| Recall (adversarial) | 0.7820 |
| Recall (benign) | 0.6606 |

### Streaming metrics (overall)

| AUROC@t | Value |
|---|---|
| @1 | **0.8191** |
| @5 | 0.8375 |
| @10 | 0.8371 |
| @20 | 0.8386 |
| @max | 0.8386 |

| EDP | Value |
|---|---|
| Median | **1 packet** |
| IQR | [1, 1] |
| Detection rate | 78.40% |

### Per-adversarial-source (verdict-level, balanced benign subsample K=10)

| Source | precision | recall | F1 | ROC AUC | n_adv |
|---|---|---|---|---|---|
| brute_force both | 0.007 | 0.026 | 0.012 | 0.447 | 1,045 |
| brute_force iat | 0.008 | 0.028 | 0.013 | 0.444 | 1,045 |
| brute_force size | 0.010 | 0.033 | 0.015 | 0.445 | 1,045 |
| ddos both | 0.227 | 1.000 | 0.370 | 1.000 | 5,781 |
| ddos iat | 0.097 | 0.366 | 0.153 | 0.544 | 5,781 |
| ddos size | 0.226 | 1.000 | 0.369 | 1.000 | 4,842 |
| dns_ddos both | 0.192 | 0.800 | 0.310 | 0.856 | 1,019 |
| dns_ddos iat | 0.194 | 0.809 | 0.313 | 0.859 | 1,019 |
| dns_ddos size | 0.192 | 0.793 | 0.310 | 0.847 | 1,019 |
| ldap_ddos both | 0.510 | 1.000 | 0.676 | 0.799 | 74,710 |
| ldap_ddos iat | 0.510 | 1.000 | 0.676 | 0.799 | 74,710 |
| ldap_ddos size | 0.510 | 1.000 | 0.676 | 0.799 | 74,710 |
| mirai both | 0.008 | 0.029 | 0.013 | 0.397 | 1,957 |
| mirai iat | 0.007 | 0.026 | 0.012 | 0.401 | 1,957 |
| mirai size | 0.059 | 0.214 | 0.093 | 0.518 | 1,957 |
| ntp_ddos both | 0.010 | 0.036 | 0.016 | 0.469 | 2,103 |
| ntp_ddos iat | 0.007 | 0.026 | 0.011 | 0.461 | 2,103 |
| ntp_ddos size | 0.009 | 0.031 | 0.014 | 0.470 | 2,103 |
| portscan both | 0.247 | 0.988 | 0.395 | 0.992 | 23,800 |
| portscan iat | 0.000 | 0.000 | 0.000 | 0.327 | 23,800 |
| portscan size | 0.247 | 0.987 | 0.395 | 0.992 | 23,800 |
| ssdp_ddos both | 0.338 | 0.905 | 0.492 | 0.819 | 40,446 |
| ssdp_ddos iat | 0.339 | 0.910 | 0.494 | 0.821 | 40,446 |
| ssdp_ddos size | 0.339 | 0.908 | 0.493 | 0.819 | 40,446 |
| ssdp_flood both | 0.227 | 1.000 | 0.370 | 0.998 | 9,831 |
| ssdp_flood iat | 0.228 | 1.000 | 0.371 | 0.999 | 9,831 |
| ssdp_flood size | 0.227 | 1.000 | 0.370 | 0.998 | 9,831 |
| syn_ddos both | 0.737 | 1.000 | 0.848 | 1.000 | 200,483 |
| syn_ddos iat | 0.209 | 0.094 | 0.130 | 0.602 | 200,483 |
| syn_ddos size | 0.737 | 1.000 | 0.848 | 1.000 | 200,483 |

Precision is computed against a balanced benign subsample, so absolute values are low where recall is low; ROC AUC is the robust per-source signal. Note the balanced-benign ROC for `dns_ddos`/`ssdp_ddos` (0.82–0.86) is *cross-source* separability — against **same-base** benign flows these floods sit at streaming AUROC ≈ 0.50 in every mode (the TCN cannot tell a perturbed flood flow from that flood's own benign flows).

## Observations

- **EDP median = 1 packet.** The TCN flags 78% of adversarial flows from the very first packet — before the RF's packet-2 checkpoint. This packet-1 earliness is the core advantage the meta-learner inherits.
- **The four reflection floods are the TCN's blind spot** (same-base streaming AUROC ≈ 0.50 on `dns_ddos`, `ldap_ddos`, `ntp_ddos`, `ssdp_ddos` in *every* mode): these one-or-two-packet spoofed UDP flows give the per-packet sequence essentially nothing to grip, and the aggregate is carried by the TCP-ish sources.
- **IAT-only perturbation is the hardest mode on TCP sources**: `portscan_iat` 0.327 (below chance — perturbed flows look *less* suspicious than benign), `syn_ddos_iat` 0.602, `ddos_iat` 0.544, vs ≈ 1.0 for the `_both`/`_size` variants of the same attacks. With few packets there is effectively one inter-packet interval, so a pure timing change is hard to separate from natural variability.
- **`brute_force` and `mirai` fail at verdict level** (ROC 0.40–0.52 balanced) despite decent same-base streaming AUROC (0.98 / 0.68–0.72): the TCN's threshold behaviour on these sources is calibrated to the wrong operating point — another reason the ensemble relies on the RF for them.
- **AUROC is roughly flat after packet 5** (0.8375 → 0.8386 at max): most of the TCN's discriminative signal is present within the first few packets of a flow.
- The RF's aggregated prefix statistics are more informative than the raw per-packet sequence for the floods and timing-only edits, which is exactly the complementarity the meta-learner is designed to exploit.

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
| `earliness_vs_accuracy.png` | Per-source AUROC@t curves |
| `confusion_matrix.png` | Binary confusion matrix |
| `roc_curve.png` | ROC curve |
| `pr_curve.png` | Precision-recall curve |
| `training_curves.png` | Loss + macro-F1 over epochs |
