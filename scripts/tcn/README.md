# Phase 3 — TCN Base Learner

> Part of [ADEPT](../../README.md). The temporal half of the ADEPT ensemble — a binary benign-vs-adversarial classifier trained on the raw `[N=10, F=1090]` packet-sequence tensor produced by [Phase 1](../packet_to_tensor/README.md). Paired with the [Random Forest](../rf/README.md) base learner, which consumes the disjoint 24-dim aggregate-statistics view, to maximise representational diversity.

## Role in the ensemble

The TCN learns the same benign-vs-adversarial distinction the RF does, but from the bit-level packet stream rather than aggregated statistics. This means timing perturbations (jitter, reordering) and fine-grained header tweaks that average out of the RF's mean/std/min/max view are still visible to the temporal model. The two share a single flow-level split (driven by `seed=42` in both `rf.yaml` and `tcn.yaml`) so a window the RF was tested on is also a window the TCN was tested on — a precondition for the meta-learner that fuses them later.

## Design

| Aspect | Choice | Rationale |
|---|---|---|
| **Input** | `sequence.npy` — `[W, N=10, F=1090]` float32 (memmap) | 1088 nPrint bits + IPAT (z-score) + Direction. Read lazily — a single source can be ~17 GB so concatenation into RAM is impossible. |
| **Architecture** | Bai et al. 2018 TCN: 3 stacked dilated causal Conv1d blocks (channels=[128,128,128], k=3, d=[1,2,4]), weight-normed, residual, with mean+max temporal pooling and a 2-class linear head | Dilations give a receptive field of 29 ≫ N=10. ~0.8 M parameters fit on any modern GPU; causal padding makes the model streaming-compatible. |
| **Labels** | Same as RF: `0` = benign, `1` = adversarial (TANTGen) | Dataset-driven via `attach_labels.py`. |
| **Benign pool** | Same 4 sources as `rf.yaml` (ddos, dos_slow, mirai, ssdp_flood) | Same coverage so RF and TCN see the same windows. |
| **Adversarial pool** | Auto-discovered, same exclusion list as RF | 4 attacks × 3 modes = 12 adversarial folders. |
| **Split** | Flow-level, source-stratified — shared with RF via `src/models/common/dataset.py` | Same `seed`, same `assemble_index`, same `flow_level_split`. |
| **Benign capping** | Total benign windows ≤ `benign_cap_ratio` × adversarial windows (flow-atomic, global) | Same value as RF (3.0). |
| **Loss** | Class-weighted `CrossEntropyLoss` | Mirrors sklearn's `class_weight='balanced'` from RF. |
| **Optimiser** | AdamW + CosineAnnealingLR, mixed precision on CUDA | Standard recipe; AMP halves activation memory. |
| **Early stopping** | Patience on val macro-F1 | Best state restored at end of training. |

## Module reference

### `src/models/common/dataset.py` — shared by RF and TCN

| Function | Purpose |
|---|---|
| `assemble_index(...)` | Build a per-window index `(source, local_idx, y, flow_ids, label_str)` over benign + adversarial sources, with flow-atomic global benign capping. **Loads no payload.** |
| `flow_level_split(sources, flow_ids, ...)` | Stratified, flow-atomic split. |
| `split_summary(sources, y, split)` | Per-split counts: total, benign, adversarial, per-source. |
| `cap_benign`, `discover_adversarial_sources`, `load_source_meta` | Shared helpers. |

### `src/models/tcn/model.py` — architecture

| Class | Purpose |
|---|---|
| `Chomp1d` | Right-side strip to enforce causality (Bai et al. 2018 §3.1). |
| `TemporalBlock` | 2× weight-normed dilated `Conv1d` + ReLU + Dropout, with residual (1×1 Conv1d if channels change). |
| `TCN` | Stacked `TemporalBlock`s + mean+max temporal pooling + linear head. `forward(x)` accepts `[B, T, F]` and returns `[B, num_classes]` logits. |

### `src/models/tcn/dataset.py` — lazy memmap reader

| Class / function | Purpose |
|---|---|
| `WindowMemmapCache` | Lazy per-source `np.memmap` pool. Pickleable to DataLoader workers (paths only at pickle time). |
| `TCNWindowDataset` | `torch.utils.data.Dataset` that returns `[T, F]` float32 tensors by reading one slice at a time from the right memmap. |
| `make_dataloaders(...)` | Build train/val/test `DataLoader`s with shared workers and pinned memory. |
| `compute_class_weights(y_train, mode)` | Inverse-frequency weights for `CrossEntropyLoss` (`balanced` matches sklearn). |

### `src/models/tcn/train.py` — full pipeline

`train(config, tensors_root, output_dir, max_windows=None)` does: `assemble_index → flow_level_split → DataLoaders → model → AdamW + scheduler → epoch loop with AMP → early stopping → persist artifacts`. Returns `{best_epoch, best_val_macro_f1, epochs_run, split_summary, train_seconds, device, n_parameters}`.

### `src/models/tcn/evaluate.py` — held-out evaluation

Loads `tcn.pt` + `dataset_index.npz` + `tcn_config.json`, runs one inference pass over the test split, then computes the same blocks the RF reports:

- **`overall`** — accuracy, per-class precision/recall/F1, macro-F1, ROC-AUC, PR-AUC.
- **`per_adversarial_source`** — `vs_all_benign`, `vs_balanced_benign` (K × n_adv subsample, K=10), `vs_same_base_benign` (only benign from the matching base attack). Variants with fewer than 50 adversarial windows are flagged `low_support`.
- **`per_label_str`** — sanity check: predicted-adversarial rate and mean adversarial probability per `label_str`.

The schema of `metrics.json` matches RF's exactly so a future side-by-side comparison script can ingest both blindly.

## CLI — `scripts/tcn/run_tcn.py`

Orchestrates `train` and/or `evaluate` from `config/tcn.yaml`, with colour-coded progress and summary output (matches the RF runner's style).

### Flags

| Flag | Default | Purpose |
|---|---|---|
| `--config` | `config/tcn.yaml` | Path to TCN config. |
| `--mode` | `both` | `train` / `eval` / `both`. |
| `--max-windows` | `None` | Cap windows per split — smoke-test knob. Skipped when omitted. |
| `--log-level` | `WARNING` | Python logging level; set `INFO` for per-epoch lines. |
| `--json` | off | Print final `overall` metrics as JSON (useful for shell piping). |

### Examples

```bash
# Smoke test (~1 minute on CPU, seconds on GPU)
python3 scripts/tcn/run_tcn.py --max-windows 2000 --mode train

# Full train + evaluate
python3 scripts/tcn/run_tcn.py

# Evaluate a previously-trained model without retraining
python3 scripts/tcn/run_tcn.py --mode eval

# Verbose per-epoch logging
python3 scripts/tcn/run_tcn.py --log-level INFO
```

### Typical end-to-end flow (after Phase 1 + RF)

```bash
# Phase 1 + 2 already done (rf.joblib + tensors present).

# Train + evaluate the TCN
python3 scripts/tcn/run_tcn.py

# Compare against RF
diff <(jq '.overall' data/models/rf/metrics.json) \
     <(jq '.overall' data/models/tcn/metrics.json)
```

## Configuration — `config/tcn.yaml`

| Key | Purpose |
|---|---|
| `tensors_dir` | Root of Phase-1 output. |
| `adversarial_dir` | Root of the adversarial tensor tree. |
| `output_dir` | Where TCN artifacts (model, plots, metrics) are written. |
| `benign_sources` | List of tensor subdir names treated as benign. |
| `adversarial_exclude` | Adversarial subdirs to skip (fuzzing/portscan/syn_dos variants). |
| `benign_cap_ratio` | Global cap on benign windows. |
| `split.{train_frac, val_frac, test_frac, seed}` | Same as `rf.yaml` for split parity. |
| `model.num_features` | Per-packet feature count (1090). |
| `model.channels` | Output channels per `TemporalBlock`. Length = depth. |
| `model.kernel_size` | Shared kernel width. |
| `model.dropout` | Dropout inside each block. |
| `training.device` | `auto` / `cuda` / `cpu`. |
| `training.epochs` | Max epochs (early stopping may end sooner). |
| `training.batch_size` | Mini-batch size. |
| `training.num_workers` | DataLoader workers. |
| `training.lr` / `training.weight_decay` | AdamW hyperparams. |
| `training.scheduler` | `cosine` or `plateau`. |
| `training.early_stop_patience` | Epochs without val_macro_f1 improvement before stopping. |
| `training.amp` | Mixed precision (effective only on CUDA). |
| `training.class_weight` | `balanced` (inverse frequency) or `none`. |

## Output files (`data/models/tcn/`)

| File | Description |
|---|---|
| `tcn.pt` | Best PyTorch `state_dict` (for downstream inference / ensembling). |
| `tcn_config.json` | Resolved model + training config. |
| `dataset_index.npz` | Per-window index — `sources`, `local_idx`, `y`, `flow_ids`, `label_str`, `feature_names`, `source_folders`, plus the `used_*_idx` slices when `--max-windows` is on. |
| `split.npz` | `train_idx`, `val_idx`, `test_idx` into `dataset_index.npz`. Equal element-wise to RF's `split.npz` under the shared seed. |
| `split_summary.json` | Per-split window counts (benign / adversarial / per-source). |
| `training_info.json` | Best epoch, total train seconds, device, effective config. |
| `training_curves.json` | Per-epoch `train_loss`, `val_loss`, `val_macro_f1`, `lr`, `seconds`. |
| `metrics.json` | Overall + per-adversarial-source (× three flavours) + per-`label_str` metrics. |
| `confusion_matrix.png` | Binary CM on the test split. |
| `roc_curve.png` | ROC curve. |
| `pr_curve.png` | Precision-recall curve. |
| `training_curves.png` | Loss + macro-F1 over epochs. |
