# Phase 2 — Random Forest Base Learner

> Part of [ADEPT](../../README.md). The statistical half of the ADEPT ensemble — a binary benign-vs-adversarial classifier trained on the 24-dim window aggregates produced by [Phase 1](../packet_to_tensor/README.md). Deliberately paired with the (planned) TCN base learner, which consumes the raw sequence tensor, to maximise representational diversity.

## Role in the ensemble

Under ADEPT's [dataset-driven label convention](../../README.md#dataset-driven-label-convention), the RF learns to distinguish:

- the **benign manifold** — every window from the Kitsune + CIC-IDS2017 tensor folders listed in `benign_sources` (the raw-dataset attacks count as benign because the upstream NIDS is expected to handle them);
- **adversarial perturbations** — every window from the TANTGen-generated subfolders auto-discovered under `adversarial_dir`.

This narrows ADEPT's specific job to catching **novel TANTGen-style threats** the upstream NIDS cannot, and means every window has a definite label with no "unknown" rows to drop.

## Design

| Aspect | Choice | Rationale |
|---|---|---|
| **Input** | `rf_stats.npy` — `[W, 24]` float32 | Uses only the aggregated per-window statistics, never the raw packet sequence — keeps RF and TCN representations disjoint. |
| **Labels** | `window_labels.csv` written by `attach_labels.py` | `0` = benign (Kitsune / CIC-IDS2017 folder), `1` = adversarial (TANTGen folder). Dataset-driven, so every window has a definite label. |
| **Benign pool** | `benign_sources` in `rf.yaml` | All Kitsune + CIC-IDS2017 tensor folders (raw-dataset attacks + Monday benign) — every one of their windows is a negative sample. |
| **Adversarial pool** | auto-discovered under `adversarial_dir` | Every subdirectory with a `metadata.json` is loaded as an adversarial source (one per TANTGen variant, e.g. `mirai_adv_both`). |
| **Split** | **Flow-level, not window-level** | Windows from the same 5-tuple flow stay in the same split — prevents leakage via flow-internal correlation. |
| **Stratification** | Per-source folder | Every source contributes a single class, so stratifying by source automatically stratifies by class while keeping each source's window fraction near its target in every split. Flows are bin-packed largest-first into the split with the greatest deficit. |
| **Benign capping** | Total benign windows ≤ `benign_cap_ratio` × adversarial windows | Applied globally (flow-atomic, across all benign sources) so the much larger raw-dataset pool doesn't dominate training. |
| **Hyperparameters** | Small grid (val macro-F1) | `n_estimators`, `max_depth`, `min_samples_leaf` — grid overrides feed into `rf_defaults`. |
| **Final fit** | `train + val`, evaluate on `test` | No data is wasted before the held-out evaluation. |

## Module reference

### `src/models/rf/dataset.py` — load, merge, split

| Function | Purpose |
|---|---|
| `load_source(folder, source_name)` | Read `rf_stats.npy`, `flow_ids.npy`, `window_labels.csv`, `metadata.json` for one source folder; drop `unknown` rows; return `(X, y, flow_ids, label_str, feature_names)`. |
| `discover_adversarial_sources(adversarial_dir)` | Return every subdirectory name under `adversarial_dir` that contains a `metadata.json`. |
| `cap_benign(y, scoped_flow_ids, ratio, rng)` | Drop whole benign flows (not individual windows) until total benign windows ≤ `ratio × n_adversarial`. Flow IDs are scoped by source so benign flows from different folders never collide. |
| `assemble(tensors_dir, benign_sources, adversarial_dir, benign_cap_ratio, seed)` | Load every benign source + every auto-discovered adversarial source into one `RFDataset` (`X, y, flow_ids, sources, label_str, feature_names`), then apply global benign capping. NaN/inf values in `X` are replaced with zero. |
| `flow_level_split(dataset, train/val/test_frac, seed)` | Stratified, flow-atomic split by source; bin-packs flows largest-first into the split with the greatest window-count deficit so totals land near targets. Asserts no flow leakage across splits. |
| `split_summary(dataset, split)` | Per-split counts: total, benign, adversarial, per-source. |

The dataclasses returned:

| Dataclass | Fields |
|---|---|
| `RFDataset` | `X` `[N,F]` float32, `y` `[N]` int8, `flow_ids` `[N]` str, `sources` `[N]` str, `label_str` `[N]` str, `feature_names: list[str]` |
| `Split` | `train_idx`, `val_idx`, `test_idx` — `int64` arrays into `RFDataset` rows |

### `src/models/rf/train.py` — grid search + final fit

| Function | Purpose |
|---|---|
| `grid_search(dataset, split, defaults, grid_cfg)` | Fit each `(n_estimators × max_depth × min_samples_leaf)` combination on `train`, score macro-F1 on `val`. Returns `(best_params, all_results)` where each result has `params`, `val_macro_f1`, `fit_seconds`. |
| `fit_final(dataset, split, defaults, best_params)` | Re-fit on `train + val` using the best params; produces the deployable classifier. |
| `train(config, tensors_root, output_dir)` | Full pipeline: `assemble → split → grid_search → fit_final → persist artifacts`. Returns `{best_params, split_summary}`. |

### `src/models/rf/evaluate.py` — held-out evaluation

Runs the fitted `rf.joblib` on the test split and produces both machine-readable metrics (`metrics.json`) and inspection plots (confusion matrix, ROC, PR, feature importance).

Per-adversarial-source metrics are reported in **three flavours** so the reader can disambiguate deployment signal from structural size penalties and from cross-capture confounds:

- **`vs_all_benign`** — this variant's windows + **all benign windows in the test set**. Reflects realistic prevalence, but precision is dominated by the total benign FP count.
- **`vs_balanced_benign`** — this variant's windows + a `K × n_adv` benign subsample (capped at total benign, K=10). Makes precision comparable across variants of different sizes.
- **`vs_same_base_benign`** — this variant's windows + only benign from the same base-attack capture (e.g. `mirai_adv_both` → benign windows from the `mirai/` folder). Removes cross-capture confounds.

Variants with fewer than 50 adversarial windows are flagged as `low_support` — their metrics should be treated as noise.

A `per_label_str` block reports the predicted-adversarial rate and mean adversarial probability for each `label_str` value present in the test set; under the dataset-driven convention this is just a sanity check (benign vs adversarial).

## CLI — `scripts/rf/run_rf.py`

Orchestrates `train` and/or `evaluate` from `config/rf.yaml`, with colour-coded progress and summary output.

### Flags

| Flag | Default | Purpose |
|---|---|---|
| `--config` | `config/rf.yaml` | Path to RF config. |
| `--mode` | `both` | `train` / `eval` / `both`. |
| `--log-level` | `WARNING` | Python logging level; set `INFO` for the library's own log messages. |
| `--json` | off | Print final `overall` metrics as JSON (useful for shell piping). |

### Examples

```bash
# Train and evaluate with the default config
python3 scripts/rf/run_rf.py

# Train only (skip eval)
python3 scripts/rf/run_rf.py --mode train

# Evaluate a previously-trained artifact without retraining
python3 scripts/rf/run_rf.py --mode eval

# Use a custom config
python3 scripts/rf/run_rf.py --config config/rf_experiment.yaml
```

### Typical end-to-end flow

```bash
# 1. Generate the TANTGen adversarial PCAPs
python3 scripts/adversarial_generation/generate_adversarials.py

# 2. Tensorise the 5 attacks whose adversarial PCAPs exist on disk + variants
python3 scripts/packet_to_tensor/process_all_attacks.py --tantgen

# 3. Train + evaluate the RF
python3 scripts/rf/run_rf.py
```

## Configuration — `config/rf.yaml`

| Key | Purpose |
|---|---|
| `tensors_dir` | Root of Phase-1 output (expects subdirs matching `benign_sources`). |
| `adversarial_dir` | Root of the adversarial tensor tree; every subdir with `metadata.json` is auto-discovered as a positive source. |
| `output_dir` | Where RF artifacts (model, plots, metrics) are written. |
| `benign_sources` | List of tensor subdir names (Kitsune + CIC-IDS2017) treated as benign. |
| `benign_cap_ratio` | Global cap on benign windows as a multiple of adversarial windows. |
| `drop_unknown` | Safety net — should never fire since `attach_labels.py` only writes `benign` / `adversarial`. |
| `split` | `train_frac` / `val_frac` / `test_frac` / `seed`. |
| `rf_defaults` | sklearn `RandomForestClassifier` kwargs applied to every fit (`n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`, `n_jobs`, `class_weight`, `random_state`). |
| `grid_search.enabled` | Toggle grid search on/off. |
| `grid_search.{n_estimators, max_depth, min_samples_leaf}` | Candidate values — Cartesian product, scored by val macro-F1. |
| `grid_search.scoring` | Currently fixed to `macro_f1` (informational). |

The default benign pool is restricted to the **5 attacks for which TANTGen adversarial PCAPs exist on disk** (`brute_force`, `ddos`, `mirai`, `portscan`, `ssdp_flood`). Other Kitsune + CIC-IDS2017 captures are excluded because they have no matching adversarial counterpart and would only inflate the benign class. Under this setup the adversarial pool yields **15 sources** (5 attacks × 3 perturbation modes: `both`, `iat`, `size`).

## Output files (`data/models/rf/`)

| File | Description |
|---|---|
| `rf.joblib` | Fitted `RandomForestClassifier` (for downstream inference / ensembling). |
| `dataset.npz` | Merged `(X, y, sources, label_str)` — lets `evaluate` run without re-assembling. |
| `split.npz` | `train_idx`, `val_idx`, `test_idx` into `dataset.npz`. |
| `feature_names.json` | Ordered list of 24 RF stat names. |
| `split_summary.json` | Per-split window counts (benign / adversarial, per-source). |
| `training_info.json` | Grid-search results, best params, effective config (minus `rf_defaults`). |
| `metrics.json` | Overall + per-adversarial-source (× three flavours) + per-`label_str` metrics. |
| `confusion_matrix.png` | Binary CM on the test split (benign vs adversarial). |
| `roc_curve.png` | ROC curve (when both classes present in test). |
| `pr_curve.png` | Precision-recall curve (when both classes present in test). |
| `feature_importance.png` | Gini importances over the 24 RF features. |

`run.log` is also produced when `run_rf.py` is invoked with output redirected — useful for archiving training console output.
