# Mini-Transformer Meta-Learner

Trains and evaluates the small causal Transformer that fuses the per-packet
TCN stream with the per-checkpoint RF stream of the same flow into a single
fused per-packet `P(adversarial)`.

## Quick start

```bash
# Train + evaluate using config/metalearner.yaml
python3 scripts/metalearner/run_metalearner.py

# Train only / evaluate only
python3 scripts/metalearner/run_metalearner.py --mode train
python3 scripts/metalearner/run_metalearner.py --mode eval
```

The meta-learner consumes the trained RF (`data/models/rf/rf.joblib`,
`dataset.npz`, `split.npz`) and the trained TCN (`data/models/tcn/tcn.pt`,
`dataset_index.npz`) directly — make sure both have been trained first
(`scripts/rf/run_rf.py`, `scripts/tcn/run_tcn.py`).

## What gets written

Under `data/models/metalearner/`:

| Path | Purpose |
|---|---|
| `metalearner.pt` | best PyTorch state_dict (selected by meta-val AUROC@10) |
| `metalearner_config.json` | resolved model + training config + best epoch |
| `split.npz` | meta-train / meta-val flow indices (sub-split of seed=42 val) |
| `training_curves.json` | per-epoch train_loss + meta-val AUROC@10 |
| `metrics.json` | last-timestep verdict + streaming overall |
| `streaming_metrics.json` | full AUROC@t / EDP / per-source slicing |
| `confusion_matrix.png`, `roc_curve.png`, `pr_curve.png` | last-timestep plots |
| `latency_vs_accuracy.png` | per-source AUROC@t curves |
| `latency_vs_accuracy_comparison.png` | RF + TCN + Meta-Learner overlaid |
| `cache/{val,test}/` | memmapped per-split token caches (auto-rebuilt when base learners change) |

## How the data flows

1. `prepare_split_caches` runs both base learners' `predict_streams()` on the
   shared seed=42 val and test splits, joins per `(source, flow_id_str)`, and
   materialises a per-packet `[L, 13]` token matrix per flow under
   `data/models/metalearner/cache/<split>/tokens.npy`.
2. `MetaLearnerDataset` memmaps the cache and serves variable-length token
   sequences via `collate_padded`.
3. The training loop trains on a 0.85/0.15 sub-split of val (seed=43); meta-val
   streaming AUROC@10 drives early stopping (patience=5).
4. Final evaluation runs on the untouched test split and produces the standard
   streaming-eval artefacts plus a three-way comparison plot.

The cache is invalidated automatically when `rf.joblib` or `tcn.pt` changes
(mtime + size proxy in `cache/cache_key.txt`).
