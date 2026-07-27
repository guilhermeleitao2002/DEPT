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
| `earliness_vs_accuracy.png` | per-source AUROC@t curves |
| `earliness_vs_accuracy_comparison.png` | RF + TCN + Meta-Learner overlaid |
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

## Results snapshot (final global run)

Best epoch 17/20 (meta-val AUROC@10 0.9333), 67,971 parameters, 3.4 h train +
0.2 h eval on one GPU. Test split (1,293,737 flows): ROC AUC 0.9334,
PR AUC 0.9866, macro F1 0.6988, precision-adv 0.9985 (1,115 benign false
positives out of 211,151), recall-adv 0.7076. Streaming: AUROC@1 0.8373 /
@2 0.9099 / @10 0.9323 / @max 0.9334, EDP median 1 packet, detection rate
(p>0.5) 77.69%. The meta-learner beats the TCN at every horizon and is the
only stream with a packet-1 verdict, but from packet 2 onward the RF stream
alone is stronger (0.9805–0.9813) — see the root README for the full
comparison and the gradient-ablation study for what the stack buys in
robustness terms.
