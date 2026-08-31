# Mini-Transformer Meta-Learner

> Part of [DEPT](../../README.md) — **phase 5**. The fusion layer: a small causal Transformer that merges the TCN's per-packet stream with the RF's per-checkpoint stream into a single per-packet `P(adversarial)`.

```bash
python3 scripts/metalearner/run_metalearner.py            # train + evaluate
python3 scripts/metalearner/run_metalearner.py --mode eval
```

Requires both base learners trained first (`data/models/rf/`, `data/models/tcn/`).

## What it does

Each packet becomes a **13-dim token**: the carry-forward RF logit, the dense TCN logit, four "where are we in the flow" features (`rf_age`, `is_rf_checkpoint`, IPAT, direction) and a compact packet summary (IPv4 length + six TCP flag bits) — deliberately *not* the full 1088-bit nPrint vector. A learnable `rf_unknown_logit` fills the RF slot before its first checkpoint. **67,971 parameters**: 2 encoder layers, 4 heads, `d_model=64`, strict causal mask, plus an auxiliary "beat the best base learner" loss.

Both base learners' `predict_streams()` are run on the shared seed=42 val and test splits and memmapped as per-flow token caches (invalidated automatically when `rf.joblib` or `tcn.pt` changes). Training uses a 0.85/0.15 sub-split of val (seed=43 — out-of-fold for both base learners) with meta-val AUROC@10 driving early stopping; final evaluation runs on the untouched test split.

## Results (test: 1,293,737 flows)

Best epoch 17/20 (meta-val AUROC@10 0.9333), 3.4 h train + 0.2 h eval on one GPU.

| Last timestep | | Streaming | |
|---|---|---|---|
| ROC AUC | 0.9334 | AUROC@1 | **0.8373** |
| PR AUC | 0.9866 | AUROC@2 / @10 | 0.9099 / 0.9323 |
| Macro F1 | 0.6988 | AUROC@max | 0.9334 |
| Precision (adv) | **0.9985** | EDP median | **1 packet** |
| Recall (adv) | 0.7076 | Detection rate (p>0.5) | 77.69% |

The meta-learner **beats the TCN at every horizon and is the only stream with a packet-1 verdict**, but from packet 2 onward the RF alone is stronger (0.9805–0.9813). Its confusion profile is precision-first — only 1,115 of 211,151 benign test flows flagged (FPR 0.53%).

So what the stack buys is not raw AUROC: it is a verdict one packet earlier, plus robustness against gradient-guided evasion (74.5% never-evaded vs 45.5% for the TCN alone — see [`scripts/gradient_ablation/`](../gradient_ablation/README.md)).

Artifacts under `data/models/metalearner/`: `metalearner.pt`, `split.npz`, `metrics.json`, `streaming_metrics.json`, `training_curves.json`, the token caches, and plots including `earliness_vs_accuracy_comparison.png` (RF + TCN + meta overlaid).
