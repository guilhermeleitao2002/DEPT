# Mini-Transformer Meta-Learner

> Part of [DEPT](../../README.md), **phase 5**. A small causal transformer that combines the TCN's per-packet stream and the RF's per-checkpoint stream into one per-packet `P(adversarial)`.

```bash
python3 scripts/metalearner/run_metalearner.py            # train + evaluate
python3 scripts/metalearner/run_metalearner.py --mode eval
```

The RF and TCN must be trained first (`data/models/rf/`, `data/models/tcn/`).

## Method

Each packet becomes a **13-dim token**: the last RF logit (carried forward), the TCN logit, `rf_age`, `is_rf_checkpoint`, IPAT, direction, the IPv4 length and six TCP flag bits. Before the first RF checkpoint, the RF slot holds a learned `rf_unknown_logit`. **67,971 parameters**: 2 encoder layers, 4 heads, `d_model=64`, causal mask, plus an auxiliary loss that penalises being less confident than the better base learner.

The base learners are run on the val and test splits and their outputs are cached as per-flow tokens (the cache is rebuilt when `rf.joblib` or `tcn.pt` change). Training uses a 0.85/0.15 split of val (seed 43), with meta-val AUROC@10 for early stopping; the test split is only used for the final evaluation.

## Results (test: 1,293,737 flows)

Best epoch 17/20 (meta-val AUROC@10 0.9333), 3.4 h training + 0.2 h evaluation on one GPU.

| Last packet | | Streaming | |
|---|---|---|---|
| ROC AUC | 0.9334 | AUROC@1 | **0.8373** |
| PR AUC | 0.9866 | AUROC@2 / @10 | 0.9099 / 0.9323 |
| Macro F1 | 0.6988 | AUROC@max | 0.9334 |
| Precision (adv) | **0.9985** | EDP median | **1 packet** |
| Recall (adv) | 0.7076 | Detection rate (p>0.5) | 77.69% |

The meta-learner is better than the TCN at every packet and gives a verdict at packet 1, but from packet 2 on the RF alone is better (0.9805–0.9813). Only 1,115 of 211,151 benign test flows are flagged (FPR 0.53%).

So the stack does not improve on the RF's AUROC. It gives a verdict one packet earlier and is harder to evade with a gradient-guided attack (74.5% of flows never evaded vs 45.5% for the TCN alone, see [`scripts/gradient_ablation/`](../gradient_ablation/README.md)).

Artifacts in `data/models/metalearner/`: `metalearner.pt`, `split.npz`, `metrics.json`, `streaming_metrics.json`, `training_curves.json`, the token caches, and the plots, including `earliness_vs_accuracy_comparison.png` (RF, TCN and meta-learner together).
