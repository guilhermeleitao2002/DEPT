#!/usr/bin/env python3
"""
ADEPT — Mini-Transformer Meta-Learner (train + eval)
====================================================

Trains a small causal Transformer that fuses the per-packet TCN stream
with the per-checkpoint RF stream of the same flow into a single fused
per-packet ``P(adversarial)``. The meta-learner trains on the seed=42
val split (out-of-fold for both base learners) and is evaluated on the
test split.

Usage
-----
    python scripts/metalearner/run_metalearner.py
    python scripts/metalearner/run_metalearner.py --mode train
    python scripts/metalearner/run_metalearner.py --mode eval
    python scripts/metalearner/run_metalearner.py --max-flows 2000   # smoke

Outputs (under <output_dir> from metalearner.yaml)
--------------------------------------------------
    metalearner.pt                             — best PyTorch state_dict
    metalearner_config.json                    — resolved model + training cfg
    split.npz                                  — meta-train / meta-val flow indices
    training_curves.json                       — per-epoch train_loss + meta-val AUROC@10
    metrics.json                               — last-timestep + streaming overall
    streaming_metrics.json                     — full AUROC@t / EDP / per-source
    confusion_matrix.png / roc_curve.png / pr_curve.png
    latency_vs_accuracy.png                    — meta-learner only
    latency_vs_accuracy_comparison.png         — RF + TCN + Meta-Learner overlaid
    cache/{val,test}/                          — memmapped token caches
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import yaml

# Cosmetic PyTorch hint: nested-tensor optimisation can't apply because
# the encoder uses pre-LN (norm_first=True), which we keep for AMP
# stability. Suppressed here only — affects this script's process, not
# the user's interpreter.
warnings.filterwarnings(
    "ignore",
    message=r".*enable_nested_tensor is True, but self\.use_nested_tensor is False.*",
    category=UserWarning,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.models.metalearner import evaluate as ml_eval  # noqa: E402
from src.models.metalearner import train as ml_train    # noqa: E402


_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def bold(t):    return _c("1", t)
def dim(t):     return _c("2", t)
def green(t):   return _c("32", t)
def yellow(t):  return _c("33", t)
def red(t):     return _c("31", t)
def cyan(t):    return _c("36", t)


W = 76


def _section(title: str) -> None:
    print(f"\n{bold('═' * W)}")
    print(f" {bold(title)}")
    print(f"{bold('═' * W)}")


def _step(n: int, total: int, msg: str) -> None:
    print(f"\n{dim('─' * W)}")
    print(f" {bold(cyan(f'[ {n} / {total} ]'))}  {bold(msg)}")
    print(f"{dim('─' * W)}")


def _row(label: str, value: str) -> None:
    print(f" {cyan(label):<20}  {value}")


def _resolve(path_str: str, repo_root: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else repo_root / p


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ADEPT meta-learner — train + evaluate the mini-Transformer",
    )
    parser.add_argument("--config", default="config/metalearner.yaml")
    parser.add_argument("--mode", choices=["train", "eval", "both"], default="both")
    parser.add_argument("--log-level", default="WARNING",
                        help="Python logging level (e.g. INFO, DEBUG)")
    parser.add_argument("--max-flows", type=int, default=None,
                        help="(Reserved) cap meta-train flows for smoke runs")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg_path = _resolve(args.config, REPO_ROOT)
    cfg = yaml.safe_load(cfg_path.read_text())
    output_dir = _resolve(cfg["output_dir"], REPO_ROOT)
    cfg["output_dir"] = str(output_dir)

    # Resolve relative paths to repo-root absolutes for consistency.
    for k in ("tensors_dir", "adversarial_dir", "rf_model_dir", "tcn_model_dir"):
        cfg[k] = str(_resolve(cfg[k], REPO_ROOT))

    _section(f"ADEPT — Meta-Learner  {dim('(mode: ' + args.mode + ')')}")
    _row("Config",        str(cfg_path))
    _row("RF model dir",  cfg["rf_model_dir"])
    _row("TCN model dir", cfg["tcn_model_dir"])
    _row("Output dir",    str(output_dir))
    print(f"{bold('═' * W)}")

    summary: dict = {}
    t0 = time.time()
    if args.mode in ("train", "both"):
        _step(1, 2 if args.mode == "both" else 1, "Training meta-learner")
        t = time.time()
        summary["train"] = ml_train.train(output_dir, cfg)
        print(f"  {green('✓')} Trained in {time.time() - t:.1f}s")
        be = summary["train"]["best_epoch"]
        bv = summary["train"]["best_meta_val_auroc"]
        np_ = summary["train"]["n_parameters"]
        print(f"  {dim('best epoch:')} {be}  "
              f"{dim('meta-val AUROC@10:')} {bv:.4f}  "
              f"{dim('params:')} {np_:,}")

    if args.mode in ("eval", "both"):
        n_steps = 2 if args.mode == "both" else 1
        _step(n_steps, n_steps, "Evaluating on held-out test split")
        t = time.time()
        m = ml_eval.evaluate(output_dir, cfg)
        summary["eval"] = m
        print(f"  {green('✓')} Evaluated in {time.time() - t:.1f}s")
        ov = m["overall"]
        print(f"  {dim('macro F1:')} {ov['macro_f1']:.4f}  "
              f"{dim('ROC AUC:')} {ov['roc_auc']:.4f}  "
              f"{dim('PR AUC:')} {ov['pr_auc']:.4f}  "
              f"{dim('recall_adv:')} {ov['recall_adversarial']:.4f}")
        sm = m["streaming"]
        edp = sm.get("edp", {})
        print(f"  {dim('streaming EDP median:')} {edp.get('edp_median')}  "
              f"{dim('detection rate:')} {edp.get('detection_rate')}")
        for t_key in ("1", "2", "5", "10", "20", "max"):
            entry = sm["auroc_at"].get(t_key, {})
            v = entry.get("auroc")
            if v is not None:
                print(f"    AUROC@{t_key:>4} = {v:.4f}")

    print(f"\n{bold('═' * W)}")
    print(f" {bold('Done')}  {dim(f'Total {time.time() - t0:.1f}s')}")
    print(f" {cyan('Artifacts')}  {output_dir}")
    print(f"{bold('═' * W)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
