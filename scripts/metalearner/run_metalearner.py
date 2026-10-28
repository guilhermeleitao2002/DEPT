#!/usr/bin/env python3
"""
DEPT — Mini-Transformer Meta-Learner (train + eval)
====================================================

Trains a small causal transformer that fuses the per-packet TCN stream and the
per-checkpoint RF stream into one per-packet ``P(adversarial)``. It is trained
on the base learners' val split and evaluated on the test split.

Usage
-----
    python scripts/metalearner/run_metalearner.py
    python scripts/metalearner/run_metalearner.py --mode train
    python scripts/metalearner/run_metalearner.py --mode eval
    python scripts/metalearner/run_metalearner.py --attack mirai

With --attack it stacks the per-attack RF and TCN (train those first) and
writes to ``data/models/per_attack/<attack>/metalearner``.

Outputs (under <output_dir> from metalearner.yaml)
--------------------------------------------------
    metalearner.pt                        — best state_dict
    metalearner_config.json               — resolved model + training config
    split.npz                             — meta-train / meta-val flow indices
    training_curves.json                  — per-epoch train_loss + meta-val AUROC@10
    metrics.json                          — last-packet + streaming overall
    streaming_metrics.json                — AUROC@t / EDP / per source
    confusion_matrix.png / roc_curve.png / pr_curve.png
    earliness_vs_accuracy.png             — meta-learner only
    earliness_vs_accuracy_comparison.png  — RF + TCN + meta-learner
    cache/{val,test}/                     — memmapped token caches
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

import yaml

# PyTorch warns that the nested-tensor fast path is off with pre-LN layers.
warnings.filterwarnings(
    "ignore",
    message=r".*enable_nested_tensor is True, but self\.use_nested_tensor is False.*",
    category=UserWarning,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.models.common.attack_scope import per_attack_models_root  # noqa: E402
from src.models.metalearner import evaluate as ml_eval  # noqa: E402
from src.models.metalearner import train as ml_train    # noqa: E402


# The meta-learner is driven entirely by this config.
CONFIG_PATH = "config/metalearner.yaml"


_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def bold(t):    return _c("1", t)
def dim(t):     return _c("2", t)
def green(t):   return _c("32", t)
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
        description="DEPT meta-learner — train + evaluate the mini-Transformer",
    )
    parser.add_argument("--mode", choices=["train", "eval", "both"], default="both")
    parser.add_argument(
        "--attack", default=None, metavar="NAME",
        help="stack the per-attack base learners from "
             "data/models/per_attack/<NAME>/{rf,tcn}; artifacts go to "
             "data/models/per_attack/<NAME>/metalearner")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg_path = _resolve(CONFIG_PATH, REPO_ROOT)
    cfg = yaml.safe_load(cfg_path.read_text())
    output_dir = _resolve(cfg["output_dir"], REPO_ROOT)
    cfg["output_dir"] = str(output_dir)

    # Resolve relative paths to repo-root absolutes for consistency.
    for k in ("tensors_dir", "adversarial_dir", "rf_model_dir", "tcn_model_dir"):
        cfg[k] = str(_resolve(cfg[k], REPO_ROOT))

    if args.attack:
        root = per_attack_models_root(REPO_ROOT, args.attack)
        cfg["rf_model_dir"] = str(root / "rf")
        cfg["tcn_model_dir"] = str(root / "tcn")
        output_dir = root / "metalearner"
        cfg["output_dir"] = str(output_dir)
        missing = [p for p in (root / "rf" / "rf.joblib", root / "tcn" / "tcn.pt")
                   if not p.exists()]
        if missing:
            for p in missing:
                print(f"ERROR: per-attack base learner missing: {p}")
            print(f"Train it first: run_rf.py / run_tcn.py --attack {args.attack}")
            return 1

    _section(f"DEPT — Meta-Learner  {dim('(mode: ' + args.mode + ')')}")
    _row("Config",        str(cfg_path))
    if args.attack:
        _row("Attack scope", f"{bold(args.attack)}  "
                             f"{dim('(per-attack run)')}")
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
