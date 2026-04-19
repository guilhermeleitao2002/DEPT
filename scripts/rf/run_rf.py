#!/usr/bin/env python3
"""
ADEPT — Random Forest Base Learner (train + eval)
==================================================

Trains the statistical base learner on the 24-dim window aggregates produced
by the Packet-to-Tensor pipeline and evaluates it on a held-out, flow-level
test split.

Usage
-----
    # Train + evaluate with default config/rf.yaml:
    python scripts/rf/run_rf.py

    # Train only:
    python scripts/rf/run_rf.py --mode train

    # Evaluate an already-trained artifact:
    python scripts/rf/run_rf.py --mode eval

    # Custom config:
    python scripts/rf/run_rf.py --config config/rf.yaml

Outputs (under <output_dir> from rf.yaml)
-----------------------------------------
    rf.joblib              — fitted sklearn RandomForestClassifier
    split.npz              — train/val/test row indices into dataset.npz
    dataset.npz            — merged X, y, attacks, label_str for reproducibility
    feature_names.json     — ordered RF stat names (24)
    split_summary.json     — per-split window counts (benign / attack / per-attack)
    training_info.json     — grid search results, best params, effective config
    metrics.json           — overall, per-attack, per-label_str metrics
    confusion_matrix.png   — binary CM (test)
    roc_curve.png          — ROC (test, when both classes present)
    pr_curve.png           — PR  (test, when both classes present)
    feature_importance.png — Gini importances over the 24 RF features
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

# Make `src` importable without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.models.rf import evaluate as rf_eval  # noqa: E402
from src.models.rf import train as rf_train  # noqa: E402


# ---------------------------------------------------------------------------
# Terminal colours (ANSI — disabled when stdout is not a tty)
# ---------------------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def bold(t):    return _c("1", t)
def dim(t):     return _c("2", t)
def green(t):   return _c("32", t)
def yellow(t):  return _c("33", t)
def red(t):     return _c("31", t)
def cyan(t):    return _c("36", t)
def magenta(t): return _c("35", t)


W = 72


def _rule(char: str = "─") -> str:
    return dim(char * W)


def _section(title: str) -> None:
    print(f"\n{bold('═' * W)}")
    print(f" {bold(title)}")
    print(f"{bold('═' * W)}")


def _step(n: int, total: int, msg: str) -> None:
    print(f"\n{dim(f'[{n}/{total}]')} {bold(msg)}")


def _resolve(path_str: str, repo_root: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else repo_root / p


# ---------------------------------------------------------------------------
# Pretty-printers
# ---------------------------------------------------------------------------

def _fmt_metric(v) -> str:
    if v is None:
        return dim("—")
    return f"{v:.4f}"


def _colour_metric(v, good: float = 0.9, ok: float = 0.75) -> str:
    if v is None:
        return dim("—")
    s = f"{v:.4f}"
    if v >= good:
        return green(s)
    if v >= ok:
        return yellow(s)
    return red(s)


def _print_config_banner(config_path: Path, config: dict,
                          tensors_root: Path, output_dir: Path,
                          mode: str) -> None:
    attacks = config.get("attacks", [])
    gs = (config.get("grid_search") or {})
    grid_on = gs.get("enabled", False)
    _section(f"ADEPT — Random Forest  {dim('(' + mode + ')')}")
    print(f" {cyan('Config')}        {config_path}")
    print(f" {cyan('Tensors dir')}   {tensors_root}")
    print(f" {cyan('Output dir')}    {output_dir}")
    print(f" {cyan('Attacks')}       {len(attacks)}  "
          f"{dim('(' + ', '.join(attacks) + ')')}")
    print(f" {cyan('Benign cap')}    {config.get('benign_cap_ratio')}× attack windows")
    split = config.get("split", {})
    print(f" {cyan('Split')}         "
          f"train={split.get('train_frac')}  "
          f"val={split.get('val_frac')}  "
          f"test={split.get('test_frac')}  "
          f"{dim('seed=' + str(split.get('seed')))}")
    print(f" {cyan('Grid search')}   {green('ON') if grid_on else yellow('OFF')}")
    print(f"{bold('═' * W)}")


def _print_split_summary(summary: dict) -> None:
    hdr = f" {'Split':<8} {'Windows':>12} {'Benign':>12} {'Attack':>12} {'Attack %':>10}"
    print(cyan(hdr))
    print(f" {'-' * 8} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 10}")
    for name in ("train", "val", "test"):
        s = summary[name]
        n = s["n_windows"]
        n_b = s["n_benign"]
        n_a = s["n_attack"]
        pct = (100.0 * n_a / n) if n else 0.0
        print(f" {name:<8} {n:>12,} {n_b:>12,} {n_a:>12,} {pct:>9.1f}%")


def _print_grid_results(training_info_path: Path) -> None:
    if not training_info_path.exists():
        return
    info = json.loads(training_info_path.read_text())
    results = info.get("grid_results") or []
    best = info.get("best_params") or {}
    if not results:
        print(f" {dim('Grid search disabled — using rf_defaults only.')}")
        if best:
            print(f" {cyan('Params')}        {best}")
        return
    # Sort by val score descending, take top 5
    results_sorted = sorted(results, key=lambda r: r["val_macro_f1"], reverse=True)
    top = results_sorted[:5]
    print(f" {cyan('Top configurations')} {dim(f'(of {len(results)})')}")
    hdr = f" {'rank':>4}  {'n_est':>6} {'max_depth':>10} {'min_leaf':>9}  {'val_f1':>9} {'fit_s':>8}"
    print(cyan(hdr))
    print(f" {'-' * 4}  {'-' * 6} {'-' * 10} {'-' * 9}  {'-' * 9} {'-' * 8}")
    for i, r in enumerate(top, 1):
        p = r["params"]
        is_best = (p == best)
        prefix = green("★") if is_best else " "
        row = (f" {i:>4}  {p['n_estimators']:>6} "
               f"{str(p['max_depth']):>10} {p['min_samples_leaf']:>9}  "
               f"{_colour_metric(r['val_macro_f1']):>9} "
               f"{r['fit_seconds']:>8.1f}")
        print(f"{prefix}{row}")
    print(f"\n {cyan('Best params')}   {bold(str(best))}")


def _print_overall_metrics(overall: dict) -> None:
    def row(label: str, v, width: int = 20) -> str:
        return f" {cyan(label + ':'):<{width + 10}} {_colour_metric(v)}"

    print(row("Accuracy", overall.get("accuracy")))
    print(row("Macro F1", overall.get("macro_f1")))
    print(row("ROC AUC", overall.get("roc_auc")))
    print(row("PR AUC", overall.get("pr_auc")))
    print()
    hdr = f" {'class':<8} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}"
    print(cyan(hdr))
    print(f" {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    print(f" {'benign':<8} {_colour_metric(overall.get('precision_benign')):>10} "
          f"{_colour_metric(overall.get('recall_benign')):>10} "
          f"{_colour_metric(overall.get('f1_benign')):>10} "
          f"{overall.get('n_benign', 0):>10,}")
    print(f" {'attack':<8} {_colour_metric(overall.get('precision_attack')):>10} "
          f"{_colour_metric(overall.get('recall_attack')):>10} "
          f"{_colour_metric(overall.get('f1_attack')):>10} "
          f"{overall.get('n_attack', 0):>10,}")


def _print_per_attack_metrics(per_attack: dict) -> None:
    if not per_attack:
        return
    print(f" {dim('precision/recall/f1 computed against a balanced benign subsample (K×n_atk); roc_auc/pr_auc use all benign.')}")
    print()
    hdr = (f" {'attack':<20} {'precision':>10} {'recall':>10} "
           f"{'f1':>10} {'roc_auc':>10} {'pr_auc':>10} {'n_atk':>8}")
    print(cyan(hdr))
    print(f" {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 10} "
          f"{'-' * 10} {'-' * 10} {'-' * 8}")
    low_support_attacks: list[str] = []
    for atk in sorted(per_attack.keys()):
        entry = per_attack[atk]
        bal = entry.get("vs_balanced_benign", {})
        all_bn = entry.get("vs_all_benign", {})
        low = entry.get("low_support", False)
        raw_label = f"{atk} *" if low else atk
        padded = f"{raw_label:<20}"
        if low:
            # Only dim the trailing " *" so width stays visually 20.
            padded = padded.replace(" *", dim(" *"), 1)
            low_support_attacks.append(atk)
        print(f" {padded} "
              f"{_colour_metric(bal.get('precision_attack')):>10} "
              f"{_colour_metric(bal.get('recall_attack')):>10} "
              f"{_colour_metric(bal.get('f1_attack')):>10} "
              f"{_colour_metric(all_bn.get('roc_auc')):>10} "
              f"{_colour_metric(all_bn.get('pr_auc')):>10} "
              f"{all_bn.get('n_attack', 0):>8,}")
    if low_support_attacks:
        print(f"\n {dim('*  low support (n_atk < 50) — metrics are unreliable, treat as noise.')}")


def _print_artifacts(output_dir: Path) -> None:
    artifacts = [
        ("rf.joblib",              "trained classifier"),
        ("split.npz",              "train/val/test indices"),
        ("dataset.npz",            "merged dataset snapshot"),
        ("feature_names.json",     "RF stat names (24)"),
        ("split_summary.json",     "split counts"),
        ("training_info.json",     "grid search + best params"),
        ("metrics.json",           "full metrics (overall/per-attack)"),
        ("confusion_matrix.png",   "binary confusion matrix"),
        ("roc_curve.png",          "ROC curve"),
        ("pr_curve.png",           "precision-recall curve"),
        ("feature_importance.png", "Gini importances"),
    ]
    print(f" {cyan('Artifacts written to')} {output_dir}")
    for name, desc in artifacts:
        p = output_dir / name
        mark = green("✓") if p.exists() else dim("·")
        size = ""
        if p.exists():
            kb = p.stat().st_size / 1024
            size = dim(f" ({kb:,.1f} KB)")
        print(f"   {mark} {name:<24} {dim(desc)}{size}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and/or evaluate the ADEPT Random Forest base learner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config/rf.yaml", help="Path to rf.yaml")
    parser.add_argument("--mode", choices=["train", "eval", "both"], default="both")
    parser.add_argument("--log-level", default="WARNING",
                        help="Python logging level (default: WARNING — keeps terminal clean)")
    parser.add_argument("--json", action="store_true",
                        help="Also print overall metrics as JSON at the end")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config_path = _resolve(args.config, REPO_ROOT)
    if not config_path.exists():
        print(f"{red(bold('ERROR:'))} config not found: {config_path}")
        return 1
    config = yaml.safe_load(config_path.read_text())
    tensors_root = _resolve(config["tensors_dir"], REPO_ROOT)
    output_dir = _resolve(config["output_dir"], REPO_ROOT)

    _print_config_banner(config_path, config, tensors_root, output_dir, args.mode)

    total_t0 = time.time()

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    train_seconds = None
    if args.mode in ("train", "both"):
        _step(1, 2 if args.mode == "both" else 1, "Training Random Forest")
        t0 = time.time()
        try:
            info = rf_train.train(config, tensors_root, output_dir)
        except Exception as e:
            print(f"  {red(bold('✗ FAILED'))} {e}")
            return 2
        train_seconds = time.time() - t0
        extra = f"  {dim(f'({train_seconds / 60:.1f} min)')}" if train_seconds >= 60 else ""
        print(f"  {green('✓')} Trained in {train_seconds:.1f}s{extra}")

        print(f"\n {bold('Split summary')}")
        _print_split_summary(info["split_summary"])

        print(f"\n {bold('Grid search')}")
        _print_grid_results(output_dir / "training_info.json")

    # -----------------------------------------------------------------------
    # Evaluate
    # -----------------------------------------------------------------------
    eval_seconds = None
    metrics = None
    if args.mode in ("eval", "both"):
        step_idx = 2 if args.mode == "both" else 1
        step_total = 2 if args.mode == "both" else 1
        _step(step_idx, step_total, "Evaluating on held-out test split")
        t0 = time.time()
        try:
            metrics = rf_eval.evaluate(output_dir)
        except FileNotFoundError as e:
            print(f"  {red(bold('✗ FAILED'))} missing artifact: {e}. "
                  f"Run {bold('--mode train')} first.")
            return 2
        except Exception as e:
            print(f"  {red(bold('✗ FAILED'))} {e}")
            return 2
        eval_seconds = time.time() - t0
        print(f"  {green('✓')} Evaluated in {eval_seconds:.1f}s")

        print(f"\n {bold('Overall metrics (test)')}")
        _print_overall_metrics(metrics["overall"])

        print(f"\n {bold('Per-attack metrics')}  "
              f"{dim('(each attack vs. all test benign)')}")
        _print_per_attack_metrics(metrics.get("per_attack", {}))

    # -----------------------------------------------------------------------
    # Wrap-up
    # -----------------------------------------------------------------------
    total_elapsed = time.time() - total_t0
    _section("Summary")
    if train_seconds is not None:
        print(f" {cyan('Train time')}    {train_seconds:.1f}s "
              f"{dim('(' + f'{train_seconds / 60:.1f} min' + ')')}")
    if eval_seconds is not None:
        print(f" {cyan('Eval time')}     {eval_seconds:.1f}s")
    print(f" {cyan('Total time')}    {total_elapsed:.1f}s "
          f"{dim('(' + f'{total_elapsed / 60:.1f} min' + ')')}")
    print()
    _print_artifacts(output_dir)
    print(f"{bold('═' * W)}\n")

    if args.json and metrics is not None:
        print(json.dumps(metrics["overall"], indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
