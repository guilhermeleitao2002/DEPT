#!/usr/bin/env python3
"""
DEPT — Random Forest Base Learner (train + eval)
==================================================

Trains the statistical base learner on the 24-dim window aggregates produced
by the Packet-to-Tensor pipeline and evaluates it on a held-out, flow-level
test split. Under DEPT's benign-vs-adversarial label convention, the
benign pool is every window from the Kitsune + CIC-IDS2017 folders listed
as `benign_sources` in rf.yaml, and the adversarial pool is every TANTGen-
generated subfolder auto-discovered under `adversarial_dir`.

Usage
-----
    # Train + evaluate (default):
    python scripts/rf/run_rf.py

    # Train only:
    python scripts/rf/run_rf.py --mode train

    # Evaluate an already-trained artifact:
    python scripts/rf/run_rf.py --mode eval

Reads config/rf.yaml and force-overwrites every artifact under <output_dir>.

Outputs (under <output_dir> from rf.yaml)
-----------------------------------------
    rf.joblib              — fitted sklearn RandomForestClassifier
    split.npz              — train/val/test row indices into dataset.npz
    dataset.npz            — merged X, y, sources, label_str for reproducibility
    feature_names.json     — ordered RF stat names (24)
    split_summary.json     — per-split window counts (benign / adversarial / per-source)
    training_info.json     — grid search results, best params, effective config
    metrics.json           — overall, per-adversarial-source, per-label_str metrics
    confusion_matrix.png   — binary CM (test, benign vs adversarial)
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


W = 76


def _rule(char: str = "─") -> str:
    return dim(char * W)


def _section(title: str) -> None:
    print(f"\n{bold('═' * W)}")
    print(f" {bold(title)}")
    print(f"{bold('═' * W)}")


def _step(n: int, total: int, msg: str) -> None:
    """Rule-bracketed step header, matching process_all_attacks' style."""
    print(f"\n{dim('─' * W)}")
    print(f" {bold(cyan(f'[ {n} / {total} ]'))}  {bold(msg)}")
    print(f"{dim('─' * W)}")


def _subhead(title: str) -> None:
    """A smaller in-step heading with a single dim rule below it."""
    print(f"\n {bold(title)}")
    print(f" {dim('─' * (W - 1))}")


def _resolve(path_str: str, repo_root: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else repo_root / p


def _fmt_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:,.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:,.1f} GB"


# ---------------------------------------------------------------------------
# Pretty-printers
# ---------------------------------------------------------------------------

def _fmt_metric(v) -> str:
    if v is None:
        return dim("—")
    return f"{v:.4f}"


def _colour_metric(v, good: float = 0.9, ok: float = 0.75, width: int = 0) -> str:
    if v is None:
        s = "—".rjust(width) if width else "—"
        return dim(s)
    s = f"{v:.4f}"
    if width:
        s = s.rjust(width)
    if v >= good:
        return green(s)
    if v >= ok:
        return yellow(s)
    return red(s)


def _print_config_banner(config_path: Path, config: dict,
                          tensors_root: Path, output_dir: Path,
                          mode: str) -> None:
    benign_sources = config.get("benign_sources", [])
    adv_dir = Path(config.get("adversarial_dir", "data/tensors/adversarial"))
    adv_names: list[str] = []
    if adv_dir.exists():
        adv_names = sorted(
            d.name for d in adv_dir.iterdir()
            if d.is_dir() and (d / "metadata.json").exists()
        )
    gs = (config.get("grid_search") or {})
    grid_on = gs.get("enabled", False)
    sp = config.get("split", {})

    _section(f"DEPT — Random Forest  {dim('(mode: ' + mode + ')')}")
    label_w = 13

    def _row(label: str, value: str) -> None:
        print(f" {cyan(f'{label:<{label_w}}')} {value}")

    _row("Config",       str(config_path))
    _row("Tensors dir",  str(tensors_root))
    _row("Adversarial",  f"{adv_dir}  "
                         f"{dim('(' + str(len(adv_names)) + ' source(s) discovered)')}")
    _row("Output dir",   str(output_dir))
    _row("Benign pool",  f"{bold(str(len(benign_sources)))} source(s)  "
                         f"{dim('(' + ', '.join(benign_sources) + ')')}")
    _row("Benign cap",   f"{bold(str(config.get('benign_cap_ratio')))}× adversarial windows")
    _row("Split",        f"train={sp.get('train_frac')}  "
                         f"val={sp.get('val_frac')}  "
                         f"test={sp.get('test_frac')}  "
                         f"{dim('seed=' + str(sp.get('seed')))}")
    _row("Grid search",  green(bold('ON')) if grid_on else yellow('OFF'))
    print(f"{bold('═' * W)}")


def _print_split_summary(summary: dict) -> None:
    _subhead("Split summary  (rows = RF checkpoints, multi-scale prefixes)")
    hdr = (f" {'Split':<8} {'Ckpts':>12} {'Flows':>10} {'Benign':>12} "
           f"{'Adversarial':>13} {'Adv %':>8}")
    print(cyan(hdr))
    print(f" {dim('-' * 8)} {dim('-' * 12)} {dim('-' * 10)} {dim('-' * 12)} "
          f"{dim('-' * 13)} {dim('-' * 8)}")

    totals = {"n_checkpoints": 0, "n_flows": 0, "n_benign": 0, "n_adversarial": 0}
    for name in ("train", "val", "test"):
        s = summary[name]
        n = s["n_checkpoints"]
        nf = s.get("n_flows", 0)
        n_b = s["n_benign"]
        n_a = s["n_adversarial"]
        pct = (100.0 * n_a / n) if n else 0.0
        totals["n_checkpoints"] += n
        totals["n_flows"] += nf
        totals["n_benign"] += n_b
        totals["n_adversarial"] += n_a
        print(f" {name:<8} {n:>12,} {dim(f'{nf:>10,}')} "
              f"{dim(f'{n_b:>12,}')} "
              f"{green(f'{n_a:>13,}')} {yellow(f'{pct:>7.1f}%')}")

    print(f" {dim('-' * 8)} {dim('-' * 12)} {dim('-' * 10)} {dim('-' * 12)} "
          f"{dim('-' * 13)} {dim('-' * 8)}")
    tn = totals["n_checkpoints"]
    tf = totals["n_flows"]
    tb = totals["n_benign"]
    ta = totals["n_adversarial"]
    pct = (100.0 * ta / tn) if tn else 0.0
    total_label = f"{'total':<8}"
    tn_str = f"{tn:>12,}"
    tf_str = f"{tf:>10,}"
    tb_str = f"{tb:>12,}"
    ta_str = f"{ta:>13,}"
    pct_str = f"{pct:>7.1f}%"
    print(f" {bold(total_label)} {bold(tn_str)} {dim(tf_str)} "
          f"{dim(tb_str)} {green(bold(ta_str))} {yellow(pct_str)}")


def _print_grid_results(training_info_path: Path) -> None:
    if not training_info_path.exists():
        return
    info = json.loads(training_info_path.read_text())
    results = info.get("grid_results") or []
    best = info.get("best_params") or {}

    _subhead("Grid search")
    if not results:
        print(f" {dim('Disabled — using rf_defaults only.')}")
        if best:
            print(f" {cyan('Params')}        {bold(str(best))}")
        return

    # Sort by val score descending, take top 5
    results_sorted = sorted(results, key=lambda r: r["val_macro_f1"], reverse=True)
    top = results_sorted[:5]
    print(f" {dim(f'Top {len(top)} of {len(results)} configuration(s):')}")
    print()
    hdr = (f"  {'':<2}{'rank':>4}  {'n_est':>6} {'max_depth':>10} "
           f"{'min_leaf':>9}  {'val_f1':>9}  {'fit_s':>8}")
    print(cyan(hdr))
    print(f"  {'':<2}{dim('-' * 4)}  {dim('-' * 6)} {dim('-' * 10)} "
          f"{dim('-' * 9)}  {dim('-' * 9)}  {dim('-' * 8)}")
    for i, r in enumerate(top, 1):
        p = r["params"]
        is_best = (p == best)
        marker = green(bold("★ ")) if is_best else "  "
        fit_s = r["fit_seconds"]
        fit_str = f"{fit_s:>8.1f}"
        row = (f"{i:>4}  {p['n_estimators']:>6} "
               f"{str(p['max_depth']):>10} {p['min_samples_leaf']:>9}  "
               f"{_colour_metric(r['val_macro_f1'], width=9)}  "
               f"{dim(fit_str)}")
        print(f"  {marker}{row}")
    print(f"\n {cyan('Best params')}  {bold(str(best))}")


def _print_overall_metrics(overall: dict) -> None:
    _subhead("Overall metrics (test)")

    # Headline row: the four scalar metrics, side-by-side.
    def _pill(label: str, v) -> str:
        return f"{cyan(label)} {_colour_metric(v)}"

    line = "   ".join([
        _pill("Acc", overall.get("accuracy")),
        _pill("Macro F1", overall.get("macro_f1")),
        _pill("ROC AUC", overall.get("roc_auc")),
        _pill("PR AUC", overall.get("pr_auc")),
    ])
    print(f" {line}\n")

    # Per-class breakdown.
    hdr = (f" {'class':<12} {'precision':>10} {'recall':>10} {'f1':>10} "
           f"{'support':>10}")
    print(cyan(hdr))
    print(f" {dim('-' * 12)} {dim('-' * 10)} {dim('-' * 10)} "
          f"{dim('-' * 10)} {dim('-' * 10)}")
    n_b = overall.get("n_benign", 0)
    n_a = overall.get("n_adversarial", 0)
    n_b_str = f"{n_b:>10,}"
    n_a_str = f"{n_a:>10,}"
    print(f" {'benign':<12} "
          f"{_colour_metric(overall.get('precision_benign'), width=10)} "
          f"{_colour_metric(overall.get('recall_benign'), width=10)} "
          f"{_colour_metric(overall.get('f1_benign'), width=10)} "
          f"{dim(n_b_str)}")
    print(f" {'adversarial':<12} "
          f"{_colour_metric(overall.get('precision_adversarial'), width=10)} "
          f"{_colour_metric(overall.get('recall_adversarial'), width=10)} "
          f"{_colour_metric(overall.get('f1_adversarial'), width=10)} "
          f"{green(n_a_str)}")


def _print_per_source_metrics(per_source: dict) -> None:
    if not per_source:
        return
    _subhead(f"Per-adversarial-source metrics  "
             f"{dim('(' + str(len(per_source)) + ' variant(s))')}")
    note = ("precision/recall/f1 computed against a balanced benign subsample "
            "(K × n_adv); roc_auc/pr_auc use all test benign.")
    print(f" {dim(note)}")
    print()
    hdr = (f" {'variant':<26} {'precision':>10} {'recall':>10} "
           f"{'f1':>10} {'roc_auc':>10} {'pr_auc':>10} {'n_adv':>8}")
    print(cyan(hdr))
    print(f" {dim('-' * 26)} {dim('-' * 10)} {dim('-' * 10)} {dim('-' * 10)} "
          f"{dim('-' * 10)} {dim('-' * 10)} {dim('-' * 8)}")

    low_support: list[str] = []

    # Group rows by base attack (mirai, ddos, …) so variants of the same
    # capture sit together; a blank line separates groups.
    def _base(src: str) -> str:
        entry = per_source.get(src, {})
        base = entry.get("base_source")
        return base if base else src

    def _mode(src: str) -> str:
        for suffix in ("_adv_both", "_adv_iat", "_adv_size"):
            if src.endswith(suffix):
                return suffix[5:]  # strip "_adv_"
        return ""

    sorted_srcs = sorted(per_source.keys(), key=lambda s: (_base(s), _mode(s), s))
    prev_base: str | None = None
    for src in sorted_srcs:
        entry = per_source[src]
        bal = entry.get("vs_balanced_benign", {})
        all_bn = entry.get("vs_all_benign", {})
        low = entry.get("low_support", False)
        base = _base(src)

        if prev_base is not None and base != prev_base:
            print()  # blank line between attack families
        prev_base = base

        # Render variant name: mode highlighted, base in dim suffix.
        mode = _mode(src)
        if mode:
            label = f"{base}  {magenta(mode)}"
            visible_len = len(base) + 2 + len(mode)
        else:
            label = src
            visible_len = len(src)
        if low:
            label = f"{label} {dim('*')}"
            visible_len += 2
            low_support.append(src)

        # Pad to 26 visible chars (account for ANSI).
        pad = max(0, 26 - visible_len)
        padded_label = label + (" " * pad)

        n_adv = all_bn.get("n_adversarial", 0)
        n_adv_str = f"{n_adv:>8,}"
        print(f" {padded_label} "
              f"{_colour_metric(bal.get('precision_adversarial'), width=10)} "
              f"{_colour_metric(bal.get('recall_adversarial'), width=10)} "
              f"{_colour_metric(bal.get('f1_adversarial'), width=10)} "
              f"{_colour_metric(all_bn.get('roc_auc'), width=10)} "
              f"{_colour_metric(all_bn.get('pr_auc'), width=10)} "
              f"{green(n_adv_str)}")

    if low_support:
        print(f"\n {dim('*  low support (n_adv < 50) — metrics are unreliable, treat as noise.')}")


def _print_artifacts(output_dir: Path) -> None:
    artifacts = [
        ("rf.joblib",              "trained classifier"),
        ("split.npz",              "train/val/test indices"),
        ("dataset.npz",            "merged dataset snapshot"),
        ("feature_names.json",     "RF stat names (24)"),
        ("split_summary.json",     "split counts"),
        ("training_info.json",     "grid search + best params"),
        ("metrics.json",           "full metrics (overall/per-adversarial-source)"),
        ("confusion_matrix.png",   "binary confusion matrix"),
        ("roc_curve.png",          "ROC curve"),
        ("pr_curve.png",           "precision-recall curve"),
        ("feature_importance.png", "Gini importances"),
    ]
    _subhead(f"Artifacts  {dim('→ ' + str(output_dir))}")
    total_bytes = 0
    n_present = 0
    for name, desc in artifacts:
        p = output_dir / name
        present = p.exists()
        if present:
            n_present += 1
            total_bytes += p.stat().st_size
        mark = green("✓") if present else dim("·")
        size = dim(f"({_fmt_size(p.stat().st_size):>10})") if present else dim("     —    ")
        print(f"   {mark} {name:<24} {size}  {dim(desc)}")
    print(f" {dim('─' * (W - 1))}")
    print(f"   {cyan('Total')}  {bold(f'{n_present}/{len(artifacts)}')} files, "
          f"{bold(_fmt_size(total_bytes))}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and/or evaluate the DEPT Random Forest base learner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["train", "eval", "both"], default="both")
    args = parser.parse_args()

    logging.basicConfig(
        level="WARNING",
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config_path = _resolve("config/rf.yaml", REPO_ROOT)
    if not config_path.exists():
        print(f"{red(bold('ERROR:'))} config not found: {config_path}")
        return 1
    config = yaml.safe_load(config_path.read_text())
    tensors_root = _resolve(config["tensors_dir"], REPO_ROOT)
    output_dir = _resolve(config["output_dir"], REPO_ROOT)
    adversarial_dir = _resolve(
        config.get("adversarial_dir", "data/tensors/adversarial"), REPO_ROOT)
    # Rewrite the config with the resolved adversarial_dir so downstream code
    # never has to worry about relative paths.
    config["adversarial_dir"] = str(adversarial_dir)

    _print_config_banner(config_path, config, tensors_root, output_dir, args.mode)

    total_t0 = time.time()

    total_steps = 2 if args.mode == "both" else 1

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    train_seconds = None
    if args.mode in ("train", "both"):
        _step(1, total_steps, "Training Random Forest")
        t0 = time.time()
        try:
            info = rf_train.train(config, tensors_root, output_dir)
        except Exception as e:
            print(f"  {red(bold('✗ FAILED'))} {e}")
            return 2
        train_seconds = time.time() - t0
        extra = (f"  {dim(f'({train_seconds / 60:.1f} min)')}"
                 if train_seconds >= 60 else "")
        print(f"  {green('✓')} Trained in {bold(f'{train_seconds:.1f}s')}{extra}")

        _print_split_summary(info["split_summary"])
        _print_grid_results(output_dir / "training_info.json")

    # -----------------------------------------------------------------------
    # Evaluate
    # -----------------------------------------------------------------------
    eval_seconds = None
    metrics = None
    if args.mode in ("eval", "both"):
        step_idx = 2 if args.mode == "both" else 1
        _step(step_idx, total_steps, "Evaluating on held-out test split")
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
        print(f"  {green('✓')} Evaluated in {bold(f'{eval_seconds:.1f}s')}")

        _print_overall_metrics(metrics["overall"])
        _print_per_source_metrics(metrics.get("per_adversarial_source", {}))

    # -----------------------------------------------------------------------
    # Wrap-up — boxed summary, matching process_all_attacks' style.
    # -----------------------------------------------------------------------
    total_elapsed = time.time() - total_t0

    print(f"\n{bold('=' * W)}")
    print(f" {bold('DEPT × Random Forest — Summary')}")
    print(f"{bold('=' * W)}")

    if metrics is not None:
        ov = metrics["overall"]
        # Headline KPIs for quick scanning.
        kpis = "   ".join([
            f"{cyan('Macro F1')} {_colour_metric(ov.get('macro_f1'))}",
            f"{cyan('ROC AUC')} {_colour_metric(ov.get('roc_auc'))}",
            f"{cyan('PR AUC')} {_colour_metric(ov.get('pr_auc'))}",
            f"{cyan('Recall-adv')} {_colour_metric(ov.get('recall_adversarial'))}",
        ])
        print(f" {kpis}")
        print(f" {dim('-' * (W - 1))}")

    if train_seconds is not None:
        pretty = (f"  {dim(f'({train_seconds / 60:.1f} min)')}"
                  if train_seconds >= 60 else "")
        print(f" {cyan('Train time')}   {bold(f'{train_seconds:>7.1f}s')}{pretty}")
    if eval_seconds is not None:
        print(f" {cyan('Eval  time')}   {bold(f'{eval_seconds:>7.1f}s')}")
    pretty_total = (f"  {dim(f'({total_elapsed / 60:.1f} min)')}"
                    if total_elapsed >= 60 else "")
    print(f" {cyan('Total time')}   {bold(f'{total_elapsed:>7.1f}s')}{pretty_total}")

    _print_artifacts(output_dir)
    print(f"{bold('=' * W)}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
