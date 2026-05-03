#!/usr/bin/env python3
"""
ADEPT — TCN Base Learner (train + eval)
=======================================

Trains the temporal base learner on the [N=10, F=1090] sequence tensors
produced by the Packet-to-Tensor pipeline and evaluates it on the same
held-out, flow-level test split as the RF (so the two can be compared
window-for-window). Under ADEPT's benign-vs-adversarial label convention,
the benign pool is every window from the Kitsune + CIC-IDS2017 folders
listed as ``benign_sources`` in tcn.yaml, and the adversarial pool is
every TANTGen-generated subfolder auto-discovered under
``adversarial_dir``.

Usage
-----
    # Train + evaluate with default config/tcn.yaml:
    python scripts/tcn/run_tcn.py

    # Train only:
    python scripts/tcn/run_tcn.py --mode train

    # Evaluate an already-trained artifact:
    python scripts/tcn/run_tcn.py --mode eval

    # Smoke test with 2000 windows per split (1 minute):
    python scripts/tcn/run_tcn.py --max-windows 2000

Outputs (under <output_dir> from tcn.yaml)
------------------------------------------
    tcn.pt                 — best PyTorch state_dict
    tcn_config.json        — resolved model + training config
    split.npz              — train/val/test row indices into dataset_index.npz
    dataset_index.npz      — per-window index (sources, local_idx, y, ...)
    split_summary.json     — per-split window counts (benign / adversarial / per-source)
    training_info.json     — best epoch, durations, device, effective config
    training_curves.json   — per-epoch train_loss / val_loss / val_macro_f1 / lr
    metrics.json           — overall, per-adversarial-source, per-label_str metrics
    confusion_matrix.png   — binary CM (test, benign vs adversarial)
    roc_curve.png          — ROC (test, when both classes present)
    pr_curve.png           — PR  (test, when both classes present)
    training_curves.png    — loss + val macro-F1 over epochs
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

from src.models.tcn import evaluate as tcn_eval  # noqa: E402
from src.models.tcn import train as tcn_train  # noqa: E402


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
    print(f"\n{dim('─' * W)}")
    print(f" {bold(cyan(f'[ {n} / {total} ]'))}  {bold(msg)}")
    print(f"{dim('─' * W)}")


def _subhead(title: str) -> None:
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


def _colour_metric(v, good: float = 0.9, ok: float = 0.75) -> str:
    if v is None:
        return dim("—")
    s = f"{v:.4f}"
    if v >= good:
        return green(s)
    if v >= ok:
        return yellow(s)
    return red(s)


def _resolve_device_label(device_cfg: str) -> str:
    if device_cfg != "auto":
        return device_cfg
    try:
        import torch
        if torch.cuda.is_available():
            return f"cuda ({torch.cuda.get_device_name(0)})"
        return "cpu"
    except Exception:
        return "auto"


def _print_config_banner(config_path: Path, config: dict,
                         tensors_root: Path, output_dir: Path,
                         mode: str, max_windows: int | None) -> None:
    benign_sources = config.get("benign_sources", [])
    adv_dir = Path(config.get("adversarial_dir", "data/tensors/adversarial"))
    adv_names: list[str] = []
    if adv_dir.exists():
        adv_names = sorted(
            d.name for d in adv_dir.iterdir()
            if d.is_dir() and (d / "metadata.json").exists()
        )
    sp = config.get("split", {})
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})

    _section(f"ADEPT — TCN  {dim('(mode: ' + mode + ')')}")
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
    _row("Device",       _resolve_device_label(str(train_cfg.get("device", "auto"))))
    _row("Epochs",       f"{bold(str(train_cfg.get('epochs')))}  "
                         f"{dim('early_stop_patience=' + str(train_cfg.get('early_stop_patience', 0)))}")
    _row("Batch",        f"{bold(str(train_cfg.get('batch_size')))}  "
                         f"{dim('num_workers=' + str(train_cfg.get('num_workers', 0)))}")
    _row("Channels",     f"{model_cfg.get('channels')}  "
                         f"{dim('kernel=' + str(model_cfg.get('kernel_size')) + '  dropout=' + str(model_cfg.get('dropout')))}")
    _row("LR",           f"{train_cfg.get('lr')}  "
                         f"{dim('wd=' + str(train_cfg.get('weight_decay')) + '  sched=' + str(train_cfg.get('scheduler')))}")
    _row("AMP",          green(bold('ON')) if train_cfg.get('amp') else yellow('OFF'))
    if max_windows:
        _row("Smoke test", yellow(f"--max-windows={max_windows}  (subsample per split)"))
    print(f"{bold('═' * W)}")


def _print_split_summary(summary: dict) -> None:
    _subhead("Split summary")
    hdr = (f" {'Split':<8} {'Windows':>12} {'Benign':>12} "
           f"{'Adversarial':>13} {'Adv %':>8}")
    print(cyan(hdr))
    print(f" {dim('-' * 8)} {dim('-' * 12)} {dim('-' * 12)} "
          f"{dim('-' * 13)} {dim('-' * 8)}")

    totals = {"n_windows": 0, "n_benign": 0, "n_adversarial": 0}
    for name in ("train", "val", "test"):
        s = summary[name]
        n = s["n_windows"]
        n_b = s["n_benign"]
        n_a = s["n_adversarial"]
        pct = (100.0 * n_a / n) if n else 0.0
        totals["n_windows"] += n
        totals["n_benign"] += n_b
        totals["n_adversarial"] += n_a
        print(f" {name:<8} {n:>12,} {dim(f'{n_b:>12,}')} "
              f"{green(f'{n_a:>13,}')} {yellow(f'{pct:>7.1f}%')}")

    print(f" {dim('-' * 8)} {dim('-' * 12)} {dim('-' * 12)} "
          f"{dim('-' * 13)} {dim('-' * 8)}")
    tn = totals["n_windows"]
    tb = totals["n_benign"]
    ta = totals["n_adversarial"]
    pct = (100.0 * ta / tn) if tn else 0.0
    total_label = f"{'total':<8}"
    tn_str = f"{tn:>12,}"
    tb_str = f"{tb:>12,}"
    ta_str = f"{ta:>13,}"
    pct_str = f"{pct:>7.1f}%"
    print(f" {bold(total_label)} {bold(tn_str)} "
          f"{dim(tb_str)} {green(bold(ta_str))} {yellow(pct_str)}")


def _print_training_curves(output_dir: Path, info: dict) -> None:
    curves_path = output_dir / "training_curves.json"
    if not curves_path.exists():
        return
    curves = json.loads(curves_path.read_text())
    if not curves:
        return

    _subhead("Training curves")
    best_epoch = info.get("best_epoch", -1)
    best_f1 = info.get("best_val_macro_f1")
    n_params = info.get("n_parameters")
    summary_bits = []
    if n_params is not None:
        summary_bits.append(f"{cyan('Params')} {bold(f'{n_params:,}')}")
    if best_f1 is not None and best_epoch is not None and best_epoch >= 0:
        summary_bits.append(
            f"{cyan('Best')} epoch {bold(str(best_epoch))} → "
            f"val_f1 {_colour_metric(best_f1)}")
    if summary_bits:
        print(f" {' '.join(summary_bits)}")
        print()

    hdr = (f"  {'epoch':>5} {'train_loss':>12} {'val_loss':>10} "
           f"{'val_f1':>9} {'lr':>10} {'sec':>7}")
    print(cyan(hdr))
    print(f"  {dim('-' * 5)} {dim('-' * 12)} {dim('-' * 10)} "
          f"{dim('-' * 9)} {dim('-' * 10)} {dim('-' * 7)}")

    for c in curves:
        ep = int(c["epoch"])
        marker = green(bold("★ ")) if ep == best_epoch else "  "
        print(f"{marker}{ep:>5} "
              f"{c['train_loss']:>12.4f} {c['val_loss']:>10.4f} "
              f"{_colour_metric(c['val_macro_f1']):>9} "
              f"{c['lr']:>10.2e} {c['seconds']:>7.1f}")


def _print_overall_metrics(overall: dict) -> None:
    _subhead("Overall metrics (test)")

    def _pill(label: str, v) -> str:
        return f"{cyan(label)} {_colour_metric(v)}"

    line = "   ".join([
        _pill("Acc", overall.get("accuracy")),
        _pill("Macro F1", overall.get("macro_f1")),
        _pill("ROC AUC", overall.get("roc_auc")),
        _pill("PR AUC", overall.get("pr_auc")),
    ])
    print(f" {line}\n")

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
          f"{_colour_metric(overall.get('precision_benign')):>10} "
          f"{_colour_metric(overall.get('recall_benign')):>10} "
          f"{_colour_metric(overall.get('f1_benign')):>10} "
          f"{dim(n_b_str)}")
    print(f" {'adversarial':<12} "
          f"{_colour_metric(overall.get('precision_adversarial')):>10} "
          f"{_colour_metric(overall.get('recall_adversarial')):>10} "
          f"{_colour_metric(overall.get('f1_adversarial')):>10} "
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

    def _base(src: str) -> str:
        entry = per_source.get(src, {})
        base = entry.get("base_source")
        return base if base else src

    def _mode(src: str) -> str:
        for suffix in ("_adv_both", "_adv_iat", "_adv_size"):
            if src.endswith(suffix):
                return suffix[5:]
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
            print()
        prev_base = base

        mode = _mode(src)
        if mode:
            label = f"{base}  {magenta(mode)}"
        else:
            label = src
        if low:
            label = f"{label} {dim('*')}"
            low_support.append(src)

        visible = src + (" *" if low else "")
        pad = max(0, 26 - len(visible))
        padded_label = label + (" " * pad)

        n_adv = all_bn.get("n_adversarial", 0)
        n_adv_str = f"{n_adv:>8,}"
        print(f" {padded_label} "
              f"{_colour_metric(bal.get('precision_adversarial')):>10} "
              f"{_colour_metric(bal.get('recall_adversarial')):>10} "
              f"{_colour_metric(bal.get('f1_adversarial')):>10} "
              f"{_colour_metric(all_bn.get('roc_auc')):>10} "
              f"{_colour_metric(all_bn.get('pr_auc')):>10} "
              f"{green(n_adv_str)}")

    if low_support:
        print(f"\n {dim('*  low support (n_adv < 50) — metrics are unreliable, treat as noise.')}")


def _print_artifacts(output_dir: Path) -> None:
    artifacts = [
        ("tcn.pt",                 "trained model state_dict"),
        ("tcn_config.json",        "resolved model + training config"),
        ("split.npz",              "train/val/test indices"),
        ("dataset_index.npz",      "per-window index (no payload)"),
        ("split_summary.json",     "split counts"),
        ("training_info.json",     "best epoch, durations, device"),
        ("training_curves.json",   "per-epoch loss + val macro-F1"),
        ("metrics.json",           "full metrics (overall/per-adversarial-source)"),
        ("confusion_matrix.png",   "binary confusion matrix"),
        ("roc_curve.png",          "ROC curve"),
        ("pr_curve.png",           "precision-recall curve"),
        ("training_curves.png",    "loss + macro-F1 over epochs"),
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
        description="Train and/or evaluate the ADEPT TCN base learner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config/tcn.yaml", help="Path to tcn.yaml")
    parser.add_argument("--mode", choices=["train", "eval", "both"], default="both")
    parser.add_argument("--max-windows", type=int, default=None,
                        help="Cap windows per split (smoke-test knob; full split if omitted)")
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
    adversarial_dir = _resolve(
        config.get("adversarial_dir", "data/tensors/adversarial"), REPO_ROOT)
    config["adversarial_dir"] = str(adversarial_dir)

    _print_config_banner(config_path, config, tensors_root, output_dir,
                          args.mode, args.max_windows)

    total_t0 = time.time()
    total_steps = 2 if args.mode == "both" else 1

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    train_seconds = None
    info: dict | None = None
    if args.mode in ("train", "both"):
        _step(1, total_steps, "Training Temporal Convolutional Network")
        t0 = time.time()
        try:
            info = tcn_train.train(config, tensors_root, output_dir,
                                    max_windows=args.max_windows)
        except Exception as e:
            print(f"  {red(bold('✗ FAILED'))} {e}")
            raise
        train_seconds = time.time() - t0
        extra = (f"  {dim(f'({train_seconds / 60:.1f} min)')}"
                 if train_seconds >= 60 else "")
        print(f"  {green('✓')} Trained in {bold(f'{train_seconds:.1f}s')}{extra}")

        _print_split_summary(info["split_summary"])
        _print_training_curves(output_dir, info)

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
            metrics = tcn_eval.evaluate(output_dir)
        except FileNotFoundError as e:
            print(f"  {red(bold('✗ FAILED'))} missing artifact: {e}. "
                  f"Run {bold('--mode train')} first.")
            return 2
        except Exception as e:
            print(f"  {red(bold('✗ FAILED'))} {e}")
            raise
        eval_seconds = time.time() - t0
        print(f"  {green('✓')} Evaluated in {bold(f'{eval_seconds:.1f}s')}")

        _print_overall_metrics(metrics["overall"])
        _print_per_source_metrics(metrics.get("per_adversarial_source", {}))

    # -----------------------------------------------------------------------
    # Wrap-up
    # -----------------------------------------------------------------------
    total_elapsed = time.time() - total_t0

    print(f"\n{bold('=' * W)}")
    print(f" {bold('ADEPT × TCN — Summary')}")
    print(f"{bold('=' * W)}")

    if metrics is not None:
        ov = metrics["overall"]
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

    if args.json and metrics is not None:
        print(json.dumps(metrics["overall"], indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
