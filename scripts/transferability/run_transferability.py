#!/usr/bin/env python3
"""
DEPT — Gradient-guided transferable evasion (RQ1)
=================================================

TANTGen, the adversary DEPT is trained on, is black-box and gradient-free. This
study adds the attacker RQ1 asks about: one that differentiates the **TCN**
directly, and whose perturbation is then scored — unchanged — against all three
targets:

  * rf     — the non-differentiable forest      (no gradient path; transfer only)
  * tcn    — the differentiable base learner    (white-box; the attacker's own target)
  * stack  — RF + TCN + meta-learner            (the heterogeneous ensemble)

The attacker pays in units it actually spends: padding bytes and delay seconds,
increase-only (the packets are already on the wire). The headline is a cost
curve — how much budget each target demands — compared at a **matched
false-positive rate**, not at a raw 0.5 threshold.

If the stack costs more than the TCN alone, heterogeneous stacking has
measurably raised the price of evasion, and RQ1 is answered by measurement.

Usage
-----
    python3 scripts/transferability/run_transferability.py
    python3 scripts/transferability/run_transferability.py --attack mirai
    python3 scripts/transferability/run_transferability.py --flows-per-source 20 \
        --budgets b0,b32

Extra-pipeline: NOT a phase of run_pipeline.sh. Reads config/transferability.yaml.
Writes data/models/transferability/{transferability_metrics.json, evasion_cost.png,
sidecar/}. Requires trained RF + TCN + meta-learner under data/models/ and the
TANTGen PCAPs under data/adversarial/.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

import yaml

warnings.filterwarnings(
    "ignore",
    message=r".*enable_nested_tensor is True, but self\.use_nested_tensor is False.*",
    category=UserWarning,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.models.transferability import evaluate as transfer_eval  # noqa: E402


_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def bold(t):   return _c("1", t)
def dim(t):    return _c("2", t)
def green(t):  return _c("32", t)
def yellow(t): return _c("33", t)
def red(t):    return _c("31", t)
def cyan(t):   return _c("36", t)


W = 84


def _section(title: str) -> None:
    print(f"\n{bold('═' * W)}")
    print(f" {bold(title)}")
    print(f"{bold('═' * W)}")


def _row(label: str, value: str) -> None:
    print(f" {cyan(f'{label:<18}')}  {value}")


def _resolve(path_str: str, repo_root: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else repo_root / p


def _fmt(v, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return dim("—")
    return f"{v:.{nd}f}"


def _pct(v, nd: int = 1) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return dim("—")
    return f"{v * 100:.{nd}f}%"


def _print_summary(s: dict) -> None:
    policy = s["report_policy"]
    budgets = s["asr"]["budgets"]
    variants = s["asr"].get("variants", ["pgd"])
    spend = s["spend"]

    print(f"\n {bold('Thresholds')}  {dim('(matched FPR on unperturbed benign flows)')}")
    for t, pol in sorted(s["thresholds"].items()):
        pretty = "  ".join(f"{k}={_fmt(v)}" for k, v in sorted(pol.items()))
        print(f"   {cyan(f'{t:<8}')} {pretty}")

    print(f"\n {bold('Detection at δ=0')}  "
          f"{dim('(the ASR denominator — a target that misses a flow cannot be evaded)')}")
    for t in ("rf", "tcn", "stack"):
        cell = (s["asr"]["overall"].get("best", {}).get(t, {})
                .get(policy, {}).get(budgets[0], {}))
        print(f"   {cyan(f'{t:<8}')} detection_rate={_fmt(cell.get('detection_rate_clean'))}"
              f"  {dim('n_detected=' + str(cell.get('n_detected', 0)) + '/' + str(cell.get('n_flows', 0)))}")

    print(f"\n {bold(f'Evasion cost — ASR @ {policy}')}  "
          f"{dim('(one perturbation crafted on the TCN, scored on all three)')}")
    hdr = f"   {'budget':<8} {'pad B/flow':>11} {'delay ms':>9}"
    for t in ("rf", "tcn", "stack"):
        hdr += f" {t.upper():>8}"
    print(cyan(hdr))
    for b in budgets:
        sp = spend.get(b, {})
        line = (f"   {b:<8} "
                f"{_fmt(sp.get('median_total_pad_bytes_per_flow'), 1):>11} "
                f"{_fmt(sp.get('median_delay_ms_per_packet'), 3):>9}")
        for t in ("rf", "tcn", "stack"):
            cell = (s["asr"]["overall"].get("best", {}).get(t, {})
                    .get(policy, {}).get(b, {}))
            line += f" {_pct(cell.get('asr'), 1):>8}"
        print(line)
    print(dim("   best-of-attacker (PGD ∪ random), monotone envelope, denominator ="))
    print(dim("   flows the target detects at δ=0."))

    if {"pgd", "random"} <= set(variants):
        print(f"\n {bold('Does the gradient help?')}  "
              f"{dim('(TCN ASR: PGD vs query-matched random search)')}")
        print(cyan(f"   {'budget':<8} {'PGD':>8} {'random':>8}"))
        for b in budgets:
            p = (s["asr"]["overall"]["pgd"]["tcn"].get(policy, {}).get(b, {}))
            r = (s["asr"]["overall"]["random"]["tcn"].get(policy, {}).get(b, {}))
            print(f"   {b:<8} {_pct(p.get('asr'), 1):>8} {_pct(r.get('asr'), 1):>8}")

    print(f"\n {bold('Minimum budget to evade')}  "
          f"{dim('(median over flows that ever evade; denominator = detected at δ=0)')}")
    mb = s["min_budget"]
    for t in ("rf", "tcn", "stack"):
        e = mb.get(t)
        if not isinstance(e, dict):
            continue
        print(f"   {cyan(f'{t:<8}')} median={bold(str(e.get('median_min_budget_id') or '—')):<14}"
              f" never-evaded={_pct(e.get('frac_never_evaded'))}"
              f"  {dim('n_detected=' + str(e.get('n_detected_unperturbed', 0)))}")

    rs = s.get("reliance_shift") or {}
    if rs:
        print(f"\n {bold('Meta-learner reliance shift')}  "
              f"{dim('(does the meta lean on the RF as the TCN is degraded?)')}")
        print(cyan(f"   {'budget':<8} {'stack p_adv':>12} {'Δ if RF clean':>14}"
                   f" {'Δ if TCN clean':>15} {'rf_reliance':>12}"))
        for b in budgets:
            e = rs.get(b)
            if not e:
                continue
            print(f"   {b:<8} {_fmt(e['median_stack_verdict']):>12}"
                  f" {_fmt(e['median_delta_if_rf_restored']):>14}"
                  f" {_fmt(e['median_delta_if_tcn_restored']):>15}"
                  f" {bold(_fmt(e['rf_reliance'])):>12}")

    ac = s.get("attack_strength_check") or {}
    if ac.get("available"):
        f = ac["frac_margin_below_zero"]
        print(f"\n {bold('Attack-strength cross-check')}  "
              f"{dim('(ART AutoPGD, relaxed, eps=' + str(ac['eps_bytes']) + ' B — is our search under-tuned?)')}")
        print(f"   {cyan('ART AutoPGD')} {_pct(f['art_autopgd'])}   "
              f"{cyan('ours PGD')} {_pct(f['ours_pgd'])}   "
              f"{cyan('ours random')} {_pct(f['ours_random'])}   "
              f"{dim('(frac below the boundary, n=' + str(ac['n_flows']) + ')')}")

    skipped = s.get("skipped_sources") or {}
    if skipped:
        print(f"\n {yellow('Skipped sources')}  {dim('(' + str(len(skipped)) + ')')}")
        for name, why in sorted(skipped.items()):
            print(f"   {name:<26} {dim(str(why)[:70])}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DEPT — gradient-guided evasion: RF vs TCN vs stack (RQ1)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--attack", default=None, metavar="NAME",
        help="restrict to one base attack (e.g. mirai); default: all sources")
    parser.add_argument(
        "--flows-per-source", type=int, default=None, metavar="N",
        help="override config flows_per_source (smaller = faster smoke run)")
    parser.add_argument(
        "--budgets", default=None, metavar="ID,ID",
        help="comma-separated budget ids from the config grid (default: all)")
    parser.add_argument(
        "--config", default="config/transferability.yaml", metavar="PATH")
    args = parser.parse_args()

    logging.basicConfig(
        level="INFO",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cfg_path = _resolve(args.config, REPO_ROOT)
    if not cfg_path.exists():
        print(f"{red(bold('ERROR:'))} config not found: {cfg_path}")
        return 1
    cfg = yaml.safe_load(cfg_path.read_text())

    for k in ("tensors_dir", "adversarial_dir", "adversarial_pcap_dir",
              "rf_model_dir", "tcn_model_dir", "metalearner_model_dir",
              "output_dir"):
        cfg[k] = str(_resolve(cfg[k], REPO_ROOT))
    cfg["_repo_root"] = str(REPO_ROOT)

    if args.flows_per_source is not None:
        cfg["flows_per_source"] = args.flows_per_source
    if args.budgets:
        want = {b.strip() for b in args.budgets.split(",") if b.strip()}
        cfg["budgets"] = [b for b in cfg["budgets"] if str(b.get("id")) in want]
        if not cfg["budgets"]:
            print(f"{red(bold('ERROR:'))} no budgets match {sorted(want)}")
            return 1

    output_dir = Path(cfg["output_dir"])

    _section("DEPT — Gradient-guided evasion  "
             + dim("(PGD on the TCN → scored on RF / TCN / stack)"))
    _row("Config", str(cfg_path))
    _row("Attack scope", args.attack or "all sources")
    _row("Flows/source", str(cfg.get("flows_per_source")))
    _row("Budget grid", ", ".join(str(b.get("id")) for b in cfg["budgets"]))
    _row("Threat model", "white-box PGD on the TCN; " + dim("no gradient path through the RF"))
    _row("Output dir", str(output_dir))
    print(f"{bold('═' * W)}")

    t0 = time.time()
    try:
        summary = transfer_eval.evaluate(cfg, output_dir, attack=args.attack)
    except Exception as e:
        print(f"  {red(bold('✗ FAILED'))} {e}")
        raise
    print(f"\n  {green('✓')} Attacked {bold(str(summary['n_flows']))} flows "
          f"in {bold(f'{time.time() - t0:.1f}s')}")
    _print_summary(summary)

    print(f"\n{bold('═' * W)}")
    print(f" {bold('DEPT × Transferability — Done')}  {dim(f'Total {time.time() - t0:.1f}s')}")
    print(f" {cyan('Artifacts')}  {output_dir}")
    print(f"{bold('═' * W)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
