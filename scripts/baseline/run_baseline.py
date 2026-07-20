#!/usr/bin/env python3
"""
DEPT — Deployment baseline (protected-NIDS "before vs after DEPT")
=================================================================

Measures how the protected NIDS behaves across three realized full-volume streams — and
therefore how DEPT performs when deployed in front of it:

  * clean       — original-attack size/timing        (oracle ceiling: NIDS working)
  * adversarial — TANTGen's perturbation              (evasion floor: detection collapses)
  * defended    — the full DEPT gated pipeline        (detector flags → purify only flagged)

Each attack is scored through the real NIDS it was built to evade (Kitsune for the
Kitsune-native attacks, TANTGen's ML-NIDS for the CIC attacks). The headline is the ASR
reduction (how much of TANTGen's attack success DEPT removes) and the detection
restoration from the adversarial floor toward the clean ceiling.

Usage
-----
    python3 scripts/baseline/run_baseline.py
    python3 scripts/baseline/run_baseline.py --attacks mirai,ddos

Reads config/baseline.yaml. Writes data/models/baseline/{baseline_metrics.json,
baseline_detection.png}. Requires trained RF + TCN + meta-learner + reconstructor, the
sibling TANTGen repo (Kitsune AfterImage .so + NIDS models), and — for the ML-NIDS
attacks — `cicflowmeter` on PATH.
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

# The package re-exports the ``evaluate`` *function* (shadowing the submodule
# of the same name), so bind it as a callable — not as a module.
from src.models.baseline import evaluate as baseline_evaluate  # noqa: E402


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
    if v is None:
        return dim("—")
    return f"{v:.{nd}f}"


def _pct(v, nd: int = 1) -> str:
    if v is None:
        return dim("—")
    return f"{v * 100:.{nd}f}%"


def _print_overall(title: str, ov: dict) -> None:
    dr = ov.get("detection_rate", {})
    asr = ov.get("asr", {})
    print(f"\n {bold(title)}")
    print(f"   {cyan('detection rate')}  "
          f"clean {_fmt(dr.get('clean'))}  "
          f"adversarial {_fmt(dr.get('adversarial'))}  "
          f"{bold('defended ' + _fmt(dr.get('defended')))}"
          + (f"  purified-all {_fmt(dr.get('purified_all'))}"
             if dr.get('purified_all') is not None else ""))
    print(f"   {cyan('attack success')}  "
          f"before DEPT {_fmt(asr.get('adversarial'))}  →  "
          f"after DEPT {bold(green(_fmt(asr.get('defended'))))}")
    print(f"   {cyan('ASR reduction')} {bold(green(_pct(ov.get('asr_reduction'))))}   "
          f"{cyan('detection restoration')} "
          f"{bold(_pct(ov.get('restoration_ratio_detection')))}   "
          f"{dim('n_flows ' + str(ov.get('n_flows', 0)))}")


def _print_baseline(summary: dict) -> None:
    _print_overall("Overall (all attacks) — before vs after DEPT", summary["overall"])
    for kind, ov in sorted((summary.get("by_kind") or {}).items()):
        _print_overall(f"By NIDS family — {kind}", ov)

    per = summary.get("per_source", {})
    if per:
        print(f"\n {bold('Per-source')}  {dim('(' + str(len(per)) + ' variant(s))')}")
        hdr = (f"   {'variant':<24} {'nids':>7} {'dr_clean':>9} {'dr_adv':>8} "
               f"{'dr_def':>8} {'asr_red':>8} {'restor':>7} {'flagged':>8}")
        print(cyan(hdr))
        for name in sorted(per.keys()):
            s = per[name]
            dr = s.get("detection_rate", {})
            gate = s.get("gate", {})
            restor = s.get("restoration_ratio_detection")
            print(f"   {name:<24} {str(s.get('kind') or '—'):>7} "
                  f"{_fmt(dr.get('clean')):>9} "
                  f"{_fmt(dr.get('adversarial')):>8} "
                  f"{_fmt(dr.get('defended')):>8} "
                  f"{_pct(s.get('asr_reduction')):>8} "
                  f"{(_fmt(restor, 2) if restor is not None else '—'):>7} "
                  f"{_pct(gate.get('flag_rate')):>8}")

    skipped = summary.get("skipped_sources", {})
    if skipped:
        print(f"\n {yellow('Skipped sources')}  {dim('(' + str(len(skipped)) + ')')}")
        for name, why in sorted(skipped.items()):
            print(f"   {name:<24} {dim(str(why))}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DEPT deployment baseline — protected-NIDS efficacy clean/perturbed/defended",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--attacks", default=None, metavar="A,B,…",
        help="comma-separated base attacks to evaluate (default: all mapped in nids.attacks)")
    args = parser.parse_args()

    logging.basicConfig(
        level="INFO",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cfg_path = _resolve("config/baseline.yaml", REPO_ROOT)
    if not cfg_path.exists():
        print(f"{red(bold('ERROR:'))} config not found: {cfg_path}")
        return 1
    cfg = yaml.safe_load(cfg_path.read_text())

    for k in ("tensors_dir", "adversarial_dir", "adversarial_pcap_dir",
              "rf_model_dir", "tcn_model_dir", "metalearner_model_dir",
              "reconstructor_model_dir", "output_dir"):
        cfg[k] = str(_resolve(cfg[k], REPO_ROOT))
    cfg["_repo_root"] = str(REPO_ROOT)
    output_dir = Path(cfg["output_dir"])

    attacks = None
    if args.attacks:
        attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]

    _section("DEPT — Deployment baseline  "
             + dim("(clean → perturbed → DEPT-defended, through the independent NIDS)"))
    _row("Config", str(cfg_path))
    _row("Attacks", ", ".join(attacks) if attacks else "all (nids.attacks)")
    _row("Detector", cfg["metalearner_model_dir"] + dim("  (gates purification)"))
    _row("Reconstructor", cfg["reconstructor_model_dir"])
    _row("Output dir", str(output_dir))
    print(f"{bold('═' * W)}")

    t0 = time.time()
    try:
        summary = baseline_evaluate(cfg, output_dir, attacks=attacks)
    except Exception as e:
        print(f"  {red(bold('✗ FAILED'))} {e}")
        raise
    print(f"\n  {green('✓')} Baseline evaluated in {bold(f'{time.time() - t0:.1f}s')}")
    _print_baseline(summary)

    print(f"\n{bold('═' * W)}")
    print(f" {bold('DEPT × Baseline — Done')}  {dim(f'Total {time.time() - t0:.1f}s')}")
    print(f" {cyan('Artifacts')}  {output_dir}")
    print(f"{bold('═' * W)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
