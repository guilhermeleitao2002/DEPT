#!/usr/bin/env python3
"""
DEPT — Phase 9: DEPT's purifier vs the diffusion purifier of Merzouk et al.
===========================================================================

Replaces DEPT's reconstructor with a published purifier — the diffusion model of
Merzouk et al., "Diffusion-based Adversarial Purification for Intrusion Detection"
(DBSec 2025) — inside the phase-8 harness, and scores both through the independent
NIDS each attack was built to evade. Same flows, same realization, same NIDS.

Stages (run in order; each skips work already on disk unless --force):

  overlays  — clean training / validation flows of the reconstructor's split, as
              (size, IPAT) vectors, plus the IPAT scale
  train     — the diffusion model (clean data only, as in the paper)
  sweep     — t* on validation flows: the step with the highest NIDS detection
  evaluate  — clean / adversarial / DEPT / diffusion (+ both behind DEPT's gate)
              through the NIDS, on the held-out test pairs; resumable per source

Usage
-----
    python3 scripts/diffusion_purifier/run_diffusion_purifier.py
    python3 scripts/diffusion_purifier/run_diffusion_purifier.py --stage train
    python3 scripts/diffusion_purifier/run_diffusion_purifier.py --stage evaluate --attacks ddos,portscan
    python3 scripts/diffusion_purifier/run_diffusion_purifier.py --stage sweep --force

Reads config/diffusion_purifier.yaml (+ reconstructor.yaml and baseline.yaml for the
shared settings). Writes under data/models/diffusion_purifier/.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=r".*enable_nested_tensor is True, but self\.use_nested_tensor is False.*",
    category=UserWarning,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.models import diffusion_purifier as dp  # noqa: E402

STAGES = ("overlays", "train", "sweep", "evaluate")
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


def _fmt(v, nd: int = 3) -> str:
    return dim("—") if v is None else f"{v:.{nd}f}"


def _pct(v, nd: int = 1) -> str:
    return dim("—") if v is None else f"{v * 100:.{nd}f}%"


_NAMES = {"clean": "clean (oracle)", "adversarial": "adversarial (TANTGen)",
          "dept_all": "DEPT purifier", "diffusion_all": "diffusion purifier",
          "dept_gated": "DEPT gate + DEPT purifier", "diffusion_gated": "DEPT gate + diffusion"}


def _print_block(title: str, d: dict) -> None:
    print(f"\n {bold(title)}")
    print(cyan(f"   {'regime':<28} {'flows':>9} {'detection':>10} {'ASR':>8} "
               f"{'ASR red.':>9} {'restored':>9}"))
    for r, dr in d["detection_rate"].items():
        print(f"   {_NAMES.get(r, r):<28} {d['n_flows'][r]:>9} {_fmt(dr):>10} "
              f"{_fmt(d['asr'][r]):>8} {_pct(d['asr_reduction'].get(r)):>9} "
              f"{_pct(d['restoration_ratio_detection'].get(r)):>9}")


def _print_summary(summary: dict) -> None:
    print(f"\n {bold('t*')} = {summary['t_star']}   {dim('split: ' + str(summary['split']))}")
    subs = summary.get("subsets", {})
    titles = {"test": "Held-out test pairs (headline)",
              "strict": "Strict subset — original flow in no training pair",
              "gated": "Gated subset — gate and purifiers both held out",
              "full": "Kitsune full streams (deployment view)"}
    for sub in ("test", "strict", "gated", "full"):
        if sub in subs:
            _print_block(f"{titles[sub]} — overall", subs[sub]["overall"])
    for kind, d in (subs.get("test", {}).get("by_kind") or {}).items():
        _print_block(f"Held-out test pairs — {kind}", d)

    per = summary.get("per_source", {})
    if per:
        print(f"\n {bold('Per source')}  {dim('(test subset detection rate)')}")
        print(cyan(f"   {'source':<24} {'nids':>7} {'clean':>7} {'adv':>7} "
                   f"{'DEPT':>7} {'diffusion':>9}"))
        for name in sorted(per):
            dr = per[name]["derived"]["test"]["detection_rate"]
            print(f"   {name:<24} {per[name]['kind']:>7} {_fmt(dr['clean']):>7} "
                  f"{_fmt(dr['adversarial']):>7} {_fmt(dr['dept_all']):>7} "
                  f"{_fmt(dr['diffusion_all']):>9}")
    lat = summary.get("latency", {})
    for bs, v in (lat.get("by_batch_size") or {}).items():
        print(f"\n {bold('Latency')} {dim('(' + str(lat.get('device')) + ', batch ' + bs + ')')}"
              f"  DEPT {v['dept_ms_per_flow']:.3f} ms/flow   diffusion "
              f"{v['diffusion_ms_per_flow']:.3f} ms/flow   "
              f"({v['diffusion_over_dept']:.1f}×)")
    skipped = summary.get("skipped_sources", {})
    if skipped:
        print(f"\n {yellow('Skipped sources')}")
        for name, why in sorted(skipped.items()):
            print(f"   {name:<24} {dim(str(why))}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DEPT phase 9 — DEPT's purifier vs the diffusion purifier (Merzouk et al.)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default="all", choices=("all",) + STAGES,
                        help="stage to run (default: all, in order)")
    parser.add_argument("--attacks", default=None, metavar="A,B,…",
                        help="evaluate: only these base attacks (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="redo the selected stage(s) even if their outputs exist")
    parser.add_argument("--config", default="config/diffusion_purifier.yaml")
    args = parser.parse_args()

    logging.basicConfig(level="INFO",
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    if not cfg_path.exists():
        print(f"{red(bold('ERROR:'))} config not found: {cfg_path}")
        return 1
    cfg = dp.load_config(cfg_path, REPO_ROOT)
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()] if args.attacks else None
    stages = STAGES if args.stage == "all" else (args.stage,)

    _section("DEPT — Phase 9  " + dim("(DEPT's purifier vs diffusion purification, "
                                       "Merzouk et al.)"))
    _row("Config", str(cfg_path))
    _row("Stages", ", ".join(stages) + ("  (--force)" if args.force else ""))
    _row("Attacks", ", ".join(attacks) if attacks else "all (nids.attacks)")
    _row("Reconstructor", cfg["reconstructor_model_dir"])
    _row("Output dir", cfg["output_dir"])
    print(f"{bold('═' * W)}")

    t_all = time.time()
    summary = None
    for stage in stages:
        t0 = time.time()
        print(f"\n {bold('▶ ' + stage)}")
        if stage == "overlays":
            info = dp.build_caches(cfg, force=args.force)
            _row("Train flows", f"{info['n_train_flows']:,} "
                 f"({info['n_train_packets']:,} packets)")
            _row("Val flows", f"{info['n_val_flows']:,}")
        elif stage == "train":
            info = dp.train(cfg, force=args.force)
            tr = info["training"]
            _row("Best step", f"{tr['best_step']} (val loss {tr['best_val_loss']:.5f})")
            _row("Parameters", f"{tr['n_parameters']:,}")
        elif stage == "sweep":
            res = dp.run_sweep(cfg, force=args.force)
            _row("t*", f"{res['t_star']}  {dim(res['selection'])}")
            rates = (res.get("nids") or {}).get("detection_rate") or {}
            if rates:
                _row("Val detection", "  ".join(
                    f"{k}={_fmt(v)}" for k, v in rates.items() if v is not None))
        elif stage == "evaluate":
            summary = dp.evaluate(cfg, attacks=attacks, force=args.force)
        print(f"  {green('✓')} {stage} done in {bold(f'{time.time() - t0:.1f}s')}")

    if summary is not None:
        _print_summary(summary)
    print(f"\n{bold('═' * W)}")
    print(f" {bold('DEPT × Phase 9 — Done')}  {dim(f'Total {time.time() - t_all:.1f}s')}")
    print(f" {cyan('Artifacts')}  {cfg['output_dir']}")
    print(f"{bold('═' * W)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
