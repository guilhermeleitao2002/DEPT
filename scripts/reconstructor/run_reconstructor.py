#!/usr/bin/env python3
"""
DEPT — Reconstructor (Purification) Transformer (train + eval)
==============================================================

Trains a small causal residual denoiser that purifies adversarially-perturbed
flows back onto the clean-traffic manifold, and evaluates it on the held-out
test pairs with three metric groups:

  * Reconstruction quality  — MSE on perturbed dims, before vs after.
  * Independent-NIDS efficacy (headline) — detection restoration through the real
    NIDS each attack evades (Kitsune / ML-NIDS), via src.models.nids.
  * Purification gain (secondary) — DEPT's own TCN P(adversarial); circular, kept
    only as an in-distribution sanity check.
  * Protocol compliance     — tshark on the realized purified packets.

Usage
-----
    python scripts/reconstructor/run_reconstructor.py
    python scripts/reconstructor/run_reconstructor.py --mode train
    python scripts/reconstructor/run_reconstructor.py --mode eval

    # Per-attack model — pairs restricted to one attack, purification gain
    # scored through the per-attack TCN from data/models/per_attack/<attack>/tcn
    # (train it first); artifacts land under
    # data/models/per_attack/<attack>/reconstructor:
    python scripts/reconstructor/run_reconstructor.py --attack mirai

Reads config/reconstructor.yaml and force-overwrites every artifact under
<output_dir>.

Outputs (under <output_dir> from reconstructor.yaml)
----------------------------------------------------
    reconstructor.pt             — best PyTorch state_dict (min val recon loss)
    reconstructor_config.json    — resolved model + training config
    split.npz                    — train/val/test pair indices (eval rebuilds
                                   the deterministic pair table from config)
    split_summary.json           — per-split / per-source pair counts
    training_curves.json         — per-epoch train/val loss
    training_info.json           — best epoch, durations, device, pair stats
    metrics.json                 — reconstruction / purification / compliance
    realized/<source>.pcap       — realized purified packets (traffic-space)
    reconstruction_error.png / purification_gain.png / training_curves.png
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

from src.models.common.attack_scope import (  # noqa: E402
    per_attack_models_root, scoped_adversarial_exclude,
)
from src.models.reconstructor import evaluate as rc_eval  # noqa: E402
from src.models.reconstructor import train as rc_train    # noqa: E402


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
    print(f" {cyan(f'{label:<16}')}  {value}")


def _resolve(path_str: str, repo_root: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else repo_root / p


def _fmt(v, nd: int = 4) -> str:
    if v is None:
        return dim("—")
    return f"{v:.{nd}f}"


def _print_recon_metrics(metrics: dict) -> None:
    ov = metrics["overall"]
    rec = ov["reconstruction"]
    print(f"\n {bold('Reconstruction (perturbed dims)')}")
    before = rec.get("perturbed_mse_before")
    after = rec.get("perturbed_mse_after")
    ratio = rec.get("reduction_ratio")
    print(f"   {cyan('MSE before')} {_fmt(before, 6)}   "
          f"{cyan('after')} {_fmt(after, 6)}   "
          f"{cyan('reduction')} {bold(_fmt(ratio, 1) + '×') if ratio else dim('—')}")

    # ---- Headline: independent NIDS (detection restoration) ----
    nids = metrics.get("nids", {})
    nov = nids.get("overall") if isinstance(nids, dict) else None
    if nov:
        print(f"\n {bold('Purification efficacy (independent NIDS — the headline metric)')}")
        print(f"   {dim('detection restored: adv (evaded) → purified → clean (oracle)')}")
        print(f"   {cyan('detection rate')}  adv {_fmt(nov.get('detection_rate_adv'))}  "
              f"purified {_fmt(nov.get('detection_rate_purified'))}  "
              f"clean {_fmt(nov.get('detection_rate_clean'))}")
        drr = nov.get('detection_restoration_ratio')
        print(f"   {cyan('detection restoration')} "
              f"{bold(green((_fmt(drr * 100, 1) + '%') if drr is not None else '—'))}   "
              f"{dim('(rate-based; aggregates Kitsune + ML-NIDS)')}")
        for kind, kk in sorted((nids.get('by_kind') or {}).items()):
            srr = kk.get('score_restoration_ratio')
            print(f"   {cyan(kind + ' score restoration')} "
                  f"{(_fmt(srr * 100, 1) + '%') if srr is not None else dim('—')}   "
                  f"{dim('mean score adv/pur/clean')} "
                  f"{_fmt(kk.get('mean_score_adv'), 3)}/"
                  f"{_fmt(kk.get('mean_score_purified'), 3)}/"
                  f"{_fmt(kk.get('mean_score_clean'), 3)}")
    elif isinstance(nids, dict) and nids.get("skipped"):
        print(f"\n {bold('Purification efficacy (independent NIDS)')}  "
              f"{yellow('skipped:')} {dim(str(nids['skipped']))}")

    # ---- Secondary: TCN (circular — internal sanity check only) ----
    if metrics.get("tcn_used"):
        pur = ov["purification"]
        print(f"\n {bold('Purification gain (through TCN — internal sanity check, circular)')}")
        print(f"   {cyan('P(adv|adv)')} {_fmt(pur['p_adv_on_adv'])}   "
              f"{cyan('P(adv|recon)')} {_fmt(pur['p_adv_on_recon'])}   "
              f"{cyan('P(adv|clean)')} {_fmt(pur['p_adv_on_clean'])}")
        print(f"   {cyan('gain')} {_fmt(pur['gain'])}   "
              f"{cyan('residual gap')} {_fmt(pur['residual_gap'])}")

    comp = ov["protocol_compliance"]
    print(f"\n {bold('Protocol compliance')}")
    print(f"   {cyan('tshark')} {_fmt(comp.get('tshark_rate'))}")

    per = metrics.get("per_source", {})
    if per:
        print(f"\n {bold('Per-source')}  {dim('(' + str(len(per)) + ' variant(s))')}")
        hdr = (f"   {'variant':<24} {'mse_after':>10} {'nids':>7} "
               f"{'det_adv':>8} {'det_pur':>8} {'det_cln':>8} "
               f"{'restor':>7} {'tshark':>7}")
        print(cyan(hdr))
        for name in sorted(per.keys()):
            s = per[name]
            r = s["reconstruction"]
            c = s["protocol_compliance"]
            k = s.get("nids", {})
            srr = k.get("score_restoration_ratio")
            print(f"   {name:<24} "
                  f"{_fmt(r.get('perturbed_mse_after'), 5):>10} "
                  f"{str(k.get('kind') or '—'):>7} "
                  f"{_fmt(k.get('detection_rate_adv'), 3):>8} "
                  f"{_fmt(k.get('detection_rate_purified'), 3):>8} "
                  f"{_fmt(k.get('detection_rate_clean'), 3):>8} "
                  f"{(_fmt(srr, 2) if srr is not None else '—'):>7} "
                  f"{_fmt(c.get('tshark_rate'), 3):>7}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DEPT reconstructor — train + evaluate the purification transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["train", "eval", "both"], default="both")
    parser.add_argument(
        "--attack", default=None, metavar="NAME",
        help="restrict pairs to one attack and score purification through the "
             "per-attack TCN from data/models/per_attack/<NAME>/tcn; artifacts "
             "go to data/models/per_attack/<NAME>/reconstructor")
    args = parser.parse_args()

    logging.basicConfig(
        level="WARNING",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg_path = _resolve("config/reconstructor.yaml", REPO_ROOT)
    if not cfg_path.exists():
        print(f"{red(bold('ERROR:'))} config not found: {cfg_path}")
        return 1
    cfg = yaml.safe_load(cfg_path.read_text())

    tensors_root = _resolve(cfg["tensors_dir"], REPO_ROOT)
    output_dir = _resolve(cfg["output_dir"], REPO_ROOT)
    cfg["output_dir"] = str(output_dir)
    for k in ("tensors_dir", "adversarial_dir", "tcn_model_dir"):
        cfg[k] = str(_resolve(cfg[k], REPO_ROOT))
    cfg.setdefault("realize", {})
    cfg["realize"]["adversarial_pcap_dir"] = str(_resolve(
        cfg["realize"].get("adversarial_pcap_dir", "data/adversarial"), REPO_ROOT))
    cfg["_repo_root"] = str(REPO_ROOT)

    if args.attack:
        try:
            cfg["adversarial_exclude"] = scoped_adversarial_exclude(
                Path(cfg["adversarial_dir"]), args.attack,
                cfg.get("adversarial_exclude", []))
        except ValueError as e:
            print(f"{red(bold('ERROR:'))} {e}")
            return 1
        root = per_attack_models_root(REPO_ROOT, args.attack)
        cfg["tcn_model_dir"] = str(root / "tcn")
        output_dir = root / "reconstructor"
        cfg["output_dir"] = str(output_dir)
        if not (root / "tcn" / "tcn.pt").exists():
            print(f"{red(bold('ERROR:'))} per-attack TCN missing: "
                  f"{root / 'tcn' / 'tcn.pt'}. "
                  f"Train it first: run_tcn.py --attack {args.attack}")
            return 1

    _section(f"DEPT — Reconstructor  {dim('(mode: ' + args.mode + ')')}")
    _row("Config", str(cfg_path))
    if args.attack:
        _row("Attack scope", f"{bold(args.attack)}  "
                             f"{dim('(per-attack run — other attacks excluded)')}")
    _row("Tensors dir", str(tensors_root))
    _row("TCN model dir", cfg["tcn_model_dir"])
    _row("Output dir", str(output_dir))
    print(f"{bold('═' * W)}")

    total_steps = 2 if args.mode == "both" else 1
    t0 = time.time()
    info = None
    if args.mode in ("train", "both"):
        _step(1, total_steps, "Training reconstructor (causal residual denoiser)")
        t = time.time()
        try:
            info = rc_train.train(cfg, tensors_root, output_dir)
        except Exception as e:
            print(f"  {red(bold('✗ FAILED'))} {e}")
            raise
        print(f"  {green('✓')} Trained in {bold(f'{time.time() - t:.1f}s')}")
        sk = info.get("split_summary", {})
        print(f"  {dim('pairs:')} {info['n_pairs']:,}  "
              f"{dim('train/val/test:')} "
              f"{sk.get('train', {}).get('n_pairs', 0):,}/"
              f"{sk.get('val', {}).get('n_pairs', 0):,}/"
              f"{sk.get('test', {}).get('n_pairs', 0):,}")
        print(f"  {dim('best epoch:')} {info['best_epoch']}  "
              f"{dim('val loss:')} {info['best_val_loss']:.5f}  "
              f"{dim('params:')} {info['n_parameters']:,}")

    metrics = None
    if args.mode in ("eval", "both"):
        step_idx = 2 if args.mode == "both" else 1
        _step(step_idx, total_steps, "Evaluating on held-out test pairs")
        t = time.time()
        try:
            metrics = rc_eval.evaluate(cfg, output_dir)
        except FileNotFoundError as e:
            print(f"  {red(bold('✗ FAILED'))} missing artifact: {e}. "
                  f"Run {bold('--mode train')} first.")
            return 2
        except Exception as e:
            print(f"  {red(bold('✗ FAILED'))} {e}")
            raise
        print(f"  {green('✓')} Evaluated in {bold(f'{time.time() - t:.1f}s')}")
        _print_recon_metrics(metrics)

    print(f"\n{bold('═' * W)}")
    print(f" {bold('DEPT × Reconstructor — Done')}  "
          f"{dim(f'Total {time.time() - t0:.1f}s')}")
    print(f" {cyan('Artifacts')}  {output_dir}")
    print(f"{bold('═' * W)}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
