#!/usr/bin/env python3
"""
ADEPT — Reconstructor (Purification) Transformer (train + eval)
==============================================================

Trains a small causal residual denoiser that purifies adversarially-perturbed
flows back onto the clean-traffic manifold, and evaluates it on the held-out
test pairs with three metric groups:

  * Reconstruction quality — MSE on perturbed dims, before vs after.
  * Purification gain       — TCN P(adversarial) on adv / recon / clean.
  * Protocol compliance     — tshark on the realized purified packets.

Usage
-----
    python scripts/reconstructor/run_reconstructor.py
    python scripts/reconstructor/run_reconstructor.py --mode train
    python scripts/reconstructor/run_reconstructor.py --mode eval

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
def magenta(t): return _c("35", t)


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

    if metrics.get("tcn_used"):
        pur = ov["purification"]
        print(f"\n {bold('Purification gain (through TCN)')}")
        print(f"   {cyan('P(adv|adv)')} {_fmt(pur['p_adv_on_adv'])}   "
              f"{cyan('P(adv|recon)')} {_fmt(pur['p_adv_on_recon'])}   "
              f"{cyan('P(adv|clean)')} {_fmt(pur['p_adv_on_clean'])}")
        print(f"   {cyan('gain')} {bold(green(_fmt(pur['gain'])))}   "
              f"{cyan('residual gap')} {_fmt(pur['residual_gap'])}")

    kit = metrics.get("kitsune", {})
    kov = kit.get("overall") if isinstance(kit, dict) else None
    if kov:
        print(f"\n {bold('Purification efficacy (through Kitsune — independent NIDS)')}")
        print(f"   {dim('anomaly score restored: adv (evaded) → purified → clean (oracle)')}")
        print(f"   {cyan('mean RMSE')}  adv {_fmt(kov.get('mean_rmse_adv'), 4)}  "
              f"purified {_fmt(kov.get('mean_rmse_purified'), 4)}  "
              f"clean {_fmt(kov.get('mean_rmse_clean'), 4)}")
        rr = kov.get('rmse_restoration_ratio')
        print(f"   {cyan('RMSE restoration')} "
              f"{bold(green((_fmt(rr * 100, 1) + '%') if rr is not None else '—'))}   "
              f"{dim('(fraction of the adv→clean anomaly gap recovered)')}")
        print(f"   {cyan('detect(adv)')} {_fmt(kov.get('detection_rate_adv'))}   "
              f"{cyan('detect(purified)')} {_fmt(kov.get('detection_rate_purified'))}   "
              f"{cyan('detect(clean)')} {_fmt(kov.get('detection_rate_clean'))}   "
              f"{dim('(threshold-dependent — see README)')}")
    elif isinstance(kit, dict) and kit.get("skipped"):
        print(f"\n {bold('Purification efficacy (Kitsune)')}  "
              f"{yellow('skipped:')} {dim(str(kit['skipped']))}")

    comp = ov["protocol_compliance"]
    print(f"\n {bold('Protocol compliance')}")
    print(f"   {cyan('tshark')} {_fmt(comp.get('tshark_rate'))}")

    per = metrics.get("per_source", {})
    if per:
        print(f"\n {bold('Per-source')}  {dim('(' + str(len(per)) + ' variant(s))')}")
        hdr = (f"   {'variant':<24} {'mse_after':>10} "
               f"{'P_recon':>8} {'gain':>7} "
               f"{'K_adv':>7} {'K_puri':>7} {'K_clean':>7} {'tshark':>7}")
        print(cyan(hdr))
        for name in sorted(per.keys()):
            s = per[name]
            r = s["reconstruction"]
            p = s["purification"]
            c = s["protocol_compliance"]
            k = s.get("kitsune", {})
            print(f"   {name:<24} "
                  f"{_fmt(r.get('perturbed_mse_after'), 5):>10} "
                  f"{_fmt(p.get('p_adv_on_recon'), 3):>8} "
                  f"{_fmt(p.get('gain'), 3):>7} "
                  f"{_fmt(k.get('detection_rate_adv'), 3):>7} "
                  f"{_fmt(k.get('detection_rate_purified'), 3):>7} "
                  f"{_fmt(k.get('detection_rate_clean'), 3):>7} "
                  f"{_fmt(c.get('tshark_rate'), 3):>7}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ADEPT reconstructor — train + evaluate the purification transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["train", "eval", "both"], default="both")
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

    _section(f"ADEPT — Reconstructor  {dim('(mode: ' + args.mode + ')')}")
    _row("Config", str(cfg_path))
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
    print(f" {bold('ADEPT × Reconstructor — Done')}  "
          f"{dim(f'Total {time.time() - t0:.1f}s')}")
    print(f" {cyan('Artifacts')}  {output_dir}")
    print(f"{bold('═' * W)}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
