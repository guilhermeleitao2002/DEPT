"""CLI entry point for training and evaluating the ADEPT Random Forest."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

# Make `src` importable without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.rf import evaluate as rf_eval  # noqa: E402
from src.models.rf import train as rf_train  # noqa: E402


def _resolve(path_str: str, repo_root: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else repo_root / p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/rf.yaml", help="Path to rf.yaml")
    parser.add_argument("--mode", choices=["train", "eval", "both"], default="both")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("run_rf")

    config_path = _resolve(args.config, REPO_ROOT)
    config = yaml.safe_load(config_path.read_text())
    tensors_root = _resolve(config["tensors_dir"], REPO_ROOT)
    output_dir = _resolve(config["output_dir"], REPO_ROOT)

    if args.mode in ("train", "both"):
        info = rf_train.train(config, tensors_root, output_dir)
        log.info("train complete. best_params=%s", info["best_params"])

    if args.mode in ("eval", "both"):
        metrics = rf_eval.evaluate(output_dir)
        overall = metrics["overall"]
        log.info(
            "eval: acc=%.4f macro_f1=%.4f roc_auc=%s pr_auc=%s",
            overall["accuracy"],
            overall["macro_f1"],
            overall["roc_auc"],
            overall["pr_auc"],
        )
        print(json.dumps(overall, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
