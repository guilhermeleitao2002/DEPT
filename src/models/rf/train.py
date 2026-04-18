"""Train the Random Forest base learner for ADEPT."""
from __future__ import annotations

import itertools
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from .dataset import RFDataset, Split, assemble, flow_level_split, split_summary


log = logging.getLogger(__name__)


def _build_rf(defaults: dict, overrides: dict | None = None) -> RandomForestClassifier:
    params = dict(defaults)
    if overrides:
        params.update(overrides)
    return RandomForestClassifier(**params)


def _grid(grid_cfg: dict) -> list[dict]:
    keys = ["n_estimators", "max_depth", "min_samples_leaf"]
    vals = [grid_cfg[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


def grid_search(
    dataset: RFDataset,
    split: Split,
    defaults: dict,
    grid_cfg: dict,
) -> tuple[dict, list[dict]]:
    """Fit each grid point on train, score macro-F1 on val. Return (best_params, all_results)."""
    X_tr, y_tr = dataset.X[split.train_idx], dataset.y[split.train_idx]
    X_va, y_va = dataset.X[split.val_idx], dataset.y[split.val_idx]

    results = []
    best = None
    for params in _grid(grid_cfg):
        t0 = time.time()
        clf = _build_rf(defaults, params)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_va)
        score = float(f1_score(y_va, y_pred, average="macro"))
        elapsed = time.time() - t0
        log.info("grid %s -> macro_f1=%.4f (%.1fs)", params, score, elapsed)
        entry = {"params": params, "val_macro_f1": score, "fit_seconds": elapsed}
        results.append(entry)
        if best is None or score > best["val_macro_f1"]:
            best = entry
    assert best is not None
    return best["params"], results


def fit_final(
    dataset: RFDataset,
    split: Split,
    defaults: dict,
    best_params: dict,
) -> RandomForestClassifier:
    X_tr, y_tr = dataset.X[split.train_idx], dataset.y[split.train_idx]
    X_va, y_va = dataset.X[split.val_idx], dataset.y[split.val_idx]
    # Final fit uses train+val so no data is wasted before evaluating on test.
    X_fit = np.concatenate([X_tr, X_va], axis=0)
    y_fit = np.concatenate([y_tr, y_va], axis=0)
    clf = _build_rf(defaults, best_params)
    t0 = time.time()
    clf.fit(X_fit, y_fit)
    log.info("final fit on %d windows in %.1fs", len(y_fit), time.time() - t0)
    return clf


def train(config: dict, tensors_root: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = assemble(
        tensors_dir=tensors_root,
        attacks=config["attacks"],
        benign_cap_ratio=float(config["benign_cap_ratio"]),
        seed=int(config["split"]["seed"]),
    )
    log.info("assembled dataset: X=%s y=%s", dataset.X.shape, dataset.y.shape)

    split = flow_level_split(
        dataset,
        train_frac=float(config["split"]["train_frac"]),
        val_frac=float(config["split"]["val_frac"]),
        test_frac=float(config["split"]["test_frac"]),
        seed=int(config["split"]["seed"]),
    )
    summary = split_summary(dataset, split)
    log.info("split summary: %s", json.dumps(summary, indent=2))

    defaults = dict(config["rf_defaults"])
    grid_cfg = config.get("grid_search", {}) or {}
    grid_results: list[dict] = []
    if grid_cfg.get("enabled", False):
        best_params, grid_results = grid_search(dataset, split, defaults, grid_cfg)
    else:
        best_params = {}
    log.info("best params: %s", best_params)

    clf = fit_final(dataset, split, defaults, best_params)

    # Persist artifacts.
    joblib.dump(clf, output_dir / "rf.joblib")
    np.savez(
        output_dir / "split.npz",
        train_idx=split.train_idx,
        val_idx=split.val_idx,
        test_idx=split.test_idx,
    )
    # Full dataset tensors persisted so evaluate.py doesn't re-assemble (cheap vs. determinism).
    np.savez(
        output_dir / "dataset.npz",
        X=dataset.X,
        y=dataset.y,
        attacks=dataset.attacks.astype(str),
        label_str=dataset.label_str.astype(str),
    )
    (output_dir / "feature_names.json").write_text(json.dumps(dataset.feature_names))
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "training_info.json").write_text(
        json.dumps(
            {
                "best_params": best_params,
                "grid_results": grid_results,
                "defaults": defaults,
                "config": {k: v for k, v in config.items() if k != "rf_defaults"},
            },
            indent=2,
            default=str,
        )
    )
    log.info("artifacts written to %s", output_dir)
    return {"best_params": best_params, "split_summary": summary}
