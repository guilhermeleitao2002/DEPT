"""Per-flow verdicts from DEPT's detector (the meta-learner) over a chosen flow set.

Runs the meta-learner over all flows of the given adversarial sources (not
only the test split) and returns the last-packet ``P(adversarial)`` per
``(source, flow_id)``. It reuses the meta-learner's token pipeline; the token
cache is written under the baseline output dir.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..common.dataset import load_source_meta
from ..common.runtime import resolve_num_workers
from ..metalearner.dataset import (
    MetaLearnerDataset,
    collate_padded,
    prepare_split_caches,
)
from ..metalearner.model import MetaLearner
from ..tcn.evaluate import _load_dataset_index, _reanchor_folder

log = logging.getLogger(__name__)


def _resolve_device(cfg: str) -> torch.device:
    if cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg)


def _sample_indices_for_sources(rf_dir: Path, tcn_dir: Path,
                                sources: set[str]) -> tuple[np.ndarray, np.ndarray]:
    """RF (per-checkpoint) and TCN (per-flow) row indices for the given sources.

    ``dataset.npz`` is a zip — indexing one key (``sources``) reads only that member,
    not the multi-GB ``X``. The TCN index stores sources vocab-encoded, so decode it
    with the TCN's own loader.
    """
    rf_src = np.load(rf_dir / "dataset.npz")["sources"].astype(str)
    tcn_src = _load_dataset_index(tcn_dir / "dataset_index.npz")["sources"].astype(str)
    rf_idx = np.where(np.isin(rf_src, list(sources)))[0].astype(np.int64)
    tcn_idx = np.where(np.isin(tcn_src, list(sources)))[0].astype(np.int64)
    return rf_idx, tcn_idx


def test_flow_keys(tcn_dir: Path, sources: set[str]) -> set[tuple[str, str]]:
    """``(source, flow_id_str)`` of the held-out **test** flows among ``sources``.

    Lets the caller report the generalization-valid subset (flows the detector /
    purifier never trained on) alongside the full-stream deployment numbers.
    """
    idx = _load_dataset_index(tcn_dir / "dataset_index.npz")
    test_idx = idx["used_test_idx"].astype(np.int64)
    src_all = idx["sources"].astype(str)
    local_all = idx["local_idx"].astype(np.int64)
    keep = test_idx[np.isin(src_all[test_idx], list(sources))]
    src_folders = idx["source_folders"]  # {source: Path} (may hold stale absolute paths)
    repo_root = Path(tcn_dir).resolve().parents[2]  # <repo>/data/models/tcn -> <repo>
    lookup: dict[str, dict[int, str]] = {}
    out: set[tuple[str, str]] = set()
    for row in keep:
        s = str(src_all[row])
        if s not in lookup:
            folder = _reanchor_folder(src_folders[s], repo_root)
            _y, fids, _ls, locals_meta, _ = load_source_meta(Path(folder), s)
            lookup[s] = {int(li): str(fid) for li, fid in zip(locals_meta, fids)}
        out.add((s, lookup[s][int(local_all[row])]))
    return out


def detector_verdicts(cfg: dict, sources: set[str]) -> dict[tuple[str, str], float]:
    """Run the meta-learner over all flows of ``sources`` → ``{(src, fid): P(adv)}``.

    ``cfg`` supplies the model dirs (``rf_model_dir``, ``tcn_model_dir``,
    ``metalearner_model_dir``), ``tensors_dir``, ``adversarial_dir``, a
    ``detector_cache_dir`` for the token cache, and an optional ``token`` block +
    ``device`` (matching the meta-learner's training config).
    """
    rf_dir = Path(cfg["rf_model_dir"])
    tcn_dir = Path(cfg["tcn_model_dir"])
    meta_dir = Path(cfg["metalearner_model_dir"])
    cache_out = Path(cfg["detector_cache_dir"])

    saved = json.loads((meta_dir / "metalearner_config.json").read_text())
    model_cfg = saved["model"]
    train_cfg = saved["training"]
    device = _resolve_device(str(cfg.get("device", train_cfg.get("device", "auto"))))

    rf_idx, tcn_idx = _sample_indices_for_sources(rf_dir, tcn_dir, sources)
    if len(tcn_idx) == 0:
        return {}

    token_cfg = cfg.get("token", {}) or {}
    # A cache split name keyed by the source set → re-runs of the same scope reuse it.
    tag = hashlib.sha1("|".join(sorted(sources)).encode()).hexdigest()[:10]
    split_name = f"detector_{tag}"
    caches = prepare_split_caches(
        output_dir=cache_out,
        rf_dir=rf_dir, tcn_dir=tcn_dir,
        tensors_dir=Path(cfg["tensors_dir"]),
        adversarial_dir=Path(cfg["adversarial_dir"]),
        splits={split_name: {"rf_sample_indices": rf_idx,
                             "tcn_sample_indices": tcn_idx}},
        rf_age_clip=int(token_cfg.get("rf_age_clip", 30)),
        pkt_len_norm=float(token_cfg.get("pkt_len_norm", 1500.0)),
        max_length=int(train_cfg.get("max_length", 200)),
    )
    cache = caches[split_name]

    model = MetaLearner(
        d_model=int(model_cfg.get("d_model", 64)),
        n_layers=int(model_cfg.get("n_layers", 2)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        dim_feedforward=int(model_cfg.get("dim_feedforward", 128)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        num_classes=int(model_cfg.get("num_classes", 2)),
        max_seq_len=int(model_cfg.get("max_seq_len", 200)),
    ).to(device)
    model.load_state_dict(torch.load(meta_dir / "metalearner.pt", map_location=device))
    model.eval()

    num_workers = resolve_num_workers(train_cfg.get("num_workers", "auto"))
    loader = DataLoader(
        MetaLearnerDataset(cache),
        batch_size=int(train_cfg.get("batch_size", 128)),
        shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=collate_padded,
    )

    last_probs: list[float] = []
    with torch.inference_mode():
        for tokens, _labels, lengths, mask in loader:
            tokens = tokens.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            logits = model(tokens, key_padding_mask=mask)
            probs = torch.softmax(logits.float(), dim=2)[:, :, 1].detach().cpu().numpy()
            for b in range(probs.shape[0]):
                L = int(lengths[b].item())
                last_probs.append(float(probs[b, L - 1]))

    # Cache flow order matches join_streams order == flow_keys.json.
    flow_keys = json.loads((cache.cache_dir / "flow_keys.json").read_text())
    if len(flow_keys) != len(last_probs):
        raise RuntimeError(
            f"detector verdict/flow-key length mismatch: {len(last_probs)} vs {len(flow_keys)}")
    return {(str(s), str(f)): p for (s, f), p in zip(flow_keys, last_probs)}


__all__ = ["detector_verdicts", "test_flow_keys"]
