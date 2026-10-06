"""Stages 1–2 of phase 9: build the purifier's training data, then train it.

Training data follows the original method: **clean flows only**, no adversarial
examples. To keep the comparison with DEPT fair, these are the clean sides of the
reconstructor's own training pairs, one entry per distinct original flow — the same
flows DEPT's purifier learned its targets from. The validation loss uses clean flows of
validation pairs whose original flow is in no training pair.

Training itself is the original's (ε-prediction MSE at a uniformly drawn step, AdamW),
with two differences: mini-batches instead of the original's full-batch epochs (the
data is several times larger than UNSW-NB15), and the loss averaged over real packets
only (flows are padded to a fixed length).
"""
from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ..reconstructor.evaluate import _load_ipat_stats  # noqa: E402
from .data import (  # noqa: E402
    file_id,
    load_arrays,
    load_pairs_and_split,
    read_clean_overlays,
    save_arrays,
    train_flow_keys,
    unique_clean_rows,
)
from .model import build_model, build_schedule  # noqa: E402
from .overlay import OverlayNorm, encode_dense, fit_norm  # noqa: E402

log = logging.getLogger(__name__)


def resolve_device(name) -> torch.device:
    name = str(name or "auto")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _cap_per_base(ps, rows: np.ndarray, cap: int | None, seed: int) -> np.ndarray:
    if not cap:
        return rows
    rng = np.random.default_rng(seed)
    bases = np.array([str(ps.pairs.base[r]) for r in rows], dtype=object)
    keep = []
    for b in sorted(set(bases.tolist())):
        sel = rows[bases == b]
        if len(sel) > int(cap):
            sel = rng.choice(sel, size=int(cap), replace=False)
        keep.append(sel)
    return np.sort(np.concatenate(keep))


def build_caches(cfg: dict, *, force: bool = False) -> dict:
    """Stage ``overlays``: clean training / validation overlays + the IPAT scale."""
    out_dir = Path(cfg["output_dir"])
    cache = out_dir / "cache"
    summary_path = cache / "summary.json"
    if summary_path.exists() and (out_dir / "overlay_norm.json").exists() and not force:
        log.info("overlay caches already built (%s) — skipping; --force rebuilds", cache)
        return json.loads(summary_path.read_text())

    t0 = time.time()
    ocfg = dict(cfg.get("overlay", {}) or {})
    T = int(cfg.get("max_length", 200))
    seed = int(cfg.get("training", {}).get("seed", 42))
    ps = load_pairs_and_split(cfg)

    rows = unique_clean_rows(ps, ps.train_idx)
    rows = _cap_per_base(ps, rows, ocfg.get("train_max_flows_per_base"), seed)
    log.info("reading %d distinct clean training flows", len(rows))
    tr = read_clean_overlays(ps.pairs, rows, T, batch_size=int(ocfg.get("batch_size", 256)),
                             num_workers=ocfg.get("num_workers", "auto"))
    ipat_mean, ipat_std = _load_ipat_stats(cfg["tensors_dir"])
    norm = fit_norm(tr["ipat_z"], tr["length"], ipat_mean=ipat_mean, ipat_std=ipat_std,
                    size_scale=float(ocfg.get("size_scale", 1500)),
                    quantile=float(ocfg.get("ipat_quantile", 0.999)))
    save_arrays(cache / "train_clean", tr, {"split": ps.origin})

    seen = train_flow_keys(ps)
    vrows = np.array([r for r in unique_clean_rows(ps, ps.val_idx)
                      if (str(ps.pairs.base[r]), str(ps.pairs.flow_id[r])) not in seen],
                     dtype=np.int64)
    if len(vrows) == 0:
        log.warning("every validation flow also appears in training — using them anyway")
        vrows = unique_clean_rows(ps, ps.val_idx)
    vmax = ocfg.get("val_max_flows", 20000)
    if vmax and len(vrows) > int(vmax):
        vrows = np.sort(np.random.default_rng(seed).choice(vrows, size=int(vmax),
                                                           replace=False))
    va = read_clean_overlays(ps.pairs, vrows, T, batch_size=int(ocfg.get("batch_size", 256)),
                             num_workers=ocfg.get("num_workers", "auto"))
    save_arrays(cache / "val_clean", va, {"split": ps.origin})

    (out_dir / "overlay_norm.json").write_text(json.dumps(norm.to_dict(), indent=2))
    summary = {
        "split": ps.origin,
        "n_train_flows": int(len(rows)),
        "n_train_packets": int(tr["length"].astype(np.int64).sum()),
        "n_val_flows": int(len(vrows)),
        "max_length": T,
        "norm": norm.to_dict(),
        "seconds": round(time.time() - t0, 1),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def _to_device(x: np.ndarray, d: np.ndarray, length: np.ndarray, device):
    return (torch.from_numpy(x).to(device=device, dtype=torch.float16),
            torch.from_numpy(d).to(device),
            torch.from_numpy(length).to(device))


def train(cfg: dict, *, force: bool = False) -> dict:
    """Stage ``train``: fit the noise predictor; keeps the best step by validation loss."""
    out_dir = Path(cfg["output_dir"])
    cache_id = file_id(out_dir / "cache" / "summary.json")
    if cache_id is None:
        raise RuntimeError("no overlay cache — run the overlays stage first")
    saved_path = out_dir / "diffusion_config.json"
    if (out_dir / "diffusion.pt").exists() and saved_path.exists() and not force:
        saved = json.loads(saved_path.read_text())
        if saved.get("cache_id") == cache_id:
            log.info("diffusion.pt already trained on this cache — skipping; --force retrains")
            return saved
        log.info("overlay cache changed since the last training — retraining")

    mcfg = dict(cfg.get("model", {}) or {})
    tcfg = dict(cfg.get("training", {}) or {})
    T = int(cfg.get("max_length", 200))
    seed = int(tcfg.get("seed", 42))
    steps = int(tcfg.get("steps", 150000))
    B = int(tcfg.get("batch_size", 4096))
    val_every = int(tcfg.get("val_every", 2500))
    log_every = int(tcfg.get("log_every", 500))
    device = resolve_device(tcfg.get("device", "auto"))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    norm = OverlayNorm.from_dict(json.loads((out_dir / "overlay_norm.json").read_text()))
    tr = load_arrays(out_dir / "cache" / "train_clean")
    va = load_arrays(out_dir / "cache" / "val_clean")
    x_tr, d_tr, len_tr = encode_dense(norm, tr["tl"], tr["ipat_z"], tr["dir"], tr["length"])
    x_va, d_va, len_va = encode_dense(norm, va["tl"], va["ipat_z"], va["dir"], va["length"])
    del tr, va
    N = int(x_tr.shape[0])
    if N == 0:
        raise RuntimeError("no training flows in the overlay cache")

    data_dev = device
    try:
        X, D, LEN = _to_device(x_tr, d_tr, len_tr, device)
    except RuntimeError as e:  # CUDA OOM → keep the data on the host
        log.warning("training data does not fit on %s (%s) — batching from host memory",
                    device, str(e).splitlines()[0])
        data_dev = torch.device("cpu")
        X, D, LEN = _to_device(x_tr, d_tr, len_tr, data_dev)
    if len(x_va) == 0:   # no held-out clean flows: monitor on a slice of training flows
        log.warning("empty validation set — monitoring the loss on 2% of training flows")
        k = max(1, N // 50)
        x_va, d_va, len_va = x_tr[:k], d_tr[:k], len_tr[:k]
    Xv, Dv, LENv = _to_device(x_va, d_va, len_va, device)
    del x_tr, d_tr, len_tr, x_va, d_va, len_va

    model = build_model(mcfg, T).to(device)
    sched = build_schedule(mcfg, device)
    use_cond = model.cond_dim > 0
    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg.get("lr", 1e-4)))
    g_idx = torch.Generator(device=data_dev).manual_seed(seed)
    g_noise = torch.Generator(device=device).manual_seed(seed + 1)
    ar_data = torch.arange(T, device=data_dev)
    ar_dev = torch.arange(T, device=device)

    def batch_from(Xs, Ds, Ls, idx, ar):
        m = ar[None, :] < Ls[idx, None]
        x0 = Xs[idx].float()
        cond = torch.cat([Ds[idx].float(), m.float()], dim=1)
        valid = torch.cat([m, m], dim=1)
        return (x0.to(device, non_blocking=True), cond.to(device, non_blocking=True),
                valid.to(device, non_blocking=True))

    # Fixed validation draws, so every check scores the same noise.
    g_val = torch.Generator(device=device).manual_seed(seed + 2)
    n_val = int(Xv.shape[0])
    t_val = sched.sample_timesteps(n_val, generator=g_val, device=device)
    eps_val = torch.randn((n_val, 2 * T), generator=g_val, device=device)

    @torch.no_grad()
    def val_loss() -> float:
        model.eval()
        num = torch.zeros((), device=device)
        den = torch.zeros((), device=device)
        for s in range(0, n_val, B):
            idx = torch.arange(s, min(s + B, n_val), device=device)
            x0, cond, valid = batch_from(Xv, Dv, LENv, idx, ar_dev)
            a_hat = sched.alpha_hat[t_val[idx]][:, None]
            x_t = torch.sqrt(a_hat) * x0 + torch.sqrt(1.0 - a_hat) * eps_val[idx]
            pred = model(x_t, t_val[idx], cond if use_cond else None)
            w = valid.float()
            num += ((pred - eps_val[idx]) ** 2 * w).sum()
            den += w.sum()
        model.train()
        return float(num / den.clamp_min(1.0))

    log.info("training on %d flows (%s), %d steps × batch %d, device=%s, params=%d",
             N, "data on " + str(data_dev), steps, B, device,
             sum(p.numel() for p in model.parameters()))
    curves = {"step": [], "train_loss": [], "val_loss": []}
    best = {"val": float("inf"), "step": 0, "state": None}
    running = torch.zeros((), device=device)   # summed on-device: no sync per step
    n_run = 0
    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        idx = torch.randint(N, (B,), generator=g_idx, device=data_dev)
        x0, cond, valid = batch_from(X, D, LEN, idx, ar_data)
        t = sched.sample_timesteps(B, generator=g_noise, device=device)
        x_t, eps = sched.noise_data(x0, t, generator=g_noise)
        pred = model(x_t, t, cond if use_cond else None)
        w = valid.float()
        loss = ((pred - eps) ** 2 * w).sum() / w.sum().clamp_min(1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running += loss.detach()
        n_run += 1
        if step % log_every == 0:
            rate = step / max(time.time() - t0, 1e-9)
            log.info("step %d/%d  loss %.5f  (%.1f steps/s, eta %.0f min)", step, steps,
                     float(running) / n_run, rate, (steps - step) / max(rate, 1e-9) / 60)
        if step % val_every == 0 or step == steps:
            vl = val_loss()
            curves["step"].append(step)
            curves["train_loss"].append(float(running) / max(n_run, 1))
            curves["val_loss"].append(vl)
            running.zero_()
            n_run = 0
            if vl < best["val"]:
                best = {"val": vl, "step": step,
                        "state": copy.deepcopy({k: v.detach().cpu()
                                                for k, v in model.state_dict().items()})}
            log.info("  val loss %.5f (best %.5f @ %d)", vl, best["val"], best["step"])

    torch.save(best["state"], out_dir / "diffusion.pt")
    info = {
        "method": "Merzouk et al., DBSec 2025 (github.com/mamerzouk/adversarial-purification)",
        "cache_id": cache_id,
        "model": mcfg,
        "max_length": T,
        "norm": norm.to_dict(),
        "training": {**tcfg, "device": str(device), "n_train_flows": N,
                     "n_val_flows": n_val, "best_step": best["step"],
                     "best_val_loss": best["val"],
                     "seconds": round(time.time() - t0, 1),
                     "n_parameters": int(sum(p.numel() for p in model.parameters()))},
    }
    (out_dir / "diffusion_config.json").write_text(json.dumps(info, indent=2))
    (out_dir / "training_curves.json").write_text(json.dumps(curves, indent=2))
    _plot_curves(curves, out_dir / "training_curves.png")
    return info


def _plot_curves(curves: dict, path: Path) -> None:
    if not curves["step"]:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curves["step"], curves["train_loss"], label="train (running mean)")
    ax.plot(curves["step"], curves["val_loss"], label="validation")
    ax.set_xlabel("step")
    ax.set_ylabel("noise-prediction MSE (real packets)")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Diffusion purifier training")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


__all__ = ["build_caches", "train", "resolve_device"]
