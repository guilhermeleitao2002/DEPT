"""Stage 3 of phase 9: choose the diffusion step ``t*`` on validation flows.

Merzouk et al. define ``t*`` as the step that maximizes the NIDS's accuracy on
adversarial examples. Here it is chosen the same way, but on validation pairs rather
than on the test data the original used:

  1. **MSE curve** (all sources, cheap): for every ``t`` in the grid, the distance of
     the purified overlay to the paired clean overlay, in the encoded space, per
     channel — with the untouched adversarial flow and DEPT's reconstructor as
     reference lines. Diagnostic only.
  2. **NIDS selection** (the ML-NIDS sources, where flows are scored independently):
     every candidate ``t`` is realized onto the same validation packets and scored
     through the real NIDS; ``t*`` is the one with the highest pooled detection rate
     (ties → the smaller ``t``). If no NIDS can run, ``t*`` falls back to the lowest
     total MSE and the file says so.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..baseline.pipeline import load_reconstructor  # noqa: E402
from ..nids import NIDSRegistry, NIDSUnavailable  # noqa: E402
from ..reconstructor.evaluate import _load_ipat_stats  # noqa: E402
from .data import (  # noqa: E402
    extract_flows,
    file_id,
    load_pairs_and_split,
    rows_by_source,
    subsample_per_source,
)
from .model import load_purifier, purify_array, purify_flows  # noqa: E402
from .overlay import encode_flows  # noqa: E402
from .realize import BranchRealizer  # noqa: E402
from .train import resolve_device  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_GRID = (2, 5, 10, 20, 30, 45, 60, 90, 150)


def adversarial_pcap(cfg: dict, base: str, mode: str) -> Path:
    return Path(cfg["adversarial_pcap_dir"]) / base / f"{base}_adversarial_{mode}.pcap"


def _masked_mse(x: np.ndarray, ref: np.ndarray, valid: np.ndarray, T: int) -> dict:
    d2 = (x - ref) ** 2
    out = {}
    for name, sl in (("size", slice(0, T)), ("ipat", slice(T, 2 * T))):
        w = valid[:, sl]
        out[name] = float(d2[:, sl][w].mean()) if w.any() else None
    out["total"] = (out["size"] or 0.0) + (out["ipat"] or 0.0)
    return out


def _mse_block(x, ref, valid, modes, T) -> dict:
    res = _masked_mse(x, ref, valid, T)
    res["by_mode"] = {m: _masked_mse(x[modes == m], ref[modes == m], valid[modes == m], T)
                      for m in sorted(set(modes.tolist()))}
    return res


def run_sweep(cfg: dict, *, force: bool = False, registry=None) -> dict:
    out_dir = Path(cfg["output_dir"])
    out_path = out_dir / "t_sweep.json"
    model_id = file_id(out_dir / "diffusion.pt")
    if model_id is None:
        raise RuntimeError("no trained diffusion.pt — run the train stage first")
    if out_path.exists() and not force:
        prev = json.loads(out_path.read_text())
        if prev.get("model_id") == model_id:
            log.info("t_sweep.json exists for this model — skipping; --force re-runs it")
            return prev
        log.info("diffusion model changed since the last sweep — re-running it")

    t0 = time.time()
    scfg = dict(cfg.get("sweep", {}) or {})
    ecfg = dict(cfg.get("evaluate", {}) or {})
    grid = sorted({int(t) for t in (scfg.get("t_grid") or DEFAULT_GRID) if int(t) > 0})
    seed = int(scfg.get("seed", 42))
    device = resolve_device(ecfg.get("device", "auto"))
    num_workers = cfg.get("num_workers", "auto")
    batch_size = int(ecfg.get("batch_size", 64))
    pbs = int(ecfg.get("purify_batch_size", 4096))

    ps = load_pairs_and_split(cfg)
    model, sched, norm, saved = load_purifier(out_dir, device)
    T = int(saved["max_length"])
    recon = load_reconstructor(Path(cfg["reconstructor_model_dir"]), device)
    result: dict = {"model_id": model_id, "grid": grid, "mse": {}, "nids": {}}

    # ---- 1. MSE to the paired clean overlay (diagnostic) -------------------------
    rows = subsample_per_source(ps, ps.val_idx, scfg.get("mse_max_flows_per_source", 2000),
                                seed)
    flows = extract_flows(ps, rows, recon_model=recon, device=device,
                          batch_size=batch_size, num_workers=num_workers, max_length=T)
    dirs = [f.direction for f in flows]
    x_adv, cond, valid = encode_flows(norm, [f.adv for f in flows], dirs, T)
    x_cln, _, _ = encode_flows(norm, [f.clean for f in flows], dirs, T)
    x_dep, _, _ = encode_flows(norm, [f.dept for f in flows], dirs, T)
    modes = np.array([f.source.rsplit("_adv_", 1)[1] for f in flows], dtype=object)
    result["mse"]["n_flows"] = len(flows)
    result["mse"]["adversarial"] = _mse_block(x_adv, x_cln, valid, modes, T)
    result["mse"]["dept"] = _mse_block(x_dep, x_cln, valid, modes, T)
    for t in grid:
        xp = purify_array(model, sched, x_adv, cond, valid, t, device=device,
                          batch_size=pbs, seed=seed + t)
        result["mse"][f"t{t}"] = _mse_block(xp, x_cln, valid, modes, T)
        log.info("t=%-4d MSE size %.5f ipat %.5f (adversarial %.5f / %.5f)", t,
                 result["mse"][f"t{t}"]["size"] or 0, result["mse"][f"t{t}"]["ipat"] or 0,
                 result["mse"]["adversarial"]["size"] or 0,
                 result["mse"]["adversarial"]["ipat"] or 0)
    del flows, x_adv, x_cln, x_dep, cond, valid

    # ---- 2. t* by NIDS detection on validation flows ----------------------------
    if scfg.get("t_star") is not None:
        result["t_star"] = int(scfg["t_star"])
        result["selection"] = "fixed in config (sweep.t_star)"
    else:
        reg = registry or NIDSRegistry(cfg.get("nids", {}), Path(cfg["_repo_root"]))
        bases = set(scfg.get("nids_bases") or [])
        wanted = {s for s in set(ps.src_names.tolist()) if s.rsplit("_adv_", 1)[0] in bases}
        nrows = subsample_per_source(ps, ps.val_idx,
                                     scfg.get("nids_max_flows_per_source", 1000),
                                     seed + 1, sources=wanted)
        ipat_mean, ipat_std = _load_ipat_stats(cfg["tensors_dir"])
        branches = ["adv", "clean", "dept"] + [f"t{t}" for t in grid]
        totals = {b: [0, 0] for b in branches}
        per_source, skipped = {}, {}
        for gname, srows in sorted(rows_by_source(ps, nrows).items()):
            base, mode = gname.rsplit("_adv_", 1)
            try:
                nids = reg.get(base)
            except NIDSUnavailable as e:
                skipped[gname] = str(e)
                continue
            fl = extract_flows(ps, srows, recon_model=recon, device=device,
                               batch_size=batch_size, num_workers=num_workers, max_length=T)
            fv = {f.fid: {"adv": f.adv, "clean": f.clean, "dept": f.dept} for f in fl}
            for t in grid:
                pur = purify_flows(model, sched, norm, [f.adv for f in fl],
                                   [f.direction for f in fl], t, device=device,
                                   batch_size=pbs, seed=seed + t)
                for f, p in zip(fl, pur):
                    fv[f.fid][f"t{t}"] = p
            rz = BranchRealizer(adversarial_pcap(cfg, base, mode), fv, ipat_mean, ipat_std,
                                mask_branches=branches)
            if rz.stats.get("packets_per_branch", 0) == 0:
                skipped[gname] = rz.stats.get("error", "no realized packets")
                continue
            per_source[gname] = {"realize": rz.stats}
            for b in branches:
                pk, ky = rz.packets(b)
                res = nids.score_flows(pk, ky)
                n, det = len(res), int(sum(r[2] for r in res))
                per_source[gname][b] = {"n": n, "detected": det}
                totals[b][0] += n
                totals[b][1] += det
            log.info("%-24s detection adv %.3f  clean %.3f  dept %.3f  best-t %s", gname,
                     _rate(per_source[gname]["adv"]), _rate(per_source[gname]["clean"]),
                     _rate(per_source[gname]["dept"]),
                     max(grid, key=lambda t: (_rate(per_source[gname][f"t{t}"]), -t)))
        rates = {b: (d / n if n else None) for b, (n, d) in totals.items()}
        result["nids"] = {"detection_rate": rates, "n_flows": {b: n for b, (n, _d) in
                                                              totals.items()},
                          "per_source": per_source, "skipped": skipped}
        if all(rates[f"t{t}"] is not None for t in grid):
            result["t_star"] = max(grid, key=lambda t: (rates[f"t{t}"], -t))
            result["selection"] = "max pooled NIDS detection on validation flows"
        else:
            result["t_star"] = min(grid, key=lambda t: (result["mse"][f"t{t}"]["total"], t))
            result["selection"] = "min total MSE (no NIDS could be run — see nids.skipped)"
            log.warning("no NIDS scores for the sweep — t* chosen by MSE instead")

    result["seconds"] = round(time.time() - t0, 1)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    _plot_sweep(result, out_dir / "t_sweep.png")
    log.info("t* = %s (%s)", result["t_star"], result["selection"])
    return result


def _rate(d: dict) -> float:
    return d["detected"] / d["n"] if d.get("n") else float("nan")


def _plot_sweep(res: dict, path: Path) -> None:
    grid = res["grid"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for ch in ("size", "ipat"):
        ax.plot(grid, [res["mse"][f"t{t}"][ch] for t in grid], marker="o",
                label=f"diffusion ({ch})")
        ax.axhline(res["mse"]["adversarial"][ch] or 0, ls=":", lw=1,
                   color="C0" if ch == "size" else "C1", label=f"no purification ({ch})")
        ax.axhline(res["mse"]["dept"][ch] or 0, ls="--", lw=1,
                   color="C0" if ch == "size" else "C1", label=f"DEPT ({ch})")
    ax.set_xscale("log")
    ax.set_xlabel("diffusion steps t")
    ax.set_ylabel("MSE to the clean flow (encoded)")
    ax.set_title("Distance to the clean flow (validation)")
    ax.legend(fontsize=7)
    ax = axes[1]
    rates = res.get("nids", {}).get("detection_rate", {})
    if rates:
        ax.plot(grid, [rates.get(f"t{t}") for t in grid], marker="o", label="diffusion")
        for b, ls in (("adv", ":"), ("dept", "--"), ("clean", "-.")):
            if rates.get(b) is not None:
                ax.axhline(rates[b], ls=ls, lw=1, color="k", label=b)
        ax.axvline(res["t_star"], color="C3", lw=1, label=f"t* = {res['t_star']}")
        ax.set_xscale("log")
        ax.set_xlabel("diffusion steps t")
        ax.set_ylabel("NIDS detection rate (pooled)")
        ax.set_title("NIDS detection (validation, ML-NIDS sources)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


__all__ = ["run_sweep", "adversarial_pcap", "DEFAULT_GRID"]
