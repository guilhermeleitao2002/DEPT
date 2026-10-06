"""Stage 4 of phase 9: DEPT's purifier vs the diffusion purifier, through the real NIDS.

For every attack × mode this realizes, onto the same adversarial packets and with one
shared keep-mask, the streams below and scores each through the NIDS the attack was
built to evade (the phase-8 registry):

  ``clean``            original attack (ceiling)
  ``adversarial``      TANTGen's perturbation (floor)
  ``dept_all``         DEPT's reconstructor applied to every flow
  ``diffusion_all``    the diffusion purifier applied to every flow (as published: no gate)
  ``dept_gated``       DEPT's meta-learner gate + reconstructor (the deployed system)
  ``diffusion_gated``  the same gate + the diffusion purifier

Metrics are reported on subsets of flows:

  * ``test``   — the reconstructor's test pairs: neither purifier trained on them. The
    headline. ML-NIDS streams contain only these flows; Kitsune streams are the full
    source (Kitsune's features are volumetric) with the metric restricted afterwards.
  * ``strict`` — test pairs whose original flow is in no training pair under any mode.
  * ``gated``  — test pairs that are also in the meta-learner's (TCN) test split, so
    both the gate and the purifiers are held out; only here are the gated streams scored.
  * ``full``   — Kitsune only: every flow of the full stream (phase 8's deployment view).

Each source's result is written to ``per_source/<source>.json`` as soon as it is done,
and a re-run skips finished sources (``--force`` recomputes them).
"""
from __future__ import annotations

import json
import logging
import time
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ..baseline.pipeline import load_reconstructor  # noqa: E402
from ..nids import NIDSRegistry, NIDSUnavailable  # noqa: E402
from ..nids.metrics import restoration_ratio  # noqa: E402
from ..reconstructor.dataset import PairedFlowDataset, collate_pairs  # noqa: E402
from ..reconstructor.evaluate import _load_ipat_stats  # noqa: E402
from ..reconstructor.realize import (  # noqa: E402
    decode_total_len_batch,
    tshark_available,
    tshark_compliance,
    write_pcap,
)
from .data import (  # noqa: E402
    extract_flows,
    file_id,
    load_pairs_and_split,
    rows_by_source,
    strict_test_keys,
)
from .model import load_purifier, purify, purify_flows  # noqa: E402
from .overlay import encode_flows  # noqa: E402
from .realize import BranchRealizer  # noqa: E402
from .sweep import adversarial_pcap  # noqa: E402
from .train import resolve_device  # noqa: E402

log = logging.getLogger(__name__)

CORE = ("clean", "adversarial", "dept_all", "diffusion_all")
GATED = ("dept_gated", "diffusion_gated")
BRANCH = {"clean": "clean", "adversarial": "adv", "dept_all": "dept",
          "diffusion_all": "diffusion", "dept_gated": "dept_gated",
          "diffusion_gated": "diffusion_gated"}
MASK_BRANCHES = ("adv", "clean", "dept", "diffusion")


def resolve_t_star(cfg: dict) -> int:
    fixed = (cfg.get("sweep", {}) or {}).get("t_star")
    if fixed is not None:
        return int(fixed)
    path = Path(cfg["output_dir"]) / "t_sweep.json"
    if not path.exists():
        raise RuntimeError(f"{path} not found — run the sweep stage first "
                           "(or set sweep.t_star in the config)")
    return int(json.loads(path.read_text())["t_star"])


def _gate(cfg: dict, target: list[str], test_fids: dict[str, set]):
    """Meta-learner verdicts + the gated subset, or ``(None, {}, info)`` if unavailable."""
    try:
        from ..baseline.detector_infer import detector_verdicts, test_flow_keys

        dcfg = {k: cfg[k] for k in ("rf_model_dir", "tcn_model_dir", "metalearner_model_dir",
                                    "tensors_dir", "adversarial_dir", "detector_cache_dir")}
        dcfg["token"] = cfg.get("token", {}) or {}
        dcfg["device"] = (cfg.get("evaluate", {}) or {}).get("device", "auto")
        verdicts = detector_verdicts(dcfg, set(target))
        tcn_test = test_flow_keys(Path(cfg["tcn_model_dir"]), set(target))
    except Exception as e:  # noqa: BLE001 — the gate is optional
        log.warning("gate unavailable (%s: %s) — gated streams skipped",
                    type(e).__name__, e)
        return None, {}, {"error": f"{type(e).__name__}: {e}"}
    gated = {s: {f for f in test_fids.get(s, set()) if (s, f) in tcn_test} for s in target}
    return verdicts, gated, {"n_verdicts": len(verdicts),
                             "n_gated_flows": int(sum(len(v) for v in gated.values()))}


def _counts(results: list, fids: set | None) -> dict:
    """Detection counts over per-flow results, optionally restricted to ``fids``."""
    n = det = 0
    s = 0.0
    for key, score, d in results:
        if fids is not None and key not in fids:
            continue
        n += 1
        det += int(d)
        s += float(score)
    return {"n_flows": n, "n_detected": det, "sum_score": s}


def derive(counts: dict, with_mean: bool = True) -> dict:
    """Rates and comparisons from ``{regime: counts}``."""
    dr = {r: (c["n_detected"] / c["n_flows"]) if c["n_flows"] else None
          for r, c in counts.items()}
    asr = {r: (1.0 - v) if v is not None else None for r, v in dr.items()}
    out = {"n_flows": {r: c["n_flows"] for r, c in counts.items()},
           "detection_rate": dr, "asr": asr,
           "asr_reduction": {}, "restoration_ratio_detection": {}}
    for r in counts:
        if r in ("clean", "adversarial"):
            continue
        a, b = asr.get("adversarial"), asr.get(r)
        out["asr_reduction"][r] = (a - b) if (a is not None and b is not None) else None
        out["restoration_ratio_detection"][r] = restoration_ratio(
            dr.get("adversarial"), dr.get(r), dr.get("clean"))
    if with_mean:
        out["mean_score"] = {r: (c["sum_score"] / c["n_flows"]) if c["n_flows"] else None
                             for r, c in counts.items()}
    return out


def _accumulate(acc: dict, counts: dict) -> None:
    for r, c in counts.items():
        a = acc.setdefault(r, {"n_flows": 0, "n_detected": 0, "sum_score": 0.0})
        for k in a:
            a[k] += c[k]


def evaluate(cfg: dict, *, attacks: list[str] | None = None, force: bool = False,
             registry=None, gate_fn=None) -> dict:
    out_dir = Path(cfg["output_dir"])
    per_dir = out_dir / "per_source"
    per_dir.mkdir(parents=True, exist_ok=True)
    ecfg = dict(cfg.get("evaluate", {}) or {})
    device = resolve_device(ecfg.get("device", "auto"))
    seed = int(ecfg.get("seed", 42))
    batch_size = int(ecfg.get("batch_size", 64))
    pbs = int(ecfg.get("purify_batch_size", 4096))
    num_workers = cfg.get("num_workers", "auto")
    threshold = float(cfg.get("detector_threshold", 0.5))

    ps = load_pairs_and_split(cfg)
    model, sched, norm, saved = load_purifier(out_dir, device)
    T = int(saved["max_length"])
    t_star = resolve_t_star(cfg)
    recon = load_reconstructor(Path(cfg["reconstructor_model_dir"]), device)
    reg = registry or NIDSRegistry(cfg.get("nids", {}), Path(cfg["_repo_root"]))
    ipat_mean, ipat_std = _load_ipat_stats(cfg["tensors_dir"])

    all_sources = sorted(set(ps.src_names.tolist()))
    want = set(attacks) if attacks else None
    target = [s for s in all_sources
              if reg.known(s.rsplit("_adv_", 1)[0])
              and (want is None or s.rsplit("_adv_", 1)[0] in want)]
    if not target:
        raise RuntimeError("phase 9: no target sources (check --attacks and nids.attacks)")
    test_rows = rows_by_source(ps, ps.test_idx)
    all_rows = rows_by_source(ps, np.arange(len(ps.pairs)))
    test_fids = {s: {str(ps.pairs.flow_id[r]) for r in test_rows.get(s, [])} for s in target}
    strict_fids: dict[str, set] = {}
    for s, f in strict_test_keys(ps):
        strict_fids.setdefault(s, set()).add(f)

    verdicts, gated_fids, gate_info = (None, {}, {"disabled": True})
    if bool(ecfg.get("gated", True)):
        verdicts, gated_fids, gate_info = (gate_fn or _gate)(cfg, target, test_fids)

    comp_sources = set(ecfg.get("compliance_sources") or [])
    comp_max = int(ecfg.get("compliance_max_packets", 20000))
    # A per-source file is reused only if it was produced by this model, t* and gate.
    run_key = {"model_id": file_id(out_dir / "diffusion.pt"), "t_star": t_star,
               "gated": verdicts is not None}
    log.info("phase 9 over %d sources, t*=%d, gate=%s", len(target), t_star,
             "on" if verdicts is not None else "off")

    summary: dict = {"t_star": t_star, "run_key": run_key,
                     "diffusion": {k: saved[k] for k in ("method", "model", "training")},
                     "split": ps.origin, "gate": gate_info, "per_source": {},
                     "skipped_sources": {}}
    for gname in target:
        cache_file = per_dir / f"{gname}.json"
        if cache_file.exists() and not force:
            per = json.loads(cache_file.read_text())
            if per.get("run_key") == run_key:
                summary["per_source"][gname] = per
                log.info("%s: cached result reused", gname)
                continue
        t_src = time.time()
        base, mode = gname.rsplit("_adv_", 1)
        try:
            nids = reg.get(base)
        except NIDSUnavailable as e:
            summary["skipped_sources"][gname] = str(e)
            continue
        kind = nids.kind
        rows = all_rows[gname] if kind == "kitsune" else test_rows.get(gname,
                                                                        np.zeros(0, np.int64))
        flows = extract_flows(ps, rows, recon_model=recon, device=device,
                              batch_size=batch_size, num_workers=num_workers, max_length=T)
        if not flows:
            summary["skipped_sources"][gname] = "no pairs"
            continue
        pur = purify_flows(model, sched, norm, [f.adv for f in flows],
                           [f.direction for f in flows], t_star, device=device,
                           batch_size=pbs, seed=seed + zlib.crc32(gname.encode()))
        fv = {f.fid: {"adv": f.adv, "clean": f.clean, "dept": f.dept, "diffusion": p}
              for f, p in zip(flows, pur)}
        del flows, pur

        g_fids = gated_fids.get(gname, set())
        gate_src = None
        regimes = list(CORE)
        if verdicts is not None:
            flagged = {fid: float(verdicts.get((gname, fid), 0.0)) >= threshold for fid in fv}
            for fid, v in fv.items():
                v["dept_gated"] = v["dept"] if flagged[fid] else v["adv"]
                v["diffusion_gated"] = v["diffusion"] if flagged[fid] else v["adv"]
            in_subset = [flagged[f] for f in g_fids if f in flagged]
            gate_src = {"flag_rate": float(np.mean(list(flagged.values()))) if flagged else None,
                        "flag_rate_gated_subset": float(np.mean(in_subset)) if in_subset
                        else None,
                        "n_gated_flows": len(in_subset)}
            regimes += list(GATED)

        rz = BranchRealizer(adversarial_pcap(cfg, base, mode), fv, ipat_mean, ipat_std,
                            MASK_BRANCHES)
        if rz.stats.get("packets_per_branch", 0) == 0:
            summary["skipped_sources"][gname] = rz.stats.get("error", "no realized packets")
            continue

        comp = {} if (gname in comp_sources and tshark_available()) else None
        results: dict[str, list] = {}
        for regime in regimes:
            branch = BRANCH[regime]
            only = g_fids if (regime in GATED and kind != "kitsune") else None
            pk, ky = rz.packets(branch, only)
            if comp is not None and branch in ("dept", "diffusion") and pk:
                pcap = out_dir / "realized" / f"{gname}_{branch}.pcap"
                if write_pcap(pk[:comp_max], pcap):
                    comp[branch] = tshark_compliance(pcap)
            results[regime] = nids.score_flows(pk, ky)
            del pk, ky

        test_scope = test_fids.get(gname, set()) if kind == "kitsune" else None
        subsets = {
            "test": {r: _counts(results[r], test_scope) for r in CORE},
            "strict": {r: _counts(results[r], strict_fids.get(gname, set())) for r in CORE},
        }
        if verdicts is not None:
            subsets["gated"] = {r: _counts(results[r], g_fids) for r in regimes}
        if kind == "kitsune":
            subsets["full"] = {r: _counts(results[r], None) for r in regimes}
        unkeyed = {r: sum(1 for k, _s, _d in results[r] if k is None) for r in regimes}

        per = {"source": gname, "base": base, "mode": mode, "kind": kind,
               "model": nids.name, "run_key": run_key, "n_pairs": int(len(rows)),
               "realize": rz.stats, "gate": gate_src, "compliance": comp,
               "unkeyed_rows": unkeyed, "subsets": subsets,
               "seconds": round(time.time() - t_src, 1)}
        cache_file.write_text(json.dumps(per, indent=2, default=str))
        summary["per_source"][gname] = per
        te = derive(subsets["test"])["detection_rate"]
        log.info("%-24s [%s] test detection: clean %s  adv %s  DEPT %s  diffusion %s  (%.0fs)",
                 gname, kind, *(_f(te[r]) for r in CORE), time.time() - t_src)
        del rz, fv, results

    if not summary["per_source"]:
        why = "; ".join(sorted(set(summary["skipped_sources"].values()))) or "no sources"
        raise RuntimeError(f"phase 9 produced no results ({why})")
    summary.update(_aggregate(summary["per_source"]))
    for gname, per in summary["per_source"].items():
        per["derived"] = {sub: derive(c) for sub, c in per["subsets"].items()}

    lat = ecfg.get("latency") or {}
    if lat.get("n_flows"):
        try:
            summary["latency"] = measure_latency(
                ps, recon, model, sched, norm, t_star, device, T=T,
                n_flows=int(lat["n_flows"]),
                batch_sizes=[int(b) for b in lat.get("batch_sizes", [1, 256])],
                repeats=int(lat.get("repeats", 20)), seed=seed)
        except Exception as e:  # noqa: BLE001 — latency is a side measurement
            log.warning("latency measurement failed: %s", e)
            summary["latency"] = {"error": str(e)}

    (out_dir / "diffusion_purifier_metrics.json").write_text(
        json.dumps(summary, indent=2, default=str))
    _plot(summary, out_dir / "diffusion_purifier_comparison.png")
    return summary


def _f(v) -> str:
    return "—" if v is None else f"{v:.3f}"


def _aggregate(per_source: dict) -> dict:
    """``subsets[subset] = {overall, by_kind}`` from the per-source counts."""
    acc: dict = {}
    for per in per_source.values():
        for sub, counts in per["subsets"].items():
            node = acc.setdefault(sub, {"overall": {}, "by_kind": {}})
            _accumulate(node["overall"], counts)
            _accumulate(node["by_kind"].setdefault(per["kind"], {}), counts)
    return {"subsets": {sub: {"overall": derive(node["overall"], with_mean=False),
                              "by_kind": {k: derive(v) for k, v in
                                          sorted(node["by_kind"].items())}}
                        for sub, node in acc.items()}}


def _time_ms(fn, device: torch.device, repeats: int) -> float:
    with torch.inference_mode():
        for _ in range(2):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) / repeats * 1000.0


def measure_latency(ps, recon, model, sched, norm, t_star: int, device, *, T: int,
                    n_flows: int, batch_sizes: list[int], repeats: int, seed: int) -> dict:
    """Wall-clock per flow, same device: DEPT's one forward pass vs ``t*`` diffusion steps."""
    rng = np.random.default_rng(seed)
    n = min(int(n_flows), len(ps.test_idx))
    rows = np.sort(rng.choice(ps.test_idx, size=n, replace=False))
    ds = PairedFlowDataset(ps.pairs, sample_indices=rows, max_length=T)
    items = [ds[i] for i in range(len(ds))]
    out = {"device": str(device) if device.type != "cuda"
           else torch.cuda.get_device_name(device), "t_star": t_star, "by_batch_size": {}}
    g = torch.Generator(device=device).manual_seed(seed)
    for bs in batch_sizes:
        if bs > len(items):
            continue
        adv, _clean, lengths, mask, _rows = collate_pairs(items[:bs])
        adv_d, pad_d = adv.to(device), (~mask).to(device)
        dept_ms = _time_ms(lambda: recon(adv_d, key_padding_mask=pad_d), device, repeats)
        adv_np = adv.numpy()
        ov = [(decode_total_len_batch(adv_np[b, :int(lengths[b])]),
               adv_np[b, :int(lengths[b]), 1088]) for b in range(bs)]
        dirs = [np.clip(np.round(adv_np[b, :int(lengths[b]), 1089]), 0, 1)
                for b in range(bs)]
        x, cond, valid = encode_flows(norm, ov, dirs, T)
        xd, cd, vd = (torch.from_numpy(x).to(device), torch.from_numpy(cond).to(device),
                      torch.from_numpy(valid).to(device))
        use_cond = model.cond_dim > 0
        diff_ms = _time_ms(lambda: purify(model, sched, xd, cd if use_cond else None, vd,
                                          t_star, generator=g), device, repeats)
        out["by_batch_size"][str(bs)] = {
            "dept_ms_per_flow": dept_ms / bs, "diffusion_ms_per_flow": diff_ms / bs,
            "diffusion_over_dept": diff_ms / dept_ms if dept_ms > 0 else None,
            "mean_packets_per_flow": float(lengths.float().mean())}
    return out


def _plot(summary: dict, path: Path) -> None:
    per = summary.get("per_source", {})
    names = sorted(per)
    if not names:
        return
    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(9, len(names) * 0.5), 8.5))
    dr = {n: derive(per[n]["subsets"]["test"])["detection_rate"] for n in names}
    w = 0.2
    colors = {"clean": "#2ca02c", "adversarial": "#d62728", "dept_all": "#1f77b4",
              "diffusion_all": "#ff7f0e"}
    labels = {"clean": "clean (oracle)", "adversarial": "adversarial (TANTGen)",
              "dept_all": "DEPT purifier", "diffusion_all": "diffusion purifier"}
    for i, r in enumerate(CORE):
        ax1.bar(x + (i - 1.5) * w, [dr[n][r] or 0 for n in names], w, label=labels[r],
                color=colors[r])
    ax1.set_ylabel("NIDS detection rate (test flows)")
    ax1.set_ylim(0, 1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=90, fontsize=7)
    ax1.legend(fontsize=8)
    ax1.set_title("NIDS detection per source")
    for i, r in enumerate(("adversarial", "dept_all", "diffusion_all")):
        ax2.bar(x + (i - 1) * 0.27, [1 - (dr[n][r] or 0) for n in names], 0.27,
                label=labels[r], color=colors[r])
    ax2.set_ylabel("attack success rate (test flows)")
    ax2.set_ylim(0, 1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=90, fontsize=7)
    ax2.legend(fontsize=8)
    ax2.set_title(f"Attack success: DEPT vs diffusion purifier (t* = {summary['t_star']})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


__all__ = ["evaluate", "derive", "resolve_t_star", "measure_latency", "CORE", "GATED"]
