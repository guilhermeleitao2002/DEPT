"""Evasion-cost results for RQ1.

The result is a cost curve: how much padding and delay the attacker must spend
to push each target below its threshold.

* Targets are compared at a matched false-positive rate on benign flows, not
  at 0.5 (:func:`matched_fpr_thresholds`).
* TANTGen's own spend varies a lot between sources, so it is reported next to
  the marginal cost (:func:`tantgen_budget`).
* :func:`reliance_shift` tests whether the meta-learner relies more on the RF
  as the TCN is attacked.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Grid over which a per-flow minimum evasion budget is searched/reported.
DEFAULT_FPR_GRID = (0.01, 0.05, 0.10)


# ──────────────────────────────────────────────────────────────────────────────
# Thresholds
# ──────────────────────────────────────────────────────────────────────────────

def matched_fpr_thresholds(
    benign_scores: dict[str, np.ndarray],
    fpr_grid: tuple[float, ...] = DEFAULT_FPR_GRID,
) -> dict[str, dict[str, float]]:
    """Per-target thresholds at matched false-positive rates on benign flows.

    The threshold at FPR ``f`` is the ``(1-f)`` quantile of the target's
    unperturbed benign verdicts.
    """
    out: dict[str, dict[str, float]] = {}
    for target, s in benign_scores.items():
        s = np.asarray(s, dtype=np.float64)
        s = s[np.isfinite(s)]
        entry: dict[str, float] = {"raw_0.5": 0.5}
        if len(s):
            for f in fpr_grid:
                entry[f"fpr_{f:g}"] = float(np.quantile(s, 1.0 - f))
        out[target] = entry
    return out


# ──────────────────────────────────────────────────────────────────────────────
# ASR
# ──────────────────────────────────────────────────────────────────────────────

def synthesize_best(records: list[dict], select_on: str = "tcn") -> list[dict]:
    """Add a ``best`` variant: the better of PGD and random search per flow.

    Neither search dominates on this surface (the 16-bit length encoding makes
    the loss jagged), so an attacker would run both. The variant is chosen on the
    TCN, the attacker's own target, and that same perturbation is then scored on
    the RF and the stack.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault((r["source"], r["flow_id"], r["budget_id"]), []).append(r)

    out = list(records)
    for (_src, _fid, _b), rows in groups.items():
        scored = [r for r in rows if np.isfinite(r["verdicts"].get(select_on, np.nan))]
        if not scored:
            continue
        winner = min(scored, key=lambda r: r["verdicts"][select_on])
        best = dict(winner)
        best["variant"] = "best"
        best["source_variant"] = winner.get("variant", "pgd")
        out.append(best)
    return out


def asr_curves(
    records: list[dict],
    thresholds: dict[str, dict[str, float]],
    budget_order: list[str],
) -> dict:
    """ASR per (variant, target, threshold policy, budget), overall and per source.

    * The denominator is the flows the target detects without perturbation;
      ``detection_rate_clean`` is reported next to it.
    * The curve is the monotone envelope: a flow counts as evaded at budget *b*
      if it evaded at any budget <= *b*, since the feasible sets are nested.
    """
    sources = sorted({r["source"] for r in records})
    variants = sorted({r.get("variant", "pgd") for r in records})
    targets = sorted({t for r in records for t in r["verdicts"]})
    # NB: budget order comes from the config grid, never from sorted() — the ids
    # are strings, and "b1460" < "b32" < "b8" alphabetically.
    budgets = [b for b in budget_order if any(r["budget_id"] == b for r in records)]

    def _slice(rows: list[dict]) -> dict:
        out: dict = {}
        for variant in variants:
            vrows = [r for r in rows if r.get("variant", "pgd") == variant]
            out[variant] = {}
            for target in targets:
                pol = thresholds.get(target, {"raw_0.5": 0.5})
                out[variant][target] = {}
                for pname, thr in pol.items():
                    out[variant][target][pname] = _envelope(
                        vrows, target, thr, budgets)
        return out

    per_source = {}
    for s in sources:
        rows = [r for r in records if r["source"] == s]
        entry = _slice(rows)
        entry["low_support"] = bool(len({r["flow_id"] for r in rows}) < 50)
        per_source[s] = entry

    return {
        "budgets": budgets,
        "variants": variants,
        "overall": _slice(records),
        "per_source": per_source,
    }


def _envelope(
    rows: list[dict],
    target: str,
    thr: float,
    budgets: list[str],
) -> dict:
    """Per-budget ASR under the nested-feasible-set (monotone) envelope."""
    # flow -> {clean verdict, budget -> verdict}
    flows: dict[str, dict] = {}
    for r in rows:
        if target not in r["verdicts"]:
            continue
        key = f"{r['source']}|{r['flow_id']}"
        e = flows.setdefault(key, {"clean": r["verdicts_clean"][target],
                                   "by_budget": {}, "nf": {}})
        e["by_budget"][r["budget_id"]] = r["verdicts"][target]
        e["nf"][r["budget_id"]] = bool(r["never_fires"].get(target, False))

    detected = [k for k, e in flows.items()
                if np.isfinite(e["clean"]) and e["clean"] >= thr]
    out: dict = {}
    evaded: set[str] = set()
    nf_seen: set[str] = set()
    for b in budgets:
        for k in detected:
            v = flows[k]["by_budget"].get(b)
            if v is not None and np.isfinite(v) and v < thr:
                evaded.add(k)
            if flows[k]["nf"].get(b):
                nf_seen.add(k)
        n_det = len(detected)
        uncond = [e["by_budget"].get(b) for e in flows.values()
                  if e["by_budget"].get(b) is not None]
        uncond = [v for v in uncond if np.isfinite(v)]
        out[b] = {
            "asr": (len(evaded) / n_det) if n_det else float("nan"),
            "asr_unconditional": (float(np.mean([v < thr for v in uncond]))
                                  if uncond else float("nan")),
            "asr_never_fires": (len(nf_seen) / n_det) if n_det else float("nan"),
            "detection_rate_clean": (n_det / len(flows)) if flows else float("nan"),
            "n_detected": int(n_det),
            "n_flows": int(len(flows)),
        }
    return out


def min_budget_per_flow(
    records: list[dict],
    thresholds: dict[str, dict[str, float]],
    budget_order: list[str],
    policy: str = "fpr_0.05",
    variant: str = "pgd",
) -> dict:
    """Smallest grid budget at which each flow evades, per target.

    Over flows detected without perturbation: the median over flows that evade,
    plus the fraction that never evade.
    """
    rows = [r for r in records if r.get("variant", "pgd") == variant]
    targets = sorted({t for r in rows for t in r["verdicts"]})
    rank = {b: i for i, b in enumerate(budget_order)}
    out: dict = {"policy": policy, "variant": variant}

    for target in targets:
        thr = thresholds.get(target, {}).get(policy, 0.5)
        first: dict[str, int] = {}
        detected: set[str] = set()
        for r in rows:
            if target not in r["verdicts"]:
                continue
            key = f"{r['source']}|{r['flow_id']}"
            if not (r["verdicts_clean"][target] >= thr):
                continue          # never detected unperturbed → nothing to evade
            detected.add(key)
            if r["verdicts"][target] < thr:
                i = rank[r["budget_id"]]
                if key not in first or i < first[key]:
                    first[key] = i
        evaded = np.asarray(sorted(first.values()), dtype=np.float64)
        n_det = len(detected)
        out[target] = {
            "threshold": float(thr),
            "n_detected_unperturbed": int(n_det),
            "n_evaded": int(len(evaded)),
            "frac_never_evaded": (float(1.0 - len(evaded) / n_det)
                                  if n_det else float("nan")),
            "median_min_budget_id": (budget_order[int(np.median(evaded))]
                                     if len(evaded) else None),
            "mean_min_budget_rank": (float(evaded.mean()) if len(evaded)
                                     else float("nan")),
        }
    return out


def spend_summary(
    records: list[dict],
    budget_order: list[str],
    variant: str = "pgd",
) -> dict:
    """Bytes padded and seconds delayed per budget (the budget is only a cap)."""
    rows_all = [r for r in records if r.get("variant", "pgd") == variant]
    budgets = [b for b in budget_order
               if any(r["budget_id"] == b for r in rows_all)]
    out = {}
    for b in budgets:
        rows = [r for r in rows_all if r["budget_id"] == b]
        pad = np.asarray([r["spend"]["median_pad_bytes"] for r in rows],
                         dtype=np.float64)
        dly = np.asarray([r["spend"]["median_delay_seconds"] for r in rows],
                         dtype=np.float64)
        tot_pad = np.asarray([r["spend"]["total_pad_bytes"] for r in rows],
                             dtype=np.float64)
        out[b] = {
            "median_pad_bytes_per_packet": float(np.nanmedian(pad)) if len(pad) else 0.0,
            "median_delay_ms_per_packet": (float(np.nanmedian(dly) * 1e3)
                                           if len(dly) else 0.0),
            "median_total_pad_bytes_per_flow": (float(np.nanmedian(tot_pad))
                                                if len(tot_pad) else 0.0),
            "n_flows": int(len(rows)),
        }
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Reliance shift
# ──────────────────────────────────────────────────────────────────────────────

def reliance_shift(
    records: list[dict],
    budget_order: list[str],
    variant: str = "pgd",
) -> dict:
    """Does the meta-learner rely more on the RF as the TCN is attacked?

    For each budget the stack's verdict is compared with two ablations:

    * ``stack_rf_clean``  — attacked TCN logit, unperturbed RF logit
    * ``stack_tcn_clean`` — unperturbed TCN logit, attacked RF logit

    With ``d_x`` the change in the verdict when slot ``x`` is reset,
    ``rf_reliance = d_rf / (d_rf + d_tcn)``.
    """
    budgets = [b for b in budget_order
               if any(r["budget_id"] == b for r in records)]
    out = {}
    for b in budgets:
        rows = [r for r in records if r["budget_id"] == b
                and r.get("variant", "pgd") == variant
                and "stack" in r["verdicts"]
                and "stack_rf_clean" in r.get("ablations", {})
                and "stack_tcn_clean" in r.get("ablations", {})]
        if not rows:
            continue
        stack = np.asarray([r["verdicts"]["stack"] for r in rows])
        rf_clean = np.asarray([r["ablations"]["stack_rf_clean"] for r in rows])
        tcn_clean = np.asarray([r["ablations"]["stack_tcn_clean"] for r in rows])

        # Change in the verdict when each slot is reset to its clean value.
        d_rf = np.abs(stack - rf_clean)
        d_tcn = np.abs(stack - tcn_clean)
        denom = d_rf + d_tcn
        share = np.divide(d_rf, denom, out=np.full_like(denom, np.nan),
                          where=denom > 1e-12)
        # At delta=0 the share is 0/0: report it as undefined.
        finite = share[np.isfinite(share)]
        out[b] = {
            "n_flows": int(len(rows)),
            "median_stack_verdict": float(np.nanmedian(stack)),
            "median_delta_if_rf_restored": float(np.nanmedian(d_rf)),
            "median_delta_if_tcn_restored": float(np.nanmedian(d_tcn)),
            "rf_reliance": (float(np.median(finite)) if finite.size
                            else float("nan")),
            "n_moved": int(finite.size),
        }
    return out


# ──────────────────────────────────────────────────────────────────────────────
# TANTGen's own budget
# ──────────────────────────────────────────────────────────────────────────────

def tantgen_budget(
    adversarial_dir: Path,
    tensors_dir: Path,
    source: str,
    max_flows: int = 400,
    ipat_std: float = 208.21998059319273,
) -> dict:
    """TANTGen's own per-packet spend on ``source``, against its clean base.

    Flows are paired by 5-tuple and only same-length pairs are compared.
    """
    from ..common.attack_scope import base_attack
    from .attack import IDX_IPAT, IDX_IPV4_TL

    base = base_attack(source)
    adv_dir = Path(adversarial_dir) / source
    cln_dir = Path(tensors_dir) / base
    if not (adv_dir.exists() and cln_dir.exists()):
        return {"error": f"missing {adv_dir} or {cln_dir}"}

    w = (1 << np.arange(15, -1, -1)).astype(np.float64)

    def _load(d: Path):
        return (np.load(d / "flow_offsets.npy"),
                np.load(d / "flow_ids.npy", allow_pickle=True),
                np.load(d / "flow_packets.npy", mmap_mode="r"))

    off_a, ids_a, pk_a = _load(adv_dir)
    off_c, ids_c, pk_c = _load(cln_dir)
    cmap = {str(k): i for i, k in enumerate(ids_c)}

    d_tl, d_iat, n = [], [], 0
    for i, k in enumerate(ids_a):
        j = cmap.get(str(k))
        if j is None:
            continue
        sa, la = off_a[i]
        sc, lc = off_c[j]
        if la != lc or la < 2:
            continue
        A = np.asarray(pk_a[int(sa):int(sa) + int(la)])
        C = np.asarray(pk_c[int(sc):int(sc) + int(lc)])
        tl_a = np.clip(np.round(A[:, IDX_IPV4_TL[0]:IDX_IPV4_TL[1]]), 0, 1) @ w
        tl_c = np.clip(np.round(C[:, IDX_IPV4_TL[0]:IDX_IPV4_TL[1]]), 0, 1) @ w
        d_tl.append(tl_a - tl_c)
        d_iat.append((A[:, IDX_IPAT].astype(np.float64)
                      - C[:, IDX_IPAT].astype(np.float64)) * ipat_std)
        n += 1
        if n >= max_flows:
            break
    if not n:
        return {"error": "no 5-tuple-matched pairs"}

    d_tl = np.concatenate(d_tl)
    d_iat = np.concatenate(d_iat)
    # Deltas up to one float32 ULP are round-off (attack.IPAT_ULP_SECONDS).
    from .attack import IPAT_ULP_SECONDS
    real = np.abs(d_iat) > IPAT_ULP_SECONDS
    return {
        "n_flows_paired": int(n),
        "n_packets": int(len(d_tl)),
        "size": {
            "frac_packets_perturbed": float((d_tl != 0).mean()),
            "mean_pad_bytes": float(d_tl.mean()),
            "max_pad_bytes": float(d_tl.max()),
        },
        "iat": {
            "frac_packets_perturbed": float(real.mean()),
            "mean_delay_seconds": float(d_iat[real].mean()) if real.any() else 0.0,
            "max_delay_seconds": float(d_iat.max()),
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_asr_curves(
    curves: dict,
    spend: dict,
    output_path: Path,
    policy: str = "fpr_0.05",
    variant: str = "best",
    title: str = "Evasion cost — gradient attack crafted on the TCN",
) -> None:
    """ASR vs spend, one line per target, for a single variant."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if variant not in curves["overall"]:
        variant = curves.get("variants", ["pgd"])[0]
    budgets = curves["budgets"]
    x = [spend.get(b, {}).get("median_total_pad_bytes_per_flow", 0.0)
         for b in budgets]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for target in ("rf", "tcn", "stack"):
        pol = curves["overall"][variant].get(target)
        if not pol or policy not in pol:
            continue
        ax.plot(x, [pol[policy][b]["asr"] for b in budgets],
                marker="o", label=target.upper())
    ax.set_xlabel("attacker spend — median total padding per flow (bytes)")
    ax.set_ylabel(f"attack success rate @ {policy}\n(flows detected unperturbed)")
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def _sidecar_path(output_dir: Path, source: str) -> Path:
    return Path(output_dir) / "sidecar" / f"{source}.npz"


def _get_sidecar(
    source: str,
    folder: Path,
    pcap_dir: Path,
    flow_ids: list[str],
    output_dir: Path,
    max_length: int,
    rebuild: bool = False,
):
    """Load the cached sidecar for ``source``, building it if absent.

    The cache is keyed by the selected flow set.
    """
    from .features import build_sidecar, load_sidecar, save_sidecar, verify_sidecar

    path = _sidecar_path(output_dir, source)
    if path.exists() and not rebuild:
        sc = load_sidecar(path)
        if set(sc.flow_ids.tolist()) >= set(flow_ids):
            return sc
        log.info("%s: cached sidecar covers a different flow set — rebuilding",
                 source)

    meta = json.loads((Path(folder) / "metadata.json").read_text())
    pcap = Path(meta["input"])
    if not pcap.exists():
        # metadata.json records an absolute path from build time; fall back to
        # the configured PCAP dir if the tree moved.
        from ..common.attack_scope import base_attack
        mode = source.rsplit("_adv_", 1)[-1]
        pcap = (Path(pcap_dir) / base_attack(source)
                / f"{base_attack(source)}_adversarial_{mode}.pcap")
    if not pcap.exists():
        raise FileNotFoundError(f"{source}: TANTGen PCAP not found ({pcap})")

    log.info("%s: building sidecar from %s", source, pcap)
    sc = build_sidecar(source, pcap, flow_ids, max_length=max_length)
    rep = verify_sidecar(
        sc, folder,
        int(meta.get("rf_first_checkpoint", 2)),
        int(meta.get("rf_checkpoint_stride", 10)),
    )
    if not rep["ok"]:
        raise RuntimeError(
            f"{source}: sidecar does not reproduce rf_checkpoints.npy at delta=0 "
            f"(max_abs_diff={rep['max_abs_diff']:.4g}, "
            f"mismatches={rep['mismatches']}). Every RF number derived from it "
            f"would be measuring this bug rather than the attack.")
    log.info("%s: sidecar verified against rf_checkpoints.npy (max_abs_diff=%.3g)",
             source, rep["max_abs_diff"])
    save_sidecar(sc, path)
    return sc


def evaluate(config: dict, output_dir: Path, attack: str | None = None) -> dict:
    """Run the study end to end and write ``gradient_ablation_metrics.json``."""
    from .attack import PGDConfig
    from .features import select_flows, test_flow_ids_from_split
    from .pipeline import (
        attack_source,
        autopgd_check,
        parse_budgets,
        score_benign_reference,
    )
    from .targets import load_models

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(config["_repo_root"])

    tensors_dir = Path(config["tensors_dir"])
    adversarial_dir = Path(config["adversarial_dir"])
    pcap_dir = Path(config["adversarial_pcap_dir"])
    acfg = config.get("attack", {})

    exclude = list(config.get("adversarial_exclude", []))
    if attack:
        from ..common.attack_scope import scoped_adversarial_exclude
        exclude = scoped_adversarial_exclude(adversarial_dir, attack, exclude)

    ipat_std = float(json.loads(
        (tensors_dir / "mean_std.json").read_text())["IPAT"]["std"])

    log.info("loading trained targets")
    models = load_models(
        Path(config["rf_model_dir"]), Path(config["tcn_model_dir"]),
        Path(config["metalearner_model_dir"]), str(config.get("device", "auto")),
    )
    models.max_length = int(config.get("max_length", models.max_length))

    log.info("scoring benign reference flows (matched-FPR thresholds)")
    benign_sources = list(config.get("benign_sources", []))
    if attack:
        benign_sources = [s for s in benign_sources if s == attack]
    benign = score_benign_reference(
        models, tensors_dir, benign_sources,
        int(config.get("benign_flows_per_source", 200)), int(config.get("seed", 42)),
    )
    thresholds = matched_fpr_thresholds(
        benign, tuple(config.get("fpr_grid", DEFAULT_FPR_GRID)))

    log.info("selecting adversarial test flows")
    test_ids = test_flow_ids_from_split(Path(config["tcn_model_dir"]), repo_root)
    selected = select_flows(
        adversarial_dir, int(config.get("flows_per_source", 300)),
        int(config.get("seed", 42)), test_ids, tuple(exclude),
        int(config.get("min_len", 2)),
    )

    budgets = parse_budgets(config["budgets"])
    pgd_cfg = PGDConfig(
        steps=int(acfg.get("steps", 40)),
        alpha_ratio=float(acfg.get("alpha_ratio", 2.5)),
        random_init=bool(acfg.get("random_init", True)),
        restarts=int(acfg.get("restarts", 2)),
        seed=int(config.get("seed", 42)),
    )

    records: list[dict] = []
    tg_budget: dict = {}
    skipped: dict[str, str] = {}
    apgd: dict = {"available": False, "reason": "not run"}

    for source, rows in selected.by_source().items():
        folder = selected.source_folders[source]
        flow_ids = [str(selected.flow_ids[r]) for r in rows]
        try:
            sc = _get_sidecar(source, folder, pcap_dir, flow_ids, output_dir,
                              models.max_length)
        except Exception as e:
            log.error("%s: skipped — %s", source, e)
            skipped[source] = str(e)
            continue

        meta = json.loads((Path(folder) / "metadata.json").read_text())

        if acfg.get("autopgd_check", True) and not apgd.get("available"):
            from ..common.dataset import load_source_meta
            _y, fids_m, _ls, li_m, _n = load_source_meta(Path(folder), source)
            lookup = {str(f): int(x) for f, x in zip(fids_m, li_m)}
            ids = [lookup[f] for f in flow_ids[:16] if f in lookup]
            if ids:
                log.info("%s: ART AutoPGD attack-strength cross-check", source)
                try:
                    apgd = autopgd_check(
                        models, Path(folder), ids,
                        float(budgets[-1].bytes_) or 128.0, ipat_std, pgd_cfg,
                        tl_cap=int(acfg.get("tl_cap", 1500)),
                    )
                    apgd["source"] = source
                except Exception as e:
                    log.warning("AutoPGD cross-check failed: %s", e)
                    apgd = {"available": False, "reason": str(e)}
        records += attack_source(
            models, source, folder, sc, flow_ids, budgets, ipat_std,
            int(meta.get("rf_first_checkpoint", 2)),
            int(meta.get("rf_checkpoint_stride", 10)),
            pgd_cfg,
            batch_size=int(config.get("batch_size", 32)),
            threshold=float(config.get("threshold", 0.5)),
            pad_byte_mode=str(acfg.get("pad_byte_mode", "random")),
            objective=str(acfg.get("objective", "final")),
            tl_cap=int(acfg.get("tl_cap", 1500)),
            variants=tuple(acfg.get("variants", ("pgd", "random"))),
        )
        tg_budget[source] = tantgen_budget(
            adversarial_dir, tensors_dir, source, ipat_std=ipat_std)

    if not records:
        raise RuntimeError(f"no flows attacked (skipped: {skipped})")

    policy = str(config.get("report_policy", "fpr_0.05"))
    budget_order = [b.id for b in budgets]
    records = synthesize_best(records)
    curves = asr_curves(records, thresholds, budget_order)
    spend = spend_summary(records, budget_order, variant="best")
    summary = {
        "n_records": len(records),
        "n_flows": len({f"{r['source']}|{r['flow_id']}" for r in records}),
        "device": str(models.device),
        "threat_model": {
            "attacker": "white-box PGD on the TCN; no gradient path through the RF",
            "perturbation": "increase-only padding (bytes) + delay (seconds)",
            "objective": str(acfg.get("objective", "final")),
            "pad_byte_mode": str(acfg.get("pad_byte_mode", "random")),
            "tl_cap": int(acfg.get("tl_cap", 1500)),
            "pgd": {"steps": pgd_cfg.steps, "alpha_ratio": pgd_cfg.alpha_ratio,
                    "random_init": pgd_cfg.random_init},
            "asr_denominator": (
                "flows the target detects unperturbed at the reported threshold"),
        },
        "benign_reference": {
            t: {"n_flows": int(len(v)),
                "median_verdict": (float(np.nanmedian(v)) if len(v) else None)}
            for t, v in benign.items()
        },
        "thresholds": thresholds,
        "report_policy": policy,
        "spend": spend,
        "asr": curves,
        "min_budget": min_budget_per_flow(
            records, thresholds, budget_order, policy, variant="best"),
        "reliance_shift": reliance_shift(records, budget_order, variant="best"),
        "attack_strength_check": apgd,
        "tantgen_budget": tg_budget,
        "skipped_sources": skipped,
    }

    (output_dir / "gradient_ablation_metrics.json").write_text(
        json.dumps(summary, indent=2, default=str))
    plot_asr_curves(curves, spend, output_dir / "evasion_cost.png", policy=policy)
    log.info("wrote %s", output_dir / "gradient_ablation_metrics.json")
    return summary


__all__ = [
    "DEFAULT_FPR_GRID",
    "matched_fpr_thresholds",
    "asr_at",
    "asr_curves",
    "min_budget_per_flow",
    "spend_summary",
    "reliance_shift",
    "tantgen_budget",
    "plot_asr_curves",
    "evaluate",
]
