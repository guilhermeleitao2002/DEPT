"""Evasion-cost measurement: the RQ1 deliverable.

The headline is not a single attack success rate but a **cost curve**: how much
padding and delay the attacker must spend to push each target below its decision
threshold. RQ1 asks whether heterogeneous stacking "measurably raises the cost",
so cost is the x-axis and it is denominated in what an attacker actually pays —
bytes on the wire and seconds of delay — not in an abstract epsilon.

Three properties of this setup drive the metric choices:

*Matched FPR, not a raw threshold.* The meta-learner early-stopped at
``best_epoch: 1``. Comparing targets at a fixed 0.5 would credit the stack for
being undertrained: a model that rarely says "adversarial" trivially resists an
attack that tries to make it say "benign". :func:`matched_fpr_thresholds` scores
unperturbed **benign** flows through each target and picks each one's threshold
at a common false-positive rate, so all three are compared at the same operating
point they would actually be deployed at.

*The marginal baseline varies wildly.* Cost here is measured *on top of* TANTGen,
whose own budget is not uniform: measured per-packet, ``mirai_adv_size`` perturbs
81% of packets by +23.6 B and ``portscan_adv_size`` 63% by +63 B, but
``dns_ddos_adv_size`` only 0.3% by +0.09 B. A marginal number alone would
mislead, so :func:`tantgen_budget` reports TANTGen's own spend beside it.

*Reliance shift is the falsifiable part.* The design claim behind DEPT is that
when the TCN is degraded, the meta-learner leans on the RF. That is testable, not
rhetorical: :func:`reliance_shift` ablates the meta's two base-learner token
slots and reports which one is carrying the verdict as the budget grows.
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

    ``benign_scores[target]`` is the flow verdict P(adversarial) over unperturbed
    benign flows. The threshold at FPR ``f`` is the ``(1-f)`` quantile of that
    distribution, so every target is compared at the same operating point rather
    than at a shared-but-arbitrary 0.5.
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
    """Add a ``best`` variant: the strongest attack the attacker could mount.

    A robustness evaluation must report the best attack available, not a
    favourite one — and here neither PGD nor random search dominates. Measured on
    ``brute_force_adv_size`` at a matched threshold, PGD reaches 35.4% at 128 B
    while query-matched random search reaches 33.3%, and at 512 B random wins
    58.3% to 12.5%. Gradient direction carries little information on this surface
    because the 16-bit total-length encoding makes the loss jagged in packet
    size, so exploration beats descent. An attacker would simply run both.

    **Selection is on the attacker's own target, never per-scored-target.** The
    variant is chosen by whichever minimises ``select_on`` (the TCN — the only
    model the attacker can differentiate), and *that* variant's perturbation is
    then read off against RF and the stack. Picking the best variant separately
    per target would silently grant the attacker a different perturbation for
    each, which is not the threat model and would overstate every ASR.
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
    """ASR per (variant, target, threshold-policy, budget), overall and per source.

    Two conventions are baked in here, and both are load-bearing.

    **The denominator is the flows the target detects unperturbed.** This is the
    standard robust-accuracy convention (attack success is measured on initially
    correct inputs), and here it is the difference between a number and an
    artifact: the TCN's reported ROC-AUC is 0.494 on ``mirai_adv_size`` and 0.232
    on ``mirai_adv_iat`` — at chance or below. An unconditional ASR would start
    near 75% before the attack did anything and would mostly be reporting flows
    the detector never caught. ``detection_rate_clean`` rides along because it is
    what makes ``asr`` readable: a target that detects nothing cannot be evaded,
    and an ASR over three flows is noise.

    **The curve is the monotone envelope.** A flow counts as evaded at budget *b*
    if it evaded at *any* budget <= *b*. This is not smoothing — it follows from
    the feasible sets being nested: an attacker allowed 1460 bytes may simply
    choose to spend 128, so a rational attacker's success cannot fall as its
    budget grows. Measured per-budget PGD does not inherit that, because the step
    size scales with the budget and the loss surface over packet size is
    genuinely jagged (the 16-bit total-length encoding flips nine bits between
    255 and 256, so neighbouring byte counts are far apart in the TCN's input).
    Without the envelope the numbers oscillate — ``ssdp_flood_adv_size`` measured
    87.5% at 128 B, 4.2% at 512 B and 97.9% at 1460 B — which would read as the
    defense strengthening when it only means the attack got sloppier.
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
    """Smallest grid budget at which each flow evades — the cost, per target.

    Restricted to flows the target detects unperturbed (see :func:`asr_at`), and
    reported as the median over those that ever evade plus the fraction that
    never do. Flows that never evade are excluded from the median rather than
    imputed at the grid cap: a censored observation is not a large one, and
    imputing would flatter whichever target resists most.
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
    """What the attacker actually spent, per budget — bytes padded, seconds delayed.

    The grid is a *cap*; PGD need not exhaust it. These are the realised costs,
    and they are the x-axis the ASR curves should be read against.
    """
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
# Reliance shift — the falsifiable design claim
# ──────────────────────────────────────────────────────────────────────────────

def reliance_shift(
    records: list[dict],
    budget_order: list[str],
    variant: str = "pgd",
) -> dict:
    """Does the meta-learner lean on the RF as the attack degrades the TCN?

    For each budget, compares the stack's verdict against two ablations replayed
    through the same meta-learner:

    * ``stack_rf_clean``  — attacked TCN logit, **unperturbed** RF logit
    * ``stack_tcn_clean`` — **unperturbed** TCN logit, attacked RF logit

    Let ``d_x`` be how far the stack's verdict moves when slot ``x`` is handed
    back its clean value. A large ``d_tcn`` means the *attacked TCN logit* was
    the thing dragging the verdict down — i.e. the stack was still listening to
    the TCN. So reliance on the RF is the complementary share:

        ``rf_reliance = d_rf / (d_rf + d_tcn)``

    If the design claim holds, this **rises** with the budget: the meta-learner
    discounts the base learner the attacker has captured and leans on the one it
    cannot reach.
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

        # How far the verdict recovers when each slot is handed back its clean
        # value. Large d_tcn => the attacked TCN logit was carrying the verdict,
        # i.e. the stack was still listening to the TCN; RF reliance is the
        # complementary share.
        d_rf = np.abs(stack - rf_clean)
        d_tcn = np.abs(stack - tcn_clean)
        denom = d_rf + d_tcn
        share = np.divide(d_rf, denom, out=np.full_like(denom, np.nan),
                          where=denom > 1e-12)
        # At delta=0 both ablations equal the stack, so every share is 0/0.
        # That is the correct answer (nothing moved, so nothing is carrying the
        # verdict), not a number to average — report it as undefined.
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
    ipat_std: float = 211047.5219908926,
) -> dict:
    """TANTGen's own per-packet spend on ``source``, paired to its clean base.

    Context for the marginal cost: the gradient attacker's budget is measured
    *on top of* this, and this varies by two orders of magnitude across sources.
    Flows are paired by canonical 5-tuple, which is why only same-length pairs
    are compared.
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
    # Deltas at or below one float32 ULP of the z-scored IPAT channel are
    # round-off, not perturbation — see attack.IPAT_ULP_SECONDS.
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
    """ASR vs realised attacker spend, one line per target.

    Plots a **single** variant. ``spend`` is per-variant (each search pays a
    different price for the same budget cap), so overlaying pgd/random/best on
    one x-axis would place two of the three lines at costs they never paid.
    """
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

    The build is the one I/O-heavy step in the study (a streaming scapy pass over
    the source PCAP), so it is cached and keyed by the selected flow set — a
    different ``flows_per_source`` or seed invalidates it.
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
                "flows the target detects unperturbed at the reported threshold "
                "(robust-accuracy convention); an unconditional ASR would mostly "
                "report flows the detector never caught — the TCN's own ROC-AUC "
                "is 0.494 on mirai_adv_size and 0.232 on mirai_adv_iat"),
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
