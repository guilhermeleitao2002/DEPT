"""Deployment baseline: how the protected NIDS behaves clean / perturbed / DEPT-defended.

This is DEPT's headline end-to-end result — the "before vs after DEPT" the dissertation's
§5.7 asks for. For every attack (× perturbation mode) we score three realized full-volume
streams through the NIDS the attack was built to evade (Kitsune / ML-NIDS, via
:class:`~src.models.nids.NIDSRegistry`):

  * **clean**       — the original-attack size/timing (oracle ceiling: the NIDS working).
  * **adversarial** — TANTGen's perturbation (evasion floor: detection collapses).
  * **defended**    — the full DEPT pipeline (:mod:`.pipeline`): the detector flags each
    flow and only flagged flows are purified; the rest pass through unchanged. Detection
    should climb back from the adversarial floor toward the clean ceiling.

A fourth **purified_all** stream (every flow purified, i.e. a perfect detector) is scored
as an upper bound. Because the defended / purified streams run DEPT models over flows some
of which were in training, the DEPT-defended numbers are also reported restricted to the
**held-out test flows** (``*_test_subset``) — the generalization-valid view — alongside the
full-stream deployment view the user asked for.

Metric = per-flow **detection rate** and **ASR** (``1 − detection_rate``) per regime; the
headline is ``asr_reduction`` (how much of TANTGen's attack success DEPT removes) and the
rate-based ``restoration_ratio_detection`` (fraction of the adv→clean detection gap
recovered).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ..nids import NIDSRegistry, NIDSUnavailable  # noqa: E402
from ..nids import metrics as nids_metrics  # noqa: E402
from ..reconstructor.dataset import build_pair_index  # noqa: E402
from ..reconstructor.evaluate import _load_ipat_stats  # noqa: E402
from ..reconstructor.realize import realize_triplet  # noqa: E402
from . import pipeline  # noqa: E402
from .detector_infer import detector_verdicts, test_flow_keys  # noqa: E402

log = logging.getLogger(__name__)

_REGIMES = ("clean", "adversarial", "defended", "purified_all")


def _resolve_device(cfg: str) -> torch.device:
    if cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg)


def _detector_cfg(config: dict) -> dict:
    return {
        "rf_model_dir": config["rf_model_dir"],
        "tcn_model_dir": config["tcn_model_dir"],
        "metalearner_model_dir": config["metalearner_model_dir"],
        "tensors_dir": config["tensors_dir"],
        "adversarial_dir": config["adversarial_dir"],
        "detector_cache_dir": str(Path(config["output_dir"]) / "detector_cache"),
        "token": config.get("token", {}),
        "device": config.get("device", "auto"),
    }


def _asr(dr):
    return (1.0 - dr) if dr is not None else None


def _source_metrics(scored: dict, scored_test, gate_stats: dict, model, mode: str,
                    rstats: dict) -> dict:
    dr = {r: scored[r].detection_rate for r in scored}
    ms = {r: scored[r].mean_score for r in scored}
    asr = {r: _asr(dr[r]) for r in dr}
    out = {
        "kind": model.kind, "model": model.name, "mode": mode,
        "model_source": getattr(model, "model_source", "?"),
        "detection_rate": dr,
        "asr": asr,
        "mean_score": ms,
        # Headline: fraction of TANTGen's attack success DEPT removes.
        "asr_reduction": nids_metrics.sub(asr.get("adversarial"), asr.get("defended")),
        # Fraction of the adv→clean detection gap the pipeline recovers (rate-based).
        "restoration_ratio_detection": nids_metrics.restoration_ratio(
            dr.get("adversarial"), dr.get("defended"), dr.get("clean")),
        "restoration_ratio_score": nids_metrics.restoration_ratio(
            ms.get("adversarial"), ms.get("defended"), ms.get("clean")),
        "gate": gate_stats,
        "n_flows": scored.get("defended").n_flows if "defended" in scored else None,
        "realize": rstats,
    }
    if scored_test is not None:
        out["test_subset"] = {
            "detection_rate_defended": scored_test.detection_rate,
            "asr_defended": _asr(scored_test.detection_rate),
            "n_flows": scored_test.n_flows,
        }
    return out


def _finalize_overall(acc: dict) -> dict:
    dr = {r: (acc[r]["n_detected"] / acc[r]["n_flows"]) if acc.get(r, {}).get("n_flows")
          else None for r in acc}
    asr = {r: _asr(dr[r]) for r in dr}
    return {
        "detection_rate": dr,
        "asr": asr,
        "asr_reduction": nids_metrics.sub(asr.get("adversarial"), asr.get("defended")),
        "restoration_ratio_detection": nids_metrics.restoration_ratio(
            dr.get("adversarial"), dr.get("defended"), dr.get("clean")),
        "n_flows": acc.get("defended", {}).get("n_flows", 0),
    }


def evaluate(config: dict, output_dir: Path, attacks: list[str] | None = None) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(config.get("_repo_root", "."))
    device = _resolve_device(str(config.get("device", "auto")))
    threshold = float(config.get("detector_threshold", 0.5))
    score_purified_all = bool(config.get("score_purified_all", True))
    score_test_subset = bool(config.get("score_test_subset", True))

    recon = pipeline.load_reconstructor(Path(config["reconstructor_model_dir"]), device)
    pairs = build_pair_index(
        tensors_dir=Path(config["tensors_dir"]),
        adversarial_dir=Path(config["adversarial_dir"]),
        adversarial_exclude=list(config.get("adversarial_exclude", [])),
        min_len=int(config.get("min_len", 2)))
    src_names = np.array(
        [f"{b}_adv_{m}" for b, m in zip(pairs.base, pairs.mode)], dtype=object)

    reg = NIDSRegistry(config.get("nids", {}), repo_root)
    all_sources = sorted(set(src_names.tolist()))
    if attacks:
        want = set(attacks)
        target = [s for s in all_sources if s.rsplit("_adv_", 1)[0] in want]
    else:
        target = [s for s in all_sources if reg.known(s.rsplit("_adv_", 1)[0])]
    if not target:
        raise RuntimeError("baseline: no target sources (check --attacks and nids.attacks)")

    log.info("baseline over %d sources: %s", len(target), ", ".join(target))
    tcn_dir = Path(config["tcn_model_dir"])
    verdicts = detector_verdicts(_detector_cfg(config), set(target))
    test_keys = test_flow_keys(tcn_dir, set(target)) if score_test_subset else set()

    pcap_dir = Path(config["adversarial_pcap_dir"])
    mean_std = _load_ipat_stats(config["tensors_dir"])
    max_length = config.get("max_length", None)
    batch_size = int(config.get("batch_size", 32))
    num_workers = config.get("num_workers", "auto")

    summary = {"threshold": threshold, "per_source": {}, "skipped_sources": {},
               "models": {}, "n_sources": len(target)}
    acc_all: dict = {}
    acc_kind: dict[str, dict] = {}

    for gname in target:
        base, mode = gname.rsplit("_adv_", 1)
        try:
            model = reg.get(base)
        except NIDSUnavailable as e:
            summary["skipped_sources"][gname] = str(e)
            continue

        rows = np.where(src_names == gname)[0]
        flow_vectors = pipeline.purify_source(
            pairs, rows, recon, device, batch_size=batch_size,
            num_workers=num_workers, max_length=max_length)
        adv_pcap = pcap_dir / base / f"{base}_adversarial_{mode}.pcap"
        branches, rstats = realize_triplet(adv_pcap, flow_vectors, mean_std[0], mean_std[1])
        if rstats.get("packets_per_branch", 0) == 0:
            summary["skipped_sources"][gname] = rstats.get("error", "no realized packets")
            continue

        streams = {
            "clean": (branches["clean"]["packets"], branches["clean"]["keys"]),
            "adversarial": (branches["adv"]["packets"], branches["adv"]["keys"]),
        }
        if score_purified_all:
            streams["purified_all"] = (
                branches["purified"]["packets"], branches["purified"]["keys"])
        d_pkts, d_keys, gate_stats = pipeline.assemble_defended(
            branches, verdicts, gname, threshold)
        streams["defended"] = (d_pkts, d_keys)

        scored = {r: model.score(pk, ky) for r, (pk, ky) in streams.items()}
        scored_test = None
        if score_test_subset:
            wanted = {fid for (s, fid) in test_keys if s == gname}
            tp, tk = pipeline.subset_by_flows(d_pkts, d_keys, wanted)
            if tp:
                scored_test = model.score(tp, tk)

        summary["per_source"][gname] = _source_metrics(
            scored, scored_test, gate_stats, model, mode, rstats)
        summary["models"][base] = model.name
        for r, bscore in scored.items():
            nids_metrics.accumulate(acc_all, r, bscore)
            nids_metrics.accumulate(acc_kind.setdefault(model.kind, {}), r, bscore)

    if not summary["per_source"]:
        why = "; ".join(sorted(set(summary["skipped_sources"].values()))) or "no sources"
        raise RuntimeError(f"baseline produced no results ({why})")

    summary["overall"] = _finalize_overall(acc_all)
    summary["by_kind"] = {k: _finalize_overall(a) for k, a in sorted(acc_kind.items())}

    (output_dir / "baseline_metrics.json").write_text(
        json.dumps(summary, indent=2, default=str))
    _plot_baseline(summary, output_dir / "baseline_detection.png")
    return summary


def _plot_baseline(summary: dict, out_path: Path) -> None:
    per = summary.get("per_source", {})
    names = [n for n in sorted(per)
             if per[n]["detection_rate"].get("defended") is not None]
    if not names:
        return
    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(9, len(names) * 0.5), 8.5))

    def _get(regime):
        return [per[n]["detection_rate"].get(regime) for n in names]

    ax1.bar(x - 0.27, _get("clean"), 0.27, label="clean (oracle)", color="#2ca02c")
    ax1.bar(x, _get("adversarial"), 0.27, label="adversarial (evaded)", color="#d62728")
    ax1.bar(x + 0.27, _get("defended"), 0.27, label="DEPT-defended", color="#1f77b4")
    ax1.set_ylabel("NIDS detection rate")
    ax1.set_ylim(0, 1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=90, fontsize=7)
    ax1.set_title("NIDS detection: clean → perturbed → DEPT-defended")
    ax1.legend(fontsize=8)

    asr_adv = [per[n]["asr"].get("adversarial") for n in names]
    asr_def = [per[n]["asr"].get("defended") for n in names]
    ax2.bar(x - 0.2, asr_adv, 0.4, label="ASR before DEPT (adversarial)", color="#d62728")
    ax2.bar(x + 0.2, asr_def, 0.4, label="ASR after DEPT (defended)", color="#1f77b4")
    ax2.set_ylabel("attack success rate (1 − detection)")
    ax2.set_ylim(0, 1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=90, fontsize=7)
    ax2.set_title("Attack Success Rate before vs after DEPT")
    ax2.legend(fontsize=8)

    fig.suptitle("DEPT deployment baseline — protected NIDS efficacy", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


__all__ = ["evaluate"]
