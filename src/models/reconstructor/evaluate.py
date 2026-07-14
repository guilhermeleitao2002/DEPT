"""Evaluate the reconstructor on the held-out test pairs.

Four metric groups (the first three match the thesis's §5.3), reported
overall and per ``<base>_adv_<mode>`` source:

  1. **Reconstruction quality** — MSE on the perturbed dimensions, *before*
     (adv vs clean) vs *after* (recon vs clean). A large ``before → after``
     drop means the denoiser removed the perturbation.
  1b. **Independent-NIDS efficacy** — *the headline metric*. Realize the
     adversarial, purified and clean size/timing onto the *same* adversarial
     packets and score each through the real NIDS each attack was built to evade:
     **Kitsune** (KitNET + AfterImage) for the Kitsune-native attacks and
     TANTGen's **ML-NIDS** (CICFlowMeter classifiers) for the CIC attacks. Neither
     ever saw DEPT's models, so this breaks the TCN's circularity. Semantics =
     *detection restoration*: the NIDS flags a flow when its anomaly signal
     (Kitsune RMSE, or ML-NIDS ``P(malicious)``) is high; the attack evades by
     suppressing it, and purification recovers the original attack's fingerprint,
     so the signal / detection rise from the (evaded) adversarial floor back
     toward the clean-original oracle. Shared with the deployment baseline via
     :mod:`src.models.nids`.
  2. **Purification gain through the TCN** — *secondary, internal sanity check*.
     Feed the adversarial, reconstructed and clean sequences through DEPT's own
     trained TCN and report the per-flow last-timestep ``P(adversarial)``. This is
     **circular** (the TCN trained on this data) — kept only as an in-distribution
     diagnostic; the independent-NIDS metric above is the credible one.
  3. **Protocol-compliance rate** — a ``tshark`` check on the *realized*
     purified packets (best-effort; needs the adversarial PCAPs and the
     tshark binary).
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from ..common.runtime import resolve_num_workers  # noqa: E402
from ..nids import NIDSRegistry, NIDSUnavailable  # noqa: E402
from ..nids import metrics as nids_metrics  # noqa: E402
from ..tcn.model import TCN  # noqa: E402
from . import realize as rz  # noqa: E402
from .dataset import (  # noqa: E402
    PairedFlowDataset,
    build_pair_index,
    collate_pairs,
)
from .model import Reconstructor  # noqa: E402


log = logging.getLogger(__name__)

_NPRINT_BITS = 1088
_IPAT_COL = 1088
_DIR_COL = 1089


def _resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def _round_recon(seq: torch.Tensor) -> torch.Tensor:
    """Project a continuous reconstruction back onto the nPrint manifold.

    nPrint bit columns → {-1, 0, 1}; direction → {0, 1}; IPAT left continuous.
    """
    out = seq.clone()
    out[..., :_NPRINT_BITS] = out[..., :_NPRINT_BITS].round().clamp_(-1.0, 1.0)
    out[..., _DIR_COL] = out[..., _DIR_COL].round().clamp_(0.0, 1.0)
    return out


def _load_reconstructor(output_dir: Path, device: torch.device) -> Reconstructor:
    cfg = json.loads((output_dir / "reconstructor_config.json").read_text())
    m = cfg.get("model", {})
    model = Reconstructor(
        num_features=int(m.get("num_features", 1090)),
        d_model=int(m.get("d_model", 128)),
        n_layers=int(m.get("n_layers", 3)),
        n_heads=int(m.get("n_heads", 4)),
        dim_feedforward=int(m.get("dim_feedforward", 256)),
        dropout=float(m.get("dropout", 0.1)),
        max_seq_len=int(m.get("max_seq_len", 200)),
    ).to(device)
    state = torch.load(output_dir / "reconstructor.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _load_tcn(tcn_dir: Path, device: torch.device):
    cfg_path = tcn_dir / "tcn_config.json"
    if not cfg_path.exists() or not (tcn_dir / "tcn.pt").exists():
        return None
    cfg = json.loads(cfg_path.read_text())
    m = cfg.get("model", {})
    model = TCN(
        num_features=int(m.get("num_features", 1090)),
        channels=tuple(m.get("channels", (128, 128, 128))),
        kernel_size=int(m.get("kernel_size", 3)),
        dropout=float(m.get("dropout", 0.15)),
        num_classes=int(m.get("num_classes", 2)),
    ).to(device)
    state = torch.load(tcn_dir / "tcn.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.inference_mode()
def _tcn_p_adv(tcn, seq: torch.Tensor, lengths: torch.Tensor,
               device: torch.device) -> np.ndarray:
    """Per-flow last-timestep P(adversarial) from the TCN over ``seq``."""
    amp = device.type == "cuda"
    with torch.amp.autocast(device_type="cuda" if amp else "cpu", enabled=amp):
        logits = tcn(seq)                              # [B, T, 2]
    B, T, C = logits.shape
    idx = (lengths.to(device) - 1).clamp(min=0).view(B, 1, 1).expand(B, 1, C)
    last = logits.gather(dim=1, index=idx).squeeze(1).float()
    return torch.softmax(last, dim=1)[:, 1].detach().cpu().numpy().astype(np.float32)


def _new_group() -> dict:
    return {
        "sse_pert_before": 0.0, "sse_pert_after": 0.0, "n_pert": 0,
        "sse_all_before": 0.0, "sse_all_after": 0.0, "n_all": 0,
        "p_adv_adv": 0.0, "p_adv_recon": 0.0, "p_adv_clean": 0.0,
        "det_adv": 0, "det_recon": 0, "n_flows": 0,
    }


def _finalize_group(g: dict) -> dict:
    nf = max(g["n_flows"], 1)
    out = {
        "n_flows": int(g["n_flows"]),
        "reconstruction": {
            "perturbed_mse_before": (g["sse_pert_before"] / g["n_pert"]) if g["n_pert"] else None,
            "perturbed_mse_after": (g["sse_pert_after"] / g["n_pert"]) if g["n_pert"] else None,
            "all_mse_before": (g["sse_all_before"] / g["n_all"]) if g["n_all"] else None,
            "all_mse_after": (g["sse_all_after"] / g["n_all"]) if g["n_all"] else None,
            "n_perturbed_elems": int(g["n_pert"]),
        },
        "purification": {
            "p_adv_on_adv": g["p_adv_adv"] / nf,
            "p_adv_on_recon": g["p_adv_recon"] / nf,
            "p_adv_on_clean": g["p_adv_clean"] / nf,
            "gain": (g["p_adv_adv"] - g["p_adv_recon"]) / nf,
            "residual_gap": (g["p_adv_recon"] - g["p_adv_clean"]) / nf,
            "detection_rate_adv": g["det_adv"] / nf,
            "detection_rate_recon": g["det_recon"] / nf,
        },
        # tshark_rate (traffic-space realized-packet compliance) is filled in
        # later by the realization pass when PCAPs + tshark are available.
        "protocol_compliance": {},
    }
    if g["n_pert"] and out["reconstruction"]["perturbed_mse_before"]:
        before = out["reconstruction"]["perturbed_mse_before"]
        after = out["reconstruction"]["perturbed_mse_after"]
        out["reconstruction"]["reduction_ratio"] = (before / after) if after else None
    return out


def evaluate(config: dict, output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    train_cfg = dict(config.get("training", {}))
    device = _resolve_device(str(train_cfg.get("device", "auto")))

    model = _load_reconstructor(output_dir, device)
    # Rebuild the (deterministic) pair table from config — its row order matches
    # the one training split, so split.npz's test_idx aligns 1:1. This avoids
    # persisting a multi-hundred-MB index, mirroring how the other DEPT models
    # re-derive their flow index at eval time.
    pairs = build_pair_index(
        tensors_dir=Path(config["tensors_dir"]),
        adversarial_dir=Path(config["adversarial_dir"]),
        adversarial_exclude=list(config.get("adversarial_exclude", [])),
        min_len=int(config.get("min_len", 2)),
    )
    test_idx = np.load(output_dir / "split.npz")["test_idx"]

    ds = PairedFlowDataset(pairs, sample_indices=test_idx,
                           max_length=train_cfg.get("max_length", None))
    num_workers = resolve_num_workers(train_cfg.get("num_workers", "auto"))
    loader = DataLoader(ds, batch_size=int(train_cfg.get("batch_size", 32)),
                        shuffle=False, num_workers=num_workers,
                        pin_memory=device.type == "cuda",
                        persistent_workers=num_workers > 0,
                        collate_fn=collate_pairs)

    tcn = _load_tcn(Path(config["tcn_model_dir"]), device)
    if tcn is None:
        log.warning("no trained TCN at %s — purification gain skipped",
                    config.get("tcn_model_dir"))

    groups: dict[str, dict] = defaultdict(_new_group)
    overall = _new_group()
    # Compact per-source reconstruction for realization: source -> {fid: (tl, ipat)}
    realize_pool: dict[str, dict[str, tuple]] = defaultdict(dict)
    # NIDS 3-way pool: source -> {fid: {"adv"/"recon"/"clean": (tl, ipat)}}. Carries
    # the adversarial and clean-target size/timing alongside the reconstruction so the
    # independent-NIDS metric can realize all three from the same adversarial packets.
    nids_pool: dict[str, dict[str, dict]] = defaultdict(dict)

    src_names = np.array(
        [f"{b}_adv_{m}" for b, m in zip(pairs.base, pairs.mode)], dtype=object)

    for adv, clean, lengths, mask, rows in loader:
        adv_d = adv.to(device, non_blocking=True)
        mask_d = mask.to(device, non_blocking=True)
        with torch.inference_mode():
            recon_d = model(adv_d, key_padding_mask=~mask_d)
        recon_round = _round_recon(recon_d)

        p_adv = p_recon = p_clean = None
        if tcn is not None:
            clean_d = clean.to(device, non_blocking=True)
            p_adv = _tcn_p_adv(tcn, adv_d, lengths, device)
            p_recon = _tcn_p_adv(tcn, recon_round, lengths, device)
            p_clean = _tcn_p_adv(tcn, clean_d, lengths, device)

        # Per-flow accumulation (CPU numpy).
        adv_np = adv.numpy()
        clean_np = clean.numpy()
        recon_np = recon_d.detach().cpu().numpy()
        recon_round_np = recon_round.detach().cpu().numpy()
        rows_np = rows.numpy()
        len_np = lengths.numpy()

        for b in range(adv_np.shape[0]):
            L = int(len_np[b])
            row = int(rows_np[b])
            gname = str(src_names[row])
            a = adv_np[b, :L]
            c = clean_np[b, :L]
            r = recon_np[b, :L]
            rr = recon_round_np[b, :L]

            pert = (a != c)
            n_pert = int(pert.sum())
            sse_pb = float((((a - c) ** 2) * pert).sum())
            sse_pa = float((((r - c) ** 2) * pert).sum())
            sse_ab = float(((a - c) ** 2).sum())
            sse_aa = float(((r - c) ** 2).sum())
            n_all = int(a.size)

            for tgt in (groups[gname], overall):
                tgt["sse_pert_before"] += sse_pb
                tgt["sse_pert_after"] += sse_pa
                tgt["n_pert"] += n_pert
                tgt["sse_all_before"] += sse_ab
                tgt["sse_all_after"] += sse_aa
                tgt["n_all"] += n_all
                tgt["n_flows"] += 1
                if tcn is not None:
                    tgt["p_adv_adv"] += float(p_adv[b])
                    tgt["p_adv_recon"] += float(p_recon[b])
                    tgt["p_adv_clean"] += float(p_clean[b])
                    tgt["det_adv"] += int(p_adv[b] >= 0.5)
                    tgt["det_recon"] += int(p_recon[b] >= 0.5)

            fid = str(pairs.flow_id[row])
            tl = rz.decode_total_len_batch(rr)
            realize_pool[gname][fid] = (tl, rr[:, _IPAT_COL].astype(np.float32))
            # Same decode for the adversarial and clean tensors (both already valid
            # nPrint, so rounding is a no-op) → three aligned size/timing overlays.
            nids_pool[gname][fid] = {
                "adv": (rz.decode_total_len_batch(a), a[:, _IPAT_COL].astype(np.float32)),
                "recon": (tl, rr[:, _IPAT_COL].astype(np.float32)),
                "clean": (rz.decode_total_len_batch(c), c[:, _IPAT_COL].astype(np.float32)),
            }

    metrics = {
        "overall": _finalize_group(overall),
        "per_source": {g: _finalize_group(s) for g, s in sorted(groups.items())},
        "device": str(device),
        "n_test_pairs": int(len(test_idx)),
        "tcn_used": tcn is not None,
    }

    # ---- Traffic-space realization + tshark protocol compliance --------------
    # Best-effort: skipped inside when the adversarial PCAPs or tshark are absent.
    metrics["realization"] = _run_realization(
        config, realize_pool, output_dir, metrics)

    # ---- Independent-NIDS efficacy (the headline metric) ---------------------
    # Realize adv / purified / clean packets and score each through the real NIDS each
    # attack was built to evade (Kitsune for the Kitsune-native attacks, ML-NIDS for
    # the CIC attacks) — detectors DEPT never trained, so this breaks the TCN's
    # circularity. Best-effort: a source is skipped-with-reason when its NIDS/deps are
    # unavailable (so the reconstruction + TCN groups still report).
    metrics["nids"] = _run_nids_eval(config, nids_pool, output_dir, metrics)

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    # ---- Plots ---------------------------------------------------------------
    _plot_reconstruction_error(metrics, output_dir / "reconstruction_error.png")
    _plot_purification_gain(metrics, output_dir / "purification_gain.png")
    _plot_nids_restoration(metrics, output_dir / "nids_restoration.png")
    _plot_training_curves(output_dir / "training_curves.json",
                          output_dir / "training_curves.png")

    return metrics


def _run_realization(config, realize_pool, output_dir, metrics):
    """Realize every reconstructed flow per source → PCAP → tshark compliance."""
    pcap_dir = Path(config.get("realize", {}).get(
        "adversarial_pcap_dir", "data/adversarial"))
    if not pcap_dir.is_absolute():
        pcap_dir = Path(config.get("_repo_root", ".")) / pcap_dir
    realized_dir = output_dir / "realized"

    if not rz.tshark_available():
        return {"skipped": "tshark not found on PATH"}

    mean_std = _load_ipat_stats(config["tensors_dir"])
    summary = {"per_source": {}, "tshark_n_packets": 0, "tshark_n_bad": 0,
               "unrealizable_l7": 0}
    for gname, flow_recon in sorted(realize_pool.items()):
        # gname = "<base>_adv_<mode>"; locate the original adversarial PCAP.
        base, mode = gname.rsplit("_adv_", 1)
        adv_pcap = pcap_dir / base / f"{base}_adversarial_{mode}.pcap"
        packets, rstats = rz.realize_source(
            adv_pcap, flow_recon, mean_std[0], mean_std[1])
        entry = {"realize": rstats}
        # Size-edited application-layer text packets (SSDP/HTTP) are excluded
        # from the realized PCAP (can't be faithfully resized by byte-overlay);
        # record the count alongside the rate so it reads "N excluded", not
        # "N failures".
        n_l7 = int(rstats.get("unrealizable_l7", 0))
        summary["unrealizable_l7"] += n_l7
        if gname in metrics["per_source"]:
            metrics["per_source"][gname]["protocol_compliance"][
                "n_unrealizable_l7"] = n_l7
        if packets:
            out_pcap = realized_dir / f"{gname}.pcap"
            if rz.write_pcap(packets, out_pcap):
                tsh = rz.tshark_compliance(out_pcap)
                entry["tshark"] = tsh
                if "n_packets" in tsh:
                    summary["tshark_n_packets"] += tsh["n_packets"]
                    summary["tshark_n_bad"] += tsh["n_bad"]
                    # Fold tshark rate into the per-source protocol block.
                    if gname in metrics["per_source"]:
                        metrics["per_source"][gname]["protocol_compliance"][
                            "tshark_rate"] = tsh.get("compliance_rate")
        summary["per_source"][gname] = entry

    n = summary["tshark_n_packets"]
    summary["tshark_compliance_rate"] = (
        float((n - summary["tshark_n_bad"]) / n) if n else None)
    if metrics["overall"]["protocol_compliance"] is not None and n:
        metrics["overall"]["protocol_compliance"]["tshark_rate"] = \
            summary["tshark_compliance_rate"]
    if metrics["overall"]["protocol_compliance"] is not None:
        metrics["overall"]["protocol_compliance"]["n_unrealizable_l7"] = \
            summary["unrealizable_l7"]
    return summary


def _load_ipat_stats(tensors_dir: Path | str) -> tuple[float, float]:
    """IPAT (mean, std) from the shared data/tensors/mean_std.json (defaults 0, 1)."""
    p = Path(tensors_dir) / "mean_std.json"
    if not p.exists():
        return 0.0, 1.0
    try:
        stats = json.loads(p.read_text())
        ip = stats.get("IPAT", {})
        return float(ip.get("mean", 0.0)), float(ip.get("std", 1.0))
    except Exception:
        return 0.0, 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Independent-NIDS efficacy (the headline metric)
# ──────────────────────────────────────────────────────────────────────────────

_NIDS_BRANCHES = ("adv", "purified", "clean")


def _abs(p, anchor: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else anchor / p


def _run_nids_eval(config, nids_pool, output_dir, metrics) -> dict:
    """Score adv / purified / clean realized packets through each attack's real NIDS.

    For each ``<base>_adv_<mode>`` source we realize three aligned packet sets (same
    flows / identity / payload; only size+timing differ) and score each through the
    NIDS ``<base>`` was built to evade — Kitsune for the Kitsune-native attacks,
    TANTGen's ML-NIDS for the CIC attacks (resolved via :class:`NIDSRegistry`).
    Detection *restoration*: purification should raise the NIDS anomaly signal from the
    (evaded) adversarial floor toward the clean-original oracle. Best-effort per source
    — a source whose NIDS or its dependencies are missing is recorded under
    ``skipped_sources`` so the reconstruction + TCN metric groups still report.

    ``score_restoration_ratio`` (per source, single NIDS) is the headline; the
    cross-family ``overall`` reads ``detection_restoration_ratio`` (rate-based, so it is
    safe to aggregate Kitsune and ML-NIDS together), with per-family numbers in
    ``by_kind``.
    """
    ncfg = dict(config.get("nids", {}))
    if not ncfg.get("enabled", True):
        return {"skipped": "nids.enabled is false"}
    repo_root = Path(config.get("_repo_root", "."))
    reg = NIDSRegistry(ncfg, repo_root)

    pcap_dir = _abs(config.get("realize", {}).get(
        "adversarial_pcap_dir", "data/adversarial"), repo_root)
    realized_dir = output_dir / "realized_nids"
    mean_std = _load_ipat_stats(config["tensors_dir"])

    summary = {"per_source": {}, "skipped_sources": {}, "models": {}}
    acc_all: dict = {}
    acc_kind: dict[str, dict] = {}

    for gname, flows in sorted(nids_pool.items()):
        base, mode = gname.rsplit("_adv_", 1)
        if not reg.known(base):
            summary["skipped_sources"][gname] = f"no NIDS mapping for base '{base}'"
            continue
        try:
            model = reg.get(base)
        except NIDSUnavailable as e:
            summary["skipped_sources"][gname] = str(e)
            continue

        adv_pcap = pcap_dir / base / f"{base}_adversarial_{mode}.pcap"
        branches, rstats = rz.realize_triplet(adv_pcap, flows, mean_std[0], mean_std[1])
        entry = {"realize": rstats, "model": model.name,
                 "model_source": getattr(model, "model_source", "?"), "kind": model.kind}
        if rstats.get("packets_per_branch", 0) == 0:
            entry["skipped"] = rstats.get("error", "no realized packets")
            summary["per_source"][gname] = entry
            continue

        scored = {}
        for br in _NIDS_BRANCHES:
            pkts, keys = branches[br]["packets"], branches[br]["keys"]
            rz.write_pcap(pkts, realized_dir / f"{gname}_{br}.pcap")
            bscore = model.score(pkts, keys)
            scored[br] = bscore
            nids_metrics.accumulate(acc_all, br, bscore)
            nids_metrics.accumulate(acc_kind.setdefault(model.kind, {}), br, bscore)

        rm = nids_metrics.finalize_restoration(
            scored["adv"], scored["purified"], scored["clean"],
            threshold=model.threshold, model_name=model.name)
        rm["kind"] = model.kind
        entry.update(rm)
        summary["per_source"][gname] = entry
        summary["models"][base] = model.name
        if gname in metrics["per_source"]:
            metrics["per_source"][gname]["nids"] = rm

    if not summary["per_source"]:
        why = "; ".join(sorted(set(summary["skipped_sources"].values()))) or "no sources"
        return {"skipped": f"all NIDS sources skipped ({why})",
                "skipped_sources": summary["skipped_sources"]}

    summary["overall"] = nids_metrics.finalize_restoration_overall(acc_all)
    summary["by_kind"] = {
        k: nids_metrics.finalize_restoration_overall(a)
        for k, a in sorted(acc_kind.items())}
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def _plot_reconstruction_error(metrics: dict, out_path: Path) -> None:
    per = metrics.get("per_source", {})
    names = [n for n in per if per[n]["reconstruction"]["perturbed_mse_before"] is not None]
    if not names:
        return
    before = [per[n]["reconstruction"]["perturbed_mse_before"] for n in names]
    after = [per[n]["reconstruction"]["perturbed_mse_after"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 0.45), 4.5))
    ax.bar(x - 0.2, before, width=0.4, label="before (adv vs clean)", color="#d62728")
    ax.bar(x + 0.2, after, width=0.4, label="after (recon vs clean)", color="#2ca02c")
    ax.set_yscale("log")
    ax.set_ylabel("MSE on perturbed dims (log)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_title("Reconstruction error on perturbed dimensions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_purification_gain(metrics: dict, out_path: Path) -> None:
    if not metrics.get("tcn_used"):
        return
    per = metrics.get("per_source", {})
    names = sorted(per.keys())
    if not names:
        return
    p_adv = [per[n]["purification"]["p_adv_on_adv"] for n in names]
    p_rec = [per[n]["purification"]["p_adv_on_recon"] for n in names]
    p_cln = [per[n]["purification"]["p_adv_on_clean"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 0.5), 4.5))
    ax.bar(x - 0.27, p_adv, width=0.27, label="P(adv | adversarial)", color="#d62728")
    ax.bar(x, p_rec, width=0.27, label="P(adv | reconstructed)", color="#1f77b4")
    ax.bar(x + 0.27, p_cln, width=0.27, label="P(adv | clean)", color="#2ca02c")
    ax.set_ylabel("TCN P(adversarial), last timestep")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_title("Purification gain (through the TCN)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_nids_restoration(metrics: dict, out_path: Path) -> None:
    """Detection restoration through the independent NIDS: detection rate + mean score."""
    nids = metrics.get("nids", {})
    per = nids.get("per_source", {})
    names = [n for n in sorted(per)
             if per[n].get("detection_rate_purified") is not None]
    if not names:
        return
    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(11, len(names) * 0.8), 5.0))

    def _tri(ax, key, ylabel, title, ylim=None):
        adv = [per[n].get(f"{key}_adv") for n in names]
        pur = [per[n].get(f"{key}_purified") for n in names]
        cln = [per[n].get(f"{key}_clean") for n in names]
        ax.bar(x - 0.27, adv, 0.27, label="adversarial (evaded)", color="#d62728")
        ax.bar(x, pur, 0.27, label="purified", color="#1f77b4")
        ax.bar(x + 0.27, cln, 0.27, label="clean (oracle)", color="#2ca02c")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=90, fontsize=7)
        ax.set_title(title)

    _tri(ax1, "detection_rate", "NIDS detection rate",
         "Detection restoration (independent NIDS)", ylim=(0, 1))
    ax1.legend(fontsize=7)
    # mean_score = Kitsune RMSE for Kitsune-native attacks, ML-NIDS P(malicious) for
    # the CIC attacks (different scales — read per-source, not across sources).
    _tri(ax2, "mean_score", "mean per-flow NIDS anomaly score",
         "Anomaly score (adv → purified → clean)")
    fig.suptitle("Purification efficacy through the independent NIDS — "
                 "purified should rise from adv toward clean", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_training_curves(curves_path: Path, out_path: Path) -> None:
    if not curves_path.exists():
        return
    curves = json.loads(curves_path.read_text())
    if not curves:
        return
    epochs = [c["epoch"] for c in curves]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, [c["train_loss"] for c in curves], label="train loss", marker="o")
    ax.plot(epochs, [c["val_loss"] for c in curves], label="val loss", marker="s")
    ax.set_xlabel("epoch")
    ax.set_ylabel("smooth-L1 reconstruction loss")
    ax.set_title("Reconstructor training curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
