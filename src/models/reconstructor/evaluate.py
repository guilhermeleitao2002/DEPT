"""Evaluate the reconstructor on the held-out test pairs.

Three metric groups (matching the thesis's §5.3), reported overall and per
``<base>_adv_<mode>`` source:

  1. **Reconstruction quality** — MSE on the perturbed dimensions, *before*
     (adv vs clean) vs *after* (recon vs clean). A large ``before → after``
     drop means the denoiser removed the perturbation.
  2. **Purification gain** — feed the adversarial, reconstructed and clean
     sequences through the trained TCN and report the per-flow last-timestep
     ``P(adversarial)``. ``gain = P(adv|adv) − P(adv|recon)``; ideally the
     reconstruction lands near the clean floor ``P(adv|clean)``.
  3. **Protocol-compliance rate** — a feature-space TCP/IP invariant check on
     the rounded reconstruction (always on), plus a ``tshark`` check on the
     *realized* purified packets (best-effort; needs the adversarial PCAPs and
     the tshark binary).
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
        "compl_checked": 0, "compl_valid": 0,
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
        "protocol_compliance": {
            "feature_space_checked": int(g["compl_checked"]),
            "feature_space_rate": (g["compl_valid"] / g["compl_checked"]) if g["compl_checked"] else None,
        },
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
    # persisting a multi-hundred-MB index, mirroring how the other ADEPT models
    # re-derive their flow index at eval time.
    pairs = build_pair_index(
        tensors_dir=Path(config["tensors_dir"]),
        adversarial_dir=Path(config["adversarial_dir"]),
        adversarial_exclude=list(config.get("adversarial_exclude", [])),
        min_len=int(config.get("min_len", 2)),
    )
    test_idx = np.load(output_dir / "split.npz")["test_idx"]
    # Honour a smoke-test cap so eval matches the trained subsample.
    max_flows = config.get("_max_flows")
    if max_flows:
        rng = np.random.default_rng(int(config["split"]["seed"]))
        if len(test_idx) > int(max_flows):
            pick = rng.choice(len(test_idx), size=int(max_flows), replace=False)
            test_idx = np.sort(test_idx[pick])

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

    realize_cfg = dict(config.get("realize", {}))
    tshark_sample = int(realize_cfg.get("tshark_sample", 300))
    do_realize = bool(realize_cfg.get("write_pcap", True)) and tshark_sample > 0

    groups: dict[str, dict] = defaultdict(_new_group)
    overall = _new_group()
    # Compact per-source reconstruction for realization: source -> {fid: (tl, ipat)}
    realize_pool: dict[str, dict[str, tuple]] = defaultdict(dict)

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

            n_chk, n_val = rz.nprint_ipv4_invariants(rr)

            for tgt in (groups[gname], overall):
                tgt["sse_pert_before"] += sse_pb
                tgt["sse_pert_after"] += sse_pa
                tgt["n_pert"] += n_pert
                tgt["sse_all_before"] += sse_ab
                tgt["sse_all_after"] += sse_aa
                tgt["n_all"] += n_all
                tgt["compl_checked"] += n_chk
                tgt["compl_valid"] += n_val
                tgt["n_flows"] += 1
                if tcn is not None:
                    tgt["p_adv_adv"] += float(p_adv[b])
                    tgt["p_adv_recon"] += float(p_recon[b])
                    tgt["p_adv_clean"] += float(p_clean[b])
                    tgt["det_adv"] += int(p_adv[b] >= 0.5)
                    tgt["det_recon"] += int(p_recon[b] >= 0.5)

            if do_realize and len(realize_pool[gname]) < tshark_sample:
                fid = str(pairs.flow_id[row])
                tl = rz.decode_total_len_batch(rr)
                realize_pool[gname][fid] = (tl, rr[:, _IPAT_COL].astype(np.float32))

    metrics = {
        "overall": _finalize_group(overall),
        "per_source": {g: _finalize_group(s) for g, s in sorted(groups.items())},
        "device": str(device),
        "n_test_pairs": int(len(test_idx)),
        "tcn_used": tcn is not None,
    }

    # ---- Traffic-space realization + tshark protocol compliance --------------
    if do_realize:
        metrics["realization"] = _run_realization(
            config, realize_pool, pairs, output_dir, realize_cfg, metrics)
    else:
        metrics["realization"] = {"skipped": "write_pcap disabled or tshark_sample=0"}

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    # ---- Plots ---------------------------------------------------------------
    _plot_reconstruction_error(metrics, output_dir / "reconstruction_error.png")
    _plot_purification_gain(metrics, output_dir / "purification_gain.png")
    _plot_training_curves(output_dir / "training_curves.json",
                          output_dir / "training_curves.png")

    return metrics


def _run_realization(config, realize_pool, pairs, output_dir, realize_cfg, metrics):
    """Realize sampled flows per source → PCAP → tshark compliance."""
    pcap_dir = Path(config.get("realize", {}).get(
        "adversarial_pcap_dir", "data/adversarial"))
    if not pcap_dir.is_absolute():
        pcap_dir = Path(config.get("_repo_root", ".")) / pcap_dir
    max_pcap = int(realize_cfg.get("max_pcap_packets", 100_000))
    realized_dir = output_dir / "realized"

    if not rz.tshark_available():
        return {"skipped": "tshark not found on PATH"}

    summary = {"per_source": {}, "tshark_n_packets": 0, "tshark_n_bad": 0}
    for gname, flow_recon in sorted(realize_pool.items()):
        # gname = "<base>_adv_<mode>"; locate the original adversarial PCAP.
        base, mode = gname.rsplit("_adv_", 1)
        adv_pcap = pcap_dir / base / f"{base}_adversarial_{mode}.pcap"
        mean_std = _load_ipat_stats(pairs.adv_folders.get(gname))
        packets, rstats = rz.realize_source(
            adv_pcap, flow_recon, mean_std[0], mean_std[1], max_pcap_packets=max_pcap)
        entry = {"realize": rstats}
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
    return summary


def _load_ipat_stats(folder: Path | None) -> tuple[float, float]:
    """IPAT (mean, std) from a source's mean_std.json (defaults 0, 1)."""
    if folder is None:
        return 0.0, 1.0
    p = Path(folder) / "mean_std.json"
    if not p.exists():
        return 0.0, 1.0
    try:
        stats = json.loads(p.read_text())
        ip = stats.get("IPAT", {})
        return float(ip.get("mean", 0.0)), float(ip.get("std", 1.0))
    except Exception:
        return 0.0, 1.0


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
