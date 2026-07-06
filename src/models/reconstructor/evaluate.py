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
  3. **Protocol-compliance rate** — a ``tshark`` check on the *realized*
     purified packets (best-effort; needs the adversarial PCAPs and the
     tshark binary).
  4. **Independent-NIDS efficacy (Kitsune)** — the headline metric. Realize the
     adversarial, purified and clean size/timing onto the *same* adversarial
     packets and score each through **Kitsune** (KitNET + AfterImage), the real
     anomaly detector TANTGen's attacks evade. Kitsune never saw ADEPT's models,
     so it breaks the TCN's circularity. Semantics = *detection restoration*
     (opposite direction to group 2): Kitsune flags a flow when ``RMSE > φ``; the
     attack evades by suppressing RMSE, and purification recovers the original
     attack's fingerprint, so RMSE / detection rise from the (evaded) adversarial
     floor back toward the clean-original oracle. Best-effort; needs the sibling
     TANTGen Kitsune tree.
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
    # persisting a multi-hundred-MB index, mirroring how the other ADEPT models
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
    # Kitsune 3-way pool: source -> {fid: {"adv"/"recon"/"clean": (tl, ipat)}}. Carries
    # the adversarial and clean-target size/timing alongside the reconstruction so the
    # independent-NIDS metric can realize all three from the same adversarial packets.
    kitsune_pool: dict[str, dict[str, dict]] = defaultdict(dict)

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
            kitsune_pool[gname][fid] = {
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
        config, realize_pool, pairs, output_dir, metrics)

    # ---- Independent-NIDS efficacy (Kitsune) ---------------------------------
    # The headline efficacy metric: realize adv / purified / clean packets and score
    # each through Kitsune (KitNET), an anomaly detector ADEPT never trained — so it
    # breaks the TCN's circularity. Best-effort: skipped inside when disabled or the
    # sibling TANTGen Kitsune tree is unavailable.
    metrics["kitsune"] = _run_kitsune_eval(
        config, kitsune_pool, pairs, output_dir, metrics)

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    # ---- Plots ---------------------------------------------------------------
    _plot_reconstruction_error(metrics, output_dir / "reconstruction_error.png")
    _plot_purification_gain(metrics, output_dir / "purification_gain.png")
    _plot_kitsune_restoration(metrics, output_dir / "kitsune_restoration.png")
    _plot_training_curves(output_dir / "training_curves.json",
                          output_dir / "training_curves.png")

    return metrics


def _run_realization(config, realize_pool, pairs, output_dir, metrics):
    """Realize every reconstructed flow per source → PCAP → tshark compliance."""
    pcap_dir = Path(config.get("realize", {}).get(
        "adversarial_pcap_dir", "data/adversarial"))
    if not pcap_dir.is_absolute():
        pcap_dir = Path(config.get("_repo_root", ".")) / pcap_dir
    realized_dir = output_dir / "realized"

    if not rz.tshark_available():
        return {"skipped": "tshark not found on PATH"}

    summary = {"per_source": {}, "tshark_n_packets": 0, "tshark_n_bad": 0,
               "unrealizable_l7": 0}
    for gname, flow_recon in sorted(realize_pool.items()):
        # gname = "<base>_adv_<mode>"; locate the original adversarial PCAP.
        base, mode = gname.rsplit("_adv_", 1)
        adv_pcap = pcap_dir / base / f"{base}_adversarial_{mode}.pcap"
        mean_std = _load_ipat_stats(pairs.adv_folders.get(gname))
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
# Independent-NIDS efficacy (Kitsune)
# ──────────────────────────────────────────────────────────────────────────────

_KITSUNE_BRANCHES = ("adv", "purified", "clean")


def _sub(a, b):
    return (a - b) if (a is not None and b is not None) else None


def _restoration_ratio(adv, pur, clean):
    """Fraction of the adv→clean anomaly-score gap the purifier recovers, in [~0, ~1].

    ``0`` = no restoration (purified ≈ adversarial), ``1`` = full restoration
    (purified ≈ clean original). Robust to Kitsune's absolute RMSE scale / threshold
    (which does not transfer across feeding regimes), unlike the detection rate.
    ``None`` when the gap is degenerate (clean ≈ adv, i.e. nothing to restore).
    """
    if adv is None or pur is None or clean is None:
        return None
    gap = clean - adv
    if abs(gap) < 1e-9:
        return None
    return (pur - adv) / gap


def _flow_agg(rmse: np.ndarray, keys: list, detection: str) -> dict:
    """Aggregate per-packet RMSE into a per-flow score (``max`` or ``mean``).

    ``rmse[i]`` and ``keys[i]`` are aligned (Kitsune consumed packets in list order);
    if extraction stopped early ``len(rmse) < len(keys)`` — we zip the common prefix.
    """
    per: dict[str, list] = defaultdict(list)
    for i in range(min(len(rmse), len(keys))):
        per[keys[i]].append(float(rmse[i]))
    reduce = np.mean if detection == "mean" else np.max
    return {k: float(reduce(v)) for k, v in per.items()}


def _branch_stats(agg: dict, threshold: float) -> dict:
    vals = np.fromiter(agg.values(), dtype=np.float64) if agg else np.zeros(0)
    n = int(vals.size)
    return {
        "mean_rmse": float(vals.mean()) if n else None,
        "detection_rate": float((vals > threshold).mean()) if n else None,
        "n_flows": n,
        "sum_rmse": float(vals.sum()),
        "n_detected": int((vals > threshold).sum()),
    }


def _resolve_kitsune_model(krun, base, tantgen_root, model_dir, benign_dir,
                           pretrained, base_to_tg, *, fm, ad, cap, seed) -> dict:
    """Locate (or train on demand) the Kitsune model for ``base``.

    Priority: TANTGen's calibrated pretrained pickle → a previously ADEPT-trained
    pickle → train a fresh one on the benign split. Returns
    ``{"kitnet", "threshold", "source"}`` or ``{"error": reason}``.
    """
    rel = pretrained.get(base)
    if rel:
        p = Path(rel)
        if not p.is_absolute():
            p = tantgen_root / p
        if p.exists():
            try:
                kn, thr = krun.load_model(p, tantgen_root)
                return {"kitnet": kn, "threshold": thr, "source": "pretrained"}
            except Exception as e:  # pragma: no cover - defensive
                return {"error": f"load pretrained failed: {e}"}

    local = model_dir / f"{base}_model.pkl"
    if local.exists():
        try:
            kn, thr = krun.load_model(local, tantgen_root)
            return {"kitnet": kn, "threshold": thr, "source": "trained"}
        except Exception as e:  # pragma: no cover - defensive
            return {"error": f"load trained failed: {e}"}

    tg = base_to_tg.get(base)
    if not tg:
        return {"error": f"no pretrained/trained model and no base_to_tg entry for '{base}'"}
    benign_pcap = benign_dir / f"{tg}_benign.pcap"
    if not benign_pcap.exists():
        return {"error": f"benign PCAP not found: {benign_pcap}"}
    try:
        log.info("training Kitsune for %s from %s", base, benign_pcap)
        krun.train_model(benign_pcap, local, tantgen_root, fm=fm, ad=ad, cap=cap, seed=seed)
        kn, thr = krun.load_model(local, tantgen_root)
        return {"kitnet": kn, "threshold": thr, "source": "trained"}
    except Exception as e:
        return {"error": f"train failed: {e}"}


def _run_kitsune_eval(config, kitsune_pool, pairs, output_dir, metrics) -> dict:
    """Score adv / purified / clean realized packets through Kitsune (independent NIDS).

    For each ``<base>_adv_<mode>`` source we realize three aligned packet sets and score
    each through the calibrated Kitsune for ``<base>``. Detection *restoration*:
    purification should raise Kitsune's RMSE / detection rate from the (evaded)
    adversarial floor back toward the clean-original attack (the oracle ceiling).
    """
    kcfg = dict(config.get("kitsune", {}))
    if not kcfg.get("enabled", False):
        return {"skipped": "kitsune.enabled is false"}
    try:
        from ..kitsune import runner as krun
    except Exception as e:  # pragma: no cover - defensive
        return {"skipped": f"kitsune adapter import failed: {e}"}

    repo_root = Path(config.get("_repo_root", "."))

    def _abs(p, anchor=repo_root):
        p = Path(p)
        return p if p.is_absolute() else anchor / p

    tantgen_root = _abs(kcfg.get("tantgen_root", ""))
    ok, why = krun.available(tantgen_root)
    if not ok:
        return {"skipped": f"kitsune unavailable: {why}"}

    model_dir = _abs(kcfg.get("model_dir", "data/models/kitsune"))
    benign_dir = _abs(kcfg.get("benign_pcap_dir", "data/original/Kitsune/Benign"), tantgen_root)
    pretrained = kcfg.get("pretrained", {}) or {}
    base_to_tg = kcfg.get("base_to_tg", {}) or {}
    # Restrict to the Kitsune-native attacks TANTGen calibrated (empty = no restriction).
    sources = set(kcfg.get("sources", []) or [])
    detection = str(kcfg.get("detection", "max"))
    fm, ad = int(kcfg.get("fm_grace", 5000)), int(kcfg.get("ad_grace", 50000))
    cap, seed = int(kcfg.get("train_cap_packets", 55000)), int(kcfg.get("seed", 42))

    pcap_dir = _abs(config.get("realize", {}).get("adversarial_pcap_dir", "data/adversarial"))
    realized_dir = output_dir / "realized_kitsune"

    model_cache: dict[str, dict] = {}
    summary = {"detection": detection, "scope": sorted(sources) if sources else "all",
               "per_source": {}, "skipped_sources": {}}
    acc = {br: {"sum_rmse": 0.0, "n_detected": 0, "n_flows": 0} for br in _KITSUNE_BRANCHES}

    for gname, flows in sorted(kitsune_pool.items()):
        base, mode = gname.rsplit("_adv_", 1)
        if sources and base not in sources:
            continue  # out of scope (only the Kitsune-native attacks are evaluated)
        if base not in model_cache:
            model_cache[base] = _resolve_kitsune_model(
                krun, base, tantgen_root, model_dir, benign_dir, pretrained,
                base_to_tg, fm=fm, ad=ad, cap=cap, seed=seed)
        resolved = model_cache[base]
        if "error" in resolved:
            summary["skipped_sources"][gname] = resolved["error"]
            continue
        kitnet, threshold, source = (
            resolved["kitnet"], resolved["threshold"], resolved["source"])

        adv_pcap = pcap_dir / base / f"{base}_adversarial_{mode}.pcap"
        mean_std = _load_ipat_stats(pairs.adv_folders.get(gname))
        branches, rstats = rz.realize_triplet(adv_pcap, flows, mean_std[0], mean_std[1])
        entry = {"realize": rstats, "threshold": threshold, "model_source": source}
        if rstats.get("packets_per_branch", 0) == 0:
            entry["skipped"] = rstats.get("error", "no realized packets")
            summary["per_source"][gname] = entry
            continue

        bs = {}
        for br in _KITSUNE_BRANCHES:
            pkts, keys = branches[br]["packets"], branches[br]["keys"]
            rz.write_pcap(pkts, realized_dir / f"{gname}_{br}.pcap")
            feats = krun.extract_features(pkts, tantgen_root)
            rmse = krun.score(kitnet, feats)
            bs[br] = _branch_stats(_flow_agg(rmse, keys, detection), threshold)
            acc[br]["sum_rmse"] += bs[br]["sum_rmse"]
            acc[br]["n_detected"] += bs[br]["n_detected"]
            acc[br]["n_flows"] += bs[br]["n_flows"]

        km = _finalize_kitsune(bs, threshold, source)
        entry.update(km)
        summary["per_source"][gname] = entry
        if gname in metrics["per_source"]:
            metrics["per_source"][gname]["kitsune"] = km

    summary["overall"] = _finalize_kitsune_overall(acc)
    return summary


def _finalize_kitsune(bs: dict, threshold: float, source: str) -> dict:
    d_adv = bs["adv"]["detection_rate"]
    d_pur = bs["purified"]["detection_rate"]
    d_cln = bs["clean"]["detection_rate"]
    r_adv, r_pur, r_cln = (
        bs["adv"]["mean_rmse"], bs["purified"]["mean_rmse"], bs["clean"]["mean_rmse"])
    return {
        # Primary (robust to Kitsune's absolute scale): continuous anomaly score.
        "mean_rmse_adv": r_adv,
        "mean_rmse_purified": r_pur,
        "mean_rmse_clean": r_cln,
        "rmse_restoration_ratio": _restoration_ratio(r_adv, r_pur, r_cln),
        # Secondary (threshold-dependent; see the feeding-regime caveat in the README).
        "detection_rate_adv": d_adv,
        "detection_rate_purified": d_pur,
        "detection_rate_clean": d_cln,
        "restoration_gain": _sub(d_pur, d_adv),   # detectability purification recovers
        "residual_gap": _sub(d_cln, d_pur),       # shortfall vs the clean oracle ceiling
        "threshold": threshold,
        "model_source": source,
        "n_flows": bs["purified"]["n_flows"],
    }


def _finalize_kitsune_overall(acc: dict) -> dict:
    def rate(br):
        n = acc[br]["n_flows"]
        return (acc[br]["n_detected"] / n) if n else None

    def mean(br):
        n = acc[br]["n_flows"]
        return (acc[br]["sum_rmse"] / n) if n else None

    d_adv, d_pur, d_cln = rate("adv"), rate("purified"), rate("clean")
    r_adv, r_pur, r_cln = mean("adv"), mean("purified"), mean("clean")
    return {
        "mean_rmse_adv": r_adv,
        "mean_rmse_purified": r_pur,
        "mean_rmse_clean": r_cln,
        "rmse_restoration_ratio": _restoration_ratio(r_adv, r_pur, r_cln),
        "detection_rate_adv": d_adv,
        "detection_rate_purified": d_pur,
        "detection_rate_clean": d_cln,
        "restoration_gain": _sub(d_pur, d_adv),
        "residual_gap": _sub(d_cln, d_pur),
        "n_flows": acc["purified"]["n_flows"],
    }


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


def _plot_kitsune_restoration(metrics: dict, out_path: Path) -> None:
    """Detection restoration through Kitsune: detection rate + mean RMSE per source."""
    kit = metrics.get("kitsune", {})
    per = kit.get("per_source", {})
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

    _tri(ax1, "detection_rate", "Kitsune detection rate (RMSE > φ)",
         "Detection restoration (independent NIDS)", ylim=(0, 1))
    ax1.legend(fontsize=7)
    _tri(ax2, "mean_rmse", "mean per-flow Kitsune RMSE",
         "Anomaly score (adv → purified → clean)")
    fig.suptitle("Purification efficacy through Kitsune (KitNET) — "
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
