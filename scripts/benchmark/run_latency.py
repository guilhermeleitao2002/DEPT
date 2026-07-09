"""DEPT — inference latency / throughput benchmark (RQ3: latency and overhead).

Measures wall-clock inference cost of every DEPT stage on the held-out
(seed=42) test split, as the system *actually* computes it: whole-flow
``[B, T, F]`` forward passes (there is no incremental / KV-cache path). For
each stage we report

  * batched throughput  — flows/s and packets/s at the model's configured
    batch size (the deployment-realistic serving figure);
  * per-flow latency    — ms per flow at batch size 1 (worst case), plus an
    amortised per-packet figure (per-flow ms / mean flow length).

Neural stages (TCN, meta-learner, reconstructor) run on GPU (device=auto);
the Random Forest is CPU (scikit-learn) inherently. The meta-learner is
reported for its own transformer forward only — its true detect cost is the
composite RF + TCN + meta, given in the ``pipeline`` block.

Bracketing the neural pipeline, two non-inference stages are also timed. On
the ingress side, the CPU ``pcap → nPrint`` preprocessing every deployment
must run *before* detection (Scapy parse + nPrint bit encoding + 5-tuple
grouping + ragged/RF tensor build) — reported per-packet/per-flow amortised,
since it is a bulk operation and, being Scapy-bound, is the practical
end-to-end bottleneck. On the egress side, the scapy traffic-space rebuild is
timed on one small source (tshark compliance excluded — that is a
compliance-time step, not inference).

Timing is done over a preloaded on-device pool so we measure compute, not disk
I/O or numpy assembly (the ingress stage likewise runs on a tmpfs scratch to
keep NAS I/O out of the figure). Parameters are hardcoded module constants
(repo convention); model dims / batch sizes / device are read from each model's
saved config. Output: ``data/models/latency/latency_metrics.json`` (overwritten
each run). A representative sample is timed by default; raise ``POOL_FLOWS`` /
``RF_CKPT_SAMPLE`` / ``PREPROCESS_MAX_PACKETS`` for a full-split run.

Usage:
    python3 scripts/benchmark/run_latency.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.models.common.latency import summarize, time_calls, time_sweep  # noqa: E402
from src.models.tcn.model import TCN  # noqa: E402
from src.models.tcn.evaluate import _load_dataset_index  # noqa: E402
from src.models.tcn.dataset import (  # noqa: E402
    FlowSequenceDataset, collate_padded as tcn_collate,
)
from src.models.metalearner.model import MetaLearner  # noqa: E402
from src.models.metalearner.dataset import (  # noqa: E402
    MetaLearnerDataset, load_token_cache, collate_padded as meta_collate,
)
from src.models.reconstructor.model import Reconstructor  # noqa: E402
from src.models.reconstructor.dataset import (  # noqa: E402
    build_pair_index, PairedFlowDataset, collate_pairs, FlowMemmapCache,
)
from src.models.reconstructor import realize as rz  # noqa: E402
from src.models.reconstructor.evaluate import _load_ipat_stats  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

# ── Benchmark parameters (hardcoded per repo convention) ─────────────────────
SEED = 42
WARMUP_SWEEPS = 1         # discarded full-pool sweeps before throughput timing
WARMUP_PER_FLOW = 2       # discarded forwards before each batch-1 timing
THROUGHPUT_REPEATS = 5    # full-pool sweeps for throughput
POOL_FLOWS = 2048         # flows preloaded on-device for timing
PERFLOW_SAMPLES = 512     # flows timed individually at batch size 1
PERFLOW_ITERS = 8         # timed forwards per single flow (median taken)
RF_CKPT_SAMPLE = 200_000  # RF checkpoint rows timed for throughput
REALIZE_SOURCE = "portscan_adv_both"  # small PCAP; scapy-rebuild sample
REALIZE_FLOWS = 1000
_IPAT_COL = 1088          # nPrint bits 0..1087, IPAT at 1088, Direction at 1089
PREPROCESS_SOURCE = "portscan_adv_both"   # small PCAP; pcap→nPrint ingress sample
PREPROCESS_MAX_PACKETS = 100_000          # packets timed for the ingress pipeline
PREPROCESS_REPEATS = 3                     # full pcap→nPrint passes (median taken)

OUTPUT_DIR = REPO_ROOT / "data" / "models" / "latency"


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sample_idx(all_idx: np.ndarray, n: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(all_idx))
    return np.asarray(all_idx)[perm[: min(n, len(all_idx))]]


# ─────────────────────────────────────────────────────────────────────────────
# Random Forest (CPU)
# ─────────────────────────────────────────────────────────────────────────────
def bench_rf() -> dict:
    import joblib

    rf_dir = REPO_ROOT / "data" / "models" / "rf"
    clf = joblib.load(rf_dir / "rf.joblib")
    data = np.load(rf_dir / "dataset.npz", allow_pickle=False)
    test_idx = np.load(rf_dir / "split.npz")["test_idx"]

    # Mean checkpoints per flow over the full test split (for per-flow cost).
    sources = data["sources"][test_idx].astype(str)
    flow_ids = data["flow_ids"][test_idx].astype(str)
    scoped = np.char.add(np.char.add(sources, "|"), flow_ids)
    n_flows = int(len(np.unique(scoped)))
    ckpts_per_flow = float(len(test_idx) / max(n_flows, 1))

    sample = _sample_idx(test_idx, RF_CKPT_SAMPLE)
    X = data["X"][sample].astype(np.float32, copy=False)

    clf.predict_proba(X[: min(4096, len(X))])  # warmup
    sweeps = []
    for _ in range(THROUGHPUT_REPEATS):
        t0 = time.perf_counter()
        clf.predict_proba(X)
        sweeps.append((time.perf_counter() - t0) * 1e3)
    sweep_ms = float(np.median(sweeps))
    n = int(len(X))
    per_ckpt_ms = sweep_ms / n
    return {
        "device": "cpu",
        "n_estimators": int(getattr(clf, "n_estimators", 0)),
        "n_checkpoints_timed": n,
        "ckpt_per_s": n / (sweep_ms / 1e3),
        "per_checkpoint_ms": per_ckpt_ms,
        "mean_checkpoints_per_flow": ckpts_per_flow,
        "per_flow_ms_est": per_ckpt_ms * ckpts_per_flow,
        "note": "CPU sklearn; per-flow = per-checkpoint x mean checkpoints/flow",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Generic on-device pool timing for a [B,T,F] forward model
# ─────────────────────────────────────────────────────────────────────────────
def _bench_seq_model(loader, device, forward) -> dict:
    """Preload a pool of batches on ``device`` and time throughput + per-flow.

    ``forward(seq)`` runs one batch forward (already-on-device tensors).
    Each loader batch is ``(seq, lengths)`` after adaptation by the caller.
    """
    pool: list[tuple[torch.Tensor, torch.Tensor]] = []
    total_flows = 0
    total_pkts = 0
    for seq, lengths in loader:
        seq = seq.to(device, non_blocking=True)
        pool.append((seq, lengths))
        total_flows += int(seq.shape[0])
        total_pkts += int(lengths.sum().item())
        if total_flows >= POOL_FLOWS:
            break

    def sweep() -> None:
        for seq, _ in pool:
            forward(seq)

    sweeps = time_sweep(sweep, device, warmup=WARMUP_SWEEPS, repeats=THROUGHPUT_REPEATS)
    sweep_ms = float(np.median(sweeps))

    # Per-flow latency at batch size 1.
    perflow_ms: list[float] = []
    perflow_len: list[int] = []
    for seq, lengths in pool:
        for b in range(int(seq.shape[0])):
            Lb = int(lengths[b].item())
            one = seq[b:b + 1, :Lb, :].contiguous()
            s = summarize(time_calls(
                lambda x=one: forward(x), device, warmup=WARMUP_PER_FLOW, iters=PERFLOW_ITERS,
            ))
            perflow_ms.append(float(s["median"]))
            perflow_len.append(Lb)
            if len(perflow_ms) >= PERFLOW_SAMPLES:
                break
        if len(perflow_ms) >= PERFLOW_SAMPLES:
            break

    mean_len = float(np.mean(perflow_len)) if perflow_len else 0.0
    per_flow = summarize(perflow_ms)
    out = {
        "device": str(device),
        "batched": {
            "batch_size": int(pool[0][0].shape[0]) if pool else 0,
            "n_flows": total_flows,
            "n_packets": total_pkts,
            "sweep_ms_median": sweep_ms,
            "flows_per_s": total_flows / (sweep_ms / 1e3),
            "pkts_per_s": total_pkts / (sweep_ms / 1e3),
            "ms_per_flow_amortized": sweep_ms / max(total_flows, 1),
        },
        "per_flow_bs1_ms": per_flow,
        "mean_flow_len": mean_len,
        "per_packet_ms_amortized": (per_flow["median"] / mean_len) if mean_len else None,
    }
    del pool
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out, per_flow["median"]


# ─────────────────────────────────────────────────────────────────────────────
# TCN
# ─────────────────────────────────────────────────────────────────────────────
def bench_tcn(device) -> tuple[dict, float]:
    tcn_dir = REPO_ROOT / "data" / "models" / "tcn"
    cfg = json.loads((tcn_dir / "tcn_config.json").read_text())
    m, t = cfg["model"], cfg["training"]
    model = TCN(
        num_features=int(m.get("num_features", 1090)),
        channels=tuple(m.get("channels", (128, 128, 128))),
        kernel_size=int(m.get("kernel_size", 3)),
        dropout=float(m.get("dropout", 0.15)),
        num_classes=int(m.get("num_classes", 2)),
    ).to(device)
    model.load_state_dict(torch.load(tcn_dir / "tcn.pt", map_location=device))
    model.eval()

    idx = _load_dataset_index(tcn_dir / "dataset_index.npz")
    pool_idx = _sample_idx(idx["used_test_idx"], POOL_FLOWS + 256)
    ds = FlowSequenceDataset(
        sources=idx["sources"], local_idx=idx["local_idx"], y=idx["y"],
        sample_indices=pool_idx, source_folders=idx["source_folders"],
        max_length=t.get("max_length", None),
    )
    loader = DataLoader(ds, batch_size=int(t.get("batch_size", 32)), shuffle=False,
                        num_workers=0, collate_fn=tcn_collate)
    use_amp = device.type == "cuda"

    def forward(seq):
        with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
            return model(seq)

    return _bench_seq_model(((s, l) for s, _y, l, _m in loader), device, forward)


# ─────────────────────────────────────────────────────────────────────────────
# Meta-learner (transformer forward only)
# ─────────────────────────────────────────────────────────────────────────────
def bench_meta(device) -> tuple[dict, float]:
    meta_dir = REPO_ROOT / "data" / "models" / "metalearner"
    cfg = json.loads((meta_dir / "metalearner_config.json").read_text())
    m, t = cfg["model"], cfg["training"]
    model = MetaLearner(
        d_model=int(m.get("d_model", 64)), n_layers=int(m.get("n_layers", 2)),
        n_heads=int(m.get("n_heads", 4)),
        dim_feedforward=int(m.get("dim_feedforward", 128)),
        dropout=float(m.get("dropout", 0.1)),
        num_classes=int(m.get("num_classes", 2)),
        max_seq_len=int(m.get("max_seq_len", 200)),
    ).to(device)
    model.load_state_dict(torch.load(meta_dir / "metalearner.pt", map_location=device))
    model.eval()

    cache_dir = meta_dir / "cache" / "test"
    if not (cache_dir / "tokens.npy").exists():
        return {"skipped": f"no token cache at {cache_dir}"}, None
    cache = load_token_cache(cache_dir)
    sample = _sample_idx(np.arange(len(cache)), POOL_FLOWS + 256)
    ds = MetaLearnerDataset(cache, sample_indices=sample)
    loader = DataLoader(ds, batch_size=int(t.get("batch_size", 128)), shuffle=False,
                        num_workers=0, collate_fn=meta_collate)

    # Preload tokens + masks on device; time model(tokens, key_padding_mask).
    pool = []
    total_flows = 0
    total_pkts = 0
    for tokens, _labels, lengths, mask in loader:
        tokens = tokens.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        pool.append((tokens, mask, lengths))
        total_flows += int(tokens.shape[0])
        total_pkts += int(lengths.sum().item())
        if total_flows >= POOL_FLOWS:
            break

    def forward(tokens, mask):
        return model(tokens, key_padding_mask=mask)

    def sweep():
        for tokens, mask, _l in pool:
            forward(tokens, mask)

    sweep_ms = float(np.median(time_sweep(sweep, device, warmup=WARMUP_SWEEPS, repeats=THROUGHPUT_REPEATS)))

    perflow_ms, perflow_len = [], []
    for tokens, mask, lengths in pool:
        for b in range(int(tokens.shape[0])):
            Lb = int(lengths[b].item())
            tk = tokens[b:b + 1, :Lb, :].contiguous()
            mk = mask[b:b + 1, :Lb].contiguous()
            s = summarize(time_calls(lambda a=tk, c=mk: forward(a, c),
                                     device, warmup=WARMUP_PER_FLOW, iters=PERFLOW_ITERS))
            perflow_ms.append(float(s["median"]))
            perflow_len.append(Lb)
            if len(perflow_ms) >= PERFLOW_SAMPLES:
                break
        if len(perflow_ms) >= PERFLOW_SAMPLES:
            break

    mean_len = float(np.mean(perflow_len)) if perflow_len else 0.0
    per_flow = summarize(perflow_ms)
    out = {
        "device": str(device),
        "kind": "transformer_forward_only",
        "batched": {
            "batch_size": int(pool[0][0].shape[0]) if pool else 0,
            "n_flows": total_flows, "n_packets": total_pkts,
            "sweep_ms_median": sweep_ms,
            "flows_per_s": total_flows / (sweep_ms / 1e3),
            "pkts_per_s": total_pkts / (sweep_ms / 1e3),
            "ms_per_flow_amortized": sweep_ms / max(total_flows, 1),
        },
        "per_flow_bs1_ms": per_flow,
        "mean_flow_len": mean_len,
        "per_packet_ms_amortized": (per_flow["median"] / mean_len) if mean_len else None,
        "note": "transformer forward only; excludes upstream RF+TCN and token build (see pipeline block)",
    }
    del pool
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out, per_flow["median"]


# ─────────────────────────────────────────────────────────────────────────────
# Reconstructor
# ─────────────────────────────────────────────────────────────────────────────
def bench_reconstructor(device) -> tuple[dict, float]:
    rec_dir = REPO_ROOT / "data" / "models" / "reconstructor"
    cfg = json.loads((rec_dir / "reconstructor_config.json").read_text())
    m = cfg["model"]
    t = cfg.get("training", {})
    model = Reconstructor(
        num_features=int(m.get("num_features", 1090)), d_model=int(m.get("d_model", 128)),
        n_layers=int(m.get("n_layers", 3)), n_heads=int(m.get("n_heads", 4)),
        dim_feedforward=int(m.get("dim_feedforward", 256)),
        dropout=float(m.get("dropout", 0.1)), max_seq_len=int(m.get("max_seq_len", 200)),
    ).to(device)
    model.load_state_dict(torch.load(rec_dir / "reconstructor.pt", map_location=device))
    model.eval()

    import yaml
    ycfg = yaml.safe_load((REPO_ROOT / "config" / "reconstructor.yaml").read_text())
    pairs = build_pair_index(
        tensors_dir=REPO_ROOT / ycfg["tensors_dir"],
        adversarial_dir=REPO_ROOT / ycfg["adversarial_dir"],
        adversarial_exclude=list(ycfg.get("adversarial_exclude", [])),
        min_len=int(ycfg.get("min_len", 2)),
    )
    test_idx = np.load(rec_dir / "split.npz")["test_idx"]
    pool_idx = _sample_idx(test_idx, POOL_FLOWS + 256)
    ds = PairedFlowDataset(pairs, sample_indices=pool_idx,
                           max_length=t.get("max_length", None))
    loader = DataLoader(ds, batch_size=int(t.get("batch_size", 32)), shuffle=False,
                        num_workers=0, collate_fn=collate_pairs)

    def forward(seq):
        # eval uses key_padding_mask = ~mask (True at padding); build from lengths.
        B, T, _ = seq.shape
        kpm = torch.ones(B, T, dtype=torch.bool, device=seq.device)
        # single-flow (bs1) and batched both pass a full-valid seq of its own T,
        # so no padded positions inside the timed tensor -> mask all-False.
        kpm[:] = False
        return model(seq, key_padding_mask=kpm)

    return _bench_seq_model(((adv, l) for adv, _c, l, _m, _r in loader),
                            device, forward)


# ─────────────────────────────────────────────────────────────────────────────
# PCAP → nPrint preprocessing (ingress: input processing before the models)
# ─────────────────────────────────────────────────────────────────────────────
def bench_preprocess() -> dict:
    """Time the raw-PCAP → nPrint tensor conversion the models depend on.

    Deployed DEPT cannot feed the RF/TCN/meta stack directly: every incoming
    flow must first be turned into the ragged per-flow nPrint tensors, exactly
    as ``run_packet_to_tensor.run()`` does, in three CPU-bound stages —

        1. extract_packet_features — sequential Scapy PCAP parse + vectorised
                                     nPrint bit encoding + identity masking;
        2. group_by_flow           — canonical 5-tuple grouping with per-flow
                                     IPAT + Direction;
        3. build_tensors           — ragged [P, 1090] assembly, IPAT z-score,
                                     and the multi-scale RF checkpoints.

    Timed on the first ``PREPROCESS_MAX_PACKETS`` packets of one real PCAP
    (device-independent — this stage never touches the GPU). The temporary
    slice and tensors live on tmpfs so the figure reflects compute, not NAS
    I/O. Reported as amortised per-packet / per-flow cost and ingress
    throughput — the practical pipeline bottleneck, since Scapy parsing is far
    slower than any neural forward.
    """
    import contextlib
    import shutil
    import subprocess
    import tempfile

    try:
        import yaml
        from src.feature_extraction.pcap_parser import extract_packet_features
        from src.feature_extraction.flow_grouper import group_by_flow
        from src.feature_extraction.tensor_builder import build_tensors

        cfg = yaml.safe_load((REPO_ROOT / "config" / "pipeline.yaml").read_text())
        mask_prefixes = cfg.get("mask_prefixes", [])
        std_cols = cfg["standardize_columns"]
        rf_base = cfg["rf_base_features"]
        rf_agg = cfg["rf_agg_functions"]
        stride = int(cfg.get("rf_checkpoint_stride", 10))
        first = int(cfg.get("rf_first_checkpoint", 2))

        base, mode = PREPROCESS_SOURCE.rsplit("_adv_", 1)
        src_pcap = (REPO_ROOT / "data" / "adversarial" / base
                    / f"{base}_adversarial_{mode}.pcap")
        if not src_pcap.exists():
            return {"skipped": f"pcap not found: {src_pcap}"}
        shared_mean_std = (REPO_ROOT / "data" / "tensors"
                           / cfg.get("mean_std_filename", "mean_std.json"))

        tmp_root = Path(tempfile.mkdtemp(prefix="dept_preprocess_"))
        try:
            # Cap to a representative slice so the (Scapy-bound) parse stays
            # bounded. editcap does a fast header-level copy; fall back to a
            # Scapy read/write slice when it is unavailable.
            slice_pcap = tmp_root / "slice.pcap"
            editcap = shutil.which("editcap")
            if editcap is not None:
                subprocess.run(
                    [editcap, "-r", str(src_pcap), str(slice_pcap),
                     f"1-{PREPROCESS_MAX_PACKETS}"],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            else:
                from scapy.all import PcapReader, PcapWriter
                with PcapReader(str(src_pcap)) as rd, PcapWriter(str(slice_pcap)) as wr:
                    for i, pkt in enumerate(rd):
                        if i >= PREPROCESS_MAX_PACKETS:
                            break
                        wr.write(pkt)

            def _one_pass(out_dir: Path) -> tuple:
                # Silence the engines' progress bars / prints during timing;
                # exceptions still propagate (only stream writes are redirected).
                with open(os.devnull, "w") as dn, \
                        contextlib.redirect_stdout(dn), \
                        contextlib.redirect_stderr(dn):
                    t0 = time.perf_counter()
                    nprint_bits, meta_df = extract_packet_features(
                        str(slice_pcap), mask_prefixes=mask_prefixes,
                    )
                    t1 = time.perf_counter()
                    meta_df, flow_indices, sort_order = group_by_flow(meta_df)
                    nprint_bits = nprint_bits[sort_order]
                    t2 = time.perf_counter()
                    res = build_tensors(
                        nprint_bits=nprint_bits, meta_df=meta_df,
                        flow_indices=flow_indices, output_dir=str(out_dir),
                        rf_checkpoint_stride=stride, rf_first_checkpoint=first,
                        rf_base_features=rf_base, rf_agg_functions=rf_agg,
                        standardize_columns=std_cols,
                        mean_std_path=str(shared_mean_std),
                        outputs=("rf", "tcn"),
                    )
                    t3 = time.perf_counter()
                return ((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3,
                        int(res["num_packets"]), int(res["num_flows"]))

            parse_ms: list[float] = []
            group_ms: list[float] = []
            build_ms: list[float] = []
            n_packets = n_flows = 0
            for r in range(max(1, PREPROCESS_REPEATS)):
                out_dir = tmp_root / f"out_{r}"
                p, g, b, n_packets, n_flows = _one_pass(out_dir)
                parse_ms.append(p)
                group_ms.append(g)
                build_ms.append(b)
                shutil.rmtree(out_dir, ignore_errors=True)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

        parse, group, build = summarize(parse_ms), summarize(group_ms), summarize(build_ms)
        total_ms = parse["median"] + group["median"] + build["median"]
        return {
            "device": "cpu",
            "source": PREPROCESS_SOURCE,
            "note": "raw pcap → nPrint ragged tensors (Scapy parse + nPrint "
                    "encode + 5-tuple group + ragged/RF tensor build); tmpfs "
                    "scratch so NAS I/O is excluded; masking + RF cadence per "
                    "config/pipeline.yaml",
            "n_packets": n_packets,
            "n_flows": n_flows,
            "workers": os.cpu_count(),
            "stage_ms_median": {
                "parse_encode": parse["median"],
                "group_by_flow": group["median"],
                "build_tensors": build["median"],
            },
            "stage_detail_ms": {
                "parse_encode": parse,
                "group_by_flow": group,
                "build_tensors": build,
            },
            "total_ms_median": total_ms,
            "ms_per_packet_amortized": total_ms / max(n_packets, 1),
            "ms_per_flow_amortized": total_ms / max(n_flows, 1),
            "pkts_per_s": (n_packets / (total_ms / 1e3)) if total_ms else None,
            "flows_per_s": (n_flows / (total_ms / 1e3)) if total_ms else None,
        }
    except Exception as e:  # best-effort — never fail the whole benchmark
        return {"skipped": f"{type(e).__name__}: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# Scapy traffic-space rebuild (no tshark)
# ─────────────────────────────────────────────────────────────────────────────
def bench_realize() -> dict:
    try:
        import yaml
        ycfg = yaml.safe_load((REPO_ROOT / "config" / "reconstructor.yaml").read_text())
        pairs = build_pair_index(
            tensors_dir=REPO_ROOT / ycfg["tensors_dir"],
            adversarial_dir=REPO_ROOT / ycfg["adversarial_dir"],
            adversarial_exclude=list(ycfg.get("adversarial_exclude", [])),
            min_len=int(ycfg.get("min_len", 2)),
        )
        gname = REALIZE_SOURCE
        if gname not in pairs.adv_folders:
            return {"skipped": f"source {gname} not in pair index"}
        base, mode = gname.rsplit("_adv_", 1)
        pcap_dir = REPO_ROOT / ycfg.get("realize", {}).get("adversarial_pcap_dir", "data/adversarial")
        adv_pcap = pcap_dir / base / f"{base}_adversarial_{mode}.pcap"
        if not adv_pcap.exists():
            return {"skipped": f"pcap not found: {adv_pcap}"}

        src_names = np.array([f"{b}_adv_{mm}" for b, mm in zip(pairs.base, pairs.mode)],
                             dtype=object)
        rows = np.where(src_names == gname)[0][:REALIZE_FLOWS]
        cache = FlowMemmapCache(pairs.adv_folders)
        flow_recon: dict[str, tuple] = {}
        for r in rows:
            r = int(r)
            seq = cache.slice(str(pairs.adv_source[r]), int(pairs.adv_local[r]))
            seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
            tl = rz.decode_total_len_batch(seq).astype(np.int64)
            ipat = seq[:, _IPAT_COL].astype(np.float32)
            flow_recon[str(pairs.flow_id[r])] = (tl, ipat)
        mean, std = _load_ipat_stats(REPO_ROOT / ycfg["tensors_dir"])

        t0 = time.perf_counter()
        packets, rstats = rz.realize_source(adv_pcap, flow_recon, mean, std)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        realized_flows = int(rstats.get("realized_flows", 0))
        n_pkts = int(rstats.get("packets", 0))
        return {
            "source": gname,
            "note": "scapy rebuild incl. single-pass PCAP scan; tshark compliance excluded",
            "pcap_packets_read": int(rstats.get("pcap_packets_read", 0)),
            "target_flows": int(rstats.get("target_flows", 0)),
            "realized_flows": realized_flows,
            "realized_packets": n_pkts,
            "elapsed_ms": elapsed_ms,
            "ms_per_flow": (elapsed_ms / realized_flows) if realized_flows else None,
            "realized_pkts_per_s": (n_pkts / (elapsed_ms / 1e3)) if n_pkts else None,
        }
    except Exception as e:  # best-effort — never fail the whole benchmark
        return {"skipped": f"{type(e).__name__}: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    device = _device()
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    print(f"[latency] device={device} gpu={gpu_name}")

    per_model: dict = {}
    print("[latency] PCAP → nPrint preprocessing (CPU ingress)…")
    preprocessing = bench_preprocess()
    print("[latency] Random Forest (CPU)…")
    per_model["rf"] = bench_rf()
    print("[latency] TCN…")
    per_model["tcn"], tcn_pf = bench_tcn(device)
    print("[latency] Meta-learner (transformer forward)…")
    per_model["metalearner"], meta_pf = bench_meta(device)
    print("[latency] Reconstructor…")
    per_model["reconstructor"], rec_pf = bench_reconstructor(device)
    print("[latency] Scapy rebuild (no tshark)…")
    realization = bench_realize()

    # Composite per-flow pipeline latency (batch-1 medians summed).
    rf_pf = per_model["rf"].get("per_flow_ms_est")
    pipeline: dict = {"note": "batch-1 per-flow medians summed; token materialization "
                              "(CPU numpy) between TCN and meta is not micro-timed"}
    if None not in (rf_pf, tcn_pf, meta_pf):
        detect = rf_pf + tcn_pf + meta_pf
        pipeline["detect_per_flow_ms"] = detect
        pipeline["detect_stage_breakdown_ms"] = {
            "rf": rf_pf, "tcn": tcn_pf, "meta_transformer": meta_pf,
        }
        if rec_pf is not None:
            pipeline["detect_plus_purify_per_flow_ms"] = detect + rec_pf
            pipeline["purify_stage_ms"] = rec_pf

    # Ingress preprocessing is upstream of every neural stage. It is a bulk
    # (amortised) per-flow cost, not a batch-1 latency, so we surface it beside
    # the detect medians rather than summing it in — and flag that end-to-end
    # throughput is ingress-bound (Scapy parsing ≪ any neural forward).
    pre_pf = preprocessing.get("ms_per_flow_amortized") if isinstance(preprocessing, dict) else None
    if pre_pf is not None:
        pipeline["preprocess_per_flow_ms_amortized"] = pre_pf
        pipeline["preprocess_note"] = (
            "pcap→nPrint ingress, amortised per flow over a bulk PCAP parse "
            "(not a batch-1 figure); dominates end-to-end cost and bounds "
            "throughput"
        )

    # Budget context: raw sustained throughput per neural stage (bottleneck-bound).
    budget = {
        "reference_note": "sustained pkts/s per stage at configured batch size; the "
                          "detect path throughput is bounded by its slowest stage — "
                          "and, end-to-end, by the CPU pcap→nPrint ingress, which is "
                          "far slower than any neural forward. A line-rate (e.g. "
                          "100 Gbps) comparison is packet-size dependent and is framed "
                          "in the dissertation, not asserted here.",
        "preprocess_pkts_per_s": preprocessing.get("pkts_per_s") if isinstance(preprocessing, dict) else None,
        "tcn_pkts_per_s": per_model["tcn"].get("batched", {}).get("pkts_per_s"),
        "meta_pkts_per_s": per_model["metalearner"].get("batched", {}).get("pkts_per_s"),
        "reconstructor_pkts_per_s": per_model["reconstructor"].get("batched", {}).get("pkts_per_s"),
        "rf_ckpt_per_s": per_model["rf"].get("ckpt_per_s"),
    }

    result = {
        "device": str(device),
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "config": {
            "seed": SEED, "warmup_sweeps": WARMUP_SWEEPS, "warmup_per_flow": WARMUP_PER_FLOW,
            "throughput_repeats": THROUGHPUT_REPEATS,
            "pool_flows": POOL_FLOWS, "per_flow_samples": PERFLOW_SAMPLES,
            "per_flow_iters": PERFLOW_ITERS, "rf_ckpt_sample": RF_CKPT_SAMPLE,
            "sample_note": "representative test-split sample; raise constants for full split",
        },
        "preprocessing": preprocessing,
        "per_model": per_model,
        "realization": realization,
        "pipeline": pipeline,
        "budget_context": budget,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "latency_metrics.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"[latency] wrote {out_path}")

    # Terminal summary.
    def _g(d, *ks):
        for k in ks:
            d = d.get(k, {}) if isinstance(d, dict) else {}
        return d
    print("\n=== summary (per-flow median latency, batch=1) ===")
    if pre_pf is not None:
        print(f"  ingress (amort): {pre_pf:.3f} ms/flow  "
              f"({preprocessing.get('pkts_per_s', 0):,.0f} pkt/s, "
              f"pcap→nPrint, not batch-1)")
    elif isinstance(preprocessing, dict) and "skipped" in preprocessing:
        print(f"  ingress        : skipped ({preprocessing['skipped']})")
    print(f"  RF (est/flow)  : {rf_pf:.3f} ms" if rf_pf else "  RF: n/a")
    print(f"  TCN            : {tcn_pf:.3f} ms" if tcn_pf else "  TCN: n/a")
    print(f"  Meta (tx only) : {meta_pf:.3f} ms" if meta_pf else "  Meta: n/a")
    print(f"  Reconstructor  : {rec_pf:.3f} ms" if rec_pf else "  Recon: n/a")
    if "detect_per_flow_ms" in pipeline:
        print(f"  detect total   : {pipeline['detect_per_flow_ms']:.3f} ms/flow")
    if "detect_plus_purify_per_flow_ms" in pipeline:
        print(f"  detect+purify  : {pipeline['detect_plus_purify_per_flow_ms']:.3f} ms/flow")
    print("=== throughput (batched, pkts/s) ===")
    print(f"  TCN   : {_g(per_model,'tcn','batched').get('pkts_per_s'):,.0f}")
    print(f"  Meta  : {_g(per_model,'metalearner','batched').get('pkts_per_s'):,.0f}")
    print(f"  Recon : {_g(per_model,'reconstructor','batched').get('pkts_per_s'):,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
