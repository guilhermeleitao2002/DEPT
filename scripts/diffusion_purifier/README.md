# Diffusion-Purifier Comparison — DEPT vs a published purifier

> Part of [DEPT](../../README.md) — **phase 9** of `run_pipeline.sh` (global mode only). Compares DEPT's purifier with a published one, the diffusion-based purifier of Merzouk et al. (DBSec 2025), inside the phase-8 harness.

```bash
python3 scripts/diffusion_purifier/run_diffusion_purifier.py                    # all stages, resumable
python3 scripts/diffusion_purifier/run_diffusion_purifier.py --stage evaluate --attacks ddos,portscan
python3 scripts/diffusion_purifier/run_diffusion_purifier.py --stage sweep --force
```

## What it answers

The [deployment baseline](../baseline/README.md) compares DEPT with doing nothing. This phase compares it with a **published defense**: the diffusion purifier of Merzouk et al., "Diffusion-based Adversarial Purification for Intrusion Detection" (DBSec 2025, [code](https://github.com/mamerzouk/adversarial-purification)). Of the NIDS purifiers in the related work, it is the only one with public code. It takes the place of DEPT's reconstructor inside the phase-8 harness: the same flows, the same realization back into packets and the same independent NIDS (Kitsune for `mirai` / `ssdp_flood`, TANTGen's ML-NIDS for the eight CIC attacks). Only the purifier changes.

## How it works

| Stage | What it does | Writes |
|---|---|---|
| `overlays` | Reads the distinct clean flows of the **reconstructor's training pairs** as (size, IPAT) arrays and fits the IPAT scale | `cache/`, `overlay_norm.json` |
| `train` | Trains the diffusion model on those clean flows only, as in the paper | `diffusion.pt`, `diffusion_config.json`, `training_curves.{json,png}` |
| `sweep` | Chooses the diffusion step `t*` on validation flows | `t_sweep.{json,png}` |
| `evaluate` | Realizes and scores every stream through the NIDS, on held-out test flows | `per_source/*.json`, `diffusion_purifier_metrics.json`, `diffusion_purifier_comparison.png` |

Each stage skips work that is already on disk (`--force` redoes it). `evaluate` writes one file per source as it goes, so an interrupted run continues where it stopped.

**Representation.** Phase 8 realizes a purifier's output from one per-packet overlay: IPv4 total length and IPAT. The diffusion model works directly in that space, not on the 1090-dim nPrint tensor. A flow of up to 200 packets becomes one vector `[size_1..size_200 | ipat_1..ipat_200]`, which is the flat-vector input the original model expects. Size is divided by 1500. IPAT is heavy-tailed, so it is log-scaled (`log1p(IPAT) / log1p(q99.9)`) instead of the original min-max.

**Model.** The model is copied from the authors' `diffusion.py` into [`model.py`](../../src/models/diffusion_purifier/model.py), with attribution. It keeps the paper's settings: a linear DDPM schedule (T = 1000, β 1e-4 → 0.02), a 10-layer × 960-unit MLP noise predictor with a sinusoidal step embedding added to every layer, the noise-prediction MSE loss, AdamW at 1e-4, and purification by noising to step `t*` and denoising back. A unit test checks that the copied forward pass and reverse process give the same output as the authors' file. Three changes are needed for variable-length flows, and each is marked `# DEPT:` in the code:
- the MLP also receives the packet directions and the padding mask, which realization cannot change;
- the loss covers real packets only, and padding is held at its known value during denoising;
- training uses mini-batches (4096) instead of the original's full-batch epochs.

**Choosing `t*`.** The paper defines `t*` as the step that maximizes NIDS accuracy on adversarial inputs, and picks it on its test data. Here every step in the grid {2, 5, 10, 20, 30, 45, 60, 90, 150} is realized onto the same **validation** packets of the ML-NIDS sources and scored through the real NIDS. `t*` is the step with the highest pooled detection. A distance-to-clean curve (with DEPT and "no purification" as reference lines) is saved next to it as a diagnostic.

**Evaluation.** Every stream is realized onto the same adversarial packets with one shared keep-mask (a packet is dropped from every stream if any overlay would corrupt an SSDP/HTTP text payload). This is phase 8's `realize_triplet` extended to more branches, one branch in memory at a time; a test checks it reproduces `realize_triplet` byte for byte.

| Stream | Meaning |
|---|---|
| `clean` / `adversarial` | Ceiling and floor. Re-scored here, because the shared keep-mask can differ slightly from phase 8's. |
| `dept_all` | DEPT's reconstructor applied to every flow |
| `diffusion_all` | The diffusion purifier applied to every flow (as published: no detector in front) |
| `dept_gated` / `diffusion_gated` | DEPT's meta-learner gate + each purifier (phase 8's defended stream, and the same with the purifier swapped) |

Metrics are the phase-8 ones (detection rate, ASR, ASR reduction, restoration ratio), on four subsets of flows:
- **`test`** (headline) — the reconstructor's test pairs, which neither purifier trained on. ML-NIDS streams contain only these flows. Kitsune streams are the full source, because Kitsune's features are volumetric, and the metric is restricted afterwards.
- **`strict`** — test pairs whose original flow appears in no training pair under any mode. The three modes are split independently, so most test pairs have a sibling (same original flow, another mode) in training, whose clean side both purifiers learned from. If the ranking holds here, the comparison does not depend on that overlap.
- **`gated`** — test pairs that are also in the meta-learner's (TCN) test split, so the gate and the purifiers are all held out. This is the only subset where the gated streams are scored.
- **`full`** — Kitsune only: every flow of the full stream (phase 8's deployment view).

It also measures **latency** per flow on the same GPU (DEPT's single forward pass against `t*` diffusion steps) and runs **tshark** on a packet sample of both purifiers for `ddos_adv_both` and `portscan_adv_both`.

## Fairness

- Same training flows: the diffusion model learns from the clean sides of the reconstructor's own training pairs. It sees no adversarial examples, as in the paper.
- Same test flows, same realization, same NIDS, same keep-mask for every stream.
- `t*` is tuned on validation flows, never on test flows.
- TANTGen does not adapt to either defense.
- Two advantages go to the baseline. It sees the whole flow at once, while DEPT's reconstructor is causal. And it borrows DEPT's realization step, while the original method stops at the feature vector.

## Results

*Not run yet.*

## Prerequisites

- Phase 6 artifacts under `data/models/reconstructor/`: `reconstructor.pt`, `reconstructor_config.json`, `split.npz`, `split_summary.json`. The run stops with an error if the rebuilt pair table does not match `split_summary.json`.
- Everything phase 8 needs to score traffic: tensorized sources, the TANTGen adversarial PCAPs, the sibling TANTGen repo with its NIDS models, the vendored cicflowmeter, and Cython for Kitsune.
- For the gated streams: the trained RF (`dataset.npz`), TCN (`dataset_index.npz`) and meta-learner. Phase 8's `data/models/baseline/detector_cache/` is reused when the source set is the same. If the gate cannot be built, the gated streams are skipped with a warning.
- `tshark` (optional) for the compliance sample.

Config: [`config/diffusion_purifier.yaml`](../../config/diffusion_purifier.yaml). The split and excludes are read from `reconstructor.yaml`, and the NIDS routing, gate and model paths from `baseline.yaml`, so this phase always matches phases 6 and 8.

**Expected runtime** (estimates): overlays ~1 h, training 1–3 h on one GPU (150 k steps; the log shows an ETA), sweep 1–2 h, evaluation 8–12 h (about 14% of phase 8's flows, 4–6 streams each).
