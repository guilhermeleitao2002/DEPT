# Deployment Baseline — protected NIDS, before vs after DEPT

> Part of [DEPT](../../README.md) — **phase 8** of `run_pipeline.sh` (global mode only). Measures DEPT as what it actually is — a defense deployed **in front of a NIDS** — rather than through DEPT's own internal detector.

```bash
python3 scripts/baseline/run_baseline.py                  # all attacks
python3 scripts/baseline/run_baseline.py --attacks mirai,ddos
```

## What it answers

Every other stage reports its own internal metrics. None of them answer the deployment question: **does putting DEPT in front of a real NIDS restore the detection TANTGen destroyed?** This scores three realized full-volume traffic streams — **clean** (original attack, oracle), **adversarial** (TANTGen's perturbation), and **defended** (the full DEPT gated pipeline) — through the NIDS each attack was actually built to evade, never DEPT's own detector. A fourth `purified_all` stream (purify everything, i.e. a perfect gate) is scored as an upper bound.

The gated pipeline is the runtime loop the architecture diagram promises: the reconstructor purifies each flow, the meta-learner emits a per-flow `P(adversarial)`, and the defended stream takes the *purified* overlay for flagged flows and the *adversarial* overlay unchanged for the rest. All three streams are realized onto the **same** packets (only size and timing differ), so the comparison is controlled.

NIDS routing uses the shared `src/models/nids/` registry: **Kitsune** for `mirai` / `ssdp_flood`, TANTGen's **ML-NIDS** flow classifiers for the eight CIC attacks. Headline metrics are `asr_reduction` and `restoration_ratio_detection` (rate-based, so it aggregates both NIDS families safely), plus `gate.flag_rate`.

Because the defended stream runs over **all** of an attack's flows for full-volume realism — Kitsune's signal is partly volumetric — it is the *deployment* view; it is therefore also reported restricted to held-out test flows as the generalization-valid view. The `clean` and `adversarial` streams involve no DEPT model, so their numbers are unconditionally valid.

## Results

30 sources, 2,704,816 flow-scores, detector threshold 0.5, 33.6 h, nothing skipped.

| Regime | detection rate | ASR |
|---|---|---|
| clean (oracle) | 0.9811 | 0.0189 |
| adversarial (TANTGen) | 0.7887 | 0.2113 |
| **DEPT-defended** | **0.8836** | **0.1164** |
| purified_all (upper bound) | 0.8831 | 0.1169 |

**ASR reduction 9.49 points; detection-restoration ratio 0.493.** On held-out test flows only the defended stream reads 0.9061 (372,321 flows) — the generalization-valid view is not worse than the deployment view. By family: ml_nids **+9.85 pts** (2.62 M flows), kitsune **−1.67 pts** (82.8 K).

- **The gain is real but narrow.** It is carried by six of thirty variants — `ddos_size` +67.4, `portscan_both` +67.0, `ddos_both` +59.7, `portscan_size` +46.8, `ssdp_flood_both` +20.0, `syn_ddos_size` +10.9 — exactly the cases where TANTGen's structural edits actually destroyed detection. Everything else moves by under a point, because TANTGen never evaded those NIDS in the first place.
- **Purifying traffic that was never evading hurts**: `dns_ddos` iat/both −22.7/−22.5, `mirai_size` −23.9, `portscan_iat` −18.2, `ddos_iat` −12.2. So the −9.5-point average is a mixture of large wins, large losses, and a majority of no-ops.
- **The gate is over-eager.** Flag rates are ~100% on every flood source and near 0% on several `_size` variants (`ssdp_ddos_size` 0.0%, `ntp_ddos_size` 0.2%) — the same reflection-flood blind spot the base learners show. `defended ≈ purified_all` for exactly this reason. A better gate — one predicting *whether purification will help*, not merely whether the flow is adversarial — is the obvious next step, left as future work.
- **The Kitsune family is net negative**, driven entirely by `mirai_size`, where TANTGen's size edit had accidentally *raised* Kitsune's detection to 0.242 and DEPT removed that self-sabotage. Kitsune's clean-flow detection on mirai is itself only 0.003, so these numbers rest on a detector that barely fires on the traffic at all.

## Prerequisites

Trained `data/models/{rf,tcn,metalearner,reconstructor}`; the sibling TANTGen repo (`nids.tantgen_root`) with its built `AfterImage*.so` and `<Attack>_*.joblib` models; `cicflowmeter` on `PATH` for the ML-NIDS attacks and `Cython`/`pyximport` for the Kitsune ones; tensorized sources under `data/tensors/` and adversarial PCAPs under `data/adversarial/`.

Writes `baseline_metrics.json`, `baseline_detection.png` and a `detector_cache/` to `data/models/baseline/`. Config: [`config/baseline.yaml`](../../config/baseline.yaml).
