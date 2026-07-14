# Deployment Baseline (protected-NIDS "before vs after DEPT")

> Part of [DEPT](../../README.md). The end-to-end evaluation that measures DEPT as what it
> actually is — a defense deployed **in front of a NIDS** — rather than only through DEPT's
> own internal detector.

## What it answers

DEPT's other stages each report their own internal metrics (RF/TCN/meta-learner detection,
reconstructor MSE). None of them answer the deployment question an examiner asks first:
**does putting DEPT in front of a real NIDS actually restore the detection that TANTGen's
perturbations destroyed?** This baseline answers it directly, for every attack, across three
realized full-volume traffic streams scored through the real NIDS the attack was built to
evade.

| Regime | Traffic | Expectation |
|---|---|---|
| **clean** | original-attack size/timing (oracle) | NIDS detects it → high detection, low ASR |
| **adversarial** | TANTGen's perturbation | evasion → detection collapses, ASR spikes |
| **defended** | the full DEPT gated pipeline | detection climbs back toward the clean ceiling |

A fourth **purified_all** stream (every flow purified — a *perfect* detector) is scored as an
upper bound.

## The independent NIDS (each attack's true target)

Traffic is scored through the NIDS each attack was actually built to evade — never DEPT's own
detector (that would be circular). Routing is the shared `nids` registry
([`src/models/nids/`](../../src/models/nids/), the same block the reconstructor eval uses):

- **Kitsune** (KitNET + AfterImage) for the Kitsune-native attacks `mirai`, `ssdp_flood`
  (evaded by the `iat` vector), reusing TANTGen's calibrated pickles.
- **ML-NIDS** (TANTGen's CICFlowMeter flow classifiers) for the 8 CIC-IDS/CIC-DDoS attacks
  (evaded by the `size` vector), reusing TANTGen's `<Attack>_RF.joblib` + scaler.

## The DEPT gated pipeline (the "defended" regime)

This is the runtime loop DEPT's architecture diagram promises. For each flow
([`src/models/baseline/pipeline.py`](../../src/models/baseline/pipeline.py)):

1. the **reconstructor** purifies the flow and its size/timing overlay is decoded;
2. the **detector** (the meta-learner — DEPT's real classifier, fusing RF + TCN) emits a
   per-flow `P(adversarial)` ([`detector_infer.py`](../../src/models/baseline/detector_infer.py));
3. the **defended stream** takes the *purified* overlay for flows flagged adversarial
   (`P(adv) ≥ detector_threshold`) and the *adversarial* overlay unchanged for the rest —
   "non-adversarial passes through immediately, adversarial passes only after purification".

The three streams are realized onto the **same** adversarial packets (only size + timing
differ; [`realize.realize_triplet`](../../src/models/reconstructor/realize.py)) so the
comparison is controlled, then scored through the NIDS.

## Metrics

Per attack (× mode) and overall, from per-flow NIDS verdicts:

- **detection rate** and **ASR** (`1 − detection_rate`) for each regime;
- **`asr_reduction`** = `ASR(adversarial) − ASR(defended)` — the headline: how much of
  TANTGen's attack success DEPT removes;
- **`restoration_ratio_detection`** = `(dr_defended − dr_adv) / (dr_clean − dr_adv)` — the
  fraction of the adv→clean detection gap the pipeline recovers (rate-based, so it aggregates
  Kitsune and ML-NIDS safely; the cross-family overall reads this);
- **`gate.flag_rate`** — the fraction of flows the detector flagged (how much traffic the
  pipeline actually purified).

### Deployment vs generalization

The `defended` / `purified_all` streams run DEPT models over **all** of an attack's flows for
full-volume realism — including flows some models saw in training, so those numbers are the
*deployment* view. The DEPT-defended stream is therefore **also** reported restricted to the
held-out **test** flows (`per_source.<src>.test_subset`, `score_test_subset: true`) — the
generalization-valid view. Regimes `clean` and `adversarial` involve no DEPT model, so their
full-stream numbers are unconditionally valid.

> **Volumetric-feeding note.** Kitsune's anomaly signal is partly volumetric (packet-rate /
> burst structure). The baseline feeds full realized streams (not a flow subset) precisely so
> its detection rates are realistic and comparable to TANTGen's own ASR numbers — the opposite
> trade-off from the reconstructor eval, which feeds only the held-out test pairs and therefore
> leads with the scale-robust score-restoration ratio instead.

## Usage

```
python3 scripts/baseline/run_baseline.py                 # all attacks in nids.attacks
python3 scripts/baseline/run_baseline.py --attacks mirai,ddos
```

Reads [`config/baseline.yaml`](../../config/baseline.yaml). Writes to
`data/models/baseline/`:

| File | Description |
|---|---|
| `baseline_metrics.json` | Per-source + overall + by-NIDS-family detection / ASR / restoration |
| `baseline_detection.png` | Detection rate (clean/adv/defended) and ASR before-vs-after per source |
| `detector_cache/` | Cached meta-learner token stream for the gate (keyed by RF/TCN mtime) |

## Prerequisites

- Trained `data/models/{rf,tcn,metalearner,reconstructor}` (run the pipeline first).
- The sibling TANTGen repo (`nids.tantgen_root`) with its built `AfterImage*.so` and the
  `src/NIDS/models/<Attack>_*.joblib` files.
- `cicflowmeter` on `PATH` for the ML-NIDS attacks (`pip install cicflowmeter`); the Kitsune
  attacks additionally need `Cython`/`pyximport` in the venv (for the AfterImage `.so`).
- Tensorized adversarial + clean sources under `data/tensors/` and the TANTGen adversarial
  PCAPs under `data/adversarial/`.
