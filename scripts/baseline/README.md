# Deployment Baseline: protected NIDS before and after DEPT

> Part of [DEPT](../../README.md), **phase 8** of `run_pipeline.sh` (global mode only). Evaluates DEPT placed in front of a NIDS.

```bash
python3 scripts/baseline/run_baseline.py                  # all attacks
python3 scripts/baseline/run_baseline.py --attacks mirai,ddos
```

## Method

The question is whether putting DEPT in front of a real NIDS restores the detection that TANTGen removed. Three full traffic streams are scored by the NIDS each attack was built to evade (not by DEPT's own detector):

- **clean**: the original attack;
- **adversarial**: TANTGen's version;
- **defended**: the gated DEPT pipeline.

A fourth stream, `purified_all` (every flow purified, i.e. a perfect detector), is an upper bound.

In the defended stream the reconstructor purifies each flow, the meta-learner gives a per-flow `P(adversarial)`, and the purified version is used only for the flows it flags; the others keep their adversarial version. All streams are realized on the same packets, so only size and timing differ.

The NIDS are taken from `src/models/nids/`: Kitsune for `mirai` / `ssdp_flood` and TANTGen's ML-NIDS for the eight CIC attacks. The main metrics are `asr_reduction` and `restoration_ratio_detection`, plus `gate.flag_rate`.

The defended stream covers all flows of each attack, including flows the models were trained on, because Kitsune's features depend on traffic volume. It is therefore also reported on the held-out test flows only. The `clean` and `adversarial` streams do not involve any DEPT model.

## Results

30 sources, 2,704,816 flow scores, detector threshold 0.5, 33.6 h.

| Stream | Detection rate | ASR |
|---|---|---|
| clean | 0.9811 | 0.0189 |
| adversarial (TANTGen) | 0.7887 | 0.2113 |
| **DEPT-defended** | **0.8836** | **0.1164** |
| purified_all (upper bound) | 0.8831 | 0.1169 |

**ASR reduction 9.49 points; detection-restoration ratio 0.493.** On the held-out test flows only, the defended stream reaches 0.9061 (372,321 flows). Per NIDS: ML-NIDS **+9.85 points** (2.62 M flows), Kitsune **−1.67 points** (82.8 K).

- **The gain comes from six of the thirty variants**: `ddos_size` +67.4, `portscan_both` +67.0, `ddos_both` +59.7, `portscan_size` +46.8, `ssdp_flood_both` +20.0, `syn_ddos_size` +10.9. These are the cases where TANTGen did evade the NIDS. The other variants change by less than a point, because TANTGen never evaded those NIDS.
- **Purifying traffic that was not evading lowers detection**: `dns_ddos` iat/both −22.7/−22.5, `mirai_size` −23.9, `portscan_iat` −18.2, `ddos_iat` −12.2.
- **The gate flags too much.** Flag rates are ~100% on every flood and close to 0% on some `_size` variants (`ssdp_ddos_size` 0.0%, `ntp_ddos_size` 0.2%), the same weakness the base learners have. This is why `defended ≈ purified_all`. A gate that predicts whether purification will help, and not only whether the flow is adversarial, is left as future work.
- **Kitsune is net negative** because of `mirai_size`: there TANTGen's size change had raised Kitsune's detection to 0.242 and DEPT removed it. Kitsune detects only 0.003 of the clean mirai flows, so these numbers are not very informative.

## Prerequisites

Trained `data/models/{rf,tcn,metalearner,reconstructor}`; the TANTGen repository (`nids.tantgen_root`) with its compiled `AfterImage*.so`, its `<Attack>_*.joblib` models and its vendored cicflowmeter (run with an interpreter that has `scapy` and `requests`); `Cython` for Kitsune; the tensors in `data/tensors/` and the adversarial PCAPs in `data/adversarial/`.

Writes `baseline_metrics.json`, `baseline_detection.png` and `detector_cache/` to `data/models/baseline/`. Config: [`config/baseline.yaml`](../../config/baseline.yaml).
