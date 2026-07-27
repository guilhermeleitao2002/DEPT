# Reconstructor (Purification) Transformer

> Part of [DEPT](../../README.md). The purification stage — a small causal (decoder-only) Transformer that denoises a flow the ensemble flagged as adversarial, mapping it back onto the clean-traffic manifold.

## Role in the defense

When DEPT's ensemble flags a flow as adversarial, the reconstructor **purifies** it before it reaches the protected NIDS. TANTGen evades detection by perturbing two quantities — packet **timing** (IPAT) and packet **size** (IPv4 total length / payload padding). The reconstructor learns the inverse: given the adversarial flow, it reconstructs the clean flow, undoing those perturbations so the underlying attack is once again detectable. Success is measured — headline first — by **detection restoration through the independent NIDS** each attack was actually built to evade: **Kitsune** (KitNET) for the Kitsune-native attacks and TANTGen's **ML-NIDS** (CICFlowMeter classifiers) for the CIC attacks, where the purified flow's anomaly signal should climb back from the evaded floor toward the original attack (Evaluation §1b, shared with the deployment baseline via `src/models/nids/`). A **purification gain through DEPT's own TCN** is *also* reported, but only as an in-distribution sanity check — it is circular (the TCN trained on this data), which is exactly why the independent-NIDS metric is the headline.

It is **causal / decoder-only** by design (mirroring the meta-learner and the thesis's deployment story): position `t` is denoised from packets `≤ t` only, so the model is KV-cacheable for streaming purification.

## Input / target

Both the input and the reconstruction target are the same `[L, 1090]` per-packet tensor every DEPT model consumes:

- **1088 nPrint bits** — tri-state `{-1, 0, 1}` of all IPv4/TCP/UDP/ICMP header bits (identity fields zeroed).
- **IPAT** — Inter-Packet Arrival Time within the flow (z-score normalized).
- **Direction** — `1` = forward, `0` = reverse.

Training examples are **matched `(adversarial → clean)` flow pairs**. For each `<base>_adv_<mode>` adversarial source the clean base folder is resolved by stripping `_adv_<mode>`; flows are paired **by their canonical 5-tuple** (`flow_ids.npy`), never by row position (TANTGen can split/merge/drop flows). Matched pairs keep equal packet counts ~99.8% of the time; the rare divergent pairs are truncated to the shorter length, and pairs below `min_len` packets are dropped. Any adversarial source with no clean base folder is skipped automatically.

## Architecture

```
adv [B, T, 1090]
    │
    ├─ Linear 1090 → d_model (128)
    ├─ Sinusoidal positional encoding
    │
    ├─ 3 × pre-LN Transformer encoder layer (4 heads, causal mask)
    │
    └─ Linear d_model → 1090   →  residual Δ
                                    │
        reconstruction = adv + Δ  ──┘
```

A **residual head** makes the identity map the easy default (the head is zero-initialised, so an untrained model reconstructs its input verbatim). Capacity is therefore spent only on the localized timing/length correction; the ~1050 untouched header bits are preserved for free.

## Training

- **Loss**: per-element **smooth-L1** between the reconstruction and the clean target, averaged within each flow (so long flows don't dominate) and masked over padding.
- **Attack-agnostic**: uniform weight over all 1090 dims — the model is supervised only to "reconstruct the clean flow" and is **never told which dimensions the attack perturbs**. With `reduction=mean` every dim's gradient is scaled equally, and on a residual head the unperturbed dims have residual-target `Δ=0`, so the learning signal concentrates on the perturbed dims without any weighting.
- **Split**: flow-atomic, source-stratified, same `seed=42` as the base learners, so the held-out test pairs never leak into any model's training set.
- **Early stopping** on the validation reconstruction loss (lower is better).

## Evaluation

Four metric groups (`metrics.json`), reported overall and per `<base>_adv_<mode>`:

1. **Reconstruction quality** — MSE on the perturbed dims, *before* (adv vs clean) vs *after* (recon vs clean). The before→after drop quantifies the denoising.
2. **Purification gain (in-distribution sanity check)** — the reconstruction (rounded back to valid nPrint) is fed through the trained **TCN** alongside the adversarial and clean flows; the per-flow last-timestep `P(adversarial)` is reported for each. `gain = P(adv|adv) − P(adv|recon)`; the residual gap to `P(adv|clean)` shows how close to the clean floor the purifier lands. This judge is DEPT's *own* detector, trained on this data — a useful sanity check, but **circular**, which is why group 4 exists.
3. **Protocol-compliance rate** — *traffic-space* (best-effort): every reconstructed test flow is **realized** back onto the original adversarial PCAP packets (overlay the reconstructed length/timing onto the real packet; scapy recomputes checksums), written to `realized/<source>.pcap`, and validated with `tshark` (`_ws.malformed` / bad-checksum count). **Application-layer text packets are excluded:** a size edit on an SSDP/HTTP-style payload (e.g. `M-SEARCH * HTTP/1.1`) keeps IP/UDP lengths and checksums valid but corrupts the L7 grammar, so tshark throws `_ws.malformed` — a raw byte-overlay simply can't faithfully resize it. Such size-edited packets (detected by SSDP port 1900 or an HTTP start line in [`realize.py:_is_l7_text`](../../src/models/reconstructor/realize.py)) are dropped from the realized PCAP and reported separately as `n_unrealizable_l7` (per-source and overall), not scored as purifier failures; `tshark_rate` is therefore over realizable packets only. Mechanism confirmed by `tshark` on a flagged `realized/ssdp_flood_adv_size.pcap` (42,585 `udp:ssdp` `M-SEARCH … Malformed`, with IP/UDP lengths and checksums all consistent). With the exclusion, `tshark_rate` rises from 0.9473 to **0.9976** and **85,379** packets (almost all `ssdp_flood` both/size) are reported as `n_unrealizable_l7`.
4. **Independent-NIDS efficacy (Kitsune + ML-NIDS)** — the **headline** efficacy metric, and an *independent* judge that breaks group 2's circularity. Each attack is scored through the real NIDS **it was built to evade**: **Kitsune** (KitNET autoencoder + AfterImage) for the Kitsune-native attacks (`mirai`, `ssdp_flood`), and TANTGen's **ML-NIDS** (per-attack CICFlowMeter flow classifiers) for the CIC-IDS/CIC-DDoS attacks. Both are trained only on their own data and never saw DEPT's models; both are reused from the sibling TANTGen repo strictly as evaluators (nothing about them enters the purifier's training — the defense stays attack-agnostic). Routing is via the shared `nids` registry in [`src/models/nids/`](../../src/models/nids/) (also used by the deployment baseline).

   *Controlled 3-way realization.* For each source we realize **three** packet sets from the **same** adversarial test-flow packets, overlaying only size+timing (via [`realize.py:realize_triplet`](../../src/models/reconstructor/realize.py)): `adv` (adversarial size/timing), `purified` (the reconstruction), and `clean` (the paired clean-target tensor = original-attack oracle). Same flows, identity, payloads and L7/`miss_len` exclusions (one shared keep-mask) → the *only* variable Kitsune sees is size+timing, exactly what the purifier touches and what AfterImage's size/jitter features respond to. Driving `clean` from the tensor (not a separate capture) also sidesteps any stale on-disk original PCAP.

   *Semantics = anomaly-signal **restoration** (opposite direction to group 2).* The NIDS flags a flow when its anomaly signal is high (Kitsune `RMSE > φ`, or the ML-NIDS classifier predicting the malicious class). The attack *evades* by **suppressing** that signal; purification recovers the original attack's fingerprint, so it should **rise** from the (evaded) adversarial floor back toward the clean-original oracle. The **primary** per-source metric is `score_restoration_ratio = (mean_score_purified − mean_score_adv) / (mean_score_clean − mean_score_adv)` — the fraction of the adv→clean gap the purifier recovers (`0` = none, `1` = full), where `score` is Kitsune RMSE or ML-NIDS `P(malicious)`. Also reported per source: `mean_score_{adv,purified,clean}`, `detection_rate_{adv,purified,clean}`, and `restoration_gain`/`residual_gap` in detection-rate terms. Because Kitsune RMSE and ML-NIDS probabilities live on different scales, the cross-family **overall** reads `detection_restoration_ratio` (rate-based, so it aggregates both NIDS safely), with per-family score-based ratios under `by_kind`. Each realized branch is scored as one stream; Kitsune per-flow aggregation is `max` RMSE by default (`nids.kitsune.detection`).

   > **Feeding-regime caveat (why the score ratio, not the detection rate, is the headline here).** Kitsune is a *volumetric* detector — much of its anomaly signal comes from packet-rate / burst structure across a full stream. The controlled 3-way design feeds only the re-timed **test-flow subset**, which does not reproduce that volume, so the absolute RMSE is compressed and *every* branch — **including the clean original attack** — can sit below TANTGen's φ, driving Kitsune detection rates toward 0. The RMSE *ordering* (adv < purified < clean) and the `score_restoration_ratio` stay robust and correctly ranked across modes (e.g. `size`, the strongest structural signal, shows the largest clean-vs-adv gap; `iat` the smallest). Treat the detection rate here as a threshold-dependent lower bound and the score restoration ratio as the efficacy number. (The ML-NIDS is flow-level, not volumetric, so its detection rate on the subset is more meaningful; the **deployment baseline** — `scripts/baseline` — additionally scores full-volume streams for realistic detection-rate/ASR numbers.)

   *Scope & models.* Every attack is scored through the NIDS it was built to evade (`nids.attacks`): `mirai` / `ssdp_flood` → **Kitsune**, reusing TANTGen's `Mirai_model.pkl` / `SSDP_Flood_model.pkl` verbatim; the 8 CIC attacks → **ML-NIDS**, reusing TANTGen's `<Attack>_RF.joblib` + `<Attack>_scaler.joblib` (`nids.default_ml_model`, default `RF`). Kitsune needs the sibling repo's built `AfterImage.so` and `Cython`/`pyximport` in the venv (the prebuilt `.so` runs `pyximport.install()` on import); ML-NIDS needs `cicflowmeter` on `PATH` (`pip install cicflowmeter`). Best-effort per source: a source whose NIDS or its dependency is missing is skipped-with-reason (recorded in `metrics.json` under `nids.skipped_sources`) so the reconstruction + TCN groups still report. Configured under `nids:` in [`config/reconstructor.yaml`](../../config/reconstructor.yaml) — the same block the deployment baseline reads.

## Results

Trained on **2,566,489** matched `(adversarial → clean)` pairs (split 1,796,542 / 384,984 / 384,963, seed=42), 20 epochs, 10.0 h on one GPU, 677,698 parameters, best epoch 19, best val reconstruction loss 3.7 × 10⁻⁴. Evaluated on the 385 K-pair held-out test split.

### Overall

| Group | Metric | Value |
|---|---|---|
| Reconstruction | perturbed-dim MSE before → after | 0.918 → **0.054** (17.0× reduction) |
| Purification (through TCN, sanity check) | P(adv): adversarial / reconstructed / clean | 0.675 / **0.428** / 0.421 |
| Purification | gain / residual gap to clean floor | **0.247** / **0.007** |
| Protocol compliance | tshark on **realizable** packets | **0.9722** (67,106 bad / 2,412,357) |
| Protocol compliance | excluded `n_unrealizable_l7` (size-edited SSDP/HTTP text) | **450,758** (96% of them `ssdp_flood` both/size; 19,677 in `ddos`) |
| **NIDS restoration (headline)** | detection rate: adv → purified (clean oracle) | **0.781 → 0.906** (0.980) |
| NIDS restoration | detection-restoration ratio / residual gap | **0.630** / 0.074 (n = 370,700 flows) |
| NIDS restoration, by family | Kitsune: 0.019 → 0.792 (clean 0.793) — ratio **0.999** | ML-NIDS: 0.808 → 0.910 (clean 0.987) — ratio 0.571 |

### Per-source (`<base> <mode>`)

`mse` = MSE on perturbed dims (before/after); `P_adv`/`P_rec` = TCN P(adversarial) on adv/recon; `gain` = P_adv − P_rec; `tsh` = tshark compliance on **realizable** packets; `l7` = packets excluded as `n_unrealizable_l7` (size-edited SSDP/HTTP text — see §3; `·` = none); `NIDS det adv→pur (clean)` = independent-NIDS detection rate on the realized triplet. `—` = not computable (no perturbed cells recorded, or all packets unrealizable).

| variant | mse_before | mse_after | P_adv | P_rec | gain | tsh | l7 | NIDS det adv→pur (clean) |
|---|---|---|---|---|---|---|---|---|
| brute_force both | 0.000 | 0.001 | 0.347 | 0.348 | −0.002 | 1.000 | · | 0.996 → 0.981 (0.996) |
| brute_force iat | 0.000 | 0.001 | 0.342 | 0.344 | −0.002 | 1.000 | · | 0.997 → 0.990 (0.997) |
| brute_force size | 1.000 | 0.188 | 0.343 | 0.345 | −0.002 | 1.000 | · | 0.993 → 0.963 (0.994) |
| ddos both | 0.928 | 0.124 | 1.000 | 0.069 | **+0.931** | 0.839 | 11,318 | **0.080 → 0.865** (0.821) |
| ddos iat | 0.302 | 0.003 | 0.431 | 0.074 | +0.357 | 0.850 | · | 0.981 → 0.957 (0.946) |
| ddos size | 1.000 | 0.117 | 0.999 | 0.062 | **+0.938** | 0.838 | 8,359 | **0.005 → 0.817** (0.828) |
| dns_ddos both | 0.000 | 0.000 | 0.777 | 0.785 | −0.008 | 0.998 | · | 0.986 → 0.762 (0.996) |
| dns_ddos iat | 0.000 | 0.000 | 0.778 | 0.785 | −0.006 | 0.999 | 2 | 0.993 → 0.756 (0.998) |
| dns_ddos size | 1.000 | 0.302 | 0.774 | 0.776 | −0.002 | 0.994 | 4 | 0.986 → 0.765 (0.986) |
| ldap_ddos both | 0.176 | 0.002 | 0.541 | 0.541 | +0.000 | 1.000 | · | 1.000 → 1.000 (1.000) |
| ldap_ddos iat | 0.081 | 0.001 | 0.541 | 0.541 | +0.000 | 1.000 | · | 0.999 → 0.999 (1.000) |
| ldap_ddos size | — | — | 0.541 | 0.541 | +0.000 | 1.000 | · | 1.000 → 1.000 (1.000) |
| mirai both | 1.890 | 1.628 | 0.272 | 0.251 | +0.020 | 0.991 | · | 0.003 → 0.001 (0.007) |
| mirai iat | 2.462 | 2.142 | 0.269 | 0.246 | +0.023 | 0.991 | · | 0.001 → 0.002 (0.006) |
| mirai size | 1.569 | 0.973 | 0.520 | 0.249 | +0.271 | 0.990 | · | 0.257 → 0.005 (0.008) |
| ntp_ddos both | 0.000 | 0.000 | 0.396 | 0.397 | −0.001 | 0.996 | · | 0.983 → 0.985 (0.998) |
| ntp_ddos iat | 0.000 | 0.000 | 0.395 | 0.395 | −0.001 | 0.996 | · | 0.983 → 0.986 (0.999) |
| ntp_ddos size | — | — | 0.396 | 0.397 | −0.001 | 0.996 | · | 0.998 → 0.988 (0.998) |
| portscan both | 0.972 | 0.105 | 0.988 | 0.153 | **+0.835** | 0.999 | 58 | **0.011 → 0.679** (0.999) |
| portscan iat | 0.000 | 0.007 | 0.154 | 0.154 | +0.000 | 0.999 | · | 0.994 → 0.811 (0.999) |
| portscan size | 1.000 | 0.104 | 0.989 | 0.154 | **+0.835** | 0.999 | 70 | **0.013 → 0.479** (0.999) |
| ssdp_ddos both | 0.000 | 0.000 | 0.618 | 0.617 | +0.001 | 0.999 | · | 0.998 → 0.998 (0.998) |
| ssdp_ddos iat | 0.000 | 0.000 | 0.619 | 0.618 | +0.001 | 0.999 | · | 0.999 → 0.998 (0.999) |
| ssdp_ddos size | — | — | 0.619 | 0.618 | +0.001 | 0.999 | · | 0.998 → 0.998 (0.998) |
| ssdp_flood both | 0.965 | 0.003 | 0.999 | 0.625 | +0.374 | 1.000 | 215,295 | 1.000 → 1.000 (1.000) |
| ssdp_flood iat | 0.110 | 0.081 | 0.999 | 0.464 | +0.535 | 1.000 | · | **0.001 → 1.000** (1.000) |
| ssdp_flood size | 1.000 | 0.003 | 0.999 | 0.450 | +0.549 | — | 215,652 | — (all packets unrealizable) |
| syn_ddos both | 0.973 | 0.129 | 1.000 | 0.077 | **+0.923** | 1.000 | · | 0.973 → 0.999 (0.998) |
| syn_ddos iat | 0.491 | 0.000 | 0.902 | 0.077 | +0.825 | 0.178 | · | 1.000 → 1.000 (0.997) |
| syn_ddos size | 1.000 | 0.144 | 1.000 | 0.068 | **+0.932** | 1.000 | · | **0.879 → 1.000** (1.000) |

### Observations

- **Restoration is strong exactly where the attack actually evaded.** The rows whose NIDS detection had collapsed — `ddos` both/size (0.08 / 0.005), `portscan` both/size (0.011 / 0.013), `ssdp_flood_iat` (0.001), `syn_ddos_size` (0.879) — are pulled back to 0.48–1.00, in several cases to or above the clean-oracle ceiling. Aggregated: Kitsune-family detection is restored 0.019 → 0.792 against a 0.793 clean ceiling (**99.9%** restoration), ML-NIDS 0.808 → 0.910 vs 0.987 (57.1% — an average diluted by the many rows where there was nothing to restore).
- **Purifying non-evaded traffic can hurt.** Where TANTGen never fooled the NIDS (`dns_ddos` all modes 0.99 adv detection, `portscan_iat` 0.994, `brute_force` ≈ 0.99), pushing the flow through the denoiser *lowers* detection by 2–24 points (`dns_ddos` both/iat → 0.76). The purifier must be **gated** — this is quantified end-to-end in the deployment baseline (`scripts/baseline`).
- **Near-no-op where there is nothing to undo.** For the reflection floods' both/iat variants, `mse_before ≈ 0` — TANTGen's edits barely move the tensor dims — so TCN gain ≈ 0 and NIDS detection stays put. The model correctly leaves these flows almost unchanged.
- **Mirai is the weakest reconstruction** (MSE 1.9–2.5 → 1.6–2.1, only ~1.3×) *and* an uninformative NIDS row: Kitsune's per-flow detection of even the clean mirai slice is ~0.8%, so there is nothing measurable to restore. The one signal — `mirai_size` adv detection 0.257 → 0.005 — means TANTGen's size edit accidentally made flows *more* visible and the purifier removed that self-sabotage (good denoising, bad for defense).
- **Protocol validity holds once SSDP/HTTP text is set aside.** Realized-packet `tshark` compliance is **97.2%** (67,106 bad / 2.41 M realizable). The **450,758** size-edited text-protocol packets (96% `ssdp_flood` both/size — `ssdp_flood_size` is *entirely* unrealizable — plus 19,677 in `ddos`) are excluded as `l7`, not failures. Two realization weak spots remain and are honest limitations: `ddos` (~84–85%: 22,623 of 140,128 realized packets flagged, checksum/length corner cases at scale) and `syn_ddos_iat` (**17.8%**, but low-support: only 927 of 10,748 pairs survive the equal-length pairing filter, and of the 2,088 realized packets 1,716 carry bad TCP checksums after the scapy re-write — a realization bug on this slice, not a purifier failure; verified `_ws.malformed` count is 0).
- **The TCN gain stays what it is: a circular sanity check.** Residual gap 0.007 through DEPT's own TCN says the manipulation signature is gone *for this detector*; the independent-NIDS group above is the metric that carries evidential weight.

## Output files (`data/models/reconstructor/`)

| File | Description |
|---|---|
| `reconstructor.pt` | Best PyTorch `state_dict` (min val reconstruction loss) |
| `reconstructor_config.json` | Resolved model + training config |
| `split.npz` | `train_idx` / `val_idx` / `test_idx` into the (deterministic) pair table that eval rebuilds from config |
| `split_summary.json` | Per-split / per-source pair counts |
| `training_info.json` | Best epoch, device, train time, pair stats, effective config |
| `training_curves.json` | Per-epoch train_loss / val_loss / lr |
| `metrics.json` | Reconstruction / **nids** (headline) / purification (TCN, secondary) / protocol-compliance, overall + per-source |
| `realized/<source>.pcap` | Realized purified packets (traffic-space, for tshark + replay) |
| `realized_nids/<source>_{adv,purified,clean}.pcap` | The 3 aligned realized packet sets scored through the independent NIDS |
| `reconstruction_error.png` | Per-source MSE on perturbed dims, before vs after |
| `purification_gain.png` | Per-source TCN `P(adv)` for adversarial / reconstructed / clean (secondary) |
| `nids_restoration.png` | Per-source independent-NIDS detection rate + mean anomaly score for adv / purified / clean |
| `training_curves.png` | Train / val reconstruction loss over epochs |

> Any Kitsune models calibrated on demand are cached under `data/models/kitsune/<base>_model.pkl`; the two Kitsune-native attacks reuse TANTGen's pickles directly, and the CIC attacks use TANTGen's ML-NIDS `.joblib` models (no calibration).
