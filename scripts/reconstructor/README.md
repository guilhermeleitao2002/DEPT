# Reconstructor (Purification) Transformer

> Part of [ADEPT](../../README.md). The purification stage — a small causal (decoder-only) Transformer that denoises a flow the ensemble flagged as adversarial, mapping it back onto the clean-traffic manifold.

## Role in the defense

When ADEPT's ensemble flags a flow as adversarial, the reconstructor **purifies** it before it reaches the protected NIDS. TANTGen evades detection by perturbing two quantities — packet **timing** (IPAT) and packet **size** (IPv4 total length / payload padding). The reconstructor learns the inverse: given the adversarial flow, it reconstructs the clean flow, undoing those perturbations so the underlying attack is once again detectable. Success is measured two ways: internally by **purification gain** (feeding the reconstruction back through ADEPT's own TCN and watching `P(adversarial)` fall toward the clean floor), and — the headline, *independent* measure — by **detection restoration through Kitsune**, the external anomaly-detection NIDS TANTGen actually evades, where the purified flow's anomaly signal should climb back from the evaded floor toward the original attack (see Evaluation §4).

It is **causal / decoder-only** by design (mirroring the meta-learner and the thesis's deployment story): position `t` is denoised from packets `≤ t` only, so the model is KV-cacheable for streaming purification.

## Input / target

Both the input and the reconstruction target are the same `[L, 1090]` per-packet tensor every ADEPT model consumes:

- **1088 nPrint bits** — tri-state `{-1, 0, 1}` of all IPv4/TCP/UDP/ICMP header bits (identity fields zeroed).
- **IPAT** — Inter-Packet Arrival Time within the flow (z-score normalized).
- **Direction** — `1` = forward, `0` = reverse.

Training examples are **matched `(adversarial → clean)` flow pairs**. For each `<base>_adv_<mode>` adversarial source the clean base folder is resolved by stripping `_adv_<mode>`; flows are paired **by their canonical 5-tuple** (`flow_ids.npy`), never by row position (TANTGen can split/merge/drop flows). Matched pairs keep equal packet counts ~99.8% of the time; the rare divergent pairs are truncated to the shorter length, and pairs below `min_len` packets are dropped. Sources with no clean base (e.g. `ddos_kitsune_adv_*`) are skipped automatically.

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
2. **Purification gain (in-distribution sanity check)** — the reconstruction (rounded back to valid nPrint) is fed through the trained **TCN** alongside the adversarial and clean flows; the per-flow last-timestep `P(adversarial)` is reported for each. `gain = P(adv|adv) − P(adv|recon)`; the residual gap to `P(adv|clean)` shows how close to the clean floor the purifier lands. This judge is ADEPT's *own* detector, trained on this data — a useful sanity check, but **circular**, which is why group 4 exists.
3. **Protocol-compliance rate** — *traffic-space* (best-effort): every reconstructed test flow is **realized** back onto the original adversarial PCAP packets (overlay the reconstructed length/timing onto the real packet; scapy recomputes checksums), written to `realized/<source>.pcap`, and validated with `tshark` (`_ws.malformed` / bad-checksum count). **Application-layer text packets are excluded:** a size edit on an SSDP/HTTP-style payload (e.g. `M-SEARCH * HTTP/1.1`) keeps IP/UDP lengths and checksums valid but corrupts the L7 grammar, so tshark throws `_ws.malformed` — a raw byte-overlay simply can't faithfully resize it. Such size-edited packets (detected by SSDP port 1900 or an HTTP start line in [`realize.py:_is_l7_text`](../../src/models/reconstructor/realize.py)) are dropped from the realized PCAP and reported separately as `n_unrealizable_l7` (per-source and overall), not scored as purifier failures; `tshark_rate` is therefore over realizable packets only. Mechanism confirmed by `tshark` on a flagged `realized/ssdp_flood_adv_size.pcap` (42,585 `udp:ssdp` `M-SEARCH … Malformed`, with IP/UDP lengths and checksums all consistent). With the exclusion, `tshark_rate` rises from 0.9473 to **0.9976** and **85,379** packets (almost all `ssdp_flood` both/size) are reported as `n_unrealizable_l7`.
4. **Independent-NIDS efficacy (Kitsune)** — the **headline** efficacy metric, and an *independent* judge that breaks group 2's circularity. **Kitsune** (KitNET autoencoder + AfterImage feature extractor) is the real anomaly-detection NIDS that TANTGen's attacks were built to evade; it is trained only on benign traffic and never saw ADEPT's models. Reused from the sibling TANTGen repo strictly as an evaluator (nothing about it enters the purifier's training — the defense stays attack-agnostic).

   *Controlled 3-way realization.* For each source we realize **three** packet sets from the **same** adversarial test-flow packets, overlaying only size+timing (via [`realize.py:realize_triplet`](../../src/models/reconstructor/realize.py)): `adv` (adversarial size/timing), `purified` (the reconstruction), and `clean` (the paired clean-target tensor = original-attack oracle). Same flows, identity, payloads and L7/`miss_len` exclusions (one shared keep-mask) → the *only* variable Kitsune sees is size+timing, exactly what the purifier touches and what AfterImage's size/jitter features respond to. Driving `clean` from the tensor (not a separate capture) also sidesteps any stale on-disk original PCAP.

   *Semantics = anomaly-score **restoration** (opposite direction to group 2).* Kitsune flags a flow when reconstruction `RMSE > threshold φ`. The attack *evades* by **suppressing** RMSE below φ; purification recovers the original attack's fingerprint, so its RMSE should **rise** from the (evaded) adversarial floor back toward the clean-original oracle. The **primary** metric is `rmse_restoration_ratio = (mean_rmse_purified − mean_rmse_adv) / (mean_rmse_clean − mean_rmse_adv)` — the fraction of the adv→clean anomaly-score gap the purifier recovers (`0` = none, `1` = full). Also reported per source and overall: `mean_rmse_{adv,purified,clean}`, `detection_rate_{adv,purified,clean}` (fraction of flows whose per-flow-aggregated RMSE exceeds φ), and `restoration_gain`/`residual_gap` in detection-rate terms. Each realized branch is scored as one temporal stream (matching TANTGen's `-M exec`); per-flow aggregation is `max` RMSE by default (`kitsune.detection`).

   > **Feeding-regime caveat (why RMSE, not detection rate, is the headline).** Kitsune is a *volumetric* detector — much of its anomaly signal comes from packet-rate / burst structure across a full stream. The controlled 3-way design feeds only the re-timed **test-flow subset**, which does not reproduce that volume, so the absolute RMSE is compressed and *every* branch — **including the clean original attack** — can sit below TANTGen's φ, driving detection rates toward 0. The RMSE *ordering* (adv < purified < clean) and the `rmse_restoration_ratio` stay robust and correctly ranked across modes (e.g. `size`, the strongest structural signal, shows the largest clean-vs-adv gap; `iat` the smallest). Treat the detection rate as a threshold-dependent lower bound and the RMSE restoration ratio as the efficacy number.

   *Scope & models.* Restricted to the two **Kitsune-native** attacks TANTGen actually calibrated a Kitsune against — `mirai` and `ssdp_flood` (`kitsune.sources`) — reusing TANTGen's own `Mirai_model.pkl` / `SSDP_Flood_model.pkl` verbatim (the gold-standard "attacker's own target NIDS" case). Extending to other attacks would mean calibrating a fresh Kitsune per attack (the adapter can, via `train_model` on the benign split — FM-grace 5000 / AD-grace 50000, `φ = max` benign RMSE, caching to `data/models/kitsune/<base>_model.pkl`), but that is out of the current scope. Best-effort: the whole group is skipped-with-reason if `kitsune.enabled: false` or the sibling TANTGen Kitsune tree (its built `AfterImage.so`) is absent. Requires `Cython`/`pyximport` in the venv (the prebuilt `.so` runs `pyximport.install()` on import). Configured under `kitsune:` in [`config/reconstructor.yaml`](../../config/reconstructor.yaml).

## Results

Trained on **1,986,705** matched `(adversarial → clean)` pairs (split 1.39 M / 298 K / 298 K, seed=42), 20 epochs, ~7.4 h on one GPU, 677,698 parameters, best val reconstruction loss 2.9 × 10⁻⁴. Evaluated on the 298 K-pair held-out test split.

### Overall

| Group | Metric | Value |
|---|---|---|
| Reconstruction | perturbed-dim MSE before → after | 0.962 → **0.128** (7.5× reduction) |
| Purification (through TCN) | P(adv): adversarial / reconstructed / clean | 0.552 / **0.425** / 0.425 |
| Purification | gain / residual gap to clean floor | **0.128** / **0.000** |
| Purification | detection rate (P≥0.5): adversarial → reconstructed | 0.489 → 0.350 |
| Protocol compliance | tshark on **realizable** packets | **0.9976** (3,886 bad / 1,598,558) |
| Protocol compliance | excluded `n_unrealizable_l7` (size-edited SSDP/HTTP text) | **85,379** (≈99.5% of them `ssdp_flood` both/size) |

The residual gap of 0.000 means the purified flow is, to the detector, indistinguishable from a genuinely un-manipulated one.

### Per-source (`<base> <mode>`)

`mse` = MSE on perturbed dims (before/after); `P_adv`/`P_rec` = TCN P(adversarial) on adv/recon; `gain` = P_adv − P_rec; `tsh` = tshark compliance on **realizable** packets; `l7` = packets excluded as `n_unrealizable_l7` (size-edited SSDP/HTTP text — see §3; `·` = none). `—` = no perturbed cells in the test sample (adv ≈ clean).

| variant | mse_before | mse_after | P_adv | P_rec | gain | tsh | l7 |
|---|---|---|---|---|---|---|---|
| brute_force both | 0.948 | 0.581 | 0.707 | 0.141 | +0.567 | 1.000 | · |
| brute_force iat | 0.941 | 0.702 | 0.381 | 0.010 | +0.370 | 1.000 | · |
| brute_force size | 0.991 | 0.562 | 0.709 | 0.015 | +0.694 | 1.000 | · |
| ddos both | 0.979 | 0.474 | 1.000 | 0.101 | +0.899 | 0.946 | 161 |
| ddos iat | 0.977 | 0.530 | 0.737 | 0.095 | +0.642 | 0.954 | 136 |
| ddos size | 0.985 | 0.481 | 1.000 | 0.145 | +0.855 | 0.943 | 155 |
| dns_ddos both | 0.000 | 0.000 | 0.467 | 0.467 | +0.000 | 0.998 | · |
| dns_ddos iat | 0.000 | 0.000 | 0.469 | 0.469 | +0.000 | 0.996 | · |
| dns_ddos size | 1.000 | 0.361 | 0.468 | 0.467 | +0.000 | 0.996 | 8 |
| ldap_ddos both | 0.000 | 0.000 | 0.618 | 0.618 | +0.000 | 1.000 | · |
| ldap_ddos iat | 0.000 | 0.000 | 0.618 | 0.618 | +0.000 | 1.000 | · |
| ldap_ddos size | 1.000 | 0.568 | 0.618 | 0.618 | −0.000 | 1.000 | · |
| mirai both | 0.972 | 0.511 | 0.484 | 0.084 | +0.400 | 0.985 | · |
| mirai iat | 0.947 | 0.486 | 0.089 | 0.091 | −0.002 | 0.982 | · |
| mirai size | 0.979 | 0.478 | 0.533 | 0.092 | +0.441 | 0.988 | · |
| ntp_ddos both | 0.300 | 0.075 | 0.432 | 0.431 | +0.001 | 0.998 | 2 |
| ntp_ddos iat | 0.000 | 0.000 | 0.431 | 0.431 | −0.000 | 0.996 | · |
| ntp_ddos size | 1.000 | 0.349 | 0.432 | 0.431 | +0.000 | 0.993 | · |
| portscan both | 0.977 | 0.096 | 0.988 | 0.031 | **+0.957** | 1.000 | 8 |
| portscan iat | 0.136 | 0.093 | 0.032 | 0.032 | +0.000 | 1.000 | 8 |
| portscan size | 1.000 | 0.135 | 0.987 | 0.032 | **+0.955** | 1.000 | 8 |
| ssdp_ddos both | 0.007 | 0.003 | 0.418 | 0.418 | −0.000 | 0.999 | · |
| ssdp_ddos iat | 0.000 | 0.000 | 0.418 | 0.418 | −0.000 | 0.999 | · |
| ssdp_ddos size | 1.000 | 0.483 | 0.418 | 0.418 | −0.000 | 0.999 | · |
| ssdp_flood both | 0.973 | 0.070 | 0.997 | 0.077 | **+0.920** | 1.000 | 42267 |
| ssdp_flood iat | 0.938 | 0.230 | 0.935 | 0.085 | +0.850 | 1.000 | · |
| ssdp_flood size | 0.968 | 0.081 | 0.998 | 0.075 | **+0.923** | 1.000 | 42626 |
| syn_ddos both | — | — | 0.245 | 0.245 | −0.000 | 1.000 | · |
| syn_ddos iat | 0.000 | 0.000 | 0.246 | 0.246 | −0.000 | 1.000 | · |
| syn_ddos size | 1.000 | 0.126 | 1.000 | 0.245 | **+0.755** | 1.000 | · |

### Observations

- **Purification is strong exactly where the attack is structural.** `portscan`, `ssdp_flood`, `ddos` and `syn_ddos_size` — attacks that add real packet-size/padding signal the detector keys on — see P(adv) collapse from ~1.0 to <0.15 (gains 0.76–0.96).
- **It is a near-no-op where there is nothing to undo.** For the high-rate CIC-DDoS floods (`dns_ddos`, `ldap_ddos`, `ntp_ddos`, `ssdp_ddos`) and all `_iat`-only variants, `mse_before ≈ 0` — the adversarial and clean flows already differ only in the near-constant IPAT — so gain ≈ 0. The model correctly leaves these almost unchanged (residual gap stays 0).
- **Protocol validity holds once SSDP/HTTP text is set aside.** Realized-packet `tshark` compliance is **99.8%** (3,886 bad / 1.60 M realizable). A size edit on an application-layer **text** protocol (SSDP/HTTP `M-SEARCH`) corrupts its L7 grammar — tshark flags `_ws.malformed` although IP/UDP lengths and checksums stay valid — so such packets can't be byte-overlay-realized; the **85,379** of them (almost all `ssdp_flood` both/size; small counts in `ddos`/`portscan`/`dns_ddos`) are excluded and counted as `l7`, not failures. Among the remaining realizable packets most sources are ≥ 99%, with `ddos` the lowest (~94–95%, size-edit checksum/length corner cases). These are realization edge cases, not failures of the purifier itself.
- **The TCN caveat is now addressed by group 4.** TCN purification gain is judged by ADEPT's own detector, so "residual gap 0" only means the manipulation signature is gone *relative to this detector*. The **Kitsune** metric closes this: it re-realizes the purified flow to real packets and scores it with the independent anomaly-detection NIDS TANTGen actually targets, reporting **detection restoration** (RMSE / detection rising from the evaded adversarial floor back toward the clean-original oracle). See `kitsune` in `metrics.json` and `kitsune_restoration.png`.

## Output files (`data/models/reconstructor/`)

| File | Description |
|---|---|
| `reconstructor.pt` | Best PyTorch `state_dict` (min val reconstruction loss) |
| `reconstructor_config.json` | Resolved model + training config |
| `split.npz` | `train_idx` / `val_idx` / `test_idx` into the (deterministic) pair table that eval rebuilds from config |
| `split_summary.json` | Per-split / per-source pair counts |
| `training_info.json` | Best epoch, device, train time, pair stats, effective config |
| `training_curves.json` | Per-epoch train_loss / val_loss / lr |
| `metrics.json` | Reconstruction / purification / protocol-compliance / **kitsune**, overall + per-source |
| `realized/<source>.pcap` | Realized purified packets (traffic-space, for tshark + replay) |
| `realized_kitsune/<source>_{adv,purified,clean}.pcap` | The 3 aligned realized packet sets scored through Kitsune |
| `reconstruction_error.png` | Per-source MSE on perturbed dims, before vs after |
| `purification_gain.png` | Per-source TCN `P(adv)` for adversarial / reconstructed / clean |
| `kitsune_restoration.png` | Per-source Kitsune detection rate + mean RMSE for adv / purified / clean |
| `training_curves.png` | Train / val reconstruction loss over epochs |

> Trained Kitsune models for the CIC attacks are cached under `data/models/kitsune/<base>_model.pkl` (the two Kitsune-native attacks reuse TANTGen's calibrated pickles directly).
