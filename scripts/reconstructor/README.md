# Reconstructor (Purification) Transformer

> Part of [DEPT](../../README.md) — **phase 6**. The purification stage: a small causal Transformer that denoises a flow the ensemble flagged as adversarial, mapping it back onto the clean-traffic manifold.

```bash
python3 scripts/reconstructor/run_reconstructor.py
```

## What it does

TANTGen evades by perturbing two quantities — packet **timing** (IPAT) and packet **size** (IPv4 total length / payload padding). The reconstructor learns the inverse: given the adversarial flow, reproduce the clean one, so the underlying attack is detectable again by the NIDS it was built to evade.

Training pairs are **matched `(adversarial → clean)` flows**, joined by canonical 5-tuple (`flow_ids.npy`), never by row position — TANTGen splits, merges and drops flows. Matched pairs keep equal packet counts ~99.8% of the time; divergent pairs are truncated to the shorter length.

Architecture: `Linear 1090 → 128`, sinusoidal PE, 3 pre-LN encoder layers (4 heads, causal mask), `Linear 128 → 1090` producing a **residual Δ**, so `reconstruction = adv + Δ`. **677,698 parameters.** The zero-initialised residual head makes identity the default: capacity goes to the localized timing/length correction and the ~1050 untouched header bits are preserved for free. Loss is per-element smooth-L1 against the clean target, uniform over all 1090 dims — the model is **never told which dims the attack perturbs**, keeping the defense attack-agnostic. Causal by design, so it is KV-cacheable for streaming purification.

Four metric groups are reported, overall and per `<base>_adv_<mode>`: reconstruction MSE on perturbed dims; purification gain through DEPT's **own** TCN (kept only as an in-distribution sanity check — it is circular); protocol compliance via `tshark` on realized PCAPs; and, as the **headline**, detection restoration through the **independent NIDS** each attack was built to evade (Kitsune for the Kitsune-native attacks, TANTGen's ML-NIDS classifiers for the CIC ones, routed by the shared `src/models/nids/` registry). For that last group three packet sets — `adv`, `purified`, `clean` — are realized from the *same* adversarial test packets, overlaying only size and timing, so the only variable the NIDS sees is what the purifier touches.

## Results

Trained on **2,566,489 matched pairs** (1.80 M / 385 K / 385 K, seed=42), 20 epochs, 10.0 h on one GPU, best epoch 19. Evaluated on the held-out 385 K-pair test split.

| Group | Result (overall) |
|---|---|
| Reconstruction — perturbed-dim MSE, adv → recon | 0.918 → **0.054** (17.0× reduction) |
| Purification gain (through DEPT's TCN, sanity check) | 0.675 → **0.428** → 0.421 clean (gain 0.247, residual gap 0.007) |
| Protocol compliance (`tshark`, realizable packets) | **97.2%** (67,106 bad of 2,412,357) |
| **NIDS restoration (headline)** — detection adv → purified | **0.781 → 0.906** (clean 0.980); **63.0% of destroyed detection restored** |

By family: Kitsune 0.019 → 0.792 against a 0.793 clean ceiling (**99.9%** restoration); ML-NIDS 0.808 → 0.910 against 0.987 (57.1%, diluted by rows where nothing needed restoring).

- **Restoration is strong exactly where the attack actually evaded**: `ddos` both/size 0.08/0.005 → 0.87/0.82, `portscan` both/size 0.011/0.013 → 0.68/0.48, `ssdp_flood_iat` 0.001 → 1.00, `syn_ddos_size` 0.879 → 1.00.
- **Purifying non-evaded traffic hurts.** Where TANTGen never fooled the NIDS (`dns_ddos` all modes, `portscan_iat`, `brute_force`), denoising *lowers* detection by 2–24 points. The purifier must be **gated** — quantified end-to-end in the [deployment baseline](../baseline/README.md).
- **Near-no-op where there is nothing to undo**: on the reflection floods' both/iat variants `mse_before ≈ 0`, and the model correctly leaves those flows almost unchanged.
- **Mirai is the weakest reconstruction** (MSE 1.9–2.5 → 1.6–2.1, ~1.3×) and an uninformative NIDS row — Kitsune detects only ~0.8% of even the *clean* mirai slice per-flow, so there is nothing measurable to restore.

**Realization caveat.** A size edit on an application-layer **text** protocol (SSDP/HTTP `M-SEARCH`) keeps IP/UDP lengths and checksums valid but corrupts the L7 grammar, so `tshark` reports `_ws.malformed`. Byte-overlay cannot faithfully resize these, so **450,758** such packets (96% of them `ssdp_flood` both/size) are reported separately as `n_unrealizable_l7` rather than scored as failures; the 97.2% is over the 2.41 M realizable packets. Two remaining weak spots are honest limitations: `ddos` (~84–85%, checksum/length corner cases at scale) and `syn_ddos_iat` (17.8%, low-support — a scapy re-write checksum bug on 2,088 packets, not a purifier failure; `_ws.malformed` is 0 there).

Artifacts under `data/models/reconstructor/`: `reconstructor.pt`, `metrics.json`, `split.npz`, `training_curves.json`, `realized/<source>.pcap`, `realized_nids/<source>_{adv,purified,clean}.pcap`, plus reconstruction-error, purification-gain, NIDS-restoration and training-curve plots.
