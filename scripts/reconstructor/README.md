# Reconstructor (Purification) Transformer

> Part of [DEPT](../../README.md), **phase 6**. A small causal transformer that purifies flows flagged as adversarial.

```bash
python3 scripts/reconstructor/run_reconstructor.py
```

## Method

TANTGen changes two things: packet **timing** (IPAT) and packet **size** (IPv4 total length, by padding the payload). The reconstructor learns to reverse this: given the adversarial flow, it outputs the clean one, so that the NIDS the attack was built to evade can detect it again.

Training pairs are **(adversarial, clean)** flows with the same canonical 5-tuple (`flow_ids.npy`), not the same row, since TANTGen can split, merge and drop flows. About 99.8% of matched pairs have the same number of packets; the others are cut to the shorter length.

Architecture: `Linear 1090 → 128`, sinusoidal positional encoding, 3 pre-LN encoder layers (4 heads, causal mask) and `Linear 128 → 1090` giving a residual Δ, so `reconstruction = adv + Δ`. **677,698 parameters.** The residual head starts at zero, so the untrained model is the identity. The loss is smooth-L1 against the clean flow over all 1090 dimensions; the model is not told which dimensions the attack changes.

Metrics, overall and per `<base>_adv_<mode>`:

- reconstruction MSE on the perturbed dimensions;
- purification gain through DEPT's own TCN (only a sanity check, since the TCN was trained on this data);
- protocol compliance with `tshark` on the realized packets;
- detection restoration through the **independent NIDS** each attack was built to evade (Kitsune for `mirai`/`ssdp_flood`, TANTGen's ML-NIDS for the CIC attacks, see `src/models/nids/`). The `adv`, `purified` and `clean` versions are realized on the same adversarial packets, changing only size and timing.

## Results

Trained on **2,566,489 matched pairs** (1.80 M / 385 K / 385 K, seed=42), 20 epochs, 10.0 h on one GPU, best at the last epoch. Evaluated on the 385 K held-out test pairs.

| Metric | Result (overall) |
|---|---|
| MSE on perturbed dims, adv → recon | 0.918 → **0.054** (17.0× lower) |
| TCN `P(adv)`: adv → recon → clean | 0.675 → **0.428** → 0.421 |
| Protocol compliance (`tshark`, realizable packets) | **97.2%** (67,106 bad of 2,412,357) |
| **Independent-NIDS detection**, adv → purified | **0.781 → 0.906** (clean 0.980), **63.0% restored** |

Per NIDS: Kitsune 0.019 → 0.792 against 0.793 clean (**99.9%** restored); ML-NIDS 0.808 → 0.910 against 0.987 (57.1%, lower because many rows had nothing to restore).

- **Restoration is large where TANTGen evaded the NIDS**: `ddos` both/size 0.08/0.005 → 0.87/0.82, `portscan` both/size 0.011/0.013 → 0.68/0.48, `ssdp_flood_iat` 0.001 → 1.00, `syn_ddos_size` 0.879 → 1.00.
- **Purifying traffic that did not evade lowers detection.** Where TANTGen never fooled the NIDS (`dns_ddos` all modes, `portscan_iat`, `brute_force`), detection drops by 2–24 points. This is why the purifier is placed behind the detector, see the [deployment baseline](../baseline/README.md).
- **Almost no change where there is nothing to undo**: on the both/iat variants of the reflection floods `mse_before ≈ 0` and the model leaves the flows almost unchanged.
- **Mirai is the weakest case** (MSE 1.9–2.5 → 1.6–2.1), and its NIDS row says little: Kitsune detects only ~0.8% of even the clean mirai flows.

**Realization.** Changing the size of an application-layer text packet (SSDP/HTTP `M-SEARCH`) keeps IP/UDP lengths and checksums valid but breaks the text, so `tshark` reports `_ws.malformed`. These **450,758** packets (96% of them `ssdp_flood` both/size) are counted separately as `n_unrealizable_l7`, and the 97.2% is over the 2.41 M other packets. The lowest compliance rates are `ddos` (~84–85%) and `syn_ddos_iat` (17.8%, 2,088 packets with a checksum problem from the scapy rewrite; no `_ws.malformed` there).

Artifacts in `data/models/reconstructor/`: `reconstructor.pt`, `metrics.json`, `split.npz`, `training_curves.json`, `realized/<source>.pcap`, `realized_nids/<source>_{adv,purified,clean}.pcap`, and the reconstruction-error, purification-gain, NIDS-restoration and training-curve plots.
