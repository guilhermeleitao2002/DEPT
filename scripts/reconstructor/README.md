# Reconstructor (Purification) Transformer

> Part of [ADEPT](../../README.md). The purification stage — a small causal (decoder-only) Transformer that denoises a flow the ensemble flagged as adversarial, mapping it back onto the clean-traffic manifold.

## Role in the defense

When ADEPT's ensemble flags a flow as adversarial, the reconstructor **purifies** it before it reaches the protected NIDS. TANTGen evades detection by perturbing two quantities — packet **timing** (IPAT) and packet **size** (IPv4 total length / payload padding). The reconstructor learns the inverse: given the adversarial flow, it reconstructs the clean flow, undoing those perturbations so the underlying attack is once again detectable. Success is measured by **purification gain** — feeding the reconstruction back through the detector and observing `P(adversarial)` fall toward the clean floor.

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
- **Pure by default** (`loss_emphasis: none`): uniform weight over all 1090 dims — the model is supervised only to "reconstruct the clean flow" and is **never told which dimensions the attack perturbs**. With `reduction=mean` every dim's gradient is scaled equally, and on a residual head the unperturbed dims have residual-target `Δ=0`, so the learning signal concentrates on the perturbed dims without any weighting.
- **Footprint ablation** (`loss_emphasis: perturbation`, off by default): up-weights dims where `clean != adv` by `perturb_weight`. A white-box experiment ("does knowing the perturbation footprint help?"), not part of the main run.
- **Split**: flow-atomic, source-stratified, same `seed=42` as the base learners, so the held-out test pairs never leak into any model's training set.
- **Early stopping** on the validation reconstruction loss (lower is better).

## Evaluation

Three metric groups (`metrics.json`), reported overall and per `<base>_adv_<mode>`:

1. **Reconstruction quality** — MSE on the perturbed dims, *before* (adv vs clean) vs *after* (recon vs clean). The before→after drop quantifies the denoising.
2. **Purification gain** — the reconstruction (rounded back to valid nPrint) is fed through the trained **TCN** alongside the adversarial and clean flows; the per-flow last-timestep `P(adversarial)` is reported for each. `gain = P(adv|adv) − P(adv|recon)`; the residual gap to `P(adv|clean)` shows how close to the clean floor the purifier lands.
3. **Protocol-compliance rate** — two checks:
   - *feature-space* (always on): decode IPv4 version / IHL / total-length from the rounded reconstruction and assert TCP/IP invariants.
   - *traffic-space* (best-effort): a sample of test flows is **realized** back onto the original adversarial PCAP packets (overlay the reconstructed length/timing onto the real packet; scapy recomputes checksums), written to `realized/<source>.pcap`, and validated with `tshark` (`_ws.malformed` / bad-checksum count).

## Output files (`data/models/reconstructor/`)

| File | Description |
|---|---|
| `reconstructor.pt` | Best PyTorch `state_dict` (min val reconstruction loss) |
| `reconstructor_config.json` | Resolved model + training config |
| `split.npz` | `train_idx` / `val_idx` / `test_idx` into the (deterministic) pair table that eval rebuilds from config |
| `split_summary.json` | Per-split / per-source pair counts |
| `training_info.json` | Best epoch, device, train time, pair stats, effective config |
| `training_curves.json` | Per-epoch train_loss / val_loss / lr |
| `metrics.json` | Reconstruction / purification / protocol-compliance, overall + per-source |
| `realized/<source>.pcap` | Realized purified packets (traffic-space, for tshark + replay) |
| `reconstruction_error.png` | Per-source MSE on perturbed dims, before vs after |
| `purification_gain.png` | Per-source TCN `P(adv)` for adversarial / reconstructed / clean |
| `training_curves.png` | Train / val reconstruction loss over epochs |
