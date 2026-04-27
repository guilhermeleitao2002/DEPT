# ADEPT × TANTGen — Adversarial PCAP Generation

This directory contains the driver that operates [TANTGen](../../../Transformer-based_Adversarial_Network_Traffic_Generator/)
as an external tool to produce adversarial versions of attack packets for the
ADEPT defense pipeline.

Files:

- [generate_adversarials.py](generate_adversarials.py) — the driver. Iterates the selected attacks, drives TANTGen end-to-end (split → surrogate → finetune → apply) and lands the resulting PCAPs under `ADEPT/data/adversarial/<attack>/`.

## Quick start

From `ADEPT/`:

```bash
# 0. One-time system + Python deps
sudo apt install tshark wireshark-common
pip install --user -r ../Transformer-based_Adversarial_Network_Traffic_Generator/requirements.txt
pip install --user captum
( cd ../Transformer-based_Adversarial_Network_Traffic_Generator/src/Kitsune/Feature_Extraction \
  && python3 setup.py build_ext --inplace )

# 1. Pretrain TANTGen base (one-time, after at least one attack split exists)
python3 scripts/adversarial_generation/generate_adversarials.py --pretrain

# 2. Generate adversarial PCAPs (6 attacks × 3 modes = 18 outputs)
python3 scripts/adversarial_generation/generate_adversarials.py
```

Outputs:

```
ADEPT/data/adversarial/<attack>/<attack>_adversarial_{both,iat,size}.pcap
```

The driver auto-skips outputs already on disk; pass `--force` to rebuild,
`--only <key>` / `--modes <m>` to scope a subset.

## Attack set

Six attacks, three modes each:

| Attack | Dataset | Notes |
|---|---|---|
| mirai | Kitsune | Botnet C&C / IoT DDoS |
| syn_dos | Kitsune | Volumetric TCP DoS — `size` mode is weak (snaplen-truncated source) |
| ssdp_flood | Kitsune | UDP amplification |
| ddos | CIC-IDS2017 | Friday afternoon DDoS window |
| dos_slow | CIC-IDS2017 | Slowloris/SlowHTTPTest/Hulk/GoldenEye composite |
| portscan | CIC-IDS2017 | Friday afternoon recon scan |

Modes: `both` perturbs IAT and packet size; `iat` perturbs only timing; `size` perturbs only payload padding.

## Why Fuzzing was excluded

Kitsune's [Fuzzing PCAP](../../data/Kitsune/pcap/Fuzzing_pcap.pcapng) is captured with `snaplen = 200 B`. That cap leaves TANTGen with effectively no room to perturb either channel:

- **size**: TANTGen pads TCP/UDP payloads with random bytes. With every packet already at the 200 B ceiling, `clamp_pad()` returns 0 and the writer emits the original packet unchanged. We confirmed `fuzzing_adversarial_size.pcap` was byte-identical to the original (same MD5).
- **iat**: Fuzzing's average IAT is ~1.7 ms with very tight variance. The transformer's predicted perturbations average to the same float at the precision we sampled, so the resulting traffic is statistically indistinguishable from the source.
- **both**: same outcome — driven by the same two limits.

Because the variants carry no measurable adversarial signal, they are excluded from both `generate_adversarials.py` and `process_all_attacks.py`. The corresponding directories under `data/adversarial/fuzzing/` and `data/tensors/fuzzing/` were deleted. SYN_DoS shares the same 200 B snaplen, but its IAT distribution is loose enough that `iat` and `both` modes still produce useful shifts; only `size` is weak.

## Manifest of decisions from the post-generation analysis

After analyzing the 21 (now 18) adversarial PCAPs, five recommendations were considered. This is the audit trail.

### Implemented

1. **Drop Fuzzing entirely.** Removed from both attack registries; physical PCAPs and tensor outputs deleted. Reasoning above.
2. **Recompute `mean_std.json` over benign + adversarial inputs jointly.** Implemented at the packet-to-tensor stage. `run_packet_to_tensor.py --fit-mean-std` accepts a list of PCAPs, accumulates IPAT mean/std with a streaming Welford / Chan parallel update (one batch at a time, RAM-bounded), and writes a single shared JSON. `process_all_attacks.py` runs this fit once before per-attack processing and passes `--mean-std data/tensors/mean_std.json` to every per-attack invocation, mirroring the file into each per-attack output dir for traceability. Smoke check on SYN_DoS adversarial-only inputs: shared fit yields IPAT mean 9.85, std 31.08 vs the previous benign-only fit's mean 0.04, std 3.24 — confirming the heavier tails are now captured.

### Not implemented (deliberate)

3. **Auto-tag no-op adversarials.** Skipped: the only known no-ops were Fuzzing's, and we removed Fuzzing outright. A future no-op detector would be redundant in this dataset.
4. **Drop `dos_slow` size/both from training.** Skipped: this is a training-stage policy, not a data-generation concern. Random-byte padding inside slow-HTTP attacks may degrade the attack's semantics, but ADEPT will still see the data and its loss curves will reveal whether the variants hurt or help. Decision deferred to whoever assembles the training split.
5. **Run a smoke-train pass and inspect purification reconstruction loss.** Skipped: the purification component is not yet in the project; this stage targets dataset preparation only.

## Re-running after a code change

Both stages are idempotent:

- `generate_adversarials.py` skips per-attack/per-mode outputs that already exist (`--force` to rebuild).
- `process_all_attacks.py` skips the shared mean/std fit if `data/tensors/mean_std.json` is present (`--refit-shared-mean-std` to refit; `--no-shared-mean-std` to disable the shared step entirely and revert to per-PCAP fits).

End-to-end, after the changes here, retraining is:

```bash
# 1. (If sources or driver changed) regenerate adversarial PCAPs
python3 scripts/adversarial_generation/generate_adversarials.py [--force]

# 2. Build tensors with a shared standardizer
python3 scripts/packet_to_tensor/process_all_attacks.py --tantgen --include-benign --force
python3 scripts/packet_to_tensor/process_all_attacks.py --adversarial --force

# 3. Train the model (driver lives elsewhere — out of scope for this README)
```
