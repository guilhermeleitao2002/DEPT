# ADEPT × TANTGen — Adversarial PCAP Generation

> Part of [ADEPT](../../README.md). Thin driver that operates [TANTGen](../../../Transformer-based_Adversarial_Network_Traffic_Generator/) (sibling repo) as an external tool to produce adversarial versions of attack PCAPs. Its outputs are picked up automatically by [Phase 1](../packet_to_tensor/README.md) via `process_all_attacks.py --tantgen` and become the positive class for [Phase 2](../rf/README.md).

TANTGen itself is documented in its own repository — this README only covers the ADEPT-side glue.

## Outputs

```
ADEPT/data/adversarial/<attack>/<attack>_adversarial_{both,iat,size}.pcap
```

The driver auto-skips outputs already on disk; pass `--force` to rebuild,
`--only <key>` / `--modes <m>` to scope a subset.

## Quick start

From `ADEPT/`:

```bash
# 0. One-time system + Python deps
sudo apt install tshark wireshark-common -y
pip3 install --user -r ../Transformer-based_Adversarial_Network_Traffic_Generator/requirements.txt
pip3 install --user captum
( cd ../Transformer-based_Adversarial_Network_Traffic_Generator/src/Kitsune/Feature_Extraction \
  && python3 setup.py build_ext --inplace )

# 1. Pretrain TANTGen base (one-time, after at least one attack split exists)
python3 scripts/adversarial_generation/generate_adversarials.py --pretrain

# 2. Generate adversarial PCAPs
python3 scripts/adversarial_generation/generate_adversarials.py
```

## Attack set

ADEPT currently consumes **6 adversarial sources** under `--tantgen`, three perturbation modes each (`both` = IAT + size, `iat` = timing only, `size` = payload padding only) — i.e. **18 PCAPs total**. Five sources pair with a matching benign source; `ddos_kitsune` is adversarial-only:

| Attack | Dataset | Notes |
|---|---|---|
| `mirai` | Kitsune | Botnet C&C / IoT DDoS |
| `ssdp_flood` | Kitsune | UDP amplification |
| `ddos` | CIC-IDS2017 | Friday afternoon DDoS window |
| `portscan` | CIC-IDS2017 | Friday afternoon recon scan |
| `brute_force` | CIC-IDS2017 | Tuesday FTP-Patator + SSH-Patator combined window |
| `ddos_kitsune` | adversarial-only | Kitsune-trained TANTGen perturbing the CIC DDoS benign manifold; no separate benign source — augments the adversarial pool only |

`data/adversarial/brute_force_kitsune/` and `data/adversarial/portscan_kitsune/` are also on disk but intentionally left out of `TANTGEN_KEYS` in `scripts/packet_to_tensor/process_all_attacks.py`; they are skipped by `build_adversarial_attacks()` until explicitly opted in.

CIC attacks reuse the same time windows as `process_all_attacks.py`, so the malicious PCAPs ADEPT trains on match the ones TANTGen perturbed.

> The `generate_adversarials.py` script's built-in `ATTACKS` list still reflects the older 6-attack TANTGen scope (`mirai`, `syn_dos`, `ssdp_flood`, `ddos`, `dos_slow`, `portscan`). The PCAPs actually consumed by ADEPT are whatever lives under `data/adversarial/<attack>/<attack>_adversarial_<mode>.pcap` — the table above is the source of truth for that on-disk set, including `brute_force` (which was produced outside the current generator script).

## CLI — `generate_adversarials.py`

| Flag | Purpose |
|---|---|
| `--only KEY [KEY...]` | Restrict to these attack keys. |
| `--exclude KEY [KEY...]` | Skip these attack keys. |
| `--modes {both,iat,size}` | Perturbation modes (default: all three). |
| `--force` | Rebuild adversarial PCAPs even if present. |
| `--pretrain` | Run the one-time TANTGen pretrain step and exit. Requires at least one attack's benign split to already exist. |
| `--dry-run` | List the selection and exit. |

The driver runs each attack through TANTGen's standard pipeline (split → surrogate → finetune → apply) and lands the final adversarial PCAP under `ADEPT/data/adversarial/<key>/`. All intermediate artifacts (benign/malicious splits, feature CSVs, checkpoints) stay inside the TANTGen repo because TANTGen modules load them from paths baked into its `config-folders.yaml`. Most stages are skip-on-exists, so re-runs only redo what's needed.

## End-to-end retraining flow

```bash
# 1. (If sources or driver changed) regenerate adversarial PCAPs
python3 scripts/adversarial_generation/generate_adversarials.py [--force]

# 2. Build tensors with a shared standardizer (benign + adversarial in one pass)
python3 scripts/packet_to_tensor/process_all_attacks.py --tantgen --include-benign --force

# 3. Train downstream models (RF today; TCN / Transformer later)
python3 scripts/rf/run_rf.py
```

## Why Fuzzing was excluded

Kitsune's [Fuzzing PCAP](../../data/Kitsune/pcap/Fuzzing_pcap.pcapng) is captured with `snaplen = 200 B`. That cap leaves TANTGen with effectively no room to perturb either channel:

- **size:** TANTGen pads TCP/UDP payloads with random bytes. With every packet already at the 200 B ceiling, `clamp_pad()` returns 0 and the writer emits the original packet unchanged. We confirmed `fuzzing_adversarial_size.pcap` was byte-identical to the original (same MD5).
- **iat:** Fuzzing's average IAT is ~1.7 ms with very tight variance. The transformer's predicted perturbations average to the same float at the precision we sampled, so the resulting traffic is statistically indistinguishable from the source.
- **both:** same outcome — driven by the same two limits.

Fuzzing is therefore excluded from both `generate_adversarials.py` and `process_all_attacks.py`. SYN_DoS shares the same 200 B snaplen, but its IAT distribution is loose enough that `iat` and `both` modes still produce useful shifts; only `size` is weak.

## Post-generation decisions (audit trail)

After analyzing the 21 (now 18) adversarial PCAPs, five recommendations were considered:

**Implemented:**

1. **Drop Fuzzing entirely.** Reasoning above.
2. **Recompute `mean_std.json` over benign + adversarial inputs jointly.** Implemented at the packet-to-tensor stage — see [Phase 1 README → shared standardizer](../packet_to_tensor/README.md#why-a-shared-mean_stdjson). Smoke check on SYN_DoS adversarial-only inputs: shared fit yields IPAT mean 9.85 / std 31.08 vs the previous benign-only 0.04 / 3.24 — confirming the heavier tails are now captured.

**Deliberately not implemented:**

3. **Auto-tag no-op adversarials** — only known no-ops were Fuzzing's, and Fuzzing was removed outright.
4. **Drop `dos_slow` size/both from training** — training-stage policy, not a data-generation concern. Deferred to whoever assembles the training split.
5. **Run a smoke-train pass and inspect purification reconstruction loss** — purification module not yet in the project; out of scope here.
