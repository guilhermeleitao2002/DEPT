# DEPT × TANTGen — Adversarial PCAP Generation

> Part of [DEPT](../../README.md). Thin driver that operates [TANTGen](../../../Transformer-based_Adversarial_Network_Traffic_Generator/) (sibling repo) as an external tool to produce adversarial versions of attack PCAPs. Its outputs are picked up automatically by [Phase 1](../packet_to_tensor/README.md) via `process_all_attacks.py` and become the positive class for [Phase 2](../rf/README.md).

TANTGen itself is documented in its own repository — this README only covers the DEPT-side glue.

## Outputs

```
DEPT/data/adversarial/<attack>/<attack>_adversarial_{both,iat,size}.pcap
```

There are no command-line options. Every run regenerates the full set — all 10
attacks × 3 modes — re-running finetune + apply and overwriting the final
adversarial PCAPs. (The one-time global pretrain and the deterministic
intermediate artifacts — benign/malicious splits, feature CSVs — are reused if
already on disk.)

## Quick start

From `DEPT/`:

```bash
# 0. One-time system + Python deps
sudo apt install tshark wireshark-common -y
pip3 install --user -r ../Transformer-based_Adversarial_Network_Traffic_Generator/requirements.txt
pip3 install --user captum
( cd ../Transformer-based_Adversarial_Network_Traffic_Generator/src/Kitsune/Feature_Extraction \
  && python3 setup.py build_ext --inplace )

# 1. Generate adversarial PCAPs. The one-time TANTGen pretrain runs first
#    automatically if its checkpoint is missing, then all 10 attacks × 3 modes.
python3 scripts/adversarial_generation/generate_adversarials.py
```

## Attack set

DEPT currently consumes **10 adversarial sources** (the `TANTGEN_KEYS` set in `scripts/packet_to_tensor/process_all_attacks.py`), three perturbation modes each (`both` = IAT + size, `iat` = timing only, `size` = payload padding only) — i.e. **30 PCAPs total**. Every source pairs with a matching benign source:

| Attack | Dataset | Notes |
|---|---|---|
| `mirai` | Kitsune | Botnet C&C / IoT DDoS |
| `ssdp_flood` | Kitsune | UDP amplification |
| `ddos` | CIC-IDS2017 | Friday afternoon DDoS window |
| `portscan` | CIC-IDS2017 | Friday afternoon recon scan |
| `brute_force` | CIC-IDS2017 | Tuesday FTP-Patator + SSH-Patator combined window |
| `ntp_ddos` | CIC-DDoS-2019 | NTP monlist amplification flood (UDP) |
| `dns_ddos` | CIC-DDoS-2019 | DNS amplification flood (UDP) |
| `ssdp_ddos` | CIC-DDoS-2019 | SSDP/UPnP amplification flood (UDP) |
| `syn_ddos` | CIC-DDoS-2019 | TCP SYN flood (distinct from the Kitsune `syn_dos`) |
| `ldap_ddos` | CIC-DDoS-2019 | CLDAP amplification flood (UDP) |

`data/adversarial/brute_force_kitsune/` and `data/adversarial/portscan_kitsune/` may also appear on disk but are intentionally left out of `TANTGEN_KEYS` in `scripts/packet_to_tensor/process_all_attacks.py`; they are skipped by `build_adversarial_attacks()` until explicitly opted in.

CIC-IDS2017 attacks reuse the same time windows as `process_all_attacks.py`, so the malicious PCAPs DEPT trains on match the ones TANTGen perturbed. The CIC-DDoS-2019 floods have no clean time window: the generator caps its malicious input to the first 1,000,000 packets (the hardcoded `MAX_MALICIOUS_PACKETS`) and `process_all_attacks.py` packet-caps the matching benign capture to the same first 1,000,000 packets, so the benign and adversarial sides remain paired and balanced.

> The `generate_adversarials.py` script's built-in `ATTACKS` list covers all 10 attacks across three families (Kitsune: `mirai`, `ssdp_flood`; CIC-IDS2017: `ddos`, `portscan`, `brute_force`; CIC-DDoS-2019: `ntp_ddos`, `dns_ddos`, `ssdp_ddos`, `syn_ddos`, `ldap_ddos`). The PCAPs actually consumed by DEPT are whatever lives under `data/adversarial/<attack>/<attack>_adversarial_<mode>.pcap`; the table above is the source of truth for that on-disk set.

## How it runs

`generate_adversarials.py` takes no arguments — the DEPT core pipeline has one fixed configuration baked into the script (the `ATTACKS` registry, the three modes, per-attack evasion margins, split ratios, and the 1,000,000-packet malicious cap `MAX_MALICIOUS_PACKETS`). It runs each attack through TANTGen's standard pipeline (split → surrogate → finetune → apply) and lands the final adversarial PCAP under `DEPT/data/adversarial/<key>/`. All intermediate artifacts (benign/malicious splits, feature CSVs, checkpoints) stay inside the TANTGen repo because TANTGen modules load them from paths baked into its `config-folders.yaml`. The deterministic intermediate stages and the one-time pretrain are skip-on-exists; the final finetune + apply always re-run, overwriting the adversarial PCAPs.

## End-to-end retraining flow

```bash
# 1. (If sources or driver changed) regenerate adversarial PCAPs
python3 scripts/adversarial_generation/generate_adversarials.py

# 2. Build tensors with a shared standardizer (benign + adversarial in one pass)
python3 scripts/packet_to_tensor/process_all_attacks.py

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
2. **Recompute `mean_std.json` over benign + adversarial inputs jointly.** Implemented at the packet-to-tensor stage — see [Phase 1 README → design decisions](../packet_to_tensor/README.md#design-decisions). Smoke check on SYN_DoS adversarial-only inputs: shared fit yields IPAT mean 9.85 / std 31.08 vs the previous benign-only 0.04 / 3.24 — confirming the heavier tails are now captured.

**Deliberately not implemented:**

3. **Auto-tag no-op adversarials** — only known no-ops were Fuzzing's, and Fuzzing was removed outright.
4. **Drop `dos_slow` size/both from training** — training-stage policy, not a data-generation concern. Deferred to whoever assembles the training split.
5. **Run a smoke-train pass and inspect purification reconstruction loss** — purification module not yet in the project; out of scope here.
