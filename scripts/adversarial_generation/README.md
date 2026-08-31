# Adversarial PCAP Generation (DEPT × TANTGen)

> Part of [DEPT](../../README.md) — **phase 1**. Thin driver that runs [TANTGen](../../../Transformer-based_Adversarial_Network_Traffic_Generator/) (sibling repo) as an external tool to produce the adversarial PCAPs that become DEPT's positive class. TANTGen itself is documented in its own repo; this covers only the DEPT-side glue.

```bash
python3 scripts/adversarial_generation/generate_adversarials.py
```

No arguments — one fixed configuration is baked into the script (`ATTACKS` registry, three modes, per-attack evasion margins, split ratios, the 1,000,000-packet `MAX_MALICIOUS_PACKETS` cap). Each attack runs TANTGen's split → surrogate → finetune → apply pipeline, landing PCAPs at `data/adversarial/<attack>/<attack>_adversarial_{both,iat,size}.pcap`. The one-time pretrain and deterministic intermediates are skip-on-exists; finetune + apply always re-run.

## Attack set

**10 attacks × 3 modes = 30 PCAPs** (`both` = timing + size, `iat` = timing only, `size` = payload padding only). Every source pairs with a matching benign source.

| Family | Attacks |
|---|---|
| Kitsune | `mirai`, `ssdp_flood` |
| CIC-IDS2017 | `ddos`, `portscan`, `brute_force` |
| CIC-DDoS-2019 | `ntp_ddos`, `dns_ddos`, `ssdp_ddos`, `syn_ddos`, `ldap_ddos` |

CIC-IDS2017 attacks reuse the same time windows as `process_all_attacks.py`; the CIC-DDoS-2019 floods have no clean window, so both sides are capped to their first 1,000,000 packets, keeping benign and adversarial paired.

**Fuzzing (Kitsune) is excluded**: its 200 B snaplen leaves no room to perturb — `size` output was byte-identical to the source (same MD5), and its ~1.7 ms IAT with tight variance made `iat`/`both` statistically indistinguishable. SYN_DoS shares the snaplen but has loose enough timing that only its `size` mode is weak.

## Results

Full generation: **96,242 s (26.7 h)** for all 30 PCAPs.

| Attack | Time | Attack | Time |
|---|---:|---|---:|
| `ntp_ddos` | 11 h 53 m | `ssdp_flood` | 1 h 31 m |
| `syn_ddos` | 5 h 14 m | `ldap_ddos` | 1 h 34 m |
| `ssdp_ddos` | 2 h 13 m | `brute_force` | 1 h 01 m |
| `ddos` | 1 h 24 m | `dns_ddos` | 50 m |
| `mirai` | 33 m | `portscan` | 31 m |

`ntp_ddos` dominates because its 37 GB raw capture must be streamed to reach the malicious cap.
