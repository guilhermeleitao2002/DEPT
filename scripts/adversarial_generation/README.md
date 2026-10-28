# Adversarial PCAP Generation (DEPT × TANTGen)

> Part of [DEPT](../../README.md), **phase 1**. Runs TANTGen (expected in `../Transformer-based_Adversarial_Network_Traffic_Generator`) to produce the adversarial PCAPs that form DEPT's positive class. TANTGen itself is documented in its own repository.

```bash
python3 scripts/adversarial_generation/generate_adversarials.py
```

There are no arguments: the attack list, the three modes, the per-attack evasion margins, the split ratios and the 1,000,000-packet `MAX_MALICIOUS_PACKETS` cap are set in the script. For each attack it runs TANTGen's split → surrogate → finetune → apply steps and writes `data/adversarial/<attack>/<attack>_adversarial_{both,iat,size}.pcap`. The one-time pretrain and the intermediate files are skipped if they exist; finetune and apply always run again.

## Attacks

**10 attacks × 3 modes = 30 PCAPs** (`both` = timing + size, `iat` = timing only, `size` = payload padding only). Each one has a matching benign source.

| Dataset | Attacks |
|---|---|
| Kitsune | `mirai`, `ssdp_flood` |
| CIC-IDS2017 | `ddos`, `portscan`, `brute_force` |
| CIC-DDoS-2019 | `ntp_ddos`, `dns_ddos`, `ssdp_ddos`, `syn_ddos`, `ldap_ddos` |

The CIC-IDS2017 attacks use the same time windows as `process_all_attacks.py`. The CIC-DDoS-2019 captures have no clean attack window, so both the benign and the adversarial side are cut to the first 1,000,000 packets.

Kitsune's Fuzzing attack is not used: its 200 B snaplen leaves no room for padding (the `size` output was identical to the input), and its very regular ~1.7 ms IAT made `iat`/`both` no different from the original.

## Runtime

Full generation took **96,242 s (26.7 h)** for the 30 PCAPs.

| Attack | Time | Attack | Time |
|---|---:|---|---:|
| `ntp_ddos` | 11 h 53 m | `ssdp_flood` | 1 h 31 m |
| `syn_ddos` | 5 h 14 m | `ldap_ddos` | 1 h 34 m |
| `ssdp_ddos` | 2 h 13 m | `brute_force` | 1 h 01 m |
| `ddos` | 1 h 24 m | `dns_ddos` | 50 m |
| `mirai` | 33 m | `portscan` | 31 m |

`ntp_ddos` takes the longest because its raw capture is 37 GB.
