# Gradient-Guided Evasion (RQ1)

> Part of [DEPT](../../README.md), **phase 7** of `run_pipeline.sh` (global mode only).

```bash
python3 scripts/gradient_ablation/run_gradient_ablation.py                 # all sources
python3 scripts/gradient_ablation/run_gradient_ablation.py --attack mirai
```

## Question

TANTGen is black-box and does not use gradients, so it cannot answer **RQ1**:

> Does heterogeneous stacking (a non-differentiable Random Forest fused with a differentiable TCN) measurably raise the cost of a transferable, gradient-guided evasion attack relative to a single-architecture baseline?

Here one perturbation is computed **white-box against the TCN** and then scored, unchanged, on three targets: the RF, the TCN and the RF+TCN+meta-learner stack. The TCN is both the single-architecture baseline and part of the stack, so the attacker knows the differentiable part exactly and has no gradient for the other.

## Setup

- **Increase-only perturbations** (padding bytes and delay seconds, on top of TANTGen's). The attacker cannot remove what is already sent, and since DEPT's positive class is TANTGen traffic, removing the perturbation would also stop the attack against the NIDS.
- **Budgets in physical units.** IPAT is z-scored with std = 208.2 s while the nPrint channels are bits, so an `eps=0.01` ball would allow ±2.08 s of delay and nothing in the bits. Only the two channels that `realize.py` can write back are changed: the IPv4 total-length bits `[16:32]` and IPAT `[1088]`.
- **Thresholds at a matched FPR**, not 0.5. The targets are calibrated very differently (median benign verdict: RF 0.043, TCN 0.379, stack 0.002). **ASR only counts flows that the target detects without perturbation.**
- **The ASR curve is the monotone envelope**: a flow evaded at budget *b* counts as evaded at every larger budget, since the feasible sets are nested.
- **Attack**: PGD with restarts, plus a `random` search with the same number of queries; the reported attacker (`best`) takes the better of the two. ART's `AutoProjectedGradientDescent` runs on the relaxed problem as a check that the search is not under-tuned. The size gradient goes through the pseudoinverse of the 16-bit length decoding.
- **Sidecar**: scoring the RF on a perturbed flow needs its 24 prefix statistics recomputed from perturbed per-packet values, which are not saved (`Payload_Length` and `Payload_Entropy` are not in nPrint). One cached scapy pass per source recomputes them; `verify_sidecar` checks that they reproduce `rf_checkpoints.npy` at δ=0.

## Results

9,000 flows (300 × 30 sources), 162,000 scored records, 6 budgets, 4.3 h. Reported at `fpr_0.05` with the `best` attacker.

| Budget | median spend / flow | RF | TCN | **Stack** |
|---|---|---|---|---|
| b8 | 13 B, 0.004 ms/pkt | 8.6% | 40.5% | **1.7%** |
| b32 | 47 B, 0.041 ms/pkt | 10.7% | 50.1% | **4.7%** |
| b128 | 167 B, 0.419 ms/pkt | 13.1% | 52.2% | **8.5%** |
| b512 | 537 B, 3.68 ms/pkt | 18.4% | 53.3% | 18.9% |
| b1460 | 1,574 B, 28.5 ms/pkt | 36.0% | 54.5% | **25.5%** |

| Target | Threshold | Detected at δ=0 | Median min budget | Never evaded |
|---|---|---|---|---|
| RF | 0.4535 | 7,510 | b512 | 64.0% |
| TCN | 0.9820 | 3,374 | **b8** | 45.5% |
| **Stack** | 0.3658 | 7,597 | b512 | **74.5%** |

The stack is the hardest target at every budget except b512, where it is level with the RF (18.9% vs 18.4%). TANTGen's own spend (~29–32 padding bytes per packet) falls between b32 and b128, where the stack's ASR is at most 8.5%.

The meta-learner relies more on the RF as the TCN is attacked: `rf_reliance` goes from 0.00 at δ=0 to 0.53–0.56 at every non-zero budget, with 5,900–6,500 of the 9,000 flows changing verdict.

Limitations:

- **The gradient does not help the attacker here.** Random search with the same number of queries beats PGD on the TCN at every budget (40.1% vs 37.2% at b8). The bit encoding of packet size makes the loss too irregular for the gradient direction to be useful.
- **Some sources cannot be measured.** At δ=0 the TCN detects almost nothing on `brute_force_*`, `ldap_ddos_*`, `ntp_ddos_*`, `ssdp_ddos_*` and others (some have `nan` ASR because no flow is detected), partly because their flows have a median of 2 packets. On these sources the stack follows the RF, including its misses on the reflection floods' `_size` variants.

Writes `gradient_ablation_metrics.json`, `evasion_cost.png` and the `sidecar/<source>.npz` cache to `data/models/gradient_ablation/`. Requires the trained RF, TCN and meta-learner, the TANTGen PCAPs, and `adversarial-robustness-toolbox` for the optional AutoPGD check.
