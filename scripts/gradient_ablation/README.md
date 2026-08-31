# Gradient Ablation — RQ1

> Part of [DEPT](../../README.md) — **phase 7** of `run_pipeline.sh` (global mode only).

```bash
python3 scripts/gradient_ablation/run_gradient_ablation.py                 # all sources
python3 scripts/gradient_ablation/run_gradient_ablation.py --attack mirai
```

## What it answers

TANTGen is black-box and gradient-free — the right attacker for training and RQ2, the wrong one for **RQ1**:

> Does heterogeneous stacking — a non-differentiable Random Forest fused with a differentiable TCN — measurably raise the cost of a transferable, gradient-guided evasion attack relative to a single-architecture baseline?

This supplies the missing attacker. One perturbation is crafted **white-box against the TCN** and then scored, unchanged, against three targets: RF, TCN, and the RF+TCN+meta stack. No surrogate is needed — the TCN *is* the single-architecture baseline and a component of the stack, so the attacker knows the differentiable half exactly and cannot differentiate the other. That asymmetry is what RQ1 is about.

**Design choices that make the numbers mean something:**

- **Increase-only δ ≥ 0** (pad bytes, delay seconds, on top of TANTGen's). The packets are already on the wire, so an attacker may pad and delay but never un-send — and since DEPT's positive class *is* "TANTGen-perturbed", an attacker allowed to *reduce* perturbation would evade at zero cost by simply not attacking.
- **Budgets in physical units, never an ε-ball.** IPAT is z-scored with std = 211,047.5 s while the nPrint channels are bits, so one `eps=0.01` would buy ±2110 s of delay and nothing in the bit channels. The perturbable surface is exactly the two channels `realize.py` can write back to the wire and TANTGen manipulates: IPv4 total-length bits `[16:32]` and IPAT `[1088]`.
- **Matched-FPR thresholds, never a raw 0.5.** The targets are calibrated very differently (median benign verdict: RF 0.043, TCN 0.379, stack 0.002), so a fixed 0.5 would credit whichever says "adversarial" least often. **ASR is conditioned on the flows each target detects at δ=0**, the standard robust-accuracy convention.
- **The ASR curve is the monotone envelope** — feasible sets are nested, and measured per-budget PGD does not inherit monotonicity (the loss over packet size is genuinely jagged: `ssdp_flood_adv_size` measured 87.5% at 128 B but 4.2% at 512 B).
- Attack is **hand-rolled PGD with restarts plus a query-matched `random` control** (reported attacker = `best` = PGD ∪ random); ART's `AutoProjectedGradientDescent` runs on the relaxed problem as a parameter-free check that the search is not under-tuned. The size gradient uses an STE on the scalar round and the pseudoinverse of the decode map, so a step means "how does the loss move if the *value* moves".
- **A sidecar is required** because scoring the RF on a perturbed flow needs its 24 prefix statistics recomputed from perturbed per-packet values, which are not on disk — `Payload_Length` and `Payload_Entropy` together carry 45.7% of the forest's decision mass and nPrint covers headers only. One cached streaming scapy pass per source re-derives them; `verify_sidecar` reproduces `rf_checkpoints.npy` at δ=0 **exactly** (max_abs_diff 0.0 on `mirai_adv_size`).

## Results

9,000 flows (300 × 30 sources), 162,000 scored records, 6 budgets, 4.3 h. Reported at `fpr_0.05` with the `best` attacker and the monotone envelope.

| Budget | median spend / flow | RF | TCN | **Stack** |
|---|---|---|---|---|
| b8 | 13 B, 0.004 ms/pkt | 8.6% | 40.5% | **1.7%** |
| b32 | 47 B, 0.041 ms/pkt | 10.7% | 50.1% | **4.7%** |
| b128 | 167 B, 0.419 ms/pkt | 13.1% | 52.2% | **8.5%** |
| b512 | 537 B, 3.68 ms/pkt | 18.4% | 53.3% | 18.9% |
| b1460 | 1,574 B, 28.5 ms/pkt | 36.0% | 54.5% | **25.5%** |

| Target | threshold | detected at δ=0 | median min budget | never evaded |
|---|---|---|---|---|
| RF | 0.4535 | 7,510 | b512 | 64.0% |
| TCN | 0.9820 | 3,374 | **b8** | 45.5% |
| **Stack** | 0.3658 | 7,597 | b512 | **74.5%** |

**RQ1 is answered affirmatively, with one qualification.** The stack is the hardest target at every budget an attacker can plausibly afford, and TANTGen's own spend (~29–32 pad-bytes/packet) sits in the b32–b128 region where the stack concedes ≤ 8.5%. The qualification is the crossover at b512, where the stack (18.9%) briefly matches the RF (18.4%) before pulling ahead again at b1460.

**The meta-learner does lean on the RF as the TCN degrades** — `rf_reliance` moves from 0.00 at δ=0 to 0.53–0.56 across every non-zero budget, with 5,900–6,500 of 9,000 flows registering a shifted verdict. This is the falsifiable form of DEPT's design claim, and it holds.

Two honest caveats. **The gradient buys the attacker nothing here**: query-matched random search *beats* PGD on the TCN at every budget (40.1% vs 37.2% at b8), because the nPrint bit-encoding of packet size makes the loss too jagged for descent direction to carry information — so the stack's advantage is measured against a strong-but-not-gradient-dependent attacker. And **only some sources can answer RQ1 at all**: at δ=0 the TCN detects essentially nothing on `brute_force_*`, `ldap_ddos_*`, `ntp_ddos_*`, `ssdp_ddos_*` and others (several have `nan` ASR from an empty denominator), partly because those sources have a **median of 2 packets** against a 29-packet receptive field. The stack usually rides the RF where the TCN is blind, and inherits the RF's reflection-flood `_size` failures too.

Writes `gradient_ablation_metrics.json`, `evasion_cost.png` and the verified `sidecar/<source>.npz` cache to `data/models/gradient_ablation/`. Requires trained RF + TCN + meta-learner, the TANTGen PCAPs, and `adversarial-robustness-toolbox` for the optional AutoPGD check.
