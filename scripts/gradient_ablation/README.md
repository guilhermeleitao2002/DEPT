# Gradient Ablation — RQ1

> Part of [DEPT](../../README.md).

**Phase 7 of `run_pipeline.sh`** (global mode only — skipped in `--per-attack`
runs, since it evaluates the global models). A comparison study, like the
deployment baseline ([`scripts/baseline/`](../baseline/README.md), phase 8);
it can also be run standalone:

```bash
python3 scripts/gradient_ablation/run_gradient_ablation.py                    # all sources
python3 scripts/gradient_ablation/run_gradient_ablation.py --attack mirai     # one attack
python3 scripts/gradient_ablation/run_gradient_ablation.py \
    --flows-per-source 24 --budgets b0,b32,b128                               # smoke run
```

## What it answers

TANTGen, the adversary DEPT is trained on, is black-box and gradient-free. That
makes it the right attacker for training and for RQ2 — and the wrong one for **RQ1**:

> Does heterogeneous stacking, a non-differentiable Random Forest fused with a
> differentiable Temporal Convolutional Network, measurably raise the cost of a
> *transferable, gradient-guided* evasion attack relative to a single-architecture
> baseline?

This study supplies the missing attacker. One perturbation is crafted against the
**TCN** and then scored, unchanged, against all three targets: **RF**, **TCN**,
and the **RF+TCN+meta stack**. If the stack costs the attacker more budget than
the TCN alone, RQ1 is answered by measurement rather than by design argument.

## Threat model

**White-box on the TCN; no gradient path through the forest.** No surrogate is
needed: the TCN *is* the single-architecture baseline, and it is a component of
the stack. So the attacker knows the differentiable half exactly and cannot
differentiate the other half — which is precisely the asymmetry RQ1 is about. The
transfer measured is TCN → RF and TCN → stack.

The perturbation is **δ ≥ 0** — padding bytes and delay seconds, per packet, on
top of TANTGen's own. Cost is therefore the *extra* budget over TANTGen.

**Increase-only** is a real constraint, not a convenience: the packets are already
on the wire, so the attacker may pad and delay but never un-send. It is also what
keeps the problem non-trivial. DEPT's positive class *is* "TANTGen-perturbed", so
an attacker permitted to *reduce* the perturbation would evade by simply not
perturbing — at zero cost, and at the price of losing the NIDS evasion that
motivated the perturbation in the first place. Empirically the constraint matches
TANTGen itself: every negative delta in its output is exactly one float32 ULP
(−3.071e-06 s), i.e. round-off, not signal.

## Attack

Budgets are in **physical units** (bytes, seconds), never an ε-ball on the tensor.
This is forced, not stylistic: the IPAT channel is z-scored with **std = 211047.5 s**
while the 1088 nPrint channels are bits in {−1, 0, 1}, so a single `eps=0.01` would
buy the attacker **±2110 seconds** of delay and nothing at all in the bit channels.

The perturbable surface is exactly two of the 1090 channels — the IPv4
total-length bits `[16:32]` and the IPAT channel `[1088]` — which are also the only
two [`realize.py`](../../src/models/reconstructor/realize.py) can write back to the
wire, and the same two TANTGen manipulates.

| Piece | Choice |
|---|---|
| Optimiser | Hand-rolled PGD (Madry et al.) with restarts, plus a query-matched `random` control |
| Reported attacker | `best` = PGD ∪ random, per flow |
| Size gradient | STE on the **scalar** round; bits via the **pseudoinverse of the decode map** |
| Feasible set | increase-only, integer bytes, clamp `[orig_tl, 1500]`, zero on non-IPv4 packets |
| Strength check | ART `AutoProjectedGradientDescent` on the relaxed problem |

**Why hand-rolled.** The feasible set above is a projection operator; ART's `mask`
kwarg is a binary include/exclude and cannot express it. ART keeps the job it does
well — a parameter-free third-party check that this search is not simply
under-tuned.

**Why not an STE over the 16 bits.** Identity-backward would hand bit 0 (place
value 32768) and bit 15 (place value 1) the same gradient, and `tl` 255 → 256 flips
nine bits at once, so an L∞ step in bit space corresponds to no physical quantity.
Instead, `_EncodeTotalLength` backpropagates through the minimum-norm Jacobian
consistent with the decode (`d bits_j / d tl = w_j / ||w||²`, `w_j = 2^(15−j)`),
which projects the bit-gradient onto the decode direction — "how does the loss move
if the *value* moves".

## The sidecar

Scoring the RF on a *perturbed* flow means recomputing its 24 prefix statistics
from perturbed per-packet values — and those values are not on disk.
`tensor_builder` builds them, feeds them to `_flow_checkpoint_stats`, and persists
**only the aggregate**. The sequence tensor does not rescue us: nPrint covers
headers only, so `Payload_Length` and `Payload_Entropy` — together **45.7% of the
trained RF's decision mass** — are absent at any price. The aggregate cannot be
inverted (four numbers per ten packets; prefix min/max non-invertible in principle).

So one streaming scapy pass per source re-derives the per-packet RF base features
and caches them under `sidecar/`. It is the only I/O-heavy step, and it is cached.

`verify_sidecar` recomputes the checkpoints at δ=0 and compares them against the
source's own `rf_checkpoints.npy`; a mismatch aborts the run. Measured on
`mirai_adv_size`: **max_abs_diff = 0.0** over 40 flows / 324 checkpoint rows — the
pass reproduces the tensorizer exactly, not merely within tolerance.

Byte histograms are stored per packet so `Payload_Entropy` is exact under **both**
padding contents. That matters: the attacker sees no payload bytes through the
TCN's gradient, so it cannot tell zero-padding (entropy falls) from random padding
(entropy rises) — yet the two move the RF's largest feature group in opposite
directions. Both are reported rather than one assumed (`attack.pad_byte_mode`).

## Metrics

**ASR is conditioned on the flows each target detects at δ=0** — the standard
robust-accuracy convention (attack success measured on initially-correct inputs).
Here it is the difference between a number and an artifact: the TCN's reported
ROC-AUC is **0.494** on `mirai_adv_size` and **0.232** on `mirai_adv_iat` — at
chance, or below. An unconditional ASR would start near 75% before the attack did
anything and would mostly be reporting flows the detector never caught.
`detection_rate_clean` is reported alongside, because a target that detects nothing
cannot be evaded.

**Thresholds are matched-FPR, never a raw 0.5.** The meta-learner early-stopped at
`best_epoch: 1`; comparing at a fixed 0.5 would credit it for rarely saying
"adversarial", which is undertraining rather than robustness. Each target's
threshold is set at a common false-positive rate on unperturbed benign flows.

**The curve is the monotone envelope.** A flow counts as evaded at budget *b* if it
evaded at any budget ≤ *b*. This follows from the feasible sets being nested — an
attacker allowed 1460 B may choose to spend 128 — and is needed because measured
per-budget PGD does not inherit it: the step size scales with the budget and the
loss surface over packet size is genuinely jagged (`ssdp_flood_adv_size` measured
87.5% at 128 B, **4.2%** at 512 B, 97.9% at 1460 B without it).

| Metric | Meaning |
|---|---|
| `asr` | evaded / detected-at-δ=0, per target, per budget |
| `min_budget` | median smallest grid budget at which a flow evades |
| `spend` | realised cost — padding bytes and delay ms actually spent |
| `reliance_shift` | does the meta lean on the RF as the TCN degrades? |
| `attack_strength_check` | ART AutoPGD vs this module's search |
| `tantgen_budget` | TANTGen's own per-source spend, for context |

`reliance_shift` ablates the meta's two base-learner token slots: with `d_x` the
distance the verdict moves when slot *x* is handed back its clean value, a large
`d_tcn` means the stack was still listening to the TCN, so
`rf_reliance = d_rf / (d_rf + d_tcn)`. It is the falsifiable form of DEPT's design
claim.

`tantgen_budget` is context the marginal cost needs: TANTGen's own spend varies by
two orders of magnitude across sources — `mirai_adv_size` perturbs 81% of packets
by +23.6 B and `portscan_adv_size` 63% by +63 B, but `dns_ddos_adv_size` only 0.3%
by +0.09 B.

## Results

<!-- TODO: populate from data/models/gradient_ablation/gradient_ablation_metrics.json -->

Not yet run at full scale. The smoke configuration (`--attack brute_force`,
24 flows/source) is reported in the thesis; see `data/models/gradient_ablation/`.

## Observations

**Only some sources can answer RQ1 at all.** A target that already misses a flow
cannot be evaded, and the TCN's δ=0 detection rate collapses to 0.000 on
`brute_force_adv_both/iat`, `ddos_adv_iat`, `dns_ddos_*`, `ssdp_ddos_*` and
`ssdp_flood_adv_iat`. Flow length compounds it: `ldap_ddos`, `portscan`,
`ssdp_ddos` and `syn_ddos` have a **median of 2 packets** against a TCN whose
receptive field is 29.

**The stack already rides the RF before any attack.** At δ=0, where the TCN is
blind and the forest is not, the stack follows the forest: `ddos_adv_iat`,
`portscan_adv_iat` and `ssdp_flood_adv_iat` all read RF 1.000 / TCN 0.000 /
**stack 1.000**. It is not universal — on `ssdp_ddos_*` (RF 1.000, TCN 0.000) the
stack collapses to **0.021**, discarding the RF's signal.

**Gradients buy the attacker little here.** Query-matched random search matches or
beats PGD on the TCN, and restarts dominate step count (35% → 60% ASR at 128 B
going from 2 restarts to 8). The nPrint bit-encoding of packet size makes the loss
jagged enough that descent direction carries little information — which is why the
reported attacker is `best` = PGD ∪ random, and why the `random` control is not a
formality. ART's AutoPGD agrees the search is not under-tuned (100% vs 100%).

**Some flows are unreachable through this surface at any budget.** The TCN's margin
on `syn_ddos_adv_size` is **91–161 logits**; padding and delay move it around but
never near the boundary, giving 0% ASR even at 1460 B/packet. Size and timing
cannot overcome that much confidence.

**Timing below ~3.07 µs is unrepresentable.** That is one float32 ULP of the
z-scored IPAT channel, and `dns_ddos`'s median IAT (2.38 µs) sits *below* it — a
representational limit of the tensor, not an architectural property of the models.

## Output files (`data/models/gradient_ablation/`)

| File | Description |
|---|---|
| `gradient_ablation_metrics.json` | All metrics: ASR curves, min budget, spend, reliance shift, thresholds, attack-strength check, TANTGen budget |
| `evasion_cost.png` | ASR vs realised attacker spend, one line per target |
| `sidecar/<source>.npz` | Cached per-packet RF base features + byte histograms (verified against `rf_checkpoints.npy` at δ=0) |

## Requires

Trained RF + TCN + meta-learner under `data/models/`, the TANTGen PCAPs under
`data/adversarial/` (the sidecar re-derives payload features from them), and
`adversarial-robustness-toolbox` for the optional AutoPGD check.
