"""Reconstructor (Purification) Transformer — ADEPT's input-purification stage.

The reconstructor is the fourth stage of ADEPT's defense. When the ensemble
flags a flow as adversarial, the reconstructor "denoises" it: a small causal
(decoder-only) Transformer maps the adversarially-perturbed per-packet
sequence back onto the manifold of normal (un-manipulated) traffic, undoing
the timing (IPAT) and size (IPv4 length) perturbations TANTGen introduces.

The module operates in two spaces:

  * **Feature-space** — the model reads/writes the same ``[L, 1090]`` nPrint +
    IPAT + Direction tensor every other ADEPT model consumes. It is trained on
    matched ``(adversarial → clean)`` flow pairs (paired by 5-tuple) with a
    residual reconstruction objective.
  * **Traffic-space (Ψ)** — a realization step (:mod:`realize`) overlays the
    reconstructed length/timing back onto the *original* adversarial PCAP
    packets, yielding replayable purified packets whose protocol validity is
    checked with ``tshark``.

Sub-modules:

  * ``model``     — causal residual denoiser (Transformer encoder + Δ head)
  * ``dataset``   — paired (adversarial, clean) per-flow dataset
  * ``train``     — training loop on the seed=42 flow-level split
  * ``realize``   — nPrint → packet realization onto the original PCAPs
  * ``evaluate``  — reconstruction quality, purification gain (through the TCN),
                     and protocol-compliance rate
"""
