"""Mini-Transformer Meta-Learner over RF + TCN streaming predictions.

The meta-learner is the third stage of ADEPT's stacking ensemble. It
consumes the per-packet TCN stream and the per-checkpoint RF stream of
the same flow and emits a fused per-packet ``P(adversarial)`` prediction
via a small causal Transformer encoder.

Sub-modules:

  * ``model``          — causal Transformer encoder (binary, per-packet)
  * ``base_inference`` — load trained RF + TCN, produce joined per-flow
                          streaming predictions on a chosen split
  * ``dataset``        — token construction, on-disk cache, PyTorch Dataset
  * ``train``          — training loop on the val split (out-of-fold for
                          both base learners)
  * ``evaluate``       — inference on the test split + comparison plots

The meta-learner produces a ``StreamingPredictions`` object compatible
with ``src/models/common/streaming_eval.py`` so the same AUROC@t / EDP
machinery used for RF and TCN works unchanged.
"""
