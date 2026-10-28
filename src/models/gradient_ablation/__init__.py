"""Gradient-guided evasion study (RQ1).

TANTGen is black-box and does not use gradients. This package adds an
attacker that differentiates the TCN and then scores the same perturbation
against the RF, the TCN and the full stack. The attacker has a gradient path
into the TCN but not into the RF, so if the stack costs it more budget than
the TCN alone, the stacking has raised the cost of evasion.

Phase 7 of ``run_pipeline.sh``. See ``scripts/gradient_ablation/README.md``.
"""
