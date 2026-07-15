"""Gradient ablation — DEPT's second adversary, priced across the ensemble (RQ1).

TANTGen, DEPT's training adversary, is black-box and gradient-free. This
package adds the complementary attacker RQ1 asks about: one that
differentiates the **TCN** directly and whose perturbation is then scored,
unchanged, against **RF**, **TCN** and the **RF+TCN+meta stack**.

The point of the comparison is the asymmetry. The attacker has a gradient
path into the TCN and none into the forest, so if the stack costs it more
budget than the TCN alone, heterogeneous stacking has measurably raised the
price of evasion — RQ1 answered by measurement rather than by design argument.

Extra-pipeline: not a phase of ``run_pipeline.sh``. See
``scripts/gradient_ablation/README.md``.
"""
