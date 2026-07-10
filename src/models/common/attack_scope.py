"""Per-attack scoping shared by the DEPT stage runners.

The default ("global") pipeline trains one encompassing model per stage on
all 10 attacks at once. Every stage runner also accepts ``--attack NAME``,
which restricts the run to a single attack: the benign pool collapses to
that attack's clean source folder and the adversarial pool to its
``<attack>_adv_{both,iat,size}`` variants. This module owns the two pieces
of plumbing every runner needs for that:

  * ``scoped_adversarial_exclude`` — extend a config's exclude list so the
    auto-discovery in the train/eval code only keeps the chosen attack's
    adversarial sources;
  * ``per_attack_models_root`` — the canonical artifact home for per-attack
    runs, ``data/models/per_attack/<attack>/{rf,tcn,metalearner,
    reconstructor,latency}``. The global artifacts stay untouched at
    ``data/models/<stage>``.
"""
from __future__ import annotations

from pathlib import Path

from .dataset import discover_adversarial_sources

PER_ATTACK_DIRNAME = "per_attack"


def base_attack(source: str) -> str:
    """Base attack of an adversarial source folder (``X_adv_mode`` → ``X``)."""
    return source.rsplit("_adv_", 1)[0] if "_adv_" in source else source


def scoped_adversarial_exclude(
    adversarial_dir: Path, attack: str, exclude: list[str] | tuple = (),
) -> list[str]:
    """Extend ``exclude`` so only ``attack``'s adversarial variants survive
    the auto-discovery under ``adversarial_dir``.

    Raises ``ValueError`` when the attack has no adversarial source on disk
    (typo, or the Packet-to-Tensor phase has not produced its tensors yet).
    """
    discovered = discover_adversarial_sources(Path(adversarial_dir))
    kept = [s for s in discovered if base_attack(s) == attack]
    if not kept:
        known = sorted({base_attack(s) for s in discovered})
        raise ValueError(
            f"no adversarial tensors for attack '{attack}' under "
            f"{adversarial_dir} (known attacks: {', '.join(known) or 'none'})")
    return sorted(set(exclude) | {s for s in discovered if base_attack(s) != attack})


def per_attack_models_root(repo_root: Path, attack: str) -> Path:
    """Artifact root for one attack's models: data/models/per_attack/<attack>."""
    return Path(repo_root) / "data" / "models" / PER_ATTACK_DIRNAME / attack


def list_trained_attacks(repo_root: Path) -> list[str]:
    """Attacks with a per-attack artifact directory (any stage) on disk."""
    root = Path(repo_root) / "data" / "models" / PER_ATTACK_DIRNAME
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir())
