"""Diffusion purifier of Merzouk et al. (DBSec 2025), vendored for the phase-9 comparison.

Source: https://github.com/mamerzouk/adversarial-purification (``diffusion.py``, itself
based on https://github.com/dome272/Diffusion-Models-pytorch and the DDPM paper).
MIT License per the repository README; copyright the original authors.

Kept from the original:
  * the linear DDPM variance schedule (defaults T=1000, beta 1e-4 → 0.02);
  * the noise predictor: a fully connected network of ``n_hidden`` ReLU layers, with a
    sinusoidal timestep embedding (projected by SiLU + Linear) added to every hidden
    layer's output;
  * the objective: MSE between the true and the predicted noise at a random step;
  * purification with ``t`` steps: noise the input to step ``t-1`` in one shot, then run
    the reverse process from ``t-1`` down to 1 (the original's
    ``noise_data(x, t-1)`` followed by ``reconstruct(model, x_t, t)``).

Changed for DEPT's variable-length flows (each marked ``# DEPT:``):
  * the noise predictor also receives the flow's condition vector (packet directions +
    packet mask), concatenated to its input — both are fixed by realization;
  * the loss (see ``train.py``) is averaged over real packets only, and padded
    positions are held at their known value (zero, noised to the current step) during
    the reverse process.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from .overlay import OverlayNorm


class DiffusionSchedule:
    """Linear-variance DDPM schedule (the original ``Diffusion`` class)."""

    def __init__(self, noise_steps: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 0.02, device: torch.device | str = "cpu") -> None:
        self.noise_steps = int(noise_steps)
        self.beta = torch.linspace(beta_start, beta_end, self.noise_steps,
                                   device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def sample_timesteps(self, n: int, *, generator=None,
                         device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.randint(1, self.noise_steps, (n,), generator=generator,
                             device=device)

    def noise_data(self, x: torch.Tensor, t: torch.Tensor, *, generator=None):
        """``x_t = sqrt(a_hat_t) x + sqrt(1 - a_hat_t) eps`` → ``(x_t, eps)``."""
        a_hat = self.alpha_hat[t][:, None]
        eps = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        return torch.sqrt(a_hat) * x + torch.sqrt(1.0 - a_hat) * eps, eps


class NoiseMLP(nn.Module):
    """The original's MLP noise predictor (``l1`` … ``l10`` + output ``l11``)."""

    def __init__(self, data_dim: int, cond_dim: int = 0, hidden_dim: int = 960,
                 n_hidden: int = 10, emb_dim: int = 256) -> None:
        super().__init__()
        if emb_dim % 2:
            raise ValueError("emb_dim must be even")
        self.data_dim = int(data_dim)
        self.cond_dim = int(cond_dim)
        self.emb_dim = int(emb_dim)
        # DEPT: the condition vector is concatenated to the noisy input.
        self.inp = nn.Sequential(nn.Linear(self.data_dim + self.cond_dim, hidden_dim),
                                 nn.ReLU())
        self.hidden = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            for _ in range(int(n_hidden) - 1))
        self.out = nn.Linear(hidden_dim, self.data_dim)
        self.emb_layer = nn.Sequential(nn.SiLU(), nn.Linear(self.emb_dim, hidden_dim))

    def pos_encoding(self, t: torch.Tensor, channels: int) -> torch.Tensor:
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=t.device).float()
                                    / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        return torch.cat([pos_enc_a, pos_enc_b], dim=-1)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.emb_layer(self.pos_encoding(t.unsqueeze(-1).float(), self.emb_dim))
        h = torch.cat([x, cond], dim=-1) if self.cond_dim else x
        h = self.inp(h) + emb
        for layer in self.hidden:
            h = layer(h) + emb
        return self.out(h)


@torch.no_grad()
def purify(model: NoiseMLP, sched: DiffusionSchedule, x0: torch.Tensor,
           cond: torch.Tensor | None, valid: torch.Tensor, t: int, *,
           generator=None) -> torch.Tensor:
    """Purify ``x0 [B, D]`` with ``t`` diffusion steps; padded dims come back as zero."""
    if t <= 0:
        return x0.clone()
    model.eval()
    pad = ~valid
    B = x0.shape[0]

    def _randn():
        return torch.randn(x0.shape, generator=generator, device=x0.device,
                           dtype=x0.dtype)

    a_hat = sched.alpha_hat[t - 1]
    x = torch.sqrt(a_hat) * x0 + torch.sqrt(1.0 - a_hat) * _randn()
    for j in range(t - 1, 0, -1):
        ts = torch.full((B,), j, device=x0.device, dtype=torch.long)
        eps_hat = model(x, ts, cond)
        alpha, alpha_hat, beta = sched.alpha[j], sched.alpha_hat[j], sched.beta[j]
        noise = _randn() if j > 1 else torch.zeros_like(x)
        x = (1.0 / torch.sqrt(alpha)) * (
            x - ((1.0 - alpha) / torch.sqrt(1.0 - alpha_hat)) * eps_hat
        ) + torch.sqrt(beta) * noise
        # DEPT: padded dims are known (zero); keep them at their noised value.
        if j > 1:
            x = torch.where(pad, torch.sqrt(1.0 - sched.alpha_hat[j - 1]) * _randn(), x)
    return torch.where(pad, torch.zeros_like(x), x)


def purify_array(model: NoiseMLP, sched: DiffusionSchedule, x, cond, valid, t: int, *,
                 device, batch_size: int = 4096, seed: int = 0):
    """Batched :func:`purify` over encoded numpy arrays ``[n, 2T]`` → numpy ``[n, 2T]``."""
    import numpy as np

    out = np.array(x, dtype=np.float32, copy=True)
    if t <= 0 or len(out) == 0:
        return out
    g = torch.Generator(device=device).manual_seed(int(seed))
    use_cond = model.cond_dim > 0
    for s in range(0, len(out), int(batch_size)):
        e = min(s + int(batch_size), len(out))
        xb = torch.from_numpy(out[s:e]).to(device)
        cb = torch.from_numpy(np.ascontiguousarray(cond[s:e])).to(device) if use_cond else None
        vb = torch.from_numpy(np.ascontiguousarray(valid[s:e])).to(device)
        out[s:e] = purify(model, sched, xb, cb, vb, int(t), generator=g).cpu().numpy()
    return out


def purify_flows(model: NoiseMLP, sched: DiffusionSchedule, norm: OverlayNorm,
                 adv_overlays: list, directions: list, t: int, *, device,
                 batch_size: int = 4096, seed: int = 0) -> list[tuple]:
    """Adversarial ``(total_len, ipat_z)`` flows → purified overlays, ready to realize."""
    from .overlay import decode_flows, encode_flows

    T = model.data_dim // 2
    x, cond, valid = encode_flows(norm, adv_overlays, directions, T)
    xp = purify_array(model, sched, x, cond, valid, t, device=device,
                      batch_size=batch_size, seed=seed)
    return decode_flows(norm, xp, [len(o[0]) for o in adv_overlays])


def build_model(model_cfg: dict, T: int) -> NoiseMLP:
    condition = bool(model_cfg.get("condition", True))
    return NoiseMLP(data_dim=2 * T, cond_dim=2 * T if condition else 0,
                    hidden_dim=int(model_cfg.get("hidden_dim", 960)),
                    n_hidden=int(model_cfg.get("n_hidden", 10)),
                    emb_dim=int(model_cfg.get("emb_dim", 256)))


def build_schedule(model_cfg: dict, device) -> DiffusionSchedule:
    return DiffusionSchedule(noise_steps=int(model_cfg.get("noise_steps", 1000)),
                             beta_start=float(model_cfg.get("beta_start", 1e-4)),
                             beta_end=float(model_cfg.get("beta_end", 0.02)),
                             device=device)


def load_purifier(model_dir: Path, device) -> tuple[NoiseMLP, DiffusionSchedule,
                                                    OverlayNorm, dict]:
    """Trained purifier from ``diffusion.pt`` + ``diffusion_config.json``."""
    model_dir = Path(model_dir)
    saved = json.loads((model_dir / "diffusion_config.json").read_text())
    model = build_model(saved["model"], int(saved["max_length"])).to(device)
    model.load_state_dict(torch.load(model_dir / "diffusion.pt", map_location=device))
    model.eval()
    sched = build_schedule(saved["model"], device)
    return model, sched, OverlayNorm.from_dict(saved["norm"]), saved


__all__ = ["DiffusionSchedule", "NoiseMLP", "purify", "purify_array", "purify_flows",
           "build_model", "build_schedule", "load_purifier"]
