"""Causal residual denoiser for DEPT's input purification.

A small decoder-only (causal) transformer with the same layers as the
meta-learner, but with a residual head:

  Per-packet sequence  ``[B, T, 1090]``  (nPrint bits + IPAT + Direction)
    │
    ├─ Linear projection 1090 → d_model
    ├─ Sinusoidal positional encoding
    │
    ├─ N × pre-LN Transformer encoder layer  (causal self-attention)
    │
    └─ Linear head  d_model → 1090   →  residual Δ
                                          │
            reconstruction = input + Δ  ──┘

With the residual head the identity map is the default (Δ≈0), so the model
only has to learn the timing and length correction. Position ``t`` is
denoised from packets ``<= t`` only.
"""
from __future__ import annotations

import math

import torch
from torch import nn


RECONSTRUCTOR_FEATURE_DIM = 1090  # 1088 nPrint bits + IPAT (z-score) + Direction


class _SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (shared with the meta-learner)."""

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, L, d]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        T = x.size(1)
        if T > self.pe.size(1):
            # Lazy extension if a longer flow appears at inference.
            d = x.size(2)
            pe = torch.zeros(T, d, device=x.device, dtype=x.dtype)
            position = torch.arange(0, T, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d, 2, dtype=torch.float, device=x.device)
                * (-math.log(10000.0) / d)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return x + pe.unsqueeze(0)
        return x + self.pe[:, :T, :].to(dtype=x.dtype)


class _EncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer with no fast-path dispatching.

    Functionally identical to ``nn.TransformerEncoderLayer(norm_first=True,
    batch_first=True)`` but does NOT inherit from it, so
    ``nn.TransformerEncoder`` does not use the fused fast path (see
    ``metalearner/model.py``).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        # Pre-norm (norm_first=True) residual blocks.
        x = src
        x2 = self.norm1(x)
        x2, _ = self.self_attn(
            x2, x2, x2,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        x = x + self.dropout1(x2)
        x2 = self.norm2(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x2))))
        x = x + self.dropout2(x2)
        return x


class Reconstructor(nn.Module):
    """Causal residual denoiser for per-packet flow purification.

    Forward inputs:
      * ``seq`` — ``[B, T, F]`` float32 adversarial per-packet sequence
        (``F = 1090``: nPrint bits + IPAT + Direction).
      * ``key_padding_mask`` — ``[B, T]`` bool, ``True`` at padded positions
        (matching ``nn.MultiheadAttention``'s convention).

    Returns:
      * ``reconstruction`` — ``[B, T, F]`` purified sequence ``seq + Δ``.
        Causal: position ``t`` only depends on inputs ``≤ t``.
    """

    def __init__(
        self,
        num_features: int = RECONSTRUCTOR_FEATURE_DIM,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 200,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )
        self.num_features = num_features
        self.d_model = d_model

        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_enc = _SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)

        encoder_layer = _EncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, num_features)
        # Zero-initialised residual head: the untrained model is the identity.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        # Upper-triangular bool mask: True = position is masked (cannot attend).
        return torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1,
        )

    def forward(
        self,
        seq: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # seq: [B, T, F]
        if seq.dim() != 3 or seq.size(-1) != self.num_features:
            raise ValueError(
                f"expected seq of shape [B, T, {self.num_features}], "
                f"got {tuple(seq.shape)}"
            )

        x = self.input_proj(seq)                 # [B, T, d_model]
        x = self.pos_enc(x)
        T = x.size(1)
        causal = self._causal_mask(T, x.device)
        x = self.encoder(x, mask=causal, src_key_padding_mask=key_padding_mask)
        delta = self.head(x)                     # [B, T, F]
        return seq + delta


def num_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "Reconstructor",
    "RECONSTRUCTOR_FEATURE_DIM",
    "num_parameters",
]
