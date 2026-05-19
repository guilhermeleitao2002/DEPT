"""Causal Mini-Transformer that fuses RF + TCN per-packet streams.

Architecture (small by design — TCN already does feature extraction; the
meta-learner's only job is fusion):

  Token in (13 dims per packet)
    │
    ├─ NaN substitution: ``p_RF_logit`` slot replaced with the learnable
    │  ``rf_unknown_logit`` parameter for packets before the first RF
    │  checkpoint emits.
    │
    ├─ Linear projection 13 → d_model
    ├─ Sinusoidal positional encoding (Vaswani-style; generalises to
    │  flow lengths beyond ``max_seq_len``).
    │
    ├─ N × TransformerEncoderLayer
    │     • multi-head self-attention with strict causal mask
    │     • pre-LN (``norm_first=True``) for AMP stability
    │
    └─ Linear head  d_model → 2  (per-timestep binary logits)

The 13-dim token layout is owned by ``dataset.py``. The model only knows
that token slot 0 is the RF logit (potentially NaN) and the rest is
finite — see ``METALEARNER_TOKEN_DIM``.
"""
from __future__ import annotations

import math

import torch
from torch import nn


METALEARNER_TOKEN_DIM = 13
RF_LOGIT_SLOT = 0  # index of p_RF_logit within each token


class _SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

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

    Functionally identical to nn.TransformerEncoderLayer(norm_first=True,
    batch_first=True) but does NOT inherit from it, so
    nn.TransformerEncoder.forward() skips the isinstance check that would
    invoke torch._transformer_encoder_layer_fwd — the fused C++ kernel that
    emits mask-type warnings on older PyTorch regardless of mask dtype.
    Sub-module names are kept identical so the state-dict layout matches.
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



class MetaLearner(nn.Module):
    """Causal Mini-Transformer for per-packet stacking of RF + TCN.

    Forward inputs:
      * ``tokens`` — ``[B, T, 13]`` float32. Slot 0 (``p_RF_logit``) may
        be ``NaN`` for packets before the first RF checkpoint; the model
        substitutes ``rf_unknown_logit`` in those positions.
      * ``key_padding_mask`` — ``[B, T]`` bool, ``True`` at padded
        positions (matching ``nn.MultiheadAttention``'s convention).

    Returns:
      * ``logits`` — ``[B, T, num_classes]`` per-timestep classification
        logits. Causal: position ``t`` only depends on inputs ``≤ t``.
    """

    def __init__(
        self,
        token_dim: int = METALEARNER_TOKEN_DIM,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        num_classes: int = 2,
        max_seq_len: int = 200,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )
        self.token_dim = token_dim
        self.d_model = d_model
        self.num_classes = num_classes

        # Learnable scalar substituted in for NaN p_RF_logit slots.
        self.rf_unknown_logit = nn.Parameter(torch.zeros(()))

        self.input_proj = nn.Linear(token_dim, d_model)
        self.pos_enc = _SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)

        encoder_layer = _EncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, num_classes)

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        # Upper-triangular bool mask: True = position is masked (cannot attend).
        return torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # tokens: [B, T, 13]
        if tokens.dim() != 3 or tokens.size(-1) != self.token_dim:
            raise ValueError(
                f"expected tokens of shape [B, T, {self.token_dim}], "
                f"got {tuple(tokens.shape)}"
            )

        # Substitute NaN p_RF_logit slots with the learnable parameter. Done
        # in float to keep the parameter gradients well-defined under AMP.
        rf_slot = tokens[..., RF_LOGIT_SLOT]
        nan_mask = torch.isnan(rf_slot)
        if nan_mask.any():
            # `where` keeps gradients flowing through rf_unknown_logit.
            tokens = tokens.clone()
            tokens[..., RF_LOGIT_SLOT] = torch.where(
                nan_mask, self.rf_unknown_logit.to(tokens.dtype), rf_slot,
            )

        x = self.input_proj(tokens)             # [B, T, d_model]
        x = self.pos_enc(x)
        T = x.size(1)
        causal = self._causal_mask(T, x.device)
        # _EncoderLayer is not a subclass of nn.TransformerEncoderLayer, so
        # TransformerEncoder.forward() fails the isinstance check and takes the
        # plain Python loop — no torch._transformer_encoder_layer_fwd, no
        # mask-type warnings.
        x = self.encoder(x, mask=causal, src_key_padding_mask=key_padding_mask)
        return self.head(x)                      # [B, T, num_classes]


def num_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "MetaLearner",
    "METALEARNER_TOKEN_DIM",
    "RF_LOGIT_SLOT",
    "num_parameters",
]
