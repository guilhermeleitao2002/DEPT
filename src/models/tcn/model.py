"""Temporal Convolutional Network (Bai et al., 2018).

Stacked dilated causal 1-D convolutions with weight normalisation and
residual connections. The model emits one logit pair per packet, and the
output at packet ``t`` depends only on packets ``<= t``.

Input ``[B, T, F]`` (F = 1088 nPrint bits + IPAT + Direction), output
``[B, T, num_classes]``.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
try:
    from torch.nn.utils.parametrizations import weight_norm  # PyTorch >= 1.13
except ImportError:
    from torch.nn.utils import weight_norm  # type: ignore[no-redef]


class Chomp1d(nn.Module):
    """Strip ``chomp`` time-steps from the right of a ``[B, C, T]`` tensor.

    Together with a padding of ``(k-1) * d`` this makes the convolution
    causal.
    """

    def __init__(self, chomp: int) -> None:
        super().__init__()
        self.chomp = int(chomp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp == 0:
            return x
        return x[..., : -self.chomp]


class TemporalBlock(nn.Module):
    """Two stacked weight-norm dilated convs with residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=pad, dilation=dilation))
        self.chomp1 = Chomp1d(pad)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      padding=pad, dilation=dilation))
        self.chomp2 = Chomp1d(pad)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample: nn.Module
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.downsample = nn.Identity()
        self.relu_out = nn.ReLU()

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        if isinstance(self.downsample, nn.Conv1d):
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.drop2(self.relu2(self.chomp2(self.conv2(out))))
        return self.relu_out(out + self.downsample(x))


class TCN(nn.Module):
    """Stack of ``TemporalBlock`` layers with a per-timestep classification head.

    The logits at position ``t`` are the verdict after the first ``t + 1``
    packets of the flow.

    Args:
        num_features: per-packet feature count ``F`` (1090 for DEPT).
        channels: output channels per block. Length = depth. Dilation at
            block i is ``2 ** i``.
        kernel_size: shared kernel width.
        dropout: dropout probability inside each block.
        num_classes: head output dimension (2 for binary).
    """

    def __init__(
        self,
        num_features: int = 1090,
        channels: Sequence[int] = (128, 128, 128),
        kernel_size: int = 3,
        dropout: float = 0.15,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.channels = list(channels)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)
        self.num_classes = int(num_classes)

        blocks: list[nn.Module] = []
        in_ch = self.num_features
        for i, out_ch in enumerate(self.channels):
            blocks.append(TemporalBlock(
                in_channels=in_ch,
                out_channels=int(out_ch),
                kernel_size=self.kernel_size,
                dilation=2 ** i,
                dropout=self.dropout,
            ))
            in_ch = int(out_ch)
        self.tcn = nn.Sequential(*blocks)

        # Per-timestep head as a 1x1 conv.
        self.head = nn.Conv1d(in_ch, self.num_classes, kernel_size=1)
        nn.init.kaiming_normal_(self.head.weight, nonlinearity="linear")
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] -> [B, F, T] for Conv1d
        x = x.transpose(1, 2).contiguous()
        h = self.tcn(x)                          # [B, C_last, T]
        logits = self.head(h)                    # [B, num_classes, T]
        return logits.transpose(1, 2).contiguous()  # [B, T, num_classes]

    @property
    def receptive_field(self) -> int:
        """Symbolic receptive field for sanity checking against flow length."""
        rf = 1
        for i in range(len(self.channels)):
            rf += 2 * (self.kernel_size - 1) * (2 ** i)
        return rf
