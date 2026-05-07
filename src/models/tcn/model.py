"""Temporal Convolutional Network for ADEPT.

Architecture follows Bai et al. 2018 ("An Empirical Evaluation of Generic
Convolutional and Recurrent Networks for Sequence Modeling"): stacked
dilated 1-D convolutions with weight normalisation, causal padding via
right-side chomp, and residual connections per block.

Input convention: DataLoader emits tensors with shape ``[B, T, F]`` to
match the on-disk ``sequence.npy`` layout (``T`` = window size, ``F`` =
1088 nPrint bits + IPAT + Direction). The forward pass permutes to the
``[B, F, T]`` channels-first layout that ``Conv1d`` expects.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class Chomp1d(nn.Module):
    """Strip ``chomp`` time-steps from the right of a ``[B, C, T]`` tensor.

    Combined with left-and-right padding of ``(k-1) * d``, this enforces
    causality (Bai et al. 2018 §3.1) — output at step t depends only on
    inputs at steps ≤ t.
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
    """Stack of ``TemporalBlock`` layers with mean+max pooling head.

    Args:
        num_features: per-packet feature count ``F`` (1090 for ADEPT).
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

        head_in = 2 * in_ch  # mean + max pooled
        self.head = nn.Linear(head_in, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] -> [B, F, T]
        x = x.transpose(1, 2).contiguous()
        h = self.tcn(x)                       # [B, C_last, T]
        pooled_mean = h.mean(dim=2)
            # [B, C_last]
        pooled_max = h.max(dim=2).values      # [B, C_last]
        pooled = torch.cat([pooled_mean, pooled_max], dim=1)
        return self.head(pooled)              # [B, num_classes]

    @property
    def receptive_field(self) -> int:
        """Symbolic receptive field for sanity checking against window size."""
        rf = 1
        for i in range(len(self.channels)):
            rf += 2 * (self.kernel_size - 1) * (2 ** i)
        return rf
