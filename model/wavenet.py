"""WaveNet-lite: a small dilated causal-convolution autoregressive model.

The model predicts a categorical distribution over the next signal symbol given
all previous symbols. Its per-sample probabilities are the shared substrate for
both project heads:

* anomaly detection -- score = negative log-likelihood of observed samples;
* compression       -- probabilities drive an entropy coder.

Design choices are dictated by the 1-8 MB / INT8 target tier: few channels, a
bounded receptive field (dilations reset every ``dilation_cycle`` layers), and
strictly causal convolutions so the same computation can run as a streaming,
memory-bounded pass on a microcontroller (Phase 3-4).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig

__all__ = ["CausalConv1d", "ResidualBlock", "WaveNetLite"]


class CausalConv1d(nn.Module):
    """1-D convolution that only sees current and past timesteps.

    Implemented as left-padding by ``(kernel_size - 1) * dilation`` so output
    length equals input length with no leakage from the future.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class ResidualBlock(nn.Module):
    """Gated-activation residual block (WaveNet-style) with a skip output."""

    def __init__(
        self,
        residual_channels: int,
        skip_channels: int,
        kernel_size: int,
        dilation: int,
    ):
        super().__init__()
        # Produce both filter and gate in one conv (2x channels), then split.
        self.conv = CausalConv1d(
            residual_channels, 2 * residual_channels, kernel_size, dilation
        )
        self.residual_proj = nn.Conv1d(residual_channels, residual_channels, 1)
        self.skip_proj = nn.Conv1d(residual_channels, skip_channels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x)
        filter_part, gate_part = h.chunk(2, dim=1)
        z = torch.tanh(filter_part) * torch.sigmoid(gate_part)
        residual = self.residual_proj(z) + x
        skip = self.skip_proj(z)
        return residual, skip


class WaveNetLite(nn.Module):
    """Small autoregressive next-symbol model over a categorical alphabet."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_bins = config.num_bins

        self.embed = nn.Embedding(config.num_bins, config.residual_channels)
        self.blocks = nn.ModuleList(
            ResidualBlock(
                config.residual_channels,
                config.skip_channels,
                config.kernel_size,
                dilation=2 ** (i % config.dilation_cycle),
            )
            for i in range(config.num_layers)
        )
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(config.skip_channels, config.skip_channels, 1),
            nn.ReLU(),
            nn.Conv1d(config.skip_channels, config.num_bins, 1),
        )

    @property
    def receptive_field(self) -> int:
        return self.config.receptive_field

    def forward(self, symbols: torch.Tensor) -> torch.Tensor:
        """Map ``(B, T)`` integer symbols to ``(B, T, num_bins)`` logits."""
        # (B, T) -> (B, C, T)
        x = self.embed(symbols).transpose(1, 2)
        skip_total = 0.0
        for block in self.blocks:
            x, skip = block(x)
            skip_total = skip_total + skip
        logits = self.head(skip_total)  # (B, num_bins, T)
        return logits.transpose(1, 2)  # (B, T, num_bins)

    def nll(
        self, inputs: torch.Tensor, targets: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        """Per-sample negative log-likelihood in **nats**.

        Args:
            inputs: ``(B, T)`` context symbols.
            targets: ``(B, T)`` next symbols to predict.
            reduction: ``"mean"``, ``"sum"`` or ``"none"`` (per-sample ``(B, T)``).
        """
        logits = self.forward(inputs)  # (B, T, V)
        loss = F.cross_entropy(
            logits.reshape(-1, self.num_bins),
            targets.reshape(-1),
            reduction="none",
        ).reshape(targets.shape)
        if reduction == "none":
            return loss
        if reduction == "sum":
            return loss.sum()
        return loss.mean()

    @torch.no_grad()
    def log_probs(self, symbols: torch.Tensor) -> torch.Tensor:
        """Return ``(B, T, num_bins)`` log-probabilities (natural log)."""
        return F.log_softmax(self.forward(symbols), dim=-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
