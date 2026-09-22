"""Assemble :class:`SignalRecord` objects into quantized symbol frames.

The output of this module is the categorical representation the autoregressive
model consumes: integer symbols in ``[0, num_bins)``. Everything here is NumPy;
the optional :class:`FrameDataset` wraps the arrays for PyTorch with a lazy import.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import DataConfig
from .cwru import SignalRecord
from .framing import frame_signal
from .quantize import quantize

__all__ = ["FrameSet", "records_to_frames", "FrameDataset"]


@dataclass
class FrameSet:
    """A batch of quantized frames with per-frame metadata.

    Attributes:
        symbols: ``(n_frames, frame_length)`` int64 symbols in ``[0, num_bins)``.
        labels: ``(n_frames,)`` int64 health labels (0 normal, 1 fault).
        names: Source record name for each frame.
    """

    symbols: np.ndarray
    labels: np.ndarray
    names: list[str]

    def __len__(self) -> int:
        return int(self.symbols.shape[0])

    def filter_label(self, label: int) -> "FrameSet":
        """Return a new FrameSet containing only frames with the given label."""
        mask = self.labels == label
        return FrameSet(
            symbols=self.symbols[mask],
            labels=self.labels[mask],
            names=[n for n, m in zip(self.names, mask) if m],
        )


def records_to_frames(records: list[SignalRecord], config: DataConfig) -> FrameSet:
    """Quantize and frame a list of records into a single :class:`FrameSet`."""
    all_symbols: list[np.ndarray] = []
    all_labels: list[int] = []
    all_names: list[str] = []
    for rec in records:
        symbols, _scale = quantize(
            rec.signal,
            num_bins=config.num_bins,
            quantizer=config.quantizer,
            mu=config.mu_law_mu,
            normalize=config.normalize,
        )
        frames = frame_signal(symbols, config.frame_length, config.hop_length)
        if frames.shape[0] == 0:
            continue
        all_symbols.append(frames)
        all_labels.extend([rec.label] * frames.shape[0])
        all_names.extend([rec.name] * frames.shape[0])

    if not all_symbols:
        return FrameSet(
            symbols=np.empty((0, config.frame_length), dtype=np.int64),
            labels=np.empty((0,), dtype=np.int64),
            names=[],
        )
    return FrameSet(
        symbols=np.concatenate(all_symbols, axis=0).astype(np.int64),
        labels=np.asarray(all_labels, dtype=np.int64),
        names=all_names,
    )


class FrameDataset:
    """PyTorch ``Dataset`` over a :class:`FrameSet` for next-symbol prediction.

    Each item yields ``(input, target)`` where ``target`` is ``input`` shifted by
    one sample (teacher forcing). Torch is imported lazily so importing this
    module does not require torch.
    """

    def __init__(self, frames: FrameSet):
        import torch  # lazy

        self._torch = torch
        self.symbols = torch.from_numpy(frames.symbols.astype(np.int64))
        self.labels = torch.from_numpy(frames.labels.astype(np.int64))

    def __len__(self) -> int:
        return int(self.symbols.shape[0])

    def __getitem__(self, idx: int):
        frame = self.symbols[idx]
        # Predict sample t+1 from samples <= t.
        return frame[:-1], frame[1:]
