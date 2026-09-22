"""Configuration dataclasses for the SignalMint pipeline.

These are plain, dependency-free dataclasses (no torch, no numpy) so they can be
imported anywhere -- including the lightweight runtime/eval paths -- and so the
Phase 0 test suite is meaningful without heavy dependencies.

The values chosen as defaults match the project's committed target tier:
1-8 MB RAM, INT8, 1-D signals, CWRU bearing vibration as the first dataset.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    """Signal framing and quantization settings.

    Attributes:
        dataset: Name of the dataset adapter to use (e.g. ``"cwru"``).
        data_dir: Directory where raw/cached dataset files live.
        sample_rate: Nominal sampling rate of the signal in Hz. CWRU drive-end
            data is commonly 12 kHz or 48 kHz.
        frame_length: Number of samples per training frame (the model's context
            window in samples).
        hop_length: Step between successive frames. Defaults to ``frame_length``
            (non-overlapping) when left as ``None``.
        num_bins: Quantization levels for the amplitude axis. INT8-friendly
            choices are 16, 32, 64, 128 or 256.
        quantizer: Amplitude quantization scheme. ``"mu_law"`` companding
            concentrates resolution near zero; ``"linear"`` is uniform.
        mu_law_mu: Companding parameter; only used when ``quantizer == "mu_law"``.
        normalize: Whether to per-frame normalize amplitude before quantizing.
    """

    dataset: str = "cwru"
    data_dir: str = "data"
    sample_rate: int = 12_000
    frame_length: int = 1_024
    hop_length: int | None = None
    num_bins: int = 256
    quantizer: str = "mu_law"
    mu_law_mu: float = 255.0
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.frame_length <= 0:
            raise ValueError("frame_length must be positive")
        if self.hop_length is None:
            self.hop_length = self.frame_length
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if self.num_bins < 2:
            raise ValueError("num_bins must be >= 2")
        if self.quantizer not in ("mu_law", "linear"):
            raise ValueError(f"unknown quantizer: {self.quantizer!r}")


@dataclass
class ModelConfig:
    """Streaming autoregressive model (WaveNet-lite dilated causal conv).

    Attributes:
        num_bins: Size of the categorical output distribution. Must match
            :attr:`DataConfig.num_bins`.
        residual_channels: Width of the residual/hidden path.
        skip_channels: Width of the skip-connection accumulator.
        kernel_size: Causal convolution kernel size (typically 2).
        num_layers: Total dilated-conv layers.
        dilation_cycle: Dilation doubles each layer and resets every
            ``dilation_cycle`` layers, bounding the receptive field.
    """

    num_bins: int = 256
    residual_channels: int = 32
    skip_channels: int = 64
    kernel_size: int = 2
    num_layers: int = 8
    dilation_cycle: int = 8

    def __post_init__(self) -> None:
        for name in ("residual_channels", "skip_channels", "num_layers"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.kernel_size < 2:
            raise ValueError("kernel_size must be >= 2 for a causal conv")
        if self.dilation_cycle <= 0:
            raise ValueError("dilation_cycle must be positive")

    @property
    def receptive_field(self) -> int:
        """Number of past samples the model can attend to."""
        rf = 1
        for layer in range(self.num_layers):
            dilation = 2 ** (layer % self.dilation_cycle)
            rf += (self.kernel_size - 1) * dilation
        return rf


@dataclass
class TrainConfig:
    """Training loop hyper-parameters."""

    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    val_fraction: float = 0.1
    seed: int = 0
    device: str = "auto"
    quantization_aware: bool = False
    checkpoint_dir: str = "artifacts"

    def __post_init__(self) -> None:
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in [0, 1)")
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")


@dataclass
class AnomalyConfig:
    """Anomaly-detection head settings.

    Attributes:
        score: Aggregation for per-sample NLL into a per-frame anomaly score.
        threshold_percentile: Percentile of the normal-data score distribution
            used to set the detection threshold.
        target_false_alarm: Target false-alarm rate for detection@FA reporting.
    """

    score: str = "mean_nll"
    threshold_percentile: float = 99.0
    target_false_alarm: float = 0.01

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold_percentile <= 100.0:
            raise ValueError("threshold_percentile must be in (0, 100]")
        if not 0.0 <= self.target_false_alarm <= 1.0:
            raise ValueError("target_false_alarm must be in [0, 1]")


@dataclass
class CompressConfig:
    """Neural-compression head settings.

    Attributes:
        coder: Entropy coder backend (``"arithmetic"`` or ``"rans"``).
        precision_bits: Fixed-point precision for the coder's cumulative
            frequency arithmetic.
    """

    coder: str = "arithmetic"
    precision_bits: int = 32

    def __post_init__(self) -> None:
        if self.coder not in ("arithmetic", "rans"):
            raise ValueError(f"unknown coder: {self.coder!r}")
        if not 8 <= self.precision_bits <= 62:
            raise ValueError("precision_bits must be in [8, 62]")


@dataclass
class QuantConfig:
    """INT8 export settings."""

    weight_bits: int = 8
    activation_bits: int = 8
    per_channel: bool = True
    symmetric: bool = True

    def __post_init__(self) -> None:
        if self.weight_bits not in (4, 8):
            raise ValueError("weight_bits must be 4 or 8")
        if self.activation_bits not in (8, 16):
            raise ValueError("activation_bits must be 8 or 16")


@dataclass
class SignalMintConfig:
    """Top-level configuration aggregating every stage of the pipeline."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    compress: CompressConfig = field(default_factory=CompressConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)

    def __post_init__(self) -> None:
        # Keep the categorical alphabet size consistent across data and model.
        if self.data.num_bins != self.model.num_bins:
            raise ValueError(
                "data.num_bins "
                f"({self.data.num_bins}) must equal model.num_bins "
                f"({self.model.num_bins})"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a nested plain-dict."""
        return asdict(self)

    def to_json(self, path: str | Path) -> Path:
        """Write the configuration to ``path`` as JSON and return the path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SignalMintConfig:
        """Rebuild a config from a nested plain-dict (inverse of :meth:`to_dict`)."""
        return cls(
            data=DataConfig(**payload.get("data", {})),
            model=ModelConfig(**payload.get("model", {})),
            train=TrainConfig(**payload.get("train", {})),
            anomaly=AnomalyConfig(**payload.get("anomaly", {})),
            compress=CompressConfig(**payload.get("compress", {})),
            quant=QuantConfig(**payload.get("quant", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> SignalMintConfig:
        """Load a config previously written with :meth:`to_json`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


def default_config() -> SignalMintConfig:
    """Return the committed default configuration for the CWRU target tier."""
    return SignalMintConfig()
