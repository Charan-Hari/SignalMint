"""Integer-quantization quality and bit-exact C parity (Phase 3)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from signalmint.config import DataConfig, ModelConfig, TrainConfig  # noqa: E402
from signalmint.data.cwru import get_synthetic_records  # noqa: E402
from signalmint.data.dataset import records_to_frames  # noqa: E402
from signalmint.quantize.int_infer import IntegerRuntime  # noqa: E402
from signalmint.quantize.parity import check_parity, find_c_compiler  # noqa: E402
from signalmint.quantize.quantizer import quantize_model  # noqa: E402
from signalmint.train import train_model  # noqa: E402


@pytest.fixture(scope="module")
def small_model():
    records = get_synthetic_records(n_normal=3, n_fault=0, n_samples=6000, seed=0)
    dc = DataConfig(frame_length=256, num_bins=16, hop_length=128)
    frames = records_to_frames(records, dc)
    mc = ModelConfig(
        num_bins=16, residual_channels=8, skip_channels=8, num_layers=4, dilation_cycle=4
    )
    res = train_model(frames, mc, TrainConfig(batch_size=16, epochs=3, learning_rate=3e-3, seed=0))
    return res.model.eval(), frames, dc


def test_integer_reference_matches_float(small_model) -> None:
    model, frames, dc = small_model
    qm = quantize_model(model)
    rt = IntegerRuntime(qm)
    frame = frames.symbols[0]
    with torch.no_grad():
        fl = model.forward(torch.from_numpy(frame[None, :].astype(np.int64)))[0].numpy()
    il = rt.logits(frame)
    agreement = float(np.mean(fl.argmax(1) == il.argmax(1)))
    assert agreement > 0.9


def test_quantized_model_shapes(small_model) -> None:
    model, frames, dc = small_model
    qm = quantize_model(model)
    assert qm.num_bins == 16
    assert len(qm.blocks) == 4
    # INT8 weights within range.
    for blk in qm.blocks:
        assert blk.conv.weight.dtype == np.int8
        assert blk.conv.weight.min() >= -127 and blk.conv.weight.max() <= 127


@pytest.mark.skipif(find_c_compiler() is None, reason="no C compiler available")
def test_c_runtime_bit_exact_parity(small_model) -> None:
    model, frames, dc = small_model
    rng = np.random.default_rng(0)
    symbols = rng.integers(0, dc.num_bins, size=96).astype(np.int64)
    result = check_parity(model, symbols)
    assert result.exact, (
        f"parity mismatch: {result.num_mismatch}/{result.num_values} values differ, "
        f"max abs diff {result.max_abs_diff} (compiler={result.compiler})"
    )
