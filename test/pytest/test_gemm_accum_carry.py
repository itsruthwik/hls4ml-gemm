"""A lowered Gemm accumulates in the type the bit-exact pass sized for the layer it replaces.

The bit-exact pass runs before the GEMM lowering and sizes a Dense's accumulator from its
quantized operands. The Gemm that stands in for the Dense must keep that accumulator rather
than fall back to the backend default: the GEMM kernels honour it, and the default overflows
on a model whose operands need more integer bits.
"""
import numpy as np
import pytest
import keras

from hls4ml.converters import convert_from_keras_model

pytest.importorskip('hgq')
from hgq.layers import QDense  # noqa: E402
from hgq.quantizer.config import QuantizerConfig  # noqa: E402


def _model():
    keras.utils.set_random_seed(3)
    iq = QuantizerConfig('kbi', 'datalane', k0=True, b0=7, i0=3, trainable=False)
    kq = QuantizerConfig('kbi', 'weight', k0=True, b0=7, i0=0, overflow_mode='WRAP', round_mode='RND', trainable=False)
    inp = keras.layers.Input((64,), name='x')
    out = QDense(16, name='fc', iq_conf=iq, kq_conf=kq, enable_oq=False)(inp)
    model = keras.Model(inp, out)
    model(np.zeros((1, 64), dtype=np.float32))
    rng = np.random.default_rng(3)
    model.get_layer('fc')._kernel.assign(rng.integers(-128, 128, size=(64, 16)).astype(np.float32) / 128.0)
    return model


def _accum(backend, strategy, tmp_path):
    hls_model = convert_from_keras_model(
        _model(), backend=backend, io_type='io_stream', output_dir=str(tmp_path / f'{backend}_{strategy}'),
        bit_exact=True, hls_config={'Model': {'Precision': 'auto', 'ReuseFactor': 1, 'Strategy': strategy}},
    )
    name = 'gemm_fc' if strategy == 'GEMM' else 'fc'
    return str(hls_model.graph[name].types['accum_t'].precision)


@pytest.mark.parametrize('backend', ['Vitis', 'Catapult'])
def test_gemm_keeps_the_bit_exact_accumulator(backend, tmp_path):
    base = _accum(backend, 'Resource', tmp_path)
    gemm = _accum(backend, 'GEMM', tmp_path)
    assert gemm == base
    # The sized accumulator is wider than the 16-bit, 6-integer-bit backend default.
    assert '16,6' not in gemm
