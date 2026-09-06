from pathlib import Path

import keras
import numpy as np
import pytest

import hls4ml
from hls4ml.converters import convert_from_keras_model

if keras.__version__ < '3.0.0':
    pytest.skip('This test requires keras 3.0.0 or higher', allow_module_level=True)

# HGQ2 ships as the `hgq2` distribution but imports as `hgq`. Skip rather than fail
# collection when it is absent, matching test_catapult_hgq2_attention.py.
pytest.importorskip('hgq')
pytest.importorskip('hgq.layers')

from hgq.config import QuantizerConfigScope
from hgq.layers import QLayerNormalization

test_path = Path(__file__).parent

seq_len = 4


def _make_model(dim):
    with QuantizerConfigScope(f0=4, i0=4):
        inp = keras.layers.Input((seq_len, dim))
        out = QLayerNormalization(name='ln')(inp)
        model = keras.Model(inp, out)
    return model


# dim is deliberately mixed power-of-2 (16) and non-power-of-2 (12, 24, 40): the LN kernel's
# mean/var divide-by-dim is a compile-time-constant fixed-point divide, not a reciprocal
# multiply, specifically so non-power-of-2 dims stay bit-exact against HGQ2's float Sum/dim
# (see jojo-track/archive/hls4ml-gemm-layernorm-consistency, "SIMPLIFIED" 2026-09-04 entry).
@pytest.mark.parametrize('dim', [12, 16, 24, 40])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_catapult_hgq2_layernorm_bit_exact(test_case_id, io_type, dim):
    np.random.seed(0)
    model = _make_model(dim)
    data = np.random.normal(0, 1, size=(50, seq_len, dim)).astype(np.float32)

    # Warm up the quantizers' calibrated ranges (EMA-based) before extracting the rsqrt LUT --
    # matches the pattern used to validate LN<->HGQ2 consistency (see
    # jojo-track/archive/hls4ml-gemm-layernorm-consistency).
    model.predict(data, verbose=0)

    r_keras = model.predict(data, verbose=0)

    output_dir = str(test_path / test_case_id)
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    model_hls = convert_from_keras_model(
        model,
        output_dir=output_dir,
        io_type=io_type,
        backend='Catapult',
        hls_config=config,
    )
    model_hls.compile()
    r_hls = model_hls.predict(data).reshape(r_keras.shape)

    max_diff = np.max(np.abs(r_keras - r_hls))
    assert max_diff == 0, f'max|delta|={max_diff}, expected bit-exact'
