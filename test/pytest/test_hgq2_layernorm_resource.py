from pathlib import Path

import keras
import numpy as np
import pytest

import hls4ml
from hls4ml.converters import convert_from_keras_model

if keras.__version__ < '3.0.0':
    pytest.skip('This test requires keras 3.0.0 or higher', allow_module_level=True)

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


# The Resource-strategy io_stream LayerNorm folds each token over ReuseFactor cycles and
# gathers sum(x) and sum(x^2) in one pass, recovering the two-pass variance by exact
# expansion. It must stay bit-exact to HGQ2 like the Latency kernel. dim mixes power-of-2
# and non-power-of-2 sizes (the divide-by-dim path); ReuseFactor covers fully parallel (1),
# a divisor of some dims (4), a non-divisor that leaves the last fold partial (7), and one
# above dim (64), where each cycle handles a single element.
@pytest.mark.parametrize('dim', [12, 16, 40])
@pytest.mark.parametrize('reuse_factor', [1, 4, 7, 64])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Catapult'])
def test_hgq2_layernorm_resource_bit_exact(test_case_id, backend, reuse_factor, dim):
    np.random.seed(0)
    model = _make_model(dim)
    data = np.random.normal(0, 1, size=(50, seq_len, dim)).astype(np.float32)

    # Warm up the quantizers' calibrated ranges before extracting the rsqrt LUT.
    model.predict(data, verbose=0)

    r_keras = model.predict(data, verbose=0)

    output_dir = str(test_path / test_case_id)
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['LayerName']['ln']['Strategy'] = 'Resource'
    config['LayerName']['ln']['ReuseFactor'] = reuse_factor
    model_hls = convert_from_keras_model(
        model,
        output_dir=output_dir,
        io_type='io_stream',
        backend=backend,
        hls_config=config,
    )
    assert model_hls.graph['ln'].get_attr('strategy') == 'resource'
    model_hls.compile()
    r_hls = model_hls.predict(data).reshape(r_keras.shape)

    max_diff = np.max(np.abs(r_keras - r_hls))
    assert max_diff == 0, f'max|delta|={max_diff}, expected bit-exact'
