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
from hgq.layers import QUnaryFunctionLUT

test_path = Path(__file__).parent

seq_len = 4


def _make_model(dim):
    with QuantizerConfigScope(f0=3, i0=3):
        inp = keras.layers.Input((seq_len, dim))
        out = QUnaryFunctionLUT(
            'gelu', name='act', allow_heterogeneous_table=False, allow_heterogeneous_input=False
        )(inp)
        model = keras.Model(inp, out)
    return model


# The Resource-strategy io_stream unary_lut folds each beat over ReuseFactor cycles and
# reads the table from BRAM copies (one dual-port copy per two lookup lanes). It must stay
# bit-exact to HGQ2 like the Latency kernel. ReuseFactor covers fully parallel (1: one lane
# per element, many copies), a divisor of dim (4), a non-divisor that leaves the last fold
# partial (7), and one above dim (64), where each cycle looks up a single element.
@pytest.mark.parametrize('dim', [12, 16])
@pytest.mark.parametrize('reuse_factor', [1, 4, 7, 64])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Catapult'])
def test_hgq2_unary_lut_resource_bit_exact(test_case_id, backend, reuse_factor, dim):
    np.random.seed(0)
    model = _make_model(dim)
    data = np.random.normal(0, 2, size=(50, seq_len, dim)).astype(np.float32)

    r_keras = model.predict(data, verbose=0)

    output_dir = str(test_path / test_case_id)
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['LayerName']['act']['Strategy'] = 'Resource'
    config['LayerName']['act']['ReuseFactor'] = reuse_factor
    model_hls = convert_from_keras_model(
        model,
        output_dir=output_dir,
        io_type='io_stream',
        backend=backend,
        hls_config=config,
    )
    assert model_hls.graph['act'].class_name == 'UnaryLUT'
    assert model_hls.graph['act'].get_attr('strategy') == 'resource'
    model_hls.compile()
    r_hls = model_hls.predict(data).reshape(r_keras.shape)

    max_diff = np.max(np.abs(r_keras - r_hls))
    assert max_diff == 0, f'max|delta|={max_diff}, expected bit-exact'
