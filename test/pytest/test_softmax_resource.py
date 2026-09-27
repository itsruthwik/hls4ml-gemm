from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import Input, Softmax
from tensorflow.keras.models import Model

import hls4ml

test_root_path = Path(__file__).parent

rows = 6


def _convert(model, backend, strategy, reuse_factor, output_dir):
    cfg = hls4ml.utils.config_from_keras_model(model, granularity='name', default_precision='ap_fixed<16,6>')
    cfg['LayerName']['softmax']['Strategy'] = strategy
    cfg['LayerName']['softmax']['ReuseFactor'] = reuse_factor
    cfg['LayerName']['softmax']['Implementation'] = 'stable'
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=cfg, io_type='io_stream', backend=backend, output_dir=output_dir
    )
    hls_model.compile()
    return hls_model


# The Resource-strategy stream softmax folds each row over ReuseFactor cycles and pipelines
# rows through max | exp + sum | normalize stages; it must reproduce the stable kernel
# exactly. ReuseFactor covers fully parallel (1: one lane per element, exp table copies per
# two lanes), a non-divisor of the row (5), and one above the row (64: one element per cycle).
@pytest.mark.parametrize('n', [8, 12])
@pytest.mark.parametrize('reuse_factor', [1, 5, 64])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Catapult'])
def test_softmax_resource_matches_stable(test_case_id, backend, reuse_factor, n):
    inp = Input(shape=(rows, n), name='inp')
    model = Model(inp, Softmax(name='softmax')(inp))
    X = np.random.default_rng(0).uniform(-8, 8, size=(20, rows, n)).astype(np.float32)

    out_dir = test_root_path / test_case_id
    ref = _convert(model, backend, 'Latency', reuse_factor, str(out_dir / 'latency'))
    res = _convert(model, backend, 'Resource', reuse_factor, str(out_dir / 'resource'))
    assert res.graph['softmax'].get_attr('strategy') == 'resource'

    np.testing.assert_array_equal(res.predict(X), ref.predict(X))
