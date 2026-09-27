from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import Add, Dense, Input, ReLU
from tensorflow.keras.models import Model

import hls4ml

test_root_path = Path(__file__).parent

seq = 6


def _model():
    inp = Input(shape=(seq, 4), name='inp')
    x = Dense(4, name='dense1')(inp)
    x = ReLU(name='relu')(x)
    x = Dense(4, name='dense2')(x)
    out = Add(name='resid')([x, inp])
    return Model(inp, out)


def _stream_depths(hls_model):
    return {
        var.name: var.pragma[1]
        for var in hls_model.output_vars.values()
        if 'StreamVariable' in type(var).__name__ and var.pragma
    }


# Model.FifoDepth replaces the whole-tensor default depth of the inter-layer io_stream FIFOs;
# LayerName.<name>.InputFifoDepth still overrides single edges on top of it.
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
def test_model_fifo_depth(test_case_id, backend):
    model = _model()
    X = np.random.default_rng(0).uniform(-1, 1, size=(10, seq, 4)).astype(np.float32)

    def convert(model_cfg_extra, layer_cfg, sub):
        cfg = hls4ml.utils.config_from_keras_model(model, granularity='name', default_precision='ap_fixed<16,6>')
        cfg['Model'].update(model_cfg_extra)
        for name, extra in layer_cfg.items():
            cfg['LayerName'][name].update(extra)
        hls_model = hls4ml.converters.convert_from_keras_model(
            model, hls_config=cfg, io_type='io_stream', backend=backend, output_dir=str(test_root_path / test_case_id / sub)
        )
        hls_model.compile()
        return hls_model

    default = convert({}, {}, 'default')
    assert set(_stream_depths(default).values()) == {seq}  # one beat per token: the whole tensor

    shallow = convert({'FifoDepth': 2}, {'resid': {'InputFifoDepth': {1: 5}}}, 'shallow')
    skip = shallow.graph['resid'].get_input_variable(shallow.graph['resid'].inputs[1])
    depths = _stream_depths(shallow)
    assert depths.pop(skip.name) == 5
    assert set(depths.values()) == {2}

    np.testing.assert_array_equal(shallow.predict(X), default.predict(X))
