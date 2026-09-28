"""GemmPackedStreams: GEMM IP edges kept packed across the graph (Vivado/Vitis io_stream).

With the knob on, an edge between a GEMM node and a GEMM or a neighbour that reads/writes
packed beats (im2col, relu, ...) carries the IP's raw ap_uint beat, and neither end runs a
separate pack/unpack process. The knob is off by default and must then change nothing; on,
the results must be bit-identical to off.
"""

import numpy as np
import pytest

import hls4ml


def _model():
    import keras

    rng = np.random.default_rng(3)
    inp = keras.layers.Input((8, 8, 3))
    x = keras.layers.Conv2D(4, 3, padding='valid', use_bias=True, name='conv')(inp)
    x = keras.layers.Activation('relu', name='conv_relu')(x)
    x = keras.layers.Reshape((36, 4), name='flat')(x)
    x = keras.layers.Dense(6, use_bias=True, name='dense')(x)
    x = keras.layers.Activation('relu', name='dense_relu')(x)
    out = keras.layers.Dense(5, use_bias=True, name='out')(x)
    model = keras.Model(inp, out)
    model.set_weights([rng.standard_normal(w.shape).astype(np.float32) * 0.3 for w in model.get_weights()])
    return model


def _convert(model, tmp_path, packed, backend='Vitis'):
    hls_model_cfg = {'Precision': 'ap_fixed<16,6>', 'ReuseFactor': 1, 'Strategy': 'Latency'}
    if packed is not None:
        hls_model_cfg['GemmPackedStreams'] = packed
    return hls4ml.converters.convert_from_keras_model(
        model,
        backend=backend,
        io_type='io_stream',
        output_dir=str(tmp_path / f'packed_{packed}_{backend}'),
        hls_config={
            'Model': hls_model_cfg,
            'LayerType': {'Conv2D': {'Strategy': 'GEMM'}, 'Dense': {'Strategy': 'GEMM'}},
        },
    )


def _packed_vars(hls_model):
    return {v.name for v in hls_model.output_vars.values() if getattr(v, 'gemm_packed', False)}


def _boundary(hls_model):
    return {v.name for v in hls_model.get_input_variables() + hls_model.get_output_variables()}


@pytest.mark.parametrize('packed', [None, False])
def test_off_marks_nothing(packed, tmp_path):
    hls_model = _convert(_model(), tmp_path, packed)
    hls_model.write()
    assert not _packed_vars(hls_model)
    cpp = (tmp_path / f'packed_{packed}_Vitis' / 'firmware' / 'myproject.cpp').read_text()
    assert 'nnet::packed<' not in cpp


def test_on_marks_gemm_edges_only(tmp_path):
    hls_model = _convert(_model(), tmp_path, True)
    packed = _packed_vars(hls_model)
    boundary = _boundary(hls_model)
    assert packed, 'expected packed GEMM edges'
    assert not (packed & boundary), 'model inputs/outputs must never be packed'
    producers = set()
    for node in hls_model.get_layers():
        for name in node.outputs:
            if node.get_output_variable(name).name in packed:
                producers.add(type(node).__name__)
    # im2col -> GEMM, GEMM -> relu and relu -> GEMM are packed on this chain.
    assert any(p.endswith('Im2Col') for p in producers)
    assert any(p.endswith('Gemm') for p in producers)
    assert any(p.endswith('Activation') for p in producers)


def test_on_bit_exact_vs_off(tmp_path):
    model = _model()
    x = np.random.default_rng(4).standard_normal((6, 8, 8, 3)).astype(np.float32) * 0.5
    y = {}
    for packed in (False, True):
        hls_model = _convert(model, tmp_path, packed)
        hls_model.compile()
        y[packed] = hls_model.predict(x)
    np.testing.assert_array_equal(y[True], y[False])
    cpp = (tmp_path / 'packed_True_Vitis' / 'firmware' / 'myproject.cpp').read_text()
    assert 'nnet::packed<' in cpp
    # Only the model's own input keeps a pack process and its output an unpack process.
    assert cpp.count('nnet::pack_stream<') <= 1
    assert cpp.count('nnet::unpack_stream<') <= 1
