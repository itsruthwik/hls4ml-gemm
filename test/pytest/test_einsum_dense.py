from pathlib import Path

import keras
import numpy as np
import pytest

from hls4ml.converters import convert_from_keras_model

if keras.__version__ < '3.0.0':
    pytest.skip('Only keras v3 is supported for now', allow_module_level=True)

from keras.layers import EinsumDense, Input

test_root_path = Path(__file__).parent


@pytest.mark.parametrize('strategy', ['latency', 'resource'])
@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis', 'Catapult'])
@pytest.mark.parametrize(
    'operation',
    [
        # eq, inp, out
        ('bi,j->bij', (8,), (8, 7), None),
        ('bi,j->bij', (8,), (8, 7), 'i'),
        ('bi,j->bij', (8,), (8, 7), 'j'),
        ('bi,io->bo', (8,), 7, None),
        ('...i,oi->...o', (4, 3), (5,), None),
        ('...abcd,bcde->...aeb', (5, 4, 3, 2), (5, 6, 4), None),
        ('...abcd,bcde->...aeb', (5, 4, 3, 2), (5, 6, 4), 'aeb'),
        ('...abcd,bcde->...aeb', (5, 4, 3, 2), (5, 6, 4), 'ab'),
        ('...abcd,bcde->...aeb', (5, 4, 3, 2), (5, 6, 4), 'a'),
        ('bqd,dk->bqk', (8, 4), (8, 5), None),
        # n_inplace > 1: one kernel slice per 'a', so a row must read its own slice
        ('...abc,acd->...abd', (3, 4, 5), (3, 4, 6), None),
    ],
)
def test_einsum_dense(test_case_id, backend, io_type, strategy, operation):
    if backend == 'Catapult' and (strategy != 'resource' or io_type != 'io_stream'):
        pytest.skip('Catapult only covers strategy=resource, io_type=io_stream')

    eq, inp_shape, out_shape, bias_axes = operation
    model = keras.Sequential(
        [Input(inp_shape), EinsumDense(eq, output_shape=out_shape, bias_axes=bias_axes, name='einsum_dense')]
    )

    if bias_axes is not None:
        layer = model.get_layer('einsum_dense')
        layer.bias.assign(keras.ops.convert_to_tensor(np.random.rand(*layer.bias.shape)))

    data = np.random.rand(1000, *inp_shape)
    output_dir = str(test_root_path / test_case_id)
    reuse_factor = 4 if strategy == 'resource' else 1
    hls_config = {'Model': {'Precision': 'ap_fixed<32,8>', 'ReuseFactor': reuse_factor, 'Strategy': strategy}}
    model_hls = convert_from_keras_model(
        model, backend=backend, output_dir=output_dir, hls_config=hls_config, io_type=io_type
    )

    model_hls.compile()
    r_keras = model.predict(data, verbose=0, batch_size=1000)  # type: ignore
    r_hls = model_hls.predict(data).reshape(r_keras.shape)  # type: ignore

    np.testing.assert_allclose(r_hls, r_keras, atol=2e-6, rtol=0)
