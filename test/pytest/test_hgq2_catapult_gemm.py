import keras
import numpy as np
import pytest
from hls4ml.converters import convert_from_keras_model
import hls4ml
from pathlib import Path

if keras.__version__ < '3.0.0':
    pytest.skip('This test requires keras 3.0.0 or higher', allow_module_level=True)

# HGQ2 ships as the `hgq2` distribution but imports as `hgq`. Skip rather than fail
# collection when it is absent, matching test_catapult_hgq2_attention.py.
pytest.importorskip('hgq')
pytest.importorskip('hgq.layers')

from hgq.config import QuantizerConfigScope
from hgq.layers import QDense, QConv1D, QConv2D

test_path = Path(__file__).parent

def _test_model_accuracy(model, data, test_case_id, gemm_ip=True, gemm_exclude=()):
    # Keras prediction
    r_keras = model.predict(data)

    # hls4ml conversion
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['Model']['Strategy'] = 'GEMM' if gemm_ip else 'Latency'

    # For Catapult GEMM IP, we often need to ensure the strategy is set to GEMM
    # and IOType is io_stream. Layers in gemm_exclude stay on the stock path — the
    # GEMM-IP requires one beat to carry the full K-row (beat width == gemm_k), so a
    # Dense fed a multi-beat activation (e.g. after Flatten) cannot use it.
    for layer in config['LayerName']:
        # Heuristic to find Dense/Conv layers
        if any(x in layer for x in ['dense', 'conv']):
             # An excluded layer must actively shadow the Model-level GEMM default.
             config['LayerName'][layer]['Strategy'] = 'GEMM' if layer not in gemm_exclude else 'Latency'

    output_dir = str(test_path / test_case_id)
    model_hls = convert_from_keras_model(
        model,
        output_dir=output_dir,
        io_type='io_stream',
        backend='Catapult',
        hls_config=config
    )

    model_hls.compile()
    r_hls = model_hls.predict(data).reshape(r_keras.shape)

    # HGQ2 models with bit_exact=True (default) should be bit-accurate
    np.testing.assert_array_equal(r_hls, r_keras)

@pytest.mark.parametrize('gemm_ip', [True])
def test_hgq2_dense_gemm(test_case_id, gemm_ip):
    with QuantizerConfigScope(f0=4, i0=4):
        inp = keras.layers.Input((10,))
        out = QDense(5, name='dense')(inp)
        model = keras.Model(inp, out)
    
    # Initialize weights implicitly by built model
    data = np.random.uniform(-1, 1, (1, 10)).astype(np.float32)
    
    _test_model_accuracy(model, data, test_case_id, gemm_ip=gemm_ip)

@pytest.mark.parametrize('gemm_ip', [True])
def test_hgq2_conv1d_gemm(test_case_id, gemm_ip):
    with QuantizerConfigScope(f0=4, i0=4):
        inp = keras.layers.Input((8, 4))
        out = QConv1D(6, 3, name='conv1d')(inp)
        model = keras.Model(inp, out)
    
    data = np.random.uniform(-1, 1, (1, 8, 4)).astype(np.float32)
    
    _test_model_accuracy(model, data, test_case_id, gemm_ip=gemm_ip)

@pytest.mark.parametrize('gemm_ip', [True])
def test_hgq2_conv2d_gemm(test_case_id, gemm_ip):
    with QuantizerConfigScope(f0=4, i0=4):
        inp = keras.layers.Input((8, 8, 3))
        out = QConv2D(4, (3, 3), name='conv2d')(inp)
        model = keras.Model(inp, out)
    
    data = np.random.uniform(-1, 1, (1, 8, 8, 3)).astype(np.float32)
    
    _test_model_accuracy(model, data, test_case_id, gemm_ip=gemm_ip)

@pytest.mark.parametrize('gemm_ip', [True])
def test_hgq2_pointwise_conv2d_gemm(test_case_id, gemm_ip):
    with QuantizerConfigScope(f0=4, i0=4):
        inp = keras.layers.Input((8, 8, 3))
        out = QConv2D(4, (1, 1), name='conv2d_pw')(inp)
        model = keras.Model(inp, out)
    
    data = np.random.uniform(-1, 1, (1, 8, 8, 3)).astype(np.float32)
    
    _test_model_accuracy(model, data, test_case_id, gemm_ip=gemm_ip)

def test_hgq2_full_model_gemm(test_case_id):
    with QuantizerConfigScope(f0=4, i0=4):
        inp = keras.layers.Input((10, 10, 1))
        x = QConv2D(4, (3, 3), name='conv2d_1')(inp)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Flatten()(x)
        out = QDense(10, name='dense_1')(x)
        model = keras.Model(inp, out)
    
    data = np.random.uniform(-1, 1, (1, 10, 10, 1)).astype(np.float32)

    # dense_1 follows Flatten, so it is fed a multi-beat activation and cannot use
    # the GEMM-IP (beat width != gemm_k); keep it on the stock io_stream Dense.
    _test_model_accuracy(model, data, test_case_id, gemm_ip=True, gemm_exclude={'dense_1'})
