import os
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Conv2D, Dense

import hls4ml
import hls4ml.converters
import hls4ml.utils

# csim (compile()) needs Catapult's libstdc++ on LD_LIBRARY_PATH; conftest.py sets it
# from atlas.env and re-execs pytest. Gate the numeric test on that being in effect so
# it skips cleanly on a standalone hls4ml checkout (no atlas.env).
_HAS_CSIM = 'catapult' in os.environ.get('LD_LIBRARY_PATH', '').lower()


def _model(layer_type):
    if layer_type == 'dense':
        return tf.keras.Sequential([Dense(2, input_shape=(4,), name='dense')])
    elif layer_type == 'conv1d':
        return tf.keras.Sequential([Conv1D(4, 3, input_shape=(10, 2), name='conv1d')])
    else:  # conv2d
        return tf.keras.Sequential([Conv2D(4, (3, 3), input_shape=(10, 10, 2), name='conv2d')])


def _gemm_config(model, precision=None):
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['IOType'] = 'io_stream'
    if precision is not None:
        config['Model']['Precision'] = precision
    for layer in config['LayerName']:
        config['LayerName'][layer]['Strategy'] = 'Resource'
        if 'conv' in layer or 'dense' in layer:
            config['LayerName'][layer]['Strategy'] = 'GEMM'
    return config


@pytest.mark.parametrize('layer_type', ['dense', 'conv1d', 'conv2d'])
def test_catapult_gemm_codegen(layer_type, tmp_path):
    """Smoke test for Catapult GEMM-IP project generation: projects write and
    gemm_config.json is generated."""
    model = _model(layer_type)
    output_dir = str(tmp_path / f'catapult_gemm_{layer_type}_prj')

    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=_gemm_config(model), output_dir=output_dir,
        backend='Catapult', part='xcku115-flvb2104-2-i', io_type='io_stream',
    )
    hls_model.write()

    prj_path = Path(output_dir)
    assert (prj_path / 'build_prj.tcl').exists()
    assert (prj_path / 'gemm_config.json').exists()
    assert (prj_path / 'firmware' / 'myproject.cpp').exists()
    assert (prj_path / 'firmware' / 'nnet_utils' / 'nnet_gemm_ip.h').exists()


@pytest.mark.skipif(not _HAS_CSIM, reason='Catapult csim env (ATLAS_LDLIB / atlas.env) not available')
@pytest.mark.parametrize('layer_type', ['dense', 'conv1d', 'conv2d'])
def test_catapult_gemm_csim(layer_type, tmp_path):
    """csim of the GEMM-IP behavioural model matches keras (io_stream, Strategy: GEMM).

    Uses the C++ behavioural GEMM core in nnet_gemm_ip.h -- no generated IP needed
    (that is only for synthesis/cosim)."""
    np.random.seed(0)
    model = _model(layer_type)
    x = np.random.rand(10, *model.input_shape[1:]).astype('float32')
    y_keras = model.predict(x, verbose=0)

    output_dir = str(tmp_path / f'catapult_gemm_csim_{layer_type}_prj')
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=_gemm_config(model, precision='ap_fixed<32,12>'),
        output_dir=output_dir, backend='Catapult', part='xcku115-flvb2104-2-i',
        io_type='io_stream',
    )
    hls_model.compile()
    y_hls = np.asarray(hls_model.predict(np.ascontiguousarray(x))).reshape(y_keras.shape)
    max_err = np.abs(y_hls - y_keras).max()
    assert max_err < 1e-2, f'{layer_type}: max_err={max_err:.3e}'
