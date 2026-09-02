"""Catapult GEMM-IP codegen tests for the Keras 2 MultiHeadAttention handler.

Mirrors test_catapult_gemm.py: exercises project generation (convert + write +
artifact checks) without compile(), so it stays free of environment-specific
GLIBCXX dependencies. Uses stock keras MultiHeadAttention so the test is
independent of the ATLAS QMultiHeadAttention layer; the handler treats both the
same (QMultiHeadAttention just additionally carries quantizer bit-widths).
"""

import os
from pathlib import Path

import keras
import numpy as np
import pytest
from keras.layers import Input
from keras.models import Model

import hls4ml
import hls4ml.converters
import hls4ml.utils

# csim (compile()) needs Catapult's libstdc++ on LD_LIBRARY_PATH; conftest.py sets it
# from atlas.env and re-execs pytest. Gate the numeric test on that being in effect so
# it skips cleanly on a standalone hls4ml checkout (no atlas.env).
_HAS_CSIM = 'catapult' in os.environ.get('LD_LIBRARY_PATH', '').lower()


def _mha_model(num_heads=2, key_dim=8, seq=6, d=16, cross=False):
    q = Input(shape=(seq, d), name='q')
    v = Input(shape=(seq, d), name='v')
    inputs = [q, v]
    layer = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, name='mha')
    if cross:
        k = Input(shape=(seq, d), name='k')
        inputs = [q, v, k]
        out = layer(q, v, k)
    else:
        out = layer(q, v)
    return Model(inputs, out)


def _config(model, gemm=None):
    # `gemm` is the GEMM-IP toggle (the parametrize passes 'GEMM' as a truthy label);
    # it selects Strategy: GEMM, the mutually-exclusive GEMM strategy value.
    cfg = hls4ml.utils.config_from_keras_model(model, backend='Catapult', granularity='name')
    if gemm:
        cfg.setdefault('Model', {})['Strategy'] = 'GEMM'
    return cfg


@pytest.mark.parametrize(
    'io_type,strategy',
    [('io_parallel', None), ('io_stream', 'GEMM'), ('io_parallel', 'GEMM')],
)
def test_mha_codegen(io_type, strategy, tmp_path):
    """MHA lowers and a project is written on both the baseline and GEMM paths."""
    model = _mha_model()
    out_dir = str(tmp_path / f'mha_{io_type}_{strategy}')
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=_config(model, strategy), output_dir=out_dir,
        backend='Catapult', io_type=io_type, part='xcku115-flvb2104-2-i',
    )
    classes = [layer.class_name for layer in hls_model.graph.values()]
    # decomposition primitives must be present regardless of backend
    assert 'Softmax' in classes
    assert classes.count('EinsumDense') + classes.count('Gemm') >= 4  # Q/K/V/O projections

    hls_model.write()
    prj = Path(out_dir)
    assert (prj / 'firmware' / 'myproject.cpp').exists()
    if strategy == 'GEMM':
        assert (prj / 'gemm_config.json').exists()


def test_mha_cross_attention(tmp_path):
    """3-input (q, v, k) cross-attention resolves value/key from call-kwargs."""
    model = _mha_model(cross=True)
    out_dir = str(tmp_path / 'mha_cross')
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=_config(model, 'GEMM'), output_dir=out_dir,
        backend='Catapult', io_type='io_stream', part='xcku115-flvb2104-2-i',
    )
    hls_model.write()
    assert (Path(out_dir) / 'gemm_config.json').exists()


def test_mha_all_kwargs(tmp_path):
    """query/value/key passed as keyword args resolve correctly (by arg name)."""
    q = Input(shape=(5, 16), name='q')
    v = Input(shape=(7, 16), name='v')
    k = Input(shape=(7, 16), name='k')
    model = Model([q, v, k], keras.layers.MultiHeadAttention(2, 8, name='mha')(query=q, value=v, key=k))
    out_dir = str(tmp_path / 'mha_kwargs')
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=_config(model, 'GEMM'), output_dir=out_dir,
        backend='Catapult', io_type='io_stream', part='xcku115-flvb2104-2-i',
    )
    hls_model.write()
    assert (Path(out_dir) / 'gemm_config.json').exists()


@pytest.mark.skipif(not _HAS_CSIM, reason='Catapult csim env (ATLAS_LDLIB / atlas.env) not available')
@pytest.mark.parametrize('io_type,strategy', [('io_parallel', None), ('io_stream', 'GEMM')])
def test_mha_csim_numeric(io_type, strategy, tmp_path):
    """csim of the decomposed MHA matches keras within softmax-table tolerance,
    on both the baseline and GEMM backends."""
    np.random.seed(0)
    model = _mha_model(num_heads=2, key_dim=8, seq=6, d=16)
    xq = np.random.randn(20, 6, 16).astype('float32')
    xv = np.random.randn(20, 6, 16).astype('float32')
    y_keras = model.predict([xq, xv], verbose=0)

    cfg = _config(model, strategy)
    cfg['Model']['Precision'] = 'ap_fixed<32,12>'  # isolate decomposition, not quantization
    out_dir = str(tmp_path / f'mha_csim_{io_type}')
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=cfg, output_dir=out_dir, backend='Catapult',
        io_type=io_type, part='xcku115-flvb2104-2-i',
    )
    hls_model.compile()
    y_hls = np.asarray(
        hls_model.predict([np.ascontiguousarray(xq), np.ascontiguousarray(xv)])
    ).reshape(y_keras.shape)
    max_err = np.abs(y_hls - y_keras).max()
    # tolerance covers hls4ml's softmax lookup-table approximation
    assert max_err < 0.3, f'{io_type}/{strategy}: max_err={max_err:.3e}'


def test_mha_rank_guard(tmp_path):
    """Rank-4 / multi-axis attention is rejected loudly, not silently mishandled."""
    q = Input(shape=(4, 6, 16), name='q')
    v = Input(shape=(4, 6, 16), name='v')
    model = Model([q, v], keras.layers.MultiHeadAttention(2, 8, attention_axes=(1, 2), name='m4')(q, v))
    with pytest.raises(AssertionError):
        hls4ml.converters.convert_from_keras_model(
            model, hls_config=_config(model), output_dir=str(tmp_path / 'g'),
            backend='Catapult', io_type='io_parallel',
        )
