"""Catapult GEMM-IP codegen tests for the Keras 2 QMultiHeadAttention handler.

Mirrors test_catapult_gemm.py: exercises project generation (convert + write +
artifact checks) plus a numeric csim check. Keras-2-venv-only: the fork's
`QMultiHeadAttention` (hls4ml.utils.qkeras_attention) needs Keras 2 + QKeras, so
this whole module skips cleanly under Keras 3 (e.g. the HGQ2 venv). Stock Keras
MultiHeadAttention is intentionally out of scope (user decision 2026-09-06); see
test_catapult_hgq2_attention.py for the native Keras 3 / HGQ2 path.
"""

import os
from pathlib import Path

import keras
import numpy as np
import pytest

if int(keras.__version__.split('.')[0]) != 2:
    pytest.skip('QMultiHeadAttention needs the Keras 2 + QKeras venv (.venv-keras2)', allow_module_level=True)

try:
    import qkeras  # noqa: F401
except ImportError:
    pytest.skip('qkeras is not installed in this venv', allow_module_level=True)

from keras.layers import Input
from keras.models import Model

import hls4ml
import hls4ml.converters
import hls4ml.utils
from hls4ml.utils.qkeras_attention import QMultiHeadAttention

# csim (compile()) needs Catapult's libstdc++ on LD_LIBRARY_PATH; conftest.py sets it
# from atlas.env and re-execs pytest. Gate the numeric test on that being in effect so
# it skips cleanly on a standalone hls4ml checkout (no atlas.env).
_HAS_CSIM = 'catapult' in os.environ.get('LD_LIBRARY_PATH', '').lower()


def _mha_model(num_heads=2, key_dim=8, seq=6, d=16, cross=False, weight_bits=8):
    q = Input(shape=(seq, d), name='q')
    v = Input(shape=(seq, d), name='v')
    inputs = [q, v]
    layer = QMultiHeadAttention(num_heads=num_heads, key_dim=key_dim, weight_bits=weight_bits, name='mha')
    if cross:
        # Equal key/value sequence length (decoder cross-attention with a matching
        # encoder length); the handler supports differing Q vs K/V lengths but this
        # keeps the test focused on kwarg resolution, not shape bookkeeping.
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
    """3-input (q, v, k) cross-attention (equal K/V seq length) resolves value/key
    from call-kwargs."""
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
    v = Input(shape=(5, 16), name='v')
    k = Input(shape=(5, 16), name='k')
    layer = QMultiHeadAttention(2, 8, weight_bits=8, name='mha')
    model = Model([q, v, k], layer(query=q, value=v, key=k))
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
    model = _mha_model(num_heads=2, key_dim=8, seq=6, d=16, weight_bits=8)
    xq = np.random.randn(20, 6, 16).astype('float32')
    xv = np.random.randn(20, 6, 16).astype('float32')
    y_keras = model.predict([xq, xv], verbose=0)

    cfg = _config(model, strategy)
    cfg['Model']['Precision'] = 'ap_fixed<32,12>'  # isolate decomposition, not quantization
    # Catapult's io_parallel softmax is the ac_math piecewise-linear kernel, which
    # static_asserts if its input has too many integer bits (the internal exp table
    # width blows up exponentially with them -- see softmax-numerics). The QK^T
    # scores here are small (bounded dot products of near-unit-scale Q/K), so a
    # handful of integer bits is exact; only the softmax's own input needs
    # narrowing, not the wide Model-wide precision used for everything else.
    cfg['LayerName']['mha_qk']['Precision']['result'] = 'ap_fixed<16,6>'
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
    layer = QMultiHeadAttention(2, 8, weight_bits=8, attention_axes=(1, 2), name='m4')
    model = Model([q, v], layer(q, v))
    with pytest.raises(AssertionError):
        hls4ml.converters.convert_from_keras_model(
            model, hls_config=_config(model), output_dir=str(tmp_path / 'g'),
            backend='Catapult', io_type='io_parallel',
        )
