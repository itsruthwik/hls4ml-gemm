import numpy as np
import keras
import pytest

from hls4ml.converters import convert_from_keras_model

pytest.importorskip('hgq')
pytest.importorskip('hgq.layers')

from hgq.config import QuantizerConfigScope
from hgq.layers import QMultiHeadAttention


@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
def test_vivado_hgq2_attention_multihead_gemm_ip(backend, io_type, tmp_path):
    """Multi-head attention on the GEMM path for Vivado/Vitis.

    SplitAttentionHeads rewrites the cluster into H per-head lanes: projections stay
    2D [seq, d_model]; a stateless HeadSplit fans each into H [seq, key_dim] streams;
    each head runs its own QK^T Gemm, Softmax and A.V Gemm; a stateless HeadMerge
    concatenates the contexts back. With 2 heads there are 4 projections + 2 QK^T +
    2 A.V = 8 Gemm nodes. Verified bit-exact on both io types (io_stream relies on the
    ported HGQ io_stream quantizer).
    """
    keras.utils.set_random_seed(7)
    with QuantizerConfigScope(f0=3, i0=2):
        q = keras.layers.Input((4, 8), name='q')
        v = keras.layers.Input((4, 8), name='v')
        k = keras.layers.Input((4, 8), name='k')
        out = QMultiHeadAttention(2, 4, name='hgq_mha', fuse='none')(q, v, k)
        model = keras.Model([q, v, k], out)

    out_dir = str(tmp_path / f'mha_{backend}_{io_type}')
    hls_model = convert_from_keras_model(
        model,
        backend=backend,
        io_type=io_type,
        output_dir=out_dir,
        hls_config={'Model': {'Precision': 'ap_fixed<16,6>', 'ReuseFactor': 1, 'Strategy': 'GEMM'}},
    )

    classes = [type(n).__name__ for n in hls_model.graph.values()]
    assert any('HeadSplit' in c for c in classes), 'SplitAttentionHeads should produce HeadSplit lanes'
    assert any('HeadMerge' in c for c in classes)
    assert sum(c.endswith('Gemm') for c in classes) == 8, 'expected 8 Gemm nodes (4 proj + 2 QK^T + 2 A.V)'

    hls_model.compile()
    rng = np.random.default_rng(6789)
    data = [rng.normal(0, 0.5, size=(4, 4, 8)).astype(np.float32) for _ in range(3)]
    keras_pred = model.predict(data, verbose=0)
    assert np.std(keras_pred) > 1e-3, 'degenerate (near-constant) reference output'
    hls_pred = hls_model.predict(data).reshape(keras_pred.shape)
    np.testing.assert_allclose(hls_pred, keras_pred, atol=0.35, rtol=0.0)

    top = (tmp_path / f'mha_{backend}_{io_type}' / 'firmware' / 'myproject.cpp').read_text()
    assert 'nnet::split_lanes' in top, 'expected the stateless per-head lane split'
    assert 'nnet::merge_lanes' in top
    assert 'nnet::einsum' not in top, 'no baseline einsum should leak through the GEMM path'


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
def test_vivado_hgq2_attention_second_operand_row_major(backend, tmp_path):
    """SecondOperandRowMajor=True streams the two-operand B operand row-major (N-wide beats,
    one contraction row per beat) instead of the default col-major (K-wide beats). Required to
    route QK^T / A.V to the mvau GEMM IP. Must stay bit-exact and mark the two-operand configs
    b_row_major=true (io_stream)."""
    keras.utils.set_random_seed(7)
    with QuantizerConfigScope(f0=3, i0=2):
        q = keras.layers.Input((4, 8), name='q')
        v = keras.layers.Input((4, 8), name='v')
        k = keras.layers.Input((4, 8), name='k')
        out = QMultiHeadAttention(2, 4, name='hgq_mha', fuse='none')(q, v, k)
        model = keras.Model([q, v, k], out)

    out_dir = str(tmp_path / f'mha_rm_{backend}')
    hls_model = convert_from_keras_model(
        model,
        backend=backend,
        io_type='io_stream',
        output_dir=out_dir,
        hls_config={'Model': {'Precision': 'ap_fixed<16,6>', 'ReuseFactor': 1, 'Strategy': 'GEMM',
                              'SecondOperandRowMajor': True}},
    )

    hls_model.compile()

    # The two-operand (QK^T / A.V) configs must declare row-major B; the weightless projection
    # configs have no b_row_major field. 2 QK^T + 2 A.V = 4 two-operand configs (2 heads).
    params = (tmp_path / f'mha_rm_{backend}' / 'firmware' / 'parameters.h').read_text()
    assert params.count('b_row_major = true') >= 4, 'expected >=4 row-major two-operand configs'
    assert 'b_row_major = false' not in params, 'no two-operand config should stay col-major'

    rng = np.random.default_rng(6789)
    data = [rng.normal(0, 0.5, size=(4, 4, 8)).astype(np.float32) for _ in range(3)]
    keras_pred = model.predict(data, verbose=0)
    assert np.std(keras_pred) > 1e-3, 'degenerate (near-constant) reference output'
    hls_pred = hls_model.predict(data).reshape(keras_pred.shape)
    np.testing.assert_allclose(hls_pred, keras_pred, atol=0.35, rtol=0.0)
