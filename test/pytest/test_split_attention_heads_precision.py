"""Unit test for SplitAttentionHeads' per-head quantizer precision handling.

Regression test for the bug where a per-head FixedPointQuantizer clone (e.g.
``mha_query_oq_h0``) re-resolved its result_t to Model.Precision.default instead
of carrying the source node's real (user-configured) numeric type. See
hls4ml/backends/fpga/passes/attention_heads.py:_clone_quantizer_head.
"""
import keras
import pytest

if keras.__version__ < '3.0.0':
    pytest.skip('This test requires Keras 3.0.0 or higher', allow_module_level=True)

pytest.importorskip('hgq')
pytest.importorskip('hgq.layers')

from hgq.config import QuantizerConfigScope
from hgq.layers import QMultiHeadAttention

from hls4ml.converters import convert_from_keras_model

NAME = 'mha'
L, D, H = 4, 8, 2
KEY_DIM = D // H

# Explicit 8-bit precision for every per-head quantizer clone this pass produces,
# keyed by its *own* (head-suffixed) name -- the pattern used by examples/mha_*.
HLS_CONFIG = {
    'Model': {
        'Precision': {'default': 'fixed<16,6,RND,WRAP,0>'},
        'ReuseFactor': 1,
        'Strategy': 'GEMM',
        'BramFactor': 1000000000,
    },
    'LayerName': {
        f'{NAME}_query_oq_h0': {'Precision': {'result': 'fixed<8,2,RND,WRAP,0>'}},
        f'{NAME}_query_oq_h1': {'Precision': {'result': 'fixed<8,2,RND,WRAP,0>'}},
        f'{NAME}_key_oq_h0': {'Precision': {'result': 'fixed<8,2,RND,WRAP,0>'}},
        f'{NAME}_key_oq_h1': {'Precision': {'result': 'fixed<8,2,RND,WRAP,0>'}},
        f'{NAME}_value_oq_h0': {'Precision': {'result': 'fixed<8,2,RND,WRAP,0>'}},
        f'{NAME}_value_oq_h1': {'Precision': {'result': 'fixed<8,2,RND,WRAP,0>'}},
        f'{NAME}_q_softmax_oq_h0': {'Precision': {'result': 'ufixed<8,1,RND,WRAP,0>'}},
        f'{NAME}_q_softmax_oq_h1': {'Precision': {'result': 'ufixed<8,1,RND,WRAP,0>'}},
        f'{NAME}_attention_output_iq_h0': {'Precision': {'result': 'fixed<8,2,RND,WRAP,0>'}},
        f'{NAME}_attention_output_iq_h1': {'Precision': {'result': 'fixed<8,2,RND,WRAP,0>'}},
    },
}

EXPECTED_HEAD_PRECISIONS = {
    f'{NAME}_query_oq_h0': 'fixed<8,2', f'{NAME}_query_oq_h1': 'fixed<8,2',
    f'{NAME}_key_oq_h0': 'fixed<8,2', f'{NAME}_key_oq_h1': 'fixed<8,2',
    f'{NAME}_value_oq_h0': 'fixed<8,2', f'{NAME}_value_oq_h1': 'fixed<8,2',
    f'{NAME}_q_softmax_oq_h0': 'ufixed<8,1', f'{NAME}_q_softmax_oq_h1': 'ufixed<8,1',
    f'{NAME}_attention_output_iq_h0': 'fixed<8,2', f'{NAME}_attention_output_iq_h1': 'fixed<8,2',
}


def _build_2head_model():
    with QuantizerConfigScope(f0=3, i0=2):
        q = keras.layers.Input((L, D), name=f'{NAME}_q_in')
        v = keras.layers.Input((L, D), name=f'{NAME}_v_in')
        out = QMultiHeadAttention(H, KEY_DIM, name=NAME, fuse='none')(q, v)
        model = keras.Model([q, v], out)
    return model


def test_split_attention_heads_precision_matches_source_not_default(tmp_path):
    """Every *_h0 / *_h1 clone the pass emits must carry its own explicit
    HLSConfig precision (or the source node's, when unconfigured) -- never
    silently fall back to Model.Precision.default fixed<16,6>."""
    model = _build_2head_model()

    hls_model = convert_from_keras_model(
        model,
        backend='Vitis',
        output_dir=str(tmp_path / 'split_attention_heads_precision'),
        io_type='io_stream',
        part='xcve2802-vsvh1760-2MP-e-S',
        hls_config=HLS_CONFIG,
        bit_exact=False,
    )

    found_any = False
    for node_name, expect in EXPECTED_HEAD_PRECISIONS.items():
        node = hls_model.graph.get(node_name)
        if node is None:
            continue
        found_any = True
        prec = str(node.get_output_variable().type.precision)
        assert prec.startswith(expect), (
            f'{node_name}: expected precision starting with {expect!r}, got {prec!r} '
            '(re-resolved to Model.Precision.default instead of the source/configured type)'
        )
        assert not prec.startswith('fixed<16,6') or expect.startswith('fixed<16,6'), (
            f'{node_name}: fell back to Model.Precision.default ({prec})'
        )

    assert found_any, 'SplitAttentionHeads did not run: no *_h0/*_h1 clone nodes found in the model graph'
