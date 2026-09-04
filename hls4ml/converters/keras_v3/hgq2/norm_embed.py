"""keras_v3 handlers for HGQ2 QLayerNormalization and QEmbedding.

The Vivado/Vitis backend already has the IR layers, C++ kernels
(`nnet_layernorm.h`, `nnet_embed.h`) and templates; these handlers just map the
HGQ2 layers onto them. `QLayerHandler.default_class_name` strips the leading 'Q',
so QLayerNormalization -> LayerNormalization and QEmbedding -> Embedding, and the
base wires the input/output fixed-point quantizers.
"""
from collections.abc import Sequence
from math import prod
from typing import TYPE_CHECKING

import numpy as np

from hls4ml.model.types import FixedPrecisionType

from ._base import QLayerHandler, fixed_quantizer_to_hls4ml_t

if TYPE_CHECKING:
    from keras import KerasTensor


def _extract_rsqrt_lut(rsqrt_table):
    """Extract HGQ2's rsqrt QUnaryFunctionLUT bit-exactly.

    HGQ2's QLayerNormalization does not compute 1/sqrt in float: it routes the
    per-token variance through a quantized LUT (`rsqrt_table`, a QUnaryFunctionLUT)
    with an input quantizer `iq` (which fixes the table's address grid) and an
    output quantizer `oq`. To be bit-exact to HGQ2, hls4ml must use *this* table,
    not its own `init_invert_sqr_table`. Mirrors QUnaryLUTHandler, specialized to
    the rsqrt table whose input (variance) is unsigned.

    Returns (table_data, table_t, addr_f, table_size): the LUT array indexed by the
    quantized variance address, its fixed-point type, the number of fractional bits
    the variance is quantized to (address scale = 2**addr_f), and the table length.
    """
    from hgq.quantizer.internal import FixedPointQuantizerBase
    from keras import ops
    from quantizers import get_fixed_quantizer_np

    iq = rsqrt_table.iq.quantizer
    assert isinstance(iq, FixedPointQuantizerBase), 'rsqrt input quantizer must be fixed-point'
    k, i, f = (int(ops.max(x)) for x in iq.kif)
    # Variance is always >= 0, so only the non-negative address range is ever hit,
    # regardless of whether the quantizer carries a sign bit (k). Build the table
    # over [0, 2**i) with 2**(i+f) entries, addressed by round(var * 2**f).
    addr_f = f
    _eps = 2.0**-f
    _max = 2.0**i - _eps
    N = int(round(_max / _eps + 1))  # == 2**(i+f)
    assert np.log2(N).is_integer(), f'rsqrt table size must be a power of 2, got {N}'

    # Address grid: every quantized variance value, in ascending order (unsigned ->
    # the raw address is just the integer var/eps, no signed reordering).
    all_var = np.linspace(0.0, float(_max), N, dtype=np.float32)
    table = rsqrt_table.activation(all_var)  # HGQ2's 1/sqrt(var + epsilon)
    oq_ndim = len(rsqrt_table.oq.quantizer.bits.shape)
    t = ops.reshape(table, (1,) * (oq_ndim - 1) + (int(table.shape[-1]),))
    table = ops.reshape(rsqrt_table.oq(t), (-1,))
    table = ops.convert_to_numpy(table)

    oq = rsqrt_table.oq.quantizer
    round_mode = oq.round_mode[2:] if oq.round_mode.startswith('S_') else oq.round_mode
    fixed_q = get_fixed_quantizer_np(round_mode, oq.overflow_mode)
    ok, oi, of = (int(ops.convert_to_numpy(x).ravel().item()) for x in oq.kif)
    table = fixed_q(table, ok, oi, of)
    table_t = FixedPrecisionType(ok + oi + of, ok + oi, bool(ok))
    return np.asarray(ops.convert_to_numpy(table)).ravel(), table_t, addr_f, N


class QLayerNormalizationHandler(QLayerHandler):
    handles = ('hgq.layers.layer_normalization.QLayerNormalization',)

    def handle(self, layer, in_tensors: Sequence['KerasTensor'], out_tensors: Sequence['KerasTensor']):
        from keras import ops

        in_shape = tuple(in_tensors[0].shape[1:])
        if len(in_shape) != 2:
            raise ValueError(
                f'hls4ml LayerNormalization needs 3D input (batch, seq, feat); got {in_tensors[0].shape} '
                f'for layer {layer.name}'
            )
        assert layer.axis in ([len(in_tensors[0].shape) - 1], [-1], -1), (
            f'Only axis=-1 LayerNormalization is supported in hls4ml (got axis={layer.axis})'
        )

        feat = in_shape[-1]
        if layer.scale:
            gamma = ops.convert_to_numpy(layer.kq(layer.ln_gamma, training=False))
        else:
            gamma = np.ones(feat, dtype='float32')
        if layer.center:
            beta = ops.convert_to_numpy(layer.bq(layer.ln_beta, training=False))
        else:
            beta = np.zeros(feat, dtype='float32')

        # Bit-exact rsqrt: use HGQ2's own quantized rsqrt LUT. Its values are 1/sqrt(var + eps)
        # already folded with epsilon and the output quantizer, so the kernel needs nothing about
        # epsilon and any epsilon value is supported.
        rsqrt_table_data, rsqrt_table_t, rsqrt_addr_f, rsqrt_table_size = _extract_rsqrt_lut(layer.rsqrt_table)

        # Bit-exact mean: HGQ2's QLayerNormalization quantizes the per-token mean via `mean_q`
        # (a homogeneous datalane quantizer). Emit its precision as mean_t so the kernel rounds
        # the computed mean to the SAME fixed-point value HGQ2 does -> (x - mean) is exact and the
        # whole LayerNorm reproduces HGQ2 bit-exactly.
        mean_t = fixed_quantizer_to_hls4ml_t(layer.mean_q.quantizer, take_max=True)

        return {
            'n_in': prod(in_shape),
            'seq_len': in_shape[-2],
            'axis': 2,
            'gamma_data': np.asarray(gamma).ravel(),
            'beta_data': np.asarray(beta).ravel(),
            'rsqrt_table_data': rsqrt_table_data,
            'rsqrt_table_t': rsqrt_table_t,
            'rsqrt_addr_f': rsqrt_addr_f,
            'table_size': rsqrt_table_size,
            'mean_t': mean_t,
        }


class QEmbeddingHandler(QLayerHandler):
    handles = ('hgq.layers.embedding.QEmbedding',)

    def handle(self, layer, in_tensors: Sequence['KerasTensor'], out_tensors: Sequence['KerasTensor']):
        from keras import ops

        emb = ops.convert_to_numpy(layer.kq(layer.embeddings, training=False))
        return {
            'n_in': in_tensors[0].shape[1],
            'vocab_size': int(layer.input_dim),
            'n_out': int(layer.output_dim),
            'embeddings_data': np.asarray(emb),
        }
