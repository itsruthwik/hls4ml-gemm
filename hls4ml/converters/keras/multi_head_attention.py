"""Keras 2 converter for QMultiHeadAttention.

QKeras has no attention layer; ATLAS supplies a Keras 2 `QMultiHeadAttention` built
on QKeras-quantized EinsumDense projections (`hls4ml.utils.qkeras_attention`).
This handler decomposes it — at parse time, entirely within the Keras 2 path —
into the same primitive nodes HGQ2's Keras 3 handler produces:

    query/key/value EinsumDense  ->  Einsum(Q.K^T)  ->  Softmax  ->  Einsum(A.V)  ->  output EinsumDense

The einsum equations and intermediate shapes are quantization-independent, so we
reconstruct a stock keras MultiHeadAttention from the numeric config to derive
them exactly (no dependency on the ATLAS layer), and read the trained weights
through the reader. Only 'QMultiHeadAttention' is registered (stock Keras 2
`MultiHeadAttention` is out of scope by design); it carries `weight_bits`, from
which this handler stamps the projection WEIGHT precision. Projection
ACTIVATION/result precision is set separately, via the hls4ml config on the
decomposed node names (see hls4ml.utils.qkeras_attention.attention_hls_config)
-- keeping this handler purely structural.

Supported: rank-3 (batch, seq, feature) self- and cross-attention, multi-head,
single attention axis. Rejected with a clear error: attention masks, rank != 3,
multi-axis attention. Works on both the non-GEMM baseline (io_parallel) and the
GEMM path (io_stream / io_parallel), since it only emits generic
EinsumDense / Einsum / Softmax nodes.
"""

from math import prod

from hls4ml.converters.keras_v2_to_hls import get_weights_data, keras_handler
from hls4ml.model.quantizers import QKerasQuantizer


def _bits_quantizer(bits, integer=0):
    """A signed quantized_bits(bits, integer, alpha=1) QKeras quantizer."""
    return QKerasQuantizer(
        {'class_name': 'quantized_bits',
         'config': {'bits': bits, 'integer': integer, 'keep_negative': True, 'alpha': 1}}
    )


def _strip_batch_dim(equation, einsum_dense):
    """Drop the leading batch axis from a keras einsum equation."""
    inps, out = equation.split('->')
    if einsum_dense:
        inp0, inp1 = inps.split(',')
        inp0, out = inp0[1:], out[1:]  # kernel (inp1) has no batch axis
        return f'{inp0},{inp1}->{out}'
    inp0, inp1 = inps.split(',')
    return f'{inp0[1:]},{inp1[1:]}->{out[1:]}'


@keras_handler('QMultiHeadAttention')
def parse_mha_layer(keras_layer, input_names, input_shapes, data_reader):
    import keras

    cfg = keras_layer['config']
    name = cfg['name']
    num_heads = cfg['num_heads']
    key_dim = cfg['key_dim']
    value_dim = cfg.get('value_dim') or key_dim
    use_bias = cfg.get('use_bias', True)
    out_shape_cfg = cfg.get('output_shape')
    attn_axes = cfg.get('attention_axes')
    # QMultiHeadAttention carries INT8 bit-widths; stock MultiHeadAttention does not
    # (None -> leave projection precision to the hls4ml config).
    weight_bits = cfg.get('weight_bits')
    act_bits = cfg.get('act_bits')
    act_int = cfg.get('act_int', 0)

    # inputs: query, value, [key]. key defaults to value when absent.
    # A 4th input means an attention_mask tensor was passed (nested in call-kwargs
    # and surfaced by the v2 parser) -- not supported here, and not supported by
    # HGQ2 either.
    assert len(input_names) in (2, 3), (
        f"MultiHeadAttention '{name}': expected 2 (Q,V) or 3 (Q,V,K) inputs, got "
        f"{len(input_names)}. Attention masks are not supported."
    )
    # Resolve query/value/key by call-arg name when available (keras records extra
    # call args like value/key in the inbound-node kwargs), falling back to the
    # positional order the parser flattened them into. Robust to mha(q, v),
    # mha(q, v, k) and the all-kwargs mha(query=q, value=v, key=k).
    argnames = []
    inbound = keras_layer.get('inbound_nodes') or []
    if inbound:
        for entry in inbound[0]:
            argnames.append(None)  # positional ref
            if len(entry) >= 4 and isinstance(entry[3], dict):
                for kw, ref in entry[3].items():
                    if isinstance(ref, list) and len(ref) >= 3 and isinstance(ref[0], str):
                        argnames.append(kw)
    # argnames aligns index-wise with input_names / input_shapes (same flattening).
    if len(argnames) != len(input_names):
        argnames = [None] * len(input_names)  # unexpected shape: fall back to positional

    def _idx(name, pos):
        return argnames.index(name) if name in argnames else pos

    qi, vi = _idx('query', 0), _idx('value', 1)
    ki = argnames.index('key') if 'key' in argnames else (2 if len(input_names) > 2 else vi)
    q_in, v_in, k_in = input_names[qi], input_names[vi], input_names[ki]
    q_shape = list(input_shapes[qi])
    v_shape = list(input_shapes[vi])
    k_shape = list(input_shapes[ki])

    # Only rank-3 (batch, seq, feature) sequence attention is supported; higher-rank
    # (e.g. 2D/image) attention would need different score/context shapes.
    assert len(q_shape) == 3, (
        f"MultiHeadAttention '{name}': only rank-3 (batch, seq, feature) inputs are "
        f"supported, got query rank {len(q_shape)}."
    )
    # A custom multi-axis attention would change the softmax reduction; only the
    # default single (last non-batch) attention axis is supported.
    assert attn_axes is None or len(tuple(attn_axes)) == 1, (
        f"MultiHeadAttention '{name}': only single-axis attention is supported, got "
        f"attention_axes={attn_axes}."
    )

    # Reconstruct a stock keras MHA to derive equations + shapes (weights unused).
    mha = keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, value_dim=value_dim,
        use_bias=use_bias, output_shape=out_shape_cfg, attention_axes=attn_axes,
    )
    qT = keras.Input(batch_shape=q_shape)
    vT = keras.Input(batch_shape=v_shape)
    kT = keras.Input(batch_shape=k_shape)
    mha(qT, vT, kT)  # builds sublayers + equations

    # This handler reads keras' private MHA internals to recover the einsum
    # equations. Guard them so a keras version that renames/moves them fails loudly
    # here instead of producing a wrong graph.
    _internals = ('_query_dense', '_key_dense', '_value_dense', '_output_dense',
                  '_dot_product_equation', '_combine_equation')
    missing = [a for a in _internals if not hasattr(mha, a)]
    assert not missing, (
        f"keras MultiHeadAttention internals {missing} not found (keras "
        f"{keras.__version__}); the MHA converter depends on them -- update the "
        f"handler for this keras version."
    )

    to_Q, to_K, to_V, to_O = mha._query_dense, mha._key_dense, mha._value_dense, mha._output_dense
    # Compute shapes explicitly from known dims (EinsumDense.compute_output_shape
    # drops the sequence length). Standard MHA: key and value share a sequence.
    seq_q, seq_k, seq_v = q_shape[1], k_shape[1], v_shape[1]
    if out_shape_cfg:
        d_out = out_shape_cfg if isinstance(out_shape_cfg, int) else list(out_shape_cfg)[-1]
    else:
        d_out = q_shape[-1]
    Q_shape = [None, seq_q, num_heads, key_dim]
    K_shape = [None, seq_k, num_heads, key_dim]
    V_shape = [None, seq_v, num_heads, value_dim]
    score_shape = [None, num_heads, seq_q, seq_k]
    pre_O_shape = [None, seq_q, num_heads, value_dim]
    O_shape = [None, seq_q, d_out]

    # Keras MHA scales the query by 1/sqrt(key_dim) before the Q.K^T dot product
    # (keras _compute_attention). Fold that scale into the query projection weights.
    # We deliberately do NOT use the softmax exp_scale for this: exp_scale only
    # scales the exp *table values*, while the table is *addressed* by the raw
    # score, so leaving the query unscaled makes the scores ~sqrt(key_dim) larger
    # and coarsens the softmax table addressing (measured: error grows ~10x).
    # Folding keeps the scores small; the query projection's weight precision must
    # carry fractional headroom for the scaled values (handled in precision setup).
    import numpy as _np

    _inv = 1.0 / _np.sqrt(float(key_dim))
    wq = get_weights_data(data_reader, name, 'query/kernel') * _inv
    wk = get_weights_data(data_reader, name, 'key/kernel')
    wv = get_weights_data(data_reader, name, 'value/kernel')
    wo = get_weights_data(data_reader, name, 'attention_output/kernel')
    bq = get_weights_data(data_reader, name, 'query/bias') if use_bias else None
    if bq is not None:
        bq = bq * _inv
    bk = get_weights_data(data_reader, name, 'key/bias') if use_bias else None
    bv = get_weights_data(data_reader, name, 'value/bias') if use_bias else None
    bo = get_weights_data(data_reader, name, 'attention_output/bias') if use_bias else None

    def _tuple(shape):
        return tuple(int(d) for d in shape[1:])

    # query / key / value projections
    q_node = {
        'class_name': 'EinsumDense', 'name': f'{name}_query', 'inputs': [q_in],
        'equation': _strip_batch_dim(to_Q.equation, True),
        'weight_data': wq, 'bias_data': bq,
        'inp_shape': _tuple(q_shape), 'out_shape': _tuple(Q_shape),
    }
    k_node = {
        'class_name': 'EinsumDense', 'name': f'{name}_key', 'inputs': [k_in],
        'equation': _strip_batch_dim(to_K.equation, True),
        'weight_data': wk, 'bias_data': bk,
        'inp_shape': _tuple(k_shape), 'out_shape': _tuple(K_shape),
    }
    v_node = {
        'class_name': 'EinsumDense', 'name': f'{name}_value', 'inputs': [v_in],
        'equation': _strip_batch_dim(to_V.equation, True),
        'weight_data': wv, 'bias_data': bv,
        'inp_shape': _tuple(v_shape), 'out_shape': _tuple(V_shape),
    }
    # Q.K^T  (equation is source(key),target(query)->product(score))
    qk_node = {
        'class_name': 'Einsum', 'name': f'{name}_qk', 'inputs': [f'{name}_key', f'{name}_query'],
        'equation': _strip_batch_dim(mha._dot_product_equation, False),
        'inp0_shape': _tuple(K_shape), 'inp1_shape': _tuple(Q_shape), 'out_shape': _tuple(score_shape),
    }
    # softmax over the key axis (last axis of the score tensor)
    score_dims = [int(d) for d in score_shape[1:]]
    sm_node = {
        'class_name': 'Softmax', 'name': f'{name}_softmax', 'inputs': [f'{name}_qk'],
        'activation': 'softmax', 'axis': -1,
        'n_in': prod(score_dims), 'n_outer': prod(score_dims[:-1]), 'n_inner': 1,
    }
    # A.V  (equation is product(score),source(value)->target(context))
    av_node = {
        'class_name': 'Einsum', 'name': f'{name}_av', 'inputs': [f'{name}_softmax', f'{name}_value'],
        'equation': _strip_batch_dim(mha._combine_equation, False),
        'inp0_shape': _tuple(score_shape), 'inp1_shape': _tuple(V_shape), 'out_shape': _tuple(pre_O_shape),
    }
    # output projection — canonical output, named after the original layer
    o_node = {
        'class_name': 'EinsumDense', 'name': name, 'inputs': [f'{name}_av'],
        'equation': _strip_batch_dim(to_O.equation, True),
        'weight_data': wo, 'bias_data': bo,
        'inp_shape': _tuple(pre_O_shape), 'out_shape': _tuple(O_shape),
    }

    # Propagate INT8 WEIGHT precision from the layer's bit-widths onto the
    # projection weights/biases. The query projection carries the folded
    # 1/sqrt(key_dim) scale, which pushes its (originally INT8) weights off the
    # integer grid; give it a few extra fractional bits so the scaled values are
    # represented without re-quant.
    if weight_bits is not None:
        for nd in (q_node, k_node, v_node, o_node):
            nd['weight_quantizer'] = _bits_quantizer(weight_bits)
            if nd['bias_data'] is not None:
                nd['bias_quantizer'] = _bits_quantizer(weight_bits)
        # Note: the query weights carry the folded 1/sqrt(key_dim) scale, so their
        # magnitude is < 1 and they use only part of the [-1, 1) grid -> a few bits
        # of effective resolution are wasted (they stay `weight_bits` WIDE, no growth).
        # The clean fix -- shifting the query fixed-point range down (fewer integer
        # bits) -- is not expressible via QKeras quantized_bits, which rejects a
        # negative `integer` (it computes tf.pow(2, integer) with int args). Reclaiming
        # that resolution would need setting the query weight precision directly.

    # NOTE (activation/result precision, step 3b): modelling each projection's
    # output quantizer must be done as the EinsumDense's *own* result precision,
    # in place -- NOT by inserting activation nodes between the projection and the
    # QK/AV einsums. The Catapult attention pass requires the einsum's inputs to be
    # EinsumDense projections directly (it errors otherwise), so inserted nodes
    # break the GEMM lowering. Tracked as remaining work.
    nodes = [q_node, k_node, v_node, qk_node, sm_node, av_node, o_node]
    shapes = [Q_shape, K_shape, V_shape, score_shape, score_shape, pre_O_shape, O_shape]
    return [(n, s) for n, s in zip(nodes, shapes)]
