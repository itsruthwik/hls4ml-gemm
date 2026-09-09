"""Keras 2 / QKeras `QMultiHeadAttention`.

QKeras ships no attention layer, so this builds one on top of QKeras quantizers
by subclassing keras' own `MultiHeadAttention` and swapping the four EinsumDense
projections (query/key/value/output) for a quantized `QEinsumDense`. The float
attention math (Q.K^T, softmax, A.V) is inherited unchanged; its accumulator /
softmax precision is set later in the hls4ml config, matching the weight/
activation scheme used elsewhere in the fork.

This mirrors HGQ2's `QMultiHeadAttention` (which does the same swap with its own
`QEinsumDense`) so hls4ml's Keras 2 converter (hls4ml/converters/keras/multi_head_attention.py)
can decompose it into the same EinsumDense / Einsum / Softmax primitive nodes as
the native Keras 3 / HGQ2 handler.

Only usable in the Keras 2 + QKeras venv (`.venv-keras2`); importing it under
Keras 3 or without qkeras installed raises a clear ImportError instead of an
opaque one from deep inside keras/qkeras.
"""

try:
    import tensorflow as tf
    from keras.layers import EinsumDense, MultiHeadAttention
    from keras.src.layers.attention.multi_head_attention import (
        _build_proj_equation,
        _get_output_shape,
    )
    from qkeras.quantizers import quantized_bits
except ImportError as exc:  # pragma: no cover - exercised only on the wrong venv
    raise ImportError(
        'hls4ml.utils.qkeras_attention requires the Keras 2 + QKeras environment '
        '(.venv-keras2): Keras 3 and/or qkeras is not importable here. '
        f'Original error: {exc}'
    ) from exc

__all__ = ['QEinsumDense', 'QMultiHeadAttention', 'attention_hls_config']


class QEinsumDense(EinsumDense):
    """EinsumDense with a QKeras-quantized kernel and (optional) output.

    Straight-through fake-quant during training; the quantizer bit-widths are
    what the hls4ml converter reads to set precisions on the lowered node.
    """

    def __init__(self, *args, kernel_quantizer=None, output_quantizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_quantizer = kernel_quantizer
        self.output_quantizer = output_quantizer

    def call(self, inputs):
        kernel = self.kernel_quantizer(self.kernel) if self.kernel_quantizer else self.kernel
        ret = tf.einsum(self.equation, inputs, kernel)
        if self.bias is not None:
            ret += self.bias
        if self.activation is not None:
            ret = self.activation(ret)
        if self.output_quantizer is not None:
            ret = self.output_quantizer(ret)
        return ret


class QMultiHeadAttention(MultiHeadAttention):
    """MultiHeadAttention with QKeras-quantized projections."""

    def __init__(self, *args, weight_bits=8, weight_int=0, act_bits=8, act_int=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        # Integer bits for the projection-kernel quantizer. Default 0 keeps the
        # historical +/-1 range ([-1,1)); pass e.g. 7 for a full-range INT8
        # (fixed<8,8>) weight, needed by callers that want large-magnitude
        # integer weights instead of the usual sub-unity fake-quant range.
        if not isinstance(weight_int, int) or not (0 <= weight_int < weight_bits):
            raise ValueError(
                f'weight_int must be an int with 0 <= weight_int < weight_bits '
                f'(got weight_int={weight_int!r}, weight_bits={weight_bits!r})'
            )
        self.weight_int = weight_int
        self.act_bits = act_bits
        # Projection outputs (Q/K/V/context) routinely exceed [-1, 1); the activation
        # quantizer needs integer headroom or it saturates. act_int sets those integer
        # bits (range +/- 2**act_int). A production layer would calibrate/learn this
        # per tensor (what HGQ2 does); this is a sane, non-saturating default.
        self.act_int = act_int

    def _quantizers(self):
        return dict(
            kernel_quantizer=quantized_bits(self.weight_bits, self.weight_int, alpha=1),
            output_quantizer=quantized_bits(self.act_bits, self.act_int, alpha=1),
        )

    def _make_output_dense(self, free_dims, common_kwargs, name=None):
        import collections.abc

        if self._output_shape:
            if not isinstance(self._output_shape, collections.abc.Sized):
                output_shape = [self._output_shape]
            else:
                output_shape = self._output_shape
        else:
            output_shape = [self._query_shape[-1]]
        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            free_dims, bound_dims=2, output_dims=len(output_shape)
        )
        return QEinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(output_rank - 1, output_shape),
            bias_axes=bias_axes if self._use_bias else None,
            name=name,
            **self._quantizers(),
            **common_kwargs,
        )

    def _build_from_signature(self, query, value, key=None):
        # Mirrors keras' MultiHeadAttention._build_from_signature but builds
        # QEinsumDense projections instead of EinsumDense.
        self._built_from_signature = True
        self._query_shape = tf.TensorShape(query.shape if hasattr(query, "shape") else query)
        self._value_shape = tf.TensorShape(value.shape if hasattr(value, "shape") else value)
        if key is None:
            self._key_shape = self._value_shape
        else:
            self._key_shape = tf.TensorShape(key.shape if hasattr(key, "shape") else key)

        from keras.src.utils import tf_utils

        with tf_utils.maybe_init_scope(self):
            free_dims = self._query_shape.rank - 1
            eq, bias_axes, output_rank = _build_proj_equation(free_dims, bound_dims=1, output_dims=2)
            self._query_dense = QEinsumDense(
                eq, output_shape=_get_output_shape(output_rank - 1, [self._num_heads, self._key_dim]),
                bias_axes=bias_axes if self._use_bias else None, name="query",
                **self._quantizers(), **self._get_common_kwargs_for_sublayer())
            eq, bias_axes, output_rank = _build_proj_equation(
                self._key_shape.rank - 1, bound_dims=1, output_dims=2)
            self._key_dense = QEinsumDense(
                eq, output_shape=_get_output_shape(output_rank - 1, [self._num_heads, self._key_dim]),
                bias_axes=bias_axes if self._use_bias else None, name="key",
                **self._quantizers(), **self._get_common_kwargs_for_sublayer())
            eq, bias_axes, output_rank = _build_proj_equation(
                self._value_shape.rank - 1, bound_dims=1, output_dims=2)
            self._value_dense = QEinsumDense(
                eq, output_shape=_get_output_shape(output_rank - 1, [self._num_heads, self._value_dim]),
                bias_axes=bias_axes if self._use_bias else None, name="value",
                **self._quantizers(), **self._get_common_kwargs_for_sublayer())
            self._build_attention(output_rank)
            self._output_dense = self._make_output_dense(
                free_dims, self._get_common_kwargs_for_sublayer(), "attention_output")

    def get_config(self):
        config = super().get_config()
        config.update(weight_bits=self.weight_bits, weight_int=self.weight_int,
                       act_bits=self.act_bits, act_int=self.act_int)
        return config


def attention_hls_config(model, backend="Catapult", wide="ap_fixed<32,12>",
                         softmax_table_size=None, softmax_impl=None):
    """Build an hls4ml config that carries each QMultiHeadAttention's activation
    precision onto its decomposed nodes.

    The handler decomposes an MHA named `mha` into `mha_query/_key/_value` (Q/K/V
    projections), `mha_qk`, `mha_softmax`, `mha_av`, and `mha` (output projection).
    Weight precision is propagated by the handler itself; this helper stamps the
    projection *output* precision (the QEinsumDense output_quantizer) from the
    layer's act_bits/act_int, and keeps the matmuls/softmax wide so only the
    modelled quantization steps (plus the softmax table) contribute error.

    Entirely config-side (no hls4ml changes) -- the converter-only path.
    """
    import hls4ml

    cfg = hls4ml.utils.config_from_keras_model(model, backend=backend, granularity="name")
    ln = cfg.setdefault("LayerName", {})

    def set_result(node_name, prec):
        ln.setdefault(node_name, {}).setdefault("Precision", {})["result"] = prec

    for layer in model.layers:
        if layer.__class__.__name__ != "QMultiHeadAttention":
            continue
        name = layer.name
        # quantized_bits(act_bits, act_int) signed -> ap_fixed<act_bits, act_int+1>
        act_t = f"ap_fixed<{layer.act_bits}, {layer.act_int + 1}>"
        for suffix in ("_query", "_key", "_value", ""):  # "" == output projection
            set_result(name + suffix, act_t)
        for suffix in ("_qk", "_softmax", "_av"):  # matmuls + softmax stay wide
            set_result(name + suffix, wide)
        # The softmax exp/inv tables synthesize as dynamic-indexed register banks;
        # at the hls4ml default (1024) Catapult's dpfsm transform blows up in time
        # and memory. Let callers shrink the tables (and pick the impl) so full
        # synthesis stays tractable.
        sm = ln.setdefault(name + "_softmax", {})
        if softmax_table_size is not None:
            sm["TableSize"] = softmax_table_size
        if softmax_impl is not None:
            sm["Implementation"] = softmax_impl
    return cfg
