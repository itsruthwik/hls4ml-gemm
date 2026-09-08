"""Materialize the softmax exp and 1/x lookup tables as constant weight arrays.

The Vivado kernels fill their tables at run time from ``std::exp`` / a float
division, which Vitis constant-folds. Catapult cannot synthesize those (the
libm calls have no hardware definition), so the Catapult kernels take the
tables as array arguments and this pass computes them here, emulating the
Vivado ``init_exp_table`` / ``init_invert_table`` arithmetic (float32 exp and
division, then the table type's rounding and saturation). The arrays are
written through the ordinary weight path, so csim loads them from ``.txt``
and synthesis sees a constant initializer it maps to a ROM.
"""

from math import ceil, log2

import numpy as np

from hls4ml.model.layers import Softmax
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.types import FixedPrecisionType, NamedType, RoundingMode, SaturationMode, WeightVariable


def quantize_to_fixed(values: np.ndarray, precision: FixedPrecisionType) -> np.ndarray:
    """Round and saturate float values into ``precision`` (returns floats)."""
    width, fractional = precision.width, precision.fractional
    scaled = np.asarray(values, dtype=np.float64) * 2.0**fractional

    rm = precision.rounding_mode
    if rm == RoundingMode.TRN:
        q = np.floor(scaled)
    elif rm == RoundingMode.TRN_ZERO:
        q = np.trunc(scaled)
    elif rm == RoundingMode.RND:
        q = np.floor(scaled + 0.5)
    elif rm == RoundingMode.RND_ZERO:
        q = np.where(scaled >= 0, np.ceil(scaled - 0.5), np.floor(scaled + 0.5))
    elif rm == RoundingMode.RND_INF:
        q = np.where(scaled >= 0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))
    elif rm == RoundingMode.RND_MIN_INF:
        q = np.ceil(scaled - 0.5)
    elif rm == RoundingMode.RND_CONV:
        q = np.rint(scaled)
    else:
        raise ValueError(f'Unsupported rounding mode {rm}')

    if precision.signed:
        lo, hi = -(2 ** (width - 1)), 2 ** (width - 1) - 1
    else:
        lo, hi = 0, 2**width - 1
    sm = precision.saturation_mode
    if sm == SaturationMode.SAT:
        q = np.clip(q, lo, hi)
    elif sm == SaturationMode.SAT_SYM:
        q = np.clip(q, -hi if precision.signed else lo, hi)
    elif sm == SaturationMode.SAT_ZERO:
        q = np.where((q < lo) | (q > hi), 0.0, q)
    elif sm == SaturationMode.WRAP:
        q = np.mod(q - lo, 2.0**width) + lo
    else:
        raise ValueError(f'Unsupported saturation mode {sm}')

    return q * 2.0**-fractional


def table_index_values(precision: FixedPrecisionType, table_size: int) -> np.ndarray:
    """Real value addressed by each table index: the index forms the top bits of a
    ``precision`` word (mirrors ``softmax_real_val_from_idx``)."""
    width, integer, signed = precision.width, precision.integer, precision.signed
    n_bits = ceil(log2(table_size))
    assert n_bits <= width, f'softmax table of {table_size} entries needs {n_bits} address bits, input has {width}'
    raw = np.arange(table_size, dtype=np.int64) << (width - n_bits)
    if signed:
        raw = np.where(raw >= 2 ** (width - 1), raw - 2**width, raw)
    return raw.astype(np.float64) * 2.0 ** (integer - width)


class SoftmaxConstTables(OptimizerPass):
    def match(self, node):
        if not isinstance(node, Softmax):
            return False
        if node.get_attr('implementation') not in ('latency', 'stable'):
            return False
        return not isinstance(node.get_attr('exp_table'), WeightVariable)

    def transform(self, model, node):
        impl = node.get_attr('implementation')
        exp_table_t: NamedType = node.get_attr('exp_table_t')
        inv_table_t: NamedType = node.get_attr('inv_table_t')
        exp_scale = float(node.get_attr('exp_scale', 1.0))
        input_prec: FixedPrecisionType = node.get_input_variable().type.precision

        # Same fallbacks the config template applies for frontends that leave these unset.
        inv_inp_t: NamedType = node.get_attr('inv_inp_t')
        if inv_inp_t is None or inv_inp_t.name == 'model_default_t':
            inv_inp_t = exp_table_t
            node.set_attr('inv_inp_t', inv_inp_t)
        table_size = int(node.get_attr('table_size'))
        exp_table_size = int(node.get_attr('exp_table_size', table_size))
        inv_table_size = int(node.get_attr('inv_table_size', table_size))

        if impl == 'stable':
            inp_norm_t: NamedType = node.get_attr('inp_norm_t')
            if inp_norm_t is None:
                width, iwidth, signed = input_prec.width, input_prec.integer, input_prec.signed
                width, iwidth = width - signed, iwidth - signed
                inp_norm_t = NamedType(f'{node.name}_inp_norm_t', FixedPrecisionType(width, iwidth, False))
                node.set_attr('inp_norm_t', inp_norm_t)
            exp_table_size = min(exp_table_size, 2 ** inp_norm_t.precision.width)
            exp_index_t = inp_norm_t.precision
        else:
            exp_index_t = input_prec
        node.set_attr('exp_table_size', exp_table_size)
        node.set_attr('inv_table_size', inv_table_size)

        # exp table: float32 exp of the (scaled, and for stable negated) index value.
        x = table_index_values(exp_index_t, exp_table_size).astype(np.float32) * np.float32(exp_scale)
        if impl == 'stable':
            x = -x
        exp_vals = np.exp(x.astype(np.float64)).astype(np.float32)
        exp_table = quantize_to_fixed(exp_vals, exp_table_t.precision)

        # invert table: float32 1/x; 1/0 pinned to the saturated value 1/(0 + eps) gives.
        x = table_index_values(inv_inp_t.precision, inv_table_size).astype(np.float32)
        with np.errstate(divide='ignore'):
            inv_vals = np.where(x == 0, np.float32(1e7), np.float32(1.0) / x).astype(np.float32)
        inv_table = quantize_to_fixed(inv_vals, inv_table_t.precision)

        node.add_weights_variable(
            name='exp_table', type_name=exp_table_t.name, precision=exp_table_t.precision, data=exp_table
        )
        node.add_weights_variable(
            name='inv_table', type_name=inv_table_t.name, precision=inv_table_t.precision, data=inv_table
        )
        return True
