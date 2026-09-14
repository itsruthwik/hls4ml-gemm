from hls4ml.backends.gemm_ip_config import GemmIPConfigTemplateBase
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.backends.fpga.passes.gemm_nodes import (
    Im2Col,
    Gemm,
)

# ---------------------------------------------------------------------------
# Im2Col templates
# ---------------------------------------------------------------------------

im2col_config_template = """struct config{index} : nnet::im2col_config {{
    static const unsigned in_height = {in_height};
    static const unsigned in_width = {in_width};
    static const unsigned n_chan = {n_chan};
    static const unsigned filt_height = {filt_height};
    static const unsigned filt_width = {filt_width};
    static const unsigned stride_height = {stride_height};
    static const unsigned stride_width = {stride_width};
    static const unsigned out_height = {out_height};
    static const unsigned out_width = {out_width};
    static const unsigned pad_top = {pad_top};
    static const unsigned pad_bottom = {pad_bottom};
    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
    static const unsigned gemm_m = {gemm_m};
    static const unsigned tile_rows = {im2col_tile_rows};
}};\n"""

im2col_function_template = 'nnet::im2col_{n_dim}d_stream<{input_t}, {output_t}, {config}>({input}, {output});'
im2col_gemm_rows_function_template = (
    'nnet::im2col_{n_dim}d_gemm_rows<{input_t}, {output_t}, {config}>({input}, {output});'
)

# io_parallel, Strategy:GEMM: the emitter writes a_row_T a_rows[gemm_m] (one packed
# K-wide row per patch), but the standalone node's output variable is the flat
# [n_patches, patch_size] array the downstream Gemm array template reads
# ({input}[row*gemm_k+kk]) — unpack into that flat layout here rather than
# widening the emitter (kept frozen; see plan.md).
im2col_gemm_rows_array_function_template = """
    {{
        typedef nnet::array<{output_scalar_t}, config{index}::filt_height * config{index}::filt_width \
* config{index}::n_chan> a_row_t;
        a_row_t a_rows[config{index}::gemm_m];
        nnet::im2col_{n_dim}d_gemm_rows_array<{input_t}, a_row_t, config{index}>({input}, a_rows);
        UNPACK_IM2COL_{index}: for (unsigned row = 0; row < config{index}::gemm_m; row++) {{
            #pragma hls_unroll
            for (unsigned kk = 0; kk < a_row_t::size; kk++) {{
                #pragma hls_unroll
                {output}[row * a_row_t::size + kk] = a_rows[row][kk];
            }}
        }}
    }}
"""


class Im2ColConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Im2Col)
        self.template = im2col_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['gemm_m'] = node.get_attr('gemm_m', None) or 1
        params['im2col_tile_rows'] = node.get_attr('im2col_tile_rows', None) or 1
        return self.template.format(**params)


class Im2ColFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Im2Col, include_header=['nnet_utils/nnet_im2col.h'])
        self.template = im2col_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['n_dim'] = 2 if node.get_attr('in_height', 1) > 1 or node.get_attr('filt_height', 1) > 1 else 1
        if node.get_attr('strategy') == 'gemm':
            io_type = node.model.config.get_config_value('IOType')
            if io_type == 'io_parallel':
                params['output_scalar_t'] = node.get_output_variable().type.precision.definition_cpp()
                return im2col_gemm_rows_array_function_template.format(**params)
            return im2col_gemm_rows_function_template.format(**params)
        return self.template.format(**params)


# ---------------------------------------------------------------------------
# GemmStream templates (Dense GEMM IP — row/column streaming)
# ---------------------------------------------------------------------------

gemm_const_weights_config_template = """struct config{index} : nnet::gemm_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned n_patches = {n_patches};
    static const unsigned gemm_m = {gemm_m};
    static const unsigned gemm_k = {n_in};
    static const unsigned gemm_n = {n_out};
    static const unsigned gemm_ip_id = {index};
    // Microarchitecture knobs consumed by the generic (behavioral-HLS) GEMM core:
    // the reuse loop trip count and the multiplier ALLOCATION cap (mirrors the
    // Vivado writer's own gemm_const_weights_config_template).
    static const unsigned reuse_factor = {reuse_factor};
    static const unsigned multiplier_limit = {multiplier_limit};
    typedef {weight_t.name} weight_t;
    typedef {bias_t.name} bias_t;
    typedef {accum_t.name} accum_t;
    template<class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};\n"""

# The io_stream const-weight GEMM has exactly ONE signature: weight-stationary /
# const_weights. A SINGLE call for BOTH csim and synth — the overload (nnet_gemm_stream.h)
# branches internally: synth + GEMM_IP_HEADER → external weight-stationary IP; csim /
# no package → native behavioral model sourcing weights from CONFIG_T::gemm_weight_cols()
# (the ROM accessor injected into the config struct). No weights on the call site.
# Bias, when this node has one, is read through the config (CONFIG_T::gemm_bias(),
# injected alongside the weight ROM) rather than a function argument -- the same
# mechanism the baked weight matrix uses, so there is only ever this one call form.
gemm_stream_const_weights_function_template = (
    'nnet::gemm_stream_const_weights<{input_t}, {output_t}, {config}>({input}, {output});'
)


# ---------------------------------------------------------------------------
# Two-operand Gemm (attention QK^T / A.V): both operands are activations, no
# constant weight ROM. weight_t is the B operand's type (only used by cast's
# binary-quantizer SFINAE); bias is a local zero array. n_inplace batches the
# per-head GEMMs; the template loops it around the (unrolled) call.
# ---------------------------------------------------------------------------
gemm_two_operand_config_template = """struct config{index} {{
    static const unsigned n_in = {gemm_k};
    static const unsigned n_out = {gemm_n};
    static const unsigned gemm_m = {gemm_m};
    static const unsigned gemm_k = {gemm_k};
    static const unsigned gemm_n = {gemm_n};
    static const unsigned gemm_ip_id = {index};
    static const unsigned n_inplace = {n_inplace};
    // The generated two-operand streaming core expects the B operand transposed,
    // and asserts CONFIG_T::transpose_weights. This bare struct does not inherit
    // nnet::gemm_config, so declare it explicitly (true for the QK^T / A.V cores).
    static const bool transpose_weights = true;
    // Microarchitecture knobs consumed by the generic (behavioral-HLS) GEMM core
    // (mirrors the Vivado writer's own gemm_two_operand_config_template).
    static const unsigned reuse_factor = {reuse_factor};
    static const unsigned multiplier_limit = {multiplier_limit};
    typedef {input1_t} weight_t;
    typedef {accum_t.name} accum_t;
    // Two-operand GEMM (QK^T / A.V) carries no bias; the cell still takes a bias
    // array, fed a zero. bias_t must be a SCALAR addable to accum_t — not output_t,
    // which under io_stream is a packed beat (nnet::array), not a scalar.
    typedef {accum_t.name} bias_t;
    template<class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};\n"""

# io_parallel: both operands are flat arrays in canonical [I, L*, C] order; pack
# per-head A rows / B columns and call the array core once per head.
gemm_array_two_op_function_template = """
    {{
        typedef nnet::array<{input0_scalar_t}, config{index}::gemm_k> a_row_t;
        typedef nnet::array<{input1_scalar_t}, config{index}::gemm_k> b_col_t;
        typedef nnet::array<{output_scalar_t}, config{index}::gemm_n> res_row_t;

        #pragma hls_unroll
        for (unsigned i = 0; i < config{index}::n_inplace; i++) {{
            a_row_t a_rows[config{index}::gemm_m];
            b_col_t b_cols[config{index}::gemm_n];
            res_row_t result_rows[config{index}::gemm_m];
            for (unsigned l0 = 0; l0 < config{index}::gemm_m; l0++) {{
                #pragma hls_unroll
                for (unsigned c = 0; c < config{index}::gemm_k; c++) {{
                    #pragma hls_unroll
                    a_rows[l0][c] = {input0}[(i * config{index}::gemm_m + l0) * config{index}::gemm_k + c];
                }}
            }}
            for (unsigned l1 = 0; l1 < config{index}::gemm_n; l1++) {{
                #pragma hls_unroll
                for (unsigned c = 0; c < config{index}::gemm_k; c++) {{
                    #pragma hls_unroll
                    b_cols[l1][c] = {input1}[(i * config{index}::gemm_n + l1) * config{index}::gemm_k + c];
                }}
            }}
            nnet::gemm_array<a_row_t, b_col_t, res_row_t, config{index}>(
                a_rows, b_cols, result_rows
            );
            for (unsigned l0 = 0; l0 < config{index}::gemm_m; l0++) {{
                #pragma hls_unroll
                for (unsigned l1 = 0; l1 < config{index}::gemm_n; l1++) {{
                    #pragma hls_unroll
                    {output}[(i * config{index}::gemm_m + l0) * config{index}::gemm_n + l1] = result_rows[l0][l1];
                }}
            }}
        }}
    }}
"""

# io_stream: both operands are ac_channels in canonical head-ordered layout; call
# the two-stream core once per head (unrolled, inlined -> one design block).
gemm_stream_two_op_function_template = """
    {{
        #pragma hls_unroll
        for (unsigned i = 0; i < config{index}::n_inplace; i++) {{
            nnet::gemm_stream<{input0_t}, {input1_t}, {output_t}, config{index}>(
                {input0}, {input1}, {output}
            );
        }}
    }}
"""

# n_inplace == 1 (the common attention-head case): no bias array, no zero-fill
# loop, no per-head loop -- a bare call. Two-operand GEMM never has a real bias.
gemm_stream_two_op_single_function_template = (
    'nnet::gemm_stream<{input0_t}, {input1_t}, {output_t}, config{index}>({input0}, {input1}, {output});'
)


def _inject_weight_rom_accessor(cfg, node):
    """Add the const-operand ROM accessor to a GEMM config struct.

    Every weight-stationary entry (stream and array alike) needs to source the constant
    operand itself when no backend package is present, so the accessor belongs to every
    const-weight GEMM config rather than to one template. Only odr-used in that csim
    case; with a package the body is never instantiated and the ROM global is dead-code
    eliminated. The ROM is #include'd above the config in parameters.h, so unqualified
    lookup resolves here.
    """
    w = node.get_weights('weight').name
    if node.get_attr('second_operand_row_major', False):
        # Row-major (SecondOperandRowMajor): one N-wide contraction row per beat.
        inject = (
            '    static const bool weights_row_major = true;\n'
            '    typedef nnet::array<weight_t, gemm_n> weight_beat_t;\n'
            f'    static weight_beat_t* gemm_weight_beats() {{ return {w}_gemm_rows; }}\n'
        )
    else:
        # Column-major (default): one K-high output column per beat. weight_col_t /
        # gemm_weight_cols() are the legacy names of the same ROM, kept for consumers
        # that predate weight_beat_t.
        inject = (
            '    static const bool weights_row_major = false;\n'
            '    typedef nnet::array<weight_t, gemm_k> weight_beat_t;\n'
            f'    static weight_beat_t* gemm_weight_beats() {{ return {w}_gemm_cols; }}\n'
            '    typedef weight_beat_t weight_col_t;\n'
            f'    static weight_col_t* gemm_weight_cols() {{ return {w}_gemm_cols; }}\n'
        )
    # Bias, like the weight ROM above, is read through the config
    # (CONFIG_T::gemm_bias()) rather than a function argument -- the same
    # mechanism, so it never becomes a port. The bias weight variable always
    # exists (added unconditionally in gemm_nodes.py); a row-varying EinsumDense
    # bias is added in the wrapper instead, so point the accessor at a zero array.
    row_varying = bool(node.get_attr('_row_varying_bias', False))
    if row_varying:
        inject += (
            '    static bias_t* gemm_bias() { static bias_t zero_bias[gemm_n] = {}; return zero_bias; }\n'
        )
    else:
        b = node.get_weights('bias').name
        inject += f'    static bias_t* gemm_bias() {{ return {b}; }}\n'
    stripped = cfg.rstrip()
    assert stripped.endswith('};'), 'unexpected gemm config layout'
    return stripped[:-2] + inject + '};\n'


class GemmConfigTemplate(GemmIPConfigTemplateBase):
    """One config for the unified Gemm node — interface-agnostic.

    Uses the ``nnet::gemm_config`` base for both io_stream and io_parallel
    (the array path already inherited it via the shared im2col config). Embeds the
    weight-ROM accessor so the CONST_WEIGHTS behavioral model (csim, no gemm-ip-gen)
    can source constant weights without them appearing on the call signature.
    """
    backend_name = 'catapult'

    def __init__(self):
        super().__init__(Gemm)
        self.template = gemm_const_weights_config_template

    def format(self, node):
        if not node.get_attr('weights_in_core', True):
            return self._format_two_operand(node)
        return _inject_weight_rom_accessor(super().format(node), node)

    def _format_two_operand(self, node):
        # Two-operand Gemm: no weight ROM; product/weight_t come from the two
        # activation inputs (attention QK^T / A.V).
        from hls4ml.backends.backend import get_backend

        inp0 = node.get_input_variable(node.inputs[0])
        inp1 = node.get_input_variable(node.inputs[1])
        rf = max(1, int(node.get_attr('reuse_factor', 1) or 1))
        gk = int(node.get_attr('gemm_k'))
        gn = int(node.get_attr('gemm_n'))
        params = {
            'index': node.index,
            'gemm_m': node.get_attr('gemm_m'),
            'gemm_k': gk,
            'gemm_n': gn,
            'n_inplace': node.get_attr('n_inplace', 1),
            'reuse_factor': rf,
            'multiplier_limit': -(-(gk * gn) // rf),
            'input1_t': inp1.type.name,
            'accum_t': node.types['accum_t'],
            'product_type': get_backend('catapult').product_type(inp0.type.precision, inp1.type.precision),
        }
        return gemm_two_operand_config_template.format(**params)


class GemmFunctionTemplate(FunctionCallTemplate):
    """One function template for the unified Gemm node.

    Dispatches the 2x2 (IOType x weights_in_core) among the four gemm_* signatures.
    Phase 1 exercises only the weight-stationary column (Dense / pointwise Conv);
    the two-operand column (attention QK^T / A.V) is wired in Phase 2.
    """

    def __init__(self):
        super().__init__(
            Gemm,
            include_header=['nnet_utils/nnet_gemm_stream.h', 'nnet_utils/nnet_gemm_ip.h'],
        )
        # No self.template: format() fully overrides the base and dispatches the
        # 2x2 (IOType x weights_in_core) among the four gemm_* signatures explicitly.

    def format(self, node):
        io_type = node.model.config.get_config_value('IOType')

        if not node.get_attr('weights_in_core', True):
            # Two-operand GEMM (attention QK^T / A.V): both operands are activations,
            # no constant weight/bias to reference.
            inp0 = node.get_input_variable(node.inputs[0])
            inp1 = node.get_input_variable(node.inputs[1])
            out_var = node.get_output_variable()
            two_op = {
                'index': node.index,
                'input0': inp0.name,
                'input1': inp1.name,
                'output': out_var.name,
                'input0_t': inp0.type.name,
                'input1_t': inp1.type.name,
                'output_t': out_var.type.name,
                'input0_scalar_t': inp0.type.precision.definition_cpp(),
                'input1_scalar_t': inp1.type.precision.definition_cpp(),
                'output_scalar_t': out_var.type.precision.definition_cpp(),
            }
            if io_type == 'io_parallel':
                return gemm_array_two_op_function_template.format(**two_op)
            if node.get_attr('n_inplace', 1) == 1:
                # Bare call: no bias array/port, no zero-fill, no per-head loop.
                return gemm_stream_two_op_single_function_template.format(**two_op)
            # n_inplace > 1: unchanged, needs its own follow-up look (plan.md).
            return gemm_stream_two_op_function_template.format(**two_op)

        params = self._default_function_params(node)
        params['w'] = node.get_weights('weight').name
        params['weight_t'] = node.get_weights('weight').type.name
        has_bias = bool(node.get_attr('has_bias', False))
        row_varying = bool(node.get_attr('_row_varying_bias', False))
        # The row-varying wrapper always needs {b}/{bias_t} to add the per-element
        # bias, regardless of has_bias (has_bias only gates the IP's own port).
        if has_bias or row_varying:
            params['b'] = node.get_weights('bias').name
            params['bias_t'] = node.get_weights('bias').type.name

        if row_varying:
            # Bias varies along the data/row axis — the per-column IP port can't
            # express it (see gemm_array_row_bias_function_template).
            if io_type == 'io_parallel':
                return gemm_array_row_bias_function_template.format(**params)
            raise NotImplementedError(
                f"Gemm '{node.name}': row-varying EinsumDense bias is not yet wired for "
                'io_stream (io_parallel only). Attention projections carry per-column bias.'
            )

        if io_type == 'io_parallel':
            return gemm_array_function_template.format(**params)
        return gemm_stream_const_weights_function_template.format(**params)


gemm_array_function_template = """
    {{
        typedef nnet::array<{input_t}, config{index}::gemm_k> a_row_t;
        typedef nnet::array<{output_t}, config{index}::gemm_n> res_row_t;

        a_row_t a_rows[config{index}::gemm_m];
        res_row_t result_rows[config{index}::gemm_m];

        PACK_A_ROWS_{index}: for (unsigned row = 0; row < config{index}::gemm_m; row++) {{
            #pragma hls_unroll
            for (unsigned kk = 0; kk < config{index}::gemm_k; kk++) {{
                #pragma hls_unroll
                a_rows[row][kk] = {input}[row * config{index}::gemm_k + kk];
            }}
        }}

        nnet::gemm_array_const_weights<a_row_t, res_row_t, config{index}>(
            a_rows, result_rows
        );

        UNPACK_C_ROWS_{index}: for (unsigned row = 0; row < config{index}::gemm_m; row++) {{
            #pragma hls_unroll
            for (unsigned col = 0; col < config{index}::gemm_n; col++) {{
                #pragma hls_unroll
                {output}[row * config{index}::gemm_n + col] = result_rows[row][col];
            }}
        }}
    }}
"""


# io_parallel const_weights GEMM with ROW-VARYING bias (EinsumDense whose bias_axes
# touches the data free axis). The IP bias PORT is one value per column, so feed it
# ZERO and add the full per-element bias ({b} sized gemm_m*gemm_n) in the unpack.
# The add is in the result (res_T) domain rather than accum_t — that keeps the four
# gemm_* core signatures intact (an accum_t add would need a fifth, accum-draining
# core); the extra rounding is <= 1 res_T LSB.
gemm_array_row_bias_function_template = """
    {{
        typedef nnet::array<{input_t}, config{index}::gemm_k> a_row_t;
        typedef nnet::array<{output_t}, config{index}::gemm_n> res_row_t;

        a_row_t a_rows[config{index}::gemm_m];
        res_row_t result_rows[config{index}::gemm_m];

        PACK_A_ROWS_{index}: for (unsigned row = 0; row < config{index}::gemm_m; row++) {{
            #pragma hls_unroll
            for (unsigned kk = 0; kk < config{index}::gemm_k; kk++) {{
                #pragma hls_unroll
                a_rows[row][kk] = {input}[row * config{index}::gemm_k + kk];
            }}
        }}

        nnet::gemm_array_const_weights<a_row_t, res_row_t, config{index}>(
            a_rows, result_rows
        );

        UNPACK_C_ROWS_{index}: for (unsigned row = 0; row < config{index}::gemm_m; row++) {{
            #pragma hls_unroll
            for (unsigned col = 0; col < config{index}::gemm_n; col++) {{
                #pragma hls_unroll
                {output}[row * config{index}::gemm_n + col] =
                    result_rows[row][col] + {b}[row * config{index}::gemm_n + col];
            }}
        }}
    }}
"""


