"""Vivado/Vitis GEMM template implementations.

Provides LayerConfigTemplate and FunctionCallTemplate classes for:
  - Im2Col (with GEMM IP row/column streaming)
  - GemmStream (row/column GEMM IP for Dense)
  - Im2ColGemmStream (fused Conv im2col + GEMM IP)

The GEMM path uses the row/column streaming contract:
  A stream: one full K-wide row per cycle
  B stream: one full K-wide column per cycle (pre-packed ROM)
  C stream: one full N-wide row per cycle
  M = n_patches, K = n_in, N = n_out

Key differences from Catapult:
  - Uses hls::stream instead of ac_channel.
  - Uses Xilinx pragmas (#pragma HLS STREAM / DATAFLOW / PIPELINE).
  - get_backend('vivado') used for product_type.
  - No #pragma hls_design block / #pragma hls_fifo_depth.
"""

from hls4ml.backends.gemm_ip_config import GemmIPConfigTemplateBase
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.backends.fpga.passes.gemm_nodes import Im2Col, Gemm, Im2ColGemm

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
}};\n"""

im2col_function_template = 'nnet::im2col_{n_dim}d_stream<{input_t}, {output_t}, {config}>({input}, {output});'
im2col_gemm_rows_function_template = (
    'nnet::im2col_{n_dim}d_gemm_rows<{input_t}, {output_t}, {config}>({input}, {output});'
)


class Im2ColConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Im2Col)
        self.template = im2col_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['gemm_m'] = node.get_attr('gemm_m', 1)
        return self.template.format(**params)


class Im2ColFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Im2Col, include_header=['nnet_utils/nnet_im2col_stream.h'])
        self.template = im2col_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['n_dim'] = 2 if node.get_attr('in_height', 1) > 1 or node.get_attr('filt_height', 1) > 1 else 1
        if node.get_attr('strategy') == 'gemm':
            return im2col_gemm_rows_function_template.format(**params)
        return self.template.format(**params)


# ---------------------------------------------------------------------------
# GemmStream templates (Dense GEMM IP — row/column streaming)
# ---------------------------------------------------------------------------

gemm_weightless_config_template = """struct config{index} : nnet::gemm_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned n_patches = {n_patches};
    static const unsigned gemm_m = {gemm_m};
    static const unsigned gemm_k = {n_in};
    static const unsigned gemm_n = {n_out};
    static const unsigned gemm_ip_id = {index};
    static const bool transpose_weights = true;
    // Microarchitecture knobs consumed by the generic (behavioral-HLS) GEMM core:
    // row-loop pipeline II and the multiplier ALLOCATION cap (mirrors nnet_dense_*).
    static const unsigned reuse_factor = {reuse_factor};
    static const unsigned multiplier_limit = {multiplier_limit};
    typedef {weight_t.name} weight_t;
    typedef {bias_t.name} bias_t;
    typedef {accum_t.name} accum_t;
    template<class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};\n"""

gemm_stream_packed_function_template = (
    'nnet::gemm_stream_weightless<{input_t}, {output_t}, {config}>'
    '({input}, {output}, {b});'
)


gemm_array_function_template = """
    {{
        typedef nnet::array<{input_t}, config{index}::gemm_k> a_row_t;
        typedef nnet::array<{output_t}, config{index}::gemm_n> res_row_t;

        a_row_t a_rows[config{index}::gemm_m];
        res_row_t result_rows[config{index}::gemm_m];
        #pragma HLS ARRAY_PARTITION variable=a_rows complete
        #pragma HLS ARRAY_PARTITION variable=result_rows complete

        PACK_A_ROWS_{index}: for (unsigned row = 0; row < config{index}::gemm_m; row++) {{
            #pragma HLS UNROLL
            for (unsigned kk = 0; kk < config{index}::gemm_k; kk++) {{
                #pragma HLS UNROLL
                a_rows[row][kk] = {input}[row * config{index}::gemm_k + kk];
            }}
        }}

        nnet::gemm_array_weightless<a_row_t, {bias_t}, res_row_t, config{index}>(
            a_rows, result_rows, {b}
        );

        UNPACK_C_ROWS_{index}: for (unsigned row = 0; row < config{index}::gemm_m; row++) {{
            #pragma HLS UNROLL
            for (unsigned col = 0; col < config{index}::gemm_n; col++) {{
                #pragma HLS UNROLL
                {output}[row * config{index}::gemm_n + col] = result_rows[row][col];
            }}
        }}
    }}
"""


gemm_array_row_bias_function_template = """
    {{
        typedef nnet::array<{input_t}, config{index}::gemm_k> a_row_t;
        typedef nnet::array<{output_t}, config{index}::gemm_n> res_row_t;

        a_row_t a_rows[config{index}::gemm_m];
        res_row_t result_rows[config{index}::gemm_m];
        #pragma HLS ARRAY_PARTITION variable=a_rows complete
        #pragma HLS ARRAY_PARTITION variable=result_rows complete

        // Row-varying EinsumDense bias: the per-column GEMM-IP bias port cannot
        // express a bias that varies across the M rows. Feed the core a zero
        // per-column bias and add the full per-element bias in the unpack loop.
        {bias_t} {output}_zero_bias[config{index}::gemm_n];
        for (unsigned col = 0; col < config{index}::gemm_n; col++) {{
            #pragma HLS UNROLL
            {output}_zero_bias[col] = 0;
        }}

        PACK_A_ROWS_{index}: for (unsigned row = 0; row < config{index}::gemm_m; row++) {{
            #pragma HLS UNROLL
            for (unsigned kk = 0; kk < config{index}::gemm_k; kk++) {{
                #pragma HLS UNROLL
                a_rows[row][kk] = {input}[row * config{index}::gemm_k + kk];
            }}
        }}

        nnet::gemm_array_weightless<a_row_t, {bias_t}, res_row_t, config{index}>(
            a_rows, result_rows, {output}_zero_bias
        );

        UNPACK_C_ROWS_{index}: for (unsigned row = 0; row < config{index}::gemm_m; row++) {{
            #pragma HLS UNROLL
            for (unsigned col = 0; col < config{index}::gemm_n; col++) {{
                #pragma HLS UNROLL
                {output}[row * config{index}::gemm_n + col] =
                    result_rows[row][col] + {b}[row * config{index}::gemm_n + col];
            }}
        }}
    }}
"""


# ---------------------------------------------------------------------------
# Two-operand Gemm (attention QK^T / A.V): both operands are activations, no
# constant weight ROM. weight_t is the B operand's type (product SFINAE only);
# bias is a local zero array. n_inplace batches the per-head GEMMs.
# ---------------------------------------------------------------------------
gemm_two_operand_config_template = """struct config{index} {{
    static const unsigned n_in = {gemm_k};
    static const unsigned n_out = {gemm_n};
    static const unsigned n_patches = {gemm_m};
    static const unsigned gemm_m = {gemm_m};
    static const unsigned gemm_k = {gemm_k};
    static const unsigned gemm_n = {gemm_n};
    static const unsigned gemm_ip_id = {index};
    static const unsigned n_inplace = {n_inplace};
    // The generated two-operand core expects the B operand transposed.
    static const bool transpose_weights = true;
    // B beat layout: false = col-major (K-wide beats, one output column/beat, default);
    // true = row-major (N-wide beats, one contraction row/beat) for the mvau IP.
    static const bool b_row_major = {b_row_major};
    static const unsigned reuse_factor = {reuse_factor};
    static const unsigned multiplier_limit = {multiplier_limit};
    typedef {input1_t} weight_t;
    typedef {accum_t.name} accum_t;
    // Two-operand GEMM carries no bias; the cell still takes a bias array, fed a
    // zero. bias_t must be a SCALAR addable to accum_t.
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

        config{index}::bias_t {output}_zero_bias[config{index}::gemm_n];
        for (unsigned n = 0; n < config{index}::gemm_n; n++) {{
            #pragma HLS UNROLL
            {output}_zero_bias[n] = 0;
        }}

        for (unsigned i = 0; i < config{index}::n_inplace; i++) {{
            #pragma HLS UNROLL
            a_row_t a_rows[config{index}::gemm_m];
            b_col_t b_cols[config{index}::gemm_n];
            res_row_t result_rows[config{index}::gemm_m];
            for (unsigned l0 = 0; l0 < config{index}::gemm_m; l0++) {{
                for (unsigned c = 0; c < config{index}::gemm_k; c++) {{
                    #pragma HLS UNROLL
                    a_rows[l0][c] = {input0}[(i * config{index}::gemm_m + l0) * config{index}::gemm_k + c];
                }}
            }}
            for (unsigned l1 = 0; l1 < config{index}::gemm_n; l1++) {{
                for (unsigned c = 0; c < config{index}::gemm_k; c++) {{
                    #pragma HLS UNROLL
                    b_cols[l1][c] = {input1}[{b_index}];
                }}
            }}
            nnet::gemm_array<a_row_t, b_col_t, config{index}::bias_t, res_row_t, config{index}>(
                a_rows, b_cols, result_rows, {output}_zero_bias
            );
            for (unsigned l0 = 0; l0 < config{index}::gemm_m; l0++) {{
                for (unsigned l1 = 0; l1 < config{index}::gemm_n; l1++) {{
                    #pragma HLS UNROLL
                    {output}[(i * config{index}::gemm_m + l0) * config{index}::gemm_n + l1] = result_rows[l0][l1];
                }}
            }}
        }}
    }}
"""


# io_stream: both operands are hls::streams in canonical head-ordered layout;
# call the two-stream core once per head (unrolled).
gemm_stream_two_op_function_template = """
    {{
        config{index}::bias_t {output}_zero_bias[config{index}::gemm_n];
        for (unsigned n = 0; n < config{index}::gemm_n; n++) {{
            #pragma HLS UNROLL
            {output}_zero_bias[n] = 0;
        }}
        for (unsigned i = 0; i < config{index}::n_inplace; i++) {{
            #pragma HLS UNROLL
            nnet::gemm_stream<{input0_t}, {input1_t}, {output_t}, config{index}>(
                {input0}, {input1}, {output}, {output}_zero_bias
            );
        }}
    }}
"""


def _format_two_operand(node):
    """Build the bare two-operand GEMM config (both operands activations, no weight
    ROM): product/weight_t come from the two activation inputs (attention QK^T / A.V)."""
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
        'b_row_major': 'true' if node.get_attr('second_operand_row_major', False) else 'false',
        'reuse_factor': rf,
        'multiplier_limit': -(-(gk * gn) // rf),
        'input1_t': inp1.type.name,
        'accum_t': node.types['accum_t'],
        'product_type': get_backend('vivado').product_type(inp0.type.precision, inp1.type.precision),
    }
    return gemm_two_operand_config_template.format(**params)


def _inject_weight_rom_accessor(cfg, node):
    """Embed the const-weight ROM accessor into a GEMM config struct so the
    weightless cores can source the constant operand from the config (matching
    Catapult), keeping the weight off the call signature. The ROM header is
    #include'd above the config in parameters.h, so unqualified lookup resolves here.
    """
    w = node.get_weights('weight').name
    inject = (
        '    typedef nnet::array<weight_t, gemm_k> weight_col_t;\n'
        f'    static weight_col_t* gemm_weight_cols() {{ return {w}_gemm_cols; }}\n'
    )
    stripped = cfg.rstrip()
    assert stripped.endswith('};'), 'unexpected gemm config layout'
    return stripped[:-2] + inject + '};\n'


class GemmConfigTemplate(GemmIPConfigTemplateBase):
    """One config for the unified Gemm node (Vivado). Uses the gemm_config
    base for both interfaces (the array path tolerates the inherited defaults).
    Embeds the weight-ROM accessor so the weightless cores source constant weights
    from the config instead of the call site."""
    backend_name = 'vivado'

    def __init__(self):
        super().__init__(Gemm)
        self.template = gemm_weightless_config_template

    def format(self, node):
        if not node.get_attr('weights_in_core', True):
            # Two-operand Gemm: no weight ROM; product/weight_t come from the two
            # activation inputs (attention QK^T / A.V).
            return _format_two_operand(node)
        return _inject_weight_rom_accessor(super().format(node), node)


class GemmFunctionTemplate(FunctionCallTemplate):
    """One function template for the unified Gemm node (Vivado).

    Dispatches the 2x2 (IOType x weights_in_core) among the four gemm_* cores:
    gemm_array_weightless / gemm_stream_weightless (const weight held by the IP) and
    gemm_array / gemm_stream (two activation operands, attention QK^T / A.V).
    """

    def __init__(self):
        super().__init__(
            Gemm,
            include_header=['nnet_utils/nnet_gemm_stream.h', 'nnet_utils/nnet_gemm_ip.h'],
        )
        self.template = gemm_array_function_template

    def format(self, node):
        io_type = node.model.config.get_config_value('IOType')

        if not node.get_attr('weights_in_core', True):
            # Two-operand GEMM (attention QK^T / A.V): both operands are activations,
            # no constant weight/bias to reference.
            inp0 = node.get_input_variable(node.inputs[0])
            inp1 = node.get_input_variable(node.inputs[1])
            out_var = node.get_output_variable()
            idx = node.index
            # io_parallel B flat-array index: col-major B is [I, L1, C] (default),
            # row-major B is [I, C, L1] (SecondOperandRowMajor) -- the transpose that made
            # B N-inner also flips how the flat array is addressed here.
            if node.get_attr('second_operand_row_major', False):
                b_index = f'(i * config{idx}::gemm_k + c) * config{idx}::gemm_n + l1'
            else:
                b_index = f'(i * config{idx}::gemm_n + l1) * config{idx}::gemm_k + c'
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
                'b_index': b_index,
            }
            if io_type == 'io_parallel':
                return gemm_array_two_op_function_template.format(**two_op)
            return gemm_stream_two_op_function_template.format(**two_op)

        params = self._default_function_params(node)
        params['w'] = node.get_weights('weight').name
        params['b'] = node.get_weights('bias').name
        params['n_out'] = node.get_attr('n_out')
        params['weight_t'] = node.get_weights('weight').type.name
        params['weight_t_name'] = node.get_weights('weight').type.name
        params['bias_t'] = node.get_weights('bias').type.name
        if io_type == 'io_parallel':
            if node.get_attr('_row_varying_bias', False):
                # EinsumDense bias that varies across the M rows: zero per-column
                # bias into the core, full per-element bias added in the unpack loop.
                return gemm_array_row_bias_function_template.format(**params)
            return gemm_array_function_template.format(**params)
        if node.get_attr('_row_varying_bias', False):
            # The per-element bias-add wrapper is only wired for io_parallel; the
            # io_stream packed path has no place to add a row-varying bias without
            # silently dropping the per-row component. EinsumDense GEMM-IP is
            # io_parallel-gated so this is unreachable in practice, but fail loudly
            # rather than emit a wrong answer (mirrors Catapult).
            raise NotImplementedError(
                f"Gemm '{node.name}': row-varying bias is not supported on the io_stream GEMM-IP "
                "path. Use io_parallel for EinsumDense layers whose bias varies across rows."
            )
        return gemm_stream_packed_function_template.format(**params)


# ---------------------------------------------------------------------------
# Im2ColGemm templates (fused Conv im2col + GEMM IP)
# ---------------------------------------------------------------------------

im2col_gemm_stream_config_template = """struct config{index}_im2col : nnet::im2col_config {{
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
}};

struct config{index}_gemm : nnet::gemm_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned n_patches = {n_patches};
    static const unsigned gemm_m = {gemm_m};
    static const unsigned gemm_k = {n_in};
    static const unsigned gemm_n = {n_out};
    static const unsigned gemm_ip_id = {index};
    static const bool transpose_weights = true;
    static const unsigned reuse_factor = {reuse_factor};
    static const unsigned multiplier_limit = {multiplier_limit};
    typedef {weight_t.name} weight_t;
    typedef {bias_t.name} bias_t;
    typedef {accum_t.name} accum_t;
    template<class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};\n"""

# Row/column streaming fused conv: im2col emits one K-wide A row per output pixel,
# then the weightless io_stream GEMM core (gemm_stream_weightless) contracts it against
# the constant kernel columns sourced from the config ROM. One behavioral / synth / cosim
# definition per the four-name contract in nnet_gemm_ip.h.
im2col_gemm_stream_function_template = """
    {{
        typedef nnet::array<{input_scalar_t}, config{index}_gemm::gemm_k> a_row_t;

        static hls::stream<a_row_t> activation_rows("activation_rows_{index}");
        #pragma HLS STREAM variable=activation_rows depth=2

        // im2col emits one K-wide A row per output pixel; the weightless GEMM core
        // sources the constant kernel columns from the config ROM (no weight arg),
        // matching the Dense/EinsumDense weightless path and Catapult's fused conv.
        nnet::im2col_{n_dim}d_gemm_rows<{input_t}, a_row_t, config{index}_im2col>({input}, activation_rows);
        nnet::gemm_stream_weightless<a_row_t, {result_t}, config{index}_gemm>(
            activation_rows, {output}, {b}
        );
    }}
"""


# io_parallel counterpart of the streaming fused template: the input is a flat
# array, so an array-interface im2col materialises a_rows[gemm_m] (no hls::stream),
# then the weightless array core is called.
im2col_gemm_array_function_template = """
    {{
        typedef nnet::array<{input_t}, config{index}_gemm::gemm_k> a_row_t;
        typedef nnet::array<{output_t}, config{index}_gemm::gemm_n> res_row_t;

        a_row_t a_rows[config{index}_gemm::gemm_m];
        res_row_t result_rows[config{index}_gemm::gemm_m];

        nnet::im2col_{n_dim}d_gemm_rows_array<{input_t}, a_row_t, config{index}_im2col>({input}, a_rows);

        nnet::gemm_array_weightless<a_row_t, {bias_t}, res_row_t, config{index}_gemm>(
            a_rows, result_rows, {b}
        );

        UNPACK_C_ROWS_{index}: for (unsigned row = 0; row < config{index}_gemm::gemm_m; row++) {{
            #pragma HLS UNROLL
            for (unsigned col = 0; col < config{index}_gemm::gemm_n; col++) {{
                #pragma HLS UNROLL
                {output}[row * config{index}_gemm::gemm_n + col] = result_rows[row][col];
            }}
        }}
    }}
"""


class Im2ColGemmConfigTemplate(GemmIPConfigTemplateBase):
    # One config for the fused Im2ColGemm node (Vivado). Embeds the weight-ROM
    # accessor into config{index}_gemm so the weightless core sources the constant
    # kernel from the config (the conv kernel is always in-core).
    backend_name = 'vivado'

    def __init__(self):
        super().__init__(Im2ColGemm)
        self.template = im2col_gemm_stream_config_template

    def format(self, node):
        return _inject_weight_rom_accessor(super().format(node), node)


class Im2ColGemmFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(
            Im2ColGemm,
            include_header=[
                'nnet_utils/nnet_im2col.h',
                'nnet_utils/nnet_im2col_stream.h',
                'nnet_utils/nnet_gemm_stream.h',
                'nnet_utils/nnet_gemm_ip.h',
            ],
        )
        self.template = im2col_gemm_stream_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['n_dim'] = 2 if node.get_attr('in_height', 1) > 1 or node.get_attr('filt_height', 1) > 1 else 1
        params['w'] = node.get_weights('weight').name
        params['b'] = node.get_weights('bias').name
        params['weight_t'] = node.get_weights('weight').type.name
        params['bias_t'] = node.get_weights('bias').type.name
        params['input_scalar_t'] = node.get_input_variable().type.precision.definition_cpp()
        params['result_t'] = node.get_output_variable(node.outputs[0]).type.name
        if node.model.config.get_config_value('IOType') == 'io_parallel':
            return im2col_gemm_array_function_template.format(**params)
        return im2col_gemm_stream_function_template.format(**params)


def register_gemm_templates(backend):
    """Explicitly register GEMM IP config/function templates for *backend*.

    Auto-discovery via ``extract_optimizers_from_path`` would find these
    template classes anyway, but explicit registration follows the established
    backend pattern and avoids ambiguity when multiple pass modules define
    templates for the same layer type.
    """
    backend.register_pass('gemm_config_template', GemmConfigTemplate)
    backend.register_pass('gemm_function_template', GemmFunctionTemplate)
    backend.register_pass('im2colgemm_config_template', Im2ColGemmConfigTemplate)
    backend.register_pass('im2colgemm_function_template', Im2ColGemmFunctionTemplate)
    backend.register_pass('im2col_config_template', Im2ColConfigTemplate)
    backend.register_pass('im2col_function_template', Im2ColFunctionTemplate)
