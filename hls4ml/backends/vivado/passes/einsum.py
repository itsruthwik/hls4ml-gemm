from math import ceil

from hls4ml.backends.backend import get_backend
from hls4ml.backends.fpga.einsum_utils import equation_row_plan
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import Einsum
from hls4ml.utils.transpose_utils import transpose_config_gen

from .reshaping_templates import transpose_config_template

# Shared Dense template
# Einsum template

# equation_row_plan moved to hls4ml.backends.fpga.einsum_utils so the Catapult backend can share
# it too; re-exported here (unused directly in this module beyond the import below) for anything
# still importing it from this path.

einsum_config_template = """
struct config{index} {{
    typedef config{index}_tpose_inp0 tpose_inp0_config;
    typedef config{index}_tpose_inp1 tpose_inp1_config;
    typedef config{index}_tpose_out tpose_out_conf;

    typedef {accum_t.name} accum_t;
    typedef {input1_t} weight_t;
    typedef {output_t} bias_t;

    // Layer Sizes
    static const unsigned n_free0 = {n_free0};
    static const unsigned n_free1 = {n_free1};
    static const unsigned n_contract = {n_contract};
    static const unsigned n_inplace = {n_inplace};
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned gemm_m = {gemm_m};
    static const unsigned gemm_n = {gemm_n};
    static const unsigned gemm_k = {gemm_k};
    static const unsigned gemm_ip_id = {index};
    static const bool transpose_weights = true;

    // Resource reuse info
    static const unsigned io_type = nnet::{iotype};
    static const unsigned strategy = nnet::{strategy};
    static const unsigned reuse_factor = {reuse_factor};
    static const unsigned multiplier_limit = {multiplier_limit};
    static const bool store_weights_in_bram = false; // NOT USED

    // io_stream kernel (nnet_einsum_stream.h). row_stream_op says which operand's free axis
    // supplies rows (0: rows are (i, l0), n_free1 outputs each; 1: rows are (i, l1), n_free0
    // outputs each). row_major_i_outer says whether the flattened row counter decodes as
    // i * ROW_COUNT + row_l (true) or row_l * n_inplace + i (false). Both are derived purely from
    // the equation's index letters -- see equation_row_plan() in the Vivado Einsum pass.
    static const unsigned row_stream_op = {row_stream_op};
    static const bool row_major_i_outer = {row_major_i_outer};

    template <class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};
"""

einsum_function_template = 'nnet::einsum<{input0_t}, {input1_t}, {output_t}, {config}>({input0}, {input1}, {output});'

einsum_include_list = ['nnet_utils/nnet_einsum.h', 'nnet_utils/nnet_einsum_stream.h']


class EinsumConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Einsum)
        self.template = einsum_config_template

    def format(self, node: Einsum):
        default_params = self._default_config_params(node)

        strategy = node.attributes['strategy']
        io_type = node.model.config.get_config_value('IOType')

        assert io_type in (
            'io_parallel',
            'io_stream',
        ), f'Einsum layer does not support io_type {io_type}'
        assert strategy.lower() in (
            'latency',
            'resource',
        ), f'Einsum layer does not support strategy {strategy}'

        # EinsumDense config
        params = default_params.copy()
        params['strategy'] = strategy
        params['n_free0'] = node.attributes['n_free0']
        params['n_free1'] = node.attributes['n_free1']
        params['n_contract'] = node.attributes['n_contract']
        params['n_inplace'] = node.attributes['n_inplace']
        params['n_in'] = node.get_attr('n_in', node.attributes['n_contract'])
        params['n_out'] = node.get_attr('n_out', node.attributes['n_free1'])
        params['gemm_m'] = node.get_attr('gemm_m', node.attributes['n_free0'])
        params['gemm_n'] = node.get_attr('gemm_n', node.attributes['n_free1'])
        params['gemm_k'] = node.get_attr('gemm_k', node.attributes['n_contract'])
        params['input1_t'] = node.get_input_variable(node.inputs[1]).type.name
        params['output_t'] = node.get_output_variable().type.name
        inp0_t = node.get_input_variable(node.inputs[0]).type.precision
        inp1_t = node.get_input_variable(node.inputs[1]).type.precision
        params['product_type'] = get_backend('vivado').product_type(inp0_t, inp1_t)

        total_mults = params['n_free0'] * params['n_free1'] * params['n_contract'] * params['n_inplace']
        params['multiplier_limit'] = ceil(total_mults / params['reuse_factor'])

        n_free0 = int(node.attributes['n_free0'])
        n_free1 = int(node.attributes['n_free1'])
        n_contract = int(node.attributes['n_contract'])
        out_interpert_shape = node.attributes['out_interpert_shape']
        inp0_shape = node.attributes['inp0_shape']
        inp1_shape = node.attributes['inp1_shape']
        inp0_tpose_idxs = node.attributes['inp0_tpose_idxs']
        inp1_tpose_idxs = node.attributes['inp1_tpose_idxs']
        out_tpose_idxs = node.attributes['out_tpose_idxs']

        if io_type == 'io_stream' and strategy.lower() == 'resource':
            # init_einsum already ran equation_row_plan and raised at conversion if it failed, so
            # this must succeed here.
            plan = equation_row_plan(node.attributes['equation'], inp0_shape, inp1_shape)
            params['row_stream_op'] = plan['row_op']
            params['row_major_i_outer'] = 'true' if plan['row_major_i_outer'] else 'false'
        else:
            # io_parallel and/or Latency don't use nnet_einsum_stream.h's kernel; these fields are
            # unused but must still be defined (referenced as static class members).
            params['row_stream_op'] = 0
            params['row_major_i_outer'] = 'true'

        einsum_conf = self.template.format(**params)

        # inp/out transpose config
        tpose_inp0_config_name = f'config{node.index}_tpose_inp0'
        tpose_inp1_config_name = f'config{node.index}_tpose_inp1'
        tpose_out_conf_name = f'config{node.index}_tpose_out'

        conf = transpose_config_gen(tpose_inp0_config_name, inp0_shape, inp0_tpose_idxs)
        inp0_tpose_conf = transpose_config_template.format(**conf)
        conf = transpose_config_gen(tpose_inp1_config_name, inp1_shape, inp1_tpose_idxs)
        inp1_tpose_conf = transpose_config_template.format(**conf)
        conf = transpose_config_gen(tpose_out_conf_name, out_interpert_shape, out_tpose_idxs)
        out_tpose_conf = transpose_config_template.format(**conf)

        return '\n\n'.join((inp0_tpose_conf, inp1_tpose_conf, out_tpose_conf, einsum_conf))


class EinsumFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Einsum, include_header=einsum_include_list)
        self.template = einsum_function_template

    def format(self, node: Einsum):
        params = {}
        params['config'] = f'config{node.index}'
        params['input0_t'] = node.get_input_variable(node.inputs[0]).type.name
        params['input1_t'] = node.get_input_variable(node.inputs[1]).type.name
        params['output_t'] = node.get_output_variable().type.name
        params['input0'] = node.get_input_variable(node.inputs[0]).name
        params['input1'] = node.get_input_variable(node.inputs[1]).name
        params['output'] = node.get_output_variable().name
        # A gemm_ip Einsum never reaches this template: LowerEinsumToGemm lowers it to a
        # Gemm node in the IR before templates run (GEMM lives in the IR now). This
        # template only serves the baseline einsum path.
        return self.template.format(**params)
