from math import ceil

from hls4ml.backends.backend import get_backend
from hls4ml.backends.fpga.einsum_utils import select_row_stream_operand
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import Einsum
from hls4ml.utils.transpose_utils import transpose_config_gen

from .reshaping_templates import transpose_config_template

# Shared Dense template
# Einsum template

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
    // io_stream + Resource: stream operand 0 one row at a time against a buffered operand 1
    // (see nnet_einsum_stream.h). True only when operand-0 and output transposes are identity.
    static const bool row_stream = {row_stream};
    static const unsigned row_stream_operand = {row_stream_operand};

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

        # Row streaming: which operand can be streamed one row at a time against the other,
        # buffered. Shared with the Catapult backend (hls4ml.backends.fpga.einsum_utils) so the
        # selection logic and its semantics are defined once.
        row_stream, row_stream_operand = select_row_stream_operand(
            io_type=io_type,
            strategy=strategy,
            n_inplace=node.attributes['n_inplace'],
            n_contract=int(node.attributes['n_contract']),
            n_free0=int(node.attributes['n_free0']),
            n_free1=int(node.attributes['n_free1']),
            inp0_shape=node.attributes['inp0_shape'],
            inp1_shape=node.attributes['inp1_shape'],
            out_shape=node.attributes['out_interpert_shape'],
            inp0_tpose_idxs=node.attributes['inp0_tpose_idxs'],
            inp1_tpose_idxs=node.attributes['inp1_tpose_idxs'],
            out_tpose_idxs=node.attributes['out_tpose_idxs'],
            in0_pack=int(node.get_input_variable(node.inputs[0]).shape[-1]),
            in1_pack=int(node.get_input_variable(node.inputs[1]).shape[-1]),
            out_pack=int(node.get_output_variable().shape[-1]),
        )
        params['row_stream'] = 'true' if row_stream else 'false'
        params['row_stream_operand'] = row_stream_operand

        einsum_conf = self.template.format(**params)

        # inp/out transpose config
        inp0_shape = node.attributes['inp0_shape']
        inp1_shape = node.attributes['inp1_shape']
        out_interpert_shape = node.attributes['out_interpert_shape']
        inp0_tpose_idxs = node.attributes['inp0_tpose_idxs']
        inp1_tpose_idxs = node.attributes['inp1_tpose_idxs']
        out_tpose_idxs = node.attributes['out_tpose_idxs']
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
