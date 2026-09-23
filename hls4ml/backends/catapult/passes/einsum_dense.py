import warnings

from hls4ml.backends.backend import get_backend
from hls4ml.backends.fpga.einsum_utils import equation_row_plan
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import EinsumDense

from .reshaping_templates import catapult_transpose_config_gen, transpose_config_template_einsum

# EinsumDense template

einsum_dense_config_template = """
struct config{index} {{
    typedef config{index}_tpose_inp tpose_inp_conf;
    typedef config{index}_tpose_out tpose_out_conf;

    typedef {accum_t.name} accum_t;
    typedef {bias_t.name} bias_t;
    typedef {weight_t.name} weight_t;

    {kernel_config};

    // Layer Sizes
    static const unsigned n_free_data = {n_free_data};
    static const unsigned n_free_kernel = {n_free_kernel};
    static const unsigned n_contract = {n_contract};
    static const unsigned n_inplace = {n_inplace};
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned gemm_m = {gemm_m};
    static const unsigned gemm_k = {gemm_k};
    static const unsigned gemm_n = {gemm_n};
    static const unsigned gemm_ip_id = {index};

    // Resource reuse info
    static const unsigned io_type = nnet::{iotype};
    static const unsigned strategy = nnet::{strategy};
    static const unsigned reuse_factor = {reuse_factor};
    static const unsigned parallelization_factor = {parallelization_factor}; // Only useful when n_inplace > 1
    static const unsigned n_zeros = {nzeros};
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor);

    // io_stream kernel (nnet_einsum_dense_stream.h). row_major_i_outer says whether the flattened
    // row counter decodes as i * n_free_data + l0 (true) or l0 * n_inplace + i (false); derived
    // purely from the equation's index letters -- see equation_row_plan() in
    // hls4ml.backends.fpga.einsum_utils.
    static const bool row_major_i_outer = {row_major_i_outer};

    template<class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};
"""

dense_config_template = """struct config{index}_dense : nnet::dense_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned gemm_m = {gemm_m};
    static const unsigned gemm_k = {gemm_k};
    static const unsigned gemm_n = {gemm_n};
    static const unsigned gemm_ip_id = {index};
    static const unsigned reuse_factor = {reuse};
    static const unsigned strategy = nnet::{strategy};
    static const unsigned n_zeros = {nzeros};
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef {accum_t.name} accum_t;
    typedef {bias_t.name} bias_t;
    typedef {weight_t.name} weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};\n"""

einsum_dense_function_template = 'nnet::einsum_dense<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'

einsum_dense_include_list = [
    'nnet_utils/nnet_einsum_dense.h',
    'nnet_utils/nnet_dense.h',
    'nnet_utils/nnet_einsum_stream.h',
    'nnet_utils/nnet_einsum_dense_stream.h',
]


class EinsumDenseConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(EinsumDense)
        self.template = einsum_dense_config_template
        self.dense_template = dense_config_template

    def dense_config(self, node: EinsumDense):
        dense_params = self._default_config_params(node)
        strategy = node.attributes['strategy']
        dense_params['strategy'] = strategy
        dense_params['n_in'] = node.attributes['n_contract']
        dense_params['n_out'] = node.attributes['n_free_kernel']
        dense_params['gemm_m'] = node.get_attr('gemm_m', node.attributes['n_free_data'])
        dense_params['gemm_k'] = node.get_attr('gemm_k', node.attributes['n_contract'])
        dense_params['gemm_n'] = node.get_attr('gemm_n', node.attributes['n_free_kernel'])
        if node.attributes['n_inplace'] == 1:
            dense_params['nzeros'] = node.get_weights('weight').nzeros  # type: ignore
        else:
            dense_params['nzeros'] = '-1; // Not making sense when kernels are switching'
        
        dense_params['product_type'] = get_backend('catapult').product_type(
            node.get_input_variable().type.precision,
            node.get_weights('weight').type.precision,  # type: ignore
        )

        dense_config = self.dense_template.format(**dense_params)
        return dense_config

    def format(self, node: EinsumDense):
        default_params = self._default_config_params(node)

        strategy = node.attributes['strategy']
        io_type = node.model.config.get_config_value('IOType')

        # Only the baseline (non-GEMM) EinsumDense reaches this template — the GEMM strategy
        # lowers EinsumDense to a Gemm node in LowerEinsumToGemm before templating. init_einsum_dense
        # (catapult_backend.py) already rejected io_stream + Latency at conversion, so any
        # io_type/strategy combination that reaches here is one the io_stream row kernel
        # (nnet_einsum_dense_stream.h) or the io_parallel array core can build. This matches the
        # Vivado/Vitis backend's accepted set exactly.
        assert io_type in ('io_parallel', 'io_stream'), f'EinsumDense layer does not support io_type {io_type}'
        assert strategy.lower() in ('latency', 'resource'), (
            'EinsumDense layer only supports Latency and Resource strategies for now'
        )

        # EinsumDense config
        params = default_params.copy()
        params['strategy'] = strategy
        params['n_free_data'] = node.attributes['n_free_data']
        params['n_free_kernel'] = node.attributes['n_free_kernel']
        params['n_contract'] = node.attributes['n_contract']
        params['n_inplace'] = node.attributes['n_inplace']
        params['n_in'] = node.get_attr('n_in', node.attributes['n_contract'])
        params['n_out'] = node.get_attr('n_out', node.attributes['n_free_kernel'])
        params['gemm_m'] = node.get_attr('gemm_m', node.attributes['n_free_data'])
        params['gemm_k'] = node.get_attr('gemm_k', node.attributes['n_contract'])
        params['gemm_n'] = node.get_attr('gemm_n', node.attributes['n_free_kernel'])
        params['product_type'] = get_backend('catapult').product_type(
            node.get_input_variable().type.precision,
            node.get_weights('weight').type.precision,
        )
        if node.attributes['n_inplace'] == 1:
            params['nzeros'] = node.get_weights('weight').nzeros
        else:
            params['nzeros'] = '-1'
        
        if strategy.lower() == 'latency' and params.get('reuse_factor', 1) > 1:
            warnings.warn(
                f"EinsumDense layer '{node.name}': ReuseFactor={params['reuse_factor']} is ignored on the "
                'Catapult backend — pragmas cannot take template-dependent II values and the einsum '
                'loops are fully unrolled. Use Catapult TCL directives to constrain resources instead.',
                stacklevel=2,
            )

        params['kernel_config'] = f'typedef config{node.index}_dense dense_conf'
        
        pf = node.attributes['parallelization_factor']
        if pf < 0:
            pf = params['n_inplace']
        params['parallelization_factor'] = pf

        if io_type == 'io_stream' and strategy.lower() == 'resource':
            # init_einsum_dense already ran equation_row_plan and raised at conversion if it
            # failed (and required row_op == 0: the data operand supplies rows), so this must
            # succeed here.
            plan = equation_row_plan(
                node.attributes['equation'], node.attributes['inp_shape'], node.attributes['kernel_shape']
            )
            params['row_major_i_outer'] = 'true' if plan['row_major_i_outer'] else 'false'
        else:
            # io_parallel and/or Latency don't use nnet_einsum_dense_stream.h's kernel; this
            # field is unused but must still be defined (referenced as a static class member).
            params['row_major_i_outer'] = 'true'

        einsum_conf = self.template.format(**params)

        # inp/out transpose config
        inp_shape = node.attributes['inp_shape']
        out_interpert_shape = node.attributes['out_interpert_shape']
        inp_tpose_idxs = node.attributes['inp_tpose_idxs']
        out_tpose_idxs = node.attributes['out_tpose_idxs']
        tpose_inp_conf_name = f'config{node.index}_tpose_inp'
        tpose_out_conf_name = f'config{node.index}_tpose_out'

        conf = catapult_transpose_config_gen(tpose_inp_conf_name, inp_shape, inp_tpose_idxs)
        inp_tpose_conf = transpose_config_template_einsum.format(**conf)
        conf = catapult_transpose_config_gen(tpose_out_conf_name, out_interpert_shape, out_tpose_idxs)
        out_tpose_conf = transpose_config_template_einsum.format(**conf)

        dense_config = self.dense_config(node)
        return '\n\n'.join((inp_tpose_conf, out_tpose_conf, dense_config, einsum_conf))


class EinsumDenseFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(EinsumDense, include_header=einsum_dense_include_list)
        self.template = einsum_dense_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['b'] = node.get_weights('bias').name
        params['w'] = node.get_weights('weight').name
        # A gemm_ip EinsumDense never reaches this template — LowerEinsumToGemm
        # lowers it to a Gemm node. Only the baseline kernel remains, and the call
        # site is identical for io_parallel and io_stream: overload resolution on
        # {input_t}/{output_t} (array vs ac_channel) picks the array core in
        # nnet_einsum_dense.h or the stream shell in nnet_einsum_dense_stream.h.
        return self.template.format(**params)


def register_einsum_dense(backend):
    # Register template passes
    backend.register_template(EinsumDenseConfigTemplate)
    backend.register_template(EinsumDenseFunctionTemplate)
