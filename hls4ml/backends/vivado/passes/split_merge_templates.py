"""Vivado/Vitis HeadSplit / HeadMerge codegen templates (hls::stream / pointer-array).

The HeadSplit / HeadMerge IR node classes live in the shared FPGA base
(``backends/fpga/passes/split_merge_nodes.py``). These per-backend templates emit
``nnet::split_lanes`` / ``merge_lanes`` (io_stream) and ``nnet::split_lanes_array`` /
``merge_lanes_array`` (io_parallel), backed by the Vivado-dialect
``templates/vivado/nnet_utils/nnet_split_merge.h``. The emitted calls are the same
names as Catapult; only the runtime header dialect (hls::stream vs ac_channel) differs.
"""

from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate, stream_type_arg
from hls4ml.backends.fpga.passes.split_merge_nodes import HeadSplit, HeadMerge


split_merge_config_template = """struct config{index} {{
    static const unsigned n_beats = {seq};
    static const unsigned seq = {seq};
    static const unsigned d_model = {d_model};
    static const unsigned key_dim = {key_dim};
    static const unsigned n_heads = {n_heads};
}};\n"""


class HeadSplitConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(HeadSplit)
        self.template = split_merge_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['seq'] = node.get_attr('seq')
        params['d_model'] = node.get_attr('d_model')
        params['key_dim'] = node.get_attr('key_dim')
        params['n_heads'] = node.get_attr('n_heads')
        return self.template.format(**params)


class HeadSplitFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(HeadSplit, include_header=['nnet_utils/nnet_split_merge.h'])

    def format(self, node):
        io_type = node.model.config.get_config_value('IOType')
        inp = node.get_input_variable()
        outs = [node.get_output_variable(o) for o in node.outputs]
        out_names = ', '.join(o.name for o in outs)
        cfg = f'config{node.index}'
        if io_type == 'io_parallel':
            in_t = inp.type.precision.definition_cpp()
            out_t = outs[0].type.precision.definition_cpp()
            return f'nnet::split_lanes_array<{in_t}, {out_t}, {cfg}>({inp.name}, {out_names});'
        return f'nnet::split_lanes<{stream_type_arg(inp)}, {stream_type_arg(outs[0])}, {cfg}>({inp.name}, {out_names});'


class HeadMergeConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(HeadMerge)
        self.template = split_merge_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['seq'] = node.get_attr('seq')
        params['d_model'] = node.get_attr('d_model')
        params['key_dim'] = node.get_attr('key_dim')
        params['n_heads'] = node.get_attr('n_heads')
        return self.template.format(**params)


class HeadMergeFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(HeadMerge, include_header=['nnet_utils/nnet_split_merge.h'])

    def format(self, node):
        io_type = node.model.config.get_config_value('IOType')
        ins = [node.get_input_variable(i) for i in node.inputs]
        in_names = ', '.join(i.name for i in ins)
        out = node.get_output_variable()
        cfg = f'config{node.index}'
        if io_type == 'io_parallel':
            head_scalar = ins[0].type.precision.definition_cpp()
            out_scalar = out.type.precision.definition_cpp()
            return f'nnet::merge_lanes_array<{head_scalar}, {out_scalar}, {cfg}>({out.name}, {in_names});'
        return f'nnet::merge_lanes<{stream_type_arg(ins[0])}, {stream_type_arg(out)}, {cfg}>({out.name}, {in_names});'
