import math

from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import Resize, Transpose, ZeroPadding1D, ZeroPadding2D

# ZeroPadding templates

zeropad1d_config_template = """struct config{index} : nnet::padding1d_config {{
    static const unsigned in_width = {in_width};
    static const unsigned n_chan = {n_chan};
    static const unsigned out_width = {out_width};
    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
}};\n"""

zeropad2d_config_template = """struct config{index} : nnet::padding2d_config {{
    static const unsigned in_height = {in_height};
    static const unsigned in_width = {in_width};
    static const unsigned n_chan = {n_chan};
    static const unsigned out_height = {out_height};
    static const unsigned out_width = {out_width};
    static const unsigned pad_top = {pad_top};
    static const unsigned pad_bottom = {pad_bottom};
    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
}};\n"""

zeropad1d_function_template = 'nnet::zeropad1d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output});'
zeropad2d_function_template = 'nnet::zeropad2d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output});'

padding_include_list = ['nnet_utils/nnet_padding.h', 'nnet_utils/nnet_padding_stream.h']


class ZeroPaddingConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__((ZeroPadding1D, ZeroPadding2D))
        self.templates = {
            'ZeroPadding1D': zeropad1d_config_template,
            'ZeroPadding2D': zeropad2d_config_template,
        }

    def format(self, node):
        params = self._default_config_params(node)
        return self.templates[node.class_name].format(**params)


class ZeroPaddingFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__((ZeroPadding1D, ZeroPadding2D), include_header=padding_include_list)
        self.templates = {
            'ZeroPadding1D': zeropad1d_function_template,
            'ZeroPadding2D': zeropad2d_function_template,
        }

    def format(self, node):
        params = self._default_function_params(node)
        params['data_format'] = 'cf' if node.get_attr('data_format') == 'channels_first' else 'cl'

        return self.templates[node.class_name].format(**params)


# Resize templates

resize_config_template = """struct config{index} : nnet::resize_config {{
    static const unsigned height = {in_height};
    static const unsigned width = {in_width};
    static const unsigned n_chan = {n_chan};
    static const unsigned new_height = {out_height};
    static const unsigned new_width = {out_width};
}};\n"""

resize_function_template = 'nnet::resize_{algorithm}<{input_t}, {config}>({input}, {output});'

resize_include_list = ['nnet_utils/nnet_image.h', 'nnet_utils/nnet_image_stream.h']


class ResizeConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Resize)
        self.template = resize_config_template

    def format(self, node):
        params = self._default_config_params(node)

        return self.template.format(**params)


class ResizeFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Resize, include_header=resize_include_list)
        self.template = resize_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['algorithm'] = node.get_attr('algorithm')

        return self.template.format(**params)


# Transpose templates

transpose_config_template = """struct config{index} : nnet::transpose_config {{
    static const unsigned depth = {depth};
    static const unsigned height = {height};
    static const unsigned width = {width};
    static constexpr unsigned perm[3] = {{{perm_str}}};
}};\n"""

transpose_function_template = 'nnet::transpose_{dim}<{input_t}, {output_t}, {config}>({input}, {output});'

# io_stream transpose: a lean config carrying only N + index_map (no
# ": nnet::transpose_config" base, so it never collides with the array
# transpose_config struct), consumed by nnet::transpose_stream.
transpose_stream_config_template = """struct config{index} {{
    static const unsigned N = {N};
    inline static constexpr unsigned index_map[{N}] = {{{index_map}}};
}};\n"""

transpose_stream_function_template = (
    'nnet::transpose_stream<{input_t}, {output_t}, {config}>({input}, {output});'
)

transpose_include_list = ['nnet_utils/nnet_array.h', 'nnet_utils/nnet_stream.h']

# For Einsum/EinsumDense
transpose_config_template_einsum = """struct {config_name} {{
    static const unsigned dims = {dims};
    static const unsigned N = {N};
    inline static constexpr unsigned from_shape[{dims}] = {{{from_shape}}};
    inline static constexpr unsigned to_shape[{dims}] = {{{to_shape}}};
    inline static constexpr unsigned perm[{dims}] = {{{perm}}};
    inline static constexpr unsigned perm_strides[{dims}] = {{{perm_strides}}};
    inline static constexpr unsigned index_map[{N}] = {{{index_map}}};
}};
"""


def catapult_transpose_config_gen(config_name, shape, perm):
    to_shape = tuple(shape[i] for i in perm)
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    perm_strides = tuple(strides[i] for i in perm)

    index_map = []
    for index in range(math.prod(shape)):
        remaining = index
        coords = [0] * len(to_shape)
        for i in range(len(to_shape) - 1, -1, -1):
            dim = to_shape[i]
            coords[i] = remaining % dim
            remaining //= dim
        input_index = 0
        for coord, perm_stride in zip(coords, perm_strides):
            input_index += coord * perm_stride
        index_map.append(input_index)

    return {
        'dims': len(shape),
        'N': math.prod(shape),
        'from_shape': ', '.join(str(x) for x in shape),
        'to_shape': ', '.join(str(x) for x in to_shape),
        'perm': ', '.join(str(x) for x in perm),
        'perm_strides': ', '.join(str(x) for x in perm_strides),
        'index_map': ', '.join(str(x) for x in index_map),
        'config_name': config_name,
    }


class TransposeConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Transpose)
        self.template = transpose_config_template

    def format(self, node):
        if node.model.config.get_config_value('IOType') == 'io_stream':
            # Streaming transpose needs the flat index_map (array config's
            # height/width/depth are meaningless once beats are reordered).
            shape = tuple(node.get_input_variable().shape)
            perm = node.get_attr('perm')
            conf = catapult_transpose_config_gen(f'config{node.index}', shape, perm)
            return transpose_stream_config_template.format(index=node.index, **conf)

        params = self._default_config_params(node)

        return self.template.format(**params)


class TransposeFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Transpose, include_header=transpose_include_list)
        self.template = transpose_function_template

    def format(self, node):
        params = self._default_function_params(node)
        if node.model.config.get_config_value('IOType') == 'io_stream':
            return transpose_stream_function_template.format(**params)
        params['dim'] = node.get_attr('dim')

        return self.template.format(**params)
