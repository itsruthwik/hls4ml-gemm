from hls4ml.backends.catapult.catapult_types import (
    CatapultArrayVariableConverter,
    CatapultInplaceArrayVariableConverter,
    CatapultInplaceStreamVariableConverter,
    CatapultStreamVariableConverter,
)
from hls4ml.backends.fpga.fpga_types import ACTypeConverter, HLSTypeConverter, StaticWeightVariableConverter
from hls4ml.model.layers import Im2Col
from hls4ml.model.optimizer import GlobalOptimizerPass
from hls4ml.model.types import InplaceTensorVariable


class TransformTypes(GlobalOptimizerPass):
    def __init__(self):
        self.type_converter = HLSTypeConverter(precision_converter=ACTypeConverter())
        self.array_var_converter = CatapultArrayVariableConverter(type_converter=self.type_converter)
        self.inplace_array_var_converter = CatapultInplaceArrayVariableConverter(type_converter=self.type_converter)
        self.stream_var_converter = CatapultStreamVariableConverter(type_converter=self.type_converter)
        self.inplace_stream_var_converter = CatapultInplaceStreamVariableConverter(type_converter=self.type_converter)
        self.weight_var_converter = StaticWeightVariableConverter(type_converter=self.type_converter)

    def transform(self, model, node):
        io_type = node.model.config.get_config_value('IOType')

        for out_name, var in node.variables.items():
            if io_type == 'io_stream':
                if isinstance(var, InplaceTensorVariable):
                    new_var = self.inplace_stream_var_converter.convert(var)
                elif isinstance(node, Im2Col) and node.get_attr('strategy') == 'gemm' and node.get_attr(
                    'im2col_tile_rows'
                ):
                    # Standalone Im2Col feeding a Gemm IP: the tile depth (rows written
                    # before the GEMM IP may backpressure) belongs on THIS channel's FIFO
                    # pragma, not the default full-n_patches depth the converter would
                    # otherwise compute from the output shape.
                    new_var = self.stream_var_converter.convert(var, depth=node.get_attr('im2col_tile_rows'))
                else:
                    new_var = self.stream_var_converter.convert(var)
            elif io_type == 'io_serial':
                new_var = self.array_var_converter.convert(var, pragma='stream')
            elif io_type == 'io_parallel':
                if out_name in node.model.inputs:
                    new_var = self.array_var_converter.convert(var, pragma='reshape')
                elif isinstance(var, InplaceTensorVariable):
                    new_var = self.inplace_array_var_converter.convert(var, pragma='')
                else:
                    new_var = self.array_var_converter.convert(var, pragma='partition')
            else:
                raise Exception(f'Unknown IOType {io_type} in {node.name} ({node.__class__.__name__})')

            node.set_attr(out_name, new_var)

        for w_name, weight in node.weights.items():
            transpose = node.model.config.get_layer_config_value(node, 'TransposeWeights', False)
            new_weight = self.weight_var_converter.convert(weight, transpose=transpose)
            node.set_attr(w_name, new_weight)

        for t_name, type in node.types.items():
            new_type = self.type_converter.convert(type)
            node.set_attr(t_name, new_type)
