import os
import sys
from warnings import warn

import numpy as np

from hls4ml.backends import FPGABackend
from hls4ml.backends.catapult.catapult_types import CatapultArrayVariableConverter
from hls4ml.backends.gemm_ip_config import is_gemm_strategy
from hls4ml.backends.fpga.fpga_types import ACTypeConverter, HLSTypeConverter
from hls4ml.model.attributes import ChoiceAttribute, ConfigurableAttribute, TypeAttribute
from hls4ml.model.flow import register_flow
from hls4ml.model.layers import (
    GRU,
    LSTM,
    Conv1D,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Einsum,
    EinsumDense,
    Embedding,
    GarNet,
    GarNetStack,
    GlobalPooling1D,
    GlobalPooling2D,
    Layer,
    LayerNormalization,
    Pooling1D,
    Pooling2D,
    SeparableConv1D,
    SeparableConv2D,
    SimpleRNN,
)
from hls4ml.backends.catapult.passes import (
    TransposeWeightsForGemmIP,
    SplitConvGemm,
    ReplaceDenseGemm,
    LowerEinsumToGemm,
    ValidateGemm,
)
from hls4ml.model.optimizer import get_backend_passes, layer_optimizer
from hls4ml.model.types import (
    FixedPrecisionType,
    IntegerPrecisionType,
    NamedType,
    PackedType,
    RoundingMode,
    SaturationMode,
)
from hls4ml.report import parse_catapult_report
from hls4ml.utils import attribute_descriptions as descriptions
from hls4ml.utils.einsum_utils import parse_einsum
from hls4ml.utils.fixed_point_utils import ceil_log2


class CatapultBackend(FPGABackend):
    def __init__(self):
        super().__init__('Catapult')
        self._register_layer_attributes()
        try:
            self.register_pass('transpose_weights_for_gemm', TransposeWeightsForGemmIP)
        except Exception:
            pass
        for name, cls in [
            ('split_conv_gemm', SplitConvGemm),
            ('replace_dense_gemm', ReplaceDenseGemm),
            ('lower_einsum_to_gemm', LowerEinsumToGemm),
            ('validate_gemm', ValidateGemm),
        ]:
            try:
                self.register_pass(name, cls)
            except Exception:
                pass
        self._register_flows()

    def _register_layer_attributes(self):
        # Add RNN-specific attributes, recurrent_reuse_factor and static implementation
        rnn_layers = [
            SimpleRNN,
            LSTM,
            GRU,
        ]

        for layer in rnn_layers:
            attrs = self.attribute_map.get(layer, [])
            attrs.append(ConfigurableAttribute('recurrent_reuse_factor', default=1, description=descriptions.reuse_factor))
            attrs.append(
                ConfigurableAttribute('static', value_type=bool, default=True, description=descriptions.recurrent_static)
            )
            attrs.append(ConfigurableAttribute('table_size', default=1024, description=descriptions.table_size))
            attrs.append(TypeAttribute('table', default=FixedPrecisionType(18, 8), description=descriptions.table_type))
            self.attribute_map[layer] = attrs

        # Add ParallelizationFactor to Conv1D/2D
        pf_layers = [
            Conv1D,
            Conv2D,
        ]

        for layer in pf_layers:
            attrs = self.attribute_map.get(layer, [])
            attrs.append(ConfigurableAttribute('parallelization_factor', default=1, description=descriptions.conv_pf))
            self.attribute_map[layer] = attrs

        # Add ConvImplementation to Convolution+Pooling layers
        cnn_layers = [Conv1D, Conv2D, SeparableConv1D, SeparableConv2D, DepthwiseConv2D, Pooling1D, Pooling2D]

        # "LineBuffer_reg" is a Conv2D-only variant holding the per-row delay lines in
        # register arrays instead of ap_shift_reg, at the cost of
        # in_width*n_chan*(filt_height-1) registers + a shift mux. Only the Conv2D
        # line-buffer path (conv_2d_cl) dispatches it; the default stays "LineBuffer".
        conv2d_impl_desc = (
            '"LineBuffer" (ap_shift_reg, default) is preferred for FPGA targets where the rolled shift '
            'infers an SRL. "LineBuffer_reg" uses a fully-unrolled register line buffer (RecII=1 feed, '
            'no SRL dependence) — useful on Catapult/ASIC flows where ap_shift_reg serializes. '
            '"Encoded" is the alternative streaming scheme. This attribute only applies to io_stream.'
        )

        for layer in cnn_layers:
            attrs = self.attribute_map.get(layer, [])
            if layer is Conv2D:
                attrs.append(
                    ChoiceAttribute(
                        'conv_implementation',
                        choices=['LineBuffer', 'LineBuffer_reg', 'Encoded'],
                        default='LineBuffer',
                        description=conv2d_impl_desc,
                    )
                )
            else:
                attrs.append(
                    ChoiceAttribute(
                        'conv_implementation',
                        choices=['LineBuffer', 'Encoded'],
                        default='LineBuffer',
                        description=descriptions.conv_implementation,
                    )
                )
            self.attribute_map[layer] = attrs

        sep_conv_layers = [SeparableConv1D, SeparableConv2D]
        for layer in sep_conv_layers:
            attrs = self.attribute_map.get(layer, [])
            attrs.append(TypeAttribute('dw_output', default=FixedPrecisionType(18, 8)))
            self.attribute_map[layer] = attrs

        # Add LayerNorm attributes
        ln_layers = [LayerNormalization]
        for layer in ln_layers:
            attrs = self.attribute_map.get(layer, [])
            attrs.append(ConfigurableAttribute('table_range_power2', default=0, description=descriptions.table_range_power2))
            attrs.append(ConfigurableAttribute('table_size', default=4096, description=descriptions.table_size))
            attrs.append(
                TypeAttribute(
                    'table',
                    default=FixedPrecisionType(
                        8, 5, signed=False, rounding_mode=RoundingMode.RND_CONV, saturation_mode=SaturationMode.SAT
                    ),
                    description=descriptions.table_type,
                )
            )
            attrs.append(
                TypeAttribute(
                    'accum',
                    # 24,8 (16 fractional bits, up from 10): the plain (non-HGQ2) LN path has no
                    # register_precision sizing, so this default accum_t is what the kernel's
                    # `/(int)dim` divide runs in. A narrow accum_t loses up to 1 LSB of the
                    # quotient (the fixed-point divide truncates internally before the
                    # destination's rounding mode applies), which is fatal for small-variance
                    # tokens where the variance itself is only a few LSBs wide. Widening accum_t
                    # instead of reworking the kernel's arithmetic keeps the divide -- required
                    # for HGQ2 bit-exactness at non-power-of-2 dim -- and keeps it synthesizable
                    # (a wide-intermediate divide was tried and failed Catapult C/RTL synthesis:
                    # no library divider component for the resulting ~80-bit quotient).
                    default=FixedPrecisionType(
                        24, 8, signed=True, rounding_mode=RoundingMode.RND_CONV, saturation_mode=SaturationMode.SAT
                    ),
                    description=descriptions.accum_type,
                )
            )
            self.attribute_map[layer] = attrs

    def _register_flows(self):
        initializers = self._get_layer_initializers()
        init_flow = register_flow('init_layers', initializers, requires=['optimize'], backend=self.name)

        streaming_passes = [
            'catapult:inplace_stream_flatten',  # Inform downstream changed packsize in case of skipping flatten
            'catapult:reshape_stream',
            'catapult:clone_output',
            'catapult:insert_zero_padding_before_conv1d',
            'catapult:insert_zero_padding_before_conv2d',
            'catapult:broadcast_stream',
        ]
        streaming_flow = register_flow('streaming', streaming_passes, requires=[init_flow], backend=self.name)

        quantization_passes = [
            'catapult:merge_batch_norm_quantized_tanh',
            'catapult:quantize_dense_output',
            'fuse_consecutive_batch_normalization',
            'catapult:xnor_pooling',
        ]
        quantization_flow = register_flow('quantization', quantization_passes, requires=[init_flow], backend=self.name)

        optimization_passes = [
            'catapult:split_conv_gemm',
            'catapult:replace_dense_gemm',
            'catapult:split_attention_heads',
            'catapult:lower_einsum_to_gemm',
            'catapult:remove_final_reshape',
            'catapult:optimize_pointwise_conv',
            'catapult:inplace_parallel_reshape',
            'catapult:inplace_stream_flatten',
            'catapult:skip_softmax',
            'catapult:fix_softmax_table_size',
            'catapult:softmax_const_tables',
            'catapult:process_fixed_point_quantizer_layer',
            'infer_precision_types',
        ]
        optimization_flow = register_flow('optimize', optimization_passes, requires=[init_flow], backend=self.name)

        catapult_types = [
            'catapult:transform_types',
            'catapult:register_bram_weights',
            'catapult:transpose_weights_for_gemm',
            'catapult:generate_conv_streaming_instructions',
            'catapult:apply_resource_strategy',
            'catapult:generate_conv_im2col',
            'catapult:apply_winograd_kernel_transformation',
            'catapult:validate_gemm',
        ]
        catapult_types_flow = register_flow('specific_types', catapult_types, requires=[init_flow], backend=self.name)

        templates = self._get_layer_templates()
        template_flow = register_flow('apply_templates', self._get_layer_templates, requires=[init_flow], backend=self.name)

        writer_passes = ['make_stamp', 'catapult:write_hls']
        self._writer_flow = register_flow('write', writer_passes, requires=['catapult:ip'], backend=self.name)

        fifo_depth_opt_passes = [
            'catapult:fifo_depth_optimization'
        ] + writer_passes  # After optimization, a new project will be written

        register_flow('fifo_depth_optimization', fifo_depth_opt_passes, requires=[self._writer_flow], backend=self.name)

        all_passes = get_backend_passes(self.name)

        extras = [
            # Ideally this should be empty
            opt_pass
            for opt_pass in all_passes
            if opt_pass
            not in initializers
            + streaming_passes
            + quantization_passes
            + optimization_passes
            + catapult_types
            + templates
            + writer_passes
            + fifo_depth_opt_passes
        ]

        if len(extras) > 0:
            for opt in extras:
                warn(f'WARNING: Optimizer "{opt}" is not part of any flow and will not be executed.')

        ip_flow_requirements = [
            'optimize',
            init_flow,
            streaming_flow,
            quantization_flow,
            optimization_flow,
            catapult_types_flow,
            template_flow,
        ]

        self._default_flow = register_flow('ip', None, requires=ip_flow_requirements, backend=self.name)

    def get_default_flow(self):
        return self._default_flow

    def get_writer_flow(self):
        return self._writer_flow

    def create_initial_config(
        self,
        tech='fpga',
        part='xcku115-flvb2104-2-i',
        asiclibs='nangate-45nm',
        fifo=None,
        clock_period=5,
        io_type='io_parallel',
    ):
        config = {}

        config['Technology'] = tech
        if tech == 'fpga':
            config['Part'] = part if part is not None else 'xcvu13p-flga2577-2-e'
        else:
            config['ASICLibs'] = asiclibs if asiclibs is not None else 'nangate-45nm'
        config['ClockPeriod'] = clock_period
        config['FIFO'] = fifo
        config['IOType'] = io_type
        config['HLSConfig'] = {}

        return config

    def build(
        self,
        model,
        reset=False,
        csim=True,
        synth=True,
        cosim=False,
        validation=False,
        vhdl=False,
        verilog=True,
        export=False,
        vsynth=False,
        fifo_opt=False,
        bitfile=False,
        ran_frame=5,
        sw_opt=False,
        power=False,
        da=False,
        bup=False,
    ):
        # print(f'ran_frame value: {ran_frame}')  # Add this line for debugging
        catapult_exe = 'catapult'
        if 'linux' in sys.platform:
            cmd = 'command -v ' + catapult_exe + ' > /dev/null'
            found = os.system(cmd)
            if found != 0:
                mgc_home = os.getenv('MGC_HOME')
                if mgc_home is not None:
                    catapult_exe = mgc_home + '/bin/catapult'
                    cmd = 'command -v ' + catapult_exe + ' > /dev/null'
                    found = os.system(cmd)
            if found != 0:
                catapult_home = os.getenv('CATAPULT_HOME')
                if catapult_home is not None:
                    catapult_exe = catapult_home + '/bin/catapult'
                    cmd = 'command -v ' + catapult_exe + ' > /dev/null'
                    found = os.system(cmd)
            if found != 0:
                raise Exception('Catapult HLS installation not found. Make sure "catapult" is on PATH.')

        curr_dir = os.getcwd()
        # this execution moves into the hls4ml-generated "output_dir" and runs the build_prj.tcl script.
        os.chdir(model.config.get_output_dir())
        ccs_args = f'"reset={reset} csim={csim} synth={synth} cosim={cosim} validation={validation}'
        ccs_args += f' export={export} vsynth={vsynth} fifo_opt={fifo_opt} bitfile={bitfile} ran_frame={ran_frame}'
        ccs_args += f' sw_opt={sw_opt} power={power} da={da} vhdl={vhdl} verilog={verilog} bup={bup}"'
        ccs_invoke = catapult_exe + " -product ultra -shell -f build_prj.tcl -eval 'set ::argv " + ccs_args + "'"
        print(ccs_invoke)
        os.system(ccs_invoke)
        os.chdir(curr_dir)

        return parse_catapult_report(model.config.get_output_dir())

    def _validate_conv_strategy(self, layer):
        if layer.model.config.pipeline_style.lower() != 'dataflow':
            print(f'WARNING: Layer {layer.name} requires "dataflow" pipeline style. Switching to "dataflow" pipeline style.')
            layer.model.config.pipeline_style = 'dataflow'

    def _validate_gemm_ip_conv_support(self, layer):
        if not is_gemm_strategy(layer):
            return

        if layer.model.config.get_config_value('IOType') not in ('io_stream', 'io_parallel'):
            raise ValueError(
                f'Layer "{layer.name}" requested Strategy: GEMM, but Catapult Conv GEMM requires '
                'IOType=io_stream or io_parallel.'
            )

        if layer.get_attr('data_format') != 'channels_last':
            raise ValueError(f'Layer "{layer.name}" requested Strategy: GEMM, but Catapult Conv GEMM requires channels_last.')
        if layer.class_name in ('Conv1D', 'PointwiseConv1D', 'Conv1DBatchnorm'):
            if layer.get_attr('dilation', 1) != 1:
                raise ValueError(
                    f'Layer "{layer.name}" requested Strategy: GEMM, but Catapult Conv GEMM does not support dilation > 1.'
                )
        else:
            if (
                layer.get_attr('dilation', 1) != 1
                or layer.get_attr('dilation_width', 1) != 1
                or layer.get_attr('dilation_height', 1) != 1
            ):
                raise ValueError(
                    f'Layer "{layer.name}" requested Strategy: GEMM, but Catapult Conv GEMM does not support dilation > 1.'
                )

        # Row/column streaming requires stride 1 and valid (zero) padding.
        stride_h = layer.get_attr('stride_height', 1)
        stride_w = layer.get_attr('stride_width', 1)
        if stride_h != 1 or stride_w != 1:
            raise ValueError(
                f'Layer "{layer.name}" requested Strategy: GEMM, but only stride=1 is supported '
                f'by the row/column GEMM IP (got stride={stride_h}x{stride_w}).'
            )
        pad_top = layer.get_attr('pad_top', 0)
        pad_bottom = layer.get_attr('pad_bottom', 0)
        pad_left = layer.get_attr('pad_left', 0)
        pad_right = layer.get_attr('pad_right', 0)
        if pad_top != 0 or pad_bottom != 0 or pad_left != 0 or pad_right != 0:
            raise ValueError(
                f'Layer "{layer.name}" requested Strategy: GEMM, but only valid padding is supported '
                f'by the row/column GEMM IP (got pad=[{pad_top},{pad_bottom},{pad_left},{pad_right}]).'
            )

    @layer_optimizer(Layer)
    def init_base_layer(self, layer):
        reuse_factor = layer.model.config.get_reuse_factor(layer)
        layer.set_attr('reuse_factor', reuse_factor)

        target_cycles = layer.model.config.get_target_cycles(layer)
        layer.set_attr('target_cycles', target_cycles)

    @layer_optimizer(Dense)
    def init_dense(self, layer):
        index_t = IntegerPrecisionType(width=1, signed=False)
        input_shape = layer.get_input_variable().shape
        # gemm_m is set by ReplaceDenseGemm (n_patches) for the GEMM path;
        # for non-GEMM, set n_patches directly.
        gemm_m = int(np.prod(input_shape[:-1])) if len(input_shape) > 1 else 1
        layer.set_attr('gemm_m', gemm_m)
        layer.set_attr('gemm_k', input_shape[-1])
        layer.set_attr('gemm_n', layer.get_attr('n_out'))
        compression = layer.model.config.get_compression(layer)
        if is_gemm_strategy(layer):
            # GEMM is a mutually-exclusive strategy: the matmul is realized by the GEMM
            # IP, not by latency/resource soft logic. ReuseFactor is left as configured.
            layer.set_attr('strategy', 'gemm')
        elif layer.model.config.is_resource_strategy(layer):
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_target_reuse_factor(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
            if compression:
                layer.set_attr('strategy', 'compressed')
                index_t = layer.get_weights('weight').type.index_precision
            else:
                layer.set_attr('strategy', 'resource')
        else:
            layer.set_attr('strategy', 'latency')
        layer.set_attr('index_t', NamedType(f'layer{layer.index}_index', index_t))

    # TODO consolidate these functions into a single `init_conv`
    @layer_optimizer(Conv1D)
    def init_conv1d(self, layer):
        if len(layer.weights['weight'].data.shape) == 2:  # This can happen if we assign weights of Dense layer to 1x1 Conv1D
            layer.weights['weight'].data = np.expand_dims(layer.weights['weight'].data, axis=(0, 1))

        if is_gemm_strategy(layer):
            layer.set_attr('strategy', 'gemm')
        elif layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_target_reuse_factor(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
        else:
            layer.set_attr('strategy', 'latency')

        out_width = layer.get_output_variable().shape[0]
        chosen_pf = layer.model.config.get_layer_config_value(layer, 'ParallelizationFactor', 1)
        valid_pf = self.get_valid_conv_partition_splits(1, out_width)
        if chosen_pf not in valid_pf:
            closest_pf = self.get_closest_reuse_factor(valid_pf, chosen_pf)
            valid_pf_str = ','.join(map(str, valid_pf))
            print(
                f'WARNING: Invalid ParallelizationFactor={chosen_pf} in layer "{layer.name}".'
                f'Using ParallelizationFactor={closest_pf} instead. Valid ParallelizationFactor(s): {valid_pf_str}.'
            )
        else:
            closest_pf = chosen_pf
        layer.set_attr('n_partitions', out_width // closest_pf)

        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

        self._validate_gemm_ip_conv_support(layer)
        self._validate_conv_strategy(layer)

    @layer_optimizer(SeparableConv1D)
    def init_sepconv1d(self, layer):
        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
        else:
            layer.set_attr('strategy', 'latency')

        layer.set_attr(
            'n_partitions', 1
        )  # TODO Once we have SeparableConv implementation for io_parallel this should be set properly
        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

        # Set the output type of the depthwise phase
        dw_out_precision, _ = layer.model.config.get_precision(layer, 'dw_output')
        dw_out_name = layer.name + '_dw_out_t'
        if layer.model.config.get_config_value('IOType') == 'io_stream':
            dw_output_t = PackedType(dw_out_name, dw_out_precision, layer.get_attr('n_chan'), n_pack=1)
        else:
            dw_output_t = NamedType(dw_out_name, dw_out_precision)
        layer.set_attr('dw_output_t', dw_output_t)

    @layer_optimizer(Conv2D)
    def init_conv2d(self, layer):
        if len(layer.weights['weight'].data.shape) == 2:  # This can happen if we assign weights of Dense layer to 1x1 Conv2D
            layer.weights['weight'].data = np.expand_dims(layer.weights['weight'].data, axis=(0, 1))

        if is_gemm_strategy(layer):
            layer.set_attr('strategy', 'gemm')
        elif layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            self.set_target_reuse_factor(layer)
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
        else:
            layer.set_attr('strategy', 'latency')

        out_height = layer.get_output_variable().shape[0]
        out_width = layer.get_output_variable().shape[1]
        chosen_pf = layer.model.config.get_layer_config_value(layer, 'ParallelizationFactor', 1)
        valid_pf = self.get_valid_conv_partition_splits(out_height, out_width)
        if chosen_pf not in valid_pf:
            closest_pf = self.get_closest_reuse_factor(valid_pf, chosen_pf)
            valid_pf_str = ','.join(map(str, valid_pf))
            print(
                f'WARNING: Invalid ParallelizationFactor={chosen_pf} in layer "{layer.name}".'
                f'Using ParallelizationFactor={closest_pf} instead. Valid ParallelizationFactor(s): {valid_pf_str}.'
            )
        else:
            closest_pf = chosen_pf
        layer.set_attr('n_partitions', out_height * out_width // closest_pf)

        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

        self._validate_gemm_ip_conv_support(layer)
        self._validate_conv_strategy(layer)

    @layer_optimizer(SeparableConv2D)
    def init_sepconv2d(self, layer):
        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
        else:
            layer.set_attr('strategy', 'latency')

        layer.set_attr(
            'n_partitions', 1
        )  # TODO Once we have SeparableConv implementation for io_parallel this should be set properly
        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

        # Set the output type of the depthwise phase
        dw_out_precision, _ = layer.model.config.get_precision(layer, 'dw_output')
        dw_out_name = layer.name + '_dw_out_t'
        if layer.model.config.get_config_value('IOType') == 'io_stream':
            dw_output_t = PackedType(dw_out_name, dw_out_precision, layer.get_attr('n_chan'), n_pack=1)
        else:
            dw_output_t = NamedType(dw_out_name, dw_out_precision)
        layer.set_attr('dw_output_t', dw_output_t)

    @layer_optimizer(DepthwiseConv2D)
    def init_depconv2d(self, layer):
        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
        else:
            layer.set_attr('strategy', 'latency')

        layer.set_attr(
            'n_partitions', 1
        )  # TODO Once we have SeparableConv implementation for io_parallel this should be set properly
        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

        # Set the output type of the depthwise phase
        dw_out_precision, _ = layer.model.config.get_precision(layer, 'dw_output')
        dw_out_name = layer.name + '_dw_out_t'
        if layer.model.config.get_config_value('IOType') == 'io_stream':
            dw_output_t = PackedType(dw_out_name, dw_out_precision, layer.get_attr('n_chan'), n_pack=1)
        else:
            dw_output_t = NamedType(dw_out_name, dw_out_precision)
        layer.set_attr('dw_output_t', dw_output_t)

    def _set_pooling_accum_t(self, layer, pool_size):
        extra_bits = ceil_log2(pool_size)
        accum_t = layer.get_attr('accum_t')
        accum_t.precision.width += extra_bits * 2
        if isinstance(accum_t.precision, FixedPrecisionType):
            accum_t.precision.integer += extra_bits

    @layer_optimizer(Pooling1D)
    def init_pooling1d(self, layer):
        pool_size = layer.get_attr('pool_width')
        self._set_pooling_accum_t(layer, pool_size)

        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

    @layer_optimizer(Pooling2D)
    def init_pooling2d(self, layer):
        pool_size = layer.get_attr('pool_height') * layer.get_attr('pool_width')
        self._set_pooling_accum_t(layer, pool_size)

        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

    @layer_optimizer(GlobalPooling1D)
    def init_global_pooling1d(self, layer):
        pool_size = layer.get_attr('n_in')
        self._set_pooling_accum_t(layer, pool_size)

    @layer_optimizer(GlobalPooling2D)
    def init_global_pooling2d(self, layer):
        pool_size = layer.get_attr('in_height') * layer.get_attr('in_width')
        self._set_pooling_accum_t(layer, pool_size)

    @layer_optimizer(EinsumDense)
    def init_einsum_dense(self, layer: EinsumDense) -> None:
        kernel: np.ndarray = layer.attributes['weight_data']
        bias: np.ndarray | None = layer.attributes['bias_data']
        equation = layer.attributes['equation']
        inp_shape = layer.attributes['inp_shape']
        out_shape = layer.attributes['out_shape']

        kernel_shape = kernel.shape
        recipe = parse_einsum(equation, inp_shape, kernel_shape)
        assert not any(recipe['direct_sum_axis']), (
            'Do not put direct sum indices (e.g., only appears in one of the operands) in the equation.'
            'Use explicit addition operator before instead.'
        )
        inp_tpose_idxs, ker_tpose_idxs = recipe['in_transpose_idxs']
        out_tpose_idxs = recipe['out_transpose_idxs']

        # Pre-transpose kernel (and bias) to save a transpose in cpp.
        # hls4ml dense acts like i,ij->j
        # parser assumes ij,j->i, so we need to transpose the kernel to match
        kernel = kernel.transpose(ker_tpose_idxs)
        kernel = kernel.reshape(recipe['I'], recipe['L1'], recipe['C']).transpose(0, 2, 1)

        def to_original_kernel(tkernel: np.ndarray) -> np.ndarray:
            _kernel = tkernel.transpose(0, 2, 1)
            _kernel = _kernel.reshape(tuple(kernel_shape[i] for i in ker_tpose_idxs))
            return _kernel.transpose(np.argsort(ker_tpose_idxs))

        if bias is not None:
            bias = np.broadcast_to(bias, out_shape).transpose(np.argsort(out_tpose_idxs))
        else:
            bias = np.zeros(out_shape).transpose(np.argsort(out_tpose_idxs))

        layer.attributes['weight_data'] = kernel
        layer.attributes['to_original_kernel'] = to_original_kernel
        layer.attributes['bias_data'] = bias
        layer.attributes['inp_tpose_idxs'] = inp_tpose_idxs
        layer.attributes['out_tpose_idxs'] = out_tpose_idxs
        layer.attributes['out_interpert_shape'] = recipe['out_interpert_shape']
        layer.attributes['n_free_data'] = recipe['L0']
        layer.attributes['n_free_kernel'] = recipe['L1']
        layer.attributes['n_inplace'] = recipe['I']
        layer.attributes['n_contract'] = recipe['C']
        layer.attributes['n_in'] = recipe['C']
        layer.attributes['n_out'] = recipe['L1']
        layer.attributes['gemm_m'] = recipe['L0']
        layer.attributes['gemm_k'] = recipe['C']
        layer.attributes['gemm_n'] = recipe['L1']
        pf = layer.attributes.get('parallelization_factor', recipe['L0'])
        layer.attributes['parallelization_factor'] = pf

        layer.add_weights(compression=layer.model.config.get_compression(layer))
        layer.add_bias()

        if is_gemm_strategy(layer):
            # LowerEinsumToGemm lowers this to a Gemm node; the streaming transpose +
            # gemm_stream cells carry the io_stream path.
            layer.set_attr('strategy', 'gemm')
            return
        strategy: str | None = layer.model.config.get_strategy(layer)
        if not strategy:
            layer.set_attr('strategy', 'latency')
            return
        if strategy.lower() in ('latency',):
            layer.set_attr('strategy', strategy)
            return
        warn(f'Invalid strategy "{strategy}" for EinsumDense layer "{layer.name}". Using "latency" strategy instead.')
        layer.set_attr('strategy', 'latency')

    @layer_optimizer(Einsum)
    def init_einsum(self, layer: Einsum) -> None:
        equation = layer.attributes['equation']
        inp0_shape = layer.attributes['inp0_shape']
        inp1_shape = layer.attributes['inp1_shape']

        recipe = parse_einsum(equation, inp0_shape, inp1_shape)
        assert not any(recipe['direct_sum_axis']), (
            'Do not put direct sum indices (e.g., only appears in one of the operands) in the equation.'
            'Use explicit addition operator before instead.'
        )
        inp0_tpose_idxs, inp1_tpose_idxs = recipe['in_transpose_idxs']
        out_tpose_idxs = recipe['out_transpose_idxs']

        layer.attributes.update(recipe)
        layer.attributes['n_free0'] = recipe['L0']
        layer.attributes['n_free1'] = recipe['L1']
        layer.attributes['n_inplace'] = recipe['I']
        layer.attributes['n_contract'] = recipe['C']
        layer.attributes['out_interpert_shape'] = recipe['out_interpert_shape']
        layer.attributes['gemm_m'] = recipe['L0']
        layer.attributes['gemm_n'] = recipe['L1']
        layer.attributes['gemm_k'] = recipe['C']
        layer.attributes['n_in'] = recipe['C']
        layer.attributes['n_out'] = recipe['L1']

        if is_gemm_strategy(layer):
            if layer.model.config.get_config_value('IOType') not in ('io_parallel', 'io_stream'):
                raise ValueError(
                    f'Layer "{layer.name}" requested Strategy: GEMM, but Catapult Einsum GEMM IP '
                    'requires IOType=io_parallel or io_stream.'
                )
            if recipe['L0'] <= 0 or recipe['L1'] <= 0 or recipe['C'] <= 0:
                raise ValueError(
                    f'Layer "{layer.name}" requested Strategy: GEMM, but its Einsum equation is not GEMM-compatible.'
                )

        layer.attributes['inp0_tpose_idxs'] = inp0_tpose_idxs
        layer.attributes['inp1_tpose_idxs'] = inp1_tpose_idxs
        layer.attributes['out_tpose_idxs'] = out_tpose_idxs

        pf = layer.attributes.get('parallelization_factor', recipe['L0'])
        layer.attributes['parallelization_factor'] = pf

        if is_gemm_strategy(layer):
            layer.set_attr('strategy', 'gemm')
            return
        strategy: str | None = layer.model.config.get_strategy(layer)
        if not strategy:
            layer.set_attr('strategy', 'latency')
            return
        if strategy.lower() in ('latency',):
            layer.set_attr('strategy', 'latency')
            return
        warn(f'Invalid strategy "{strategy}" for Einsum layer "{layer.name}". Using "latency" strategy instead.')
        layer.set_attr('strategy', 'latency')

    @layer_optimizer(Embedding)
    def init_embed(self, layer):
        if layer.attributes['n_in'] is None:
            raise Exception('Input length of Embedding layer must be specified.')

    @layer_optimizer(LSTM)
    def init_lstm(self, layer):
        # TODO Allow getting recurrent reuse factor from the config
        reuse_factor = layer.model.config.get_reuse_factor(layer)
        layer.set_attr('recurrent_reuse_factor', reuse_factor)

        if layer.model.config.is_resource_strategy(layer):
            n_in, n_out, n_in_recr, n_out_recr = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
            self.set_closest_reuse_factor(layer, n_in_recr, n_out_recr, attribute='recurrent_reuse_factor')
            layer.set_attr('strategy', 'resource')
        else:
            layer.set_attr('strategy', 'latency')

        layer.set_attr('index_t', NamedType(f'layer{layer.index}_index', IntegerPrecisionType(width=1, signed=False)))

    @layer_optimizer(GRU)
    def init_gru(self, layer):
        reuse_factor = layer.model.config.get_reuse_factor(layer)
        layer.set_attr('recurrent_reuse_factor', reuse_factor)

        if layer.model.config.is_resource_strategy(layer):
            n_in, n_out, n_in_recr, n_out_recr = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
            self.set_closest_reuse_factor(layer, n_in_recr, n_out_recr, attribute='recurrent_reuse_factor')
            layer.set_attr('strategy', 'resource')
        else:
            layer.set_attr('strategy', 'latency')

        layer.set_attr('index_t', NamedType(f'layer{layer.index}_index', IntegerPrecisionType(width=1, signed=False)))

    @layer_optimizer(GarNet)
    def init_garnet(self, layer):
        reuse_factor = layer.attributes['reuse_factor']

        var_converter = CatapultArrayVariableConverter(
            type_converter=HLSTypeConverter(precision_converter=ACTypeConverter())
        )

        # A bit controversial but we are going to set the partitioning of the input here
        in_layer = layer.model.graph[layer.inputs[0]]
        in_var = layer.get_input_variable(layer.inputs[0])
        partition_factor = in_var.shape[1] * (in_var.shape[0] // reuse_factor)
        in_pragma = ('partition', 'cyclic', partition_factor)
        new_in_var = var_converter.convert(in_var, pragma=in_pragma)
        in_layer.set_attr(layer.inputs[0], new_in_var)

        if layer.attributes['collapse']:
            out_pragma = 'partition'
        else:
            partition_factor = layer._output_features * (layer.attributes['n_vertices'] // reuse_factor)
            out_pragma = ('partition', 'cyclic', partition_factor)

        out_name, out_var = next(iter(layer.variables.items()))
        new_out_var = var_converter.convert(out_var, pragma=out_pragma)

        layer.set_attr(out_name, new_out_var)

    @layer_optimizer(GarNetStack)
    def init_garnet_stack(self, layer):
        self.init_garnet(layer)
