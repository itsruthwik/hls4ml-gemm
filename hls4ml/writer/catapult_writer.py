import glob
import os
import stat
import tarfile
import json
from collections import OrderedDict
from copy import copy
from pathlib import Path
from shutil import copyfile, copytree, rmtree

import numpy as np

from hls4ml.writer.gemm_ip_weights import (
    gemm_ip_weight_basename,
    gemm_ip_weight_layout,
    write_gemm_ip_weight_cols,
)
import yaml

from hls4ml.backends import get_backend
from hls4ml.backends.fpga.passes.gemm_nodes import Gemm, Im2ColGemm
from hls4ml.model.layers import EinsumDense, Einsum
from hls4ml.writer.writers import Writer

config_filename = 'hls4ml_config.yml'


class CatapultWriter(Writer):
    @staticmethod
    def _uses_gemm_ip(model):
        return any(bool(node.get_attr('strategy') == 'gemm') for node in model.graph.values())

    @staticmethod
    def _as_list(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    def _layer_input_variables(self, layer):
        return [layer.get_input_variable(input_name) for input_name in layer.inputs]

    def _layer_output_variables(self, layer):
        return [layer.get_output_variable(output_name) for output_name in layer.outputs]

    @staticmethod
    def _resolve_inplace_variable(var):
        """Follow InplaceTensorVariable links to the variable that actually owns storage."""
        from hls4ml.model.types import InplaceTensorVariable

        while isinstance(var, InplaceTensorVariable):
            var = var.input_var
        return var

    def _emit_catapult_stage_wrapper(self, layer):
        func_list = self._as_list(layer.get_attr('function_cpp', None))
        if len(func_list) != 1:
            return ''

        io_vars = self._layer_input_variables(layer) + self._layer_output_variables(layer)
        if not io_vars:
            return ''

        params = []
        pragmas = []
        for var in io_vars:
            resolved = self._resolve_inplace_variable(var)
            if resolved is not var:
                # An inplace alias defines as 'auto& alias = parent', which is not a
                # valid parameter. Declare the parameter with the parent's type but
                # keep the alias name so the wrapped function body still compiles;
                # the call site passes the parent channel.
                proxy = copy(resolved)
                proxy.name = var.name
                params.append(proxy.definition_cpp(as_reference=True))
            else:
                params.append(var.definition_cpp(as_reference=True))
            if getattr(var, 'pragma', None):
                pragmas.append('    ' + self._make_array_pragma(var, layer.model) + '\n')

        # Each stage is a Catapult block. No pipeline pragma on the block itself: Vitis has no
        # function-level II on a dataflow stage either, and each layer header pipelines its own
        # driver loop at II=reuse_factor. A block-level II serialises the inner loop bodies at
        # rf>1 and over-constrains the Resource accumulate feedback at 1.
        wrapper = '#pragma hls_design block\n'
        wrapper += f'void {layer.name}_stage(\n'
        wrapper += ',\n'.join(f'    {param}' for param in params)
        wrapper += '\n) {\n'
        wrapper += ''.join(pragmas)
        wrapper += f'    {func_list[0].split("//", 1)[0].rstrip()}\n'
        wrapper += '}\n\n'
        return wrapper

    @staticmethod
    def _is_gemm_ip_weight(layer, weights):
        if not bool(layer.get_attr('strategy') == 'gemm'):
            return False
        try:
            return weights.name == layer.get_weights('weight').name
        except Exception:
            return False

    def print_gemm_ip_weight_beats_to_cpp(self, var, layer, odir):
        """Write GEMM-IP packed weight columns (shared implementation)."""
        write_gemm_ip_weight_cols(var, layer, odir)

    def print_array_to_cpp(self, var, odir, write_txt_file=True):
        """Write a weights array to C++ header files.

        Args:
            var (WeightVariable): Weight to write
            odir (str): Output directory
            write_txt_file (bool, optional): Write txt files in addition to .h files. Defaults to True.
        """

        h_file = open(f'{odir}/firmware/weights/{var.name}.h', 'w')
        if write_txt_file:
            txt_file = open(f'{odir}/firmware/weights/{var.name}.txt', 'w')

        # meta data
        h_file.write(f'//Numpy array shape {var.shape}\n')
        h_file.write(f'//Min {np.min(var.min):.12f}\n')
        h_file.write(f'//Max {np.max(var.max):.12f}\n')
        h_file.write(f'//Number of zeros {var.nzeros}\n')
        h_file.write('\n')

        h_file.write(f'#ifndef {var.name.upper()}_H_\n')
        h_file.write(f'#define {var.name.upper()}_H_\n')
        h_file.write('\n')

        if write_txt_file:
            h_file.write('#ifndef __SYNTHESIS__\n')
            h_file.write('// global extern pointer only - actual array allocated in myproject_test.cpp\n')
            h_file.write('extern ' + var.definition_cpp() + ';\n')
            h_file.write('#else\n')

        h_file.write(var.definition_cpp() + ' = {')

        # fill c++ array.
        # not including internal brackets for multidimensional case
        sep = ''
        if getattr(var, 'transpose', False):
            # If transpose is requested, we iterate in column-major order
            # This is handled by the __iter__ method in WeightVariable
            pass
            
        for x in var:
            h_file.write(sep + x)
            if write_txt_file:
                txt_file.write(sep + x)
            sep = ', '
        h_file.write('};\n')
        if write_txt_file:
            h_file.write('#endif\n')
            txt_file.close()
        h_file.write('\n#endif\n')
        h_file.close()

    def write_output_dir(self, model):
        """Write the base output directory

        Args:
            model (ModelGraph): the hls4ml model.
        """
        if not os.path.isdir(f'{model.config.get_output_dir()}/firmware/weights'):
            os.makedirs(f'{model.config.get_output_dir()}/firmware/weights')

    @staticmethod
    def _make_array_pragma(variable, model):
        """
        Layers in hls_model.py can specify output array partitioning through the `pragma` attribute.
        If `pragma` is a string: options are 'partition', 'reshape', or 'stream'.
        If `pragma` is a tuple: (mode, type, factor) where mode is 'partition' or 'reshape', type is
        'complete', 'cyclic', or 'block', and factor is an integer only used when the type is not 'complete'.
        """

        config = variable.pragma
        if type(config) is tuple:
            mode = config[0]
            if mode in ['partition', 'reshape']:
                typ = config[1]
                if typ != 'complete':
                    factor = config[2]
            elif mode == 'stream':
                depth = config[1]
        else:
            mode = config
            typ = 'complete'
            factor = 0

        if mode in ['partition', 'reshape']:
            if typ == 'complete':
                template = '// #pragma HLS ARRAY_{mode} variable={name} {type} dim={dim}'
            else:
                template = '// #pragma HLS ARRAY_{mode} variable={name} {type} factor={factor} dim={dim}'

            return template.format(mode=mode.upper(), name=variable.name, type=typ, factor=factor, dim=0)

        elif mode == 'stream':
            fifo = model.config.get_config_value('FIFO')
            if fifo is not None:
                retstr = f'#pragma hls_resource {variable.name}:cns variables="{variable.name}"'
                retstr += f' map_to_module="{fifo}" // depth="{depth}"'
                return retstr
            else:
                return ''
        else:
            return ''

    @staticmethod
    def _make_array_fifo_pragma(variable, model):
        config = variable.pragma
        factor = ''
        if type(config) is tuple:
            mode = config[0]
            if mode in ['partition', 'reshape']:
                typ = config[1]
                if typ != 'complete':
                    factor = config[2]
            elif mode == 'stream':
                depth = config[1]
        else:
            mode = config
            typ = 'complete'
            factor = 0

        if mode == 'stream':
            return f'#pragma hls_fifo_depth {depth}'
        else:
            return ''

    def write_project_cpp(self, model):
        """Write the main architecture source file (myproject.cpp)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        filedir = os.path.dirname(os.path.abspath(__file__))

        fout = open(f'{model.config.get_output_dir()}/firmware/layer_summary.txt', 'w')
        outstr = ''
        outstr = outstr + '{}'.format('Layer Name').ljust(25)
        outstr = outstr + '  {}'.format('Layer Class').ljust(20)
        outstr = outstr + '  {}'.format('Input Type').ljust(40)
        outstr = outstr + '  {}'.format('Input Shape').ljust(15)
        outstr = outstr + '  {}'.format('Output Type').ljust(40)
        outstr = outstr + '  {}'.format('Output Shape').ljust(15)
        # outstr = outstr + "  {}".format("Weight Type").ljust(24)
        # outstr = outstr + "  {}".format("Bias Type").ljust(24)
        outstr = outstr + '  {}'.format('Filter Shape').ljust(15)
        outstr = outstr + '  {}'.format('Stride').ljust(10)
        outstr = outstr + '  {}'.format('IOType').ljust(15)
        outstr = outstr + '  {}'.format('Reuse').ljust(10)

        fout.write(outstr + '\n')
        input_shape = ''
        input_datatype = ''
        for layer in model.get_layers():
            datatype = layer.get_output_variable().type.precision.definition_cpp() + ' '
            shape = ''
            # layer.get_output_variable().type.precision.width
            # layer.get_output_variable().type.precision.integer
            # layer.get_output_variable().type.precision.sign
            for v in layer.get_output_variable().shape:
                shape = shape + '[' + str(v) + ']'

            if layer.attributes.layer.class_name != 'Input':
                my_class_name = layer.class_name
                if layer.attributes.layer.class_name == 'Activation':
                    my_class_name = layer.get_attr('activation')

                # filter_datatype = ""
                # print(layer.weights.__dir__())
                # layer_precision = layer.get_layer_precision()
                # for wname, weights in layer.weights.items():
                #    print(wname)
                #    print(weights.type.name)
                #    print(weights.type.precision.definition_cpp())
                #    #print(weights.type.precision.__dir__())
                #    print(weights.type.precision.width)
                #    if 'ACFixed' in weights.type.precision.__class__:
                #        print(weights.type.precision.integer)
                #        print(weights.type.precision.signed)
                #    print(weights.data_length)

                filter = ''
                filt_width = layer.get_attr('filt_width')
                filt_height = layer.get_attr('filt_height')
                if filt_width is not None:
                    filter = '[' + str(filt_width) + ']'
                if filt_height is not None:
                    filter = filter + '[' + str(filt_height) + ']'

                stride = ''
                stride_width = layer.get_attr('stride_width')
                if stride_width is not None:
                    stride = str(stride_width)

                outstr = ''
                outstr = outstr + f'{layer.name}'.ljust(25)
                outstr = outstr + f'  {my_class_name}'.ljust(20)
                outstr = outstr + f'  {input_datatype}'.ljust(40)
                outstr = outstr + f'  {input_shape}'.ljust(15)
                outstr = outstr + f'  {datatype}'.ljust(40)
                outstr = outstr + f'  {shape}'.ljust(15)
                # outstr = outstr + "  {}".format("weight type").ljust(24)
                # outstr = outstr + "  {}".format("bias type").ljust(24)
                outstr = outstr + f'  {filter}'.ljust(15)
                outstr = outstr + f'  {stride}'.ljust(10)
                outstr = outstr + '  {}'.format(layer.model.config.get_config_value('IOType')).ljust(15)
                outstr = outstr + f'  {str(layer.model.config.get_reuse_factor(layer))}'.ljust(10)
                fout.write(outstr + '\n')

            input_shape = shape
            input_datatype = datatype

        fout.close()

        f = open(os.path.join(filedir, '../templates/catapult/firmware/myproject.cpp'))
        fout = open(f'{model.config.get_output_dir()}/firmware/{model.config.get_project_name()}.cpp', 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']
        io_type = model.config.get_config_value('IOType')
        stream_channel_scope = model.config.get_writer_config().get('StreamChannelScope', 'local_static')
        stream_function_style = model.config.get_writer_config().get('StreamFunctionStyle', None)
        use_stream_stage_wrappers = io_type == 'io_stream' and stream_function_style != 'inline'
        use_parallel_stage_wrappers = False
        use_stage_wrappers = use_stream_stage_wrappers or use_parallel_stage_wrappers
        use_global_stream_channels = io_type in ('io_serial', 'io_stream') and stream_channel_scope == 'global'

        indent = '    '

        for line in f.readlines():
            # Add headers to weights and biases
            if 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())
            elif '// hls-fpga-machine-learning insert global-layer-declarations' in line:
                newline = line
                if use_global_stream_channels:
                    declared_vars = set()
                    for layer in model.get_layers():
                        for var in layer.get_variables():
                            if var in model_inputs or var in model_outputs:
                                continue
                            def_cpp = var.definition_cpp()
                            if def_cpp is not None and def_cpp not in declared_vars:
                                declared_vars.add(def_cpp)
                                newline += def_cpp + ';\n'
                    newline += '\n'
            elif '// hls-fpga-machine-learning insert stage-wrapper-definitions' in line:
                newline = line
                if use_stage_wrappers:
                    for layer in model.get_layers():
                        newline += self._emit_catapult_stage_wrapper(layer)
            elif '// hls-fpga-machine-learning insert header' in line:
                inputs_str = ', '.join([i.definition_cpp(as_reference=True) for i in model_inputs])
                outputs_str = ', '.join([o.definition_cpp(as_reference=True) for o in model_outputs])
                brams_str = ', \n'.join([indent + b.definition_cpp(as_reference=False) for b in model_brams])

                newline = ''
                newline += indent + inputs_str + ',\n'
                newline += indent + outputs_str
                if len(model_brams) > 0:
                    newline += ',\n' + brams_str
                newline += '\n'

            elif '// hls-fpga-machine-learning insert load weights' in line:
                newline = line
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        if w.weight_class == 'CompressedWeightVariable':
                            newline += indent + '    nnet::load_compressed_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                                w.type.name, w.nonzeros, w.name, w.name
                            )
                        elif w.weight_class == 'ExponentWeightVariable':
                            newline += indent + '    nnet::load_exponent_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                                w.type.name, w.data_length, w.name, w.name
                            )
                        else:
                            newline += indent + '    nnet::load_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                                w.type.name, w.data_length, w.name, w.name
                            )

            # Add Interface Synthesis resource pragmas
            elif '// hls-fpga-machine-learning insert IFSynPragmas' in line:
                newline = line
                all_inputs = [i.name for i in model_inputs]
                all_outputs = [o.name for o in model_outputs]
                all_brams = [b.name for b in model_brams]

                if io_type == 'io_serial' or io_type == 'io_stream':
                    # Eventually this will be amba.ccs_axi4stream_in and amba.ccs_axi4stream_out
                    for dut_input in all_inputs:
                        newline += f'#pragma hls_resource {dut_input}:rsc variables="{dut_input}"'
                        newline += ' map_to_module="ccs_ioport.ccs_in_wait"\n'
                    for dut_output in all_outputs:
                        newline += f'#pragma hls_resource {dut_output}:rsc variables="{dut_output}"'
                        newline += ' map_to_module="ccs_ioport.ccs_out_wait"\n'

            # Add input/output type
            elif '// hls-fpga-machine-learning insert IO' in line:
                newline = line
                all_inputs = [i.name for i in model_inputs]
                all_outputs = [o.name for o in model_outputs]
                all_brams = [b.name for b in model_brams]

                if io_type == 'io_parallel':
                    for i in model_inputs:
                        newline += indent + self._make_array_pragma(i, model) + '\n'
                    for o in model_outputs:
                        newline += indent + self._make_array_pragma(o, model) + '\n'
                    # TODO discussed adding a handle for setting the interface mode for individual input and output arrays
                    # Probably the handle doesn't need to be exposed to the user but should be just set in hls_model.py
                    newline += indent + '// #pragma HLS INTERFACE ap_vld port={},{} \n'.format(
                        ','.join(all_inputs), ','.join(all_outputs)
                    )
                    if model.config.model_strategy.lower() == 'dataflow':
                        newline += indent + '// #pragma HLS DATAFLOW \n'
                    else:
                        newline += indent + '// #pragma HLS PIPELINE \n'
                if io_type == 'io_stream':
                    newline += indent + '// #pragma HLS INTERFACE axis port={},{} \n'.format(
                        ','.join(all_inputs), ','.join(all_outputs)
                    )
                    if all_brams:
                        newline += indent + '// #pragma HLS INTERFACE bram port={} \n'.format(','.join(all_brams))
                    newline += indent + '// #pragma HLS DATAFLOW \n'

            elif '// hls-fpga-machine-learning insert layers' in line:
                newline = line + '\n'
                for layer in model.get_layers():
                    vars = layer.get_variables()
                    for var in vars:
                        if var not in model_inputs and var not in model_outputs:
                            def_cpp = var.definition_cpp()
                            if def_cpp is not None:
                                depth = 1
                                if var.pragma and type(var.pragma) is tuple and var.pragma[0] == 'stream':
                                    depth = var.pragma[1]
                                
                                if var.pragma:
                                    newline += '    ' + self._make_array_fifo_pragma(var, model) + '\n'
                                if io_type == 'io_serial' or io_type == 'io_stream':
                                    if not use_global_stream_channels:
                                        newline += f'    static {def_cpp};\n'
                                else:
                                    newline += '    ' + def_cpp + '; \n'
                                if var.pragma:
                                    newline += '    ' + self._make_array_pragma(var, model) + '\n'
                    func = layer.get_attr('function_cpp', None)
                    if func:
                        if not isinstance(func, (list, set)):
                            func = [func]
                        if use_stage_wrappers and len(func) == 1:
                            call_vars = self._layer_input_variables(layer) + self._layer_output_variables(layer)
                            # Inplace aliases own no storage; pass the parent channel.
                            call_args = ', '.join(self._resolve_inplace_variable(var).name for var in call_vars)
                            newline += f'    {layer.name}_stage({call_args}); // {layer.name}\n'
                        elif len(func) == 1:
                            newline += '    ' + func[0] + ' // ' + layer.name + '\n'
                        else:
                            newline += '    // ' + layer.name + '\n'
                            for line in func:
                                newline += '    ' + line + '\n'
                        if model.config.trace_output and layer.get_attr('trace', False):
                            newline += '#ifndef __SYNTHESIS__\n'
                            for var in vars:
                                newline += '    nnet::save_layer_output<{}>({}, "{}", {});\n'.format(
                                    var.type.name, var.name, layer.name, var.size_cpp()
                                )
                            newline += '#endif\n'
                        newline += '\n'

            # Just copy line
            else:
                newline = line

            fout.write(newline)

        f.close()
        fout.close()

    def write_project_header(self, model):
        """Write the main architecture header file (myproject.h)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/catapult/firmware/myproject.h'))
        fout = open(f'{model.config.get_output_dir()}/firmware/{model.config.get_project_name()}.h', 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        indent = '    '

        for line in f.readlines():
            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT', format(model.config.get_project_name().upper()))
            elif 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())
            elif '// hls-fpga-machine-learning insert header' in line:
                inputs_str = ', '.join([i.definition_cpp(as_reference=True) for i in model_inputs])
                outputs_str = ', '.join([o.definition_cpp(as_reference=True) for o in model_outputs])
                brams_str = ', \n'.join([indent + b.definition_cpp(as_reference=False) for b in model_brams])

                newline = ''
                newline += indent + inputs_str + ',\n'
                newline += indent + outputs_str
                if len(model_brams) > 0:
                    newline += ',\n' + brams_str
                newline += '\n'
            else:
                newline = line
            fout.write(newline)

        f.close()
        fout.close()

    def write_defines(self, model):
        """Write the C++ type definitions file (defines.h)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/catapult/firmware/defines.h'))
        fout = open(f'{model.config.get_output_dir()}/firmware/defines.h', 'w')

        for line in f.readlines():
            if '// hls-fpga-machine-learning insert layer-precision' in line:
                newline = line
                all_precision = OrderedDict()
                for layer in model.get_layers():
                    layer_precision = layer.get_layer_precision()
                    for type_name, type_var in layer_precision.items():
                        # Ensure that layer's types doesn't override existing types
                        # This can happen in case of InplaceVariable types
                        if type_name not in all_precision:
                            all_precision[type_name] = type_var
                for used_type in all_precision.values():
                    newline += used_type.definition_cpp()

            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    def write_parameters(self, model):
        """Write the C++ layer config file (parameters.h)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/catapult/firmware/parameters.h'))
        fout = open(f'{model.config.get_output_dir()}/firmware/parameters.h', 'w')

        for line in f.readlines():
            if '// hls-fpga-machine-learning insert includes' in line:
                newline = line
                for include in sorted(set(sum((layer.get_attr('include_header', []) for layer in model.get_layers()), []))):
                    newline += '#include "%s"\n' % include

            elif '// hls-fpga-machine-learning insert weights' in line:
                newline = line
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        if w.storage.lower() != 'bram':
                            newline += f'#include "weights/{w.name}.h"\n'
                            # ROM header included for both paths: the legacy streamed-weight
                            # call and the weight-stationary csim behavioral model both use it.
                            # (In synth-with-package the weight-stationary call is const_weights and
                            # this static array is unused → dead-code-eliminated.)
                            if self._is_gemm_ip_weight(layer, w):
                                newline += f'#include "weights/{gemm_ip_weight_basename(w, layer)}.h"\n'

            elif '// hls-fpga-machine-learning insert layer-config' in line:
                newline = line
                for layer in model.get_layers():
                    config = layer.get_attr('config_cpp', None)
                    if config:
                        newline += '// ' + layer.name + '\n'
                        newline += config + '\n'
            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    def write_weights(self, model):
        """Write the weights into header files

        Args:
            model (ModelGraph): the hls4ml model.
        """
        for layer in model.get_layers():
            for weights in layer.get_weights():
                self.print_array_to_cpp(weights, model.config.get_output_dir())
                if self._is_gemm_ip_weight(layer, weights):
                    # ROM header: source of truth for the legacy streamed-weight path AND
                    # the weight-stationary csim behavioral model (native, no gemm-ip-gen).
                    self.print_gemm_ip_weight_beats_to_cpp(weights, layer, model.config.get_output_dir())
                    if bool(layer.get_attr('weights_in_core', False)):
                        # Weight-stationary also emits raw-bits .dat for the external
                        # generator to bake into the synth const_weights core.
                        from hls4ml.writer.gemm_ip_weights import write_gemm_ip_weight_dat
                        write_gemm_ip_weight_dat(weights, layer, model.config.get_output_dir())

    def __make_dat_file(self, original_path, project_path):
        """
        Convert other input/output data types into a dat file, which is
        a text file with the falttened matrix printed out. Note that ' ' is
        assumed to be the delimiter.
        """

        # Take in data from current supported data files
        if original_path[-3:] == 'npy':
            data = np.load(original_path)
        else:
            raise Exception('Unsupported input/output data files.')

        # Faltten data, just keep first dimension
        data = data.reshape(data.shape[0], -1)

        def print_data(f):
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    f.write(str(data[i][j]) + ' ')
                f.write('\n')

        # Print out in dat file
        with open(project_path, 'w') as f:
            print_data(f)

    def write_test_bench(self, model):
        """Write the testbench files (myproject_test.cpp and input/output .dat files)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        filedir = os.path.dirname(os.path.abspath(__file__))

        if not os.path.exists(f'{model.config.get_output_dir()}/tb_data/'):
            os.mkdir(f'{model.config.get_output_dir()}/tb_data/')

        input_data = model.config.get_config_value('InputData')
        output_predictions = model.config.get_config_value('OutputPredictions')

        if input_data:
            if input_data[-3:] == 'dat':
                copyfile(input_data, f'{model.config.get_output_dir()}/tb_data/tb_input_features.dat')
            else:
                self.__make_dat_file(input_data, f'{model.config.get_output_dir()}/tb_data/tb_input_features.dat')

        if output_predictions:
            if output_predictions[-3:] == 'dat':
                copyfile(output_predictions, f'{model.config.get_output_dir()}/tb_data/tb_output_predictions.dat')
            else:
                self.__make_dat_file(
                    output_predictions, f'{model.config.get_output_dir()}/tb_data/tb_output_predictions.dat'
                )

        f = open(os.path.join(filedir, '../templates/catapult/myproject_test.cpp'))
        fout = open(f'{model.config.get_output_dir()}/{model.config.get_project_name()}_test.cpp', 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        for line in f.readlines():
            indent = ' ' * (len(line) - len(line.lstrip(' ')))

            # Insert numbers
            if 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())
            elif '// hls-fpga-machine-learning insert bram' in line:
                newline = line
                for bram in model_brams:
                    newline += f'#include "firmware/weights/{bram.name}.h"\n'

            elif '// hls-fpga-machine-learning insert declare weights' in line:
                newline = line
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        newline += w.definition_cpp() + ';\n'

            elif '// hls-fpga-machine-learning insert load weights' in line:
                newline = line
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        if w.weight_class == 'CompressedWeightVariable':
                            newline += indent + '    nnet::load_compressed_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                                w.type.name, w.nonzeros, w.name, w.name
                            )
                        elif w.weight_class == 'ExponentWeightVariable':
                            newline += indent + '    nnet::load_exponent_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                                w.type.name, w.data_length, w.name, w.name
                            )
                        else:
                            newline += indent + '    nnet::load_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                                w.type.name, w.data_length, w.name, w.name
                            )

            elif '// hls-fpga-machine-learning insert data' in line:
                newline = line
                offset = 0
                for inp in model_inputs:
                    newline += '      ' + inp.definition_cpp() + ';\n'
                    newline += '      nnet::copy_data<float, {}, {}, {}>(in, {});\n'.format(
                        inp.type.name, offset, inp.size_cpp(), inp.name
                    )
                    offset += inp.size()
                for out in model_outputs:
                    newline += '      ' + out.definition_cpp() + ';\n'
            elif '// hls-fpga-machine-learning insert random' in line:
                newline = line
                for inp in model_inputs:
                    newline += '    ' + inp.definition_cpp() + ';\n'
                    newline += f'    nnet::fill_random<{inp.type.name}, {inp.size_cpp()}>({inp.name});\n'
                for out in model_outputs:
                    newline += '    ' + out.definition_cpp() + ';\n'
            elif '// hls-fpga-machine-learning insert zero' in line:
                newline = line
                for inp in model_inputs:
                    newline += '    ' + inp.definition_cpp() + ';\n'
                    newline += f'    nnet::fill_zero<{inp.type.name}, {inp.size_cpp()}>({inp.name});\n'
                for out in model_outputs:
                    newline += '    ' + out.definition_cpp() + ';\n'
            elif '// hls-fpga-machine-learning insert top-level-function' in line:
                newline = line

                input_vars = ','.join([i.name for i in model_inputs])
                output_vars = ','.join([o.name for o in model_outputs])
                bram_vars = ','.join([b.name for b in model_brams])

                # Concatenate the input, output, and bram variables. Filter out empty/null values
                all_vars = ','.join(filter(None, [input_vars, output_vars, bram_vars]))

                top_level = indent + f'{model.config.get_project_name()}({all_vars});\n'

                newline += top_level
            elif '// hls-fpga-machine-learning insert predictions' in line:
                newline = line
                for out in model_outputs:
                    newline += indent + f'for(int i = 0; i < {out.size_cpp()}; i++) {{\n'
                    newline += indent + '  std::cout << pr[i] << " ";\n'
                    newline += indent + '}\n'
                    newline += indent + 'std::cout << std::endl;\n'
            elif '// hls-fpga-machine-learning insert tb-output' in line:
                newline = line
                for out in model_outputs:
                    newline += indent + 'nnet::print_result<{}, {}>({}, fout);\n'.format(
                        out.type.name, out.size_cpp(), out.name
                    )  # TODO enable this
            elif (
                '// hls-fpga-machine-learning insert output' in line
                or '// hls-fpga-machine-learning insert quantized' in line
            ):
                newline = line
                for out in model_outputs:
                    newline += indent + 'nnet::print_result<{}, {}>({}, std::cout, true);\n'.format(
                        out.type.name, out.size_cpp(), out.name
                    )
            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    def write_bridge(self, model):
        """Write the Python-C++ bridge (myproject_bridge.cpp)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/catapult/myproject_bridge.cpp'))
        fout = open(f'{model.config.get_output_dir()}/{model.config.get_project_name()}_bridge.cpp', 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        indent = '    '

        for line in f.readlines():
            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT', format(model.config.get_project_name().upper()))
            elif 'myproject' in line:
                newline = line.replace('myproject', format(model.config.get_project_name()))
            elif '// hls-fpga-machine-learning insert weights dir' in line:
                weights_dir = (Path(fout.name).parent / 'firmware/weights').resolve()
                newline = f'static std::string s_weights_dir = "{weights_dir}";\n'
            elif '// hls-fpga-machine-learning insert bram' in line:
                newline = line
                for bram in model_brams:
                    newline += f'#include "firmware/weights/{bram.name}.h"\n'
            elif '// hls-fpga-machine-learning insert declare weights' in line:
                newline = line
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        newline += w.definition_cpp() + ';\n'
            elif '// hls-fpga-machine-learning insert header' in line:
                dtype = line.split('#', 1)[1].strip()
                inputs_str = ', '.join([f'{dtype} {i.name}[{i.size_cpp()}]' for i in model_inputs])
                outputs_str = ', '.join([f'{dtype} {o.name}[{o.size_cpp()}]' for o in model_outputs])

                newline = ''
                newline += indent + inputs_str + ',\n'
                newline += indent + outputs_str + '\n'
            elif '// hls-fpga-machine-learning insert wrapper' in line:
                dtype = line.split('#', 1)[1].strip()
                newline = ''
                for i in model_inputs:
                    newline += indent + '{var};\n'.format(var=i.definition_cpp(name_suffix='_ap'))
                    newline += indent + 'nnet::convert_data<{}, {}, {}>({}, {}_ap);\n'.format(
                        dtype, i.type.name, i.size_cpp(), i.name, i.name
                    )
                newline += '\n'

                for o in model_outputs:
                    newline += indent + '{var};\n'.format(var=o.definition_cpp(name_suffix='_ap'))

                newline += '\n'

                input_vars = ','.join([i.name + '_ap' for i in model_inputs])
                bram_vars = ','.join([b.name for b in model_brams])
                output_vars = ','.join([o.name + '_ap' for o in model_outputs])

                # Concatenate the input, output, and bram variables. Filter out empty/null values
                all_vars = ','.join(filter(None, [input_vars, output_vars, bram_vars]))

                top_level = indent + f'{model.config.get_project_name()}({all_vars});\n'
                newline += top_level

                newline += '\n'

                for o in model_outputs:
                    newline += indent + 'nnet::convert_data<{}, {}, {}>({}_ap, {});\n'.format(
                        o.type.name, dtype, o.size_cpp(), o.name, o.name
                    )
            elif '// hls-fpga-machine-learning insert trace_outputs' in line:
                newline = ''
                for layer in model.get_layers():
                    func = layer.get_attr('function_cpp', None)
                    if func and model.config.trace_output and layer.get_attr('trace', False):
                        vars = layer.get_variables()
                        for var in vars:
                            newline += (
                                indent
                                + 'nnet::trace_outputs->insert(std::pair<std::string, void *>('
                                + f'"{layer.name}", (void *) malloc({var.size_cpp()} * element_size)));\n'
                            )

            else:
                newline = line
            fout.write(newline)

        f.close()
        fout.close()

    def write_build_script(self, model):
        """Write the TCL/Shell build scripts.

        Args:
            model (ModelGraph): the hls4ml model.
        """

        filedir = Path(__file__).parent

        # build_prj.tcl
        srcpath = (filedir / '../templates/catapult/build_prj.tcl').resolve()
        dstpath = Path(f'{model.config.get_output_dir()}/build_prj.tcl').resolve()
        
        gemm_ip_pkg = model.config.get_writer_config().get('GemmIpPackage')
        if gemm_ip_pkg is None:
            gemm_ip_pkg = model.config.get_config_value('GemmIpPackage')
        if gemm_ip_pkg is None:
            # Direct access fallback
            try:
                gemm_ip_pkg = model.config.config['HLSConfig']['Model'].get('GemmIpPackage')
            except Exception:
                pass
        
        compiler_flags = '-DRANDOM_FRAMES=$opt(ran_frame)'

        # GEMM IP: decide whether the gemm-ip-gen package is used AT BUILD TIME from the
        # package dir's existence (not baked here), honoring the nnet_gemm_ip.h header
        # contract — with a package, -DGEMM_IP_HEADER pulls in the packaged cores; WITHOUT
        # one, the macro is left undefined so a package-free csim compiles the behavioral
        # model (`#else` branch). Only synth/cosim/validation need the package. $sfd is the
        # project dir; the SCVerify compiles run from a nested solution dir, so normalize to
        # an absolute path (evaluated when the tcl is sourced) for a stable include.
        uses_gemm_ip = self._uses_gemm_ip(model)
        if uses_gemm_ip:
            if gemm_ip_pkg:
                gemm_pkg_dir_expr = '[file normalize {' + str(Path(gemm_ip_pkg).resolve()) + '}]'
            else:
                gemm_pkg_dir_expr = '[file normalize $sfd/gemm_pkg]'

        with open(srcpath) as src, open(dstpath, 'w') as dst:
            for line in src.readlines():
                indent = line[: len(line) - len(line.lstrip())]
                line = line.replace('myproject', model.config.get_project_name())
                line = line.replace('CATAPULT_DIR', model.config.get_project_dir())
                if '#hls-fpga-machine-learning insert techlibs' in line:
                    if model.config.get_config_value('Technology') is None:
                        if model.config.get_config_value('Part') is not None:
                            line = indent + 'setup_xilinx_part {{{}}}\n'.format(model.config.get_config_value('Part'))
                        elif model.config.get_config_value('ASICLibs') is not None:
                            line = indent + 'setup_asic_libs {{{}}}\n'.format(model.config.get_config_value('ASICLibs'))
                    else:
                        if model.config.get_config_value('Technology') == 'asic':
                            line = indent + 'setup_asic_libs {{{}}}\n'.format(model.config.get_config_value('ASICLibs'))
                        else:
                            line = indent + 'setup_xilinx_part {{{}}}\n'.format(model.config.get_config_value('Part'))
                elif '#hls-fpga-machine-learning insert invoke_args' in line:
                    # The writer copies InputData/OutputPredictions into tb_data/ under
                    # canonical names, so the testbench args must reference those names —
                    # not the raw config value (which may be an absolute source path and
                    # would produce a malformed $sfd/tb_data/<abspath>, silently falling
                    # back to random frames).
                    tb_in_file = model.config.get_config_value('InputData')
                    tb_out_file = model.config.get_config_value('OutputPredictions')
                    invoke_args = '$sfd/firmware/weights'
                    if tb_in_file is not None:
                        invoke_args = invoke_args + ' $sfd/tb_data/tb_input_features.dat'
                    if tb_out_file is not None:
                        invoke_args = invoke_args + ' $sfd/tb_data/tb_output_predictions.dat'
                    line = indent + f'flow package option set /SCVerify/INVOKE_ARGS "{invoke_args}"\n'
                elif 'set hls_clock_period 5' in line:
                    line = indent + 'set hls_clock_period {}\n'.format(model.config.get_config_value('ClockPeriod'))
                elif 'options set Input/CompilerFlags' in line:
                    # Presence-driven const softmax LUTs: when the two-pass header has been
                    # generated next to the streaming activations, compile the exp/invert
                    # tables as ROM (-DHLS4ML_SOFTMAX_CONST_TABLES) instead of a runtime build
                    # that Catapult otherwise schedules into the streamed core (inflating the
                    # softmax block throughput). Absent -> plain runtime build (unchanged).
                    line = indent + f'set _hls4ml_cxxflags "{compiler_flags}"\n'
                    # Define _gemm_has_pkg unconditionally so the blackbox guard below can
                    # reference it even when GEMM IP is unused.
                    line += indent + 'set _gemm_has_pkg 0\n'
                    if uses_gemm_ip:
                        # Package presence gates -DGEMM_IP_HEADER (else behavioral csim).
                        line += (
                            indent + f'set _gemm_pkg_dir {gemm_pkg_dir_expr}\n'
                            + indent + 'if { [file isdirectory $_gemm_pkg_dir] } {\n'
                            + indent + '  append _hls4ml_cxxflags " -DBLACKBOX_FLOW -DGEMM_IP_HEADER -I$_gemm_pkg_dir"\n'
                            + indent + '  set _gemm_has_pkg 1\n'
                            + indent + '}\n'
                        )
                    line += (
                        indent + 'if { [file exists $sfd/firmware/nnet_utils/softmax_const_tables.h] } {\n'
                        + indent + '  append _hls4ml_cxxflags " -DHLS4ML_SOFTMAX_CONST_TABLES"\n'
                        + indent + '  logfile message "hls4ml: softmax_const_tables.h present -> softmax LUTs as ROM\\n" info\n'
                        + indent + '}\n'
                        + indent + 'options set Input/CompilerFlags $_hls4ml_cxxflags\n'
                    )
                elif '#hls-fpga-machine-learning insert blackboxes' in line:
                    if uses_gemm_ip:
                        # Only synth/cosim/validation need the gemm-ip-gen package; a
                        # package-free csim compiles the behavioral model (GEMM_IP_HEADER
                        # left undefined above). Fail loudly only for the RTL flows.
                        line = (
                            indent + 'if { !$_gemm_has_pkg && ($opt(synth) || $opt(cosim) || $opt(validation)) } {\n'
                            + indent + '  logfile message "GEMM IP is used but $_gemm_pkg_dir was not found. Run gemm-ip-gen before synth/cosim." error\n'
                            + indent + '  exit 1\n'
                            + indent + '}\n'
                            + indent + 'if { $_gemm_has_pkg } {\n'
                            + indent + '  logfile message "GEMM IP package resolved at $_gemm_pkg_dir through GEMM_IP_HEADER and ac_blackbox bindings." info\n'
                            + indent + '} else {\n'
                            + indent + '  logfile message "No GEMM IP package found; csim uses the behavioral model (synth/cosim require the package)." info\n'
                            + indent + '}\n'
                        )
                    else:
                        line = ''
                dst.write(line)

        # Optional bottom-up Tcl script
        build_bup_tcl_src = (filedir / '../templates/catapult/build_prj_bup.tcl').resolve()
        build_bup_tcl_dst = Path(f'{model.config.get_output_dir()}/build_prj_bup.tcl').resolve()
        if build_bup_tcl_src.exists():
            copyfile(build_bup_tcl_src, build_bup_tcl_dst)

        # Optional bottom-up YAML flow description
        build_bup_yml_src = (filedir / '../templates/catapult/build_prj_bup.yml').resolve()
        build_bup_yml_dst = Path(f'{model.config.get_output_dir()}/build_prj_bup.yml').resolve()
        if build_bup_yml_src.exists():
            with open(build_bup_yml_src) as src, open(build_bup_yml_dst, 'w') as dst:
                for line in src.readlines():
                    indent = line[: len(line) - len(line.lstrip())]
                    line = line.replace('myproject', model.config.get_project_name())
                    line = line.replace('CATAPULT_DIR', model.config.get_project_dir())
                    if '#hls-fpga-machine-learning insert build_options' in line:
                        line = ''
                        build_options = {
                            'reset': 0,
                            'csim': 0,
                            'synth': 1,
                            'cosim': 0,
                            'validation': 0,
                            'vhdl': 1,
                            'verilog': 1,
                            'export': 0,
                            'vsynth': 0,
                            'bitfile': 0,
                            'fifo_opt': 0,
                            'ran_frame': 2,
                            'sw_opt': 0,
                            'power': 0,
                            'da': 0,
                            'bup': 1,
                        }
                        for key, value in build_options.items():
                            line += indent + f'{key}: {value}\n'
                    elif '#hls-fpga-machine-learning insert techlibs' in line:
                        if model.config.get_config_value('Technology') is None:
                            if model.config.get_config_value('Part') is not None:
                                line = indent + 'setup_xilinx_part {{{}}}\n'.format(model.config.get_config_value('Part'))
                            elif model.config.get_config_value('ASICLibs') is not None:
                                line = indent + 'setup_asic_libs {{{}}}\n'.format(
                                    model.config.get_config_value('ASICLibs')
                                )
                        else:
                            if model.config.get_config_value('Technology') == 'asic':
                                line = indent + 'setup_asic_libs {{{}}}\n'.format(
                                    model.config.get_config_value('ASICLibs')
                                )
                            else:
                                line = indent + 'setup_xilinx_part {{{}}}\n'.format(model.config.get_config_value('Part'))
                    elif '#hls-fpga-machine-learning insert invoke_args' in line:
                        tb_in_file = model.config.get_config_value('InputData')
                        tb_out_file = model.config.get_config_value('OutputPredictions')
                        invoke_args = '$sfd/firmware/weights'
                        if tb_in_file is not None:
                            invoke_args = invoke_args + ' $sfd/tb_data/tb_input_features.dat'
                        if tb_out_file is not None:
                            invoke_args = invoke_args + ' $sfd/tb_data/tb_output_predictions.dat'
                        line = indent + f'flow package option set /SCVerify/INVOKE_ARGS "{invoke_args}"\n'
                    elif 'set hls_clock_period 5' in line:
                        line = indent + 'set hls_clock_period {}\n'.format(model.config.get_config_value('ClockPeriod'))
                    elif 'options set Input/CompilerFlags' in line:
                        line = indent + f'options set Input/CompilerFlags "{compiler_flags}"\n'
                    dst.write(line)

        # build_lib.sh
        build_lib_src = (filedir / '../templates/catapult/build_lib.sh').resolve()
        build_lib_dst = Path(f'{model.config.get_output_dir()}/build_lib.sh').resolve()
        with open(build_lib_src) as src, open(build_lib_dst, 'w') as dst:
            for line in src.readlines():
                line = line.replace('myproject', model.config.get_project_name())
                line = line.replace('mystamp', model.config.get_config_value('Stamp'))

                dst.write(line)
        build_lib_dst.chmod(build_lib_dst.stat().st_mode | stat.S_IEXEC)

        # Optional VRA helper
        build_vra_src = (filedir / '../templates/catapult/build_vra.sh').resolve()
        build_vra_dst = Path(f'{model.config.get_output_dir()}/build_vra.sh').resolve()
        if build_vra_src.exists():
            with open(build_vra_src) as src, open(build_vra_dst, 'w') as dst:
                for line in src.readlines():
                    line = line.replace('myproject', model.config.get_project_name())
                    line = line.replace('mystamp', model.config.get_config_value('Stamp'))
                    if model.config.get_config_value('InputData') is not None:
                        line = line.replace(
                            'tb_input_features.dat', 'tb_data/' + os.path.basename(model.config.get_config_value('InputData'))
                        )
                    if model.config.get_config_value('OutputPredictions') is not None:
                        line = line.replace(
                            'tb_output_predictions.dat',
                            'tb_data/' + os.path.basename(model.config.get_config_value('OutputPredictions')),
                        )
                    dst.write(line)
            build_vra_dst.chmod(build_vra_dst.stat().st_mode | stat.S_IEXEC)

    def write_nnet_utils(self, model):
        """Copy the nnet_utils, AP types headers and any custom source to the project output directory

        Args:
            model (ModelGraph): the hls4ml model.
        """

        # nnet_utils
        filedir = os.path.dirname(os.path.abspath(__file__))

        srcpath = os.path.join(filedir, '../templates/catapult/nnet_utils/')
        dstpath = f'{model.config.get_output_dir()}/firmware/nnet_utils/'

        if not os.path.exists(dstpath):
            os.mkdir(dstpath)

        headers = [os.path.basename(h) for h in glob.glob(srcpath + '*.h')]

        if model.config.get_config_value('DontCopyNNET') is not None:
            h = 'nnet_code_gen.h'
            copyfile(srcpath + h, dstpath + h)
            return

        for h in headers:
            copyfile(srcpath + h, dstpath + h)

        # Copy behavioral subdirectory
        beh_srcpath = os.path.join(srcpath, 'behavioral/')
        beh_dstpath = os.path.join(dstpath, 'behavioral/')
        if os.path.exists(beh_srcpath):
            if os.path.exists(beh_dstpath):
                rmtree(beh_dstpath)
            copytree(beh_srcpath, beh_dstpath)

        print('Copying NNET files to local firmware directory')

        filedir = os.path.dirname(os.path.abspath(__file__))
        for pkg in ('ac_types', 'ac_math', 'ac_simutils'):
            dstpath = f'{model.config.get_output_dir()}/firmware/{pkg}/'

            # backward compatibility, look in root dir
            srcpath = os.path.join(filedir, '../../' + pkg + '/')
            if not os.path.exists(srcpath):
                # look next in Catapult-specific templates
                srcpath = os.path.join(filedir, '../templates/catapult/' + pkg + '/')

            if os.path.exists(srcpath):
                if os.path.exists(dstpath):
                    rmtree(dstpath)
                print('... copying AC ' + pkg + ' headers from ' + srcpath)
                copytree(srcpath, dstpath)
            else:
                print('... skipping copy of ' + pkg + ' headers - assumed to located in Catapult install tree')

        # custom source
        filedir = os.path.dirname(os.path.abspath(__file__))

        custom_source = get_backend('Catapult').get_custom_source()
        for dst, srcpath in custom_source.items():
            dstpath = f'{model.config.get_output_dir()}/firmware/{dst}'
            copyfile(srcpath, dstpath)

    def write_generated_code(self, model):
        """Write the generated code (nnet_code_gen.h)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        path = f'{model.config.get_output_dir()}/firmware/nnet_utils/nnet_code_gen.h'
        f = open(path)
        contents = f.readlines()
        f.close()
        f = open(path, 'w')

        for line in contents:
            if '// hls4ml insert code' in line:
                newline = line
                for layer in model.get_layers():
                    for generated_code in layer.code.values():
                        newline += str(generated_code)
            else:
                newline = line
            f.write(newline)
        f.close()

    def write_yml(self, model):
        """Write the config to the YAML file

        Args:
            model (ModelGraph): the hls4ml model.
        """

        def keras_model_representer(dumper, keras_model):
            model_path = model.config.get_output_dir() + '/keras_model.keras'
            keras_model.save(model_path)
            return dumper.represent_scalar('!keras_model', model_path)

        try:
            import keras

            KerasModel = keras.models.Model

            yaml.add_multi_representer(KerasModel, keras_model_representer)
        except Exception:
            pass

        with open(model.config.get_output_dir() + '/' + config_filename, 'w') as file:
            yaml.dump(model.config.config, file)

    def write_tar(self, model):
        """Write the generated project as a .tar.gz archive

        Args:
            model (ModelGraph): the hls4ml model.
        """

        if not os.path.exists(model.config.get_output_dir() + '.tar.gz'):
            with tarfile.open(model.config.get_output_dir() + '.tar.gz', mode='w:gz') as archive:
                archive.add(model.config.get_output_dir(), recursive=True)
        else:
            print('Project .tar.gz archive already exists')

    def write_hls(self, model):
        self.write_output_dir(model)
        self.write_project_cpp(model)
        self.write_project_header(model)
        self.write_weights(model)
        self.write_defines(model)
        self.write_parameters(model)
        self.write_test_bench(model)
        self.write_bridge(model)
        self.write_build_script(model)
        self.write_nnet_utils(model)
        self.write_generated_code(model)
        self.write_yml(model)
        self.write_tar(model)
        self.write_gemm_config(model)

    @staticmethod
    def _gemm_ip_interface(node):
        # Interface is DERIVED from IOType, never guessed: io_parallel -> array,
        # io_stream -> stream, uniformly for the unified Gemm/Im2ColGemm nodes and
        # the (Phase 1) Einsum/EinsumDense GEMM-IP layers. Because the same IOType
        # drives the instantiated call in the template, the declared interface can
        # never disagree with the core that is actually built.
        io_type = node.model.config.get_config_value('IOType')
        is_gemm_ip = isinstance(node, (Gemm, Im2ColGemm, Einsum, EinsumDense)) and bool(
            node.get_attr('strategy') == 'gemm'
        )
        if is_gemm_ip and io_type == 'io_parallel':
            return 'array'
        return 'stream'

    @staticmethod
    def _gemm_ip_protocol(interface, weight_layout='column_major'):
        # weight_layout: how a weight-stationary IP's constant operand is packed
        # (SecondOperandRowMajor): column_major = w_gemm_cols[N][K] (one K-high output
        # column per beat, default); row_major = w_gemm_rows[K][N] (one N-wide
        # contraction row per beat).
        if interface == 'array':
            return {
                'kind': 'catapult_ccore_array',
                'input_valid': 'scheduled_en_and_in_valid',
                'output_hold': 'fifo_until_en',
                'internal_run': 'self_timed_after_start',
                'result_order': 'row_major',
                'input_beat_order': 'row_major',
                'weight_layout': weight_layout,
            }
        # Row/column streaming contract:
        #   A stream: one K-wide row per cycle (row_major)
        #   B stream: one beat per cycle from the packed ROM (weight_layout)
        #   C stream: one N-wide row per cycle (row_major)
        return {
            'kind': 'catapult_ac_channel_stream',
            'result_order': 'row_major',
            'input_beat_order': 'row_major',
            'weight_layout': weight_layout,
        }

    @staticmethod
    def _gemm_ip_blackbox(node):
        return {
            'entity': f'{node.name}_core',
            'rtl': f'{node.name}/{node.name}_core.v',
            'clock': 'clk',
            'reset': 'rst',
            'reset_active': 'high',
            'start': 'en',
        }

    def _gemm_ip_metadata(self, node):
        interface = self._gemm_ip_interface(node)
        return {
            'layer_name': node.name,
            'gemm_ip_id': node.name,
            'gemm_ip_index': node.index,
            'interface': interface,
            'protocol': self._gemm_ip_protocol(interface, gemm_ip_weight_layout(node)),
            'blackbox': self._gemm_ip_blackbox(node),
            # The generator derives the blackbox combinational-delay budget
            # from the project clock so the wrapper schedule and the core
            # share one timing contract.
            'clock_period_ns': node.model.config.get_config_value('ClockPeriod'),
            # How the constant operand is packed (SecondOperandRowMajor): the ROM
            # header / .dat beat order; also mirrored in protocol.weight_layout.
            'weight_layout': gemm_ip_weight_layout(node),
        }

    def write_gemm_config(self, model):
        """Write a JSON file containing details of GEMM templates used in the design."""
        import json
        gemm_info = {}
        for node in model.graph.values():
            use_gemm_ip = bool(node.get_attr('strategy') == 'gemm')
            if use_gemm_ip:
                if isinstance(node, Einsum):
                    gemm_info[node.name] = {
                        'type': node.class_name,
                        'n_in': node.get_attr('n_in'),
                        'n_out': node.get_attr('n_out'),
                        'gemm_m': node.get_attr('gemm_m'),
                        'gemm_k': node.get_attr('gemm_k'),
                        'gemm_n': node.get_attr('gemm_n'),
                        'n_inplace': node.get_attr('n_inplace'),
                        'transpose_weights': True,
                        # Einsum (QK^T / A.V) is two-operand with no bias; bias is never
                        # in the IP here.
                        'bias_in_core': False,
                        'input_precision': str(node.get_input_variable(node.inputs[0]).type.precision),
                        'rhs_precision': str(node.get_input_variable(node.inputs[1]).type.precision),
                        'output_precision': str(node.get_output_variable().type.precision),
                        'weight_precision': str(node.get_input_variable(node.inputs[1]).type.precision),
                        'bias_precision': None,
                        'accum_precision': str(node.types['accum_t'].precision) if 'accum_t' in node.types else None,
                    }
                    gemm_info[node.name].update(self._gemm_ip_metadata(node))
                elif isinstance(node, EinsumDense):
                    gemm_info[node.name] = {
                        'type': node.class_name,
                        'n_in': node.get_attr('n_in'),
                        'n_out': node.get_attr('n_out'),
                        'gemm_m': node.get_attr('gemm_m', node.get_attr('n_free_data')),
                        'gemm_k': node.get_attr('gemm_k', node.get_attr('n_contract')),
                        'gemm_n': node.get_attr('gemm_n', node.get_attr('n_free_kernel')),
                        # EinsumDense-specific shape fields; useful for RTL tooling
                        # that needs to know the original contraction dimensions
                        # rather than the packed GEMM tile sizes.
                        'n_free_data': node.get_attr('n_free_data'),
                        'n_free_kernel': node.get_attr('n_free_kernel'),
                        'n_contract': node.get_attr('n_contract'),
                        'n_inplace': node.get_attr('n_inplace'),
                        'transpose_weights': True,
                        # The einsum_dense GEMM path feeds the IP a zero bias and adds
                        # the (possibly per-element) bias in the wrapper, so the IP omits it.
                        'bias_in_core': False,
                        'input_precision': str(node.get_input_variable().type.precision),
                        'output_precision': str(node.get_output_variable().type.precision),
                        'weight_precision': str(node.get_weights('weight').type.precision),
                        'bias_precision': str(node.get_weights('bias').type.precision) if node.get_weights('bias') else None,
                        'accum_precision': str(node.types['accum_t'].precision) if 'accum_t' in node.types else None,
                    }
                    gemm_info[node.name].update(self._gemm_ip_metadata(node))
                else:
                    gemm_info[node.name] = {
                        'type': node.class_name,
                        'n_in': node.get_attr('n_in'),
                        'n_out': node.get_attr('n_out'),
                        'gemm_m': node.get_attr('gemm_m', node.get_attr('n_patches', 1)),
                        'gemm_k': node.get_attr('gemm_k', node.get_attr('n_in')),
                        'gemm_n': node.get_attr('gemm_n', node.get_attr('n_out')),
                        'transpose_weights': bool(node.get_attr('strategy') == 'gemm')
                        or node.model.config.get_layer_config_value(node, 'TransposeWeights', False),
                        # Weight-stationary: the external GEMM IP holds the packed
                        # weights internally and hls4ml calls the const_weights signature.
                        'weights_in_core': bool(node.get_attr('weights_in_core', False)),
                        # Whether the IP itself should include the bias adder. True only
                        # for the per-column weight-stationary case, where hls4ml feeds
                        # the real bias to the IP's bias port. False when hls4ml feeds a
                        # zero and adds bias in the generated wrapper instead: two-operand
                        # GEMM (QK^T / A.V, no bias) and row-varying EinsumDense bias
                        # (per-element add the per-column port can't express). gemm-ip-gen
                        # honors this to omit the bias adder/port (a later gemm-ip-gen phase).
                        'bias_in_core': bool(node.get_attr('weights_in_core', False))
                        and not bool(node.get_attr('_row_varying_bias', False)),
                        'input_precision': str(node.get_input_variable().type.precision),
                        'output_precision': str(node.get_output_variable().type.precision),
                        # Two-operand Gemm (attention QK^T / A.V) has no constant weight/bias.
                        'weight_precision': str(node.get_weights('weight').type.precision)
                        if node.get_attr('weight') is not None else None,
                        'bias_precision': str(node.get_weights('bias').type.precision)
                        if node.get_attr('bias') is not None else None,
                        'accum_precision': str(node.types['accum_t'].precision) if 'accum_t' in node.types else None,
                    }
                    gemm_info[node.name].update(self._gemm_ip_metadata(node))
                # Weight-stationary: point the external generator at the raw-bits .dat.
                if bool(node.get_attr('weights_in_core', False)):
                    try:
                        wname = node.get_weights('weight').name
                        # Path is relative to gemm_config.json (in output_dir); the .dat
                        # lives under output_dir/firmware/weights/.
                        basename = gemm_ip_weight_basename(node.get_weights('weight'), node)
                        gemm_info[node.name]['weight_file'] = f'firmware/weights/{basename}.dat'
                        gemm_info[node.name]['weight_layout'] = gemm_ip_weight_layout(node)
                    except Exception:
                        pass

        if gemm_info:
            output_dir = model.config.get_output_dir()
            with open(f'{output_dir}/gemm_config.json', 'w') as f:
                json.dump(gemm_info, f, indent=4)
            print(f'Wrote GEMM configuration to {output_dir}/gemm_config.json')
