import glob
import os
import stat
import tarfile
from collections import OrderedDict
from pathlib import Path
from shutil import copyfile, copytree, rmtree

import numpy as np

from hls4ml.writer.gemm_ip_weights import (
    gemm_ip_weight_basename,
    gemm_ip_weight_layout,
    write_gemm_ip_weight_cols,
)
import yaml

from hls4ml.backends.fpga.gemm.gemm_nodes import Gemm
from hls4ml.model.layers import Einsum, EinsumDense
from hls4ml.writer.gemm_ip_json import write_gemm_config_json
from hls4ml.writer.writers import Writer

config_filename = 'hls4ml_config.yml'


class VivadoWriter(Writer):
    @staticmethod
    def _uses_gemm_ip(model):
        """Return True if any layer in the model uses GEMM IP."""
        return any(bool(node.get_attr('strategy') == 'gemm') for node in model.graph.values())

    @staticmethod
    def _gemm_ip_metadata(node):
        interface = VivadoWriter._gemm_ip_interface(node)
        return {
            'layer_name': node.name,
            'gemm_ip_id': node.name,
            'gemm_ip_index': node.index,
            # The one function the firmware calls for this node (io_stream): the package
            # must define exactly this name on packed bit streams (see nnet_gemm_pack.h).
            'function': f'gemm_stream_{node.name}',
            'interface': interface,
            'protocol': VivadoWriter._gemm_ip_protocol(interface, gemm_ip_weight_layout(node)),
            'blackbox': VivadoWriter._gemm_ip_blackbox(node),
            # The generator derives the blackbox combinational-delay budget
            # from the project clock so the wrapper schedule and the core
            # share one timing contract.
            'clock_period_ns': node.model.config.get_config_value('ClockPeriod'),
            'part': node.model.config.get_config_value('Part'),
            'clock_uncertainty': node.model.config.get_config_value('ClockUncertainty'),
            # Config resolved onto the node at GEMM lowering (resolve-then-store).
            'strategy': node.get_attr('strategy', 'latency'),
            'reuse_factor': node.get_attr('reuse_factor', 1),
            'parallelization_factor': node.get_attr('parallelization_factor', 1),
            'target_cycles': node.get_attr('target_cycles', None),
            'sparse': bool(node.get_attr('sparse', False)),
            # Derived from the bias tensor itself (non-all-zero) in gemm_nodes.py;
            # the single source of truth for whether a real bias exists.
            'has_bias': bool(node.get_attr('has_bias', False)),
            # SecondOperandRowMajor, resolved per layer. Two-operand GEMM: the B beat
            # layout the emitted call uses (row_major = N-wide beats, one contraction
            # row/beat, mvau IP; col_major = K-wide beats, one output column/beat,
            # default). Weight-stationary GEMM: how the constant operand is packed into
            # the ROM header / .dat (see weight_layout, mirrored in protocol).
            'second_operand_row_major': bool(node.get_attr('second_operand_row_major', False)),
            'second_operand_beat_order': (
                'row_major' if node.get_attr('second_operand_row_major', False) else 'col_major'
            ),
            'weight_layout': gemm_ip_weight_layout(node),
        }

    @staticmethod
    def _gemm_ip_interface(node):
        io_type = node.model.config.get_config_value('IOType')
        if isinstance(node, (Gemm, Einsum, EinsumDense)) and io_type == 'io_parallel' and bool(
            node.get_attr('strategy') == 'gemm'
        ):
            return 'array'
        return 'stream'

    @staticmethod
    def _gemm_ip_protocol(interface, weight_layout='column_major'):
        protocol = {
            'result_order': 'row_major',
            'input_beat_order': 'row_major',
            'weight_layout': weight_layout,
        }
        if interface == 'array':
            return {
                'kind': 'xilinx_hls_array',
                'input_valid': 'ap_vld',
                'output_hold': 'ap_stable',
                'internal_run': 'self_timed_after_start',
                **protocol,
            }
        return {'kind': 'xilinx_hls_stream', **protocol}

    @staticmethod
    def _gemm_ip_blackbox(node):
        return {
            'entity': f'{node.name}_wrapper',
            'rtl': f'{node.name}/{node.name}.v',
            'clock': 'ap_clk',
            'reset': 'ap_rst',
            'reset_active': 'high',
            'start': 'ap_start',
            'done': 'ap_done',
            'idle': 'ap_idle',
            'ready': 'ap_ready',
        }

    def write_gemm_config(self, model):
        """Write a JSON file containing details of GEMM IP templates used in the design."""
        from hls4ml.model.layers import Im2Col

        gemm_info = {}
        for node in model.graph.values():
            # Im2Col carries strategy='gemm' too (it selects the im2col_*_gemm_rows
            # emitter), but it is a reshaping node, not a GEMM-IP consumer -- it
            # never gets a manifest entry.
            use_gemm_ip = bool(node.get_attr('strategy') == 'gemm') and not isinstance(node, Im2Col)
            if not use_gemm_ip:
                continue

            has_weights = 'weight' in node.weights
            has_bias = 'bias' in node.weights
            if isinstance(node, Einsum):
                info = {
                    'type': node.class_name,
                    'n_in': node.get_attr('n_in'),
                    'n_out': node.get_attr('n_out'),
                    'gemm_m': node.get_attr('gemm_m'),
                    'gemm_k': node.get_attr('gemm_k'),
                    'gemm_n': node.get_attr('gemm_n'),
                    'n_inplace': node.get_attr('n_inplace'),
                    'transpose_weights': True,
                    # Einsum (QK^T / A.V) is two-operand with no bias tensor at all.
                    'has_bias': bool(node.get_attr('has_bias', False)),
                    'input_precision': str(node.get_input_variable(node.inputs[0]).type.precision),
                    'rhs_precision': str(node.get_input_variable(node.inputs[1]).type.precision),
                    'output_precision': str(node.get_output_variable().type.precision),
                    'weight_precision': str(node.get_input_variable(node.inputs[1]).type.precision),
                    'bias_precision': None,
                    'accum_precision': str(node.types['accum_t'].precision) if 'accum_t' in node.types else None,
                }
            elif isinstance(node, EinsumDense):
                info = {
                    'type': node.class_name,
                    'n_in': node.get_attr('n_in'),
                    'n_out': node.get_attr('n_out'),
                    'gemm_m': node.get_attr('gemm_m', node.get_attr('n_free_data')),
                    'gemm_k': node.get_attr('gemm_k', node.get_attr('n_contract')),
                    'gemm_n': node.get_attr('gemm_n', node.get_attr('n_free_kernel')),
                    'n_free_data': node.get_attr('n_free_data'),
                    'n_free_kernel': node.get_attr('n_free_kernel'),
                    'n_contract': node.get_attr('n_contract'),
                    'n_inplace': node.get_attr('n_inplace'),
                    'transpose_weights': True,
                    # has_bias is the single source of truth for the external IP: the
                    # bias tensor is non-all-zero AND it lives on the per-column
                    # weight-stationary port (a row-varying EinsumDense bias can't be
                    # expressed there -- hls4ml adds it in the generated wrapper instead,
                    # so the IP sees no bias at all in that case).
                    'has_bias': bool(node.get_attr('has_bias', False))
                    and not bool(node.get_attr('_row_varying_bias', False)),
                    'input_precision': str(node.get_input_variable().type.precision),
                    'output_precision': str(node.get_output_variable().type.precision),
                    'weight_precision': str(node.get_weights('weight').type.precision) if has_weights else None,
                    'bias_precision': str(node.get_weights('bias').type.precision) if has_bias else None,
                    # Raw per-column bias values, baked as a compile-time constant by
                    # the external IP generator (gemm-ip-gen) instead of a runtime port.
                    # Only present when the collapsed 'has_bias' above is True.
                    'bias': (
                        [float(x) for x in np.asarray(node.get_weights('bias').data).reshape(-1)]
                        if has_bias and not bool(node.get_attr('_row_varying_bias', False))
                        else None
                    ),
                    'accum_precision': str(node.types['accum_t'].precision) if 'accum_t' in node.types else None,
                }
            else:
                info = {
                    'type': node.class_name,
                    'n_in': node.get_attr('n_in'),
                    'n_out': node.get_attr('n_out'),
                    'gemm_m': node.get_attr('gemm_m', node.get_attr('n_patches', 1)),
                    'gemm_k': node.get_attr('gemm_k', node.get_attr('n_in')),
                    'gemm_n': node.get_attr('gemm_n', node.get_attr('n_out')),
                    'transpose_weights': use_gemm_ip
                    or node.model.config.get_layer_config_value(node, 'TransposeWeights', False),
                    # Weight-stationary: the external GEMM IP holds the packed
                    # weights internally and hls4ml calls the const_weights signature.
                    'weights_in_core': bool(node.get_attr('weights_in_core', False)),
                    # has_bias is the single source of truth, derived from the node's
                    # bias tensor (non-all-zero) in gemm_nodes.py, AND not row-varying
                    # (a row-varying EinsumDense bias can't live on the per-column
                    # weight-stationary port -- hls4ml adds it in the generated wrapper
                    # instead, so the external IP sees no bias in that case). Two-operand
                    # GEMM (QK^T / A.V) is always has_bias False by construction.
                    'has_bias': bool(node.get_attr('has_bias', False))
                    and not bool(node.get_attr('_row_varying_bias', False)),
                    'input_precision': str(node.get_input_variable().type.precision),
                    'output_precision': str(node.get_output_variable().type.precision),
                    # Weight precision: the static weight tensor for a weight-stationary
                    # GEMM; for a two-operand GEMM (QK^T / A.V, no static weight) the
                    # "weight" is the second runtime operand B = inputs[1], so take its
                    # precision -- the MVU weight port carries B, and gemm-ip-gen needs
                    # its width/frac to size the core and the requant scale.
                    'weight_precision': (
                        str(node.get_weights('weight').type.precision) if has_weights
                        else str(node.get_input_variable(node.inputs[1]).type.precision)
                        if len(node.inputs) > 1 else None
                    ),
                    'bias_precision': str(node.get_weights('bias').type.precision) if has_bias else None,
                    # Raw per-column bias values, baked as a compile-time constant by
                    # the external IP generator (gemm-ip-gen) instead of a runtime port.
                    # Only present when the collapsed 'has_bias' above is True.
                    'bias': (
                        [float(x) for x in np.asarray(node.get_weights('bias').data).reshape(-1)]
                        if has_bias and not bool(node.get_attr('_row_varying_bias', False))
                        else None
                    ),
                    'accum_precision': str(node.types['accum_t'].precision) if 'accum_t' in node.types else None,
                }
            if use_gemm_ip:
                info.update(self._gemm_ip_metadata(node))
            # Weight-stationary: point the external generator at the raw-bits .dat.
            if bool(node.get_attr('weights_in_core', False)):
                try:
                    wname = node.get_weights('weight').name
                    # Path is relative to gemm_config.json (in output_dir); the .dat
                    # lives under output_dir/firmware/weights/.
                    basename = gemm_ip_weight_basename(node.get_weights('weight'), node)
                    info['weight_file'] = f'firmware/weights/{basename}.dat'
                    info['weight_layout'] = gemm_ip_weight_layout(node)
                except Exception:
                    pass
            gemm_info[node.name] = info

        if gemm_info:
            output_dir = model.config.get_output_dir()
            write_gemm_config_json(f'{output_dir}/gemm_config.json', gemm_info)
            print(f'Wrote GEMM configuration to {output_dir}/gemm_config.json')

    def print_array_to_cpp(self, var, odir, namespace=None, write_txt_file=True):
        """Write a weights array to C++ header files.

        Args:
            var (WeightVariable): Weight to write
            odir (str): Output directory
            namespace (str, optional): Writes a namespace for the weights to avoid clashes with global variables.
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

        if namespace is not None:
            h_file.write(f'namespace {namespace} {{\n\n')

        if write_txt_file:
            h_file.write('#ifndef __SYNTHESIS__\n')
            h_file.write(var.definition_cpp() + ';\n')
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
        h_file.write('};\n\n')

        if write_txt_file:
            h_file.write('#endif\n')
            txt_file.close()

        if namespace is not None:
            h_file.write('}\n\n')

        h_file.write('\n#endif\n')
        h_file.close()

    def write_project_dir(self, model):
        """Write the base project directory

        Args:
            model (ModelGraph): the hls4ml model.
        """
        if not os.path.isdir(f'{model.config.get_output_dir()}/firmware/weights'):
            os.makedirs(f'{model.config.get_output_dir()}/firmware/weights')

    @staticmethod
    def _make_array_pragma(variable):
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
                template = '#pragma HLS ARRAY_{mode} variable={name} {type} dim={dim}'
            else:
                template = '#pragma HLS ARRAY_{mode} variable={name} {type} factor={factor} dim={dim}'

            return template.format(mode=mode.upper(), name=variable.name, type=typ, factor=factor, dim=0)

        elif mode == 'stream':
            return f'#pragma HLS STREAM variable={variable.name} depth={depth}'

    def write_project_cpp(self, model):
        """Write the main architecture source file (myproject.cpp)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        filedir = os.path.dirname(os.path.abspath(__file__))

        f = open(os.path.join(filedir, '../templates/vivado/firmware/myproject.cpp'))
        fout = open(f'{model.config.get_output_dir()}/firmware/{model.config.get_project_name()}.cpp', 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        indent = '    '

        for line in f.readlines():
            # Add headers to weights and biases
            if 'myproject' in line:
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

            elif '// hls-fpga-machine-learning insert namespace-start' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += f'namespace {namespace} {{\n'

            elif '// hls-fpga-machine-learning insert namespace-end' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += '}\n'

            elif '// hls-fpga-machine-learning insert load weights' in line:
                newline = line
                if model.config.get_writer_config()['WriteWeightsTxt']:
                    newline += '#ifndef __SYNTHESIS__\n'
                    newline += '    static bool loaded_weights = false;\n'
                    newline += '    if (!loaded_weights) {\n'

                    for layer in model.get_layers():
                        for w in layer.get_weights():
                            if w.weight_class == 'CompressedWeightVariable':
                                newline += (
                                    indent
                                    + '    nnet::load_compressed_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                                        w.type.name, w.nonzeros, w.name, w.name
                                    )
                                )
                            elif w.weight_class == 'ExponentWeightVariable':
                                newline += (
                                    indent
                                    + '    nnet::load_exponent_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                                        w.type.name, w.data_length, w.name, w.name
                                    )
                                )
                            else:
                                newline += indent + '    nnet::load_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                                    w.type.name, w.data_length, w.name, w.name
                                )

                    newline += '        loaded_weights = true;'
                    newline += '    }\n'
                    newline += '#endif'

            # Add input/output type
            elif '// hls-fpga-machine-learning insert IO' in line:
                newline = line
                all_inputs = [i.name for i in model_inputs]
                all_outputs = [o.name for o in model_outputs]
                all_brams = [b.name for b in model_brams]
                io_type = model.config.get_config_value('IOType')

                pipeline_style = model.config.pipeline_style
                pipeline_ii = model.config.pipeline_ii
                pipeline_pragma = indent + f'#pragma HLS {pipeline_style.upper()}'
                if pipeline_style == 'pipeline' and pipeline_ii is not None:
                    pipeline_pragma += f' II={pipeline_ii}\n'
                else:
                    pipeline_pragma += '\n'

                if io_type == 'io_parallel':
                    for i in model_inputs:
                        newline += indent + self._make_array_pragma(i) + '\n'
                    for o in model_outputs:
                        newline += indent + self._make_array_pragma(o) + '\n'
                    # TODO discussed adding a handle for setting the interface mode for individual input and output arrays
                    # Probably the handle doesn't need to be exposed to the user but should be just set in hls_model.py
                    newline += indent + '#pragma HLS INTERFACE ap_vld port={},{} \n'.format(
                        ','.join(all_inputs), ','.join(all_outputs)
                    )
                    if all_brams:
                        newline += indent + '#pragma HLS INTERFACE bram port={} \n'.format(','.join(all_brams))
                    newline += pipeline_pragma

                if io_type == 'io_stream':
                    newline += indent + '#pragma HLS INTERFACE axis port={},{} \n'.format(
                        ','.join(all_inputs), ','.join(all_outputs)
                    )
                    if all_brams:
                        newline += indent + '#pragma HLS INTERFACE bram port={} \n'.format(','.join(all_brams))
                    newline += pipeline_pragma

            elif '// hls-fpga-machine-learning insert layers' in line:
                newline = line + '\n'
                for layer in model.get_layers():
                    vars = layer.get_variables()
                    for var in vars:
                        if var not in model_inputs and var not in model_outputs:
                            def_cpp = var.definition_cpp()
                            if def_cpp is not None:
                                newline += '    ' + def_cpp + ';\n'
                                if var.pragma:
                                    newline += '    ' + self._make_array_pragma(var) + '\n\n'
                for layer in model.get_layers():
                    func = layer.get_attr('function_cpp', None)
                    if func:
                        if not isinstance(func, (list, set)):
                            func = [func]
                        if len(func) == 1:
                            newline += '    ' + func[0] + ' // ' + layer.name + '\n'
                        else:
                            newline += '    // ' + layer.name + '\n'
                            for line in func:
                                newline += '    ' + line + '\n'
                        if model.config.trace_output and layer.get_attr('trace', False):
                            vars = layer.get_variables()
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
        f = open(os.path.join(filedir, '../templates/vivado/firmware/myproject.h'))
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

            elif '// hls-fpga-machine-learning insert namespace-start' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += f'namespace {namespace} {{\n'

            elif '// hls-fpga-machine-learning insert namespace-end' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += '}\n'

            elif '// hls-fpga-machine-learning insert emulator-defines' in line:
                newline = line

                if model.config.get_writer_config().get('WriteEmulationConstants', False):
                    brams_def_str = ', '.join([b.definition_cpp(as_reference=False) for b in model_brams])
                    brams_call_str = ', '.join([b.name for b in model_brams])

                    if model.config.get_config_value('IOType') == 'io_stream':
                        input_call_str = ', '.join([f'std::get<{n}>(inputs)' for n in range(len(model_inputs))])
                        output_call_str = ', '.join([f'std::get<{n}>(outputs)' for n in range(len(model_outputs))])
                    else:
                        input_call_str = ', '.join([f'std::get<{n}>(inputs).data()' for n in range(len(model_inputs))])
                        output_call_str = ', '.join([f'std::get<{n}>(outputs).data()' for n in range(len(model_outputs))])

                    newline += (
                        f'\ninline void {model.config.get_project_name()}_emulator('
                        'inputs_t& inputs, outputs_t& outputs'  # the inputs_t should ideally be const
                    )
                    if len(model_brams) > 0:
                        newline += ',\n' + brams_def_str
                    newline += ') {\n'
                    newline += indent + model.config.get_project_name() + '(\n'
                    newline += indent + indent + input_call_str + ',\n'
                    newline += indent + indent + output_call_str
                    if len(model_brams) > 0:
                        newline += ',\n' + indent + indent + brams_call_str
                    newline += '\n' + indent + ');\n}\n'

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
        f = open(os.path.join(filedir, '../templates/vivado/firmware/defines.h'))
        fout = open(f'{model.config.get_output_dir()}/firmware/defines.h', 'w')

        for line in f.readlines():
            if '// hls-fpga-machine-learning insert headers' in line:
                uses_stdfloat = False
                uses_apfloat = False
                for layer in model.get_layers():
                    layer_precision = layer.get_layer_precision()
                    for type_var in layer_precision.values():
                        cpp = type_var.definition_cpp()
                        if 'std::' in cpp:
                            uses_stdfloat = True
                            break
                        if 'ap_float' in cpp:
                            uses_apfloat = True
                            break
                if uses_stdfloat:
                    newline = line + '#include <stdfloat>\n'
                if uses_apfloat:
                    newline = line + '#include "ap_float.h"\n'

            elif '// hls-fpga-machine-learning insert layer-precision' in line:
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

            elif '// hls-fpga-machine-learning insert namespace-start' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += f'namespace {namespace} {{\n'

            elif '// hls-fpga-machine-learning insert namespace-end' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += '}\n'

            elif '// hls-fpga-machine-learning insert emulator-defines' in line:
                newline = line

                if model.config.get_writer_config().get('WriteEmulationConstants', False):
                    if model.config.get_config_value('IOType') == 'io_stream':
                        input_types = [f'hls::stream<{v.type.name}>' for v in model.get_input_variables()]
                        output_types = [f'hls::stream<{v.type.name}>' for v in model.get_output_variables()]
                    else:
                        input_types = [f'std::array<{v.type.name}, {v.size_cpp()}>' for v in model.get_input_variables()]
                        output_types = [f'std::array<{v.type.name}, {v.size_cpp()}>' for v in model.get_output_variables()]
                    input_types_str = ', '.join(input_types)
                    output_types_str = ', '.join(output_types)
                    newline += '\n' + f'using inputs_t = std::tuple<{input_types_str}>;'
                    newline += '\n' + f'using outputs_t = std::tuple<{output_types_str}>;\n'
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
        f = open(os.path.join(filedir, '../templates/vivado/firmware/parameters.h'))
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
                            if self._is_gemm_ip_weight(layer, w):
                                newline += f'#include "weights/{gemm_ip_weight_basename(w, layer)}.h"\n'

            elif '// hls-fpga-machine-learning insert layer-config' in line:
                newline = line
                for layer in model.get_layers():
                    config = layer.get_attr('config_cpp', None)
                    if config:
                        newline += '// ' + layer.name + '\n'
                        newline += config + '\n'

            elif '// hls-fpga-machine-learning insert gemm-ip' in line:
                newline = line + self._gemm_ip_hook(model)

            elif '// hls-fpga-machine-learning insert namespace-start' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += f'namespace {namespace} {{\n'

            elif '// hls-fpga-machine-learning insert namespace-end' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += '}\n'

            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    def _gemm_ip_stream_nodes(self, model):
        """The io_stream GEMM nodes whose IP the firmware calls as gemm_stream_<name>."""
        if model.config.get_config_value('IOType') != 'io_stream':
            return []
        return [node for node in model.get_layers()
                if isinstance(node, Gemm) and node.get_attr('strategy') == 'gemm']

    def _gemm_ip_hook(self, model):
        """Text for the gemm-ip hook in parameters.h, after the layer configs.

        With a package (-DGEMM_IP_HEADER) the package's combined header supplies every
        gemm_stream_<name>: a declaration for an RTL blackbox, a definition in terms of
        the node's config for a soft-logic target -- which is why it must come after the
        configs. Without a package, csim gets the same names defined here on the
        behavioral kernels (nnet_gemm_pack.h), and synthesis gets declarations only, so
        a package-less synthesis fails loudly at link rather than silently.
        """
        nodes = self._gemm_ip_stream_nodes(model)
        if not nodes:
            return ''
        decls, defs = [], []
        for node in nodes:
            fn = f'gemm_stream_{node.name}'
            out_t = node.get_output_variable().type.name
            cfg = f'config{node.index}'
            if node.get_attr('weights_in_core', True):
                in_t = node.get_input_variable().type.name
                sig = (f'hls::stream<ap_uint<nnet::gemm_packed_bits<{in_t}>::value> > &a, '
                       f'hls::stream<ap_uint<nnet::gemm_packed_bits<{out_t}>::value> > &p')
                body = f'    nnet::gemm_stream_packed_const_weights<{in_t}, {out_t}, {cfg}>(a, p);'
            else:
                in0_t = node.get_input_variable(node.inputs[0]).type.name
                in1_t = node.get_input_variable(node.inputs[1]).type.name
                sig = (f'hls::stream<ap_uint<nnet::gemm_packed_bits<{in0_t}>::value> > &a, '
                       f'hls::stream<ap_uint<nnet::gemm_packed_bits<{in1_t}>::value> > &b, '
                       f'hls::stream<ap_uint<nnet::gemm_packed_bits<{out_t}>::value> > &p')
                body = f'    nnet::gemm_stream_packed<{in0_t}, {in1_t}, {out_t}, {cfg}>(a, b, p);'
            decls.append(f'void {fn}({sig});')
            defs.append(f'inline void {fn}({sig}) {{\n{body}\n}}')
        return (
            '// GEMM IP entry points, one concrete function per io_stream GEMM node.\n'
            '#if defined(GEMM_IP_HEADER)\n'
            '#include "gemm_ip_combined.h"\n'
            '#elif !defined(__SYNTHESIS__)\n'
            '// no package: csim on the behavioral kernels\n'
            + '\n'.join(defs) + '\n'
            '#else\n'
            '// no package under synthesis: declarations only -> loud link failure (intended)\n'
            + '\n'.join(decls) + '\n'
            '#endif\n'
        )

    def _is_gemm_ip_weight(self, layer, weights):
        if not bool(layer.get_attr('strategy') == 'gemm'):
            return False
        try:
            return weights.name == layer.get_weights('weight').name
        except Exception:
            return False

    def print_gemm_ip_weight_beats_to_cpp(self, var, layer, odir):
        """Write GEMM-IP packed weight columns (shared implementation)."""
        write_gemm_ip_weight_cols(var, layer, odir)

    def write_weights(self, model):
        """Write the weights into header files

        Args:
            model (ModelGraph): the hls4ml model.
        """
        namespace = model.config.get_writer_config().get('Namespace', None)
        write_txt = model.config.get_writer_config().get('WriteWeightsTxt', True)
        for layer in model.get_layers():
            for weights in layer.get_weights():
                self.print_array_to_cpp(
                    weights, model.config.get_output_dir(), namespace=namespace, write_txt_file=write_txt
                )
                if self._is_gemm_ip_weight(layer, weights):
                    # ROM header: source of truth for the streamed-weight path AND
                    # the weight-stationary csim behavioral model (native, no gemm-ip-gen).
                    self.print_gemm_ip_weight_beats_to_cpp(weights, layer, model.config.get_output_dir())
                    if bool(layer.get_attr('weights_in_core', False)):
                        # Weight-stationary also emits raw-bits .dat for the external
                        # generator to bake into the synth const_weights core.
                        from hls4ml.writer.gemm_ip_weights import write_gemm_ip_weight_dat

                        write_gemm_ip_weight_dat(weights, layer, model.config.get_output_dir())

    def write_multigraph_weights(self, model):
        """Write the weights into header files

        Args:
            model (MultiModelGraph): the hls4ml multigraph model.
        """
        namespace = model.config.get_writer_config().get('Namespace', None)
        write_txt = model.config.get_writer_config().get('WriteWeightsTxt', True)
        for g in model.graphs:
            for layer in g.get_layers():
                for weights in layer.get_weights():
                    self.print_array_to_cpp(
                        weights, model.config.get_output_dir(), namespace=namespace, write_txt_file=write_txt
                    )

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

        f = open(os.path.join(filedir, '../templates/vivado/myproject_test.cpp'))
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

            elif '// hls-fpga-machine-learning insert load weights' in line:
                newline = line
                if model_brams and model.config.get_writer_config()['WriteWeightsTxt']:
                    for bram in model_brams:
                        newline += indent + 'nnet::load_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                            bram.type.name, bram.data_length, bram.name, bram.name
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

            elif '// hls-fpga-machine-learning insert zero' in line:
                newline = line
                for inp in model_inputs:
                    newline += indent + inp.definition_cpp() + ';\n'
                    newline += indent + f'nnet::fill_zero<{inp.type.name}, {inp.size_cpp()}>({inp.name});\n'
                for out in model_outputs:
                    newline += indent + out.definition_cpp() + ';\n'

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
                tb_stream = model.config.get_writer_config().get('TBOutputStream', 'both')
                if tb_stream != 'stdout':
                    for out in model_outputs:
                        newline += indent + 'nnet::print_result<{}, {}>({}, fout);\n'.format(
                            out.type.name, out.size_cpp(), out.name
                        )  # TODO enable this

            elif (
                '// hls-fpga-machine-learning insert output' in line
                or '// hls-fpga-machine-learning insert quantized' in line
            ):
                newline = line
                tb_stream = model.config.get_writer_config().get('TBOutputStream', 'both')
                keep_output = str(tb_stream != 'stdout').lower()  # We keep output if we need to write it to file too.
                if tb_stream != 'file':
                    for out in model_outputs:
                        newline += indent + 'nnet::print_result<{}, {}>({}, std::cout, {});\n'.format(
                            out.type.name, out.size_cpp(), out.name, keep_output
                        )

            elif '// hls-fpga-machine-learning insert namespace' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += indent + f'using namespace {namespace};\n'

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
        f = open(os.path.join(filedir, '../templates/vivado/myproject_bridge.cpp'))
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

            elif '// hls-fpga-machine-learning insert bram' in line:
                newline = line
                for bram in model_brams:
                    newline += f'#include "firmware/weights/{bram.name}.h"\n'

            elif '// hls-fpga-machine-learning insert header' in line:
                dtype = line.split('#', 1)[1].strip()
                inputs_str = ', '.join([f'{dtype} *{i.name}' for i in model_inputs])
                outputs_str = ', '.join([f'{dtype} *{o.name}' for o in model_outputs])

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

            elif '// hls-fpga-machine-learning insert namespace' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += indent + f'using namespace {namespace};\n'

            else:
                newline = line
            fout.write(newline)

        f.close()
        fout.close()

    def write_bridge_multigraph(self, model):
        """Write the Python-C++ bridge (myproject_stitched_bridge.cpp)
        Args:
            model (MultiModelGraph): the hls4ml multigraph model.
        """

        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/vivado/myproject_bridge.cpp'))
        fout = open(f'{model.config.get_output_dir()}/{model.config.get_project_name()}_bridge.cpp', 'w')
        model_inputs = model.graphs[0].get_input_variables()
        model_outputs = model.graphs[-1].get_output_variables()
        model_brams = [var for var in model.graphs[0].get_weight_variables() if var.storage.lower() == 'bram']

        indent = '    '

        for line in f.readlines():
            newline = ''
            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT', format(model.config.get_project_name().upper()))
            elif 'firmware/myproject' in line:
                for graph_idx, g in enumerate(model.graphs):
                    newline += '#undef DEFINES_H_\n'
                    if len(g.outputs) == 1:
                        newline += '#define result_t ' + 'result_graph' + str(graph_idx + 1) + '_t\n'
                    newline += line.replace('myproject', format(model.graphs[graph_idx].config.get_project_name()))
                    if len(g.outputs) == 1:
                        newline += (
                            'typedef result_graph' + str(graph_idx + 1) + '_t graph' + str(graph_idx + 1) + '_result_t;\n'
                        )
                        newline += '#undef result_t\n\n' if graph_idx < len(model.graphs) - 1 else '\n'
                newline += '\n'
            elif 'myproject' in line:
                newline = line.replace('myproject', format(model.config.get_project_name()))

            elif '// hls-fpga-machine-learning insert bram' in line:
                newline = line
                for bram in model_brams:
                    newline += f'#include "firmware/weights/{bram.name}.h"\n'

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

                for idx, g in enumerate(model.graphs):
                    for o in g.get_output_variables():
                        definition = o.definition_cpp(name_suffix='_ap')
                        if len(g.outputs) == 1:
                            parts = definition.split(' ', 1)
                            datatype = 'graph' + str(idx + 1) + '_result_t'
                            if parts[0].startswith('hls::stream'):
                                modified_definition = 'hls::stream<' + datatype + '> ' + parts[1]
                            else:
                                modified_definition = datatype + ' ' + parts[1]
                            newline += indent + f'{modified_definition};\n'
                        else:
                            newline += indent + f'{definition};\n'

                newline += '\n'

                top_level = ''
                output_vars = ''
                for idx, g in enumerate(model.graphs):
                    if idx == 0:
                        input_vars = ','.join([i.name + '_ap' for i in g.get_input_variables()])
                    else:
                        input_vars = output_vars
                    bram_vars = ','.join(
                        [b.name for b in [var for var in g.get_weight_variables() if var.storage.lower() == 'bram']]
                    )
                    output_vars = ','.join([o.name + '_ap' for o in g.get_output_variables()])
                    # Concatenate the input, output, and bram variables. Filter out empty/null values
                    all_vars = ','.join(filter(None, [input_vars, output_vars, bram_vars]))
                    top_level += indent + f'{g.config.get_project_name()}({all_vars});\n'
                newline += top_level

                newline += '\n'

                for o in model_outputs:
                    if len(model.graphs[-1].outputs) == 1:
                        newline += indent + 'nnet::convert_data<{}, {}, {}>({}_ap, {});\n'.format(
                            datatype, dtype, o.size_cpp(), o.name, o.name
                        )
                    else:
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

            elif '// hls-fpga-machine-learning insert namespace' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += indent + f'using namespace {namespace};\n'

            elif '// hls-fpga-machine-learning insert tb_input_writer' in line:
                funcs = [
                    ('float', 'dump_tb_inputs_float'),
                    ('double', 'dump_tb_inputs_double'),
                ]
                newline = ''
                for dtype, funcname in funcs:
                    newline += f'void {funcname}(\n'
                    newline += '    const char* output_path'
                    for inp in model_inputs:
                        newline += f',\n    {dtype} {inp.name}[{inp.size_cpp()}]'
                    newline += '\n) {\n\n'

                    for inp in model_inputs:
                        decl = inp.definition_cpp(name_suffix='_ap').strip()
                        ap = inp.name + '_ap'
                        if decl.startswith('hls::stream'):
                            newline += f'    {decl};\n'
                        else:
                            newline += f'    {inp.type.name} {ap}[{inp.size_cpp()}];\n'
                        newline += f'    nnet::convert_data<{dtype}, {inp.type.name}, {inp.size_cpp()}>({inp.name}, {ap});\n'
                    newline += '\n'
                    newline += f'    std::ofstream fout(std::string(output_path) + "/{inp.name}_input_data.txt");\n'

                    for inp in model_inputs:
                        decl = inp.definition_cpp(name_suffix='_ap').strip()
                        shape = inp.shape

                        if decl.startswith('hls::stream'):
                            if len(shape) == 1:
                                N = shape[0]
                                newline += f'    for(int i = 0; i < {N}; i++) {{\n'
                                newline += f'        auto temp = {inp.name}_ap.read();\n'
                                newline += f'        ap_uint<{inp.type.name}::value_type::width> bits = temp[0].range();\n'
                                newline += f"        fout << bits.to_uint() << (i+1<{N} ? ' ' : '\\n');\n"
                                newline += '    }\n'
                            else:
                                inputs_list = model.nn_config['inputs']
                                fifo_depth = next((e['fifo_depth'] for e in inputs_list if e['name'] == inp.name), None)
                                batch_size = next((e['batch_size'] for e in inputs_list if e['name'] == inp.name), None)
                                newline += f'    for(int r = 0; r < {fifo_depth}; r++) {{\n'
                                newline += f'        auto temp = {inp.name}_ap.read();\n'
                                newline += f'        for(int c = 0; c < {batch_size}; c++) {{\n'
                                newline += (
                                    f'            ap_uint<{inp.type.name}::value_type::width> bits = temp[c].range();\n'
                                )
                                newline += f"            fout << bits.to_uint() << (c+1<{batch_size} ? ' ' : '\\n');\n"
                                newline += '        }\n'
                                newline += '    }\n'
                        else:
                            ap = inp.name + '_ap'
                            N = inp.size_cpp()
                            newline += f'    for(int i = 0; i < {N}; i++) {{\n'
                            newline += f'        ap_uint<{inp.type.name}::width> bits = {ap}[i].range();\n'
                            newline += f"        fout << bits.to_uint() << (i+1<{N} ? ' ' : '\\n');\n"
                            newline += '    }\n'
                    newline += '    fout.close();\n'
                    newline += '}\n'
            else:
                newline = line
            fout.write(newline)

        f.close()
        fout.close()

    def _write_build_prj_tcl(self, model, srcpath, dstpath):
        """Generate build_prj.tcl from a template with GEMM IP cflags/blackbox injection."""

        gemm_ip_pkg = model.config.get_writer_config().get('GemmIpPackage')
        if gemm_ip_pkg is None:
            gemm_ip_pkg = model.config.get_config_value('GemmIpPackage')
        if gemm_ip_pkg is None:
            try:
                gemm_ip_pkg = model.config.config['HLSConfig']['Model'].get('GemmIpPackage')
            except Exception:
                pass

        uses_gemm_ip = self._uses_gemm_ip(model)

        # GEMM IP: whether the gemm-ip-gen package is used is decided AT BUILD TIME from the
        # package dir's existence, not baked at write time. This honors the header contract in
        # nnet_gemm_ip.h: with a package -> -DGEMM_IP_HEADER pulls in the packaged cores (synth
        # + cosim + csim); WITHOUT one, GEMM_IP_HEADER is left undefined so a package-free
        # csim compiles the behavioral model (the `#else` branch). Only synth/cosim genuinely
        # need the package, so only those error when it is absent.
        #
        # hls4ml stays target-blind: it just provides the include path + the source hook, and
        # gemm-ip-gen's per-target gemm_ip_sources.tcl decides HOW the IP enters the build
        # (RTL-blackbox `add_files -blackbox`, or a header-only no-op for a behavioral-HLS
        # target). $tcldir is the build script's dir (== project dir); normalize to absolute so
        # the include resolves from the nested csim/cosim solution dir the HLS tool builds from.
        if gemm_ip_pkg:
            pkg_dir_expr = '[file normalize {' + str(Path(gemm_ip_pkg).resolve()) + '}]'
        else:
            pkg_dir_expr = '[file normalize $tcldir/gemm_pkg]'

        preamble = (
            f'set _gemm_pkg_dir {pkg_dir_expr}\n'
            'if { [file isdirectory $_gemm_pkg_dir] } {\n'
            '    set _gemm_cflags " -DBLACKBOX_FLOW -DGEMM_IP_HEADER -I$_gemm_pkg_dir"\n'
            '    set _gemm_has_pkg 1\n'
            '} else {\n'
            '    set _gemm_cflags ""\n'
            '    set _gemm_has_pkg 0\n'
            '}\n'
        )
        blackbox_block = (
            'if { !$_gemm_has_pkg && ($opt(synth) || $opt(cosim)) } {\n'
            '    puts "ERROR: GEMM IP is used but $_gemm_pkg_dir was not found. '
            'Run gemm-ip-gen before synth/cosim."\n'
            '    exit 1\n'
            '}\n'
            'if { $_gemm_has_pkg } {\n'
            '    source $_gemm_pkg_dir/gemm_ip_sources.tcl\n'
            '    puts "\\[hls4ml\\] GEMM IP package resolved at $_gemm_pkg_dir"\n'
            '} else {\n'
            '    puts "\\[hls4ml\\] No GEMM IP package found; csim uses the behavioral model '
            '(synth/cosim require the package)."\n'
            '}\n'
        )

        with open(srcpath) as src, open(dstpath, 'w') as dst:
            for line in src:
                if 'add_files firmware/${project_name}.cpp -cflags "-std=c++0x' in line:
                    if uses_gemm_ip:
                        dst.write(preamble)
                        line = line.replace('-I firmware"', '-I firmware$_gemm_cflags"')
                    dst.write(line)
                elif '#hls-fpga-machine-learning insert blackboxes' in line:
                    dst.write(line)
                    if uses_gemm_ip:
                        dst.write(blackbox_block)
                else:
                    dst.write(line)

    def write_build_script(self, model):
        """Write the TCL/Shell build scripts (project.tcl, build_prj.tcl, vivado_synth.tcl, build_lib.sh)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        filedir = Path(__file__).parent

        # project.tcl
        prj_tcl_dst = Path(f'{model.config.get_output_dir()}/project.tcl')
        with open(prj_tcl_dst, 'w') as f:
            f.write('variable project_name\n')
            f.write(f'set project_name "{model.config.get_project_name()}"\n')
            f.write('variable backend\n')
            f.write('set backend "vivado"\n')
            f.write('variable part\n')
            f.write('set part "{}"\n'.format(model.config.get_config_value('Part')))
            f.write('variable clock_period\n')
            f.write('set clock_period {}\n'.format(model.config.get_config_value('ClockPeriod')))
            f.write('variable clock_uncertainty\n')
            f.write('set clock_uncertainty {}\n'.format(model.config.get_config_value('ClockUncertainty', '12.5%')))
            f.write('variable version\n')
            f.write('set version "{}"\n'.format(model.config.get_config_value('Version', '1.0.0')))
            f.write('variable maximum_size\n')
            f.write('set maximum_size {}\n'.format(model.config.get_config_value('MaximumSize', '4096')))

        # build_prj.tcl
        srcpath = (filedir / '../templates/vivado/build_prj.tcl').resolve()
        dstpath = f'{model.config.get_output_dir()}/build_prj.tcl'
        self._write_build_prj_tcl(model, srcpath, dstpath)

        # vivado_synth.tcl
        srcpath = (filedir / '../templates/vivado/vivado_synth.tcl').resolve()
        dstpath = f'{model.config.get_output_dir()}/vivado_synth.tcl'
        copyfile(srcpath, dstpath)

        # build_lib.sh
        build_lib_src = (filedir / '../templates/vivado/build_lib.sh').resolve()
        build_lib_dst = Path(f'{model.config.get_output_dir()}/build_lib.sh').resolve()
        with open(build_lib_src) as src, open(build_lib_dst, 'w') as dst:
            for line in src.readlines():
                line = line.replace('myproject', model.config.get_project_name())
                line = line.replace('mystamp', model.config.get_config_value('Stamp') or '')

                dst.write(line)
        build_lib_dst.chmod(build_lib_dst.stat().st_mode | stat.S_IEXEC)

    def write_build_script_multigraph(self, model):
        """Write the build script (build_lib.sh) for stitched multigraph project
        Args:
            model (MultiModelGraph): the hls4ml multigraph model.
        """
        filedir = Path(__file__).parent
        os.makedirs(model.config.get_output_dir(), exist_ok=True)
        build_lib_src = (filedir / '../templates/vivado/build_lib_multigraph.sh').resolve()
        build_lib_dst = Path(f'{model.config.get_output_dir()}/build_lib.sh').resolve()
        graph_project_names = ' '.join(f'"{g.config.get_output_dir().split("/")[-1]}"' for g in model.graphs)

        with open(build_lib_src) as src, open(build_lib_dst, 'w') as dst:
            for line in src.readlines():
                line = line.replace('myproject', model.config.config['OriginalProjectName'])
                line = line.replace('myproject_stitched', model.config.config['ProjectName'])
                line = line.replace('mystamp', model.config.config['Stamp'])
                line = line.replace('mygraph_name_list', graph_project_names)
                dst.write(line)
        os.chmod(build_lib_dst, os.stat(build_lib_dst).st_mode | stat.S_IEXEC)

    def write_nnet_utils(self, model):
        """Copy the nnet_utils, AP types headers and any custom source to the project output directory

        Args:
            model (ModelGraph): the hls4ml model.
        """

        # nnet_utils
        filedir = os.path.dirname(os.path.abspath(__file__))

        srcpath = os.path.join(filedir, '../templates/vivado/nnet_utils/')
        dstpath = f'{model.config.get_output_dir()}/firmware/nnet_utils/'

        if not os.path.exists(dstpath):
            os.mkdir(dstpath)

        headers = [os.path.basename(h) for h in glob.glob(srcpath + '*.h')]

        for h in headers:
            copyfile(srcpath + h, dstpath + h)

        # ap_types
        filedir = os.path.dirname(os.path.abspath(__file__))

        srcpath = os.path.join(filedir, '../templates/vivado/ap_types/')
        dstpath = f'{model.config.get_output_dir()}/firmware/ap_types/'

        if os.path.exists(dstpath):
            rmtree(dstpath)

        copytree(srcpath, dstpath)

        # custom source
        filedir = os.path.dirname(os.path.abspath(__file__))

        custom_source = model.config.backend.get_custom_source()
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
        namespace = model.config.get_writer_config().get('Namespace', None)

        for line in contents:
            if '// hls4ml insert code' in line:
                newline = line
                for layer in model.get_layers():
                    for generated_code in layer.code.values():
                        newline += str(generated_code)
            else:
                newline = line
            if namespace is not None:
                if 'namespace nnet' in newline:
                    newline = newline.replace('namespace nnet', f'namespace {namespace}')
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

        write_tar = model.config.get_writer_config().get('WriteTar', False)
        if write_tar:
            tar_path = model.config.get_output_dir() + '.tar.gz'
            if os.path.exists(tar_path):
                os.remove(tar_path)
            with tarfile.open(tar_path, mode='w:gz') as archive:
                archive.add(model.config.get_output_dir(), recursive=True, arcname='')

    def write_hls(self, model, is_multigraph=False):
        if not is_multigraph:
            self.write_project_dir(model)
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
            self.write_gemm_config(model)
            self.write_tar(model)
        else:
            self.write_project_dir(model)
            self.write_build_script_multigraph(model)
            self.write_bridge_multigraph(model)
            self.write_multigraph_weights(model)
