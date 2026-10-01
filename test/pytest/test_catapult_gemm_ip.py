import json
from pathlib import Path

import pytest
import tensorflow as tf

import hls4ml

test_root_path = Path(__file__).parent


def _make_dense_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(3, input_shape=(4,), name='dense'))
    model.compile(optimizer='adam', loss='mse')
    return model


def _make_rank2_dense_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(3, input_shape=(5, 2), name='dense2d'))
    model.compile(optimizer='adam', loss='mse')
    return model


def _make_rank3_dense_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(5, input_shape=(2, 3, 4), name='dense3d'))
    model.compile(optimizer='adam', loss='mse')
    return model


def _make_pointwise_conv1d_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(3, 1, input_shape=(5, 2), name='pointwise'))
    model.compile(optimizer='adam', loss='mse')
    return model


def _make_pointwise_conv2d_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(4, (1, 1), input_shape=(3, 5, 2), name='pointwise2d'))
    model.compile(optimizer='adam', loss='mse')
    return model


def _make_general_conv1d_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(3, 2, input_shape=(5, 2), name='conv1d_general'))
    model.compile(optimizer='adam', loss='mse')
    return model


def _make_two_dense_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(16, input_shape=(16,), name='dense1'))
    model.add(tf.keras.layers.ReLU(name='dense1_relu'))
    model.add(tf.keras.layers.Dense(8, name='dense2'))
    model.add(tf.keras.layers.ReLU(name='dense2_relu'))
    model.compile(optimizer='adam', loss='mse')
    return model


@pytest.mark.parametrize(
    'io_type, gemm_ip, expected',
    [
        ('io_stream', False, 'false'),
        ('io_stream', True, 'true'),
        ('io_parallel', False, 'false'),
    ],
)
def test_catapult_dense_gemm_ip_config_codegen(test_case_id, io_type, gemm_ip, expected):
    model = _make_dense_model()
    config = hls4ml.utils.config_from_keras_model(model)
    config['Model']['Strategy'] = 'GEMM' if gemm_ip else 'Latency'

    output_dir = test_root_path / test_case_id
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(output_dir),
        io_type=io_type,
        backend='Catapult',
    )

    hls_model.write()

    parameters = output_dir / 'firmware' / 'parameters.h'
    parameters_text = parameters.read_text()

    if io_type == 'io_stream':
        # GemmIP Dense routes through the Gemm node (nnet::gemm_stream);
        # the native dense stream header carries no GEMM branch.
        dense_stream = output_dir / 'firmware' / 'nnet_utils' / 'nnet_dense_stream.h'
        dense_stream_text = dense_stream.read_text()
        assert 'use_gemm_ip' not in dense_stream_text

        gemm_ip_header = output_dir / 'firmware' / 'nnet_utils' / 'nnet_gemm_ip.h'
        assert gemm_ip_header.exists()

        myproject_text = (output_dir / 'firmware' / 'myproject.cpp').read_text()
        if gemm_ip:
            assert 'nnet::gemm_stream_const_weights<' in myproject_text
        else:
            assert 'nnet::gemm_stream_const_weights<' not in myproject_text


def test_catapult_dense_gemm_ip_io_parallel_routes_to_gemm_array(test_case_id):
    """io_parallel Dense + GemmIP uses the array-interface Gemm node
    (it used to raise; the io_parallel array path superseded that)."""
    from hls4ml.backends.fpga.passes.gemm_nodes import Gemm

    model = _make_dense_model()
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['LayerName']['dense']['Strategy'] = 'GEMM'

    output_dir = test_root_path / test_case_id
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(output_dir),
        io_type='io_parallel',
        backend='Catapult',
    )
    has_gemm_array = any(isinstance(n, Gemm) for n in hls_model.graph.values())
    assert has_gemm_array, 'Expected io_parallel Dense + GemmIP to produce a Gemm node'

    # A Dense kernel is constant regardless of IOType, so Gemm is weight-stationary
    # and must take the const_weights ARRAY entry. Compile it: gemm_array_const_weights is a
    # template, so without an instantiation it is never type-checked and can rot silently.
    hls_model.write()
    myproject_text = (output_dir / 'firmware' / 'myproject.cpp').read_text()
    assert 'nnet::gemm_array_const_weights<' in myproject_text
    assert 'gemm_weight_cols()' in (output_dir / 'firmware' / 'parameters.h').read_text()

    import numpy as np

    hls_model.compile()
    x = np.random.default_rng(0).random((1, 4)).astype('float32')
    y_keras = np.asarray(model.predict(x, verbose=0)).flatten()
    y_hls = np.asarray(hls_model.predict(x)).flatten()
    assert np.allclose(y_keras, y_hls, atol=5e-2), f'{y_keras} vs {y_hls}'


def test_catapult_dense_gemm_metadata_rank1(test_case_id):
    model = _make_dense_model()
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['LayerName']['dense']['Strategy'] = 'GEMM'

    output_dir = test_root_path / test_case_id
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(output_dir),
        io_type='io_stream',
        backend='Catapult',
    )

    hls_model.write()

    parameters_text = (output_dir / 'firmware' / 'parameters.h').read_text()
    gemm_ip_text = (output_dir / 'firmware' / 'nnet_utils' / 'nnet_gemm_ip.h').read_text()

    assert 'static const unsigned gemm_m = 1;' in parameters_text
    assert 'static const unsigned gemm_k = 4;' in parameters_text
    assert 'static const unsigned gemm_n = 3;' in parameters_text
    # Four-name GEMM layer: nnet_gemm_ip.h defines the two array entries; the old
    # wrapper/indirection names are gone.
    assert 'void gemm_array' in gemm_ip_text
    assert 'void gemm_array_const_weights' in gemm_ip_text
    assert 'stream_gemm_ip' not in gemm_ip_text


def test_catapult_rank2_dense_default_rewrites_to_pointwise_conv(test_case_id):
    model = _make_rank2_dense_model()
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')

    output_dir = test_root_path / test_case_id
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(output_dir),
        io_type='io_stream',
        backend='Catapult',
    )

    layers = {layer.name: layer for layer in hls_model.get_layers()}
    assert layers['dense2d'].class_name == 'PointwiseConv1D'


def test_catapult_rank2_dense_gemm_preserves_dense_and_emits_shape(test_case_id):
    model = _make_rank2_dense_model()
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['LayerName']['dense2d']['Strategy'] = 'GEMM'

    output_dir = test_root_path / test_case_id
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(output_dir),
        io_type='io_stream',
        backend='Catapult',
    )

    layers = {layer.name: layer for layer in hls_model.get_layers()}
    assert layers['gemm_dense2d'].class_name == 'Gemm'
    assert layers['gemm_dense2d'].get_output_variable().shape == [5, 3]

    hls_model.write()

    parameters_text = (output_dir / 'firmware' / 'parameters.h').read_text()
    defines_text = (output_dir / 'firmware' / 'defines.h').read_text()

    assert 'static const unsigned gemm_m = 5;' in parameters_text
    assert 'static const unsigned gemm_k = 2;' in parameters_text
    assert 'static const unsigned gemm_n = 3;' in parameters_text
    assert 'typedef nnet::array<' in defines_text
    assert ', 2*1> input_t;' in defines_text
    assert ', 3*1> result_t;' in defines_text


def test_catapult_dense_gemm_inherits_model_and_layer_precision(test_case_id):
    """Model-level weight/bias/accum reach the synthetic GEMM node; result comes from LayerName.

    `weight`, `bias` and `accum` are resolved from Model level by
    `_mirror_precision_to_gemm_node`, which skips the auto-generated per-layer 'auto'.
    `result` is different: `HLSConfig.get_precision` checks LayerName *before* Model, and
    `config_from_keras_model(granularity='name')` auto-generates
    `Precision: {'result': 'auto'}` — 'auto' is a value, not an absence, so it short-circuits
    the lookup and a Model-level `result` is never consulted. That shadowing is upstream hls4ml
    behaviour and applies identically to the non-GEMM path, so `result` is set where it is
    actually read.
    """
    model = _make_rank2_dense_model()
    config = hls4ml.utils.config_from_keras_model(model, backend='Catapult', granularity='name')
    config['Model']['Precision'] = {
        'default': 'ac_fixed<16,6,true>',
        'weight': 'ac_int<8,true>',
        'bias': 'ac_int<8,true>',
        'accum': 'ac_fixed<32,16,true>',
    }
    config['LayerName']['dense2d']['Strategy'] = 'GEMM'
    config['LayerName']['dense2d']['Precision']['result'] = 'ac_int<8,true>'

    output_dir = test_root_path / test_case_id
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(output_dir),
        io_type='io_stream',
        backend='Catapult',
    )

    hls_model.write()

    defines_text = (output_dir / 'firmware' / 'defines.h').read_text()
    with open(output_dir / 'gemm_config.json') as f:
        gemm_config = json.load(f)

    assert 'typedef ac_int<8, true> gemm_dense2d_weight_t;' in defines_text
    assert 'typedef ac_int<8, true> gemm_dense2d_bias_t;' in defines_text
    assert 'typedef ac_fixed<32,16,true> gemm_dense2d_accum_t;' in defines_text
    assert 'typedef nnet::array<ac_int<8, true>, 3*1> result_t;' in defines_text

    dense_entry = gemm_config['gemm_dense2d']
    assert dense_entry['weight_precision'] == 'int<8>'
    assert dense_entry['bias_precision'] == 'int<8>'
    assert dense_entry['output_precision'] == 'int<8>'
    assert dense_entry['accum_precision'] == 'fixed<32,16,TRN,WRAP,0>'


def test_catapult_rank3_dense_gemm_preserves_shape_and_metadata(test_case_id):
    model = _make_rank3_dense_model()
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['LayerName']['dense3d']['Strategy'] = 'GEMM'

    output_dir = test_root_path / test_case_id
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(output_dir),
        io_type='io_stream',
        backend='Catapult',
    )

    layers = {layer.name: layer for layer in hls_model.get_layers()}
    assert layers['gemm_dense3d'].class_name == 'Gemm'
    # Rank-3 output dims are preserved (not flattened to [n_patches, n_out])
    assert layers['gemm_dense3d'].get_output_variable().shape == [2, 3, 5]

    hls_model.write()

    parameters_text = (output_dir / 'firmware' / 'parameters.h').read_text()
    defines_text = (output_dir / 'firmware' / 'defines.h').read_text()

    assert 'static const unsigned gemm_m = 6;' in parameters_text
    assert 'static const unsigned gemm_k = 4;' in parameters_text
    assert 'static const unsigned gemm_n = 5;' in parameters_text
    assert ', 4*1> input_t;' in defines_text
    assert ', 5*1> result_t;' in defines_text


def test_catapult_dense_gemm_codegen_transposes_weights(test_case_id):
    model = _make_dense_model()
    dense = model.get_layer('dense')
    weight_values = tf.constant(
        [
            [0.0, 1.0, 2.0],
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0],
        ],
        dtype=tf.keras.backend.floatx(),
    )
    bias_values = tf.constant([0.5, 1.5, 2.5], dtype=tf.keras.backend.floatx())
    dense.set_weights([weight_values.numpy(), bias_values.numpy()])

    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['LayerName']['dense']['Strategy'] = 'GEMM'

    output_dir = test_root_path / test_case_id
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(output_dir),
        io_type='io_stream',
        backend='Catapult',
    )

    hls_model.write()

    weight_file = next((output_dir / 'firmware' / 'weights').glob('w*.txt'))
    serialized_weights = [float(value.strip()) for value in weight_file.read_text().split(',') if value.strip()]

    assert serialized_weights == [
        0.0,
        10.0,
        20.0,
        30.0,
        1.0,
        11.0,
        21.0,
        31.0,
        2.0,
        12.0,
        22.0,
        32.0,
    ]


def test_catapult_resource_dense_pins_packed_weight_rom_width(test_case_id):
    # Resource RF 2 stores the 4x3 kernel as 2 packed words of 6 weights; the build script must
    # pin that ROM's WORD_WIDTH to one word, or Catapult may split it and replicate the ROM.
    model = _make_dense_model()
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['Model']['Strategy'] = 'Resource'
    config['Model']['ReuseFactor'] = 2
    config['LayerName']['dense']['Strategy'] = 'Resource'
    config['LayerName']['dense']['ReuseFactor'] = 2
    config['LayerName']['dense']['Precision']['weight'] = 'fixed<8,1>'

    output_dir = test_root_path / test_case_id
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=str(output_dir), io_type='io_stream', backend='Catapult'
    )
    hls_model.write()

    weight_name = hls_model.graph['dense'].get_weights('weight').name
    header = (output_dir / 'firmware' / 'weights' / f'{weight_name}.h').read_text()
    assert 'nnet::array<' in header and f'{weight_name}[2]' in header
    tcl = (output_dir / 'build_prj.tcl').read_text()
    assert f'set packed_weight_roms {{{weight_name} 48}}' in tcl
    assert 'keep_packed_weight_roms_wide $design $packed_weight_roms' in tcl


def test_catapult_pointwise_conv1d_gemm_ip_codegen(test_case_id):
    model = _make_pointwise_conv1d_model()
    config = hls4ml.utils.config_from_keras_model(model)
    config['Model']['Strategy'] = 'GEMM'

    output_dir = test_root_path / test_case_id
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(output_dir),
        io_type='io_stream',
        backend='Catapult',
    )

    hls_model.write()

    parameters_text = (output_dir / 'firmware' / 'parameters.h').read_text()
    myproject_text = (output_dir / 'firmware' / 'myproject.cpp').read_text()

    assert 'static const unsigned gemm_m = 5;' in parameters_text
    assert 'static const unsigned gemm_k = 2;' in parameters_text
    assert 'static const unsigned gemm_n = 3;' in parameters_text
    assert 'gemm_pointwise_stage(' in myproject_text
    assert 'nnet::gemm_stream_const_weights<' in myproject_text


def test_catapult_pointwise_conv2d_gemm_ip_codegen(test_case_id):
    model = _make_pointwise_conv2d_model()
    config = hls4ml.utils.config_from_keras_model(model)
    config['Model']['Strategy'] = 'GEMM'

    output_dir = test_root_path / test_case_id
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(output_dir),
        io_type='io_stream',
        backend='Catapult',
    )

    hls_model.write()

    parameters_text = (output_dir / 'firmware' / 'parameters.h').read_text()
    myproject_text = (output_dir / 'firmware' / 'myproject.cpp').read_text()

    assert 'static const unsigned gemm_m = 15;' in parameters_text
    assert 'static const unsigned gemm_k = 2;' in parameters_text
    assert 'static const unsigned gemm_n = 4;' in parameters_text
    assert 'gemm_pointwise2d_stage(' in myproject_text
    assert 'nnet::gemm_stream_const_weights<' in myproject_text


def test_catapult_general_conv1d_gemm_ip_codegen(test_case_id):
    model = _make_general_conv1d_model()
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['LayerName']['conv1d_general']['Strategy'] = 'GEMM'

    output_dir = test_root_path / test_case_id
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(output_dir),
        io_type='io_stream',
        backend='Catapult',
    )

    hls_model.write()

    myproject_text = (output_dir / 'firmware' / 'myproject.cpp').read_text()
    gemm_ip_text = (output_dir / 'firmware' / 'nnet_utils' / 'nnet_gemm_ip.h').read_text()
    parameters_text = (output_dir / 'firmware' / 'parameters.h').read_text()

    # General conv routes through a standalone Im2Col node feeding a pure Gemm node
    # (SplitConvGemm splits, no fusion). Assert the node kinds, not their names.
    from hls4ml.backends.fpga.passes.gemm_nodes import Gemm
    from hls4ml.model.layers import Im2Col

    assert any(isinstance(layer, Im2Col) for layer in hls_model.get_layers()), \
        'General conv should produce a standalone Im2Col node'
    assert any(isinstance(layer, Gemm) for layer in hls_model.get_layers()), \
        'General conv should produce a Gemm node'
    assert 'nnet::im2col_1d_gemm_rows<' in myproject_text
    assert 'nnet::gemm_stream_const_weights<' in myproject_text
    # Four-name GEMM layer: the old weight-column feed / wrapper helpers are gone.
    assert 'void gemm_array' in gemm_ip_text
    assert 'stream_gemm_packed_weight_cols' not in gemm_ip_text
    assert 'stream_gemm_ip' not in gemm_ip_text
    assert 'static const unsigned gemm_k = 4;' in parameters_text
    assert 'static const unsigned gemm_n = 3;' in parameters_text
    # Conv kernels must be transposed to [F, W*C] before the packed columns are written.
    assert (output_dir / 'firmware' / 'weights').is_dir()
    cols = list((output_dir / 'firmware' / 'weights').glob('*_gemm_cols.h'))
    assert cols, 'expected packed GEMM weight columns for the fused conv stage'


def test_catapult_io_stream_stage_wrapper_codegen(test_case_id):
    model = _make_two_dense_model()
    config = hls4ml.utils.config_from_keras_model(model)

    output_dir = test_root_path / test_case_id
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(output_dir),
        io_type='io_stream',
        backend='Catapult',
    )

    hls_model.write()

    myproject_text = (output_dir / 'firmware' / 'myproject.cpp').read_text()

    assert '#pragma hls_design block\nvoid dense1_stage(' in myproject_text
    assert '#pragma hls_design block\nvoid dense1_relu_stage(' in myproject_text
    assert '#pragma hls_design block\nvoid dense2_stage(' in myproject_text
    assert '#pragma hls_design block\nvoid dense2_relu_stage(' in myproject_text
    assert 'void CCS_BLOCK(dense1_stage)' not in myproject_text
    assert '#pragma hls_design ccore' not in myproject_text

    assert 'static ac_channel<layer2_t> layer2_out' in myproject_text
    assert 'static ac_channel<layer3_t> layer3_out' in myproject_text
    assert 'static ac_channel<layer4_t> layer4_out' in myproject_text

    assert 'dense1_stage(' in myproject_text
    assert 'dense1_relu_stage(layer2_out, layer3_out); // dense1_relu' in myproject_text
    assert 'dense2_stage(layer3_out, layer4_out); // dense2' in myproject_text
    assert 'dense2_relu_stage(layer4_out, layer5_out); // dense2_relu' in myproject_text


def test_catapult_gemm_strategy_emits_contract_and_behavioral_ip(test_case_id):
    model = _make_two_dense_model()
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['Model']['Strategy'] = 'GEMM'

    for layer_name in ('dense1', 'dense2'):
        layer_cfg = config['LayerName'][layer_name]
        layer_cfg['Strategy'] = 'GEMM'
        layer_cfg['ReuseFactor'] = 1
        layer_cfg['TransposeWeights'] = True

    output_dir = test_root_path / test_case_id
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(output_dir),
        io_type='io_stream',
        backend='Catapult',
    )

    hls_model.write()

    gemm_config = output_dir / 'gemm_config.json'
    assert gemm_config.exists()

    gemm_config_data = json.loads(gemm_config.read_text())
    assert set(gemm_config_data) == {'gemm_dense1', 'gemm_dense2'}
    assert gemm_config_data['gemm_dense1']['transpose_weights'] is True
    assert gemm_config_data['gemm_dense2']['transpose_weights'] is True
    for layer_name in ('gemm_dense1', 'gemm_dense2'):
        entry = gemm_config_data[layer_name]
        assert entry['layer_name'] == layer_name
        assert entry['gemm_ip_id'] == layer_name
        assert isinstance(entry['gemm_ip_index'], int)
        assert entry['interface'] == 'stream'
        assert entry['protocol']['kind'] == 'catapult_ac_channel_stream'
        assert entry['protocol']['input_beat_order'] == 'row_major'
        assert entry['protocol']['weight_layout'] == 'column_major'
        assert entry['blackbox']['entity'] == f'{layer_name}_core'
        assert entry['blackbox']['rtl'] == f'{layer_name}/{layer_name}_core.v'
    gemm_ip_header = output_dir / 'firmware' / 'nnet_utils' / 'nnet_gemm_ip.h'
    assert gemm_ip_header.exists()
    gemm_ip_text = gemm_ip_header.read_text()
    # Four-name GEMM layer: entry point == synth core == csim behavioral share one
    # name each; the behavioral now lives inline in the four entries.
    assert 'void gemm_array' in gemm_ip_text
    assert 'void gemm_array_const_weights' in gemm_ip_text
    assert 'stream_gemm_ip' not in gemm_ip_text
    assert 'stream_gemm_ip_const_weights' not in gemm_ip_text
