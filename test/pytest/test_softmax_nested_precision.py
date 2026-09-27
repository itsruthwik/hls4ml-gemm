import pytest

import hls4ml


keras = pytest.importorskip('keras')


def _softmax_model():
    inputs = keras.layers.Input((5,), name='input_features')
    outputs = keras.layers.Softmax(name='softmax')(inputs)
    return keras.Model(inputs, outputs)


@pytest.mark.parametrize('backend', ['Vitis', 'Catapult'])
def test_nested_softmax_precisions_reach_graph_and_typedefs(backend, tmp_path):
    model = _softmax_model()
    config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend=backend)
    precision = config['LayerName']['softmax']['Precision']
    precision['exp_table'] = 'fixed<13,4,RND,SAT>'
    precision['inv_table'] = 'ufixed<14,2,RND,SAT>'
    precision['inv_inp'] = 'ufixed<18,3,RND,SAT>'
    output_dir = tmp_path / backend.lower()

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        backend=backend,
        io_type='io_stream',
        output_dir=str(output_dir),
    )
    softmax = hls_model.graph['softmax']

    assert str(softmax.get_attr('exp_table_t').precision) == 'fixed<13,4,RND,SAT,0>'
    assert str(softmax.get_attr('inv_table_t').precision) == 'ufixed<14,2,RND,SAT,0>'
    assert str(softmax.get_attr('inv_inp_t').precision) == 'ufixed<18,3,RND,SAT,0>'

    hls_model.write()
    defines = (output_dir / 'firmware' / 'defines.h').read_text()
    if backend == 'Vitis':
        assert 'typedef ap_fixed<13,4,AP_RND,AP_SAT,0> softmax_exp_table_t;' in defines
        assert 'typedef ap_ufixed<14,2,AP_RND,AP_SAT,0> softmax_inv_table_t;' in defines
        assert 'typedef ap_ufixed<18,3,AP_RND,AP_SAT,0> softmax_inv_inp_t;' in defines
    else:
        assert 'typedef ac_fixed<13,4,true,AC_RND,AC_SAT> softmax_exp_table_t;' in defines
        assert 'typedef ac_fixed<14,2,false,AC_RND,AC_SAT> softmax_inv_table_t;' in defines
        assert 'typedef ac_fixed<18,3,false,AC_RND,AC_SAT> softmax_inv_inp_t;' in defines


@pytest.mark.parametrize('backend', ['Vitis', 'Catapult'])
def test_softmax_type_defaults_do_not_fall_back_to_model_precision(backend, tmp_path):
    hls_model = hls4ml.converters.convert_from_keras_model(
        _softmax_model(),
        hls_config={'Model': {'Precision': 'fixed<10,4>', 'ReuseFactor': 1, 'Strategy': 'Latency'}},
        backend=backend,
        io_type='io_stream',
        output_dir=str(tmp_path / backend.lower()),
    )
    softmax = hls_model.graph['softmax']

    assert str(softmax.get_attr('exp_table_t').precision) == 'fixed<18,8,RND,SAT,0>'
    assert str(softmax.get_attr('inv_table_t').precision) == 'fixed<18,8,RND,SAT,0>'
    assert str(softmax.get_attr('inv_inp_t').precision) == 'fixed<18,8,RND,SAT,0>'


def test_converter_supplied_softmax_type_wins_over_nested_precision(tmp_path):
    model = _softmax_model()
    config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend='Vitis')
    config['LayerName']['softmax']['Precision']['inv_inp'] = 'ufixed<18,3,RND,SAT>'
    config['LayerName']['softmax']['inv_inp_t'] = 'fixed<11,5,RND,WRAP>'

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        backend='Vitis',
        output_dir=str(tmp_path / 'preserve'),
    )

    assert str(hls_model.graph['softmax'].get_attr('inv_inp_t').precision) == 'fixed<11,5,RND,WRAP,0>'
