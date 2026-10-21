"""FabricMultipliers: Model-level knob that builds every HLS-inferred multiplier in LUTs.

It is a project-wide `config_op mul -impl fabric` in build_prj.tcl, switched by a
project.tcl variable, so it must default off and read a string 'False' as off.
"""

import pytest

import hls4ml


def _write(tmp_path, backend, value):
    import keras

    inp = keras.layers.Input((4,))
    out = keras.layers.Dense(3, name='dense')(inp)
    model = keras.Model(inp, out)
    cfg = {'Precision': 'ap_fixed<16,6>', 'ReuseFactor': 1}
    if value is not None:
        cfg['FabricMultipliers'] = value
    prj = tmp_path / f'fm_{backend}_{value}'
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, backend=backend, io_type='io_parallel', output_dir=str(prj), hls_config={'Model': cfg}
    )
    hls_model.write()
    return prj


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
@pytest.mark.parametrize('value, expected', [(None, 0), (False, 0), ('False', 0), (True, 1), ('true', 1)])
def test_fabric_multipliers_knob(backend, value, expected, tmp_path):
    prj = _write(tmp_path, backend, value)
    assert f'set fabric_multipliers {expected}\n' in (prj / 'project.tcl').read_text()
    build = (prj / 'build_prj.tcl').read_text()
    assert 'config_op mul -impl fabric' in build
    assert '$fabric_multipliers' in build
