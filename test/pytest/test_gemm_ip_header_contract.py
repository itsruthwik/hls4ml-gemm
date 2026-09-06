import hls4ml
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import os
import pytest

def test_gemm_ip_header_contract_emits_flags(tmp_path):
    """
    Verifies that a GEMM-IP enabled model generates a build_prj.tcl with
    the correct -DGEMM_IP_HEADER and package include path, rather than 
    a value-based absolute path macro.
    """
    inp = keras.Input(shape=(8,), name='input_1')
    out = keras.layers.Dense(4, name='dense_target', use_bias=True)(inp)
    model = keras.Model(inputs=inp, outputs=out)

    config = hls4ml.utils.config_from_keras_model(model, backend='Catapult', granularity='name')
    config['LayerName']['dense_target']['Strategy'] = 'GEMM'
    config['LayerName']['dense_target']['Strategy'] = 'GEMM'
    config['Model']['Strategy'] = 'GEMM'
    
    # Use a dummy path
    dummy_pkg = tmp_path / 'dummy_gemm_ip_pkg'
    config['Model']['GemmIpPackage'] = str(dummy_pkg)
    config['GemmIpPackage'] = str(dummy_pkg)

    prj_dir = tmp_path / 'gemm_ip_header_test_prj'
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(prj_dir),
        backend='Catapult',
        io_type='io_stream',
    )
    hls_model.write()

    build_tcl_path = prj_dir / 'build_prj.tcl'
    assert build_tcl_path.exists(), "build_prj.tcl should exist"
    
    tcl_content = build_tcl_path.read_text()
    
    # We should have the marker macro (emitted inside the build-time package-presence guard
    # so a package-free csim omits it and falls back to the behavioral model).
    assert '-DGEMM_IP_HEADER' in tcl_content
    # We should NOT have a value assignment like -DGEMM_IP_HEADER="..."
    assert '-DGEMM_IP_HEADER=' not in tcl_content
    assert '-DGEMM_IP_HEADER\\=' not in tcl_content

    # The package dir is resolved into a tcl var (via [file normalize {<path>}]) and the
    # include is -I$_gemm_pkg_dir; assert the path is present and the var-based include is used.
    assert str(dummy_pkg.resolve()) in tcl_content
    assert '-I$_gemm_pkg_dir' in tcl_content

def test_gemm_ip_header_not_emitted_for_native(tmp_path):
    """
    Verifies that a native non-GEMM model does not generate -DGEMM_IP_HEADER.
    """
    inp = keras.Input(shape=(8,), name='input_1')
    out = keras.layers.Dense(4, name='dense_target', use_bias=True)(inp)
    model = keras.Model(inputs=inp, outputs=out)

    config = hls4ml.utils.config_from_keras_model(model, backend='Catapult', granularity='name')
    
    prj_dir = tmp_path / 'native_test_prj'
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(prj_dir),
        backend='Catapult',
        io_type='io_stream',
    )
    hls_model.write()

    build_tcl_path = prj_dir / 'build_prj.tcl'
    assert build_tcl_path.exists(), "build_prj.tcl should exist"
    
    tcl_content = build_tcl_path.read_text()
    
    assert '-DGEMM_IP_HEADER' not in tcl_content
