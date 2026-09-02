"""
test_int8_gemm_ip_precision.py
==============================
Regression tests for INT8 precision override support in the Catapult GEMM IP
backend (Strategy: GEMM).

Root cause (see local-docs/HGQ2_Mixed_Precision_Report.md):
  When a Dense or Conv layer is transformed into a Gemm / Im2ColGemm
  node, the new synthetic node previously could not find its LayerName precision
  entries in HLSConfig because its node name (e.g. 'gemm_dense_target') differed
  from the original source name ('dense_target').

  The fix adds _mirror_precision_to_gemm_node() which is called *before* make_node()
  in each optimizer pass so that initialize() -> add_weights_variable() -> get_precision()
  finds the correct per-layer entries.

Configuration API notes
-----------------------
- Use hls4ml.utils.config_from_keras_model() to build the flat config dict.
- The flat dict has top-level keys 'Model' and 'LayerName' (no 'HLSConfig' wrapper
  in the user-facing dict).
- The GEMM IP path is opted into with Strategy: 'GEMM' (Model/LayerType/LayerName),
  a mutually-exclusive strategy value resolved by the stock get_strategy chain.
- Pass io_type='io_stream' to convert_from_keras_model() for GEMM layers.

Tests
-----
1. Dense INT8 GemmIP  — weight_t / bias_t must be ac_int<8, true>
2. Dense GemmIP no override — falls back gracefully to model default
3. Dense GemmIP=False (baseline) — Dense node not replaced, weight_t is INT8 on
   the original Dense node (regression guard that baseline path is unaffected)
4. Precision isolation — INT8 override on one Dense does NOT leak to sibling
5. Conv1D INT8 GemmIP — fused Im2ColGemm has weight_t = ac_int<8, true>
"""

import os
import shutil

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dense_model(name='dense_target'):
    """Return a tiny Keras Dense model."""
    import os; os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    import tensorflow as tf; tf.get_logger().setLevel('ERROR')
    from tensorflow import keras
    inp = keras.Input(shape=(8,), name='input_1')
    out = keras.layers.Dense(4, name=name, use_bias=True)(inp)
    return keras.Model(inputs=inp, outputs=out)


def _make_multi_dense_model():
    """Two Dense layers on the same input (for isolation test)."""
    import os; os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    import tensorflow as tf; tf.get_logger().setLevel('ERROR')
    from tensorflow import keras
    inp = keras.Input(shape=(8,), name='input_1')
    a = keras.layers.Dense(4, name='dense_int8')(inp)
    b = keras.layers.Dense(4, name='dense_default')(inp)
    out = keras.layers.Add(name='add')([a, b])
    return keras.Model(inputs=inp, outputs=out)


def _make_conv1d_model(name='conv_target'):
    """Tiny Keras Conv1D model.

    Uses valid padding: the row/column GEMM IP supports valid padding only, and these
    tests are about weight/accum precision, so padding is incidental to their intent.
    """
    import os; os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    import tensorflow as tf; tf.get_logger().setLevel('ERROR')
    from tensorflow import keras
    inp = keras.Input(shape=(16, 4), name='input_1')
    out = keras.layers.Conv1D(8, 3, padding='valid', name=name, use_bias=True)(inp)
    return keras.Model(inputs=inp, outputs=out)


def _precision_cpp(node, weight_name='weight'):
    """Return the C++ definition string for a named weight's precision."""
    w = node.weights[weight_name]
    if hasattr(w.type.precision, 'definition_cpp'):
        return w.type.precision.definition_cpp()
    return str(w.type.precision)


def _weight_type_name(node):
    return node.weights['weight'].type.name


def _bias_type_name(node):
    return node.weights['bias'].type.name


# ---------------------------------------------------------------------------
# Test 1: Dense INT8 GemmIP=True — weight_t and bias_t must be ac_int<8, true>
# ---------------------------------------------------------------------------

def test_dense_int8_gemm_ip_weight_type(tmp_path):
    """
    After conversion with GemmIP=True and an INT8 LayerName override for
    'dense_target', the synthetic 'gemm_dense_target' Gemm node must
    have weight_t = bias_t = ac_int<8, true>.
    """
    import hls4ml

    model = _make_dense_model()
    config = hls4ml.utils.config_from_keras_model(model, backend='Catapult', granularity='name')
    config['LayerName']['dense_target']['Strategy'] = 'GEMM'
    config['LayerName']['dense_target']['Precision'] = {
        'weight': 'ac_int<8>',
        'bias':   'ac_int<8>',
    }

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(tmp_path / 'dense_int8_gemm'),
        backend='Catapult',
        io_type='io_stream',
    )

    gemm_node = next(
        (n for n in hls_model.get_layers() if 'gemm_dense_target' in n.name),
        None,
    )
    assert gemm_node is not None, \
        "Expected a Gemm node named 'gemm_dense_target' after conversion."

    w_cpp = _precision_cpp(gemm_node, 'weight')
    b_cpp = _precision_cpp(gemm_node, 'bias')

    assert 'ac_int<8' in w_cpp, \
        f"weight_t should be ac_int<8,true> but got: {w_cpp!r}"
    assert 'ac_int<8' in b_cpp, \
        f"bias_t should be ac_int<8,true> but got: {b_cpp!r}"

    # Must NOT still be the global model default
    assert 'ac_fixed<16' not in w_cpp, \
        f"weight_t must not be model_default_t but got: {w_cpp!r}"
    assert 'ac_fixed<16' not in b_cpp, \
        f"bias_t must not be model_default_t but got: {b_cpp!r}"

    # Type name must be layer-specific, not the generic model_default_t
    assert _weight_type_name(gemm_node) != 'model_default_t', \
        "weight type name must not be 'model_default_t'"
    assert _bias_type_name(gemm_node) != 'model_default_t', \
        "bias type name must not be 'model_default_t'"


# ---------------------------------------------------------------------------
# Test 2: Dense GemmIP=True without precision override — falls back to default
# ---------------------------------------------------------------------------

def test_dense_gemm_ip_no_override_uses_default(tmp_path):
    """
    When GemmIP=True but no LayerName precision override is given, weight_t /
    bias_t should fall back to the model-global default.  The fix must be a
    no-op when there is nothing to mirror.
    """
    import hls4ml

    model = _make_dense_model()
    config = hls4ml.utils.config_from_keras_model(model, backend='Catapult', granularity='name')
    config['LayerName']['dense_target']['Strategy'] = 'GEMM'
    # No Precision override — leave at 'auto'

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(tmp_path / 'dense_default_gemm'),
        backend='Catapult',
        io_type='io_stream',
    )

    gemm_node = next(
        (n for n in hls_model.get_layers() if 'gemm_dense_target' in n.name),
        None,
    )
    assert gemm_node is not None, "Expected a Gemm node for 'dense_target'."

    # Should resolve to something concrete (not crash).
    w_cpp = _precision_cpp(gemm_node, 'weight')
    assert w_cpp is not None and len(w_cpp) > 0, \
        "weight_t precision string must be non-empty."

    # Should NOT be INT8 (we gave no override)
    assert 'ac_int<8' not in w_cpp, \
        f"Without an override, weight_t must not be ac_int<8>, but got: {w_cpp!r}"


# ---------------------------------------------------------------------------
# Test 3: GemmIP=False baseline — Dense not replaced, precision works on orig node
# ---------------------------------------------------------------------------

def test_dense_non_gemm_ip_path_unaffected(tmp_path):
    """
    With GemmIP=False (default) the Dense layer is NOT transformed into a
    Gemm node.  The INT8 override should still be visible on the original
    Dense node, and no Gemm node should appear.  This is a regression
    guard that the baseline non-GEMM-IP path is completely unaffected.
    """
    import hls4ml

    model = _make_dense_model()
    config = hls4ml.utils.config_from_keras_model(model, backend='Catapult', granularity='name')
    config['LayerName']['dense_target']['Precision'] = {
        'weight': 'ac_int<8>',
        'bias':   'ac_int<8>',
    }
    # No GEMM strategy — the default (Latency) leaves the Dense on the stock path

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(tmp_path / 'dense_int8_no_gemm'),
        backend='Catapult',
        io_type='io_parallel',
    )

    # No Gemm node should exist
    from hls4ml.backends.fpga.passes.gemm_nodes import Gemm
    for node in hls_model.get_layers():
        assert not isinstance(node, Gemm), \
            f"Unexpected Gemm node '{node.name}' when GemmIP=False."

    # Original Dense node should have INT8 weight type
    dense_node = next(
        (n for n in hls_model.get_layers() if n.name == 'dense_target'),
        None,
    )
    assert dense_node is not None, "Original 'dense_target' node not found."

    w_cpp = _precision_cpp(dense_node, 'weight')
    assert 'ac_int<8' in w_cpp, \
        f"Dense (GemmIP=False) weight_t should be ac_int<8,true> but got: {w_cpp!r}"


# ---------------------------------------------------------------------------
# Test 4: Precision isolation — INT8 on one Dense must not leak to sibling
# ---------------------------------------------------------------------------

def test_precision_isolation_between_dense_layers(tmp_path):
    """
    INT8 override on 'dense_int8' must not contaminate the sibling 'dense_default'
    layer (which should keep the global model default precision).
    """
    import hls4ml

    model = _make_multi_dense_model()
    config = hls4ml.utils.config_from_keras_model(model, backend='Catapult', granularity='name')
    config['LayerName']['dense_int8']['Strategy'] = 'GEMM'
    config['LayerName']['dense_int8']['Precision'] = {
        'weight': 'ac_int<8>',
        'bias':   'ac_int<8>',
    }
    config['LayerName']['dense_default']['Strategy'] = 'GEMM'
    # No precision override for dense_default

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(tmp_path / 'isolation_test'),
        backend='Catapult',
        io_type='io_stream',
    )

    int8_node = next(
        (n for n in hls_model.get_layers() if 'gemm_dense_int8' in n.name),
        None,
    )
    default_node = next(
        (n for n in hls_model.get_layers() if 'gemm_dense_default' in n.name),
        None,
    )

    assert int8_node is not None,    "Gemm for 'dense_int8' not found."
    assert default_node is not None, "Gemm for 'dense_default' not found."

    int8_w_cpp    = _precision_cpp(int8_node,    'weight')
    default_w_cpp = _precision_cpp(default_node, 'weight')

    assert 'ac_int<8' in int8_w_cpp, \
        f"dense_int8 weight_t should be ac_int<8> but got: {int8_w_cpp!r}"

    assert 'ac_int<8' not in default_w_cpp, \
        f"dense_default weight_t must NOT be ac_int<8> (precision leaked!) but got: {default_w_cpp!r}"


# ---------------------------------------------------------------------------
# Test 5: Conv1D INT8 GemmIP=True — fused node weight_t = ac_int<8, true>
# ---------------------------------------------------------------------------

def test_conv1d_int8_gemm_ip_weight_type(tmp_path):
    """
    For Conv1D with GemmIP=True and an INT8 LayerName override, the fused
    Im2ColGemm node must have weight_t = ac_int<8, true>.
    """
    import hls4ml
    from hls4ml.backends.fpga.passes.gemm_nodes import Im2ColGemm

    model = _make_conv1d_model()
    config = hls4ml.utils.config_from_keras_model(model, backend='Catapult', granularity='name')
    config['LayerName']['conv_target']['Strategy'] = 'GEMM'
    config['LayerName']['conv_target']['Precision'] = {
        'weight': 'ac_int<8>',
        'bias':   'ac_int<8>',
    }

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(tmp_path / 'conv1d_int8_gemm'),
        backend='Catapult',
        io_type='io_stream',
    )

    # The SplitConvGemm + FuseIm2ColGemm passes produce an Im2ColGemm node.
    fused_node = next(
        (n for n in hls_model.get_layers()
         if isinstance(n, Im2ColGemm) and 'conv_target' in n.name),
        None,
    )
    if fused_node is None:
        # Fallback: any non-original node that mentions conv_target
        fused_node = next(
            (n for n in hls_model.get_layers()
             if 'conv_target' in n.name and n.name != 'conv_target'
             and hasattr(n, 'weights') and 'weight' in n.weights),
            None,
        )

    assert fused_node is not None, \
        "Expected a fused Im2ColGemm node for 'conv_target' but none found."

    w_cpp = _precision_cpp(fused_node, 'weight')
    b_cpp = _precision_cpp(fused_node, 'bias')

    assert 'ac_int<8' in w_cpp, \
        f"Conv1D (GemmIP) weight_t should be ac_int<8,true> but got: {w_cpp!r}"
    assert 'ac_int<8' in b_cpp, \
        f"Conv1D (GemmIP) bias_t should be ac_int<8,true> but got: {b_cpp!r}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v', '--tb=short']))

# ---------------------------------------------------------------------------
# Test 6: Dense GEMM-IP with result=ac_int<8> and accum=ac_fixed<18,8>
# ---------------------------------------------------------------------------

def test_dense_gemm_ip_accum_override(tmp_path):
    """
    Dense GEMM-IP with result=ac_int<8> and accum=ac_fixed<18,8> emits
    distinct result and accumulator precision.
    """
    import hls4ml

    model = _make_dense_model()
    config = hls4ml.utils.config_from_keras_model(model, backend='Catapult', granularity='name')
    config['LayerName']['dense_target']['Strategy'] = 'GEMM'
    config['LayerName']['dense_target']['Precision'] = {
        'weight': 'ac_int<8>',
        'bias':   'ac_int<8>',
        'result': 'ac_int<8>',
        'accum':  'ac_fixed<18,8>',
    }

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(tmp_path / 'dense_accum_gemm'),
        backend='Catapult',
        io_type='io_stream',
    )

    gemm_node = next(
        (n for n in hls_model.get_layers() if 'gemm_dense_target' in n.name),
        None,
    )
    assert gemm_node is not None, "Expected a Gemm node."

    accum_t = gemm_node.types.get('accum_t')
    assert accum_t is not None, "accum_t not found in node.types"
    accum_cpp = accum_t.precision.definition_cpp() if hasattr(accum_t.precision, 'definition_cpp') else str(accum_t.precision)
    assert 'ac_fixed<18,8' in accum_cpp.replace(' ', ''), f"accum_t should be ac_fixed<18,8> but got: {accum_cpp!r}"
    assert 'ac_int<8' not in accum_cpp, "accum_t should not be the same as result (ac_int<8>)"
    
    res_t = gemm_node.types.get('result_t')
    res_cpp = res_t.precision.definition_cpp() if hasattr(res_t.precision, 'definition_cpp') else str(res_t.precision)
    assert 'ac_int<8' in res_cpp, f"result_t should be ac_int<8> but got: {res_cpp!r}"


# ---------------------------------------------------------------------------
# Test 7: Conv1D GEMM-IP fused path emits distinct accumulator precision
# ---------------------------------------------------------------------------

def test_conv1d_gemm_ip_accum_override(tmp_path):
    """
    Conv1D GEMM-IP fused path emits distinct accumulator precision.
    """
    import hls4ml
    from hls4ml.backends.fpga.passes.gemm_nodes import Im2ColGemm

    model = _make_conv1d_model()
    config = hls4ml.utils.config_from_keras_model(model, backend='Catapult', granularity='name')
    config['LayerName']['conv_target']['Strategy'] = 'GEMM'
    config['LayerName']['conv_target']['Precision'] = {
        'weight': 'ac_int<8>',
        'bias':   'ac_int<8>',
        'result': 'ac_int<8>',
        'accum':  'ac_fixed<18,8>',
    }

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(tmp_path / 'conv1d_accum_gemm'),
        backend='Catapult',
        io_type='io_stream',
    )

    fused_node = next(
        (n for n in hls_model.get_layers()
         if isinstance(n, Im2ColGemm) and 'conv_target' in n.name),
        None,
    )
    if fused_node is None:
        fused_node = next(
            (n for n in hls_model.get_layers()
             if 'conv_target' in n.name and n.name != 'conv_target'
             and hasattr(n, 'weights') and 'weight' in n.weights),
            None,
        )

    assert fused_node is not None, "Expected a fused Im2ColGemm node."

    accum_t = fused_node.types.get('accum_t')
    assert accum_t is not None, "accum_t not found in node.types"
    accum_cpp = accum_t.precision.definition_cpp() if hasattr(accum_t.precision, 'definition_cpp') else str(accum_t.precision)
    assert 'ac_fixed<18,8' in accum_cpp.replace(' ', ''), f"accum_t should be ac_fixed<18,8> but got: {accum_cpp!r}"
    assert 'ac_int<8' not in accum_cpp, "accum_t should not be the same as result (ac_int<8>)"

