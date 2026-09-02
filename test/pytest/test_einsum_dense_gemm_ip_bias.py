"""Functional check of the EinsumDense GEMM-IP bias semantics (both backends).

The EinsumDense bias is per output element and may vary across the
n_free_data rows (Keras bias_axes can include the data free axis).  The GEMM
IP bias port applies one value per column to every row, so the template must
add the per-element bias outside the IP.  A row-0 broadcast (the historical
bug) makes every row reuse the first row's bias, which this test rejects:
the bias rows are spaced 0.5 apart while the comparison tolerance is 0.1.

Also serves as the build-verification of the EinsumDense GEMM-IP csim on
each backend (it caught the wrong-order weight-column packing and the
weights-header include resolution along the way).
"""

import json
import os
import subprocess
import sys

import pytest

MGC_HOME = '/home/tools/siemens/catapult/Mgc_home'
CATAPULT_GCC_LIB = f'{MGC_HOME}/pkgs/dcs_gcc/gcc-13.4.0/lib64'

BACKEND_PRECISION = {
    'Catapult': 'ac_fixed<16,6,true>',
    'Vivado': 'ap_fixed<16,6>',
}


@pytest.mark.parametrize('backend', ['Catapult', 'Vivado'])
def test_einsum_dense_gemm_ip_row_varying_bias(tmp_path, backend):
    output_dir = tmp_path / f'einsum_dense_gemm_ip_bias_{backend.lower()}_prj'
    env = os.environ.copy()
    if backend == 'Catapult':
        env.setdefault('MGC_HOME', MGC_HOME)  # build_lib.sh resolves headers via $MGC_HOME
        env['LD_LIBRARY_PATH'] = f'{CATAPULT_GCC_LIB}:{env.get("LD_LIBRARY_PATH", "")}'

    script = f'''
import numpy as np
import keras
from hls4ml.converters import convert_from_keras_model

inp = keras.layers.Input((4, 8))
layer = keras.layers.EinsumDense('abc,cd->abd', output_shape=(4, 6), bias_axes='bd')
out = layer(inp)
model = keras.Model(inp, out)

kernel = np.linspace(-0.4, 0.4, 8 * 6, dtype=np.float32).reshape(8, 6)
# Bias varies along the data free axis (rows 0.0, 0.5, 1.0, 1.5): a row-0
# broadcast leaves rows 1-3 off by up to 1.5, far beyond the 0.1 tolerance.
bias = np.tile(np.arange(4, dtype=np.float32).reshape(4, 1) * 0.5, (1, 6))
layer.set_weights([kernel, bias])

hls_model = convert_from_keras_model(
    model,
    backend={backend!r},
    output_dir={str(output_dir)!r},
    io_type='io_parallel',
    hls_config={{
        'Model': {{'Precision': {BACKEND_PRECISION[backend]!r}, 'ReuseFactor': 1, 'Strategy': 'Latency'}},
        'LayerType': {{'EinsumDense': {{'Strategy': 'GEMM'}}}},
    }},
)
hls_model.compile()

rng = np.random.default_rng(1357)
data = rng.normal(0, 0.25, size=(3, 4, 8)).astype(np.float32)
keras_prediction = model.predict(data, verbose=0)
hls_prediction = hls_model.predict(data).reshape(keras_prediction.shape)

print('max_abs', float(np.max(np.abs(hls_prediction - keras_prediction))))
np.testing.assert_allclose(hls_prediction, keras_prediction, atol=0.1, rtol=0.0)
'''

    result = subprocess.run([sys.executable, '-c', script], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr

    top = (output_dir / 'firmware/myproject.cpp').read_text()
    if backend == 'Catapult':
        # Catapult lowers EinsumDense to a weightless Gemm (LowerEinsumToGemm); the
        # row-varying bias is added per-element in the wrapper around gemm_array_weightless.
        assert 'nnet::gemm_array_weightless<' in top, 'expected the lowered GEMM-IP path'
        assert 'nnet::einsum_dense_gemm_ip<' not in top
        # bias_in_core contract (consumed by gemm-ip-gen): a row-varying bias is added
        # per-element in the wrapper, NOT on the IP's per-column bias port, so the IP
        # must be told to leave its bias adder off even though weights are in-core.
        with open(output_dir / 'gemm_config.json') as f:
            gemm_config = json.load(f)
        gemm_entries = [e for e in gemm_config.values() if e['type'] == 'Gemm']
        assert gemm_entries, 'expected a lowered Gemm node in gemm_config.json'
        assert all(e['weights_in_core'] for e in gemm_entries)
        assert all(e['bias_in_core'] is False for e in gemm_entries)
    else:
        # Vivado now unifies with Catapult: EinsumDense lowers to a weightless Gemm
        # (LowerEinsumToGemm) and is materialized by the shared Gemm codegen. The einsum
        # template no longer emits GEMM. After the ROM-accessor migration the Vivado
        # weightless array core is gemm_array_weightless (config-sourced weights, no weight
        # argument) — the same name/shape as Catapult.
        assert 'nnet::einsum_dense_gemm_ip<' not in top, 'einsum template must no longer emit GEMM'
        assert 'nnet::gemm_array_weightless<' in top, 'expected the lowered weightless GEMM-IP array path'
        # Row-varying bias: zero per-column bias into the core, full per-element bias
        # added in the unpack loop (see gemm_array_row_bias_function_template).
        assert '_zero_bias' in top, 'expected the row-varying per-element bias-add path'
        with open(output_dir / 'gemm_config.json') as f:
            gemm_config = json.load(f)
        gemm_entries = [e for e in gemm_config.values() if e['type'] == 'Gemm']
        assert gemm_entries, 'expected a lowered Gemm node in gemm_config.json'


def test_einsum_dense_gemm_ip_row_varying_bias_io_stream_raises_loudly(tmp_path):
    """io_stream row-varying EinsumDense bias must fail LOUDLY, not silently mis-lower.

    The per-element wrapper bias-add is only wired for io_parallel; the io_stream
    Gemm function template raises NotImplementedError rather than falling back to the
    per-column IP port (which would drop the per-row bias component — exactly the
    Phase-3 regression). Attention never hits this (projection bias is per-column),
    so the guard stays a guard; this locks it as a loud failure, not a wrong answer.
    No compile: the raise fires at codegen.
    """
    import keras
    import numpy as np
    import pytest as _pytest
    from hls4ml.converters import convert_from_keras_model

    inp = keras.layers.Input((4, 8))
    layer = keras.layers.EinsumDense('abc,cd->abd', output_shape=(4, 6), bias_axes='bd')
    out = layer(inp)
    model = keras.Model(inp, out)
    kernel = np.linspace(-0.4, 0.4, 8 * 6, dtype=np.float32).reshape(8, 6)
    bias = np.tile(np.arange(4, dtype=np.float32).reshape(4, 1) * 0.5, (1, 6))
    layer.set_weights([kernel, bias])

    with _pytest.raises(NotImplementedError, match='row-varying'):
        hls_model = convert_from_keras_model(
            model,
            backend='Catapult',
            io_type='io_stream',
            output_dir=str(tmp_path / 'row_varying_io_stream_prj'),
            hls_config={
                'Model': {'Precision': 'ac_fixed<16,6,true>', 'ReuseFactor': 1, 'Strategy': 'Latency'},
                'LayerType': {'EinsumDense': {'Strategy': 'GEMM'}},
            },
        )
        hls_model.write()
