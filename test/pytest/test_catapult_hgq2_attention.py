import glob
import json
import os
import subprocess
import sys

import keras
import pytest

from hls4ml.converters import convert_from_keras_model

if keras.__version__ < '3.0.0':
    pytest.skip('This test requires Keras 3.0.0 or higher', allow_module_level=True)

hgq = pytest.importorskip('hgq')
pytest.importorskip('hgq.layers')

from hgq.config import QuantizerConfigScope
from hgq.layers import QMultiHeadAttention


def _catapult_csim_env():
    """Environment for loading the hls4ml csim library: Catapult's bundled gcc links it
    against a newer libstdc++ than the system one, so its runtime must lead
    LD_LIBRARY_PATH. Derived from MGC_HOME rather than a machine-specific path; the
    test skips when the toolchain is not configured."""
    mgc_home = os.environ.get('MGC_HOME')
    libs = sorted(glob.glob(os.path.join(mgc_home, 'pkgs', 'dcs_gcc', 'gcc-*', 'lib64'))) if mgc_home else []
    if not libs:
        pytest.skip('Catapult toolchain (MGC_HOME with its bundled gcc) not configured')
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = f'{libs[-1]}:{env.get("LD_LIBRARY_PATH", "")}'
    return env


def _make_hgq_mha_model():
    with QuantizerConfigScope(f0=3, i0=2):
        q = keras.layers.Input((4, 8), name='q')
        v = keras.layers.Input((4, 8), name='v')
        k = keras.layers.Input((4, 8), name='k')
        out = QMultiHeadAttention(1, 4, name='hgq_mha', fuse='none')(q, v, k)
        return keras.Model([q, v, k], out)


def test_catapult_hgq2_attention_codegen_and_compile(tmp_path):
    model = _make_hgq_mha_model()
    output_dir = tmp_path / 'catapult_hgq2_mha_prj'

    hls_model = convert_from_keras_model(
        model,
        backend='Catapult',
        output_dir=str(output_dir),
        io_type='io_parallel',
        hls_config={'Model': {'Precision': 'ac_fixed<16,6,true>', 'ReuseFactor': 1, 'Strategy': 'Latency'}},
    )

    layer_classes = [layer.class_name for layer in hls_model.get_layers()]
    assert 'EinsumDense' in layer_classes
    assert 'Einsum' in layer_classes
    assert 'Softmax' in layer_classes

    hls_model.write()

    parameters = (output_dir / 'firmware/parameters.h').read_text()
    top = (output_dir / 'firmware/myproject.cpp').read_text()
    assert '#include "nnet_utils/nnet_einsum.h"' in parameters
    assert '#include "nnet_utils/nnet_einsum_dense.h"' in parameters
    assert 'nnet::einsum<' in top
    assert 'nnet::einsum_dense<' in top
    assert 'nnet::softmax_multidim<' in top

    env = os.environ.copy()
    env.setdefault('MGC_HOME', '/nonexistent')
    result = subprocess.run(['bash', 'build_lib.sh'], cwd=output_dir, env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_catapult_hgq2_attention_numeric_csim(tmp_path):
    output_dir = tmp_path / 'catapult_hgq2_mha_numeric_prj'
    env = _catapult_csim_env()

    script = f'''
import numpy as np
import keras
from hgq.config import QuantizerConfigScope
from hgq.layers import QMultiHeadAttention
from hls4ml.converters import convert_from_keras_model

with QuantizerConfigScope(f0=3, i0=2):
    q = keras.layers.Input((4, 8), name='q')
    v = keras.layers.Input((4, 8), name='v')
    k = keras.layers.Input((4, 8), name='k')
    out = QMultiHeadAttention(1, 4, name='hgq_mha', fuse='none')(q, v, k)
    model = keras.Model([q, v, k], out)

hls_model = convert_from_keras_model(
    model,
    backend='Catapult',
    output_dir={str(output_dir)!r},
    io_type='io_parallel',
    hls_config={{'Model': {{'Precision': 'ac_fixed<16,6,true>', 'ReuseFactor': 1, 'Strategy': 'Latency'}}}},
)
hls_model.compile()

rng = np.random.default_rng(12345)
data = [rng.normal(0, 0.25, size=(3, 4, 8)).astype(np.float32) for _ in range(3)]
keras_prediction = model.predict(data, verbose=0)
hls_prediction = hls_model.predict(data).reshape(keras_prediction.shape)

print('max_abs', float(np.max(np.abs(hls_prediction - keras_prediction))))
print('mean_abs', float(np.mean(np.abs(hls_prediction - keras_prediction))))
print('std_hls', float(np.std(hls_prediction)))
np.testing.assert_allclose(hls_prediction, keras_prediction, atol=0.35, rtol=0.0)
assert np.std(hls_prediction) > 1e-3
'''

    result = subprocess.run([sys.executable, '-c', script], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_catapult_hgq2_attention_einsum_gemm_ip_codegen_and_csim(tmp_path):
    output_dir = tmp_path / 'catapult_hgq2_mha_einsum_gemm_ip_prj'
    env = _catapult_csim_env()

    script = f'''
import numpy as np
import keras
from hgq.config import QuantizerConfigScope
from hgq.layers import QMultiHeadAttention
from hls4ml.converters import convert_from_keras_model

with QuantizerConfigScope(f0=3, i0=2):
    q = keras.layers.Input((4, 8), name='q')
    v = keras.layers.Input((4, 8), name='v')
    k = keras.layers.Input((4, 8), name='k')
    out = QMultiHeadAttention(1, 4, name='hgq_mha', fuse='none')(q, v, k)
    model = keras.Model([q, v, k], out)

hls_model = convert_from_keras_model(
    model,
    backend='Catapult',
    output_dir={str(output_dir)!r},
    io_type='io_parallel',
    hls_config={{
        'Model': {{'Precision': 'ac_fixed<16,6,true>', 'ReuseFactor': 1, 'Strategy': 'GEMM'}},
    }},
)
hls_model.compile()

rng = np.random.default_rng(6789)
data = [rng.normal(0, 0.25, size=(2, 4, 8)).astype(np.float32) for _ in range(3)]
keras_prediction = model.predict(data, verbose=0)
hls_prediction = hls_model.predict(data).reshape(keras_prediction.shape)

print('max_abs', float(np.max(np.abs(hls_prediction - keras_prediction))))
print('mean_abs', float(np.mean(np.abs(hls_prediction - keras_prediction))))
np.testing.assert_allclose(hls_prediction, keras_prediction, atol=0.35, rtol=0.0)
'''

    result = subprocess.run([sys.executable, '-c', script], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr

    parameters = (output_dir / 'firmware/parameters.h').read_text()
    top = (output_dir / 'firmware/myproject.cpp').read_text()
    # Full attention GEMM: LowerEinsumToGemm turns the Q/K/V/O projections into
    # const_weights Gemm and the QK^T / A.V einsums into two-operand Gemm, inserting
    # Transpose nodes for the non-identity operand/output permutations.
    assert 'nnet::gemm_array_const_weights<' in top  # projections
    assert 'nnet::gemm_array<' in top             # two-operand QK^T / A.V
    assert 'nnet::transpose_' in top              # inserted operand/output transposes
    assert 'nnet::einsum_gemm_ip<' not in top

    with open(output_dir / 'gemm_config.json') as f:
        gemm_config = json.load(f)
    gemm_entries = [entry for entry in gemm_config.values() if entry['type'] == 'Gemm']
    assert len(gemm_entries) == 6  # 4 projections + QK^T + A.V
    assert {entry['interface'] for entry in gemm_entries} == {'array'}
    # 4 const_weights projections + 2 two-operand matmuls.
    const_weights = [e for e in gemm_entries if e['weights_in_core']]
    two_operand = [e for e in gemm_entries if not e['weights_in_core']]
    assert len(const_weights) == 4
    assert len(two_operand) == 2
    # bias_in_core contract (consumed by gemm-ip-gen): the IP owns the bias adder
    # only for the per-column weight-stationary projections; two-operand matmuls
    # carry no bias, so the port is off.
    assert all(e['bias_in_core'] is True for e in const_weights)
    assert all(e['bias_in_core'] is False for e in two_operand)


def test_catapult_hgq2_attention_io_stream_gemm_ip_codegen_and_csim(tmp_path):
    # The headline capability: io_stream MHA on the GEMM path. LowerEinsumToGemm
    # turns the projections into const_weights Gemm and QK^T / A.V into two-operand
    # Gemm; under io_stream the projections become gemm_stream_const_weights, the
    # matmuls gemm_stream, and the inserted operand/output Transposes stream via
    # nnet::transpose_stream (full reorder buffer, reusing the io_parallel index_map).
    output_dir = tmp_path / 'catapult_hgq2_mha_io_stream_gemm_ip_prj'
    env = _catapult_csim_env()

    script = f'''
import numpy as np
import keras
from hgq.config import QuantizerConfigScope
from hgq.layers import QMultiHeadAttention
from hls4ml.converters import convert_from_keras_model

with QuantizerConfigScope(f0=3, i0=2):
    q = keras.layers.Input((4, 8), name='q')
    v = keras.layers.Input((4, 8), name='v')
    k = keras.layers.Input((4, 8), name='k')
    out = QMultiHeadAttention(1, 4, name='hgq_mha', fuse='none')(q, v, k)
    model = keras.Model([q, v, k], out)

hls_model = convert_from_keras_model(
    model,
    backend='Catapult',
    output_dir={str(output_dir)!r},
    io_type='io_stream',
    hls_config={{
        'Model': {{'Precision': 'ac_fixed<16,6,true>', 'ReuseFactor': 1, 'Strategy': 'GEMM'}},
    }},
)
hls_model.compile()

rng = np.random.default_rng(6789)
data = [rng.normal(0, 0.25, size=(2, 4, 8)).astype(np.float32) for _ in range(3)]
keras_prediction = model.predict(data, verbose=0)
hls_prediction = hls_model.predict(data).reshape(keras_prediction.shape)

print('max_abs', float(np.max(np.abs(hls_prediction - keras_prediction))))
print('mean_abs', float(np.mean(np.abs(hls_prediction - keras_prediction))))
np.testing.assert_allclose(hls_prediction, keras_prediction, atol=0.35, rtol=0.0)
assert np.std(hls_prediction) > 1e-3
'''

    result = subprocess.run([sys.executable, '-c', script], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr

    top = (output_dir / 'firmware/myproject.cpp').read_text()
    # Streaming GEMM cells + the streaming transpose, and NO array-interface leak.
    assert 'nnet::gemm_stream_const_weights<' in top  # projections
    assert 'nnet::gemm_stream<' in top             # two-operand QK^T / A.V
    assert 'nnet::transpose_stream<' in top        # inserted operand/output transposes
    assert 'nnet::gemm_array' not in top
    assert 'nnet::transpose_3d<' not in top
    assert 'nnet::einsum' not in top

    with open(output_dir / 'gemm_config.json') as f:
        gemm_config = json.load(f)
    gemm_entries = [entry for entry in gemm_config.values() if entry['type'] == 'Gemm']
    assert len(gemm_entries) == 6  # 4 projections + QK^T + A.V
    assert {entry['interface'] for entry in gemm_entries} == {'stream'}
    # bias_in_core contract is interface-independent: True for the per-column
    # weight-stationary projections, False for the two-operand matmuls.
    assert all(e['bias_in_core'] is True for e in gemm_entries if e['weights_in_core'])
    assert all(e['bias_in_core'] is False for e in gemm_entries if not e['weights_in_core'])


def test_catapult_hgq2_attention_projection_gemm_ip_codegen_and_csim(tmp_path):
    output_dir = tmp_path / 'catapult_hgq2_mha_projection_gemm_ip_prj'
    env = _catapult_csim_env()

    script = f'''
import numpy as np
import keras
from hgq.config import QuantizerConfigScope
from hgq.layers import QMultiHeadAttention
from hls4ml.converters import convert_from_keras_model

with QuantizerConfigScope(f0=3, i0=2):
    q = keras.layers.Input((4, 8), name='q')
    v = keras.layers.Input((4, 8), name='v')
    k = keras.layers.Input((4, 8), name='k')
    out = QMultiHeadAttention(1, 4, name='hgq_mha', fuse='none')(q, v, k)
    model = keras.Model([q, v, k], out)

hls_model = convert_from_keras_model(
    model,
    backend='Catapult',
    output_dir={str(output_dir)!r},
    io_type='io_parallel',
    hls_config={{
        'Model': {{'Precision': 'ac_fixed<16,6,true>', 'ReuseFactor': 1, 'Strategy': 'Latency'}},
        'LayerType': {{'EinsumDense': {{'Strategy': 'GEMM'}}}},
    }},
)
hls_model.compile()

rng = np.random.default_rng(2468)
data = [rng.normal(0, 0.25, size=(2, 4, 8)).astype(np.float32) for _ in range(3)]
keras_prediction = model.predict(data, verbose=0)
hls_prediction = hls_model.predict(data).reshape(keras_prediction.shape)

print('max_abs', float(np.max(np.abs(hls_prediction - keras_prediction))))
print('mean_abs', float(np.mean(np.abs(hls_prediction - keras_prediction))))
np.testing.assert_allclose(hls_prediction, keras_prediction, atol=0.35, rtol=0.0)
'''

    result = subprocess.run([sys.executable, '-c', script], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr

    parameters = (output_dir / 'firmware/parameters.h').read_text()
    top = (output_dir / 'firmware/myproject.cpp').read_text()
    # Projections are lowered by LowerEinsumToGemm to const_weights Gemm nodes; io_parallel
    # takes the array entry.
    assert 'nnet::gemm_array_const_weights<' in top
    assert 'nnet::einsum_dense_gemm_ip<' not in top
    assert '_gemm_cols.h' in parameters

    with open(output_dir / 'gemm_config.json') as f:
        gemm_config = json.load(f)
    # The four Q/K/V/O projections are now unified Gemm nodes (const_weights, array interface).
    projection_entries = [entry for entry in gemm_config.values() if entry['type'] == 'Gemm']
    assert len(projection_entries) == 4
    assert {entry['gemm_m'] for entry in projection_entries} == {4}
    assert {entry['gemm_k'] for entry in projection_entries} == {4, 8}
    assert {entry['gemm_n'] for entry in projection_entries} == {4, 8}
    assert {entry['interface'] for entry in projection_entries} == {'array'}
    assert {entry['protocol']['kind'] for entry in projection_entries} == {'catapult_ccore_array'}
    assert all(entry['protocol']['input_valid'] == 'scheduled_en_and_in_valid' for entry in projection_entries)
    assert all(entry['protocol']['output_hold'] == 'fifo_until_en' for entry in projection_entries)
    assert all(entry['protocol']['internal_run'] == 'self_timed_after_start' for entry in projection_entries)
    assert all(entry['gemm_ip_id'] == name for name, entry in gemm_config.items() if entry['type'] == 'Gemm')
    assert all(entry['blackbox']['entity'] == f'{name}_core' for name, entry in gemm_config.items() if entry['type'] == 'Gemm')


@pytest.mark.parametrize('io_type', ['io_parallel', 'io_stream'])
def test_catapult_hgq2_attention_multihead_gemm_ip(io_type, tmp_path):
    # Headline: MULTI-head attention on the GEMM path. SplitAttentionHeads rewrites the
    # cluster into H per-head lanes: projections stay 2D [seq, d_model]; a stateless
    # HeadSplit fans each into H [seq, key_dim] streams; each head runs its own QK^T
    # Gemm, Softmax and A.V Gemm; a stateless HeadMerge concatenates the contexts back.
    # The head-move transposes are gone (head is a within-beat lane) — the only residual
    # reorder is the per-head A.V V-transpose. With 2 heads there are 4 projections +
    # 2 QK^T + 2 A.V = 8 Gemm nodes, and exactly 2 (io_stream) V-transposes.
    output_dir = tmp_path / f'catapult_hgq2_mha_multihead_{io_type}_prj'
    env = _catapult_csim_env()

    script = f'''
import numpy as np
import keras
from hgq.config import QuantizerConfigScope
from hgq.layers import QMultiHeadAttention
from hls4ml.converters import convert_from_keras_model

with QuantizerConfigScope(f0=3, i0=2):
    q = keras.layers.Input((4, 8), name='q')
    v = keras.layers.Input((4, 8), name='v')
    k = keras.layers.Input((4, 8), name='k')
    out = QMultiHeadAttention(2, 4, name='hgq_mha', fuse='none')(q, v, k)
    model = keras.Model([q, v, k], out)

hls_model = convert_from_keras_model(
    model,
    backend='Catapult',
    output_dir={str(output_dir)!r},
    io_type={io_type!r},
    hls_config={{
        'Model': {{'Precision': 'ac_fixed<16,6,true>', 'ReuseFactor': 1, 'Strategy': 'GEMM'}},
    }},
)
hls_model.compile()

rng = np.random.default_rng(6789)
data = [rng.normal(0, 0.25, size=(2, 4, 8)).astype(np.float32) for _ in range(3)]
keras_prediction = model.predict(data, verbose=0)
hls_prediction = hls_model.predict(data).reshape(keras_prediction.shape)

print('max_abs', float(np.max(np.abs(hls_prediction - keras_prediction))))
np.testing.assert_allclose(hls_prediction, keras_prediction, atol=0.35, rtol=0.0)
assert np.std(hls_prediction) > 1e-3
'''

    result = subprocess.run([sys.executable, '-c', script], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr

    top = (output_dir / 'firmware/myproject.cpp').read_text()
    # Stateless per-head lane split/merge replace the head-move transposes.
    assert 'nnet::split_lanes' in top
    assert 'nnet::merge_lanes' in top
    # No baseline einsum leaks through.
    assert 'nnet::einsum' not in top

    if io_type == 'io_stream':
        # The ONLY streaming reorder buffers left are the two per-head A.V V-transposes;
        # every head-move transpose is gone.
        assert top.count('nnet::transpose_stream<') == 2
        assert 'nnet::gemm_stream<' in top          # per-head QK^T / A.V
        assert 'nnet::gemm_stream_const_weights<' in top  # projections

    with open(output_dir / 'gemm_config.json') as f:
        gemm_config = json.load(f)
    gemm_entries = [entry for entry in gemm_config.values() if entry['type'] == 'Gemm']
    assert len(gemm_entries) == 8  # 4 projections + 2 QK^T + 2 A.V
    const_weights = [e for e in gemm_entries if e['weights_in_core']]
    two_operand = [e for e in gemm_entries if not e['weights_in_core']]
    assert len(const_weights) == 4   # Q/K/V/O projections
    assert len(two_operand) == 4  # 2 heads x (QK^T + A.V)
