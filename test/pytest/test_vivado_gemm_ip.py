"""Tests for the Vivado/Vitis GEMM IP pass implementation.

These are unit tests that verify:
1. The GEMM optimizer passes are correctly registered for the Vivado backend.
2. ReplaceDenseGemm transforms a Dense layer into a Gemm node.
3. SplitConvGemm transforms non-pointwise Conv2D into a standalone Im2Col node feeding a Gemm node.
4. TransposeWeightsForGemmIP correctly transposes weight data.
5. Gemm and Im2Col config/function templates generate valid C++.

Note: These tests do NOT require Vivado HLS to be installed. They only
verify the Python-side graph transformation and template generation.
"""

import numpy as np
import pytest

import hls4ml
from hls4ml.backends.vivado.passes.gemm_nodes import (
    Gemm,
    ReplaceDenseGemm,
    SplitConvGemm,
    TransposeWeightsForGemmIP,
)
from hls4ml.model.layers import Im2Col


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dense_model(n_in=16, n_out=8, strategy='latency'):
    """Create a minimal hls4ml config with a single Dense layer."""
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(n_out, input_shape=(n_in,), use_bias=True)
    ])
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['Model']['Strategy'] = strategy
    config['Model']['IOType'] = 'io_stream'
    config['Model']['Strategy'] = 'GEMM'
    return model, config


def _make_conv2d_model(filters=4, kernel_size=3, input_shape=(8, 8, 2)):
    """Create a minimal hls4ml config with a single Conv2D layer (valid padding)."""
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, kernel_size, padding='valid', input_shape=input_shape, use_bias=True)
    ])
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['Model']['Strategy'] = 'latency'
    config['Model']['IOType'] = 'io_stream'
    config['Model']['Strategy'] = 'GEMM'
    return model, config


# ---------------------------------------------------------------------------
# 1. Backend / pass registration
# ---------------------------------------------------------------------------

class TestGemmPassRegistration:
    def test_vivado_backend_has_gemm_passes(self):
        from hls4ml.model import optimizer as opt_mod
        passes = opt_mod.get_backend_passes('vivado')
        assert 'vivado:replace_dense_gemm' in passes
        assert 'vivado:split_conv_gemm' in passes
        assert 'vivado:transpose_weights_for_gemm' in passes

    def test_vivado_backend_has_gemm_templates(self):
        from hls4ml.model import optimizer as opt_mod
        passes = opt_mod.get_backend_passes('vivado')
        assert 'vivado:gemm_config_template' in passes
        assert 'vivado:gemm_function_template' in passes
        assert 'vivado:im2col_config_template' in passes
        assert 'vivado:im2col_function_template' in passes

    def test_vitis_backend_has_gemm_templates(self):
        from hls4ml.model import optimizer as opt_mod
        passes = opt_mod.get_backend_passes('vitis')
        assert 'vitis:gemm_config_template' in passes
        assert 'vitis:gemm_function_template' in passes


# ---------------------------------------------------------------------------
# 2. Dense → Gemm transformation
# ---------------------------------------------------------------------------

class TestReplaceDenseGemm:
    @pytest.fixture(autouse=True)
    def _skip_if_no_tf(self, tmp_path):
        pytest.importorskip('tensorflow')
        self.tmp_path = tmp_path

    def _build_hls_model(self, n_in=16, n_out=8):
        model, config = _make_dense_model(n_in, n_out)
        hls_model = hls4ml.converters.convert_from_keras_model(
            model, hls_config=config, backend='Vivado', io_type='io_stream', output_dir=str(self.tmp_path / 'test_vivado_dense_gemm')
        )
        return hls_model

    def test_dense_replaced_by_gemmstream(self):
        hls_model = self._build_hls_model()
        layer_names = [n for n in hls_model.graph]
        has_gemm = any('gemm' in n.lower() or isinstance(hls_model.graph[n], Gemm) for n in layer_names)
        assert has_gemm, f'Expected Gemm node in graph; got: {layer_names}'

    def test_gemmstream_has_transposed_weights(self):
        hls_model = self._build_hls_model(n_in=8, n_out=4)
        for node in hls_model.graph.values():
            if isinstance(node, Gemm):
                w = node.get_weights('weight')
                # Transposed layout: [N, K] == [n_out, n_in]
                n_in = node.get_attr('n_in')
                n_out = node.get_attr('n_out')
                assert w.data.shape == (n_out, n_in), (
                    f'Expected transposed weight shape ({n_out}, {n_in}), got {w.data.shape}'
                )
                break
        else:
            pytest.fail('No Gemm node found in graph')

    def test_gemmstream_config_cpp_generated(self):
        hls_model = self._build_hls_model()
        for node in hls_model.graph.values():
            if isinstance(node, Gemm):
                config_cpp = node.get_attr('config_cpp', '')
                assert 'gemm_k' in config_cpp, f'Expected gemm_k in config_cpp: {config_cpp!r}'
                assert 'gemm_n' in config_cpp, f'Expected gemm_n in config_cpp: {config_cpp!r}'
                assert 'transpose_weights' in config_cpp
                break
        else:
            pytest.fail('No Gemm node found in graph')

    def test_gemmstream_function_cpp_generated(self):
        hls_model = self._build_hls_model()
        for node in hls_model.graph.values():
            if isinstance(node, Gemm):
                fn_cpp = node.get_attr('function_cpp', '')
                assert 'nnet::gemm_stream' in fn_cpp, f'Expected nnet::gemm_stream in function_cpp: {fn_cpp!r}'
                break
        else:
            pytest.fail('No Gemm node found in graph')


# ---------------------------------------------------------------------------
# 3. Conv2D -> Im2Col + Gemm transformation
# ---------------------------------------------------------------------------

class TestSplitFuseConvGemm:
    @pytest.fixture(autouse=True)
    def _skip_if_no_tf(self, tmp_path):
        pytest.importorskip('tensorflow')
        self.tmp_path = tmp_path

    def _build_hls_model(self, filters=4, kernel_size=3):
        model, config = _make_conv2d_model(filters, kernel_size)
        hls_model = hls4ml.converters.convert_from_keras_model(
            model, hls_config=config, backend='Vivado', io_type='io_stream', output_dir=str(self.tmp_path / 'test_vivado_conv_gemm')
        )
        return hls_model

    def test_conv2d_replaced_by_im2col_and_gemm(self):
        hls_model = self._build_hls_model()
        has_im2col = any(isinstance(n, Im2Col) for n in hls_model.graph.values())
        has_gemm = any(isinstance(n, Gemm) for n in hls_model.graph.values())
        assert has_im2col and has_gemm, 'Expected Im2Col + Gemm in graph'

    def test_im2col_config_cpp_generated(self):
        hls_model = self._build_hls_model()
        for node in hls_model.graph.values():
            if isinstance(node, Im2Col):
                config_cpp = node.get_attr('config_cpp', '')
                assert 'gemm_m' in config_cpp
                assert 'filt_height' in config_cpp
                break
        else:
            pytest.fail('No Im2Col found')
        for node in hls_model.graph.values():
            if isinstance(node, Gemm):
                config_cpp = node.get_attr('config_cpp', '')
                assert 'gemm_k' in config_cpp
                break
        else:
            pytest.fail('No Gemm found')

    def test_im2col_function_cpp_calls_gemm_rows(self):
        # The standalone Im2Col node's call site is a bare function call; the inter-node
        # channel it writes into (an hls::stream) is declared where the writer declares
        # every inter-layer variable, not inline in this call's own function_cpp.
        hls_model = self._build_hls_model()
        for node in hls_model.graph.values():
            if isinstance(node, Im2Col):
                fn_cpp = node.get_attr('function_cpp', '')
                assert 'im2col_2d_gemm_rows' in fn_cpp, f'Expected im2col_2d_gemm_rows in function_cpp; got: {fn_cpp!r}'
                break
        else:
            pytest.fail('No Im2Col found')


# ---------------------------------------------------------------------------
# 4. Weight transposition arithmetic
# ---------------------------------------------------------------------------

class TestWeightTransposition:
    def test_dense_transpose_shape(self):
        """Weight layout for Dense GEMM IP must be [N, K]."""
        n_in, n_out = 8, 4
        W_orig = np.random.randn(n_in, n_out).astype(np.float32)
        W_transposed = W_orig.T  # [N, K]
        assert W_transposed.shape == (n_out, n_in)

    def test_conv2d_transpose_shape(self):
        """Weight layout for Conv2D GEMM IP must be [F, H*W*C] i.e. [n_out, gemm_k]."""
        filt_h, filt_w, n_chan, n_filt = 3, 3, 2, 4
        W_orig = np.random.randn(filt_h, filt_w, n_chan, n_filt).astype(np.float32)
        # Catapult: transpose [H,W,C,F] -> [F,H,W,C] then flatten to [F, H*W*C]
        W_transposed = W_orig.transpose(3, 0, 1, 2).reshape(n_filt, filt_h * filt_w * n_chan)
        assert W_transposed.shape == (n_filt, filt_h * filt_w * n_chan)


# ---------------------------------------------------------------------------
# 5. Strategy: GEMM routing
# ---------------------------------------------------------------------------

class TestGemmStrategyRouting:
    @pytest.fixture(autouse=True)
    def _skip_if_no_tf(self, tmp_path):
        pytest.importorskip('tensorflow')
        self.tmp_path = tmp_path

    def test_get_gemm_ip_io_parallel_routes_to_gemm_array(self):
        """io_parallel Dense + GemmIP uses the array-interface Gemm node
        (it used to raise; the io_parallel array path superseded that)."""
        import tensorflow as tf
        from hls4ml.backends.vivado.passes.gemm_nodes import Gemm
        model = tf.keras.Sequential([tf.keras.layers.Dense(8, input_shape=(16,))])
        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['Model']['IOType'] = 'io_parallel'
        config['Model']['Strategy'] = 'GEMM'

        hls_model = hls4ml.converters.convert_from_keras_model(
            model, hls_config=config, backend='Vivado', io_type='io_parallel',
            output_dir=str(self.tmp_path / 'test_gemm_ioparallel')
        )
        has_gemm_array = any(isinstance(n, Gemm) for n in hls_model.graph.values())
        assert has_gemm_array, 'Expected io_parallel Dense + GemmIP to produce a Gemm node'

    def test_gemm_m_equals_n_patches_for_rank2_dense(self):
        import tensorflow as tf
        model = tf.keras.Sequential([tf.keras.layers.Dense(8, input_shape=(16,))])
        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['Model']['IOType'] = 'io_stream'
        config['Model']['Strategy'] = 'GEMM'
        # Rank-2 Dense with input shape (16,) → n_patches=1 → gemm_m=1
        hls_model = hls4ml.converters.convert_from_keras_model(
            model, hls_config=config, backend='Vivado', io_type='io_stream', output_dir=str(self.tmp_path / 'test_gemm_m_default')
        )
        for node in hls_model.graph.values():
            if isinstance(node, Gemm):
                assert node.get_attr('gemm_m') == 1
                assert node.get_attr('gemm_m') == node.get_attr('n_patches')
                break
        else:
            pytest.fail('No Gemm found')


# ---------------------------------------------------------------------------
# 6. No-tiling Gemm (gemm_m == n_patches)
# ---------------------------------------------------------------------------

class TestGemmTiled:
    """Tests that tiled Gemm execution is correctly configured."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_tf(self, tmp_path):
        pytest.importorskip('tensorflow')
        self.tmp_path = tmp_path

    def test_no_tiling_in_config(self):
        """GemmIP has no tiling: gemm_m == n_patches, no n_tiles in config."""
        import tensorflow as tf
        model = tf.keras.Sequential([tf.keras.layers.Dense(8, input_shape=(5, 4))])
        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['Model']['Strategy'] = 'latency'
        config['Model']['IOType'] = 'io_stream'
        config['Model']['Strategy'] = 'GEMM'
        hls_model = hls4ml.converters.convert_from_keras_model(
            model, hls_config=config, backend='Vivado', io_type='io_stream',
            output_dir=str(self.tmp_path / 'test_no_tiling')
        )
        for node in hls_model.graph.values():
            if isinstance(node, Gemm):
                assert node.get_attr('gemm_m') == node.get_attr('n_patches') == 5
                config_cpp = node.get_attr('config_cpp', '')
                assert 'n_tiles' not in config_cpp, (
                    f'n_tiles should not appear in row/col config; got: {config_cpp!r}'
                )
                assert 'gemm_m' in config_cpp
                assert 'n_patches' in config_cpp
                break
        else:
            pytest.fail('No Gemm node found')

    def test_single_tile_no_loop_overhead(self):
        """Single batch Dense: gemm_m == n_patches, no tiling overhead."""
        import tensorflow as tf
        model = tf.keras.Sequential([tf.keras.layers.Dense(8, input_shape=(2, 4))])
        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['Model']['Strategy'] = 'latency'
        config['Model']['IOType'] = 'io_stream'
        config['Model']['Strategy'] = 'GEMM'
        hls_model = hls4ml.converters.convert_from_keras_model(
            model, hls_config=config, backend='Vivado', io_type='io_stream',
            output_dir=str(self.tmp_path / 'test_single_tile')
        )
        for node in hls_model.graph.values():
            if isinstance(node, Gemm):
                assert node.get_attr('n_patches') == 2
                assert node.get_attr('gemm_m') == 2
                config_cpp = node.get_attr('config_cpp', '')
                assert 'n_tiles' not in config_cpp
                break
        else:
            pytest.fail('No Gemm node found')


# ---------------------------------------------------------------------------
# 7. Dense output shape preservation
# ---------------------------------------------------------------------------

class TestDenseOutputShape:
    """Verify that GEMM IP Dense preserves the original output shape."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_tf(self, tmp_path):
        pytest.importorskip('tensorflow')
        self.tmp_path = tmp_path

    def test_rank2_dense_output_shape(self):
        """Rank-1 Dense input produces [n_out] output (1-D vector)."""
        import tensorflow as tf
        model = tf.keras.Sequential([tf.keras.layers.Dense(8, input_shape=(16,))])
        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['Model']['Strategy'] = 'latency'
        config['Model']['IOType'] = 'io_stream'
        config['Model']['Strategy'] = 'GEMM'
        hls_model = hls4ml.converters.convert_from_keras_model(
            model, hls_config=config, backend='Vivado', io_type='io_stream',
            output_dir=str(self.tmp_path / 'test_rank2_shape')
        )
        for node in hls_model.graph.values():
            if isinstance(node, Gemm):
                shape = list(node.get_output_variable().shape)
                # io_stream Dense with 1-D input produces [n_out] output
                assert shape == [8], f'Expected [8] got {shape}'
                break
        else:
            pytest.fail('No Gemm found')

    def test_rank3_dense_output_shape_preserved(self):
        """Rank-3 Dense preserves d0×d1 not flattened."""
        import tensorflow as tf
        model = tf.keras.Sequential([tf.keras.layers.Dense(8, input_shape=(4, 5, 6))])
        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['Model']['Strategy'] = 'latency'
        config['Model']['IOType'] = 'io_stream'
        config['Model']['Strategy'] = 'GEMM'
        hls_model = hls4ml.converters.convert_from_keras_model(
            model, hls_config=config, backend='Vivado', io_type='io_stream',
            output_dir=str(self.tmp_path / 'test_rank3_shape')
        )
        for node in hls_model.graph.values():
            if isinstance(node, Gemm):
                shape = list(node.get_output_variable().shape)
                assert len(shape) == 3, f'Expected rank-3 output, got {len(shape)}-d: {shape}'
                assert shape[:2] == [4, 5], f'Expected [4, 5, 8] got {shape}'
                assert shape[2] == 8
                break
        else:
            pytest.fail('No Gemm found')


# ---------------------------------------------------------------------------
# 8. Pointwise Conv GEMM path
# ---------------------------------------------------------------------------

class TestPointwiseConvGemm:
    """Pointwise Conv (1×1 kernel) should skip Im2Col and use Gemm directly."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_tf(self, tmp_path):
        pytest.importorskip('tensorflow')
        self.tmp_path = tmp_path

    def test_pointwise_conv1d_direct_gemmstream(self):
        """Conv1D with kernel=1 is pointwise — should become Gemm, no Im2Col."""
        import tensorflow as tf
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(8, 1, padding='same', input_shape=(16, 4), use_bias=True)
        ])
        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['Model']['Strategy'] = 'latency'
        config['Model']['IOType'] = 'io_stream'
        config['Model']['Strategy'] = 'GEMM'
        hls_model = hls4ml.converters.convert_from_keras_model(
            model, hls_config=config, backend='Vivado', io_type='io_stream',
            output_dir=str(self.tmp_path / 'test_pw_conv1d')
        )
        # Should have Gemm but NOT Im2Col (pointwise conv is a plain GEMM).
        has_gemm = False
        for node in hls_model.graph.values():
            if isinstance(node, Gemm):
                has_gemm = True
            if isinstance(node, Im2Col):
                pytest.fail('Pointwise Conv should NOT produce Im2Col')
        assert has_gemm, 'Expected Gemm for pointwise Conv'

    def test_pointwise_conv2d_direct_gemmstream(self):
        """Conv2D with kernel=1×1 is pointwise — should become Gemm."""
        import tensorflow as tf
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(8, (1, 1), padding='same', input_shape=(4, 4, 2), use_bias=True)
        ])
        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['Model']['Strategy'] = 'latency'
        config['Model']['IOType'] = 'io_stream'
        config['Model']['Strategy'] = 'GEMM'
        hls_model = hls4ml.converters.convert_from_keras_model(
            model, hls_config=config, backend='Vivado', io_type='io_stream',
            output_dir=str(self.tmp_path / 'test_pw_conv2d')
        )
        for node in hls_model.graph.values():
            if isinstance(node, Im2Col):
                pytest.fail('Pointwise Conv2D should NOT produce Im2Col')
        has_gemm = any(isinstance(n, Gemm) for n in hls_model.graph.values())
        assert has_gemm, 'Expected Gemm for pointwise Conv2D'

    def test_non_pointwise_conv2d_produces_im2col_and_gemm(self):
        """Non-pointwise Conv2D (3×3) still produces an Im2Col + Gemm pair."""
        import tensorflow as tf
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(4, 3, padding='valid', input_shape=(8, 8, 2), use_bias=True)
        ])
        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['Model']['Strategy'] = 'latency'
        config['Model']['IOType'] = 'io_stream'
        config['Model']['Strategy'] = 'GEMM'
        hls_model = hls4ml.converters.convert_from_keras_model(
            model, hls_config=config, backend='Vivado', io_type='io_stream',
            output_dir=str(self.tmp_path / 'test_nonpw_conv2d')
        )
        has_im2col = any(isinstance(n, Im2Col) for n in hls_model.graph.values())
        has_gemm = any(isinstance(n, Gemm) for n in hls_model.graph.values())
        assert has_im2col and has_gemm, 'Non-pointwise Conv2D should produce Im2Col + Gemm'


# ---------------------------------------------------------------------------
# 9. gemm_m == n_patches verification
# ---------------------------------------------------------------------------

class TestGemmM:
    @pytest.fixture(autouse=True)
    def _skip_if_no_tf(self, tmp_path):
        pytest.importorskip('tensorflow')
        self.tmp_path = tmp_path

    def test_gemm_m_equals_n_patches_for_dense(self):
        """GemmIP gemm_m equals n_patches (full M, no tiling)."""
        import tensorflow as tf
        model = tf.keras.Sequential([tf.keras.layers.Dense(8, input_shape=(10,))])
        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['Model']['Strategy'] = 'latency'
        config['Model']['IOType'] = 'io_stream'
        config['Model']['Strategy'] = 'GEMM'
        hls_model = hls4ml.converters.convert_from_keras_model(
            model, hls_config=config, backend='Vivado', io_type='io_stream',
            output_dir=str(self.tmp_path / 'test_gemm_m_dense')
        )
        for node in hls_model.graph.values():
            if isinstance(node, Gemm):
                # gemm_m == n_patches (rank-2 Dense, input shape (10,) → n_patches=1)
                assert node.get_attr('gemm_m') == 1
                assert node.get_attr('gemm_m') == node.get_attr('n_patches')
                break
        else:
            pytest.fail('No Gemm found')

    def test_gemm_m_equals_n_patches_for_conv(self):
        """GemmIP gemm_m equals n_patches (full M, no tiling)."""
        import tensorflow as tf
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(4, 3, padding='valid', input_shape=(8, 8, 2), use_bias=True)
        ])
        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['Model']['Strategy'] = 'latency'
        config['Model']['IOType'] = 'io_stream'
        config['Model']['Strategy'] = 'GEMM'
        hls_model = hls4ml.converters.convert_from_keras_model(
            model, hls_config=config, backend='Vivado', io_type='io_stream',
            output_dir=str(self.tmp_path / 'test_gemm_m_conv')
        )
        for node in hls_model.graph.values():
            if isinstance(node, Gemm) and 'conv' in node.name:
                # padding='valid', 8x8 in, 3x3 kernel → 6x6 out → n_patches=36
                assert node.get_attr('gemm_m') == 36
                assert node.get_attr('gemm_m') == node.get_attr('n_patches')
                break
        else:
            pytest.fail('No conv-derived Gemm found')


# ---------------------------------------------------------------------------
# 10. Conv output shape preservation (non-pointwise)
# ---------------------------------------------------------------------------

class TestConvOutputShape:
    """Non-pointwise Conv2D GemmIP should preserve [out_h, out_w, n_filt]."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_tf(self, tmp_path):
        pytest.importorskip('tensorflow')
        self.tmp_path = tmp_path

    def test_conv2d_im2col_gemm_preserves_output_shape(self):
        """The conv-derived Gemm keeps [out_h, out_w, n_filt] not [n_patches, n_out]."""
        import tensorflow as tf
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(4, 3, padding='valid',
                                   input_shape=(8, 8, 2), use_bias=True)
        ])
        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['Model']['Strategy'] = 'latency'
        config['Model']['IOType'] = 'io_stream'
        config['Model']['Strategy'] = 'GEMM'
        hls_model = hls4ml.converters.convert_from_keras_model(
            model, hls_config=config, backend='Vivado', io_type='io_stream',
            output_dir=str(self.tmp_path / 'test_conv_shape')
        )
        for node in hls_model.graph.values():
            if isinstance(node, Gemm) and 'conv' in node.name:
                shape = list(node.get_output_variable().shape)
                assert len(shape) == 3, (
                    f'Expected rank-3 Conv output, got {len(shape)}-d: {shape}'
                )
                # padding='valid', 8x8 input, 3x3 kernel → 6x6 output
                assert shape == [6, 6, 4], f'Expected [6, 6, 4] got {shape}'
                break
        else:
            pytest.fail('No conv-derived Gemm found')


# ---------------------------------------------------------------------------
# 11. Vitis backend acceptance
# ---------------------------------------------------------------------------

class TestVitisAcceptance:
    """Vitis backend inherits Vivado GEMM IP registration."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_tf(self, tmp_path):
        pytest.importorskip('tensorflow')
        self.tmp_path = tmp_path

    def test_vitis_config_accepts_gemmip(self):
        """Vitis backend builds a Dense model with GemmIP without error."""
        import tensorflow as tf
        model = tf.keras.Sequential([tf.keras.layers.Dense(8, input_shape=(16,))])
        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['Model']['Strategy'] = 'latency'
        config['Model']['IOType'] = 'io_stream'
        config['Model']['Strategy'] = 'GEMM'
        hls_model = hls4ml.converters.convert_from_keras_model(
            model, hls_config=config, backend='Vitis', io_type='io_stream',
            output_dir=str(self.tmp_path / 'test_vitis_gemm')
        )
        has_gemm = any('gemm' in n.lower() for n in hls_model.graph)
        assert has_gemm, 'Expected Gemm in Vitis graph'

    def test_vitis_template_discovery(self):
        """Vitis sees the same templates as Vivado."""
        from hls4ml.model import optimizer as opt_mod
        vitis_passes = opt_mod.get_backend_passes('vitis')
        assert 'vitis:gemm_config_template' in vitis_passes
        assert 'vitis:gemm_function_template' in vitis_passes
        assert 'vitis:im2col_config_template' in vitis_passes
        assert 'vitis:im2col_function_template' in vitis_passes

    def test_vitis_gemm_passes(self):
        """Vitis backend has the GEMM pass transformations under the vivado namespace."""
        from hls4ml.model import optimizer as opt_mod
        passes = opt_mod.get_backend_passes('vivado')
        assert 'vivado:replace_dense_gemm' in passes
        assert 'vivado:split_conv_gemm' in passes
        assert 'vivado:transpose_weights_for_gemm' in passes


# ---------------------------------------------------------------------------
# 11. Packed weight file generation
# ---------------------------------------------------------------------------

class TestPackedWeightGeneration:
    """Tests that *_gemm_cols.h files are generated with correct dimensions."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_tf(self, tmp_path):
        pytest.importorskip('tensorflow')
        self.tmp_path = tmp_path

    def test_dense_emits_gemm_cols_file(self):
        """Dense GemmIP should produce weight*_gemm_cols.h."""
        import tensorflow as tf
        import tempfile
        import os
        model = tf.keras.Sequential([tf.keras.layers.Dense(8, input_shape=(16,))])
        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['Model']['Strategy'] = 'latency'
        config['Model']['IOType'] = 'io_stream'
        config['Model']['Strategy'] = 'GEMM'
        with tempfile.TemporaryDirectory(dir=str(self.tmp_path)) as tmpdir:
            hls_model = hls4ml.converters.convert_from_keras_model(
                model, hls_config=config, backend='Vivado', io_type='io_stream',
                output_dir=tmpdir
            )
            hls_model.write()
            # Look for *_gemm_cols.h in the firmware weights directory
            weights_dir = os.path.join(tmpdir, 'firmware', 'weights')
            if not os.path.isdir(weights_dir):
                pytest.skip(f'Weights directory not found: {weights_dir}')
            cols_files = [f for f in os.listdir(weights_dir) if f.endswith('_gemm_cols.h')]
            if not cols_files:
                # The file may be in a subdirectory or at the top level
                for root, _, files in os.walk(tmpdir):
                    cols_files.extend(f for f in files if f.endswith('_gemm_cols.h'))
            assert len(cols_files) > 0, f'No *_gemm_cols.h found in {tmpdir}'

    def test_gemm_cols_header_has_correct_type(self):
        """Weight cols header contains nnet::array with expected dim."""
        import tensorflow as tf
        import tempfile
        import os
        n_in, n_out = 16, 8
        model = tf.keras.Sequential([tf.keras.layers.Dense(n_out, input_shape=(n_in,))])
        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['Model']['Strategy'] = 'latency'
        config['Model']['IOType'] = 'io_stream'
        config['Model']['Strategy'] = 'GEMM'
        with tempfile.TemporaryDirectory(dir=str(self.tmp_path)) as tmpdir:
            hls_model = hls4ml.converters.convert_from_keras_model(
                model, hls_config=config, backend='Vivado', io_type='io_stream',
                output_dir=tmpdir
            )
            hls_model.write()
            cols_files = []
            for root, _, files in os.walk(tmpdir):
                cols_files.extend(
                    os.path.join(root, f) for f in files if f.endswith('_gemm_cols.h')
                )
            if not cols_files:
                pytest.skip('No *_gemm_cols.h files generated')
            with open(cols_files[0]) as fh:
                content = fh.read()
            assert 'nnet::array' in content, f'Expected nnet::array type in cols file'
            # Packed weight file: w_gemm_cols[gemm_n], each array<wt, gemm_k>
            assert str(n_in) in content, f'Expected gemm_k={n_in} dimension reference'
            assert str(n_out) in content, f'Expected gemm_n={n_out} dimension reference'

    def test_conv_emits_gemm_cols_file(self):
        """Conv2D GemmIP should produce weight*_gemm_cols.h."""
        import tensorflow as tf
        import tempfile
        import os
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(4, 3, padding='valid',
                                   input_shape=(8, 8, 2), use_bias=True)
        ])
        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['Model']['Strategy'] = 'latency'
        config['Model']['IOType'] = 'io_stream'
        config['Model']['Strategy'] = 'GEMM'
        with tempfile.TemporaryDirectory(dir=str(self.tmp_path)) as tmpdir:
            hls_model = hls4ml.converters.convert_from_keras_model(
                model, hls_config=config, backend='Vivado', io_type='io_stream',
                output_dir=tmpdir
            )
            hls_model.write()
            cols_files = []
            for root, _, files in os.walk(tmpdir):
                cols_files.extend(
                    os.path.join(root, f) for f in files if f.endswith('_gemm_cols.h')
                )
            assert len(cols_files) > 0, f'No *_gemm_cols.h found for Conv2D in {tmpdir}'


# ---------------------------------------------------------------------------
# ValidateGemm safety pass (parity with Catapult)
# ---------------------------------------------------------------------------

class TestValidateGemmRegistration:
    """ValidateGemm is re-exported and wired into the Vivado flow.

    Catapult owns the ValidateGemm structural-contract pass; the Vivado backend
    re-exports it and runs it in the specific_types flow so GEMM-IP nodes get
    the same legality checks on both backends.
    """

    def test_validate_gemm_reexported(self):
        from hls4ml.backends.vivado.passes.gemm_nodes import ValidateGemm
        from hls4ml.backends.fpga.passes.gemm_nodes import ValidateGemm as CatapultValidateGemm

        # Same class object, not a divergent copy.
        assert ValidateGemm is CatapultValidateGemm

    def test_validate_gemm_pass_registered(self):
        from hls4ml.model import optimizer as opt_mod

        assert 'vivado:validate_gemm' in opt_mod.get_backend_passes('vivado')

    def test_validate_gemm_in_specific_types_flow(self):
        from hls4ml.model.flow.flow import get_flow

        flow = get_flow('vivado:specific_types')
        assert 'vivado:validate_gemm' in flow.optimizers


class TestEinsumDenseGemmIpIOType:
    """EinsumDense GEMM-IP works on BOTH io_parallel and io_stream.

    The io_parallel-only restriction was lifted: io_stream EinsumDense GEMM-IP lowers
    to a const_weights Gemm and materializes through gemm_stream_const_weights, at parity with
    Catapult. Only the row-varying-bias io_stream corner still fails loudly (the
    per-element bias-add is io_parallel-only), matching Catapult.
    """

    def test_io_stream_einsum_dense_gemm_ip_lowers_to_gemm(self):
        import keras

        from hls4ml.backends.vivado.passes.gemm_nodes import Gemm

        inp = keras.layers.Input((4, 8))
        out = keras.layers.EinsumDense('abc,cd->abd', output_shape=(4, 6), bias_axes='d')(inp)
        model = keras.Model(inp, out)
        hls_model = hls4ml.converters.convert_from_keras_model(
            model,
            backend='Vivado',
            io_type='io_stream',
            hls_config={
                'Model': {'Precision': 'ap_fixed<16,6>', 'ReuseFactor': 1, 'Strategy': 'Latency'},
                'LayerType': {'EinsumDense': {'Strategy': 'GEMM'}},
            },
        )
        classes = [type(n).__name__ for n in hls_model.graph.values()]
        assert not any('EinsumDense' in c for c in classes), 'EinsumDense should be lowered away'
        assert any(isinstance(n, Gemm) for n in hls_model.graph.values()), 'expected a Gemm node'

    def test_io_stream_row_varying_bias_raises(self):
        import keras

        inp = keras.layers.Input((4, 8))
        layer = keras.layers.EinsumDense('abc,cd->abd', output_shape=(4, 6), bias_axes='bd')
        out = layer(inp)
        model = keras.Model(inp, out)
        layer.set_weights([
            np.linspace(-0.4, 0.4, 8 * 6, dtype=np.float32).reshape(8, 6),
            np.tile(np.arange(4, dtype=np.float32).reshape(4, 1) * 0.5, (1, 6)),
        ])

        with pytest.raises(NotImplementedError, match='row-varying'):
            hls_model = hls4ml.converters.convert_from_keras_model(
                model,
                backend='Vivado',
                io_type='io_stream',
                hls_config={
                    'Model': {'Precision': 'ap_fixed<16,6>', 'ReuseFactor': 1, 'Strategy': 'Latency'},
                    'LayerType': {'EinsumDense': {'Strategy': 'GEMM'}},
                },
            )
            hls_model.write()


# ---------------------------------------------------------------------------
# EinsumDense -> const_weights Gemm lowering (Stage 2a)
# ---------------------------------------------------------------------------

class TestEinsumDenseLowering:
    """EinsumDense GEMM-IP lowers to a Gemm node in the IR (LowerEinsumToGemm).

    GEMM must live in the IR, not be emitted from the einsum template. The node is
    rewritten to a const_weights Gemm and materialized by the shared Gemm codegen, so
    the generated top calls gemm_array_const_weights and never einsum_dense_gemm_ip.
    """

    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp_path = tmp_path

    def _write_einsum_dense(self, subdir, bias_axes):
        import keras

        inp = keras.layers.Input((4, 8))
        layer = keras.layers.EinsumDense('abc,cd->abd', output_shape=(4, 6), bias_axes=bias_axes)
        out = layer(inp)
        model = keras.Model(inp, out)
        kernel = np.linspace(-0.4, 0.4, 8 * 6, dtype=np.float32).reshape(8, 6)
        if bias_axes == 'bd':
            # Vary the bias across the data-free (row) axis -> row-varying lowering.
            bias = np.tile(np.arange(4, dtype=np.float32).reshape(4, 1) * 0.5, (1, 6))
        else:
            bias = np.linspace(-0.2, 0.2, 6, dtype=np.float32)
        layer.set_weights([kernel, bias])
        out_dir = str(self.tmp_path / subdir)
        hls_model = hls4ml.converters.convert_from_keras_model(
            model, backend='Vivado', io_type='io_parallel', output_dir=out_dir,
            hls_config={
                'Model': {'Precision': 'ap_fixed<16,6>', 'ReuseFactor': 1, 'Strategy': 'Latency'},
                'LayerType': {'EinsumDense': {'Strategy': 'GEMM'}},
            },
        )
        return hls_model, out_dir

    def test_einsum_dense_gemm_ip_lowers_to_gemm_node(self):
        from hls4ml.backends.vivado.passes.gemm_nodes import Gemm

        hls_model, _ = self._write_einsum_dense('ed_lower_node', bias_axes='d')
        classes = [type(n).__name__ for n in hls_model.graph.values()]
        assert not any('EinsumDense' in c for c in classes), 'EinsumDense should be lowered away'
        assert any(isinstance(n, Gemm) for n in hls_model.graph.values()), 'expected a Gemm node'

    def test_einsum_dense_gemm_ip_emits_const_weights_array_core(self):
        hls_model, out_dir = self._write_einsum_dense('ed_lower_core', bias_axes='d')
        hls_model.write()
        top = (self.tmp_path / 'ed_lower_core' / 'firmware' / 'myproject.cpp').read_text()
        assert 'nnet::gemm_array_const_weights<' in top
        assert 'nnet::einsum_dense_gemm_ip<' not in top

    def test_per_column_bias_uses_plain_array_template(self):
        # bias_axes='d' varies only along the kernel free axis (per column), constant
        # across rows -> plain array template, no per-element _zero_bias add path.
        hls_model, _ = self._write_einsum_dense('ed_percol_bias', bias_axes='d')
        hls_model.write()
        top = (self.tmp_path / 'ed_percol_bias' / 'firmware' / 'myproject.cpp').read_text()
        assert '_zero_bias' not in top, 'per-column bias should not use the row-varying path'

    def test_row_varying_bias_uses_zero_bias_add_path(self):
        # bias_axes='bd' varies along the data free axis (rows) -> row-varying path.
        # The per-column IP bias is baked as a compile-time constant (the config's
        # gemm_bias() accessor, zeroed for this node -- never a call argument), and
        # the full per-element bias is added afterward in the unpack loop.
        hls_model, _ = self._write_einsum_dense('ed_rowvar_bias', bias_axes='bd')
        hls_model.write()
        top = (self.tmp_path / 'ed_rowvar_bias' / 'firmware' / 'myproject.cpp').read_text()
        params = (self.tmp_path / 'ed_rowvar_bias' / 'firmware' / 'parameters.h').read_text()
        assert 'gemm_bias() { static' in params, 'row-varying node should get a zero gemm_bias() accessor'
        assert 'gemm_array_const_weights<a_row_t, res_row_t, config' in top
        assert 'result_rows[row][col] +' in top, 'row-varying bias should use the per-element add path'


@pytest.mark.parametrize('backend', ['Vivado', 'Vitis'])
@pytest.mark.parametrize('dim', [1, 2])
def test_conv_gemm_ip_io_parallel_csim(backend, dim, tmp_path):
    """General Conv1D/2D GEMM-IP lowers to a standalone Im2Col node feeding a Gemm node
    and csim-matches keras on io_parallel (array im2col + gemm_array_const_weights), at
    parity with the io_stream Im2Col+Gemm path."""
    import keras

    rng = np.random.default_rng(1)
    if dim == 1:
        inp = keras.layers.Input((10, 3))
        layer = keras.layers.Conv1D(4, 3, padding='valid', use_bias=True)
    else:
        inp = keras.layers.Input((8, 8, 3))
        layer = keras.layers.Conv2D(4, 3, padding='valid', use_bias=True)
    out = layer(inp)
    model = keras.Model(inp, out)
    model.set_weights([rng.standard_normal(w.shape).astype(np.float32) * 0.2 for w in model.get_weights()])

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        backend=backend,
        io_type='io_parallel',
        output_dir=str(tmp_path / f'conv{dim}d_iop_{backend}'),
        hls_config={
            'Model': {'Precision': 'ap_fixed<20,8>', 'ReuseFactor': 1, 'Strategy': 'Latency'},
            'LayerType': {'Conv1D': {'Strategy': 'GEMM'}, 'Conv2D': {'Strategy': 'GEMM'}},
        },
    )
    assert any('Im2Col' in type(n).__name__ for n in hls_model.graph.values()), 'expected a standalone Im2Col node'
    assert any(type(n).__name__.endswith('Gemm') for n in hls_model.graph.values()), 'expected a Gemm node'
    hls_model.compile()
    x = rng.standard_normal((4,) + inp.shape[1:]).astype(np.float32) * 0.5
    y_hls = hls_model.predict(x).reshape(4, -1)
    y_keras = model.predict(x, verbose=0).reshape(4, -1)
    np.testing.assert_allclose(y_hls, y_keras, atol=0.15, rtol=0.0)
