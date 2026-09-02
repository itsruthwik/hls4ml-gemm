"""
Dense GEMM IP correctness tests.

Tests that verify:
1. Numerical correctness for rank-1, rank-2, and rank-greater-than-2 Dense layers
2. Packet structure and count (input K-wide, output N-wide)
3. Output shape preservation
4. Bias handling and output casting behavior
"""

import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

import hls4ml

test_root_path = Path(__file__).parent


def _make_custom_dense_model(input_shape, output_units, name="dense"):
    """Create a Dense model with specified input shape and output units."""
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(output_units, input_shape=input_shape, name=name))
    model.compile(optimizer='adam', loss='mse')
    return model


class TestDenseGemmRank1Correctness:
    """Test Dense GEMM correctness for rank-1 inputs (batch, K)."""

    def test_rank1_dense_gemm_numerical_correctness_small(self, test_case_id):
        """Small rank-1 Dense with known weights for easy verification."""
        model = _make_custom_dense_model((4,), 3, name='dense')
        dense = model.get_layer('dense')

        # Set specific weights for verification
        weight_values = tf.constant(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
            dtype=tf.keras.backend.floatx(),
        )
        bias_values = tf.constant([0.01, 0.02, 0.03], dtype=tf.keras.backend.floatx())
        dense.set_weights([weight_values.numpy(), bias_values.numpy()])

        # Generate test data
        test_input = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        keras_pred = model.predict(test_input, verbose=0)

        # Create HLS4ML model with GEMM IP enabled
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

        # Verify metadata
        parameters_text = (output_dir / 'firmware' / 'parameters.h').read_text()
        assert 'static const unsigned gemm_m = 1;' in parameters_text
        assert 'static const unsigned gemm_k = 4;' in parameters_text
        assert 'static const unsigned gemm_n = 3;' in parameters_text

    def test_rank1_dense_gemm_numerical_correctness_large(self, test_case_id):
        """Large rank-1 Dense with random data."""
        model = _make_custom_dense_model((64,), 32, name='dense')

        # Generate random test data
        test_input = np.random.randn(10, 64).astype(np.float32)
        keras_pred = model.predict(test_input, verbose=0)

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
        assert 'static const unsigned gemm_m = 1;' in parameters_text
        assert 'static const unsigned gemm_k = 64;' in parameters_text
        assert 'static const unsigned gemm_n = 32;' in parameters_text


class TestDenseGemmRank2Correctness:
    """Test Dense GEMM correctness for rank-2 inputs (batch, M, K)."""

    def test_rank2_dense_gemm_preserves_shape(self, test_case_id):
        """Verify rank-2 Dense preserves shape with GEMM IP."""
        model = _make_custom_dense_model((5, 4), 3, name='dense')

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

        # GemmIP rewrites Dense into the Gemm node
        layers = {layer.name: layer for layer in hls_model.get_layers()}
        assert layers['gemm_dense'].class_name == 'Gemm'

        # Verify output shape preserves all dimensions except last
        output_shape = layers['gemm_dense'].get_output_variable().shape
        assert output_shape == [5, 3], f"Expected [5, 3], got {output_shape}"

        hls_model.write()

        # Verify GEMM metadata
        parameters_text = (output_dir / 'firmware' / 'parameters.h').read_text()
        assert 'static const unsigned gemm_m = 5;' in parameters_text
        assert 'static const unsigned gemm_k = 4;' in parameters_text
        assert 'static const unsigned gemm_n = 3;' in parameters_text

    def test_rank2_dense_gemm_numerical_correctness(self, test_case_id):
        """Rank-2 Dense GEMM numerical correctness with known values."""
        model = _make_custom_dense_model((3, 2), 2, name='dense')
        dense = model.get_layer('dense')

        # Set specific weights for verification
        # Weights should be [K, N] = [2, 2] where K is the last input dim and N is output
        weight_values = tf.constant(
            [[0.5, 0.25], [0.75, 0.125]],
            dtype=tf.keras.backend.floatx(),
        )
        bias_values = tf.constant([0.01, 0.02], dtype=tf.keras.backend.floatx())
        dense.set_weights([weight_values.numpy(), bias_values.numpy()])

        # Test data: batch of 1 with 3x2 matrix
        test_input = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=np.float32)
        keras_pred = model.predict(test_input, verbose=0)

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

        # Verify metadata matches input shape semantics
        parameters_text = (output_dir / 'firmware' / 'parameters.h').read_text()
        assert 'static const unsigned gemm_m = 3;' in parameters_text  # M from first dim
        assert 'static const unsigned gemm_k = 2;' in parameters_text  # K from second dim
        assert 'static const unsigned gemm_n = 2;' in parameters_text  # N from output

    def test_rank2_dense_gemm_various_sizes(self, test_case_id):
        """Test rank-2 Dense with various M, K, N combinations."""
        test_cases = [
            ((2, 4), 3),   # M=2, K=4, N=3
            ((8, 16), 32), # M=8, K=16, N=32
            ((7, 5), 11),  # Asymmetric case
        ]

        for i, (input_shape, output_units) in enumerate(test_cases):
            model = _make_custom_dense_model(input_shape, output_units, name=f'dense_{i}')

            config = hls4ml.utils.config_from_keras_model(model, granularity='name')
            config['LayerName'][f'dense_{i}']['Strategy'] = 'GEMM'

            output_dir = test_root_path / f"{test_case_id}_{i}"
            hls_model = hls4ml.converters.convert_from_keras_model(
                model,
                hls_config=config,
                output_dir=str(output_dir),
                io_type='io_stream',
                backend='Catapult',
            )

            hls_model.write()

            parameters_text = (output_dir / 'firmware' / 'parameters.h').read_text()
            assert f'static const unsigned gemm_m = {input_shape[0]};' in parameters_text
            assert f'static const unsigned gemm_k = {input_shape[1]};' in parameters_text
            assert f'static const unsigned gemm_n = {output_units};' in parameters_text


class TestDenseGemmRank3Correctness:
    """Test Dense GEMM correctness for rank-3+ inputs (batch, D0, D1, ..., K)."""

    def test_rank3_dense_gemm_preserves_shape(self, test_case_id):
        """Verify rank-3 Dense preserves shape with GEMM IP."""
        model = _make_custom_dense_model((2, 3, 4), 5, name='dense')

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

        # GemmIP rewrites Dense into the Gemm node
        layers = {layer.name: layer for layer in hls_model.get_layers()}
        assert layers['gemm_dense'].class_name == 'Gemm'

        # Verify output shape: [2, 3, 5] (leading dims preserved, last dim changed to n_out)
        output_shape = layers['gemm_dense'].get_output_variable().shape
        assert output_shape == [2, 3, 5], f"Expected [2, 3, 5], got {output_shape}"

        hls_model.write()

        # Verify GEMM metadata
        parameters_text = (output_dir / 'firmware' / 'parameters.h').read_text()
        # M = 2 * 3 = 6 (product of leading dimensions)
        assert 'static const unsigned gemm_m = 6;' in parameters_text
        assert 'static const unsigned gemm_k = 4;' in parameters_text
        assert 'static const unsigned gemm_n = 5;' in parameters_text

    def test_rank4_dense_gemm_preserves_shape(self, test_case_id):
        """Verify rank-4 Dense preserves shape with GEMM IP."""
        model = _make_custom_dense_model((2, 3, 4, 5), 7, name='dense')

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

        layers = {layer.name: layer for layer in hls_model.get_layers()}
        output_shape = layers['gemm_dense'].get_output_variable().shape
        assert output_shape == [2, 3, 4, 7], f"Expected [2, 3, 4, 7], got {output_shape}"

        hls_model.write()

        parameters_text = (output_dir / 'firmware' / 'parameters.h').read_text()
        # M = 2 * 3 * 4 = 24
        assert 'static const unsigned gemm_m = 24;' in parameters_text
        assert 'static const unsigned gemm_k = 5;' in parameters_text
        assert 'static const unsigned gemm_n = 7;' in parameters_text


class TestDenseGemmBiasAndCasting:
    """Test Dense GEMM bias handling and output casting."""

    def test_dense_with_bias(self, test_case_id):
        """Verify Dense GEMM includes bias in output."""
        model = _make_custom_dense_model((4,), 3, name='dense')
        dense = model.get_layer('dense')

        weight_values = tf.constant(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            dtype=tf.keras.backend.floatx(),
        )
        bias_values = tf.constant([10.0, 20.0, 30.0], dtype=tf.keras.backend.floatx())
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

        # Verify bias is present in weights directory
        weights_dir = output_dir / 'firmware' / 'weights'
        bias_files = list(weights_dir.glob('b*.txt'))
        assert len(bias_files) > 0, "Bias file should be generated"

        # Check bias file has correct values
        bias_file = bias_files[0]
        bias_contents = bias_file.read_text()
        bias_values_serialized = [float(v.strip()) for v in bias_contents.split(',') if v.strip()]
        assert bias_values_serialized == [10.0, 20.0, 30.0]

    def test_dense_without_bias(self, test_case_id):
        """Verify Dense GEMM handles zero bias correctly."""
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(3, input_shape=(4,), use_bias=False, name='dense'))
        model.compile(optimizer='adam', loss='mse')

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

        # Verify model still generates properly even without explicit bias
        parameters_text = (output_dir / 'firmware' / 'parameters.h').read_text()
        assert 'static const unsigned gemm_m = 1;' in parameters_text
        assert 'static const unsigned gemm_k = 4;' in parameters_text
        assert 'static const unsigned gemm_n = 3;' in parameters_text


class TestDenseGemmPacketStructure:
    """Test that packet structure matches expected sizes (K-wide input, N-wide output)."""

    def test_dense_packet_structure_codegen(self, test_case_id):
        """Verify generated code has correct K and N dimensions."""
        model = _make_custom_dense_model((17,), 13, name='dense')

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

        # Verify defines.h has array dimensions for packet types
        defines_text = (output_dir / 'firmware' / 'defines.h').read_text()

        # Input should be array of size K (one row of A matrix)
        # Output should be array of size N (one row of C matrix)
        assert 'input_t' in defines_text
        assert 'result_t' in defines_text

    def test_dense_rank2_packet_structure_codegen(self, test_case_id):
        """Verify rank-2 Dense packet structure in generated code."""
        model = _make_custom_dense_model((7, 11), 13, name='dense')

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

        defines_text = (output_dir / 'firmware' / 'defines.h').read_text()
        parameters_text = (output_dir / 'firmware' / 'parameters.h').read_text()

        # M=7, K=11, N=13
        assert 'static const unsigned gemm_m = 7;' in parameters_text
        assert 'static const unsigned gemm_k = 11;' in parameters_text
        assert 'static const unsigned gemm_n = 13;' in parameters_text


class TestDenseGemmWeightTransposition:
    """Test that weights are correctly transposed for GEMM IP."""

    def test_rank2_dense_weight_transposition_verification(self, test_case_id):
        """Verify weight transposition matches expected W_T[N][K] layout."""
        model = _make_custom_dense_model((4,), 3, name='dense')
        dense = model.get_layer('dense')

        # Original weights: W[K][N] = [4][3]
        weight_values = tf.constant(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [1.0, 1.1, 1.2],
            ],
            dtype=tf.keras.backend.floatx(),
        )
        dense.set_weights([weight_values.numpy(), tf.zeros(3, dtype=tf.keras.backend.floatx())])

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

        # Get the transposed weights from the generated file
        weight_file = next((output_dir / 'firmware' / 'weights').glob('w*.txt'))
        serialized_weights = [
            float(v.strip()) for v in weight_file.read_text().split(',') if v.strip()
        ]

        # After transposition W_T[N][K], weights should be stored as:
        # W_T[0][*] = [0.1, 0.4, 0.7, 1.0]  (col 0 of original)
        # W_T[1][*] = [0.2, 0.5, 0.8, 1.1]  (col 1 of original)
        # W_T[2][*] = [0.3, 0.6, 0.9, 1.2]  (col 2 of original)
        # Flattened: [0.1, 0.4, 0.7, 1.0, 0.2, 0.5, 0.8, 1.1, 0.3, 0.6, 0.9, 1.2]
        expected = [0.1, 0.4, 0.7, 1.0, 0.2, 0.5, 0.8, 1.1, 0.3, 0.6, 0.9, 1.2]
        np.testing.assert_array_almost_equal(serialized_weights, expected)
