"""
Conv GEMM IP tests for general (non-pointwise) convolutions.

Tests codegen and validation for:
1. General Conv1D with various kernel sizes, strides, padding, dilation
2. General Conv2D with various kernel sizes, strides, padding, dilation
3. Im2col GEMM metadata emission
4. Weight transposition for Conv GEMM IP
5. Rejection of unsupported configurations (channels_first, etc.)
"""

from pathlib import Path

import pytest
import tensorflow as tf

import hls4ml

test_root_path = Path(__file__).parent


def _make_general_conv1d_model(input_shape, kernel_size, n_filters, strides=1, padding='same', dilation_rate=1, name='conv1d'):
    """Create a Conv1D model with specified parameters."""
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Conv1D(
            n_filters,
            kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            input_shape=input_shape,
            name=name,
        )
    )
    model.compile(optimizer='adam', loss='mse')
    return model


def _make_general_conv2d_model(input_shape, kernel_size, n_filters, strides=(1, 1), padding='same', dilation_rate=(1, 1), name='conv2d'):
    """Create a Conv2D model with specified parameters."""
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Conv2D(
            n_filters,
            kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            input_shape=input_shape,
            name=name,
        )
    )
    model.compile(optimizer='adam', loss='mse')
    return model


class TestGeneralConv1DGemmIP:
    """Test Conv1D GEMM IP for general (non-pointwise) kernels."""

    def test_conv1d_kernel3_gemm_ip_metadata(self, test_case_id):
        """Conv1D with kernel_size=3 should emit correct GEMM metadata."""
        # Input: [batch, in_width=10, n_chan=3]
        # Kernel: [filt_width=3, n_chan=3, n_filt=8]
        # Output (valid padding): [batch, out_width=8, n_filt=8]
        # (The row/col GEMM IP only supports valid padding.)
        model = _make_general_conv1d_model((10, 3), kernel_size=3, n_filters=8, padding='valid', name='conv1d')

        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['LayerName']['conv1d']['Strategy'] = 'GEMM'

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

        # For Conv1D: M = out_width, K = filt_width * n_chan, N = n_filt
        assert 'static const unsigned gemm_m = 8;' in parameters_text   # out_width with valid padding
        assert 'static const unsigned gemm_k = 9;' in parameters_text   # 3 * 3 = filt_width * n_chan
        assert 'static const unsigned gemm_n = 8;' in parameters_text   # n_filt

    def test_conv1d_stride_rejects_gemm_ip(self, test_case_id):
        """The row/col GEMM IP supports stride=1 only; stride>1 must be rejected."""
        model = _make_general_conv1d_model((12, 4), kernel_size=2, n_filters=6, strides=2, padding='valid', name='conv1d')

        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['LayerName']['conv1d']['Strategy'] = 'GEMM'

        output_dir = test_root_path / test_case_id
        with pytest.raises(ValueError, match='only stride=1'):
            hls4ml.converters.convert_from_keras_model(
                model,
                hls_config=config,
                output_dir=str(output_dir),
                io_type='io_stream',
                backend='Catapult',
            )

    def test_conv1d_channels_first_rejects_gemm_ip(self, test_case_id):
        """Conv1D with channels_first should reject GEMM IP for now."""
        pytest.skip("channels_first Conv1D GEMM support deferred")


class TestGeneralConv2DGemmIP:
    """Test Conv2D GEMM IP for general (non-pointwise) kernels."""

    def test_conv2d_kernel3x3_gemm_ip_metadata(self, test_case_id):
        """Conv2D with kernel_size=(3,3) should emit correct GEMM metadata."""
        # Input: [batch, in_height=8, in_width=8, n_chan=3]
        # Kernel: [filt_height=3, filt_width=3, n_chan=3, n_filt=16]
        # Output (valid padding): [batch, out_height=6, out_width=6, n_filt=16]
        # (The row/col GEMM IP only supports valid padding.)
        model = _make_general_conv2d_model((8, 8, 3), kernel_size=(3, 3), n_filters=16, padding='valid', name='conv2d')

        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['LayerName']['conv2d']['Strategy'] = 'GEMM'

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

        # For Conv2D: M = out_height * out_width, K = filt_height * filt_width * n_chan, N = n_filt
        assert 'static const unsigned gemm_m = 36;' in parameters_text   # 6 * 6
        assert 'static const unsigned gemm_k = 27;' in parameters_text   # 3 * 3 * 3
        assert 'static const unsigned gemm_n = 16;' in parameters_text   # n_filt
        # General conv routes through the fused im2col + GEMM stage. The stage wrapper is
        # named gemm_<layer>_stage and holds the im2col call; myproject only invokes it.
        myproject_text = (output_dir / 'firmware' / 'myproject.cpp').read_text()
        # Names of intermediate variables depend on global layer indices, which shift with
        # test ordering — assert on the stage wrapper itself, not on its argument names.
        assert 'void gemm_conv2d_stage(' in myproject_text
        assert myproject_text.count('gemm_conv2d_stage(') >= 2, 'stage should be defined and invoked'
        assert 'nnet::im2col_2d_gemm_rows<' in myproject_text

    def test_conv2d_stride_rejects_gemm_ip(self, test_case_id):
        """The row/col GEMM IP supports stride=1 only; stride>1 must be rejected."""
        model = _make_general_conv2d_model((8, 8, 3), kernel_size=(2, 2), n_filters=8, strides=(2, 2), padding='valid', name='conv2d')

        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['LayerName']['conv2d']['Strategy'] = 'GEMM'

        output_dir = test_root_path / test_case_id
        with pytest.raises(ValueError, match='only stride=1'):
            hls4ml.converters.convert_from_keras_model(
                model,
                hls_config=config,
                output_dir=str(output_dir),
                io_type='io_stream',
                backend='Catapult',
            )

    def test_conv2d_rectangular_kernel_gemm_ip_metadata(self, test_case_id):
        """Conv2D with rectangular kernel should emit correct GEMM metadata."""
        # Input: [batch, in_height=7, in_width=10, n_chan=4]
        # Kernel: [filt_height=2, filt_width=3, n_chan=4, n_filt=12]
        # Output (valid padding): [batch, out_height=6, out_width=8, n_filt=12]
        model = _make_general_conv2d_model((7, 10, 4), kernel_size=(2, 3), n_filters=12, padding='valid', name='conv2d')

        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['LayerName']['conv2d']['Strategy'] = 'GEMM'

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

        # M = 6 * 8 = 48
        assert 'static const unsigned gemm_m = 48;' in parameters_text
        assert 'static const unsigned gemm_k = 24;' in parameters_text   # 2 * 3 * 4
        assert 'static const unsigned gemm_n = 12;' in parameters_text


class TestConvGemmWeightTransposition:
    """Test that Conv weights are correctly transposed for GEMM IP."""

    def test_conv1d_weight_transposition_codegen(self, test_case_id):
        """Verify Conv1D weights are transposed for GEMM IP."""
        model = _make_general_conv1d_model((5, 2), kernel_size=2, n_filters=3, padding='valid', name='conv1d')

        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['LayerName']['conv1d']['Strategy'] = 'GEMM'

        output_dir = test_root_path / test_case_id
        hls_model = hls4ml.converters.convert_from_keras_model(
            model,
            hls_config=config,
            output_dir=str(output_dir),
            io_type='io_stream',
            backend='Catapult',
        )

        hls_model.write()

        # Verify weight file exists
        weight_files = list((output_dir / 'firmware' / 'weights').glob('w*.txt'))
        assert len(weight_files) > 0, "Weight file should be generated"


class TestConvGemmValidation:
    """Test Conv GEMM IP validation and error handling."""

    def test_conv1d_io_parallel_routes_to_array_gemm(self, test_case_id):
        """Conv1D GEMM IP under io_parallel takes the weightless ARRAY path (no ac_channel)."""
        model = _make_general_conv1d_model((10, 3), kernel_size=3, n_filters=8, padding='valid', name='conv1d')

        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['LayerName']['conv1d']['Strategy'] = 'GEMM'

        output_dir = test_root_path / test_case_id
        hls_model = hls4ml.converters.convert_from_keras_model(
            model,
            hls_config=config,
            output_dir=str(output_dir),
            io_type='io_parallel',
            backend='Catapult',
        )
        hls_model.write()

        myproject_text = (output_dir / 'firmware/myproject.cpp').read_text()
        assert 'nnet::gemm_array_weightless<' in myproject_text
        assert 'nnet::im2col_1d_gemm_rows_array<' in myproject_text
        # io_parallel must not take the streaming entry
        assert 'nnet::gemm_stream_weightless<' not in myproject_text
        assert 'ac_channel' not in myproject_text

    def test_conv2d_io_parallel_routes_to_array_gemm(self, test_case_id):
        """Conv2D GEMM IP under io_parallel takes the weightless ARRAY path (no ac_channel)."""
        model = _make_general_conv2d_model((8, 8, 3), kernel_size=(3, 3), n_filters=16, padding='valid', name='conv2d')

        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['LayerName']['conv2d']['Strategy'] = 'GEMM'

        output_dir = test_root_path / test_case_id
        hls_model = hls4ml.converters.convert_from_keras_model(
            model,
            hls_config=config,
            output_dir=str(output_dir),
            io_type='io_parallel',
            backend='Catapult',
        )
        hls_model.write()

        myproject_text = (output_dir / 'firmware/myproject.cpp').read_text()
        assert 'nnet::gemm_array_weightless<' in myproject_text
        assert 'nnet::im2col_2d_gemm_rows_array<' in myproject_text
        assert 'nnet::gemm_stream_weightless<' not in myproject_text


class TestConvGemmPacketStructure:
    """Test Conv GEMM IP packet width and count validation."""

    def test_conv1d_packet_structure_codegen(self, test_case_id):
        """Verify Conv1D generates correct packet structures for im2col rows and output."""
        # Conv1D: M=out_width, K=kernel*channels, N=filters
        # Input packets should be K-wide, output packets N-wide
        model = _make_general_conv1d_model((10, 3), kernel_size=3, n_filters=8, padding='valid', name='conv1d')

        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['LayerName']['conv1d']['Strategy'] = 'GEMM'

        output_dir = test_root_path / test_case_id
        hls_model = hls4ml.converters.convert_from_keras_model(
            model,
            hls_config=config,
            output_dir=str(output_dir),
            io_type='io_stream',
            backend='Catapult',
        )
        hls_model.write()

        # Check parameters.h for packet dimensions
        params_file = output_dir / 'firmware/parameters.h'
        assert params_file.exists(), "parameters.h should exist"

        with open(params_file, 'r') as f:
            content = f.read()
            # K should be kernel_size * n_chan = 3 * 3 = 9
            assert 'gemm_k = 9' in content or 'gemm_k=9' in content, "Conv1D gemm_k should be kernel*n_chan"
            # N should be n_filters = 8
            assert 'gemm_n = 8' in content or 'gemm_n=8' in content, "Conv1D gemm_n should be n_filters"

    def test_conv2d_packet_structure_codegen(self, test_case_id):
        """Verify Conv2D generates correct packet structures for im2col rows and output."""
        # Conv2D: M=out_height*out_width, K=kernel_h*kernel_w*channels, N=filters
        # Input packets (im2col rows) should be K-wide, output packets N-wide
        model = _make_general_conv2d_model((8, 8, 3), kernel_size=(3, 3), n_filters=16, padding='valid', name='conv2d')

        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['LayerName']['conv2d']['Strategy'] = 'GEMM'

        output_dir = test_root_path / test_case_id
        hls_model = hls4ml.converters.convert_from_keras_model(
            model,
            hls_config=config,
            output_dir=str(output_dir),
            io_type='io_stream',
            backend='Catapult',
        )
        hls_model.write()

        # Check parameters.h for packet dimensions
        params_file = output_dir / 'firmware/parameters.h'
        assert params_file.exists(), "parameters.h should exist"

        with open(params_file, 'r') as f:
            content = f.read()
            # K should be kernel_h * kernel_w * n_chan = 3 * 3 * 3 = 27
            assert 'gemm_k = 27' in content or 'gemm_k=27' in content, "Conv2D gemm_k should be kernel_h*kernel_w*n_chan"
            # N should be n_filters = 16
            assert 'gemm_n = 16' in content or 'gemm_n=16' in content, "Conv2D gemm_n should be n_filters"


class TestConvGemmValidationErrors:
    """Test Conv GEMM IP validation and error handling."""

    def test_conv1d_channels_first_rejects_gemm_ip_clearly(self, test_case_id):
        """Verify Conv1D with channels_first data format rejects GEMM IP with clear error."""
        # Create a simple Conv1D model with channels_first via Sequential API
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv1D(8, kernel_size=3, padding='valid', 
                                          data_format='channels_first', 
                                          input_shape=(3, 10), name='conv1d'))

        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['LayerName']['conv1d']['Strategy'] = 'GEMM'

        output_dir = test_root_path / test_case_id
        # Should fail validation due to channels_first
        with pytest.raises(ValueError, match='channels_last|data format|unsupported'):
            hls4ml.converters.convert_from_keras_model(
                model,
                hls_config=config,
                output_dir=str(output_dir),
                io_type='io_stream',
                backend='Catapult',
            )

    def test_conv2d_channels_first_rejects_gemm_ip_clearly(self, test_case_id):
        """Verify Conv2D with channels_first data format rejects GEMM IP with clear error."""
        # Create a simple Conv2D model with channels_first via Sequential API
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding='valid',
                                          data_format='channels_first',
                                          input_shape=(3, 8, 8), name='conv2d'))

        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['LayerName']['conv2d']['Strategy'] = 'GEMM'

        output_dir = test_root_path / test_case_id
        # Should fail validation due to channels_first
        with pytest.raises(ValueError, match='channels_last|data format|unsupported'):
            hls4ml.converters.convert_from_keras_model(
                model,
                hls_config=config,
                output_dir=str(output_dir),
                io_type='io_stream',
                backend='Catapult',
            )

    def test_conv1d_with_various_valid_configs_succeeds(self, test_case_id):
        """Verify various valid Conv1D configs with GEMM IP generate code successfully."""
        # Test multiple valid configurations
        # The row/col GEMM IP supports valid padding and stride=1 only.
        configs_to_test = [
            {'kernel_size': 1, 'padding': 'valid', 'strides': 1},
            {'kernel_size': 3, 'padding': 'valid', 'strides': 1},
            {'kernel_size': 5, 'padding': 'valid', 'strides': 1},
        ]

        for cfg in configs_to_test:
            model = _make_general_conv1d_model((10, 3), kernel_size=cfg['kernel_size'], 
                                              n_filters=8, padding=cfg['padding'],
                                              strides=cfg['strides'], name='conv1d')

            config = hls4ml.utils.config_from_keras_model(model, granularity='name')
            config['LayerName']['conv1d']['Strategy'] = 'GEMM'

            output_dir = test_root_path / test_case_id / f"k{cfg['kernel_size']}_p{cfg['padding']}_s{cfg['strides']}"
            # Should succeed for all valid configs
            hls_model = hls4ml.converters.convert_from_keras_model(
                model,
                hls_config=config,
                output_dir=str(output_dir),
                io_type='io_stream',
                backend='Catapult',
            )
            hls_model.write()

            # Verify myproject.h was created
            assert (output_dir / 'firmware/myproject.h').exists(), f"Should generate code for Conv1D config {cfg}"

    def test_conv1d_dilation_rejects_gemm_ip(self, test_case_id):
        """Conv1D GEMM IP should reject non-unit dilation until the stream path supports it."""
        model = _make_general_conv1d_model((10, 3), kernel_size=3, n_filters=8, padding='valid', dilation_rate=2, name='conv1d')

        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['LayerName']['conv1d']['Strategy'] = 'GEMM'

        output_dir = test_root_path / test_case_id
        with pytest.raises(ValueError, match=r'dilation > 1|unit dilation'):
            hls4ml.converters.convert_from_keras_model(
                model,
                hls_config=config,
                output_dir=str(output_dir),
                io_type='io_stream',
                backend='Catapult',
            )

    def test_conv2d_with_various_valid_configs_succeeds(self, test_case_id):
        """Verify various valid Conv2D configs with GEMM IP generate code successfully."""
        # Test multiple valid configurations
        # The row/col GEMM IP supports valid padding and stride=1 only.
        configs_to_test = [
            {'kernel_size': (1, 1), 'padding': 'valid', 'strides': (1, 1)},
            {'kernel_size': (3, 3), 'padding': 'valid', 'strides': (1, 1)},
            {'kernel_size': (2, 3), 'padding': 'valid', 'strides': (1, 1)},  # rectangular kernel
        ]

        for cfg in configs_to_test:
            model = _make_general_conv2d_model((8, 8, 3), kernel_size=cfg['kernel_size'],
                                              n_filters=16, padding=cfg['padding'],
                                              strides=cfg['strides'], name='conv2d')

            config = hls4ml.utils.config_from_keras_model(model, granularity='name')
            config['LayerName']['conv2d']['Strategy'] = 'GEMM'

            k_str = f"k{cfg['kernel_size'][0]}x{cfg['kernel_size'][1]}"
            s_str = f"s{cfg['strides'][0]}x{cfg['strides'][1]}"
            output_dir = test_root_path / test_case_id / f"{k_str}_p{cfg['padding']}_{s_str}"
            # Should succeed for all valid configs
            hls_model = hls4ml.converters.convert_from_keras_model(
                model,
                hls_config=config,
                output_dir=str(output_dir),
                io_type='io_stream',
                backend='Catapult',
            )
            hls_model.write()

            # Verify myproject.h was created
            assert (output_dir / 'firmware/myproject.h').exists(), f"Should generate code for Conv2D config {cfg}"

    def test_conv2d_dilation_rejects_gemm_ip(self, test_case_id):
        """Conv2D GEMM IP should reject non-unit dilation until the stream path supports it."""
        model = _make_general_conv2d_model(
            (8, 8, 3), kernel_size=(3, 3), n_filters=16, padding='valid', dilation_rate=(2, 2), name='conv2d'
        )

        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['LayerName']['conv2d']['Strategy'] = 'GEMM'

        output_dir = test_root_path / test_case_id
        with pytest.raises(ValueError, match=r'dilation > 1|unit dilation'):
            hls4ml.converters.convert_from_keras_model(
                model,
                hls_config=config,
                output_dir=str(output_dir),
                io_type='io_stream',
                backend='Catapult',
            )


class TestPointwiseConvGemmIP:
    """Test pointwise (kernel_size=1) Conv GEMM IP with padding and stride support."""

    def test_pointwise_conv1d_valid_padding_stride1_codegen(self, test_case_id):
        """Pointwise Conv1D with valid padding, stride=1 should generate correct GEMM metadata."""
        # Pointwise: kernel_size=1, padding='valid', stride=1
        # M = out_width, K = 1 * n_chan = n_chan, N = n_filt
        model = _make_general_conv1d_model((10, 3), kernel_size=1, n_filters=8, 
                                          padding='valid', strides=1, name='conv1d')

        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['LayerName']['conv1d']['Strategy'] = 'GEMM'

        output_dir = test_root_path / test_case_id
        hls_model = hls4ml.converters.convert_from_keras_model(
            model,
            hls_config=config,
            output_dir=str(output_dir),
            io_type='io_stream',
            backend='Catapult',
        )
        hls_model.write()

        # Verify metadata: for pointwise, K = n_chan
        params_file = output_dir / 'firmware/parameters.h'
        assert params_file.exists(), "parameters.h should exist"
        with open(params_file, 'r') as f:
            content = f.read()
            # K should be n_chan for pointwise (kernel_size * n_chan = 1 * 3 = 3)
            assert 'gemm_k = 3' in content or 'gemm_k=3' in content, "Pointwise Conv1D gemm_k should be n_chan"
            # N should be n_filt = 8
            assert 'gemm_n = 8' in content or 'gemm_n=8' in content, "Pointwise Conv1D gemm_n should be n_filt"

        # Pointwise conv routes directly through the Gemm stage.
        myproject_text = (output_dir / 'firmware' / 'myproject.cpp').read_text()
        assert 'nnet::gemm_stream_weightless<' in myproject_text

    def test_pointwise_conv1d_same_padding_stride1_codegen(self, test_case_id):
        """Pointwise Conv1D with same padding, stride=1 should preserve width in GEMM metadata."""
        # Pointwise with same padding: M = out_width (same as in_width), K = n_chan, N = n_filt
        model = _make_general_conv1d_model((10, 3), kernel_size=1, n_filters=8, 
                                          padding='same', strides=1, name='conv1d')

        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['LayerName']['conv1d']['Strategy'] = 'GEMM'

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
        params_file = output_dir / 'firmware/parameters.h'
        assert params_file.exists(), "parameters.h should exist"
        with open(params_file, 'r') as f:
            content = f.read()
            assert 'gemm_k = 3' in content or 'gemm_k=3' in content, "Pointwise Conv1D gemm_k should be n_chan"
            assert 'gemm_n = 8' in content or 'gemm_n=8' in content, "Pointwise Conv1D gemm_n should be n_filt"

    def test_pointwise_conv1d_stride_rejects_gemm_ip(self, test_case_id):
        """The row/col GEMM IP supports stride=1 only, including for pointwise conv."""
        model = _make_general_conv1d_model((10, 3), kernel_size=1, n_filters=8,
                                          padding='valid', strides=2, name='conv1d')

        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['LayerName']['conv1d']['Strategy'] = 'GEMM'

        output_dir = test_root_path / test_case_id
        with pytest.raises(ValueError, match='only stride=1'):
            hls4ml.converters.convert_from_keras_model(
                model,
                hls_config=config,
                output_dir=str(output_dir),
                io_type='io_stream',
                backend='Catapult',
            )

    def test_pointwise_conv2d_valid_padding_stride1_codegen(self, test_case_id):
        """Pointwise Conv2D with valid padding, stride=1 should generate correct GEMM metadata."""
        # Pointwise Conv2D: kernel_size=(1,1), padding='valid', stride=1
        # M = out_height * out_width, K = 1 * 1 * n_chan = n_chan, N = n_filt
        model = _make_general_conv2d_model((8, 8, 3), kernel_size=(1, 1), n_filters=16, 
                                          padding='valid', strides=(1, 1), name='conv2d')

        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['LayerName']['conv2d']['Strategy'] = 'GEMM'

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
        params_file = output_dir / 'firmware/parameters.h'
        assert params_file.exists(), "parameters.h should exist"
        with open(params_file, 'r') as f:
            content = f.read()
            # K should be n_chan for pointwise (kernel_h * kernel_w * n_chan = 1 * 1 * 3 = 3)
            assert 'gemm_k = 3' in content or 'gemm_k=3' in content, "Pointwise Conv2D gemm_k should be n_chan"
            # N should be n_filt = 16
            assert 'gemm_n = 16' in content or 'gemm_n=16' in content, "Pointwise Conv2D gemm_n should be n_filt"

    def test_pointwise_conv2d_same_padding_codegen(self, test_case_id):
        """Pointwise Conv2D with same padding should preserve spatial dimensions in GEMM metadata."""
        # Pointwise Conv2D with same padding: M = out_height * out_width (same as in), K = n_chan, N = n_filt
        model = _make_general_conv2d_model((8, 8, 3), kernel_size=(1, 1), n_filters=16, 
                                          padding='same', strides=(1, 1), name='conv2d')

        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['LayerName']['conv2d']['Strategy'] = 'GEMM'

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
        params_file = output_dir / 'firmware/parameters.h'
        assert params_file.exists(), "parameters.h should exist"
        with open(params_file, 'r') as f:
            content = f.read()
            # M = out_height * out_width = 8 * 8 = 64
            assert 'gemm_m = 64' in content or 'gemm_m=64' in content, "Pointwise Conv2D with same padding should have M=64"
            assert 'gemm_k = 3' in content or 'gemm_k=3' in content, "Pointwise Conv2D gemm_k should be n_chan"

    def test_pointwise_conv2d_stride_rejects_gemm_ip(self, test_case_id):
        """The row/col GEMM IP supports stride=1 only, including for pointwise conv."""
        model = _make_general_conv2d_model((8, 8, 3), kernel_size=(1, 1), n_filters=16,
                                          padding='valid', strides=(2, 2), name='conv2d')

        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['LayerName']['conv2d']['Strategy'] = 'GEMM'

        output_dir = test_root_path / test_case_id
        with pytest.raises(ValueError, match='only stride=1'):
            hls4ml.converters.convert_from_keras_model(
                model,
                hls_config=config,
                output_dir=str(output_dir),
                io_type='io_stream',
                backend='Catapult',
            )

    def test_pointwise_conv1d_stride_matrix_validated(self, test_case_id):
        """Pointwise Conv1D: stride=1 generates code; stride>1 is rejected.

        kernel=1 'same' padding adds no pad, so it is accepted at stride=1.
        """
        configs_to_test = [
            {'stride': 1, 'padding': 'valid', 'ok': True},
            {'stride': 1, 'padding': 'same', 'ok': True},
            {'stride': 2, 'padding': 'valid', 'ok': False},
            {'stride': 3, 'padding': 'valid', 'ok': False},
        ]

        for i, cfg in enumerate(configs_to_test):
            model = _make_general_conv1d_model((10, 3), kernel_size=1, n_filters=8,
                                              padding=cfg['padding'], strides=cfg['stride'], name='conv1d')

            config = hls4ml.utils.config_from_keras_model(model, granularity='name')
            config['LayerName']['conv1d']['Strategy'] = 'GEMM'

            output_dir = test_root_path / test_case_id / f"s{cfg['stride']}_p{cfg['padding']}"
            if cfg['ok']:
                hls_model = hls4ml.converters.convert_from_keras_model(
                    model,
                    hls_config=config,
                    output_dir=str(output_dir),
                    io_type='io_stream',
                    backend='Catapult',
                )
                hls_model.write()
                assert (output_dir / 'firmware/myproject.h').exists()
            else:
                with pytest.raises(ValueError, match='only stride=1'):
                    hls4ml.converters.convert_from_keras_model(
                        model,
                        hls_config=config,
                        output_dir=str(output_dir),
                        io_type='io_stream',
                        backend='Catapult',
                    )

    def test_pointwise_conv2d_stride_matrix_validated(self, test_case_id):
        """Pointwise Conv2D: stride=(1,1) generates code; any stride>1 is rejected."""
        configs_to_test = [
            {'stride': (1, 1), 'padding': 'valid', 'ok': True},
            {'stride': (1, 1), 'padding': 'same', 'ok': True},
            {'stride': (2, 2), 'padding': 'valid', 'ok': False},
            {'stride': (1, 2), 'padding': 'valid', 'ok': False},
            {'stride': (2, 1), 'padding': 'valid', 'ok': False},
        ]

        for i, cfg in enumerate(configs_to_test):
            model = _make_general_conv2d_model((8, 8, 3), kernel_size=(1, 1), n_filters=16,
                                              padding=cfg['padding'], strides=cfg['stride'], name='conv2d')

            config = hls4ml.utils.config_from_keras_model(model, granularity='name')
            config['LayerName']['conv2d']['Strategy'] = 'GEMM'

            s_str = f"s{cfg['stride'][0]}x{cfg['stride'][1]}"
            output_dir = test_root_path / test_case_id / f"{s_str}_p{cfg['padding']}"
            if cfg['ok']:
                hls_model = hls4ml.converters.convert_from_keras_model(
                    model,
                    hls_config=config,
                    output_dir=str(output_dir),
                    io_type='io_stream',
                    backend='Catapult',
                )
                hls_model.write()
                assert (output_dir / 'firmware/myproject.h').exists()
            else:
                with pytest.raises(ValueError, match='only stride=1'):
                    hls4ml.converters.convert_from_keras_model(
                        model,
                        hls_config=config,
                        output_dir=str(output_dir),
                        io_type='io_stream',
                        backend='Catapult',
                    )
