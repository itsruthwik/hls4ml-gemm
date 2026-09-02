"""
Test Conv im2col GEMM integration with stream templates.

This test suite verifies that Conv1D/2D stream templates properly integrate
the im2col GEMM IP path, buffering streaming input and calling the im2col
GEMM functions correctly.
"""

import pytest
import tempfile
import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Conv2D
from tensorflow.keras.models import Model

# Import hls4ml
import hls4ml
from hls4ml.backends.catapult.catapult_backend import CatapultBackend


class TestConv1DIm2colIntegration:
    """Test Conv1D im2col GEMM integration."""

    def test_conv1d_gemm_ip_template_integration_generated_code(self):
        """Verify Conv1D generates correct buffering and im2col call."""
        # Create simple Conv1D model
        input_layer = Input(shape=(10, 3))
        conv = Conv1D(8, kernel_size=3, padding='valid', data_format='channels_last', name='conv1d')(input_layer)
        model = Model(inputs=input_layer, outputs=conv)
        model.compile(optimizer='adam', loss='mse')

        # Configure for GEMM IP. Strategy: GEMM is the opt-in, set per layer name —
        # LayerName entries carry no 'class_name' key, so filtering on one silently
        # disables GEMM and leaves this on the traditional path.
        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['Model']['ReuseFactor'] = 1
        config['LayerName']['conv1d']['Strategy'] = 'GEMM'

        # Generate HLS code
        with tempfile.TemporaryDirectory() as tmpdir:
            hls_model = hls4ml.converters.convert_from_keras_model(
                model,
                hls_config=config,
                output_dir=tmpdir,
                io_type='io_stream',
                backend='Catapult',
            )

            # Guard against silently falling back to the traditional conv path.
            node_types = [type(n).__name__ for n in hls_model.graph.values()]
            assert any('Im2ColGemm' in t for t in node_types), \
                f"GEMM IP path not taken; graph is {node_types}"

            hls_model.write()

            # General conv routes through the fused im2col + GEMM stage wrapper, which
            # holds the im2col call; myproject.cpp defines and invokes it.
            myproject_cpp = os.path.join(tmpdir, 'firmware/myproject.cpp')
            assert os.path.exists(myproject_cpp), "myproject.cpp should exist"

            with open(myproject_cpp, 'r') as f:
                content = f.read()
                assert 'void gemm_conv1d_stage(' in content, "Should define the GEMM stage wrapper"
                assert 'nnet::im2col_1d_gemm_rows<' in content, "Stage should perform im2col"
                assert 'conv_1d_cl' not in content, "GEMM path must not fall back to conv_1d_cl"

    def test_conv1d_gemm_ip_without_gemm_uses_traditional_path(self):
        """Verify Conv1D falls back to the traditional path without Strategy: GEMM."""
        # Create simple Conv1D model
        input_layer = Input(shape=(10, 3))
        conv = Conv1D(8, kernel_size=3, padding='valid', data_format='channels_last')(input_layer)
        model = Model(inputs=input_layer, outputs=conv)
        model.compile(optimizer='adam', loss='mse')

        # Configure without GEMM IP
        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['Model']['Strategy'] = 'io_stream'
        config['Model']['ReuseFactor'] = 1

        # Explicitly disable GEMM IP
        for layer_cfg in config['LayerName'].values():
            if layer_cfg.get('class_name') == 'Conv1D':
                layer_cfg['Strategy'] = 'Latency'

        # Generate HLS code
        with tempfile.TemporaryDirectory() as tmpdir:
            hls_model = hls4ml.converters.convert_from_keras_model(
                model,
                hls_config=config,
                output_dir=tmpdir,
                io_type='io_stream',
                backend='Catapult',
            )
            hls_model.write()
            myproject_file = os.path.join(tmpdir, 'firmware/myproject.h')
            assert os.path.exists(myproject_file), "myproject.h should exist"


class TestConv2DIm2colIntegration:
    """Test Conv2D im2col GEMM integration."""

    def test_conv2d_gemm_ip_template_integration_generated_code(self):
        """Verify Conv2D generates correct buffering and im2col call."""
        # Create simple Conv2D model
        input_layer = Input(shape=(8, 8, 3))
        conv = Conv2D(16, kernel_size=(3, 3), padding='valid', data_format='channels_last', name='conv2d')(input_layer)
        model = Model(inputs=input_layer, outputs=conv)
        model.compile(optimizer='adam', loss='mse')

        # Configure for GEMM IP. Strategy: GEMM is the opt-in, set per layer name —
        # LayerName entries carry no 'class_name' key, so filtering on one silently
        # disables GEMM and leaves this on the traditional path.
        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['Model']['ReuseFactor'] = 1
        config['LayerName']['conv2d']['Strategy'] = 'GEMM'

        # Generate HLS code
        with tempfile.TemporaryDirectory() as tmpdir:
            hls_model = hls4ml.converters.convert_from_keras_model(
                model,
                hls_config=config,
                output_dir=tmpdir,
                io_type='io_stream',
                backend='Catapult',
            )

            # Guard against silently falling back to the traditional conv path.
            node_types = [type(n).__name__ for n in hls_model.graph.values()]
            assert any('Im2ColGemm' in t for t in node_types), \
                f"GEMM IP path not taken; graph is {node_types}"

            hls_model.write()

            # General conv routes through the fused im2col + GEMM stage wrapper, which
            # holds the im2col call; myproject.cpp defines and invokes it.
            myproject_cpp = os.path.join(tmpdir, 'firmware/myproject.cpp')
            assert os.path.exists(myproject_cpp), "myproject.cpp should exist"

            with open(myproject_cpp, 'r') as f:
                content = f.read()
                assert 'void gemm_conv2d_stage(' in content, "Should define the GEMM stage wrapper"
                assert 'nnet::im2col_2d_gemm_rows<' in content, "Stage should perform im2col"
                assert 'conv_2d_cl' not in content, "GEMM path must not fall back to conv_2d_cl"

    def test_conv2d_gemm_ip_without_gemm_uses_traditional_path(self):
        """Verify Conv2D falls back to the traditional path without Strategy: GEMM."""
        # Create simple Conv2D model
        input_layer = Input(shape=(8, 8, 3))
        conv = Conv2D(16, kernel_size=(3, 3), padding='valid', data_format='channels_last')(input_layer)
        model = Model(inputs=input_layer, outputs=conv)
        model.compile(optimizer='adam', loss='mse')

        # Configure without GEMM IP
        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['Model']['Strategy'] = 'io_stream'
        config['Model']['ReuseFactor'] = 1

        # Explicitly disable GEMM IP
        for layer_cfg in config['LayerName'].values():
            if layer_cfg.get('class_name') == 'Conv2D':
                layer_cfg['Strategy'] = 'Latency'

        # Generate HLS code
        with tempfile.TemporaryDirectory() as tmpdir:
            hls_model = hls4ml.converters.convert_from_keras_model(
                model,
                hls_config=config,
                output_dir=tmpdir,
                io_type='io_stream',
                backend='Catapult',
            )
            hls_model.write()

            # Verify it generates successfully without errors
            # (backward compatibility test)
            myproject_file = os.path.join(tmpdir, 'firmware/myproject.h')
            assert os.path.exists(myproject_file), "myproject.h should exist"


class TestConvGemmIpWithPadding:
    """Test Conv GEMM IP with padding and stride."""

    def test_conv1d_gemm_ip_with_padding_generates_code(self):
        """Verify Conv1D with padding generates code (padding should be supported)."""
        # Create Conv1D model with padding
        input_layer = Input(shape=(10, 3))
        conv = Conv1D(8, kernel_size=3, padding='same', data_format='channels_last')(input_layer)
        model = Model(inputs=input_layer, outputs=conv)
        model.compile(optimizer='adam', loss='mse')

        # Configure for GEMM IP
        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['Model']['Strategy'] = 'io_stream'
        config['Model']['ReuseFactor'] = 1

        # Enable GEMM IP for Conv layer
        for layer_cfg in config['LayerName'].values():
            if layer_cfg.get('class_name') == 'Conv1D':
                layer_cfg['Strategy'] = 'GEMM'

        # Generate HLS code - should succeed
        with tempfile.TemporaryDirectory() as tmpdir:
            hls_model = hls4ml.converters.convert_from_keras_model(
                model,
                hls_config=config,
                output_dir=tmpdir,
                io_type='io_stream',
                backend='Catapult',
            )
            hls_model.write()
            myproject_file = os.path.join(tmpdir, 'firmware/myproject.h')
            assert os.path.exists(myproject_file), "myproject.h should exist for Conv1D with padding"

    def test_conv2d_gemm_ip_with_stride_generates_code(self):
        """Verify Conv2D with stride generates code (stride should be supported)."""
        # Create Conv2D model with stride
        input_layer = Input(shape=(8, 8, 3))
        conv = Conv2D(16, kernel_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last')(input_layer)
        model = Model(inputs=input_layer, outputs=conv)
        model.compile(optimizer='adam', loss='mse')

        # Configure for GEMM IP
        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        config['Model']['Strategy'] = 'io_stream'
        config['Model']['ReuseFactor'] = 1

        # Enable GEMM IP for Conv layer
        for layer_cfg in config['LayerName'].values():
            if layer_cfg.get('class_name') == 'Conv2D':
                layer_cfg['Strategy'] = 'GEMM'

        # Generate HLS code - should succeed
        with tempfile.TemporaryDirectory() as tmpdir:
            hls_model = hls4ml.converters.convert_from_keras_model(
                model,
                hls_config=config,
                output_dir=tmpdir,
                io_type='io_stream',
                backend='Catapult',
            )
            hls_model.write()
            myproject_file = os.path.join(tmpdir, 'firmware/myproject.h')
            assert os.path.exists(myproject_file), "myproject.h should exist for Conv2D with stride"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
