"""Shared GEMM-IP packed weight-column writer.

Used by both the Catapult and Vivado/Vitis writers so the packing logic cannot
silently diverge between backends.
"""

import numpy as np

from hls4ml.model.layers import EinsumDense


def write_gemm_ip_weight_cols(var, layer, odir):
    """Write GEMM-IP packed weight columns.

    The scalar weight header remains the source of truth for native and
    simulation paths. This extra header is consumed only by GEMM-IP
    synthesis calls to avoid repacking flat scalar weights in hardware.

    Weight storage is column-major: ``w_gemm_cols[gemm_n]`` where each
    entry is an ``array<weight_t, gemm_k>`` holding one column of the
    [K, N] weight matrix.
    """

    gemm_k = layer.get_attr('gemm_k', layer.get_attr('n_in'))
    gemm_n = layer.get_attr('gemm_n', layer.get_attr('n_out'))
    beat_name = f'{var.name}_gemm_cols'
    guard = f'{beat_name.upper()}_H_'

    weight_data = np.asarray(var.data)

    # EinsumDense stores its kernel as [n_inplace, n_contract, n_free_kernel].
    # The gemm_cols header holds a single [K, N] block, so peel a leading
    # unit in-place dimension; refuse anything we cannot represent rather
    # than falling through to a layout guess.
    if weight_data.ndim == 3:
        if weight_data.shape[0] != 1:
            raise NotImplementedError(
                f'{var.name}: GEMM-IP weight columns with n_inplace={weight_data.shape[0]} '
                'are not supported (one [K, N] block per header).'
            )
        weight_data = weight_data[0]

    with open(f'{odir}/firmware/weights/{beat_name}.h', 'w') as h_file:
        h_file.write(f'// Packed GEMM-IP weight columns for {var.name}\n')
        h_file.write(f'// Source numpy array shape {var.shape}\n\n')
        h_file.write(f'#ifndef {guard}\n')
        h_file.write(f'#define {guard}\n\n')
        # File-relative include: resolves from firmware/weights/ on every
        # backend regardless of the build's -I flags (the Vivado csim has
        # none that reach nnet_utils, and the bare "nnet_utils/..." form only
        # worked on Catapult via the copy shipped inside MGC_HOME).
        h_file.write('#include "../nnet_utils/nnet_types.h"\n\n')
        h_file.write(f'static nnet::array<{var.type.name}, {gemm_k}> {beat_name}[{gemm_n}] = {{')

        # Column-major packing: w_gemm_cols[n][k] = the weight multiplying
        # input k for output n. The SOURCE layout is decided by the layer type,
        # never sniffed from the shape — a square kernel is ambiguous and the
        # old shape guess silently packed transposed weights for square
        # EinsumDense kernels (e.g. MHA projections).
        #   - EinsumDense kernels are [n_contract, n_free_kernel] = [K, N];
        #     TransposeWeightsForGemmIP does not touch them.
        #   - Dense/conv kernels reach the writer transposed/flattened to
        #     [N, K] by TransposeWeightsForGemmIP.
        # EinsumDense keeps its [K, N] kernel whether it is still an EinsumDense layer
        # or has been lowered to a Gemm node (LowerEinsumToGemm sets _original_type).
        kn_source = isinstance(layer, EinsumDense) or layer.get_attr('_original_type') == 'EinsumDense'
        if kn_source and weight_data.shape != (gemm_k, gemm_n):
            raise NotImplementedError(
                f'{var.name}: EinsumDense GEMM-IP weights expected [K, N] = '
                f'({gemm_k}, {gemm_n}), got {weight_data.shape}.'
            )
        sep = ''
        for n in range(gemm_n):
            h_file.write(sep + '{')
            col_sep = ''
            for k in range(gemm_k):
                if kn_source:
                    value = weight_data[k, n]
                elif weight_data.shape == (gemm_n, gemm_k):
                    value = weight_data[n, k]
                else:
                    value = weight_data.reshape(-1)[n * gemm_k + k]
                h_file.write(col_sep + var.precision_fmt.format(value))
                col_sep = ', '
            h_file.write('}')
            sep = ', '
        h_file.write('};\n\n')
        h_file.write('#endif\n')


def _fixed_point_int(value, precision):
    """Raw fixed-point integer bit-pattern of *value* under *precision*.

    Weight-stationary cells hand the exact quantized bits to the external GEMM IP
    generator with no float re-quantization round-trip (keeps csim/cosim bit-exact,
    per the plan's single-arithmetic-authority decision). For a fixed<W,I> type the
    stored integer is round(value * 2**(W-I)); integer types have 0 fractional bits.
    """
    frac = getattr(precision, 'fractional', 0) or 0
    return int(np.round(float(value) * (2.0 ** frac)))


def write_gemm_ip_weight_dat(var, layer, odir):
    """Write column-major raw fixed-point weight integers for the GEMM IP generator.

    Same ``[gemm_n][gemm_k]`` column-major order as :func:`write_gemm_ip_weight_cols`,
    but as raw integer bit-patterns (one column per line, ``gemm_k`` space-separated
    integers) instead of a C++ ROM header. Consumed out-of-band by gemm-ip-gen for
    weight-stationary layers; never ``#include``d into the hls4ml project.
    """
    gemm_k = layer.get_attr('gemm_k', layer.get_attr('n_in'))
    gemm_n = layer.get_attr('gemm_n', layer.get_attr('n_out'))
    dat_name = f'{var.name}_gemm_cols'

    weight_data = np.asarray(var.data)
    if weight_data.ndim == 3:
        if weight_data.shape[0] != 1:
            raise NotImplementedError(
                f'{var.name}: GEMM-IP weight .dat with n_inplace={weight_data.shape[0]} '
                'is not supported (one [K, N] block per file).'
            )
        weight_data = weight_data[0]

    kn_source = isinstance(layer, EinsumDense) or layer.get_attr('_original_type') == 'EinsumDense'
    precision = var.type.precision
    with open(f'{odir}/firmware/weights/{dat_name}.dat', 'w') as f:
        for n in range(gemm_n):
            row = []
            for k in range(gemm_k):
                if kn_source:
                    value = weight_data[k, n]
                elif weight_data.shape == (gemm_n, gemm_k):
                    value = weight_data[n, k]
                else:
                    value = weight_data.reshape(-1)[n * gemm_k + k]
                row.append(str(_fixed_point_int(value, precision)))
            f.write(' '.join(row) + '\n')
