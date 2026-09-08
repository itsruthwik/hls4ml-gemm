"""Shared GEMM-IP packed weight writer.

Used by both the Catapult and Vivado/Vitis writers so the packing logic cannot
silently diverge between backends.

Layout is selected per layer by ``SecondOperandRowMajor`` (resolved onto the node
as ``second_operand_row_major``), the same knob that flips the B beat order of a
two-operand GEMM. For a weight-stationary GEMM the constant operand IS the second
operand, so the knob picks how its ROM / ``.dat`` are packed:

- column-major (default): ``<w>_gemm_cols[gemm_n]``, each beat an
  ``array<weight_t, gemm_k>`` holding one output column (``cols[n][k] = W[k][n]``).
- row-major: ``<w>_gemm_rows[gemm_k]``, each beat an ``array<weight_t, gemm_n>``
  holding one contraction row (``rows[k][n] = W[k][n]``).

The manifest (``gemm_config.json``) reports the choice as ``weight_layout``.
"""

import numpy as np

from hls4ml.model.layers import EinsumDense


def gemm_ip_weight_row_major(layer):
    """True if the layer's constant operand is packed row-major (SecondOperandRowMajor)."""
    return bool(layer.get_attr('second_operand_row_major', False))


def gemm_ip_weight_layout(layer):
    """Manifest spelling of the packed weight layout: ``row_major`` | ``column_major``."""
    return 'row_major' if gemm_ip_weight_row_major(layer) else 'column_major'


def gemm_ip_weight_basename(var, layer):
    """Basename (no extension) shared by the packed ROM header and the raw-int ``.dat``."""
    return f'{var.name}_gemm_rows' if gemm_ip_weight_row_major(layer) else f'{var.name}_gemm_cols'


def _single_kn_block(var, weight_data):
    # EinsumDense stores its kernel as [n_inplace, n_contract, n_free_kernel].
    # The packed header holds a single [K, N] block, so peel a leading unit
    # in-place dimension; refuse anything we cannot represent rather than
    # falling through to a layout guess.
    if weight_data.ndim == 3:
        if weight_data.shape[0] != 1:
            raise NotImplementedError(
                f'{var.name}: GEMM-IP packed weights with n_inplace={weight_data.shape[0]} '
                'are not supported (one [K, N] block per file).'
            )
        weight_data = weight_data[0]
    return weight_data


def _kn_reader(var, layer, weight_data, gemm_k, gemm_n):
    """Return ``value(k, n)`` = the weight multiplying input k for output n.

    The SOURCE layout is decided by the layer type, never sniffed from the shape —
    a square kernel is ambiguous and a shape guess silently packed transposed
    weights for square EinsumDense kernels (e.g. MHA projections).
      - EinsumDense kernels are [n_contract, n_free_kernel] = [K, N];
        TransposeWeightsForGemmIP does not touch them (whether still an EinsumDense
        layer or lowered to a Gemm node, which sets _original_type).
      - Dense/conv kernels reach the writer transposed/flattened to [N, K] by
        TransposeWeightsForGemmIP.
    """
    kn_source = isinstance(layer, EinsumDense) or layer.get_attr('_original_type') == 'EinsumDense'
    if kn_source:
        if weight_data.shape != (gemm_k, gemm_n):
            raise NotImplementedError(
                f'{var.name}: EinsumDense GEMM-IP weights expected [K, N] = '
                f'({gemm_k}, {gemm_n}), got {weight_data.shape}.'
            )
        return lambda k, n: weight_data[k, n]
    if weight_data.shape == (gemm_n, gemm_k):
        return lambda k, n: weight_data[n, k]
    flat = weight_data.reshape(-1)
    return lambda k, n: flat[n * gemm_k + k]


def _beat_iter(row_major, gemm_k, gemm_n):
    """(outer, inner) index pairs: one beat per outer index.

    column-major: beat n holds k = 0..K-1;  row-major: beat k holds n = 0..N-1.
    """
    if row_major:
        return gemm_k, gemm_n, (lambda outer, inner: (outer, inner))   # (k, n)
    return gemm_n, gemm_k, (lambda outer, inner: (inner, outer))       # (k, n)


def write_gemm_ip_weight_cols(var, layer, odir):
    """Write the GEMM-IP packed weight ROM header (see the module docstring for layout).

    The scalar weight header remains the source of truth for native and
    simulation paths. This extra header is consumed only by GEMM-IP
    synthesis calls to avoid repacking flat scalar weights in hardware.
    """

    gemm_k = layer.get_attr('gemm_k', layer.get_attr('n_in'))
    gemm_n = layer.get_attr('gemm_n', layer.get_attr('n_out'))
    row_major = gemm_ip_weight_row_major(layer)
    beat_name = gemm_ip_weight_basename(var, layer)
    guard = f'{beat_name.upper()}_H_'

    weight_data = _single_kn_block(var, np.asarray(var.data))
    value = _kn_reader(var, layer, weight_data, gemm_k, gemm_n)
    n_beats, beat_len, kn = _beat_iter(row_major, gemm_k, gemm_n)

    with open(f'{odir}/firmware/weights/{beat_name}.h', 'w') as h_file:
        layout = 'rows (row-major, one contraction row per beat)' if row_major \
            else 'columns (column-major, one output column per beat)'
        h_file.write(f'// Packed GEMM-IP weight {layout} for {var.name}\n')
        h_file.write(f'// Source numpy array shape {var.shape}\n\n')
        h_file.write(f'#ifndef {guard}\n')
        h_file.write(f'#define {guard}\n\n')
        # File-relative include: resolves from firmware/weights/ on every
        # backend regardless of the build's -I flags (the Vivado csim has
        # none that reach nnet_utils, and the bare "nnet_utils/..." form only
        # worked on Catapult via the copy shipped inside MGC_HOME).
        h_file.write('#include "../nnet_utils/nnet_types.h"\n\n')
        h_file.write(f'static nnet::array<{var.type.name}, {beat_len}> {beat_name}[{n_beats}] = {{')
        sep = ''
        for outer in range(n_beats):
            h_file.write(sep + '{')
            col_sep = ''
            for inner in range(beat_len):
                k, n = kn(outer, inner)
                h_file.write(col_sep + var.precision_fmt.format(value(k, n)))
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
    """Write raw fixed-point weight integers for the GEMM IP generator.

    Same beat order as :func:`write_gemm_ip_weight_cols` (column-major: one output
    column per line, ``gemm_k`` integers; row-major: one contraction row per line,
    ``gemm_n`` integers) but as raw integer bit-patterns instead of a C++ ROM header.
    Consumed out-of-band by gemm-ip-gen for weight-stationary layers (which reads the
    manifest's ``weight_layout`` to decode it); never ``#include``d into the hls4ml
    project.
    """
    gemm_k = layer.get_attr('gemm_k', layer.get_attr('n_in'))
    gemm_n = layer.get_attr('gemm_n', layer.get_attr('n_out'))
    dat_name = gemm_ip_weight_basename(var, layer)

    weight_data = _single_kn_block(var, np.asarray(var.data))
    value = _kn_reader(var, layer, weight_data, gemm_k, gemm_n)
    n_beats, beat_len, kn = _beat_iter(gemm_ip_weight_row_major(layer), gemm_k, gemm_n)
    precision = var.type.precision
    with open(f'{odir}/firmware/weights/{dat_name}.dat', 'w') as f:
        for outer in range(n_beats):
            row = []
            for inner in range(beat_len):
                k, n = kn(outer, inner)
                row.append(str(_fixed_point_int(value(k, n), precision)))
            f.write(' '.join(row) + '\n')
