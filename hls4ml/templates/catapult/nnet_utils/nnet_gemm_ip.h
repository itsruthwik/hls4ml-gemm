#ifndef NNET_GEMM_IP_H_
#define NNET_GEMM_IP_H_

// Catapult GEMM IP — exactly FOUR entry points, ONE name each across codegen,
// synthesis, and csim:
//   gemm_stream             io_stream,   two activation operands   (nnet_gemm_stream.h)
//   gemm_stream_const_weights  io_stream,   constant operand in the IP (nnet_gemm_stream.h)
//   gemm_array              io_parallel, two activation operands   (this file)
//   gemm_array_const_weights   io_parallel, constant operand in the IP (this file)
//
// Each name has ONE definition, selected by build mode:
//   GEMM_IP_HEADER (package)  -> gemm-ip-gen provides it (synth + cosim)
//   no package, csim          -> hls4ml behavioral model below
//   no package, synth         -> declaration only -> loud link failure (intended)
//
// Row/column contract: A rows of width gemm_k, B columns of height gemm_k, C rows
// of width gemm_n. M = n_patches, K = n_in, N = n_out. For the const_weights entries
// the constant operand is the IP's own; csim sources it from
// CONFIG_T::gemm_weight_beats() (a ROM accessor the writer injects into the config; beat
// layout per CONFIG_T::weights_row_major, read via gemm_weight_at).
//
// Argument order is uniform across all four: activation operand(s), then the
// result (out array / out channel), then biases LAST.

#include "ac_channel.h"
#include "nnet_common.h"
#include "nnet_mult.h"
#include "nnet_types.h"

// The external package defines the four names for synthesis / cosim.
#ifdef GEMM_IP_HEADER_VALUE
#include GEMM_IP_HEADER_VALUE
#elif defined(GEMM_IP_HEADER)
#include "gemm_ip_combined.h"
#endif

namespace nnet {

// ---------------------------------------------------------------------------
// Constant-operand ROM access, layout-agnostic. The writer packs a weight-stationary
// GEMM's weights per SecondOperandRowMajor (CONFIG_T::weights_row_major):
//   false (default): gemm_weight_beats()[n][k] = W[k][n]  (K-high output columns)
//   true           : gemm_weight_beats()[k][n] = W[k][n]  (N-wide contraction rows)
// Both index forms are type-valid on either beat array; the dead branch folds away.
// ---------------------------------------------------------------------------
template <typename CONFIG_T>
inline typename CONFIG_T::weight_t gemm_weight_at(typename CONFIG_T::weight_beat_t *beats,
                                                  unsigned k, unsigned n) {
    if (CONFIG_T::weights_row_major) {
        return beats[k][n];
    }
    return beats[n][k];
}

#if !defined(GEMM_IP_HEADER)
#if !defined(__SYNTHESIS__)

// ---------------------------------------------------------------------------
// gemm_array — io_parallel, TWO activation operands (attention QK^T / A.V).
// Both A rows and B columns are passed in as arrays.
// ---------------------------------------------------------------------------
// Two-operand GEMM (QK^T / A.V) never has a real bias -- there is only ever this
// one signature, no bias argument at all.
template <class a_row_T, class b_col_T, class res_row_T, typename CONFIG_T>
void gemm_array(a_row_T a_rows[CONFIG_T::gemm_m],
                b_col_T weight_cols[CONFIG_T::gemm_n],
                res_row_T results[CONFIG_T::gemm_m]) {
    static_assert(a_row_T::size == CONFIG_T::gemm_k, "A row width must equal gemm_k.");
    static_assert(b_col_T::size == CONFIG_T::gemm_k, "B column height must equal gemm_k.");
    static_assert(res_row_T::size == CONFIG_T::gemm_n, "C row width must equal gemm_n.");

    for (unsigned int m = 0; m < CONFIG_T::gemm_m; m++) {
        res_row_T c_row;
        for (unsigned int n = 0; n < CONFIG_T::gemm_n; n++) {
            typename CONFIG_T::accum_t accum = 0;
            for (unsigned int k = 0; k < CONFIG_T::gemm_k; k++) {
                accum += CONFIG_T::template product<typename a_row_T::value_type, typename b_col_T::value_type>::product(
                    a_rows[m][k], weight_cols[n][k]);
            }
            c_row[n] = cast<typename a_row_T::value_type, typename res_row_T::value_type, CONFIG_T>(accum);
        }
        results[m] = c_row;
    }
}

// ---------------------------------------------------------------------------
// gemm_array_const_weights — io_parallel, constant operand held by the IP
// (Dense / Conv / EinsumDense projections). csim sources the columns from the
// config ROM; no weight argument on the signature.
// ---------------------------------------------------------------------------
// Bias, like the weight ROM, is read through the config (CONFIG_T::gemm_bias(),
// injected by the writer alongside gemm_weight_beats()) rather than a function
// argument -- the same "compile-time constant" mechanism, so there is only ever
// this one signature.
template <class a_row_T, class res_row_T, typename CONFIG_T>
void gemm_array_const_weights(a_row_T a_rows[CONFIG_T::gemm_m],
                           res_row_T results[CONFIG_T::gemm_m]) {
    static_assert(a_row_T::size == CONFIG_T::gemm_k, "A row width must equal gemm_k.");
    static_assert(res_row_T::size == CONFIG_T::gemm_n, "C row width must equal gemm_n.");

    typename CONFIG_T::weight_beat_t *weights = CONFIG_T::gemm_weight_beats();
    typename CONFIG_T::bias_t *biases = CONFIG_T::gemm_bias();
    for (unsigned int m = 0; m < CONFIG_T::gemm_m; m++) {
        res_row_T c_row;
        for (unsigned int n = 0; n < CONFIG_T::gemm_n; n++) {
            typename CONFIG_T::accum_t accum = 0;
            for (unsigned int k = 0; k < CONFIG_T::gemm_k; k++) {
                accum += CONFIG_T::template product<typename a_row_T::value_type,
                                                    typename CONFIG_T::weight_t>::product(
                    a_rows[m][k], gemm_weight_at<CONFIG_T>(weights, k, n));
            }
            accum += biases[n];
            c_row[n] = cast<typename a_row_T::value_type, typename res_row_T::value_type, CONFIG_T>(accum);
        }
        results[m] = c_row;
    }
}

#else // __SYNTHESIS__ without a package: declaration only -> loud link failure.

template <class a_row_T, class b_col_T, class res_row_T, typename CONFIG_T>
void gemm_array(a_row_T a_rows[CONFIG_T::gemm_m], b_col_T weight_cols[CONFIG_T::gemm_n],
                res_row_T results[CONFIG_T::gemm_m]);

template <class a_row_T, class res_row_T, typename CONFIG_T>
void gemm_array_const_weights(a_row_T a_rows[CONFIG_T::gemm_m], res_row_T results[CONFIG_T::gemm_m]);

#endif // __SYNTHESIS__
#endif // GEMM_IP_HEADER

} // namespace nnet

#endif
