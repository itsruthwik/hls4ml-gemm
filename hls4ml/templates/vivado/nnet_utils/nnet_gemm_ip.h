#ifndef NNET_GEMM_IP_H_
#define NNET_GEMM_IP_H_

// Vivado/Vitis GEMM IP — exactly FOUR entry points, ONE name each across codegen,
// synthesis, and csim (parity with the Catapult four-name contract):
//   gemm_stream             io_stream,   two activation operands    (nnet_gemm_stream.h)
//   gemm_stream_weightless  io_stream,   constant operand in the IP (nnet_gemm_stream.h)
//   gemm_array              io_parallel, two activation operands    (this file)
//   gemm_array_weightless   io_parallel, constant operand in the IP (this file)
//
// Each name has ONE definition, selected by build mode:
//   GEMM_IP_HEADER (package)  -> gemm-ip-gen provides it (synth + cosim)
//   no package, csim          -> hls4ml behavioral model below
//   no package, synth         -> declaration only -> loud link failure (intended)
//
// There is no gemm_ip_stream/gemm_ip_array dispatcher: the four public names ARE the
// frontend/backend seam, exactly as on Catapult. Row/column contract: A rows of width
// gemm_k, B columns of height gemm_k, C rows of width gemm_n. M = n_patches, K = n_in,
// N = n_out. For the weightless entries the constant operand is the IP's own; csim
// sources it from CONFIG_T::gemm_weight_cols() (a ROM accessor the writer injects into
// the config). Argument order is uniform: activation operand(s), then the result, then
// biases LAST.

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_mult.h"
#include "nnet_types.h"

// Always include the behavioral csim model first (the no-package csim path uses it).
#include "nnet_gemm_behavioral.h"

// The external package defines the four names for synthesis / cosim.
#ifdef GEMM_IP_HEADER_VALUE
#include GEMM_IP_HEADER_VALUE
#elif defined(GEMM_IP_HEADER)
#include "gemm_ip_combined.h"
#endif

namespace nnet {

#if !defined(GEMM_IP_HEADER)
#if !defined(__SYNTHESIS__)

// ---------------------------------------------------------------------------
// gemm_array_weightless — io_parallel, constant operand held by the IP
// (Dense / Conv / EinsumDense projections). No weight argument: the columns come
// from the config ROM via CONFIG_T::gemm_weight_cols(). csim computes directly.
// ---------------------------------------------------------------------------
template <class a_row_T, class bias_T, class res_row_T, typename CONFIG_T>
void gemm_array_weightless(a_row_T a_rows[CONFIG_T::gemm_m],
                           res_row_T results[CONFIG_T::gemm_m],
                           bias_T biases[CONFIG_T::gemm_n]) {
    static_assert(a_row_T::size == CONFIG_T::gemm_k, "A row width must equal gemm_k.");
    static_assert(res_row_T::size == CONFIG_T::gemm_n, "C row width must equal gemm_n.");

    typename CONFIG_T::weight_col_t *weight_cols = CONFIG_T::gemm_weight_cols();
    gemm_ip_array_sim<a_row_T, typename CONFIG_T::weight_col_t, bias_T, res_row_T, CONFIG_T>(
        a_rows, weight_cols, biases, results);
}

// ---------------------------------------------------------------------------
// gemm_array — io_parallel, TWO activation operands (attention QK^T / A.V).
// Both A rows and B columns are activation arrays; no constant weight.
// ---------------------------------------------------------------------------
template <class a_row_T, class b_col_T, class bias_T, class res_row_T, typename CONFIG_T>
void gemm_array(a_row_T a_rows[CONFIG_T::gemm_m],
                b_col_T b_cols[CONFIG_T::gemm_n],
                res_row_T results[CONFIG_T::gemm_m],
                bias_T biases[CONFIG_T::gemm_n]) {
    static_assert(a_row_T::size == CONFIG_T::gemm_k, "A row width must equal gemm_k.");
    static_assert(b_col_T::size == CONFIG_T::gemm_k, "B column height must equal gemm_k.");
    static_assert(res_row_T::size == CONFIG_T::gemm_n, "C row width must equal gemm_n.");

    for (unsigned int m = 0; m < CONFIG_T::gemm_m; m++) {
        res_row_T c_row;
        for (unsigned int n = 0; n < CONFIG_T::gemm_n; n++) {
            typename CONFIG_T::accum_t accum = 0;
            for (unsigned int k = 0; k < CONFIG_T::gemm_k; k++) {
                accum += CONFIG_T::template product<typename a_row_T::value_type, typename b_col_T::value_type>::product(
                    a_rows[m][k], b_cols[n][k]);
            }
            accum += biases[n];
            c_row[n] = cast<typename a_row_T::value_type, typename res_row_T::value_type, CONFIG_T>(accum);
        }
        results[m] = c_row;
    }
}

#else // __SYNTHESIS__ without a package: declaration only -> loud link failure.

template <class a_row_T, class bias_T, class res_row_T, typename CONFIG_T>
void gemm_array_weightless(a_row_T a_rows[CONFIG_T::gemm_m], res_row_T results[CONFIG_T::gemm_m],
                           bias_T biases[CONFIG_T::gemm_n]);

template <class a_row_T, class b_col_T, class bias_T, class res_row_T, typename CONFIG_T>
void gemm_array(a_row_T a_rows[CONFIG_T::gemm_m], b_col_T b_cols[CONFIG_T::gemm_n],
                res_row_T results[CONFIG_T::gemm_m], bias_T biases[CONFIG_T::gemm_n]);

#endif // __SYNTHESIS__
#endif // GEMM_IP_HEADER

} // namespace nnet

#endif // NNET_GEMM_IP_H_
