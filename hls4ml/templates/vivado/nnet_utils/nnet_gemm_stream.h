#ifndef NNET_GEMM_STREAM_H_
#define NNET_GEMM_STREAM_H_

// Vivado/Vitis GEMM IP — io_stream entry points. See nnet_gemm_ip.h for the four-name
// contract and the build-mode selection. This file defines the two streaming entries;
// nnet_gemm_ip.h defines the two array entries. Uses hls::stream (not ac_channel).

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_gemm_ip.h"

namespace nnet {

struct gemm_config {
    static const unsigned n_in      = 10;
    static const unsigned n_out     = 10;
    static const unsigned n_patches = 1;
    static const unsigned gemm_m    = 1;
    static const unsigned gemm_k    = 10;
    static const unsigned gemm_n    = 10;
    static const bool     transpose_weights   = true;
    static const bool     b_row_major         = false;  // two-operand B beat layout (default col-major)

    typedef float weight_t;
    typedef float bias_t;
    typedef float accum_t;
};

#if !defined(GEMM_IP_HEADER)
#if !defined(__SYNTHESIS__)

// ---------------------------------------------------------------------------
// gemm_stream_const_weights — io_stream, constant operand held by the IP
// (Dense / Conv / EinsumDense projections). No weight argument on the signature:
// the columns come from the config ROM via CONFIG_T::gemm_weight_cols(). A streams in
// one K-wide row per beat (gemm_k may span several narrower beats). No tiling:
// gemm_m == n_patches.
// ---------------------------------------------------------------------------
template <class data_T, class res_T, typename CONFIG_T>
void gemm_stream_const_weights(hls::stream<data_T> &data_stream, hls::stream<res_T> &res_stream,
                            typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
    typedef nnet::array<typename data_T::value_type, CONFIG_T::gemm_k> a_row_T;
    typedef typename CONFIG_T::weight_col_t b_col_T;
    static_assert(CONFIG_T::gemm_m == CONFIG_T::n_patches,
                  "gemm_stream expects gemm_m == n_patches (no tiling).");

    typename CONFIG_T::weight_col_t *weight_cols = CONFIG_T::gemm_weight_cols();

    // Simulation: source the constant columns from the config ROM (like Catapult).
    hls::stream<a_row_T> a_row_stream("a_row_stream_sim");
    for (unsigned int m = 0; m < CONFIG_T::gemm_m; m++) {
        a_row_T a_row;
        for (unsigned int kp = 0; kp < CONFIG_T::gemm_k / data_T::size; kp++) {
            data_T a_pack = data_stream.read();
            for (unsigned int k = 0; k < data_T::size; k++) {
                a_row[kp * data_T::size + k] = a_pack[k];
            }
        }
        a_row_stream.write(a_row);
    }
    gemm_ip_stream_sim<a_row_T, b_col_T, typename CONFIG_T::bias_t, res_T, CONFIG_T>(
        a_row_stream, weight_cols, biases, res_stream);
}

// ---------------------------------------------------------------------------
// gemm_stream — io_stream, TWO activation operands (attention QK^T / A.V).
// A and B both stream in; B is read into local storage (the IP's operand residency),
// then C = A * B streams out. No constant weight. The B beat layout is selected by
// CONFIG_T::b_row_major (dispatched below; -std=c++0x has no if constexpr, and the two
// layouts carry different static_asserts, so use a bool-specialized helper):
//   false (default) : col-major -- gemm_n beats, each K-wide (b[n][k] = B[k][n]).
//   true            : row-major -- gemm_k beats, each N-wide (b[k][n] = B[k][n]); this is
//                     the layout the mvau IP consumes (one contraction row per beat).
// ---------------------------------------------------------------------------
template <bool B_ROW_MAJOR> struct gemm_stream_two_op;

template <> struct gemm_stream_two_op<false> {  // col-major B: K-wide beats, gemm_n of them
    template <class data0_T, class data1_T, class res_T, typename CONFIG_T>
    static void run(hls::stream<data0_T> &a_stream, hls::stream<data1_T> &b_stream,
                    hls::stream<res_T> &res_stream, typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
        static_assert(data1_T::size == CONFIG_T::gemm_k, "col-major B column height must equal gemm_k.");
        data1_T b_cols[CONFIG_T::gemm_n];
        for (unsigned int n = 0; n < CONFIG_T::gemm_n; n++) {
            b_cols[n] = b_stream.read();
        }
        for (unsigned int m = 0; m < CONFIG_T::gemm_m; m++) {
            data0_T a_row = a_stream.read();
            res_T c_row;
            for (unsigned int n = 0; n < CONFIG_T::gemm_n; n++) {
                typename CONFIG_T::accum_t accum = 0;
                for (unsigned int k = 0; k < CONFIG_T::gemm_k; k++) {
                    accum += CONFIG_T::template product<typename data0_T::value_type,
                                                        typename data1_T::value_type>::product(a_row[k], b_cols[n][k]);
                }
                accum += biases[n];
                c_row[n] = cast<typename data0_T::value_type, typename res_T::value_type, CONFIG_T>(accum);
            }
            res_stream.write(c_row);
        }
    }
};

template <> struct gemm_stream_two_op<true> {  // row-major B: N-wide beats, gemm_k of them
    template <class data0_T, class data1_T, class res_T, typename CONFIG_T>
    static void run(hls::stream<data0_T> &a_stream, hls::stream<data1_T> &b_stream,
                    hls::stream<res_T> &res_stream, typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
        static_assert(data1_T::size == CONFIG_T::gemm_n, "row-major B row width must equal gemm_n.");
        data1_T b_rows[CONFIG_T::gemm_k];
        for (unsigned int k = 0; k < CONFIG_T::gemm_k; k++) {
            b_rows[k] = b_stream.read();
        }
        for (unsigned int m = 0; m < CONFIG_T::gemm_m; m++) {
            data0_T a_row = a_stream.read();
            res_T c_row;
            for (unsigned int n = 0; n < CONFIG_T::gemm_n; n++) {
                typename CONFIG_T::accum_t accum = 0;
                for (unsigned int k = 0; k < CONFIG_T::gemm_k; k++) {
                    accum += CONFIG_T::template product<typename data0_T::value_type,
                                                        typename data1_T::value_type>::product(a_row[k], b_rows[k][n]);
                }
                accum += biases[n];
                c_row[n] = cast<typename data0_T::value_type, typename res_T::value_type, CONFIG_T>(accum);
            }
            res_stream.write(c_row);
        }
    }
};

template <class data0_T, class data1_T, class res_T, typename CONFIG_T>
void gemm_stream(hls::stream<data0_T> &a_stream, hls::stream<data1_T> &b_stream,
                 hls::stream<res_T> &res_stream,
                 typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
    static_assert(data0_T::size == CONFIG_T::gemm_k, "A row width must equal gemm_k.");
    static_assert(res_T::size == CONFIG_T::gemm_n, "C row width must equal gemm_n.");
    gemm_stream_two_op<CONFIG_T::b_row_major>::template run<data0_T, data1_T, res_T, CONFIG_T>(
        a_stream, b_stream, res_stream, biases);
}

#else // __SYNTHESIS__ without a package: declaration only -> loud link failure.

template <class data_T, class res_T, typename CONFIG_T>
void gemm_stream_const_weights(hls::stream<data_T> &data_stream, hls::stream<res_T> &res_stream,
                            typename CONFIG_T::bias_t biases[CONFIG_T::n_out]);

template <class data0_T, class data1_T, class res_T, typename CONFIG_T>
void gemm_stream(hls::stream<data0_T> &a_stream, hls::stream<data1_T> &b_stream,
                 hls::stream<res_T> &res_stream, typename CONFIG_T::bias_t biases[CONFIG_T::n_out]);

#endif // __SYNTHESIS__
#endif // GEMM_IP_HEADER

} // namespace nnet

#endif // NNET_GEMM_STREAM_H_
