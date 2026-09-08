#ifndef NNET_GEMM_STREAM_H_
#define NNET_GEMM_STREAM_H_

// Catapult GEMM IP — io_stream entry points. See nnet_gemm_ip.h for the four-name
// contract and the build-mode selection. This file defines the two streaming
// entries; nnet_gemm_ip.h defines the two array entries.

#include "ac_channel.h"
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

    typedef float weight_t;
    typedef float bias_t;
    typedef float accum_t;
};

#if !defined(GEMM_IP_HEADER)
#if !defined(__SYNTHESIS__)

// ---------------------------------------------------------------------------
// gemm_stream_const_weights — io_stream, constant operand held by the IP
// (Dense / Conv / EinsumDense projections). A streams in one K-wide row per beat;
// the constant columns come from the config ROM. csim sources them from
// CONFIG_T::gemm_weight_beats() (either beat layout, see gemm_weight_at); synth binds the
// IP's own weights.
//
// The K-wide A row may arrive as one beat (feature vector == last dim, the common
// Dense case) OR as several narrower beats (gemm_k = P * beat), which happens when
// the contraction is a flattened multi-dim activation streamed one last-dim slice
// per beat (e.g. Conv/Pool -> Flatten -> Dense). Both are gathered here into a full
// K-wide row before the contraction; gemm_k must be a whole number of beats.
// ---------------------------------------------------------------------------
template <class data_T, class res_T, typename CONFIG_T>
void gemm_stream_const_weights(ac_channel<data_T> &data_stream, ac_channel<res_T> &res_stream,
                            typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
    static_assert(CONFIG_T::gemm_k % data_T::size == 0, "gemm_k must be a whole number of input beats.");
    static_assert(res_T::size == CONFIG_T::gemm_n, "C row width must equal gemm_n.");
    static_assert(CONFIG_T::gemm_m == CONFIG_T::n_patches, "gemm expects gemm_m == n_patches (no tiling).");

    typedef typename data_T::value_type a_val_T;
    static const unsigned PACKETS = CONFIG_T::gemm_k / data_T::size;

    typename CONFIG_T::weight_beat_t *weights = CONFIG_T::gemm_weight_beats();
    for (unsigned int m = 0; m < CONFIG_T::gemm_m; m++) {
        a_val_T a_row[CONFIG_T::gemm_k];
        for (unsigned int kp = 0; kp < PACKETS; kp++) {
            data_T beat = data_stream.read();
            for (unsigned int k = 0; k < data_T::size; k++) {
                a_row[kp * data_T::size + k] = beat[k];
            }
        }
        res_T c_row;
        for (unsigned int n = 0; n < CONFIG_T::gemm_n; n++) {
            typename CONFIG_T::accum_t accum = 0;
            for (unsigned int k = 0; k < CONFIG_T::gemm_k; k++) {
                accum += CONFIG_T::template product<a_val_T, typename CONFIG_T::weight_t>::product(
                    a_row[k], gemm_weight_at<CONFIG_T>(weights, k, n));
            }
            accum += biases[n];
            c_row[n] = cast<a_val_T, typename res_T::value_type, CONFIG_T>(accum);
        }
        res_stream.write(c_row);
    }
}

// ---------------------------------------------------------------------------
// gemm_stream — io_stream, TWO activation operands (attention QK^T / A.V).
// NON-buffered: A and B both stream in; the wrapper does not drain B. The
// mandatory operand residency (B is reused across A's M rows) lives inside the IP
// — modelled here by reading the B columns into local storage, which is the IP's
// business, not a hls4ml-side buffer stage. No separate concurrent feed process.
// ---------------------------------------------------------------------------
template <class data0_T, class data1_T, class res_T, typename CONFIG_T>
void gemm_stream(ac_channel<data0_T> &a_stream, ac_channel<data1_T> &b_stream,
                 ac_channel<res_T> &res_stream,
                 typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
    static_assert(data0_T::size == CONFIG_T::gemm_k, "A row width must equal gemm_k.");
    static_assert(data1_T::size == CONFIG_T::gemm_k, "B column height must equal gemm_k.");
    static_assert(res_T::size == CONFIG_T::gemm_n, "C row width must equal gemm_n.");

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
                accum += CONFIG_T::template product<typename data0_T::value_type, typename data1_T::value_type>::product(
                    a_row[k], b_cols[n][k]);
            }
            accum += biases[n];
            c_row[n] = cast<typename data0_T::value_type, typename res_T::value_type, CONFIG_T>(accum);
        }
        res_stream.write(c_row);
    }
}

#else // __SYNTHESIS__ without a package: declaration only -> loud link failure.

template <class data_T, class res_T, typename CONFIG_T>
void gemm_stream_const_weights(ac_channel<data_T> &data_stream, ac_channel<res_T> &res_stream,
                            typename CONFIG_T::bias_t biases[CONFIG_T::n_out]);

template <class data0_T, class data1_T, class res_T, typename CONFIG_T>
void gemm_stream(ac_channel<data0_T> &a_stream, ac_channel<data1_T> &b_stream,
                 ac_channel<res_T> &res_stream, typename CONFIG_T::bias_t biases[CONFIG_T::n_out]);

#endif // __SYNTHESIS__
#endif // GEMM_IP_HEADER

} // namespace nnet

#endif
