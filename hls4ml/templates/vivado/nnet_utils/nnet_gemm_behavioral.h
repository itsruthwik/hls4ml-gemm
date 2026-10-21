#ifndef NNET_GEMM_BEHAVIORAL_H_
#define NNET_GEMM_BEHAVIORAL_H_

// Vivado/Vitis behavioral GEMM model for C simulation.
// Uses hls::stream instead of ac_channel.
// Only compiled when __SYNTHESIS__ is NOT defined.

#ifndef __SYNTHESIS__

#include "hls_stream.h"
#include "nnet_mult.h"

namespace nnet {


// Simulation-only: row/column GEMM IP behavioral model.
// Reads a_row_T rows from A stream (one per cycle, each width gemm_k),
// reads weight columns from ROM (gemm_n columns, each height gemm_k),
// computes C = A * B and writes res_row_T rows (one per cycle, each width gemm_n).
template <class a_row_T, class b_col_T, class bias_T, class res_row_T, typename CONFIG_T>
void gemm_ip_stream_sim(hls::stream<a_row_T> &a_rows,
                        b_col_T weight_cols[CONFIG_T::gemm_n],
                        bias_T biases[CONFIG_T::gemm_n],
                        hls::stream<res_row_T> &c_rows) {

    // Read all A rows into a local buffer (one a_row_T per row, each gemm_k wide).
    typename a_row_T::value_type activations[CONFIG_T::gemm_m][CONFIG_T::gemm_k];
    for (unsigned int m = 0; m < CONFIG_T::gemm_m; m++) {
        a_row_T a_row = a_rows.read();
        for (unsigned int k = 0; k < CONFIG_T::gemm_k; k++) {
            activations[m][k] = a_row[k];
        }
    }

    // Compute C = A * B and emit one row at a time.
    // Weight layout: weight_cols[n][k] = B[k][n] (column-major).
    for (unsigned int m = 0; m < CONFIG_T::gemm_m; m++) {
        res_row_T c_row;
        for (unsigned int n = 0; n < CONFIG_T::gemm_n; n++) {
            typename CONFIG_T::accum_t accum = 0;
            for (unsigned int k = 0; k < CONFIG_T::gemm_k; k++) {
                accum += CONFIG_T::template product<typename a_row_T::value_type, typename b_col_T::value_type>::product(
                    activations[m][k], weight_cols[n][k]);
            }
            accum += biases[n];
            c_row[n] = cast<typename a_row_T::value_type, typename res_row_T::value_type, CONFIG_T>(accum);
        }
        c_rows.write(c_row);
    }
}

template <class a_row_T, class b_col_T, class bias_T, class res_row_T, typename CONFIG_T>
void gemm_ip_array_sim(a_row_T a_rows[CONFIG_T::gemm_m],
                       b_col_T weight_cols[CONFIG_T::gemm_n],
                       bias_T biases[CONFIG_T::gemm_n],
                       res_row_T results[CONFIG_T::gemm_m]) {
    for (unsigned int m = 0; m < CONFIG_T::gemm_m; m++) {
        res_row_T c_row;
        for (unsigned int n = 0; n < CONFIG_T::gemm_n; n++) {
            typename CONFIG_T::accum_t accum = 0;
            for (unsigned int k = 0; k < CONFIG_T::gemm_k; k++) {
                accum += CONFIG_T::template product<typename a_row_T::value_type, typename b_col_T::value_type>::product(
                    a_rows[m][k], weight_cols[n][k]);
            }
            accum += biases[n];
            c_row[n] = cast<typename a_row_T::value_type, typename res_row_T::value_type, CONFIG_T>(accum);
        }
        results[m] = c_row;
    }
}

} // namespace nnet

#endif // __SYNTHESIS__

#endif // NNET_GEMM_BEHAVIORAL_H_
