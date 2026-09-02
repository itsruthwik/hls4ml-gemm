#ifndef NNET_SPLIT_MERGE_H_
#define NNET_SPLIT_MERGE_H_

#include "hls_stream.h"
#include "nnet_types.h"

// Head-lane Split / Merge for multi-head attention on the GEMM path (Vivado/Vitis
// dialect: hls::stream for io_stream, flat-array reindex for io_parallel; the
// Catapult copy uses ac_channel).
//
// A projection emits its output as a token-per-beat stream [seq, d_model]
// (beat = d_model). Multi-head attention needs each head's [seq, key_dim] block.
// Because head lives INSIDE a beat (d_model = H * key_dim, contiguous lanes),
// separating heads is a stateless lane slice — one input beat produces H output
// beats in the same step, no buffer and no reorder. This lets the head-move
// transpose be dropped entirely: the GEMM nodes stay uniform (they consume clean
// per-head [seq, key_dim]); all head-lane handling lives here.
//
// The io_stream forms take the H output/input channels ENUMERATED (variadic),
// matching how hls4ml declares each layer output as a separate hls::stream. The
// io_parallel forms are flat-array reindex. Both are stateless.

namespace nnet {

// ---- io_stream ------------------------------------------------------------

// Split a d_model-wide stream into H head-lane streams (key_dim each), given as
// enumerated output channels. data_T = array<T, d_model>, res_T = array<T, key_dim>.
template <class data_T, class res_T, typename CONFIG_T, class... Outs>
void split_lanes(hls::stream<data_T> &data, Outs &... outs) {
    const unsigned H = sizeof...(outs);
    hls::stream<res_T> *out_ch[] = {&outs...};
    for (unsigned b = 0; b < CONFIG_T::n_beats; b++) {
        data_T in = data.read();
        for (unsigned h = 0; h < H; h++) {
            #pragma HLS UNROLL
            res_T out;
            for (unsigned k = 0; k < res_T::size; k++) {
                #pragma HLS UNROLL
                out[k] = in[h * res_T::size + k];
            }
            out_ch[h]->write(out);
        }
    }
}

// Inverse: concatenate H head-lane streams (key_dim each, enumerated) into one
// d_model-wide stream. data_T = array<T, key_dim>, res_T = array<T, d_model>.
template <class data_T, class res_T, typename CONFIG_T, class... Ins>
void merge_lanes(hls::stream<res_T> &res, Ins &... ins) {
    const unsigned H = sizeof...(ins);
    hls::stream<data_T> *in_ch[] = {&ins...};
    for (unsigned b = 0; b < CONFIG_T::n_beats; b++) {
        res_T out;
        for (unsigned h = 0; h < H; h++) {
            #pragma HLS UNROLL
            data_T beat = in_ch[h]->read();
            for (unsigned k = 0; k < data_T::size; k++) {
                #pragma HLS UNROLL
                out[h * data_T::size + k] = beat[k];
            }
        }
        res.write(out);
    }
}

// ---- io_parallel ----------------------------------------------------------

// Flat-array gather: [seq, d_model] -> H x [seq, key_dim]. Enumerated outputs.
// CONFIG_T::seq, ::d_model, ::key_dim. All indices compile-time -> wiring.
template <class data_T, class res_T, typename CONFIG_T, class... Outs>
void split_lanes_array(const data_T *data, Outs *... outs) {
    const unsigned H = sizeof...(outs);
    res_T *out_a[] = {outs...};
    for (unsigned s = 0; s < CONFIG_T::seq; s++) {
        for (unsigned h = 0; h < H; h++) {
            #pragma HLS UNROLL
            for (unsigned k = 0; k < CONFIG_T::key_dim; k++) {
                #pragma HLS UNROLL
                out_a[h][s * CONFIG_T::key_dim + k] = data[s * CONFIG_T::d_model + h * CONFIG_T::key_dim + k];
            }
        }
    }
}

// Inverse scatter: H x [seq, key_dim] -> [seq, d_model].
template <class data_T, class res_T, typename CONFIG_T, class... Ins>
void merge_lanes_array(res_T *res, Ins *... ins) {
    const unsigned H = sizeof...(ins);
    const data_T *in_a[] = {ins...};
    for (unsigned s = 0; s < CONFIG_T::seq; s++) {
        for (unsigned h = 0; h < H; h++) {
            #pragma HLS UNROLL
            for (unsigned k = 0; k < CONFIG_T::key_dim; k++) {
                #pragma HLS UNROLL
                res[s * CONFIG_T::d_model + h * CONFIG_T::key_dim + k] = in_a[h][s * CONFIG_T::key_dim + k];
            }
        }
    }
}

} // namespace nnet
#endif
