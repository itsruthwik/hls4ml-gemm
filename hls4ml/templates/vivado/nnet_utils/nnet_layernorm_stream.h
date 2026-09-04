#ifndef NNET_LAYERNORM_STREAM_H_
#define NNET_LAYERNORM_STREAM_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_layernorm.h"
#include "nnet_types.h"

namespace nnet {

// ****************************************************
//       Streaming Layer Normalization
// ****************************************************
//
// LayerNorm normalizes each token (the `dim = n_in / seq_len` features on the
// normalized axis) independently. In io_stream each stream element carries one
// full token (`data_T` is an nnet::array of `dim` values), so one read == one
// token: buffer it, run the shared per-token kernel `layernorm_1d`, write it
// back. Overloads the io_parallel `layernormalize` by argument type (hls::stream
// vs flat array), matching the batchnorm/activation streaming convention.

template <class data_T, class res_T, typename CONFIG_T>
void layernormalize(hls::stream<data_T> &data, hls::stream<res_T> &res,
                    typename CONFIG_T::scale_t scale[CONFIG_T::n_in / CONFIG_T::seq_len],
                    typename CONFIG_T::bias_t bias[CONFIG_T::n_in / CONFIG_T::seq_len],
                    typename CONFIG_T::table_t rsqrt_table[CONFIG_T::table_size]) {
    static const unsigned dim = CONFIG_T::n_in / CONFIG_T::seq_len;

    #pragma HLS ARRAY_PARTITION variable=scale complete
    #pragma HLS ARRAY_PARTITION variable=bias complete

LayerNormSeqLoop:
    for (int j = 0; j < CONFIG_T::seq_len; ++j) {
        #pragma HLS PIPELINE

        data_T in_pack = data.read();
        res_T out_pack;
        PRAGMA_DATA_PACK(out_pack)

        typename data_T::value_type in_buf[dim];
        typename res_T::value_type out_buf[dim];
        #pragma HLS ARRAY_PARTITION variable=in_buf complete
        #pragma HLS ARRAY_PARTITION variable=out_buf complete

    LayerNormLoad:
        for (int i = 0; i < dim; ++i) {
            #pragma HLS UNROLL
            in_buf[i] = in_pack[i];
        }

        layernorm_1d<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(in_buf, out_buf, scale, bias,
                                                                                        rsqrt_table);

    LayerNormStore:
        for (int i = 0; i < dim; ++i) {
            #pragma HLS UNROLL
            out_pack[i] = out_buf[i];
        }

        res.write(out_pack);
    }
}

} // namespace nnet

#endif
