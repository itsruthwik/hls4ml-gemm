#ifndef NNET_EINSUM_DENSE_STREAM_H_
#define NNET_EINSUM_DENSE_STREAM_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_einsum_dense.h"

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void einsum_dense(
    hls::stream<data_T> &data_stream, hls::stream<res_T> &res_stream,
    typename CONFIG_T::dense_conf::weight_t weights[CONFIG_T::n_free_kernel * CONFIG_T::n_contract * CONFIG_T::n_inplace],
    typename CONFIG_T::dense_conf::bias_t biases[CONFIG_T::n_free_data * CONFIG_T::n_free_kernel * CONFIG_T::n_inplace]) {

    typename data_T::value_type data[CONFIG_T::tpose_inp_conf::N];
    #pragma HLS ARRAY_PARTITION variable=data complete

    typename res_T::value_type res[CONFIG_T::tpose_out_conf::N];
    #pragma HLS ARRAY_PARTITION variable=res complete

DataPrepare:
    for (int i_in = 0; i_in < CONFIG_T::tpose_inp_conf::N / data_T::size; i_in++) {
        if (CONFIG_T::tpose_inp_conf::N / data_T::size > 1) {
            #pragma HLS PIPELINE
        }
        data_T data_pack = data_stream.read();
    DataPack:
        for (int i_pack = 0; i_pack < data_T::size; i_pack++) {
            #pragma HLS UNROLL
            data[i_in * data_T::size + i_pack] = data_pack[i_pack];
        }
    }

    nnet::einsum_dense<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(data, res, weights, biases);

ResWrite:
    for (unsigned i_out = 0; i_out < CONFIG_T::tpose_out_conf::N / res_T::size; i_out++) {
        if (CONFIG_T::tpose_out_conf::N / res_T::size > 1) {
            #pragma HLS PIPELINE
        }
        res_T res_pack;
        PRAGMA_DATA_PACK(res_pack)
    ResPack:
        for (int i_pack = 0; i_pack < res_T::size; i_pack++) {
            #pragma HLS UNROLL
            res_pack[i_pack] = res[i_out * res_T::size + i_pack];
        }
        res_stream.write(res_pack);
    }
}

} // namespace nnet

#endif
