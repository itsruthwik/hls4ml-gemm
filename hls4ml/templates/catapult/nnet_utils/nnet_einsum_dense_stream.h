#ifndef NNET_EINSUM_DENSE_STREAM_H_
#define NNET_EINSUM_DENSE_STREAM_H_

#include "nnet_common.h"
#include "nnet_einsum_dense.h"
#include "nnet_types.h"
#include <ac_channel.h>

namespace nnet {

// Streaming EinsumDense shell: drain the input stream into the flat array the
// existing array-core nnet::einsum_dense already operates on (transpose ->
// per-row dense (Latency or Resource, picked by nnet::dense on CONFIG_T::dense_conf::strategy)
// -> transpose), then re-stream the output. Bit-exact with the io_parallel path
// by construction — same core, same math, only the I/O is packed/unpacked beat by beat
// the way nnet_dense_stream.h does it for plain Dense.
template <class data_T, class res_T, typename CONFIG_T>
void einsum_dense(
    ac_channel<data_T> &data_stream, ac_channel<res_T> &res_stream,
    typename CONFIG_T::dense_conf::weight_t weights[CONFIG_T::n_free_kernel * CONFIG_T::n_contract * CONFIG_T::n_inplace],
    typename CONFIG_T::dense_conf::bias_t biases[CONFIG_T::n_free_data * CONFIG_T::n_free_kernel * CONFIG_T::n_inplace]) {

    constexpr unsigned N_IN = CONFIG_T::n_free_data * CONFIG_T::n_contract * CONFIG_T::n_inplace;
    constexpr unsigned N_OUT = CONFIG_T::n_free_data * CONFIG_T::n_free_kernel * CONFIG_T::n_inplace;

    typename data_T::value_type data[N_IN];
    typename res_T::value_type res[N_OUT];

    #pragma hls_pipeline_init_interval 1
DataPrepare:
    for (unsigned i_in = 0; i_in < N_IN / data_T::size; i_in++) {
        data_T data_pack = data_stream.read();
    #pragma hls_unroll
    DataPack:
        for (unsigned i_pack = 0; i_pack < data_T::size; i_pack++) {
            data[i_in * data_T::size + i_pack] = data_pack[i_pack];
        }
    }

    nnet::einsum_dense<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(data, res, weights, biases);

    #pragma hls_pipeline_init_interval 1
ResWrite:
    for (unsigned i_out = 0; i_out < N_OUT / res_T::size; i_out++) {
        res_T res_pack;
    #pragma hls_unroll
    ResPack:
        for (unsigned i_pack = 0; i_pack < res_T::size; i_pack++) {
            res_pack[i_pack] = res[i_out * res_T::size + i_pack];
        }
        res_stream.write(res_pack);
    }
}

} // namespace nnet

#endif
