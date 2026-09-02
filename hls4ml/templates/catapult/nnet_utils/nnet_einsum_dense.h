#ifndef NNET_EINSUM_DENSE_H_
#define NNET_EINSUM_DENSE_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_dense_stream.h"
#include "nnet_helpers.h"
#include "nnet_mult.h"
#include "nnet_transpose.h"

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void einsum_dense(
    data_T data[CONFIG_T::n_free_data * CONFIG_T::n_contract * CONFIG_T::n_inplace],
    res_T res[CONFIG_T::n_free_data * CONFIG_T::n_free_kernel * CONFIG_T::n_inplace],
    typename CONFIG_T::dense_conf::weight_t weights[CONFIG_T::n_free_kernel * CONFIG_T::n_contract * CONFIG_T::n_inplace],
    typename CONFIG_T::dense_conf::bias_t biases[CONFIG_T::n_free_data * CONFIG_T::n_free_kernel * CONFIG_T::n_inplace]) {
    data_T inp_tpose[CONFIG_T::n_free_data * CONFIG_T::n_contract * CONFIG_T::n_inplace];
    res_T out_tpose[CONFIG_T::n_free_data * CONFIG_T::n_free_kernel * CONFIG_T::n_inplace];
    res_T out_buffer[CONFIG_T::n_free_kernel];
    // Catapult does not use the Vivado ARRAY_PARTITION pragma syntax here.

    nnet::transpose<data_T, data_T, typename CONFIG_T::tpose_inp_conf>(data, inp_tpose);

    constexpr unsigned L0 = CONFIG_T::n_free_data;
    constexpr unsigned L1 = CONFIG_T::n_free_kernel;
    constexpr unsigned C = CONFIG_T::n_contract;
    constexpr unsigned I = CONFIG_T::n_inplace;

    for (unsigned l0 = 0; l0 < L0; l0++) {
        #pragma hls_unroll
        for (unsigned i = 0; i < I; i++) {
            #pragma hls_unroll
            dense<data_T, res_T, typename CONFIG_T::dense_conf>(&inp_tpose[(i * L0 + l0) * C], out_buffer,
                                                                &weights[(i * L1 * C)], &biases[((i * L0 + l0) * L1)]);
            for (unsigned j = 0; j < L1; j++) {
                #pragma hls_unroll
                out_tpose[(i * L0 + l0) * L1 + j] = out_buffer[j];
            }
        }
    }

    nnet::transpose<res_T, res_T, typename CONFIG_T::tpose_out_conf>(out_tpose, res);
}

} // namespace nnet


#endif
