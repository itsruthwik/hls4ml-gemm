#ifndef NNET_EINSUM_DENSE_STREAM_H_
#define NNET_EINSUM_DENSE_STREAM_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_einsum_dense.h"
#include "nnet_einsum_stream.h"
#include "nnet_mult.h"
#include "nnet_types.h"
#include <ac_channel.h>

namespace nnet {

// io_stream EinsumDense, Resource strategy only (io_stream Latency is rejected at conversion; see
// init_einsum_dense in the Catapult backend). Weights are constant, so unlike
// nnet_einsum_stream.h's two-operand kernel there is only one operand to stream in: the data
// input. The kernel unpacks it into a flat array (DataPrepare, as in the Catapult two-operand
// Einsum kernel) then runs it through nnet::transpose with the already-generated tpose_inp_conf --
// the same config and the same canonical (I, L0, C) layout the io_parallel array core
// (nnet_einsum_dense.h) uses. It then computes one row (one (i, l0) pair) at a time by calling
// nnet::dense<>() with CONFIG_T::dense_conf -- the exact same call nnet_einsum_dense.h's
// io_parallel core makes per free-data index -- so weights stay in the single (I, L1, C) layout
// for both io types and bias add / cast<>() happen inside that same dense kernel, giving
// bit-exact results with io_parallel.
template <class data_T, class res_T, typename CONFIG_T>
void einsum_dense(
    ac_channel<data_T> &data_stream, ac_channel<res_T> &res_stream,
    typename CONFIG_T::dense_conf::weight_t weights[CONFIG_T::n_free_kernel * CONFIG_T::n_contract * CONFIG_T::n_inplace],
    typename CONFIG_T::dense_conf::bias_t biases[CONFIG_T::n_free_data * CONFIG_T::n_free_kernel * CONFIG_T::n_inplace]) {
    static_assert(CONFIG_T::strategy == nnet::resource, "io_stream EinsumDense requires Strategy=Resource");

    constexpr unsigned I = CONFIG_T::n_inplace;
    constexpr unsigned L0 = CONFIG_T::n_free_data;
    constexpr unsigned L1 = CONFIG_T::n_free_kernel;
    constexpr unsigned C = CONFIG_T::n_contract;
    constexpr unsigned row_count = L0;

    typename data_T::value_type data[CONFIG_T::tpose_inp_conf::N];

#pragma hls_pipeline_init_interval 1
DataPrepare:
    for (unsigned t = 0; t < CONFIG_T::tpose_inp_conf::N / data_T::size; t++) {
        data_T pack = data_stream.read();
    #pragma hls_unroll
    DataPack:
        for (unsigned p = 0; p < data_T::size; p++) {
            data[t * data_T::size + p] = pack[p];
        }
    }

    typename data_T::value_type inp_tpose[CONFIG_T::tpose_inp_conf::N]; // canonical (I, L0, C)
    nnet::transpose<typename data_T::value_type, typename data_T::value_type, typename CONFIG_T::tpose_inp_conf>(
        data, inp_tpose);

    typename res_T::value_type out_buffer[L1];

#pragma hls_pipeline_init_interval 1
RowLoop:
    for (unsigned slot = 0; slot < I * L0; slot++) {
        unsigned i, l0;
        if (CONFIG_T::row_major_i_outer) {
            i = slot / row_count;
            l0 = slot % row_count;
        } else {
            l0 = slot / I;
            i = slot % I;
        }

        nnet::dense<typename data_T::value_type, typename res_T::value_type, typename CONFIG_T::dense_conf>(
            &inp_tpose[(i * L0 + l0) * C], out_buffer, &weights[i * L1 * C], &biases[(i * L0 + l0) * L1]);

    WriteRow:
        for (unsigned i_out = 0; i_out < L1 / res_T::size; i_out++) {
            res_T res_pack;
        #pragma hls_unroll
        WritePack:
            for (unsigned i_pack = 0; i_pack < res_T::size; i_pack++) {
                res_pack[i_pack] = out_buffer[i_out * res_T::size + i_pack];
            }
            res_stream.write(res_pack);
        }
    }
}

} // namespace nnet

#endif
