#ifndef NNET_EINSUM_DENSE_STREAM_H_
#define NNET_EINSUM_DENSE_STREAM_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_einsum_dense.h"
#include "nnet_einsum_stream.h"
#include "nnet_mult.h"
#include <type_traits>

namespace nnet {

// io_stream EinsumDense, Resource strategy. Weights are constant, so unlike
// nnet_einsum_stream.h's two-operand kernel there is only one operand to stream in: the data
// input. The kernel unpacks it into a flat array (DataPrepare, as in nnet_dense_stream.h) and runs
// it through nnet::transpose with the already-generated tpose_inp_conf -- the same config and the
// same canonical (I, L0, C) layout the io_parallel array core (nnet_einsum_dense.h) uses. It then
// computes one row (one (i, l0) pair) at a time by calling nnet::dense<>() with CONFIG_T::dense_conf
// -- the exact same call nnet_einsum_dense.h's io_parallel core makes per free-data index -- so
// weights stay in the single (I, L1, C) Resource layout for both io types (ApplyResourceStrategy no
// longer has an io_stream-only permutation branch) and bias add / cast<>() happen inside that same
// dense kernel, giving bit-exact results with io_parallel.
template <class data_T, class res_T, typename CONFIG_T>
typename std::enable_if<CONFIG_T::strategy == nnet::resource, void>::type einsum_dense(
    hls::stream<data_T> &data_stream, hls::stream<res_T> &res_stream,
    typename CONFIG_T::dense_conf::weight_t weights[CONFIG_T::n_free_kernel * CONFIG_T::n_contract * CONFIG_T::n_inplace],
    typename CONFIG_T::dense_conf::bias_t biases[CONFIG_T::n_free_data * CONFIG_T::n_free_kernel * CONFIG_T::n_inplace]) {
    constexpr unsigned I = CONFIG_T::n_inplace;
    constexpr unsigned L0 = CONFIG_T::n_free_data;
    constexpr unsigned L1 = CONFIG_T::n_free_kernel;
    constexpr unsigned C = CONFIG_T::n_contract;
    constexpr unsigned row_count = L0;

    typename data_T::value_type data[CONFIG_T::tpose_inp_conf::N];
    #pragma HLS ARRAY_PARTITION variable = data complete

DataPrepare:
    for (unsigned t = 0; t < CONFIG_T::tpose_inp_conf::N / data_T::size; t++) {
        if (CONFIG_T::tpose_inp_conf::N / data_T::size > 1) {
            #pragma HLS PIPELINE II = 1
        }
        data_T pack = data_stream.read();
    DataPack:
        for (unsigned p = 0; p < data_T::size; p++) {
            #pragma HLS UNROLL
            data[t * data_T::size + p] = pack[p];
        }
    }

    typename data_T::value_type inp_tpose[CONFIG_T::tpose_inp_conf::N]; // canonical (I, L0, C)
    #pragma HLS ARRAY_PARTITION variable = inp_tpose complete
    nnet::transpose<typename data_T::value_type, typename data_T::value_type, typename CONFIG_T::tpose_inp_conf>(
        data, inp_tpose);

    typename res_T::value_type out_buffer[L1];
    #pragma HLS ARRAY_PARTITION variable = out_buffer complete

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
            PRAGMA_DATA_PACK(res_pack)
            for (unsigned i_pack = 0; i_pack < res_T::size; i_pack++) {
                #pragma HLS UNROLL
                res_pack[i_pack] = out_buffer[i_out * res_T::size + i_pack];
            }
            res_stream.write(res_pack);
        }
    }
}

// io_stream EinsumDense, Latency strategy (the recommended strategy at ReuseFactor 1). Each row is
// one call of the Latency dense core (nnet::dense_latency: a single function-level pipeline at
// II = ReuseFactor, fully unrolled), so consecutive rows overlap in one row loop pipelined at
// II = ReuseFactor; a row costs one II, not a pipeline fill and drain. Weights stay in the natural
// (I, C, L1) layout the Latency core reads (ApplyResourceStrategy transposes Resource weights only).
//
// direct_rows (set at conversion when the input stream already delivers each row's C values
// contiguously, in row order, as whole beats, and rows leave in that order): each row's beats are
// read, computed and written in the same loop iteration, with no frame buffer -- rows and frames
// flow through back to back. Otherwise the frame is buffered and transposed into canonical
// (I, L0, C) order first, as in the Resource kernel, and the rows are then computed back to back.
template <class data_T, class res_T, typename CONFIG_T>
typename std::enable_if<CONFIG_T::strategy == nnet::latency, void>::type einsum_dense(
    hls::stream<data_T> &data_stream, hls::stream<res_T> &res_stream,
    typename CONFIG_T::dense_conf::weight_t weights[CONFIG_T::n_free_kernel * CONFIG_T::n_contract * CONFIG_T::n_inplace],
    typename CONFIG_T::dense_conf::bias_t biases[CONFIG_T::n_free_data * CONFIG_T::n_free_kernel * CONFIG_T::n_inplace]) {
    constexpr unsigned I = CONFIG_T::n_inplace;
    constexpr unsigned L0 = CONFIG_T::n_free_data;
    constexpr unsigned L1 = CONFIG_T::n_free_kernel;
    constexpr unsigned C = CONFIG_T::n_contract;
    constexpr unsigned n_rows = I * L0;

    typename data_T::value_type inp_tpose[CONFIG_T::direct_rows ? 1 : CONFIG_T::tpose_inp_conf::N];
    #pragma HLS ARRAY_PARTITION variable = inp_tpose complete
    if (!CONFIG_T::direct_rows) {
        typename data_T::value_type data[CONFIG_T::tpose_inp_conf::N];
        #pragma HLS ARRAY_PARTITION variable = data complete
    DataPrepareL:
        for (unsigned t = 0; t < CONFIG_T::tpose_inp_conf::N / data_T::size; t++) {
            #pragma HLS PIPELINE II = 1
            data_T pack = data_stream.read();
            for (unsigned p = 0; p < data_T::size; p++) {
                #pragma HLS UNROLL
                data[t * data_T::size + p] = pack[p];
            }
        }
        nnet::transpose<typename data_T::value_type, typename data_T::value_type, typename CONFIG_T::tpose_inp_conf>(
            data, inp_tpose);
    }

RowLoopL:
    for (unsigned slot = 0; slot < n_rows; slot++) {
        #pragma HLS PIPELINE II = CONFIG_T::reuse_factor
        unsigned i, l0;
        if (CONFIG_T::row_major_i_outer) {
            i = slot / L0;
            l0 = slot % L0;
        } else {
            l0 = slot / I;
            i = slot % I;
        }

        typename data_T::value_type row_in[C];
        #pragma HLS ARRAY_PARTITION variable = row_in complete
        if (CONFIG_T::direct_rows) {
        ReadRow:
            for (unsigned b = 0; b < C / data_T::size; b++) {
                data_T pack = data_stream.read();
                for (unsigned p = 0; p < data_T::size; p++) {
                    #pragma HLS UNROLL
                    row_in[b * data_T::size + p] = pack[p];
                }
            }
        } else {
            for (unsigned c = 0; c < C; c++) {
                #pragma HLS UNROLL
                row_in[c] = inp_tpose[(i * L0 + l0) * C + c];
            }
        }

        typename CONFIG_T::dense_conf::bias_t row_bias[L1];
        #pragma HLS ARRAY_PARTITION variable = row_bias complete
        for (unsigned j = 0; j < L1; j++) {
            #pragma HLS UNROLL
            row_bias[j] = biases[(i * L0 + l0) * L1 + j];
        }

        typename res_T::value_type out_buffer[L1];
        #pragma HLS ARRAY_PARTITION variable = out_buffer complete
        nnet::dense<typename data_T::value_type, typename res_T::value_type, typename CONFIG_T::dense_conf>(
            row_in, out_buffer, &weights[i * C * L1], row_bias);

    WriteRowL:
        for (unsigned i_out = 0; i_out < L1 / res_T::size; i_out++) {
            res_T res_pack;
            PRAGMA_DATA_PACK(res_pack)
            for (unsigned i_pack = 0; i_pack < res_T::size; i_pack++) {
                #pragma HLS UNROLL
                res_pack[i_pack] = out_buffer[i_out * res_T::size + i_pack];
            }
            res_stream.write(res_pack);
        }
    }
}

} // namespace nnet

#endif
