#ifndef NNET_EINSUM_STREAM_H_
#define NNET_EINSUM_STREAM_H_

#include "nnet_common.h"
#include "nnet_mult.h"
#include "nnet_transpose.h"
#include "nnet_types.h"
#include <ac_channel.h>

namespace nnet {

// Streaming einsum for the 2-operand matmul case (attention QK^T / A.V).
//
// I/O are ac_channel<nnet::array<...>> so the layer composes with io_stream
// neighbours and presents NARROW ports (one beat at a time) instead of the full
// io_parallel matrices. The contraction is buffered internally and is computed
// with the EXACT same math as nnet::einsum (io_parallel) -> bit-exact. Both input
// streams are drained fully (packing-agnostic) before the contraction; this also
// breaks the two-input ordering hazard (size the input channel FIFOs >= their
// beat counts; tiny for attention S <= 32). Per-row pipelining is a later
// refinement and not required for correctness/coverage.
template <typename data0_T, typename data1_T, typename res_T, typename CONFIG_T>
void einsum_stream(ac_channel<data0_T> &data0_stream, ac_channel<data1_T> &data1_stream,
                   ac_channel<res_T> &res_stream) {

    typedef typename data0_T::value_type a_T;
    typedef typename data1_T::value_type b_T;
    typedef typename res_T::value_type c_T;

    constexpr unsigned N0 = CONFIG_T::tpose_inp0_config::N;
    constexpr unsigned N1 = CONFIG_T::tpose_inp1_config::N;
    constexpr unsigned NO = CONFIG_T::tpose_out_conf::N;

    a_T raw0[N0];
    b_T raw1[N1];

    // Drain both input streams fully into flat buffers (packing-agnostic).
    #pragma hls_pipeline_init_interval 1
ReadInp0:
    for (unsigned i = 0; i < N0 / data0_T::size; i++) {
        data0_T beat = data0_stream.read();
        #pragma hls_unroll
        for (unsigned p = 0; p < data0_T::size; p++) {
            raw0[i * data0_T::size + p] = beat[p];
        }
    }
    #pragma hls_pipeline_init_interval 1
ReadInp1:
    for (unsigned i = 0; i < N1 / data1_T::size; i++) {
        data1_T beat = data1_stream.read();
        #pragma hls_unroll
        for (unsigned p = 0; p < data1_T::size; p++) {
            raw1[i * data1_T::size + p] = beat[p];
        }
    }

    a_T tpose_i0[N0];
    b_T tpose_i1[N1];
    nnet::transpose<a_T, a_T, typename CONFIG_T::tpose_inp0_config>(raw0, tpose_i0);
    nnet::transpose<b_T, b_T, typename CONFIG_T::tpose_inp1_config>(raw1, tpose_i1);

    constexpr unsigned L0 = CONFIG_T::n_free0;
    constexpr unsigned L1 = CONFIG_T::n_free1;
    constexpr unsigned C = CONFIG_T::n_contract;
    constexpr unsigned I = CONFIG_T::n_inplace;

    // Contraction: identical structure/pragmas to nnet::einsum (bit-exact).
    c_T tpose_o[NO];
    typename CONFIG_T::accum_t accum_buf;
    #pragma hls_unroll
    for (unsigned i = 0; i < I; i++) {
        #pragma hls_unroll
        for (unsigned l0 = 0; l0 < L0; l0++) {
            #pragma hls_unroll
            for (unsigned l1 = 0; l1 < L1; l1++) {
                accum_buf = 0;
                #pragma hls_unroll
                for (unsigned c = 0; c < C; c++) {
                    a_T a = tpose_i0[(i * L0 + l0) * C + c];
                    b_T b = tpose_i1[i * L1 * C + l1 * C + c];
                    accum_buf += CONFIG_T::template product<a_T, b_T>::product(a, b);
                }
                tpose_o[(i * L0 + l0) * L1 + l1] = accum_buf;
            }
        }
    }

    c_T out_flat[NO];
    nnet::transpose<c_T, c_T, typename CONFIG_T::tpose_out_conf>(tpose_o, out_flat);

    // Emit output beats in the same packing the writer expects.
    #pragma hls_pipeline_init_interval 1
WriteOut:
    for (unsigned i = 0; i < NO / res_T::size; i++) {
        res_T beat;
        #pragma hls_unroll
        for (unsigned p = 0; p < res_T::size; p++) {
            beat[p] = out_flat[i * res_T::size + p];
        }
        res_stream.write(beat);
    }
}

// NOTE: the io_stream einsum GEMM path (einsum_gemm_ip_stream) was retired. The
// GEMM lowering now happens in the IR (Einsum -> Gemm node), so the four gemm_*
// entry points are the only GEMM functions. This header keeps only the baseline
// streaming einsum above.


} // namespace nnet

#endif
