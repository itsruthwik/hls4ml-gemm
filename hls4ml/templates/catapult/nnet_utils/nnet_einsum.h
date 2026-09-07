#ifndef NNET_EINSUM_H_
#define NNET_EINSUM_H_

#include "nnet_common.h"
#include "nnet_mult.h"
#include "nnet_transpose.h"

namespace nnet {

template <typename data0_T, typename data1_T, typename res_T, typename CONFIG_T>
void einsum(const data0_T data0[CONFIG_T::tpose_inp0_config::N], const data1_T data1[CONFIG_T::tpose_inp1_config::N],
            res_T res[CONFIG_T::tpose_out_conf::N]) {

    // Vivado: function-level PIPELINE II=reuse_factor with every loop unrolled. Catapult has
    // no loop left to pipeline once all four are unrolled, so only the unrolls are expressed.

    data0_T tpose_i0[CONFIG_T::tpose_inp0_config::N];
    data1_T tpose_i1[CONFIG_T::tpose_inp1_config::N];
    res_T tpose_o[CONFIG_T::tpose_out_conf::N];

    // Catapult does not use the Vivado ARRAY_PARTITION pragma syntax here.

    nnet::transpose<data0_T, data0_T, typename CONFIG_T::tpose_inp0_config>(data0, tpose_i0);
    nnet::transpose<data1_T, data1_T, typename CONFIG_T::tpose_inp1_config>(data1, tpose_i1);

    constexpr unsigned L0 = CONFIG_T::n_free0;
    constexpr unsigned L1 = CONFIG_T::n_free1;
    constexpr unsigned C = CONFIG_T::n_contract;
    constexpr unsigned I = CONFIG_T::n_inplace;

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
                    data0_T a = tpose_i0[(i * L0 + l0) * C + c];
                    data1_T b = tpose_i1[i * L1 * C + l1 * C + c];
                    accum_buf += CONFIG_T::template product<data0_T, data1_T>::product(a, b);
                }
                tpose_o[(i * L0 + l0) * L1 + l1] = accum_buf;
            }
        }
    }

    nnet::transpose<res_T, res_T, typename CONFIG_T::tpose_out_conf>(tpose_o, res);
}

} // namespace nnet


#endif
