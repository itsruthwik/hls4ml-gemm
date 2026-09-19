#ifndef NNET_EINSUM_H_
#define NNET_EINSUM_H_

#include "nnet_common.h"
#include "nnet_mult.h"
#include "nnet_transpose.h"

namespace nnet {

template <typename data0_T, typename data1_T, typename res_T, typename CONFIG_T>
void einsum_latency(const data0_T tpose_i0[CONFIG_T::tpose_inp0_config::N],
                     const data1_T tpose_i1[CONFIG_T::tpose_inp1_config::N],
                     res_T tpose_o[CONFIG_T::tpose_out_conf::N]) {

    // Vivado: function-level PIPELINE II=reuse_factor with every loop unrolled. Catapult has
    // no loop left to pipeline once all four are unrolled, so only the unrolls are expressed.

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
}

// Resource-strategy contraction core, ported from the Vivado backend (nnet_einsum.h there).
// Shares multipliers across ReuseLoop iterations like nnet::dense_resource_rf_leq_nin, with one
// structural rule the HLS scheduler depends on: every unrolled multiplier lane accumulates into
// its OWN partial sum (lane_acc[im], index static per lane), and only the operand index (the
// contraction position c) moves with the reuse counter ir. Lanes are reduced per output after the
// loop. Accumulating straight into acc[out] with an index derived from ir made hundreds of lanes
// alias the same registers through runtime indices each cycle on Vitis (RTL != C on multi-layer
// dataflow designs); the per-lane form has no such aliasing and is used here for the same reason.
//
// Numerics are identical to einsum_latency: each product is rounded into accum_t once
// (translation-invariant on the accum grid) and accum_t adds wrap, so the association order of
// the adds does not change the result.
//
// Requires reuse_factor <= n_contract and n_contract % reuse_factor == 0; the backend clamps
// ReuseFactor to a divisor of n_contract for Einsum layers.
template <typename data0_T, typename data1_T, typename res_T, typename CONFIG_T>
void einsum_resource(const data0_T tpose_i0[CONFIG_T::tpose_inp0_config::N],
                      const data1_T tpose_i1[CONFIG_T::tpose_inp1_config::N],
                      res_T tpose_o[CONFIG_T::tpose_out_conf::N]) {

    constexpr unsigned L0 = CONFIG_T::n_free0;
    constexpr unsigned L1 = CONFIG_T::n_free1;
    constexpr unsigned C = CONFIG_T::n_contract;
    constexpr unsigned I = CONFIG_T::n_inplace;

    constexpr unsigned n_out = I * L0 * L1;
    constexpr unsigned rufactor = CONFIG_T::reuse_factor;
    static_assert(rufactor >= 1 && rufactor <= C && C % rufactor == 0,
                  "einsum_resource: reuse_factor must divide n_contract");
    constexpr unsigned G = C / rufactor; // multiplier lanes per output
    constexpr unsigned n_lanes = n_out * G;

    typename CONFIG_T::accum_t lane_acc[n_lanes];

#pragma hls_unroll
InitAccum:
    for (unsigned im = 0; im < n_lanes; im++) {
        lane_acc[im] = 0;
    }

#pragma hls_pipeline_init_interval 1
ReuseLoop:
    for (unsigned ir = 0; ir < rufactor; ir++) {

    #pragma hls_unroll
    MultLoop:
        for (unsigned im = 0; im < n_lanes; im++) {

            const unsigned out = im / G; // static per lane
            const unsigned c = (im % G) * rufactor + ir;
            const unsigned l1 = out % L1;
            const unsigned tmp = out / L1;
            const unsigned l0 = tmp % L0;
            const unsigned i = tmp / L0;

            data0_T a = tpose_i0[(i * L0 + l0) * C + c];
            data1_T b = tpose_i1[i * L1 * C + l1 * C + c];
            auto mult = CONFIG_T::template product<data0_T, data1_T>::product(a, b);
            lane_acc[im] += mult;
        }
    }

#pragma hls_unroll
Result:
    for (unsigned out = 0; out < n_out; out++) {
        typename CONFIG_T::accum_t acc = lane_acc[out * G];
        for (unsigned g = 1; g < G; g++) {
            acc += lane_acc[out * G + g];
        }
        tpose_o[out] = acc;
    }
}

template <typename data0_T, typename data1_T, typename res_T, typename CONFIG_T>
void einsum(const data0_T data0[CONFIG_T::tpose_inp0_config::N], const data1_T data1[CONFIG_T::tpose_inp1_config::N],
            res_T res[CONFIG_T::tpose_out_conf::N]) {

    data0_T tpose_i0[CONFIG_T::tpose_inp0_config::N];
    data1_T tpose_i1[CONFIG_T::tpose_inp1_config::N];
    res_T tpose_o[CONFIG_T::tpose_out_conf::N];

    // Catapult does not use the Vivado ARRAY_PARTITION pragma syntax here.

    nnet::transpose<data0_T, data0_T, typename CONFIG_T::tpose_inp0_config>(data0, tpose_i0);
    nnet::transpose<data1_T, data1_T, typename CONFIG_T::tpose_inp1_config>(data1, tpose_i1);

    if (CONFIG_T::strategy == nnet::resource) {
        einsum_resource<data0_T, data1_T, res_T, CONFIG_T>(tpose_i0, tpose_i1, tpose_o);
    } else {
        einsum_latency<data0_T, data1_T, res_T, CONFIG_T>(tpose_i0, tpose_i1, tpose_o);
    }

    nnet::transpose<res_T, res_T, typename CONFIG_T::tpose_out_conf>(tpose_o, res);
}

} // namespace nnet


#endif
