#ifndef NNET_EINSUM_H_
#define NNET_EINSUM_H_

#include "nnet_common.h"
#include "nnet_gemm_ip.h"
#include "nnet_mult.h"
#include "nnet_transpose.h"

namespace nnet {

struct config_einsum {
    typedef void tpose_inp0_config;
    typedef void tpose_inp1_config;
    typedef void tpose_out_conf;

    // Layer Sizes
    static const unsigned n_free0;
    static const unsigned n_free1;
    static const unsigned n_contract;
    static const unsigned n_inplace;

    // Resource reuse info
    static const unsigned io_type;
    static const unsigned strategy;
    static const unsigned reuse_factor;
    static const unsigned multiplier_limit;

    template <class x_T, class y_T> using product = nnet::product::mult<x_T, y_T>;
};

template <typename data0_T, typename data1_T, typename res_T, typename CONFIG_T>
void einsum_latency(const data0_T tpose_i0[CONFIG_T::tpose_inp0_config::N],
                     const data1_T tpose_i1[CONFIG_T::tpose_inp1_config::N],
                     res_T tpose_o[CONFIG_T::tpose_out_conf::N]) {

    #pragma HLS PIPELINE II = CONFIG_T::reuse_factor
    #pragma HLS ALLOCATION operation instances = mul limit = CONFIG_T::multiplier_limit

    // for l0 in range(L0):
    //     for i in range(I):
    //             output[(i*L0+l0)*L1:(i*L0+l0+1)*L1] = input1[i*L1*C:(i+1)*L1*C].reshape((L1,C)) @
    //             input0[(i*L0+l0)*C:(i*L0+l0+1)*C]

    constexpr unsigned L0 = CONFIG_T::n_free0;
    constexpr unsigned L1 = CONFIG_T::n_free1;
    constexpr unsigned C = CONFIG_T::n_contract;
    constexpr unsigned I = CONFIG_T::n_inplace;

    typename CONFIG_T::accum_t accum_buf;
    for (unsigned i = 0; i < I; i++) {
        #pragma HLS UNROLL
        for (unsigned l0 = 0; l0 < L0; l0++) {
            #pragma HLS UNROLL
            for (unsigned l1 = 0; l1 < L1; l1++) {
                #pragma HLS UNROLL
                accum_buf = 0;
                for (unsigned c = 0; c < C; c++) {
                    #pragma HLS UNROLL
                    data0_T a = tpose_i0[(i * L0 + l0) * C + c];
                    data1_T b = tpose_i1[i * L1 * C + l1 * C + c];
                    accum_buf += CONFIG_T::template product<data0_T, data1_T>::product(a, b);
                }
                tpose_o[(i * L0 + l0) * L1 + l1] = accum_buf;
            }
        }
    }
}

// Resource-strategy contraction core. Shares multipliers across ReuseLoop iterations
// like nnet::dense_resource_rf_leq_nin, with one structural rule that the HLS
// scheduler depends on: every unrolled multiplier lane accumulates into its OWN
// partial sum (lane_acc[im], index static per lane), and only the operand index
// (the contraction position c) moves with the reuse counter ir. Lanes are reduced
// per output after the loop. Accumulating straight into acc[out] with an index
// derived from ir made hundreds of lanes alias the same registers through
// runtime indices each cycle; csim serialises those read-modify-writes correctly
// but Vitis scheduled them as independent, giving RTL != C on multi-layer
// dataflow designs (mha_small, 2026-09). The per-lane form has no such aliasing.
//
// Numerics are identical to einsum_latency: each product is rounded into accum_t
// once (translation-invariant on the accum grid) and accum_t adds wrap, so the
// association order of the adds does not change the result. (A saturating
// accum_t would make the order observable; hls4ml accumulators wrap.)
//
// Requires reuse_factor <= n_contract and n_contract % reuse_factor == 0; the
// backend clamps ReuseFactor to a divisor of n_contract for Einsum layers.
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
    #pragma HLS ARRAY_PARTITION variable = lane_acc complete

InitAccum:
    for (unsigned im = 0; im < n_lanes; im++) {
        #pragma HLS UNROLL
        lane_acc[im] = 0;
    }

ReuseLoop:
    for (unsigned ir = 0; ir < rufactor; ir++) {
        #pragma HLS PIPELINE II = 1

    MultLoop:
        for (unsigned im = 0; im < n_lanes; im++) {
            #pragma HLS UNROLL

            const unsigned out = im / G; // static per lane
            const unsigned c = (im % G) * rufactor + ir;
            const unsigned l1 = out % L1;
            const unsigned tmp = out / L1;
            const unsigned l0 = tmp % L0;
            const unsigned i = tmp / L0;

            data0_T a = tpose_i0[(i * L0 + l0) * C + c];
            data1_T b = tpose_i1[i * L1 * C + l1 * C + c];
            // Natural-width product, quantised to accum_t only when it lands in lane_acc (same as
            // einsum_latency). Casting to accum_t here would split the multiply from the add and
            // Vitis then builds a standalone 8x8 multiplier in fabric instead of a fused DSP MAC.
            auto mult = CONFIG_T::template product<data0_T, data1_T>::product(a, b);
            lane_acc[im] += mult;
        }
    }

Result:
    for (unsigned out = 0; out < n_out; out++) {
        #pragma HLS UNROLL
        typename CONFIG_T::accum_t acc = lane_acc[out * G];
        for (unsigned g = 1; g < G; g++) {
            #pragma HLS UNROLL
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

    #pragma HLS ARRAY_PARTITION variable = tpose_i0 complete
    #pragma HLS ARRAY_PARTITION variable = tpose_i1 complete
    #pragma HLS ARRAY_PARTITION variable = tpose_o complete

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
