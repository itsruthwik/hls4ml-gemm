#ifndef NNET_EINSUM_STREAM_H_
#define NNET_EINSUM_STREAM_H_

#include "nnet_common.h"
#include "nnet_einsum.h"
#include "nnet_mult.h"
#include "nnet_transpose.h"
#include "nnet_types.h"
#include <ac_channel.h>
#include <type_traits>

namespace nnet {

// io_stream Einsum, Resource strategy only (io_stream Latency is rejected at conversion; see
// init_einsum in the Catapult backend). One kernel, no drain-to-array-core fallback and no
// buffered output path: every row streams out the moment it is computed. The Catapult Einsum pass
// (equation_row_plan, shared with the Vivado/Vitis backend -- see
// hls4ml.backends.fpga.einsum_utils) derives, purely from the equation's index letters, which
// operand supplies rows and how the row loop nests; conversion raises before codegen if neither
// operand's rows can be streamed directly. Two phases, ported from the Vivado/Vitis kernel of the
// same name:
//
// 1. DataPrepare: unpack both operand streams into flat arrays, then run each through
//    nnet::transpose with the already-generated tpose_inp0_config / tpose_inp1_config -- the same
//    configs and the same canonical (I, L, C) layout the io_parallel array core (nnet::einsum in
//    nnet_einsum.h) uses. The two operand streams have no data dependency on each other, so they
//    are read from one fused loop that runs to the longer operand's beat count and reads each
//    stream only while it still has beats left, rather than draining one operand's stream fully
//    before starting the other, which would deadlock against an upstream producer filling both
//    channels concurrently.
//
// 2. A row is CONFIG_T::row_stream_op's operand's free axis fixed at one value (and i fixed),
//    giving the other operand's whole free axis as outputs: op 0 -> row (i, l0), n_free1 outputs;
//    op 1 -> row (i, l1), n_free0 outputs. Row loop flattened with the reuse step (ir) in one
//    pipelined II=1 loop, reusing einsum_resource's lane-sharing scheme: OUT_LEN * (C / RF) lanes,
//    each with its own accumulator (out index static per lane), walking C with the reuse counter,
//    indexing the transposed flat arrays with the same (i, l, c) -> offset arithmetic
//    einsum_resource uses. Catapult does not share unrolled multipliers just because the II
//    allows it, so this reuse walk (im % multscale) is what makes the lane sharing explicit; an
//    II=1 loop alone would not do it. The flattened row counter decodes directly into (i, row_l)
//    -- CONFIG_T::row_major_i_outer picks i*ROW_COUNT+row_l or row_l*n_inplace+i, whichever the
//    equation's output index order calls for (see equation_row_plan) -- so rows are visited,
//    computed and streamed out in exactly the order the output stream needs.
template <class data0_T, class data1_T, class res_T, typename CONFIG_T, unsigned OP> struct einsum_stream_impl;

// Unpack both operand streams into flat arrays, one fused loop running to the longer operand's
// beat count so neither stream is drained before the other.
template <class data0_T, class data1_T, typename CONFIG_T>
void einsum_stream_read(ac_channel<data0_T> &data0_stream, ac_channel<data1_T> &data1_stream,
                        typename data0_T::value_type data0[CONFIG_T::tpose_inp0_config::N],
                        typename data1_T::value_type data1[CONFIG_T::tpose_inp1_config::N]) {
    constexpr unsigned n0_beats = CONFIG_T::tpose_inp0_config::N / data0_T::size;
    constexpr unsigned n1_beats = CONFIG_T::tpose_inp1_config::N / data1_T::size;
    constexpr unsigned n_beats = n0_beats > n1_beats ? n0_beats : n1_beats;
#pragma hls_pipeline_init_interval 1
DataPrepare:
    for (unsigned t = 0; t < n_beats; t++) {
        if (t < n0_beats) {
            data0_T pack = data0_stream.read();
        #pragma hls_unroll
        DataPack0:
            for (unsigned p = 0; p < data0_T::size; p++) {
                data0[t * data0_T::size + p] = pack[p];
            }
        }
        if (t < n1_beats) {
            data1_T pack = data1_stream.read();
        #pragma hls_unroll
        DataPack1:
            for (unsigned p = 0; p < data1_T::size; p++) {
                data1[t * data1_T::size + p] = pack[p];
            }
        }
    }
}

// Shared row/reuse compute loop: tpose_i0/tpose_i1 are the transposed, canonical (I, L, C)-flat
// arrays nnet::transpose produced (the same layout and the same (i, l0/l1, c) -> flat-offset
// arithmetic as einsum_resource in nnet_einsum.h). ROW_IS_OP0 selects which operand supplies the
// row (its free index fixed at row_l) and which is fully read out per row (its whole free axis,
// OUT_LEN long).
template <class data0_T, class data1_T, class res_T, typename CONFIG_T, unsigned OUT_LEN, unsigned ROW_IS_OP0>
void einsum_stream_rows(typename data0_T::value_type tpose_i0[CONFIG_T::tpose_inp0_config::N],
                        typename data1_T::value_type tpose_i1[CONFIG_T::tpose_inp1_config::N],
                        ac_channel<res_T> &res_stream) {
    constexpr unsigned L0 = CONFIG_T::n_free0;
    constexpr unsigned L1 = CONFIG_T::n_free1;
    constexpr unsigned C = CONFIG_T::n_contract;
    constexpr unsigned rufactor = CONFIG_T::reuse_factor;
    constexpr unsigned multscale = C / rufactor; // lanes per output
    constexpr unsigned block_factor = OUT_LEN * multscale;
    constexpr unsigned N_ROWS = (ROW_IS_OP0 ? CONFIG_T::n_free0 : CONFIG_T::n_free1) * CONFIG_T::n_inplace;

    typedef typename data0_T::value_type in0_T;
    typedef typename data1_T::value_type in1_T;

    // Per-lane accumulators (einsum_resource's scheme, nnet_einsum.h): each of the block_factor
    // lanes owns its own register, so the loop-carried path from one RowReuseLoop iteration to the
    // next is a single multiply-add, not a multscale-wide adder chain feeding one shared acc[out]
    // register.
    typename CONFIG_T::accum_t lane_acc[block_factor];

    unsigned ir = 0;
    unsigned i = 0, row_l = 0;
#pragma hls_pipeline_init_interval 1
RowReuseLoop:
    for (unsigned t = 0; t < N_ROWS * rufactor; t++) {

        if (ir == 0) {
            unsigned slot = t / rufactor;
            constexpr unsigned row_count = ROW_IS_OP0 ? CONFIG_T::n_free0 : CONFIG_T::n_free1;
            if (CONFIG_T::row_major_i_outer) {
                i = slot / row_count;
                row_l = slot % row_count;
            } else {
                row_l = slot / CONFIG_T::n_inplace;
                i = slot % CONFIG_T::n_inplace;
            }
        }

    #pragma hls_unroll
    MultLoop:
        for (unsigned im = 0; im < block_factor; im++) {
            const unsigned out = im / multscale;                  // static per lane
            const unsigned c = ir + rufactor * (im % multscale); // walks the contraction axis
            typename CONFIG_T::accum_t mult;
            if (ROW_IS_OP0) {
                in0_T a = tpose_i0[(i * L0 + row_l) * C + c];
                in1_T b = tpose_i1[i * L1 * C + out * C + c];
                mult = CONFIG_T::template product<in0_T, in1_T>::product(a, b);
            } else {
                in0_T a = tpose_i0[(i * L0 + out) * C + c];
                in1_T b = tpose_i1[i * L1 * C + row_l * C + c];
                mult = CONFIG_T::template product<in0_T, in1_T>::product(a, b);
            }
            // Loop-carried path is exactly this one add (mirrors einsum_resource): lane_acc[im]
            // is cleared once, on the first reuse step of each row, by starting the accumulation
            // from `mult` alone instead of `lane_acc[im] + mult`.
            lane_acc[im] = (ir == 0) ? mult : (typename CONFIG_T::accum_t)(lane_acc[im] + mult);
        }

        if (ir == rufactor - 1) {
            res_T res_pack;
        #pragma hls_unroll
        WriteRow:
            for (unsigned i_out = 0; i_out < OUT_LEN / res_T::size; i_out++) {
                for (unsigned i_pack = 0; i_pack < res_T::size; i_pack++) {
                    unsigned out = i_out * res_T::size + i_pack;
                    typename CONFIG_T::accum_t acc = lane_acc[out * multscale];
                Reduce:
                    for (unsigned g = 1; g < multscale; g++) {
                        acc += lane_acc[out * multscale + g];
                    }
                    res_pack[i_pack] = acc;
                }
                res_stream.write(res_pack);
            }
            ir = 0;
        } else {
            ir++;
        }
    }
}

// row_stream_op == 0: rows are (i, l0), each n_free1 outputs. Operand 0 is the row array, operand
// 1 is the buffered array (all n_free1 outputs read every cycle).
template <class data0_T, class data1_T, class res_T, typename CONFIG_T>
struct einsum_stream_impl<data0_T, data1_T, res_T, CONFIG_T, 0> {
    static void run(ac_channel<data0_T> &data0_stream, ac_channel<data1_T> &data1_stream,
                    ac_channel<res_T> &res_stream) {
        static_assert(CONFIG_T::strategy == nnet::resource, "io_stream Einsum requires Strategy=Resource");
        static_assert(CONFIG_T::reuse_factor >= 1 && CONFIG_T::reuse_factor <= CONFIG_T::n_contract &&
                          CONFIG_T::n_contract % CONFIG_T::reuse_factor == 0,
                      "einsum stream: reuse_factor must divide n_contract");
        static_assert(CONFIG_T::n_free1 % res_T::size == 0, "einsum stream: output pack size must divide the row length");

        typename data0_T::value_type data0[CONFIG_T::tpose_inp0_config::N];
        typename data1_T::value_type data1[CONFIG_T::tpose_inp1_config::N];
        typename data0_T::value_type tpose_i0[CONFIG_T::tpose_inp0_config::N]; // row array
        typename data1_T::value_type tpose_i1[CONFIG_T::tpose_inp1_config::N]; // buffered

        einsum_stream_read<data0_T, data1_T, CONFIG_T>(data0_stream, data1_stream, data0, data1);
        nnet::transpose<typename data0_T::value_type, typename data0_T::value_type,
                        typename CONFIG_T::tpose_inp0_config>(data0, tpose_i0);
        nnet::transpose<typename data1_T::value_type, typename data1_T::value_type,
                        typename CONFIG_T::tpose_inp1_config>(data1, tpose_i1);
        einsum_stream_rows<data0_T, data1_T, res_T, CONFIG_T, CONFIG_T::n_free1, 1>(tpose_i0, tpose_i1, res_stream);
    }
};

// row_stream_op == 1: rows are (i, l1), each n_free0 outputs. Mirror of the above with the
// operand roles swapped.
template <class data0_T, class data1_T, class res_T, typename CONFIG_T>
struct einsum_stream_impl<data0_T, data1_T, res_T, CONFIG_T, 1> {
    static void run(ac_channel<data0_T> &data0_stream, ac_channel<data1_T> &data1_stream,
                    ac_channel<res_T> &res_stream) {
        static_assert(CONFIG_T::strategy == nnet::resource, "io_stream Einsum requires Strategy=Resource");
        static_assert(CONFIG_T::reuse_factor >= 1 && CONFIG_T::reuse_factor <= CONFIG_T::n_contract &&
                          CONFIG_T::n_contract % CONFIG_T::reuse_factor == 0,
                      "einsum stream: reuse_factor must divide n_contract");
        static_assert(CONFIG_T::n_free0 % res_T::size == 0, "einsum stream: output pack size must divide the row length");

        typename data0_T::value_type data0[CONFIG_T::tpose_inp0_config::N];
        typename data1_T::value_type data1[CONFIG_T::tpose_inp1_config::N];
        typename data0_T::value_type tpose_i0[CONFIG_T::tpose_inp0_config::N]; // buffered
        typename data1_T::value_type tpose_i1[CONFIG_T::tpose_inp1_config::N]; // row array

        einsum_stream_read<data0_T, data1_T, CONFIG_T>(data0_stream, data1_stream, data0, data1);
        nnet::transpose<typename data0_T::value_type, typename data0_T::value_type,
                        typename CONFIG_T::tpose_inp0_config>(data0, tpose_i0);
        nnet::transpose<typename data1_T::value_type, typename data1_T::value_type,
                        typename CONFIG_T::tpose_inp1_config>(data1, tpose_i1);
        einsum_stream_rows<data0_T, data1_T, res_T, CONFIG_T, CONFIG_T::n_free0, 0>(tpose_i0, tpose_i1, res_stream);
    }
};

template <class data0_T, class data1_T, class res_T, typename CONFIG_T>
void einsum(ac_channel<data0_T> &data0_stream, ac_channel<data1_T> &data1_stream, ac_channel<res_T> &res_stream) {
    einsum_stream_impl<data0_T, data1_T, res_T, CONFIG_T, CONFIG_T::row_stream_op>::run(data0_stream, data1_stream,
                                                                                        res_stream);
}

} // namespace nnet

#endif
