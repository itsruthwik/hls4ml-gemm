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

// Row-streamed two-operand einsum (Resource strategy), ported from the Vivado backend's
// nnet_einsum_stream.h. Each output row is a Dense-style Resource matvec
//   out_row[o] = sum_c row[c] * buf[o * C + c]
// where buf is the OTHER operand, buffered once (with its transpose applied on the buffer) in
// the (n_rows_out, C) layout dense_resource_rf_leq_nin reads. The lane sharing is that kernel's
// ReuseLoop/MultLoop scheme. STREAM_OP says which einsum operand the row belongs to, so the
// product is formed with the operand types in their declared order.
template <typename data0_T, typename data1_T, typename res_T, typename CONFIG_T, unsigned N_OUT, unsigned STREAM_OP>
struct einsum_row_resource {
    typedef typename std::conditional<STREAM_OP == 0, data0_T, data1_T>::type row_T;
    typedef typename std::conditional<STREAM_OP == 0, data1_T, data0_T>::type buf_T;
    // Operand ordering helpers: the product template is <data0_T, data1_T>; hand it the row and
    // buffer elements in that order whichever operand is being streamed.
    template <unsigned OP = STREAM_OP>
    static typename std::enable_if<OP == 0, data0_T>::type pick0(const row_T &r, const buf_T &) { return r; }
    template <unsigned OP = STREAM_OP>
    static typename std::enable_if<OP != 0, data0_T>::type pick0(const row_T &, const buf_T &b) { return b; }
    template <unsigned OP = STREAM_OP>
    static typename std::enable_if<OP == 0, data1_T>::type pick1(const row_T &, const buf_T &b) { return b; }
    template <unsigned OP = STREAM_OP>
    static typename std::enable_if<OP != 0, data1_T>::type pick1(const row_T &r, const buf_T &) { return r; }
};

// Drain one whole packed stream into a flat array.
template <class pack_T, unsigned N>
void einsum_drain(ac_channel<pack_T> &in_stream, typename pack_T::value_type out[N]) {
#pragma hls_pipeline_init_interval 1
DrainLoop:
    for (unsigned i_in = 0; i_in < N / pack_T::size; i_in++) {
        pack_T data_pack = in_stream.read();
    #pragma hls_unroll
    DrainPack:
        for (unsigned i_pack = 0; i_pack < pack_T::size; i_pack++) {
            out[i_in * pack_T::size + i_pack] = data_pack[i_pack];
        }
    }
}

// io_stream Resource path: buffer one operand once (applying its transpose on the buffer), then
// stream the other operand one row at a time through einsum_row_resource and emit each output
// row as soon as it is done. The backend picks the streamed operand (CONFIG_T::row_stream_operand)
// from the effective output order: rows of operand 0 when the output is (L0, L1) contiguous, rows
// of operand 1 when it is (L1, L0). Size-1 axes are ignored in that decision. n_inplace must be 1.
template <class data0_T, class data1_T, class res_T, typename CONFIG_T, bool ROW_STREAM>
struct einsum_stream_impl;

template <class data0_T, class data1_T, class res_T, typename CONFIG_T>
struct einsum_stream_impl<data0_T, data1_T, res_T, CONFIG_T, true> {
    static constexpr unsigned OP = CONFIG_T::row_stream_operand;
    typedef typename std::conditional<OP == 0, data0_T, data1_T>::type row_pack_T;
    typedef typename std::conditional<OP == 0, data1_T, data0_T>::type buf_pack_T;
    static constexpr unsigned N_ROWS = OP == 0 ? CONFIG_T::n_free0 : CONFIG_T::n_free1;
    static constexpr unsigned N_OUT = OP == 0 ? CONFIG_T::n_free1 : CONFIG_T::n_free0;
    static constexpr unsigned C = CONFIG_T::n_contract;
    typedef typename std::conditional<OP == 0, typename CONFIG_T::tpose_inp1_config,
                                      typename CONFIG_T::tpose_inp0_config>::type buf_tpose_conf;
    static constexpr unsigned N_BUF = OP == 0 ? CONFIG_T::tpose_inp1_config::N : CONFIG_T::tpose_inp0_config::N;

    static void run(ac_channel<data0_T> &data0_stream, ac_channel<data1_T> &data1_stream,
                    ac_channel<res_T> &res_stream) {
        static_assert(CONFIG_T::n_inplace == 1, "einsum_stream_impl: n_inplace must be 1");
        static_assert(C % row_pack_T::size == 0,
                      "einsum_stream_impl: streamed operand pack size must divide n_contract");
        static_assert(N_OUT % res_T::size == 0,
                      "einsum_stream_impl: output pack size must divide the output row length");

        ac_channel<row_pack_T> &row_stream = select_stream<OP>(data0_stream, data1_stream);
        ac_channel<buf_pack_T> &buf_stream = select_stream<1 - OP>(data0_stream, data1_stream);

        typename buf_pack_T::value_type raw[N_BUF];
        typename buf_pack_T::value_type buf[N_OUT * C];

        einsum_drain<buf_pack_T, N_BUF>(buf_stream, raw);
        nnet::transpose<typename buf_pack_T::value_type, typename buf_pack_T::value_type, buf_tpose_conf>(raw, buf);

        // One flat pipelined loop over (row, reuse step). Reads a row on its first reuse step,
        // runs one lane pass per step, writes the row on its last step. Rows overlap in the
        // pipeline, so the module takes ~N_ROWS * reuse_factor cycles plus depth while keeping
        // only C / reuse_factor lanes per output (a separate row loop around a rolled reuse loop
        // would serialise rows and cost far more latency).
        constexpr unsigned rufactor = CONFIG_T::reuse_factor;
        static_assert(rufactor >= 1 && rufactor <= C && C % rufactor == 0,
                      "einsum_stream_impl: reuse_factor must divide n_contract");
        constexpr unsigned multscale = C / rufactor;         // lanes per output
        constexpr unsigned block_factor = N_OUT * multscale; // lanes
        typedef typename row_pack_T::value_type row_T;
        typedef einsum_row_resource<typename data0_T::value_type, typename data1_T::value_type,
                                    typename res_T::value_type, CONFIG_T, N_OUT, OP>
            ops;

        row_T row[C];
        typename CONFIG_T::accum_t acc[N_OUT];

        unsigned ir = 0;
    #pragma hls_pipeline_init_interval 1
    RowReuseLoop:
        for (unsigned t = 0; t < N_ROWS * rufactor; t++) {

            if (ir == 0) {
            #pragma hls_unroll
            ReadRow:
                for (unsigned i_in = 0; i_in < C / row_pack_T::size; i_in++) {
                    row_pack_T data_pack = row_stream.read();
                #pragma hls_unroll
                    for (unsigned i_pack = 0; i_pack < row_pack_T::size; i_pack++) {
                        row[i_in * row_pack_T::size + i_pack] = data_pack[i_pack];
                    }
                }
            #pragma hls_unroll
            InitAccum:
                for (unsigned o = 0; o < N_OUT; o++) {
                    acc[o] = 0;
                }
            }

            // Sum this step's lanes feed-forward first, so the loop-carried path through acc is a
            // single adder; chaining every lane into acc directly cannot be scheduled at II 1.
            typename CONFIG_T::accum_t step[N_OUT];
        #pragma hls_unroll
        InitStep:
            for (unsigned o = 0; o < N_OUT; o++) {
                step[o] = 0;
            }

        #pragma hls_unroll
        MultLoop:
            for (unsigned im = 0; im < block_factor; im++) {
                const unsigned out_index = im / multscale;           // static per lane
                const unsigned c = ir + rufactor * (im % multscale); // walks the contraction axis
                step[out_index] += CONFIG_T::template product<typename data0_T::value_type, typename data1_T::value_type>::product(
                    ops::pick0(row[c], buf[out_index * C + c]), ops::pick1(row[c], buf[out_index * C + c]));
            }

        #pragma hls_unroll
        StepAccum:
            for (unsigned o = 0; o < N_OUT; o++) {
                acc[o] += step[o];
            }

            if (ir == rufactor - 1) {
            #pragma hls_unroll
            WriteRow:
                for (unsigned i_out = 0; i_out < N_OUT / res_T::size; i_out++) {
                    res_T res_pack;
                #pragma hls_unroll
                    for (unsigned i_pack = 0; i_pack < res_T::size; i_pack++) {
                        res_pack[i_pack] = acc[i_out * res_T::size + i_pack];
                    }
                    res_stream.write(res_pack);
                }
                ir = 0;
            } else {
                ir++;
            }
        }
    }

  private:
    template <unsigned WHICH>
    static typename std::enable_if<WHICH == 0, ac_channel<data0_T> &>::type select_stream(ac_channel<data0_T> &s0,
                                                                                          ac_channel<data1_T> &) {
        return s0;
    }
    template <unsigned WHICH>
    static typename std::enable_if<WHICH == 1, ac_channel<data1_T> &>::type select_stream(ac_channel<data0_T> &,
                                                                                          ac_channel<data1_T> &s1) {
        return s1;
    }
};

// Fallback (Latency, or transposes/tilings that break row order): drain both operands, run the
// array core (nnet::einsum, which itself dispatches Latency vs Resource on CONFIG_T::strategy),
// re-stream the result.
template <class data0_T, class data1_T, class res_T, typename CONFIG_T>
struct einsum_stream_impl<data0_T, data1_T, res_T, CONFIG_T, false> {
    static void run(ac_channel<data0_T> &data0_stream, ac_channel<data1_T> &data1_stream,
                    ac_channel<res_T> &res_stream) {
        typename data0_T::value_type data0[CONFIG_T::tpose_inp0_config::N];
        typename data1_T::value_type data1[CONFIG_T::tpose_inp1_config::N];
        typename res_T::value_type res[CONFIG_T::tpose_out_conf::N];

        einsum_drain<data0_T, CONFIG_T::tpose_inp0_config::N>(data0_stream, data0);
        einsum_drain<data1_T, CONFIG_T::tpose_inp1_config::N>(data1_stream, data1);

        nnet::einsum<typename data0_T::value_type, typename data1_T::value_type, typename res_T::value_type, CONFIG_T>(
            data0, data1, res);

    #pragma hls_pipeline_init_interval 1
    ResWrite:
        for (unsigned i_out = 0; i_out < CONFIG_T::tpose_out_conf::N / res_T::size; i_out++) {
            res_T res_pack;
        #pragma hls_unroll
        ResPack:
            for (unsigned i_pack = 0; i_pack < res_T::size; i_pack++) {
                res_pack[i_pack] = res[i_out * res_T::size + i_pack];
            }
            res_stream.write(res_pack);
        }
    }
};

template <class data0_T, class data1_T, class res_T, typename CONFIG_T>
void einsum_stream(ac_channel<data0_T> &data0_stream, ac_channel<data1_T> &data1_stream,
                   ac_channel<res_T> &res_stream) {
    einsum_stream_impl<data0_T, data1_T, res_T, CONFIG_T,
                       (CONFIG_T::strategy == nnet::resource && CONFIG_T::row_stream)>::run(data0_stream, data1_stream,
                                                                                            res_stream);
}

} // namespace nnet

#endif
