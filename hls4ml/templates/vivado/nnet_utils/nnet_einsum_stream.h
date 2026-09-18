#ifndef NNET_EINSUM_STREAM_H_
#define NNET_EINSUM_STREAM_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_einsum.h"
#include <type_traits>

namespace nnet {

// Row-streamed two-operand einsum: each output row is a Dense-style Resource matvec
//   out_row[o] = sum_c row[c] * buf[o * C + c]          (buf is (n_rows_out, C), i.e. the
// (n_out, n_in) layout dense_resource_rf_leq_nin reads). The lane sharing is that kernel's
// ReuseLoop/MultLoop scheme, copied without its constant-weight pragmas (function_instantiate,
// ARRAY_RESHAPE, ROM resource) because buf is runtime data drained from the other input stream.
// Requires reuse_factor <= C and C % reuse_factor == 0 (backend clamps ReuseFactor to a divisor
// of n_contract for Einsum layers), giving C / reuse_factor lanes per output.
// STREAM_OP says which einsum operand the row belongs to, so the product is formed with the
// operand types in their declared order.
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
void einsum_drain(hls::stream<pack_T> &in_stream, typename pack_T::value_type out[N]) {
DrainLoop:
    for (unsigned i_in = 0; i_in < N / pack_T::size; i_in++) {
        if (N / pack_T::size > 1) {
            #pragma HLS PIPELINE
        }
        pack_T data_pack = in_stream.read();
    DrainPack:
        for (unsigned i_pack = 0; i_pack < pack_T::size; i_pack++) {
            #pragma HLS UNROLL
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

    static void run(hls::stream<data0_T> &data0_stream, hls::stream<data1_T> &data1_stream,
                    hls::stream<res_T> &res_stream) {
        static_assert(CONFIG_T::n_inplace == 1, "einsum_stream_rows: n_inplace must be 1");
        static_assert(C % row_pack_T::size == 0, "einsum_stream_rows: streamed operand pack size must divide n_contract");
        static_assert(N_OUT % res_T::size == 0, "einsum_stream_rows: output pack size must divide the output row length");

        hls::stream<row_pack_T> &row_stream = select_stream<OP>(data0_stream, data1_stream);
        hls::stream<buf_pack_T> &buf_stream = select_stream<1 - OP>(data0_stream, data1_stream);

        typename buf_pack_T::value_type raw[N_BUF];
        #pragma HLS ARRAY_PARTITION variable = raw complete
        typename buf_pack_T::value_type buf[N_OUT * C];
        #pragma HLS ARRAY_PARTITION variable = buf complete

        einsum_drain<buf_pack_T, N_BUF>(buf_stream, raw);
        nnet::transpose<typename buf_pack_T::value_type, typename buf_pack_T::value_type, buf_tpose_conf>(raw, buf);

        // One flat pipelined loop over (row, reuse step). Reads a row on its first reuse step,
        // runs one lane pass per step, writes the row on its last step. Rows overlap in the
        // pipeline, so the module takes ~N_ROWS * reuse_factor cycles plus depth while keeping
        // only C / reuse_factor lanes per output (a separate row loop around a rolled reuse
        // loop serialised rows and cost ~10x the latency).
        constexpr unsigned rufactor = CONFIG_T::reuse_factor;
        static_assert(rufactor >= 1 && rufactor <= C && C % rufactor == 0,
                      "einsum_stream_rows: reuse_factor must divide n_contract");
        constexpr unsigned multscale = C / rufactor;         // lanes per output
        constexpr unsigned block_factor = N_OUT * multscale; // lanes
        typedef typename row_pack_T::value_type row_T;
        typedef typename buf_pack_T::value_type buf_T;
        typedef einsum_row_resource<typename data0_T::value_type, typename data1_T::value_type,
                                    typename res_T::value_type, CONFIG_T, N_OUT, OP>
            ops;

        row_T row[C];
        #pragma HLS ARRAY_PARTITION variable = row complete
        typename CONFIG_T::accum_t acc[N_OUT];
        #pragma HLS ARRAY_PARTITION variable = acc complete

        unsigned ir = 0;
    RowReuseLoop:
        for (unsigned t = 0; t < N_ROWS * rufactor; t++) {
            #pragma HLS PIPELINE II = 1

            if (ir == 0) {
            ReadRow:
                for (unsigned i_in = 0; i_in < C / row_pack_T::size; i_in++) {
                    #pragma HLS UNROLL
                    row_pack_T data_pack = row_stream.read();
                    for (unsigned i_pack = 0; i_pack < row_pack_T::size; i_pack++) {
                        #pragma HLS UNROLL
                        row[i_in * row_pack_T::size + i_pack] = data_pack[i_pack];
                    }
                }
            InitAccum:
                for (unsigned o = 0; o < N_OUT; o++) {
                    #pragma HLS UNROLL
                    acc[o] = 0;
                }
            }

        MultLoop:
            for (unsigned im = 0; im < block_factor; im++) {
                #pragma HLS UNROLL
                const unsigned out_index = im / multscale;           // static per lane
                const unsigned c = ir + rufactor * (im % multscale); // walks the contraction axis
                acc[out_index] += CONFIG_T::template product<typename data0_T::value_type, typename data1_T::value_type>::product(
                    ops::pick0(row[c], buf[out_index * C + c]), ops::pick1(row[c], buf[out_index * C + c]));
            }

            if (ir == rufactor - 1) {
            WriteRow:
                for (unsigned i_out = 0; i_out < N_OUT / res_T::size; i_out++) {
                    #pragma HLS UNROLL
                    res_T res_pack;
                    PRAGMA_DATA_PACK(res_pack)
                    for (unsigned i_pack = 0; i_pack < res_T::size; i_pack++) {
                        #pragma HLS UNROLL
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
    static typename std::enable_if<WHICH == 0, hls::stream<data0_T> &>::type select_stream(hls::stream<data0_T> &s0,
                                                                                          hls::stream<data1_T> &) {
        return s0;
    }
    template <unsigned WHICH>
    static typename std::enable_if<WHICH == 1, hls::stream<data1_T> &>::type select_stream(hls::stream<data0_T> &,
                                                                                          hls::stream<data1_T> &s1) {
        return s1;
    }
};

// Fallback (Latency, or transposes that break row order): drain both operands, run the array
// core, re-stream the result.
template <class data0_T, class data1_T, class res_T, typename CONFIG_T>
struct einsum_stream_impl<data0_T, data1_T, res_T, CONFIG_T, false> {
    static void run(hls::stream<data0_T> &data0_stream, hls::stream<data1_T> &data1_stream,
                    hls::stream<res_T> &res_stream) {
        typename data0_T::value_type data0[CONFIG_T::tpose_inp0_config::N];
        #pragma HLS ARRAY_PARTITION variable = data0 complete
        typename data1_T::value_type data1[CONFIG_T::tpose_inp1_config::N];
        #pragma HLS ARRAY_PARTITION variable = data1 complete
        typename res_T::value_type res[CONFIG_T::tpose_out_conf::N];
        #pragma HLS ARRAY_PARTITION variable = res complete

        einsum_drain<data0_T, CONFIG_T::tpose_inp0_config::N>(data0_stream, data0);
        einsum_drain<data1_T, CONFIG_T::tpose_inp1_config::N>(data1_stream, data1);

        nnet::einsum<typename data0_T::value_type, typename data1_T::value_type, typename res_T::value_type, CONFIG_T>(
            data0, data1, res);

    ResWrite:
        for (unsigned i_out = 0; i_out < CONFIG_T::tpose_out_conf::N / res_T::size; i_out++) {
            if (CONFIG_T::tpose_out_conf::N / res_T::size > 1) {
                #pragma HLS PIPELINE
            }
            res_T res_pack;
            PRAGMA_DATA_PACK(res_pack)
        ResPack:
            for (int i_pack = 0; i_pack < res_T::size; i_pack++) {
                #pragma HLS UNROLL
                res_pack[i_pack] = res[i_out * res_T::size + i_pack];
            }
            res_stream.write(res_pack);
        }
    }
};

template <class data0_T, class data1_T, class res_T, typename CONFIG_T>
void einsum(hls::stream<data0_T> &data0_stream, hls::stream<data1_T> &data1_stream, hls::stream<res_T> &res_stream) {
    einsum_stream_impl<data0_T, data1_T, res_T, CONFIG_T,
                       (CONFIG_T::strategy == nnet::resource && CONFIG_T::row_stream)>::run(data0_stream, data1_stream,
                                                                                            res_stream);
}

} // namespace nnet

#endif
