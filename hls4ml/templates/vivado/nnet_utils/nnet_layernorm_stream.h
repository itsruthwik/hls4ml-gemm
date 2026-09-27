#ifndef NNET_LAYERNORM_STREAM_H_
#define NNET_LAYERNORM_STREAM_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_layernorm.h"
#include "nnet_types.h"

namespace nnet {

// ****************************************************
//       Streaming Layer Normalization
// ****************************************************
//
// LayerNorm normalizes each token (the `dim = n_in / seq_len` features on the
// normalized axis) independently. In io_stream each stream element carries one
// full token (`data_T` is an nnet::array of `dim` values), so one read == one
// token. Overloads the io_parallel `layernormalize` by argument type (hls::stream
// vs flat array), matching the batchnorm/activation streaming convention.
//
// Two implementations, selected by CONFIG_T::strategy like dense:
//  - latency:  buffer the token and run the shared per-token kernel `layernorm_1d`,
//              fully parallel (one token per cycle).
//  - resource: fold each token over reuse_factor cycles, ceil(dim / reuse_factor)
//              elements per cycle, with consecutive tokens overlapped (see below).

template <class data_T, class res_T, typename CONFIG_T>
void layernormalize_latency(hls::stream<data_T> &data, hls::stream<res_T> &res,
                            typename CONFIG_T::scale_t scale[CONFIG_T::n_in / CONFIG_T::seq_len],
                            typename CONFIG_T::bias_t bias[CONFIG_T::n_in / CONFIG_T::seq_len],
                            typename CONFIG_T::table_t rsqrt_table[CONFIG_T::table_size]) {
    static const unsigned dim = CONFIG_T::n_in / CONFIG_T::seq_len;

    #pragma HLS ARRAY_PARTITION variable=scale complete
    #pragma HLS ARRAY_PARTITION variable=bias complete

LayerNormSeqLoop:
    for (int j = 0; j < CONFIG_T::seq_len; ++j) {
        #pragma HLS PIPELINE

        data_T in_pack = data.read();
        res_T out_pack;
        PRAGMA_DATA_PACK(out_pack)

        typename data_T::value_type in_buf[dim];
        typename res_T::value_type out_buf[dim];
        #pragma HLS ARRAY_PARTITION variable=in_buf complete
        #pragma HLS ARRAY_PARTITION variable=out_buf complete

    LayerNormLoad:
        for (int i = 0; i < dim; ++i) {
            #pragma HLS UNROLL
            in_buf[i] = in_pack[i];
        }

        layernorm_1d<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(in_buf, out_buf, scale, bias,
                                                                                        rsqrt_table);

    LayerNormStore:
        for (int i = 0; i < dim; ++i) {
            #pragma HLS UNROLL
            out_pack[i] = out_buf[i];
        }

        res.write(out_pack);
    }
}

// Resource strategy. The per-element work is folded over reuse_factor cycles, and the
// token passes through three DATAFLOW stages so consecutive tokens overlap and the
// per-token scalar work (divides, rsqrt lookup) costs no throughput:
//   stats     -- sum(x) and sum(x^2) in one pass, lanes-wide partial sums
//   scalar    -- mean_q, variance, rsqrt table lookup (once per token)
//   normalize -- (x - mean_q) * inv_std * scale + bias, lanes-wide
// Bit-exact to `layernorm_1d`: its two-pass variance sum((x - mean_q)^2) is recovered by
// the exact expansion sum(x^2) - 2*mean_q*sum(x) + dim*mean_q^2. sum_t and sum2_t hold the
// sums exactly and ap_fixed carries each product at full width, so nothing is rounded
// until the same accum_t assignment the two-pass kernel makes.

namespace layernorm_resource {

template <class data_T, typename CONFIG_T> struct stat_msg {
    data_T x;
    typename CONFIG_T::sum_t sum;
    typename CONFIG_T::sum2_t sum2;
};

template <class data_T, typename CONFIG_T> struct norm_msg {
    data_T x;
    typename CONFIG_T::mean_t mean_q;
    typename CONFIG_T::table_t deno;
};

template <class data_T, typename CONFIG_T>
void stats(hls::stream<data_T> &data, hls::stream<stat_msg<data_T, CONFIG_T>> &out) {
    static const unsigned dim = CONFIG_T::n_in / CONFIG_T::seq_len;
    static const unsigned rf = CONFIG_T::reuse_factor;
    static const unsigned lanes = DIV_ROUNDUP(dim, rf);

LayerNormStatsSeq:
    for (int j = 0; j < CONFIG_T::seq_len; ++j) {
        stat_msg<data_T, CONFIG_T> m;
        m.x = data.read();

        typename CONFIG_T::sum_t acc[lanes];
        typename CONFIG_T::sum2_t acc2[lanes];
        #pragma HLS ARRAY_PARTITION variable=acc complete
        #pragma HLS ARRAY_PARTITION variable=acc2 complete
        for (int l = 0; l < lanes; ++l) {
            #pragma HLS UNROLL
            acc[l] = 0;
            acc2[l] = 0;
        }

    LayerNormStats:
        for (int c = 0; c < rf; ++c) {
            #pragma HLS PIPELINE II=1
            for (int l = 0; l < lanes; ++l) {
                #pragma HLS UNROLL
                int i = c * lanes + l;
                if (i < dim) {
                    typename data_T::value_type v = m.x[i];
                    acc[l] += v;
                    acc2[l] += v * v;
                }
            }
        }

        m.sum = 0;
        m.sum2 = 0;
        for (int l = 0; l < lanes; ++l) {
            #pragma HLS UNROLL
            m.sum += acc[l];
            m.sum2 += acc2[l];
        }
        out.write(m);
    }
}

template <class data_T, typename CONFIG_T>
void scalar(hls::stream<stat_msg<data_T, CONFIG_T>> &in, hls::stream<norm_msg<data_T, CONFIG_T>> &out,
            typename CONFIG_T::table_t rsqrt_table[CONFIG_T::table_size]) {
    static const unsigned dim = CONFIG_T::n_in / CONFIG_T::seq_len;

LayerNormScalarSeq:
    for (int j = 0; j < CONFIG_T::seq_len; ++j) {
        stat_msg<data_T, CONFIG_T> m = in.read();

        // Same divide-and-round as layernorm_1d (see the notes there on why it stays a divide).
        typename CONFIG_T::accum_t mean = static_cast<typename CONFIG_T::accum_t>(m.sum) / (int)dim;
        typename CONFIG_T::mean_t mean_q = mean;

        // Exact sum((x - mean_q)^2): the value layernorm_1d accumulates term by term.
        typename CONFIG_T::accum_t sum_cache2 =
            m.sum2 - (mean_q * m.sum) * (ap_uint<2>)2 + (mean_q * mean_q) * (ap_uint<16>)dim;
        typename CONFIG_T::accum_t var = sum_cache2 / (int)dim;

        // HGQ2 rsqrt table address, as in layernorm_1d.
        ap_fixed<64, 32> index_val =
            (ap_fixed<64, 32>)var * (ap_fixed<64, 32>)(1 << CONFIG_T::rsqrt_addr_f) + (ap_fixed<64, 32>)0.5;
        int index = (int)index_val;
        if (index < 0)
            index = 0;
        if (index > (int)CONFIG_T::table_size - 1)
            index = CONFIG_T::table_size - 1;

        norm_msg<data_T, CONFIG_T> o;
        o.x = m.x;
        o.mean_q = mean_q;
        o.deno = rsqrt_table[index];
        out.write(o);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void normalize(hls::stream<norm_msg<data_T, CONFIG_T>> &in, hls::stream<res_T> &res,
               typename CONFIG_T::scale_t scale[CONFIG_T::n_in / CONFIG_T::seq_len],
               typename CONFIG_T::bias_t bias[CONFIG_T::n_in / CONFIG_T::seq_len]) {
    static const unsigned dim = CONFIG_T::n_in / CONFIG_T::seq_len;
    static const unsigned rf = CONFIG_T::reuse_factor;
    static const unsigned lanes = DIV_ROUNDUP(dim, rf);

    #pragma HLS ARRAY_PARTITION variable=scale cyclic factor=lanes
    #pragma HLS ARRAY_PARTITION variable=bias cyclic factor=lanes

LayerNormNormalizeSeq:
    for (int j = 0; j < CONFIG_T::seq_len; ++j) {
        norm_msg<data_T, CONFIG_T> m = in.read();
        res_T out_pack;
        PRAGMA_DATA_PACK(out_pack)

    LayerNormNormalize:
        for (int c = 0; c < rf; ++c) {
            #pragma HLS PIPELINE II=1
            for (int l = 0; l < lanes; ++l) {
                #pragma HLS UNROLL
                int i = c * lanes + l;
                if (i < dim) {
                    typename CONFIG_T::norm_t data_diff = static_cast<typename CONFIG_T::norm_t>(
                        static_cast<typename CONFIG_T::accum_t>(m.x[i]) - m.mean_q);
                    out_pack[i] = data_diff * m.deno * scale[i] + bias[i];
                }
            }
        }
        res.write(out_pack);
    }
}

} // namespace layernorm_resource

template <class data_T, class res_T, typename CONFIG_T>
void layernormalize_resource(hls::stream<data_T> &data, hls::stream<res_T> &res,
                             typename CONFIG_T::scale_t scale[CONFIG_T::n_in / CONFIG_T::seq_len],
                             typename CONFIG_T::bias_t bias[CONFIG_T::n_in / CONFIG_T::seq_len],
                             typename CONFIG_T::table_t rsqrt_table[CONFIG_T::table_size]) {
    #pragma HLS DATAFLOW

    hls::stream<layernorm_resource::stat_msg<data_T, CONFIG_T>> stat_stream("layernorm_stat");
    hls::stream<layernorm_resource::norm_msg<data_T, CONFIG_T>> norm_stream("layernorm_norm");
    #pragma HLS STREAM variable=stat_stream depth=2
    #pragma HLS STREAM variable=norm_stream depth=2

    layernorm_resource::stats<data_T, CONFIG_T>(data, stat_stream);
    layernorm_resource::scalar<data_T, CONFIG_T>(stat_stream, norm_stream, rsqrt_table);
    layernorm_resource::normalize<data_T, res_T, CONFIG_T>(norm_stream, res, scale, bias);
}

template <class data_T, class res_T, typename CONFIG_T>
void layernormalize(hls::stream<data_T> &data, hls::stream<res_T> &res,
                    typename CONFIG_T::scale_t scale[CONFIG_T::n_in / CONFIG_T::seq_len],
                    typename CONFIG_T::bias_t bias[CONFIG_T::n_in / CONFIG_T::seq_len],
                    typename CONFIG_T::table_t rsqrt_table[CONFIG_T::table_size]) {
    #pragma HLS INLINE
    if (CONFIG_T::strategy == nnet::resource) {
        layernormalize_resource<data_T, res_T, CONFIG_T>(data, res, scale, bias, rsqrt_table);
    } else {
        layernormalize_latency<data_T, res_T, CONFIG_T>(data, res, scale, bias, rsqrt_table);
    }
}

} // namespace nnet

#endif
