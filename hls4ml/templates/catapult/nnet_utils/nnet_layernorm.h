#ifndef NNET_LAYERNORM_H_
#define NNET_LAYERNORM_H_

#include "ac_channel.h"
#include "ac_fixed.h"
#include "ac_int.h"
#include "nnet_common.h"
#include "nnet_dense.h"
#include <math.h>

namespace nnet {

struct layernorm_config {
    // Internal data type definitions
    typedef float bias_t;
    typedef float scale_t;
    typedef float accum_t;
    typedef float table_t;
    typedef float mean_t;
    typedef float norm_t;

    // Layer Sizes
    static const unsigned n_in = 20;
    static const unsigned seq_len = 4;
    static const unsigned axis = 2;
    static const unsigned table_size = 1024;
    static const unsigned rsqrt_addr_f = 0;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;

    template <class x_T, class y_T> using product = nnet::product::mult<x_T, y_T>;
};

// Bit-exact to HGQ2's QLayerNormalization: the reciprocal-std is NOT recomputed in float
// here. HGQ2 routes the per-token variance through a quantized LUT (rsqrt_table), whose
// values already fold in epsilon and the output quantizer, and which the converter transfers
// verbatim. This kernel reproduces that exactly: it addresses the SAME table by the SAME
// quantized-variance index (round(var * 2^rsqrt_addr_f), saturated). Because the table is
// data-driven, the kernel needs nothing about epsilon (any value works).
template <class data_T, class res_T, typename CONFIG_T>
void layernorm_1d(data_T data[CONFIG_T::n_in / CONFIG_T::seq_len], res_T res[CONFIG_T::n_in / CONFIG_T::seq_len],
                  typename CONFIG_T::scale_t scale[CONFIG_T::n_in / CONFIG_T::seq_len],
                  typename CONFIG_T::bias_t bias[CONFIG_T::n_in / CONFIG_T::seq_len],
                  typename CONFIG_T::table_t rsqrt_table[CONFIG_T::table_size]) {
    //#pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    //#pragma HLS ARRAY_PARTITION variable=data complete
    //#pragma HLS ARRAY_PARTITION variable=res complete
    typename CONFIG_T::table_t deno_inver = 0;

    static const unsigned dim = CONFIG_T::n_in / CONFIG_T::seq_len;
    typename CONFIG_T::accum_t sum_cache = 0;
    typename CONFIG_T::accum_t sum_cache2 = 0;
    typename CONFIG_T::accum_t var, mean, diff;
    // data_diff (x - mean) is sized at its exact minimal precision (norm_t), not accum_t, so the
    // squared-diff and result multiplies run at operand width -- the standard hls4ml convention
    // (multiply at operand precision, accumulate the SUM in accum_t; see product<> in nnet_mult.h).
    // Lossless (norm_t holds x - mean_q exactly), so it stays bit-exact.
    typename CONFIG_T::norm_t data_diff[dim];

    //#pragma HLS ARRAY_PARTITION variable=data_diff complete

LAYERNORM_1D_SUM:
    for (int i = 0; i < dim; ++i) {
        sum_cache += static_cast<typename CONFIG_T::accum_t>(data[i]);
    }
    // Divide by dim: dim is a compile-time constant, so HLS lowers this fixed-point divide to a
    // multiply-by-reciprocal + shift (no divider, no float). Dividing rounds the QUOTIENT to
    // accum_t (error ~2^-accum_f, unscaled), which tracks HGQ2's float Sum/dim; a fixed 1/dim
    // reciprocal-multiply instead rounds the reciprocal (error ~Sum*2^-accum_f) and was what made
    // the rare per-token rsqrt index flip. Same idiom as average pooling's `y /= length`.
    mean = sum_cache / (int)dim;
    // Quantize the mean to HGQ2's mean_q precision so (x - mean) is bit-exact to HGQ2.
    typename CONFIG_T::mean_t mean_q = mean;

LAYERNORM_1D_VAR:
    for (int i = 0; i < dim; ++i) {
        data_diff[i] = static_cast<typename CONFIG_T::norm_t>(static_cast<typename CONFIG_T::accum_t>(data[i]) - mean_q);
        diff = data_diff[i] * data_diff[i];
        sum_cache2 += diff;
    }
    var = sum_cache2 / (int)dim;

    // HGQ2 address: quantize the variance to rsqrt_addr_f fractional bits (round),
    // then saturate into the unsigned table range [0, table_size).
    // CATAPULT_PORT: ac_fixed has no implicit conversion to int, so the round-half-up +
    // saturate is done through an explicit ac_int cast (mirrors the Vivado (int) cast, same
    // truncate-toward-zero-after-add-0.5 semantics for the values this ever sees: var >= 0).
    // The scale-by-2^rsqrt_addr_f must NOT be done in accum_t: accum_t's integer width is sized
    // for the LN reduction (sum/variance), which is independent of table_size, so for a large
    // table (e.g. table_size=4096 -> 2^12) a narrow accum_t saturates the constant
    // 2^rsqrt_addr_f to its max representable value and silently corrupts every index. Route the
    // address arithmetic through a dedicated wide type instead.
    ac_fixed<64, 32, true> addr_val =
        (ac_fixed<64, 32, true>)var * (ac_fixed<64, 32, true>)(1 << CONFIG_T::rsqrt_addr_f) + (ac_fixed<64, 32, true>)0.5;
    ac_int<32, true> index = addr_val.to_int();
    if (index < 0)
        index = 0;
    if (index > (ac_int<32, true>)CONFIG_T::table_size - 1)
        index = CONFIG_T::table_size - 1;
    deno_inver = rsqrt_table[index.to_int()];

LAYERNORM_1D_RESULT:
    for (int i = 0; i < dim; ++i) {
        res[i] = data_diff[i] * deno_inver * scale[i] + bias[i];
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void layernormalize(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in],
                    typename CONFIG_T::scale_t scale[CONFIG_T::n_in / CONFIG_T::seq_len],
                    typename CONFIG_T::bias_t bias[CONFIG_T::n_in / CONFIG_T::seq_len],
                    typename CONFIG_T::table_t rsqrt_table[CONFIG_T::table_size]) {
    static const unsigned dim = CONFIG_T::n_in / CONFIG_T::seq_len;
    data_T in_val[dim];
    res_T outval[dim];

    //#pragma HLS ARRAY_PARTITION variable=scale complete
    //#pragma HLS ARRAY_PARTITION variable=bias complete
    //#pragma HLS ARRAY_PARTITION variable=in_val complete
    //#pragma HLS ARRAY_PARTITION variable=outval complete

LAYERNORM_SEQ_LOOP:
    for (int j = 0; j < CONFIG_T::seq_len; ++j) {
        //#pragma HLS PIPELINE
    LAYERNORM_LOAD:
        for (int i = 0; i < dim; ++i) {
            //#pragma HLS UNROLL
            in_val[i] = data[j * dim + i];
        }
        layernorm_1d<data_T, res_T, CONFIG_T>(in_val, outval, scale, bias, rsqrt_table);
    LAYERNORM_STORE:
        for (int i = 0; i < dim; ++i) {
            //#pragma HLS UNROLL
            res[j * dim + i] = outval[i];
        }
    }
}

} // namespace nnet

#endif
