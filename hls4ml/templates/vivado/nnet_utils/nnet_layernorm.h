#ifndef NNET_LAYERNORM_H_
#define NNET_LAYERNORM_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_dense.h"
#include <math.h>

#include "hls_math.h"

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
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    #pragma HLS ARRAY_PARTITION variable=data complete
    #pragma HLS ARRAY_PARTITION variable=res complete
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

    #pragma HLS ARRAY_PARTITION variable=data_diff complete

LAYERNORM_1D_SUM:
    for (int i = 0; i < dim; ++i) {
        sum_cache += static_cast<typename CONFIG_T::accum_t>(data[i]);
    }
    // Divide by dim: dim is a compile-time constant, so HLS lowers this fixed-point divide to a
    // multiply-by-reciprocal + shift (no divider, no float). Dividing rounds the QUOTIENT to
    // accum_t (error ~2^-accum_f, unscaled), which tracks HGQ2's float Sum/dim; a fixed 1/dim
    // reciprocal-multiply instead rounds the reciprocal (error ~Sum*2^-accum_f) and was what made
    // the rare per-token rsqrt index flip. Same idiom as average pooling's `y /= length`.
    // NOTE: keep this divide directly in accum_t -- do NOT promote the operands to a wide
    // intermediate type. That was tried (a fixed-point divide of two narrow accum_t values does
    // lose precision versus true round-to-nearest) but a wide ac_fixed/ap_fixed quotient
    // synthesizes to a divider Catapult's default resource library has no component for
    // (measured: 'div(80,1,37,0,37)' fails C/RTL synthesis -- CRAAS-6). accum_t is sized
    // explicitly for the HGQ2 path (register_precision) and is bit-exact there regardless of
    // width; the plain (non-HGQ2) path's precision is fixed by widening its DEFAULT accum_t
    // (see vivado_backend/catapult_backend's LayerNorm attribute registration), not by
    // reworking this arithmetic.
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
    // The scale-by-2^rsqrt_addr_f must NOT be done in accum_t: accum_t's integer width is sized
    // for the LN reduction (sum/variance), which is independent of table_size, so for a large
    // table (e.g. table_size=4096 -> 2^12) a narrow accum_t (e.g. ap_fixed<14,4>) saturates the
    // constant 2^rsqrt_addr_f to its max representable value and silently corrupts every index.
    // Route the address arithmetic through a dedicated wide type instead.
    ap_fixed<64, 32> index_val = (ap_fixed<64, 32>)var * (ap_fixed<64, 32>)(1 << CONFIG_T::rsqrt_addr_f) + (ap_fixed<64, 32>)0.5;
    int index = (int)index_val;
    if (index < 0)
        index = 0;
    if (index > (int)CONFIG_T::table_size - 1)
        index = CONFIG_T::table_size - 1;
    deno_inver = rsqrt_table[index];

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

    #pragma HLS ARRAY_PARTITION variable=scale complete
    #pragma HLS ARRAY_PARTITION variable=bias complete
    #pragma HLS ARRAY_PARTITION variable=in_val complete
    #pragma HLS ARRAY_PARTITION variable=outval complete

LAYERNORM_SEQ_LOOP:
    for (int j = 0; j < CONFIG_T::seq_len; ++j) {
        #pragma HLS PIPELINE
    LAYERNORM_LOAD:
        for (int i = 0; i < dim; ++i) {
            #pragma HLS UNROLL
            in_val[i] = data[j * dim + i];
        }
        layernorm_1d<data_T, res_T, CONFIG_T>(in_val, outval, scale, bias, rsqrt_table);
    LAYERNORM_STORE:
        for (int i = 0; i < dim; ++i) {
            #pragma HLS UNROLL
            res[j * dim + i] = outval[i];
        }
    }
}

} // namespace nnet

#endif
