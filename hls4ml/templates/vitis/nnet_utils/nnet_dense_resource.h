#ifndef NNET_DENSE_RESOURCE_H_
#define NNET_DENSE_RESOURCE_H_

// Vitis-specific override of the Vivado backend's nnet_dense_resource.h.
//
// On Versal parts, Vitis HLS 2024.1 and 2025.2 pack narrow (e.g. 8-bit weight)
// multiply-accumulate chains from these Resource-strategy kernels into a
// DSP58 dot-product primitive (dsp_dotpra3 / dsp58_dp). That primitive is
// multi-cycle (observed NUM_STAGE=4), but it gets fed back into a
// `#pragma HLS PIPELINE II=1` loop-carried accumulator that was scheduled
// assuming a low-latency add -- the resulting RTL silently diverges from the
// C model (RTL != C in cosim) even though csim is exact. Pinning the
// accumulate `add` to `impl=fabric` prevents the tool from fusing it with the
// multiply into a dsp_dotpra3 macro; pairing that with an explicit
// `impl=dsp` binding on the product keeps the multiplies on plain
// single-product DSP58 multipliers (as they are on non-Versal parts) instead
// of falling back to fabric-only multiplication. Both bindings are applied
// below to every product-accumulate site in this file.
//
// Standalone reproducer, root-cause writeup, and before/after measurements:
// temp_space/vitis-resource/dsp58_dotpra3_bugreport/ (relative to the ATLAS
// repo root).

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_mult.h"
#include <assert.h>
#include <math.h>

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void dense_resource_rf_leq_nin(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                               typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                               typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {

    const int rufactor = CONFIG_T::reuse_factor;
    const int multfactor = MIN(CONFIG_T::n_in, CONFIG_T::reuse_factor);
    const int multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in * CONFIG_T::n_out, multfactor);
    const int block_factor = DIV_ROUNDUP(CONFIG_T::n_in * CONFIG_T::n_out, CONFIG_T::reuse_factor);
    const int multscale = multiplier_limit / CONFIG_T::n_out;
    const int nin = CONFIG_T::n_in;
    const int nout = CONFIG_T::n_out;

    assert((multiplier_limit % nout == 0 || rufactor >= nin) && "The current Reuse Factor is not allowed");
    assert((multiplier_limit == block_factor) && "This function is correct only for RF <= N_IN");

    #pragma HLS function_instantiate variable=weights,biases
    #pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    #pragma HLS ARRAY_PARTITION variable=biases complete

    if (CONFIG_T::reuse_factor > 1) {
        #pragma HLS RESOURCE variable=weights core=ROM_nP_BRAM
    }

    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=acc complete
    #pragma HLS bind_op variable=acc op=add impl=fabric

InitAccum:
    for (int iacc = 0; iacc < nout; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
    }

ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        #pragma HLS PIPELINE II=1 rewind

        int w_index = ir;
        int in_index = ir;
        int out_index = 0;
        int acc_step = 0;

    MultLoop:
        for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL

            typename CONFIG_T::accum_t mult = static_cast<typename CONFIG_T::accum_t>(
                CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[in_index], weights[w_index]));
            #pragma HLS bind_op variable=mult op=mul impl=dsp

            acc[out_index] += mult;

            // Increment w_index
            w_index += rufactor;
            // Increment in_index
            in_index += rufactor;
            if (in_index >= nin) {
                in_index = ir;
            }
            // Increment out_index
            if (acc_step + 1 >= multscale) {
                acc_step = 0;
                out_index++;
            } else {
                acc_step++;
            }
        }
    }

// Cast to "res_t" type
Result:
    for (int ires = 0; ires < CONFIG_T::n_out; ires++) {
        #pragma HLS UNROLL
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void dense_resource_rf_gt_nin_rem0(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                                   typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                                   typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {

    const int rufactor = MIN(CONFIG_T::reuse_factor, CONFIG_T::n_in * CONFIG_T::n_out);
    const int multfactor = MIN(CONFIG_T::n_in, CONFIG_T::reuse_factor);
    const int multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in * CONFIG_T::n_out, multfactor);
    const int block_factor = DIV_ROUNDUP(CONFIG_T::n_in * CONFIG_T::n_out, CONFIG_T::reuse_factor);
    const int multscale = multiplier_limit / CONFIG_T::n_out;
    const int nin = CONFIG_T::n_in;
    const int nout = CONFIG_T::n_out;

    assert((multiplier_limit % nout == 0 || rufactor >= nin) && "The current Reuse Factor is not allowed");
    assert((rufactor > nin && rufactor % nin == 0) && "This function is correct only for RF > N_IN && RF % N_IN == 0");

    #pragma HLS function_instantiate variable=weights,biases
    #pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    #pragma HLS ARRAY_PARTITION variable=biases complete

    if (CONFIG_T::reuse_factor > 1) {
        #pragma HLS RESOURCE variable=weights core=ROM_nP_BRAM
    }

    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=acc complete
    #pragma HLS bind_op variable=acc op=add impl=fabric

InitAccum:
    for (int iacc = 0; iacc < nout; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
    }

    int w_index;
    int in_index = 0;
    int out_index;
    int outstep = 0;
    const int outscale = rufactor / nin;

    int outidx[rufactor];
IndexLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        outidx[ir] = outstep;
        if ((ir + 1) % nin == 0) {
            outstep++;
        }
    }

ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        #pragma HLS PIPELINE II=1 rewind

        w_index = ir;
        out_index = outidx[ir] /*outstep*/;

    MultLoop:
        for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL
            typename CONFIG_T::accum_t mult = static_cast<typename CONFIG_T::accum_t>(
                CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[in_index], weights[w_index]));
            #pragma HLS bind_op variable=mult op=mul impl=dsp

            acc[out_index] += mult;

            w_index += rufactor;
            if (w_index >= CONFIG_T::n_in * CONFIG_T::n_out)
                break; // check out of bounds
            out_index += outscale;
        }

        in_index++;
        if (in_index >= nin) {
            in_index = 0;
            // outstep++; // This causes a huge increase in scheduling and RTL generation times, hence the above workaround.
        }
    }

// Cast to "res_t" type
Result:
    for (int ires = 0; ires < CONFIG_T::n_out; ires++) {
        #pragma HLS UNROLL
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void dense_resource_rf_gt_nin(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                              typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                              typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {

    const int rufactor = CONFIG_T::reuse_factor;
    const int multfactor = MIN(CONFIG_T::n_in, CONFIG_T::reuse_factor);
    const int multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in * CONFIG_T::n_out, multfactor);
    const int block_factor = DIV_ROUNDUP(CONFIG_T::n_in * CONFIG_T::n_out, CONFIG_T::reuse_factor);
    const int multscale = multiplier_limit / CONFIG_T::n_out;
    const int nin = CONFIG_T::n_in;
    const int nout = CONFIG_T::n_out;

    assert((multiplier_limit % nout == 0 || rufactor >= nin) && "The current Reuse Factor is not allowed");
    assert((rufactor > nin) && "This function is correct only for RF > N_IN");

    #pragma HLS function_instantiate variable=weights,biases
    #pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    #pragma HLS ARRAY_PARTITION variable=biases complete

    if (CONFIG_T::reuse_factor > 1) {
        #pragma HLS RESOURCE variable=weights core=ROM_nP_BRAM
    }

    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=acc complete

InitAccum:
    for (int iacc = 0; iacc < nout; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
    }

ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        #pragma HLS PIPELINE II=1 rewind
        typename CONFIG_T::accum_t tmpmult[block_factor];
        #pragma HLS ARRAY_PARTITION variable=tmpmult complete
        #pragma HLS bind_op variable=tmpmult op=mul impl=dsp

    MultLoop:
        for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL
            int w_index = ir + rufactor * im;
            int in_index = w_index % nin;
            if (w_index >= CONFIG_T::n_in * CONFIG_T::n_out)
                continue; // check out of bounds
            tmpmult[im] =
                CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[in_index], weights[w_index]);
        }

        typename CONFIG_T::accum_t mult[multiplier_limit];
        #pragma HLS ARRAY_PARTITION variable=mult complete
        #pragma HLS bind_op variable=mult op=add impl=fabric

    ResetMult:
        for (int imult = 0; imult < multiplier_limit; imult++) {
            #pragma HLS UNROLL
            mult[imult] = 0;
        }

    AccumLoop1:
        for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL
            int w_index = ir + rufactor * im;
            int out_index = w_index / multfactor;
            if (out_index >= multiplier_limit)
                continue; // check out of bounds
            mult[out_index] += tmpmult[im];
        }

    AccumLoop2:
        for (int im = 0; im < multiplier_limit; im++) {
            #pragma HLS UNROLL
            // int out_index = im/multscale; // This is the general case
            // acc[out_index] += mult[im];
            acc[im] += mult[im]; // If RF > N_IN then multiplier_limit == n_out
        }
    }

// Cast to "res_t" type
Result:
    for (int ires = 0; ires < CONFIG_T::n_out; ires++) {
        #pragma HLS UNROLL
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void dense_resource(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                    typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                    typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {

    #pragma HLS INLINE recursive

    if (CONFIG_T::reuse_factor <= CONFIG_T::n_in) {
        dense_resource_rf_leq_nin<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else if (CONFIG_T::reuse_factor % CONFIG_T::n_in == 0) {
        dense_resource_rf_gt_nin_rem0<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else {
        dense_resource_rf_gt_nin<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
}

} // namespace nnet

#endif
