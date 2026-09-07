
#ifndef NNET_DENSE_RESOURCE_H_
#define NNET_DENSE_RESOURCE_H_

#include "ac_channel.h"
#include "nnet_common.h"
#include "nnet_mult.h"
#include <assert.h>
#include <math.h>

namespace nnet {

// Balanced adder tree over N consecutive elements (used to reduce split accumulators).
template <class T, int N> struct tree_sum_t {
    static T sum(const T *p) { return tree_sum_t<T, N / 2>::sum(p) + tree_sum_t<T, N - N / 2>::sum(p + N / 2); }
};
template <class T> struct tree_sum_t<T, 1> {
    static T sum(const T *p) { return p[0]; }
};

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


    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    typename CONFIG_T::accum_t acc_part[CONFIG_T::n_out][multscale];

#pragma hls_unroll
InitAccum:
    for (int iacc = 0; iacc < nout; iacc++) {
        acc[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
    #pragma hls_unroll
    InitPart:
        for (int ip = 0; ip < multscale; ip++) {
            acc_part[iacc][ip] = 0;
        }
    }

#pragma hls_pipeline_init_interval 1
ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++) {

        int w_index = ir;
        int in_index = ir;
        int out_index = 0;
        int acc_step = 0;

    #pragma hls_unroll
    MultLoop:
        for (int im = 0; im < block_factor; im++) {
            // One partial accumulator per (output, slot): each loop-carried chain is a single
            // multiply-add. Summing the multscale products into one acc per iteration makes
            // Catapult fuse the whole chain into the II=1 feedback, which cannot close at 5 ns.
            // Bit-exact: accum_t is wrap-mode fixed point without rounding (order-independent).
            // rf==1 has a single reuse iteration, so there is no loop-carried chain to break;
            // keep the plain accumulate there (the split form produced RTL that failed cosim at
            // rf==1 while its C model was exact).
            if (rufactor == 1) {
                acc[out_index] += static_cast<typename CONFIG_T::accum_t>(
                    CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[in_index], weights[w_index]));
            } else {
                acc_part[out_index][acc_step] += static_cast<typename CONFIG_T::accum_t>(
                    CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[in_index], weights[w_index]));
            }

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
    if (rufactor > 1) {
    #pragma hls_unroll
    ReducePart:
        for (int io = 0; io < nout; io++) {
            acc[io] += tree_sum_t<typename CONFIG_T::accum_t, multscale>::sum(acc_part[io]);
        }
    }

#pragma hls_unroll
Result:
    for (unsigned int ires = 0; ires < CONFIG_T::n_out; ires++) {
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
    const int nin = CONFIG_T::n_in;
    const int nout = CONFIG_T::n_out;

    assert((multiplier_limit % nout == 0 || rufactor >= nin) && "The current Reuse Factor is not allowed");
    assert((rufactor > nin && rufactor % nin == 0) && "This function is correct only for RF > N_IN && RF % N_IN == 0");


    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];

#pragma hls_unroll
InitAccum:
    for (int iacc = 0; iacc < nout; iacc++) {
        acc[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
    }

    unsigned int w_index;
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

#pragma hls_pipeline_init_interval 1
ReuseLoop:
    for (unsigned int ir = 0; ir < rufactor; ir++) {

        w_index = ir;
        out_index = outidx[ir] /*outstep*/;

    #pragma hls_unroll
    MultLoop:
        for (unsigned int im = 0; im < block_factor; im++) {
            acc[out_index] += static_cast<typename CONFIG_T::accum_t>(
                CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[in_index], weights[w_index]));

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
#pragma hls_unroll
Result:
    for (unsigned int ires = 0; ires < CONFIG_T::n_out; ires++) {
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
    const int nin = CONFIG_T::n_in;
    const int nout = CONFIG_T::n_out;

    assert((multiplier_limit % nout == 0 || rufactor >= nin) && "The current Reuse Factor is not allowed");
    assert((rufactor > nin) && "This function is correct only for RF > N_IN");


    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];

#pragma hls_unroll
InitAccum:
    for (int iacc = 0; iacc < nout; iacc++) {
        acc[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
    }

#pragma hls_pipeline_init_interval 1
ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        typename CONFIG_T::accum_t tmpmult[block_factor];

    #pragma hls_unroll
    MultLoop:
        for (int im = 0; im < block_factor; im++) {
            unsigned int w_index = ir + rufactor * im;
            int in_index = w_index % nin;
            if (w_index >= CONFIG_T::n_in * CONFIG_T::n_out)
                continue; // check out of bounds
            tmpmult[im] =
                CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[in_index], weights[w_index]);
        }

        typename CONFIG_T::accum_t mult[multiplier_limit];

    #pragma hls_unroll
    ResetMult:
        for (int imult = 0; imult < multiplier_limit; imult++) {
            mult[imult] = 0;
        }

    #pragma hls_unroll
    AccumLoop1:
        for (int im = 0; im < block_factor; im++) {
            int w_index = ir + rufactor * im;
            int out_index = w_index / multfactor;
            if (out_index >= multiplier_limit)
                continue; // check out of bounds
            mult[out_index] += tmpmult[im];
        }

    #pragma hls_unroll
    AccumLoop2:
        for (int im = 0; im < multiplier_limit; im++) {
            // int out_index = im/multscale; // This is the general case
            // acc[out_index] += mult[im];
            acc[im] += mult[im]; // If RF > N_IN then multiplier_limit == n_out
        }
    }

// Cast to "res_t" type
#pragma hls_unroll
Result:
    for (unsigned int ires = 0; ires < CONFIG_T::n_out; ires++) {
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void dense_resource(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                    typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                    typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {


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
