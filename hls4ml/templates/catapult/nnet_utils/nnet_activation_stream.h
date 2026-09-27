
// Change History:
//   2022-06-30  dgburnette - Cleaned up code to separate AC Math from LUT code.
//                            Activation functions not implemented in AC Math will assert.
//   2022-06-28  dgburnette - Replaced AP Types with AC Datatypes.

#ifndef NNET_ACTIVATION_STREAM_H_
#define NNET_ACTIVATION_STREAM_H_

#include "ac_channel.h"
#include "ac_fixed.h"
#include "nnet_activation.h"
#include "nnet_common.h"
#include "nnet_stream.h"
#include "nnet_types.h"
#include <ac_math/ac_elu_pwl.h>
#include <ac_math/ac_pow_pwl.h>
#include <ac_math/ac_relu.h>
#include <ac_math/ac_selu_pwl.h>
#include <ac_math/ac_sigmoid_pwl.h>
#include <ac_math/ac_softmax_pwl.h>
#include <ac_math/ac_softplus_pwl.h>
#include <ac_math/ac_softsign_pwl.h>
#include <ac_math/ac_tanh_pwl.h>
#include <ac_std_float.h>
#include <cmath>

namespace nnet {

// *************************************************
//       LINEAR Activation
// *************************************************
// Adding this to work around problem with Catapult and SR model where the output channel appears to be inout
template <class data_T, class res_T, typename CONFIG_T> void linear(ac_channel<data_T> &data, ac_channel<res_T> &res) {
#pragma hls_pipeline_init_interval 1
LinearActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;

    #pragma hls_unroll
    LinearPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            out_data[j] = in_data[j];
        }

        res.write(out_data);
    }
}

// *************************************************
//       RELU Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T> void relu(ac_channel<data_T> &data, ac_channel<res_T> &res) {
#pragma hls_pipeline_init_interval 1
ReLUActLoop:
    for (unsigned int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;

    #pragma hls_unroll
    ReLUPackLoop:
        for (unsigned int j = 0; j < res_T::size; j++) {
#ifndef USE_AC_MATH
            if (in_data[j] > 0)
                out_data[j] = in_data[j];
            else
                out_data[j] = 0;
#else
            ac_math::ac_relu(in_data[j], out_data[j]);
#endif
        }

        res.write(out_data);
    }
}

// *************************************************
//       Sigmoid Activation
// *************************************************
#ifndef USE_AC_MATH

template <class data_T, class res_T, typename CONFIG_T> void sigmoid(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t sigmoid_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t sigmoid_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_sigmoid_table<CONFIG_T, CONFIG_T::table_size>(sigmoid_table);
        initialized = true;
    }

#pragma hls_pipeline_init_interval 1
SigmoidActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;

    #pragma hls_unroll
    SigmoidPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            int data_round = in_data[j].to_double() * (int)CONFIG_T::table_size / 16;
            int index = data_round + 8 * (int)CONFIG_T::table_size / 16;
            if (index < 0)
                index = 0;
            else if (index > CONFIG_T::table_size - 1)
                index = (int)CONFIG_T::table_size - 1;
            out_data[j] = sigmoid_table[index];
        }

        res.write(out_data);
    }
}

#else

template <class data_T, class res_T, typename CONFIG_T> void sigmoid(ac_channel<data_T> &data, ac_channel<res_T> &res) {
SigmoidActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;
    SigmoidPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            // ac_math::ac_sigmoid_pwl(in_data[j], out_data[j]);
            ac_sigmoid_pwl_wrapper(in_data[j], out_data[j]);
        }
        res.write(out_data);
    }
}

#endif

// *************************************************
//       Softmax Activation
// *************************************************
//
// Streaming twins of the table-driven kernels in nnet_activation.h: one input
// pack is one softmax row (axis == -1), so the row length is data_T::size. The
// exp / 1/x tables arrive as constant weight arrays (catapult:softmax_const_tables).
// Define HLS4ML_SOFTMAX_AC_MATH for the ac_math piecewise-linear kernel instead.

#ifndef HLS4ML_SOFTMAX_AC_MATH

template <class data_T, class res_T, typename CONFIG_T>
void softmax_latency(ac_channel<data_T> &data, ac_channel<res_T> &res,
                     typename CONFIG_T::exp_table_t exp_table[CONFIG_T::exp_table_size],
                     typename CONFIG_T::inv_table_t invert_table[CONFIG_T::inv_table_size]) {
    // Reuse factor folds the normalisation multipliers (Vivado's rule: size / multiplier_limit
    // steps per row). Catapult does not share unrolled multipliers just because the II allows
    // it, so the sharing is explicit: one flat II 1 loop over (row, reuse step) with
    // size / steps multiplier lanes, reading a row on its first step and writing it on its last.
    constexpr unsigned multiplier_limit = DIV_ROUNDUP(data_T::size, CONFIG_T::reuse_factor);
    constexpr unsigned rufactor = data_T::size / multiplier_limit;
    constexpr unsigned multscale = DIV_ROUNDUP(data_T::size, rufactor); // multiplier lanes

    typename CONFIG_T::accum_t exp_res[data_T::size];
    typename CONFIG_T::inv_table_t inv_exp_sum = 0;
    res_T out_pack;
    unsigned ir = 0;
    #pragma hls_pipeline_init_interval 1
SoftmaxExpLoop:
    for (unsigned t = 0; t < (CONFIG_T::n_in / data_T::size) * rufactor; t++) {
        if (ir == 0) {
            data_T in_pack = data.read();

            // Calculate all the e^x's
            typename CONFIG_T::inv_inp_t exp_sum(0);
            #pragma hls_unroll
        SoftmaxExpPackLoop:
            for (unsigned j = 0; j < data_T::size; j++) {
                unsigned x = softmax_idx_from_real_val<typename data_T::value_type, CONFIG_T::exp_table_size>(in_pack[j]);
                exp_res[j] = exp_table[x];
            }

            // Explicitly sum the results with an adder tree.
            // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
            Op_add<typename CONFIG_T::accum_t> op_add;
            exp_sum = reduce<typename CONFIG_T::accum_t, data_T::size, Op_add<typename CONFIG_T::accum_t>>(exp_res, op_add);

            inv_exp_sum =
                invert_table[softmax_idx_from_real_val<typename CONFIG_T::inv_inp_t, CONFIG_T::inv_table_size>(exp_sum)];
        }

        #pragma hls_unroll
    SoftmaxInvPackLoop:
        for (unsigned k = 0; k < multscale; k++) {
            const unsigned j = ir * multscale + k; // lane k serves one output per reuse step
            if (j < res_T::size) {
                out_pack[j] = exp_res[j] * inv_exp_sum;
            }
        }

        if (ir == rufactor - 1) {
            res.write(out_pack);
            ir = 0;
        } else {
            ir++;
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void softmax_stable(ac_channel<data_T> &data, ac_channel<res_T> &res,
                    typename CONFIG_T::exp_table_t exp_table[CONFIG_T::exp_table_size],
                    typename CONFIG_T::inv_table_t invert_table[CONFIG_T::inv_table_size]) {
    // Reuse factor folds the normalisation multipliers (Vivado's rule: size / multiplier_limit
    // steps per row). Catapult does not share unrolled multipliers just because the II allows
    // it, so the sharing is explicit: one flat II 1 loop over (row, reuse step) with
    // size / steps multiplier lanes, reading a row on its first step and writing it on its last.
    constexpr unsigned multiplier_limit = DIV_ROUNDUP(data_T::size, CONFIG_T::reuse_factor);
    constexpr unsigned rufactor = data_T::size / multiplier_limit;
    constexpr unsigned multscale = DIV_ROUNDUP(data_T::size, rufactor); // multiplier lanes

    typename CONFIG_T::accum_t exp_res[data_T::size];
    typename CONFIG_T::inv_table_t inv_exp_sum = 0;
    res_T out_pack;
    unsigned ir = 0;
    #pragma hls_pipeline_init_interval 1
SoftmaxArrayLoop:
    for (unsigned t = 0; t < (CONFIG_T::n_in / data_T::size) * rufactor; t++) {
        if (ir == 0) {
            data_T in_pack = data.read();

            typename data_T::value_type data_array[data_T::size];
            #pragma hls_unroll
        SoftmaxArrayPackLoop:
            for (unsigned j = 0; j < data_T::size; j++) {
                data_array[j] = in_pack[j];
            }

            // Find the max and compute all delta(x_i, x_max)
            Op_max<typename data_T::value_type> op_max;
            typename data_T::value_type x_max =
                reduce<typename data_T::value_type, data_T::size, Op_max<typename data_T::value_type>>(data_array, op_max);

            typename CONFIG_T::inp_norm_t d_xi_xmax[data_T::size];
            #pragma hls_unroll
            for (unsigned j = 0; j < data_T::size; j++) {
                d_xi_xmax[j] = x_max - data_array[j];
            }

            // Calculate all the e^x's
            typename CONFIG_T::inv_inp_t exp_sum(0);
            #pragma hls_unroll
            for (unsigned j = 0; j < data_T::size; j++) {
                unsigned x = softmax_idx_from_real_val<typename CONFIG_T::inp_norm_t, CONFIG_T::exp_table_size>(d_xi_xmax[j]);
                exp_res[j] = exp_table[x];
            }

            // Explicitly sum the results with an adder tree.
            // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
            Op_add<typename CONFIG_T::accum_t> op_add;
            exp_sum = reduce<typename CONFIG_T::accum_t, data_T::size, Op_add<typename CONFIG_T::accum_t>>(exp_res, op_add);

            inv_exp_sum =
                invert_table[softmax_idx_from_real_val<typename CONFIG_T::inv_inp_t, CONFIG_T::inv_table_size>(exp_sum)];
        }

        #pragma hls_unroll
    SoftmaxInvPackLoop:
        for (unsigned k = 0; k < multscale; k++) {
            const unsigned j = ir * multscale + k; // lane k serves one output per reuse step
            if (j < res_T::size) {
                out_pack[j] = exp_res[j] * inv_exp_sum;
            }
        }

        if (ir == rufactor - 1) {
            res.write(out_pack);
            ir = 0;
        } else {
            ir++;
        }
    }
}

#else

// ac_math piecewise-linear softmax, one input pack per row.
template <class data_T, class res_T, typename CONFIG_T>
void softmax_ac_math(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    typename data_T::value_type data_cache[data_T::size];
    typename res_T::value_type res_cache[res_T::size];
    #pragma hls_pipeline_init_interval 1
SoftmaxInitLoop:
    for (unsigned s = 0; s < CONFIG_T::n_in / data_T::size; s++) {
        data_T in_pack = data.read();

        #pragma hls_unroll
    SoftmaxInitPackLoop:
        for (unsigned j = 0; j < data_T::size; j++) {
            data_cache[j] = in_pack[j];
        }

        res_T out_pack;
        ac_softmax_pwl_wrapper(data_cache, res_cache);

        #pragma hls_unroll
    SoftmaxResPackLoop:
        for (unsigned j = 0; j < res_T::size; j++) {
            out_pack[j] = res_cache[j];
        }

        res.write(out_pack);
    }
}

#endif // HLS4ML_SOFTMAX_AC_MATH

template <class data_T, class res_T, typename CONFIG_T>
void softmax_legacy(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    // Initialize the lookup table
#ifdef __SYNTHESIS__
    bool initialized = false;
    typename CONFIG_T::table_t exp_table[CONFIG_T::exp_table_size];
    typename CONFIG_T::table_t invert_table[CONFIG_T::inv_table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t exp_table[CONFIG_T::exp_table_size];
    static typename CONFIG_T::table_t invert_table[CONFIG_T::inv_table_size];
#endif
    if (!initialized) {
        init_exp_table_legacy<CONFIG_T, CONFIG_T::exp_table_size>(exp_table);
        init_invert_table_legacy<CONFIG_T, CONFIG_T::inv_table_size>(invert_table);
        initialized = true;
    }

    // Index into the lookup table based on data for exponentials
    typename CONFIG_T::table_t exp_res[data_T::size];
    typename CONFIG_T::table_t exp_diff_res;
    typename data_T::value_type data_cache[data_T::size];

    #pragma hls_pipeline_init_interval 1
SoftmaxInitLoop:
    for (unsigned s = 0; s < CONFIG_T::n_in / data_T::size; s++) {
        data_T in_pack = data.read();
        #pragma hls_unroll
    SoftmaxInitPackLoop:
        for (unsigned j = 0; j < data_T::size; j++) {
            data_cache[j] = in_pack[j];
            exp_res[j] = 0;
        }

        #pragma hls_unroll
    SoftmaxExpLoop:
        for (int i = 0; i < data_T::size; i++) {
            #pragma hls_unroll
        SoftmaxExpInner:
            for (int j = 0; j < data_T::size; j++) {
                if (i == j) {
                    exp_diff_res = 1;
                } else {
                    int data_round =
                        (data_cache[j].to_double() - data_cache[i].to_double()) * (int)CONFIG_T::exp_table_size / 16;
                    int index = data_round + 8 * (int)CONFIG_T::exp_table_size / 16;
                    if (index < 0)
                        index = 0;
                    if (index > CONFIG_T::exp_table_size - 1)
                        index = (int)CONFIG_T::exp_table_size - 1;
                    exp_diff_res = exp_table[index];
                }
                exp_res[i] += exp_diff_res;
            }
        }

        res_T out_pack;
        #pragma hls_unroll
    SoftmaxInvPackLoop:
        for (unsigned j = 0; j < res_T::size; j++) {
            int exp_res_index = exp_res[j].to_double() * (int)CONFIG_T::inv_table_size / 64;
            if (exp_res_index < 0)
                exp_res_index = 0;
            if (exp_res_index > CONFIG_T::inv_table_size - 1)
                exp_res_index = (int)CONFIG_T::inv_table_size - 1;
            out_pack[j] = (typename res_T::value_type)invert_table[exp_res_index];
        }
        res.write(out_pack);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void softmax_argmax(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    #pragma hls_pipeline_init_interval 1
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;

        #pragma hls_unroll
        for (int j = 0; j < res_T::size; j++) {
            out_data[j] = (typename res_T::value_type)0;
        }

        typename data_T::value_type maximum = in_data[0];
        int idx = 0;

        #pragma hls_unroll
        for (int j = 1; j < res_T::size; j++) {
            if (in_data[j] > maximum) {
                maximum = in_data[j];
                idx = j;
            }
        }

        out_data[idx] = (typename res_T::value_type)1;
        res.write(out_data);
    }
}

#ifndef HLS4ML_SOFTMAX_AC_MATH
// Resource strategy for the stable softmax. One row (one beat) is accepted every
// reuse_factor cycles: lanes = ceil(size / reuse_factor) elements per cycle, and rows overlap
// across three blocks (max | exp + sum | normalize) joined by channels, each one flat II 1
// loop over (row, fold step). Only lane groups and one scalar per row flow between blocks.
// Bit-exact to softmax_stable: the exp sum is exact in accum_t and rounded once into
// inv_inp_t, so the summation order doesn't matter.
namespace softmax_resource {

template <class data_T, typename CONFIG_T> struct fold {
    static const unsigned lanes = DIV_ROUNDUP(data_T::size, CONFIG_T::reuse_factor);
    static const unsigned steps = DIV_ROUNDUP(data_T::size, lanes);
    static const unsigned rows = CONFIG_T::n_in / data_T::size;
    typedef array<typename data_T::value_type, lanes> x_group_t;
    typedef array<typename CONFIG_T::accum_t, lanes> e_group_t;
};

#pragma hls_design block
template <class data_T, typename CONFIG_T>
void row_max(ac_channel<data_T> &data, ac_channel<typename fold<data_T, CONFIG_T>::x_group_t> &xs,
             ac_channel<typename data_T::value_type> &maxs) {
    typedef fold<data_T, CONFIG_T> F;
    data_T x;
    typename data_T::value_type mx[F::lanes];
    unsigned s = 0;
    #pragma hls_pipeline_init_interval 1
SoftmaxMaxLoop:
    for (unsigned n = 0; n < F::rows * F::steps; n++) {
        if (s == 0)
            x = data.read();
        typename F::x_group_t g;
        #pragma hls_unroll yes
        for (unsigned l = 0; l < F::lanes; l++) {
            unsigned j = s * F::lanes + l;
            typename data_T::value_type v = (j < data_T::size) ? x[j] : x[0];
            g[l] = v;
            if (s == 0 || v > mx[l])
                mx[l] = v;
        }
        xs.write(g);
        if (s == F::steps - 1) {
            typename data_T::value_type m = mx[0];
            #pragma hls_unroll yes
            for (unsigned l = 1; l < F::lanes; l++) {
                if (mx[l] > m)
                    m = mx[l];
            }
            maxs.write(m);
            s = 0;
        } else {
            s++;
        }
    }
}

#pragma hls_design block
template <class data_T, typename CONFIG_T>
void row_exp(ac_channel<typename fold<data_T, CONFIG_T>::x_group_t> &xs, ac_channel<typename data_T::value_type> &maxs,
             ac_channel<typename fold<data_T, CONFIG_T>::e_group_t> &es,
             ac_channel<typename CONFIG_T::inv_table_t> &invs,
             typename CONFIG_T::exp_table_t exp_table[CONFIG_T::exp_table_size],
             typename CONFIG_T::inv_table_t invert_table[CONFIG_T::inv_table_size]) {
    typedef fold<data_T, CONFIG_T> F;
    typename data_T::value_type x_max = 0;
    typename CONFIG_T::accum_t part[F::lanes];
    unsigned s = 0;
    #pragma hls_pipeline_init_interval 1
SoftmaxExpLoop:
    for (unsigned n = 0; n < F::rows * F::steps; n++) {
        if (s == 0)
            x_max = maxs.read();
        typename F::x_group_t g = xs.read();
        typename F::e_group_t eg;
        #pragma hls_unroll yes
        for (unsigned l = 0; l < F::lanes; l++) {
            unsigned j = s * F::lanes + l;
            typename CONFIG_T::accum_t acc = (s == 0) ? typename CONFIG_T::accum_t(0) : part[l];
            typename CONFIG_T::accum_t ev = 0;
            if (j < data_T::size) {
                typename CONFIG_T::inp_norm_t d = x_max - g[l];
                ev = exp_table[softmax_idx_from_real_val<typename CONFIG_T::inp_norm_t, CONFIG_T::exp_table_size>(d)];
                acc += ev;
            }
            eg[l] = ev;
            part[l] = acc;
        }
        es.write(eg);
        if (s == F::steps - 1) {
            typename CONFIG_T::accum_t sum = 0;
            #pragma hls_unroll yes
            for (unsigned l = 0; l < F::lanes; l++) {
                sum += part[l];
            }
            typename CONFIG_T::inv_inp_t exp_sum = sum;
            invs.write(invert_table[softmax_idx_from_real_val<typename CONFIG_T::inv_inp_t, CONFIG_T::inv_table_size>(exp_sum)]);
            s = 0;
        } else {
            s++;
        }
    }
}

#pragma hls_design block
template <class data_T, class res_T, typename CONFIG_T>
void row_normalize(ac_channel<typename fold<data_T, CONFIG_T>::e_group_t> &es,
                   ac_channel<typename CONFIG_T::inv_table_t> &invs, ac_channel<res_T> &res) {
    typedef fold<data_T, CONFIG_T> F;
    res_T y;
    typename CONFIG_T::inv_table_t inv = 0;
    unsigned s = 0;
    #pragma hls_pipeline_init_interval 1
SoftmaxNormalizeLoop:
    for (unsigned n = 0; n < F::rows * F::steps; n++) {
        if (s == 0)
            inv = invs.read();
        typename F::e_group_t eg = es.read();
        #pragma hls_unroll yes
        for (unsigned l = 0; l < F::lanes; l++) {
            unsigned j = s * F::lanes + l;
            if (j < res_T::size)
                y[j] = eg[l] * inv;
        }
        if (s == F::steps - 1) {
            res.write(y);
            s = 0;
        } else {
            s++;
        }
    }
}

} // namespace softmax_resource

template <class data_T, class res_T, typename CONFIG_T>
void softmax_stable_resource(ac_channel<data_T> &data, ac_channel<res_T> &res,
                             typename CONFIG_T::exp_table_t exp_table[CONFIG_T::exp_table_size],
                             typename CONFIG_T::inv_table_t invert_table[CONFIG_T::inv_table_size]) {
    typedef softmax_resource::fold<data_T, CONFIG_T> F;
    static ac_channel<typename F::x_group_t> x_stream;
    static ac_channel<typename data_T::value_type> max_stream;
    static ac_channel<typename F::e_group_t> e_stream;
    static ac_channel<typename CONFIG_T::inv_table_t> inv_stream;

    softmax_resource::row_max<data_T, CONFIG_T>(data, x_stream, max_stream);
    softmax_resource::row_exp<data_T, CONFIG_T>(x_stream, max_stream, e_stream, inv_stream, exp_table, invert_table);
    softmax_resource::row_normalize<data_T, res_T, CONFIG_T>(e_stream, inv_stream, res);
}
#endif // HLS4ML_SOFTMAX_AC_MATH

// Table forms (latency / stable): tables come in as constant weight arrays.
template <class data_T, class res_T, typename CONFIG_T>
void softmax(ac_channel<data_T> &data, ac_channel<res_T> &res,
             typename CONFIG_T::exp_table_t exp_table[CONFIG_T::exp_table_size],
             typename CONFIG_T::inv_table_t invert_table[CONFIG_T::inv_table_size]) {
    static_assert(CONFIG_T::axis == -1, "io_stream softmax normalizes along the last axis only");
    static_assert(CONFIG_T::implementation == softmax_implementation::latency ||
                      CONFIG_T::implementation == softmax_implementation::stable,
                  "table softmax called for a non-table implementation");
#ifdef HLS4ML_SOFTMAX_AC_MATH
    (void)exp_table;
    (void)invert_table;
    softmax_ac_math<data_T, res_T, CONFIG_T>(data, res);
#else
    if constexpr (CONFIG_T::implementation == softmax_implementation::latency) {
        softmax_latency<data_T, res_T, CONFIG_T>(data, res, exp_table, invert_table);
    } else if constexpr (CONFIG_T::strategy == nnet::resource) {
        softmax_stable_resource<data_T, res_T, CONFIG_T>(data, res, exp_table, invert_table);
    } else {
        softmax_stable<data_T, res_T, CONFIG_T>(data, res, exp_table, invert_table);
    }
#endif
}

// Table-free forms (legacy / argmax).
template <class data_T, class res_T, typename CONFIG_T> void softmax(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    static_assert(CONFIG_T::axis == -1, "io_stream softmax normalizes along the last axis only");
    static_assert(CONFIG_T::implementation == softmax_implementation::legacy ||
                      CONFIG_T::implementation == softmax_implementation::argmax,
                  "latency / stable softmax needs its exp and invert tables");
    if constexpr (CONFIG_T::implementation == softmax_implementation::legacy) {
        softmax_legacy<data_T, res_T, CONFIG_T>(data, res);
    } else {
        softmax_argmax<data_T, res_T, CONFIG_T>(data, res);
    }
}

// *************************************************
//       TanH Activation
// *************************************************

#ifndef USE_AC_MATH

template <class data_T, class res_T, typename CONFIG_T> void tanh(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t tanh_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t tanh_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_tanh_table<CONFIG_T, CONFIG_T::table_size>(tanh_table);
        initialized = true;
    }

#pragma hls_pipeline_init_interval 1
TanHActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;

    #pragma hls_unroll
    TanHPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            int data_round = in_data[j].to_double() * (int)CONFIG_T::table_size / 8;
            int index = data_round + 4 * (int)CONFIG_T::table_size / 8;
            if (index < 0)
                index = 0;
            else if (index > CONFIG_T::table_size - 1)
                index = (int)CONFIG_T::table_size - 1;
            out_data[j] = tanh_table[index];
        }

        res.write(out_data);
    }
}

#else

template <class data_T, class res_T, typename CONFIG_T> void tanh(ac_channel<data_T> &data, ac_channel<res_T> &res) {
TanHActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {

        data_T in_data = data.read();
        res_T out_data;
    TanHPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            // int data_round = in_data[j]*CONFIG_T::table_size/8;
            ac_math::ac_tanh_pwl(in_data[j], out_data[j]);
        }
        res.write(out_data);
    }
}

#endif

// *************************************************
//       UnaryLUT Activation
// *************************************************
// Two implementations, selected by CONFIG_T::strategy like dense:
//  - latency:  every element of a beat looked up in parallel from the table.
//  - resource: the beat is folded over reuse_factor cycles, ceil(size / reuse_factor)
//              lookups per cycle, each from a memory copy of the table (a dual-port copy
//              serves two lanes).
template <class data_T, class res_T, typename CONFIG_T>
void unary_lut_latency(ac_channel<data_T> &data, ac_channel<res_T> &res,
                       typename CONFIG_T::table_t table[CONFIG_T::table_size]) {
    // Vivado: #pragma HLS PIPELINE II=CONFIG_T::reuse_factor (Catapult takes a constexpr name)
    constexpr int ce_reuse_factor = CONFIG_T::reuse_factor;
    (void)ce_reuse_factor;
    // Vitis treats PIPELINE II=reuse_factor as a target it may beat (it achieves 1 here); Catapult
    // pipelines at exactly the requested II, so the stream driver runs at 1 and the reuse
    // factor sets the rate through the inner reuse loop where it applies.
    #pragma hls_pipeline_init_interval 1
UnaryLUTActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;

        #pragma hls_unroll
    UnaryLUTPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            // CATAPULT_PORT
            // Vivado: get_index_unary_lut<...>(in_data[j].V);  // .V == raw fixed-point word
            unsigned index = get_index_unary_lut<CONFIG_T::table_size>(in_data[j]);
            out_data[j] = table[index];
        }

        res.write(out_data);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void unary_lut_resource(ac_channel<data_T> &data, ac_channel<res_T> &res,
                        typename CONFIG_T::table_t table[CONFIG_T::table_size]) {
    static const unsigned rf = CONFIG_T::reuse_factor;
    static const unsigned lanes = DIV_ROUNDUP(data_T::size, rf);
    static const unsigned n_copies = DIV_ROUNDUP(lanes, 2);

    // Memory copies of the table, filled on the first call (the table is constant).
    static typename CONFIG_T::table_t table_mem[n_copies][CONFIG_T::table_size];
    static bool table_loaded = false;
    if (!table_loaded) {
        #pragma hls_pipeline_init_interval 1
    UnaryLUTLoadTable:
        for (int t = 0; t < CONFIG_T::table_size; t++) {
            #pragma hls_unroll yes
            for (int k = 0; k < n_copies; k++) {
                table_mem[k][t] = table[t];
            }
        }
        table_loaded = true;
    }

    data_T in_data;
    res_T out_data;

    // One flat loop over (beat, fold step) so consecutive beats don't restart the pipeline.
    unsigned c = 0;
    #pragma hls_pipeline_init_interval 1
UnaryLUTFoldLoop:
    for (int n = 0; n < CONFIG_T::n_in / data_T::size * rf; n++) {
        if (c == 0)
            in_data = data.read();
        #pragma hls_unroll yes
        for (int l = 0; l < lanes; l++) {
            int j = c * lanes + l;
            if (j < data_T::size) {
                unsigned index = get_index_unary_lut<CONFIG_T::table_size>(in_data[j]);
                out_data[j] = table_mem[l / 2][index];
            }
        }
        if (c == rf - 1) {
            res.write(out_data);
            c = 0;
        } else {
            c++;
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void unary_lut(ac_channel<data_T> &data, ac_channel<res_T> &res, typename CONFIG_T::table_t table[CONFIG_T::table_size]) {
    if constexpr (CONFIG_T::strategy == nnet::resource) {
        unary_lut_resource<data_T, res_T, CONFIG_T>(data, res, table);
    } else {
        unary_lut_latency<data_T, res_T, CONFIG_T>(data, res, table);
    }
}

// *************************************************
//       Hard sigmoid Activation
// *************************************************

template <class data_T, class res_T, typename CONFIG_T> void hard_sigmoid(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    typename data_T::value_type slope = (typename data_T::value_type)0.2;
    typename data_T::value_type shift = (typename data_T::value_type)0.5;

#pragma hls_pipeline_init_interval 1
HardSigmoidActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;

    #pragma hls_unroll
    HardSigmoidPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            typename data_T::value_type datareg = slope * in_data[j] + shift;
            if (datareg > 1)
                datareg = 1;
            else if (datareg < 0)
                datareg = 0;
            out_data[j] = datareg;
        }

        res.write(out_data);
    }
}

// *************************************************
//       Hard TanH Activation
// *************************************************

template <class data_T, class res_T, typename CONFIG_T> void hard_tanh(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    // typename data_T::value_type slope = (typename data_T::value_type) 0.2;
    // typename data_T::value_type shift = (typename data_T::value_type) 0.5;

#pragma hls_pipeline_init_interval 1
HardTanhActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;
        // PRAGMA_DATA_PACK(out_data)

    #pragma hls_unroll
    HardTanhPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            auto sigmoid = CONFIG_T::slope * in_data[j] + CONFIG_T::shift;
            if (sigmoid > 1)
                sigmoid = 1;
            else if (sigmoid < 0)
                sigmoid = 0;
            out_data[j] = 2 * sigmoid - 1;
        }

        res.write(out_data);
    }
}

// *************************************************
//       Leaky RELU Activation
// *************************************************
template <class data_T, class param_T, class res_T, typename CONFIG_T>
void leaky_relu(ac_channel<data_T> &data, param_T alpha, ac_channel<res_T> &res) {
#pragma hls_pipeline_init_interval 1
LeakyReLUActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;

    #pragma hls_unroll
    LeakyReLUPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            if (in_data[j] > 0)
                out_data[j] = in_data[j];
            else
                out_data[j] = alpha * in_data[j];
        }
        res.write(out_data);
    }
}

// *************************************************
//       Thresholded RELU Activation
// *************************************************

template <class data_T, class param_T, class res_T, typename CONFIG_T>
void thresholded_relu(ac_channel<data_T> &data, param_T theta, ac_channel<res_T> &res) {
#pragma hls_pipeline_init_interval 1
ThresholdedReLUActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;

    #pragma hls_unroll
    ThresholdedReLUPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            if (in_data[j] > theta)
                out_data[j] = in_data[j];
            else
                out_data[j] = 0;
        }

        res.write(out_data);
    }
}

// *************************************************
//       Softplus Activation
// *************************************************

#ifndef USE_AC_MATH

template <class data_T, class res_T, typename CONFIG_T> void softplus(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t softplus_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t softplus_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_softplus_table<CONFIG_T, CONFIG_T::table_size>(softplus_table);
        initialized = true;
    }

#pragma hls_pipeline_init_interval 1
SoftplusActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;

    #pragma hls_unroll
    SoftplusPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            int data_round = in_data[j].to_double() * (int)CONFIG_T::table_size / 16;
            int index = data_round + 8 * (int)CONFIG_T::table_size / 16;
            if (index < 0)
                index = 0;
            else if (index > CONFIG_T::table_size - 1)
                index = (int)CONFIG_T::table_size - 1;
            out_data[j] = softplus_table[index];
        }
        res.write(out_data);
    }
}

#else

template <class data_T, class res_T, typename CONFIG_T> void softplus(ac_channel<data_T> &data, ac_channel<res_T> &res) {
SoftplusActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;
    SoftplusPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            ac_softplus_pwl_wrapper(in_data[j], out_data[j]);
        }
        res.write(out_data);
    }
}

#endif

// *************************************************
//       Softsign Activation
// *************************************************

#ifndef USE_AC_MATH

template <class data_T, class res_T, typename CONFIG_T> void softsign(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t softsign_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t softsign_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_softsign_table<CONFIG_T, CONFIG_T::table_size>(softsign_table);
        initialized = true;
    }

#pragma hls_pipeline_init_interval 1
SoftsignActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;

    #pragma hls_unroll
    SoftsignPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            int data_round = in_data[j].to_double() * (int)CONFIG_T::table_size / 16;
            int index = data_round + 8 * (int)CONFIG_T::table_size / 16;
            if (index < 0)
                index = 0;
            else if (index > CONFIG_T::table_size - 1)
                index = (int)CONFIG_T::table_size - 1;
            out_data[j] = softsign_table[index];
        }
        res.write(out_data);
    }
}

#else

template <class data_T, class res_T, typename CONFIG_T> void softsign(ac_channel<data_T> &data, ac_channel<res_T> &res) {
SoftsignActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;
    SoftsignPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            ac_math::ac_softsign_pwl(in_data[j], out_data[j]);
        }
        res.write(out_data);
    }
}

#endif

// *************************************************
//       ELU Activation
// *************************************************

#ifndef USE_AC_MATH

template <class data_T, class param_T, class res_T, typename CONFIG_T>
void elu(ac_channel<data_T> &data, param_T alpha, ac_channel<res_T> &res) {
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t elu_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t elu_table[CONFIG_T::table_size];
#endif

    if (!initialized) {
        init_elu_table<CONFIG_T, CONFIG_T::table_size>(elu_table);
        initialized = true;
    }

#pragma hls_pipeline_init_interval 1
EluActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;

    #pragma hls_unroll
    EluPackLoop:
        for (int j = 0; j < res_T::size; j++) {

            typename data_T::value_type datareg = in_data[j];
            if (datareg >= 0) {
                out_data[j] = datareg;
            } else {
                int index = (int)datareg.to_double() * (int)CONFIG_T::table_size / -8;
                if (index > CONFIG_T::table_size - 1)
                    index = CONFIG_T::table_size - 1;
                out_data[j] = alpha * elu_table[index];
            }
        }
        res.write(out_data);
    }
}

#else
template <class data_T, class param_T, class res_T, typename CONFIG_T>
void elu(ac_channel<data_T> &data, param_T alpha, ac_channel<res_T> &res) {
EluActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;
    EluPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            ac_math::ac_elu_pwl(in_data[j], out_data[j], alpha);
        }
        res.write(out_data);
    }
}

#endif

// *************************************************
//       SELU Activation
// *************************************************

#ifndef USE_AC_MATH

template <class data_T, class res_T, typename CONFIG_T> void selu(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t selu_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t selu_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_selu_table<CONFIG_T, CONFIG_T::table_size>(selu_table);
        initialized = true;
    }

#pragma hls_pipeline_init_interval 1
SeluActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;

    #pragma hls_unroll
    SeluPackLoop:
        for (int j = 0; j < res_T::size; j++) {

            typename data_T::value_type datareg = in_data[j];
            if (datareg >= 0) {
                out_data[j] = (typename data_T::value_type)1.0507009873554804934193349852946 * datareg;
            } else {
                int index = (int)datareg.to_double() * (int)CONFIG_T::table_size / -8;
                if (index > CONFIG_T::table_size - 1)
                    index = (int)CONFIG_T::table_size - 1;
                out_data[j] = selu_table[index];
            }
        }
        res.write(out_data);
    }
}

#else

template <class data_T, class res_T, typename CONFIG_T> void selu(ac_channel<data_T> &data, ac_channel<res_T> &res) {
SeluActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;
    SeluPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            ac_math::ac_selu_pwl(in_data[j], out_data[j]);
        }
        res.write(out_data);
    }
}

#endif

// *************************************************
//       PReLU Activation
// *************************************************
template <class data_T, class param_T, class res_T, typename CONFIG_T>
void prelu(ac_channel<data_T> &data, const param_T alpha[CONFIG_T::n_in], ac_channel<res_T> &res) {
#pragma hls_pipeline_init_interval 1
PReLUActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;

    #pragma hls_unroll
    PReLUPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            if (in_data[j] > 0)
                out_data[j] = in_data[j];
            else
                out_data[j] = alpha[i * res_T::size + j] * in_data[j];
        }
        res.write(out_data);
    }
}

// *************************************************
//       Binary TanH Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T> void binary_tanh(ac_channel<data_T> &data, ac_channel<res_T> &res) {
    using cache_T = ac_int<2, true>;
#pragma hls_pipeline_init_interval 1
PReLUActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;

    #pragma hls_unroll
    PReLUPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            cache_T cache;

            if (in_data[j] >= 0)
                cache = 1;
            else
                cache = -1;

            out_data[j] = binary_cast<cache_T, typename res_T::value_type>(cache);
        }
        res.write(out_data);
    }
}

// *************************************************
//       Ternary TanH Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T> void ternary_tanh(ac_channel<data_T> &data, ac_channel<res_T> &res) {
#pragma hls_pipeline_init_interval 1
PReLUActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;

    #pragma hls_unroll
    PReLUPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            if (in_data[j] > 1)
                out_data[j] = (typename res_T::value_type)1;
            else if (in_data[j] <= -1)
                out_data[j] = (typename res_T::value_type) - 1;
            else
                out_data[j] = (typename res_T::value_type)0;
        }
        res.write(out_data);
    }
}

} // namespace nnet

#endif
