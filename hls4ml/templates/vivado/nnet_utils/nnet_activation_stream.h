#ifndef NNET_ACTIVATION_STREAM_H_
#define NNET_ACTIVATION_STREAM_H_

#include "ap_fixed.h"
#include "hls_stream.h"
#include "nnet_activation.h"
#include "nnet_common.h"
#include "nnet_stream.h"
#include "nnet_types.h"
#include <cmath>

namespace nnet {

// *************************************************
//       LINEAR Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T> void linear(hls::stream<data_T> &data, hls::stream<res_T> &res) {
LinearActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    LinearPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            out_data[j] = in_data[j];
        }

        res.write(out_data);
    }
}

// *************************************************
//       RELU Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T> void relu(hls::stream<data_T> &data, hls::stream<res_T> &res) {
ReLUActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    ReLUPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            if (in_data[j] > 0)
                out_data[j] = in_data[j];
            else
                out_data[j] = 0;
        }

        res.write(out_data);
    }
}

// *************************************************
//       Sigmoid Activation
// *************************************************

template <class data_T, class res_T, typename CONFIG_T> void sigmoid(hls::stream<data_T> &data, hls::stream<res_T> &res) {
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

SigmoidActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    SigmoidPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            int data_round = in_data[j] * CONFIG_T::table_size / 16;
            int index = data_round + 8 * CONFIG_T::table_size / 16;
            if (index < 0)
                index = 0;
            else if (index > CONFIG_T::table_size - 1)
                index = CONFIG_T::table_size - 1;
            out_data[j] = sigmoid_table[index];
        }

        res.write(out_data);
    }
}

// *************************************************
//       Softmax Activation
// *************************************************

template <class data_T, class res_T, typename CONFIG_T>
void softmax_latency(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    // Initialize the lookup tables
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::exp_table_t exp_table[CONFIG_T::exp_table_size];
    typename CONFIG_T::inv_table_t invert_table[CONFIG_T::inv_table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::exp_table_t exp_table[CONFIG_T::exp_table_size];
    static typename CONFIG_T::inv_table_t invert_table[CONFIG_T::inv_table_size];

#endif
    if (!initialized) {
        // Note we are exponentiating the inputs, which have type data_T
        init_exp_table<typename data_T::value_type, CONFIG_T>(exp_table);
        // Note we are inverting the exponentials, which have type exp_table_t
        init_invert_table<typename CONFIG_T::inv_inp_t, CONFIG_T>(invert_table);
        initialized = true;
    }

    constexpr unsigned multiplier_limit = DIV_ROUNDUP(data_T::size, CONFIG_T::reuse_factor);
    constexpr unsigned ii = data_T::size / multiplier_limit;

    // Calculate all the e^x's
    typename CONFIG_T::accum_t exp_res[data_T::size];
    #pragma HLS array_partition variable=exp_res complete
    typename CONFIG_T::inv_inp_t exp_sum(0);
SoftmaxExpLoop:
    for (unsigned i = 0; i < CONFIG_T::n_in / data_T::size; i++) {
        #pragma HLS PIPELINE II=ii rewind

        data_T in_pack = data.read();
    SoftmaxExpPackLoop:
        for (unsigned j = 0; j < data_T::size; j++) {
            #pragma HLS UNROLL
            unsigned x = softmax_idx_from_real_val<typename data_T::value_type, CONFIG_T::exp_table_size>(in_pack[j]);
            exp_res[j] = exp_table[x];
        }

        // Explicitly sum the results with an adder tree.
        // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
        Op_add<typename CONFIG_T::accum_t> op_add;
        exp_sum = reduce<typename CONFIG_T::accum_t, data_T::size, Op_add<typename CONFIG_T::accum_t>>(exp_res, op_add);

        typename CONFIG_T::inv_table_t inv_exp_sum =
            invert_table[softmax_idx_from_real_val<typename CONFIG_T::inv_inp_t, CONFIG_T::inv_table_size>(exp_sum)];

        res_T out_pack;
        PRAGMA_DATA_PACK(out_pack)

    SoftmaxInvPackLoop:
        for (unsigned j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            #pragma HLS ALLOCATION operation instances=mul limit=multiplier_limit
            out_pack[j] = exp_res[j] * inv_exp_sum;
        }
        res.write(out_pack);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void softmax_stable(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    // Initialize the lookup tables
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::exp_table_t exp_table[CONFIG_T::exp_table_size];
    typename CONFIG_T::inv_table_t invert_table[CONFIG_T::inv_table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::exp_table_t exp_table[CONFIG_T::exp_table_size];
    static typename CONFIG_T::inv_table_t invert_table[CONFIG_T::inv_table_size];

#endif
    if (!initialized) {
        // Note we are exponentiating the inputs, which have type data_T
        init_exp_table<typename CONFIG_T::inp_norm_t, CONFIG_T>(exp_table, true);
        // Note we are inverting the exponentials, which have type exp_table_t
        init_invert_table<typename CONFIG_T::inv_inp_t, CONFIG_T>(invert_table);
        initialized = true;
    }

    constexpr unsigned multiplier_limit = DIV_ROUNDUP(data_T::size, CONFIG_T::reuse_factor);
    constexpr unsigned ii = data_T::size / multiplier_limit;

    typename data_T::value_type data_array[data_T::size];
#pragma HLS ARRAY_PARTITION variable=data_array complete
SoftmaxArrayLoop:
    for (unsigned i = 0; i < CONFIG_T::n_in / data_T::size; i++) {
        #pragma HLS PIPELINE II=ii rewind

        data_T in_pack = data.read();
    SoftmaxArrayPackLoop:
        for (unsigned j = 0; j < data_T::size; j++) {
            #pragma HLS UNROLL
            data_array[j] = in_pack[j];
        }

        // Find the max and compute all delta(x_i, x_max)
        Op_max<typename data_T::value_type> op_max;
        typename data_T::value_type x_max =
            reduce<typename data_T::value_type, data_T::size, Op_max<typename data_T::value_type>>(data_array, op_max);

        typename CONFIG_T::inp_norm_t d_xi_xmax[data_T::size];
        for (unsigned j = 0; j < data_T::size; j++) {
            #pragma HLS UNROLL
            d_xi_xmax[j] = x_max - data_array[j];
        }

        // Calculate all the e^x's
        typename CONFIG_T::accum_t exp_res[data_T::size];
        #pragma HLS ARRAY_PARTITION variable=exp_res complete
        typename CONFIG_T::inv_inp_t exp_sum(0);
        for (unsigned j = 0; j < data_T::size; j++) {
            #pragma HLS UNROLL
            unsigned x = softmax_idx_from_real_val<typename CONFIG_T::inp_norm_t, CONFIG_T::exp_table_size>(d_xi_xmax[j]);
            exp_res[j] = exp_table[x];
        }

        // Explicitly sum the results with an adder tree.
        // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
        Op_add<typename CONFIG_T::accum_t> op_add;
        exp_sum = reduce<typename CONFIG_T::accum_t, data_T::size, Op_add<typename CONFIG_T::accum_t>>(exp_res, op_add);

        typename CONFIG_T::inv_table_t inv_exp_sum =
            invert_table[softmax_idx_from_real_val<typename CONFIG_T::inv_inp_t, CONFIG_T::inv_table_size>(exp_sum)];

        res_T out_pack;
        PRAGMA_DATA_PACK(out_pack)

    SoftmaxInvPackLoop:
        for (unsigned j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            #pragma HLS ALLOCATION operation instances=mul limit=multiplier_limit
            out_pack[j] = exp_res[j] * inv_exp_sum;
        }
        res.write(out_pack);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void softmax_legacy(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t exp_table[CONFIG_T::table_size];
    typename CONFIG_T::table_t invert_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t exp_table[CONFIG_T::table_size];
    static typename CONFIG_T::table_t invert_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_exp_table_legacy<CONFIG_T, CONFIG_T::table_size>(exp_table);
        init_invert_table_legacy<CONFIG_T, CONFIG_T::table_size>(invert_table);
        initialized = true;
    }

    // Index into the lookup table based on data for exponentials
    typename CONFIG_T::table_t exp_res[data_T::size];
    typename CONFIG_T::table_t exp_diff_res;
    typename data_T::value_type data_cache[data_T::size];

SoftmaxInitLoop:
    for (unsigned s = 0; s < CONFIG_T::n_in / data_T::size; s++) {
        #pragma HLS PIPELINE
        data_T in_pack = data.read();
    SoftmaxInitPackLoop:
        for (unsigned j = 0; j < data_T::size; j++) {
            #pragma HLS UNROLL
            data_cache[j] = in_pack[j];
            exp_res[j] = 0;
        }

    SoftmaxExpLoop:
        for (int i = 0; i < data_T::size; i++) {
        #pragma HLS UNROLL
        SoftmaxExpInner:
            for (int j = 0; j < data_T::size; j++) {
                #pragma HLS UNROLL

                if (i == j) {
                    exp_diff_res = 1;
                } else {
                    int data_round = (data_cache[j] - data_cache[i]) * CONFIG_T::table_size / 16;
                    int index = data_round + 8 * CONFIG_T::table_size / 16;
                    if (index < 0)
                        index = 0;
                    if (index > CONFIG_T::table_size - 1)
                        index = CONFIG_T::table_size - 1;
                    exp_diff_res = exp_table[index];
                }

                exp_res[i] += exp_diff_res;
            }
        }

        res_T out_pack;
        PRAGMA_DATA_PACK(out_pack)

    SoftmaxInvPackLoop:
        for (unsigned j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL

            int exp_res_index = exp_res[j] * CONFIG_T::table_size / 64;
            if (exp_res_index < 0)
                exp_res_index = 0;
            if (exp_res_index > CONFIG_T::table_size - 1)
                exp_res_index = CONFIG_T::table_size - 1;

            out_pack[j] = (typename res_T::value_type)invert_table[exp_res_index];
        }
        res.write(out_pack);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void softmax_argmax(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE
        data_T in_data = data.read();
        res_T out_data;

        for (int i = 0; i < res_T::size; i++) {
            #pragma HLS UNROLL
            out_data[i] = (typename res_T::value_type)0;
        }

        typename data_T::value_type maximum = in_data[0];
        int idx = 0;

        for (int i = 1; i < res_T::size; i++) {
            #pragma HLS PIPELINE
            if (in_data[i] > maximum) {
                maximum = in_data[i];
                idx = i;
            }
        }

        out_data[idx] = (typename res_T::value_type)1;
        res.write(out_data);
    }
}

// Resource strategy for the stable softmax. One row (one beat) is accepted every
// reuse_factor cycles: lanes = ceil(size / reuse_factor) elements per cycle, and rows overlap
// across three DATAFLOW stages (max | exp + sum | normalize), each one flat II=1 loop over
// (row, fold step). Only lane groups and one scalar per row flow between stages, through
// FIFOs two rows deep, so no stage holds more than one row. The tables live in BRAM, the
// exp table replicated per two lanes. Bit-exact to softmax_stable: the exp sum is exact in
// accum_t and rounded once into inv_inp_t, so the summation order doesn't matter.
namespace softmax_resource {

template <class data_T, typename CONFIG_T> struct fold {
    static const unsigned lanes = DIV_ROUNDUP(data_T::size, CONFIG_T::reuse_factor);
    static const unsigned steps = DIV_ROUNDUP(data_T::size, lanes);
    static const unsigned rows = CONFIG_T::n_in / data_T::size;
    static const unsigned exp_copies = DIV_ROUNDUP(lanes, 2);
    typedef array<typename data_T::value_type, lanes> x_group_t;
    typedef array<typename CONFIG_T::accum_t, lanes> e_group_t;
};

template <class data_T, typename CONFIG_T>
void row_max(hls::stream<data_T> &data, hls::stream<typename fold<data_T, CONFIG_T>::x_group_t> &xs,
             hls::stream<typename data_T::value_type> &maxs) {
    typedef fold<data_T, CONFIG_T> F;
    data_T x;
    typename data_T::value_type mx[F::lanes];
    #pragma HLS ARRAY_PARTITION variable=mx complete
    unsigned s = 0;
SoftmaxMaxLoop:
    for (unsigned n = 0; n < F::rows * F::steps; n++) {
        #pragma HLS PIPELINE II=1
        if (s == 0)
            x = data.read();
        typename F::x_group_t g;
        for (unsigned l = 0; l < F::lanes; l++) {
            #pragma HLS UNROLL
            unsigned j = s * F::lanes + l;
            typename data_T::value_type v = (j < data_T::size) ? x[j] : x[0];
            g[l] = v;
            if (s == 0 || v > mx[l])
                mx[l] = v;
        }
        xs.write(g);
        if (s == F::steps - 1) {
            typename data_T::value_type m = mx[0];
            for (unsigned l = 1; l < F::lanes; l++) {
                #pragma HLS UNROLL
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

template <class data_T, typename CONFIG_T>
void row_exp(hls::stream<typename fold<data_T, CONFIG_T>::x_group_t> &xs, hls::stream<typename data_T::value_type> &maxs,
             hls::stream<typename fold<data_T, CONFIG_T>::e_group_t> &es,
             hls::stream<typename CONFIG_T::inv_table_t> &invs) {
    typedef fold<data_T, CONFIG_T> F;
    // Tables built as softmax_stable builds them (plain locals under synthesis, so the
    // constant init loop is evaluated at compile time into ROM contents).
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::exp_table_t exp_table[F::exp_copies][CONFIG_T::exp_table_size];
    typename CONFIG_T::inv_table_t invert_table[CONFIG_T::inv_table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::exp_table_t exp_table[F::exp_copies][CONFIG_T::exp_table_size];
    static typename CONFIG_T::inv_table_t invert_table[CONFIG_T::inv_table_size];
#endif
    #pragma HLS ARRAY_PARTITION variable=exp_table dim=1 complete
    #pragma HLS BIND_STORAGE variable=exp_table type=rom_2p impl=bram
    #pragma HLS BIND_STORAGE variable=invert_table type=rom_1p impl=bram
    if (!initialized) {
        for (unsigned k = 0; k < F::exp_copies; k++)
            init_exp_table<typename CONFIG_T::inp_norm_t, CONFIG_T>(exp_table[k], true);
        init_invert_table<typename CONFIG_T::inv_inp_t, CONFIG_T>(invert_table);
        initialized = true;
    }

    typename data_T::value_type x_max = 0;
    typename CONFIG_T::accum_t part[F::lanes];
    #pragma HLS ARRAY_PARTITION variable=part complete
    unsigned s = 0;
SoftmaxExpLoop:
    for (unsigned n = 0; n < F::rows * F::steps; n++) {
        #pragma HLS PIPELINE II=1
        if (s == 0)
            x_max = maxs.read();
        typename F::x_group_t g = xs.read();
        typename F::e_group_t eg;
        for (unsigned l = 0; l < F::lanes; l++) {
            #pragma HLS UNROLL
            unsigned j = s * F::lanes + l;
            typename CONFIG_T::accum_t acc = (s == 0) ? typename CONFIG_T::accum_t(0) : part[l];
            typename CONFIG_T::accum_t ev = 0;
            if (j < data_T::size) {
                typename CONFIG_T::inp_norm_t d = x_max - g[l];
                ev = exp_table[l / 2][softmax_idx_from_real_val<typename CONFIG_T::inp_norm_t, CONFIG_T::exp_table_size>(d)];
                acc += ev;
            }
            eg[l] = ev;
            part[l] = acc;
        }
        es.write(eg);
        if (s == F::steps - 1) {
            typename CONFIG_T::accum_t sum = 0;
            for (unsigned l = 0; l < F::lanes; l++) {
                #pragma HLS UNROLL
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

template <class data_T, class res_T, typename CONFIG_T>
void row_normalize(hls::stream<typename fold<data_T, CONFIG_T>::e_group_t> &es,
                   hls::stream<typename CONFIG_T::inv_table_t> &invs, hls::stream<res_T> &res) {
    typedef fold<data_T, CONFIG_T> F;
    res_T y;
    PRAGMA_DATA_PACK(y)
    typename CONFIG_T::inv_table_t inv = 0;
    unsigned s = 0;
SoftmaxNormalizeLoop:
    for (unsigned n = 0; n < F::rows * F::steps; n++) {
        #pragma HLS PIPELINE II=1
        if (s == 0)
            inv = invs.read();
        typename F::e_group_t eg = es.read();
        for (unsigned l = 0; l < F::lanes; l++) {
            #pragma HLS UNROLL
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
void softmax_stable_resource(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    #pragma HLS DATAFLOW
    typedef softmax_resource::fold<data_T, CONFIG_T> F;
    static const unsigned group_depth = 2 * F::steps;

    hls::stream<typename F::x_group_t> x_stream("softmax_x");
    hls::stream<typename data_T::value_type> max_stream("softmax_max");
    hls::stream<typename F::e_group_t> e_stream("softmax_e");
    hls::stream<typename CONFIG_T::inv_table_t> inv_stream("softmax_inv");
    #pragma HLS STREAM variable=x_stream depth=group_depth
    #pragma HLS STREAM variable=max_stream depth=2
    #pragma HLS STREAM variable=e_stream depth=group_depth
    #pragma HLS STREAM variable=inv_stream depth=2

    softmax_resource::row_max<data_T, CONFIG_T>(data, x_stream, max_stream);
    softmax_resource::row_exp<data_T, CONFIG_T>(x_stream, max_stream, e_stream, inv_stream);
    softmax_resource::row_normalize<data_T, res_T, CONFIG_T>(e_stream, inv_stream, res);
}

template <class data_T, class res_T, typename CONFIG_T> void softmax(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    // Inline the dispatcher: otherwise it is the dataflow process and, being an
    // unpipelined wrapper around the kernel, it re-runs the kernel's full latency
    // every frame; inlining makes the kernel's rewound row loop the process body.
    #pragma HLS INLINE
    assert(CONFIG_T::axis == -1);

    switch (CONFIG_T::implementation) {
    case softmax_implementation::latency:
        softmax_latency<data_T, res_T, CONFIG_T>(data, res);
        break;
    case softmax_implementation::stable:
        if (CONFIG_T::strategy == nnet::resource) {
            softmax_stable_resource<data_T, res_T, CONFIG_T>(data, res);
        } else {
            softmax_stable<data_T, res_T, CONFIG_T>(data, res);
        }
        break;
    case softmax_implementation::legacy:
        softmax_legacy<data_T, res_T, CONFIG_T>(data, res);
        break;
    case softmax_implementation::argmax:
        softmax_argmax<data_T, res_T, CONFIG_T>(data, res);
        break;
    }
}

// *************************************************
//       TanH Activation
// *************************************************

template <class data_T, class res_T, typename CONFIG_T> void tanh(hls::stream<data_T> &data, hls::stream<res_T> &res) {
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

TanHActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    TanHPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            int data_round = in_data[j] * CONFIG_T::table_size / 8;
            int index = data_round + 4 * CONFIG_T::table_size / 8;
            if (index < 0)
                index = 0;
            else if (index > CONFIG_T::table_size - 1)
                index = CONFIG_T::table_size - 1;
            out_data[j] = tanh_table[index];
        }

        res.write(out_data);
    }
}

// *************************************************
//       UnaryLUT Activation
// *************************************************

// Two implementations, selected by CONFIG_T::strategy like dense:
//  - latency:  every element of a beat looked up in parallel from a register table.
//  - resource: the beat is folded over reuse_factor cycles, ceil(size / reuse_factor)
//              lookups per cycle, each from a BRAM copy of the table (a dual-port copy
//              serves two lanes).

template <class data_T, class res_T, typename CONFIG_T>
void unary_lut_latency(hls::stream<data_T> &data, hls::stream<res_T> &res,
                       typename CONFIG_T::table_t table[CONFIG_T::table_size]) {
    #pragma HLS function_instantiate variable=table
    #pragma HLS ARRAY_PARTITION variable=table complete

UnaryLUTActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor rewind

        data_T in_data = data.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    UnaryLUTPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            unsigned index = get_index_unary_lut<CONFIG_T::table_size>(in_data[j]);
            out_data[j] = table[index];
        }

        res.write(out_data);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void unary_lut_resource(hls::stream<data_T> &data, hls::stream<res_T> &res,
                        typename CONFIG_T::table_t table[CONFIG_T::table_size]) {
    static const unsigned rf = CONFIG_T::reuse_factor;
    static const unsigned lanes = DIV_ROUNDUP(data_T::size, rf);
    static const unsigned n_copies = DIV_ROUNDUP(lanes, 2);

    // BRAM copies of the table, filled on the first call (the table is constant).
    static typename CONFIG_T::table_t table_bram[n_copies][CONFIG_T::table_size];
    #pragma HLS ARRAY_PARTITION variable=table_bram dim=1 complete
    #pragma HLS BIND_STORAGE variable=table_bram type=ram_2p impl=bram
    static bool table_loaded = false;
    if (!table_loaded) {
    UnaryLUTLoadTable:
        for (int t = 0; t < CONFIG_T::table_size; t++) {
            #pragma HLS PIPELINE II=1
            for (int k = 0; k < n_copies; k++) {
                #pragma HLS UNROLL
                table_bram[k][t] = table[t];
            }
        }
        table_loaded = true;
    }

    data_T in_data;
    res_T out_data;
    PRAGMA_DATA_PACK(out_data)

    // One flat loop over (beat, fold step) so consecutive beats don't restart the pipeline.
    unsigned c = 0;
UnaryLUTFoldLoop:
    for (int n = 0; n < CONFIG_T::n_in / data_T::size * rf; n++) {
        #pragma HLS PIPELINE II=1
        if (c == 0)
            in_data = data.read();
        for (int l = 0; l < lanes; l++) {
            #pragma HLS UNROLL
            int j = c * lanes + l;
            if (j < data_T::size) {
                unsigned index = get_index_unary_lut<CONFIG_T::table_size>(in_data[j]);
                out_data[j] = table_bram[l / 2][index];
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
void unary_lut(hls::stream<data_T> &data, hls::stream<res_T> &res, typename CONFIG_T::table_t table[CONFIG_T::table_size]) {
    #pragma HLS INLINE
    if (CONFIG_T::strategy == nnet::resource) {
        unary_lut_resource<data_T, res_T, CONFIG_T>(data, res, table);
    } else {
        unary_lut_latency<data_T, res_T, CONFIG_T>(data, res, table);
    }
}

// *************************************************
//       Hard sigmoid Activation
// *************************************************

template <class data_T, class res_T, typename CONFIG_T>
void hard_sigmoid(hls::stream<data_T> &data, hls::stream<res_T> &res) {

HardSigmoidActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    HardSigmoidPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            auto datareg = CONFIG_T::slope * in_data[j] + CONFIG_T::shift;
            if (datareg > 1)
                datareg = 1;
            else if (datareg < 0)
                datareg = 0;
            out_data[j] = datareg;
        }

        res.write(out_data);
    }
}

template <class data_T, class res_T, typename CONFIG_T> void hard_tanh(hls::stream<data_T> &data, hls::stream<res_T> &res) {

HardSigmoidActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    HardSigmoidPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
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
void leaky_relu(hls::stream<data_T> &data, param_T alpha, hls::stream<res_T> &res) {
LeakyReLUActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    LeakyReLUPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
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
void thresholded_relu(hls::stream<data_T> &data, param_T theta, hls::stream<res_T> &res) {
ThresholdedReLUActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    ThresholdedReLUPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
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

template <class data_T, class res_T, typename CONFIG_T> void softplus(hls::stream<data_T> &data, hls::stream<res_T> &res) {
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

SoftplusActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    SoftplusPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            int data_round = in_data[j] * CONFIG_T::table_size / 16;
            int index = data_round + 8 * CONFIG_T::table_size / 16;
            if (index < 0)
                index = 0;
            else if (index > CONFIG_T::table_size - 1)
                index = CONFIG_T::table_size - 1;
            out_data[j] = softplus_table[index];
        }
        res.write(out_data);
    }
}

// *************************************************
//       Softsign Activation
// *************************************************

template <class data_T, class res_T, typename CONFIG_T> void softsign(hls::stream<data_T> &data, hls::stream<res_T> &res) {
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

SoftsignActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    SoftsignPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            int data_round = in_data[j] * CONFIG_T::table_size / 16;
            int index = data_round + 8 * CONFIG_T::table_size / 16;
            if (index < 0)
                index = 0;
            else if (index > CONFIG_T::table_size - 1)
                index = CONFIG_T::table_size - 1;
            out_data[j] = softsign_table[index];
        }
        res.write(out_data);
    }
}

// *************************************************
//       ELU Activation
// *************************************************
template <class data_T, class param_T, class res_T, typename CONFIG_T>
void elu(hls::stream<data_T> &data, param_T alpha, hls::stream<res_T> &res) {
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

EluActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    EluPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL

            typename data_T::value_type datareg = in_data[j];
            if (datareg >= 0) {
                out_data[j] = datareg;
            } else {
                int index = datareg * CONFIG_T::table_size / -8;
                if (index > CONFIG_T::table_size - 1)
                    index = CONFIG_T::table_size - 1;
                out_data[j] = alpha * elu_table[index];
            }
        }
        res.write(out_data);
    }
}

template <class data_T, class res_T, typename CONFIG_T> void elu(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    elu<data_T, ap_uint<1>, res_T, CONFIG_T>(data, 1.0, res);
}

// *************************************************
//       SELU Activation
// *************************************************

template <class data_T, class res_T, typename CONFIG_T> void selu(hls::stream<data_T> &data, hls::stream<res_T> &res) {
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

SeluActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    SeluPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL

            typename data_T::value_type datareg = in_data[j];
            if (datareg >= 0) {
                out_data[j] = (typename data_T::value_type)1.0507009873554804934193349852946 * datareg;
            } else {
                int index = datareg * CONFIG_T::table_size / -8;
                if (index > CONFIG_T::table_size - 1)
                    index = CONFIG_T::table_size - 1;
                out_data[j] = selu_table[index];
            }
        }
        res.write(out_data);
    }
}

// *************************************************
//       PReLU Activation
// *************************************************

template <class data_T, class param_T, class res_T, typename CONFIG_T>
void prelu(hls::stream<data_T> &data, const param_T alpha[CONFIG_T::n_in], hls::stream<res_T> &res) {
PReLUActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    PReLUPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
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
template <class data_T, class res_T, typename CONFIG_T>
void binary_tanh(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    using cache_T = ap_int<2>;
PReLUActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        cache_T cache;
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    PReLUPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
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
template <class data_T, class res_T, typename CONFIG_T>
void ternary_tanh(hls::stream<data_T> &data, hls::stream<res_T> &res) {
PReLUActLoop:
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        PRAGMA_DATA_PACK(out_data)

    PReLUPackLoop:
        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
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
