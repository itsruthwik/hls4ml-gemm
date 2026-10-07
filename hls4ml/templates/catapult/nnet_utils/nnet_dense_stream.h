#ifndef NNET_DENSE_STREAM_H_
#define NNET_DENSE_STREAM_H_

#include "ac_channel.h"
#include "nnet_common.h"
#include "nnet_types.h"
#include <assert.h>
#include <math.h>

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void dense_wrapper(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                   typename weight_store<CONFIG_T>::type weights[weight_store<CONFIG_T>::size],
                   typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
    if constexpr (CONFIG_T::strategy == nnet::latency) { // dense_latency takes flat weights only
        constexpr int ce_reuse_factor = CONFIG_T::reuse_factor;
        (void)ce_reuse_factor;
        #pragma hls_pipeline_init_interval ce_reuse_factor
        dense_latency<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else {
        dense_resource<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void dense(ac_channel<data_T> &data_stream, ac_channel<res_T> &res_stream,
           typename weight_store<CONFIG_T>::type weights[weight_store<CONFIG_T>::size],
           typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
    // data is written by the unrolled DataPack loop and read fully in parallel by dense_wrapper's
    // unrolled inner loops; mirrors the Vivado ARRAY_PARTITION variable=data complete.
    #pragma hls_resource data:rsc variables="data" map_to_module="[Register]"
    typename data_T::value_type data[CONFIG_T::n_in];

    // res is written fully in parallel by dense_wrapper's unrolled inner loops and read by the
    // unrolled ResPack loop; mirrors the Vivado ARRAY_PARTITION variable=res complete.
    #pragma hls_resource res:rsc variables="res" map_to_module="[Register]"
    typename res_T::value_type res[CONFIG_T::n_out];

#pragma hls_pipeline_init_interval 1
DataPrepare:
    for (unsigned int i_in = 0; i_in < CONFIG_T::n_in / data_T::size; i_in++) {
        data_T data_pack = data_stream.read();
    #pragma hls_unroll
    DataPack:
        for (unsigned int i_pack = 0; i_pack < data_T::size; i_pack++) {
            data[i_in * data_T::size + i_pack] = data_pack[i_pack];
        }
    }

    dense_wrapper<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(data, res, weights, biases);

#pragma hls_pipeline_init_interval 1
ResWrite:
    for (unsigned i_out = 0; i_out < CONFIG_T::n_out / res_T::size; i_out++) {
        res_T res_pack;
    #pragma hls_unroll
    ResPack:
        for (unsigned int i_pack = 0; i_pack < res_T::size; i_pack++) {
            res_pack[i_pack] = res[i_out * res_T::size + i_pack];
        }
        res_stream.write(res_pack);
    }
}

// Stream Dense with Latency strategy: one inference per call, written so that the stage block around it
// (pipelined at II 1 by the Catapult writer) overlaps inferences. The input read and the output write are
// unrolled rather than separately pipelined, and the multiply-add is the unpipelined dense_latency_core, so
// the whole read -> multiply-add -> write path is one pipeline stage sequence in the block's main loop
// (separately pipelined sections each cost a step of the main loop's interval). A multi-beat input or output
// still works; its extra channel accesses lengthen the interval.
template <class data_T, class res_T, typename CONFIG_T>
void dense_overlap(ac_channel<data_T> &data_stream, ac_channel<res_T> &res_stream,
                   typename weight_store<CONFIG_T>::type weights[weight_store<CONFIG_T>::size],
                   typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
    #pragma hls_resource data:rsc variables="data" map_to_module="[Register]"
    typename data_T::value_type data[CONFIG_T::n_in];
    #pragma hls_resource res:rsc variables="res" map_to_module="[Register]"
    typename res_T::value_type res[CONFIG_T::n_out];

#pragma hls_unroll yes
DataPrepare:
    for (unsigned int i_in = 0; i_in < CONFIG_T::n_in / data_T::size; i_in++) {
        data_T data_pack = data_stream.read();
    #pragma hls_unroll
    DataPack:
        for (unsigned int i_pack = 0; i_pack < data_T::size; i_pack++) {
            data[i_in * data_T::size + i_pack] = data_pack[i_pack];
        }
    }

    dense_latency_core<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(data, res, weights, biases);

#pragma hls_unroll yes
ResWrite:
    for (unsigned i_out = 0; i_out < CONFIG_T::n_out / res_T::size; i_out++) {
        res_T res_pack;
    #pragma hls_unroll
    ResPack:
        for (unsigned int i_pack = 0; i_pack < res_T::size; i_pack++) {
            res_pack[i_pack] = res[i_out * res_T::size + i_pack];
        }
        res_stream.write(res_pack);
    }
}

} // namespace nnet

#endif
