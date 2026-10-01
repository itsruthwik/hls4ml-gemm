#ifndef NNET_PADDING_STREAM_H_
#define NNET_PADDING_STREAM_H_

#include <math.h>

namespace nnet {

template <class res_T, typename CONFIG_T> void fill_zero(hls::stream<res_T> &res) {
    #pragma HLS INLINE
    res_T res_part;
    for (int c = 0; c < CONFIG_T::n_chan; c++) {
        #pragma HLS UNROLL
        res_part[c] = 0;
    }
    res.write(res_part);
}

template <class data_T, class res_T, typename CONFIG_T> void fill_data(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    #pragma HLS INLINE
    data_T data_part = data.read();
    res_T res_part;
    for (int c = 0; c < CONFIG_T::n_chan; c++) {
        #pragma HLS UNROLL
        res_part[c] = data_part[c];
    }
    res.write(res_part);
}

// One flat loop over the output grid, one beat per cycle: a border position writes zeros, an
// interior one copies the next input beat. The nested per-region loops this replaces had no
// pipeline pragma, so Vitis auto-pipelined the row loop and unrolled a whole padded row of
// stream reads and writes into one body (thousands of instructions per instance), which made
// its front-end optimisation the long pole of every conv design's csynth.
template <class data_T, class res_T, typename CONFIG_T>
void zeropad1d_cl(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    #pragma HLS INLINE off
PadWidth:
    for (unsigned j = 0; j < CONFIG_T::out_width; j++) {
        #pragma HLS PIPELINE II=1
        const bool inside = j >= CONFIG_T::pad_left && j < CONFIG_T::pad_left + CONFIG_T::in_width;
        data_T data_part;
        if (inside) {
            data_part = data.read();
        }
        res_T res_part;
    PadChannels:
        for (unsigned c = 0; c < CONFIG_T::n_chan; c++) {
            #pragma HLS UNROLL
            res_part[c] = inside ? typename res_T::value_type(data_part[c]) : typename res_T::value_type(0);
        }
        res.write(res_part);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void zeropad2d_cl(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    #pragma HLS INLINE off
PadHeight:
    for (unsigned i = 0; i < CONFIG_T::out_height; i++) {
        const bool row_inside = i >= CONFIG_T::pad_top && i < CONFIG_T::pad_top + CONFIG_T::in_height;
    PadWidth:
        for (unsigned j = 0; j < CONFIG_T::out_width; j++) {
            #pragma HLS PIPELINE II=1
            const bool inside = row_inside && j >= CONFIG_T::pad_left && j < CONFIG_T::pad_left + CONFIG_T::in_width;
            data_T data_part;
            if (inside) {
                data_part = data.read();
            }
            res_T res_part;
        PadChannels:
            for (unsigned c = 0; c < CONFIG_T::n_chan; c++) {
                #pragma HLS UNROLL
                res_part[c] = inside ? typename res_T::value_type(data_part[c]) : typename res_T::value_type(0);
            }
            res.write(res_part);
        }
    }
}

} // namespace nnet

#endif
