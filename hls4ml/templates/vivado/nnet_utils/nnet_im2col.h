#ifndef NNET_IM2COL_H_
#define NNET_IM2COL_H_

// Vivado/Vitis io_parallel im2col row generators for the fused Conv GEMM-IP path.
// The io_stream counterparts live in nnet_im2col_stream.h. These take a flat
// channels_last input array (random access) and materialize a_rows[gemm_m], so
// no hls::stream / shift-register machinery is needed. Assumes the conv-GEMM
// guard's constraints: stride 1, valid (zero) padding, dilation 1.

#include "nnet_common.h"

namespace nnet {

// ---------------------------------------------------------------------------
// im2col_1d_gemm_rows_array — io_parallel counterpart of im2col_1d_gemm_rows.
// Row element order is kernel-col-major, channel-minor: a_row[kw * n_chan + c],
// to match the GEMM weight transposition ([W,C,F] -> [F, W*C]); the streaming
// path (im2col_1d_gemm_rows) packs the identical order.
// ---------------------------------------------------------------------------
template <class data_T, class a_row_T, typename CONFIG_T>
void im2col_1d_gemm_rows_array(const data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
                               a_row_T a_rows[CONFIG_T::out_width]) {
    static_assert(a_row_T::size == CONFIG_T::filt_width * CONFIG_T::n_chan,
                  "A row width must equal filt_width * n_chan");
GemmRowsWidth:
    for (unsigned ow = 0; ow < CONFIG_T::out_width; ow++) {
        a_row_T out_pack;
    GemmRowsFiltWidth:
        for (unsigned kw = 0; kw < CONFIG_T::filt_width; kw++) {
            #pragma HLS UNROLL
        GemmRowsChan:
            for (unsigned c = 0; c < CONFIG_T::n_chan; c++) {
                #pragma HLS UNROLL
                out_pack[kw * CONFIG_T::n_chan + c] = data[(ow + kw) * CONFIG_T::n_chan + c];
            }
        }
        a_rows[ow] = out_pack;
    }
}

// ---------------------------------------------------------------------------
// im2col_2d_gemm_rows_array — io_parallel counterpart of im2col_2d_gemm_rows.
// Row element order is spatial-row-major then channel:
//   a_row[(kh * filt_width + kw) * n_chan + c]
// to match the Conv2D GEMM weight transposition ([H,W,C,F] -> [F, H*W*C]).
// Output rows are raster-ordered (oh * out_width + ow) = gemm_m order.
// ---------------------------------------------------------------------------
template <class data_T, class a_row_T, typename CONFIG_T>
void im2col_2d_gemm_rows_array(const data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
                               a_row_T a_rows[CONFIG_T::out_height * CONFIG_T::out_width]) {
    static_assert(a_row_T::size == CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan,
                  "A row width must equal filt_height * filt_width * n_chan");
GemmRowsHeight:
    for (unsigned oh = 0; oh < CONFIG_T::out_height; oh++) {
    GemmRowsWidth:
        for (unsigned ow = 0; ow < CONFIG_T::out_width; ow++) {
            a_row_T out_pack;
        GemmRowsFiltHeight:
            for (unsigned kh = 0; kh < CONFIG_T::filt_height; kh++) {
                #pragma HLS UNROLL
            GemmRowsFiltWidth:
                for (unsigned kw = 0; kw < CONFIG_T::filt_width; kw++) {
                    #pragma HLS UNROLL
                GemmRowsChan:
                    for (unsigned c = 0; c < CONFIG_T::n_chan; c++) {
                        #pragma HLS UNROLL
                        out_pack[(kh * CONFIG_T::filt_width + kw) * CONFIG_T::n_chan + c] =
                            data[((oh + kh) * CONFIG_T::in_width + (ow + kw)) * CONFIG_T::n_chan + c];
                    }
                }
            }
            a_rows[oh * CONFIG_T::out_width + ow] = out_pack;
        }
    }
}

} // namespace nnet

#endif // NNET_IM2COL_H_
