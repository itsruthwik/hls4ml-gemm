#ifndef NNET_IM2COL_STREAM_H_
#define NNET_IM2COL_STREAM_H_

// Vivado/Vitis im2col stream utilities.
// Provides standard im2col_2d_cl, im2col_1d/2d_stream for non-GEMM paths,
// and im2col_1d/2d_gemm_rows for row/column GEMM-IP streaming.
// All stream arguments use hls::stream instead of ac_channel.

#include "ap_int.h"
#include "ap_shift_reg.h"
#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_conv_stream.h"
#include "nnet_stream_beat.h"

namespace nnet {

// Minimal signed width that holds [-(N-1) .. N-1] plus a sign bit, used to
// right-size the im2col_gemm_rows position/shift counters below instead of
// native 32-bit int.
constexpr int im2col_idx_bits_for(int n, int b = 2, int cap = 4) {
    return (cap > n) ? b : im2col_idx_bits_for(n, b + 1, cap << 1);
}
template <int N> struct im2col_idx_bits {
    static constexpr int value = im2col_idx_bits_for(N) + 1; // + sign bit headroom
};

// ---------------------------------------------------------------------------
// im2col_config — base configuration struct used by im2cl and gemm_rows templates.
// ---------------------------------------------------------------------------
struct im2col_config {
    static const unsigned in_height   = 10;
    static const unsigned in_width    = 10;
    static const unsigned n_chan      = 1;
    static const unsigned filt_height = 3;
    static const unsigned filt_width  = 3;
    static const unsigned stride_height = 1;
    static const unsigned stride_width  = 1;
    static const unsigned out_height  = 8;
    static const unsigned out_width   = 8;
    static const unsigned pad_top     = 0;
    static const unsigned pad_bottom  = 0;
    static const unsigned pad_left    = 0;
    static const unsigned pad_right   = 0;
    static const unsigned gemm_m      = 1;
    static const unsigned reuse_factor = 1;
};

// ---------------------------------------------------------------------------
// im2col_2d_cl — existing 2-D im2col (channels-last, pixel-at-a-time).
// ---------------------------------------------------------------------------
template <class data_T, class res_T, typename CONFIG_T>
void im2col_2d_cl(
    hls::stream<data_T> &data,
    hls::stream<res_T>  &res) {

    static ap_shift_reg<typename data_T::value_type, CONFIG_T::in_width> line_buffer[MAX(CONFIG_T::filt_height - 1, 1)]
                                                                                    [CONFIG_T::n_chan];
    #pragma HLS ARRAY_PARTITION variable = line_buffer complete dim = 2

    const static int lShiftX = CONFIG_T::filt_width  - 1;
    const static int lShiftY = CONFIG_T::filt_height - 1;

    static int pX = 0, pY = 0, sX = 0, sY = 0;

    static typename data_T::value_type kernel_data[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan];
    #pragma HLS ARRAY_PARTITION variable = kernel_data complete

ReadInputHeight:
    for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
    ReadInputWidth:
        for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width; i_iw++) {
            #pragma HLS LOOP_FLATTEN
            #pragma HLS PIPELINE II=1

            nnet::shift_line_buffer<data_T, CONFIG_T>(data.read(), line_buffer, kernel_data);

            if ((sX - lShiftX) == 0 && (sY - lShiftY) == 0 && pY > lShiftY - 1 && pX > lShiftX - 1) {
                res_T res_pack;
                PRAGMA_DATA_PACK(res_pack)
            PackLoop:
                for (unsigned i_ic = 0; i_ic < CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan; i_ic++) {
                    #pragma HLS UNROLL
                    res_pack[i_ic] = kernel_data[i_ic];
                }
                res.write(res_pack);
            }

            if (pX + 1 == (int)CONFIG_T::in_width) {
                pX = 0; sX = 0; pY++; sY++;
                if (sY == (int)CONFIG_T::stride_height) sY = 0;
            } else {
                pX++; sX++;
                if (sX == (int)CONFIG_T::stride_width) sX = 0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// im2col_1d_stream — standard 1-D row stream (non-GEMM path).
// ---------------------------------------------------------------------------
template <class data_T, class im2col_row_t, typename CONFIG_T>
void im2col_1d_stream(hls::stream<data_T> &data, hls::stream<im2col_row_t> &a_stream) {
    #pragma HLS INLINE off
    typedef typename data_T::value_type data_element_t;
    int pX = 0, sX = 0;
    data_element_t kernel_data[CONFIG_T::filt_width * CONFIG_T::n_chan] = {};
    const static int lShiftX = CONFIG_T::filt_width - 1;

ReadInputWidth:
    for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width / (data_T::size / CONFIG_T::n_chan); i_iw++) {
        data_T data_pack = data.read();
        for (unsigned p = 0; p < data_T::size / CONFIG_T::n_chan; p++) {
            nnet::array<data_element_t, CONFIG_T::n_chan> pixel_pack;
        PackChannels:
            for (unsigned c = 0; c < CONFIG_T::n_chan; c++) {
                #pragma HLS UNROLL
                pixel_pack[c] = data_pack[p * CONFIG_T::n_chan + c];
            }
            nnet::kernel_shift_1d<decltype(pixel_pack), CONFIG_T>(pixel_pack, kernel_data);

            if ((sX - lShiftX) == 0 && pX > lShiftX - 1) {
                im2col_row_t out_pack;
            PackLoop:
                for (unsigned i = 0; i < CONFIG_T::filt_width * CONFIG_T::n_chan; i++) {
                    #pragma HLS UNROLL
                    out_pack[i] = kernel_data[i];
                }
                a_stream.write(out_pack);
            }

            if (pX + 1 == (int)CONFIG_T::in_width) {
                pX = 0; sX = 0;
            } else {
                pX++;
                sX = ((sX - lShiftX) == 0) ? sX - (int)CONFIG_T::stride_width + 1 : sX + 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// im2col_2d_stream — standard 2-D row stream (non-GEMM path).
// ---------------------------------------------------------------------------
template <class data_T, class im2col_row_t, typename CONFIG_T>
void im2col_2d_stream(hls::stream<data_T> &data, hls::stream<im2col_row_t> &a_stream) {
    #pragma HLS INLINE off
    typedef typename data_T::value_type data_element_t;
    static ap_shift_reg<data_element_t, CONFIG_T::in_width> line_buffer[MAX(CONFIG_T::filt_height - 1, 1)]
                                                                        [CONFIG_T::n_chan];
    int pX = 0, pY = 0, sX = 0, sY = 0;
    data_element_t kernel_data[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan] = {};
    const static int lShiftX = CONFIG_T::filt_width  - 1;
    const static int lShiftY = CONFIG_T::filt_height - 1;

ReadInputHeight:
    for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
    ReadInputWidth:
        for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width / (data_T::size / CONFIG_T::n_chan); i_iw++) {
            data_T data_pack = data.read();
            for (unsigned p = 0; p < data_T::size / CONFIG_T::n_chan; p++) {
                nnet::array<data_element_t, CONFIG_T::n_chan> pixel_pack;
            PackChannels:
                for (unsigned c = 0; c < CONFIG_T::n_chan; c++) {
                    #pragma HLS UNROLL
                    pixel_pack[c] = data_pack[p * CONFIG_T::n_chan + c];
                }
                nnet::shift_line_buffer<decltype(pixel_pack), CONFIG_T>(pixel_pack, line_buffer, kernel_data);

                if ((sX - lShiftX) == 0 && (sY - lShiftY) == 0 && pY > lShiftY - 1 && pX > lShiftX - 1) {
                    im2col_row_t out_pack;
                PackLoop:
                    for (unsigned i = 0; i < CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan; i++) {
                        #pragma HLS UNROLL
                        out_pack[i] = kernel_data[i];
                    }
                    a_stream.write(out_pack);
                }

                if (pX + 1 == (int)CONFIG_T::in_width) {
                    pX = 0; sX = 0;
                    if (pY + 1 == (int)CONFIG_T::in_height) {
                        pY = 0; sY = 0;
                    } else {
                        pY++;
                        sY = ((sY - lShiftY) == 0) ? sY - (int)CONFIG_T::stride_height + 1 : sY + 1;
                    }
                } else {
                    pX++;
                    sX = ((sX - lShiftX) == 0) ? sX - (int)CONFIG_T::stride_width + 1 : sX + 1;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// im2col_1d_gemm_rows — 1-D im2col that emits one full K-wide row per valid
// output pixel.  Designed for row/column streaming GEMM IPs.
// Output stream: one a_row_T per output spatial position, where
//   a_row_T = nnet::array<input_scalar_t, filt_width * n_chan>
// Flattening order: k = kw * n_chan + c
// ---------------------------------------------------------------------------
template <class data_T, class a_row_io_T, typename CONFIG_T>
void im2col_1d_gemm_rows(hls::stream<data_T> &data, hls::stream<typename beat_io<a_row_io_T>::elem_t> &a_rows) {
    // a_row_io_T is the row type, or nnet::packed<row type> when the row feeds a GEMM IP packed.
    typedef typename beat_io<a_row_io_T>::array_t a_row_T;
    #pragma HLS INLINE off

    static_assert(a_row_T::size == CONFIG_T::filt_width * CONFIG_T::n_chan,
                  "A row width must equal filt_width * n_chan");

    typedef typename data_T::value_type data_element_t;
    // pX/sX only ever hold values in [-stride, in_width], but a native 32-bit
    // int keeps all 32 bits live through the "== 0" compares and the
    // conditional +1/reset updates below, so the position and shift counters
    // synthesize as wide adders/compares even though their upper bits are
    // always zero. Narrowing to just enough bits removes that dead width.
    constexpr int idxBits = im2col_idx_bits<CONFIG_T::in_width>::value;
    ap_int<idxBits> pX = 0, sX = 0;
    data_element_t kernel_data[CONFIG_T::filt_width * CONFIG_T::n_chan] = {};
    #pragma HLS ARRAY_PARTITION variable=kernel_data complete dim=1
    // Keep the shift-window bound and stride/width constants the same narrow
    // width as pX/sX, so mixing them in the compares/updates below does not
    // silently re-widen the arithmetic back to 32-bit int.
    const static ap_int<idxBits> lShiftX  = CONFIG_T::filt_width - 1;
    const static ap_int<idxBits> strideWidth = CONFIG_T::stride_width;
    const static ap_int<idxBits> inWidthIdx  = CONFIG_T::in_width;
    const static ap_int<idxBits> one = 1;
    const static ap_int<idxBits> lShiftXm1 = lShiftX - one;

ReadInputWidth:
    for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width / (data_T::size / CONFIG_T::n_chan); i_iw++) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        data_T data_pack = data.read();
        for (unsigned p = 0; p < data_T::size / CONFIG_T::n_chan; p++) {
            nnet::array<data_element_t, CONFIG_T::n_chan> pixel_pack;
        PackChannels:
            for (unsigned c = 0; c < CONFIG_T::n_chan; c++) {
                #pragma HLS UNROLL
                pixel_pack[c] = data_pack[p * CONFIG_T::n_chan + c];
            }
            nnet::kernel_shift_1d<decltype(pixel_pack), CONFIG_T>(pixel_pack, kernel_data);

            if (sX == lShiftX && pX > lShiftXm1) {
                a_row_T out_pack;
            PackLoop:
                for (unsigned i = 0; i < CONFIG_T::filt_width * CONFIG_T::n_chan; i++) {
                    #pragma HLS UNROLL
                    out_pack[i] = kernel_data[i];
                }
                beat_io<a_row_io_T>::write(a_rows, out_pack);
            }

            if (pX + one == inWidthIdx) {
                pX = 0; sX = 0;
            } else {
                pX++;
                sX = (sX == lShiftX) ? ap_int<idxBits>(sX - strideWidth + one) : ap_int<idxBits>(sX + one);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// im2col_2d_gemm_rows — 2-D im2col that emits one full K-wide row per valid
// output pixel.  Designed for row/column streaming GEMM IPs.
// Output stream: one a_row_T per output spatial position, where
//   a_row_T = nnet::array<input_scalar_t, filt_height * filt_width * n_chan>
// Flattening order: k = ((kh * filt_width) + kw) * n_chan + c
// ---------------------------------------------------------------------------
template <class data_T, class a_row_io_T, typename CONFIG_T>
void im2col_2d_gemm_rows(hls::stream<data_T> &data, hls::stream<typename beat_io<a_row_io_T>::elem_t> &a_rows) {
    // a_row_io_T is the row type, or nnet::packed<row type> when the row feeds a GEMM IP packed.
    typedef typename beat_io<a_row_io_T>::array_t a_row_T;
    #pragma HLS INLINE off

    static_assert(a_row_T::size == CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan,
                  "A row width must equal filt_height * filt_width * n_chan");

    typedef typename data_T::value_type data_element_t;
    static ap_shift_reg<data_element_t, CONFIG_T::in_width> line_buffer[MAX(CONFIG_T::filt_height - 1, 1)]
                                                                            [CONFIG_T::n_chan];
    #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=2
    // pX/pY/sX/sY only ever hold values in [-stride, MAX(in_height, in_width)],
    // but a native 32-bit int keeps all 32 bits live through the "== 0"
    // compares and the conditional +1/reset updates below, so the
    // row/column position and shift counters synthesize as wide
    // adders/compares even though their upper bits are always zero.
    // Narrowing to just enough bits removes that dead width from the
    // recurrence.
    constexpr int idxBits = im2col_idx_bits<MAX(CONFIG_T::in_height, CONFIG_T::in_width)>::value;
    ap_int<idxBits> pX = 0, pY = 0, sX = 0, sY = 0;
    data_element_t kernel_data[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan] = {};
    #pragma HLS ARRAY_PARTITION variable=kernel_data complete dim=1
    // Keep the shift-window bounds and stride/dimension constants the same
    // narrow width as pX/pY/sX/sY, so mixing them in the compares/updates
    // below does not silently re-widen the arithmetic back to 32-bit int.
    const static ap_int<idxBits> lShiftX = CONFIG_T::filt_width  - 1;
    const static ap_int<idxBits> lShiftY = CONFIG_T::filt_height - 1;
    const static ap_int<idxBits> strideWidth  = CONFIG_T::stride_width;
    const static ap_int<idxBits> strideHeight = CONFIG_T::stride_height;
    const static ap_int<idxBits> inWidthIdx   = CONFIG_T::in_width;
    const static ap_int<idxBits> inHeightIdx  = CONFIG_T::in_height;
    const static ap_int<idxBits> one = 1;
    const static ap_int<idxBits> lShiftXm1 = lShiftX - one;
    const static ap_int<idxBits> lShiftYm1 = lShiftY - one;

ReadInputHeight:
    for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
    ReadInputWidth:
        for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width / (data_T::size / CONFIG_T::n_chan); i_iw++) {
            #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
            data_T data_pack = data.read();
            for (unsigned p = 0; p < data_T::size / CONFIG_T::n_chan; p++) {
                nnet::array<data_element_t, CONFIG_T::n_chan> pixel_pack;
            PackChannels:
                for (unsigned c = 0; c < CONFIG_T::n_chan; c++) {
                    #pragma HLS UNROLL
                    pixel_pack[c] = data_pack[p * CONFIG_T::n_chan + c];
                }
                nnet::shift_line_buffer<decltype(pixel_pack), CONFIG_T>(pixel_pack, line_buffer, kernel_data);

                if (sX == lShiftX && sY == lShiftY && pY > lShiftYm1 && pX > lShiftXm1) {
                    a_row_T out_pack;
                PackLoop:
                    for (unsigned i = 0; i < CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan; i++) {
                        #pragma HLS UNROLL
                        out_pack[i] = kernel_data[i];
                    }
                    beat_io<a_row_io_T>::write(a_rows, out_pack);
                }

                if (pX + one == inWidthIdx) {
                    pX = 0; sX = 0;
                    if (pY + one == inHeightIdx) {
                        pY = 0; sY = 0;
                    } else {
                        pY++;
                        sY = (sY == lShiftY) ? ap_int<idxBits>(sY - strideHeight + one) : ap_int<idxBits>(sY + one);
                    }
                } else {
                    pX++;
                    sX = (sX == lShiftX) ? ap_int<idxBits>(sX - strideWidth + one) : ap_int<idxBits>(sX + one);
                }
            }
        }
    }
}

} // namespace nnet

#endif // NNET_IM2COL_STREAM_H_
