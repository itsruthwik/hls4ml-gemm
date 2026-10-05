#ifndef NNET_IM2COL_H_
#define NNET_IM2COL_H_

#include "ac_channel.h"
#include "ap_shift_reg.h"
#include "nnet_common.h"
#include "nnet_conv_stream.h"

namespace nnet {

struct im2col_config {
    static const unsigned in_height = 10;
    static const unsigned in_width = 10;
    static const unsigned n_chan = 1;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 8;
    static const unsigned out_width = 8;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned gemm_m = 1;
    static const unsigned tile_rows = 1;
    static const unsigned reuse_factor = 1;
};

template <class a_row_T, class data_T, unsigned N, typename CONFIG_T>
void write_im2col_row(const data_T (&kernel_data)[N], ac_channel<a_row_T> &a_rows) {
    static_assert(a_row_T::size == N, "im2col row width must match kernel window width");
    a_row_T out_pack;
PackIm2ColRow:
    #pragma hls_unroll
    for (unsigned i = 0; i < N; i++) {
        out_pack[i] = kernel_data[i];
    }
    a_rows.write(out_pack);
}

// Pixel and stride counters sized to their range. With 32-bit `int` counters the emit test and the
// counter updates are 32-bit subtracts and compares, which Catapult splits over several pipeline
// stages at a 2 ns clock; every later stage then holds its own copy of the kernel window that
// write_im2col_row reads. sX / sY take values in [filt - stride, filt - 1] (signed when stride > filt).
template <unsigned EXTENT> struct im2col_pos_t { typedef ac_int<ac::nbits<(EXTENT > 1 ? EXTENT - 1 : 1)>::val, false> type; };
template <unsigned FILT, unsigned STRIDE> struct im2col_stride_t {
    typedef ac_int<ac::nbits<FILT + STRIDE>::val + 1, true> type;
};

// ---------------------------------------------------------------------------
// im2col_1d_gemm_rows — 1-D im2col that emits one full K-wide row per valid
// output pixel.  Designed for row/column streaming GEMM IPs.
// Output: one a_row_T per output spatial position, where
//   a_row_T = nnet::array<input_scalar_t, filt_width * n_chan>
// ---------------------------------------------------------------------------
#pragma hls_design block
template <class data_T, class a_row_T, typename CONFIG_T>
void im2col_1d_gemm_rows(ac_channel<data_T> &data, ac_channel<a_row_T> &a_rows) {
    static_assert(a_row_T::size == CONFIG_T::filt_width * CONFIG_T::n_chan,
                  "A row width must equal filt_width * n_chan");

    typedef typename data_T::value_type data_element_t;
    typedef typename im2col_pos_t<CONFIG_T::in_width>::type px_t;
    typedef typename im2col_stride_t<CONFIG_T::filt_width, CONFIG_T::stride_width>::type sx_t;
    typedef typename im2col_pos_t<CONFIG_T::tile_rows>::type tile_t;
    px_t pX = 0;
    sx_t sX = 0;
    tile_t tile_row = 0;
    data_element_t kernel_data[CONFIG_T::filt_width * CONFIG_T::n_chan] = {};
    const static int lShiftX = CONFIG_T::filt_width - 1;

    static_assert(CONFIG_T::tile_rows >= 1 && CONFIG_T::tile_rows <= CONFIG_T::gemm_m,
                  "tile_rows must satisfy 1 <= tile_rows <= gemm_m");

    // Tile semantics: within a tile (tile_rows consecutive emitted a_rows) this pixel
    // loop is gapless (II=1, no stalls) — the downstream activation_rows channel is
    // sized to tile_rows deep so a full tile always drains without backpressure.
    // GEMM-IP backpressure is only expected/allowed to be visible at tile boundaries.
ReadInputWidth:
    // Pipeline the spatial driver loop (Catapult does not auto-flatten the nest the way Vivado does,
    // so pipelining only the inner ReadInputPack leaves this loop rolled). Unroll the inner pack loop.
    // II is a literal 1 (gapless): Catapult rejects a symbolic pragma argument such as
    // CONFIG_T::reuse_factor (CIN-93) and then leaves the loop rolled.
    #pragma hls_pipeline_init_interval 1
    for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width / (data_T::size / CONFIG_T::n_chan); i_iw++) {
        data_T data_pack = data.read();
    ReadInputPack:
        #pragma hls_unroll
        for (unsigned p = 0; p < data_T::size / CONFIG_T::n_chan; p++) {
            nnet::array<data_element_t, CONFIG_T::n_chan> pixel_pack;
        PackChannels:
            #pragma hls_unroll
            for (unsigned c = 0; c < CONFIG_T::n_chan; c++) {
                pixel_pack[c] = data_pack[p * CONFIG_T::n_chan + c];
            }
            kernel_shift_1d<decltype(pixel_pack), CONFIG_T>(pixel_pack, kernel_data);

            // Comparisons against constants, no subtracts (same conditions as before).
            const bool sx_hit = (sX == lShiftX);
            if (sx_hit && pX >= lShiftX) {
                write_im2col_row<a_row_T, data_element_t, CONFIG_T::filt_width * CONFIG_T::n_chan, CONFIG_T>(
                    kernel_data, a_rows);
                tile_row = (tile_row == CONFIG_T::tile_rows - 1) ? tile_t(0) : tile_t(tile_row + 1);
            }

            if (pX == CONFIG_T::in_width - 1) {
                pX = 0;
                sX = 0;
            } else {
                pX = pX + 1;
                sX = sx_hit ? sx_t(lShiftX - (int)CONFIG_T::stride_width + 1) : sx_t(sX + 1);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// im2col_2d_gemm_rows — 2-D im2col that emits one full K-wide row per valid
// output pixel.  Designed for row/column streaming GEMM IPs.
// Output: one a_row_T per output spatial position, where
//   a_row_T = nnet::array<input_scalar_t, filt_height * filt_width * n_chan>
// ---------------------------------------------------------------------------
// Register line-buffer shift: per-row delay lines held in register arrays. The
// window layout written into kernel_window matches kernel_shift_2d.
template <class data_T, typename CONFIG_T>
void gemm_shift_line_buffer_reg(
    const data_T &in_elem,
    typename data_T::value_type line_buffer[MAX(CONFIG_T::filt_height - 1, 1)][CONFIG_T::n_chan]
                                           [CONFIG_T::in_width],
    typename data_T::value_type kernel_window[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan]) {
    typedef typename data_T::value_type T;
    T shift_buffer[CONFIG_T::filt_height][CONFIG_T::n_chan];

UpdateBufferReg:
    #pragma hls_unroll
    for (unsigned i_ic = 0; i_ic < CONFIG_T::n_chan; i_ic++) {
        shift_buffer[CONFIG_T::filt_height - 1][i_ic] = in_elem[i_ic];
    }

LineBufferRegChan:
    #pragma hls_unroll
    for (unsigned i_ic = 0; i_ic < CONFIG_T::n_chan; i_ic++) {
    LineBufferRegRow:
        #pragma hls_unroll
        for (unsigned i_ih = 1; i_ih < CONFIG_T::filt_height; i_ih++) {
            // Pop the oldest (in_width-ago) element of this delay line, shift the
            // register row toward the tail, push the incoming column element at 0.
            T pop_elem = line_buffer[i_ih - 1][i_ic][CONFIG_T::in_width - 1];
        LineBufferRegShift:
            #pragma hls_unroll
            for (int j = CONFIG_T::in_width - 1; j > 0; j--) {
                line_buffer[i_ih - 1][i_ic][j] = line_buffer[i_ih - 1][i_ic][j - 1];
            }
            line_buffer[i_ih - 1][i_ic][0] = shift_buffer[CONFIG_T::filt_height - i_ih][i_ic];
            shift_buffer[CONFIG_T::filt_height - i_ih - 1][i_ic] = pop_elem;
        }
    }
    kernel_shift_2d<data_T, CONFIG_T>(shift_buffer, kernel_window);
}

#pragma hls_design block
template <class data_T, class a_row_T, typename CONFIG_T>
void im2col_2d_gemm_rows(ac_channel<data_T> &data, ac_channel<a_row_T> &a_rows) {
    static_assert(a_row_T::size == CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan,
                  "A row width must equal filt_height * filt_width * n_chan");

    typedef typename data_T::value_type data_element_t;
    // GEMM-only register line buffer (replaces ap_shift_reg to break the RecII~=5
    // recurrence). Fully partitioned: MAX(filt_height-1,1) delay rows x n_chan x
    // in_width registers, no mux needed (each element has a single source, unlike an
    // ap_shift_reg-backed line buffer which needs a read-address mux per tap).
    // Not static: the pixel loop covers the whole frame and no row is emitted until
    // pY >= filt_height - 1 and pX >= filt_width - 1, by which point every tap it reads was
    // written in this frame, so nothing needs to persist across frames. As a static, Catapult
    // kept the persistent copy and a loop-carried copy of every element (twice the registers).
    data_element_t line_buffer[MAX(CONFIG_T::filt_height - 1, 1)][CONFIG_T::n_chan][CONFIG_T::in_width];
    typedef typename im2col_pos_t<CONFIG_T::in_width>::type px_t;
    typedef typename im2col_pos_t<CONFIG_T::in_height>::type py_t;
    typedef typename im2col_stride_t<CONFIG_T::filt_width, CONFIG_T::stride_width>::type sx_t;
    typedef typename im2col_stride_t<CONFIG_T::filt_height, CONFIG_T::stride_height>::type sy_t;
    typedef typename im2col_pos_t<CONFIG_T::tile_rows>::type tile_t;
    px_t pX = 0;
    py_t pY = 0;
    sx_t sX = 0;
    sy_t sY = 0;
    tile_t tile_row = 0;
    data_element_t kernel_data[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan] = {};
    const static int lShiftX = CONFIG_T::filt_width - 1;
    const static int lShiftY = CONFIG_T::filt_height - 1;

    static_assert(CONFIG_T::tile_rows >= 1 && CONFIG_T::tile_rows <= CONFIG_T::gemm_m,
                  "tile_rows must satisfy 1 <= tile_rows <= gemm_m");

    // Tile semantics: within a tile (tile_rows consecutive emitted a_rows) the pixel
    // loop below is gapless (II=1, no stalls) — the downstream activation_rows
    // channel is sized to tile_rows deep so a full tile always drains without
    // backpressure. GEMM-IP backpressure is only expected/allowed to be visible at
    // tile boundaries, between tiles.
ReadInputPixels:
    // Literal II, as in im2col_1d_gemm_rows: Catapult rejects a symbolic pragma argument.
    #pragma hls_pipeline_init_interval 1
    for (unsigned i_beat = 0; i_beat < CONFIG_T::in_height * CONFIG_T::in_width / (data_T::size / CONFIG_T::n_chan);
         i_beat++) {
        data_T data_pack = data.read();
    ReadInputPack:
        #pragma hls_unroll
        for (unsigned p = 0; p < data_T::size / CONFIG_T::n_chan; p++) {
            nnet::array<data_element_t, CONFIG_T::n_chan> pixel_pack;
        PackChannels:
            #pragma hls_unroll
            for (unsigned c = 0; c < CONFIG_T::n_chan; c++) {
                pixel_pack[c] = data_pack[p * CONFIG_T::n_chan + c];
            }
            gemm_shift_line_buffer_reg<decltype(pixel_pack), CONFIG_T>(pixel_pack, line_buffer, kernel_data);

            // Comparisons against constants, no subtracts (same conditions as before).
            const bool sx_hit = (sX == lShiftX);
            const bool sy_hit = (sY == lShiftY);
            if (sx_hit && sy_hit && pY >= lShiftY && pX >= lShiftX) {
                write_im2col_row<a_row_T, data_element_t,
                                  CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan, CONFIG_T>(
                    kernel_data, a_rows);
                tile_row = (tile_row == CONFIG_T::tile_rows - 1) ? tile_t(0) : tile_t(tile_row + 1);
            }

            if (pX == CONFIG_T::in_width - 1) {
                pX = 0;
                sX = 0;
                if (pY == CONFIG_T::in_height - 1) {
                    pY = 0;
                    sY = 0;
                } else {
                    pY = pY + 1;
                    sY = sy_hit ? sy_t(lShiftY - (int)CONFIG_T::stride_height + 1) : sy_t(sY + 1);
                }
            } else {
                pX = pX + 1;
                sX = sx_hit ? sx_t(lShiftX - (int)CONFIG_T::stride_width + 1) : sx_t(sX + 1);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// im2col_1d_gemm_rows_array — io_parallel counterpart of im2col_1d_gemm_rows.
// The input arrives as a flat channels_last array (random access), so no
// ac_channel / shift-register machinery is needed: gather each K-wide row by
// direct indexing. Row element order is kernel-col-major, channel-minor
//   a_row[kw * n_chan + c]
// to match the GEMM weight transposition ([W,C,F] -> [F, W*C]); the streaming
// path (kernel_shift_1d) packs the identical order. Assumes the conv-GEMM
// guard's constraints: valid (zero) padding, dilation 1 (any stride).
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
        #pragma hls_unroll
        for (unsigned kw = 0; kw < CONFIG_T::filt_width; kw++) {
        GemmRowsChan:
            #pragma hls_unroll
            for (unsigned c = 0; c < CONFIG_T::n_chan; c++) {
                out_pack[kw * CONFIG_T::n_chan + c] = data[(ow * CONFIG_T::stride_width + kw) * CONFIG_T::n_chan + c];
            }
        }
        a_rows[ow] = out_pack;
    }
}

// ---------------------------------------------------------------------------
// im2col_2d_gemm_rows_array — io_parallel counterpart of im2col_2d_gemm_rows.
// Row element order is spatial-row-major then channel
//   a_row[(kh * filt_width + kw) * n_chan + c]
// to match the Conv2D GEMM weight transposition ([H,W,C,F] -> [F, H*W*C]).
// Output rows are raster-ordered (oh * out_width + ow) = gemm_m order.
// Assumes valid padding, dilation 1 (the conv-GEMM guard); any stride.
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
            #pragma hls_unroll
            for (unsigned kh = 0; kh < CONFIG_T::filt_height; kh++) {
            GemmRowsFiltWidth:
                #pragma hls_unroll
                for (unsigned kw = 0; kw < CONFIG_T::filt_width; kw++) {
                GemmRowsChan:
                    #pragma hls_unroll
                    for (unsigned c = 0; c < CONFIG_T::n_chan; c++) {
                        out_pack[(kh * CONFIG_T::filt_width + kw) * CONFIG_T::n_chan + c] =
                            data[((oh * CONFIG_T::stride_height + kh) * CONFIG_T::in_width +
                                  (ow * CONFIG_T::stride_width + kw)) * CONFIG_T::n_chan + c];
                    }
                }
            }
            a_rows[oh * CONFIG_T::out_width + ow] = out_pack;
        }
    }
}

} // namespace nnet

#endif
