#ifndef NNET_IM2COL_STREAM_H_
#define NNET_IM2COL_STREAM_H_

#include "ap_shift_reg.h"
#include "ac_channel.h"
#include "nnet_common.h"
#include "nnet_conv_stream.h"

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void im2col_2d_cl(
    ac_channel<data_T> &data,
    ac_channel<res_T> &res) {
    
    static ap_shift_reg<typename data_T::value_type, CONFIG_T::in_width> line_buffer[MAX(CONFIG_T::filt_height - 1, 1)]
                                                                                    [CONFIG_T::n_chan];
    // Catapult does not use the Vivado ARRAY_PARTITION pragma syntax here.

    // Thresholds
    const static int lShiftX = CONFIG_T::filt_width - 1;
    const static int lShiftY = CONFIG_T::filt_height - 1;

    // Counters
    static int pX = 0; // Pixel X
    static int pY = 0; // Pixel Y

    static int sX = 0; // Stride X
    static int sY = 0; // Stride Y

    static typename data_T::value_type kernel_data[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan];
    // Catapult does not use the Vivado ARRAY_PARTITION pragma syntax here.

ReadInputHeight:
    for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
    ReadInputWidth:
        for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width; i_iw++) {
            #pragma hls_pipeline_init_interval 1

            // Add pixel to buffer
            nnet::shift_line_buffer<data_T, CONFIG_T>(data.read(), line_buffer, kernel_data);

            // Check to see if we have a full kernel
            if ((sX - lShiftX) == 0 && (sY - lShiftY) == 0 && pY > lShiftY - 1 && pX > lShiftX - 1) {
                res_T res_pack;
                PRAGMA_DATA_PACK(res_pack)

            PackLoop:
                for (unsigned i_ic = 0; i_ic < CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan; i_ic++) {
                    #pragma hls_unroll
                    res_pack[i_ic] = kernel_data[i_ic];
                }
                res.write(res_pack);
            }

            // Counter management
            if (pX + 1 == CONFIG_T::in_width) {
                pX = 0;
                sX = 0;
                pY++;
                sY++;
                if (sY == CONFIG_T::stride_height)
                    sY = 0;
            } else {
                pX++;
                sX++;
                if (sX == CONFIG_T::stride_width)
                    sX = 0;
            }
        }
    }
}

} // namespace nnet

#endif
