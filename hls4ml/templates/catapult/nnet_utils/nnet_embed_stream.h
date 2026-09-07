#ifndef NNET_EMBED_STREAM_H_
#define NNET_EMBED_STREAM_H_

#include "ac_channel.h"
#include "nnet_common.h"
#include "nnet_helpers.h"

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void embedding(ac_channel<data_T> &data, ac_channel<res_T> &res,
               typename CONFIG_T::embeddings_t embeddings[CONFIG_T::vocab_size * CONFIG_T::n_out]) {
    data_T in_data = data.read();
    constexpr int ce_reuse_factor = CONFIG_T::reuse_factor;
    (void)ce_reuse_factor;
// Vitis treats PIPELINE II=reuse_factor as a target it may beat (it achieves 1 here); Catapult
    // pipelines at exactly the requested II, so the stream driver runs at 1 and the reuse
    // factor sets the rate through the inner reuse loop where it applies.
    #pragma hls_pipeline_init_interval 1
InputSequence:
    for (int j = 0; j < data_T::size; j++) {

        res_T res_pack;

    #pragma hls_unroll
    DenseEmbedding:
        for (int i = 0; i < CONFIG_T::n_out; i++) {
            res_pack[i] = embeddings[in_data[j] * CONFIG_T::n_out + i];
        }
        res.write(res_pack);
    }
}

} // namespace nnet

#endif
