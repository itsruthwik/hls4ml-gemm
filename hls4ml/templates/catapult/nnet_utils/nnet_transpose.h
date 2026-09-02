#ifndef NNET_TRANSPOSE_H_
#define NNET_TRANSPOSE_H_

namespace nnet {

// Interface documentation only: the shape an einsum transpose CONFIG_T is expected
// to provide (the generated configs are plain structs matching this, they do NOT
// inherit it). Named distinctly from nnet_array.h's `transpose_config` (the base the
// array Transpose layer DOES inherit) so both headers can coexist in one translation
// unit — e.g. a Transpose layer alongside a baseline einsum (partial-GEMM configs).
struct einsum_transpose_config {
    static const unsigned dims;
    static const unsigned N;
    static const unsigned *const from_shape;
    static const unsigned *const to_shape;
    static const unsigned *const perm;
    static const unsigned *const perm_strides;
    static const unsigned *const index_map;
};

template <typename CONFIG_T> unsigned transfer_idx(unsigned index) {
    // Given output idx in c-order flat array, return input idx
    unsigned idx = 0;
    for (int i = (int)CONFIG_T::dims - 1; i >= 0; i--) {
        idx += (index % CONFIG_T::to_shape[i]) * CONFIG_T::perm_strides[i];
        index /= CONFIG_T::to_shape[i];
    }
    return idx;
}

template <typename data_T, typename res_T, typename CONFIG_T>
void transpose(const data_T data[CONFIG_T::N], res_T res[CONFIG_T::N]) {
    // NOTE: hls_unroll must PRECEDE the for to bind to it (Catapult). Previously
    // it sat inside the loop body and silently failed to bind (CIN-319), so the
    // transpose ran rolled at II=1 over N elements — ~64 cycles each, and with
    // three transposes per io_stream einsum GEMM stage that was ~190 cyc/stage of
    // pure overhead. Unrolled it is ~1 cycle (combinational reindex).
    #pragma hls_unroll
    for (unsigned i = 0; i < CONFIG_T::N; i++) {
        unsigned idx = CONFIG_T::index_map[i];
        res[i] = data[idx];
    }
}

} // namespace nnet

#endif
