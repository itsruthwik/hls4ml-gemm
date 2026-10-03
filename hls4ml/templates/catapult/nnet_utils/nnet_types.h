#ifndef NNET_TYPES_H_
#define NNET_TYPES_H_

#include <assert.h>
#include <cstddef>
#include <cstdio>

#include "ac_int.h"

namespace nnet {

// Fixed-size array
template <typename T, unsigned N> struct array {
    typedef T value_type;
    static const unsigned size = N;

    T data[N];

    T &operator[](size_t pos) { return data[pos]; }

    const T &operator[](size_t pos) const { return data[pos]; }

    array &operator=(const array &other) {
        if (&other == this)
            return *this;

        assert(N == other.size && "Array sizes must match.");

        #pragma hls_unroll
        for (unsigned i = 0; i < N; i++) {
            data[i] = other[i];
        }
        return *this;
    }

    bool operator==(const array &other) const {
        if (N != other.size) {
            return false;
        }

        for (unsigned i = 0; i < N; i++) {
            if (data[i] != other[i]) {
                return false;
            }
        }

        return true;
    }

    bool operator!=(const array &other) const { return !(*this == other); }
};

// Storage of the weights one nnet::dense_resource call reads. Flat by default; when the writer
// stores a layer's weights block-major (CONFIG_T::block_major_weights) they are reuse_factor
// words of n_in*n_out/reuse_factor weights, word ir holding everything ReuseLoop iteration ir
// reads. This is the Catapult counterpart of the Vivado ARRAY_RESHAPE block factor=block_factor:
// one wide read per iteration from an array that is reuse_factor deep. A word is a single
// unsigned ac_int holding the weights' bit patterns side by side (lane im at bits
// [im*width, (im+1)*width)), not an nnet::array: Catapult splits a struct of weights into its
// elements and builds one ROM copy per lane, whereas an integer word maps to one ROM row.
template <class CONFIG_T, bool packed = CONFIG_T::block_major_weights> struct weight_store {
    typedef typename CONFIG_T::weight_t type;
    static const unsigned size = CONFIG_T::n_in * CONFIG_T::n_out;
};
template <class CONFIG_T> struct weight_store<CONFIG_T, true> {
    static const unsigned lanes = CONFIG_T::n_in * CONFIG_T::n_out / CONFIG_T::reuse_factor;
    typedef ac_int<lanes * CONFIG_T::weight_t::width, false> type;
    static const unsigned size = CONFIG_T::reuse_factor;
};

// Generic lookup-table implementation, for use in approximations of math functions
template <typename T, unsigned N, T (*func)(T)> class lookup_table {
  public:
    lookup_table(T from, T to) : range_start(from), range_end(to), base_div(ac_int<16, false>(N) / T(to - from)) {
        T step = (range_end - range_start) / ac_int<16, false>(N);
        for (size_t i = 0; i < N; i++) {
            T num = range_start + ac_int<16, false>(i) * step;
            T sample = func(num);
            samples[i] = sample;
        }
    }

    T operator()(T n) const {
        int index = (n - range_start) * base_div;
        if (index < 0)
            index = 0;
        else if (index > N - 1)
            index = N - 1;
        return samples[index];
    }

  private:
    T samples[N];
    const T range_start, range_end;
    ac_fixed<20, 16, true> base_div;
};

} // namespace nnet

#endif
