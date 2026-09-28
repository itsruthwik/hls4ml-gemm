#ifndef NNET_STREAM_BEAT_H_
#define NNET_STREAM_BEAT_H_

// Stream beats that may travel packed (Vivado/Vitis, io_stream).
//
// A GEMM IP takes hls::stream<ap_uint<W>> operands: one nnet::array beat with lane 0 in the
// low bits and every lane at its own exact width. An edge between a GEMM and a neighbour that
// supports it can stay in that packed form, so the neighbour converts inside its own pipeline
// and no separate pack/unpack process (with its own fill, drain and start sync) sits on the
// edge. Such an edge is named by the tag nnet::packed<T> in place of the array type T in a
// layer's template arguments; beat_io<> maps either form to the stream element, the array
// type the layer computes on, and the read/write between them.

#include "ap_int.h"
#include "hls_stream.h"

namespace nnet {

template <class T> struct gemm_packed_bits {
    static const unsigned value = T::size * T::value_type::width;
};

// Raw bit copies between one array beat and its packed form; never a value conversion.
template <class T> ap_uint<gemm_packed_bits<T>::value> pack_beat(const T &beat) {
    #pragma HLS INLINE
    ap_uint<gemm_packed_bits<T>::value> bits;
    for (int j = 0; j < (int)T::size; j++) {
        #pragma HLS UNROLL
        bits.range(j * T::value_type::width + T::value_type::width - 1, j * T::value_type::width) =
            beat[j].range(T::value_type::width - 1, 0);
    }
    return bits;
}

template <class T> T unpack_beat(const ap_uint<gemm_packed_bits<T>::value> &bits) {
    #pragma HLS INLINE
    T beat;
    for (int j = 0; j < (int)T::size; j++) {
        #pragma HLS UNROLL
        typename T::value_type v;
        v.range(T::value_type::width - 1, 0) =
            bits.range(j * T::value_type::width + T::value_type::width - 1, j * T::value_type::width);
        beat[j] = v;
    }
    return beat;
}

// Tag: the array type T carried packed on this edge.
template <class T> struct packed {};

template <class T> struct beat_io {
    typedef T array_t;
    typedef T elem_t;
    static T read(hls::stream<elem_t> &s) {
        #pragma HLS INLINE
        return s.read();
    }
    static void write(hls::stream<elem_t> &s, const T &beat) {
        #pragma HLS INLINE
        s.write(beat);
    }
};

template <class T> struct beat_io<packed<T> > {
    typedef T array_t;
    typedef ap_uint<gemm_packed_bits<T>::value> elem_t;
    static T read(hls::stream<elem_t> &s) {
        #pragma HLS INLINE
        return unpack_beat<T>(s.read());
    }
    static void write(hls::stream<elem_t> &s, const T &beat) {
        #pragma HLS INLINE
        s.write(pack_beat<T>(beat));
    }
};

} // namespace nnet

#endif
