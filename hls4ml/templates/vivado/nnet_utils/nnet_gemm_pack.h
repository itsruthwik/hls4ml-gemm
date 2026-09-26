#ifndef NNET_GEMM_PACK_H_
#define NNET_GEMM_PACK_H_

// Packed-bit-stream boundary of a GEMM IP (Vivado/Vitis, io_stream).
//
// A GEMM node's IP is a concrete function, gemm_stream_<layer>, taking
// hls::stream<ap_uint<W>> operands: one full unpadded row (or, for the second
// operand of a two-operand GEMM, one full row or column) per beat, packed lane 0 in
// the low bits. Bit widths are exact -- elements * element width -- so a
// gemm-ip-gen package and this header compute the same number from the same config.
// hls4ml keeps its array streams everywhere else and converts right around the call
// with the two processes below; an RTL-blackbox IP then sits directly in the top
// dataflow region with no nested region and no wrapper of its own (Vitis will not
// hand a top-level argument straight to a blackbox, so the pack/unpack processes are
// always present, even where the GEMM is the model's first or last layer).

#include "ap_int.h"
#include "hls_stream.h"

namespace nnet {

template <class T> struct gemm_packed_bits {
    static const unsigned value = T::size * T::value_type::width;
};

// array beats -> packed beats, N_BEATS per invocation. Pure bit copy (raw pattern,
// never a value conversion), one beat per cycle, free-running across invocations.
template <class data_T, unsigned N_BEATS>
void pack_stream(hls::stream<data_T> &in, hls::stream<ap_uint<gemm_packed_bits<data_T>::value> > &out) {
    const unsigned W = data_T::value_type::width;
PACK: for (unsigned i = 0; i < N_BEATS; i++) {
        #pragma HLS PIPELINE II=1 rewind
        data_T beat = in.read();
        ap_uint<gemm_packed_bits<data_T>::value> bits;
        for (unsigned j = 0; j < data_T::size; j++) {
            #pragma HLS UNROLL
            bits.range(j * W + W - 1, j * W) = beat[j].range(W - 1, 0);
        }
        out.write(bits);
    }
}

// packed beats -> array beats, N_BEATS per invocation. The IP already emitted each
// lane at the result type's own width (requantized), so this loads the raw bit
// pattern with .range() and never re-converts the value.
template <class res_T, unsigned N_BEATS>
void unpack_stream(hls::stream<ap_uint<gemm_packed_bits<res_T>::value> > &in, hls::stream<res_T> &out) {
    const unsigned W = res_T::value_type::width;
UNPACK: for (unsigned i = 0; i < N_BEATS; i++) {
        #pragma HLS PIPELINE II=1 rewind
        ap_uint<gemm_packed_bits<res_T>::value> bits = in.read();
        res_T beat;
        for (unsigned j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            typename res_T::value_type v;
            v.range(W - 1, 0) = bits.range(j * W + W - 1, j * W);
            beat[j] = v;
        }
        out.write(beat);
    }
}

// The array-stream kernels these bodies wrap. Declared here (not included) because
// nnet_gemm_stream.h includes nnet_gemm_ip.h, which includes this file; the
// definitions come from nnet_gemm_stream.h (no package, csim) or from a soft-logic
// target's package header, both parsed later.
template <class data_T, class res_T, typename CONFIG_T>
void gemm_stream_const_weights(hls::stream<data_T> &data_stream, hls::stream<res_T> &res_stream);
template <class data0_T, class data1_T, class res_T, typename CONFIG_T>
void gemm_stream(hls::stream<data0_T> &a_stream, hls::stream<data1_T> &b_stream,
                 hls::stream<res_T> &res_stream);

// ---------------------------------------------------------------------------
// Packed-stream bodies for a GEMM computed in C++ rather than in an RTL blackbox:
// unpack, run the templated array-stream kernel (CONFIG_T selects the layer), pack.
// Used by hls4ml's own no-package csim stubs and by gemm-ip-gen's generic (soft
// logic) Vitis target, so both name the same two functions.
// ---------------------------------------------------------------------------
template <class data_T, class res_T, typename CONFIG_T>
void gemm_stream_packed_const_weights(hls::stream<ap_uint<gemm_packed_bits<data_T>::value> > &a,
                                      hls::stream<ap_uint<gemm_packed_bits<res_T>::value> > &p) {
    #pragma HLS DATAFLOW
    hls::stream<data_T> a_rows;
    hls::stream<res_T> c_rows;
    #pragma HLS STREAM variable=a_rows depth=2
    #pragma HLS STREAM variable=c_rows depth=2
    unpack_stream<data_T, CONFIG_T::gemm_m>(a, a_rows);
    gemm_stream_const_weights<data_T, res_T, CONFIG_T>(a_rows, c_rows);
    pack_stream<res_T, CONFIG_T::gemm_m>(c_rows, p);
}

template <class data0_T, class data1_T, class res_T, typename CONFIG_T>
void gemm_stream_packed(hls::stream<ap_uint<gemm_packed_bits<data0_T>::value> > &a,
                        hls::stream<ap_uint<gemm_packed_bits<data1_T>::value> > &b,
                        hls::stream<ap_uint<gemm_packed_bits<res_T>::value> > &p) {
    #pragma HLS DATAFLOW
    hls::stream<data0_T> a_rows;
    hls::stream<data1_T> b_beats;
    hls::stream<res_T> c_rows;
    #pragma HLS STREAM variable=a_rows depth=2
    #pragma HLS STREAM variable=b_beats depth=2
    #pragma HLS STREAM variable=c_rows depth=2
    unpack_stream<data0_T, CONFIG_T::gemm_m>(a, a_rows);
    unpack_stream<data1_T, (CONFIG_T::b_row_major ? CONFIG_T::gemm_k : CONFIG_T::gemm_n)>(b, b_beats);
    gemm_stream<data0_T, data1_T, res_T, CONFIG_T>(a_rows, b_beats, c_rows);
    pack_stream<res_T, CONFIG_T::gemm_m>(c_rows, p);
}

} // namespace nnet

#endif
