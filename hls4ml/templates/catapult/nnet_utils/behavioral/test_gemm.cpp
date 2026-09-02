#include <iostream>
#include <vector>
#include "ac_int.h"
#include "ac_fixed.h"
#include "ac_channel.h"
#include "../nnet_helpers.h"
#include "../nnet_types.h"

// Mock CONFIG_T
struct MockConfig {
    static const unsigned n_in = 4;
    static const unsigned n_out = 2;
    static const unsigned gemm_m = 1;
    static const unsigned gemm_k = 4;
    static const unsigned gemm_n = 2;
    typedef float weight_t;
    typedef float bias_t;
    typedef float accum_t;

    template<class T1, class T2>
    struct product_impl {
        static float product(T1 a, T2 b) { return a * b; }
    };
    template<class T1, class T2>
    using product = product_impl<T1, T2>;
};

// Data types
typedef nnet::array<float, 4> data_T;
typedef nnet::array<float, 2> res_T;
typedef nnet::array<float, 4> weight_T;

// Include the streaming GEMM entry points (gemm_stream lives here; it pulls in
// nnet_gemm_ip.h). No package / no __SYNTHESIS__ -> the inline behavioral is used.
#include "../nnet_gemm_stream.h"

int main() {
    ac_channel<data_T> a_stream;
    ac_channel<weight_T> bt_stream;
    float biases[2] = {1.0, 2.0};
    ac_channel<res_T> c_stream;

    // Input data
    data_T in;
    for(int i=0; i<4; i++) in[i] = i + 1; // 1, 2, 3, 4
    a_stream.write(in);

    // Weights (Transposed)
    // Row 0: 0.1, 0.2, 0.3, 0.4
    // Row 1: 0.5, 0.6, 0.7, 0.8
    weight_T w0, w1;
    for(int i=0; i<4; i++) w0[i] = 0.1 * (i+1);
    for(int i=0; i<4; i++) w1[i] = 0.1 * (i+5);
    bt_stream.write(w0);
    bt_stream.write(w1);

    // Call the two-operand streaming GEMM entry (A stream + B stream, non-buffered).
    nnet::gemm_stream<data_T, weight_T, res_T, MockConfig>(a_stream, bt_stream, c_stream, biases);

    // Check result
    res_T out = c_stream.read();
    
    // Expected:
    // Out[0] = (1*0.1 + 2*0.2 + 3*0.3 + 4*0.4) + 1.0 = (0.1 + 0.4 + 0.9 + 1.6) + 1.0 = 3.0 + 1.0 = 4.0
    // Out[1] = (1*0.5 + 2*0.6 + 3*0.7 + 4*0.8) + 2.0 = (0.5 + 1.2 + 2.1 + 3.2) + 2.0 = 7.0 + 2.0 = 9.0

    std::cout << "Out[0]: " << out[0] << " (Expected 4.0)" << std::endl;
    std::cout << "Out[1]: " << out[1] << " (Expected 9.0)" << std::endl;

    if (out[0] == 4.0 && out[1] == 9.0) {
        std::cout << "SUCCESS" << std::endl;
        return 0;
    } else {
        std::cout << "FAILURE" << std::endl;
        return 1;
    }
}
