#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

static std::string s_weights_dir;

const char *get_weights_dir() { return s_weights_dir.c_str(); }

#include "firmware/myproject.h"
#include "nnet_utils/nnet_helpers.h"
// #include "firmware/parameters.h"

#include <mc_scverify.h>

// hls-fpga-machine-learning insert bram

#define CHECKPOINT 5000

#ifndef RANDOM_FRAMES
#define RANDOM_FRAMES 1
#endif

// hls-fpga-machine-learning insert declare weights

namespace nnet {
bool trace_enabled = true;
std::map<std::string, void *> *trace_outputs = NULL;
size_t trace_type_size = sizeof(double);

// Defined here rather than in nnet_helpers.h: Catapult puts its own bundled nnet_utils ahead of
// the project's copy on the testbench include path, so helpers added to the project's
// nnet_helpers.h are not seen.
// Raw integer code of a fixed-point value: its bit pattern as a sign-extended (signed types) or
// plain (unsigned types) integer, i.e. value * 2^frac_bits exactly, with no decimal formatting.
template <class T> long long fixed_raw_code(const T &v) { return v.template slc<T::width>(0).to_int64(); }

// print_result, but writing each output's raw integer code (fixed_raw_code). Used for the
// tb_data/*.raw files, which compare exactly against integer golden codes.
template <class res_T, size_t SIZE> void print_result_raw(res_T result[SIZE], std::ostream &out, bool keep = false) {
    for (unsigned i = 0; i < SIZE; i++) {
        out << fixed_raw_code(result[i]) << " ";
    }
    out << std::endl;
}

template <class res_T, size_t SIZE> void print_result_raw(ac_channel<res_T> &result, std::ostream &out, bool keep = false) {
    if (!keep) {
        while (result.available(1)) {
            res_T res_pack = result.read();
            for (unsigned int j = 0; j < res_T::size; j++) {
                out << fixed_raw_code(res_pack[j]) << " ";
            }
        }
        out << std::endl;
    } else {
        if (result.debug_size() >= SIZE / res_T::size) {
            for (unsigned int i = 0; i < SIZE / res_T::size; i++) {
                res_T res_pack = result[i]; // peek
                for (unsigned int j = 0; j < res_T::size; j++) {
                    out << fixed_raw_code(res_pack[j]) << " ";
                }
            }
            out << std::endl;
        }
    }
}
} // namespace nnet

CCS_MAIN(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Error - too few arguments" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <weights_dir> <tb_input_features> <tb_output_predictions>" << std::endl;
        std::cerr << "Where: <weights_dir>           - string pathname to directory containing wN.txt and bN.txt files"
                  << std::endl;
        std::cerr << "       <tb_input_features>     - string pathname to tb_input_features.dat (optional)" << std::endl;
        std::cerr << "       <tb_output_predictions> - string pathname to tb_output_predictions.dat (optional)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "If no testbench input/prediction data provided, random input data will be generated" << std::endl;
        CCS_RETURN(-1);
    }
    s_weights_dir = argv[1];
    std::cout << "  Weights directory: " << s_weights_dir << std::endl;

    std::string tb_in;
    std::string tb_out;
    std::ifstream fin;
    std::ifstream fpr;
    bool use_random = false;
    if (argc == 2) {
        std::cout << "No testbench files provided - Using random input data" << std::endl;
        use_random = true;
    } else {
        tb_in = argv[2];
        tb_out = argv[3];
        std::cout << "  Test Feature Data: " << tb_in << std::endl;
        std::cout << "  Test Predictions : " << tb_out << std::endl;

        // load input data from text file
        fin.open(tb_in);
        // load predictions from text file
        fpr.open(tb_out);
        if (!fin.is_open() || !fpr.is_open()) {
            use_random = true;
        }
    }

#if defined(RTL_SIM) || defined(CCS_DUT_RTL)
    std::string RESULTS_LOG = "tb_data/rtl_cosim_results.log";
#else
    std::string RESULTS_LOG = "tb_data/csim_results.log";
#endif
    std::ofstream fout(RESULTS_LOG);
    // The same outputs as raw integer codes (value * 2^frac_bits), for exact comparison.
    std::string RAW_LOG = RESULTS_LOG.substr(0, RESULTS_LOG.size() - 4) + ".raw";
    std::ofstream fraw(RAW_LOG);

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        loaded_weights = true;
    }
#endif
    std::string iline;
    std::string pline;
    int e = 0;

    if (!use_random) {
        while (std::getline(fin, iline) && std::getline(fpr, pline)) {
            if (e % CHECKPOINT == 0)
                std::cout << "Processing input " << e << std::endl;
            char *cstr = const_cast<char *>(iline.c_str());
            char *current;
            std::vector<float> in;
            current = strtok(cstr, " ");
            while (current != NULL) {
                in.push_back(atof(current));
                current = strtok(NULL, " ");
            }
            cstr = const_cast<char *>(pline.c_str());
            std::vector<float> pr;
            current = strtok(cstr, " ");
            while (current != NULL) {
                pr.push_back(atof(current));
                current = strtok(NULL, " ");
            }
            //    std::cout << "    Input feature map size = " << in.size() << " Output predictions size = " << pr.size() <<
            //    std::endl;

            // hls-fpga-machine-learning insert data

            // hls-fpga-machine-learning insert top-level-function

            if (e % CHECKPOINT == 0) {
                std::cout << "Predictions" << std::endl;
                // hls-fpga-machine-learning insert predictions
                std::cout << "Quantized predictions" << std::endl;
                // hls-fpga-machine-learning insert quantized
            }
            e++;

            // hls-fpga-machine-learning insert tb-output
        }
        if (fin.is_open()) {
            fin.close();
        }
        if (fpr.is_open()) {
            fpr.close();
        }
    } else {
        std::cout << "INFO: Unable to open input/predictions file(s) so feeding random values" << std::endl;
        std::cout << "Number of Frames Passed from the tcl= " << RANDOM_FRAMES << std::endl;

        if (RANDOM_FRAMES > 0) {
            for (unsigned int k = 0; k < RANDOM_FRAMES; k++) {
                // hls-fpga-machine-learning insert random

                // hls-fpga-machine-learning insert top-level-function

                // hls-fpga-machine-learning insert output

                // hls-fpga-machine-learning insert tb-output
            }
        } else {
            // hls-fpga-machine-learning insert zero

            // hls-fpga-machine-learning insert top-level-function

            // hls-fpga-machine-learning insert output

            // hls-fpga-machine-learning insert tb-output
        }
    }

    fout.close();
    fraw.close();
    std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

    return 0;
}
