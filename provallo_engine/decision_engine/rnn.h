#ifndef __RNN_H__
#define __RNN_H__

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <algorithm>
#include <iterator>
#include <chrono>
#include <thread>
#include <mutex>

#include "matrix.h"
#include "utils.h"
//RNN Denoiser model 
//
#define WEIGHTS_SCALE (1.f/256)

#define MAX_NEURONS 128

#define ACTIVATION_TANH    0
#define ACTIVATION_SIGMOID 1
#define ACTIVATION_RELU    2
#define ACTIVATION_NONE    3


using namespace std;

namespace provallo {
    typedef signed char rnn_weight;

    typedef float float_t;
    
    struct DenseLayer;
    struct GRULayer;

    struct RNNModel {
    size_t input_dense_size;
    const DenseLayer *input_dense;

    size_t vad_gru_size;
    const GRULayer *vad_gru;

    size_t noise_gru_size;
    const GRULayer *noise_gru;

    size_t denoise_gru_size;
    const GRULayer *denoise_gru;

    size_t denoise_output_size;
    const DenseLayer *denoise_output;

    size_t vad_output_size;
    const DenseLayer *vad_output;
    };
    typedef matrix<real_t> Matrix;
    struct DenseLayer {
        size_t input_size;
        size_t output_size;
        Matrix weights;
        Matrix biases;
    };

    struct RNNState {
              const RNNModel *model;
        real_t *vad_gru_state;
        real_t *noise_gru_state;
        real_t *denoise_gru_state;
     };
    class RNN {
    public:
        RNN(int input_size, int hidden_size, int output_size, int seq_length, int learning_rate, int epochs, int batch_size, int num_layers, int num_threads);
        ~RNN();
        void forward(vector<vector<int>> &data, vector<vector<int>> &labels);
        void backward(vector<vector<int>> &data, vector<vector<int>> &labels);
        void update();
        void compute_input(const real_t *input, real_t *output); 
        void compute_dense_layer(DenseLayer *layer, const real_t *input, real_t *output); 
        void compute_gru_layer(GRULayer *layer, const real_t *input, real_t *output, real_t *state);
        void compute_rnn(RNNState *state, const real_t *input, real_t *output);
        void compute_rnn_vad(RNNState *state, const real_t *input, real_t *output);
        void compute_rnn_denoise(RNNState *state, const real_t *input, real_t *output);
        void compute_rnn_noise(RNNState *state, const real_t *input, real_t *output);
        void compute_rnn_output(RNNState *state, const real_t *input, real_t *output);
        
        void train(vector<vector<int>> &data, vector<vector<int>> &labels);
        void test(vector<vector<int>> &data, vector<vector<int>> &labels);
        void save(string filename);
        void load(string filename);     

        void set_learning_rate(real_t learning_rate);
     private:   
        void init();
        void init_model();
        void init_state();
        void init_weights();
        void init_weights_dense(DenseLayer *layer, size_t input_size, size_t output_size);
        void init_weights_gru(GRULayer *layer, size_t input_size, size_t output_size);
        void init_weights_rnn(RNNModel *model, size_t input_size, size_t hidden_size, size_t output_size, size_t num_layers);
        void init_weights_rnn_vad(RNNModel *model, size_t input_size, size_t hidden_size, size_t output_size, size_t num_layers);
        void init_weights_rnn_denoise(RNNModel *model, size_t input_size, size_t hidden_size, size_t output_size, size_t num_layers);
        void init_weights_rnn_noise(RNNModel *model, size_t input_size, size_t hidden_size, size_t output_size, size_t num_layers);
        void init_weights_rnn_output(RNNModel *model, size_t input_size, size_t hidden_size, size_t output_size, size_t num_layers);
        void init_state_rnn(RNNState *state, const RNNModel *model);
        void init_state_rnn_vad(RNNState *state, const RNNModel *model);
        void init_state_rnn_denoise(RNNState *state, const RNNModel *model);
        void init_state_rnn_noise(RNNState *state, const RNNModel *model);
        void init_state_rnn_output(RNNState *state, const RNNModel *model);
        
        RNNModel *model;
        RNNState *state;
        real_t *input;
        real_t *output;
        real_t *labels;
        real_t *loss;
        real_t *loss_output;
        real_t *loss_denoise;
        real_t *loss_vad;
        real_t *loss_noise;

        real_t *loss_denoise_grad;
        real_t *loss_vad_grad;
        real_t *loss_noise_grad;

        real_t *loss_output_grad;
        real_t *loss_grad;
        real_t *output_grad;
        real_t *denoise_output_grad;
        real_t *vad_output_grad;
        real_t *noise_output_grad;
        real_t *denoise_output;
        real_t *vad_output;
        real_t *noise_output;
        real_t *denoise_gru_state;
        real_t *vad_gru_state;
        real_t *noise_gru_state;
        real_t *denoise_gru_state_grad;
        real_t *vad_gru_state_grad;
        real_t *noise_gru_state_grad;
        real_t denoise_gru_state_grad_sum;
        real_t vad_gru_state_grad_sum;
        real_t noise_gru_state_grad_sum;
        real_t *denoise_gru_state_grad_sum_arr;
        real_t *vad_gru_state_grad_sum_arr;
        real_t *noise_gru_state_grad_sum_arr;
        

     };

}// namespace provallo

#endif 