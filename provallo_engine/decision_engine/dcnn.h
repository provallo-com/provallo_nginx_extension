#ifndef _DCNN_H_
#define _DCNN_H_

//refactor neural_helper.h :
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <numeric>
#include <complex>
#include <valarray>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <ctime>
 
#include "matrix.h"
#include "utils.h"
#include "info_helper.h"
//implement FFT for sampling
#include "sampling_helper.h"

namespace provallo 
{
    //refactor neural_helper.h :
    //convolutional neural network 
    //convolutional layer
    //pooling layer
    //fully connected layer
    //dropout layer
    //batch normalization layer
    //noisy layer
    //output layer
    //input layer
    //zero padding layer
    //neural network:
    template <typename T>
    using type_matrix = matrix<T>;

    template <typename T> 
    class neuro_layer 
    {   
        
    public:

        //supported loss functions:
        enum loss_func_type {
            mse,
            cross_entropy , c4_loss
        };
        //supported optimizers: 
        enum optimizer_type {
            sgd,
            momentum,
            nesterov,
            adagrad,
            rmsprop,
            adam
        };
        //supported weight initialization types:
        enum weight_init_type {
            uniform,
            normal,
            xavier,
            he
        };
        //supported activation functions:
        enum activation_function {
            sigmoid,
            tanh,
            relu,
            leaky_relu,
            elu,
            selu,
            softmax,
            identity,
            linear
        };
        //supported running modes:
        enum sample_type {
            single,
            batch,
            mini_batch,
            stochastic
        };
        //supported layer types:
        enum layer_type {
            input,
            convolution,
            pooling,
            pooling_average,
            zero_padding,
            dropout,
            batch_normalization,
            noisy,
            output
        };
        //supported weight update types:
        enum weight_update_type {
            gradient_descent,
            gradient_descent_momentum,
            gradient_descent_nesterov,
            adagrad,
            rmsprop,
            adam
        };
        //supported sample types:
        enum sample_type
        {
            audio_matrix,
            video_matrix,
            image_matrix,
            text_matrix,
            none
        };
        //neural layer:
        using ptr= std::unique_ptr<neuro_layer>;
        using real_type = real_t;
        using type_matrix = matrix<T>;
    
        virtual type_matrix
        process_layer (const type_matrix &inputs)=0;
        virtual type_matrix
        layer_backpropagation (const type_matrix &xnPartialDerivative, real_t step) =0;
        virtual type_matrix
        layer_backdrop_invariant (const type_matrix & xnPartialDerivative) =0;
        virtual type_matrix
        mini_batch_layer_backdrop (const type_matrix & xnPartialDerivative, real_t step) =0;
        virtual void
        update_layer_weights (size_t minibatchSize = 1) =0;

        virtual void
        reset () =0;
        virtual size_t
        get_input_size () =0;

        virtual neuro_layer<T>* next_layer () =0;
        virtual neuro_layer<T>* previous_layer () =0;
        virtual void set_next_layer (neuro_layer<T>* next) =0;
        virtual void set_previous_layer (neuro_layer<T>* previous) =0;
        layer_type get_layer_type () const =0;
        activation_function get_activation_function () const =0;
        virtual void set_activation_function (activation_function function) =0;
        virtual void set_sample_type (sample_type type) =0;
        virtual void init ( size_t inputSize, size_t outputSize, size_t batchSize = 1) =0;

     protected:
        virtual type_matrix
        //derivative of the activation function unary matrix operation with the activation function as the unary function 
        fnDerivativeMatrix () const =0;
        //current activation function corresponding to the layer: 
        std::function<real_type(real_type)> _activationFunction;
        
        std::function<real_type(real_type)> _activationFunctionDerivative;        

        //available activation functions:

        std::vector<std::function<real_type(real_type)>> _activationFunctions;
          //current activation function derivative:
        
        //current loss function:
        std::function<real_type(real_type,real_type)> _lossFunction;
        //available loss functions:
        std::vector<std::function<real_type(real_type,real_type)>> _lossFunctions;
        //current optimizer:
        std::function<real_type(real_type,real_type)> _optimizer;
        //available optimizers:
        std::vector<std::function<real_type(real_type,real_type)>> _optimizers;
        
         
    }/*neuro_layer*/;

    //implement neuro layers for neural network:
    //input layers:
    template<typename T>
    class convolutional_operation
    {
    public:
        using real_type = real_t;
        using type_matrix = matrix<T>;
        type_matrix
        convolutional_layer (const type_matrix &inputs)
        {
            type_matrix ret = inputs * inputs.transpose() ;
            
            ret.unaryExpr ([](real_type val) { return 1.0 / (1.0 + std::exp(-val)); }); 
            ret.unaryExpr ([](real_type val) { return val > 0.5 ? 1.0 : 0.0; }); 
            ret = ret * ret.transpose();
            return ret;
        }
    }/*convolutional_operation*/;
    
    template <typename T>
    class input_layer : public neuro_layer<T>
    {
    public:
        using real_type = real_t;
        using type_matrix = matrix<T>;
        
        //input layer constructor:
        input_layer (size_t inputSize, size_t outputSize, size_t batchSize = 1) : _inputSize (inputSize), _outputSize (outputSize), _batchSize (batchSize)
        {
            _activationFunction = _activationFunctions[neuro_layer<T>::activation_function::identity];
            _activationFunctionDerivative = _activationFunctions[neuro_layer<T>::activation_function::identity];
            _lossFunction = _lossFunctions[neuro_layer<T>::loss_func_type::mse];
            _optimizer = _optimizers[neuro_layer<T>::optimizer_type::sgd];
        } 
        //process layer:
        type_matrix
        process_layer (const type_matrix &inputs) override
        {
            return inputs;
        }

        type_matrix
        layer_backpropagation (const type_matrix &xnPartialDerivative, real_t step) override
        {
            //input layer does not have a backpropagation process attached to it 
            return xnPartialDerivative;
        }

        type_matrix
        layer_backdrop_invariant (const type_matrix & xnPartialDerivative) override
        {
            //input layer does not have a backpropagation invariant 
            return xnPartialDerivative;
        }

        type_matrix
        mini_batch_layer_backdrop (const type_matrix & xnPartialDerivative, real_t step) override
        {
            //input layer does not have a mini batch backpropagation process attached to it 
            return xnPartialDerivative;
        }
        type_matrix 
        get_input_size () override
        {
            return _inputSize;
        }
        void
        update_layer_weights (size_t minibatchSize = 1) override
        {
            this->mini_batch_layer_backdrop (minibatchSize, 0.0); 

        }
        void
        reset () override
        {
            //input layer does not have a reset process attached to it 
        }
        neuro_layer<T>*
        next_layer () override
        {
            return _nextLayer;
        }
        neuro_layer<T>*
        previous_layer () override
        {
            return _previousLayer;
        }
        void
        set_next_layer (neuro_layer<T>* next) override
        {
            _nextLayer = next;
        }
        void
        set_previous_layer (neuro_layer<T>* previous) override
        {
            _previousLayer = previous;
        }
        layer_type
        get_layer_type () const override
        {
            return layer_type::input;
        }
        activation_function
        get_activation_function () const override
        {
            return _activationFunction;
        }
        void
        set_activation_function (activation_function function) override
        {
            _activationFunction = _activationFunctions[function];
            _activationFunctionDerivative = _activationFunctions[function];
        }
        void
        set_sample_type (sample_type type) override
        {
            _sampleType = type;
        }
        void
        init ( size_t inputSize, size_t outputSize, size_t batchSize = 1) override
        {
            _inputSize = inputSize;
            _outputSize = outputSize;
            _batchSize = batchSize;
        }
    private:
        type_matrix
        fnDerivativeMatrix () const override
        {
            return type_matrix (_inputSize, _batchSize, 1.0);
        }
        size_t _inputSize;
        size_t _outputSize;
        size_t _batchSize;
        sample_type _sampleType;
        neuro_layer<T>* _nextLayer;
        neuro_layer<T>* _previousLayer;
    }/*input_layer*/;

    //implement convolutional layer:
    template <typename T>
    class convolution
    {
    public:
        using real_type = real_t;
        using type_matrix = matrix<T>;
        //convolutional layer constructor:
        convolution (size_t inputSize, size_t outputSize, size_t batchSize = 1) : _inputSize (inputSize), _outputSize (outputSize), _batchSize (batchSize)
        {
            _activationFunction = _activationFunctions[neuro_layer<T>::activation_function::relu];
            _activationFunctionDerivative = _activationFunctions[neuro_layer<T>::activation_function::relu];
            _lossFunction = _lossFunctions[neuro_layer<T>::loss_func_type::mse];
            _optimizer = _optimizers[neuro_layer<T>::optimizer_type::sgd];
        }
        //process layer:
        type_matrix
        process_layer (const type_matrix &inputs) override
        {
            return inputs;
        }

        type_matrix
        layer_backpropagation (const type_matrix &xnPartialDerivative, real_t step) override
        {
            //convolutional layer does not have a backpropagation process attached to it 
            return xnPartialDerivative;
        }

        type_matrix
        layer_backdrop_invariant (const type_matrix & xnPartialDerivative) override
        {
            //convolutional layer does not have a backpropagation invariant 
            return xnPartialDerivative;
        }

        type_matrix
        mini_batch_layer_backdrop (const type_matrix & xnPartialDerivative, real_t step) override
        {
            //convolutional layer does not have a mini batch backpropagation process attached to it 
            return xnPartialDerivative;
        }
        type_matrix 
        get_input_size () override
        {
            return _inputSize;
        }
        void
        update_layer_weights (size_t minibatchSize = 1) override
        {
            this->mini_batch_layer_backdrop (minibatchSize, 0.0); 

        }
        void
        reset () override
        {
            //convolutional layer does not have a reset process attached to it 
        }
        neuro_layer<T>*
        next_layer () override
        {
            return _nextLayer;
        }
        neuro_layer<T>*
        previous_layer () override
        {
            return _previousLayer;
        }
        void
        set_next_layer (neuro_layer<T>* next) override
        {
            _nextLayer  = next;
        }
        void
        set_previous_layer (neuro_layer<T>* previous) override
        {
            _previousLayer = previous;
        }       
        layer_type  
        get_layer_type () const override
        {
            return layer_type::convolution;
        }       
        activation_function
        get_activation_function () const override
        {
            return _activationFunction;
        }   
        void
        set_activation_function (activation_function function) override
        {
            _activationFunction = _activationFunctions[function];
            _activationFunctionDerivative = _activationFunctions[function];
        }   
        void    
        set_sample_type (sample_type type) override
        {
            _sampleType = type;
        }   
        void    
        init ( size_t inputSize, size_t outputSize, size_t batchSize = 1) override
        {
            _inputSize = inputSize;
            _outputSize = outputSize;
            _batchSize = batchSize;
        }   
        //process layer use the convolutional layer to process the input matrix: 
        type_matrix
        process_layer (const type_matrix &inputs) override
        {
            convolutional_operation<T> convOp;
            return convOp.convolutional_layer (inputs); 
            
        }

    private:    
        type_matrix
        fnDerivativeMatrix () const override
        {
            return type_matrix (_inputSize, _batchSize, 1.0);
        }   
        size_t _inputSize;
        size_t _outputSize;
        size_t _batchSize;
        sample_type _sampleType;
        neuro_layer<T>* _nextLayer;
        neuro_layer<T>* _previousLayer;
    }/*convolution*/;

    //implement pooling layer:
    template <typename T>
    class pooling
    {   
    public:
        using real_type = real_t;
        using type_matrix = matrix<T>;
        //pooling layer constructor:
        pooling (size_t inputSize, size_t outputSize, size_t batchSize = 1) : _inputSize (inputSize), _outputSize (outputSize), _batchSize (batchSize)
        {
            _activationFunction = _activationFunctions[neuro_layer<T>::activation_function::relu];
            _activationFunctionDerivative = _activationFunctions[neuro_layer<T>::activation_function::relu];
            _lossFunction = _lossFunctions[neuro_layer<T>::loss_func_type::mse];
            _optimizer = _optimizers[neuro_layer<T>::optimizer_type::sgd];
        }
        //process layer:
        type_matrix
        process_layer (const type_matrix &inputs) override
        {
            return inputs;
        }

        type_matrix
        layer_backpropagation (const type_matrix &xnPartialDerivative, real_t step) override
        {
            //pooling layer does not have a backpropagation process attached to it 
            return xnPartialDerivative;
        }

        type_matrix
        layer_backdrop_invariant (const type_matrix & xnPartialDerivative) override
        {
            //pooling layer does not have a backpropagation invariant 
            return xnPartialDerivative;
        }

        type_matrix
        mini_batch_layer_backdrop (const type_matrix & xnPartialDerivative, real_t step) override
        {
            //pooling layer does not have a mini batch backpropagation process attached to it 
            return xnPartialDerivative;
        }
        type_matrix 
        get_input_size () override
        {
            return _inputSize;
        }
        void
        update_layer_weights (size_t minibatchSize = 1) override
        {
            this->mini_batch_layer_backdrop (minibatchSize, 0.0); 

        }
        void
        reset () override
        {
            //pooling layer does not have a reset process attached to it 
        }
        neuro_layer<T>*
        next_layer () override
        {
            return _nextLayer;
        }
        neuro_layer<T>*
        previous_layer () override
        {
            return _previousLayer;
        }
        void
        set_next_layer (neuro_layer<T>* next) override
        {
            _nextLayer = next;
        }   
        void
        set_previous_layer (neuro_layer<T>* previous) override
        {
            _previousLayer = previous;
        }   
        layer_type  
        get_layer_type () const override
        {
            return layer_type::pooling;
        }   
        activation_function
        get_activation_function () const override
        {
            return _activationFunction;
        }   
        void
        set_activation_function (activation_function function) override
        {
            _activationFunction = _activationFunctions[function];
            _activationFunctionDerivative = _activationFunctions[function];
        }   
        void
        set_sample_type (sample_type type) override
        {
            _sampleType = type;
        }   
        void
        init ( size_t inputSize, size_t outputSize, size_t batchSize = 1) override
        {
            _inputSize = inputSize;
            _outputSize = outputSize;
            _batchSize = batchSize;
        }   
    private:
        type_matrix
        fnDerivativeMatrix () const override
        {
            return type_matrix (_inputSize, _batchSize, 1.0);
        }   
        size_t _inputSize;
        size_t _outputSize;
        size_t _batchSize;
        enum sample_type _sampleType;
        neuro_layer<T>* _nextLayer;
        neuro_layer<T>* _previousLayer;
    }/*pooling*/;

    //implement pooling average layer:

    template <typename T>
    class pooling_average
    {
    public:
        using real_type = real_t;
        using type_matrix = matrix<T>;
        //pooling average layer constructor:
        pooling_average (size_t inputSize, size_t outputSize, size_t batchSize = 1) : _inputSize (inputSize), _outputSize (outputSize), _batchSize (batchSize)
        {
            _activationFunction = _activationFunctions[neuro_layer<T>::activation_function::relu];
            _activationFunctionDerivative = _activationFunctions[neuro_layer<T>::activation_function::relu];
            _lossFunction = _lossFunctions[neuro_layer<T>::loss_func_type::mse];
            _optimizer = _optimizers[neuro_layer<T>::optimizer_type::sgd];
        }
        //process layer:
        type_matrix
        process_layer (const type_matrix &inputs) override
        {
            return inputs;
        }

        type_matrix
        layer_backpropagation (const type_matrix &xnPartialDerivative, real_t step) override
        {
            //pooling average layer does not have a backpropagation process attached to it 
            return xnPartialDerivative;
        }

        type_matrix
        layer_backdrop_invariant (const type_matrix & xnPartialDerivative) override
        {
            //pooling average layer does not have a backpropagation invariant 
            return xnPartialDerivative;
        }

        type_matrix
        mini_batch_layer_backdrop (const type_matrix & xnPartialDerivative, real_t step) override
        {
            //pooling average layer does not have a mini batch backpropagation process attached to it 
            return xnPartialDerivative;
        }
        type_matrix 
        get_input_size () override
        {
            return _inputSize;
        }
        void
        update_layer_weights (size_t minibatchSize = 1) override
        {
            this->mini_batch_layer_backdrop (minibatchSize, 0.0); 

        }
        void
        reset () override
        {
            //pooling average layer does not have a reset process attached to it 
        }
        neuro_layer<T>*
        next_layer () override
        {
            return _nextLayer;
        }
        neuro_layer<T>*
        previous_layer () override
        {
            return _previousLayer;
        }
        void
        set_next_layer (neuro_layer<T>* next) override
        {
            _next
        }
        void
        set_previous_layer (neuro_layer<T>* previous) override
        {
            _previousLayer = previous;
        }   
        layer_type
        get_layer_type () const override
        {
            return layer_type::pooling_average;
        }   
        activation_function
        get_activation_function () const override
        {
            return _activationFunction;
        }   
        void
        set_activation_function (activation_function function) override
        {
            _activationFunction = _activationFunctions[function];
            _activationFunctionDerivative = _activationFunctions[function];
        }   
        void
        set_sample_type (sample_type type) override
        {
            _sampleType = type;
        }   
        void
        init ( size_t inputSize, size_t outputSize, size_t batchSize = 1) override
        {
            _inputSize = inputSize;
            _outputSize = outputSize;
            _batchSize = batchSize;
        }
    private:

        type_matrix
        fnDerivativeMatrix () const override
        {
            return type_matrix (_inputSize, _batchSize, 1.0);
        }
        size_t _inputSize;
        size_t _outputSize;
        size_t _batchSize;
        sample_type _sampleType;
        neuro_layer<T>* _nextLayer;
        neuro_layer<T>* _previousLayer;
    }/*pooling_average*/;


    //implement zero padding layer:
    template <typename T>
    class zero_padding
    {
        
    };
    //implement dropout layer:
    template <typename T>
    class dropout
    {
    };
    //implement batch normalization layer:
    template <typename T>
    class batch_normalization
    {
    };
    //implement noisy layer:
    template <typename T>
    class noisy
    {
    };
    //implement output layer:
    template <typename T>
    class output
    {
    };
}/*provallo*/

#endif