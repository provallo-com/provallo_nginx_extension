/*
 * neuralhelper.cpp
 *
 *  Created on: May 28, 2023
 *      Author: kardon
 */

#include "neuralhelper.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <thread>
namespace provallo
{
    using type_matrix = matrix<float>;

    //  Sigmoid
    ActivationFun
    sigmoid(float lambda)
    {
        return [=](float x)
        {
            return 1.f / (1.f + exp(-lambda * x));
        };
    }
    //  Heavyside
    ActivationFun
    heavyside(float gapAbscissa)
    {
        return [=](float x)
        {
            return (x < gapAbscissa) ? 0 : 1;
        };
    }
    //  Hyperbolic Tangent
    ActivationFun
    hyperTan()
    {
        return [=](float x)
        {
            return tanh(2 * x / 3);
        };
    }
    //  reLu
    ActivationFun
    reLu()
    {
        return [=](float x)
        {
            if (x > 0.f)
                return x;
            else
                return 0.f;
        };
    }
    //  Leaky ReLu
    ActivationFun
    reLuLeaky(float lambda)
    {
        return [=](float x)
        {
            if (x > 0)
                return x;
            else
                return lambda * x;
        };
    }
    //  Mean Squared Error Function
    ErrorFun
    coutDiscr()
    {
        return [=](type_matrix v1, type_matrix v2)
        {
            float res = 0;
            for (size_t i = 0; i < v1.size1(); i++)
            {
                for (size_t j = 0; j < v2.size2(); j++)

                    res += -log(0.0000001 + fabs(v1(i, j) - (1 - v2(i, j)))); // Permet d'inverser le desiredOutput -> donc ajout du signe -
            }
            return res;
        };
    }
    //  quadratic error function
    ErrorFun
    coutGen()
    {
        return [](type_matrix v1, type_matrix v2)
        {
            float res = 0;
            for (size_t i = 0; i < v1.size1(); i++)
            {
                for (size_t j = 0; j < v2.size2(); j++)

                    res += -log(0.0000001 + v1(i, j)); // Permet d'inverser le desiredOutput -> donc ajout du signe -
            }
            return res;
        };
    }
    // Cross-Entropy Error Function

    ErrorFun
    genMinMax()
    {
        return [](type_matrix v1, type_matrix v2)
        {
            float res = 0;
            for (size_t i = 0; i < v1.size1(); i++)
            {

                for (size_t j = 0; j < v2.size2(); j++)
                {

                    res += log(1 - v1(i, j) + 0.0000001);
                }
            }
            return res;
        };
    }
    // Kullback-Leibler divergence (KL divergence)

    ErrorFun
    genKLDiv()
    {
        return [](type_matrix v1, type_matrix v2)
        {
            float res = 0;
            for (size_t i = 0; i < v1.size1(); i++)
                for (size_t j = 0; j < v2.size2(); j++)
                    res += -exp(-(1 / 1.0f) * log((1 / v1(i, j)) - 1));

            return res;
        };
    }

    zero_pad_layer::~zero_pad_layer()
    {
    }

    zero_pad_layer::zero_pad_layer(const size_t &input_size,
                                   const size_t &outputSize,
                                   const size_t &ZeroPadType) : _input_dim((size_t)sqrt(input_size)), _output_dim((size_t)sqrt(outputSize)), _zeropad_type(ZeroPadType)
    {
        if (input_size > outputSize)
            throw "Size not matching for ZeroPadding (Input must be smaller than Output) !";
        _propagation_matrix = type_matrix::Zero(input_size, outputSize);
        _backprop_matrix = type_matrix::Zero(outputSize, input_size);
        size_t outputDimension = sqrt(outputSize);
        if (ZeroPadType == 0) // CLassique
        {
            _zero_padding_tail = (((_output_dim)-_input_dim) / 2);
            for (size_t i(0); i < _input_dim; i++)
            {
                for (size_t  j(0); j < _input_dim; j++)
                {
                    _propagation_matrix(
                        i * (_input_dim) + j,
                        _zero_padding_tail * (outputDimension + 1) + outputDimension * (i) + j) = 1;
                    _backprop_matrix(
                        _zero_padding_tail * (outputDimension + 1) + outputDimension * (i) + j,
                        i * (_input_dim) + j) = 1;
                }
            }
        }
        else
        {
	
			
		//            if ((_output_dim - _input_dim) / (_input_dim + 1) != (size_t)(_output_dim - _input_dim) / (_input_dim + 1))
		//        throw std::runtime_error("Deconvolution size mismatch !");
            _zero_padding_tail = ((_output_dim - _input_dim) / (_input_dim + 1));
            for (size_t i(0); i < _input_dim; i++)
            {
                for (size_t  j(0); j < _input_dim; j++)
                {
                    _propagation_matrix(
                        i * (_input_dim) + j,
                        _zero_padding_tail * (_output_dim + 1) + (_zero_padding_tail + 1) * (_output_dim) * (i) + j * (_zero_padding_tail + 1)) = 1;
                    _backprop_matrix(
                        _zero_padding_tail * (_output_dim + 1) + (_zero_padding_tail + 1) * (_output_dim) * (i) + j * (_zero_padding_tail + 1),
                        i * (_input_dim) + j) = 1;
                }
            }
        }
    }
    // propagation
    type_matrix zero_pad_layer::process_layer(const type_matrix &inputs)

    {

        return inputs * _propagation_matrix;
    }

    type_matrix zero_pad_layer::layer_backpropagation(const type_matrix &xnPartialDerivative, float step)
    {
        return xnPartialDerivative * _backprop_matrix;
    }

    type_matrix zero_pad_layer::layer_backdrop_invariant(const type_matrix &xnPartialDerivative)
    {
        // return x(n-1)PartialDerivative
        return xnPartialDerivative * _backprop_matrix;
    }

    type_matrix zero_pad_layer::mini_batch_layer_backdrop(const type_matrix &xnPartialDerivative, float step)
    {
        return xnPartialDerivative * _backprop_matrix;
    }

    //****************AUTRES****************
    //**************************************

    void zero_pad_layer::update_layer_weights(size_t minibatchSize)
    {
        std::cout << "[=] WARNING, useless function being used " << std::endl;
    }

    type_matrix zero_pad_layer::fnDerivativeMatrix() const
    {

        type_matrix output = _backprop_matrix;

        // implement derivativeMatrix + diagonal array operator
        output.clear();
        return output;
    }

    void zero_pad_layer::reset()
    {
        this->_propagation_matrix.clear();
    }

    size_t zero_pad_layer::get_input_size()
    {
        return (static_cast<size_t>(_input_dim * _input_dim));
    }

    noisy_layer::noisy_layer(size_t inputSize, size_t outputSize, std::function<float(float)> activationF, size_t descentType)
        : fully_connected_layer(inputSize, outputSize, activationF, descentType), _noise_weights(type_matrix::Random(1, outputSize)), _noise_buffer(type_matrix::Zero(_noise_weights.rows(), 1)), _noise_variation_sum(type_matrix::Zero(1, outputSize))
    {
    }

    type_matrix noisy_layer::process_layer(const type_matrix &inputs)
    {
        _buffer_input = inputs;
        _noise_buffer = type_matrix::Random(_noise_weights.rows(), 1);
        _buffer_activation_level = (inputs * _weight) + (_noise_buffer * _noise_weights) - _bias;
        type_matrix output = _buffer_activation_level;

        for (unsigned int i(0); i < output.size2(); i++)
            output(0, i) = _activation_functions(output(0, i));

        return output;
    }

    type_matrix noisy_layer::layer_backpropagation(const type_matrix &xnPartialDerivative, float step)
    {
        type_matrix ynPartialDerivative = xnPartialDerivative * fnDerivativeMatrix();

        // update weights
        type_matrix wnPartialDerivative = (_buffer_input.transpose()) * ynPartialDerivative;
        type_matrix noisePartialDerivative = (_noise_buffer.transpose()) * ynPartialDerivative;
        _sum_bias_variation = _sum_bias_variation + (step * ynPartialDerivative);
        _sum_weight_variation = _sum_weight_variation + (step * wnPartialDerivative);
        _noise_variation_sum = _noise_variation_sum + (step * noisePartialDerivative);
        update_layer_weights();

        // return x(n-1)PartialDerivative
        return ynPartialDerivative * _weight.transpose();
    }

    void noisy_layer::update_layer_weights(size_t minibatchSize)
    {
        fully_connected_layer::update_layer_weights(minibatchSize);
        for (auto i : _noise_variation_sum)
            i = i / double(minibatchSize);

        _sum_weight_variation.set_zero();
    }

    fully_connected_layer::fully_connected_layer(size_t inputSize, size_t outputSize, std::function<float(float)> activationF, size_t descentType)
        : _weight(type_matrix::Random(inputSize, outputSize)), _bias(type_matrix::Random(1, outputSize)) // ligne
          ,
          _activation_functions(activationF), _buffer_activation_level(type_matrix::Zero(1, outputSize)) // ligne
          ,
          _buffer_input(type_matrix::Zero(1, inputSize)) // ligne
          ,
          _sum_weight_variation(type_matrix::Zero(inputSize, outputSize)), _sum_bias_variation(type_matrix::Zero(1, outputSize)), _adaptive_weight_step(type_matrix::Constant(inputSize, outputSize, 1)), _adaptive_bias_step(type_matrix::Constant(1, outputSize, 1)), _adaptive_weight_second_step(type_matrix::Constant(inputSize, outputSize, 1)), _adaptive_bias_second_step(type_matrix::Constant(1, outputSize, 1)), _descent_type(descentType), _update_count(0)
    {
    }

    fully_connected_layer::fully_connected_layer(size_t inputSize, size_t outputSize, type_matrix weight, type_matrix bias, std::function<float(float)> activationF, size_t descentType)
        : _weight(weight), _bias(bias) // ligne
          ,
          _activation_functions(activationF), _buffer_activation_level(type_matrix::Zero(1, outputSize)) // ligne
          ,
          _buffer_input(type_matrix::Zero(1, inputSize)) // ligne
          ,
          _sum_weight_variation(type_matrix::Zero(inputSize, outputSize)), _sum_bias_variation(type_matrix::Zero(1, outputSize)), _adaptive_weight_step(type_matrix::Constant(inputSize, outputSize, 1)), _adaptive_bias_step(type_matrix::Constant(1, outputSize, 1)), _adaptive_weight_second_step(type_matrix::Constant(inputSize, outputSize, 1)), _adaptive_bias_second_step(type_matrix::Constant(1, outputSize, 1)), _descent_type(descentType), _update_count(0)
    {
    }

    // fw propogation

    type_matrix fully_connected_layer::process_layer(const type_matrix &inputs)
    {
        _buffer_input = inputs;
        _buffer_activation_level = inputs * _weight - _bias;
        type_matrix output = _buffer_activation_level;

        // size-->size1
        for (unsigned int i(0); i < output.size1(); i++)
            output(0, i) = _activation_functions(output(0, i));

        return output;
    }
    // back propogation :

    type_matrix fully_connected_layer::layer_backpropagation(const type_matrix &xnPartialDerivative, float step)
    {

        type_matrix ynPartialDerivative = xnPartialDerivative * fnDerivativeMatrix();

        type_matrix wnPartialDerivative = (_buffer_input.transpose()) * ynPartialDerivative;
        switch (_descent_type)
        {
        case 1: // RMS Prop
            updateFirstMomentStep(wnPartialDerivative, ynPartialDerivative, step);
            _sum_bias_variation = _sum_bias_variation + (float(1.) / (_adaptive_bias_step.sqrt()) + (ynPartialDerivative * 0.000001));
            _sum_weight_variation = _sum_weight_variation + (float(1.) / (_adaptive_bias_step.sqrt()) + (ynPartialDerivative * 0.000001));

            // mSumBiasVariation += ((1.0/(sqrt(mAdaptativeBiasStep.array()+0.000001)))*ynPartialDerivative.array()).matrix();
            // mSumWeightVariation += ((1.0/(sqrt(mAdaptativeWeightStep.array()+0.000001)))*wnPartialDerivative.array()).matrix();
            update_layer_weights();
            break;
        case 2: // Adam
            _update_count++;
            updateFirstMomentStep(wnPartialDerivative, ynPartialDerivative, step);
            updateSecondMomentStep(wnPartialDerivative, ynPartialDerivative, step);
            _sum_bias_variation = _sum_bias_variation + (float(1.) / (_adaptive_bias_second_step.sqrt() * _adaptive_bias_step));
            _sum_weight_variation = _sum_weight_variation + (float(1.) / (_adaptive_weight_second_step.sqrt() * _adaptive_bias_step));
            /// mSumBiasVariation += ((1.0/(sqrt(((mAdaptativeBiasSecondStep).array())+0.000001)))*mAdaptativeBiasStep.array()).matrix();
            // mSumWeightVariation += ((1.0/(sqrt(((mAdaptativeWeightSecondStep).array())+0.000001)))*mAdaptativeWeightStep.array()).matrix();
            update_layer_weights();
            break;
        default:
            _sum_bias_variation = _sum_bias_variation + (step * ynPartialDerivative);
            _sum_weight_variation = _sum_weight_variation + (step * wnPartialDerivative);
            update_layer_weights();
            break;
        }
        // ret  x(n-1)PartialDerivative
        return ynPartialDerivative * _weight.transpose();
    }

    type_matrix fully_connected_layer::layer_backdrop_invariant(const type_matrix &xnPartialDerivative)
    {
        // Calcul de ynPartialDerivative
        type_matrix ynPartialDerivative = xnPartialDerivative * fnDerivativeMatrix();

        // return x(n-1)PartialDerivative

        return ynPartialDerivative * _weight.transpose();
    }

    type_matrix fully_connected_layer::mini_batch_layer_backdrop(const type_matrix &xnPartialDerivative, float step)
    {
        // Same as layerBackprop but no weight updating

        // Calcul de ynPartialDerivative
        type_matrix ynPartialDerivative = xnPartialDerivative * fnDerivativeMatrix();

        // Calcul de wnPartialDerivative et somation des erreurs
        type_matrix wnPartialDerivative = (_buffer_input.transpose()) * ynPartialDerivative;
        _sum_bias_variation = _sum_bias_variation + (step * ynPartialDerivative);
        _sum_weight_variation = _sum_weight_variation + (step * wnPartialDerivative);
        // Pas de mise à jour au sein de la backprop

        // return x(n-1)PartialDerivative
        return ynPartialDerivative * _weight.transpose();
    }

    //****************AUTRES****************
    //**************************************

    void fully_connected_layer::update_layer_weights(size_t minibatchSize)
    {
        _weight = _weight - _sum_weight_variation / double(minibatchSize);
        _bias = _bias + _sum_bias_variation / double(minibatchSize);

        // reset des buffer
        _sum_weight_variation.set_zero();
        _sum_bias_variation.set_zero();
    }

    type_matrix fully_connected_layer::fnDerivativeMatrix() const
    {
        auto fnDerivated = [this](float x, float dx)
        { return (_activation_functions(x + dx) - _activation_functions(x)) / dx; };
        //-->size-->size1
        type_matrix fnDerivativeMat(_buffer_activation_level.size1(), 1);
        for (size_t i(0); i < _buffer_activation_level.size1(); ++i)

            fnDerivativeMat(0, i) = fnDerivated(_buffer_activation_level(i, 0), real_type(0.05));
        fnDerivativeMat = fnDerivativeMat * fnDerivativeMat;
        return fnDerivativeMat;
        // return (fnDerivativeMat.as_diagonal()*fnDerivativeMat);
    }

    void fully_connected_layer::updateFirstMomentStep(const type_matrix &wnPartialDerivative, const type_matrix &ynPartialDerivative, float step)
    {
        _adaptive_weight_step = (step * _adaptive_weight_step) + (1 - step) * wnPartialDerivative;
        _adaptive_bias_step = (step * _adaptive_weight_step) + (1 - step) * wnPartialDerivative;

        // mAdaptativeWeightStep = step*mAdaptativeWeightStep + (1-step)*fabs(wnPartialDerivative);
        // mAdaptativeBiasStep = step*mAdaptativeBiasStep + (1-step)*fabs(ynPartialDerivative);
    }

    void fully_connected_layer::updateSecondMomentStep(const type_matrix &wnPartialDerivative, const type_matrix &ynPartialDerivative, float step)
    {

        _adaptive_weight_second_step = real_type(.99) * _adaptive_weight_step + (real_type(.01) * (wnPartialDerivative * wnPartialDerivative));
        // mAdaptativeWeightSecondStep = 0.99*mAdaptativeWeightStep + (1-0.99)*(wnPartialDerivative.array()*wnPartialDerivative.array());
        // mAdaptativeBiasSecondStep = 0.99*mAdaptativeBiasStep + (1-0.99)*(ynPartialDerivative.array()*ynPartialDerivative.array());
        _adaptive_bias_second_step = real_type(.99) * _adaptive_bias_step + (real_type(.01) * (ynPartialDerivative * ynPartialDerivative));
    }

    void fully_connected_layer::reset()
    {
        _weight = type_matrix::Random(_weight.rows(), _weight.cols());
        _bias = type_matrix::Random(1, _bias.cols());
        // mWeight = type_matrix::Random(mWeight.rows(), mWeight.cols());
        // mBias = type_matrix::Random(1,mBias.cols());
    }

    size_t fully_connected_layer::get_input_size()
    {
        return _weight.rows();
    }

    neural_net::neural_net() {}

    neural_net::neural_net(const std::vector<size_t> &layerTypes, const std::vector<size_t> &layerSizes, const std::vector<size_t> &layerChannels, const std::vector<std::vector<size_t>> &layerArg, const std::vector<ActivationFun> &activationFuns, size_t descentType)
    {
        if (layerSizes.size() != activationFuns.size() + 1)
            throw std::logic_error("NeuralNetwork::NeuralNetwork error - Sizes of parameters do not match");

        for (size_t i(0); i < layerSizes.size() - 1; ++i)
        {
            neuron_layer *new_layer = nullptr;
            new_layer = (layerTypes[i] == 0) ? (neuron_layer *)new fully_connected_layer(layerSizes[i], layerSizes[i + 1], activationFuns[i], descentType) : (layerTypes[i] == 2) ? (neuron_layer *)new convolution_layer(layerSizes[i], layerChannels[i], sqrt(layerSizes[i]) - sqrt(layerSizes[i + 1]) + 1, layerChannels[i + 1])
                                                                                                                                                         : (layerTypes[i] == 3)   ? (neuron_layer *)new noisy_layer(layerSizes[i], layerSizes[i + 1], activationFuns[i], descentType)
                                                                                                                                                         : (layerTypes[i] == 4)   ? (neuron_layer *)new zero_pad_layer(layerSizes[i], layerSizes[i + 1], layerArg[i][0])
                                                                                                                                                                                  : nullptr;

            if (new_layer)
                push_back(neuron_layer::ptr(new_layer));
        }
    }

    neural_net::neural_net(const std::vector<size_t> &layerSizes, const std::vector<matrix<float>> &weightVector, const std::vector<matrix<float>> &biasVector, const std::vector<ActivationFun> &activationFuns, size_t descentType)
    {

        if (layerSizes.size() != activationFuns.size() + 1)
            throw std::logic_error("NeuralNetwork::NeuralNetwork error - Sizes of parameters do not match");

        for (size_t i(0); i < layerSizes.size() - 1; ++i)
            // size_t tailleImg, size_t _Channels,
            //  		       const std::vector<type_matrix>& weight, std::function<float
            // 		       (float)> activationF = sigmoid (10.f)
            this->push_back(
                neuron_layer::ptr((layerSizes[i] == 0) ? (neuron_layer *)new fully_connected_layer(layerSizes[i], layerSizes[i + 1], weightVector[i], biasVector[i], activationFuns[i], descentType) : (layerSizes[i] == 2) ? (neuron_layer *)new convolution_layer(layerSizes[i], layerSizes[i + 1], weightVector, activationFuns[i])
                                                                                                                                                                                                                            : // @suppress("Ambiguous problem")
                                                                                                                                                                                                       (layerSizes[i] == 3) ? (neuron_layer *)new noisy_layer(layerSizes[i], layerSizes[i + 1], activationFuns[i], descentType)
                                                                                                                                                                                                   : (layerSizes[i] == 4)   ? (neuron_layer *)new zero_pad_layer(layerSizes[i], layerSizes[i + 1], 0)
                                                                                                                                                                                                                            : nullptr));
    }

    // convolution stuff:
    std::recursive_mutex convolution_operator::mtx;

    void convolution_operator::operator()(int id)
    {
        auto input = _input_matrix;
        size_t inputDimension = sqrt(input.cols());
        size_t filtreDimension = sqrt(_filter_matrix.cols());
        type_matrix filtreConv = type_matrix::Zero(input.cols(), (inputDimension - filtreDimension + 1) * (inputDimension - filtreDimension + 1));
        for (size_t k = 0; k < inputDimension - filtreDimension + 1; k++) // For each vertical offset of the filter

        {
            for (size_t l = 0; l < inputDimension - filtreDimension + 1; l++) // For each lateral offset of the filter...
            {
                for (size_t i = 0; i < filtreDimension; i++) // Filter's Line path...
                {
                    for (size_t j = 0; j < filtreDimension; j++) // traverse columns
                    {
                        filtreConv(l + j + (i + k) * inputDimension, l + k * (inputDimension - filtreDimension + 1)) = (_filter_matrix)(id, i * filtreDimension + j); // On crée une matrice de poids qui va nous permettre de lancer un calcul qui sera parallélisable
                    }
                }
            }
        }

        float *temp = type_matrix((filtreConv * input)).row(id);
        std::lock_guard<std::recursive_mutex> locker_(mtx);

        // mtx.lock();
        if (_sum_lines)
        {
            for (size_t col = 0; col < _result->cols(); ++col)
                _result->operator()(0, col) = _result->operator()(0, col) * (_result->operator()(0, col) + temp[col]);
        }
        else
        {
            for (size_t col = 0; col < _result->cols(); ++col)
            {

                (*_result).row(id)[col] = temp[col];
            }
        }
        // mtx.unlock();
    }

    convolution_layer::convolution_layer(size_t tailleImg, size_t nbChannels, size_t dimensionFiltre, size_t nbFiltres, std::function<float(float)> activationF)
        : _dimension_input((int)sqrt(tailleImg)), _weight_matrix(std::vector<type_matrix>()), mBias(type_matrix::Random(1, nbFiltres)) // ligne
          ,
          _activation_functions(activationF), _buffer_activation_level(type_matrix::Zero(nbFiltres, (_dimension_input - dimensionFiltre + 1) * (_dimension_input - dimensionFiltre + 1))) // ligne
          ,
          _buffer_input(type_matrix::Zero(nbChannels, tailleImg)) // ligne
          ,
          _sum_weight_variation(std::vector<type_matrix>()), _sum_bias_variation(type_matrix::Zero(1, nbFiltres)), _input_dimension(_dimension_input), _input_channels(nbChannels)
    {
        // weight matrices
        for (size_t i(0); i < nbFiltres; i++)
        {
            _weight_matrix.push_back(type_matrix::Random(nbChannels, dimensionFiltre * dimensionFiltre));
            _sum_weight_variation.push_back(type_matrix::Zero(nbChannels, dimensionFiltre * dimensionFiltre));
        }
    }

    convolution_layer::convolution_layer(size_t tailleImg, size_t nbChannels, const std::vector<type_matrix> &weight, std::function<float(float)> activationF)
        : _dimension_input((size_t)sqrt(tailleImg)), _weight_matrix(weight), mBias(type_matrix::Random(1, weight.size())) // ligne
          ,
          _activation_functions(activationF), _buffer_activation_level(type_matrix::Zero(weight.size(), (_dimension_input - (size_t)sqrt(weight[0].cols()) + 1) * (_dimension_input - (size_t)sqrt(weight[0].cols()) + 1))) // ligne
          ,
          _buffer_input(type_matrix::Zero(nbChannels, tailleImg)) // ligne
          ,
          _sum_weight_variation(std::vector<type_matrix>()), _sum_bias_variation(type_matrix::Zero(1, weight.size())), _input_dimension(_dimension_input), _input_channels(nbChannels)
    {
        // weight matrices
        for (size_t i(0); i < weight.size(); i++)
        {
            for (size_t j(0); j < weight[0].cols(); j++)
                _weight_matrix[i](0, j) = weight[i](0, j);
            _sum_weight_variation.push_back(type_matrix::Zero(nbChannels, weight[0].cols()));
        }
    }

    type_matrix convolution_layer::process_layer(const type_matrix &inputs)
    {
        _buffer_input = inputs;
        if (inputs.cols() != _input_dimension * _input_dimension)
            throw;
        for (size_t n = 0; n < _weight_matrix.size(); n++) // Pour chaque filtre...
        {
            auto temp = convolution_layer::convolution(inputs, _weight_matrix[n], true);
            for (size_t col = 0; col < _buffer_activation_level.cols(); ++col)
                _buffer_activation_level.row(n)[col] = temp(col, 0);
        }
        type_matrix output = _buffer_activation_level;

        for (size_t  i(0); i < output.cols(); i++)
            for (size_t j(0); j < output.rows(); j++)
                output(j, i) = _activation_functions(output(j, i));

        return output;
    }
    type_matrix convolution_layer::layer_backpropagation(const type_matrix &xnPartialDerivative, float step)
    {
        // calculate ynPartialDerivative
        type_matrix ynPartialDerivative = xnPartialDerivative * fnDerivativeMatrix();

        size_t  weightDimension = sqrt(_weight_matrix[0].cols());
        size_t ynDerivDimension = sqrt(ynPartialDerivative.cols());

        // Mise à jour des poids
        for (size_t i(0); i < _sum_weight_variation.size(); i++) // Pour chaque filtre...
        {
            // type_matrix ynPartialDerivativeCarree = type_matrix::One(mBufferInput.rows(),1)*ynPartialDerivative.row(i);
            // mSumWeightVariation[i] =  mSumWeightVariation[i] + ( step*convolution_layer::convolution(mBufferInput,ynPartialDerivativeCarree, false));
            _sum_weight_variation[i] = _sum_weight_variation[i] + (step * convolution_layer::convolution(_buffer_input, ynPartialDerivative, false));
        }
        // mSumBiasVariation += step*ynPartialDerivative;
        _sum_bias_variation = _sum_bias_variation + (step * ynPartialDerivative);

        // return the x(n-1) PartialDerivative
        size_t incrementDimension = 2 * (weightDimension - 1);
        size_t zeroPaddingDimension = incrementDimension + ynDerivDimension;

        type_matrix ynZeroPadding = type_matrix::Zero(ynPartialDerivative.rows(), zeroPaddingDimension * zeroPaddingDimension);
        for (size_t n = 0; n < ynPartialDerivative.rows(); n++) // Pour chaque channel...
        {
            for (size_t i = 0; i < ynDerivDimension; i++)
            {
                for (size_t  j = 0; j < ynDerivDimension; j++)
                {
                    ynZeroPadding(n, (i + incrementDimension / 2) * zeroPaddingDimension + j + incrementDimension / 2) = ynPartialDerivative(n, i * ynDerivDimension + j);
                }
            }
        }
        type_matrix resultat = type_matrix::Zero(_input_channels, _buffer_input.cols()); // input tail
        for (size_t i = 0; i < _weight_matrix.size(); i++)                               // for the channels
        {
            type_matrix ynZeroPaddingCarree = type_matrix::One(_weight_matrix[i].rows(), ynZeroPadding.rows()) * ynZeroPadding;
            auto reverse = _weight_matrix[i].reverse();
            auto result = reverse;

            provallo::convolution_operator o(ynZeroPaddingCarree, reverse, &result, false);

            // We sum all the errors of a line for all the channels
            o(i);
            resultat = resultat + type_matrix(o);
        }
        update_layer_weights();
        return resultat;
    }
    type_matrix convolution_layer::convolution(const type_matrix &input, const type_matrix &filtre, bool sommerLignes)
    {
        std::vector<std::thread> threads;
        size_t inputDimension = sqrt(input.cols());
        size_t filtreDimension = sqrt(filtre.cols());
        size_t  nbChannels = filtre.rows();
        if (sommerLignes)
            nbChannels = 1;
        type_matrix *resultat(new type_matrix(type_matrix::Zero(nbChannels, (inputDimension - filtreDimension + 1) * (inputDimension - filtreDimension + 1))));
        provallo::convolution_operator conv(input, filtre, resultat, sommerLignes);
        for (size_t n = 0; n < filtre.rows(); n++) // Pour chaque channel...
        {
            threads.push_back(std::thread(conv, n));
        }
        // std::cout << "synchronizing all threads...\n";
        for (auto &th : threads)
            th.join();
        return (*resultat);
    }

    std::ostream &operator<<(std::ostream &flux, neural_net network)
    {
        /*NeuralNetwork::iterator it;
        for (it = network.begin(); it != network.end(); it++)
        {
            flux << *it << "\n" << std::endl;
        }*/
        // log network
        std::cout << "[+] neural_net [" << ptrdiff_t(&network) << "]";

        for (std::list<std::unique_ptr<provallo::neuron_layer>>::iterator it = network.begin(); it != network.end(); ++it)
        {
            flux << it->get() << std::endl;
        }
        return flux;
    }

    // neural helper functions :

    neural_helper::neural_helper()
    {
    }

    neural_helper::neural_helper(neural_net::ptr generator, neural_net::ptr discriminator, size_t genFun)
        : _generator(std::move(generator)), _discriminator(std::move(discriminator)), _error_func_dis(coutDiscr())
    {
        switch (genFun)
        {
        case 0:
            _error_func_gen = coutGen();
            break;
        case 1:
            _error_func_gen = genMinMax();
            break;
        case 2:
            _error_func_gen = genKLDiv();
            break;
        default:
            _error_func_gen = coutGen();
            break;
        }
    }

    neural_helper::neural_helper(neural_net *generator, neural_net *discriminator, size_t genFun)
        : _generator(generator), _discriminator(discriminator), _error_func_dis(coutDiscr())
    {
        switch (genFun)
        {
        case 0:
            _error_func_gen = coutGen();
            break;
        case 1:
            _error_func_gen = genMinMax();
            break;
        case 2:
            _error_func_gen = genKLDiv();
            break;
        default:
            _error_func_gen = coutGen();
            break;
        }
    }

    void neural_helper::backpropDiscriminator(matrix<float> input, matrix<float> desiredOutput, float step, float dx)
    {
        matrix<float> xnPartialDerivative = calculateInitialErrorVector(_discriminator->processNetwork(input), desiredOutput, dx);

        propagateError(_discriminator, xnPartialDerivative, step);
    }

    void neural_helper::backpropGenerator(matrix<float> input, matrix<float> desiredOutput, float step, float dx)
    {
        matrix<float> xnPartialDerivative = calculateInitialErrorVectorGen(_discriminator->processNetwork(input), desiredOutput, dx);
        xnPartialDerivative = propagateErrorDiscriminatorInvariant(xnPartialDerivative);
        propagateError(_generator, xnPartialDerivative, step);
    }

    void neural_helper::minibatchDiscriminatorBackprop(neural_net::ptr network, matrix<float> input, matrix<float> desiredOutput, float step, float dx)
    // Same as backpropDiscriminator but no weight updating
    {
        matrix<float> xnPartialDerivative = calculateInitialErrorVector(_discriminator->processNetwork(input), desiredOutput, dx);

        propagateErrorMinibatch(network, xnPartialDerivative, step);
    }

    void neural_helper::minibatchGeneratorBackprop(neural_net::ptr network, matrix<float> input, matrix<float> desiredOutput, float step, float dx)
    // Same as backpropGenerator but no weight updating
    {
        matrix<float> xnPartialDerivative = calculateInitialErrorVectorGen(_discriminator->processNetwork(input), desiredOutput, dx);
        xnPartialDerivative = propagateErrorMinibatch(_discriminator, xnPartialDerivative, 0);
        propagateErrorMinibatch(network, xnPartialDerivative, step);
    }

    void neural_helper::updateNetworkWeights(neural_net::ptr network, size_t minibatchSize)
    {
        for (auto itr = network->rbegin(); itr != network->rend(); ++itr)
            (*itr)->update_layer_weights(minibatchSize);
    }

    void neural_helper::propagateError(neural_net::ptr network, matrix<float> xnPartialDerivative, float step)
    {
        for (auto itr = network->rbegin(); itr != network->rend(); ++itr)
        {
            xnPartialDerivative = (*itr)->layer_backpropagation(xnPartialDerivative, step);
        }
    }

    matrix<float> neural_helper::propagateErrorMinibatch(neural_net::ptr network, matrix<float> xnPartialDerivative, float step)
    {
        for (auto itr = network->rbegin(); itr != network->rend(); ++itr)
        {
            xnPartialDerivative = (*itr)->mini_batch_layer_backdrop(xnPartialDerivative, step);
        }
        return xnPartialDerivative;
    }

    //[[deprecated]] //use propagateErrorMinibatch(mDiscriminator, xnPartialDerivative, 0) instead
    matrix<float> neural_helper::propagateErrorDiscriminatorInvariant(matrix<float> xnPartialDerivative)
    {
        for (auto itr = _discriminator->rbegin(); itr != _discriminator->rend(); ++itr)
        {
            xnPartialDerivative = (*itr)->layer_backdrop_invariant(xnPartialDerivative);
        }
        return xnPartialDerivative;
    }

    matrix<float> neural_helper::calculateInitialErrorVectorGen(matrix<float> output, matrix<float> desiredOutput, float dx)
    {
        size_t output_size = output.size1() * output.size2();

        matrix<float> errorVect = matrix<float>::Zero(1, output_size);

        for (size_t i(0); i < output_size; ++i)
        {
            matrix<float> deltaX(matrix<float>::Zero(1, output_size));
            deltaX(0, i) = dx;
            errorVect(0, i) = std::min(100.f, (_error_func_gen(output + deltaX, desiredOutput) - _error_func_gen(output, desiredOutput)) / dx);
        }
        return errorVect;
    }

    matrix<float> neural_helper::calculateInitialErrorVector(matrix<float> output, matrix<float> desiredOutput, float dx)
    {
        size_t output_size = output.size1() * output.size2();
        matrix<float> errorVect = matrix<float>::Zero(1, output_size);

        for (size_t i(0); i < output_size; ++i)
        {
            matrix<float> deltaX(matrix<float>::Zero(1, output_size));
            deltaX(0, i) = dx;
            errorVect(0, i) = std::min(100.f, (_error_func_dis(output + deltaX, desiredOutput) - _error_func_dis(output, desiredOutput)) / dx);
        }
        return errorVect;
    }

    error_processor::error_processor()
        : mErrorsGen(),
          mErrorsDis()
    {
    }

    error_processor::error_statistics error_processor::processData() const
    {
        error_statistics data;

        // Calcul de la moyenne
        data.meanGen = std::accumulate(mErrorsGen.begin(), mErrorsGen.end(), 0.f) / (static_cast<float>(mErrorsGen.size()));
        data.meanDis = std::accumulate(mErrorsDis.begin(), mErrorsDis.end(), 0.f) / (static_cast<float>(mErrorsGen.size()));

        // Calcul d'écart type
        float deviationGen{0};
        float deviationDis{0};

        if (mErrorsGen.size() != 1)
        {
            std::for_each(mErrorsGen.begin(), mErrorsGen.end(), [&](float x)
                          { deviationGen += pow(x - data.meanGen, 2); });
            data.deviationGen = sqrt(deviationGen / static_cast<float>(mErrorsGen.size() - 1));
        }
        if (mErrorsDis.size() != 1)
        {
            std::for_each(mErrorsDis.begin(), mErrorsDis.end(), [&](float x)
                          { deviationDis += pow(x - data.meanDis, 2); });
            data.deviationDis = sqrt(deviationDis / static_cast<float>(mErrorsDis.size() - 1));
        }
        // Calcul d'interval de confiance
        data.confidenceRangeGen = 2 * data.deviationGen / (sqrt(static_cast<float>(mErrorsGen.size())));
        data.confidenceRangeDis = 2 * data.deviationDis / (sqrt(static_cast<float>(mErrorsDis.size())));

        return data;
    }

    void error_processor::addResultGen(float result)
    {
        mErrorsGen.push_back(result);
    }

    void error_processor::addResultDis(float result)
    {
        mErrorsDis.push_back(result);
    }

    source_processor::source_processor(const std::string &CSVFileNameRes, const std::string &CSVFileNameImg)
        : mCSVRes(CSVFileNameRes + ".csv"), mCSVImg(CSVFileNameImg + ".csv")
    {
        mCSVRes << "Teach index"
                << "MeanGen"
                << "MeanDis"
                << "DeviationGen"
                << "ConfidenceRangeGen"
                << "DeviationDis"
                << "ConfidenceRangeDis"
                << ""
                << "";
    }

    error_processor &source_processor::operator[](size_t teachIndex)
    {
        if (teachIndex > mErrorStats.size())
            throw std::logic_error("StatsCollector::operator[] - Erreur : Indice d'apprentissage trop grand");

        if (teachIndex == mErrorStats.size())
            mErrorStats.push_back(error_processor());

        return mErrorStats[teachIndex];
    }

    void source_processor::exportData(bool mustProcessData)
    {
        if (!mustProcessData)
            throw std::logic_error("Not implemented yet");

        for (size_t index{0}; index < mErrorStats.size(); ++index)
        {
            error_processor::error_statistics data(mErrorStats[index].processData());

            mCSVRes << index << data.meanGen << data.meanDis << data.deviationGen << data.confidenceRangeGen << data.deviationDis << data.confidenceRangeDis << endrow;
        }
    }

    void source_processor::exportImage(const matrix<float> &image, size_t teachIndex, size_t sizeSide)
    {
        mCSVImg << "#" << teachIndex << endrow;
        for (size_t j(0); j < sizeSide * sizeSide; j++)
        {
            mCSVImg << image(j, teachIndex);
            if (j % sizeSide == sizeSide - 1)
                mCSVImg << endrow;
        }
        mCSVImg << endrow;
    }

    csvfile *source_processor::getCSVFile()
    {
        return &mCSVRes;
    }

    std::ofstream &operator<<(std::ofstream &stream, const learning_task::task_configuration &_configuration)
    {
        /*
        const std::string separator = " : ";
        stream << "step" << separator << _configuration.step << endrow;
        stream << "dx" << separator << _configuration.dx << endrow;

        stream << "Experiments" << separator << _configuration._Experiments << endrow;
        stream << "LoopsPerExperiment" << separator << _configuration._LoopsPerExperiment << endrow;
        stream << "TeachingsPerLoop" << separator << _configuration._TeachingsPerLoop << endrow;
        stream << "DisTeach" << separator << _configuration._DisTeach << endrow;
        stream << "GenTeach" << separator << _configuration._GenTeach << endrow;
        stream << "DisTest" << separator << _configuration._DisTest << endrow;
        stream << "GenTest" << separator << _configuration._GenTest << endrow;
        stream << "labelTrainSize" << separator << _configuration.labelTrainSize << endrow;
        stream << "labelTestSize" << separator << _configuration.labelTestSize << endrow;
        stream << "ImgParIntervalleImg" << separator << _configuration.ImgParIntervalleImg << endrow;
        stream << "minibatchSize" << separator << _configuration.minibatchSize << endrow;
        stream << "genFunction" << separator << _configuration.genFunction << endrow;
        stream << "descent" << separator << _configuration.descent << endrow;
        stream << "descentTypeGen" << separator << _configuration.descentTypeGen << endrow;
        stream << "descentTypeDis" << separator << _configuration.descentTypeDis << endrow;
        stream << "descentType" << separator << _configuration.descentType << endrow;
        stream << "sigmoidParameter"    << separator << _configuration.sigmoidParameter << endrow;
        stream << "databaseToUse" << separator << _configuration.databaseToUse << endrow;
            */

        return stream;
    }
    std::ifstream &operator>>(std::ifstream &stream, learning_task::task_configuration &_configuration)
    {
        /*                const std::string separator = " : ";

                 //load source processor configuration
                stream >>std::string("step") >> separator >> _configuration.step >> endrow;
                stream >>std::string("dx") >> separator >> _configuration.dx>> endrow;
                stream >> "Experiments" >> separator >> _configuration._Experiments >> endrow;
                stream >> "LoopsPerExperiment" >> separator >> _configuration._LoopsPerExperiment >> endrow;
                stream >> "TeachingsPerLoop" >> separator >> _configuration._TeachingsPerLoop >> endrow;
                stream >> "DisTeach" >> separator >> _DisTeach >> endrow;
                stream >> "GenTeach" >> separator >> _GenTeach >> endrow;
                stream >> "DisTest" >> separator >> _DisTest >> endrow;
                stream >> "GenTest" >> separator >> _GenTest >> endrow;
                stream >> "labelTrainSize" >> separator >> _configuration.labelTrainSize >> endrow;
                stream >> "labelTestSize" >> separator >> _configuration.labelTestSize >> endrow;
                stream >> "intervalleImg" >> separator >> _configuration.intervalleImg >> endrow;
                stream >> "ImgParIntervalleImg" >> separator >> _configuration._ImgParIntervalleImg >> endrow;
                stream >> "minibatchSize" >> separator >> _configuration.minibatchSize >> endrow;
                stream >> "genFunction" >> separator >> _configuration.genFunction >> endrow;
                stream >> "descent" >> separator >> _configuration.descent >> endrow;
                stream >> "descentTypeGen" >> separator >> _configuration.descentTypeGen >> endrow;
                stream >> "descentTypeDis" >> separator >> _configuration.descentTypeDis >> endrow;
                stream >> "descentType" >> separator >> _configuration.descentType >> endrow;
                stream >>std::string( "sigmoidParameter" )>> separator >> _configuration.sigmoidParameter >> endrow;
                stream >> std::string("databaseToUse") >> separator >> _configuration.databaseToUse >> endrow;

         */

        return stream;
    }

    learning_task::learning_task() : _configuration()
    {
        // Charge la configuration de l'application
        // loadConfig();
        /// A réparer, cette fonctionnalité est pétée (et n'a jamais marché en fait)
        /// _sourceprocessor : _sourceprocessor(mConfig.CSVFileNameResult,mConfig.CSVFileNameImage);
        //*(_statscollector.getCSVFile()) << "Step" << _configuration.step << "dx" << _configuration.dx << endrow;
        (*mSourceProcessor.getCSVFile()) << "Step" << _configuration.step << "dx" << _configuration.dx << endrow;

        try
        {
            if (_configuration.databaseToUse == "mnist")
            {
                sample_source::Ptr inputProvider(new MnistProvider(_configuration.chiffresATracer, 6000, 1000));
                mTeachingBatchDis = inputProvider->trainingBatch();
                mTestingBatchDis = inputProvider->testingBatch();
            }
            else if (_configuration.databaseToUse == "cifar10")
            {
                // Cifar10Provider::CifarLabel CifVehicle =    Cifar10Provider::CifarLabel::airplane | Cifar10Provider::CifarLabel::automobile | Cifar10Provider::CifarLabel::ship | Cifar10Provider::CifarLabel::truck;
                cifar10_source::Utype cifs_animals(
                    cifar10_source::Utype(cifar10_source::CifarLabel::bird) | cifar10_source::Utype(cifar10_source::CifarLabel::cat) |
                    cifar10_source::Utype(cifar10_source::CifarLabel::deer) | cifar10_source::Utype(cifar10_source::CifarLabel::dog) | cifar10_source::Utype(cifar10_source::CifarLabel::horse) | cifar10_source::Utype(cifar10_source::CifarLabel::frog));

                cifar10_source::CifarLabel CifAnimals = cifar10_source::CifarLabel(cifs_animals);

                // Cifar10Provider::CifarLabel CifAll = CifAnimals | CifVehicle;
                sample_source::Ptr inputProvider(new cifar10_source(CifAnimals, 10000, 10000));
                mTeachingBatchDis = inputProvider->trainingBatch(1);
                mTestingBatchDis = inputProvider->testingBatch(1);
            }
            else
            {
                std::cout << "Application::Application error : databaseToUse is unknown (" << stderr << ")" << std::endl;
                exit(EXIT_FAILURE);
            }

            // Création du vecteur de bruit pour les tests du générateur
            std::vector<matrix<float>> vectorTest;
            for (size_t i(0); i < _configuration._GenTest; i++)
            {
                matrix<float> noise = matrix<float>::Random(_configuration.genLayerNbChannels[0], _configuration.genLayerSizes[0]);
                vectorTest.push_back(noise);
            }

            // Création du Batch de Test du générateur
            for (size_t i(0); i < _configuration._GenTest; i++)
            {
                matrix<float> outputTest = matrix<float>::Zero(1, 1);
                outputTest(0, 0) = 0.;
                mTestingBatchGen.push_back(learning_task::Sample(vectorTest[i], outputTest));
            }
            std::cout << "Chargement du Batch de test effectué !" << std::endl;

            if (_configuration.networkAreImported)
            {
                /*mDiscriminator = NeuralNetwork::Ptr(importNeuralNetwork(mConfig.discriminatorPath,Functions::sigmoid(mConfig.sigmoidParameter)));
             std::cout << "Chargement du Discriminateur effectué !" << std::endl;

             mGenerator = NeuralNetwork::Ptr(importNeuralNetwork(mConfig.generatorPath,Functions::sigmoid(mConfig.sigmoidParameter)));
                std::cout << "Chargement du Générateur effectué !" << std::endl;*/
            }
            else
            {
                // Construction du réseau de neurones
                // Le Generateur
                std::vector<ActivationFun> funsGen;
                for (size_t i(0); i < _configuration.genLayerSizes.size() - 1; i++)
                {
                    if (i == _configuration.genLayerSizes.size() - 2)
                        funsGen.push_back(sigmoid(0.1f));
                    else
                        funsGen.push_back(sigmoid(0.1f));
                }
                mGenerator = (new neural_net(_configuration.genLayerTypes, _configuration.genLayerSizes, _configuration.genLayerNbChannels, _configuration.genLayerArgs, funsGen, _configuration.descentTypeGen));
                // Le Discriminateur
                std::vector<ActivationFun> funsDis;
                for (size_t i(0); i < _configuration.disLayerSizes.size() - 1; i++)
                    funsDis.push_back(sigmoid(0.1f));
                mDiscriminator = (new neural_net(_configuration.disLayerTypes, _configuration.disLayerSizes, _configuration.disLayerNbChannels, _configuration.disLayerArgs, funsDis, _configuration.descentTypeDis));
            }
            mTeacher = neural_helper(mGenerator, mDiscriminator, _configuration.genFunction);
        }
        catch (const std::exception &ex)
        {
            std::cout << "Exception was thrown: " << ex.what() << std::endl;
        }
    }
    learning_task::learning_task(const std::string &configFile)
    {
        std::ifstream file(configFile);

        if (!file.is_open())
        {
            std::cout << "Application::Application error : configFile can't be opened (" << stderr << ")" << std::endl;
            exit(EXIT_FAILURE);
        }
        // read configuration from   file
        file >> _configuration;
        file.close();
        // run experiments

        runExperiments();
    }
    //**************EXPERIENCES*************
    //**************************************

    void learning_task::runExperiments()
    {
        for (unsigned int index{0}; index < _configuration._Experiments; ++index)
        {

            if (!_configuration.networkAreImported)
            {
                resetExperiment();
                std::cout << "Réseau réinitialisé !" << std::endl;
            }

            if (_configuration.typeOfExperiment == "stochastic")
            {
                runSingleStochasticExperiment();
            }
            else if (_configuration.typeOfExperiment == "minibatch")
            {
                runSingleMinibatchExperiment();
            }
            else
            {
                std::cout << "Application::runExperiments error : typeOfExperiment is unknown (" << stderr << ")" << std::endl;
                exit(EXIT_FAILURE);
            }
            export_weights();
            std::cout << "Exp num. " << (index + 1) << " finie !" << std::endl;
        }

        mSourceProcessor.exportData(true);
    }

    void learning_task::runSingleStochasticExperiment()
    {
        mSourceProcessor[0].addResultGen(runTestGen());

        for (unsigned int loopIndex{0}; loopIndex < _configuration._LoopsPerExperiment; ++loopIndex)
        {
            std::cout << "Apprentissage num. : " << (loopIndex)*_configuration._TeachingsPerLoop << std::endl;
            runStochasticTeach();
            auto scoreGen = runTestGen();
            auto scoreDis = runTestDis(_configuration._DisTest);

            mSourceProcessor[loopIndex + 1].addResultGen(scoreGen);
            mSourceProcessor[loopIndex + 1].addResultDis(scoreDis);
            std::cout << "Le scoreGen est de " << scoreGen << " et le scoreDis de " << scoreDis << " !" << std::endl;
            // Création Image
            if (loopIndex % _configuration.intervalleImg == 0)
            {
                matrix<float> input;
                for (unsigned int i(0); i < _configuration._ImgParIntervalleImg; i++)
                {
                    input = matrix<float>::Random(_configuration.genLayerNbChannels[0], _configuration.genLayerSizes[0]);
                    mSourceProcessor.exportImage(mGenerator->processNetwork(input), loopIndex * _configuration._TeachingsPerLoop, _configuration.imageSizeSide);
                }
            }
        }
    }

    void learning_task::runSingleMinibatchExperiment()
    {
        mSourceProcessor[0].addResultGen(runTestGen());
        for (unsigned int loopIndex{0}; loopIndex < _configuration._LoopsPerExperiment; ++loopIndex)
        {
            std::cout << "learning # : " << (loopIndex)*_configuration._TeachingsPerLoop << std::endl;
            runMinibatchTeach();
            auto scoreGen = runTestGen();
            auto scoreDis = runTestDis(_configuration._DisTest);
            mSourceProcessor[loopIndex + 1].addResultGen(scoreGen);
            mSourceProcessor[loopIndex + 1].addResultDis(scoreDis);
            std::cout << "[#]Generator score " << scoreGen << "  # Discirminator score " << scoreDis << " !" << std::endl;
            if (loopIndex % _configuration.intervalleImg == 0)
            {
                matrix<float> input;
                for (unsigned int i(0); i < _configuration._ImgParIntervalleImg; i++)
                {
                    input = matrix<float>::Random(_configuration.genLayerNbChannels[0], _configuration.genLayerSizes[0]);
                    mSourceProcessor.exportImage(mGenerator->processNetwork(input), loopIndex * _configuration._TeachingsPerLoop, _configuration.imageSizeSide);
                }
            }
        }
    }

    void learning_task::resetExperiment()
    {
        mGenerator->reset();
        mDiscriminator->reset();
    }

    //************APPRENTISSAGE*************
    //**************************************

    void learning_task::runStochasticTeach()
    {
        std::uniform_int_distribution<> distribution(0, static_cast<int>(mTeachingBatchDis.size()) - 1);
        std::mt19937 randomEngine((std::random_device())());

        for (size_t index{0}; index < _configuration._TeachingsPerLoop; index++)
        {
            matrix<float> noiseInput = matrix<float>::Random(_configuration.genLayerNbChannels[0], _configuration.genLayerSizes[0]);
            matrix<float> desiredOutput = matrix<float>(1, 1);

            for (size_t i(0); i < _configuration._GenTeach; i++)
            {

                matrix<float> noiseInput = matrix<float>::Random(_configuration.genLayerNbChannels[0], _configuration.genLayerSizes[0]);
                matrix<float> input = mGenerator->processNetwork(noiseInput);
                noiseInput = matrix<float>::Random(_configuration.genLayerNbChannels[0], _configuration.genLayerSizes[0]);
                input = mGenerator->processNetwork(noiseInput);

                desiredOutput(0, 0) = 1;
                mTeacher.backpropGenerator(input, desiredOutput, _configuration.step, _configuration.dx);
            }
            for (size_t i(0); i < _configuration._DisTeach; i++)
            {
                noiseInput = matrix<float>::Random(_configuration.genLayerNbChannels[0], _configuration.genLayerSizes[0]);
                Sample sample(mTeachingBatchDis[distribution(randomEngine)]);
                mTeacher.backpropDiscriminator(sample.first, sample.second, _configuration.step, _configuration.dx);

                matrix<float> input = mGenerator->processNetwork(noiseInput);
                desiredOutput(0, 0) = 0;
                mTeacher.backpropDiscriminator(input, desiredOutput, _configuration.step, _configuration.dx);
            }
        }
    }

    void learning_task::runMinibatchTeach()
    {
        unsigned int minibatchWeightingCoefficient = _configuration.useAverageForBatchlearning ? _configuration.minibatchSize : 1;

        for (unsigned int index{0}; index < _configuration._TeachingsPerLoop; index++)
        {
            matrix<float> desiredOutput0 = matrix<float>(1, 1);
            matrix<float> desiredOutput1 = matrix<float>(1, 1);
            desiredOutput0(0, 0) = 0;
            desiredOutput1(0, 0) = 1;
            for (unsigned long k(0); k < 1; ++k)
            {
                Minibatch generatedImagesFromNoiseMinibatch = sampleGeneratedImagesFromNoiseMinibatch(); //"Sample minibatch of batchSize noise samples {z_1, ..., z_m} from noise prior p_g(z)"
                Minibatch exampleMinibatch = sampleMinibatch(mTeachingBatchDis);                         //"Sample minibatch of batchSize examples {x_1, ..., x_m} from data-generating distribution p_data(x)

                //"Update the discriminator by ascending its stochastic gradient"
                for (unsigned long i(0); i < _configuration.minibatchSize; ++i)
                {
                    Sample falseimagesample(generatedImagesFromNoiseMinibatch[i]);
                    Sample trueimagesample(exampleMinibatch[i]);
                    mTeacher.minibatchDiscriminatorBackprop(std::shared_ptr<neural_net>(mDiscriminator), falseimagesample.first, desiredOutput0, _configuration.step, _configuration.dx);
                    mTeacher.minibatchDiscriminatorBackprop(std::shared_ptr<neural_net>(mDiscriminator), trueimagesample.first, desiredOutput1, _configuration.step, _configuration.dx);
                }
                mTeacher.updateNetworkWeights(std::shared_ptr<neural_net>(mDiscriminator), minibatchWeightingCoefficient);
            }
            Minibatch generatedImagesFromNoiseMinibatch = sampleGeneratedImagesFromNoiseMinibatch(); //"Sample minibatch of batchSize noise samples {z_1, ..., z_m} from noise prior p_g(z)"

            for (std::vector<Sample>::iterator itr = generatedImagesFromNoiseMinibatch.begin(); itr != generatedImagesFromNoiseMinibatch.end(); ++itr) //"Update the generator by descending the stochastic gradient"
            {
                Sample sample(*itr);
                mTeacher.minibatchGeneratorBackprop(std::shared_ptr<neural_net>(mGenerator), sample.first, desiredOutput1, _configuration.step, _configuration.dx);
            }
            mTeacher.updateNetworkWeights(std::shared_ptr<neural_net>(mGenerator), minibatchWeightingCoefficient);
        }
    }

    float learning_task::runTestGen(int limit, bool returnErrorRate)
    {
        float errorMean{0};
        if (returnErrorRate)
        {
            for (std::vector<Sample>::iterator itr = mTestingBatchGen.begin(); itr != mTestingBatchGen.end() && limit-- != 0; ++itr)
            {
                matrix<float> output(mDiscriminator->processNetwork(mGenerator->processNetwork(itr->first)));
                errorMean += sqrt((output - itr->second).squaredNorm());
            }
        }
        return errorMean / static_cast<float>(mTestingBatchGen.size());
    }

    float learning_task::runTestDis(int limit, bool returnErrorRate)
    {
        int i(limit);
        float errorMean{0};
        if (returnErrorRate)
        {
            for (std::vector<Sample>::iterator itr = mTestingBatchDis.begin(); itr != mTestingBatchDis.end() && i-- != 0; ++itr)
            {
                matrix<float> output(mDiscriminator->processNetwork(itr->first));
                errorMean += sqrt((output).squaredNorm());
            }
        }
        return errorMean / static_cast<float>(std::min((int)mTestingBatchDis.size(), limit));
    }

#if 0
   [[deprecated]]
   float learning_task::gameScore(size_t nbImages)
   {
   	float mean = 0;
   	for (int i(0); i < nbImages; i++)
   	{
   		mean += (mDiscriminator->processNetwork(mGenerator->processNetwork(    matrix<float>::Random(1, mGenerator->get_input_size()))))(0);
   	}
   	return(mean/(float)nbImages);
   }
   [[deprecated]]
    matrix<float> learning_task::genProcessing(    matrix<float> input)
   {
   	return(mGenerator->processNetwork(input));
   }

#endif

    learning_task::Minibatch learning_task::sampleMinibatch(learning_task::Batch batch)
    {
        learning_task::Minibatch minibatch(_configuration.minibatchSize);

        // Tirage aléatoire sans remise
        std::vector<size_t> randomizedIntVector(batch.size());
        std::iota(randomizedIntVector.begin(), randomizedIntVector.end(), 0); // fills in with first int numbers starting at 0
        std::random_shuffle(randomizedIntVector.begin(), randomizedIntVector.end());

        for (unsigned long i(0); i < _configuration.minibatchSize; ++i)
        {
            minibatch[i] = batch[randomizedIntVector[i]];
        }
        return minibatch;
    }

    learning_task::Minibatch learning_task::sampleGeneratedImagesFromNoiseMinibatch()
    {
        learning_task::Minibatch generatedImagesFromNoiseMinibatch(_configuration.minibatchSize);

        matrix<float> desiredOutput(1, 1);
        desiredOutput(0, 0) = 1;

        for (unsigned long i(0); i < _configuration.minibatchSize; ++i)
        {
            matrix<float> noiseInput = matrix<float>::Random(1, mGenerator->get_input_size());
            matrix<float> input = mGenerator->processNetwork(noiseInput);

            Sample imageSample = std::make_pair(input, desiredOutput);

            generatedImagesFromNoiseMinibatch[i] = imageSample;
        }
        return generatedImagesFromNoiseMinibatch;
    }

    //************CONFIGURATION*************
    //**************************************
#ifdef RAPIDJSON
    void learning_task::loadConfig(const std::string &configFileName)
    {
        std::stringstream ss;
        std::ifstream inputStream(configFileName);
        if (!inputStream)
        {
            throw std::runtime_error("Application::loadConfig Error - Failed to load " + configFileName);
        }
        ss << inputStream.rdbuf();
        inputStream.close();
        rapidjson::Document doc;
        rapidjson::ParseResult ok(doc.Parse(ss.str().c_str()));
        if (!ok)
        {
            std::cout << stderr << "JSON parse error: %s (%u)" << rapidjson::GetParseError_En(ok.Code()) << ok.Offset() << std::endl;
            exit(EXIT_FAILURE);
        }

        setConfig(doc);
    }

    void learning_task::setConfig(rapidjson::Document &document)
    {
        _configuration.step = document["step"].GetFloat();
        _configuration.dx = document["dx"].GetFloat();
        _configuration.sigmoidParameter = document["sigmoidParameter"].GetFloat();

        _configuration.networkAreImported = document["networkAreImported"].GetBool();

        /*    auto layersSizesDis = document["layersSizesDis"].GetArray();
            for(rapidjson::SizeType i = 0; i < layersSizesDis.Size(); i++)
                mConfig.disLayerSizes.push_back(layersSizesDis[i].GetUint());

            auto layersSizesGen = document["layersSizesGen"].GetArray();
            for(rapidjson::SizeType i = 0; i < layersSizesGen.Size(); i++)
                mConfig.genLayerSizes.push_back(layersSizesGen[i].GetUint());*/

        _configuration.databaseToUse = document["databaseToUse"].GetString();
        if (_configuration.databaseToUse == "mnist")
            _configuration.imageSizeSide = 28;
        else if (_configuration.databaseToUse == "cifar10")
            _configuration.imageSizeSide = 32;

        auto chiffresATracer = document["chiffreATracer"].GetArray();
        for (rapidjson::SizeType i(0); i < chiffresATracer.Size(); i++)
            _configuration.chiffresATracer.push_back(chiffresATracer[i].GetUint());

        auto classesCifar = document["classesCifar"].GetArray();
        for (rapidjson::SizeType i(0); i < classesCifar.Size(); i++)
            _configuration.classesCifar.push_back(classesCifar[i].GetString());

        /*    auto layersTypesDis = document["layersTypesDis"].GetArray();
            for(rapidjson::SizeType i = 0; i < layersTypesDis.Size(); i++)
                mConfig.disLayerTypes.push_back(layersTypesDis[i].GetUint());

            auto layersTypesGen = document["layersTypesGen"].GetArray();
            for(rapidjson::SizeType i = 0; i < layersTypesGen.Size(); i++)
                mConfig.genLayerTypes.push_back(layersTypesGen[i].GetUint());*/

        auto chiffreATracer = document["chiffreATracer"].GetArray();
        for (rapidjson::SizeType i = 0; i < chiffreATracer.Size(); i++)
            _configuration.chiffresATracer.push_back(chiffresATracer[i].GetUint());

        auto layersDis = document["layersDis"].GetArray();
        for (int i(0); layersDis.Size() > i; i++)
        {
            _configuration.disLayerTypes.push_back(layersDis[i]["layerType"].GetUint());
            _configuration.disLayerSizes.push_back(layersDis[i]["inputSize"].GetUint());
            _configuration.disLayerNbChannels.push_back(layersDis[i]["inputChannels"].GetUint());
            _configuration.disLayerArgs.push_back(std::vector<unsigned int>());
            for (int j(0); (layersDis[i]["arguments"].GetArray()).Size() > j; j++)
            {
                _configuration.disLayerArgs[i].push_back(((layersDis[i].GetObject())["arguments"].GetArray())[j].GetUint());
            }
        }

        auto layersGen = document["layersGen"].GetArray();
        for (int i(0); layersGen.Size() > i; i++)
        {
            _configuration.genLayerTypes.push_back(layersGen[i]["layerType"].GetUint());
            _configuration.genLayerSizes.push_back(layersGen[i]["inputSize"].GetUint());
            _configuration.genLayerNbChannels.push_back(layersGen[i]["inputChannels"].GetUint());
            _configuration.genLayerArgs.push_back(std::vector<unsigned int>());
            for (int j(0); (layersGen[i]["arguments"].GetArray()).Size() > j; j++)
            {
                _configuration.genLayerArgs[i].push_back(((layersGen[i].GetObject())["arguments"].GetArray())[j].GetUint());
            }
        }

        _configuration.nbExperiments = document["nbExperiments"].GetUint();
        _configuration.nbLoopsPerExperiment = document["nbLoopsPerExperiment"].GetUint();
        _configuration.nbTeachingsPerLoop = document["nbTeachingsPerLoop"].GetUint();
        _configuration.nbDisTeach = document["nbDisTeach"].GetUint();
        _configuration.nbGenTeach = document["nbGenTeach"].GetUint();
        _configuration.nbDisTest = document["nbDisTest"].GetUint();
        _configuration.nbGenTest = document["nbGenTest"].GetUint();
        _configuration.labelTrainSize = document["labelTrainSize"].GetUint();
        _configuration.labelTestSize = document["labelTestSize"].GetUint();
        _configuration.intervalleImg = document["intervalleImg"].GetUint();
        _configuration.nbImgParIntervalleImg = document["nbImgParIntervalleImg"].GetUint();

        _configuration.minibatchSize = document["minibatchSize"].GetUint();
        _configuration.genFunction = document["genFunction"].GetUint();

        _configuration.descentTypeGen = document["descentTypeGen"].GetUint();
        _configuration.descentTypeDis = document["descentTypeDis"].GetUint();

        _configuration.generatorPath = document["generatorPath"].GetString();
        _configuration.discriminatorPath = document["discriminatorPath"].GetString();

        _configuration.generatorDest = document["generatorDest"].GetString();
        _configuration.discriminatorDest = document["discriminatorDest"].GetString();
        _configuration.CSVFileNameResult = document["CSVFileNameResult"].GetString();
        _configuration.CSVFileNameImage = document["CSVFileNameImage"].GetString();

        _configuration.typeOfExperiment = document["typeOfExperiment"].GetString();
        _configuration.useAverageForBatchlearning = document["useAverageForBatchlearning"].GetBool();
    }
#endif

    void learning_task::export_weights()
    {
        csvfile csvGen(_configuration.generatorDest);
        for (unsigned int i(0); i < _configuration.genLayerSizes.size(); i++)
            csvGen << _configuration.genLayerSizes[i];
        csvGen << endrow;
        // csvGen << *mGenerator;

        csvfile csvDis(_configuration.discriminatorDest);
        for (unsigned int i(0); i < _configuration.disLayerSizes.size(); i++)
            csvDis << _configuration.disLayerSizes[i];
        csvDis << endrow;
        // csvDis << *mDiscriminator;
        std::cout << "weights were exported !" << std::endl;
    }

    neural_net *learning_task::deserialize_neural_network(std::string networkPath, ActivationFun activationFun)
    {
        std::ifstream ifs(networkPath);
        std::string a;
        // data
        std::vector<matrix<float>> neuralNetwork;
        std::vector<matrix<float>> bias;
        // type
        std::vector<size_t> taille;
        neural_net *ret = nullptr;
        if (ifs.good())
        {
            unsigned int k = 0;
            getline(ifs, a, '\n');
            std::string b = "";
            for (unsigned int i(0); i < a.length(); i++)
            {
                if (a[i] == ';')
                {
                    if (b != "")

                    {
                        taille.push_back(stoi(b));
                        b = "";
                    }
                }
                else
                    b = b + a[i];
            }

            for (size_t i(0); i < taille.size() - 1; i++)
            {
                neuralNetwork.push_back(matrix<float>::Zero(taille[i], taille[i + 1]));
                bias.push_back(matrix<float>::Zero(1, taille[i + 1]));
            }
            std::vector<ActivationFun> activationFunVector;
            for (size_t i(0); i < taille.size() - 1; i++)
            {
                activationFunVector.push_back(activationFun);
            }
            size_t i = 0;
            size_t j = 0;
            while (k < taille.size() - 1)
            {
                std::getline(ifs, a, '\n');
                if (a != "")
                {
                    for (auto itr = a.begin(); itr != a.end(); itr++)
                    {
                        if (*itr == ';')
                        {
                            if (b != "")
                            {
                                neuralNetwork[k](j, i) = (std::stof(b));
                                i++;
                                b = "";
                            }
                        }
                        else
                            b = b + *itr;
                    }
                    j++;
                    i = 0;
                }
                else
                {
                    j = 0;
                    std::getline(ifs, a, '\n');
                    for (auto itr = a.begin(); itr != a.end(); itr++)
                    {
                        if (*itr == ';')
                        {
                            if (b != "")
                            {
                                bias[k](0, j) = (std::stof(b));
                                j++;
                                b = "";
                            }
                        }
                        else
                            b = b + *itr;
                    }
                    j = 0;
                    std::getline(ifs, a, '\n');
                    std::getline(ifs, a, '\n');
                    k = k + 1;
                }
            }
            ifs.close();

            ret = new neural_net(taille, neuralNetwork, bias, activationFunVector);
        }
        return ret;
    }

    cifar10_source::cifar10_source(CifarLabel labels, size_t labelTrainSize, size_t labelTestSize)
        : sample_source(labelTrainSize, labelTestSize), mDataset(cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>()), mLabels(labels)
    {
        if (mLabelTrainSize > 50000)
            throw std::logic_error("Erreur : dépassement d'indice sur le batch de train");
        if (mLabelTestSize > 10000)
            throw std::logic_error("Erreur : dépassement d'indice sur le batch de test");
    }

    sample_source::Batch cifar10_source::trainingBatch(bool greyLevel) const
    {
        std::cout << "Création du Batch d'entrainement Cifar10 du discriminateur" << std::endl;

        Batch trainingBatch;

        matrix<float> outputTrain = matrix<float>::Zero(1, 1);
        outputTrain(0, 0) = 1.0;

        // Compteur permet de compter le nombre d'images dans le batch, pour ne pas dépasser mLabelTrainSize
        unsigned int compteur(0);
        for (unsigned int i(0); i < 50000 && compteur < mLabelTrainSize; i++)
        {
            if (cifar10_source::matchLabelWithId(mLabels, mDataset.training_labels[i]))
            {
                trainingBatch.push_back(Sample(getMatrix(i, true, greyLevel), outputTrain));
                compteur++;
            }
        }

        std::cout << "Chargement du Batch d'entrainement Cifar10 effectué !" << std::endl;
        return trainingBatch;
    }

    sample_source::Batch cifar10_source::testingBatch(bool greyLevel) const
    {

        std::cout << "Création du Batch de test Cifar10 du discriminateur" << std::endl;

        Batch testBatch;

        matrix<float> outputTest = matrix<float>::Zero(1, 1);
        outputTest(0, 0) = 1.0;

        // Compteur permet de compter le nombre d'images dans le batch, pour ne pas dépasser mLabelTestSize
        unsigned int compteur(0);
        for (unsigned int i(0); i < 50000 && compteur < mLabelTestSize; i++)
        {
            if (cifar10_source::matchLabelWithId(mLabels, mDataset.test_labels[i]))
            {
                testBatch.push_back(Sample(getMatrix(i, true, greyLevel), outputTest));
                compteur++;
            }
        }

        std::cout << "Chargement du Batch de test Cifar10 effectué !" << std::endl;

        return testBatch;
    }

    matrix<float> cifar10_source::getMatrix(size_t index, bool isTrainOrTestRequired, bool greyLevel) const
    {
        // Le 3072 est hardcodé car c'est la taille d'une image cifar (32 * 32 = 1024 (nombre de pixels) et 1024 * 3 = 3072 (3 couleurs))
        unsigned int cifarSize;
        auto dSet = isTrainOrTestRequired ? &mDataset.training_images : &mDataset.test_images;

        if (greyLevel)
        {
            cifarSize = 1024;
            matrix<float> mat = matrix<float>::Zero(1, cifarSize);
            for (unsigned int pixel(0); pixel < cifarSize; pixel++)
            {
                mat(0, pixel) = (*dSet)[index][pixel] * 0.2126 + (*dSet)[index][pixel + 1024] * 0.7152 + (*dSet)[index][pixel + 2048] * 0.0722;
            }
            return mat;
        }
        else
        {
            cifarSize = 3072;
            matrix<float> mat = matrix<float>::Zero(1, cifarSize);
            for (unsigned int pixel(0); pixel < cifarSize; pixel++)
            {
                mat(0, pixel) = (*dSet)[index][pixel];
            }
            return mat;
        }
    }

    bool cifar10_source::matchLabelWithId(CifarLabel label, uint8_t id)
    {
        // Grâce à l'operation &, on regarde si id correspond à un flag actif (i.e bit à 1) dans label
        auto match = static_cast<cifar10_source::CifarLabel>(uint16_t(label) & (1 << id));

        // On convertit le résultat en booléen : on return donc false quand tous les bits sont à 0 (i.e aucun flag correspondant dans label)
        return static_cast<bool>(match);
    }

    MnistProvider::MnistProvider(const std::vector<size_t> &labels, size_t labelTrainSize, size_t labelTestSize)
        : sample_source(labelTrainSize, labelTestSize), mLabels{0}
    {
        std::cout << "Chargement de MNIST" << std::endl;

        mnist_reader readerTrain("MNIST/train-images-60k", "MNIST/train-labels-60k");
        readerTrain.ReadMNIST(mImageTrain, mLabelTrain);

        mnist_reader readerTest("MNIST/test-images-10k", "MNIST/test-labels-10k");
        readerTest.ReadMNIST(mImageTest, mLabelTest);

        for (size_t i(0); i < labels.size(); i++)
            mLabels[labels[i]] = 1;
    }

    sample_source::Batch MnistProvider::trainingBatch(bool) const
    {
        std::cout << "Création du Batch d'entrainement du discriminateur" << std::endl;

        Batch trainingBatch;

        matrix<float> outputTrain = matrix<float>::Zero(1, 1);
        outputTrain(0, 0) = 1.0;

        size_t compteur(0);

        for (unsigned int i(0); i < 60000 && compteur < mLabelTrainSize; i++)
        {
            if (mLabels[*mLabelTrain[i]])
            {
                trainingBatch.push_back(Sample(mImageTrain[i], outputTrain));
                compteur++;
            }
        }
        std::cout << "Chargement du Batch d'entrainement effectué !" << std::endl;
        return trainingBatch;
    }

    sample_source::Batch MnistProvider::testingBatch(bool) const
    {
        std::cout << "Création du Batch de test du discriminateur" << std::endl;

        Batch testingBatch;

        matrix<float> outputTest = matrix<float>::Zero(1, 1);
        outputTest(0, 0) = 1.0;

        size_t compteur(0);
        for (size_t i(0); i < mLabelTestSize; i++)
        {
            if (mLabels[*mLabelTest[i]])
            {
                testingBatch.push_back(Sample(mImageTest[i], outputTest));
                compteur++;
            }
        }
        std::cout << "Chargement du Batch de test effectué !" << std::endl;

        return testingBatch;
    }

    std::ostream &operator<<(std::ostream &flux, cifar10_source::CifarLabel label)
    {
        flux << static_cast<int>(label);
        return flux;
    }

    mnist_reader::mnist_reader(const std::string &pathImage, const std::string &pathLabel)
    {
        mFullPathImage = pathImage;
        mFullPathLabel = pathLabel;
    }

    int mnist_reader::reverseInt(int i)
    {
        unsigned char ch1, ch2, ch3, ch4;
        ch1 = i & 255;
        ch2 = (i >> 8) & 255;
        ch3 = (i >> 16) & 255;
        ch4 = (i >> 24) & 255;
        return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
    }

    void mnist_reader::ReadMNIST(std::vector<matrix<float>> &mnist, matrix<size_t> &label)
    {
        std::ifstream file(mFullPathImage, std::ios::binary);
        if (file.is_open())
        {
            int magic_number = 0;
            int number_of_images = 0;
            int n_rows = 0;
            int n_cols = 0;
            file.read((char *)&magic_number, sizeof(magic_number));
            magic_number = reverseInt(magic_number);
            file.read((char *)&number_of_images, sizeof(number_of_images));
            number_of_images = reverseInt(number_of_images);
            file.read((char *)&n_rows, sizeof(n_rows));
            n_rows = reverseInt(n_rows);
            file.read((char *)&n_cols, sizeof(n_cols));
            n_cols = reverseInt(n_cols);
            mnist.resize(number_of_images);
            for (auto itr(mnist.begin()); itr != mnist.end(); itr++)
            {
                itr->resize(1, n_rows * n_cols);
            }
            for (int i = 0; i < number_of_images; ++i)
            {
                for (int r = 0; r < n_rows; ++r)
                {
                    for (int c = 0; c < n_cols; ++c)
                    {
                        unsigned char temp = 0;
                        file.read((char *)&temp, sizeof(temp));
                        mnist[i](0, (n_rows * r) + c) = ((float)temp) / 255;
                    }
                }
            }
        }
        else
            throw std::runtime_error("mnist_reader::ReadMnist - Unable to open file : " + mFullPathImage);

        std::ifstream file2(mFullPathLabel, std::ios::binary);
        if (file2.is_open())
        {
            int magic_number = 0;
            int number_of_images = 0;
            file2.read((char *)&magic_number, sizeof(magic_number));
            magic_number = reverseInt(magic_number);
            file2.read((char *)&number_of_images, sizeof(number_of_images));
            number_of_images = reverseInt(number_of_images);
            label.resize(1, number_of_images);
            for (int i = 0; i < number_of_images; ++i)
            {
                unsigned char temp = 0;
                file2.read((char *)&temp, sizeof(temp));
                label(0, i) = (int)temp;
            }
        }
        else
            throw std::runtime_error("mnist_reader::ReadMnist - Unable to open file : " + mFullPathLabel);
    }
} /* namespace provallo */
