/*
 * neuralhelper.h
 *
 *  Created on: May 28, 2023
 *      Author:  Yaniv Karta
 *      Frenchy native  GANs implementation
 */

#ifndef DECISION_ENGINE_NEURALHELPER_H_
#define DECISION_ENGINE_NEURALHELPER_H_
#include <functional>
#include <list>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "../util/csv_file.h"
#include "../third_party/cifar10.h"
#include "matrix.h"
#include "utils.h" //real_t type


namespace provallo
{
  using ActivationFun = std::function<real_t(real_t)>;
  using ErrorFun = std::function<real_t(matrix<real_t>, matrix<real_t>)>;

  ActivationFun
  sigmoid (real_t lambda);
  ActivationFun
  heavyside (real_t gapAbscissa);
  ActivationFun
  hyperTan ();
  ActivationFun
  reLu ();
  ActivationFun
  reLuLeaky (real_t lambda = 0.01);
  ErrorFun
  l2Norm ();
  ErrorFun
  coutDiscr ();
  ErrorFun
  coutGen ();
  ErrorFun
  genMinMax ();
  ErrorFun
  genKLDiv ();


  //  template<class T> class matrix;
  class neuron_layer
  {
  public:
    using ptr= std::unique_ptr<neuron_layer>;

    using real_type = real_t;

    using type_matrix = matrix<real_type>;
 

    //default constructor
    neuron_layer () = default;
    // copy constructor
    neuron_layer (const neuron_layer&) = delete;
    // move constructor
    neuron_layer (neuron_layer&&) = delete;
    // copy assignment
    neuron_layer& operator= (const neuron_layer&) = delete;


    virtual
    ~neuron_layer () = default;
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
  private:
     virtual type_matrix
    fnDerivativeMatrix () const =0;
     size_t last_step_count=0;
     
  };
  // 
  class fully_connected_layer : public neuron_layer
  {
  public:

    fully_connected_layer (size_t inputSize, size_t outputSize,
			   std::function<real_t
			   (real_t)> activationF = sigmoid (10.f),
			   size_t descentType = 0);
    fully_connected_layer (size_t inputSize, size_t outputSize,
			   type_matrix weight, type_matrix bias,
			   std::function<real_t
			   (real_t)> activationF = sigmoid (10.f),
			   size_t descentType = 0);


    
    virtual type_matrix
    process_layer (const type_matrix& inputs);

    virtual type_matrix
    layer_backpropagation (const type_matrix & xnPartialDerivative, real_t step);

    type_matrix
    layer_backdrop_invariant (const type_matrix & xnPartialDerivative);

    type_matrix
    mini_batch_layer_backdrop (const type_matrix & xnPartialDerivative, real_t step);

    virtual void
    update_layer_weights (size_t minibatchSize = 1);

    void
    reset ();

    size_t
    get_input_size ();

  protected:
    type_matrix
    fnDerivativeMatrix () const;
    void
    updateFirstMomentStep (const type_matrix& wnPartialDerivative,
			   const type_matrix& ynPartialDerivative, real_t step);

    void
    updateSecondMomentStep (const type_matrix& wnPartialDerivative,
			    const type_matrix& ynPartialDerivative, real_t step);

    type_matrix _weight;

    type_matrix _bias;

    std::function<real_t
    (real_t)> _activation_functions;

    type_matrix _buffer_activation_level;

    type_matrix _buffer_input;

    type_matrix _sum_weight_variation;

    type_matrix _sum_bias_variation;

    type_matrix _adaptive_weight_step;

    type_matrix _adaptive_bias_step;

    type_matrix _adaptive_weight_second_step;

    type_matrix _adaptive_bias_second_step;

    size_t _descent_type;

    size_t _update_count;
  };

  class convolution_operator
  {
  protected:
    static std::recursive_mutex mtx;

  public:
    using type_matrix = matrix<real_t>;

    explicit convolution_operator (type_matrix input, type_matrix filtre,
		  type_matrix* resultat, bool sum_lines):
		  _input_matrix(input),_filter_matrix(filtre),_result(resultat), _sum_lines(sum_lines),_deallocate(false)
    {

        if (_result==nullptr){
          _result = new type_matrix(input);
        _deallocate = true;
      }
    }

    void
    operator() (int id);

    operator type_matrix () {
      type_matrix empty(0,0);
      if(_result)
	return *_result;
      return empty;
    }
    operator const type_matrix ()const {
      type_matrix empty(0,0);
      if(_result)
	return *_result;
      return empty;
    }
  
    type_matrix _input_matrix;
    type_matrix _filter_matrix;
//    std::shared_ptr<type_matrix> _result;


    type_matrix* _result;
    bool _sum_lines;
    bool _deallocate;

    virtual ~convolution_operator(){
      if(_deallocate)
        if(_result)
            delete _result;
          }
  };
  //

  class convolution_layer : public neuron_layer
  {
  public:

    convolution_layer (size_t inputSize, size_t _Channels,
		       size_t dimensionFiltre, size_t _Filtres,
		       std::function<real_t
		       (real_t)> activationF = sigmoid (10.f));

    convolution_layer (size_t tailleImg, size_t _Channels,
		       const std::vector<type_matrix>& weight, std::function<real_t
		       (real_t)> activationF = sigmoid (10.f));



    ~convolution_layer (){}

    type_matrix
    process_layer (const type_matrix& inputs);

    type_matrix
    layer_backpropagation (const type_matrix & xnPartialDerivative, real_t step);

    type_matrix
    layer_backdrop_invariant (const type_matrix & xnPartialDerivative){return type_matrix::value_type(1.)-(xnPartialDerivative );}//FIXME:}

    type_matrix
    mini_batch_layer_backdrop (const type_matrix & xnPartialDerivative, real_t step){return xnPartialDerivative*step;}//FIXME:}

    void
    update_layer_weights (size_t minibatchSize = 1)
	{
		if ( minibatchSize >0 ) {
		 for (size_t i = 0; i < _weight_matrix.size (); ++i)
		        {
		          _weight_matrix[i] =_weight_matrix[i] + ( _sum_weight_variation[i] / type_matrix::value_type(minibatchSize)  );
		          //_bias_matrix[i] = _bias_matrix[i] + _sum_bias_variation[i] / type_matrix::value_type(minibatchSize)  ;

	
		        }
		}
              
	}

    void
    reset (){}

    size_t
    get_input_size (){return 0;}

    static type_matrix
    convolution (const type_matrix& input, const type_matrix &filter,
		 bool sommerLignes = true);

    static type_matrix
    convolutionMonothreade (const type_matrix& input, const type_matrix& filtre,
			    bool sommerLignes = true);

  public:
    //friend std::ostream& operator<<(std::ostream& flux, convolution_layer nl);

  private:
    type_matrix
    fnDerivativeMatrix () const{type_matrix m;return m;}

  private:

    size_t _dimension_input;

    std::vector<type_matrix> _weight_matrix;

    type_matrix mBias;

    std::function<real_t
    (real_t)> _activation_functions;

    type_matrix _buffer_activation_level;

    type_matrix _buffer_input;

    std::vector<type_matrix> _sum_weight_variation;

    type_matrix _sum_bias_variation;

    size_t _input_dimension;

    size_t _input_channels;

  };
  //
  class noisy_layer : public fully_connected_layer
  {
  public:

    noisy_layer (size_t inputSize, size_t outputSize,
		 std::function<real_t
		 (real_t)> activationF = sigmoid (10.f),
		 size_t descentType = 0);

    type_matrix
    process_layer (const type_matrix& inputs);

    type_matrix
    layer_backpropagation (const type_matrix & xnPartialDerivative, real_t step);

    void
    update_layer_weights  (size_t minibatchSize = 1);

  private:
    type_matrix _noise_weights;

    type_matrix _noise_buffer;

    type_matrix _noise_variation_sum;

  };
  //
  class zero_pad_layer : public neuron_layer
  {
  public:

    zero_pad_layer (const size_t& inputSize, const size_t& outputSize,
		    const size_t& descentType =size_t(0));

    ~zero_pad_layer ();

    virtual type_matrix
    process_layer (const type_matrix& inputs);

    virtual type_matrix
    layer_backpropagation (const type_matrix & xnPartialDerivative, real_t step);

    type_matrix
    layer_backdrop_invariant (const type_matrix & xnPartialDerivative);

    type_matrix
    mini_batch_layer_backdrop (const type_matrix & xnPartialDerivative, real_t step);

    virtual void
    update_layer_weights (size_t minibatchSize = 1);

    void
    reset ();

    size_t
    get_input_size ();

  public:

    //friend std::ostream& operator<<(std::ostream& flux, zerop_pad_layer nl);

  protected:
    type_matrix
    fnDerivativeMatrix () const;

  protected:
    type_matrix _propagation_matrix;

    type_matrix _backprop_matrix;

    size_t _input_dim;

    size_t _output_dim;

    size_t _zero_padding_tail;

    size_t _zeropad_type;

  };
  //
  class max_pooling_layer : public neuron_layer
  {
  public:

    max_pooling_layer (size_t inputSize, size_t outputSize,
		       std::function<real_t
		       (real_t)> activationF = sigmoid (10.f));

    max_pooling_layer (size_t inputSize, size_t outputSize,
		       type_matrix weight, type_matrix bias, std::function<real_t
		       (real_t)> activationF = sigmoid (10.f));

    virtual ~max_pooling_layer ();

    type_matrix
    process_layer (const type_matrix& inputs);
    type_matrix
    layerBackprop (type_matrix xnPartialDerivative, real_t step);
    type_matrix
    layerBackpropInvariant (type_matrix xnPartialDerivative);

    type_matrix
    minibatchLayerBackprop (type_matrix xnPartialDerivative, real_t step);

    void
    update_layer_weights (size_t minibatchSize = 1);

    void
    reset ();

    size_t
    get_input_size ();

  public:
    //friend std::ostream& operator<<(std::ostream& flux, FullConnectedLayer nl); 

  private:
    type_matrix
    fnDerivativeMatrix () const;

  private:
    type_matrix _weight;

    type_matrix _bias;

    std::function<real_t
    (real_t)> _activation_functions;

    type_matrix _buffer_activation_level;

    type_matrix _buffer_input;

    type_matrix _sum_weight_variation;

    type_matrix _sum_bias_variation;

    type_matrix _adaptive_weight_step;

    type_matrix _adaptive_step_bias;

    size_t _descent_type;

  };
  //
  class neural_net : public std::list<neuron_layer::ptr>
  {
  public:
    using ptr = std::shared_ptr<neural_net>;
    using type_matrix = matrix<real_t>;
    using ActivationFun = std::function<real_t(real_t)>;
    using ErrorFun = std::function<real_t(matrix<real_t>, matrix<real_t>)>;
    

  public:

    neural_net ();
    neural_net (   const std::vector<size_t>& layerSizes,
		   const std::vector<matrix<real_t>> &weightVector,
		   const std::vector<matrix<real_t>> &biasVector,
		   const std::vector<ActivationFun> & activationFuns,
		   size_t descentType=0 );

    neural_net (const std::vector<size_t>& layerTypes,
		const std::vector<size_t>& layerSizes,
		const std::vector<size_t>& layerChannels,
		const std::vector<std::vector<size_t>> &layerArgs,
		const std::vector<ActivationFun>& activationFuns,
		size_t descentType = 0);


    explicit neural_net (std::vector<size_t> layerSizes);


    template<typename Container>
      neural_net (Container layerList):std::list<neuron_layer::ptr> ()  ,_step(0.2),_dx(0.05),_descent_type(0)
      {
        for (auto itr = layerList.begin (); itr != layerList.end (); ++itr)
          push_back (neuron_layer::ptr( *itr));
        //
        _xnPartialDerivative = type_matrix::Zero (get_input_size (), 1);
        _xnPartialDerivativeGen = type_matrix::Zero (get_input_size (), 1);
        _xnPartialDerivativeDis = type_matrix::Zero (get_input_size (), 1);
        _xnPartialDerivativeDisInvariant = type_matrix::Zero (get_input_size (), 1);
        _xnPartialDerivativeGenInvariant = type_matrix::Zero (get_input_size (), 1);
        //
        _buffer_activation_level = type_matrix::Zero (get_input_size (), 1);
        _buffer_input = type_matrix::Zero (get_input_size (), 1);
        _buffer_input_gen = type_matrix::Zero (get_input_size (), 1);
        _buffer_input_dis = type_matrix::Zero (get_input_size (), 1);
        _buffer_input_dis_invariant = type_matrix::Zero (get_input_size (), 1);
        _buffer_input_gen_invariant = type_matrix::Zero (get_input_size (), 1);
        //
        _sum_weight_variation = type_matrix::Zero (get_input_size (), 1);
        _sum_bias_variation = type_matrix::Zero (get_input_size (), 1);
        _sum_weight_variation_gen = type_matrix::Zero (get_input_size (), 1);
        _sum_bias_variation_gen = type_matrix::Zero (get_input_size (), 1);
        _sum_weight_variation_dis = type_matrix::Zero (get_input_size (), 1);
        _sum_bias_variation_dis = type_matrix::Zero (get_input_size (), 1);
        _sum_weight_variation_dis_invariant = type_matrix::Zero (get_input_size (), 1);
        _sum_bias_variation_dis_invariant = type_matrix::Zero (get_input_size (), 1);
        _sum_weight_variation_gen_invariant = type_matrix::Zero (get_input_size (), 1);
        _sum_bias_variation_gen_invariant = type_matrix::Zero (get_input_size (), 1);
        //
        _adaptive_weight_step = type_matrix::Zero (get_input_size (), 1);
        _adaptive_bias_step = type_matrix::Zero (get_input_size (), 1);
        _adaptive_weight_step_gen = type_matrix::Zero (get_input_size (), 1);
        _adaptive_bias_step_gen = type_matrix::Zero (get_input_size (), 1);
        _adaptive_weight_step_dis = type_matrix::Zero (get_input_size (), 1);
        _adaptive_bias_step_dis = type_matrix::Zero (get_input_size (), 1);
        _adaptive_weight_step_dis_invariant = type_matrix::Zero (get_input_size (), 1);
        _adaptive_bias_step_dis_invariant = type_matrix::Zero (get_input_size (), 1);
        _adaptive_weight_step_gen_invariant = type_matrix::Zero (get_input_size (), 1);
        _adaptive_bias_step_gen_invariant = type_matrix::Zero (get_input_size (), 1);
        // 
      }

    type_matrix
    processNetwork (type_matrix input)
    {
            
      for (auto itr = begin (); itr != end (); ++itr){
        input = (*itr)->process_layer (input);
        //update partial second deriviatives 
        _xnPartialDerivative = (*itr)->layer_backpropagation (_xnPartialDerivative, _step);
        _xnPartialDerivativeGen = (*itr)->layer_backdrop_invariant (_xnPartialDerivativeGen);
        _xnPartialDerivativeDis = (*itr)->layer_backdrop_invariant (_xnPartialDerivativeDis);
        _xnPartialDerivativeDisInvariant = (*itr)->layer_backdrop_invariant (_xnPartialDerivativeDisInvariant);
        _xnPartialDerivativeGenInvariant = (*itr)->layer_backdrop_invariant (_xnPartialDerivativeGenInvariant);
        //update steps :
        _adaptive_weight_step = _adaptive_weight_step + _xnPartialDerivative.cwiseAbs ();
        _adaptive_bias_step = _adaptive_bias_step + _xnPartialDerivative.cwiseAbs ();
        _adaptive_weight_step_gen = _adaptive_weight_step_gen + _xnPartialDerivativeGen.cwiseAbs ();
        _adaptive_bias_step_gen = _adaptive_bias_step_gen + _xnPartialDerivativeGen.cwiseAbs ();
        _adaptive_weight_step_dis = _adaptive_weight_step_dis + _xnPartialDerivativeDis.cwiseAbs ();
        _adaptive_bias_step_dis = _adaptive_bias_step_dis + _xnPartialDerivativeDis.cwiseAbs ();
        _adaptive_weight_step_dis_invariant = _adaptive_weight_step_dis_invariant + _xnPartialDerivativeDisInvariant.cwiseAbs ();
        _adaptive_bias_step_dis_invariant = _adaptive_bias_step_dis_invariant + _xnPartialDerivativeDisInvariant.cwiseAbs ();
        _adaptive_weight_step_gen_invariant = _adaptive_weight_step_gen_invariant + _xnPartialDerivativeGenInvariant.cwiseAbs ();
        _adaptive_bias_step_gen_invariant = _adaptive_bias_step_gen_invariant + _xnPartialDerivativeGenInvariant.cwiseAbs ();
        //update buffers
         _buffer_input = input;
         _buffer_input_gen = input;
         _buffer_input_dis = input;
         _buffer_input_dis_invariant = input;
         _buffer_input_gen_invariant = input;
        
        //update sums
        _sum_weight_variation = _sum_weight_variation + _xnPartialDerivative;
        _sum_bias_variation = _sum_bias_variation + _xnPartialDerivative;
        _sum_weight_variation_gen = _sum_weight_variation_gen + _xnPartialDerivativeGen;
        _sum_bias_variation_gen = _sum_bias_variation_gen + _xnPartialDerivativeGen;
        _sum_weight_variation_dis = _sum_weight_variation_dis + _xnPartialDerivativeDis;
        _sum_bias_variation_dis = _sum_bias_variation_dis + _xnPartialDerivativeDis;
        _sum_weight_variation_dis_invariant = _sum_weight_variation_dis_invariant + _xnPartialDerivativeDisInvariant;
        _sum_bias_variation_dis_invariant = _sum_bias_variation_dis_invariant + _xnPartialDerivativeDisInvariant;
        _sum_weight_variation_gen_invariant = _sum_weight_variation_gen_invariant + _xnPartialDerivativeGenInvariant;
        _sum_bias_variation_gen_invariant = _sum_bias_variation_gen_invariant + _xnPartialDerivativeGenInvariant;

        //update steps
        _adaptive_weight_step = _adaptive_weight_step + _xnPartialDerivative.cwiseAbs ();
        _adaptive_bias_step = _adaptive_bias_step + _xnPartialDerivative.cwiseAbs ();
        _adaptive_weight_step_gen = _adaptive_weight_step_gen + _xnPartialDerivativeGen.cwiseAbs ();
        _adaptive_bias_step_gen = _adaptive_bias_step_gen + _xnPartialDerivativeGen.cwiseAbs ();
        _adaptive_weight_step_dis = _adaptive_weight_step_dis + _xnPartialDerivativeDis.cwiseAbs ();
        _adaptive_bias_step_dis = _adaptive_bias_step_dis + _xnPartialDerivativeDis.cwiseAbs ();
        _adaptive_weight_step_dis_invariant = _adaptive_weight_step_dis_invariant + _xnPartialDerivativeDisInvariant.cwiseAbs ();
        _adaptive_bias_step_dis_invariant = _adaptive_bias_step_dis_invariant + _xnPartialDerivativeDisInvariant.cwiseAbs ();
        _adaptive_weight_step_gen_invariant = _adaptive_weight_step_gen_invariant + _xnPartialDerivativeGenInvariant.cwiseAbs ();
        _adaptive_bias_step_gen_invariant = _adaptive_bias_step_gen_invariant + _xnPartialDerivativeGenInvariant.cwiseAbs ();
        
        //

        //std::cout << "input: " << input << std::endl;
        
      }
      return input;
    }

    type_matrix
    processNetwork ()
    {
      type_matrix input = type_matrix::Random ((*begin ())->get_input_size (),
					       1);
      for (auto itr = begin (); itr != end (); ++itr)
	input = (*itr)->process_layer (input);

      return input;

    }
    void
    reset ()
    {
      for (auto itr = begin (); itr != end (); ++itr)
          	(*itr)->reset ();

    }
    size_t
    get_input_size ()
    {
      return (*(begin ()))->get_input_size ();

    }

  public:
    friend std::ostream&
    operator<< (std::ostream &flux, neural_net network);
  private:
    real_t _step;
    real_t _dx;
    size_t _descent_type;
    
    //errors 
    type_matrix _xnPartialDerivative;
    type_matrix _xnPartialDerivativeGen;
    type_matrix _xnPartialDerivativeDis;
    type_matrix _xnPartialDerivativeDisInvariant;
    type_matrix _xnPartialDerivativeGenInvariant;
    
    //buffers
    type_matrix _buffer_activation_level;
    type_matrix _buffer_input;
    type_matrix _buffer_input_gen;
    type_matrix _buffer_input_dis;
    type_matrix _buffer_input_dis_invariant;
    type_matrix _buffer_input_gen_invariant;
    
    //sums
    type_matrix _sum_weight_variation;
    type_matrix _sum_bias_variation;
    type_matrix _sum_weight_variation_gen;
    type_matrix _sum_bias_variation_gen;
    type_matrix _sum_weight_variation_dis;
    type_matrix _sum_bias_variation_dis;
    type_matrix _sum_weight_variation_dis_invariant;
    type_matrix _sum_bias_variation_dis_invariant;
    type_matrix _sum_weight_variation_gen_invariant;
    type_matrix _sum_bias_variation_gen_invariant;
    

    //adaptive steps
    type_matrix _adaptive_weight_step;
    type_matrix _adaptive_bias_step;
    type_matrix _adaptive_weight_step_gen;
    type_matrix _adaptive_bias_step_gen;
    type_matrix _adaptive_weight_step_dis;
    type_matrix _adaptive_bias_step_dis;
    type_matrix _adaptive_weight_step_dis_invariant;
    type_matrix _adaptive_bias_step_dis_invariant;
    type_matrix _adaptive_weight_step_gen_invariant;
    type_matrix _adaptive_bias_step_gen_invariant;

  };
  //
  class neural_helper
  {

  public:
    neural_helper();
    neural_helper (neural_net *generator, neural_net *discriminator,
		   size_t genFunType);

    neural_helper(const neural_helper& other):_generator(other._generator), 
    _discriminator(other._discriminator), _error_func_dis(other._error_func_dis),_error_func_gen(other._error_func_gen){}

    neural_helper(neural_helper&& other):_generator(std::move(other._generator)), 
    _discriminator(std::move(other._discriminator)), _error_func_dis(other._error_func_dis),_error_func_gen(other._error_func_gen){}    

    


    neural_helper&  operator=(const neural_helper& other)
    {
        _generator = other._generator;
        _discriminator = other._discriminator;
        _error_func_dis = other._error_func_dis;
        _error_func_gen = other._error_func_gen;
  
       return *this;
    }
    neural_helper&  operator=(neural_helper&& other){
      _generator = std::move(other._generator);
      _discriminator = std::move(other._discriminator);
       return *this;
    }
    
    inline const neural_net::ptr getGenerator() const{
      return _generator;
    }
    const neural_net::ptr getDiscriminator() const{
      return _discriminator;
    }


    neural_helper (neural_net::ptr generator, neural_net::ptr  discriminator,
    		   size_t genFun);

    void
    backpropDiscriminator (matrix<real_t> input, matrix<real_t> desiredOutput,
			   real_t step = 0.2, real_t dx = 0.05);

    void
    backpropGenerator (matrix<real_t> input, matrix<real_t> desiredOutput,
		       real_t step = 0.2, real_t dx = 0.05);

    void
    minibatchDiscriminatorBackprop (neural_net::ptr network,
				    matrix<real_t> input,
				    matrix<real_t> desiredOutput, real_t step =
					0.2,
				    real_t dx = 0.05);

    void
    minibatchGeneratorBackprop (neural_net::ptr network, matrix<real_t> input,
				matrix<real_t> desiredOutput, real_t step = 0.2,
				real_t dx = 0.05);

    void
    updateNetworkWeights (neural_net::ptr network, size_t minibatchSize =
			      1);

  private:

    void
    propagateError (neural_net::ptr network, matrix<real_t> xnPartialDerivative, real_t step);

    matrix<real_t>
    propagateErrorMinibatch (neural_net::ptr network,
			     matrix<real_t> xnPartialDerivative, real_t step);

    matrix<real_t>
    propagateErrorDiscriminatorInvariant (matrix<real_t> xnPartialDerivative);

    matrix<real_t>
    calculateInitialErrorVector (matrix<real_t> output,
				 matrix<real_t> desiredOutput, real_t dx);

    matrix<real_t>
    calculateInitialErrorVectorGen (matrix<real_t> output,
				    matrix<real_t> desiredOutput, real_t dx);

  private:
    neural_net::ptr _generator;
    neural_net::ptr _discriminator;

    ErrorFun _error_func_dis;
    ErrorFun _error_func_gen;
    size_t gen_func_size;

  };
  //
  class sample_source
  {
      public:
          /// Un alias pour désigner un pointeur sur in InputProvider
          using Ptr = std::unique_ptr<sample_source>;
          /// Un alias pour désigner un donnée (Entrée, Sortie)
          using Sample = std::pair<matrix<real_t>, matrix<real_t>>;
          /// Un alias pour désigner un batch de données (Entrée, Sortie)
          using Batch = std::vector<Sample>;
          /// Un alias pour désigner un minibatch de données (Entrée, Sortie)
          using Minibatch = Batch;

      public:
          sample_source(size_t labelTrainSize, size_t labelTestSize)
              : mLabelTrainSize(labelTrainSize)
              , mLabelTestSize(labelTestSize) {}

          virtual Batch trainingBatch(bool greyLevel = 0) const =0;
          virtual Batch testingBatch(bool greyLevel = 0) const =0;
          void resize(size_t labelTrainSize, size_t labelTestSize)
          {
              mLabelTrainSize = labelTrainSize;
              mLabelTestSize = labelTestSize;
          }
      protected:
          size_t mLabelTrainSize;
          size_t mLabelTestSize;

  };
  ///

  class MnistProvider : public sample_source
  {
      public:
          MnistProvider(const std::vector<size_t>& labels, size_t labelTrainSize = 60000, size_t labelTestSize = 10000);

          Batch trainingBatch(bool greyLevel = 0) const;
          Batch testingBatch(bool greyLevel = 0) const;

      private:
          std::array<size_t, 10>    mLabels;

          std::vector<matrix<real_t>>    mImageTrain;
          matrix<size_t>                 mLabelTrain;

          std::vector<matrix<real_t>>    mImageTest;
          matrix<size_t>                 mLabelTest;

  };


  class cifar10_source : public sample_source
  {
  public:
      // N'importe quel type d'entier peut remplacer uint16_t, mais il faut qu'il soit sur au moins 9 bits
      enum class CifarLabel : uint16_t
      {
          airplane = 1 << 0, // = 1
          automobile = 1 << 1, // = 2
          bird = 1 << 2, // = 4
          cat = 1 << 3, // = 8
          deer = 1 << 4, // = 16
          dog = 1 << 5, // = 32
          frog = 1 << 6, // = 64
          horse = 1 << 7, // = 128
          ship = 1 << 8, // = 256
          truck = 1 << 9 // = 512
      };
      using Utype = std::underlying_type<cifar10_source::CifarLabel>::type;

  public:
      /// Constructeur
      /**
       * @param labels une combinaison des 10 labels listés dans l'enum CifarLabel, qui correspondent aux classes avec lesquelles on veut travailler
       * @param labelTrainSize le nombre d'éléments du set de train que l'on veut utiliser
       * @param labelTestSize le nombre d'éléments du set de test que l'on veut utiliser
       */
      cifar10_source(CifarLabel labels, size_t labelTrainSize = 50000, size_t labelTestSize = 10000);

      /// Retourne le batch de training
      /**
      * @param greyLevel : détermine si l'on travaille sur les images en couleur ou en niveau de gris
      */
      Batch trainingBatch(bool greyLevel) const;
      /// Retourne le batch de test
      /**
      * @param greyLevel : détermine si l'on travaille sur les images en couleur ou en niveau de gris
      */
      Batch testingBatch(bool greyLevel) const;

  private:
      /// Permet de faire la conversion label (airplaine, dog...) vers un id cifar entre 0 et 9
      /**
       * @param label une combinaison des 10 labels listés dans l'enum CifarLabel
       * @param id un label numérique entre 0 et 9 associé aux images sur lesquelles on veut travailler
       * @return vrai si id correspond bien à une classe d'image qu'on veut traiter
       */
      static bool matchLabelWithId(CifarLabel label, uint8_t id);

      /// Retourne la ième matrice de training ou de test
      /**
       * @param index l'indice de l'image à recuperer dans le set
       * @param isTrainOrTestRequired true si on veut une image de training et false pour une de test
       * @param greyLevel : détermine si l'on travaille sur les images en couleur ou en niveau de gris
       * @return la matrice correspondant à l'image d'indice index dans le set spécifié
       */
      matrix<real_t> getMatrix(size_t index, bool isTrainOrTestRequired = 1, bool greyLevel = 0) const;

  private:
      // Résolution automatique de type parce que j'ai la flemme
      decltype(cifar::read_dataset()) mDataset;
      CifarLabel                      mLabels;

  };

  //source class for .names files
  class names_source : public sample_source
  {
    enum col_type {continuous,discrete,ignore};    
           struct col_desc {
            std::string name;
            size_t index; 
            col_type type;
            union desc_union {

                struct {
                    std::string type;
                    std::string min;
                    std::string max;
                } continuous;

                struct {
                    std::vector<std::string> values;
                } discrete;


                desc_union() {} 
                ~desc_union() {}
                
            } u; 

            
         };

 

    public:
 
        names_source(const std::string& namesFile);
 
        names_source(const names_source& other)    ;
        names_source& operator=(const names_source& other)   ;
        names_source(names_source&& other)    ;
        names_source& operator=(names_source&& other)   ;

        //sample_source impl. pure virtual 
        Batch trainingBatch(bool greyLevel) const override;
        Batch testingBatch(bool greyLevel) const override;
     protected:

        void readNamesFile();
        void readTrainFile();
        void readTestFile();
        
        matrix<real_t> mImageTrain;
        matrix<real_t> mImageTest;
        std::vector<size_t> mLabelTrain;
        std::vector<size_t> mLabelTest;
        std::vector<std::string> mLabelNames;
        std::vector<std::string> ColumnNames;
        std::vector<col_desc> mColumnDesc;
        std::map<size_t /*column*/,std::map<std::string,real_t>> mDiscreteMap;

        std::string mNamesFile;
        std::string mTestFile;  //.test file
        std::string mTrainFile; //.data file
        size_t mNbClasses;
        size_t mNbTrain;
        size_t mNbTest;
        size_t targetColumn;
        size_t nClasses;//number of classes
        size_t nFeatures;//number of features
        size_t nTrain;//number of training examples
        
        std::vector<size_t> mLabels;
        std::vector<size_t> mLabelsTrain;
        std::vector<size_t> mLabelsTest;
        std::vector<std::string> mLabelNamesTrain;
        std::vector<std::string> mLabelNamesTest;
        
        std::vector<matrix<real_t>> mImageTrainVector;
        std::vector<matrix<real_t>> mImageTestVector;
    
        real_t get_value(size_t col,const std::string &token);

        
   };
  class mnist_reader
  {
  public:
      mnist_reader(const std::string& full_path_image, const  std::string& full_path_label);
      void ReadMNIST(std::vector<matrix<real_t>> &mnist, matrix<size_t> &label);

  private:
      static int reverseInt (int i);
      std::string mFullPathImage;
      std::string mFullPathLabel;
      
  };

  class error_processor
  {
      public:
          struct error_statistics
          {
              real_t meanGen;
              real_t meanDis;
              real_t deviationGen;
              real_t confidenceRangeGen;
  			real_t deviationDis;
  			real_t confidenceRangeDis;

  		};

      public:
                          error_processor();

          error_statistics   processData() const;
          void            addResultGen(real_t result);
          void            addResultDis(real_t result);

      private:


      private:
          std::vector<real_t> mErrorsGen;
          std::vector<real_t> mErrorsDis;

  };

  class source_processor
  {
      public:
          source_processor(const std::string& CSVFileNameRes = "resultat", const std::string& CSVFileNameImg = "image");

          error_processor& operator[](size_t teachIndex);

          void exportData(bool mustProcessData = true);

          void exportImage(const matrix<real_t>& image, size_t teachIndex, size_t sizeSide);

          csvfile* getCSVFile();

      private:
          std::vector<error_processor> mErrorStats;
          csvfile                     mCSVRes;
          csvfile                     mCSVImg;
  };



  class learning_task
  {
      public:
          struct task_configuration
          {
 
              real_t step;
              real_t dx;
  	          real_t sigmoidParameter;

              bool networkAreImported;
              bool useAverageForBatchlearning;
              int descent;
              size_t _Experiments;
              size_t _LoopsPerExperiment;
              size_t _TeachingsPerLoop;
              size_t _DisTeach;
              size_t _GenTeach;
              size_t _DisTest;
              size_t _GenTest;
              size_t labelTrainSize;
              size_t labelTestSize;
              size_t intervalleImg;
              size_t _ImgParIntervalleImg;
  	          size_t minibatchSize;
              size_t genFunction;
              size_t descentTypeGen;
              size_t descentTypeDis;
              size_t imageSizeSide;
               std::string generatorPath;
              std::string discriminatorPath;
              std::string generatorDest;
              std::string discriminatorDest;
              std::string typeOfExperiment;
              std::string CSVFileNameImage;
              std::string CSVFileNameResult;
              std::string databaseToUse;
              std::vector<size_t> chiffresATracer;
              std::vector<std::string> classesCifar;
              std::vector<size_t> disLayerSizes;
              std::vector<size_t> genLayerSizes;
              std::vector<size_t> disLayerNbChannels;
              std::vector<size_t> genLayerNbChannels;
              std::vector<std::vector<size_t>> disLayerArgs;
              std::vector<std::vector<size_t>> genLayerArgs;
              std::vector<size_t> disLayerTypes;
              std::vector<size_t> genLayerTypes;

              friend std::ofstream& operator << (std::ofstream& stream , const task_configuration& config);
              friend std::ifstream& operator >> (std::ifstream& stream, task_configuration& config);  

          };
    
      public:
          /// Un alias pour désigner un donnée (Entrée, Sortie)
          using Sample = std::pair<matrix<real_t>,matrix<real_t>>;
          /// Un alias pour désigner un batch de données (Entrée, Sortie)
          using Batch = std::vector<Sample>;
  		    /// Un alias pour désigner un minibatch de données (Entrée, Sortie)
  	      using Minibatch = Batch;

      public:
          /// Constructeur par batchs
          /**
           * Ce constructeur supervise le projet par rapport au réseau de neurones donné et aux batchs de tests et d'apprentissages donnés en paramètre
           */
          learning_task();
          learning_task( const learning_task::task_configuration& config );           /// Constructeur par fonction modèle
          learning_task(neural_net::ptr network,
                        std::function<matrix<real_t>(matrix<real_t>)> modelFunction,
                        std::vector<matrix<real_t>> teachingInputs,
                        std::vector<matrix<real_t>> testingInputs);
          
          //file path constructor
          learning_task(const std::string& configuration_file);

 
  		void runExperiments();

          void runSingleStochasticExperiment();

  		void runSingleMinibatchExperiment();

  		void resetExperiment();


          /// Effectue une run d'apprentissage par méthode stochastique
          /**
           * Effectue une run d'apprentissage dont le nombre d'apprentissages est passé en paramètres
           */
          void runStochasticTeach();

  		/// Effectue une run d'apprentissage par la méthode par batch
  		/**
  		 * Effectue une run d'apprentissage dont le nombre d'apprentissages est passé en paramètres
  		 */
  		void runMinibatchTeach();

          /// Effectue une run de tests sur D(G(z))
          /**
           * Effectue une run de test sur le batch de test
  		 * @param limit  permet de limiter le nombre d'entrées de tests
  		 * @param returnErrorRate deprecated
           */
          real_t runTestGen(int limit = -1, bool returnErrorRate = 1);

          /// Effectue une run de tests sur D(x)
          /**
           * Effectue une run de test sur le batch de test
  		 * @param limit permet de limiter le nombre d'entrées de tests
  		 * @param returnErrorRate deprecated
           */
          real_t runTestDis(int limit = -1, bool returnErrorRate = 1);

          /// Effectue une approximation du score des réseaux
          //[deprecated] real_t gameScore(int nbImages);

          /// Génère une image à partir d'un input
          /**
           * Effectue un process de l'input par le Generateur
           * @param input un vecteur colonne, généralement, du bruit blanc
           */
          //[deprecated]          matrix<real_t> genProcessing(matrix<real_t> input);

  	private:
  		/// Génère un minibatch à partir d'un batch
  		/**
  		 * Génère un sous-ensemble du batch d'apprentissage ou du batch à partir de celui-ci
  		 * @param batch le batch d'apprentissage ou le batch de test
  		 */
  		Minibatch sampleMinibatch(Batch batch);


  		/// Génère un minibatch d'images obtenues par le générateur
  		/**
  		 * Génère un minibatch d'images obtenues par le générateur
  		 */
  		Minibatch sampleGeneratedImagesFromNoiseMinibatch();
   //// Configuration
       private:
          /// Fonction pour charger la configuration de l'application
#ifdef RAPIDJSON

  		void loadConfig(const std::string& configFileName = "config.json");

          void setConfig(rapidjson::Document& document);
#endif
          void export_weights();
          neural_net* deserialize_neural_network(std::string networkPath, ActivationFun activationFun);

      private:


          neural_net*  mDiscriminator;
          neural_net*  mGenerator;
          /// Le teacher qui permet de superviser l'apprentissage des réseaux
          neural_helper             mTeacher;

          /// Le batch contenant tous les samples d'apprentissage du projet
          Batch               mTeachingBatchDis;
          /// Le batch contenant tous les samples de test du projet
          Batch               mTestingBatchDis;
          
          Batch               mTestingBatchGen;
 
          source_processor  mSourceProcessor;
          
          task_configuration              _configuration;
  };


} /* namespace provallo */

#endif /* DECISION_ENGINE_NEURALHELPER_H_ */
