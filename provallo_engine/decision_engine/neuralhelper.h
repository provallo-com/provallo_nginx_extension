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
#include "matrix.h"
#include "../util/csv_file.h"
#include "../third_party/cifar10.h"
#include <thread>
#include <mutex>
#include <condition_variable>
namespace provallo
{
  using ActivationFun = std::function<float(float)>;
  using ErrorFun = std::function<float(matrix<float>, matrix<float>)>;

  ActivationFun
  sigmoid (float lambda);
  ActivationFun
  heavyside (float gapAbscissa);
  ActivationFun
  hyperTan ();
  ActivationFun
  reLu ();
  ActivationFun
  reLuLeaky (float lambda = 0.01);
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

    using real_type = float;

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
    layer_backpropagation (const type_matrix &xnPartialDerivative, float step) =0;
    virtual type_matrix
    layer_backdrop_invariant (const type_matrix & xnPartialDerivative) =0;
    virtual type_matrix
    mini_batch_layer_backdrop (const type_matrix & xnPartialDerivative, float step) =0;
    virtual void
    update_layer_weights (size_t minibatchSize = 1) =0;

    virtual void
    reset () =0;
    virtual size_t
    get_input_size () =0;
  private:

    virtual type_matrix
    fnDerivativeMatrix () const =0;
  };
  // 
  class fully_connected_layer : public neuron_layer
  {
  public:

    fully_connected_layer (size_t inputSize, size_t outputSize,
			   std::function<float
			   (float)> activationF = sigmoid (10.f),
			   size_t descentType = 0);
    fully_connected_layer (size_t inputSize, size_t outputSize,
			   type_matrix weight, type_matrix bias,
			   std::function<float
			   (float)> activationF = sigmoid (10.f),
			   size_t descentType = 0);


    
    virtual type_matrix
    process_layer (const type_matrix& inputs);

    virtual type_matrix
    layer_backpropagation (const type_matrix & xnPartialDerivative, float step);

    type_matrix
    layer_backdrop_invariant (const type_matrix & xnPartialDerivative);

    type_matrix
    mini_batch_layer_backdrop (const type_matrix & xnPartialDerivative, float step);

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
			   const type_matrix& ynPartialDerivative, float step);

    void
    updateSecondMomentStep (const type_matrix& wnPartialDerivative,
			    const type_matrix& ynPartialDerivative, float step);

    type_matrix _weight;

    type_matrix _bias;

    std::function<float
    (float)> _activation_functions;

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
    using type_matrix = matrix<float>;

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
		       std::function<float
		       (float)> activationF = sigmoid (10.f));

    convolution_layer (size_t tailleImg, size_t _Channels,
		       const std::vector<type_matrix>& weight, std::function<float
		       (float)> activationF = sigmoid (10.f));



    ~convolution_layer (){}

    type_matrix
    process_layer (const type_matrix& inputs);

    type_matrix
    layer_backpropagation (const type_matrix & xnPartialDerivative, float step);

    type_matrix
    layer_backdrop_invariant (const type_matrix & xnPartialDerivative){return xnPartialDerivative;}//FIXME:}

    type_matrix
    mini_batch_layer_backdrop (const type_matrix & xnPartialDerivative, float step){return xnPartialDerivative;}//FIXME:}

    void
    update_layer_weights (size_t minibatchSize = 1){}

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

    std::function<float
    (float)> _activation_functions;

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
		 std::function<float
		 (float)> activationF = sigmoid (10.f),
		 size_t descentType = 0);

    type_matrix
    process_layer (const type_matrix& inputs);

    type_matrix
    layer_backpropagation (const type_matrix & xnPartialDerivative, float step);

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
    layer_backpropagation (const type_matrix & xnPartialDerivative, float step);

    type_matrix
    layer_backdrop_invariant (const type_matrix & xnPartialDerivative);

    type_matrix
    mini_batch_layer_backdrop (const type_matrix & xnPartialDerivative, float step);

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
		       std::function<float
		       (float)> activationF = sigmoid (10.f));

    max_pooling_layer (size_t inputSize, size_t outputSize,
		       type_matrix weight, type_matrix bias, std::function<float
		       (float)> activationF = sigmoid (10.f));

    virtual ~max_pooling_layer ();

    type_matrix
    process_layer (const type_matrix& inputs);
    type_matrix
    layerBackprop (type_matrix xnPartialDerivative, float step);
    type_matrix
    layerBackpropInvariant (type_matrix xnPartialDerivative);

    type_matrix
    minibatchLayerBackprop (type_matrix xnPartialDerivative, float step);

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

    std::function<float
    (float)> _activation_functions;

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
    using type_matrix = matrix<float>;

  public:

    neural_net ();
    neural_net (   const std::vector<size_t>& layerSizes,
		   const std::vector<matrix<float>> &weightVector,
		   const std::vector<matrix<float>> &biasVector,
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
      neural_net (Container layerList)
      {
	for (auto itr = layerList.begin (); itr != layerList.end (); ++itr)
	  push_back (neuron_layer::ptr( *itr));
      }

    type_matrix
    processNetwork (type_matrix input)
    {
      for (auto itr = begin (); itr != end (); ++itr)
	input = (*itr)->process_layer (input);
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
    backpropDiscriminator (matrix<float> input, matrix<float> desiredOutput,
			   float step = 0.2, float dx = 0.05);

    void
    backpropGenerator (matrix<float> input, matrix<float> desiredOutput,
		       float step = 0.2, float dx = 0.05);

    void
    minibatchDiscriminatorBackprop (neural_net::ptr network,
				    matrix<float> input,
				    matrix<float> desiredOutput, float step =
					0.2,
				    float dx = 0.05);

    void
    minibatchGeneratorBackprop (neural_net::ptr network, matrix<float> input,
				matrix<float> desiredOutput, float step = 0.2,
				float dx = 0.05);

    void
    updateNetworkWeights (neural_net::ptr network, size_t minibatchSize =
			      1);

  private:

    void
    propagateError (neural_net::ptr network, matrix<float> xnPartialDerivative, float step);

    matrix<float>
    propagateErrorMinibatch (neural_net::ptr network,
			     matrix<float> xnPartialDerivative, float step);

    matrix<float>
    propagateErrorDiscriminatorInvariant (matrix<float> xnPartialDerivative);

    matrix<float>
    calculateInitialErrorVector (matrix<float> output,
				 matrix<float> desiredOutput, float dx);

    matrix<float>
    calculateInitialErrorVectorGen (matrix<float> output,
				    matrix<float> desiredOutput, float dx);

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
          using Sample = std::pair<matrix<float>, matrix<float>>;
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

          std::vector<matrix<float>>    mImageTrain;
          matrix<size_t>                 mLabelTrain;

          std::vector<matrix<float>>    mImageTest;
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
      matrix<float> getMatrix(size_t index, bool isTrainOrTestRequired = 1, bool greyLevel = 0) const;

  private:
      // Résolution automatique de type parce que j'ai la flemme
      decltype(cifar::read_dataset()) mDataset;
      CifarLabel                      mLabels;

  };


  class mnist_reader
  {
  public:
      mnist_reader(const std::string& full_path_image, const  std::string& full_path_label);
      void ReadMNIST(std::vector<matrix<float>> &mnist, matrix<size_t> &label);

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
              float meanGen;
              float meanDis;
              float deviationGen;
              float confidenceRangeGen;
  			float deviationDis;
  			float confidenceRangeDis;

  		};

      public:
                          error_processor();

          error_statistics   processData() const;
          void            addResultGen(float result);
          void            addResultDis(float result);

      private:


      private:
          std::vector<float> mErrorsGen;
          std::vector<float> mErrorsDis;

  };

  class source_processor
  {
      public:
          source_processor(const std::string& CSVFileNameRes = "resultat", const std::string& CSVFileNameImg = "image");

          error_processor& operator[](size_t teachIndex);

          void exportData(bool mustProcessData = true);

          void exportImage(const matrix<float>& image, size_t teachIndex, size_t sizeSide);

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
 
              float step;
              float dx;
  	          float sigmoidParameter;

              bool networkAreImported;
              bool useAverageForBatchlearning;

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
          using Sample = std::pair<matrix<float>,matrix<float>>;
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

          /// Constructeur par fonction modèle
          learning_task(neural_net::ptr network,
                        std::function<matrix<float>(matrix<float>)> modelFunction,
                        std::vector<matrix<float>> teachingInputs,
                        std::vector<matrix<float>> testingInputs);
          
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
          float runTestGen(int limit = -1, bool returnErrorRate = 1);

          /// Effectue une run de tests sur D(x)
          /**
           * Effectue une run de test sur le batch de test
  		 * @param limit permet de limiter le nombre d'entrées de tests
  		 * @param returnErrorRate deprecated
           */
          float runTestDis(int limit = -1, bool returnErrorRate = 1);

          /// Effectue une approximation du score des réseaux
          float gameScore(int nbImages);

          /// Génère une image à partir d'un input
          /**
           * Effectue un process de l'input par le Generateur
           * @param input un vecteur colonne, généralement, du bruit blanc
           */
          matrix<float> genProcessing(matrix<float> input);

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
