/*
 * pipelinebuilder.h
 *
 *  Created on: Jun 19, 2023
 *      Author: kardon
 */

#ifndef DECISION_ENGINE_PIPELINEBUILDER_H_
#define DECISION_ENGINE_PIPELINEBUILDER_H_
#include "matrix.h"
#include "classifier.h"
#include "hmm.h"

#include "../parsers/parser.h" //for the encoders/decoders
#include "../util/singleton.h"//for
#include "autoencoder.h"
#include "neuralhelper.h"
#include <vector>
#include <iostream>
#include <set>
#include <cstdio>

namespace provallo
{

    //dataset enums
    enum data_type : uint8_t{ DATATYPE_NUMERIC,DATATYPE_TEXT,DATATYPE_IMAGE,DATATYPE_AUDIO,DATATYPE_VIDEO,DATATYPE_TIME_SERIES,DATATYPE_GENERIC } ;
    enum data_format: uint8_t{ TEXT,CSV,TSV,ARFF,LIBSVM,LIBFM,LIBFFM,TFRECORD,CAFFE2,TORCH,MXNET,ONNX,HDF5,NPY } ; 
    enum dataset_purpose: uint8_t{ BUILD_TRAIN,TRAIN,OPTIMIZE_TRAIN,TEST,OPTIMIZE_TEST,VALIDATE,XVALIDATE } ;
    enum dataset_type: uint8_t{ STATIC,DYNAMIC } ;
    enum dataset_source: uint8_t{ FILE,STREAM,GENERATOR };
    enum dataset_status: uint8_t{ UNINITIALIZED,INITIALIZED,LOADED,PROCESSED };
    enum dataset_mode: uint8_t{ READ,WRITE } ;
    //dataset classes
     
  namespace lda
  {
      
    class LDA
    {
      size_t 
      _n_topics;
      size_t _n_features ;
      size_t _n_samples;
      size_t _n_components;
      size_t _n_top_words;
      size_t _n_iter;
      size_t _n_jobs;
      size_t _random_state;
      real_t _alpha;
      real_t _beta;
      real_t _eta;
      real_t _gamma;
      real_t _theta;
      real_t _lambda;
      real_t _learning_decay;
      real_t _learning_offset;
      size_t _max_doc_update_iter;
      size_t _total_samples;
      real_t _mean_change_tol;
      bool _verbose;


      //analyzed data
      std::vector<real_t> _lda_data;
      std::vector<real_t> _lda_components;
      std::vector<real_t> _lda_explained_variance;

      std::vector<real_t> _lda_explained_variance_ratio;
      std::vector<real_t> _lda_singular_values;
      std::vector<real_t> _lda_noise_variance;
      std::vector<real_t> _lda_mean;
      std::vector<real_t> _lda_covariance;
      std::vector<real_t> _lda_precision;
      std::vector<real_t> _lda_whiten;
      std::vector<real_t> _lda_transform;
      std::vector<real_t> _lda_transformed_data;
       
      public:
      LDA():  _n_topics(10),
              _n_features(1000),
              _n_samples(1000),
              _n_components(10),
              _n_top_words(10),
              _n_iter(10),
              _n_jobs(1),
              _random_state(0),
              _alpha(0.1),
              _beta(0.1),
              _eta(0.1),
              _gamma(0.1),
              _theta(0.1),
              _lambda(0.1),
              _learning_decay(0.7),
              _learning_offset(10),
              _max_doc_update_iter(100),
              _total_samples(1000),
              _mean_change_tol(0.001),
              _verbose(false)
      {

      } 
      LDA(size_t n_topics,
          size_t n_features,
          size_t n_samples,
          size_t n_components,
          size_t n_top_words,
          size_t n_iter,
          size_t n_jobs,
          size_t random_state,
          real_t alpha,
          real_t beta,
          real_t eta,
          real_t gamma,
          real_t theta,
          real_t lambda,
          real_t learning_decay,
          real_t learning_offset,
          size_t max_doc_update_iter,
          size_t total_samples,
          real_t mean_change_tol,
          bool verbose):  _n_topics(n_topics),
                          _n_features(n_features),
                          _n_samples(n_samples),
                          _n_components(n_components),
                          _n_top_words(n_top_words),
                          _n_iter(n_iter),
                          _n_jobs(n_jobs),
                          _random_state(random_state),
                          _alpha(alpha),
                          _beta(beta),
                          _eta(eta),
                          _gamma(gamma),
                          _theta(theta),
                          _lambda(lambda),
                          _learning_decay(learning_decay),
                          _learning_offset(learning_offset),
                          _max_doc_update_iter(max_doc_update_iter),
                          _total_samples(total_samples),
                          _mean_change_tol(mean_change_tol),
                          _verbose(verbose)
      {

      } 
      LDA(const LDA &other)
      {
        _n_topics=other._n_topics;
        _n_features=other._n_features;
        _n_samples=other._n_samples;
        _n_components=other._n_components;
        _n_top_words=other._n_top_words;
        _n_iter=other._n_iter;
        _n_jobs=other._n_jobs;
        _random_state=other._random_state;
        _alpha=other._alpha;
        _beta=other._beta;
        _eta=other._eta;
        _gamma=other._gamma;
        _theta=other._theta;
        _lambda=other._lambda;
        _learning_decay=other._learning_decay;
        _learning_offset=other._learning_offset;
        _max_doc_update_iter=other._max_doc_update_iter;
        _total_samples=other._total_samples;
        _mean_change_tol=other._mean_change_tol;
        _verbose=other._verbose;
      } 

    //fit vector<real_t> 
    std::vector<real_t> fit(const std::vector<std::vector<real_t>>& data)
    {
      real_t prb_topic_given_document = 0.0;
        real_t prb_word_given_topic = 0.0;
        real_t prb_word_given_topic_and_document = 0.0;
        real_t prb_word_given_topic_and_document_sum = 0.0;
        real_t prb_word_given_topic_and_document_sum_total = 0.0;
        real_t prb_word_given_topic_and_document_sum_total_sum= 0.0;
        real_t prb_word_given_topic_and_document_sum_total_sum_total = 0.0; 
        //real_t prb_word_given_topic_and_document_sum_total_sum = 0.0;
        //real_t prb_word_given_topic_and_document_sum_total_sum_total = 0.0;
        
      std::vector<real_t> result;
      //implement fit: 
      //start LDA:
      //update the probabilities :
     // p(word w with topic t) = p(topic t | document d) * p(word w | topic t)
       for ( const auto& sample :data )
       {
        //update the probabilities :
        //p(word w with topic t) = p(topic t | document d) * p(word w | topic t)

        //add components
        for(const auto& bow_probability : sample)
        {
          //calculate the probabilities :
          _n_topics = sample.size();
          _n_features = sample.size();
          _n_samples =data.size();
          _n_components = sample.size();
          _n_top_words = sample.size();
          _n_iter = sample.size()*sample.size()*data.size();

          prb_topic_given_document = bow_probability/_n_topics;
          prb_word_given_topic = bow_probability/_n_topics;
          prb_word_given_topic_and_document = prb_topic_given_document * prb_word_given_topic; 
          prb_word_given_topic_and_document_sum = prb_word_given_topic_and_document_sum + prb_word_given_topic_and_document;
          prb_word_given_topic_and_document_sum_total = prb_word_given_topic_and_document_sum_total + prb_word_given_topic_and_document_sum;
          prb_word_given_topic_and_document_sum_total_sum =
          prb_word_given_topic_and_document_sum_total_sum + prb_word_given_topic_and_document_sum_total ;

          prb_word_given_topic_and_document_sum_total_sum_total = prb_word_given_topic_and_document_sum_total_sum_total + prb_word_given_topic_and_document_sum_total_sum;

          //add components:
          _lda_components.push_back(prb_word_given_topic_and_document_sum_total_sum_total); 
          //add explained_variance
          _lda_explained_variance.push_back(prb_word_given_topic_and_document_sum_total_sum_total);
          //add explained_variance_ratio
          _lda_explained_variance_ratio.push_back(prb_word_given_topic_and_document_sum_total_sum_total);
          //add singular_values
          _lda_singular_values.push_back(prb_word_given_topic_and_document_sum_total_sum_total);
          //add noise_variance
          _lda_noise_variance.push_back(prb_word_given_topic_and_document_sum_total_sum_total);
          //add mean
          _lda_mean.push_back(prb_word_given_topic_and_document_sum_total_sum_total);
          //add covariance
          _lda_covariance.push_back(prb_word_given_topic_and_document_sum_total_sum_total);
          //add precision
          _lda_precision.push_back(prb_word_given_topic_and_document_sum_total_sum_total);
          //add whiten
          _lda_whiten.push_back(prb_word_given_topic_and_document_sum_total_sum_total);
          //add transform
          _lda_transform.push_back(prb_word_given_topic_and_document_sum_total_sum_total);
          //add data
          _lda_data.push_back(prb_word_given_topic_and_document_sum_total_sum_total);
          //add transformed_data
          _lda_transformed_data.push_back(prb_word_given_topic_and_document_sum_total_sum_total);
        }//end of for
          
      }//end of for
      //return result

      //normalize lda values :
      //normalize lda components
      real_t lda_components_sum = 0.0;
      for(const auto& lda_component : _lda_components)
      {
        lda_components_sum = lda_components_sum + lda_component;
      }
      for(auto& lda_component : _lda_components)
      {
        lda_component = lda_component/lda_components_sum;
      }
      //normalize lda explained_variance
      real_t lda_explained_variance_sum = 0.0;
      for(const auto& lda_explained_variance : _lda_explained_variance)
      {
        lda_explained_variance_sum = lda_explained_variance_sum + lda_explained_variance;
      }   
      for(auto& lda_explained_variance : _lda_explained_variance)
      {
        lda_explained_variance = lda_explained_variance/lda_explained_variance_sum;
      } 
      //normalize lda explained_variance_ratio  
      real_t lda_explained_variance_ratio_sum = 0.0;  
      for(const auto& lda_explained_variance_ratio : _lda_explained_variance_ratio)
      {
        lda_explained_variance_ratio_sum = lda_explained_variance_ratio_sum + lda_explained_variance_ratio;
      } 
      for(auto& lda_explained_variance_ratio : _lda_explained_variance_ratio)
      {
        lda_explained_variance_ratio = lda_explained_variance_ratio/lda_explained_variance_ratio_sum;
      } 
      //normalize lda singular_values 
      real_t lda_singular_values_sum = 0.0;
      for(const auto& lda_singular_values : _lda_singular_values)
      {
        lda_singular_values_sum = lda_singular_values_sum + lda_singular_values;
      } 
      for(auto& lda_singular_values : _lda_singular_values)
      {
        lda_singular_values = lda_singular_values/lda_singular_values_sum;
      } 
      //normalize lda noise_variance  
      real_t lda_noise_variance_sum = 0.0;  
      for(const auto& lda_noise_variance : _lda_noise_variance)
      {
        lda_noise_variance_sum = lda_noise_variance_sum + lda_noise_variance;
      } 
      for(auto& lda_noise_variance : _lda_noise_variance)
      {
        lda_noise_variance = lda_noise_variance/lda_noise_variance_sum;
      } 
      //normalize lda mean
      real_t lda_mean_sum = 0.0;  
      for(const auto& lda_mean : _lda_mean)
      {
        lda_mean_sum = lda_mean_sum + lda_mean;
      } 
      for(auto& lda_mean : _lda_mean)
      {
        lda_mean = lda_mean/lda_mean_sum;
      }
      //normalize lda covariance
      real_t lda_covariance_sum = 0.0;
      for(const auto& lda_covariance : _lda_covariance)
      {
        lda_covariance_sum = lda_covariance_sum + lda_covariance;
      }
      for(auto& lda_covariance : _lda_covariance)
      {
        lda_covariance = lda_covariance/lda_covariance_sum;
      }
      //normalize lda precision
      real_t lda_precision_sum = 0.0;
      for(const auto& lda_precision : _lda_precision)
      {
        lda_precision_sum = lda_precision_sum + lda_precision;
      }
      for(auto& lda_precision : _lda_precision)
      {
        lda_precision = lda_precision/lda_precision_sum;
      }
      //normalize lda whiten
      real_t lda_whiten_sum = 0.0;
      for(const auto& lda_whiten : _lda_whiten)
      {
        lda_whiten_sum = lda_whiten_sum + lda_whiten;
      }
      for(auto& lda_whiten : _lda_whiten)
      {
        lda_whiten = lda_whiten/lda_whiten_sum;
      }
      //normalize lda transform
      real_t lda_transform_sum = 0.0;
      for(const auto& lda_transform : _lda_transform)
      {
        lda_transform_sum = lda_transform_sum + lda_transform;
      }
      for(auto& lda_transform : _lda_transform)
      {
        lda_transform = lda_transform/lda_transform_sum;
      }
      //normalize lda data
      real_t lda_data_sum = 0.0;
      for(const auto& lda_data : _lda_data)
      {
        lda_data_sum = lda_data_sum + lda_data;
      }
      for(auto& lda_data : _lda_data)
      {
        lda_data = lda_data/lda_data_sum;
      }
      //normalize lda transformed_data
      real_t lda_transformed_data_sum = 0.0;
      for(const auto& lda_transformed_data : _lda_transformed_data)
      {
        lda_transformed_data_sum = lda_transformed_data_sum + lda_transformed_data;
      }
      for(auto& lda_transformed_data : _lda_transformed_data)
      {
        lda_transformed_data = lda_transformed_data/lda_transformed_data_sum;
      }
      
      
      //update results:
      result = _lda_transformed_data;

      //return result

      return result;
       
    }
    //predict vector<real_t>
    std::vector<real_t> predict(const std::vector<std::vector<real_t>>& data)
    {
      std::vector<real_t> result;
      //implement predict:
      //start LDA:
      for (const auto& d: data)
      {
        for(const auto& bow_prob : d) 
        {
          // calculate the prediction
          real_t prediction = bow_prob/_n_topics;
          //add the prediction to the result
          result.push_back(prediction);
          
        } 
      }
      //normalize result:
      real_t result_sum = 0.0;
      for(const auto& r : result)
      {
        result_sum = result_sum + r;
      } 
      for(auto& r : result)
      {
        r = r/result_sum;
      } 


      //return result
      return result;
    }
    //mean 
    std::vector<real_t> mean()
    {
      return _lda_mean; 
    }
    //covariance
    std::vector<real_t> covariance()
    {
      return _lda_covariance; 

    }
    //precision
    std::vector<real_t> precision()
    {
      //implement precision:
      //start LDA:
      //create a model
     return _lda_precision;
    }
    //whiten
    std::vector<real_t> whiten()
    {
      //implement whiten:
      //start LDA:
      //create a model
      return _lda_whiten;

    }
    //transform vector<real_t>
    std::vector<real_t> transform(const std::vector<std::vector<real_t>>& data)
    {
      std::vector<real_t> result;
      //implement transform:
      //start LDA:
      for (const auto& d: data)
      {
        //create a model
        for(const auto& bow_prob : d) 
        {
          //calculate the transform
          real_t transform = bow_prob/_n_topics;
          //add the transform to the result
          result.push_back(transform);
          
        } 
      }
      //return result
      return result;
    }
    //components
    std::vector<real_t> components()
    {
      
      return _lda_components;


    }
    //explained_variance
    std::vector<real_t> explained_variance()
    {
    
      return _lda_explained_variance;


    }
    //explained_variance_ratio
    std::vector<real_t> explained_variance_ratio()
    {
      
        return _lda_explained_variance_ratio;
    }
    //singular_values
    std::vector<real_t> singular_values()
    {
        
        return _lda_singular_values;
  

    } 
    //noise_variance
    std::vector<real_t> noise_variance()
    {
        
          return _lda_noise_variance;

    }   
    //mean_change_tol
    real_t mean_change_tol()
    {
      //implement mean_change_tol:
      //start LDA:
      //create a model
      return _mean_change_tol;
    } 
    //max_doc_update_iter 
   size_t max_doc_update_iter()
    {
      //implement max_doc_update_iter :
      //start LDA:
      //create a model
      return _max_doc_update_iter;

    } 
    //learning_offset 
    real_t learning_offset()
    {
      //implement learning_offset :
      //start LDA:
      //create a model
      return _learning_offset;

    } 
    //learning_decay  
    real_t learning_decay()
    {
      //implement learning_decay  :
      //start LDA:
      //create a model
      return _learning_decay;

    } 
    //lambda  
    real_t lambda()
    {
      //implement lambda  :
      //start LDA:
      //create a model
      return _lambda;

    } 
    //theta 
    real_t theta()
    {
      //implement theta :
      //start LDA:
      //create a model
      return _theta;

    } 
    //gamma
    real_t gamma()
    {
      //implement gamma:
      //start LDA:
      //create a model
      return _gamma;

    }
    //eta
    real_t eta()
    {
      //implement eta:
      //start LDA:
      //create a model
      return _eta;
    }   
    //beta
    real_t beta()
    {
      //implement beta:
      //start LDA:
      //create a model
      return _beta;
    } 
    //alpha 
    real_t alpha()
    {
      //implement alpha :
      //start LDA:
      //create a model
      return _alpha;


    }   
    //random_state
    size_t random_state()
    {
      //implement random_state:
      //start LDA:
      //create a model
      return _random_state;

    }   
    //n_jobs    
    size_t n_jobs()
    {
      //implement n_jobs    :
      //start LDA:
      //create a model
      return _n_jobs;

    } 
    //n_iter
    size_t n_iter()
    {
      //implement n_iter:
      //start LDA:
      //create a model
      return _n_iter;
    } 
    //n_top_words
     size_t n_top_words()
    {
      //implement n_top_words:
      //start LDA:
      //create a model
      return _n_top_words;
    } 
    //n_components
    size_t n_components()
    {
      //implement n_components:
      //start LDA:
      //create a model
      return _n_components;

    } 
    //n_samples
    size_t n_samples()
    {
      //implement n_samples:
      //start LDA:
      //create a model
      return _n_samples;
    } 
    //n_features  
    size_t n_features()
    {
      //implement n_features  :
      //start LDA:
      //create a model
      return _n_features;

    } 
    //n_topics  
    size_t  n_topics()
    {
      //implement n_topics  :
      //start LDA:
      //create a model
      return _n_topics;

    } 
    //
    //
    //transform vector<real_t>
    //    _lda_data = lda.transform(bow);
    //_lda_components = lda.components();
    //_lda_explained_variance = lda.explained_variance();
    //_lda_explained_variance_ratio = lda.explained_variance_ratio();
    //_lda_singular_values = lda.singular_values();
    //_lda_noise_variance = lda.noise_variance();
    //_lda_mean = lda.mean();
    //_lda_covariance = lda.covariance();
    //_lda_precision = lda.precision();
    //_lda_whiten = lda.whiten();

    };//end of LDA
   }//end of namespace lda
  
  //python style estimators (fit/predict)
  enum vectorizer_type  : uint8_t {
          TFIDF=0,
          STANDARD_SCALER,
          MIN_MAX_SCALER,
          PCA,
          ONE_HOT_VECTORIZER,
          NEURAL_TRANSFORMER,
          AERONATIC_QARTERION, 
          SVD_OPERATOR,
          NGRAM_HMM_TRANSFORMER,
          HPLANE_TRANSFORMER,
          HUFFMAN_TRANSFORMER,
          HMM_TRANSFORMER,
          REGRESSION_TRANSFORMER,
          UMAP_VECTORIZER,
          TSNE_VECTORIZER,
          AUTOENCODER_VECTORIZER,
          LDA_VECTORIZER,
          UNKNOWN_VECTORIZER,
          NULL_VECTORIZER
          };
  //forward declaration
  template <typename vector_src, typename real_x>   class vectorizer;
  template < typename real_x>
  class estimator
  {
  protected:
    
  public:
    estimator() = default;
    estimator(const estimator<real_x> &other){ UNDEF_REFERENCE(other);UNDEF_REFERENCE2(other);}
    estimator(estimator<real_x> &&other){UNDEF_REFERENCE(other);UNDEF_REFERENCE2(other);}
 
    virtual ~estimator(){}

    virtual  vectorizer_type get_type()const=0;
    estimator<real_x>& operator= (const estimator<real_x> &other);
    
    estimator<real_x>&
        operator= (estimator<real_x> &&other);
  };


  template< typename real_x>
   class transform_estimator : public estimator <real_x>
   {
    
    matrix<real_x> _data;

   public:
    transform_estimator(){}
    transform_estimator(transform_estimator<real_x> &&other):estimator<real_x>(other) 
    {
      _data= std::move(other._data);
    } 

    transform_estimator(const transform_estimator<real_x> &other):estimator<real_x>(other)
    {
      _data=other._data;
    } 
    virtual ~transform_estimator(){};

    virtual  vectorizer_type get_type()const=0;

    //Learn and estimate the parameters of the transformation
    virtual  std::vector<real_x> fit ( const matrix<real_x>& train_data )=0;

    //Apply the learned transformation to new data
    virtual std::vector<real_x> transform(const matrix<real_x>& test_data)=0;
    //transform()	Apply the learned transformation to new data	transformed_data = estimator.transform(X)

    //transformed_data = estimator.transform()

    //fit_transform()	Learn the parameters and apply the transformation to new data	transformed_data = estimator.fit_transform(X)	transformed_data = estimator.fit_transform(data)
   };
  //vectorizer implements transofrm_estimator 


#define DEFAULT_IMPL(X) \
UNDEF_REFERENCE(X); \
UNDEF_REFERENCE2(X); \
return std::vector<real_t>();

//vector
std::ifstream& operator>>(std::ifstream& is, std::vector<real_t>& obj);
std::ofstream& operator<<(std::ofstream& os, const std::vector<real_t>& obj);

//matrix is already contained.


  template <typename vector_src, typename real_x>
  class vectorizer : public transform_estimator<real_x>
  {
    protected:
    vectorizer_type _type;
    vector_src _data;
    std::vector<real_x> _transformed_data;
    std::vector<real_x> _fitted_data;
    std::vector<real_x> _predicted_data;
    //avoid incomplete type
    template <typename vector_src2, typename real_x2> friend class vectorizer;

    public:
    vectorizer(vectorizer_type type):transform_estimator<real_x>()  
    {
      _type=type;
    }
    vectorizer(vectorizer<vector_src,real_x> &&other):transform_estimator<real_x>(other)  
    {
      _type=other._type;
      _data=other._data;
      _transformed_data=other._transformed_data;
      _fitted_data=other._fitted_data;
      _predicted_data=other._predicted_data;
    }
    vectorizer(const vectorizer<vector_src,real_x> &other):transform_estimator<real_x>(other) 
    {
      _type=other._type;
      _data=other._data;
      _transformed_data=other._transformed_data;
      _fitted_data=other._fitted_data;
      _predicted_data=other._predicted_data;
    }

    
    vectorizer<vector_src,real_x>& operator= (const vectorizer<vector_src,real_x> &other)
    {
      _type=other._type;
      _data=other._data;
      _transformed_data=other._transformed_data;
      _fitted_data=other._fitted_data;
      _predicted_data=other._predicted_data;
      return *this;
    }
    vectorizer<vector_src,real_x>&
        operator= (vectorizer<vector_src,real_x> &&other)
        {
          _type=other._type;
          _data=other._data;
          _transformed_data=other._transformed_data;
          _fitted_data=other._fitted_data;
          _predicted_data=other._predicted_data;
          return *this;
        }

    const vector_src& get_data() const 
    {
      return _data;

    }
    virtual void add_document(const vector_src& doc)
    {
      _data+=doc;
       
    }
    vector_src& get_data() 
    {
      return _data;
    }

    void set_data(vector_src& data)
    {
      _data=data;
    } 
    void set_data(vector_src&& data)
    {
      _data=data;
    }
    virtual void set_type(vectorizer_type type)
    {
      _type=type;
    }
    virtual void fit()
    {
      _fitted_data=fit({_data});
    }
    virtual size_t get_output_size()const
    {
      return _fitted_data.size();
    }
    virtual  vectorizer_type get_type()const
    {
      return _type;
    }
    virtual void process_documents()
    {
      _fitted_data=fit({_data});
      _transformed_data=transform({_data});
      _predicted_data=predict({_data});

    } 

    //predict single source 
    virtual  std::vector<real_x> predict(const vector_src& data_){DEFAULT_IMPL(data_);}

    //fit:
    virtual  std::vector<real_x> fit( const std::vector<vector_src>& data_){DEFAULT_IMPL(data_);}
    virtual  std::vector<real_x> predict(const std::vector<vector_src>& data_){DEFAULT_IMPL(data_);}
    virtual  std::vector<real_x> transform(const std::vector<vector_src>& data_){DEFAULT_IMPL(data_);}
    virtual  std::vector<real_x> fit_transform(const std::vector<vector_src>& data_){ DEFAULT_IMPL(data_);} 
    //inverse:
    virtual  std::vector<real_x> fit( const provallo::matrix<real_x>&data_ ){DEFAULT_IMPL(data_);}
    virtual  std::vector<real_x> predict(const provallo::matrix<real_x>& data_){ DEFAULT_IMPL(data_);}
    virtual  std::vector<real_x> transform(const provallo::matrix<real_x>& data_){ DEFAULT_IMPL(data_);}
    void clear()
    {
      _data.clear();
      _transformed_data.clear();
      _fitted_data.clear();
      _predicted_data.clear();
    }
    virtual ~vectorizer() = default;
    friend std::ifstream& operator>>(std::ifstream& is, vectorizer<vector_src,real_x>& vec)
    {
      std::string tmp;
       std::string line;
       size_t type  = (size_t)vec._type;

        std::getline(is,line);
        std::istringstream iss(line);
        iss>>tmp>> type;
        vec._type = (vectorizer_type)type;
        std::getline(is,line);
        iss.str(line);
        iss.clear();
        is>>vec._data;
        std::getline(is,line);
        is>>vec._transformed_data;
        std::getline(is,line);
        iss.str(line);
        iss.clear();
        is>>vec._fitted_data;
        std::getline(is,line);
        iss.str(line);
        iss.clear();
        is>>vec._predicted_data;

       return is;
       
    } 
    friend std::ofstream& operator<< (std::ofstream& os, const vectorizer<vector_src,real_x>& vec)
    {
      os<<"type:"<<vec._type<<std::endl;
      os<<"data:"<<vec._data<<std::endl;
      os<<"transformed_data:"<<vec._transformed_data;
      os<<"fitted:"<<vec._fitted_data;
      os<<"predicted_data:"<<vec._predicted_data;
      return os;
    } 
    virtual void save (const std::string& filename)
    {
      std::ofstream ofs(filename);
      save(ofs);
    }
    virtual void save(std::ofstream& ofs)
    {
       ofs<<*this;
      ofs.close();
    }
    virtual void load(const std::string& filename)
    {
      std::ifstream ifs(filename);
      load(ifs);
    }
    virtual void load(std::ifstream& ifs)
    {
      ifs>>*this;
      ifs.close();
    }
    //gnu plot
      virtual void gnuplot(const std::string& filename)
      {
        {
        std::ofstream ofs(filename+".dat");
        //serialize fitted, transformed and predicted data 
        for(size_t i=0;i<_fitted_data.size();i++)
        {
          std::string index =  std::to_string(i);
          std::string fitted = std::to_string(_fitted_data[i]); 
          std::string transformed = i<_transformed_data.size()?std::to_string(_transformed_data[i]):"0";
          std::string predicted = i<_predicted_data.size()? std::to_string(_predicted_data[i]):"0";
          ofs<<index<<" "<<fitted<<" "<<transformed<<" "<<predicted<<std::endl;

          ofs<<_fitted_data[i]<<" "<<_transformed_data[i]<<" "<<_predicted_data[i]<<std::endl;
        }
        
        
        ofs<<_fitted_data<<std::endl;
        ofs<<_transformed_data<<std::endl;
        ofs<<_predicted_data<<std::endl;
        ofs.close();
        }
        std::ofstream ofs(filename);
        gnuplot(ofs);

      }
    void gnuplot(std::ofstream& ofs)
    { 
      //plot data
      std::string filename = ofs.getloc().name();
      ofs<<"set terminal png"<<std::endl;
      ofs<<"set output \""<<filename<<".png\""<<std::endl;
      
      ofs<<"set title \"vectorizer\""<<std::endl; 
      ofs<<"set xlabel \"index\""<<std::endl;
      ofs<<"set ylabel \"value\""<<std::endl;
      ofs<<"set grid"<<std::endl;
      ofs<<"set key"<<std::endl;

      ofs<<"plot \""<<filename<<".dat\" using 1:2 with lines title \"fitted\", \""<<filename<<".dat\" using 1:3 with lines title \"transformed\", \""<<filename<<".dat\" using 1:4 with lines title \"predicted\""<<std::endl;
      ofs.close();
      


    }
   };//end of vectorizer
//helper class for tfidf vectorizer
class tfidf 
{
  protected:
  std::vector<std::string> _documents;
  std::vector<std::string> _vocabulary;

  std::vector<real_t> _tf;
  std::vector<real_t> _idf;
  std::vector<std::vector<real_t>> _tfidf;
  public:
  tfidf();
  tfidf(const std::vector<std::string>& corpus );
  tfidf(const tfidf &other); //copy constructor
  tfidf(tfidf &&other); //move constructor
  tfidf& operator= (const tfidf &other);
  tfidf&
      operator= (tfidf &&other);
  ~tfidf();


  //setters/getters
  // manual vocabulary settings 
  void set_vocabulary(const std::vector<std::string>& vocabulary);
  void add_vocabulary(const std::string& word);
  void add_vocabulary(const std::vector<std::string>& vocabulary);

  //manual document settings

  void set_documents(const std::vector<std::string>& documents);
  void add_document(const std::string& document);
  void add_documents(const std::vector<std::string>& documents);


  //calculate tfidf
  void  process_documents();

  //override get output size:
  

  //inverse transform
    std::string inverse_transform(const std::vector<real_t>& vector);


    std::vector<std::string> inverse_transform (const std::vector<std::vector<real_t>> &corpus) ;
  //transform
    std::vector<real_t> transform(const std::string& document);
   //transform
    std::vector<std::vector<real_t> >  transform(const std::vector<std::string>& document);
    
    void clear();

   //get_tf
    std::vector<real_t> get_tf( )const;

  //get_idf
    std::vector<real_t> get_idf( )const;

  //get_tfidf 
    std::vector<std::vector<real_t>> get_tfidf( )const;

  //get_vocabulary
    const std::vector<std::string>& get_vocabulary()const ;

  //get_documents
    const std::vector<std::string>& get_documents()const;
  
  
  //get_tf_matrix
    std::vector<std::vector<real_t>> get_tf_matrix();

  //get_idf_matrix
    std::vector<std::vector<real_t>> get_idf_matrix();

  //get_tf_idf_matrix
    std::vector<std::vector<real_t>> get_tf_idf_matrix();

};

class tfidf_vectorizer : public vectorizer<std::string, real_t>
{
  protected:
  class tfidf _tfidf;
  typedef vectorizer<std::string, real_t> base_t;
  public:
  
  tfidf_vectorizer();
  tfidf_vectorizer(const tfidf_vectorizer &other);
  tfidf_vectorizer( const std::vector<std::string> &corpus );
  tfidf_vectorizer(tfidf_vectorizer &&other); //move constructor
  tfidf_vectorizer& operator= (const tfidf_vectorizer &other);
  tfidf_vectorizer&
      operator= (tfidf_vectorizer &&other);
  
    virtual  std::vector<std::vector<real_t>> fit( const std::vector<std::vector<std::string>> &documents )
    {
      std::vector<std::string> corpus;
      for (auto &document : documents)
      {
        for (auto &word : document)
        {
          corpus.push_back(word);
        }
      }
      _tfidf.set_documents(corpus);
      _tfidf.process_documents();

      return _tfidf.get_tfidf();
         
   }
  //override get output size:
  virtual size_t get_output_size()const override
  {
    
    return this->_tfidf.get_vocabulary().size();
  }


  virtual  std::vector<real_t> fit( const std::vector<std::string>&documents )override;
  virtual  std::vector<real_t> predict(const std::vector<std::string>&documents )override;
  virtual  std::vector<real_t> transform(const std::vector<std::string>&documents)override;
  virtual  std::vector<real_t> fit_transform(const std::vector<std::string>&documents)override;
 
  //for use with inverse transformation matrices 
  virtual std::vector<real_t> fit( const provallo::matrix<real_t>& )override;
  virtual std::vector<real_t> predict(const provallo::matrix<real_t>& )override;
  virtual std::vector<real_t> transform(const provallo::matrix<real_t>& )override;
  virtual ~tfidf_vectorizer();
  virtual void gnuplot(const std::string& filename)override
  {
    // describe get tf,idf and vocabulary  
    std::ofstream ofs(filename);
    ofs<<"set terminal png"<<std::endl;
    ofs<<"set output 'tfidf_vectorizer.png'"<<std::endl;
    ofs<<"plot '-' with lines"<<std::endl;
    for (auto& d : _tfidf.get_tf())
    {
      ofs<<d<<std::endl;
    }
  }
    //case by case

  virtual std::vector<real_t> predict (const std::string&)  ;
  virtual std::vector<real_t> transform(const std::string&);

  protected: 
  

};
class scaler;
class standard_scaler_vectorizer : public vectorizer<std::string, real_t>
{
  protected:
  std::vector<real_t> _mean;
  std::vector<real_t> _variance;
  std::vector<real_t> _standard_deviation;
  std::vector<real_t> _standardized_data;

  public:

  standard_scaler_vectorizer();
  standard_scaler_vectorizer(standard_scaler_vectorizer &&other); //move constructor
  standard_scaler_vectorizer& operator= (const standard_scaler_vectorizer &other);
  standard_scaler_vectorizer&
      operator= (standard_scaler_vectorizer &&other);
  
  //
  virtual  std::vector<real_t> fit( const std::vector<std::string>&documents );
  virtual  std::vector<real_t> predict(const std::vector<std::string>&documents );
  virtual  std::vector<real_t> transform(const std::vector<std::string>&documents);
  virtual  std::vector<real_t> fit_transform(const std::vector<std::string>&documents);
 
  //for use with inverse transformation matrices 
  
  virtual  std::vector<real_t> fit( const provallo::matrix<real_t>& );
  virtual  std::vector<real_t> predict(const provallo::matrix<real_t>& );
  virtual  std::vector<real_t> transform(const provallo::matrix<real_t>& );
   
  virtual std::vector<real_t> predict (const std::string &doc);

  virtual ~standard_scaler_vectorizer();
};


class min_max_scaler_vectorizer : public vectorizer<std::string, real_t>
{
  protected:
  std::vector<real_t> _min;
  std::vector<real_t> _max;
  std::vector<real_t> _min_max_data;

  public:



  min_max_scaler_vectorizer();
  min_max_scaler_vectorizer(min_max_scaler_vectorizer &&other); //move constructor
  min_max_scaler_vectorizer& operator= (const min_max_scaler_vectorizer &other);
  min_max_scaler_vectorizer&
      operator= (min_max_scaler_vectorizer &&other);


  virtual  std::vector<real_t> fit( const std::vector<std::string>&documents );
  virtual  std::vector<real_t> predict(const std::vector<std::string>&documents );
  virtual  std::vector<real_t> transform(const std::vector<std::string>&documents);
  virtual  std::vector<real_t> fit_transform(const std::vector<std::string>&documents);
 
  //for use with inverse transformation matrices 
  virtual  std::vector<real_t> fit( const provallo::matrix<real_t>& );
  virtual  std::vector<real_t> predict(const provallo::matrix<real_t>& );
  virtual  std::vector<real_t> transform(const provallo::matrix<real_t>& );
 

  virtual ~min_max_scaler_vectorizer();
};

class normalizer_vectorizer : public vectorizer<std::string, real_t>
{
  protected:
  std::vector<real_t> _norm;
  std::vector<real_t> _normalized_data;
  


  public:

  normalizer_vectorizer();
  normalizer_vectorizer(normalizer_vectorizer &&other); //move constructor
  normalizer_vectorizer& operator= (const normalizer_vectorizer &other);
  normalizer_vectorizer&
      operator= (normalizer_vectorizer &&other);
  
  virtual std::vector<real_t> fit( const provallo::matrix<real_t>& );
  virtual std::vector<real_t> predict(const provallo::matrix<real_t>& );
  virtual std::vector<real_t> transform(const provallo::matrix<real_t>& );
 

  virtual  std::vector<real_t> fit( const std::vector<std::string>&documents );
  virtual  std::vector<real_t> predict(const std::vector<std::string>&documents );
  virtual  std::vector<real_t> transform(const std::vector<std::string>&documents);
  virtual  std::vector<real_t> fit_transform(const std::vector<std::string>&documents);
 
  //for use with inverse transformation matrices 
  virtual ~normalizer_vectorizer();

 
};
//BoW 
class bag_of_words 
{
  //bag of words
  public:
 
  size_t get_number_of_words() const;
  size_t get_number_of_documents() const;
  size_t get_number_of_unique_tokens()const;
  size_t get_number_of_tokens() const;
  const std::vector<std::string>& get_vocabulary() const;
  std::vector<real_t> get_bag_of_words() const; 
  std::vector<real_t> get_document_vector(const std::string&doc) const; 
  const matrix<real_t>& get_matrix() const{ return _bow_matrix; } 
  //clear
  void clear();

  void process_documents();

  bag_of_words();


  //initialize with a vocabulary
  explicit bag_of_words(const std::vector<std::string>&);
  //copy constructor
  bag_of_words(const bag_of_words &other);
  bag_of_words(bag_of_words &&other); //move constructor
  bag_of_words& operator= (const bag_of_words &other);
  bag_of_words&
      operator= (bag_of_words &&other);
  //fit
    std::vector<real_t> fit(const std::string &doc);
    std::vector<std::vector<real_t>> fit(const std::vector<std::string>&documents);
    std::vector<std::vector<real_t>> fit(const provallo::matrix<real_t>&);
  //transform
    std::vector<real_t> transform(const std::string&doc);

    std::vector<std::vector<real_t>> transform(const std::vector<std::string>&documents);
    std::vector<std::vector<real_t>> transform(const provallo::matrix<real_t>&);
  //fit_transform
    std::vector<std::vector<real_t>> fit_transform(const std::vector<std::string>&documents);
    std::vector<std::vector<real_t>> fit_transform(const provallo::matrix<real_t>&);
  //inverse_transform
  //virtual std::string inverse_transform( real_t bow_value);
    std::string inverse_transform(const std::vector<real_t>&); 
    std::vector<std::string> inverse_transform(const std::vector<std::vector<real_t>>&);
    std::vector<std::string> inverse_transform(const provallo::matrix<real_t>&);
  //predict
    std::vector<std::vector<real_t>> predict(const std::vector<std::string>&documents);
    std::vector<real_t> predict(const std::string& doc);
    std::vector<std::vector<real_t>> predict(const provallo::matrix<real_t>&);
     ~bag_of_words()= default;
  virtual void add_document(const std::string&);
  virtual void process_document(const std::string&);
  virtual void process_documents(const std::vector<std::string>&);  
  //dump
  virtual void dump(std::ostream& os) const;
    //dump
  friend std::ostream& operator<<(std::ostream& os, const bag_of_words& bow); 
  //save
  friend std::ofstream& operator<<(std::ofstream& os, const bag_of_words& bow); 
  //load
  friend std::istream& operator>>(std::istream& is, bag_of_words& bow);
  //explicit load
  virtual void load(std::ifstream& is);
  //explicit save
  virtual void save(std::ofstream& os) const;
  //gnuplot - save simple word cloud on gnuplot.
  virtual void gnuplot(const std::string& filename)
  {
    std::string tmp = filename+".dat";

    {

    std::ofstream ofs(tmp);
    //reverse transform the _bow_matrix
    //save _bow 
    size_t i=0;
    const real_t epsilon = 1e-6;
    ofs<<"#word frequency size"<<std::endl;
    for ( auto& word : _vocabulary) 
    {
      auto& frequency  = _bow[i++];
      //avoid :enhanced text mode parser - ignoring spurious text
      if (frequency==0) continue;

      size_t size = word.size()*frequency/(_vocabulary.size()-1.0);
      if (size==0) size=1;

      ofs<<std::to_string(i)<<" "<< std::to_string(frequency+epsilon)<<  " "<<std::to_string( size )<< std::endl;

     }

     ofs.close();
    //gnuplot
    }

    std::ofstream ofs(filename);
    
    ofs<<"set terminal png"<<std::endl;
    //create a gif animation:
    ofs<<"set terminal gif animate delay 10"<<std::endl;
    ofs<<"set output '"<< filename<<"_bow.gif'"<<std::endl;
    ofs<<"set title 'Bag of Words'"<<std::endl;
    ofs<<"set xlabel 'Words'"<<std::endl;
    ofs<<"set ylabel 'Frequency'"<<std::endl;
    ofs<<"set zlabel 'Size'"<<std::endl;
    ofs<<"set style fill solid"<<std::endl;
    //set rotation
    ofs<<"set view 60,30,1,1"<<std::endl;
    ofs<<"set xrange [0:]"<<std::endl;
    ofs<<"set yrange [0:]"<<std::endl;
    ofs<<"set zrange [0:]"<<std::endl;
    ofs<<"set grid"<<std::endl;
    ofs<<"set key off"<<std::endl;
    ofs<<"set dgrid3d 1000,1000,1000"<<std::endl;
    ofs<<"set hidden3d"<<std::endl;
    ofs<<"set palette rgbformulae 22,13,-31"<<std::endl;
    ofs<<"set pm3d depthorder"<<std::endl;
    ofs<<"set pm3d interpolate 0,0"<<std::endl;
    ofs<<"set pm3d at b"<<std::endl;
    ofs<<"set pm3d corners2color c1"<<std::endl;
    //ofs<<"set pm3d lighting phong specular 0.5"<<std::endl;
    //plot the .dat file
    //ofs<<"splot '"<<tmp<<"' using 1:2:3   with points pt 7 ps 1.5 lc rgb variable"<<std::endl;
    //rotate view every 10 degrees 
    
    
    //plot animated gif

    ofs<<"set terminal gif animate delay 10"<<std::endl;
    ofs<<"set output '"<< filename<<"_bow.gif'"<<std::endl;
    
    ofs<<"do for [i=0:360] {"<<std::endl;
    ofs<<"set view i,30,1,1"<<std::endl;
    ofs<<"splot '"<<tmp<<"' using 1:2:3:3 with points pointtype 7   notitle"<<std::endl;
    
    ofs<<"}"<<std::endl;
    ofs<<"unset multiplot"<<std::endl;
    ofs<<"unset pm3d"<<std::endl;

    
    ofs<<"unset dgrid3d"<<std::endl;

    //finish:
    ofs<<"unset output"<<std::endl;
    ofs<<"unset terminal"<<std::endl;
    ofs.close();
    //splot the data
    //colors : 0.01-0.1 yellow, 0.1-0.5 green, 0.5-0.9 blue, 0.9-1.0 red
    //set palette rgbformulae 22,13,-31
    // fix both constant expression required for rgb,and too many axis requested  (>7e+03) :
    //ofs<<"splot '"<<tmp<<"' using 1:2:3   with points pt 7 ps 1.5 lc rgb variable"<<std::endl;
    //avoid  warning: Too many axis ticks requested (>7e+03)
    
 
  }
  protected:
  //bag of words
  std::vector<std::string> _vocabulary;
  std::vector<real_t> _bow;
  std::vector<std::vector<real_t>> _bow_transformed;
  std::vector<std::vector<real_t>> _bow_transformed_inverse;

  matrix<real_t> _bow_matrix;

  size_t num_classes;
  size_t num_features;
  size_t num_words;
  size_t num_docs;
  size_t num_samples;
  size_t num_tokens;
  size_t num_unique_tokens;
 
};



class hashed_bag_of_words 
{
  //bag of words
  public:
 
  size_t get_number_of_words() const ;
  size_t get_number_of_documents() const;
  size_t get_number_of_unique_tokens()const;
  size_t get_number_of_tokens() const;
  const std::vector<std::string>& get_vocabulary() const;
  std::vector<real_t> get_bag_of_words() const{ return _bow;}
  std::vector<real_t> get_document_vector(const std::string&doc) const; 
  const matrix<real_t>& get_matrix() const{ return _bow_matrix; } 
  //clear
  void clear();

  void process_documents(){}; //do nothing

  hashed_bag_of_words();    //initialize with a vocabulary
  explicit hashed_bag_of_words(const std::vector<std::string>&);
  //copy constructor
  hashed_bag_of_words(const hashed_bag_of_words &other);
  hashed_bag_of_words(hashed_bag_of_words &&other); //move constructor
  hashed_bag_of_words& operator= (const hashed_bag_of_words &other);
  hashed_bag_of_words&
      operator= (hashed_bag_of_words &&other);
  //fit
    std::vector<real_t> fit(const std::string &doc);
    std::vector<std::vector<real_t>> fit(const std::vector<std::string>&documents);
    std::vector<std::vector<real_t>> fit(const provallo::matrix<real_t>&);  
  //transform
    std::vector<real_t> transform(const std::string&doc);

    std::vector<std::vector<real_t>> transform(const std::vector<std::string>&documents);
    std::vector<std::vector<real_t>> transform(const provallo::matrix<real_t>&);  
  //fit_transform
    std::vector<std::vector<real_t>> fit_transform(const std::vector<std::string>&documents);
    std::vector<std::vector<real_t>> fit_transform(const provallo::matrix<real_t>&);  
  //inverse_transform
  //virtual std::string inverse_transform( real_t bow_value);
    std::string inverse_transform(const std::vector<real_t>&); 
    std::vector<std::string> inverse_transform(const std::vector<std::vector<real_t>>&);
    std::vector<std::string> inverse_transform(const provallo::matrix<real_t>&);  
  //predict
    std::vector<std::vector<real_t>> predict(const std::vector<std::string>&documents);
    std::vector<real_t> predict(const std::string& doc);
    std::vector<std::vector<real_t>> predict(const provallo::matrix<real_t>&);  
     ~hashed_bag_of_words()= default;
  virtual void add_document(const std::string&);
  virtual void process_document(const std::string&);
  virtual void process_documents(const std::vector<std::string>&);
  //dump
  virtual void dump(std::ostream& os) const;
    //dump    
  friend std::ostream& operator<<(std::ostream& os, const hashed_bag_of_words& bow);
  //save
  friend std::ofstream& operator<<(std::ofstream& os, const hashed_bag_of_words& bow);
  //load
  friend std::istream& operator>>(std::istream& is, hashed_bag_of_words& bow);
  //explicit load
  virtual void load(std::ifstream& is);
  //explicit save
  virtual void save(std::ofstream& os) const;
  //gnuplot - save simple word cloud on gnuplot : 
    virtual void gnuplot(const std::string& filename)
    {   
      std::string tmp = filename+".dat";
      {
      std::ofstream ofs(tmp);
      //reverse transform the _bow_matrix
      //save _bow 
      size_t i=0;
      const real_t epsilon = 1e-6;
      ofs<<"#word frequency size"<<std::endl;
      for ( auto& word : _vocabulary) 
      {
        auto& frequency  = _bow[i++];
        //avoid :enhanced text mode parser - ignoring spurious text
        if (frequency==0) continue;

        size_t size = word.size()*frequency/(_vocabulary.size()-1.0);
        if (size==0) size=1;

        ofs<<std::to_string(i)<<" "<< std::to_string(frequency+epsilon)<<  " "<<std::to_string( size )<< std::endl;

      }

      ofs.close();
      //gnuplot
      }

      std::ofstream ofs(filename);

      ofs<<"set terminal png"<<std::endl;
      //create a gif animation:
      ofs<<"set terminal gif animate delay 10"<<std::endl;
      ofs<<"set output '"<< filename<<"_bow.gif'"<<std::endl;
      ofs<<"set title 'Bag of Words'"<<std::endl;
      ofs<<"set xlabel 'Words'"<<std::endl;
      ofs<<"set ylabel 'Frequency'"<<std::endl;
      ofs<<"set zlabel 'Size'"<<std::endl;
      ofs<<"set style fill solid"<<std::endl;
      //set rotation
      ofs<<"set view 60,30,1,1"<<std::endl;
      ofs<<"set xrange [0:]"<<std::endl;
      ofs<<"set yrange [0:]"<<std::endl;
      ofs<<"set zrange [0:]"<<std::endl;
      ofs<<"set grid"<<std::endl;
      ofs<<"set key off"<<std::endl;
      ofs<<"set dgrid3d 1000,1000,1000"<<std::endl;
      ofs<<"set hidden3d"<<std::endl;
      ofs<<"set palette rgbformulae 22,13,-31"<<std::endl;
      ofs<<"set pm3d depthorder"<<std::endl;
      ofs<<"set pm3d interpolate 0,0"<<std::endl;
      ofs<<"set pm3d at b"<<      std::endl;  
      ofs<<"set pm3d corners2color c1"<<std::endl;
      //ofs<<"set pm3d lighting phong specular 0.5"<<std::endl; 



      //plot the .dat file  
      //ofs<<"splot '"<<tmp<<"' using 1:2:3   with points pt 7 ps 1.5 lc rgb variable"<<std::endl;  
      //rotate view every 10 degrees
      ofs<<"set terminal gif animate delay 10"<<std::endl;
      ofs<<"set output '"<< filename<<"_bow.gif'"<<std::endl;
      
      ofs<<"do for [i=0:360] {"<<std::endl;
      ofs<<"set view i,30,1,1"<<std::endl;
      ofs<<"splot '"<<tmp<<"' using 1:2:3:3 with points pointtype 7   notitle"<<std::endl;
      
      ofs<<"}"<<std::endl;
      ofs<<"unset multiplot"<<std::endl;
      ofs<<"unset pm3d"<<std::endl;
      
    }
  private: 
  //bag of words
  std::vector<std::string> _vocabulary;
  std::vector<real_t> _bow;
  matrix<real_t> _bow_matrix;

  //hashes :
  std::map<uint64_t, size_t> _hash_map;
    std::vector<uint64_t> _hashes;



  //hash functions

  uint64_t hash(const std::string& word);
  size_t hash_index(const std::string& word);
  size_t hash_index(uint64_t hash);



  size_t num_classes;
  size_t num_features;
  size_t num_words;
  size_t num_docs;
  size_t num_samples;
  size_t num_tokens;
  size_t num_unique_tokens;
  
  //inverses and transformations
  std::vector<std::vector<real_t>> _bow_transformed;
  std::vector<std::vector<real_t>> _bow_transformed_inverse;
  std::vector<uint64_t> _hashes_transformed;
  std::vector<uint64_t> _hashes_transformed_inverse;

};
//one hot vectorizer helper class
class one_hot_vectorizer :   public vectorizer<std::string, real_t>
{
  public:
  one_hot_vectorizer();
  //initialize with a vocabulary
  explicit one_hot_vectorizer(const std::vector<std::string>&); 
  //copy constructor
  one_hot_vectorizer(const one_hot_vectorizer &other);
  one_hot_vectorizer(one_hot_vectorizer &&other); //move constructor
  one_hot_vectorizer& operator= (const one_hot_vectorizer &other);
  one_hot_vectorizer&
      operator= (one_hot_vectorizer &&other);
  //fit
  //override vectorizer vtable : 
      //fit:
    virtual  std::vector<std::vector<real_t>> fit( const std::vector<std::vector<std::string>>&documents ); 
    virtual  std::vector<real_t> fit( const std::vector<std::string>& data_);
    virtual  std::vector<real_t> predict(const std::vector<std::string>& data_);//{DEFAULT_IMPL(data_);}
    virtual  std::vector<real_t> transform(const std::vector<std::string>& data_);//{DEFAULT_IMPL(data_);}
    virtual  std::vector<real_t> fit_transform(const std::vector<std::string>& data_);//{ DEFAULT_IMPL(data_);} 
    //inverse:
    virtual  std::vector<real_t> fit( const provallo::matrix<real_t>&data_ );//{DEFAULT_IMPL(data_);}
    virtual  std::vector<real_t> predict(const provallo::matrix<real_t>& data_);//{ DEFAULT_IMPL(data_);}
    virtual  std::vector<real_t> transform(const provallo::matrix<real_t>& data_);///{ DEFAULT_IMPL(data_);}
    
    virtual  std::vector<real_t> fit_transform(const provallo::matrix<real_t>& data_);//{ DEFAULT_IMPL(data_);}  
    //inverse:
    virtual  std::vector<std::string> inverse_transform(const std::vector<real_t>& data_);//{ DEFAULT_IMPL(data_);}
    virtual  std::vector<std::string> inverse_transform(const provallo::matrix<real_t>& data_);//{ DEFAULT_IMPL(data_);}
    //predict:
    virtual  std::vector<real_t> predict(const std::string& data_);
  virtual   ~one_hot_vectorizer();
  //helper functions : 
  void clear();
  size_t get_num_of_samples()const;
  size_t get_num_of_tokens()const;
  size_t get_num_of_unique_tokens()const;
  size_t get_num_of_words()const;
  size_t get_num_of_docs()const;
  size_t get_num_of_features()const;
  size_t get_num_of_classes()const;

  protected:
  //bag of words
  bag_of_words _bow;
  std::vector<std::string> _vocabulary;
  size_t num_classes;
  size_t num_features;
  size_t num_words;
  size_t num_docs;
  size_t num_samples;
  size_t num_tokens;
  size_t num_unique_tokens;

  matrix<real_t> _matrix;

  //initialize 
  void initialize(); 
  //add document and process document helper functions 
  virtual void add_document(const std::string& doc);
  //override process document
  virtual void process_document(const std::string& doc);
  //override process_documents 
  virtual void process_documents(const std::vector<std::string>& docs); 

  virtual std::vector<real_t> fit(const std::string& single_doc);

  virtual void process_documents()
  {
    //add data_src if not added already
    if(_data.length()>0)
    {
      _bow.add_document(_data);

    }
  
      


    _bow.process_documents();
    this->num_docs = _bow.get_number_of_documents();
    this->num_words = _bow.get_number_of_words();
    this->num_tokens = _bow.get_number_of_tokens();
    this->num_unique_tokens = _bow.get_number_of_unique_tokens();
    this->num_samples = _bow.get_number_of_documents();

    _matrix = _bow.get_matrix();
    _transformed_data = _bow.get_bag_of_words();
    _fitted_data = _bow.get_bag_of_words();
    _predicted_data = _bow.get_bag_of_words();
    //normalize data to one-hot vector
    for (size_t i = 0; i < _transformed_data.size(); i++)
    {
      _transformed_data[i] = _transformed_data[i] > 0 ? 1.0 : 0.0;
    }
    for (size_t i = 0; i < _fitted_data.size(); i++)
    {
      _fitted_data[i] = _fitted_data[i] > 0 ? 1.0 : 0.0;
    }
    for (size_t i = 0; i < _predicted_data.size(); i++)
    {
      _predicted_data[i] = _predicted_data[i] > 0 ? 1 : 0.0;
    }

    //end of normalize data to one-hot vector


    
  }
  virtual void gnuplot(const std::string& filename)
  {
    //draw onehot vector  
    //draw bag of words
    _bow.gnuplot(filename+"_bow_cloud.gnuplot");
    
    //draw onehot vector:

    std::ofstream ofs(filename+"_onehot.dat");
    ofs<<"#index value"<<std::endl;
    for (size_t i = 0; i < _transformed_data.size(); i++)
    {
      ofs<<std::to_string(i)<<" "<<std::to_string(_transformed_data[i])<<std::endl;
    }
    ofs.close();
    
    //gnuplot
    ofs.open(filename+"_onehot.gnuplot");
    ofs<<"set terminal png"<<std::endl;

    ofs<<"set output '"<<filename<<"_onehot.png'"<<std::endl;
    ofs<<"set title 'One Hot Vector'"<<std::endl;
    ofs<<"set xlabel 'Words'"<<std::endl;

    ofs<<"set ylabel 'Frequency'"<<std::endl;
    ofs<<"set grid"<<std::endl;
    ofs<<"set key off"<<std::endl;
    ofs<<"set xrange [0:"<<_transformed_data.size()<<"]"<<std::endl;
    ofs<<"set yrange [0:1]"<<std::endl;
    ofs<<"set xtics 1"<<std::endl;
    ofs<<"set ytics 0.1"<<std::endl;
    ofs<<"set style fill solid"<<std::endl;
    ofs<<"plot '"<<filename<<"_onehot.dat' using 1:2 with lines title 'onehot'"<<std::endl;
    ofs.close();
    


  }
 };


//PCA vectorizer helper class
class principal_component_analysis
{
  
  public:
  principal_component_analysis();
  explicit principal_component_analysis(const std::vector<std::string>&);
  explicit principal_component_analysis(const std::vector<std::vector<real_t>>&);
  explicit principal_component_analysis(const provallo::matrix<real_t>&);
  principal_component_analysis(const principal_component_analysis &other);

  principal_component_analysis(principal_component_analysis &&other); //move constructor
  principal_component_analysis& operator= (const principal_component_analysis &other);
  principal_component_analysis&
      operator= (principal_component_analysis &&other);

  //fit 
  virtual std::vector<real_t> fit( const provallo::matrix<real_t>& );
  virtual std::vector<real_t> predict(const provallo::matrix<real_t>& );
  virtual std::vector<real_t> transform(const provallo::matrix<real_t>& );

 //strings return fit({tokens:string1})

  virtual std::vector<real_t> fit( const std::string& );
  virtual std::vector<real_t> predict( const std::string& );
  virtual std::vector<real_t> transform( const std::string& );

// 'document' return fit({tokens:s1,tokens:s2,...sN})
  
  virtual std::vector<std::vector<real_t>> fit( const std::vector<std::string>& );
  virtual std::vector<std::vector<real_t>> predict(const std::vector<std::string>& );
  virtual std::vector<std::vector<real_t>> transform(const std::vector<std::string>& );

  //documents return fit{documents1,documents2,...documentsN}

  virtual std::vector<std::vector<real_t>>  fit( const std::vector<std::vector<std::string>>& );
  virtual std::vector<std::vector<real_t>>  predict(const std::vector<std::vector<std::string>>& );
  virtual std::vector<std::vector<real_t>>  transform(const std::vector<std::vector<std::string>>&);

  void QRDecomposition(const matrix<real_t>& mtx,matrix<real_t>& Q,  matrix<real_t>& R); 

  std::vector<real_t> mul ( const matrix<real_t>& mtx, const std::vector<real_t>& vec)
  {
    std::vector<real_t> result;
    for (size_t i= 0; i < mtx.rows(); i++)
    {
      real_t sum = 0;
      for (size_t j = 0; j < mtx.cols(); j++)
      {
        sum += mtx(i,j) * vec[j];
      }
      result[i]=sum;
    }
    return result;
  }

  //update pca data when fitting :
  void update_pca( const std::vector<real_t>& data);
  void gnuplot(const std::string& filename)
  {
    //save data in a file
    std::ofstream ofs(filename+".dat");
    //save scatter plot data
    //set scatter plot data : 
     /*
    std::array<std::vector<real_t>,9> _scatter_plot_data;
    _scatter_plot_data[0] = _covariance_matrix;
    _scatter_plot_data[1] = _eigen_values;
    _scatter_plot_data[2] = _eigen_vectors;
    _scatter_plot_data[3] = _pca_components;
    _scatter_plot_data[4] = _pca_explained_variance;

    _scatter_plot_data[5] = _pca_singular_values;
 
    _scatter_plot_data[6] = _pca_components;
    _scatter_plot_data[7] = _pca_explained_variance;
    _scatter_plot_data[8] = _pca_singular_values;

    */
     //end of set scatter plot data

    
    
    ofs<<"#x y value covariance eigen_values eigen_vectors pca_components pca_explained_variance pca_singular_values"<<std::endl; 

    //x - sample index
    //y - feature index
    //value - value of the feature
    //covariance - covariance matrix at (x,y)
    //eigen_values - eigen values at (x,y)
    //eigen_vectors - eigen vectors at (x,y)
    //pca_components - pca components at (x,y)
    //pca_explained_variance - pca explained variance at (x,y)

    //pca_singular_values - pca singular values at (x,y)
    //set scatter plot data
     
    //end of save scatter plot data
    ofs.close();
    //gnuplot
    std::ofstream ofs2(filename);
    ofs2<<"set terminal png"<<std::endl;
    ofs2<<"set output '"<<filename<<"_pca.png'"<<std::endl;
    ofs2<<"set title 'Principal Component Analysis'"<<std::endl;
    ofs2<<"set xlabel 'Data'"<<std::endl;
    ofs2<<"set ylabel 'Components'"<<std::endl;
    ofs2<<"set zlabel 'Variance'"<<std::endl; 
    ofs2<<"set style fill solid"<<std::endl;
    ofs2<<"set pm3d depthorder"<<std::endl;
    //set rotation
    ofs2<<"set view 60,30,1,1"<<std::endl;
    ofs2<<"set xrange [0:]"<<std::endl;
    ofs2<<"set yrange [0:]"<<std::endl;
    ofs2<<"set zrange [0:]"<<std::endl;
    ofs2<<"set grid"<<std::endl;
    ofs2<<"set key off"<<std::endl;
    ofs2<<"set dgrid3d 1000,1000,1000"<<std::endl;
    ofs2<<"set hidden3d"<<std::endl;
    ofs2<<"set palette rgbformulae 22,13,-31"<<std::endl;
    ofs2<<"set pm3d depthorder"<<std::endl;
    ofs2<<"set pm3d interpolate 0,0"<<std::endl;
    ofs2<<"set pm3d at b"<<std::endl;
    ofs2<<"set pm3d corners2color c1"<<std::endl;
    //ofs2<<"set pm3d lighting phong specular 0.5"<<std::endl;
    //plot the .dat file:
    //set multiplot 
    ofs2<<"set multiplot"<<std::endl;
    //set contour
    ofs2<<"set contour"<<std::endl;
    
    //plot scatter, with eigen values,eigen vectors, scatter plot data, pca components, explained variance, singular values 
    ofs2<<"splot '"<<filename<<".dat' using 1:2:3:3 with points pointtype 7   notitle"<<std::endl;
    //plot covariance matrix
    ofs2<<"splot '"<<filename<<".dat' using 1:2:4:4 with points pointtype 7   notitle"<<std::endl;
    //plot eigen values
    ofs2<<"splot '"<<filename<<".dat' using 1:2:5:5 with points pointtype 7   notitle"<<std::endl;
    //plot eigen vectors
    ofs2<<"splot '"<<filename<<".dat' using 1:2:6:6 with points pointtype 7   notitle"<<std::endl;
    //plot pca components
    ofs2<<"splot '"<<filename<<".dat' using 1:2:7:7 with points pointtype 7   notitle"<<std::endl;
    //plot explained variance
    ofs2<<"splot '"<<filename<<".dat' using 1:2:8:8 with points pointtype 7   notitle"<<std::endl;
    //plot singular values
    ofs2<<"splot '"<<filename<<".dat' using 1:2:9:9 with points pointtype 7   notitle"<<std::endl;
    //end of plot scatter, with eigen values,eigen vectors, scatter plot data, pca components, explained variance, singular values
    //end of plot the .dat file
    //end of set multiplot
    ofs2<<"unset multiplot"<<std::endl;
    ofs2<<"unset pm3d"<<std::endl;
    ofs2<<"unset dgrid3d"<<std::endl;
    //finish:
    ofs2<<"unset output"<<std::endl;
    ofs2<<"unset terminal"<<std::endl;
    ofs2.close();
    
  }

  void gauss_jordan_elimination(matrix<real_t>&elim  )
  {
      for ( size_t col =0;col < elim.cols();col++)
      {


          size_t max_v = col;
          real_t max_value = elim(col,col);

          for (size_t row =col + 1;row < elim.rows();row++)
          {
              if ( fabs ( elim(row,col) ) > fabs(max_value) ) 
              {

                  max_value = elim(row,col);
                  max_v = row;
              }
              
          	  if(row!=max_v)
	          {
        	          for (size_t ix = 0; ix < elim.rows(); ++ix)
	                    std::swap(elim.element(col,ix), elim.element (max_v,ix));

	          }
	  }

          

    //scale
    real_t scale = elim(col,col);
    for ( size_t ix = 0;ix < elim.rows();ix++)
    {
        elim.element(col,ix) /= scale;
    }
    for ( size_t iy=0;iy < elim.rows();iy++)
    {
        if ( iy != col)
        {
            real_t scale = elim(iy,col);
            if(scale==0.0)continue;

            for ( size_t ix = 0;ix < elim.rows();ix++)
            {
                elim.element(iy,ix) -= scale * elim.element(col,ix);
            }
        }
    }
    //end of scale


     }//end of for loop

  } //end of gauss_jordan_elimination


  matrix<real_t> gram_schmidt(const matrix<real_t>& mtx)
  {
    matrix<real_t> result(mtx.rows(),mtx.cols());
    matrix<real_t> Q(mtx.rows(),mtx.cols());
    matrix<real_t> R(mtx.rows(),mtx.cols());
    QRDecomposition(mtx,Q,R);
    result = Q * R;
    return result;
  }   

 
  //for use with inverse transformation matrices 
  //destructor
   virtual ~principal_component_analysis();
   
    
  protected:


    std::vector<real_t> _mean;
    std::vector<real_t> _variance;
    std::vector<real_t> _standard_deviation;
    std::vector<real_t> _standardized_data;
    std::vector<real_t> _covariance_matrix;
    std::vector<real_t> _eigen_values;
    std::vector<real_t> _eigen_vectors;
    std::vector<real_t> _pca_data;
    std::vector<real_t> _pca_components;
    std::vector<real_t> _pca_explained_variance;
    std::vector<real_t> _pca_explained_variance_ratio;
    std::vector<real_t> _pca_singular_values;
    std::vector<real_t> _pca_noise_variance;
    std::vector<real_t> _pca_mean;
    size_t _pca_n_components_;
    size_t  _pca_n_features_;
    size_t  _pca_n_samples_;

    matrix<real_t> _pca_components_matrix;
    matrix<real_t> _pca_explained_variance_matrix;
    matrix<real_t> _pca_explained_variance_ratio_matrix;
    matrix<real_t> _pca_singular_values_matrix;
    //vocabulary

  //pca_mean

   //use bow_vectorizer to transform documents into a matrix
  //use pca to transform matrix into a matrix of principal components
  //vocabularies
  typedef std::map<std::string,size_t> vocabulary_t;
  typedef std::map<size_t,std::string> reverse_vocabulary_t;
  vocabulary_t _vocabulary;
  reverse_vocabulary_t _reverse_vocabulary;   
  std::vector<std::vector<real_t> > _bow_matrix; //bag of words matrix
  std::vector<std::string> _words;

  //words vector

};


class pca_vectorizer : public vectorizer<std::string, real_t>
{
  protected:
  public:
  pca_vectorizer();
  pca_vectorizer(pca_vectorizer &&other); //move constructor
  pca_vectorizer& operator= (const pca_vectorizer &other);
  pca_vectorizer&
      operator= (pca_vectorizer &&other);
  virtual  std::vector<real_t> predict(const std::string& data_);

  virtual  std::vector<real_t> fit( const std::vector<std::string>&documents );
  virtual  std::vector<real_t> predict(const std::vector<std::string>&documents );
  virtual  std::vector<real_t> transform(const std::vector<std::string>&documents);
  virtual  std::vector<real_t> fit_transform(const std::vector<std::string>&documents);
 
  //for use with inverse transformation matrices 
  virtual std::vector<real_t> fit( const provallo::matrix<real_t>& ); 
  virtual std::vector<real_t> predict(const provallo::matrix<real_t>& );
  virtual std::vector<real_t> transform(const provallo::matrix<real_t>& );
 


  //vector<vector>> impl simply calls the vector<string> impl for each document.
  //each document contains strings , each string contains tokens, each token is a word. 

  virtual std::vector<std::vector<real_t>>  fit( const std::vector<std::vector<std::string>>& );
  virtual std::vector<std::vector<real_t>>  predict(const std::vector<std::vector<std::string>>& );
  virtual std::vector<std::vector<real_t>>  transform(const std::vector<std::vector<std::string>>&);


  //override get_type
  virtual vectorizer_type get_type() const ;
  
  virtual ~pca_vectorizer();



  //gnu plot:
  virtual void gnuplot(const std::string & filename);

  private:

  principal_component_analysis _pca;
    bag_of_words _bow;//bag of words vectorizer
  
};

class lda_vectorizer : public vectorizer<std::string, real_t>
{
  protected: 

  void initialize();

  public:
  lda_vectorizer();
  lda_vectorizer(lda_vectorizer &&other); //move constructor
  lda_vectorizer& operator= (const lda_vectorizer &other);
  lda_vectorizer&
      operator= (lda_vectorizer &&other);

//fit single  document
  std::vector<real_t> fit(const std::string& single_doc );
  void process_document(const std::string& single_doc ); 
  void process_documents(const std::vector<std::string>& documents );

  //override get_output_size:
  virtual size_t get_output_size()const override
  {
    if(_n_topics>0)
    return _n_topics;
    else if(_fitted_data.size()>0) 
    return _fitted_data.size();
    else if(_lda_data.size()>0) 
    return _lda_data.size();
    else if(_lda_components.size()>0)
    return _lda_components.size();

    return _bow.get_number_of_tokens()  ;
  }

  virtual  std::vector<real_t> fit( const std::vector<std::string>&documents );
  virtual  std::vector<real_t> predict(const std::vector<std::string>&documents );
  virtual  std::vector<real_t> transform(const std::vector<std::string>&documents);
  
  //for use with inverse transformation matrices 
  
  virtual std::vector<real_t> fit( const provallo::matrix<real_t>& );
  virtual std::vector<real_t> predict(const provallo::matrix<real_t>& );
  virtual std::vector<real_t> transform(const provallo::matrix<real_t>& );
   virtual  std::vector<real_t> transform(const std::vector<std::vector<std::string>>&documents);

  //fit_transform 
  std::vector<real_t> fit_transform( const std::vector<std::string>& data_);
  std::vector<std::vector<real_t>> fit_transform( const provallo::matrix<real_t>& data_);


  //for vector<vector<string>> 
  std::vector<std::vector<real_t>> fit( const std::vector<std::vector<std::string>>& data_); 
  std::vector<std::vector<real_t>> predict( const std::vector<std::vector<std::string>>& data_); 
  virtual void process_documents()  ;
  virtual std::vector<real_t> predict(const std::string& data_);
  //virtual std::vector<real_t> fit_transform(const std::vector<std::string>&documents); 
  //virtual std::vector<std::vector<real_t>> fit_transform(const matrix<real_t>&documents); 
  virtual void clear();

  virtual void load(std::ifstream& in);
  //load additional parameters
  virtual void save(std::ofstream& out)const;

  virtual ~lda_vectorizer();
  private:
  bag_of_words  _bow;
  std::vector<std::string> _vocabulary;
  std::vector<std::string> _labels;
  real_t _alpha   =  0.1;
  real_t _beta  =  0.1;
  real_t _eta   =  0.1;
  real_t _gamma =  1.0;
  real_t _theta =  0.5;
  real_t _lambda  =  1.0;
  size_t _n_iter  =  10;
  size_t _n_topics    =  10;
  size_t _n_features        =  1000;
  size_t _n_samples     =  1000 ;
  size_t _n_components        =  10 ;
  size_t _n_top_words   =  10 ;
  size_t _n_jobs    =  1  ;
  size_t _random_state    =  0;
  real_t _doc_topic_prior     =  0.1  ;
  real_t _topic_word_prior        =  0.7  ;
  real_t _learning_decay    =  0.7  ;
  real_t _learning_offset       =  10.0 ;
  size_t _max_doc_update_iter   =  100  ;
  size_t _total_samples   =  0  ;
  real_t _mean_change_tol   =  0.0;
  size_t _num_docs  =  0;
  size_t _num_words=  0;
  size_t  num_unique_tokens=  0;
  size_t _verbose = 0;
  std::vector<real_t> _lda_data;
  std::vector<real_t> _lda_components;
  std::vector<real_t> _lda_explained_variance;
  std::vector<real_t> _lda_explained_variance_ratio;
  std::vector<real_t> _lda_singular_values;
  std::vector<real_t> _lda_noise_variance;
  std::vector<real_t> _lda_mean;
  std::vector<real_t> _lda_covariance;
  std::vector<real_t> _lda_precision;
  std::vector<real_t> _lda_whiten;
  size_t _lda_n_components_ = 0;
  size_t  _lda_n_features_ = 0;
  size_t  _lda_n_samples_  = 0;
  //vocabulary
  //lda
  //bow

   //predict single doc 
  
   friend std::ostream& operator<<(std::ostream& os, const lda_vectorizer& lda)
  {
    // write out individual members of s with an end of line between each one 
    os <<"[lda_vectorizer]"<<std::endl;
    os <<"alpha"<<std::to_string( lda._alpha) <<std::endl;
    os <<"beta"<<std::to_string( lda._beta) <<std::endl;
    os <<"eta"<<std::to_string( lda._eta) <<std::endl;
    os <<"gamma"<<std::to_string( lda._gamma) <<std::endl;
    os <<"theta"<<std::to_string( lda._theta) <<std::endl;
    os <<"lambda"<<std::to_string( lda._lambda) <<std::endl;
    os <<"n_iter"<<std::to_string( lda._n_iter) <<std::endl;
    os <<"n_topics"<<std::to_string( lda._n_topics) <<std::endl;
    os <<"n_features"<<std::to_string( lda._n_features) <<std::endl;
    os <<"n_samples"<<std::to_string( lda._n_samples) <<std::endl;
    os <<"n_components"<<std::to_string( lda._n_components) <<std::endl;
    os <<"n_top_words"<<std::to_string( lda._n_top_words) <<std::endl;
    os <<"n_jobs"<<std::to_string( lda._n_jobs) <<std::endl;
    os <<"random_state"<<std::to_string( lda._random_state) <<std::endl;
    os <<"doc_topic_prior"<<std::to_string( lda._doc_topic_prior) <<std::endl;
    os <<"topic_word_prior"<<std::to_string( lda._topic_word_prior) <<std::endl;
    os <<"learning_decay"<<std::to_string( lda._learning_decay) <<std::endl;
    os <<"learning_offset"<<std::to_string( lda._learning_offset) <<std::endl;
    os <<"max_doc_update_iter"<<std::to_string( lda._max_doc_update_iter) <<std::endl;
    os <<"total_samples"<<std::to_string( lda._total_samples) <<std::endl;
    os <<"mean_change_tol"<<std::to_string( lda._mean_change_tol) <<std::endl;

    os <<"num_docs"<<std::to_string( lda._num_docs) <<std::endl;
    os <<"num_words"<<std::to_string( lda._num_words) <<std::endl;
    os <<"num_unique_tokens"<<std::to_string( lda.num_unique_tokens) <<std::endl;
    os <<"verbose"<<std::to_string( lda._verbose) <<std::endl;
    os <<"lda_data :"<<  lda._lda_data  <<std::endl;
    os <<"lda_components :"<<  lda._lda_components  <<std::endl;
    os <<"lda_explained_variance :"<<  lda._lda_explained_variance  <<std::endl;
    os <<"lda_explained_variance_ratio :"<<  lda._lda_explained_variance_ratio  <<std::endl;


    return os;
  }
   
  //lda
  //bow
  //inverse_transform
  //inverse_transform_matrix
  //inverse_transform_matrix_
  
  //print
  
  virtual void dump(std::ostream& out) const;
  //

  //gnuplot - creates gnuplot instructions to plot lda diagram using the data
  virtual void gnuplot(const std::string& filename)
  {


    //plot _bow 
    _bow.gnuplot(filename+"_bow_cloud.gnuplot");
    //save tmp lda data to file
    {
      std::ofstream ofs(filename+"_lda_data.dat");
      ofs<<"#lda data,lda_component, importance"<<std::endl;
      for (auto& d : _lda_data)
      { 
        auto component_index =  (&d - &_lda_data[0] )% _lda_components.size();
        auto importance = d * _lda_components[component_index];

        ofs<<d<<" " << std::to_string( _lda_components[component_index] )<< " "<<std::to_string(importance)<<std::endl;

        }
      ofs.close();
    }
    //plot _lda_data
    
    std::ofstream ofs(filename);
    ofs<<"reset session"<<std::endl;
    ofs<<"set terminal png"<<std::endl;
    ofs<<"set output '"<<filename.c_str()<<"_lda_vectorizer.png'"<<std::endl;
    ofs<<"set title 'Latent Dirichlet Allocation'"<<std::endl;
    ofs<<"set xlabel 'Samples'"<<std::endl;
    ofs<<"set ylabel 'Features'"<<std::endl;
    ofs<<"set xtics rotate by -45"<<std::endl;
    ofs<<"set grid"<<std::endl;
    ofs<<"set style fill solid"<<std::endl;
    ofs<<"set boxwidth 0.5"<<std::endl;
    //ofs<<"set yrange [0:]"<<std::endl;
    //ofs<<"set xrange [0:"<<_lda_data.size()<<"]"<<std::endl;
    //ofs<<"set zrange [0:]"<<std::endl;

    ofs<<"set key off"<<std::endl;
    ofs<<"set table $LDA_DATA"<<std::endl;
    ofs<<"\t set samples "<<_lda_data.size()<<std::endl;
    //load samples from lda data file  to LDA_DATA table

    ofs<<"\t plot '"<<filename<<"_lda_data.dat' using 1:2:3 with table"<<std::endl;
 
    ofs<<"# for each datapoint: how many other datapoints are within radius R"<<std::endl;
    ofs<<"R = 0.5     # Radius to check"<<std::endl;
    //distribution of x,y coordinates over z plane
    ofs<<"set xrange [-1:1]"<<std::endl;
    ofs<<"set yrange [-1:1]"<<std::endl;
    ofs<<"set size ratio -1   # same screen units for x and y"<<std::endl;
    ofs<<"set palette rgb 33,13,10"<<std::endl;
    ofs<<"Dist(x0,y0,x1,y1) = sqrt((x1-x0)**2 + (y1-y0)**2) "<<std::endl; 

   // ofs<<"set print $LDA_DATA"<<std::endl;
    ofs<<"set print $Density"<<std::endl;

    ofs<<"do for [i=1:|$LDA_DATA|] {"<<std::endl;
    ofs<<"\t x0 = real(word($LDA_DATA[i],1))"<<std::endl;
    ofs<<"\t y0 = real(word($LDA_DATA[i],2))"<<std::endl;
    ofs<<"\t z0 = real(word($LDA_DATA[i],3))"<<std::endl;
    //plot 3d density with x,y,z and density 
    ofs<<"\t c  = 0"<<std::endl;
    ofs<<"\t stats $LDA_DATA u (Dist(x0,y0,$1,$2)<=R ? c=c+1 : 0) nooutput"<<std::endl;
    ofs<<"\t d = c / (pi * R**2)             # density: points per unit area"<<std::endl;
    ofs<<"\t print sprintf(\"%g %g %g %d\", x0, y0, z0, d)"<<std::endl;
    //plot x0,y0,z0 with density d
    ofs<<"}"<<std::endl;
    ofs<<"unset table"<<std::endl;

 
    //plot :
   // ofs<<"set xrange [-1:1]"<<std::endl;
   // ofs<<"set yrange [-1:1]"<<std::endl;
   // ofs<<"set zrange [-1:1]"<<std::endl;
    ofs<<"set palette rgb 33,13,10"<<std::endl;
    ofs<<"set pm3d depthorder"<<std::endl;
    ofs<<"set hidden3d"<<std::endl;

    ofs<<"set dgrid3d 5000,5000 qnorm 2"<<std::endl;
    ofs<<"set ticslevel 0"<<std::endl;

    ofs<<"set view 60,30,1,1"<<std::endl;
    ofs<<"set isosamples 50"<<std::endl;
    ofs<<"set contour base"<<std::endl;
    ofs<<"set cntrparam levels 10"<<std::endl;
    ofs<<"set size ratio -1   # same screen units for x and y"<<std::endl;
    //plot 3d density with x,y,z and density
 //plot $Density :
    ofs<<"plot $Density u 1:2:3:4 w pm3d notitle"<<std::endl;
    ofs.close();
  }//end of gnuplot

  };

//helper class for tsne 
class tsne
{
  private:
  std::vector<real_t> _tsne_data;
  std::vector<real_t> _tsne_components;
  std::vector<real_t> _tsne_explained_variance;
  std::vector<real_t> _tsne_explained_variance_ratio;
  std::vector<real_t> _tsne_singular_values;
  std::vector<real_t> _tsne_noise_variance;
  std::vector<real_t> _tsne_mean;
  std::vector<real_t> _tsne_n_components;
  std::vector<real_t> _tsne_n_features;
  std::vector<real_t> _tsne_n_samples;
  
  public:
  tsne();
  tsne(tsne &&other); //move constructor
  tsne& operator= (const tsne &other);
  tsne&
      operator= (tsne &&other);
  virtual ~tsne();
  //fit
  
  //fit_transform
  
  //get_params

  //inverse_transform

  //set_params

  //transform

  
};

class tsne_vectorizer : public vectorizer<std::string, real_t>
{
  protected:
  std::vector<real_t> _tsne_data;
  std::vector<real_t> _tsne_components;
  std::vector<real_t> _tsne_explained_variance;
  std::vector<real_t> _tsne_explained_variance_ratio;
  std::vector<real_t> _tsne_singular_values;
  std::vector<real_t> _tsne_noise_variance;
  std::vector<real_t> _tsne_mean;
  std::vector<real_t> _tsne_n_components;
  std::vector<real_t> _tsne_n_features;
  std::vector<real_t> _tsne_n_samples;
  std::vector<real_t> _tsne_n_components_;
  std::vector<real_t> _tsne_n_features_;
  std::vector<real_t> _tsne_n_samples_;




  public:
  tsne_vectorizer();
  tsne_vectorizer(tsne_vectorizer &&other); //move constructor
  tsne_vectorizer& operator= (const tsne_vectorizer &other);
  tsne_vectorizer&
      operator= (tsne_vectorizer &&other);
  

    
  virtual  std::vector<real_t> fit( const std::vector<std::string>&documents );
  virtual  std::vector<real_t> predict(const std::vector<std::string>&documents );
  virtual  std::vector<real_t> transform(const std::vector<std::string>&documents);
  virtual  std::vector<real_t> fit_transform(const std::vector<std::string>&documents);
 
  //for use with inverse transformation matrices 
  virtual std::vector<real_t> fit( const provallo::matrix<real_t>& );
  virtual std::vector<real_t> predict(const provallo::matrix<real_t>& );
  virtual std::vector<real_t> transform(const provallo::matrix<real_t>& );
  //load/save 
  virtual void dump(std::ostream& out) const;
  virtual void load(std::ifstream& in);
  virtual void save(std::ofstream& out)const;
  //vector<vector>> impl simply calls the vector<string> impl for each document.
  //each document contains strings , each string contains tokens, each token is a word.
  virtual std::vector<std::vector<real_t>>  fit( const std::vector<std::vector<std::string>>& );
  virtual std::vector<std::vector<real_t>>  predict(const std::vector<std::vector<std::string>>& );
  virtual std::vector<std::vector<real_t>>  transform(const std::vector<std::vector<std::string>>&);
  //override get_type
  virtual vectorizer_type get_type() const ;
  //inverse_transform
  virtual std::vector<std::string> inverse_transform(const std::vector<real_t>& data);
  virtual std::vector<std::string> inverse_transform(const provallo::matrix<real_t>& data);
  //print
  


  //gnuplot - creates gnuplot instructions to plot tsne diagram using the data 

  virtual void gnuplot(const std::string& filename)
  {
    //
    
    std::string dat_filename = filename+".dat";
    {
      std::ofstream ofs(dat_filename);
      ofs<<"#tsne data,tsne_component, importance"<<std::endl;
      for (auto& d : _tsne_data)
      { 
        auto component_index =  (&d - &_tsne_data[0] )% _tsne_components.size();
        auto importance = d * _tsne_components[component_index];

        ofs<<d<<" " << std::to_string( _tsne_components[component_index] )<< " "<<std::to_string(importance)<<std::endl;

        }
      ofs.close();
    }
    //plot _tsne_data
    std::ofstream ofs(filename);
    ofs<<"reset session"<<std::endl;
    ofs<<"set terminal png"<<std::endl;
    ofs<<"set output '"<<filename.c_str()<<"_tsne_vectorizer.png'"<<std::endl;
    ofs<<"set title 't-distributed Stochastic Neighbor Embedding'"<<std::endl;
    ofs<<"set xlabel 'Samples'"<<std::endl;
    ofs<<"set ylabel 'Features'"<<std::endl;
    ofs<<"set xtics rotate by -45"<<std::endl;
    ofs<<"set grid"<<std::endl;
    ofs<<"set style fill solid"<<std::endl;


    ofs<<"set key off"<<std::endl;
    ofs<<"set table $TSNE_DATA"<<std::endl;
    ofs<<"\t set samples "<<_tsne_data.size()<<std::endl;
    //load samples from tsne data file  to TSNE_DATA table

    ofs<<"\t plot '"<<dat_filename<<"' using 1:2:3 with table"<<std::endl;
    //plot :
    ofs<<"splot $TSNE_DATA u 1:2:3 w p pt 7 lc palette z notitle"<<std::endl;
    ofs.close();
    //plot _bow later when bow is implemented
    // _bow.gnuplot(filename+"_bow_cloud.gnuplot");    
    
    //std::ofstream out(filename);
    //draw tsne animation 
    //out<<"reset session"<<std::endl;
    
  }

  virtual ~tsne_vectorizer();
};

  //uniform manifold approximation and projection 
  //helper class for umap
  class umap : public vectorizer<std::string, real_t> 
  {
    private:
    std::vector<real_t> _umap_data;
    std::vector<real_t> _umap_components;
    std::vector<real_t> _umap_explained_variance;
    std::vector<real_t> _umap_explained_variance_ratio;
    std::vector<real_t> _umap_singular_values;
    std::vector<real_t> _umap_noise_variance;
    std::vector<real_t> _umap_mean;
    std::vector<real_t> _umap_n_components;
    std::vector<real_t> _umap_n_features;
    std::vector<real_t> _umap_n_samples;
    std::vector<real_t> _umap_n_components_;
    std::vector<real_t> _umap_n_features_;
    std::vector<real_t> _umap_n_samples_;

    public:
    umap();
    umap(umap &&other); //move constructor
    umap& operator= (const umap &other);
    umap&
        operator= (umap &&other);
    virtual ~umap();
    //fit 

    virtual std::vector<real_t> fit( const std::vector<std::string>&documents ); 
    //transform 
    virtual std::vector<real_t> transform(const std::vector<std::string>&documents); 

    //fit_transform
    virtual std::vector<real_t> fit_transform(const std::vector<std::string>&documents); 
    //get_params

    //inverse_transform
    virtual std::vector<std::string> inverse_transform(const std::vector<real_t>& data);
    virtual std::vector<std::string> inverse_transform(const provallo::matrix<real_t>& data); 

    //override vectorizer vtable :
    //predict
    virtual std::vector<real_t> predict(const std::vector<std::string>&documents );

    //set_params

    //get type
    virtual vectorizer_type get_type() const ;


   };

  enum regressor_type :  uint16_t{ LINEAR_REGRESSION, LDA_REGRESSION, BAYESIAN_REGRESSION, RIDGE_REGRESSION, LASSO_REGRESSION, ELASTIC_NET_REGRESSION, SGD_REGRESSION, PASSIVE_AGGRESSIVE_REGRESSION, RANSAC_REGRESSION, HUBER_REGRESSION, THEIL_SEN_REGRESSION, POLYNOMIAL_REGRESSION, STOCHASTIC_REGRESSION, PERCEPTRON_REGRESSION, LOGISTIC_REGRESSION, SGD_CLASSIFIER, PASSIVE_AGGRESSIVE_CLASSIFIER, PERCEPTRON_CLASSIFIER, KNN_CLASSIFIER, NEAREST_CENTROID_CLASSIFIER, GAUSSIAN_NAIVE_BAYES_CLASSIFIER, MULTINOMIAL_NAIVE_BAYES_CLASSIFIER, BERNOULLI_NAIVE_BAYES_CLASSIFIER, DECISION_TREE_CLASSIFIER, RANDOM_FOREST_CLASSIFIER, EXTRA_TREES_CLASSIFIER, GRADIENT_BOOSTING_CLASSIFIER, ADABOOST_CLASSIFIER, SUPPORT_VECTOR_CLASSIFIER, SUPPORT_VECTOR_REGRESSION,NEURAL_NETWORK_AUTOENCODER_VECTOR_TO_SEQUENCE, NEURAL_NETWORK_AUTOENCODER_VECTOR_TO_VECTOR, NEURAL_NETWORK_AUTOENCODER_SEQUENCE_TO_SEQUENCE };
  //regressor 
  //============================================================================

  //model hirerchy - get coefficients, intercepts, etc. from base class 
  template <typename T, typename U> 
  class model : public vectorizer<T,U> {

      protected:
      bool is_fitted;
      regressor_type _regressor_type;

      std::vector<U> _model_data; 
      std::vector<U> _coefficients;
      std::vector<U> _intercepts;
      std::vector<U> _residuals;
      std::vector<U> _r_squared;
      std::vector<U> _score;
      std::vector<U> _n_iter;
      std::vector<U> _n_features;
      std::vector<U> _n_samples;
      std::vector<U> _n_components;
      std::vector<U> _n_components_;
      std::vector<U> _n_features_;
      std::vector<U> _n_samples_;
      std::vector<U> _loss;
      std::vector<U> _loss_function;

      public:
      model();
      model(model &&other); //move constructor
      model& operator= (const model &other);
      model&
          operator= (model &&other);
      //get coefficients:
      virtual std::vector<U> get_coefficients() const;
      virtual std::vector<U> get_intercepts() const;
      virtual std::vector<U> get_residuals() const;
      virtual std::vector<U> get_r_squared() const;
      virtual std::vector<U> get_score() const;
      virtual std::vector<U> get_n_iter() const;
      virtual std::vector<U> get_n_features() const;
      virtual std::vector<U> get_n_samples() const;
      virtual std::vector<U> get_n_components() const;
      virtual std::vector<U> get_n_components_() const;
      virtual std::vector<U> get_n_features_() const;
      virtual std::vector<U> get_n_samples_() const;
      virtual std::vector<U> get_loss() const;
      virtual std::vector<U> get_loss_function() const;

     //fit matrix, return model
      virtual model<T,U> fit(const provallo::matrix<T>&, const provallo::matrix<T>&); 
      virtual model<T,U> fit(const provallo::matrix<T>&, const std::vector<T>&);
      virtual model<T,U> fit(const provallo::matrix<T>&, const std::vector<T>&, const std::vector<T>&);
      virtual model<T,U> fit(const provallo::matrix<T>&, const std::vector<T>&, const std::vector<T>&, const std::vector<T>&);
      virtual model<T,U> fit(const provallo::matrix<T>&, const std::vector<T>&, const std::vector<T>&, const std::vector<T>&, const std::vector<T>&); 

      //fit vector, return model
      // breaks matrix into attributes + target, then calls fit(matrix, vector or matrix)

      virtual model<T,U> fit(const matrix<T>&);


      //predict matrix, return vector
      virtual std::vector<U> predict(const provallo::matrix<T>&);
      virtual std::vector<U> predict(const std::vector<T>&);
      virtual std::vector<U> predict(const std::vector<T>&, const std::vector<T>&);
      virtual std::vector<U> predict(const std::vector<T>&, const std::vector<T>&, const std::vector<T>&);
      virtual std::vector<U> predict(const std::vector<T>&, const std::vector<T>&, const std::vector<T>&, const std::vector<T>&);
      

      virtual ~model();
            
  }; //end model
  template <class T, class U>
  class ridge_regression_model : public model<T,U> 
  {
    protected:
    //model data resides on base class for easier serialization.
    //

    public:
    ridge_regression_model();
    ridge_regression_model(ridge_regression_model &&other); //move constructor
    ridge_regression_model& operator= (const ridge_regression_model &other);

    ridge_regression_model&
        operator= (ridge_regression_model &&other);
    //get coefficients:
    virtual std::vector<U> get_coefficients() const;
    virtual std::vector<U> get_intercepts() const;

    virtual std::vector<U> get_residuals() const;
    virtual std::vector<U> get_r_squared() const;
    virtual std::vector<U> get_score() const;
    virtual std::vector<U> get_n_iter() const;
    virtual std::vector<U> get_n_features() const;
    virtual std::vector<U> get_n_samples() const;
    virtual std::vector<U> get_n_components() const;


    //set coefficients used for deserialization:

    virtual void set_coefficients(std::vector<U> coefficients);
    virtual void set_intercepts(std::vector<U> intercepts);
    virtual void set_residuals(std::vector<U> residuals);
    virtual void set_r_squared(std::vector<U> r_squared);
    virtual void set_score(std::vector<U> score);
    virtual void set_n_iter(std::vector<U> n_iter);
    virtual void set_n_features(std::vector<U> n_features);
    virtual void set_n_samples(std::vector<U> n_samples);
    virtual void set_n_components(std::vector<U> n_components);

    


    //fit data to model
    virtual std::vector<U> fit( const std::vector<T>& );
    virtual std::vector<U> predict(const std::vector<T>& );
    virtual std::vector<U> transform(const std::vector<T>& );
    virtual std::vector<U> fit_transform(const std::vector<T>& );
    virtual std::vector<U> fit( const std::vector<T>&, const std::vector<U>& );
    virtual std::vector<U> predict(const std::vector<T>&, const std::vector<U>& );
    virtual std::vector<U> transform(const std::vector<T>&, const std::vector<U>& );
    virtual std::vector<U> fit_transform(const std::vector<T>&, const std::vector<U>& );
    virtual ~ridge_regression_model();
    
  };
  template <class T, class U>
  class lasso_regression_model : public model<T,U> 
  {
    protected:
    //model data resides on base class for easier serialization.
    //

    public:
    lasso_regression_model();
    lasso_regression_model(lasso_regression_model &&other); //move constructor
    lasso_regression_model& operator= (const lasso_regression_model &other);

    lasso_regression_model&
        operator= (lasso_regression_model &&other);
    //get coefficients:
    virtual std::vector<U> get_coefficients() const;
    virtual std::vector<U> get_intercepts() const;

    virtual std::vector<U> get_residuals() const;
    virtual std::vector<U> get_r_squared() const;
    virtual std::vector<U> get_score() const;
    virtual std::vector<U> get_n_iter() const;
    virtual std::vector<U> get_n_features() const;
    virtual std::vector<U> get_n_samples() const;
    virtual std::vector<U> get_n_components() const;
    
    //set coefficients used for deserialization:

    virtual void set_coefficients(std::vector<U> coefficients);
    virtual void set_intercepts(std::vector<U> intercepts);

    virtual void set_residuals(std::vector<U> residuals);
    virtual void set_r_squared(std::vector<U> r_squared);

    virtual void set_score(std::vector<U> score);
    virtual void set_n_iter(std::vector<U> n_iter);
    virtual void set_n_features(std::vector<U> n_features);
    virtual void set_n_samples(std::vector<U> n_samples);
    virtual void set_n_components(std::vector<U> n_components);



    //fit data to model 

    virtual std::vector<U> fit( const std::vector<T>& );
    virtual std::vector<U> predict(const std::vector<T>& );
    virtual std::vector<U> transform(const std::vector<T>& );
    virtual std::vector<U> fit_transform(const std::vector<T>& );
    virtual std::vector<U> fit( const std::vector<T>&, const std::vector<U>& );
    virtual std::vector<U> predict(const std::vector<T>&, const std::vector<U>& );
    virtual std::vector<U> transform(const std::vector<T>&, const std::vector<U>& );
    virtual std::vector<U> fit_transform(const std::vector<T>&, const std::vector<U>& );
    virtual ~lasso_regression_model();



  };
  template <class T, class U>
    class elastic_net_model : public model<T,U>
  {
      public:
      elastic_net_model();
      elastic_net_model(elastic_net_model &&other); //move constructor
      elastic_net_model& operator= (const elastic_net_model &other);

      elastic_net_model&
          operator= (elastic_net_model &&other);
      //get coefficients:
      virtual std::vector<U> get_coefficients() const;
      virtual std::vector<U> get_intercepts() const;
      
      virtual std::vector<U> get_residuals() const;
      virtual std::vector<U> get_r_squared() const;
      virtual std::vector<U> get_score() const;
      virtual std::vector<U> get_n_iter() const;
      virtual std::vector<U> get_n_features() const;
      virtual std::vector<U> get_n_samples() const;
      virtual std::vector<U> get_n_components() const;
      //set coefficients used for deserialization:

      virtual void set_coefficients(std::vector<U> coefficients);
      virtual void set_intercepts(std::vector<U> intercepts);

      virtual void set_residuals(std::vector<U> residuals);
      virtual void set_r_squared(std::vector<U> r_squared);

      virtual void set_score(std::vector<U> score);
      virtual void set_n_iter(std::vector<U> n_iter);
      virtual void set_n_features(std::vector<U> n_features);
      virtual void set_n_samples(std::vector<U> n_samples);
      virtual void set_n_components(std::vector<U> n_components);



      //fit data to model
      

      virtual std::vector<U> fit( const std::vector<T>& );
      virtual std::vector<U> predict(const std::vector<T>& );
      virtual std::vector<U> transform(const std::vector<T>& );
      virtual std::vector<U> fit_transform(const std::vector<T>& );
      virtual std::vector<U> fit( const std::vector<T>&, const std::vector<U>& );
      virtual std::vector<U> predict(const std::vector<T>&, const std::vector<U>& );
      virtual std::vector<U> transform(const std::vector<T>&, const std::vector<U>& );
      virtual std::vector<U> fit_transform(const std::vector<T>&, const std::vector<U>& );
      virtual ~elastic_net_model();

  };

  template <class T,class U> 
  class linear_regression_model : public model<T,U>
  {
    public:
    linear_regression_model();
    linear_regression_model(linear_regression_model &&other); //move constructor
    linear_regression_model& operator= (const linear_regression_model &other);

    linear_regression_model&
        operator= (linear_regression_model &&other);
    //get coefficients:
    virtual std::vector<U> get_coefficients() const;
    virtual std::vector<U> get_intercepts() const;
    
    virtual std::vector<U> get_residuals() const;
    virtual std::vector<U> get_r_squared() const;
    virtual std::vector<U> get_score() const;
    virtual std::vector<U> get_n_iter() const;
    virtual std::vector<U> get_n_features() const;
    virtual std::vector<U> get_n_samples() const;
    virtual std::vector<U> get_n_components() const;
    //set coefficients used for deserialization:

    virtual void set_coefficients(std::vector<U> coefficients);
    virtual void set_intercepts(std::vector<U> intercepts);

    virtual void set_residuals(std::vector<U> residuals);
    virtual void set_r_squared(std::vector<U> r_squared);

    virtual void set_score(std::vector<U> score);
    virtual void set_n_iter(std::vector<U> n_iter);
    virtual void set_n_features(std::vector<U> n_features);
    virtual void set_n_samples(std::vector<U> n_samples);
    virtual void set_n_components(std::vector<U> n_components);     //fit data to model
    virtual std::vector<U> fit( const std::vector<T>& );
    virtual std::vector<U> predict(const std::vector<T>& );
    virtual std::vector<U> transform(const std::vector<T>& );
    virtual std::vector<U> fit_transform(const std::vector<T>& );
    virtual std::vector<U> fit( const std::vector<T>&, const std::vector<U>& );
    virtual std::vector<U> predict(const std::vector<T>&, const std::vector<U>& );
    virtual std::vector<U> transform(const std::vector<T>&, const std::vector<U>& );
    virtual std::vector<U> fit_transform(const std::vector<T>&, const std::vector<U>& );
    virtual ~linear_regression_model();

  };
  
  class text_model : public model<std::string, real_t>
  {
    private:
    std::vector<real_t> _model_data;
    std::vector<real_t> _model_components;

    public:
    text_model();
    text_model(text_model &&other); //move constructor

    text_model& operator= (const text_model &other);
    text_model&
        operator= (text_model &&other);
    //get coefficients: 
    virtual std::vector<real_t> get_coefficients() const;
    virtual std::vector<real_t> get_intercepts() const;
    virtual std::vector<real_t> get_residuals() const;
    virtual std::vector<real_t> get_r_squared() const;
    virtual std::vector<real_t> get_score() const;
    virtual std::vector<real_t> get_n_iter() const;
    virtual std::vector<real_t> get_loss() const;
    virtual std::vector<real_t> get_t() const;
    virtual std::vector<real_t> get_p() const;
    virtual std::vector<real_t> get_std_err() const;
    virtual std::vector<real_t> get_confidence_intervals() const;

    //fit data to model
    virtual std::vector<real_t> fit( const std::vector<std::string>&documents );
    virtual std::vector<real_t> predict(const std::vector<std::string>&documents );
    virtual std::vector<real_t> transform(const std::vector<std::string>&documents );
    virtual std::vector<real_t> fit_transform(const std::vector<std::string>&documents );
    virtual std::vector<real_t> fit( const provallo::matrix<real_t>& );
    virtual std::vector<real_t> predict(const provallo::matrix<real_t>& );
    virtual std::vector<real_t> transform(const provallo::matrix<real_t>& );
    virtual std::vector<real_t> fit_transform(const provallo::matrix<real_t>& );
    virtual ~text_model();
  };


  class regressor :   public model<std::string, real_t>
  {
  protected:
    regressor_type _regressor_type;
    

    std::vector<real_t> _regressor_data;
    std::vector<real_t> _regressor_components;
    std::vector<real_t> _regressor_explained_variance;
    std::vector<real_t> _regressor_explained_variance_ratio;
    std::vector<real_t> _regressor_singular_values;
    std::vector<real_t> _regressor_noise_variance;
    std::vector<real_t> _regressor_mean;
    std::vector<real_t> _regressor_n_components;
    std::vector<real_t> _regressor_n_features;
    std::vector<real_t> _regressor_n_samples;
    std::vector<real_t> _regressor_n_components_;
    std::vector<real_t> _regressor_n_features_;
    std::vector<real_t> _regressor_n_samples_;
    std::vector<real_t> _regressor_coefficients;  
    std::vector<real_t> _regressor_intercepts;
    std::vector<real_t> _regressor_residuals;
    std::vector<real_t> _regressor_r_squared;
    std::vector<real_t> _regressor_score;
    std::vector<real_t> _regressor_n_iter;
    std::vector<real_t> _regressor_loss;
    std::vector<real_t> _regressor_t;
    std::vector<real_t> _regressor_p;
    std::vector<real_t> _regressor_std_err;
    std::vector<real_t> _regressor_confidence_intervals;

    //functions to apply to the data
    std::map<std::string,std::function<std::vector<real_t> regressor::*( const matrix<real_t>  & foot, const std::vector<real_t> & target )>> _regressor_functions; 
    //functions to apply to the data
    std::vector<std::function< std::vector<real_t> regressor::*( matrix<real_t>)>> _regressor_inverse_functions; //functions to apply to the data
    //functions to apply to the data
    std::vector<std::function<std::vector<real_t> regressor::*( matrix<real_t>)>> _regressor_transform_functions; //functions to apply to the data 
    //functions to apply to the data
    std::vector<std::function<std::vector<real_t> regressor::*( matrix<real_t>)>> _regressor_fit_transform_functions; //functions to apply to the data
    public: 
    regressor(const regressor_type &type = LINEAR_REGRESSION);
    regressor(regressor &&other); //move constructor
    regressor& operator= (const regressor &other);
    regressor&
        operator= (regressor &&other);
    virtual  std::vector<real_t> fit( const std::vector<std::string>&documents );

    virtual  std::vector<real_t> predict(const std::vector<std::string>&documents );
    virtual  std::vector<real_t> transform(const std::vector<std::string>&documents);
    virtual  std::vector<real_t> fit_transform(const std::vector<std::string>&documents);


    //for use with inverse transformation matrices
    virtual std::vector<real_t> fit( const provallo::matrix<real_t>& );
    virtual std::vector<real_t> predict(const provallo::matrix<real_t>& );
    virtual std::vector<real_t> transform(const provallo::matrix<real_t>& );
    regressor_type get_regressor_type()const;
    //build
    virtual std::vector<real_t> build( const std::vector<std::string>&documents );
    virtual std::vector<real_t> build( const provallo::matrix<real_t>& );
    //get coefficients:
    virtual std::vector<real_t> get_coefficients() const;
    virtual std::vector<real_t> get_intercepts() const;
    virtual std::vector<real_t> get_residuals() const;
    virtual std::vector<real_t> get_r_squared() const;
    virtual std::vector<real_t> get_score() const;
    virtual std::vector<real_t> get_n_iter() const;
    virtual std::vector<real_t> get_loss() const;
    virtual std::vector<real_t> get_t() const;
    virtual std::vector<real_t> get_p() const;
    virtual std::vector<real_t> get_std_err() const;
    virtual std::vector<real_t> get_confidence_intervals() const;
    //get regressor data
    virtual std::vector<real_t> get_regressor_data() const;

    virtual ~regressor();
    //private regressor functions :
    private:
     
    //linear_regression functions
    std::vector<real_t> linear_regression_fit( const matrix<real_t>  & foot, const std::vector<real_t> & target );
    std::vector<real_t> linear_regression_predict( matrix<real_t>  foot );
    std::vector<real_t> linear_regression_transform( matrix<real_t>  foot );
    std::vector<real_t> linear_regression_fit_transform( matrix<real_t>  foot );
    //ridge_regression functions
    std::vector<real_t> ridge_regression_fit( const matrix<real_t>  & foot, const std::vector<real_t> & target );
    std::vector<real_t> ridge_regression_predict( matrix<real_t>  foot );
    std::vector<real_t> ridge_regression_transform( matrix<real_t>  foot );
    std::vector<real_t> ridge_regression_fit_transform( matrix<real_t>  foot );
    //lasso_regression functions
    std::vector<real_t> lasso_regression_fit( const matrix<real_t>  & foot, const std::vector<real_t> & target );
    std::vector<real_t> lasso_regression_predict( matrix<real_t>  foot );
    std::vector<real_t> lasso_regression_transform( matrix<real_t>  foot );
    std::vector<real_t> lasso_regression_fit_transform( matrix<real_t>  foot );
    //elastic_net functions
    std::vector<real_t> elastic_net_fit( const matrix<real_t>  & foot, const std::vector<real_t> & target );
    std::vector<real_t> elastic_net_predict( matrix<real_t>  foot );
    std::vector<real_t> elastic_net_transform( matrix<real_t>  foot );
    std::vector<real_t> elastic_net_fit_transform( matrix<real_t>  foot );
    //bayesian_ridge functions
    std::vector<real_t> bayesian_ridge_fit( const matrix<real_t>  & foot, const std::vector<real_t> & target );
    std::vector<real_t> bayesian_ridge_predict( matrix<real_t>  foot );
    std::vector<real_t> bayesian_ridge_transform( matrix<real_t>  foot );
    std::vector<real_t> bayesian_ridge_fit_transform( matrix<real_t>  foot );
    //logistic_regression functions

    std::vector<real_t> logistic_regression_fit( const matrix<real_t>  & foot, const std::vector<real_t> & target );
    std::vector<real_t> logistic_regression_predict( matrix<real_t>  foot );
    std::vector<real_t> logistic_regression_transform( matrix<real_t>  foot );
    std::vector<real_t> logistic_regression_fit_transform( matrix<real_t>  foot );

    //svm functions

    std::vector<real_t> svm_fit( const matrix<real_t>  & foot, const std::vector<real_t> & target );
    std::vector<real_t> svm_predict( matrix<real_t>  foot );
    std::vector<real_t> svm_transform( matrix<real_t>  foot );
    std::vector<real_t> svm_fit_transform( matrix<real_t>  foot );
    
    //decision_tree functions
    
    std::vector<real_t> decision_tree_fit( const matrix<real_t>  & foot, const std::vector<real_t> & target );
    std::vector<real_t> decision_tree_predict( matrix<real_t>  foot );
    std::vector<real_t> decision_tree_transform( matrix<real_t>  foot );
    std::vector<real_t> decision_tree_fit_transform( matrix<real_t>  foot );

    //random_forest functions

    std::vector<real_t> random_forest_fit( const matrix<real_t>  & foot, const std::vector<real_t> & target );
    std::vector<real_t> random_forest_predict( matrix<real_t>  foot );
    std::vector<real_t> random_forest_transform( matrix<real_t>  foot );
    std::vector<real_t> random_forest_fit_transform( matrix<real_t>  foot );

    //gradient_boosting functions

    std::vector<real_t> gradient_boosting_fit( const matrix<real_t>  & foot, const std::vector<real_t> & target );
    std::vector<real_t> gradient_boosting_predict( matrix<real_t>  foot );
    std::vector<real_t> gradient_boosting_transform( matrix<real_t>  foot );
    std::vector<real_t> gradient_boosting_fit_transform( matrix<real_t>  foot );

    //knn functions
    
    std::vector<real_t> knn_fit( const matrix<real_t>  & foot, const std::vector<real_t> & target );
    std::vector<real_t> knn_predict( matrix<real_t>  foot );
    std::vector<real_t> knn_transform( matrix<real_t>  foot );
    std::vector<real_t> knn_fit_transform( matrix<real_t>  foot );

    //gaussian_process functions

    std::vector<real_t> gaussian_process_fit( const matrix<real_t>  & foot, const std::vector<real_t> & target );
    std::vector<real_t> gaussian_process_predict( matrix<real_t>  foot );
    std::vector<real_t> gaussian_process_transform( matrix<real_t>  foot );
    std::vector<real_t> gaussian_process_fit_transform( matrix<real_t>  foot );

    //ada_boost functions

    std::vector<real_t> ada_boost_fit( const matrix<real_t>  & foot, const std::vector<real_t> & target );
    std::vector<real_t> ada_boost_predict( matrix<real_t>  foot );
    std::vector<real_t> ada_boost_transform( matrix<real_t>  foot );
    std::vector<real_t> ada_boost_fit_transform( matrix<real_t>  foot );
    
    //mlp functions

    std::vector<real_t> mlp_fit( const matrix<real_t>  & foot, const std::vector<real_t> & target );
    std::vector<real_t> mlp_predict( matrix<real_t>  foot );
    std::vector<real_t> mlp_transform( matrix<real_t>  foot );
    std::vector<real_t> mlp_fit_transform( matrix<real_t>  foot );

    //kmeans functions

    std::vector<real_t> kmeans_fit( const matrix<real_t>  & foot, const std::vector<real_t> & target );
    std::vector<real_t> kmeans_predict( matrix<real_t>  foot );
    std::vector<real_t> kmeans_transform( matrix<real_t>  foot );
    std::vector<real_t> kmeans_fit_transform( matrix<real_t>  foot );


    //pca functions uses vectoizers to transform data

    std::vector<real_t> pca_fit( const matrix<real_t>  & foot, const std::vector<real_t> & target );
    std::vector<real_t> pca_predict( matrix<real_t>  foot );
    std::vector<real_t> pca_transform( matrix<real_t>  foot );
    std::vector<real_t> pca_fit_transform( matrix<real_t>  foot );

    //nmf functions uses vectoizers to transform data

    std::vector<real_t> nmf_fit( const matrix<real_t>  & foot, const std::vector<real_t> & target );
    std::vector<real_t> nmf_predict( matrix<real_t>  foot );
    std::vector<real_t> nmf_transform( matrix<real_t>  foot );

    //lda functions uses vectoizers to transform data
    
    std::vector<real_t> lda_fit( const matrix<real_t>  & foot, const std::vector<real_t> & target );
    std::vector<real_t> lda_predict( matrix<real_t>  foot );
    std::vector<real_t> lda_transform( matrix<real_t>  foot );
    std::vector<real_t> lda_fit_transform( matrix<real_t>  foot );

    //qda functions uses vectoizers to transform data
    
    std::vector<real_t> qda_fit( const matrix<real_t>  & foot, const std::vector<real_t> & target );
    std::vector<real_t> qda_predict( matrix<real_t>  foot );
    std::vector<real_t> qda_transform( matrix<real_t>  foot );
    std::vector<real_t> qda_fit_transform( matrix<real_t>  foot );

    //svd functions uses matrix.h implementation of singular value decomposition to transform data
    std::vector<real_t> svd_fit( const matrix<real_t>  & foot, const std::vector<real_t> & target );
    std::vector<real_t> svd_predict( matrix<real_t>  foot );
    std::vector<real_t> svd_transform( matrix<real_t>  foot );
    std::vector<real_t> svd_fit_transform( matrix<real_t>  foot );

    //jacobi functions uses matrix.h implementation of jacobi eigenvalue algorithm to transform data
    std::vector<real_t> jacobi_fit( const matrix<real_t>  & foot, const std::vector<real_t> & target );
    std::vector<real_t> jacobi_predict( matrix<real_t>  foot );
    std::vector<real_t> jacobi_transform( matrix<real_t>  foot );
    std::vector<real_t> jacobi_fit_transform( matrix<real_t>  foot );

    //svr functions uses vectoizers to transform data
    std::vector<real_t> svr_fit( const matrix<real_t>  & foot, const std::vector<real_t> & target );
    std::vector<real_t> svr_predict( matrix<real_t>  foot );
    std::vector<real_t> svr_transform( matrix<real_t>  foot );
    std::vector<real_t> svr_fit_transform( matrix<real_t>  foot );

  };

  struct stage_descriptor
  {
    public:
    uint64_t stage_id;
    std::string name;
    std::string type;
    std::string parameters;
    std::string input;
    std::string output;
    std::string input_type;
    std::string output_type;
    std::string input_parameters;
    std::string output_parameters;
    uint64_t previous_stage; //previous stage id
    uint64_t next_stage;  //next stage id
    
    //constructor from data.

    stage_descriptor (const std::string& line) : stage_id(0), name(""), type(""), parameters(""), input(""), output(""), input_type(""), output_type(""), input_parameters(""), output_parameters(""), previous_stage(0), next_stage(0) 
    {
      std::stringstream ss(line);
      ss >> stage_id >> name >> type >> parameters >> input >> output >> input_type >> output_type >> input_parameters >> output_parameters;  
    }
    //default
    stage_descriptor () : stage_id(0), name(""), type(""), parameters(""), input(""), output(""), input_type(""), output_type(""), input_parameters(""), output_parameters(""), previous_stage(0) , next_stage(0)  {} 

    //not pure virtual, public struct, but should be implemented in derived classes if needed.
    //additional data is optional per stage.
    virtual void load_additional_data(const std::string& data) {UNDEF_REFERENCE(data);UNDEF_REFERENCE2(data);}
    virtual void save_additional_data(std::string& data) {UNDEF_REFERENCE(data);UNDEF_REFERENCE2(data);}
    inline void set_descriptor(const stage_descriptor& cpy) 
    {
      stage_id = cpy.stage_id;
      name = cpy.name;
      type = cpy.type;
      parameters = cpy.parameters;
      input = cpy.input;
      output = cpy.output;
      input_type = cpy.input_type;
      output_type = cpy.output_type;
      input_parameters = cpy.input_parameters;
      output_parameters = cpy.output_parameters;
      previous_stage = cpy.previous_stage;
      next_stage = cpy.next_stage;

    }    
    friend std::ostream& operator<<(std::ostream& os, const stage_descriptor& sd);
    friend std::istream& operator>>(std::istream& is, stage_descriptor& sd);

    virtual const std::string get_additional_data() const { return ""; }
    virtual void set_additional_data(const std::string& data) { UNDEF_REFERENCE(data);UNDEF_REFERENCE2(data);}
    void set_stage_id(uint64_t id) { stage_id = id; }
    size_t get_stage_id() const { return stage_id; }
    void set_name(const std::string& n) { name = n; } 
    //process data 
    virtual void process_data(const std::string& data) { UNDEF_REFERENCE(data);UNDEF_REFERENCE2(data);}
    virtual void process_data(const matrix<real_t>& data) { UNDEF_REFERENCE(data);UNDEF_REFERENCE2(data);}
    virtual void process_data(const std::vector<real_t>& data) { UNDEF_REFERENCE(data);UNDEF_REFERENCE2(data);}
    virtual void process_data(const std::vector<std::vector<real_t> >& data) { UNDEF_REFERENCE(data);UNDEF_REFERENCE2(data);}
    virtual void set_parameter(const std::string& parameter_name, const std::string& parameter_value)
    {
        if(parameters_map.find(parameter_name) == parameters_map.end())
        {
          parameters_map[parameter_name] = parameter_value;
        }
        else
        {
          parameters_map.insert(std::make_pair(parameter_name, parameter_value));

        }
      
    } 
    
    virtual ~stage_descriptor() {}
    std::unordered_map<std::string, std::string> parameters_map;

    //category of the stage
    const std::string get_category() const { return type; } 
  
  };


  
  class dataset_stage : public stage_descriptor
  {
    
    public : 

    dataset_stage();

    virtual ~dataset_stage();
    protected:
    //additional data 
    std::string data;
    public:
    virtual void load_additional_data(const std::string& data);
    virtual void save_additional_data(std::string& data);
    
    //additional data for data sets would be :
    //1. set/subset of the data.
    //2. the data types.
    //3. the data parameters.
    
    virtual const std::string get_additional_data() const;
    virtual void set_additional_data(const std::string& data);

    //process data
    static dataset_stage* build( );
    enum dataset_type get_type() const { return _dataset_type; }
    enum data_format get_format() const { return _format; }
    enum data_type get_data_type() const { return _data_type; }
    enum dataset_source get_source() const { return _dataset_source; }
    enum dataset_status get_status() const { return _dataset_status; }
    enum dataset_mode get_mode() const { return _dataset_mode; }
    void set_type(enum dataset_type t) { _dataset_type = t; }
    void set_format(enum data_format f) { _format = f; }
    

     bool is_static() const { return _dataset_type == STATIC; }
    bool is_dynamic() const { return _dataset_type == DYNAMIC; }
    bool is_labeled() const { return _labels.size() > 0; }   
    std::vector<size_t> get_labels() const { return _labels; } 
    std::vector<std::string> get_label_names() const { return _label_names; }
    void set_label_names(const std::vector<std::string>& names) { _label_names = names; } 

    class_dist get_class_dist() const { return _class_dist; } 
    enum dataset_purpose purpose() const { return _purpose; } 
    void purpose(dataset_purpose p) { _purpose = p; }
    bool is_build_train() const { return _purpose == BUILD_TRAIN; }
    bool is_train() const { return _purpose == TRAIN; }
    bool is_optimize_train() const { return _purpose == OPTIMIZE_TRAIN; }
    bool is_test() const { return _purpose == TEST; }
    bool is_optimize_test() const { return _purpose == OPTIMIZE_TEST; }
    bool is_validate() const { return _purpose == VALIDATE; }
    bool is_xvalidate() const { return _purpose == XVALIDATE; }
    bool is_text() const { return _format == TEXT; }
    bool is_binary() const { return _format != TEXT&&_format!=CSV; }
    bool is_csv() const { return _format == CSV; }
    bool is_libsvm() const { return _format == LIBSVM; }
    bool is_libfm() const { return _format == LIBFM; }
    bool is_libffm() const { return _format == LIBFFM; }
    //tensor flow 
    bool is_tfrecord() const { return _format == TFRECORD; }
    bool is_tsv() const { return _format == TSV; }
    bool is_torch() const { return _format == TORCH; }
     bool is_caffe2() const { return _format == CAFFE2; }
    bool is_mxnet() const { return _format == MXNET; }
    bool is_onnx() const { return _format == ONNX; }
     bool is_hdf5() const { return _format == HDF5; }
     bool is_npy() const { return _format == NPY; }     

    //process data
    // implements the virtual functions of stage_descriptor 
    // decents should override process_data with matrix_base
    // 
    // load file 
    virtual void process_data(const std::string& data); 
    //adapter for matrix<real_t>

    virtual void process_data(const matrix<real_t>& data);
    //adapter for vector<real_t>
    virtual void process_data(const std::vector<real_t>& data);
    //adapter for vector<vector<real_t> >
    virtual void process_data(const std::vector<std::vector<real_t> >& data);

 
    //adapter for dataset_ptr or matrix_base 
    //converts matrix_base to matrix<real_t> and processes the dataset. 

    virtual void process_data(matrix_base& data); 
    private:
    class_dist _class_dist;
    //labels:
    std::vector<size_t> _labels;
    std::vector<std::string> _label_names; 
    std::vector<std::string> _column_names; 
    std::vector<std::string> _label_values;
    std::vector<std::string> _classes;
    std::vector<std::string> _features; //or column_names... 
    std::map<std::string,size_t> number_of_samples_per_label;
    std::map<size_t,size_t> number_of_samples_per_class;
    std::map<std::string,size_t> number_of_samples_per_feature;
    std::map<std::string,size_t> number_of_samples_per_column;
    std::map<std::string,size_t> number_of_samples_per_row;
    std::map<std::string,size_t> number_of_samples_per_attribute;
    std::map<std::string,size_t> number_of_samples_per_instance;

    enum data_type _data_type;
    enum data_format  _format; 
    enum dataset_purpose  _purpose;
    enum dataset_type  _dataset_type;
    enum dataset_source  _dataset_source;
    enum dataset_status  _dataset_status;
    enum dataset_mode  _dataset_mode;
    
     //data:

    matrix<real_t> _data;    //data matrix
    dataset_base* _dataset; //pointer to the data in the file
    std::string _data_file; //file name of the source data file
    std::vector<size_t> _number_of_samples_per_label;
    size_t _number_of_labels;
    size_t _number_of_samples ;
    size_t _number_of_attributes ;
    size_t _number_of_classes ;
    size_t _num_of_clusters;
    size_t _num_of_outliers;
    size_t _num_of_noise;
    size_t _num_of_features;
    size_t _num_of_unlabelled;
    size_t _num_of_labelled;
    size_t _num_of_test;
    size_t _num_of_train;
    size_t _num_of_validate;
    size_t _num_of_optimize;
    size_t _num_of_xvalidate;
    size_t _num_of_build_train;
    size_t _num_of_optimize_train;
    size_t _num_of_optimize_test;
 

    //additional data

    std::vector<std::string> additional_data; //[type,source,format,modes,labels,classes,attributes,rows,columns,values,classes,labels,attributes,rows,columns,values]
    //helper functions 
    matrix<real_t> load_csv(const std::string& file_name) ;
    matrix<real_t> load_txt(const std::string& file_name);
    matrix<real_t> load_libsvm(const std::string& file_name); 
    matrix<real_t> load_libfm(const std::string& file_name);
    matrix<real_t> load_libffm(const std::string& file_name);
    matrix<real_t> load_tfrecord(const std::string& file_name);
    matrix<real_t> load_tsv(const std::string& file_name);
    matrix<real_t> load_torch(const std::string& file_name);
    matrix<real_t> load_caffe2(const std::string& file_name);
    matrix<real_t> load_mxnet(const std::string& file_name);
    matrix<real_t> load_onnx(const std::string& file_name);
    matrix<real_t> load_hdf5(const std::string& file_name);
    matrix<real_t> load_npy(const std::string& file_name);
    matrix<real_t> load_matrix(const std::string& file_name);
    matrix<real_t> load_dataset(const std::string& file_name);
    matrix<real_t> load_data(const std::string& file_name);
    matrix<real_t> load(const std::string& file_name);

    //dataset stage doesn't save in any of the formats
    //it serializes to our own format 
    //
    //information about the data:
    
    public:
    //set descriptive information about the data set 
    void set_classes(const std::vector<std::string>& classes) { _classes = classes; }
    void set_features(const std::vector<std::string>& features){_column_names = features;}
    /*TODO sample information:
      void set_number_of_samples_per_label(const std::vector<size_t>& number_of_samples_per_label) { _number_of_samples_per_label = number_of_samples_per_label; } 
      void set_number_of_samples_per_class(const std::map<size_t,size_t>& number_of_samples_per_class) { _number_of_samples_per_class = number_of_samples_per_class; }
      void set_number_of_samples_per_feature(const std::map<std::string,size_t>& number_of_samples_per_feature) { _number_of_samples_per_feature = number_of_samples_per_feature; }
      void set_number_of_samples_per_column(const std::map<std::string,size_t>& number_of_samples_per_column) { _number_of_samples_per_column = number_of_samples_per_column; }
      void set_number_of_samples_per_row(const std::map<std::string,size_t>& number_of_samples_per_row) { _number_of_samples_per_row = number_of_samples_per_row; }
      void set_number_of_samples_per_attribute(const std::map<std::string,size_t>& number_of_samples_per_attribute) { _number_of_samples_per_attribute = number_of_samples_per_attribute; } 
      void set_number_of_samples_per_instance(const std::map<std::string,size_t>& number_of_samples_per_instance) { _number_of_samples_per_instance = number_of_samples_per_instance; } 
      void set_number_of_samples_per_sample(const std::map<std::string,size_t>& number_of_samples_per_sample) { _number_of_samples_per_sample = number_of_samples_per_sample; }
      void set_number_of_samples_per_label(const std::map<std::string,size_t>& number_of_samples_per_label) { _number_of_samples_per_label = number_of_samples_per_label; }
      */


  };


  class dynamic_dataset_stage : public dataset_stage
  {

    //dynamic dataset stage is a dataset stage that can be modified before or during the training process. 
    //it is used for example to add new data to the training set during the training process. 
    //it is also used to add new data to the training set during the training process. 

    public : 
    dynamic_dataset_stage();
    virtual ~dynamic_dataset_stage();
      public:
    virtual void load_additional_data(const std::string& data);
    virtual void save_additional_data(std::string& data); 
    static dynamic_dataset_stage* build( );

  };
  
  class static_dataset_stage : public dataset_stage
  {
    public : 
    static_dataset_stage();
    virtual ~static_dataset_stage();
      public:
    virtual void load_additional_data(const std::string& data);
    virtual void save_additional_data(std::string& data); 
    static static_dataset_stage* build( );

  };  
  
  class feature_extraction_parameters 
  {
    //comprehensive list of feature extraction parameters.
  public :
    feature_extraction_parameters() : features_length(0), parameters(""), feature_extraction_method(nullptr)
     {
 
        //initialize the feature extraction method.
        auto empty_fcp = [](matrix<real_t> data) -> std::vector<real_t>  { 
                     matrix<real_t> ev=data.eigenvalues(),ev2=data.eigenvectors() ; return std::vector<real_t>(ev.begin(),ev.end());  }; 
        feature_extraction_method = empty_fcp;

        
     }
    feature_extraction_parameters(const feature_extraction_parameters& cpy)
    {
      features_length = cpy.features_length;
      parameters = cpy.parameters;
      feature_extraction_method = cpy.feature_extraction_method;
    }
    feature_extraction_parameters& operator=(const feature_extraction_parameters& cpy)
    {
      features_length = cpy.features_length;
      parameters = cpy.parameters;
      feature_extraction_method = cpy.feature_extraction_method;
      return *this;
    }
    void set_feature_extraction_method(std::function< std::vector<real_t>  (matrix<real_t> ) > fcp) { feature_extraction_method = fcp; }
    std::function< std::vector<real_t>  (matrix<real_t> ) > get_feature_extraction_method() const 
    { return feature_extraction_method; }
    void set_features_length(size_t length)
     { features_length = length; }
    size_t get_features_length() const 
    { return features_length; }

    virtual ~feature_extraction_parameters() {}

  private: 
    //size_t method_id
    size_t features_length;
    //feature extraction parameters.
    std::string parameters;
      //feature extraction method.
    std::function< std::vector<real_t> (matrix<real_t> ) > feature_extraction_method  ;

  };
  //end of feature_extraction_parameters class.

  //feature extraction methods :

  template <typename vec_src,typename real_>
  class feature_transformer;
  template <typename vec_src,typename real_>
  class feature_selector;
  template <typename vec_src,typename real_>
  class feature_aggregator ;
  template <typename vec_src,typename real_>
  class feature_normalizer ;
  template <typename vec_src,typename real_>
  class feature_weighter;
  template <typename vec_src,typename real_>
  class feature_binarizer ;
  template <typename vec_src,typename real_>
  class feature_filter ;
  template <typename vec_src,typename real_>
  class feature_reducer ;
  template <typename vec_src,typename real_>
  class feature_combiner ;
  


  template <typename vec_src,typename real_>
  class feature_extractor 
  {
      //feature extractor is a class that extracts features from a vector of real_t.
      //it is used to extract features from a dataset.

    public:
      feature_extractor() {}
      virtual ~feature_extractor() {}
      virtual std::vector<real_> extract_features(const vec_src& data) const = 0;
      virtual size_t get_features_length() const = 0;
      virtual void set_features_length(size_t length) = 0;
      virtual void set_feature_extraction_method(std::function< std::vector<real_>  (matrix<real_> ) > fcp) = 0;
      virtual std::function< std::vector<real_>  (matrix<real_> ) > get_feature_extraction_method() const = 0;
      virtual void set_parameters(const std::string& parameters) = 0;
      virtual std::string get_parameters() const = 0;
      virtual void set_feature_extraction_method_id(size_t id) = 0;
      virtual size_t get_feature_extraction_method_id() const = 0;
      virtual void set_feature_extraction_method_name(const std::string& name) = 0;
      virtual std::string get_feature_extraction_method_name() const = 0;
      virtual void set_feature_extraction_method_parameters(const std::string& parameters) = 0;
      //extract features from a dataset: 
      virtual matrix<real_> extract_features(const matrix<real_>& data) const = 0;
    protected:
    feature_extraction_parameters _feature_extraction_parameters;

      //
  };
  //implement verctorized concrete feature extractors here,usa pca as an example.
  class pca_feature_extractor  : public feature_extractor<std::string,real_t> 
  {
    public:
      pca_feature_extractor() {}
      virtual ~pca_feature_extractor() {}
      virtual std::vector<real_t> extract_features(const std::string& data) const ;
      virtual size_t get_features_length() const ;
      virtual void set_features_length(size_t length) ;
      virtual void set_feature_extraction_method(std::function< std::vector<real_t>  (matrix<real_t> ) > fcp) ;
      virtual std::function< std::vector<real_t>  (matrix<real_t> ) > get_feature_extraction_method() const ;
      virtual void set_parameters(const std::string& parameters) ;
      virtual std::string get_parameters() const ;
      virtual void set_feature_extraction_method_id(size_t id) ;
      virtual size_t get_feature_extraction_method_id() const ;
      virtual void set_feature_extraction_method_name(const std::string& name) ;
      virtual std::string get_feature_extraction_method_name() const ;
      virtual void set_feature_extraction_method_parameters(const std::string& parameters) ;
      //extract features from a dataset: 
      virtual matrix<real_t> extract_features(const matrix<real_t>& data) const ;

  };
  class independent_component_analysis_feature_extractor : public feature_extractor<std::string,real_t>
  {
    public:
      independent_component_analysis_feature_extractor() {}
      virtual ~independent_component_analysis_feature_extractor() {}
      virtual std::vector<real_t> extract_features(const std::string& data) const ;
      virtual size_t get_features_length() const ;
      virtual void set_features_length(size_t length) ;
      virtual void set_feature_extraction_method(std::function< std::vector<real_t>  (matrix<real_t> ) > fcp) ;
      virtual std::function< std::vector<real_t>  (matrix<real_t> ) > get_feature_extraction_method() const ;
      virtual void set_parameters(const std::string& parameters) ;
      virtual std::string get_parameters() const ;
      virtual void set_feature_extraction_method_id(size_t id) ;
      virtual size_t get_feature_extraction_method_id() const ;
      virtual void set_feature_extraction_method_name(const std::string& name) ;
      virtual std::string get_feature_extraction_method_name() const ;
      virtual void set_feature_extraction_method_parameters(const std::string& parameters) ;
      //extract features from a dataset: 
      virtual matrix<real_t> extract_features(const matrix<real_t>& data) const ;

  };
  class lda_feature_extractor : public feature_extractor<std::string,real_t>
  {
    public:
      lda_feature_extractor() {}
      virtual ~lda_feature_extractor() {}
      virtual std::vector<real_t> extract_features(const std::string& data) const ;
      virtual size_t get_features_length() const ;
      virtual void set_features_length(size_t length) ;
      virtual void set_feature_extraction_method(std::function< std::vector<real_t>  (matrix<real_t> ) > fcp) ;
      virtual std::function< std::vector<real_t>  (matrix<real_t> ) > get_feature_extraction_method() const ;
      virtual void set_parameters(const std::string& parameters) ;
      virtual std::string get_parameters() const ;
      virtual void set_feature_extraction_method_id(size_t id) ;
      virtual size_t get_feature_extraction_method_id() const ;
      virtual void set_feature_extraction_method_name(const std::string& name) ;
      virtual std::string get_feature_extraction_method_name() const ;
      virtual void set_feature_extraction_method_parameters(const std::string& parameters) ;
      //extract features from a dataset: 
      virtual matrix<real_t> extract_features(const matrix<real_t>& data) const ;

  };
  template<class vec_src,class real_>
  class feature_transformer{
      //transform features from a dataset:
      public:
      virtual matrix<real_> transform_features(const matrix<real_>& data) const = 0;
      //transform features from a dataset:
      virtual matrix<real_> transform_features(const vec_src& data) const = 0;
      virtual std::string get_feature_transformation_method_name() const = 0;
      
  };
  //feature transformers :

  template <typename vec_src,typename real_>
  class feature_engineering 
  {
     public:
    feature_engineering():_extractor(nullptr),_transformer(nullptr),
    _selector(nullptr),_aggregator(nullptr),_weighter(nullptr),
    _binarizer(nullptr),_filter(nullptr),_reducer(nullptr)
    {}
    virtual ~feature_engineering() {}

    void set_extractor(  feature_extractor<vec_src,real_>* extractor) { _extractor = extractor; }
    void set_transformer(  feature_transformer<vec_src,real_>* transformer) { _transformer = transformer; }
    void set_selector(  feature_selector<vec_src,real_>*selector) { _selector = selector; }
    void set_aggregator(  feature_aggregator<vec_src,real_>* aggregator) { _aggregator = aggregator; }
    void set_normalizer(  feature_normalizer<vec_src,real_>* normalizer) { _normalizer = normalizer; }
    void set_weighter(  feature_weighter<vec_src,real_>* weighter) { _weighter = weighter; }
    void set_binarizer(  feature_binarizer<vec_src,real_>* binarizer) { _binarizer = binarizer; }
    void set_filter(  feature_filter<vec_src,real_>*filter) { _filter = filter; }
    void set_reducer(  feature_reducer<vec_src,real_>* reducer) { _reducer = reducer; }
    feature_extractor<vec_src,real_>* get_extractor() { return _extractor; }
    feature_transformer<vec_src,real_>* get_transformer() { return _transformer; }
    feature_selector<vec_src,real_>* get_selector() { return _selector; }
    feature_aggregator<vec_src,real_>* get_aggregator() { return _aggregator; }
    feature_normalizer<vec_src,real_>* get_normalizer() { return _normalizer; }
    feature_weighter<vec_src,real_>*  get_weighter() { return _weighter; }
    feature_binarizer<vec_src,real_>*  get_binarizer() { return _binarizer; }
    feature_filter<vec_src,real_>*  get_filter() { return _filter; }
    feature_reducer<vec_src,real_>*  get_reducer() { return _reducer; }
    void extract(const vec_src& src, matrix<real_>& dst) { _extractor!=nullptr? _extractor.extract(src, dst) :nops++; }
    void transform(const vec_src& src, matrix<real_>& dst) {_transformer!=nullptr?  _transformer.transform(src, dst):nops++; }
    void select(const vec_src& src, matrix<real_>& dst) { _selector!=nullptr? _selector.select(src, dst):nops++; }
    void aggregate(const vec_src& src, matrix<real_>& dst) { _aggregator!=nullptr?_aggregator.aggregate(src, dst):nops++; }
    void normalize(const vec_src& src, matrix<real_>& dst) { _normalizer!=nullptr? _normalizer.normalize(src, dst):nops++; }
    void weight(const vec_src& src, matrix<real_>& dst) { _weighter!=nullptr? _weighter.weight(src, dst):nops++; }
    void binarize(const vec_src& src, matrix<real_>& dst) { _binarizer!=nullptr? _binarizer.binarize(src, dst):nops++; }
    void filter(const vec_src& src, matrix<real_>& dst) {  _filter!=nullptr?_filter.filter(src, dst):nops++; }
    void reduce(const vec_src& src, matrix<real_>& dst) { _reducer!=nullptr?_reducer.reduce(src, dst):nops++; }
    
    void extract(const vec_src& src, matrix<real_>& dst, const feature_extraction_parameters& params) { _extractor.extract(src, dst, params) ; }
    void transform(const vec_src& src, matrix<real_>& dst, const feature_extraction_parameters& params) { _transformer.transform(src, dst, params); }
    void select(const vec_src& src, matrix<real_>& dst, const feature_extraction_parameters& params) { _selector.select(src, dst, params); }

    void aggregate(const vec_src& src, matrix<real_>& dst, const feature_extraction_parameters& params) { _aggregator.aggregate(src, dst, params); }
    void normalize(const vec_src& src, matrix<real_>& dst, const feature_extraction_parameters& params) { _normalizer.normalize(src, dst, params); }
    void weight(const vec_src& src, matrix<real_>& dst, const feature_extraction_parameters& params) { _weighter.weight(src, dst, params); }
    void binarize(const vec_src& src, matrix<real_>& dst, const feature_extraction_parameters& params) { _binarizer.binarize(src, dst, params); }
    void filter(const vec_src& src, matrix<real_>& dst, const feature_extraction_parameters& params) { _filter.filter(src, dst, params); }
    void reduce(const vec_src& src, matrix<real_>& dst, const feature_extraction_parameters& params) { _reducer.reduce(src, dst, params); }
    void extract(const vec_src& src, matrix<real_>& dst, const std::string& params) { _extractor.extract(src, dst, params); }
    void transform(const vec_src& src, matrix<real_>& dst, const std::string& params) { _transformer.transform(src, dst, params); }
    void select(const vec_src& src, matrix<real_>& dst, const std::string& params) { _selector.select(src, dst, params); }

    void aggregate(const vec_src& src, matrix<real_>& dst, const std::string& params) { _aggregator.aggregate(src, dst, params); }
    void normalize(const vec_src& src, matrix<real_>& dst, const std::string& params) { _normalizer.normalize(src, dst, params); }

    void weight(const vec_src& src, matrix<real_>& dst, const std::string& params) { _weighter.weight(src, dst, params); }
    void binarize(const vec_src& src, matrix<real_>& dst, const std::string& params) { _binarizer.binarize(src, dst, params); }
    void filter(const vec_src& src, matrix<real_>& dst, const std::string& params) { _filter.filter(src, dst, params); }
    void reduce(const vec_src& src, matrix<real_>& dst, const std::string& params) { _reducer.reduce(src, dst, params); }
    void extract(const vec_src& src, matrix<real_>& dst, const std::vector<std::string>& params) { _extractor.extract(src, dst, params); }
    void transform(const vec_src& src, matrix<real_>& dst, const std::vector<std::string>& params) { _transformer.transform(src, dst, params); }

    void select(const vec_src& src, matrix<real_>& dst, const std::vector<std::string>& params) { _selector.select(src, dst, params); }
    void aggregate(const vec_src& src, matrix<real_>& dst, const std::vector<std::string>& params) { _aggregator.aggregate(src, dst, params); }
    void normalize(const vec_src& src, matrix<real_>& dst, const std::vector<std::string>& params) { _normalizer.normalize(src, dst, params); }
    void weight(const vec_src& src, matrix<real_>& dst, const std::vector<std::string>& params) { _weighter.weight(src, dst, params); }
    void binarize(const vec_src& src, matrix<real_>& dst, const std::vector<std::string>& params) { _binarizer.binarize(src, dst, params); }
    void filter(const vec_src& src, matrix<real_>& dst, const std::vector<std::string>& params) { _filter.filter(src, dst, params); }
    void reduce(const vec_src& src, matrix<real_>& dst, const std::vector<std::string>& params) { _reducer.reduce(src, dst, params); }
    
    
    private :

    feature_extractor<vec_src,real_>* _extractor;
    feature_transformer<vec_src,real_>* _transformer;
    feature_selector<vec_src,real_>* _selector;
    feature_aggregator<vec_src,real_>* _aggregator;
    feature_normalizer<vec_src,real_>* _normalizer;
    feature_weighter<vec_src,real_>* _weighter;
    feature_binarizer<vec_src,real_>* _binarizer;
    feature_filter<vec_src,real_>* _filter;
    feature_reducer<vec_src,real_>* _reducer;
    //feature_map with indexes to the matrix
    std::map<std::string,std::vector<std::pair<size_t,size_t>> > _feature_map;
    matrix<real_> _data;

    //optional:
    std::set<std::string> _feature_names;
    std::map<std::pair<std::string,std::string>,std::string> _feature_names_map;
    size_t nops = 0;

  };
  //end of feature_extraction_parameters class.
  class feature_stage : public stage_descriptor
  {
    private :

    feature_extraction_parameters _extraction_functors;
    feature_engineering<std::string,real_t> _engineering_functors;
    
    std::vector<std::string> _functor_names; //initialize the feature 
    //extractor methods here.
     template <typename T>
    std::vector<T> parse_string(const std::string& d) const
    {
      std::vector<T> v;
      tokenize(d, v, " ");
      return v;
    } 

    protected:
    matrix<real_t> data;

    public : 
    feature_stage()
    {
      
      typedef std::function< std::vector<real_t>  (matrix<real_t> ) > fcp;

      fcp fcp1_sum = [ ](matrix<real_t> ds) -> std::vector<real_t> {            //return vector of probabilities for each column in the data.
       
        std::vector<real_t> ret (ds.size2(), 0.0);
        for (size_t i = 0; i < ds.size2(); ++i)
        {
          real_t sum = 0.0;
          for (size_t j = 0; j < ds.size1(); ++j)
          {
            sum += ds(j,i)  ;
          }
          ret[i] = sum / ds.size1();
        } return ret;
        };
        _extraction_functors.set_feature_extraction_method(fcp1_sum);
        _extraction_functors.set_features_length(1);
        _functor_names.push_back("sum");
 
    } 
    feature_stage(const feature_stage& cpy) : stage_descriptor(cpy)
    {
      data = cpy.data;  
    
      _extraction_functors = cpy._extraction_functors;

    }
    virtual ~feature_stage()=default;
    public:
    virtual void load_additional_data(const std::string& data);
    virtual void save_additional_data(std::string& data);

    static feature_stage* build( );

    matrix<real_t> get_data() const { return data; }
    void set_data(const matrix<real_t>& data) { this->data = data; }

    //process data input sources:
    virtual void process_data(const std::string& data);
    virtual void process_data(const matrix<real_t>& data);
    virtual void process_data(const std::vector<real_t>& data);
    virtual void process_data(const std::vector<std::vector<real_t> >& data);
     

    virtual matrix<real_t>  extract_features(const std::string& d) 
    { 

 
      //parse the string into a vector.
        
       size_t rows = 0;
       size_t cols = 0;
       
       std::vector<real_t> v = parse_string<real_t>(d); 

        if (v.size() == 0)
          return matrix<real_t>();

        if (data.size1() == 0)
        {
          rows = 1;
          cols = v.size();
        }
        else
        {
          rows = data.rows();
          cols = data.cols();
        } 
        if  ( v.size() != cols)
          throw std::runtime_error("feature_stage::extract_features: data size does not match the feature size");

        if (cols == 0)
          return matrix<real_t>(1, 1 );

        //if we only have one row, we return the vector as a row vector.
        if (rows == 1)
        {
          matrix<real_t> m(1, cols);
          for (size_t i = 0; i < cols; i++)
            m(0, i) = v[i];
          return m;
        }
        else
        {
          //go over the rows of the matrix and the sample vector and add the sample vector to the matrix.
          //calculate the correlation between the sample vector and the matrix.
          matrix<real_t> corr ( data.correlation() );
          matrix<real_t> m(rows + 1, cols);
          for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < cols; j++)
              m(i, j) = data(i, j);
          for (size_t j = 0; j < cols; j++)
            m(rows, j) = v[j];
          return m;

        } 

    }
    //returns the correlation between the data and the input
    //or the input and the input columns on zero shot learning.
    virtual matrix<real_t>  extract_features(const matrix<real_t>& d) 
    { 
      //if the data is empty, we return the input as the data.
      if (data.size1() == 0)
      {
        data = d;
        return data;
      }
      else
      {
        //go over the rows of the matrix and the sample vector and add the sample vector to the matrix.
        //calculate the correlation between the sample vector and the matrix.
        matrix<real_t> corr ( data.correlation() );
        matrix<real_t> m(data.rows() + 1, data.cols());
        for (size_t i = 0; i < data.rows(); i++)
          for (size_t j = 0; j < data.cols(); j++)
            m(i, j) = data(i, j);
        for (size_t j = 0; j < data.cols(); j++)
          m(data.rows(), j) = d(0, j);
        return m;
      }
    } 


    //returns the correlation between the data and the input
    //or the input and the input columns on zero shot learning.

    virtual matrix<real_t>  extract_features(const std::vector<real_t>& d) 
    { 
      //if the data is empty, we return the input as the data.
      if (data.size1() == 0)
      {
        data = matrix<real_t>(1, d.size());
        for (size_t i = 0; i < d.size(); i++)
          data(0, i) = d[i];
        return data;
      }
      else
      {
        //go over the rows of the matrix and the sample vector and add the sample vector to the matrix.
        //calculate the correlation between the sample vector and the matrix.
        matrix<real_t> corr ( data.correlation() );
        matrix<real_t> m(data.rows() + 1, data.cols());
        for (size_t i = 0; i < data.rows(); i++)
          for (size_t j = 0; j < data.cols(); j++)
            m(i, j) = data(i, j);
        for (size_t j = 0; j < data.cols(); j++)
          m(data.rows(), j) = d[j];
        return m;
      }
    }
    
  };
 
//fill data and labels for classifiers, regressors,vectorizers and clusterers.

  class classifier_stage : public stage_descriptor
  {
    public : 
    
    classifier_stage();
    classifier_stage(const std::string& name);
    virtual ~classifier_stage();
    protected:
    matrix<real_t> data;
    std::vector<size_t> labels;
    typedef provallo::classifier classifier_t;
    std::vector<classifier_t> classifiers;

    public:
    //loads data and labels
    virtual void load_additional_data(const std::string& data);

    //save data and labels
    virtual void save_additional_data(std::string& data);
    
    static classifier_stage* build( );
    protected:
    classifier_factory* factory;

  };
  class regressor_stage : public stage_descriptor
  { 
    public : 
    regressor_stage();
    virtual ~regressor_stage();
    protected:
    matrix<real_t> data;
    std::vector<real_t> labels;

    public:
    //loads data and labels
    virtual void load_additional_data(const std::string& data);
    //save data and labels
    virtual void save_additional_data(std::string& data);
        static regressor_stage* build( ); 

  };
  class outlier_stage : public stage_descriptor
  {
    public : 
    outlier_stage();
    virtual ~outlier_stage();
    protected:
    
    column_sampler<real_t> data;
    
    public:
    virtual void load_additional_data(const std::string& data);
    virtual void save_additional_data(std::string& data);
        static outlier_stage* build( ); 

  };
  class vectorizer_stage : public stage_descriptor
  {
    public : 
    vectorizer_stage();
    virtual ~vectorizer_stage();
     static vectorizer_stage* build( );
         virtual void load_additional_data(const std::string& data);
    virtual void save_additional_data(std::string& data);
    void initialize();
    private:
    vectorizer_type vtype;
    size_t ngram; //
    size_t min_df;//
    size_t max_df;//
    size_t max_features;//
    size_t max_norm;//
    size_t norm;//
    size_t binary;//
    size_t use_idf;//
    size_t smooth_idf;//
    size_t sublinear_tf;//

    std::vector<vectorizer<std::string, real_t> *> vectorizers;
    feature_engineering<std::string,real_t> fe;
    //feature info - for each feature, we have a vector of strings that describe the feature. 
    //for example, for a word, we have the word itself, the part of speech, the lemma, the stem, etc. 
    //for a number, we have the number itself, the number of digits, the number of letters, etc. 
    //for a date, we have the date itself, the day of the week, the month, the year, etc.
    //for a time, we have the time itself, the hour, the minute, the second, etc.
    //for a currency, we have the currency itself, the currency code, the currency symbol, etc.
    //for a location, we have the location itself, the country, the city, the state, the zip code, etc.
    
    std::map<std::string,std::vector<std::string> > feature_info; 
    int feature_engineering_type;
    int feature_selection_type;
    
  };
  class encoder_stage : public stage_descriptor
  {
    public : 
    encoder_stage();
    virtual ~encoder_stage();
     static encoder_stage* build( );
         virtual void load_additional_data(const std::string& data);
    virtual void save_additional_data(std::string& data);



  };

  class decoder_stage : public stage_descriptor
  {
    public : 
    decoder_stage();
    virtual ~decoder_stage();
     static decoder_stage* build( );
         virtual void load_additional_data(const std::string& data);
    virtual void save_additional_data(std::string& data);

  };
  class estimator_stage : public stage_descriptor
  {
    public : 
    estimator_stage();
    virtual ~estimator_stage();
        static estimator_stage* build( );  
            virtual void load_additional_data(const std::string& data);
    virtual void save_additional_data(std::string& data);

  };
  class transform_estimator_stage : public stage_descriptor
  {
    public : 
    transform_estimator_stage();
    virtual ~transform_estimator_stage();

    static transform_estimator_stage* build( );
    virtual void load_additional_data(const std::string& data);
    virtual void save_additional_data(std::string& data);


  };
  class cluster_stage : public stage_descriptor
  {
 


    private: 
    //data for clustering
    matrix<real_t> data;

    //labels for clustering
    std::vector<size_t> labels;

    //number of clusters
    size_t n_clusters;

    //number of iterations
    size_t n_init;

    //number of initializations

    size_t max_iter;


    //tolerance
    real_t tol;
    //random state
    int random_state;
    //verbosity
    int verbose;
    //precompute distances
    bool precompute_distances;
    //copy x
    bool copy_x;
    //algorithm
    std::string algorithm;
    //cluster functions 
    typedef std::vector<size_t> (cluster_stage::*cluster_t)(const matrix<real_t>&, size_t, size_t, size_t, size_t ,size_t) ; 
    //distance functions
    typedef std::vector<real_t> (cluster_stage::*metric_t)(const matrix<real_t>& data )  ; 
    //cluster - kmeans, minibatchkmeans, affinitypropagation, mean_shift, spectral_clustering, agglomerative_clustering, dbscan, birch, gaussian_mixture
    std::map<std::string, cluster_t > algorithms;
    //cluster params
    std::map<std::string, std::string> cluster_params;
    //metric functions:  euclidean, manhattan, cosine, correlation, hamming, jaccard, dice, kulsinski, rogerstanimoto, russellrao, sokalmichener, sokalsneath, yule 
    
    std::map<std::string, metric_t > metrics;
    //metric params
    std::map<std::string, std::string> metric_params;
    typedef std::vector<real_t> (cluster_stage::*score_t)(const matrix<real_t>& data, std::vector<size_t> labels, size_t met, size_t sample_size, size_t random_state, size_t n_jobs);//(const matrix<real_t>&, std::vector<size_t>, size_t, size_t, size_t, size_t, size_t);  
    std::map<std::string,score_t> scores;
    std::map<std::string,std::vector<std::pair<size_t,real_t>> >score_importance; //each score activated is mapped to a vector of pairs (n_clusters, score) 


    //init
    std::string init;
    //n jobs
    int n_jobs;
    //cluster centers
    matrix<real_t> cluster_centers;
    //inertia
    real_t inertia;
    //number of iterations
    size_t n_iter;
    //labels
    std::vector<size_t> labels_;
    //silhouette
    real_t silhouette;
    //calinski harabaz
    real_t calinski_harabaz;
    //davies bouldin
    real_t davies_bouldin;
    //dunn
    real_t dunn;
    //dunn2
    real_t dunn2;


 
     real_t pearson_correlation ( const std::vector<real_t>& x, const std::vector<real_t>& y,real_t mean_x=0.0/*x[0]*/, real_t mean_y=0.0/*y[0]*/) ;

      //metrics to be mapped:
      std::vector<real_t> euclidean_distances(const matrix<real_t>& data);
      std::vector<real_t> manhattan_distances(const matrix<real_t>& data);
      std::vector<real_t> chebyshev_distances(const matrix<real_t>& data);
      std::vector<real_t> minkowski_distances(const matrix<real_t>& data);
      std::vector<real_t> wminkowski_distances(const matrix<real_t>& data);
      std::vector<real_t> seuclidean_distances(const matrix<real_t>& data);
      std::vector<real_t> mahalanobis_distances(const matrix<real_t>& data);
      std::vector<real_t> correlation_distances(const matrix<real_t>& data);
      std::vector<real_t> cosine_distances(const matrix<real_t>& data);
      std::vector<real_t> hamming_distances(const matrix<real_t>& data);
      std::vector<real_t> jaccard_distances(const matrix<real_t>& data);
      std::vector<real_t> dice_distances(const matrix<real_t>& data);
      std::vector<real_t> kulsinski_distances(const matrix<real_t>& data);
      std::vector<real_t> rogerstanimoto_distances(const matrix<real_t>& data);
      std::vector<real_t> russellrao_distances(const matrix<real_t>& data);
      std::vector<real_t> sokalmichener_distances(const matrix<real_t>& data);
      std::vector<real_t> sokalsneath_distances(const matrix<real_t>& data);
      std::vector<real_t> yule_distances(const matrix<real_t>& data);
      std::vector<real_t> braycurtis_distances(const matrix<real_t>& data);
      std::vector<real_t> canberra_distances(const matrix<real_t>& data);
      std::vector<real_t> haversine_distances(const matrix<real_t>& data);
      std::vector<real_t> matcing_distances(const matrix<real_t>& data);
      std::vector<real_t> hellinger_distances(const matrix<real_t>& data);
      std::vector<real_t> jensenshannon_distances(const matrix<real_t>& data);
      std::vector<real_t> rbf_distances(const matrix<real_t>& data);
      std::vector<real_t> spearman_distances(const matrix<real_t>& data);
      std::vector<real_t> kendall_distances(const matrix<real_t>& data);
      std::vector<real_t> squared_sum_distances(const matrix<real_t>& data);
      //algorithm functions (cluster_t signature)
      /*
      std::vector<size_t> kmeans(const matrix<real_t>& data, size_t n_clusters, size_t n_init, size_t max_iter, real_t tol, int random_state, int verbose, bool precompute_distances, bool copy_x);
      std::vector<size_t> minibatchkmeans(const matrix<real_t>& data, size_t n_clusters, size_t batch_size, size_t n_init, size_t max_iter, real_t tol, int random_state, int verbose, bool precompute_distances, bool copy_x);
      std::vector<size_t> affinitypropagation(const matrix<real_t>& data, real_t damping, size_t max_iter, real_t convergence_iter, real_t preference, int verbose, bool copy, bool affinity);
      std::vector<size_t> meanshift(const matrix<real_t>& data, real_t bandwidth, size_t max_iter, real_t cluster_all, int bin_seeding, int min_bin_freq, bool cluster_all_);
      std::vector<size_t> spectralclustering(const matrix<real_t>& data, size_t n_clusters, size_t n_init, size_t max_iter, real_t tol, int random_state, int verbose, bool eigen_solver, bool assign_labels, bool degree, bool coef0, bool kernel_params, bool n_jobs);
      std::vector<size_t> agglomerativeclustering(const matrix<real_t>& data, size_t n_clusters, size_t linkage, size_t affinity, size_t memory, size_t connectivity, size_t compute_full_tree, size_t pooling_func, size_t distance_threshold);
      std::vector<size_t> dbscan(const matrix<real_t>& data, real_t eps, size_t min_samples, size_t metric, size_t metric_params, size_t algorithm, size_t leaf_size, size_t p, size_t n_jobs);
      std::vector<size_t> optics(const matrix<real_t>& data, real_t eps, size_t min_samples, size_t metric, size_t metric_params, size_t algorithm, size_t leaf_size, size_t p, size_t n_jobs);
       std::vector<size_t> gaussianmixture(const matrix<real_t>& data, size_t n_components, size_t covariance_type, size_t tol, size_t reg_covar, size_t max_iter, size_t n_init, size_t init_params, size_t weights_init, size_t means_init, size_t precisions_init, size_t random_state, size_t warm_start, size_t verbose, size_t verbose_interval);
      std::vector<size_t> birch(const matrix<real_t>& data, size_t n_clusters, size_t threshold, size_t branching_factor, size_t compute_labels, size_t copy, size_t n_jobs);
      std::vector<size_t> bicluster(const matrix<real_t>& data, size_t n_clusters, size_t threshold, size_t branching_factor, size_t compute_labels, size_t copy, size_t n_jobs);
      std::vector<size_t> ward(const matrix<real_t>& data, size_t n_clusters, size_t threshold, size_t branching_factor, size_t compute_labels, size_t copy, size_t n_jobs);
      std::vector<size_t> spectralbiclustering(const matrix<real_t>& data, size_t n_clusters, size_t threshold, size_t branching_factor, size_t compute_labels, size_t copy, size_t n_jobs);
      std::vector<size_t> spectralco_clustering(const matrix<real_t>& data, size_t n_clusters, size_t threshold, size_t branching_factor, size_t compute_labels, size_t copy, size_t n_jobs);
 
      */
      
      
      std::vector<size_t> kmeans(const matrix<real_t>&, size_t, size_t, size_t, size_t ,size_t);
      std::vector<size_t> minibatchkmeans(const matrix<real_t>&, size_t, size_t, size_t, size_t ,size_t);
      std::vector<size_t> affinitypropagation(const matrix<real_t>&, size_t, size_t, size_t, size_t ,size_t);
      std::vector<size_t> meanshift(const matrix<real_t>&, size_t, size_t, size_t, size_t ,size_t);
      std::vector<size_t> spectralclustering(const matrix<real_t>&, size_t, size_t, size_t, size_t ,size_t);

      std::vector<size_t> agglomerativeclustering(const matrix<real_t>&, size_t, size_t, size_t, size_t ,size_t);
      std::vector<size_t> dbscan(const matrix<real_t>&, size_t, size_t, size_t, size_t ,size_t);
      std::vector<size_t> optics(const matrix<real_t>&, size_t, size_t, size_t, size_t ,size_t);
      std::vector<size_t> gaussianmixture(const matrix<real_t>&, size_t, size_t, size_t, size_t ,size_t);
      std::vector<size_t> birch(const matrix<real_t>&, size_t, size_t, size_t, size_t ,size_t);
      std::vector<size_t> bicluster(const matrix<real_t>&, size_t, size_t, size_t, size_t ,size_t);
      std::vector<size_t> ward(const matrix<real_t>&, size_t, size_t, size_t, size_t ,size_t);
      std::vector<size_t> spectralbiclustering(const matrix<real_t>&, size_t, size_t, size_t, size_t ,size_t);
      std::vector<size_t> spectralco_clustering(const matrix<real_t>&, size_t, size_t, size_t, size_t ,size_t);

      

      //score functions with score_t signatures:

      std::vector<real_t> silhouette_score(const matrix<real_t>& data, std::vector<size_t> labels, size_t met, size_t sample_size, size_t random_state, size_t n_jobs);

      std::vector<real_t> calinski_harabasz_score(const matrix<real_t>& data, std::vector<size_t> labels, size_t met, size_t sample_size, size_t random_state, size_t n_jobs);
      std::vector<real_t> davies_bouldin_score(const matrix<real_t>& data, std::vector<size_t> labels, size_t met, size_t sample_size, size_t random_state, size_t n_jobs);
      std::vector<real_t> adjusted_rand_score(const matrix<real_t>& data, std::vector<size_t> labels, size_t met, size_t sample_size, size_t random_state, size_t n_jobs);
      std::vector<real_t> adjusted_mutual_info_score(const matrix<real_t>& data, std::vector<size_t> labels, size_t met, size_t sample_size, size_t random_state, size_t n_jobs);
      std::vector<real_t> homogeneity_completeness_v_measure(const matrix<real_t>& data, std::vector<size_t> labels, size_t met, size_t sample_size, size_t random_state, size_t n_jobs);
      std::vector<real_t> completeness_score(const matrix<real_t>& data, std::vector<size_t> labels, size_t met, size_t sample_size, size_t random_state, size_t n_jobs);
      std::vector<real_t> v_measure_score(const matrix<real_t>& data, std::vector<size_t> labels, size_t met, size_t sample_size, size_t random_state, size_t n_jobs);
      std::vector<real_t> fowlkes_mallows_score(const matrix<real_t>& data, std::vector<size_t> labels, size_t met, size_t sample_size, size_t random_state, size_t n_jobs);
      std::vector<real_t> mutual_info_score(const matrix<real_t>& data, std::vector<size_t> labels, size_t met, size_t sample_size, size_t random_state, size_t n_jobs);
      std::vector<real_t> normalized_mutual_info_score(const matrix<real_t>& data, std::vector<size_t> labels, size_t met, size_t sample_size, size_t random_state, size_t n_jobs);
      std::vector<real_t> rand_score(const matrix<real_t>& data, std::vector<size_t> labels, size_t contingency,size_t sample_size, size_t random_state, size_t n_jobs); //ignore parameters 
      std::vector<real_t> homogeneity_score(const matrix<real_t>& data, std::vector<size_t> labels, size_t contingency,size_t sample_size, size_t random_state, size_t n_jobs); //ignore parameters 



             
      std::map<std::string,std::vector<std::pair<size_t,real_t>> >column_importance;
      std::map<std::string,std::vector<std::pair<size_t,real_t>> >row_importance;
      std::map<std::string,std::vector<std::pair<size_t,real_t>> >feature_importance;

    
      //scoring signatures : 
      
    public : 

     
 
 
    cluster_stage()
    {

      //metrics to be mapped:
 
      metrics.insert(std::make_pair("euclidean",metric_t(&cluster_stage::euclidean_distances)));
      metrics.insert(std::make_pair("manhattan",metric_t(&cluster_stage::manhattan_distances)));
      metrics.insert(std::make_pair("chebyshev",metric_t(&cluster_stage::chebyshev_distances)));
      metrics.insert(std::make_pair("minkowski",metric_t(&cluster_stage::minkowski_distances))); 
      metrics.insert(std::make_pair("wminkowski",metric_t(&cluster_stage::wminkowski_distances)));
      metrics.insert(std::make_pair("seuclidean",metric_t(&cluster_stage::seuclidean_distances)));
      metrics.insert(std::make_pair("mahalanobis",metric_t(&cluster_stage::mahalanobis_distances)));
      metrics.insert(std::make_pair("hamming",metric_t(&cluster_stage::hamming_distances)));
      metrics.insert(std::make_pair("jaccard",metric_t(&cluster_stage::jaccard_distances)));
      metrics.insert(std::make_pair("dice",metric_t(&cluster_stage::dice_distances)));
      metrics.insert(std::make_pair("kulsinski",metric_t(&cluster_stage::kulsinski_distances)));
      metrics.insert(std::make_pair("rogerstanimoto",metric_t(&cluster_stage::rogerstanimoto_distances)));
      metrics.insert(std::make_pair("russellrao",metric_t(&cluster_stage::russellrao_distances)));
      metrics.insert(std::make_pair("sokalmichener",metric_t(&cluster_stage::sokalmichener_distances)));
      metrics.insert(std::make_pair("sokalsneath",metric_t(&cluster_stage::sokalsneath_distances)));
      metrics.insert(std::make_pair("yule",metric_t(&cluster_stage::yule_distances)));
      metrics.insert(std::make_pair("braycurtis",metric_t(&cluster_stage::braycurtis_distances)));
      metrics.insert(std::make_pair("canberra",metric_t(&cluster_stage::canberra_distances)));
      metrics.insert(std::make_pair("correlation",metric_t(&cluster_stage::correlation_distances)));
      metrics.insert(std::make_pair("cosine",metric_t(&cluster_stage::cosine_distances)));
      metrics.insert(std::make_pair("haversine",metric_t(&cluster_stage::haversine_distances)));
      metrics.insert(std::make_pair("squared_sum",metric_t(&cluster_stage::squared_sum_distances))); 


      //algorithms to be mapped:
      algorithms.insert(std::make_pair("kmeans",cluster_t(&cluster_stage::kmeans)));
      //algorithms.insert(std::make_pair("kmeans++",cluster_t(&cluster_stage::kmeans_plus_plus)));
      //algorithms.insert(std::make_pair("kmeans||",cluster_t(&cluster_stage::kmeans_parallel)));
      algorithms.insert(std::make_pair("birch",cluster_t(&cluster_stage::birch)));

//      algorithms.insert(std::make_pair("kmeans++",cluster_t(&cluster_stage::kmeans_plus_plus)));
  //    algorithms.insert(std::make_pair("kmeans||",cluster_t(&cluster_stage::kmeans_parallel)));
       
      
      scores.insert(std::make_pair("silhouette",score_t(&cluster_stage::silhouette_score)));
      scores.insert(std::make_pair("calinskiharabasz",score_t(&cluster_stage::calinski_harabasz_score)));
      scores.insert(std::make_pair("daviesbouldin",score_t(&cluster_stage::davies_bouldin_score)));
      //scores.insert(std::make_pair("dunn",score_t(&cluster_stage::dunn_score)));
      //scores.insert(std::make_pair("xiebeni",score_t(&cluster_stage::xie_beni_score)));
      scores.insert(std::make_pair("adjustedrand",score_t(&cluster_stage::adjusted_rand_score)));
      scores.insert(std::make_pair("adjustedmutualinfo",score_t(&cluster_stage::adjusted_mutual_info_score)));
      scores.insert(std::make_pair("completeness",score_t(&cluster_stage::completeness_score)));
      scores.insert(std::make_pair("fowlkesmallows",score_t(&cluster_stage::fowlkes_mallows_score)));
      //scores.insert(std::make_pair("homogeneity",score_t(&cluster_stage::homogeneity_score)));
      scores.insert(std::make_pair("mutualinfo",score_t(&cluster_stage::mutual_info_score)));
      scores.insert(std::make_pair("normalizedmutualinfo",score_t(&cluster_stage::normalized_mutual_info_score)));
      scores.insert(std::make_pair("vmeasure",score_t(&cluster_stage::v_measure_score)));
      //scores.insert(std::make_pair("homogeneitycompleteness",score_t(&cluster_stage::homogeneity_completeness_score)));
      scores.insert(std::make_pair("fowlkesmallows",score_t(&cluster_stage::fowlkes_mallows_score)));
      scores.insert(std::make_pair("silhouette",score_t(&cluster_stage::silhouette_score)));
      
      //initializers to be mapped:

      
    }

    virtual ~cluster_stage()=default;
    //calculate distance matrix 
    //loads data and labels
    virtual void load_additional_data(const std::string& data);
     //save data and labels
    virtual void save_additional_data(std::string& data);
 
    static cluster_stage* build( ); 

    //getters

    //get data
    matrix<real_t>& get_data();

    //get data
    const matrix<real_t>& get_data()const ;

    //get labels
    const std::vector<size_t>& get_labels()const;
    
    size_t get_n_clusters()const;

    //get n_init
    size_t get_n_init()const;

    //get max_iter
    size_t get_max_iter()const;

    //get tol

    real_t get_tol()const;

    //get random_state
    int get_random_state()const;

    //get verbose
    int get_verbose()const;

    //get precompute_distances
    bool get_precompute_distances()const;

    //get copy_x
    bool get_copy_x()const;

    //get algorithm
    std::string get_algorithm()const;

    //get metric
    std::string get_metric()const;

    //get init
    std::string get_init()const;

    //get n_jobs
    int get_n_jobs()const;


    //get cluster_centers
    matrix<real_t>& get_cluster_centers()const;

    //get inertia
    real_t get_inertia();

    //get n_iter
    size_t get_n_iter();

    //get labels_
    std::vector<size_t>& get_labels_()const;
    

    std::vector<real_t> calculate_distance_matrix(const matrix<real_t>& data, const std::string& m);
  };
  class transform_cluster_stage : public stage_descriptor
  {
    public : 
    transform_cluster_stage();
    virtual ~transform_cluster_stage();
        static transform_cluster_stage* build( );
    virtual void load_additional_data(const std::string& data);
    virtual void save_additional_data(std::string& data);
  };
  class dimentionality_reduction_stage  : public stage_descriptor
  {
    public : 
    dimentionality_reduction_stage();
    virtual ~dimentionality_reduction_stage();
        static dimentionality_reduction_stage* build( );
            virtual void load_additional_data(const std::string& data);
    virtual void save_additional_data(std::string& data);


  };  
  class transform_dimentionality_reduction_stage  : public stage_descriptor
  {
    public : 
    transform_dimentionality_reduction_stage();
    virtual ~transform_dimentionality_reduction_stage();
        static transform_dimentionality_reduction_stage* build( );
            virtual void load_additional_data(const std::string& data);
    virtual void save_additional_data(std::string& data);


  };    
  

  //normalizer 
  class normalizer_stage : public stage_descriptor
  {
    public : 
    normalizer_stage();
    virtual ~normalizer_stage();
    static normalizer_stage* build( );
    virtual void load_additional_data(const std::string& data);
    virtual void save_additional_data(std::string& data);

  };
  //standardizer
  class standardizer_stage : public stage_descriptor
  {
    public : 
    standardizer_stage();
    virtual ~standardizer_stage();
    static standardizer_stage* build( );
        virtual void load_additional_data(const std::string& data);
    virtual void save_additional_data(std::string& data);

  };
  //split
  class split_stage : public stage_descriptor
  {
    public : 
    split_stage();
    virtual ~split_stage();
    static split_stage* build( );
  };
  class merge_stage : public stage_descriptor
  {
    public : 
    merge_stage();
    virtual ~merge_stage();
    static merge_stage* build( );
        virtual void load_additional_data(const std::string& data);
    virtual void save_additional_data(std::string& data);



  };

  class join_stage : public stage_descriptor
  {
    public : 
    join_stage();
    virtual ~join_stage();
    static join_stage* build( );
        virtual void load_additional_data(const std::string& data);
    virtual void save_additional_data(std::string& data);

  };
  class concat_stage : public stage_descriptor
  {
    public : 
    concat_stage();
    virtual ~concat_stage();
    static concat_stage* build( );
        virtual void load_additional_data(const std::string& data);
    virtual void save_additional_data(std::string& data);


  };
  class group_stage : public stage_descriptor
  {
    public : 
    group_stage();
    virtual ~group_stage();
    static group_stage* build( );
        virtual void load_additional_data(const std::string& data);
    virtual void save_additional_data(std::string& data);


  };
  class sort_stage : public stage_descriptor
  {
    public : 
    sort_stage();
    virtual ~sort_stage();
    static sort_stage* build( );
        virtual void load_additional_data(const std::string& data);
    virtual void save_additional_data(std::string& data);

  };
  class filter_stage : public stage_descriptor
  {
    public : 
    filter_stage();
    virtual ~filter_stage();
    static filter_stage* build( );
        virtual void load_additional_data(const std::string& data);
    virtual void save_additional_data(std::string& data);

  };
  class sample_stage : public stage_descriptor
  {
    public : 
    sample_stage();
    virtual ~sample_stage();
    static sample_stage* build( );
        virtual void load_additional_data(const std::string& data);
    virtual void save_additional_data(std::string& data);

  };
  class shuffle_stage : public stage_descriptor
  {
    public : 
    static shuffle_stage* build( );
    shuffle_stage();
    virtual ~shuffle_stage();
    protected :
    size_t seed;
    public:
    virtual void load_additional_data(const std::string& data);
    virtual void save_additional_data(std::string& data);

  };
  class splitrows_stage : public stage_descriptor
  {
    public : 
    splitrows_stage();
    virtual ~splitrows_stage();
    //split rows
    virtual void load_additional_data(const std::string& data);
    virtual void save_additional_data(std::string& data);
    
    private:
    size_t seed;
    size_t num_splits;


    //splits are indexes of the matrix rows that are split into different matrices 
 
    std::vector<std::pair<size_t,size_t> > splits; 
    

    public:
    static splitrows_stage* build( );
    
  };
  class splitcols_stage : public stage_descriptor
  {
    public : 
    splitcols_stage();
    virtual ~splitcols_stage();
    virtual void load_additional_data(const std::string& data);
    virtual void save_additional_data(std::string& data);
    static splitcols_stage* build( );

  };
  class pivot_stage : public stage_descriptor
  {
    public : 
    pivot_stage();
    virtual ~pivot_stage();

    static pivot_stage* build( );
    private:
    //criteria for split 
    enum ColCriterion  colum_criterion; 
    enum GainCriterion gain_criterion;
    //splitting criteria
    size_t num_splits;
    size_t min_samples_split;
    size_t min_samples_leaf;
    size_t max_depth;
    size_t max_features;
    size_t seed;
    //splits are indexes of the matrix rows that are split into different matrices

    typedef std::pair<size_t,size_t> sample_index_t;
    //list of samples that are split into different matrices

    typedef std::vector<sample_index_t> sample_index_list_t; 
    //list of samples that are split into different matrices 
    std::vector<sample_index_list_t> splits;
    //list of samples that are split into different matrices
    matrix<real_t> pivot_matrix;
    //imputer for missing values
    struct Imputer imputer;
  };

  
  class unpivot_stage : public stage_descriptor
  {
    public : 
    unpivot_stage();
    virtual ~unpivot_stage();
    static  unpivot_stage* build( );
  };
  class transpose_stage : public stage_descriptor
  {
    public : 
    transpose_stage();
    virtual ~transpose_stage();
    static  transpose_stage* build( );
  };
  class flatten_stage : public stage_descriptor
  {
    public : 
    flatten_stage();
    virtual ~flatten_stage();
    static  flatten_stage* build( );
  };
  class groupby_stage : public stage_descriptor
  {
    public : 
    groupby_stage();
    virtual ~groupby_stage();
    static  groupby_stage* build( );
  };
  class aggregate_stage : public stage_descriptor
  {
    public : 
    aggregate_stage();
    virtual ~aggregate_stage();
    static  aggregate_stage* build( );
  };
  class pivotagg_stage : public stage_descriptor
  {
    public : 
    pivotagg_stage();
    virtual ~pivotagg_stage();
    static  pivotagg_stage* build( );

  };
  class unpivotagg_stage : public stage_descriptor
  {
    public : 
    unpivotagg_stage();
    virtual ~unpivotagg_stage();
    static  unpivotagg_stage* build( );

  };
  class joinagg_stage : public stage_descriptor
  {
    public : 
    joinagg_stage();
    virtual ~joinagg_stage();
    static  joinagg_stage* build( );
  };
  class concatagg_stage : public stage_descriptor
  {
    public : 
    concatagg_stage();
    virtual ~concatagg_stage();
    static  concatagg_stage* build( );

  };
  class sortagg_stage : public stage_descriptor
  {
    public : 
    sortagg_stage();
    virtual ~sortagg_stage();
    static  sortagg_stage* build( );
  };
  class filteragg_stage : public stage_descriptor
  {
    public : 
    filteragg_stage();
    virtual ~filteragg_stage();
    static  filteragg_stage* build( );

  };
  class sampleagg_stage : public stage_descriptor
  {
    public : 
    sampleagg_stage();
    virtual ~sampleagg_stage();
    static  sampleagg_stage* build( );

  };
  class shuffleagg_stage : public stage_descriptor
  {
    public : 
    shuffleagg_stage();
    virtual ~shuffleagg_stage();
    static  shuffleagg_stage* build( );

  };
  class splitrowsagg_stage : public stage_descriptor
  {
    public : 
    splitrowsagg_stage();
    virtual ~splitrowsagg_stage();
    static  splitrowsagg_stage* build( );
  };
  class splitcolsagg_stage : public stage_descriptor
  {
    public : 
    splitcolsagg_stage();
    virtual ~splitcolsagg_stage();
    static  splitcolsagg_stage* build( );
  };
  class transposeagg_stage : public stage_descriptor
  {
    public : 
    transposeagg_stage();
    virtual ~transposeagg_stage();
  };
  class flattenagg_stage : public stage_descriptor
  {
    public : 
    flattenagg_stage();
    virtual ~flattenagg_stage();
    static  flattenagg_stage* build( );
  };
  class groupbyagg_stage : public stage_descriptor
  {
    public : 
    groupbyagg_stage();
    virtual ~groupbyagg_stage();
    static  groupbyagg_stage* build( );
  };
  class aggregateagg_stage : public stage_descriptor
  {
    public : 
    aggregateagg_stage();
    virtual ~aggregateagg_stage();
    static  aggregateagg_stage* build( );
  };


  class classdist_stage : public stage_descriptor
  {
    public : 
    
    classdist_stage();
    static  classdist_stage* build( );  

    virtual ~classdist_stage();
  };
  //Knowledge transfer stage implmenets the following algorithms 
  //https://arxiv.org/pdf/1511.06440.pdf
  //https://arxiv.org/pdf/2006.16331.pdf
  //where the transfer function, accuracy and loss function are defined as follows 
  //Transfer function
  //f(x) = x
  //Accuracy function
  //a(x) = 1 if x > 0.5 else 0
  //Loss function
  //l(x) = (1 − x)^2
  //The transfer function is the identity function
  //The accuracy function is the step function
  //The loss function is the squared error function
  
  //Training can be formulated as minimizing the error function
  //J(X; θ) := ∑a∈X`(a; θ) 
  //where X` is the training set and θ is the model parameters
  //The error function is minimized by gradient descent
  //θ := θ − α∇θJ(X; θ)
  //where α is the learning rate
  //The gradient is computed by backpropagation
  //∇θJ(X; θ) = ∑a∈X`∇θ(a; θ)
  //where ∇θ(a; θ) is the gradient of the error function for a single training example
  

  class knowledge_transfer_stage : public stage_descriptor
  {
    public : 
    enum transfer_strategy{
      NONE,
      TRANSFER,
      ADAPT,
      HYPERPARAMETER_TRANSFER,
      HYPERPARAMETER_ADAPT,
      HYPERPARAMETER_TRANSFER_ADAPT
    };
    enum knowledge_transfer_function_t{
      COPY,
      LINEAR,
      EXPONENTIAL,
      LOGARITHMIC,
      SIGMOID
    };
    enum transfer_accuracy_t{
      ACCURACY,
      F1,
      PRECISION,
      RECALL,
      ROC_AUC,
      PR_AUC,
      KAPPA,
      MCC,
      BAC,
      BMAC,
      BACC,
      BMA
    };

    struct accuracy_t{
      transfer_accuracy_t accuracy;
      real_t value;
    };

    struct measure_t{
      transfer_strategy strategy;
      knowledge_transfer_function_t transfer_function;
      //list of accuracies for the success of the transfer 
      //and the harmonic infomation transfer

      std::vector<accuracy_t> accuracies;

    }; 

    knowledge_transfer_stage();
    static  knowledge_transfer_stage* build( );  

    virtual ~knowledge_transfer_stage();
    
    private:

    typedef std::pair< matrix<real_t> ,matrix<real_t> > knowledge_adapter_t;
    typedef std::pair<std::vector<real_t>,std::vector<real_t> > knowledge_transfer_hyperparam_t;




    
    //define the transfer function
    //define model to transfer
    //define the output model
    std::vector<knowledge_adapter_t> knowledge_adapters; 
    std::vector<knowledge_transfer_hyperparam_t> knowledge_transfer_hyperparam_adapters; 
    //define the transfer function
    typedef   real_t (knowledge_transfer_stage::*knowledge_transfer_function_func)(real_t,real_t,real_t,real_t ) ; 
     //transfer function
    knowledge_transfer_function_t transfer_function;

    //transfer function parameters
    real_t transfer_function_alpha;
    real_t transfer_function_beta;
    real_t transfer_function_gamma;
    real_t transfer_function_delta;
    //transfer function
    knowledge_transfer_function_t transfer_function_hyperparam;
    //transfer function parameters
    real_t transfer_function_hyperparam_alpha;
    real_t transfer_function_hyperparam_beta;
    real_t transfer_function_hyperparam_gamma;
    real_t transfer_function_hyperparam_delta;

    //errors , error gradients and error deltas 
    
    std::vector<real_t> errors;
    std::vector<real_t> error_gradients;
    std::vector<real_t> error_deltas;
    
    //create pairwise distributions for the transfer function 
    //and the hyperparameter transfer function
    std::vector<real_t> pairwise_distributions;
    std::vector<real_t> pairwise_distributions_hyperparam;

    //create pairwise distributions for the transfer function 
    //and the hyperparameter transfer function
    std::vector<real_t> pairwise_distributions_gradients;
    std::vector<real_t> pairwise_distributions_hyperparam_gradients; 
    //variational auto encoder for the transfer function 
    //and the hyperparameter transfer function

    variational_auto_encoder<real_t>* transfer_function_autoencoder; 
    variational_auto_encoder<real_t>* transfer_function_hyperparam_autoencoder; 
    //transfer function
    knowledge_transfer_function_func transfer_function_func; 
    knowledge_transfer_function_func transfer_function_hyperparam_func; 
    //evaluation function
    transfer_accuracy_t transfer_accuracy;
    transfer_accuracy_t transfer_accuracy_hyperparam;
    
    //measure accuracy of the transfer function:
    //accuracy function
    typedef   real_t (knowledge_transfer_stage::*transfer_accuracy_func)(real_t,real_t ) ; 
    transfer_accuracy_func _accuracy_func;
    transfer_accuracy_func _accuracy_hyperparam_func;
    //accuracy function parameters
    real_t transfer_accuracy_alpha;
    real_t transfer_accuracy_beta;
    real_t transfer_accuracy_gamma;
    real_t transfer_accuracy_delta;
    //accuracy function parameters
    real_t transfer_accuracy_hyperparam_alpha;
    real_t transfer_accuracy_hyperparam_beta;
    real_t transfer_accuracy_hyperparam_gamma;
    real_t transfer_accuracy_hyperparam_delta;

    //implement functions same signature fit the hyperparameter transfer function:
    //F1 score:
    real_t transfer_accuracy_f1(real_t tp,real_t fp,real_t fn,real_t tn); 
    //accuracy:
    real_t transfer_accuracy_accuracy(real_t tp,real_t fp,real_t fn,real_t tn);
    //precision:
    real_t transfer_accuracy_precision(real_t tp,real_t fp,real_t fn,real_t tn);
    //recall
    real_t transfer_accuracy_recall(real_t tp,real_t fp,real_t fn,real_t tn);
    //ROC AUC
    real_t transfer_accuracy_roc_auc(real_t tp,real_t fp,real_t fn,real_t tn);
    //PR AUC
    real_t transfer_accuracy_pr_auc(real_t tp,real_t fp,real_t fn,real_t tn);
    //Kappa 
    real_t transfer_accuracy_kappa(real_t tp,real_t fp,real_t fn,real_t tn);

    //informdness
    real_t transfer_accuracy_informedness(real_t tp,real_t fp,real_t fn,real_t tn);
    
    //Matthews correlation coefficient
    real_t transfer_accuracy_mcc(real_t tp,real_t fp,real_t fn,real_t tn);
    //Balanced accuracy
    real_t transfer_accuracy_bac(real_t tp,real_t fp,real_t fn,real_t tn);
    //Balanced multiclass accuracy
    real_t transfer_accuracy_bmac(real_t tp,real_t fp,real_t fn,real_t tn);
    //Balanced multiclass recall
    real_t transfer_accuracy_bmrc(real_t tp,real_t fp,real_t fn,real_t tn);
    //Balanced multiclass precision
    real_t transfer_accuracy_bmpc(real_t tp,real_t fp,real_t fn,real_t tn);
    //Balanced multiclass F1 score
    real_t transfer_accuracy_bmf1(real_t tp,real_t fp,real_t fn,real_t tn);
    //Balanced multiclass Matthews correlation coefficient
    real_t transfer_accuracy_bmmcc(real_t tp,real_t fp,real_t fn,real_t tn);
    //Balanced multiclass Kappa
    real_t transfer_accuracy_bmkappa(real_t tp,real_t fp,real_t fn,real_t tn);
    //Balanced multiclass ROC AUC
    real_t transfer_accuracy_bmroc_auc(real_t tp,real_t fp,real_t fn,real_t tn);
    //Balanced multiclass PR AUC
    real_t transfer_accuracy_bmpr_auc(real_t tp,real_t fp,real_t fn,real_t tn);
    //specificity
    real_t transfer_accuracy_specificity(real_t tp,real_t fp,real_t fn,real_t tn);
    //sensitivity
    real_t transfer_accuracy_sensitivity(real_t tp,real_t fp,real_t fn,real_t tn);
    //F1 score:
    real_t transfer_accuracy_f1 (const std::vector<real_t> &tp,const std::vector<real_t> &fp,const std::vector<real_t> &fn,const std::vector<real_t> &tn);  
    //accuracy:
    real_t transfer_accuracy_accuracy (const std::vector<real_t> &tp,const std::vector<real_t> &fp,const std::vector<real_t> &fn,const std::vector<real_t> &tn);
    //precision:
    real_t transfer_accuracy_precision (const std::vector<real_t> &tp,const std::vector<real_t> &fp,const std::vector<real_t> &fn,const std::vector<real_t> &tn);
    //recall
    real_t transfer_accuracy_recall (const std::vector<real_t> &tp,const std::vector<real_t> &fp,const std::vector<real_t> &fn,const std::vector<real_t> &tn);
    //ROC AUC
    real_t transfer_accuracy_roc_auc (const std::vector<real_t> &tp,const std::vector<real_t> &fp,const std::vector<real_t> &fn,const std::vector<real_t> &tn);
    //  PR AUC
    real_t transfer_accuracy_pr_auc (const std::vector<real_t> &tp,const std::vector<real_t> &fp,const std::vector<real_t> &fn,const std::vector<real_t> &tn);
    //  Kappa
    real_t transfer_accuracy_kappa (const std::vector<real_t> &tp,const std::vector<real_t> &fp,const std::vector<real_t> &fn,const std::vector<real_t> &tn);

    //informdness
    real_t transfer_accuracy_informedness (const std::vector<real_t> &tp,const std::vector<real_t> &fp,const std::vector<real_t> &fn,const std::vector<real_t> &tn);

  
    //  Matthews correlation coefficient
    real_t transfer_accuracy_mcc (const std::vector<real_t> &tp,const std::vector<real_t> &fp,const std::vector<real_t> &fn,const std::vector<real_t> &tn);
    //balanced accuracy
    real_t transfer_accuracy_bac (const std::vector<real_t> &tp,const std::vector<real_t> &fp,const std::vector<real_t> &fn,const std::vector<real_t> &tn);
    //multiclass
    real_t transfer_accuracy_bmac (const std::vector<real_t> &tp,const std::vector<real_t> &fp,const std::vector<real_t> &fn,const std::vector<real_t> &tn);
    //recall for multiclass
    real_t transfer_accuracy_bmrc (const std::vector<real_t> &tp,const std::vector<real_t> &fp,const std::vector<real_t> &fn,const std::vector<real_t> &tn);
    //precision for multiclass
    real_t transfer_accuracy_bmpc (const std::vector<real_t> &tp,const std::vector<real_t> &fp,const std::vector<real_t> &fn,const std::vector<real_t> &tn);
    //f1 for multiclass
    real_t transfer_accuracy_bmf1 (const std::vector<real_t> &tp,const std::vector<real_t> &fp,const std::vector<real_t> &fn,const std::vector<real_t> &tn);
    //mcc
    real_t transfer_accuracy_bmmcc (const std::vector<real_t> &tp,const std::vector<real_t> &fp,const std::vector<real_t> &fn,const std::vector<real_t> &tn);
    //kappa 
    real_t transfer_accuracy_bmkappa (const std::vector<real_t> &tp,const std::vector<real_t> &fp,const std::vector<real_t> &fn,const std::vector<real_t> &tn);
    //broc auc
    real_t transfer_accuracy_bmroc_auc (const std::vector<real_t> &tp,const std::vector<real_t> &fp,const std::vector<real_t> &fn,const std::vector<real_t> &tn);
    //bpr auc
    real_t transfer_accuracy_bmpr_auc (const std::vector<real_t> &tp,const std::vector<real_t> &fp,const std::vector<real_t> &fn,const std::vector<real_t> &tn);

    //specificity 
    real_t transfer_accuracy_specificity (const std::vector<real_t> &tp,const std::vector<real_t> &fp,const std::vector<real_t> &fn,const std::vector<real_t> &tn);
    //sensitivity
    real_t transfer_accuracy_sensitivity (const std::vector<real_t> &tp,const std::vector<real_t> &fp,const std::vector<real_t> &fn,const std::vector<real_t> &tn);
    // fawlkes mallows index  
    real_t transfer_accuracy_fawlkes_mallows_index (const std::vector<real_t> &tp,const std::vector<real_t> &fp,const std::vector<real_t> &fn); 
    
    real_t transfer_accuracy_markedness(const std::vector<real_t> &tp,const std::vector<real_t> &fp,const std::vector<real_t> &fn); 
    real_t transfer_accuracy_gmean(const std::vector<real_t> &tp,const std::vector<real_t> &fp,const std::vector<real_t> &fn);
    
    real_t transfer_accuracy_score;
    real_t transfer_f1_score;  
    real_t transfer_precision_score ;
    real_t transfer_recall_score ;
    real_t transfer_tp_score ;
    real_t transfer_fp_score ;
    real_t transfer_fn_score ;
    real_t transfer_tn_score ;

    
  

    };  
    //utility class to build stages using ::build() method mapped to stage type.
  class stage_factory 
  {
    public:
    typedef std::unordered_map<std::string, stage_descriptor* (*)()> stage_map;
    static stage_map& get_map()
    {
      static stage_map map;
      return map;
    }
    static stage_descriptor* build(const std::string& type)
    {
      stage_map& map = get_map();
      stage_map::iterator it = map.find(type);
      if(it == map.end())
      {
        return NULL;
      }
      return it->second();
    }
    static void register_stage(const std::string& type, stage_descriptor* (*builder)())
    {
      stage_map& map = get_map();
      map[type] = builder;
    }
    stage_descriptor* create_stage(const std::string& category,const std::string& opname)
    {
      //stage is created using the factory if the stage is not registered, it will return NULL. 
      
      stage_descriptor* stage = build( category);
      if(stage)
      {
        stage->name = opname;
      }
      return stage;
      stage->set_parameter("name", opname);
      stage->set_parameter("type", category);
      return stage;
    } 

  };
  //singleton stage factory

  class stage_factory_singleton : public singleton<stage_factory_singleton>
  {
    std::mutex _mtx;
    std::map<std::string, std::function<stage_descriptor *()>> _stage_map;

  public:
    stage_descriptor *build_stage(const stage_descriptor &stage)
    {
      // build the stage from the stage descriptor
      stage_descriptor *new_stage = nullptr;
      // create stage from the stage descriptor

      auto THIS = stage_factory_singleton::get_instance();
      // lock the mutex
      std::lock_guard<std::mutex> lock(THIS->_mtx);

      if (THIS->_stage_map.find(stage.name) != THIS->_stage_map.end())
      {
        new_stage = THIS->_stage_map[stage.name](); // returns new_stage

        // copy the stage descriptor
        new_stage->set_descriptor(stage);
      }
      else
        throw std::runtime_error("stage not found");

      return new_stage;
    }
    std::vector<std::string> get_stage_names()
    {
      std::lock_guard<std::mutex> lock(_mtx);    
      std::vector<std::string> names;
      for (auto &stage : _stage_map)
      {
        names.push_back(stage.first);
      }
      return names;
    }
    std::vector<std::string> get_stage_categories()
    {
      std::vector<std::string> categories;
      std::lock_guard<std::mutex> lock(_mtx);
    
      for (auto &stage : _stage_map)
      {
        categories.push_back(stage.second()->get_category());
      }
      return categories;
    } 

    std::vector<std::string> get_types(const std::string& category)
    {
      std::vector<std::string> types;
      std::lock_guard<std::mutex> lock(_mtx);
      for (auto &stage : _stage_map)
      {
        if(stage.second()->get_category() == category)
        {
          types.push_back(stage.first);
        }
      }
      return types; 
    }
    void register_stage(const std::string &name, std::function<stage_descriptor *()> builder)
    {
      // lock the mutex
      std::lock_guard<std::mutex> lock(_mtx);
      _stage_map[name] = builder;
    } 



    stage_factory_singleton() : _mtx(), _stage_map()
    {
      // initialize the stage map

      _stage_map["dataset"] = &dataset_stage::build;
      _stage_map["vectorizer"] = &vectorizer_stage::build;
      _stage_map["feature"] = &feature_stage::build;
      _stage_map["cluster"] = &cluster_stage::build;
      _stage_map["classifier"] = &classifier_stage::build;
      _stage_map["regressor"] = &regressor_stage::build;
      _stage_map["dimensionality_reduction"] = &dimentionality_reduction_stage::build;
      //evaluation
      _stage_map["evaluation"] = &knowledge_transfer_stage::build;
      
      _stage_map["encoder"] = &encoder_stage::build;
      _stage_map["decoder"] = &decoder_stage::build;
      _stage_map["normalizer"] = &normalizer_stage::build;
      _stage_map["filter"] = &filter_stage::build;
      
    }
    // create stage from the stage descriptor
  }; // end of stage factory

  class pipeline
  { 
    public:
    
    std::vector<pipeline*> _pipelines;//container for aggregated pipeline stages 
 
    std::vector<stage_descriptor*> _stages;

    bool load_stage(  std::ifstream& stage_file, const stage_descriptor& stage ) ;
    
    friend std::ostream& operator<<(std::ostream& os, const pipeline& p);
    friend std::ifstream & operator>>(std::ifstream & is, pipeline & p);
  
    bool load_from_file(const std::string& filename);
    //recursively save all pipeline elements to the file
    bool save_to_file(const std::string& filename);
    
    //build all pipeline elements from the file and configure the settings
    bool load_from_string(std::string pipeline_string);
    //save all pipeline elements to the file
    bool save_to_string(std::string &pipeline_string);
    //construct the pipeline from the elements
    bool build();
    //run the pipeline
    bool run();
    //run the pipeline
    bool run(const std::string& input, std::string& output);
    //constructor
    pipeline();
    
      //get/set stages 
      void add_stage(stage_descriptor* stage);
      void add_stage(const std::string& category,const std::string& opname);

  
      void remove_stage(uint64_t stage_id); 
  
      stage_descriptor* get_stage(uint64_t stage_id);
      stage_descriptor* get_stage( const std::string& stage_name); //could be multiple stages with the same name, return the first one 
      stage_descriptor* get_next_stage(uint64_t stage_id);
      stage_descriptor* get_previous_stage(uint64_t stage_id);
      stage_descriptor* get_first_stage();
      stage_descriptor* get_last_stage();


    //get/set number of stages
     size_t get_number_of_stages() const;
    
    //get/set pipelines

    std::string get_pipeline_name() const{return _pipe_name;}
    void set_pipeline_name(const std::string& name){_pipe_name=name;}
    inline uint64_t get_pipeline_id() const{return _pipe_id;}
    inline void set_pipeline_id(uint64_t id){_pipe_id=id;}
    
    //file load constructor 
    explicit pipeline(const std::string& file_);
    //destructor
    virtual ~pipeline();
    pipeline* get_pipeline(uint64_t pipeline_id);
    pipeline* get_pipeline(const std::string& pipeline_name);

    private : 

    uint64_t _pipe_id; 
    std::string _pipe_name;


  };

  //aggregate a pipline of pipeline stages
  class pipeline_stage : public stage_descriptor
  {
    public : 
    pipeline_stage();
    pipeline_stage(const std::string& name);
    virtual ~pipeline_stage();
    static pipeline_stage* build( );
        virtual void load_additional_data(const std::string& data);
    virtual void save_additional_data(std::string& data);
    pipeline * get_pipeline(){return _pipeline;}
    //when pipeline is already loaded and we want to place it inside another pipeline
    void set_pipeline(pipeline * p){_pipeline=p;}

    private:
    pipeline * _pipeline;

  };
   
  class pipeline_builder
  { 
     // a pipeline stage can have multipe pipeline descriptors for each pipeline 
     //  a pipeline descriptor can have multiple stages
     public:
    //build 
    
    //add stage manually: 

    //read_pipelines from file
    pipeline_builder& read_pipelines(const std::string & filename);
    //write_pipelines to file
    pipeline_builder& write_pipelines(const std::string & filename);
    void add_pipeline(pipeline* pipeline);
    void add_pipeline(const std::string & pipeline_name,bool load_from_file=false);
    void set_current_pipeline(const std::string & pipeline_name);
    void set_current_pipeline(uint64_t index);

    void remove_pipeline(const std::string & pipeline_name);
    void remove_pipeline(uint64_t index);


    pipeline* get_pipeline(const std::string & pipeline_name);
    pipeline* get_pipeline(uint64_t index);
    pipeline* get_current_pipeline();
    const pipeline* get_current_pipeline() const;


    size_t get_number_of_pipelines() const;
    //returns the number of stages in the current pipeline
    size_t get_current_pipeline_size() const;
    //returns the number of stages in all the pipelines
    size_t get_number_of_stages() const;
    
    
    pipeline_builder (pipeline_builder &&other);
    pipeline_builder&
    operator= (const pipeline_builder &other);
    pipeline_builder&
    operator= (pipeline_builder &&other);
    pipeline_builder (const pipeline_builder &other);
    virtual
    ~pipeline_builder ();
    pipeline_builder ();
    pipeline_builder (const std::string & filename , bool load_from_file=false);
    pipeline_builder (const std::string & filename , const std::string & pipeline_name, bool load_from_file=false);


    friend std::ostream & operator<< (std::ostream & os, const pipeline_builder & p);
    friend std::ifstream & operator>> (std::ifstream & is, pipeline_builder & p);
    bool build();
    bool run();
    bool run(const std::string& input, std::string& output);
    bool load_from_file(std::string filename);
    bool save_to_file(std::string filename);
    protected:
    std::vector<pipeline*> _pipelines;
    pipeline* _current_pipeline;
    std::string _filename;
    std::string _name;

 };

 class meta_builder : public pipeline_builder
 {
   public : 
   //initialize the meta-builder with the meta-file
   meta_builder(const std::ifstream& meta_file);
   //initialize a new meta-builder settings and saves it to the meta-file
   meta_builder(const std::string& meta_name, const std::string& meta_file_path);

   virtual ~meta_builder();

   bool build();
   bool run();
   bool run(const std::string& input, std::string& output);

   bool load_from_file(std::string filename);
   bool save_to_file(std::string filename);
     //generate stages by GAN :
    pipeline* generate_pipeline();
    //generate stages by GAN :
    pipeline* generate_pipeline(const std::string& pipeline_name);
    //generate stages by GAN :
    pipeline* generate_pipeline(const std::string& pipeline_name, const std::string& input, std::string& output);
  private :
  //meta-parameters :

  struct meta_weight{
  //weights for the meta-parameters :
  real_t accuracy; //accuracy of the pipeline
  real_t loss; //loss of the pipeline
  real_t cost; //cost of the pipeline
  real_t complexity;//complexity of the pipeline
  } _meta_weight, _meta_weight_step, _meta_weight_threshold;


  //meta-parameters :
  //number of pipelines to generate
  uint64_t _number_of_pipelines;
  //number of threads per pipeline
  uint64_t _number_of_threads;
  //number of maximum stages per pipeline
  uint64_t _max_number_of_stages;
  //number of maximum stages per pipeline
  uint64_t _min_number_of_stages; // should be 2 at least, otherwise it is not a pipeline
  //maximum numbers of aggregated pipelines to generate per pipeline
  uint64_t _max_number_of_aggregated_pipelines;
  //learning task configuration
  learning_task _learning_task;
  std::map<std::string, auto_encoder<real_t>> _stage_output_autoencoders;
  std::map<std::string, auto_encoder<real_t>> _stage_input_autoencoders;
  //
  neural_helper _neural_helper; //helper for neural networks
  

 };

  //pretrained models loaders: 
  //load the tokenizer and 
  //the pre-trained word embeddings 
  //transforms it using the pre-trained word embeddings


  template <typename T> 
  class AutoTokenizer 
  {
    //import auto tokenizer from hugging face 
    const std::string _model_name; 
    const std::string _model_path; 
    const std::string _tokenizer_path; 
    const std::string _word_embeddings_path; 
    const std::string _word_embeddings_model; 
    const std::string _word_embeddings_type; 

    public :
    AutoTokenizer(const std::string& model_name, const std::string& model_path, const std::string& tokenizer_path, const std::string& word_embeddings_path, const std::string& word_embeddings_model, const std::string& word_embeddings_type)   
    : _model_name(model_name), _model_path(model_path), _tokenizer_path(tokenizer_path), _word_embeddings_path(word_embeddings_path), _word_embeddings_model(word_embeddings_model), _word_embeddings_type(word_embeddings_type)
    {
      //load the tokenizer and the pre-trained word embeddings 
      //transforms it using the pre-trained word embeddings
    } 
    //load the tokenizer and the pre-trained word embeddings from the pre-trained path : 
    void load(const std::string& model_name, const std::string& model_path, const std::string& tokenizer_path, const std::string& word_embeddings_path, const std::string& word_embeddings_model, const std::string& path)
    {
 
    } 
  };

} /* namespace provallo */

#endif /* DECISION_ENGINE_PIPELINEBUILDER_H_ */
