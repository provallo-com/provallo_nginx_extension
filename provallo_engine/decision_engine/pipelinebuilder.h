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
#include "../parsers/parser.h" //for the encoders/decoders
#include "autoencoder.h"
#include "neuralhelper.h"
#include <vector>
#include <iostream>
#include <set>
namespace provallo
{

  //python style estimators (fit/predict)
  enum vectorizer_type  : uint8_t {
      TFIDF,
      STANDARD_SCALER,
      MIN_MAX_SCALER,
      PCA,
      NeuralTransformer,
      AeroNautics, // tensor operator pitch/yaw/roll matrices
      SingularValueDecomposition,
      HPlane,
      Huffman,
      Hmm,Regressor
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


  template <typename vector_src, typename real_x>
  class vectorizer : public transform_estimator<real_x>
  {
    protected:
    vectorizer_type _type;
    vector_src _data;
    std::vector<real_x> _transformed_data;
    matrix<real_x> _fitted_data;
    std::vector<real_x> _predicted_data;
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
 

    virtual  vectorizer_type get_type()const
    {
      return _type;
    }
    //fit:
    virtual  std::vector<real_x> fit( const std::vector<vector_src>& data_){DEFAULT_IMPL(data_);}
    virtual  std::vector<real_x> predict(const std::vector<vector_src>& data_){DEFAULT_IMPL(data_);}
    virtual  std::vector<real_x> transform(const std::vector<vector_src>& data_){DEFAULT_IMPL(data_);}
    virtual  std::vector<real_x> fit_transform(const std::vector<vector_src>& data_){ DEFAULT_IMPL(data_);}



    //inverse:
    virtual  std::vector<real_x> fit( const provallo::matrix<real_x>&data_ ){DEFAULT_IMPL(data_);}
    virtual  std::vector<real_x> predict(const provallo::matrix<real_x>& data_){ DEFAULT_IMPL(data_);}

    virtual  std::vector<real_x> transform(const provallo::matrix<real_x>& data_){ DEFAULT_IMPL(data_);}
    virtual ~vectorizer(){
      
    }

  };

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


  //inverse transform
    std::string inverse_transform(const std::vector<real_t>& vector);


    std::vector<std::string> inverse_transform (const std::vector<std::vector<real_t>> &corpus) ;
  //transform
    std::vector<real_t> transform(const std::string& document);

//transform
    std::vector<std::vector<real_t> >  transform(const std::vector<std::string>& document);
    

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

  virtual  std::vector<real_t> fit( const std::vector<std::string>&documents );
  virtual  std::vector<real_t> predict(const std::vector<std::string>&documents );
  virtual  std::vector<real_t> transform(const std::vector<std::string>&documents);
  virtual  std::vector<real_t> fit_transform(const std::vector<std::string>&documents);
 
  //for use with inverse transformation matrices 
  virtual std::vector<real_t> fit( const provallo::matrix<real_t>& );
  virtual std::vector<real_t> predict(const provallo::matrix<real_t>& );
  virtual std::vector<real_t> transform(const provallo::matrix<real_t>& );
  virtual ~tfidf_vectorizer();

  protected: 
  //case by case
  std::vector<real_t> predict (const std::string&);
  std::vector<real_t> transform(const std::string&);

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

class principal_component_analysis
{

  public:
  principal_component_analysis();
  principal_component_analysis(principal_component_analysis &&other); //move constructor
  principal_component_analysis& operator= (const principal_component_analysis &other);
  principal_component_analysis&
      operator= (principal_component_analysis &&other);

  //fit 
  virtual std::vector<real_t> fit( const provallo::matrix<real_t>& );
  virtual std::vector<real_t> predict(const provallo::matrix<real_t>& );
  virtual std::vector<real_t> transform(const provallo::matrix<real_t>& );
  
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

      }//end of for loop

  }
  matrix<real_t> gram_schmidt(const matrix<real_t>& mtx)
  {
    matrix<real_t> result(mtx.rows(),mtx.cols());
    matrix<real_t> Q(mtx.rows(),mtx.cols());
    matrix<real_t> R(mtx.rows(),mtx.cols());
    QRDecomposition(mtx,Q,R);
    result = Q * R;
    return result;
  }   


  virtual  std::vector<real_t> fit( const std::vector<std::string>&documents );
  virtual  std::vector<real_t> predict(const std::vector<std::string>&documents );
  virtual  std::vector<real_t> transform(const std::vector<std::string>&documents);
  virtual  std::vector<real_t> fit_transform(const std::vector<std::string>&documents);
 
  //for use with inverse transformation matrices 
  //destructor
   virtual ~principal_component_analysis();
  
  
  protected:
  std::vector<real_t> _mean;
  std::vector<real_t> _variance;
  std::vector<real_t> _standard_deviation;
  std::vector<real_t> _standardized_data;
  matrix<real_t> _covariance_matrix;
  matrix<real_t>  _eigen_values;
  matrix<real_t>  _eigen_vectors;
  matrix<real_t> _pca_data;
  matrix<real_t> _pca_components;
  matrix<real_t> _pca_explained_variance;

  matrix<real_t> _pca_explained_variance_ratio;
  matrix<real_t> _pca_singular_values;
  std::vector<real_t> _pca_noise_variance;
  real_t _pca_mean;
  std::vector<real_t> _pca_n_components;
  std::vector<real_t> _pca_n_features;
  std::vector<real_t> _pca_n_samples;
  std::vector<real_t> _pca_n_components_;
  std::vector<real_t> _pca_n_features_;
  std::vector<real_t> _pca_n_samples_;
 

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
  
  virtual  std::vector<real_t> fit( const std::vector<std::string>&documents );
  virtual  std::vector<real_t> predict(const std::vector<std::string>&documents );
  virtual  std::vector<real_t> transform(const std::vector<std::string>&documents);
  virtual  std::vector<real_t> fit_transform(const std::vector<std::string>&documents);
 
  //for use with inverse transformation matrices 
  virtual std::vector<real_t> fit( const provallo::matrix<real_t>& ); 
  virtual std::vector<real_t> predict(const provallo::matrix<real_t>& );
  virtual std::vector<real_t> transform(const provallo::matrix<real_t>& );
 

  //override get_type
  virtual vectorizer_type get_type() const ;
  
  virtual ~pca_vectorizer();
  principal_component_analysis _pca;
};

class lda_vectorizer : public vectorizer<std::string, real_t>
{
  protected:
  std::vector<real_t> _lda_data;
  std::vector<real_t> _lda_components;
  std::vector<real_t> _lda_explained_variance;
  std::vector<real_t> _lda_explained_variance_ratio;
  std::vector<real_t> _lda_singular_values;
  std::vector<real_t> _lda_noise_variance;
  std::vector<real_t> _lda_mean;
  std::vector<real_t> _lda_n_components;
  std::vector<real_t> _lda_n_features;
  std::vector<real_t> _lda_n_samples;
  std::vector<real_t> _lda_n_components_;
  std::vector<real_t> _lda_n_features_;
  std::vector<real_t> _lda_n_samples_;




  public:
  lda_vectorizer();
  lda_vectorizer(lda_vectorizer &&other); //move constructor
  lda_vectorizer& operator= (const lda_vectorizer &other);
  lda_vectorizer&
      operator= (lda_vectorizer &&other);

  
  
  virtual  std::vector<real_t> fit( const std::vector<std::string>&documents );
  virtual  std::vector<real_t> predict(const std::vector<std::string>&documents );
  virtual  std::vector<real_t> transform(const std::vector<std::string>&documents);
  virtual  std::vector<real_t> fit_transform(const std::vector<std::string>&documents);
 
  //for use with inverse transformation matrices 
  
  virtual std::vector<real_t> fit( const provallo::matrix<real_t>& );
  virtual std::vector<real_t> predict(const provallo::matrix<real_t>& );
  virtual std::vector<real_t> transform(const provallo::matrix<real_t>& );
 

  virtual ~lda_vectorizer();

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
 
  virtual ~tsne_vectorizer();
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


  class regressor : public vectorizer<std::string, real_t>
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
  
    virtual ~regressor();
    //private regressor functions :
    private:
    //linear_regression functions
    std::vector<real_t> *linear_regression_fit( const matrix<real_t>  & foot, const std::vector<real_t> & target );
    std::vector<real_t> *linear_regression_predict( matrix<real_t>  foot );
    std::vector<real_t> *linear_regression_transform( matrix<real_t>  foot );
    std::vector<real_t> *linear_regression_fit_transform( matrix<real_t>  foot );
    //ridge_regression functions
    std::vector<real_t> *ridge_regression_fit( const matrix<real_t>  & foot, const std::vector<real_t> & target );
    std::vector<real_t> *ridge_regression_predict( matrix<real_t>  foot );
    std::vector<real_t> *ridge_regression_transform( matrix<real_t>  foot );
    std::vector<real_t> *ridge_regression_fit_transform( matrix<real_t>  foot );
    //lasso_regression functions
    std::vector<real_t> *lasso_regression_fit( const matrix<real_t>  & foot, const std::vector<real_t> & target );
    std::vector<real_t> *lasso_regression_predict( matrix<real_t>  foot );
    std::vector<real_t> *lasso_regression_transform( matrix<real_t>  foot );
    std::vector<real_t> *lasso_regression_fit_transform( matrix<real_t>  foot );
    //elastic_net functions
    std::vector<real_t> *elastic_net_fit( const matrix<real_t>  & foot, const std::vector<real_t> & target );
    std::vector<real_t> *elastic_net_predict( matrix<real_t>  foot );
    std::vector<real_t> *elastic_net_transform( matrix<real_t>  foot );
    std::vector<real_t> *elastic_net_fit_transform( matrix<real_t>  foot );
    //bayesian_ridge functions
    std::vector<real_t> *bayesian_ridge_fit( const matrix<real_t>  & foot, const std::vector<real_t> & target );
    std::vector<real_t> *bayesian_ridge_predict( matrix<real_t>  foot );
    std::vector<real_t> *bayesian_ridge_transform( matrix<real_t>  foot );
    std::vector<real_t> *bayesian_ridge_fit_transform( matrix<real_t>  foot );
    //logistic_regression functions

    std::vector<real_t> *logistic_regression_fit( const matrix<real_t>  & foot, const std::vector<real_t> & target );
    std::vector<real_t> *logistic_regression_predict( matrix<real_t>  foot );
    std::vector<real_t> *logistic_regression_transform( matrix<real_t>  foot );
    std::vector<real_t> *logistic_regression_fit_transform( matrix<real_t>  foot );

    //svm functions

    std::vector<real_t> *svm_fit( const matrix<real_t>  & foot, const std::vector<real_t> & target );
    std::vector<real_t> *svm_predict( matrix<real_t>  foot );
    std::vector<real_t> *svm_transform( matrix<real_t>  foot );
    std::vector<real_t> *svm_fit_transform( matrix<real_t>  foot );
    
    //decision_tree functions
    
    std::vector<real_t> *decision_tree_fit( const matrix<real_t>  & foot, const std::vector<real_t> & target );
    std::vector<real_t> *decision_tree_predict( matrix<real_t>  foot );
    std::vector<real_t> *decision_tree_transform( matrix<real_t>  foot );
    std::vector<real_t> *decision_tree_fit_transform( matrix<real_t>  foot );

    //random_forest functions

    std::vector<real_t> *random_forest_fit( const matrix<real_t>  & foot, const std::vector<real_t> & target );
    std::vector<real_t> *random_forest_predict( matrix<real_t>  foot );
    std::vector<real_t> *random_forest_transform( matrix<real_t>  foot );
    std::vector<real_t> *random_forest_fit_transform( matrix<real_t>  foot );

    //gradient_boosting functions

     
  };

  struct stage_descriptor
  {
    public:
    size_t stage_id;
    std::string name;
    std::string type;
    std::string parameters;
    std::string input;
    std::string output;
    std::string input_type;
    std::string output_type;
    std::string input_parameters;
    std::string output_parameters;
    //constructor from data.

    stage_descriptor (const std::string& line) : stage_id(0), name(""), type(""), parameters(""), input(""), output(""), input_type(""), output_type(""), input_parameters(""), output_parameters("")
    {
      std::stringstream ss(line);
      ss >> stage_id >> name >> type >> parameters >> input >> output >> input_type >> output_type >> input_parameters >> output_parameters;  
    }
    //default
    stage_descriptor () : stage_id(0), name(""), type(""), parameters(""), input(""), output(""), input_type(""), output_type(""), input_parameters(""), output_parameters("") {} 

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

    }    
    friend std::ostream& operator<<(std::ostream& os, const stage_descriptor& sd);
    friend std::istream& operator>>(std::istream& is, stage_descriptor& sd);

    virtual const std::string get_additional_data() const { return ""; }
    virtual void set_additional_data(const std::string& data) { UNDEF_REFERENCE(data);UNDEF_REFERENCE2(data);}
    virtual ~stage_descriptor() {}

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

    static dataset_stage* build( );

  };


  class dynamic_dataset_stage : public stage_descriptor
  {
    public : 
    dynamic_dataset_stage();
    virtual ~dynamic_dataset_stage();
    protected:
    matrix<real_t> data;
    public:
    virtual void load_additional_data(const std::string& data);
    virtual void save_additional_data(std::string& data); 
    static dynamic_dataset_stage* build( );

  };
  
  class static_dataset_stage : public stage_descriptor
  {
    public : 
    static_dataset_stage();
    virtual ~static_dataset_stage();
    protected:
    matrix<real_t> data;
    public:
    virtual void load_additional_data(const std::string& data);
    virtual void save_additional_data(std::string& data); 
    static static_dataset_stage* build( );

  };  

  class feature_stage : public stage_descriptor
  {
    public : 
    feature_stage();
    virtual ~feature_stage();
    protected:
    matrix<real_t> data;
    public:
    virtual void load_additional_data(const std::string& data);
    virtual void save_additional_data(std::string& data);

    static feature_stage* build( );

  };
 

  class classifier_stage : public stage_descriptor
  {
    public : 
    classifier_stage();
    virtual ~classifier_stage();
    protected:
    matrix<real_t> data;
    std::vector<size_t> labels;

    public:
    //loads data and labels
    virtual void load_additional_data(const std::string& data);

    //save data and labels
    virtual void save_additional_data(std::string& data);
    
    static classifier_stage* build( );


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
    public : 
    cluster_stage();
    virtual ~cluster_stage();
        static cluster_stage* build( );
            virtual void load_additional_data(const std::string& data);
    virtual void save_additional_data(std::string& data);



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
  
  //aggregate a pipline of pipeline stages
  class pipeline_stage : public stage_descriptor
  {
    public : 
    pipeline_stage();
    virtual ~pipeline_stage();
    static pipeline_stage* build( );
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

  class pipeline
  { 
    public:
    
    std::vector<pipeline_stage*> _pipelines;
 
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
 
      void remove_stage(uint64_t stage_id); 
      stage_descriptor* get_stage(uint64_t stage_id);
      stage_descriptor* get_stage( const std::string& stage_name); //could be multiple stages with the same name, return the first one 
      stage_descriptor* get_next_stage(uint64_t stage_id);
      stage_descriptor* get_previous_stage(uint64_t stage_id);
      stage_descriptor* get_first_stage();
      stage_descriptor* get_last_stage();

     size_t get_number_of_stages() const;
    
    //get/set pipelines

    std::string get_pipeline_name() const;
    void set_pipeline_name(const std::string& name);
    inline uint64_t get_pipeline_id() const{return _pipe_id;}
    inline void set_pipeline_id(uint64_t id){_pipe_id=id;}

    //file load constructor 
    explicit pipeline(const std::string& file_);
    //destructor
    virtual ~pipeline();
    private : 
    uint64_t _pipe_id; 
    std::string _pipe_name;
    

  };
   
  class pipeline_builder
  { 
     // a pipeline stage can have multipe pipeline descriptors for each pipeline 
     //  a pipeline descriptor can have multiple stages
     protected:
    std::vector<pipeline*> _pipelines;
    pipeline* _current_pipeline;
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

    
    
    pipeline_builder (pipeline_builder &&other);
    pipeline_builder&
    operator= (const pipeline_builder &other);
    pipeline_builder&
    operator= (pipeline_builder &&other);
    pipeline_builder (const pipeline_builder &other);
    virtual
    ~pipeline_builder ();
    pipeline_builder ();
    friend std::ostream & operator<< (std::ostream & os, const pipeline_builder & p);
    friend std::ifstream & operator>> (std::ifstream & is, pipeline_builder & p);
    bool build();
    bool run();
    bool run(const std::string& input, std::string& output);
    bool load_from_file(std::string filename);
    bool save_to_file(std::string filename);

 };

 class meta_builder : public pipeline_builder
 {
   public : 
   meta_builder();
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

 };
} /* namespace provallo */

#endif /* DECISION_ENGINE_PIPELINEBUILDER_H_ */
