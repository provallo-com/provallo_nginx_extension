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
      NEURAL_TRANSFORMER,
      AERONATIC_QARTERION, // tensor operator pitch/yaw/roll matrices
      SVD_OPERATOR,
      HPLANE_TRANSFORMER,
      HUFFMAN_TRANSFORMER,
      HMM_TRANSFORMER,REGRESSION_TRANSFORMER
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
    
    virtual ~vectorizer() = default;
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
//pca helper class
class bag_of_words 
{
  //bag of words
  public:
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
  protected:
  //bag of words
  std::vector<std::string> _vocabulary;
  std::vector<real_t> _bow;
  std::vector<std::vector<real_t>> _bow_transformed;
  std::vector<std::vector<real_t>> _bow_transformed_inverse;

  matrix<real_t> _bow_matrix;


  //add document and process document helper functions 
  virtual void add_document(const std::string&);
  virtual void process_document(const std::string&);



};



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
  private:

  principal_component_analysis _pca;
    bag_of_words _bow;//bag of words vectorizer
  
};

class lda_vectorizer : public vectorizer<std::string, real_t>
{
  protected: 



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
  private:
  //labels 
  std::vector<std::string> _labels;
  real_t _alpha;
  real_t _beta;
  real_t _eta;
  real_t _gamma;
  real_t _theta;
  real_t _lambda;
  size_t _n_iter;
  size_t _n_topics;
  size_t _n_features;
  size_t _n_samples;
  size_t _n_components;
  size_t _n_top_words;
  size_t _n_jobs;
  size_t _random_state;
  real_t _doc_topic_prior;
  real_t _topic_word_prior;
  real_t _learning_decay;
  real_t _learning_offset;
  size_t _max_doc_update_iter;
  size_t _total_samples;
  real_t _mean_change_tol;
  



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
    return os;
  }
   
  //lda
  //bow
  //inverse_transform
  //inverse_transform_matrix
  //inverse_transform_matrix_
  
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
    void set_stage_id(uint64_t id) { stage_id = id; }
    size_t get_stage_id() const { return stage_id; }

    //process data 
    virtual void process_data(const std::string& data) { UNDEF_REFERENCE(data);UNDEF_REFERENCE2(data);}
    virtual void process_data(const matrix<real_t>& data) { UNDEF_REFERENCE(data);UNDEF_REFERENCE2(data);}
    virtual void process_data(const std::vector<real_t>& data) { UNDEF_REFERENCE(data);UNDEF_REFERENCE2(data);}
    virtual void process_data(const std::vector<std::vector<real_t> >& data) { UNDEF_REFERENCE(data);UNDEF_REFERENCE2(data);}

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
    
    //additional data for data sets would be :
    //1. set/subset of the data.
    //2. the data types.
    //3. the data parameters.
    
    virtual const std::string get_additional_data() const;
    virtual void set_additional_data(const std::string& data);


    static dataset_stage* build( );
    enum dataset_type { STATIC, DYNAMIC } _type;
    enum dataset_purpose { BUILD_TRAIN, TRAIN, OPTIMIZE_TRAIN, TEST,OPTIMIZE_TEST, VALIDATE,XVALIDATE } _purpose; 
    dataset_type type() const { return _type; }
    
    void type(dataset_type t) { _type = t; }
    bool is_static() const { return _type == STATIC; }
    bool is_dynamic() const { return _type == DYNAMIC; }
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


    //process data
    // implements the virtual functions of stage_descriptor 
    // decents should override process_data with matrix_base
    // 
    virtual void process_data(const std::string& data); 
    virtual void process_data(const matrix<real_t>& data);
    virtual void process_data(const std::vector<real_t>& data);
    virtual void process_data(const std::vector<std::vector<real_t> >& data);

    //adapter for dataset_ptr or matrix_base 
    //converts matrix_base to matrix<real_t> and processes the dataset. 

    virtual void process_data(matrix_base& data); 
    private:
    class_dist _class_dist;
    //labels:
    std::vector<size_t> _labels;
    std::vector<std::string> _label_names; 

    public:
    void set_labels(const std::vector<std::string>& labels);

  };


  class dynamic_dataset_stage : public dataset_stage
  {

    //dynamic dataset stage is a dataset stage that can be modified before or during the training process. 
    //it is used for example to add new data to the training set during the training process. 
    //it is also used to add new data to the training set during the training process. 

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
  
  class static_dataset_stage : public dataset_stage
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
  
  class feature_extraction_paramters 
  {
    //comprehensive list of feature extraction parameters.
  public :
    feature_extraction_paramters() : features_length(0), parameters(""), feature_extraction_method(nullptr)
     {
 
        //initialize the feature extraction method.
        auto empty_fcp = [](matrix<real_t> data) -> std::vector<real_t>  { 
                     matrix<real_t> ev=data.eigenvalues(),ev2=data.eigenvectors() ; return std::vector<real_t>(ev.begin(),ev.end());  }; 
        feature_extraction_method = empty_fcp;

        
     }
    feature_extraction_paramters(const feature_extraction_paramters& cpy)
    {
      features_length = cpy.features_length;
      parameters = cpy.parameters;
      feature_extraction_method = cpy.feature_extraction_method;
    }
    feature_extraction_paramters& operator=(const feature_extraction_paramters& cpy)
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

    virtual ~feature_extraction_paramters() {}

  private: 
    //size_t method_id
    size_t features_length;
    //feature extraction parameters.
    std::string parameters;
      //feature extraction method.
    std::function< std::vector<real_t> (matrix<real_t> ) > feature_extraction_method  ;

  };

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
  


  //end of feature_extraction_paramters class.
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
    feature_extraction_paramters _feature_extraction_paramters;

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
    
    void extract(const vec_src& src, matrix<real_>& dst, const feature_extraction_paramters& params) { _extractor.extract(src, dst, params) ; }
    void transform(const vec_src& src, matrix<real_>& dst, const feature_extraction_paramters& params) { _transformer.transform(src, dst, params); }
    void select(const vec_src& src, matrix<real_>& dst, const feature_extraction_paramters& params) { _selector.select(src, dst, params); }

    void aggregate(const vec_src& src, matrix<real_>& dst, const feature_extraction_paramters& params) { _aggregator.aggregate(src, dst, params); }
    void normalize(const vec_src& src, matrix<real_>& dst, const feature_extraction_paramters& params) { _normalizer.normalize(src, dst, params); }
    void weight(const vec_src& src, matrix<real_>& dst, const feature_extraction_paramters& params) { _weighter.weight(src, dst, params); }
    void binarize(const vec_src& src, matrix<real_>& dst, const feature_extraction_paramters& params) { _binarizer.binarize(src, dst, params); }
    void filter(const vec_src& src, matrix<real_>& dst, const feature_extraction_paramters& params) { _filter.filter(src, dst, params); }
    void reduce(const vec_src& src, matrix<real_>& dst, const feature_extraction_paramters& params) { _reducer.reduce(src, dst, params); }
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
  //end of feature_extraction_paramters class.
  class feature_stage : public stage_descriptor
  {
    private :

    feature_extraction_paramters _extraction_functors;
    feature_engineering<std::string,real_t> _engineering_functors;


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

    } 
    feature_stage(const feature_stage& cpy) : stage_descriptor(cpy)
    {
      data = cpy.data;  
    
      _extraction_functors = cpy._extraction_functors;

    }
    virtual ~feature_stage();
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
    
    vectorizer_type type;
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
    std::vector<feature_extractor<std::string, real_t> *> extractors;
    std::vector<feature_transformer<std::string, real_t> *> transformers;
    std::vector<feature_selector<std::string, real_t> *> selectors;
    std::vector<feature_aggregator<std::string, real_t> *> aggregators;
    std::vector<feature_normalizer<std::string, real_t> *> normalizers;
    std::vector<feature_weighter<std::string, real_t> *> weighters;
    std::vector<feature_binarizer<std::string, real_t> *> binarizers;
    std::vector<feature_filter<std::string, real_t> *> filters;
    std::vector<feature_reducer<std::string, real_t> *> reducers;

    
    
 
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

    virtual ~cluster_stage();
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
  //Knoledge transfer stage implmenets the following algorithms 
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

    variational_auto_encoder<real_t> transfer_function_autoencoder; 
    variational_auto_encoder<real_t> transfer_function_hyperparam_autoencoder; 
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


    //get/set number of stages
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

  learning_task _learning_task;
  std::map<std::string, auto_encoder<real_t>> _stage_output_autoencoders;
  std::map<std::string, auto_encoder<real_t>> _stage_input_autoencoders;
  
  real_t _accuracy; //accuracy of the pipeline
  real_t _loss; //loss of the pipeline
  real_t _cost; //cost of the pipeline
  real_t _complexity;//complexity of the pipeline

  real_t _accuracy_weight; //accuracy of the pipeline
  real_t _loss_weight; //loss of the pipeline
  real_t _cost_weight; //cost of the pipeline
  real_t _complexity_weight;//complexity of the pipeline

  real_t _accuracy_threshold; //accuracy of the pipeline
  real_t _loss_threshold; //loss of the pipeline
  real_t _cost_threshold; //cost of the pipeline
  real_t _complexity_threshold;//complexity of the pipeline

  real_t _accuracy_weight_step; //accuracy of the pipeline
  real_t _loss_weight_step; //loss of the pipeline
  real_t _cost_weight_step; //cost of the pipeline
  real_t _complexity_weight_step;//complexity of the pipeline
   
 
 };


} /* namespace provallo */

#endif /* DECISION_ENGINE_PIPELINEBUILDER_H_ */
