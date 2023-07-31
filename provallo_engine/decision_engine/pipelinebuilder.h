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
#include <iostream>
#include <set>
namespace provallo
{

  //python style estimators (fit/predict)
  enum vectorizer_type  {
      TFIDF,
      STANDARD_SCALER,
      MIN_MAX_SCALER,
      PCA,
      NeuralTransformer,
      AeroNautics, // tensor operator pitch/yaw/roll matrices
      SingularValueDecomposition,
      HPlane,
      Huffman,
      Hmm
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

  class pipeline_builder
  {

    std::set<estimator<Float>*> _estimators;
    std::set<transform_estimator<Float>*> _transform_estimators;
    std::set<encoder*> _encoders;
    std::set<decoder*> _decoders;
    std::set<vectorizer<Float,Float>*> _vectorizers;
    std::set<vectorizer<Float,Float>*> _transform_vectorizers;

    std::set<classifier*> _classifiers; //classifiers
    std::set<iso_forest*> _iso_forests; //outliers filters 
    

    
  public:
    pipeline_builder (pipeline_builder &&other);
    pipeline_builder&
    operator= (const pipeline_builder &other);
    pipeline_builder&
    operator= (pipeline_builder &&other);
    pipeline_builder (const pipeline_builder &other);
    virtual
    ~pipeline_builder ();
    pipeline_builder ();
 };


} /* namespace provallo */

#endif /* DECISION_ENGINE_PIPELINEBUILDER_H_ */
