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
      StandardScaler,
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
    estimator();
    estimator(estimator<real_x> &&other);


  public:

    virtual  std::vector<real_x> fit( const provallo::matrix<real_x>& )=0;
    virtual  std::vector<real_x> predict(const provallo::matrix<real_x>& )=0;
    virtual ~estimator()=0;
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
    virtual  vectorizer_type get_type()const;

    //Learn and estimate the parameters of the transformation
    virtual  std::vector<real_x> fit ( const matrix<real_x>& train_data );

    //Apply the learned transformation to new data
    virtual std::vector<real_x> transform(const matrix<real_x>& test_data);
    //Learn the parameters and apply the transformation to new data
    virtual matrix<real_x> fit_transform();

    //transform()	Apply the learned transformation to new data	transformed_data = estimator.transform(X)

    //transformed_data = estimator.transform()

    //fit_transform()	Learn the parameters and apply the transformation to new data	transformed_data = estimator.fit_transform(X)	transformed_data = estimator.fit_transform(data)

   };
  //vectorizer implements transofrm_estimator 
  template <typename vector_src, typename real_x>
  class vectorizer : public transform_estimator<real_x>
  {
    protected:
    vectorizer_type _type;
    vector_src _data;
    std::vector<real_x> _transformed_data;
    std::vector<real_x> _fitted_data;
    std::vector<real_x> _predicted_data;
    public:
    vectorizer(vectorizer_type type);

    vectorizer(vectorizer<vector_src,real_x> &&other); //move constructor
    vectorizer(const vectorizer<vector_src,real_x> &other); //copy constructor
    
    vectorizer<vector_src,real_x>& operator= (const vectorizer<vector_src,real_x> &other);
    vectorizer<vector_src,real_x>&
        operator= (vectorizer<vector_src,real_x> &&other);

    const vector_src& get_data() const ;
    vector_src& get_data() ;

    void set_data(vector_src& data);  

    virtual  vectorizer_type get_type()const;
    virtual  std::vector<real_x> fit( const provallo::matrix<real_x>& );
    virtual  std::vector<real_x> predict(const provallo::matrix<real_x>& );
    virtual  std::vector<real_x> transform(const provallo::matrix<real_x>& );
    virtual  std::vector<real_x> fit_transform(const provallo::matrix<real_x>& );
    virtual ~vectorizer();

  };

class tfidf 
{
  protected:
  std::vector<std::string> _documents;
  std::vector<std::string> _vocabulary;
  std::vector<std::vector<double>> _tfidf;
  std::vector<double> _idf;
  std::vector<double> _tf;
  
  public:
  tfidf();
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
    std::string inverse_transform(const std::vector<double>& vector);
  //transform
    std::vector<double> transform(const std::string& document);

   //get_tf
    std::vector<double> get_tf( )const;

  //get_idf
    std::vector<double> get_idf( )const;

  //get_tfidf 
    std::vector<std::vector<double>> get_tfidf( )const;

  //get_vocabulary
    const std::vector<std::string>& get_vocabulary()const ;

  //get_documents
    const std::vector<std::string>& get_documents()const;
  
  
  //get_tf_matrix
    std::vector<std::vector<double>> get_tf_matrix();

  //get_idf_matrix
    std::vector<std::vector<double>> get_idf_matrix();

  //get_tf_idf_matrix
    std::vector<std::vector<double>> get_tf_idf_matrix();

  

  
};

class tfidf_vectorizer : public vectorizer<std::string, double>
{
  protected:
  class tfidf _tfidf;

  public:
  
  tfidf_vectorizer();
  tfidf_vectorizer(const tfidf_vectorizer &other);
  tfidf_vectorizer(tfidf_vectorizer &&other); //move constructor
  tfidf_vectorizer& operator= (const tfidf_vectorizer &other);
  tfidf_vectorizer&
      operator= (tfidf_vectorizer &&other);
  
  virtual  std::vector<double> fit( const std::vector<std::string>&documents );
  virtual  std::vector<double> predict(const std::vector<std::string>&documents );
  virtual  std::vector<double> transform(const std::vector<std::string>&documents);
  virtual  std::vector<double> fit_transform(const std::vector<std::string>&documents);
 
  //for use with inverse transformation matrices 
  virtual std::vector<double> fit( const provallo::matrix<double>& );
  virtual std::vector<double> predict(const provallo::matrix<double>& );
  virtual std::vector<double> transform(const provallo::matrix<double>& );
  virtual std::vector<double> fit_transform(const provallo::matrix<double>& );


  virtual ~tfidf_vectorizer();
};
class scaler;
class standard_scaler_vectorizer : public vectorizer<std::string, double>
{
  protected:
  std::vector<double> _mean;
  std::vector<double> _variance;
  std::vector<double> _standard_deviation;
  std::vector<double> _standardized_data;

  public:

  standard_scaler_vectorizer();
  standard_scaler_vectorizer(standard_scaler_vectorizer &&other); //move constructor
  standard_scaler_vectorizer& operator= (const standard_scaler_vectorizer &other);
  standard_scaler_vectorizer&
      operator= (standard_scaler_vectorizer &&other);
  

  virtual  std::vector<double> fit( const provallo::matrix<double>& );
  virtual  std::vector<double> predict(const provallo::matrix<double>& );
  virtual  std::vector<double> transform(const provallo::matrix<double>& );
  virtual  std::vector<double> fit_transform(const provallo::matrix<double>& );


  virtual ~standard_scaler_vectorizer();
};


class min_max_scaler_vectorizer : public vectorizer<std::string, double>
{
  protected:
  std::vector<double> _min;
  std::vector<double> _max;
  std::vector<double> _min_max_data;

  public:



  min_max_scaler_vectorizer();
  min_max_scaler_vectorizer(min_max_scaler_vectorizer &&other); //move constructor
  min_max_scaler_vectorizer& operator= (const min_max_scaler_vectorizer &other);
  min_max_scaler_vectorizer&
      operator= (min_max_scaler_vectorizer &&other);


  virtual  std::vector<double> fit( const provallo::matrix<double>& );
  virtual  std::vector<double> predict(const provallo::matrix<double>& );
  virtual  std::vector<double> transform(const provallo::matrix<double>& );
  virtual  std::vector<double> fit_transform(const provallo::matrix<double>& );


  virtual ~min_max_scaler_vectorizer();
};

class normalizer_vectorizer : public vectorizer<std::string, double>
{
  protected:
  std::vector<double> _norm;
  std::vector<double> _normalized_data;
  


  public:

  normalizer_vectorizer();
  normalizer_vectorizer(normalizer_vectorizer &&other); //move constructor
  normalizer_vectorizer& operator= (const normalizer_vectorizer &other);
  normalizer_vectorizer&
      operator= (normalizer_vectorizer &&other);
  
  virtual std::vector<double> fit( const provallo::matrix<double>& );
  virtual std::vector<double> predict(const provallo::matrix<double>& );
  virtual std::vector<double> transform(const provallo::matrix<double>& );
  virtual std::vector<double> fit_transform(const provallo::matrix<double>& );


  virtual ~normalizer_vectorizer();




};

class principal_component_analysis
{

  std::vector<double> _mean;
  std::vector<double> _variance;
  std::vector<double> _standard_deviation;
  std::vector<double> _standardized_data;
  std::vector<double> _covariance_matrix;
  std::vector<double> _eigen_values;
  std::vector<double> _eigen_vectors;
  std::vector<double> _pca_data;
  std::vector<double> _pca_components;
  std::vector<double> _pca_explained_variance;
  std::vector<double> _pca_explained_variance_ratio;
  std::vector<double> _pca_singular_values;
  std::vector<double> _pca_noise_variance;
  std::vector<double> _pca_mean;
  std::vector<double> _pca_n_components;
  std::vector<double> _pca_n_features;
  std::vector<double> _pca_n_samples;
  std::vector<double> _pca_n_components_;
  std::vector<double> _pca_n_features_;
  std::vector<double> _pca_n_samples_;

  public:
  principal_component_analysis();
  principal_component_analysis(principal_component_analysis &&other); //move constructor
  principal_component_analysis& operator= (const principal_component_analysis &other);
  principal_component_analysis&
      operator= (principal_component_analysis &&other);

  //fit 
  virtual std::vector<double> fit( const provallo::matrix<double>& );
  virtual std::vector<double> predict(const provallo::matrix<double>& );
  virtual std::vector<double> transform(const provallo::matrix<double>& );
  virtual std::vector<double> fit_transform(const provallo::matrix<double>& );

  //getters
  virtual std::vector<double> get_mean();
  virtual std::vector<double> get_variance();
  virtual std::vector<double> get_standard_deviation();
  virtual std::vector<double> get_standardized_data();
  virtual std::vector<double> get_covariance_matrix();
  virtual std::vector<double> get_eigen_values();
  virtual std::vector<double> get_eigen_vectors();
  virtual std::vector<double> get_pca_data();
  virtual std::vector<double> get_pca_components();
  virtual std::vector<double> get_pca_explained_variance();
  virtual std::vector<double> get_pca_explained_variance_ratio();
  virtual std::vector<double> get_pca_singular_values();
  virtual std::vector<double> get_pca_noise_variance();
  virtual std::vector<double> get_pca_mean();
  virtual std::vector<double> get_pca_n_components();
  virtual std::vector<double> get_pca_n_features();
  virtual std::vector<double> get_pca_n_samples();
  virtual std::vector<double> get_pca_n_components_();
  virtual std::vector<double> get_pca_n_features_();
  virtual std::vector<double> get_pca_n_samples_();
  //setters
  virtual void set_mean(std::vector<double> );
  virtual void set_variance(std::vector<double> );
  virtual void set_standard_deviation(std::vector<double> );
  virtual void set_standardized_data(std::vector<double> );
  virtual void set_covariance_matrix(std::vector<double> );
  virtual void set_eigen_values(std::vector<double> );
  virtual void set_eigen_vectors(std::vector<double> );
  virtual void set_pca_data(std::vector<double> );
  virtual void set_pca_components(std::vector<double> );
  virtual void set_pca_explained_variance(std::vector<double> );
  virtual void set_pca_explained_variance_ratio(std::vector<double> );
  virtual void set_pca_singular_values(std::vector<double> );
  virtual void set_pca_noise_variance(std::vector<double> );
  virtual void set_pca_mean(std::vector<double> );
  virtual void set_pca_n_components(std::vector<double> );
  virtual void set_pca_n_features(std::vector<double> );
  virtual void set_pca_n_samples(std::vector<double> );
  virtual void set_pca_n_components_(std::vector<double> );
  virtual void set_pca_n_features_(std::vector<double> );
  virtual void set_pca_n_samples_(std::vector<double> );

  void QRDecomposition(const matrix<double>& mtx,matrix<double>& Q,  matrix<double>& R);
  std::vector<double> mul ( const matrix<double>& mtx, const std::vector<double>& vec)
  {
    std::vector<double> result;
    for (int i = 0; i < mtx.rows(); i++)
    {
      double sum = 0;
      for (int j = 0; j < mtx.cols(); j++)
      {
        sum += mtx(i,j) * vec[j];
      }
      result[i]=sum;
    }
    return result;
  }
  void gauss_jordan_elimination(matrix<double>&elim  )
  {
      for ( size_t col =0;col < elim.cols();col++)
      {


          size_t max_v = col;
          double max_value = elim(col,col);

          for (size_t row =col + 1;row < elim.rows();row++)
          {
              if ( fabs ( elim(row,col) ) > fabs(max_value) ) 
              {

                  max_value = elim(row,col);
                  max_v = row;
              }
              
          }
               //swap the rows
          if(row!=max_v)
          {
                  for (size_t ix = 0; ix < rows(); ++ix)
                    std::swap(elim.element(col,ix), elim.element (max_v,ix);
                    

          }

          //scale
          double scale = elim(col,col);
          for ( size_t ix = 0;ix < elim.rows();ix++)
          {
              elim.element(col,ix) /= scale;
          }
          for ( size_t iy=0;iy < elim.rows();iy++)
          {
              if ( iy != col)
              {
                  double scale = elim(iy,col);
                  if(scale==0.0)continue;

                  for ( size_t ix = 0;ix < elim.rows();ix++)
                  {
                      elim.element(iy,ix) -= scale * elim.element(col,ix);
                  }
              }
          }

      }//end of for loop

  }
  matrix<double> gram_schmidt(const matrix<double>& mtx)
  {
    matrix<double> result(mtx.rows(),mtx.cols());
    matrix<double> Q(mtx.rows(),mtx.cols());
    matrix<double> R(mtx.rows(),mtx.cols());
    QRDecomposition(mtx,Q,R);
    result = Q * R;
    return result;
  }   
//destructor
   virtual ~principal_component_analysis();


};


class pca_vectorizer : public vectorizer<std::string, double>
{
  protected:

  

  public:
  pca_vectorizer();
  pca_vectorizer(pca_vectorizer &&other); //move constructor
  pca_vectorizer& operator= (const pca_vectorizer &other);
  pca_vectorizer&
      operator= (pca_vectorizer &&other);
  
  virtual std::vector<double> fit( const provallo::matrix<double>& ); 
  virtual std::vector<double> predict(const provallo::matrix<double>& );
  virtual std::vector<double> transform(const provallo::matrix<double>& );
  virtual std::vector<double> fit_transform(const provallo::matrix<double>& );

  
  virtual ~pca_vectorizer();

};

class lda_vectorizer : public vectorizer<std::string, double>
{
  protected:
  std::vector<double> _lda_data;
  std::vector<double> _lda_components;
  std::vector<double> _lda_explained_variance;
  std::vector<double> _lda_explained_variance_ratio;
  std::vector<double> _lda_singular_values;
  std::vector<double> _lda_noise_variance;
  std::vector<double> _lda_mean;
  std::vector<double> _lda_n_components;
  std::vector<double> _lda_n_features;
  std::vector<double> _lda_n_samples;
  std::vector<double> _lda_n_components_;
  std::vector<double> _lda_n_features_;
  std::vector<double> _lda_n_samples_;




  public:
  lda_vectorizer();
  lda_vectorizer(lda_vectorizer &&other); //move constructor
  lda_vectorizer& operator= (const lda_vectorizer &other);
  lda_vectorizer&
      operator= (lda_vectorizer &&other);

  virtual std::vector<double> fit( const provallo::matrix<double>& );
  virtual std::vector<double> predict(const provallo::matrix<double>& );
  virtual std::vector<double> transform(const provallo::matrix<double>& );
  virtual std::vector<double> fit_transform(const provallo::matrix<double>& );


  virtual ~lda_vectorizer();

};


class tsne_vectorizer : public vectorizer<std::string, double>
{
  protected:
  std::vector<double> _tsne_data;
  std::vector<double> _tsne_components;
  std::vector<double> _tsne_explained_variance;
  std::vector<double> _tsne_explained_variance_ratio;
  std::vector<double> _tsne_singular_values;
  std::vector<double> _tsne_noise_variance;
  std::vector<double> _tsne_mean;
  std::vector<double> _tsne_n_components;
  std::vector<double> _tsne_n_features;
  std::vector<double> _tsne_n_samples;
  std::vector<double> _tsne_n_components_;
  std::vector<double> _tsne_n_features_;
  std::vector<double> _tsne_n_samples_;




  public:
  tsne_vectorizer();
  tsne_vectorizer(tsne_vectorizer &&other); //move constructor
  tsne_vectorizer& operator= (const tsne_vectorizer &other);
  tsne_vectorizer&
      operator= (tsne_vectorizer &&other);
      
  virtual std::vector<double> fit( const provallo::matrix<double>& );
  virtual std::vector<double> predict(const provallo::matrix<double>& );
  virtual std::vector<double> transform(const provallo::matrix<double>& );
  virtual std::vector<double> fit_transform(const provallo::matrix<double>& );

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
