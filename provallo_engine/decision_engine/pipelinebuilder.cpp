/*
 * pipelinebuilder.cpp
 *
 *  Created on: Jun 19, 2023
 *      Author: kardon
 */

#include "pipelinebuilder.h"

namespace provallo
{

  // vectorizers implementation:

  tfidf::tfidf(const tfidf &other) : _tf(other._tf), _idf(other._idf)
  {
  }
  tfidf::tfidf(tfidf &&other) : _tf(std::move(other._tf)), _idf(std::move(other._idf))
  {
  }
  tfidf &
  tfidf::operator=(const tfidf &other)
  {
    if (this != &other)
    {

      _tf = other._tf;
      _idf = other._idf;
    }
    return *this;
  }
  tfidf &
  tfidf::operator=(tfidf &&other)
  {
    if (this != &other)
    {
      _tf = std::move(other._tf);
      _idf = std::move(other._idf);
    }
    return *this;
  }
  tfidf::tfidf()
  {
  }
  tfidf::~tfidf()
  {
  }

  // tfidf implementation:
  void tfidf::process_documents()
  {
    this->_vocabulary.clear();
    this->_tf.clear();
    this->_idf.clear();
    this->_tfidf.clear();
    // calculate vocabulary
    for (auto &doc : _documents)
    {
      std::vector<std::string> words;
      tokenize(doc, words, " ,;.:-_()[]{}!?\"\'\n\t");

      for (auto &word : words)
      {
        if (std::find(_vocabulary.begin(), _vocabulary.end(), word) == _vocabulary.end())
        {
          _vocabulary.push_back(word);
        }
      }
    }

    // calculate term frequency
    for (auto &word : _vocabulary)
    {
      std::map<std::string, real_t> tf_doc;
      for (auto &doc : _documents)
      {
        std::vector<std::string> words;
        tokenize(doc, words, " ,;.:-_()[]{}!?\"\'\n\t");
        for (auto &w : words)
        {
          if (w == word)
          {
            if (tf_doc.find(word) == tf_doc.end())
            {
              tf_doc[word] = 1.0;
            }
            else
            {
              tf_doc[word] += 1.0;
            }
          }
        }
      }
      // normalize tf
      real_t sum = 0.0;
      for (auto &doc_pair : tf_doc)
      {
        sum += doc_pair.second;
      }
      for (auto &doc_pair : tf_doc)
      {
        doc_pair.second /= sum;
      }

      for (auto &doc_pair : tf_doc)
        _tf.push_back(doc_pair.second);
    }
    // calculate idf

    for (auto &word : _vocabulary)
    {
      real_t count = 0.0;
      for (auto &doc : _documents)
      {
        std::vector<std::string> words;
        tokenize(doc, words, " ,;.:-_()[]{}!?\"\'\n\t");
        for (auto &w : words)
        {
          if (w == word)
          {
            count++;
            break;
          }
        }
      }
      _idf.push_back(provallo::log<2>(real_t(_documents.size()) / real_t(count)));
    }
    // calculate tfidf
    _tfidf.resize(_vocabulary.size());
    for (uint32_t i = 0; i < _tf.size() && i < _idf.size(); ++i)
    {
      _tfidf[i].push_back(_tf[i] * _idf[i]);
    }
    // done
    return;
  }
  // tfidf::get_tfidf
  std::vector<std::vector<real_t>>
  tfidf::get_tfidf() const
  {
    return _tfidf;
  }

  // tfidf::get_vocabulary
  const std::vector<std::string> &
  tfidf::get_vocabulary() const
  {
    return _vocabulary;
  }
  // tfidf::get_tf
  std::vector<real_t>
  tfidf::get_tf() const
  {
    return _tf;
  }
  // tfidf::get_idf
  std::vector<real_t>
  tfidf::get_idf() const
  {
    return _idf;
  }
  // tfidf::get_documents
  const std::vector<std::string> &
  tfidf::get_documents() const
  {
    return _documents;
  }
  // tfidf::set_documents
  void
  tfidf::set_documents(const std::vector<std::string> &documents)
  {
    if (_documents.size() != documents.size())
      _documents.clear();

    _documents = documents;
    this->process_documents();
  }

  // tfidf::add_document
  void
  tfidf::add_document(const std::string &doc)
  {
    std::vector<std::string> words;
    tokenize(doc, words, " ,;.:-_()[]{}!?\"\'\n\t");
    for (auto &word : words)
    {

      // push unique words to vocabulary

      if (std::find(_vocabulary.begin(), _vocabulary.end(), word) == _vocabulary.end())
      {
        _vocabulary.push_back(word);
      }
    }
  }

  std::vector<std::vector<real_t> > tfidf::transform(const std::vector<std::string>& docs)
  {
    std::vector<std::vector<real_t> > result;
    for (auto &doc : docs)
    {
      result.push_back(this->transform(doc));
    }
    return result;
  }


  std::vector<real_t> tfidf::transform(const std::string &doc)
  {
    std::vector<real_t> result;
    std::vector<std::string> words;
    tokenize(doc, words, " ,;.:-_()[]{}!?\"\'\n\t");
    for (auto &word : _vocabulary)
    {
      real_t count = 0.0;
      for (auto &w : words)
      {
        if (w == word)
        {
          count++;
        }
      }
      result.push_back(count);
    }
    for (uint32_t i = 0; i < result.size() && i < _idf.size(); ++i)
    {
      result[i] *= _idf[i];
    }
    return result;
  }
  
  tfidf_vectorizer::tfidf_vectorizer(const tfidf_vectorizer &other) : vectorizer(other), _tfidf(other._tfidf)
  {
  }
  tfidf_vectorizer &
  tfidf_vectorizer::operator=(const tfidf_vectorizer &other)
  {
    if (this != &other)
    {
      _tfidf = other._tfidf;
    }

    return *this;
  }
  tfidf_vectorizer &
  tfidf_vectorizer::operator=(tfidf_vectorizer &&other)
  {
    if (this != &other)
    {
      _tfidf = std::move(other._tfidf);
    }
    return *this;
  }
  tfidf_vectorizer::tfidf_vectorizer(tfidf_vectorizer &&other) : vectorizer(std::move(other)), _tfidf(std::move(other._tfidf))
  {
  }
  tfidf_vectorizer::tfidf_vectorizer() : vectorizer(TFIDF)
  {
  }
  tfidf_vectorizer::~tfidf_vectorizer()
  {
  }

  // tfidf_vectorizer::fit
  std::vector<real_t>
  tfidf_vectorizer::fit(const std::vector<std::string> &corpus)
  {

    for (auto &doc : corpus)
    {
      _tfidf.add_document(doc);
    }

    _tfidf.process_documents();

    return _tfidf.get_tf();
  }
  // tfidf_vectorizer::fit_transform



  std::vector<real_t>
  tfidf_vectorizer::fit_transform(const std::vector<std::string> &corpus)
  {
    std::vector<real_t> ret = fit(corpus);
    // transform :
    provallo::matrix<real_t> ret_matrix(ret.size(), 1);
    for (uint32_t i = 0; i < ret.size(); ++i)
    {
      ret_matrix(i, 0) = ret[i];
    }

    return transform(ret_matrix);
  }

  //
  // predict

  std::vector<real_t>
  tfidf_vectorizer::predict(const std::string &doc)
  {
    return transform(doc);
  }
  std::vector<real_t>
  tfidf_vectorizer::predict(const std::vector<std::string> &corpus)
  {
    std::vector<real_t> predict;
    std::vector<real_t> result;
    for (auto &doc : corpus)
    {
      predict = transform(doc);
      for (auto &res : predict)
      {
        result.push_back(res);
      }
    }
    return result;
  }

  // tfidf_vectorizer::tfidf_vectorizer

  tfidf_vectorizer::tfidf_vectorizer(const std::vector<std::string> &corpus) : vectorizer(TFIDF)
  {
    fit(corpus);
  }

  std::vector<std::string> tfidf::inverse_transform(const std::vector<std::vector<real_t>> &corpus)
  {
    std::vector<std::string> result;
    for (auto &doc : corpus)
    {
      result.push_back(inverse_transform(doc));
    }
    return result;
  }
  std::string tfidf::inverse_transform(const std::vector<real_t> &doc)
  {
    for (auto prob : doc)
    {
      auto it = std::find(_tf.begin(), _tf.end(), prob);
      if (it != _tf.end())
      {

        size_t index = it - _tf.begin();
        if (index < _vocabulary.size())
          return _vocabulary[(it - doc.begin())];
      }
    }
    return "";
  }

  // tfidf_vectorizer::transform
  std::vector<real_t>
  tfidf_vectorizer::transform(const std::string &doc)
  {

    return _tfidf.transform(doc);
  }

  standard_scaler_vectorizer &
  standard_scaler_vectorizer::operator=(const standard_scaler_vectorizer &other)
  {
    if (this != &other)
    {
      this->_data = other._data;
      this->_fitted_data = other._fitted_data;
      this->_mean = other._mean;
      this->_predicted_data = other._predicted_data;
      this->_type = other._type;
      this->_variance = other._variance;
      // this->_scaler = other._scaler;
    }

    return *this;
  }
  standard_scaler_vectorizer &
  standard_scaler_vectorizer::operator=(standard_scaler_vectorizer &&other)
  {
    if (this != &other)
    {

      if (this != &other)
      {
        this->_data = std::move(other._data);

        this->_fitted_data = std::move(other._fitted_data);
        this->_mean = std::move(other._mean);
        this->_predicted_data = std::move(other._predicted_data);
        this->_type = std::move(other._type);

        this->_variance = std::move(other._variance);
      }
    }
    return *this;
  }
  standard_scaler_vectorizer::standard_scaler_vectorizer(standard_scaler_vectorizer &&other) : vectorizer(std::move(other))
  {
  }
  standard_scaler_vectorizer::standard_scaler_vectorizer() : vectorizer(STANDARD_SCALER)
  {
  }
  standard_scaler_vectorizer::~standard_scaler_vectorizer()
  {
  }

  // standard_scaler_vectorizer::fit
  std::vector<real_t>
  standard_scaler_vectorizer::fit(const matrix<real_t> &corpus)
  {
    const matrix<real_t> &data = corpus;
    _mean.resize(data.cols());
    _variance.resize(data.cols());

    for (size_t i = 0; i < data.cols(); ++i)
    {
      _mean[i] = data.col_mean(i);

      _variance[i] = data.col_variance(i);
    }
    _fitted_data.resize(data.rows(), data.cols());

    for (size_t i = 0; i < data.rows(); ++i)
    {
      for (size_t j = 0; j < data.cols(); ++j)
      {
        _fitted_data(i, j) = (data(i, j) - _mean.at(j)) / _variance.at(j);
      }
    }

    return _mean;
  }
  // standard_scaler_vectorizer::fit_transform
  /*
  std::vector<real_t>
  standard_scaler_vectorizer::fit_transform(const matrix<real_t> &corpus)
  {
    std::vector<real_t> ret = fit(corpus);
    // transform :
    provallo::matrix<real_t> ret_matrix(ret.size(), 1);
    for (uint32_t i = 0; i < ret.size(); ++i)
    {
      ret_matrix(i, 0) = ret[i];
    }

    return transform(ret_matrix);
  }&*/

   /*
  std::vector<real_t>
  pca_vectorizer::fit_transform(const matrix<real_t> &corpus)
  {
    std::vector<real_t> ret = fit(corpus);
    // transform :
    provallo::matrix<real_t> ret_matrix(ret.size(), 1);
    for (uint32_t i = 0; i < ret.size(); ++i)
    {
      ret_matrix(i, 0) = ret[i];
    }

    return transform(ret_matrix);
  }
  */
  pca_vectorizer &
  pca_vectorizer::operator=(const pca_vectorizer &other)
  {
    if (this != &other)
    {
      this->_pca = other._pca;
    }

    return *this;
  }
  pca_vectorizer &
  pca_vectorizer::operator=(pca_vectorizer &&other)
  {
    if (this != &other)
    {
      _pca = std::move(other._pca);
    }
    return *this;
  }
  pca_vectorizer::pca_vectorizer(pca_vectorizer &&other) : vectorizer(std::move(other)), _pca(std::move(other._pca))
  {
    //  
    // _pca = std::move(other._pca);
    //


  }
  pca_vectorizer::pca_vectorizer() : vectorizer(PCA)
  {
    //  
    // _pca = std::move(other._pca);
    // _pca = std::move(other._pca);
  }
  pca_vectorizer::~pca_vectorizer()
  {
    //  

  }
  // pca_vectorizer::fit
  std::vector<real_t>
  pca_vectorizer::fit(const provallo::matrix<real_t> &data)
  {
    return _pca.fit(data);
  }
  // standard_scaler_vectorizer::fit
  std::vector<real_t>
  standard_scaler_vectorizer::fit(const std::vector<std::string> &corpus)
  {
    // todo: IMPLEMENT FIT
    std::vector<real_t> ret;
    for (auto &doc : corpus)
    {
      ret.push_back(doc.size());
    }
    return ret;
  }
  // standard_scaler_vectorizer::fit_transform
  std::vector<real_t>
  standard_scaler_vectorizer::fit_transform(const std::vector<std::string> &corpus)
  {
    return transform(matrix<real_t>(fit(corpus)));
  }
  // standard_scaler_vectorizer::predict
  std::vector<real_t>
  standard_scaler_vectorizer::predict(const std::string &doc)
  {
    return predict(std::vector<std::string>{doc});
  }

  // pca_vectorizer::fit
  std::vector<real_t> pca_vectorizer::fit(const std::vector<std::string> &corpus)
  {
    std::vector<real_t> ret;
    for (auto &doc : corpus)
    {
      ret.push_back(doc.size());
    }
    return ret;
  }

  pipeline_builder &
  pipeline_builder::operator=(const pipeline_builder &other)
  {
    if (this != &other)
    {

      // copy everything :
      this->_estimators = other._estimators;
      this->_transform_estimators = other._transform_estimators;
      this->_encoders = other._encoders;
      this->_decoders = other._decoders;
      this->_vectorizers = other._vectorizers;
      this->_transform_vectorizers = other._transform_vectorizers;
      this->_classifiers = other._classifiers;
      this->_iso_forests = other._iso_forests;
    }
    return *this;
  }

  pipeline_builder &
  pipeline_builder::operator=(pipeline_builder &&other)
  { // move assignment

    if (this != &other)
    {

      this->_estimators = std::move(other._estimators);
      this->_transform_estimators = std::move(other._transform_estimators);
      this->_encoders = std::move(other._encoders);
      this->_decoders = std::move(other._decoders);
      this->_vectorizers = std::move(other._vectorizers);
      this->_transform_vectorizers = std::move(other._transform_vectorizers);
      this->_classifiers = std::move(other._classifiers);
      this->_iso_forests = std::move(other._iso_forests);
    }
    return *this;
  }

  pipeline_builder::pipeline_builder(const pipeline_builder &other) : _estimators(other._estimators), _transform_estimators(other._transform_estimators), _encoders(other._encoders), _decoders(other._decoders), _vectorizers(other._vectorizers), _transform_vectorizers(other._transform_vectorizers), _classifiers(other._classifiers), _iso_forests(other._iso_forests)
  {
    // copy everything :

  }

  pipeline_builder::~pipeline_builder()
  {
    // clear everything :
    _estimators.clear();
    _transform_estimators.clear();
    _encoders.clear();
    _decoders.clear();
    _vectorizers.clear();
    _transform_vectorizers.clear();
    _classifiers.clear();
    _iso_forests.clear();
  }

  pipeline_builder::pipeline_builder(): _estimators(), _transform_estimators(), _encoders(), _decoders(), _vectorizers(), _transform_vectorizers(), _classifiers(), _iso_forests()
  {
    // clear everything :
    _estimators.clear();
    _transform_estimators.clear();
    _encoders.clear();
    _decoders.clear();
    _vectorizers.clear();
    _transform_vectorizers.clear();
    _classifiers.clear();
    _iso_forests.clear();
  }

  /*  std::vector<real_t> _mean;
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
    std::vector<real_t> _pca_n_components;
    std::vector<real_t> _pca_n_features;
    std::vector<real_t> _pca_n_samples;
    std::vector<real_t> _pca_n_components_;
    std::vector<real_t> _pca_n_features_;
    std::vector<real_t> _pca_n_samples_;
  */

  // principal_component_analysis implementation:

  principal_component_analysis::principal_component_analysis()
  {
    // default constructor
  }

  principal_component_analysis::~principal_component_analysis()
  {
    // default destructor
  }
  principal_component_analysis::principal_component_analysis(  principal_component_analysis&& other):
  _mean(std::move(other._mean)),
  _variance(std::move(other._variance)),
  _standard_deviation(std::move(other._standard_deviation)),
  _standardized_data(std::move(other._standardized_data)),
  _covariance_matrix(std::move(other._covariance_matrix)),
  _eigen_values(std::move(other._eigen_values)),
  _eigen_vectors(std::move(other._eigen_vectors)),
  _pca_data(std::move(other._pca_data)),
  _pca_components(std::move(other._pca_components)),
  _pca_explained_variance(std::move(other._pca_explained_variance)),
  _pca_explained_variance_ratio(std::move(other._pca_explained_variance_ratio)),
  _pca_singular_values(std::move(other._pca_singular_values)),
  _pca_noise_variance(std::move(other._pca_noise_variance)),
  _pca_mean(std::move(other._pca_mean)),
  _pca_n_components(std::move(other._pca_n_components)),
  _pca_n_features(std::move(other._pca_n_features)),
  _pca_n_samples(std::move(other._pca_n_samples)),
  _pca_n_components_(std::move(other._pca_n_components_)),
  _pca_n_features_(std::move(other._pca_n_features_)),
  _pca_n_samples_(std::move(other._pca_n_samples_))
  {
  }



  principal_component_analysis &
  principal_component_analysis::operator=(const principal_component_analysis &other)
  {

    if (this != &other)
    {

      // copy everything :
      this->_mean = other._mean;
      this->_variance = other._variance;
      this->_standard_deviation = other._standard_deviation;
      this->_standardized_data = other._standardized_data;
      this->_covariance_matrix = other._covariance_matrix;
      this->_eigen_values = other._eigen_values;
      this->_eigen_vectors = other._eigen_vectors;
      this->_pca_data = other._pca_data;
      this->_pca_components = other._pca_components;
      this->_pca_explained_variance = other._pca_explained_variance;
      this->_pca_explained_variance_ratio = other._pca_explained_variance_ratio;
      this->_pca_singular_values = other._pca_singular_values;
      this->_pca_noise_variance = other._pca_noise_variance;
      this->_pca_mean = other._pca_mean;
      this->_pca_n_components = other._pca_n_components;
      this->_pca_n_features = other._pca_n_features;
      this->_pca_n_samples = other._pca_n_samples;
      this->_pca_n_components_ = other._pca_n_components_;
      this->_pca_n_features_ = other._pca_n_features_;
      this->_pca_n_samples_ = other._pca_n_samples_;
    }
    return *this;
  }

  principal_component_analysis &
  principal_component_analysis::operator=(principal_component_analysis &&other)
  {
    //
    if (this != &other)
    {

      // copy everything :
      this->_mean = std::move(other._mean);
      this->_variance = std::move(other._variance);
      this->_standard_deviation = std::move(other._standard_deviation);
      this->_standardized_data = std::move(other._standardized_data);
      this->_covariance_matrix = std::move(other._covariance_matrix);
      this->_eigen_values = std::move(other._eigen_values);
      this->_eigen_vectors = std::move(other._eigen_vectors);
      this->_pca_data = std::move(other._pca_data);
      this->_pca_components = std::move(other._pca_components);
      this->_pca_explained_variance = std::move(other._pca_explained_variance);
      this->_pca_explained_variance_ratio = std::move(other._pca_explained_variance_ratio);
      this->_pca_singular_values = std::move(other._pca_singular_values);
      this->_pca_noise_variance = std::move(other._pca_noise_variance);
      this->_pca_mean = std::move(other._pca_mean);
      this->_pca_n_components = std::move(other._pca_n_components);
      this->_pca_n_features = std::move(other._pca_n_features);
      this->_pca_n_samples = std::move(other._pca_n_samples);
      this->_pca_n_components_ = std::move(other._pca_n_components_);
      this->_pca_n_features_ = std::move(other._pca_n_features_);
      this->_pca_n_samples_ = std::move(other._pca_n_samples_);
    }
    return *this;
  }
  // copy constructor

  // move constructor

  // copy assignment operator

  // move assignment operator

  // principal_component_analysis implementation:
  // virtual  std::vector<real_t> fit( const std::vector<std::string>&documents );
  std::vector<real_t> principal_component_analysis::fit(const std::vector<std::string> &documents)
  {
    std::vector<real_t> result;
    for ( auto &document : documents )
    {
      result.push_back(.1*document.size());
    }
    return result;
  }  


  //QR decomposition
  void principal_component_analysis::QRDecomposition(const matrix<real_t>& mtx,matrix<real_t>& Q,  matrix<real_t>& R)
  {
    //extract Q and R from the QR decomposition
    //Q is an orthogonal matrix
    //R is an upper triangular matrix

    //get the number of rows and columns
    size_t rows = mtx.rows();
    size_t cols = mtx.cols();

    //initialize Q and R
    Q = matrix<real_t>(rows,cols);
    R = matrix<real_t>(cols,cols);

    //initialize the first column of Q
    for(size_t i = 0; i < rows; ++i)
    {
      Q(i,0) = mtx(i,0);
    }

    //calculate the norm of the first column of Q
    real_t norm = 0.0;
    for(size_t i = 0; i < rows; ++i)
    {
      norm += Q(i,0) * Q(i,0);
    }

    //set the norm of the first column of Q to 1
    norm = std::sqrt(norm);
    for(size_t i = 0; i < rows; ++i)
    {
      Q(i,0) /= norm;
    } 

    //calculate the elements of R
    for(size_t i = 0; i < cols; ++i)
    {
      for(size_t j = i; j < cols; ++j)
      {
        R(i,j) = 0.0;
        for(size_t k = 0; k < rows; ++k)
        {
          R(i,j) += mtx(k,i) * Q(k,j);
        }
      }
    } 

    //calculate the elements of Q
    for(size_t i = 1; i < cols; ++i)
    {
      for(size_t j = 0; j < rows; ++j)
      {
        Q(j,i) = 0.0;
        for(size_t k = 0; k < rows; ++k)
        {
          Q(j,i) += mtx(k,i) * R(k,j);
        }
      }
    } 
    //get the transpose of Q
    Q = Q.transpose();

    //get the transpose of R
    R = R.transpose();

    //get the absolute value of the elements of R
    for(size_t i = 0; i < cols; ++i)
    {
      for(size_t j = 0; j < cols; ++j)
      {
        R(i,j) = std::abs(R(i,j));
      }
    } 
    //get the sign of the elements of R
    for(size_t i = 0; i < cols; ++i)
    {
      if(R(i,i) < 0.0)
      {
        for(size_t j = 0; j < rows; ++j)
        {
          Q(j,i) = -Q(j,i);
        }
        for(size_t j = 0; j < cols; ++j)
        {
          R(i,j) = -R(i,j);
        }
      }
    } 

  }

  //implementation of the PCA algorithm

  //undefined references : principal_component_analysis::predict, 
  //principal_component_analysis::transform(provallo::matrix<real_t> const&)
  std::vector<real_t>
  principal_component_analysis::transform(provallo::matrix<real_t> const& data_)
  {
    //get the number of rows and columns
    size_t rows = data_.rows();
    size_t cols = data_.cols();

    //initialize the transformed data
    matrix<real_t> transformed_data(rows,cols);

    //transform the data
    for(size_t i = 0; i < rows; ++i)
    {
      //initialize the transformed data
      for(size_t j = 0; j < cols; ++j)
      { 
 
        transformed_data(i,j) = 0.0;
        for(size_t k = 0; k < cols; ++k)
        {
          //  
          transformed_data(i,j) += data_(i,k) * this->_pca_components(k,j); 
        }
      }
    } 

    //return the transformed data
    return std::vector<real_t>(transformed_data.begin(), transformed_data.end());
  }

  //fit   
   std::vector<real_t> principal_component_analysis::fit( const provallo::matrix<real_t>& data_ )
   {
    //
    //get the number of rows and columns
    size_t rows = data_.rows();
    size_t cols = data_.cols();

    //initialize the mean vector
    std::vector<real_t> mean(cols,0.0);

    //calculate the mean vector
    for(size_t i = 0; i < cols; ++i)
    {
      for(size_t j = 0; j < rows; ++j)
      {
        mean[i] += data_(j,i);
      }
      mean[i] /= rows;
    } 

    //initialize the centered data
    matrix<real_t> centered_data(rows,cols);

    //center the data
    for(size_t i = 0; i < rows; ++i)
    {
      for(size_t j = 0; j < cols; ++j)
      {
        centered_data(i,j) = data_(i,j) - mean[j];
      }
    }

    //initialize the covariance matrix
    matrix<real_t> covariance_matrix(cols,cols);
    
    //calculate the covariance matrix
    for(size_t i = 0; i < cols; ++i)
    {
      for(size_t j = i; j < cols; ++j)
      {
        covariance_matrix(i,j) = 0.0;
        for(size_t k = 0; k < rows; ++k)
        {
          covariance_matrix(i,j) += centered_data(k,i) * centered_data(k,j);
        }
        covariance_matrix(i,j) /= rows;
        covariance_matrix(j,i) = covariance_matrix(i,j);
      }
    }   

    //initialize the eigenvalues and eigenvectors
    std::vector<real_t> eigenvalues(cols,0.0);
    matrix<real_t> eigenvectors(cols,cols);
 

    /// nrot : number of rotations
    //
    //    template <typename T>
    //void jacobi(const matrix<T> &a1, std::vector<T> &d, matrix<T> &v, size_t &nrot)
    //initialize the sorted eigenvalues and eigenvectors
    size_t nrot = eigenvalues.size()*eigenvalues.size(); //cols*cols
    jacobi(covariance_matrix, eigenvalues, eigenvectors, nrot);

    std::vector<real_t> sorted_eigenvalues(eigenvalues);
    matrix<real_t> sorted_eigenvectors(eigenvectors);

    //sort the eigenvalues and eigenvectors
    std::sort(sorted_eigenvalues.begin(), sorted_eigenvalues.end(), std::greater<real_t>());
    
    for(size_t i = 0; i < cols; ++i)
    {
      for(size_t j = 0; j < cols; ++j)
      {
        if(eigenvalues[j] == sorted_eigenvalues[i])
        {
          for(size_t k = 0; k < cols; ++k)
          {
            sorted_eigenvectors(k,i) = eigenvectors(k,j);
          }
          eigenvalues[j] = 0.0;
          break;
        }
      }
    }   
    //initialize the pca components
    matrix<real_t> pca_components(cols,cols);
    //get the pca components
    for(size_t i = 0; i < cols; ++i)
    {
      for(size_t j = 0; j < cols; ++j)
      {
        pca_components(i,j) = sorted_eigenvectors(i,j);
      }
    }

    //set the pca components
    this->_pca_components = pca_components;

    //return the pca components
    return std::vector<real_t>(pca_components.begin(), pca_components.end());



 

   }
   std::vector<real_t>
  principal_component_analysis::predict(const provallo::matrix<real_t>& data_ ) 
  {
    //get the number of rows and columns
    size_t rows = data_.rows();
    size_t cols = data_.cols();

    //initialize the transformed data
    matrix<real_t> transformed_data(rows,cols);

    //transform the data
    for(size_t i = 0; i < rows; ++i)
    {
      for(size_t j = 0; j < cols; ++j)
      {
        transformed_data(i,j) = 0.0;
        for(size_t k = 0; k < cols; ++k)
        {
          transformed_data(i,j) += data_(i,k) * this->_pca_components(k,j);
        }
      }
    } 

    //return the transformed data
    return std::vector<real_t>(transformed_data.begin(), transformed_data.end());
   }
 
  
  std::vector<real_t> principal_component_analysis::predict(const std::vector<std::string>&documents )
  {
    UNDEF_REFERENCE(documents);
    UNDEF_REFERENCE2(documents);
    return std::vector<real_t>(); 
  }

  std::vector<real_t> principal_component_analysis::transform(const std::vector<std::string>&documents )
  {
    
    UNDEF_REFERENCE(documents);
    UNDEF_REFERENCE2(documents);
    return std::vector<real_t>(); 
  }

  std::vector<real_t> principal_component_analysis::fit_transform(const std::vector<std::string>&documents )  
  {
    UNDEF_REFERENCE(documents);
    UNDEF_REFERENCE2(documents);
    return std::vector<real_t>();
  }

  //pca_vectorizer :: predict
  // pca_vectorizer :: fit_transform
  //pca_vectorizer :: transform

  std::vector<real_t> pca_vectorizer::predict(provallo::matrix<real_t> const& data_)
  {
    return _pca.predict(data_);
  }
 
  std::vector<real_t> pca_vectorizer::transform(provallo::matrix<real_t> const& data_)  {
    return _pca.transform(data_);
  }

  std::vector<real_t> pca_vectorizer::predict(std::vector<std::string> const& documents)
  {
    return _pca.predict(documents);
  } 



  std::vector<real_t> pca_vectorizer::fit_transform(std::vector<std::string> const& documents)
  {
    return _pca.fit_transform(documents);
  }
  vectorizer_type pca_vectorizer::get_type() const
  {
    return vectorizer_type::PCA;
  }
  std::vector<real_t> pca_vectorizer::transform(std::vector<std::string> const& documents)
  {
    return _pca.transform(documents);
  } 
  std::vector<real_t>  standard_scaler_vectorizer::predict(provallo::matrix<real_t> const& data_)
  {
      //use _fitted_data,mean and std to transform the data 
      //get the number of rows and columns
      size_t rows = data_.rows();
      size_t cols = data_.cols();
      matrix<real_t> _std =data_.std();
      //initialize the transformed data
      matrix<real_t> transformed_data(rows,cols);



      //transform the data
      for(size_t i = 0; i < rows; ++i)
      {
        for(size_t j = 0; j < cols; ++j)
        {
          transformed_data(i,j) = (data_(i,j) - _mean[j]) / _std(i,j )  ;
        }
      } 
      //return the transformed data
      return std::vector<real_t>(transformed_data.begin(), transformed_data.end());
       
   }



  //tfidf_vectorizer
  std::vector<real_t> tfidf_vectorizer::predict(provallo::matrix<real_t> const& data_)
  {
    UNDEF_REFERENCE(data_);
    return std::vector<real_t>();
  }
  
    std::vector<real_t> standard_scaler_vectorizer::transform(const provallo::matrix<real_t>& data_matrix)  { 
    //use _fitted_data,mean and std to transform the data
    //get the number of rows and columns
    size_t rows = data_matrix.rows();
    size_t cols = data_matrix.cols();
    matrix<real_t> _std =_fitted_data.std();
    //initialize the transformed data
    matrix<real_t> transformed_data(rows,cols);
    //transform the data
    for(size_t i = 0; i < rows; ++i)
    {
      for(size_t j = 0; j < cols; ++j)
      {
        transformed_data(i,j) = (data_matrix(i,j) - _mean[j]) / _std(i,j )  ;
      }
    }
    //return the transformed data
    return std::vector<real_t>(transformed_data.begin(), transformed_data.end());
  }

  std::vector<real_t> standard_scaler_vectorizer::predict(const std::vector<std::string>& documents)
  {
    //use _fitted_data,mean and std to transform the data
    //get the number of rows and columns
    size_t rows = documents.size();

    //initialize the transformed data
    matrix<real_t> transformed_data(rows,1);
    //transform the data
    matrix<real_t> _std = _fitted_data.std();
    for(size_t i = 0; i < rows; ++i)
    {
      transformed_data(i,0) = (std::stod(documents[i]) - _mean[0]) /_std (0,0 );

    }

   //return the transformed data
    return std::vector<real_t>(transformed_data.begin(), transformed_data.end());
  }

  std::vector<real_t> standard_scaler_vectorizer::transform(const  std::vector<std::string>& documents)
  {
    //use _fitted_data,mean and std to transform the data
    //get the number of rows and columns
    size_t rows = documents.size();

    //initialize the transformed data
    matrix<real_t> transformed_data(rows,1);
    //transform the data
    matrix<real_t> _std = _fitted_data.std();
    for(size_t i = 0; i < rows; ++i)
    {
      transformed_data(i,0) = (std::stod(documents[i]) - _mean[0]) /_std (0,0 );

    }

   //return the transformed data
    return std::vector<real_t>(transformed_data.begin(), transformed_data.end());       


  }
template<>
std::vector<real_t>
provallo::vectorizer<std::string, real_t>::predict ( const std::vector<std::string> & documents ) 
{
  //use _fitted_data,mean and std to transform the data
  //get the number of rows and columns
  size_t rows = documents.size();
  
  //initialize the transformed data
  matrix<real_t> transformed_data(matrix<real_t>::Zero(rows,1 ));
  //transform the data
  matrix<real_t> _std = _fitted_data.std();
   real_t _mean = _fitted_data.mean(); 

  for(size_t i = 0; i < rows; ++i)
  {
    transformed_data(i,0) = (std::stod(documents[i]) - _mean /_std (0,0 ) );
    
  }
  //return the transformed data
  return std::vector<real_t>(transformed_data.begin(), transformed_data.end());
}
 
/*/usr/bin/ld: CMakeFiles/provallo_engine.dir/decision_engine/pipelinebuilder.cpp.o:(.data.rel.ro._ZTVN8provallo16tfidf_vectorizerE[_ZTVN8provallo16tfidf_vectorizerE]+0x28): undefined reference to `provallo::tfidf_vectorizer::fit(provallo::matrix<double> const&)'
/usr/bin/ld: CMakeFiles/provallo_engine.dir/decision_engine/pipelinebuilder.cpp.o:(.data.rel.ro._ZTVN8provallo16tfidf_vectorizerE[_ZTVN8provallo16tfidf_vectorizerE]+0x30): undefined reference to `provallo::tfidf_vectorizer::transform(provallo::matrix<double> const&)'
/usr/bin/ld: CMakeFiles/provallo_engine.dir/decision_engine/pipelinebuilder.cpp.o:(.data.rel.ro._ZTVN8provallo16tfidf_vectorizerE[_ZTVN8provallo16tfidf_vectorizerE]+0x50): undefined reference to `provallo::tfidf_vectorizer::transform(std::vector<std::string, std::allocator<std::string> > const&)'
/usr/bin/ld: CMakeFiles/provallo_engine.dir/decision_engine/pipelinebuilder.cpp.o:(.data.rel.ro._ZTVN8provallo10vectorizerISsdEE[_ZTVN8provallo10vectorizerISsdEE]+0x28): undefined reference to `provallo::vectorizer<std::string, double>::fit(provallo::matrix<double> const&)'
/usr/bin/ld: CMakeFiles/provallo_engine.dir/decision_engine/pipelinebuilder.cpp.o:(.data.rel.ro._ZTVN8provallo10vectorizerISsdEE[_ZTVN8provallo10vectorizerISsdEE]+0x30): undefined reference to `provallo::vectorizer<std::string, double>::transform(provallo::matrix<double> const&)'
/usr/bin/ld: CMakeFiles/provallo_engine.dir/decision_engine/pipelinebuilder.cpp.o:(.data.rel.ro._ZTVN8provallo10vectorizerISsdEE[_ZTVN8provallo10vectorizerISsdEE]+0x40): undefined reference to `provallo::vectorizer<std::string, double>::fit(std::vector<std::string, std::allocator<std::string> > const&)'
/usr/bin/ld: CMakeFiles/provallo_engine.dir/decision_engine/pipelinebuilder.cpp.o:(.data.rel.ro._ZTVN8provallo10vectorizerISsdEE[_ZTVN8provallo10vectorizerISsdEE]+0x50): undefined reference to `provallo::vectorizer<std::string, double>::transform(std::vector<std::string, std::allocator<std::string> > const&)'
/usr/bin/ld: CMakeFiles/provallo_engine.dir/decision_engine/pipelinebuilder.cpp.o:(.data.rel.ro._ZTVN8provallo10vectorizerISsdEE[_ZTVN8provallo10vectorizerISsdEE]+0x58): undefined reference to `provallo::vectorizer<std::string, double>::fit_transform(std::vector<std::string, std::allocator<std::string> > const&)'
/usr/bin/ld: CMakeFiles/provallo_engine.dir/decision_engine/pipelinebuilder.cpp.o:(.data.rel.ro._ZTVN8provallo10vectorizerISsdEE[_ZTVN8provallo10vectorizerISsdEE]+0x60): undefined reference to `provallo::vectorizer<std::string, double>::predict(provallo::matrix<double> const&)'
/usr/bin/ld: CMakeFiles/provallo_engine.dir/decision_engine/pipelinebuilder.cpp.o:(.data.rel.ro._ZTVN8provallo19transform_estimatorIdEE[_ZTVN8provallo19transform_estimatorIdEE]+0x20): undefined reference to `provallo::transform_estimator<double>::get_type() const'
*/



  std::vector<real_t> tfidf_vectorizer::transform(const  std::vector<std::string>& documents)
  {
    //use _fitted_data,mean and std to transform the data
    //get the number of rows and columns
    size_t rows = documents.size();

    //initialize the transformed data
    std::vector<std::vector<real_t>> transformed_data (_tfidf.transform(documents));
    std::vector<real_t> transformed_ret (rows,0.0);
    //go over the data and fill the transformed data
    for(size_t i = 0; i < rows; ++i)
    {
      for(size_t j = 0; j < transformed_data[i].size() ; ++j)
          transformed_ret[i] += transformed_data[i][j] / real_t( transformed_data[i].size() );

    }

    //flatten the data
    
    //return the transformed data
    return transformed_ret;
    }

  std::vector<real_t> tfidf_vectorizer::transform(  const matrix<real_t> & trans)
  {
    //use _tfidf to transform the data

    size_t rows = trans.rows();
    size_t cols = trans.cols();


    //use _tfidf inverse_transform to transform the data 
    std::vector<std::vector<real_t>> ret_vec (rows, std::vector<real_t>(cols, 0.0) );
    //fill ret_vec with trans data
    for ( size_t i=0;i<rows ;i++)
      for(size_t j=0;j<cols;++j)
        ret_vec[i][j]  = trans(i,j);
    //transform the data

    std::vector<std::string> transformed_data = _tfidf.inverse_transform(ret_vec); // _tfidf.inverse_transform( )
    //go over the data and fill the ret_vec with the transformed data
    for(size_t i = 0; i < rows; ++i)
    {
      for(size_t j = 0; j < cols; ++j)
      { 
        if(i*cols + j < transformed_data.size())
          ret_vec[i][j] = std::stod(transformed_data[i*cols + j]);
        else

          ret_vec[i][j] = 0.0;  
            
      }

    }
    //return the transformed data
     std::vector<real_t> ret_val(ret_vec.size()*ret_vec[0].size(),0.0);
    for(size_t i = 0; i < rows; ++i)
    {
      for(size_t j = 0; j < cols; ++j)
      { 
        ret_val[i*cols + j] = ret_vec[i][j];
      }

    }
    return ret_val;

  }

  
  std::vector<real_t> tfidf_vectorizer::fit(const provallo::matrix<double> & foot)
  {
      //not implemented.

      return std::vector<real_t>(foot.begin(), foot.end());
  }



  /*/usr/bin/ld: CMakeFiles/provallo_engine.dir/decision_engine/pipelinebuilder.cpp.o:(.data.rel.ro._ZTVN8provallo10vectorizerISsdEE[_ZTVN8provallo10vectorizerISsdEE]+0x28): undefined reference to `provallo::vectorizer<std::string, double>::fit(provallo::matrix<double> const&)'
  /usr/bin/ld: CMakeFiles/provallo_engine.dir/decision_engine/pipelinebuilder.cpp.o:(.data.rel.ro._ZTVN8provallo10vectorizerISsdEE[_ZTVN8provallo10vectorizerISsdEE]+0x30): undefined reference to `provallo::vectorizer<std::string, double>::transform(provallo::matrix<double> const&)'
  /usr/bin/ld: CMakeFiles/provallo_engine.dir/decision_engine/pipelinebuilder.cpp.o:(.data.rel.ro._ZTVN8provallo10vectorizerISsdEE[_ZTVN8provallo10vectorizerISsdEE]+0x40): undefined reference to `provallo::vectorizer<std::string, double>::fit(std::vector<std::string, std::allocator<std::string> > const&)'
  /usr/bin/ld: CMakeFiles/provallo_engine.dir/decision_engine/pipelinebuilder.cpp.o:(.data.rel.ro._ZTVN8provallo10vectorizerISsdEE[_ZTVN8provallo10vectorizerISsdEE]+0x50): undefined reference to `provallo::vectorizer<std::string, double>::transform(std::vector<std::string, std::allocator<std::string> > const&)'
  /usr/bin/ld: CMakeFiles/provallo_engine.dir/decision_engine/pipelinebuilder.cpp.o:(.data.rel.ro._ZTVN8provallo10vectorizerISsdEE[_ZTVN8provallo10vectorizerISsdEE]+0x58): undefined reference to `provallo::vectorizer<std::string, double>::fit_transform(std::vector<std::string, std::allocator<std::string> > const&)'
  /usr/bin/ld: CMakeFiles/provallo_engine.dir/decision_engine/pipelinebuilder.cpp.o:(.data.rel.ro._ZTVN8provallo10vectorizerISsdEE[_ZTVN8provallo10vectorizerISsdEE]+0x60): undefined reference to `provallo::vectorizer<std::string, double>::predict(provallo::matrix<double> const&)'
  /usr/bin/ld: CMakeFiles/provallo_engine.dir/decision_engine/pipelinebuilder.cpp.o:(.data.rel.ro._ZTVN8provallo19transform_estimatorIdEE[_ZTVN8provallo19transform_estimatorIdEE]+0x20): undefined reference to `provallo::transform_estimator<double>::get_type() const'
  */


} /* namespace provallo */
