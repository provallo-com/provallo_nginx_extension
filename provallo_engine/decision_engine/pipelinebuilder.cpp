/*
 * pipelinebuilder.h
 *
 *  Created on: Jun 19, 2023
 *      Author: kardon
 */

#include "pipelinebuilder.h"
#include "matrix.h"

#include <iostream>

#include <algorithm>

namespace provallo
{

  // bag of words implementation:
  bag_of_words::bag_of_words() : _vocabulary(), _bow()
  {
  }

  // constructor from vocabulary
  bag_of_words::bag_of_words(const std::vector<std::string> &vocabulary) : _vocabulary(vocabulary), _bow(vocabulary.size(), 0.0)
  {

    // std::cout<<"vocabulary size: "<<_vocabulary.size()<<std::endl;
    // set bag of words values on the tokens
    for (size_t i = 0; i < _vocabulary.size(); ++i)
    {
      std::string token = _vocabulary[i];

      for (size_t j = i + 1; j < _vocabulary.size(); ++j)
      {
        if (_vocabulary[j] == token)
        {
          _bow[i] += 1.0;
          _bow[j] = 0.0;
        }
      }
      for (size_t j = 0; j < i; ++j)
      {
        if (_vocabulary[j] == token)
        {
          _bow[i] += 1.0;
          _bow[j] = 0.0;
        }
      }

      _bow[i] = _bow[i] / _vocabulary.size();
    }
    // update the bow matrix
    // matrix is the number of components x samples
    _bow_matrix = matrix<real_t>(_vocabulary.size(), 1);
    for (size_t i = 0; i < _vocabulary.size(); ++i)
    {
      _bow_matrix(i, 0) = _bow[i];
    }

    // std::cout<<"bow size: "<<_bow.size()<<std::endl;
    // std::cout<<"bow: "<<_bow<<std::endl;
  }
  // copy constructor
  bag_of_words::bag_of_words(const bag_of_words &other)
  {
    _vocabulary = other._vocabulary;
    _bow = other._bow;
    _bow_matrix = other._bow_matrix;
    _bow_transformed = other._bow_transformed;
    _bow_transformed_inverse = other._bow_transformed_inverse;
  }
  // move constructor
  bag_of_words::bag_of_words(bag_of_words &&other)
  {
    _vocabulary = std::move(other._vocabulary);
    _bow = std::move(other._bow);
    _bow_matrix = std::move(other._bow_matrix);
    _bow_transformed = std::move(other._bow_transformed);
    _bow_transformed_inverse = std::move(other._bow_transformed_inverse);

    other._vocabulary.clear();
    other._bow.clear();
    other._bow_matrix.clear();
    other._bow_transformed.clear();
    other._bow_transformed_inverse.clear();

    other._vocabulary.shrink_to_fit();
    other._bow.shrink_to_fit();
    other._bow_transformed.shrink_to_fit();
    other._bow_transformed_inverse.shrink_to_fit();

    // std::cout<<"bow size: "<<_bow.size()<<std::endl;
    
  }

  // assignment
  bag_of_words &
  bag_of_words::operator=(const bag_of_words &other)
  {
    if (this != &other)
    {
      _vocabulary = other._vocabulary;
      _bow = other._bow;
      _bow_matrix = other._bow_matrix;
      _bow_transformed = other._bow_transformed;
      _bow_transformed_inverse = other._bow_transformed_inverse;
    }
    return *this;
  }
  // move assignment
  bag_of_words &
  bag_of_words::operator=(bag_of_words &&other)
  {
    if (this != &other)
    {
      _vocabulary = std::move(other._vocabulary);
      _bow = std::move(other._bow);
      _bow_matrix = std::move(other._bow_matrix);
      _bow_transformed = std::move(other._bow_transformed);
      _bow_transformed_inverse = std::move(other._bow_transformed_inverse);
    }
    return *this;
  }
  //process the documents
  //  virtual void add_document(const std::string&);
  void bag_of_words::add_document(const std::string & doc)  
  {
    // tokenize the document
    std::vector<std::string> tokens;
    tokenize(doc, tokens);
    // update vocabulary and bow matrices
    for (auto token : tokens)
    {
      // find token in vocabulary and update its bow value
      // or add it to the vocabulary if it doesn't exists
      auto it = std::find(_vocabulary.begin(), _vocabulary.end(), token);
      if (it != _vocabulary.end())
      {
        // update bow value
        size_t index = std::distance(_vocabulary.begin(), it);
        _bow[index] += 1.0;
      }
      else
      {
        // add token to vocabulary
        _vocabulary.push_back(token);
        // update bow value
        _bow.push_back(1.0);
      }
    } // end for
    // update bow matrix
    if ( _bow_matrix.size1() == 0 && _bow_matrix.size2() == 0)
    {
      _bow_matrix = matrix<real_t>(_vocabulary.size(), 1);
    }
    else
    {
      _bow_matrix.resize(_vocabulary.size(), 1);
    }
    _bow_matrix = matrix<real_t>(_vocabulary.size(), 1);
    for (size_t i = 0; i < _vocabulary.size(); ++i)
    {
      _bow_matrix(i, 0) = _bow[i];
    }
    // std::cout<<"bow matrix: "<<_bow_matrix<<std::endl;
    return ;
  }
  //  virtual void process_document(const std::string&);
  void bag_of_words::process_document(const std::string & doc) 
  {
    //same as add_document
    add_document(doc);
    return ;
  }
  //  virtual void process_documents(const std::vector<std::string>&);
  // implementation of bag of words:
  std::vector<std::vector<real_t>> bag_of_words::fit(const std::vector<std::string> &documents)
  {
    
    std::vector<std::vector<real_t>> result(documents.size());

    if (_vocabulary.size() > 0)
    {

      // update vocabulary and bow matrices
      for (auto document : documents)
      {
        std::vector<std::string> tokens;
        tokenize(document, tokens);
        for (auto token : tokens)
        {
          // find token in vocabulary and update its bow value
          // or add it to the vocabulary if it doesn't exists
          auto it = std::find(_vocabulary.begin(), _vocabulary.end(), token);
          if (it != _vocabulary.end())
          {
            // update bow value
            size_t index = std::distance(_vocabulary.begin(), it);
            _bow[index] += 1.0;
          }
          else
          {
            // add token to vocabulary
            _vocabulary.push_back(token);
            // update bow value
            _bow.push_back(1.0);
          }
        }
        // update bow matrix
      }
    }
    else
    {
      // update vocabulary and bow matrices
      for (auto document : documents)
      {
        std::vector<std::string> tokens;
        tokenize(document, tokens);
        for (auto token : tokens)
        {
          // find token in vocabulary and update its bow value
          // or add it to the vocabulary if it doesn't exists
          auto it = std::find(_vocabulary.begin(), _vocabulary.end(), token);
          if (it != _vocabulary.end())
          {
            // update bow value
            size_t index = std::distance(_vocabulary.begin(), it);
            _bow[index] += 1.0;
          }
          else
          {
            // add token to vocabulary
            _vocabulary.push_back(token);
            // update bow value
            _bow.push_back(1.0);
          }
        }
        // update bow matrix
      }
    }
    // normalize bow
    for (size_t i = 0; i < _bow.size(); ++i)
    {
      _bow[i] = _bow[i] / _vocabulary.size();
    }

    // update the bow matrix - copy, add new row and delete old one

    matrix<real_t> new_bow = matrix<real_t>(_bow_matrix.size1() + 1, _bow_matrix.size2());

    for (size_t i = 0; i < _bow_matrix.size1(); ++i)
    {
      for (size_t j = 0; j < _bow_matrix.size2(); ++j)
      {
        new_bow(i, j) = _bow_matrix(i, j);
      }
    }
    for (size_t j = 0; j < _bow_matrix.size2(); ++j)
    {
      new_bow(_bow_matrix.size1(), j) = _bow[j];
    }
    _bow_matrix = new_bow;

    _bow_matrix = matrix<real_t>(_vocabulary.size(), 1);
    for (size_t i = 0; i < _vocabulary.size(); ++i)
    {
      _bow_matrix(i, 0) = _bow[i];
    }

    return result;
  }
  std::vector<std::vector<real_t>> bag_of_words::fit(const provallo::matrix<real_t> &mat)
  {
    std::vector<std::vector<real_t>> result(mat.rows(), std::vector<real_t>(mat.cols(), 0.0));

    // already a matrix, just return it formatted as a vector of vectors
    for (size_t i = 0; i < mat.rows(); i++)
    {
      for (size_t j = 0; j < mat.cols(); j++)
      {
        result[i][j] = mat(i, j);
      }
    }
    return result;
  }
  std::vector<real_t> bag_of_words::fit(const std::string& doc)
  {
    std::vector<real_t> result;
    std::vector<std::string> tokens;
    tokenize(doc, tokens);

    // for each token in the document
    // find it in the vocabulary
    // if it exists, add its bow value to the result
    // otherwise add 0.0
    for (auto token : tokens)
    {
      auto it = std::find(_vocabulary.begin(), _vocabulary.end(), token);
      if (it != _vocabulary.end())
      {
        // update bow value
        size_t index = std::distance(_vocabulary.begin(), it);
        result.push_back(_bow[index]);
      }
      else
      {
        // add token to vocabulary
        result.push_back(0.0);
      }
    }
    return result;  
  }
  std::vector<real_t> bag_of_words::transform(const std::string &doc)
  {
    std::vector<real_t> result;
    std::vector<std::string> tokens;
    tokenize(doc, tokens);

    // for each token in the document
    // find it in the vocabulary
    // if it exists, add its bow value to the result
    // otherwise add 0.0
    for (auto token : tokens)
    {
      auto it = std::find(_vocabulary.begin(), _vocabulary.end(), token);
      if (it != _vocabulary.end())
      {
        // update bow value
        size_t index = std::distance(_vocabulary.begin(), it);
        result.push_back(_bow[index]);
      }
      else
      {
        // add token to vocabulary
        result.push_back(0.0);
      }
    }
    return result;
  }

    // transform
  std::vector<std::vector<real_t>> bag_of_words::transform(const std::vector<std::string> &documents)
  {
    std::vector<std::vector<real_t>> result;
    for (auto document : documents)
    {
      result.push_back(transform(document));
    }

    return result;
  }
  std::vector<std::vector<real_t>> bag_of_words::transform(const provallo::matrix<real_t> & mat)
  {
    std::vector<std::vector<real_t>> result;
    return transform(fit(mat));

  }

  // fit_transform

  std::vector<std::vector<real_t>> bag_of_words::fit_transform(const std::vector<std::string> &documents)
  {
     std::vector<std::vector<real_t>> result;
     for(auto document : documents)
     {
       result.push_back(fit(document));
     }
      return result;  
 
   }

  std::vector<std::vector<real_t>> bag_of_words::fit_transform(const provallo::matrix<real_t> & mat)
  {
    std::vector<std::vector<real_t>> result;
    return transform(fit(mat));

    
  }
  std::vector<real_t> bag_of_words::predict(const std::string &doc)
  {
    return transform(doc);
  }
  std::vector<std::vector<real_t>> bag_of_words::predict(const std::vector<std::string> &documents)
  {
    return transform(documents);
  }
  std::vector<std::vector<real_t>> bag_of_words::predict(const provallo::matrix<real_t> & mat)
  {
    return transform(mat);
  }
  // inverse_transform
  std::string bag_of_words::inverse_transform(const std::vector<real_t>& reverse_tokens) 
  {
    std::string result;
    for (auto token : reverse_tokens)
    {
      auto it = std::find(_bow.begin(), _bow.end(), token);
      if (it != _bow.end())
      {
        // update bow value
        size_t index = std::distance(_bow.begin(), it);
        result += _vocabulary[index];
      }
      else
      {
        // add token to vocabulary
        result += " ";
      }
    }
    return result;
  }
  std::vector<std::string> bag_of_words::inverse_transform(const std::vector<std::vector<real_t>>& mat)
  {
    std::vector<std::string> result;
    for (auto document : mat)
    {
      result.push_back(inverse_transform(document));
    }
    return result;
  }
  //vtable functions:

  std::vector<std::string> bag_of_words::inverse_transform(const provallo::matrix<real_t>& mat)
  {
    std::vector<std::string> result;
    return inverse_transform(fit(mat));
  }
  // inverse_transform end
  // bag of words end
  
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
              tf_doc.insert(std::make_pair(word, 1.0));
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

  std::vector<std::vector<real_t>> tfidf::transform(const std::vector<std::string> &docs)
  {
    std::vector<std::vector<real_t>> result;
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
  pipeline_builder::operator=(pipeline_builder &&other)
  { // move assignment

    if (this != &other)
    {

      // copy everything :
      this->_pipelines = std::move(other._pipelines);
    }

    return *this;
  }

  pipeline_builder::pipeline_builder(const pipeline_builder &other) : _pipelines(other._pipelines)
  {
    // copy everything :
  }
  pipeline_builder::~pipeline_builder()
  {
    // clear everything :
  }

  pipeline_builder::pipeline_builder() : _current_pipeline(nullptr){}; // default constructor

  pipeline_builder::pipeline_builder(pipeline_builder &&other) : _pipelines(std::move(other._pipelines))
  {
    // move everything :
    // _pipelines = std::move(other._pipelines);
  }

  // load /save pipelines
  bool pipeline_builder::load_from_file(std::string filename)
  {
    bool ret = false;
    std::ifstream ifs(filename);
    // check if file exists :
    if (ifs.good())
    {
      // clear everything :
      _pipelines.clear();
      while (ifs.is_open())
      {
        // read pipeline :
        pipeline *p = new pipeline();
        ifs >> *p;
        _pipelines.push_back(p);
      } // end while
      ret = true;
    }
    return ret;
  }

  bool pipeline_builder::save_to_file(std::string filename)
  {
    bool ret = false;
    std::ofstream ofs(filename);
    // check if file exists :
    if (ofs.good())
    {
      for (auto &p : _pipelines)
      {
        ofs << *p;
      }
      ret = true;
    }
    return ret;
  }

  // pipeline_builder::add_pipeline

  void pipeline_builder::add_pipeline(const std::string &pipeline_name, bool load_from_file)
  {
    pipeline *p = new pipeline;
    if (load_from_file)
    {
      p->load_from_file(pipeline_name);
    }
    else
      p->set_pipeline_name(pipeline_name);

    add_pipeline(p);
  }

  void pipeline_builder::set_current_pipeline(uint64_t index)
  {
    auto it = std::find_if(_pipelines.begin(), _pipelines.end(), [index](pipeline *p)
                           { return p->get_pipeline_id() == index; });
    if (it != _pipelines.end())
    {
      _current_pipeline = *it;
    }
  }
  void pipeline_builder::set_current_pipeline(const std::string &pipeline_name)
  {
    // find index in pipelines
    auto it = std::find_if(_pipelines.begin(), _pipelines.end(), [pipeline_name](pipeline *p)
                           { return p->get_pipeline_name() == pipeline_name; });

    if (it != _pipelines.end())
      _current_pipeline = *it;
    else
    {
      add_pipeline(pipeline_name, false);
      _current_pipeline = _pipelines.back();
    }
  }

  /*
  void pipeline_builder::set_current_pipeline(pipeline* pipeline)
  {
    //find index in pipelines
    auto it = std::find_if(_pipelines.begin(),_pipelines.end(),[pipeline](pipeline* p){return p==pipeline;});
    if(it!=_pipelines.end())
    {
      _current_pipeline = std::distance(_pipelines.begin(),it);
    }
  }

  */

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
    size_t _pca_n_components_;
    size_t  _pca_n_features_;
    size_t  _pca_n_samples_;
  */

  // principal_component_analysis implementation:

  principal_component_analysis::principal_component_analysis() : _pca_n_components_(0), _pca_n_features_(0), _pca_n_samples_(0) {}
  
  //copy constructor:
  principal_component_analysis::principal_component_analysis(const principal_component_analysis& other) :
  _mean(other._mean),
  _variance(other._variance),
  _standard_deviation(other._standard_deviation),
  _standardized_data(other._standardized_data),

  _covariance_matrix(other._covariance_matrix),
  _eigen_values(other._eigen_values),
  _eigen_vectors(other._eigen_vectors),
  _pca_data(other._pca_data),
  _pca_components(other._pca_components),
  _pca_explained_variance(other._pca_explained_variance),
  _pca_explained_variance_ratio(other._pca_explained_variance_ratio),
  _pca_singular_values(other._pca_singular_values),
  _pca_noise_variance(other._pca_noise_variance),
  _pca_mean(other._pca_mean),
  _pca_n_components_(other._pca_n_components_),
  _pca_n_features_(other._pca_n_features_),
  _pca_n_samples_(other._pca_n_samples_)
  {}
  //move constructor:
  principal_component_analysis::principal_component_analysis(principal_component_analysis&& other) :  
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

  _pca_n_components_(std::move(other._pca_n_components_)),
  _pca_n_features_(std::move(other._pca_n_features_)),
  _pca_n_samples_(std::move(other._pca_n_samples_))
  {}

  //copy assignment

  principal_component_analysis::~principal_component_analysis() {
    //no allocation
  }
  //assignment 
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
      this->_pca_n_components_ = other._pca_n_components_;
      this->_pca_n_features_ = other._pca_n_features_;
      this->_pca_n_samples_ = other._pca_n_samples_;
      
     _pca_components_matrix = std::move(other._pca_components_matrix);
     _pca_explained_variance_matrix = std::move(other._pca_explained_variance_matrix);
     _pca_explained_variance_ratio_matrix = std::move(other._pca_explained_variance_ratio_matrix);
     _pca_singular_values_matrix = std::move(other._pca_singular_values_matrix);
    

      
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
      this->_pca_n_components_ = std::move(other._pca_n_components_);
      this->_pca_n_features_ = std::move(other._pca_n_features_);
      this->_pca_n_samples_ = std::move(other._pca_n_samples_);
      //

     _pca_components_matrix = std::move(other._pca_components_matrix);
     _pca_explained_variance_matrix = std::move(other._pca_explained_variance_matrix);
     _pca_explained_variance_ratio_matrix = std::move(other._pca_explained_variance_ratio_matrix);
     _pca_singular_values_matrix = std::move(other._pca_singular_values_matrix);
    
    }
    return *this;
  }
  // copy constructor

  // move constructor

  // copy assignment operator

  // move assignment operator

  // principal_component_analysis implementation:
  // virtual  std::vector<real_t> fit( const std::string &document) 
  std::vector<real_t> principal_component_analysis::fit ( const std::string& document) 
  {
     // tokenize the document into words
    std::vector<std::string> words ;
     tokenize(document,words, " ");
    for( auto & word : words )
    {
      auto occurances = std::count(words.begin(), words.end(), word);
        
      if  ( _vocabulary.find(word) == _vocabulary.end() )
      {
        //add the word to the vocabulary
        
        _vocabulary.insert(std::make_pair(word, occurances / _vocabulary.size()));
        //
        //update the word in the vocabulary
        
      }
      else
      {
        //update the word in the vocabulary
        _vocabulary[word] += (occurances / _vocabulary.size());
      }
    }
    //normalize the vocabulary
    for (auto &word : _vocabulary)
    {
      word.second /= _vocabulary.size();
    }
    // create a vector of real_t to store the results
    std::vector<real_t> results;
    // for each word in the vocabulary
    for (auto &word : _vocabulary)
    {
      // add the word to the results
      results.push_back(word.second);
    }
    // return the results
    return results;
  }
    // create a vector of real_t to store the results

  
  std::vector<real_t> principal_component_analysis::transform ( const std::string& doc) 
  {
    //TODO: implement the transform method without modifying the vocabulary.
    // create a vector of real_t to store the results

    std::vector<real_t> results;
    // tokenize the document into words
    std::vector<std::string> words ;
     tokenize(doc,words, " ");
    for( auto & word : words )
    {
      auto occurances = std::count(words.begin(), words.end(), word);
        
      if  ( _vocabulary.find(word) == _vocabulary.end() )
      {
        //add the word to the vocabulary
        
        _vocabulary.insert(std::make_pair(word, occurances / _vocabulary.size()));
        //
        //update the word in the vocabulary
        
      }
      else
      {
        //update the word in the vocabulary
        _vocabulary[word] += (occurances / _vocabulary.size());
      }
    }   
    //normalize the vocabulary
    for (auto &word : _vocabulary)
    {
      word.second /= _vocabulary.size();
    }
    // for each word in the vocabulary
    for (auto &word : _vocabulary)
    {
      // add the word to the results
      results.push_back(word.second);
    }
    // return the results
    return results;

  }
  // transform method 
  std::vector<std::vector<real_t>> principal_component_analysis::transform ( const std::vector<std::string>& docs) 
  { 
    //return the results of the transform method for each document in the vector of documents 
    std::vector<std::vector<real_t>> results(docs.size());
    size_t i=0;
    for (auto &doc : docs)
    {
        auto res = transform(doc);
        for (auto &r : res)
        {
          results[i].push_back(r);
        }
        i++;
    } 
    // return the results
    return results;

  }
  
  std::vector<std::vector<real_t>> principal_component_analysis::transform(const  std::vector<std::vector<std::string> > &data_)
  {
    std::vector<std::vector<real_t>> ret(data_.size());
    //first , transform the documents into vectors of real_t 
    std::vector<std::vector<real_t>> data(data_.size());
    size_t i=0;
    for (auto &doc : data_)
    {
        auto res = transform(doc);
        for (auto &r : res)
        {
          for (auto & d : r )
          data[i].push_back(d);
        }
        i++;
    }
    //then , transform the vectors of real_t into the new space
    for (auto &d : data)
    {
      auto res = transform(d);
      for (auto &r : res)
      {
        ret[i].push_back(r);
      }
      i++;
    }
    //return the results
    

    return ret;
  }

   
  // virtual  std::vector<std::vector<real_t>> fit( const std::vector<std::string> &documents)
  std::vector<std::vector<real_t>> principal_component_analysis::fit(const std::vector<std::string> &documents)
  {
    //for each set of documents aggregate the results to a vector of vectors instead of a vector of vectors of vectors 
    std::vector<std::vector<real_t>> results(documents.size());
    size_t i=0;
    for (auto &document : documents)
    {
        auto res = fit(document);
        for (auto &r : res)
        {
          results[i].push_back(r);
        }
        i++;
    }
    return results;
  }

  // QR decomposition
  void principal_component_analysis::QRDecomposition(const matrix<real_t> &mtx, matrix<real_t> &Q, matrix<real_t> &R)
  {
    // extract Q and R from the QR decomposition
    // Q is an orthogonal matrix
    // R is an upper triangular matrix

    // get the number of rows and columns
    size_t rows = mtx.rows();
    size_t cols = mtx.cols();

    // initialize Q and R
    Q = matrix<real_t>(rows, cols);
    R = matrix<real_t>(cols, cols);

    // initialize the first column of Q
    for (size_t i = 0; i < rows; ++i)
    {
      Q(i, 0) = mtx(i, 0);
    }

    // calculate the norm of the first column of Q
    real_t norm = 0.0;
    for (size_t i = 0; i < rows; ++i)
    {
      norm += Q(i, 0) * Q(i, 0);
    }

    // set the norm of the first column of Q to 1
    norm = std::sqrt(norm);
    for (size_t i = 0; i < rows; ++i)
    {
      Q(i, 0) /= norm;
    }

    // calculate the elements of R
    for (size_t i = 0; i < cols; ++i)
    {
      for (size_t j = i; j < cols; ++j)
      {
        R(i, j) = 0.0;
        for (size_t k = 0; k < rows; ++k)
        {
          R(i, j) += mtx(k, i) * Q(k, j);
        }
      }
    }

    // calculate the elements of Q
    for (size_t i = 1; i < cols; ++i)
    {
      for (size_t j = 0; j < rows; ++j)
      {
        Q(j, i) = 0.0;
        for (size_t k = 0; k < rows; ++k)
        {
          Q(j, i) += mtx(k, i) * R(k, j);
        }
      }
    }
    // get the transpose of Q
    Q = Q.transpose();

    // get the transpose of R
    R = R.transpose();

    // get the absolute value of the elements of R
    for (size_t i = 0; i < cols; ++i)
    {
      for (size_t j = 0; j < cols; ++j)
      {
        R(i, j) = std::abs(R(i, j));
      }
    }
    // get the sign of the elements of R
    for (size_t i = 0; i < cols; ++i)
    {
      if (R(i, i) < 0.0)
      {
        for (size_t j = 0; j < rows; ++j)
        {
          Q(j, i) = -Q(j, i);
        }
        for (size_t j = 0; j < cols; ++j)
        {
          R(i, j) = -R(i, j);
        }
      }
    }
  }

  // implementation of the PCA algorithm

  // undefined references : principal_component_analysis::predict,
  // principal_component_analysis::transform(provallo::matrix<real_t> const&)
  std::vector<real_t>
  principal_component_analysis::transform(provallo::matrix<real_t> const &data_)
  {
    // get the number of rows and columns
    size_t rows = data_.rows();
    size_t cols = data_.cols();

    // initialize the transformed data
    matrix<real_t> transformed_data(rows, cols);

    // transform the data
    for (size_t i = 0; i < rows; ++i)
    {
      // initialize the transformed data
      for (size_t j = 0; j < cols; ++j)
      {

        transformed_data(i, j) = 0.0;
        for (size_t k = 0; k < cols; ++k)
        {
          //
          transformed_data(i, j) += data_(i, k) * this->_pca_components_matrix(k, j);
        }
      }
    } // end of the loop over the rows
    // return the transformed data
    return std::vector<real_t>(transformed_data.begin(), transformed_data.end());
  }

  // fit
  std::vector<real_t> principal_component_analysis::fit(const provallo::matrix<real_t> &data_)
  {
    //
    // get the number of rows and columns
    size_t rows = data_.rows();
    size_t cols = data_.cols();
    std::vector<real_t> ret(cols, 0.0);
    // initialize the mean vector
    std::vector<real_t> mean(cols, 0.0);

    // calculate the mean vector
    for (size_t i = 0; i < cols; ++i)
    {
      for (size_t j = 0; j < rows; ++j)
      {
        mean[i] += data_(j, i);
      }
      mean[i] /= rows;
    }

    // initialize the centered data
    matrix<real_t> centered_data(rows, cols);

    // center the data
    for (size_t i = 0; i < rows; ++i)
    {
      for (size_t j = 0; j < cols; ++j)
      {
        centered_data(i, j) = data_(i, j) - mean[j];
      }
    }

    // initialize the covariance matrix
    matrix<real_t> covariance_matrix(cols, cols);

    // calculate the covariance matrix
    for (size_t i = 0; i < cols; ++i)
    {
      for (size_t j = i; j < cols; ++j)
      {
        covariance_matrix(i, j) = 0.0;
        for (size_t k = 0; k < rows; ++k)
        {
          covariance_matrix(i, j) += centered_data(k, i) * centered_data(k, j);
        }
        covariance_matrix(i, j) /= rows;
        covariance_matrix(j, i) = covariance_matrix(i, j);
      }
    } // end of the loop over the columns

 
    //QR Decomposition
    matrix<real_t> eigenvalues;
    matrix <real_t> eigenvectors_transpose;
    matrix<real_t> eigenvectors;

    // get the eigenvalues and eigenvectors
    this->QRDecomposition(covariance_matrix, eigenvalues, eigenvectors);
    // get the eigenvectors
    eigenvectors_transpose = eigenvectors.transpose();
    // get the eigenvalues
    for (size_t i = 0; i < cols; ++i)
    {
      ret[i] = eigenvalues(i, i);
    }
    // get the principal components matrix
    this->_pca_components_matrix = eigenvectors_transpose;
    // return the eigenvalues
    return ret;

    
  }
  std::vector<real_t>
  principal_component_analysis::predict(const provallo::matrix<real_t> &data_)
  {
    // get the number of rows and columns
    size_t rows = data_.rows();
    size_t cols = data_.cols();

    // initialize the transformed data
    matrix<real_t> transformed_data(rows, cols);

    // transform the data
    for (size_t i = 0; i < rows; ++i)
    {
      for (size_t j = 0; j < cols; ++j)
      {
        transformed_data(i, j) = 0.0;
        for (size_t k = 0; k < cols; ++k)
        {
          transformed_data(i, j) += data_(i, k) * this->_pca_components_matrix(k, j);
        }
      }
    }
  
    // return the transformed data
    return std::vector<real_t>(transformed_data.begin(), transformed_data.end());
  }

  std::vector<std::vector<real_t>> principal_component_analysis::predict(const std::vector<std::string> &documents)
  {
    // transform the documents
    std::vector<std::vector<real_t>> ret;
    for (size_t i = 0; i < documents.size(); ++i)
    {
      ret.push_back(this->predict(documents[i]));
    }
     return ret;
  } //

  std::vector<real_t> principal_component_analysis::predict(const std::string &document)
  { // parse the document

    std::vector<real_t> ret;
    // transform the document
    std::vector<std::string> tokens;
    
    tokenize(document, tokens ," ");
    
    ret.resize(tokens.size()+1, 0.0);

    //search each token and update or return 0.
    for (size_t i = 0; i < tokens.size(); ++i)
    {
      auto it = this->_vocabulary.find(tokens[i]);
      if (it != this->_vocabulary.end())
      {
        ret[it->second] += 1.0;
      }
      else
      {
        ret[tokens.size()] += 1.0;
      }
    }

    //    std::cout << "ret.size() = " << ret.size() << std::endl;  
    // transform the document
    ret = this->predict(ret);

    return ret;
  } 

  //
  //  principal_component_analysis::fit(std::vector<std::vector<std::string, std::allocator<std::string> >, std::allocator<std::vector<std::string, std::allocator<std::string> > > > const&)'

  std::vector<std::vector<real_t>> principal_component_analysis::fit ( const std::vector<std::vector<std::string> > &documents)
  {
    // get the number of documents
    size_t n_documents = documents.size();
    // initialize the transformed documents
    std::vector<std::vector<real_t>> ret(n_documents);
    // transform the documents
    for (size_t i = 0; i < n_documents; ++i)
    {
      //fit each document in the documents vector
      std::vector<std::vector<real_t>> fitret ( this->fit(documents[i])  );

      //copy the result
      ret[i].resize(fitret.size()*fitret[0].size(), 0.0);
      for (size_t j = 0; j < fitret.size(); ++j)
      {
        std::copy(fitret[j].begin(), fitret[j].end(), ret[i].begin() + j*fitret[j].size());
      }
    }
    // return the transformed documents
    return ret;
  }
  std::vector<std::vector<real_t>> principal_component_analysis::predict(const std::vector<std::vector<std::string>> &documents)
  {
    // get the number of documents
    size_t n_documents = documents.size();
    // initialize the transformed documents
    std::vector<std::vector<real_t>> ret(n_documents);
    // transform the documents
    for (size_t i = 0; i < n_documents; ++i)
    {
      //fit each document in the documents vector
      std::vector<std::vector<real_t>> fitret ( this->predict(documents[i])  );


      //copy the result
      ret[i].resize(fitret.size()*fitret[0].size(), 0.0);
      for (size_t j = 0; j < fitret.size(); ++j)
      {
        std::copy(fitret[j].begin(), fitret[j].end(), ret[i].begin() + j*fitret[j].size());
      }
      

    }

    // return the transformed documents
    return ret;
  }   


  std::vector<real_t> pca_vectorizer::predict(provallo::matrix<real_t> const &data_)
  {
    return _pca.predict(data_);
 }

  std::vector<real_t> pca_vectorizer::transform(provallo::matrix<real_t> const &data_)
  {
    return _pca.transform(data_);
  }

  std::vector<real_t> pca_vectorizer::predict(std::vector<std::string> const &documents)
  {
    std::vector<std::vector<real_t>> ret = _pca.predict(documents);
    std::vector<real_t> ret_value;
    //flatten the vector
    for (size_t i = 0; i < ret.size(); ++i)
    {
      for (size_t j = 0; j < ret[i].size(); ++j)
      {
        ret_value.push_back(ret[i][j]);
      }
    }
    return ret_value;
  }

  std::vector<real_t> pca_vectorizer::fit_transform(std::vector<std::string> const &documents)
  {
   std::vector<std::vector<real_t>> ret = _pca.fit(documents);
   std::vector<real_t> ret_value;
    //flatten the vector
    for (size_t i = 0; i < ret.size(); ++i)
    {
      for (size_t j = 0; j < ret[i].size(); ++j)
      {
        ret_value.push_back(ret[i][j]);
      }
    }
    return ret_value;

  }
  vectorizer_type pca_vectorizer::get_type() const
  {
    return vectorizer_type::PCA;
  }
  std::vector<real_t> pca_vectorizer::transform(std::vector<std::string> const &documents)
  {
    std::vector<std::vector<real_t>> ret = _pca.transform(documents);
    std::vector<real_t> ret_value;
    //flatten the vector
    for (size_t i = 0; i < ret.size(); ++i)
    {
      for (size_t j = 0; j < ret[i].size(); ++j)
      {
        ret_value.push_back(ret[i][j]);
      }
    }
    return ret_value;
  }
  std::vector<real_t> standard_scaler_vectorizer::predict(provallo::matrix<real_t> const &data_)
  {
    // use _fitted_data,mean and std to transform the data
    // get the number of rows and columns
    size_t rows = data_.rows();
    size_t cols = data_.cols();
    matrix<real_t> _std = data_.std();
    // initialize the transformed data
    matrix<real_t> transformed_data(rows, cols);

    // transform the data
    for (size_t i = 0; i < rows; ++i)
    {
      for (size_t j = 0; j < cols; ++j)
      {
        transformed_data(i, j) = (data_(i, j) - _mean[j]) / _std(i, j);
      }
    }
    // return the transformed data
    return std::vector<real_t>(transformed_data.begin(), transformed_data.end());
  }

  // tfidf_vectorizer
  std::vector<real_t> tfidf_vectorizer::predict(provallo::matrix<real_t> const &data_)
  {
    UNDEF_REFERENCE(data_);
    return std::vector<real_t>();
  }

  std::vector<real_t> standard_scaler_vectorizer::transform(const provallo::matrix<real_t> &data_matrix)
  {
    // use _fitted_data,mean and std to transform the data
    // get the number of rows and columns
    size_t rows = data_matrix.rows();
    size_t cols = data_matrix.cols();
    matrix<real_t> _std = _fitted_data.std();
    // initialize the transformed data
    matrix<real_t> transformed_data(rows, cols);
    // transform the data
    for (size_t i = 0; i < rows; ++i)
    {
      for (size_t j = 0; j < cols; ++j)
      {
        transformed_data(i, j) = (data_matrix(i, j) - _mean[j]) / _std(i, j);
      }
    }
    // return the transformed data
    return std::vector<real_t>(transformed_data.begin(), transformed_data.end());
  }

  std::vector<real_t> standard_scaler_vectorizer::predict(const std::vector<std::string> &documents)
  {
    // use _fitted_data,mean and std to transform the data
    // get the number of rows and columns
    size_t rows = documents.size();

    // initialize the transformed data
    matrix<real_t> transformed_data(rows, 1);
    // transform the data
    matrix<real_t> _std = _fitted_data.std();
    for (size_t i = 0; i < rows; ++i)
    {
      transformed_data(i, 0) = (std::stod(documents[i]) - _mean[0]) / _std(0, 0);
    }

    // return the transformed data
    return std::vector<real_t>(transformed_data.begin(), transformed_data.end());
  }

  std::vector<real_t> standard_scaler_vectorizer::transform(const std::vector<std::string> &documents)
  {
    // use _fitted_data,mean and std to transform the data
    // get the number of rows and columns
    size_t rows = documents.size();

    // initialize the transformed data
    matrix<real_t> transformed_data(rows, 1);
    // transform the data
    matrix<real_t> _std = _fitted_data.std();
    for (size_t i = 0; i < rows; ++i)
    {
      transformed_data(i, 0) = (std::stod(documents[i]) - _mean[0]) / _std(0, 0);
    }

    // return the transformed data
    return std::vector<real_t>(transformed_data.begin(), transformed_data.end());
  }

  template <>
  std::vector<real_t>
  provallo::vectorizer<std::string, real_t>::predict(const std::vector<std::string> &documents)
  {
    // use _fitted_data,mean and std to transform the data
    // get the number of rows and columns
    size_t rows = documents.size();

    // initialize the transformed data
    matrix<real_t> transformed_data(matrix<real_t>::Zero(rows, 1));
    // transform the data
    matrix<real_t> _std = _fitted_data.std();
    real_t _mean = _fitted_data.mean();

    for (size_t i = 0; i < rows; ++i)
    {
      transformed_data(i, 0) = (std::stod(documents[i]) - _mean / _std(0, 0));
    }
    // return the transformed data
    return std::vector<real_t>(transformed_data.begin(), transformed_data.end());
  }

  std::vector<real_t> tfidf_vectorizer::transform(const std::vector<std::string> &documents)
  {
    // use _fitted_data,mean and std to transform the data
    // get the number of rows and columns
    size_t rows = documents.size();

    // initialize the transformed data
    std::vector<std::vector<real_t>> transformed_data(_tfidf.transform(documents));
    std::vector<real_t> transformed_ret(rows, 0.0);
    // go over the data and fill the transformed data
    for (size_t i = 0; i < rows; ++i)
    {
      for (size_t j = 0; j < transformed_data[i].size(); ++j)
        transformed_ret[i] += transformed_data[i][j] / real_t(transformed_data[i].size());
    }

    // flatten the data

    // return the transformed data
    return transformed_ret;
  }

  std::vector<real_t> tfidf_vectorizer::transform(const matrix<real_t> &trans)
  {
    // use _tfidf to transform the data

    size_t rows = trans.rows();
    size_t cols = trans.cols();

    // use _tfidf inverse_transform to transform the data
    std::vector<std::vector<real_t>> ret_vec(rows, std::vector<real_t>(cols, 0.0));
    // fill ret_vec with trans data
    for (size_t i = 0; i < rows; i++)
      for (size_t j = 0; j < cols; ++j)
        ret_vec[i][j] = trans(i, j);
    // transform the data

    std::vector<std::string> transformed_data = _tfidf.inverse_transform(ret_vec); // _tfidf.inverse_transform( )
    // go over the data and fill the ret_vec with the transformed data
    for (size_t i = 0; i < rows; ++i)
    {
      for (size_t j = 0; j < cols; ++j)
      {
        if (i * cols + j < transformed_data.size())
          ret_vec[i][j] = std::stod(transformed_data[i * cols + j]);
        else

          ret_vec[i][j] = 0.0;
      }
    }
    // return the transformed data
    std::vector<real_t> ret_val(ret_vec.size() * ret_vec[0].size(), 0.0);
    for (size_t i = 0; i < rows; ++i)
    {
      for (size_t j = 0; j < cols; ++j)
      {
        ret_val[i * cols + j] = ret_vec[i][j];
      }
    }
    return ret_val;
  }

  std::vector<real_t> tfidf_vectorizer::fit(const provallo::matrix<double> &foot)
  {
    // not implemented.

    return std::vector<real_t>(foot.begin(), foot.end());
  }

  // pipeline implementation :
  // initialize parent class with the given parameters

  pipeline::pipeline()
  {
    // don't do anything until we have some stages to add
  }

  // load the stages from the given file
  pipeline::pipeline(const std::string &filename)
  {
    // load the stages from the given file
    if (!load_from_file(filename))
      throw std::runtime_error("could not load the pipeline from the given file");
  }

  bool pipeline::load_stage(std::ifstream &stage_file, const stage_descriptor &stage)
  {
    bool ret_val = false;
    // find the additional data if any for the stage and load it
    std::string line;
    // build object from the stage descriptor
    stage_descriptor *new_stage = nullptr;

    // create stage from the stage descriptor
    new_stage = stage_factory_singleton::get_instance()->build_stage(stage);
    if (new_stage == nullptr)
      throw std::runtime_error("stage not found");
    // load the additional data

    std::getline(stage_file, line);
    new_stage->load_additional_data(line);
    ret_val = true;
    return ret_val;
  }
  bool pipeline::load_from_file(const std::string &filename)
  {
    std::ifstream file(filename);
    if (!file.is_open())
      return false;
    std::string line;
    while (std::getline(file, line))
    {
      // parse the line and add the stage to the pipeline from each stage descriptor :
      // the stage descriptor is a string with the following format :
      //     size_t stage_id;
      //      std::string name;
      //  std::string type;
      //  std::string parameters;
      //  std::string input;
      //  std::string output;
      //  std::string input_type;
      //  std::string output_type;
      //  std::string input_parameters;
      //  std::string output_parameters;

      // parse the line and create a stage descriptor
      stage_descriptor stage(line);
      // load the stage from the file
      if (!load_stage(file, stage))
        return false;
    }
    return true;
  }
  bool pipeline::save_to_file(const std::string &filename)
  {
    // not implemented
    std::ofstream file(filename);
    if (!file.is_open())
      return false;
    // save the stages to the file
    for (auto stage : _stages)
    {
      file << *stage << std::endl;
      file << stage->get_additional_data() << std::endl;
      file << std::endl;
    }
    file.close();
    return true;
  }

  // pipeline destructor
  pipeline::~pipeline()
  {
    // delete all the stages
    for (auto stage : _stages)
      delete stage;
  }
  // add a stage to the pipeline
  void pipeline::add_stage(stage_descriptor *stage)
  {
    _stages.push_back(stage);
  }
  // remove a stage from the pipeline
  void pipeline::remove_stage(size_t stage_id)
  {
    // find the stage with the given id
    auto it = std::find_if(_stages.begin(), _stages.end(), [stage_id](stage_descriptor *stage)
                           { return stage->stage_id == stage_id; });
    if (it != _stages.end())
    {
      // remove the stage from the vector
      _stages.erase(it);
      // delete the stage
      if (*it != nullptr)
        delete *it;
    }
  }
  // get the stage with the given id
  stage_descriptor *pipeline::get_stage(uint64_t stage_id)
  {
    // find the stage with the given id
    auto it = std::find_if(_stages.begin(), _stages.end(), [stage_id](stage_descriptor *stage)
                           { return stage->stage_id == stage_id; });
    if (it != _stages.end())
    {
      // return the stage
      return *it;
    }
    return nullptr;
  }
  // get the stage with the given name
  stage_descriptor *pipeline::get_stage(const std::string &stage_name)
  {
    // find the stage with the given name
    auto it = std::find_if(_stages.begin(), _stages.end(), [stage_name](stage_descriptor *stage)
                           { return stage->name == stage_name; });
    if (it != _stages.end())
    {
      // return the stage
      return *it;
    }
    return nullptr;
  }
  // get the next stage
  stage_descriptor *pipeline::get_next_stage(uint64_t stage_id)
  {
    // find the stage with the given id
    auto it = std::find_if(_stages.begin(), _stages.end(), [stage_id](stage_descriptor *stage)
                           { return stage->stage_id == stage_id; });
    if (it != _stages.end())
    {
      // return the next stage
      return *(++it);
    }
    return nullptr;
  }
  // get the previous stage
  stage_descriptor *pipeline::get_previous_stage(uint64_t stage_id)
  {
    // find the stage with the given id
    auto it = std::find_if(_stages.begin(), _stages.end(), [stage_id](stage_descriptor *stage)
                           { return stage->stage_id == stage_id; });
    if (it != _stages.end())
    {
      // return the previous stage
      return *(--it);
    }
    return nullptr;
  }
  // get the first stage
  stage_descriptor *pipeline::get_first_stage()
  {
    if (_stages.size() > 0)
    {
      // return the first stage
      return _stages[0];
    }
    return nullptr;
  }
  // get the last stage
  stage_descriptor *pipeline::get_last_stage()
  {
    if (_stages.size() > 0)
    {
      // return the last stage
      return _stages[_stages.size() - 1];
    }
    return nullptr;
  }
  // get the number of stages
  // from the pipeline and the pipelines it contains.
  size_t pipeline::get_number_of_stages() const
  {
     size_t stages = 0;
    for(pipeline* pipe : _pipelines)
    {

      if(pipe&& pipe->get_number_of_stages() > 0 &&pipe!=this)
      {
        stages+= pipe->get_number_of_stages();
      }
    }
    //there are stages for each aggregated pipeline
    //and the stages of the current pipeline
    //minus the stages that describes the pipelines

    return _stages.size()+stages-_pipelines.size();
  }
  std::ofstream &operator<<(std::ofstream &os, const stage_descriptor &stage)
  {
    // write the stage
    std::string stage_id = "stage_id: ";
    std::string name = "name : ";
    std::string type = "type : ";
    std::string parameters = "parameters : ";
    std::string input = "input : ";
    std::string output = "output : ";
    std::string input_type = "input_type : ";
    std::string output_type = "output_type : ";
    std::string input_parameters = "input_parameters : ";
    std::string output_parameters = "output_parameters : ";
    // write the stage id
    os << stage_id.c_str() << stage.stage_id << std::endl;
    os << name.c_str() << stage.name.c_str() << std::endl;
    os << type.c_str() << stage.type.c_str() << std::endl;
    os << parameters.c_str() << stage.parameters.c_str() << std::endl;
    os << input.c_str() << stage.input.c_str() << std::endl;
    os << output.c_str() << stage.output.c_str() << std::endl;
    os << input_type.c_str() << stage.input_type.c_str() << std::endl;
    os << output_type.c_str() << stage.output_type.c_str() << std::endl;
    os << input_parameters.c_str() << stage.input_parameters.c_str() << std::endl;
    os << output_parameters.c_str() << stage.output_parameters.c_str() << std::endl;

    return os;
  }
  // stage descriptor ifstream

  std::ifstream &operator>>(std::ifstream &is, stage_descriptor &stage)
  {
    std::string stage_id = "stage_id: ";
    std::string name = "name : ";
    std::string type = "type : ";
    std::string parameters = "parameters : ";
    std::string input = "input : ";
    std::string output = "output : ";
    std::string input_type = "input_type : ";
    std::string output_type = "output_type : ";
    std::string input_parameters = "input_parameters : ";
    std::string output_parameters = "output_parameters : ";
    // read the stage id
    is >> stage_id >> stage.stage_id;
    is >> name >> stage.name;
    is >> type >> stage.type;
    is >> parameters >> stage.parameters;
    is >> input >> stage.input;
    is >> output >> stage.output;
    is >> input_type >> stage.input_type;
    is >> output_type >> stage.output_type;
    is >> input_parameters >> stage.input_parameters;
    is >> output_parameters >> stage.output_parameters;

    // make sure that the stage is valid
    if (stage.stage_id == 0 || stage.name == "" || stage.type == "")
      throw std::runtime_error("invalid stage");

    // read the stage additional data after construction of the object.
    return is;
  }

  // pipeline ifstream

  std::ifstream &operator>>(std::ifstream &is, pipeline &p)
  {
    // read the number of stages
    size_t number_of_stages;
    is >> number_of_stages;
    // read the stages
    for (size_t i = 0; i < number_of_stages; i++)
    {
      // read the stage
      stage_descriptor stage;
      is >> stage;
      stage_descriptor *new_stage = nullptr;
      // create the stage
      new_stage = stage_factory_singleton::get_instance()->build_stage(stage);
      if (new_stage == nullptr)
        throw std::runtime_error("stage not found");

      // add the stage to the pipeline
      p.add_stage(new_stage);
    }
    return is;
  }
  // pipeline ofstream
  std::ofstream &operator<<(std::ofstream &os, const pipeline &p)
  {

    os << "name:" << p.get_pipeline_name() << std::endl;
    os << "id:" << p.get_pipeline_id() << std::endl;

    // first see if any pipelines are aggregated in this pipeline
    for (auto pipeline : p._pipelines)
    {
      os << *pipeline << std::endl;
    }

    os << "#stages:" << p.get_number_of_stages() << std::endl;
    // write the stages
    for (auto stage : p._stages)
    {

      os << *stage << std::endl;
    }
    return os;
  }

  std::ostream &operator<<(std::ostream &os, const pipeline &p)
  {
    os << "pipeline name: " << p.get_pipeline_name() << std::endl;
    os << "pipeline id: " << p.get_pipeline_id() << std::endl;
    // write the stages
    os << "#pipelines: " << p._pipelines.size() << std::endl;
    for (auto pipeline : p._pipelines)
    {
      os << *pipeline << std::endl;
    }
    os << "#stages: " << p._stages.size() << std::endl;

    for (auto stage : p._stages)
    {
      os << *stage << std::endl;
    }
    return os;
  }

  // set_pipeline_name
  void pipeline::set_pipeline_name(const std::string &name_)
  {
    _pipe_name = name_;
  }
  // get_pipeline_name
  std::string pipeline::get_pipeline_name() const
  {
    return _pipe_name;
  }
  void pipeline_builder::add_pipeline(pipeline *pipe)
  {
    if(pipe != nullptr)
      _pipelines.push_back(pipe);

    if (_current_pipeline != nullptr &&pipe!=_current_pipeline)
      _current_pipeline = pipe;


  }
  // pipeline_builder constructor
  
  pipeline_builder::pipeline_builder (const std::string & filename , bool load_from_file): _filename(filename), _name(filename) 
  {
          // read the pipeline from the file
        if (load_from_file)
        {
         std::ifstream is(filename);

        if (!is.is_open()){
             throw std::runtime_error("could not open file");
        }
        is >> *this;
        is.close();        
        }
        else
        {
            // create a new pipeline
            _current_pipeline = new pipeline( );
            _current_pipeline->set_pipeline_name(filename);
            // add the pipeline to the builder
            add_pipeline(_current_pipeline);
        } 
       // 

   }
    // pipeline_builder constructor

    pipeline_builder::pipeline_builder (const std::string & filename , const std::string & pipeline_name, bool load_from_file) : _filename(filename) , _name(pipeline_name)    
    { 
      // read the pipeline from the file
        if (load_from_file)
        {
         std::ifstream is(_filename);

        if (!is.is_open()){
             throw std::runtime_error("could not open file");
        }
        is >> *this;
        is.close();        
        }
        else
        {
            // create a new pipeline
            _current_pipeline = new pipeline(pipeline_name);
            // add the pipeline to the builder
            add_pipeline(_current_pipeline);
            

        } 
       // 

    }


  std::ifstream &operator>>(std::ifstream &is, pipeline_builder &pb)
  {
    // read the number of pipelines
    size_t number_of_pipelines;
    is >> number_of_pipelines;
    // read the pipelines
    for (size_t i = 0; i < number_of_pipelines; i++)
    {
      // read the pipeline
      pipeline *pipe = new pipeline();
      is >> *pipe;
      // add the pipeline to the builder
      pb.add_pipeline(pipe);
    }
    return is;
  }
  // stage descritor ifstream/ofstream

  std::ostream &operator<<(std::ostream &out, const stage_descriptor &stage)
  {
    // write the stage
    out << "stage_id: " << stage.stage_id << std::endl;
    out << "name : " << stage.name << std::endl;
    out << "type : " << stage.type << std::endl;
    out << "parameters : " << stage.parameters << std::endl;
    out << "input : " << stage.input << std::endl;
    out << "output : " << stage.output << std::endl;
    out << "input_type : " << stage.input_type << std::endl;
    out << "output_type : " << stage.output_type << std::endl;
    out << "input_parameters : " << stage.input_parameters << std::endl;
    out << "output_parameters : " << stage.output_parameters << std::endl;
    return out;
  }
  // stage descriptor ofstream
  //     friend std::ostream& operator<<(std::ostream& os, const stage_descriptor& sd);
  //    friend std::istream& operator>>(std::istream& is, stage_descriptor& sd);

  // stage descriptor ifstream/ofstream
  // cluster stage metrics :

  // https://www.sciencedirect.com/science/article/pii/S016786551730166X
  // pearson

  real_t cluster_stage::pearson_correlation(const std::vector<real_t> &x, const std::vector<real_t> &y, real_t mean_x, real_t mean_y)
  {
    real_t sum_sq_x = 0.0;
    real_t sum_sq_y = 0.0;
    real_t sum_coproduct = 0.0;

    for (size_t i = 2; i < x.size() + 1; i += 1)
    {
      real_t sweep = (i - 1.0) / i;
      real_t delta_x = x[i - 1] - mean_x;
      real_t delta_y = y[i - 1] - mean_y;
      sum_sq_x += delta_x * delta_x * sweep;
      sum_sq_y += delta_y * delta_y * sweep;
      sum_coproduct += delta_x * delta_y * sweep;
      mean_x += delta_x / i;
      mean_y += delta_y / i;
    }
    real_t pop_sd_x = sqrt(sum_sq_x / x.size());
    real_t pop_sd_y = sqrt(sum_sq_y / x.size());
    real_t cov_x_y = sum_coproduct / x.size();
    return cov_x_y / (pop_sd_x * pop_sd_y);
  }

  std::vector<real_t> cluster_stage::rogerstanimoto_distances(const matrix<real_t> &data)
  {
    // first define correlation matrices for the two objects
    // then calculate the distance between the two correlation matrices
    // Expressions for finding the most informative correlation matrices via cost functions
    // of the form Jρ(K) = ρ ‖A − K‖2
    // F + F (dN (K)).
    // The cost function Jρ(K) is a function of the correlation matrix K, and the
    // function dN (K) is the derivative of the cost function with respect to the correlation
    // russellrao_distances - Quantifying the Informativeness of Similarity Measurements
    // https://arxiv.org/pdf/1802.02547.pdf

    std::vector<real_t> ret(data.rows() * data.rows());
    std::vector<real_t> x(data.cols());
    std::vector<real_t> y(data.cols());

    matrix<real_t> xcorr(data.cols(), data.cols());
    matrix<real_t> ycorr(data.cols(), data.cols());

    // calculate the correlation matrices xcorr and ycorr for x and y
    // correlation matrix is a square matrix that contains the Pearson product-moment correlation coefficients (often abbreviated as Pearson's r), which measure the linear dependence between pairs of features.
    // Pearson's r is a measure of the linear correlation between two variables X and Y. It's value lies between -1 and +1, -1 indicating total negative linear correlation, 0 indicating no linear correlation, and +1 indicating total positive linear correlation.

    // Pearson's r is calculated as follows:
    // r = (nΣxy - (Σx)(Σy)) / sqrt((nΣx2 - (Σx)2)(nΣy2 - (Σy)2))
    // where Σxy is the sum of the products of the corresponding values of the two data sets, Σx is the sum of the first data set, Σy is the sum of the second data set, Σx2 is the sum of the squares of the first data set, Σy2 is the sum of the squares of the second data set, and n is the number of values in each data set.

    for (size_t i = 0; i < data.rows(); ++i)
      for (size_t j = 0; j < data.rows(); ++j)
      {
        for (size_t k = 0; k < data.cols(); ++k)
        {
          x[k] = data(i, k);
          y[k] = data(j, k);
        }
        xcorr(i, j) = pearson_correlation(x, y);
        ycorr(i, j) = pearson_correlation(y, x);
      }
    // calculate the distance between the two correlation matrices and fill the ret vector

    for (size_t i = 0; i < xcorr.rows() && i < ycorr.rows(); ++i)
      for (size_t j = 0; j < xcorr.cols() && j < ycorr.cols(); ++j)
      {
        ret[i * xcorr.rows() + j] = 1.0 - xcorr(i, j) * ycorr(i, j);
      }

    return ret;
  }

  std::vector<real_t> cluster_stage::russellrao_distances(const matrix<real_t> &data)
  {
    // first define correlation matrices for the two objects
    // then calculate the distance between the two correlation matrices
    // Expressions for finding the most informative correlation matrices via cost functions
    // of the form Jρ(K) = ρ ‖A − K‖2
    // F + F (dN (K)).
    // The cost function Jρ(K) is a function of the correlation matrix K, and the
    // function dN (K) is the derivative of the cost function with respect to the correlation
    std::vector<real_t> ret(data.rows());
    std::vector<real_t> x(data.cols());

    matrix<real_t> xcorr(data.cols(), data.cols());
    matrix<real_t> ycorr(data.cols(), data.cols());
    // calculate the correlation matrices xcorr and ycorr for x and y

    for (size_t i = 0; i < data.rows(); ++i)
    {
      for (size_t k = 0; k < data.cols(); ++k)
      {
        x[k] = data(i, k);
        ycorr(i, k) = data(i, k);
        xcorr(i, k) = data(i, k);
      }
      ret[i] = 1.0 - (real_t)std::count(x.begin(), x.end(), 1) / x.size();
    }

    // return the ret vector

    for (size_t i = 0; i < xcorr.rows() && i < ycorr.rows(); ++i)
      for (size_t j = 0; j < xcorr.cols() && j < ycorr.cols(); ++j)
      {
        ret[i * xcorr.rows() + j] = 1.0 - xcorr(i, j) * ycorr(i, j);
      }
    // return the ret vector
    return ret;
  }

  // kulsinski_distances - A Generalized Formula for Computing Pairwise Distances for Multi-Valued Attributes
  std::vector<real_t> cluster_stage::kulsinski_distances(const matrix<real_t> &data)
  {
    // continous kulsinski coefficient cK=|X intersection Y |/ c*|X| + |Y| - |X intersection Y|

    std::vector<real_t> ret(data.rows() * data.rows());
    std::vector<real_t> x(data.cols());
    std::vector<real_t> y(data.cols());

    // initialize x and y from data
    for (size_t i = 0; i < data.rows(); i++)
    {
      real_t c = 0;
      real_t c1 = 0;
      real_t c2 = 0;
      for (size_t k = 0; k < data.cols(); k++)
      {
        x[k] = data(i, k);
        y[k] = data(i, k);
        if (x[k] == 1 && y[k] == 1)
        {
          c++;
        }
        if (x[k] == 1)
        {
          c1++;
        }
        if (y[k] == 1)
        {
          c2++;
        }
        ret[i] += (c / (c1 + c2 - c)) / data.rows();
      }
    }
    // return the ret vector
    return ret;
  }

  // dice
  std::vector<real_t> cluster_stage::dice_distances(const matrix<real_t> &data)
  {
    // continous dice coefficient cDC=2 |X intersection Y |/ c*|X| + |Y|

    std::vector<real_t> ret(data.rows() * data.rows());
    std::vector<real_t> x(data.cols());
    std::vector<real_t> y(data.cols());
    // initialize x and y from data
    for (size_t i = 0; i < data.rows(); i++)
    {
      real_t c = 0;
      real_t c1 = 0;
      real_t c2 = 0;
      for (size_t k = 0; k < data.cols(); k++)
      {
        x[k] = data(i, k);
        y[k] = data(i, k);
        if (x[k] == 1 && y[k] == 1)
        {
          c++;
        }
        if (x[k] == 1)
        {
          c1++;
        }
        if (y[k] == 1)
        {
          c2++;
        }
        ret[i] += (2 * c / (c1 + c2)) / data.rows();
      }
    }
    return ret;
  }

  std::vector<real_t> cluster_stage::sokalsneath_distances(const matrix<real_t> &data)
  {
    // calculate sokalsneath distance of the data matrix:
    // dxy = (a + b) / (a + b + 2c)

    std::vector<real_t> ret(data.rows() * data.rows());
    // for each pair of vectors

    real_t a = 0;
    real_t b = 0;
    real_t c = 0;
    for (size_t i = 0; i < data.rows(); i++)
      for (size_t j = 0; j < data.rows(); j++)
      {
        a = 0;
        b = 0;
        c = 0;
        for (size_t k = 0; k < data.cols(); k++)
        {
          if (data(i, k) == data(j, k))
          {
            if (data(i, k) == 1)
            {
              a++;
            }
            else
            {
              b++;
            }
          }
          else
          {
            c++;
          }
        }
        ret[i * data.rows() + j] = (a + b) / (a + b + 2 * c);
      }

    // return the result
    return ret;
  }
  std::vector<real_t> cluster_stage::sokalmichener_distances(const matrix<real_t> &data)
  {
    // calculate sokalmichener distance of the data matrix:
    // dxy = (a + b) / (a + b + 2c)
    std::vector<real_t> ret(data.rows() * data.rows());
    // for each pair of vectors
    //--------------------------------------------------------------------
    for (size_t i = 0; i < data.rows(); i++)
    {
      for (size_t j = i; j < data.rows(); j++)
      {
        // calculate a, b, c
        real_t a = 0;
        real_t b = 0;
        real_t c = 0;
        for (size_t k = 0; k < data.cols(); k++)
        {
          if (data(i, k) == data(j, k))
          {
            if (data(i, k) == 1)
            {
              a++;
            }
            else
            {
              b++;
            }
          }
          else
          {
            c++;
          }
        }
        // calculate distance
        ret[i * data.rows() + j] = (a + b) / (a + b + 2 * c);
        ret[j * data.rows() + i] = ret[i * data.rows() + j];
      }
    }
    return ret;
  }
  std::vector<real_t> cluster_stage::haversine_distances(const matrix<real_t> &data)
  {
    // calculate haversine distance of the data matrix:
    // dxy = 2 * asin(sqrt(sin((lat1-lat2)/2)^2 + cos(lat1) * cos(lat2) * sin((lon1-lon2)/2)^2))
    std::vector<real_t> ret(data.rows() * data.rows());
    // extract lon and lat from data
    std::vector<real_t> lon(data.rows());
    std::vector<real_t> lat(data.rows());
    for (size_t i = 0; i < data.rows(); i++)
    {
      lon[i] = data(i, 0);
      lat[i] = data(i, 1);
    }
    // for each pair of vectors
    //--------------------------------------------------------------------
    for (size_t i = 0; i < data.rows(); i++)
    {
      for (size_t j = 0; j < data.rows(); j++)
      {
        // initialize haversine distance
        real_t haversine = 0;
        // calculate haversine distance
        real_t dlon = lon[i] - lon[j];
        real_t dlat = lat[i] - lat[j];
        real_t a = pow(sin(dlat / 2), 2) + cos(lat[i]) * cos(lat[j]) * pow(sin(dlon / 2), 2);
        haversine = 2 * asin(sqrt(a));
        // store haversine distance
        ret[i * data.rows() + j] = haversine;
      }
    } // end for each pair of vectors
    return ret;
  }
  std::vector<real_t> cluster_stage::yule_distances(const matrix<real_t> &data)
  {
    // calculate yule distance of the data matrix:
    std::vector<real_t> ret(data.rows() * data.rows());
    // calculate the dissimilarity between two probability distributions based on their overlap
    // The Yule distance between two probability distributions, P and Q is given by: The distance ranges from 0 to 1, with 0 indicating that the two distributions are identical and 1 indicating that they have no overlap

    std::vector<real_t> p(data.cols());
    std::vector<real_t> q(data.cols());
    // for each pair of vectors
    //--------------------------------------------------------------------

    for (size_t i = 0; i < data.rows(); i++)
    {
      for (size_t j = 0; j < data.rows(); j++)
      {
        // initialize yule distance
        real_t yule = 0.0;
        // for each coordinate
        for (size_t k = 0; k < data.cols(); k++)
        {
          // calculate the probability of the kth coordinate of the ith vector
          p[k] = data(i, k) / data.row_sum(i);
          // calculate the probability of the kth coordinate of the jth vector
          q[k] = data(j, k) / data.row_sum(j);
          // calculate the yule distance
          yule += (p[k] * q[k]) / (p[k] + q[k]);
        }
        // set the yule distance
        ret[i * data.rows() + j] = 1.0 - yule;
      }
    }
    return ret;
  }

  //
  // braycurtis
  // The Bray curtis distance has a nice property that if all coordinates are postive,
  // its value is between zero and one.
  // Zero bray curtis represent exact similar coordinate. If both objects are in the zero coordinates, the Bray curtis distance is undefined.

  std::vector<real_t> cluster_stage::braycurtis_distances(const matrix<real_t> &data)
  {
    // calculate bray curtis distance of the data matrix:
    std::vector<real_t> ret(data.rows() * data.rows());
    // bray curtis distance is the sum of the absolute difference between the elements of the vectors
    // for each pair of vectors

    // for each pair of vectors
    //--------------------------------------------------------------------

    for (size_t i = 0; i < data.rows(); i++)
    {
      for (size_t j = 0; j < data.rows(); j++)
      {
        // initialize bray curtis distance to 0
        ret[i * data.rows() + j] = 0.0;

        // for each element of the vectors
        for (size_t k = 0; k < data.cols(); k++)
        {
          // calculate bray curtis distance
          ret[i * data.rows() + j] += std::abs(data(i, k) - data(j, k)) / (std::abs(data(i, k)) + std::abs(data(j, k)));
        }
      }
    }
    //--------------------------------------------------------------------
    return ret;
  }

  // canberra_distances
  std::vector<real_t> cluster_stage::canberra_distances(const matrix<real_t> &data)
  {
    // is a numerical measure of the distance between pairs of points in a vector space, introduced in 1966 and refined in 1967 by Godfrey N
    // calculate canberra distance of the data matrix:
    std::vector<real_t> ret(data.rows() * data.rows());
    // canberra distance is the sum of the absolute difference between the elements of the vectors
    // for each pair of vectors

    for (size_t i = 0; i < data.rows(); i++)
    {
      for (size_t j = 0; j < data.rows(); j++)
      {
        // initialize canberra distance to 0
        ret[i * data.rows() + j] = 0.0;

        for (size_t k = 0; k < data.cols(); k++)
        {
          ret[i * data.rows() + j] += std::abs(data(i, k) - data(j, k)) / (std::abs(data(i, k)) + std::abs(data(j, k)));
        }
      }
    }
    return ret;
    // no score for this metric
  }

  std::vector<real_t> cluster_stage::chebyshev_distances(const matrix<real_t> &data)
  {
    // calculate chebyshev distance of the data matrix
    std::vector<real_t> ret(data.rows() * data.rows());
    // chebeshev distance is the maximum of the absolute difference between the elements of the vectors
    // for each pair of vectors

    for (size_t i = 0; i < data.rows(); i++)
    {
      for (size_t j = 0; j < data.rows(); j++)
      {
        // initialize chebeshev distance to 0
        ret[i * data.rows() + j] = 0.0;

        for (size_t k = 0; k < data.cols(); k++)
        {
          ret[i * data.rows() + j] = std::max(ret[i * data.rows() + j], std::abs(data(i, k) - data(j, k)));
        }
        ret[i * data.rows() + j] = std::abs(ret[i * data.rows() + j]);
      }
      ret[i * data.rows() + i] = 0.0;
    }
    // update the distance matrix
    // and the local score matrix

    return ret;
  }

  std::vector<real_t> cluster_stage::minkowski_distances(const matrix<real_t> &data)
  {
    // calculate minkowski distance of the data matrix
    std::vector<real_t> ret(data.rows() * data.rows());
    for (size_t i = 0; i < data.rows(); i++)
    {
      for (size_t j = 0; j < data.rows(); j++)
      {
        // initialize minkowski distance to 0
        ret[i * data.rows() + j] = 0.0;
        for (size_t k = 0; k < data.cols(); k++)
        {
          ret[i * data.rows() + j] += std::pow(std::abs(data(i, k) - data(j, k)), 3);
        }
        ret[i * data.rows() + j] = std::pow(ret[i * data.rows() + j], 1.0 / 3.0);
      }
    }
    return ret;
  }
  std::vector<real_t> cluster_stage::wminkowski_distances(const matrix<real_t> &data)
  {
    std::vector<real_t> ret(data.rows() * data.rows());
    // calculate wminkowski distance of the data matrix
    for (size_t i = 0; i < data.rows(); i++)
    {
      for (size_t j = 0; j < data.rows(); j++)
      {
        // initialize wminkowski distance to 0
        ret[i * data.rows() + j] = 0.0;
        for (size_t k = 0; k < data.cols(); k++)
        {
          ret[i * data.rows() + j] += std::pow(std::abs(data(i, k) - data(j, k)), 3);
        }
        ret[i * data.rows() + j] = std::pow(ret[i * data.rows() + j], 1.0 / 3.0);
      }
    }

    return ret;
  }
  std::vector<real_t> cluster_stage::seuclidean_distances(const matrix<real_t> &data)
  {
    // calculate seuclidean distance of the data matrix
    std::vector<real_t> ret(data.rows() * data.rows());
    for (size_t i = 0; i < data.rows(); i++)
    {
      for (size_t j = 0; j < data.rows(); j++)
      {
        // initialize seuclidean distance to 0
        ret[i * data.rows() + j] = 0.0;
        for (size_t k = 0; k < data.cols(); k++)
        {
          ret[i * data.rows() + j] += std::pow(std::abs(data(i, k) - data(j, k)), 2);
        }
        ret[i * data.rows() + j] = std::sqrt(ret[i * data.rows() + j] / data.cols());
      }
      ret[i * data.rows() + i] = 0.0;
    }
    // return the seuclidean distance matrix

    return ret;
  }
  std::vector<real_t> cluster_stage::hamming_distances(const matrix<real_t> &data)
  {
    // calculate hamming distance of the data matrix
    std::vector<real_t> ret(data.rows() * data.rows());
    for (size_t i = 0; i < data.rows(); i++)
    {
      for (size_t j = 0; j < data.rows(); j++)
      {
        // initialize hamming distance to 0
        ret[i * data.rows() + j] = 0.0;
        for (size_t k = 0; k < data.cols(); k++)
        {
          ret[i * data.rows() + j] += (data(i, k) != data(j, k));
        }
        ret[i * data.rows() + j] /= data.cols();
      }
    }
    return ret;
  }
  std::vector<real_t> cluster_stage::jaccard_distances(const matrix<real_t> &data)
  {
    // calculate jaccard distance of the data matrix
    // the difference between jaccard and hamming distance is that jaccard distance is normalized
    std::vector<real_t> ret(data.rows() * data.rows());
    for (size_t i = 0; i < data.rows(); i++)
    {
      for (size_t j = 0; j < data.rows(); j++)
      {
        // initialize jaccard distance to 0
        ret[i * data.rows() + j] = 0.0;
        for (size_t k = 0; k < data.cols(); k++)
        {
          ret[i * data.rows() + j] += (data(i, k) != data(j, k));
        }
        // normalize jaccard distance
        ret[i * data.rows() + j] /= (data.cols() - ret[i * data.rows() + j]);
      }
    }
    return ret;
  }
  // spearman
  std::vector<real_t> cluster_stage::spearman_distances(const matrix<real_t> &data)
  {

    matrix<real_t> rank_data(data.rows(), data.cols());
    std::vector<size_t> rank(data.cols());

    // initialize rank
    for (size_t i = 0; i < rank.size(); i++)
    {
      rank[i] = rank.size() - i;
    }
    // calculate rank of each column
    for (size_t i = 0; i < data.cols(); i++)
    {
      for (size_t j = 0; j < data.rows(); j++)
      {
        for (size_t k = 0; k < data.rows(); k++)
        {
          if (data(i, j) > data(i, k))
          {
            rank[j]--;
          }
          else if (data(i, j) < data(i, k))
          {
            rank[k]--;
          }
        }
      }
      // sort the rank
      std::sort(rank.begin(), rank.end());
      // assign the rank to the rank_data matrix
      for (size_t j = 0; j < data.rows(); j++)
      {
        rank_data(i, j) = rank[j];
      }
      // reset the rank
      for (size_t j = 0; j < rank.size(); j++)
      {
        rank[j] = rank.size() - j;
      }
    }
    // calculate spearman distance of the rank_data matrix
    return cluster_stage::euclidean_distances(matrix<real_t>(rank_data * rank_data.transpose()) / data.cols());
  }

  // kendall
  std::vector<real_t> cluster_stage::kendall_distances(const matrix<real_t> &data)
  {
    // calculate kendall distance of the data matrix
    std::vector<real_t> ret(data.rows() * data.rows());
    matrix<real_t> rank_data(data.rows(), data.cols());
    std::vector<size_t> rank(data.cols());

    // initialize rank
    for (size_t i = 0; i < rank.size(); i++)
    {
      rank[i] = rank.size() - i;
    }
    // calculate rank of each column
    for (size_t i = 0; i < data.cols(); i++)
    {
      for (size_t j = 0; j < data.rows(); j++)
      {
        for (size_t k = 0; k < data.rows(); k++)
        {
          if (data(j, i) > data(k, i))
          {
            rank[j]--;
          }
          else if (data(j, i) < data(k, i))
          {
            rank[k]--;
          }
        }
      }
      // copy rank to rank_data
      for (size_t j = 0; j < data.rows(); j++)
      {
        rank_data(j, i) = rank[j];
      }
      // reset rank
      for (size_t j = 0; j < rank.size(); j++)
      {
        rank[j] = rank.size() - j;
      }
    }
    // calculate kendall distance of rank_data
    // calculate kendall distance of the data matrix

    for (size_t i = 0; i < data.rows(); i++)
    {
      for (size_t j = 0; j < data.rows(); j++)
      {
        // initialize kendall distance to 0
        ret[i * data.rows() + j] = 0.0;
        for (size_t k = 0; k < data.cols(); k++)
        {
          ret[i * data.rows() + j] += (rank_data(i, k) - rank_data(j, k)) * (rank_data(i, k) - rank_data(j, k));
        }
        // normalize kendall distance
        ret[i * data.rows() + j] /= (2 * data.cols() * (data.cols() - 1));
      }
    } // calculate kendall distance of rank_data
    // return kendall distance:

    return ret;
  }
  // cosine
  std::vector<real_t> cluster_stage::cosine_distances(const matrix<real_t> &data)
  {
    // calculate cosine distance of the data matrix
    std::vector<real_t> ret(data.rows() * data.rows());
    // calculate cosine distance of the data matrix

    for (size_t i = 0; i < data.rows(); i++)
    {
      for (size_t j = 0; j < data.rows(); j++)
      {
        // initialize cosine distance to 0
        ret[i * data.rows() + j] = 0.0;
        for (size_t k = 0; k < data.cols(); k++)
        {
          ret[i * data.rows() + j] += data(i, k) * data(j, k);
        }
        // normalize cosine distance:
        ret[i * data.rows() + j] /= sqrt(data.sum_row(i) * data.sum_row(j));

        // cosine distance is between 0 and 1
        ret[i * data.rows() + j] = 1.0 - ret[i * data.rows() + j];
        // if cosine distance is negative, set it to 0
        if (ret[i * data.rows() + j] < 0.0)
        {
          ret[i * data.rows() + j] = 0.0;
        }
        // if cosine distance is greater than 1, set it to 1
        if (ret[i * data.rows() + j] > 1.0)
        {
          ret[i * data.rows() + j] = 1.0;
        }
        // if cosine distance is nan, set it to 0
        if (ret[i * data.rows() + j] != ret[i * data.rows() + j])
        {
          ret[i * data.rows() + j] = 0.0;
        }
        // if cosine distance is inf/-inf, set it to 0
        if (ret[i * data.rows() + j] == std::numeric_limits<real_t>::infinity() || ret[i * data.rows() + j] == -std::numeric_limits<real_t>::infinity())
        {
          ret[i * data.rows() + j] = 0.0;
        }
        // finish calculating cosine distance

      } // end for j
    }   // end for i

    return ret; // return cosine distance
  }             // end cosine distance
  // correlation
  std::vector<real_t> cluster_stage::correlation_distances(const matrix<real_t> &data)
  {
    // calculate correlation distance of the data matrix
    std::vector<real_t> ret(data.rows() * data.rows());
    // calculate correlation distance of the data matrix
    // calculate mean of each column
    std::vector<real_t> mean(data.cols());
    for (size_t i = 0; i < data.cols(); i++)
    {
      mean[i] = 0.0;
      for (size_t j = 0; j < data.rows(); j++)
      {
        mean[i] += data(j, i);
      }
      mean[i] /= data.rows();
    }
    // calculate standard deviation of each column
    std::vector<real_t> std(data.cols());
    for (size_t i = 0; i < data.cols(); i++)
    {
      std[i] = 0.0;
      for (size_t j = 0; j < data.rows(); j++)
      {
        std[i] += (data(j, i) - mean[i]) * (data(j, i) - mean[i]);
      }
      std[i] /= data.rows();
      std[i] = sqrt(std[i]);
    }
    // calculate correlation distance of the data matrix
    for (size_t i = 0; i < data.rows(); i++)
    {
      for (size_t j = 0; j < data.rows(); j++)
      {
        // initialize correlation distance to 0
        ret[i * data.rows() + j] = 0.0;
        for (size_t k = 0; k < data.cols(); k++)
        {
          ret[i * data.rows() + j] += (data(i, k) - mean[k]) * (data(j, k) - mean[k]);
        }
        // normalize correlation distance
        ret[i * data.rows() + j] /= (std[i] * std[j]);
        // correlation distance is between -1 and 1
        ret[i * data.rows() + j] = 1.0 - ret[i * data.rows() + j];
        // if correlation distance is negative, set it to 0
        if (ret[i * data.rows() + j] < 0.0)
        {
          ret[i * data.rows() + j] = 0.0;
        }

      } // end for j
    }   // end for i
    // return correlation distance
    return ret;
    // return correlation_distances(data, mean, std);
  }
  // manhattan
  std::vector<real_t> cluster_stage::manhattan_distances(const matrix<real_t> &data)
  {
    // calculate manhattan distance of the data matrix
    std::vector<real_t> ret(data.rows() * data.rows());
    // calculate manhattan distance of the data matrix
    for (size_t i = 0; i < data.rows(); i++)
    {
      for (size_t j = 0; j < data.rows(); j++)
      {
        // initialize manhattan distance to 0
        ret[i * data.rows() + j] = 0.0;
        for (size_t k = 0; k < data.cols(); k++)
        {
          ret[i * data.rows() + j] += fabs(data(i, k) - data(j, k));
        }
      } // end for j
    }   // end for i
    // return manhattan distance
    return ret;
  }
  // euclidean
  std::vector<real_t> cluster_stage::euclidean_distances(const matrix<real_t> &data)
  {
    // calculate euclidean distance of the data matrix
    std::vector<real_t> ret(data.rows() * data.rows());
    // calculate euclidean distance of the data matrix

    for (size_t i = 0; i < data.rows(); i++)
    {
      for (size_t j = 0; j < data.rows(); j++)
      {
        // initialize euclidean distance to 0
        ret[i * data.rows() + j] = 0.0;
        for (size_t k = 0; k < data.cols(); k++)
        {
          ret[i * data.rows() + j] += (data(i, k) - data(j, k)) * (data(i, k) - data(j, k));
        }
        // euclidean distance is between 0 and 1
        ret[i * data.rows() + j] = sqrt(ret[i * data.rows() + j]);
      } // end for j
    }   // end for i

    // return euclidean distance
    return ret;
  }
  // mahalanobis
  std::vector<real_t> cluster_stage::mahalanobis_distances(const matrix<real_t> &data)
  {
    // calculate mahalanobis distance of the data matrix
    std::vector<real_t> ret(data.rows() * data.rows());
    // calculate mahalanobis distance of the data matrix
    // calculate mean of each column
    std::vector<real_t> std(data.cols());
    std::vector<real_t> mean(data.cols());
    for (size_t i = 0; i < data.cols(); i++)
    {
      mean[i] = 0.0;
      for (size_t j = 0; j < data.rows(); j++)
      {
        mean[i] += data(j, i);
      }
      mean[i] /= data.rows();
    }
    // calculate standard deviation of each column

    for (size_t i = 0; i < data.cols(); i++)
    {
      std[i] = 0.0;
      for (size_t j = 0; j < data.rows(); j++)
      {
        std[i] += (data(j, i) - mean[i]) * (data(j, i) - mean[i]);
      }
      std[i] /= data.rows();
      std[i] = sqrt(std[i]);
    }
    // calculate mahalanobis distance of the data matrix
    for (size_t i = 0; i < data.rows(); i++)
    {
      for (size_t j = 0; j < data.rows(); j++)
      {
        // initialize mahalanobis distance to 0
        ret[i * data.rows() + j] = 0.0;
        for (size_t k = 0; k < data.cols(); k++)
        {
          ret[i * data.rows() + j] += (data(i, k) - mean[k]) * (data(j, k) - mean[k]);
        }
        // normalize mahalanobis distance
        ret[i * data.rows() + j] /= (std[i] * std[j]);
        // mahalanobis distance is between 0 and 1
        ret[i * data.rows() + j] = sqrt(ret[i * data.rows() + j]);
      } // end for j
    }   // end for i
    // return mahalanobis distance
    return ret;
  }

  // matching

  // dice

  // jaccard

  // calculate distance matrix
  std::vector<real_t> cluster_stage::calculate_distance_matrix(const matrix<real_t> &data, const std::string &m)
  {
    if (metrics.find(m) != metrics.end())
    {
      // calculate distance matrix return this reference to the function

      return (this->*metrics[m])(data);
    }
    else
    {
      // calculate distance matrix
      return euclidean_distances(data);
    }
    // calculate distance matrix

  } // end calculate_distance_matrix
  //transfer learning stage implemnenation

  //evaluation metrics 
  //transfer_accuracy_f1 implementation :
real_t knowledge_transfer_stage::transfer_accuracy_f1(const std::vector<real_t>& tp,
const std::vector<real_t>& fp, const std::vector<real_t>& fn, const std::vector<real_t>& tn)
{


  //calculate accuracy
  real_t accuracy = 0.0;
  for (size_t i = 0; i < tp.size(); i++)
  {
    accuracy += (tp[i] + tn[i]) / (tp[i] + tn[i] + fp[i] + fn[i]);
  }
  accuracy /= tp.size();
  //calculate f1
  real_t f1 = 0.0;
  for (size_t i = 0; i < tp.size(); i++)
  {
    f1 += (2 * tp[i]) / (2 * tp[i] + fp[i] + fn[i]);
  }
  f1 /= tp.size();
  //return accuracy and f1
  return (accuracy + f1) / 2.0;   
}
//transfer_accuracy_accuracy implementation :
real_t knowledge_transfer_stage::transfer_accuracy_accuracy(const std::vector<real_t>& tp,  
const std::vector<real_t>& fp, const std::vector<real_t>& fn, const std::vector<real_t>& tn)
{
  //calculate accuracy
  real_t accuracy = 0.0;
  for (size_t i = 0; i < tp.size(); i++)
  {
    accuracy += (tp[i] + tn[i]) / (tp[i] + tn[i] + fp[i] + fn[i]);
  }
  accuracy /= tp.size();
  //return accuracy
  return accuracy;      
}
//transfer_accuracy_precision
real_t knowledge_transfer_stage::transfer_accuracy_precision(const std::vector<real_t>& tp,
const std::vector<real_t>& fp, const std::vector<real_t>& fn, const std::vector<real_t>& tn)
{
  UNDEF_REFERENCE(fn);
  UNDEF_REFERENCE2(tn);
  //calculate precision
  real_t precision = 0.0;
  for (size_t i = 0; i < tp.size(); i++)
  {
    precision += (tp[i]) / (tp[i] + fp[i]);
  }
  precision /= tp.size();
  //return precision
  return precision;      
}
//transfer_accuracy_recall
real_t knowledge_transfer_stage::transfer_accuracy_recall(const std::vector<real_t>& tp,
const std::vector<real_t>& fp, const std::vector<real_t>& fn, const std::vector<real_t>& tn)
{
  UNDEF_REFERENCE(fp);
  UNDEF_REFERENCE2(tn);

  //calculate recall
  real_t recall = 0.0;
  for (size_t i = 0; i < tp.size(); i++)
  {
    recall += (tp[i]) / (tp[i] + fn[i]);
  }
  recall /= tp.size();
  //return recall
  return recall;      
}
//transfer_accuracy_auc
real_t knowledge_transfer_stage::transfer_accuracy_roc_auc(const std::vector<real_t>& tp,
const std::vector<real_t>& fp, const std::vector<real_t>& fn, const std::vector<real_t>& tn)
{
  UNDEF_REFERENCE(tn);
  UNDEF_REFERENCE2(fp);

  //calculate auc
  real_t auc = 0.0;
  for (size_t i = 0; i < tp.size(); i++)
  {
    auc += (tp[i]) / (tp[i] + fn[i]);
  }
  auc /= tp.size();
  //return auc
  return auc;

}

//transfer_accuracy_pr_auc
real_t knowledge_transfer_stage::transfer_accuracy_pr_auc(const std::vector<real_t>& tp,
                                                          const std::vector<real_t> &fp, 
                                                          const std::vector<real_t> &fn,
                                                          const std::vector<real_t> &tn)
{
  //calculate pr auc
  //undef what is not needed
  UNDEF_REFERENCE(tn);
  UNDEF_REFERENCE2(fp);

  real_t pr_auc = 0.0;
  for (size_t i = 0; i < tp.size(); i++)
  {
    pr_auc += (tp[i]) / (tp[i] + fn[i]);
  }
  pr_auc /= tp.size();
  //return pr auc
  return pr_auc;

}
real_t  knowledge_transfer_stage::transfer_accuracy_kappa(const std::vector<real_t>& tp,
const std::vector<real_t>& fp, const std::vector<real_t>& fn, const std::vector<real_t>& tn
)
{
 //calculate kappa
  real_t kappa = 0.0;
  for (size_t i = 0; i < tp.size(); i++)
  {
    kappa += (tp[i] + tn[i]) / (tp[i] + tn[i] + fp[i] + fn[i]);
  }
  kappa /= tp.size();
  //return kappa
  return kappa;

}
real_t knowledge_transfer_stage::transfer_accuracy_mcc(const std::vector<real_t>& tp,
                                                       const std::vector<real_t> &fp, 
                                                        const std::vector<real_t> &fn,
                                                        const std::vector<real_t> &tn)
{ 
  //calculate mcc
  real_t mcc = 0.0;
  for (size_t i = 0; i < tp.size(); i++)
  {
    mcc += (tp[i] + tn[i]) / (tp[i] + tn[i] + fp[i] + fn[i]);
  }
  mcc /= tp.size();
  //return mcc
  return mcc;

} 
real_t knowledge_transfer_stage::transfer_accuracy_informedness(const std::vector<real_t>& tp,
const std::vector<real_t>& fp, const std::vector<real_t>& fn, const std::vector<real_t>& tn)
{
  //calculate informedness
  real_t informedness = 0.0;
  for (size_t i = 0; i < tp.size(); i++)
  {
    informedness += (tp[i] + tn[i]) / (tp[i] + tn[i] + fp[i] + fn[i]);
  }
  informedness /= tp.size();
  //return informedness
  return informedness;    
}






} // namespace provallo
