/*
 * pipelinebuilder.h
 *
 *  Created on: Jun 19, 2023
 *      Author: kardon
 */

#include "pipelinebuilder.h"
#include "utils.h"
#include "../util/csv_file.h" //load csv into matrix_base*

#include "matrix.h"
#include "classifier.h"
#include <iostream>

#include <algorithm>

namespace provallo
{

  // bag of words implementation:
  bag_of_words::bag_of_words() : _vocabulary(), _bow()
  {
      // std::cout<<"bow size: "<<_bow.size()<<std::endl;
      //  
  }

  std::vector<real_t> bag_of_words::get_bag_of_words() const
  {
    return _bow;
  }
  // constructor from vocabulary
  bag_of_words::bag_of_words(const std::vector<std::string> &vocabulary) : _vocabulary(vocabulary), _bow(vocabulary.size(), 0.0)
  {

    // std::cout<<"vocabulary size: "<<_vocabulary.size()<<std::endl;
    // set bag of words values on the tokens
   
    size_t token_count = 0;
    size_t vocabulary_size = _vocabulary.size();
    size_t number_of_documents = vocabulary_size;
    size_t feature_count = 0;
    size_t sample_count = vocabulary_size;
    size_t sample_size = 1;
 

    for (size_t i = 0; i < vocabulary_size; ++i)
    {
      std::string token = _vocabulary[i];
        for (size_t j = i + 1; j < _vocabulary.size(); ++j)
      {
        //calculate the number of times the token appears in the vocabulary 
        // and update the bow vector
        //
        if (_vocabulary[j] == token)
        {
          _bow[i] += 1.0;
          _bow[j] = 0.0;
         //set token index to zero
          token_count++;

        }
        else
        {

          _bow[j] =   1.0 / real_t(_vocabulary.size()) ;


 

        }
   
      }
      for (size_t j = 0; j < i; ++j)
      {
        //calculate the number of times the token appears in the vocabulary
        // and update the bow vector

        if (_vocabulary[j] == token)
        {
          _bow[i] += 1.0;
          _bow[j] = 0.0;
        }
        else
        {
          _bow[j] = 1.0 / real_t(_vocabulary.size()) ;
        }


      }


      _bow[i] = _bow[i] / _vocabulary.size();
      //update samples,tokens,features counters 
      feature_count++;
      sample_count++;
      sample_size++;
         
      // std::cout<<"token: "<<token<<std::endl;
      // std::cout<<"token count: "<<token_count<<std::endl;
      // std::cout<<"vocabulary size: "<<vocabulary_size<<std::endl;
      // std::cout<<"feature count: "<<feature_count<<std::endl;
      // std::cout<<"sample count: "<<sample_count<<std::endl;

    }
    // update the bow matrix
    // matrix is the number of components x samples
    // 
  
    _bow_matrix = matrix<real_t>(_vocabulary.size(), 1);
    for (size_t i = 0; i < _vocabulary.size(); ++i)
    {
      // std::cout<<"bow size: "<<_bow.size()<<std::endl; 

      _bow_matrix(i, 0) = _bow[i];
    }
    this->num_tokens = token_count;
    this->num_features = feature_count;
    this->num_words = vocabulary_size;
    this->num_docs = number_of_documents;
    this->num_samples = sample_size;

    
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
  //get vocabulary 
  const std::vector<std::string>& bag_of_words::get_vocabulary() const
  {
    return _vocabulary;
  }
  //get_number_of_documents
  size_t bag_of_words::get_number_of_documents() const
  {
    return _bow_matrix.size1()*_bow_matrix.size2() ;
  }
  //get_number_of_words 
  size_t bag_of_words::get_number_of_words() const
  {
    return _vocabulary.size();
  }
  //get_number_of_tokens
  size_t bag_of_words::get_number_of_tokens() const
  {
    return _bow.size();
  }
  size_t bag_of_words::get_number_of_unique_tokens() const
  {
    return _vocabulary.size();
  } 
  
  //clear
  void bag_of_words::clear()
  {
    _vocabulary.clear();
    _bow.clear();
    _bow_matrix.clear();
    _bow_transformed.clear();
    _bow_transformed_inverse.clear();

    _vocabulary.shrink_to_fit();
    _bow.shrink_to_fit();
    _bow_transformed.shrink_to_fit();
    _bow_transformed_inverse.shrink_to_fit();
  }
  //process the documents
  //  virtual void add_document(const std::string&);
  void bag_of_words::add_document(const std::string & doc)  
  {
    // tokenize the document
    std::vector<std::string> tokens;
    tokenize(doc, tokens);
    size_t token_count = 0;
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
        token_count++;

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
    this->num_features = _vocabulary.size()-1;
    this->num_samples = 1;
    // std::cout<<"bow size: "<<_bow.size()<<std::endl;
    this->num_words = _bow.size();
    this->num_tokens+=token_count;
    // std::cout<<"bow matrix: "<<_bow_matrix<<std::endl;
    return ;
  }
  //transform a doc to a vec
  void bag_of_words::process_documents() 
  {
    //add_document
    //already processed.
    
    return ;

  }
  //  virtual void process_document(const std::string&);
  void bag_of_words::process_document(const std::string & doc) 
  {
    //same as add_document
    add_document(doc);
    this->num_docs++;

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
          token = reduce(token,"");
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
          token = reduce(token,"");
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
    this->num_features = _vocabulary.size()-1;
    this->num_samples = num_docs;
    this->num_words = _bow.size();
    this->num_tokens = _vocabulary.size();

    // std::cout<<"bow size: "<<_bow.size()<<std::endl;
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
    this->num_features = _vocabulary.size()-1;
    this->num_samples = 1;
    // std::cout<<"bow size: "<<_bow.size()<<std::endl;
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
      token = reduce(token,"");
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
        real_t val = 1/real_t(_vocabulary.size()-1);
        _vocabulary.push_back(token);
        _bow.push_back(val);
         result.push_back(val);

      }
    }
    this->num_features = _vocabulary.size()-1;
    this->num_samples += 1;
    //update unique tokens
    this->num_tokens = _vocabulary.size();
    // std::cout<<"bow size: "<<_bow.size()<<std::endl;
    //transform results to lda
    
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

    this->num_words = _vocabulary.size(); 
    this->num_docs = documents.size();
    this->num_samples = documents.size();
    this->num_features = _vocabulary.size()-1;
    this->num_tokens = _bow.size();
    
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
    std::vector<std::vector<real_t>> result(documents.size());
    for(auto& doc : documents)
    {
      result.push_back(transform(doc));
    }
    return result;
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
  // bag_of_words dump
  void bag_of_words::dump(std::ostream &out) const
  {
    out << "vocabulary: " << std::endl;
    for (auto &word : _vocabulary)
    {
      out << word << std::endl;
    }
    out << "bow: " << std::endl;
    for (auto &b : _bow)
    {
      out << b << std::endl;
    }
    out << "bow matrix: " << std::endl;
    out << _bow_matrix << std::endl;
    out << "bow transformed: " << std::endl;
    out << _bow_transformed << std::endl;
    out << "bow transformed inverse: " << std::endl;
    out << _bow_transformed_inverse << std::endl;
  }
  
  // bag_of_words load
   void bag_of_words::load(std::ifstream& is)
   {
      //load vocabulary
      std::string line;
      while(std::getline(is,line))
      {
        if (line.empty()||line=="\n"||line=="\r"||line=="\r\n")
          break;
        _vocabulary.push_back(line);
      }
      //load bow
      while(std::getline(is,line))
      {
        _bow.push_back(std::stod(line));
      }
      //load bow matrix
      is>>_bow_matrix;
      //load bow transformed


      //load the number of vectors in the bow transformed 
      size_t num_vectors;
      is>>num_vectors;

      //load each vector in the bow transformed
      for(size_t i=0;i<num_vectors;++i)
      {
        std::vector<real_t> vec;
        for(size_t j=0;j<_vocabulary.size();++j)
        {
          real_t value;
          is>>value;
          vec.push_back(value);
        }
        _bow_transformed.push_back(vec);
      } 
      //load bow transformed inverse

      is>>num_vectors;
      for (size_t i=0;i<num_vectors ;++i)
      {
        std::vector<real_t> vec;
        for(size_t j=0;j<_vocabulary.size();++j)
        {
          real_t value;
          is>>value;
          vec.push_back(value);
        }
        _bow_transformed_inverse.push_back(vec);
      }

      //done
   }
  //explicit save
   void bag_of_words::save(std::ofstream& os) const
   {
      //save vocabulary
      for(auto& word:_vocabulary)
      {
        os<<word<<std::endl;
      } 
      //save bow
      for(auto& b:_bow)
      {
        os<<b<<std::endl;
      }
      //save bow matrix
      os<<_bow_matrix<<std::endl;
      //save bow transformed
      os<<_bow_transformed.size()<<std::endl;
      for(auto& vec:_bow_transformed)
      {
        for(auto& value:vec)
        {
          os<<value<<" ";
        }
        os<<std::endl;
      }
      //save bow transformed inverse
      os<<_bow_transformed_inverse.size()<<std::endl;
      for(auto& vec:_bow_transformed_inverse)
      {
        for(auto& value:vec)
        {
          os<<value<<" ";
        }
        os<<std::endl;
      }
      //done
 
  }

  
  
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
  //tfidf::clear
  void tfidf::clear()
  {
    _documents.clear();
    _vocabulary.clear();
    _tf.clear();
    _idf.clear();
    _tfidf.clear();
  }
  // tfidf implementation:
  void tfidf::process_documents()
  {
    if( _documents.size() == 0)
      return;

    
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
     
    // clear data
    _documents.clear();
    // done processing

    
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
    std::vector<std::string> words;
    tokenize(doc, words, " ,;.:-_()[]{}!?\"\'\n\t");
    
    std::vector<real_t> result(words.size(),0.0);

    size_t v_index=0;
    for (auto &word : _vocabulary)
    {
  
      size_t res_index=0;

      for (auto &w : words)
      {

        if (w == word)
        {
          result[res_index ]+= 1.0;
          result[res_index ]*= _idf[v_index];

        }
        else
        {
          result[res_index%words.size()]+= 0.0;
        }


        res_index++;
      }
      v_index++;
    }
    
    // normalize tfidf
    real_t sum = 0.0;
    for (auto &value : result)
    {
      sum += value;
    } 
    for (auto &value : result)
    {
      value /= sum;
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


    //update fitted_data and transformed_data of parent class
    _fitted_data = _tfidf.get_idf();
    _transformed_data = _tfidf.get_tf();

    return _tfidf.get_tf();
  }
  // tfidf_vectorizer::fit_transform

  std::vector<real_t>
  tfidf_vectorizer::fit_transform(const std::vector<std::string> &corpus)
  {
    std::vector<real_t> ret = fit(corpus);
    // transform :
     

    return transform(ret);
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
      result.reserve( predict.size()); 
      size_t res_index=0;
      for (auto &res : predict)
      {
        result[res_index%result.size()]+= res;
        res_index++;
      }
    }
    return result;
  }

  // tfidf_vectorizer::tfidf_vectorizer

  tfidf_vectorizer::tfidf_vectorizer(const std::vector<std::string> &corpus) : vectorizer(TFIDF)
  {
    // fit :   
    for (auto &doc : corpus)
    {
      _tfidf.add_document(doc);
    }

    _tfidf.process_documents();
    //update fitted_data and transformed_data of parent class
    _fitted_data = _tfidf.get_idf();
    _transformed_data = _tfidf.get_tf();

      
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
    if(_tfidf.get_vocabulary().size() == 0)
    _tfidf.process_documents();

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
        _fitted_data[i*data.size2()+j] = (data(i, j) - _mean.at(j)) / _variance.at(j);
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

  std::vector<std::vector<real_t>> pca_vectorizer::fit( const std::vector<std::vector<std::string>>& documents  )
  {
      std::vector<std::vector<real_t>> ret(documents.size()); 
      for (size_t i = 0; i < documents.size(); ++i)
      {
        ret[i] = fit(documents[i]);
      }

      return ret; 
      
  }

  std::vector<real_t> pca_vectorizer::predict(const std::string& doc) 
  {

    return predict(std::vector<std::string>{doc});
  }

  std::vector<std::vector<real_t>>  pca_vectorizer::predict(const std::vector<std::vector<std::string>>& documents)
  {
    std::vector<std::vector<real_t>> ret;
    
    for (size_t i = 0; i < documents.size(); ++i)
    {
      ret[i] = predict(documents[i]);
    }
    
    return ret;

  }
  std::vector<std::vector<real_t>>  pca_vectorizer::transform(const std::vector<std::vector<std::string>>& documents)
  {
    std::vector<std::vector<real_t>> ret;
    
    for (size_t i = 0; i < documents.size(); ++i)
    {
      ret[i] = transform(documents[i]);
    }
    
    return ret;

  }


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

  void pipeline_builder::add_pipeline(pipeline *pipeline)
  {
    _pipelines.push_back(pipeline);
    _current_pipeline = pipeline;
  }
  //get number of pipelines
  uint64_t pipeline_builder::get_number_of_pipelines() const
  {
    return _pipelines.size();
  }
  //get current pipeline
  pipeline *pipeline_builder::get_current_pipeline()  
  {
    return _current_pipeline;
  }
  const pipeline *pipeline_builder::get_current_pipeline() const
  {
    return _current_pipeline;
  }

  //get pipeline by index
  pipeline *pipeline_builder::get_pipeline(uint64_t index)  
  {
    auto it = std::find_if(_pipelines.begin(), _pipelines.end(), [index](pipeline *p)
                           { return p->get_pipeline_id() == index; });
    if (it != _pipelines.end())
    {
      return *it;
    }
    return nullptr;
  }
  //get pipeline by name
  pipeline *pipeline_builder::get_pipeline(const std::string &pipeline_name)  
  {
    auto it = std::find_if(_pipelines.begin(), _pipelines.end(), [pipeline_name](pipeline *p)
                           { return p->get_pipeline_name() == pipeline_name; });
    if (it != _pipelines.end())
    {
      return *it;
    }
    return nullptr;
  }
  
  //get number of stages in current pipeline
  uint64_t pipeline_builder::get_number_of_stages() const
  {
    if (_current_pipeline)
    {
      return _current_pipeline->get_number_of_stages();
    }
    return 0;
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
    //update n_samples, n_features and n_components 
    _pca_n_samples_++;
    _pca_n_features_ = _vocabulary.size();

    _pca_n_components_ =size_t( _vocabulary.size()/real_t(_pca_n_samples_)  );

    //update pca 
    this->update_pca ( results);
    //update results with eigen values
    results.resize(_pca_n_features_+1);
    for (size_t i=0; i<_pca_n_features_; i++)
    {
      results[i] = _eigen_values[i];
    }
    results[_pca_n_features_] = _pca_n_components_;
    //return results
     return results;
  }
    // create a vector of real_t to store the results
  void principal_component_analysis::update_pca(const std::vector<real_t>& occurance_data)
  {
      //resize to accomodate the new data

      _pca_data.resize(_pca_n_samples_*_pca_n_features_);


      //update _pca_component_matrix and _pca_explained_variance_matrix
      _pca_components_matrix.resize(_pca_n_features_,_pca_n_features_);
      _pca_explained_variance_matrix.resize(_pca_n_features_,_pca_n_samples_ );
      
      //update the pca data

      //add the new data to the pca data
      for (size_t i=0; i<occurance_data.size(); i++)
      {
        _pca_data[i] = occurance_data[i];
      }
      //update the mean
      _pca_mean.resize(_pca_n_features_);
      for (size_t i=0; i<_pca_n_features_; i++)
      {
        _pca_mean[i] = _pca_data[i]; 
        for (size_t j=1; j<_pca_n_samples_; j++)
        {
          _pca_mean[i] += _pca_data[j*_pca_n_features_+i];
        }
        _pca_mean[i] /= _pca_n_samples_;

      }
      //update the standardized data
      _standardized_data.resize(_pca_n_samples_*_pca_n_features_);
      for (size_t i=0; i<_pca_n_samples_; i++)
      {
        for (size_t j=0; j<_pca_n_features_; j++)
        {
          _standardized_data[i*_pca_n_features_+j] = (_pca_data[i*_pca_n_features_+j] - _pca_mean[j]) / _pca_n_samples_;
        }
      }
      //update the covariance matrix
      _covariance_matrix.resize(_pca_n_features_*_pca_n_features_);
      for (size_t i=0; i<_pca_n_features_; i++)
      {
        for (size_t j=0; j<_pca_n_features_; j++)
        {
          _covariance_matrix[i*_pca_n_features_+j] = 0.0;
          for (size_t k=0; k<_pca_n_samples_; k++)
          {
            _covariance_matrix[i*_pca_n_features_+j] += _standardized_data[k*_pca_n_features_+i] * _standardized_data[k*_pca_n_features_+j];
          }
          _covariance_matrix[i*_pca_n_features_+j] /= _pca_n_samples_ - 1;
        }
      } 
      //update the eigen values and eigen vectors
      _eigen_values.resize(_pca_n_features_);
      _eigen_vectors.resize(_pca_n_features_*_pca_n_features_);
      _pca_components.resize(_pca_n_features_*_pca_n_features_);
      _pca_explained_variance.resize(_pca_n_features_);
      _pca_explained_variance_ratio.resize(_pca_n_features_);
      _pca_singular_values.resize(_pca_n_features_);
      _pca_noise_variance.resize(_pca_n_features_);
      _pca_components_matrix.resize(_pca_n_features_,_pca_n_features_);
      _pca_explained_variance_matrix.resize(_pca_n_features_,_pca_n_samples_ ); 
      _pca_explained_variance_ratio_matrix.resize(_pca_n_features_,_pca_n_samples_ );
      _pca_singular_values_matrix.resize(_pca_n_features_,  _pca_n_samples_ );

      //initialize the eigen vectors

      for (size_t i=0; i<_pca_n_features_; i++)
      {
        for (size_t j=0; j<_pca_n_features_; j++)
        {
          _eigen_vectors[i*_pca_n_features_+j] = 0.0;
        }
        _eigen_vectors[i*_pca_n_features_+i] = 1.0;

        //initialize the eigen values
        _eigen_values[i] = _covariance_matrix[i*_pca_n_features_+i];

      } 

      //update _pca_components_matrix and _pca_explained_variance_matrix
      _pca_components_matrix.resize(_pca_n_features_,_pca_n_features_);
      for(size_t i=0;i<_pca_n_features_;i++)
      {
        for(size_t j=0;j<_pca_n_features_;j++)
        {
          _pca_components_matrix(i,j) = _eigen_vectors[i*_pca_n_features_+j];
        } 
      }
       //update the pca components
      for (size_t i=0; i<_pca_n_features_; i++)
      {
        for (size_t j=0; j<_pca_n_features_; j++)
        {
          _pca_components[i*_pca_n_features_+j] = _eigen_vectors[i*_pca_n_features_+j];
        }
      }
       //update the feat x samples matrices: 
      for (size_t i=0; i<_pca_n_features_; i++)
      {
        for (size_t j=0; j<_pca_n_samples_; j++)
        {
          _pca_explained_variance_matrix(i,j) = _pca_explained_variance[i*_pca_n_samples_+j];
          _pca_explained_variance_ratio_matrix(i,j) = _pca_explained_variance_ratio[i*_pca_n_samples_+j];
          _pca_singular_values_matrix(i,j) = _pca_singular_values[i*_pca_n_samples_+j];
        }
      }   
      //update pca noise variance
      _pca_noise_variance.resize(_pca_n_features_);
      for (size_t i=0; i<_pca_n_features_; i++)
      {
        _pca_noise_variance[i] = 0.0;
        for (size_t j=0; j<_pca_n_features_; j++)
        {
          _pca_noise_variance[i] += _covariance_matrix[i*_pca_n_features_+j];
        }
        _pca_noise_variance[i] /= _pca_n_features_;
        _pca_noise_variance[i] -= _eigen_values[i];
        _pca_explained_variance[i] = _eigen_values[i];
        _pca_explained_variance_ratio[i] = _eigen_values[i] / _pca_n_features_;
        _pca_singular_values[i] = std::sqrt(_eigen_values[i]);

      }  

      //update the pca data
      //use QRDecomposition to update the pca data
      matrix<real_t> Q;
      matrix<real_t> R;
      QRDecomposition(_pca_components_matrix, Q, R);
      //update the pca data
      _pca_data.resize(_pca_n_samples_*_pca_n_features_);
      for (size_t i=0; i<_pca_n_samples_; i++)
      {
        for (size_t j=0; j<_pca_n_features_; j++)
        {
          _pca_data[i*_pca_n_features_+j] = 0.0;
          for (size_t k=0; k<_pca_n_features_; k++)
          {
            _pca_data[i*_pca_n_features_+j] += _standardized_data[i*_pca_n_features_+k] * _pca_components[k*_pca_n_features_+j];
          }
        }
      } 



       //update the pca components

      for (size_t i=0; i<_pca_n_features_; i++)
      {
        for (size_t j=0; j<_pca_n_features_; j++)
        {
          _pca_components[i*_pca_n_features_+j] = 0.0;
          for (size_t k=0; k<_pca_n_features_; k++)
          {
            _pca_components[i*_pca_n_features_+j] += _pca_components_matrix(i,k) * R(k,j);
          }
        }
      } 

      //update the feat x samples matrices:
      for (size_t i=0; i<_pca_n_features_; i++)
      {
        for (size_t j=0; j<_pca_n_samples_; j++)
        {
          _pca_explained_variance_matrix(i,j) = 0.0;
          _pca_explained_variance_ratio_matrix(i,j) = 0.0;
          _pca_singular_values_matrix(i,j) = 0.0;
          for (size_t k=0; k<_pca_n_features_; k++)
          {
            _pca_explained_variance_matrix(i,j) += _pca_explained_variance_matrix(i,k) * R(k,j);
            _pca_explained_variance_ratio_matrix(i,j) += _pca_explained_variance_ratio_matrix(i,k) * R(k,j);
            _pca_singular_values_matrix(i,j) += _pca_singular_values_matrix(i,k) * R(k,j);
          }
        }
      }
      //update pca noise variance
      _pca_noise_variance.resize(_pca_n_features_);
      for (size_t i=0; i<_pca_n_features_; i++)
      {
        _pca_noise_variance[i] = 0.0;
        for (size_t j=0; j<_pca_n_features_; j++)
        {
          _pca_noise_variance[i] += _covariance_matrix[i*_pca_n_features_+j];
        }
        _pca_noise_variance[i] /= _pca_n_features_;
        _pca_noise_variance[i] -= _eigen_values[i];
        _pca_explained_variance[i] = _eigen_values[i];
        _pca_explained_variance_ratio[i] = _eigen_values[i] / _pca_n_features_;
        _pca_singular_values[i] = std::sqrt(_eigen_values[i]);

      }   

      
  }
  
  std::vector<real_t> principal_component_analysis::transform ( const std::string& doc) 
  {
    // tokenize the document into words
 
    //
    std::vector<std::string> words ;
      tokenize(doc,words, " ");
    // create a vector of real_t to store the results
    std::vector<real_t> results(words.size());
    // for each word in the vocabulary
    size_t wordindex=0;
    for (auto &word : words)
    {
      // add the word to the results
      if (_vocabulary.find(word) != _vocabulary.end() ) 
      results [wordindex] = _vocabulary[word]; 
      else
      results [wordindex] = 0.0;
      wordindex++;
    }
    //return the results
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
    
    
    //update fitted data
    _pca_data.resize(rows*cols);
    for (size_t i=0; i<rows; i++)
    {
      for (size_t j=0; j<cols; j++)
      {
        _pca_data[i*cols+j] = data_(i,j);
      }
    }
    //update the mean
    _pca_mean.resize(cols);
    for (size_t i=0; i<cols; i++)
    {
      _pca_mean[i] = mean[i];
    }
    //update the standardized data
    _standardized_data.resize(rows*cols);
    for (size_t i=0; i<rows; i++)
    {
      for (size_t j=0; j<cols; j++)
      {
        _standardized_data[i*cols+j] = centered_data(i,j);
      }
    }
    //update the covariance matrix
    _covariance_matrix.resize(cols*cols);
    for (size_t i=0; i<cols; i++)
    {
      for (size_t j=0; j<cols; j++)
      {
        _covariance_matrix[i*cols+j] = covariance_matrix(i,j);
      }
    }    
    
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
    
    //resize component matrix if not at the right size
    if (this->_pca_components_matrix.rows() != cols || this->_pca_components_matrix.cols() != cols)
    {
      this->_pca_components_matrix = matrix<real_t>(cols, cols);
      //fill with components
      bool b =  _pca_components.size() >= cols*cols ;
      for (size_t i = 0; i < cols; ++i)
      {
        for (size_t j = 0; j < cols; ++j)
        {
          if (b){
                this->_pca_components_matrix(i, j) = this->_pca_components[i * cols + j];
          }
          else
          {
            this->_pca_components_matrix(i, j) = 0.0;
          } 

        }
      }
    }//end of resize component matrix


 
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
    std::vector<std::vector<real_t>> ret ;
    std::vector<real_t> ret_value;

    for (auto & doc : documents )
      ret.push_back( _pca.predict(doc));

    //flatten the vector
    for (size_t i = 0; i < ret.size(); ++i)
    {
      for (size_t j = 0; j < ret[i].size(); ++j)
      {
        ret_value.push_back(std::round(ret[i][j]));
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
  
  
  
  
  void pca_vectorizer::gnuplot(const std::string& filename)
  {
    //use pca helper to gnuplot: 
    _pca.gnuplot(filename);

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
      // use _fitted_data,mean and std to transform the data
    // get the number of rows and columns
    size_t rows = data_.rows();
    size_t cols = data_.cols();
    matrix<real_t> _std = data_.std();
    // initialize the transformed data
    matrix<real_t> transformed_data(rows, cols);
    std::vector<real_t> _mean;
    // transform the data
    for (size_t i = 0; i < rows; ++i)
    {
      _mean.push_back(data_.mean());
      for (size_t j = 0; j < cols; ++j)
      {
        transformed_data(i, j) = (data_(i, j) - _mean[j]) / _std(i, j);
      }
    } 
    // return the transformed data
    return std::vector<real_t>(transformed_data.begin(), transformed_data.end());

  }

  std::vector<real_t> standard_scaler_vectorizer::transform(const provallo::matrix<real_t> &data_matrix)
  {
    // use _fitted_data,mean and std to transform the data
    // get the number of rows and columns
    size_t rows = data_matrix.rows();
    size_t cols = data_matrix.cols();
    matrix<real_t> fitted_data(_fitted_data.size(),1);
    //copy the data 
    std::copy(fitted_data.begin(), fitted_data.end(), _fitted_data.begin()); 
    // initialize the transformed data
    //get mean, std, covariance matrix 
    matrix<real_t> _std = data_matrix.std();
    matrix<real_t> _cov = data_matrix.covariance();
    real_t _mean = data_matrix.mean();

    matrix<real_t> transformed_data(rows, cols);
    // transform the data
    for (size_t i = 0; i < rows; ++i)
    {
      for (size_t j = 0; j < cols; ++j)
      {
        transformed_data(i, j) = (data_matrix(i, j) - _mean) / _std(i, j);
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
    matrix<real_t> fitted(_fitted_data.size(),1);
    //copy the data
    std::copy(fitted.begin(), fitted.end(), _fitted_data.begin());

    // transform the data
    matrix<real_t> _std = fitted.std();
    real_t _mean = fitted.mean();
    for (size_t i = 0; i < rows; ++i)
    {
      transformed_data(i, 0) = (std::stod(documents[i]) - _mean) / _std(0, 0);
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
    matrix<real_t> fitted(_fitted_data.size(),1);
    //copy the data
    std::copy(fitted.begin(), fitted.end(), _fitted_data.begin());

    
     matrix<real_t> transformed_data(rows, 1);
    // transform the data
    matrix<real_t> _std = fitted.std();
    real_t _mean = fitted.mean();
    for (size_t i = 0; i < rows; ++i)
    {
      transformed_data(i, 0) = (std::stod(documents[i]) - _mean) / _std(0, 0);
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
    matrix<real_t> fitted ( _fitted_data.size(),1);
    std::copy(fitted.begin(), fitted.end(), _fitted_data.begin());


    matrix<real_t> _std = fitted.std();
    real_t _mean = fitted.mean();

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
  void pipeline::add_stage(const std::string& category, const std::string& name)  // add a stage to the pipeline 
  {
    // create a stage descriptor from the given parameters
    // find mapped types of stages from the given category


    if(category=="pipeline")
    {
        pipeline_stage *stage = new pipeline_stage();
        stage->name = name;
        stage->type = "pipeline";
        //stage id is   sample from chrono system time as uint64_t
        stage->stage_id = std::chrono::system_clock::now().time_since_epoch().count();
       if(_stages.size()>1) { 

        if(_stages.back()->next_stage==0)
          _stages.back()->next_stage = stage->stage_id;
        else
          
          {
            stage->next_stage = _stages.back()->next_stage; 
            _stages.back()->next_stage = stage->stage_id;
             stage->previous_stage = _stages.back()->stage_id;
              
          }

       }else
        {
          stage->next_stage = 0;
          stage->previous_stage = 0;
        }//end of if(_stages.size()>1) 

        stage->input = "";
        stage->output = "";
        stage->input_type = "";
        stage->output_type = "";
        stage->input_parameters = "";
        stage->output_parameters = "";
        stage->parameters = "";
         
        // create a stage from the stage descriptor with empty initialization
        _stages.push_back(stage);
        _pipelines.push_back(stage->get_pipeline());

        return;
    }

    std::vector<std::string> types = stage_factory_singleton::get_instance()->get_types(category); 
    // find the type of the stage with the given name
    auto it = std::find_if(types.begin(), types.end(), [name](const std::string& type) { return type == name; });
    if (it != types.end())
    {
      // create a stage descriptor from the given parameters
      stage_descriptor stage;
      stage.name = name;
      stage.type = *it;
      stage.stage_id = std::chrono::system_clock::now().time_since_epoch().count();
      stage.input = "";
      stage.output = "";
      stage.input_type = "";
      stage.output_type = "";
      stage.input_parameters = "";
      stage.output_parameters = "";
      stage.parameters = ""; 
      stage.next_stage=0;
      stage.previous_stage =   0;
      stage.next_stage =  0; 
      // create a stage from the stage descriptor with empty initialization
      stage_descriptor *new_stage = stage_factory_singleton::get_instance()->build_stage(stage);
      if (new_stage == nullptr)
        throw std::runtime_error("stage not found");
      // add the stage to the pipeline 
      add_stage (new_stage);

     }
    else { 
      //create stage from category and name
      stage_descriptor stage;
      stage.name = name;
      stage.type = category;
      stage.stage_id = std::chrono::system_clock::now().time_since_epoch().count();
      stage.input = "";
      stage.output = "";
      stage.input_type = "";
      stage.output_type = "";
      stage.input_parameters = "";
      stage.output_parameters = "";
      stage.parameters = "";
      // create a stage from the stage descriptor with empty initialization
      stage_descriptor *new_stage = stage_factory_singleton::get_instance()->build_stage(stage);
      if (new_stage == nullptr)
         throw std::runtime_error("stage not found");
      // add the stage to the pipeline  
      add_stage (new_stage);

     }
    
  }
  // add a stage to the pipeline
  void pipeline::add_stage(stage_descriptor *stage)
  {
    // add the stage to the pipeline

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
  // ------------------------------------------------
  //
  // get the number of stages
  // from the pipeline and the pipelines it contains.
  // ------------------------------------------------
  size_t pipeline::get_number_of_stages() const
  {
     size_t stages = 0;
    for(pipeline* pipe : _pipelines)
    {
      //  
      //
      //

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


  //get pipeline by name  
  pipeline* pipeline::get_pipeline(const std::string& name)
  {
    // find the pipeline with the given name
    auto it = std::find_if(_pipelines.begin(), _pipelines.end(), [name](pipeline *pipe)
                           { return pipe->get_pipeline_name() == name; });
    if (it != _pipelines.end())
    {
      // return the pipeline
      return *it;
    }
    return nullptr;
  }
  //get pipeline by id
  pipeline* pipeline::get_pipeline(uint64_t id)
  {
    // find the pipeline with the given id
    auto it = std::find_if(_pipelines.begin(), _pipelines.end(), [id](pipeline *pipe)
                           { return pipe->get_pipeline_id() == id; });
    if (it != _pipelines.end())
    {
      // return the pipeline
      return *it;
    }
    return nullptr;
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
 // squared sum distance 
    std::vector<real_t> cluster_stage::squared_sum_distances(const matrix<real_t> &data) 
    { 
      matrix<real_t> sqsum;
      std::vector<real_t> ret;
      sqsum = data * transpose(data);
      for (size_t i = 0; i < data.rows(); i++) {
        for (size_t j = 0; j < data.rows(); j++) {
          ret[i * data.rows() + j] = sqsum(i, i) + sqsum(j, j) - 2 * sqsum(i, j);
        }
      }
      return ret;
    }

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
  bool pipeline::build()
  {
    //build all stages
    
    return true;
  }
  //pipeline_builder::build
  bool pipeline_builder::build()
  { 
    for (auto& pipeline : _pipelines)
    {
      if (!pipeline->build())
      {
        return false;
      }
    }
    return true;

  }
  
  //pipeline_stage constructor
  pipeline_stage::pipeline_stage(const std::string& name) :_pipeline(nullptr)
  {
    //empty
    this->_pipeline = new pipeline();
    _pipeline->set_pipeline_name(name);

  }   
  //default constructor
  pipeline_stage::pipeline_stage():_pipeline(nullptr)  {
    //empty
    this->_pipeline = new pipeline();
    _pipeline->set_pipeline_name("pipeline_stage");

  } 
  //pipeline_stage destructor
  pipeline_stage::~pipeline_stage()
  {
    //empty
    delete _pipeline;
  } 
  //pipeline_stage -load additional data loads the file with the stages this pipeline contains  
  void pipeline_stage::load_additional_data(const std::string& file_name)
  {
    //empty

    UNDEF_REFERENCE(file_name);

    std::ifstream file(file_name);
    if (file.is_open())
    {
      if(!_pipeline) _pipeline = new pipeline();
      file>>*_pipeline;
      file.close();

    }
    else
    {
      std::cout << "Unable to open file " << file_name << std::endl;
    } 
  } 
  void pipeline_stage::save_additional_data(  std::string& filename)
  {
    std::ofstream file(filename);
    if (file.is_open())
    {
      file << *_pipeline;
      file.close();
    }
    else
    {
      std::cout << "Unable to open file " << filename << std::endl;
    }

  }
 
  //additional data for vectorizer_stage :
  //vectorizer_stage::load_additional_data implementation:
  void vectorizer_stage::load_additional_data(const std::string& file_name)
  {
    //empty 
    UNDEF_REFERENCE(file_name);
    //load vectorizer
    std::ifstream file(file_name);
    std::string tmp = "vectorizer_count:";
    if (file.is_open())
    {
      std::string line;
      std::getline(file, line);
      size_t count;
      if (line.find(tmp) != std::string::npos)
      {
        std::string scount = line.substr(tmp.size());
        count = std::stoi(scount);
      }
      else
      {
        std::cout << "Unable to find vectorizer count in file " << file_name << std::endl;
        file.close();
        return;
      }
      if ( vectorizers.size() ) 
      {
        for (auto& vectorizer : vectorizers)
        {
          delete vectorizer;
        }
        vectorizers.clear();
      }

      vectorizers.resize(count);
      for (  size_t i=0;i<count ;i++ ) 
      {
        
        vectorizers[i] = new vectorizer<std::string,real_t>(UNKNOWN_VECTORIZER);        
        
        file>>*vectorizers[i];
        
      } 
      //read feature engineering
      std::getline(file, line);
      tmp = "feature_engineering:";
      if (line.find(tmp) != std::string::npos)
      {
        std::string sfeature_engineering = line.substr(tmp.size());
        feature_engineering_type = std::stoi(sfeature_engineering);
      }
      else
      {
        std::cout << "Unable to find feature_engineering in file " << file_name << std::endl;
        file.close();
        return;
      } 
      //read feature selection
      std::getline(file, line);
      tmp  ="feature_selection:";
      if (line.find(tmp) != std::string::npos)
      {
        std::string sfeature_selection = line.substr(tmp.size());
        feature_selection_type = std::stoi(sfeature_selection);
      }
      else
      {
        std::cout << "Unable to find feature_selection in file " << file_name << std::endl;
        file.close();
        return;
      } 
      file.close();
    }
    else
    {
      std::cout << "Unable to open file " << file_name << std::endl;
    }
    //load vectorizer
  }
  
  //vectorizer_stage::save_additional_data implementation:
  //save vectorizer_stage additional data
  void vectorizer_stage::save_additional_data(  std::string& filename)
  {
    std::ofstream file(filename);
    //save vectorizers count
    if (file.is_open())
    {
      file << "vectorizer_count:" << vectorizers.size() << std::endl;
      for (auto& vectorizer : vectorizers)
      {
        file << *vectorizer;
      }
      file << "feature_engineering:" << feature_engineering_type << std::endl;
      file << "feature_selection:" << feature_selection_type << std::endl;
      file.close();
    }
    else
    {
      std::cout << "Unable to open file " << filename << std::endl;
    } 
    
  }
   //stage build functions vectorizer,feature_stage,cluster_stage,classifier_stage,regressor_stage ,etc..
 
  vectorizer_stage* vectorizer_stage::build()
  { 
     //build vectorizer stage
     vectorizer_stage* stage =  new vectorizer_stage;
     //set vectorizer stage name
     //return stage
     stage->type = "vectorizer_stage";
     //make sure it's initialized 
     stage->initialize();
     return stage;  

  }

  feature_stage* feature_stage::build()
  {
    //build feature stage
    feature_stage* stage = new feature_stage;
    //set feature stage name
    //return stage
     stage->type = "feature_stage";
    return stage;
  }

  cluster_stage* cluster_stage::build()
  {
    //build cluster stage
    cluster_stage* stage = new cluster_stage;
    //set cluster stage name
    //return stage
     stage->type = "cluster_stage";
    return stage;
  }

  classifier_stage* classifier_stage::build()
  {
    //build classifier stage
    classifier_stage* stage = new classifier_stage();
    
    //set classifier stage name
    //return stage
    //stage->set_descriptor(this);

    stage->type = "classifier_stage";
    return stage;
  }

  regressor_stage* regressor_stage::build()
  {
    //build regressor stage
    regressor_stage* stage = new regressor_stage;
    //set regressor stage name
    //return stage
 
    stage->type = "regressor_stage";
    return stage;
  }

  knowledge_transfer_stage* knowledge_transfer_stage::build()
  {
    //build knowledge transfer stage
    knowledge_transfer_stage* stage = new knowledge_transfer_stage;
    //set knowledge transfer stage name
    //return stage
 
    stage->type = "knowledge_transfer_stage";
    return stage;
  }
  dataset_stage* dataset_stage::build()
  {
    //build dataset stage
    dataset_stage* stage = new dataset_stage;
    //set dataset stage name
    //return stage
 
    stage->type = "dataset_stage";
    return stage;
  }
  dimentionality_reduction_stage * dimentionality_reduction_stage::build()
  {
    //build dimentionality reduction stage
    dimentionality_reduction_stage* stage = new dimentionality_reduction_stage;
    //set dimentionality reduction stage name
    //return stage
 
    stage->type = "dimentionality_reduction_stage";
    return stage;
  }
  //encoder stage 
  encoder_stage* encoder_stage::build()
  {
    //build encoder stage
    encoder_stage* stage = new encoder_stage;
    //set encoder stage name
    //return stage
 
    stage->type = "encoder_stage";
    return stage;
  } 
  //decoder stage
  decoder_stage* decoder_stage::build()
  {
    //build decoder stage
    decoder_stage* stage = new decoder_stage;
    //set decoder stage name
    //return stage
 
    stage->type = "decoder_stage";
    return stage;
  } 
  //normalizer
  normalizer_stage* normalizer_stage::build()
  {
    //build normalizer stage
    normalizer_stage* stage = new normalizer_stage;
    //set normalizer stage name
    //return stage
 
    stage->type = "normalizer_stage";
    return stage;
  }   


  //filter

  filter_stage::filter_stage()
  {
    name = "filter_stage";
    //empty
    //    this->_filter = new filter_base();

    //_filter->set_filter_name("filter_stage");

  }
  //destructor
  filter_stage::~filter_stage()
  {
    //empty
  
  }
  //load additional data


  filter_stage* filter_stage::build()
  {
    //build filter stage
    filter_stage* stage = new filter_stage;
    //set filter stage name
    //return stage
 
    stage->type = "filter_stage";
    return stage;
  } 
  //load additional stage data 
  void filter_stage::load_additional_data(const std::string& data)
  {
    UNDEF_REFERENCE(data);
    UNDEF_REFERENCE2(data);
    //empty
  }
  void filter_stage::save_additional_data(  std::string& data)
  {
    UNDEF_REFERENCE(data);
    UNDEF_REFERENCE2(data);
    
    //empty

  }
  //normalizer stage 
  normalizer_stage::normalizer_stage() 
  {
    name = "normalizer_stage";
    //empty
    //this->_normalizer = new normalizer_base();
    //_normalizer->set_normalizer_name("normalizer_stage");
  }
  //destructor
  normalizer_stage::~normalizer_stage()
  {
    //empty
  }
  //load additional data
  void normalizer_stage::load_additional_data(const std::string& data)
  {
    //data contains parameters as of what to normalize.
    UNDEF_REFERENCE(data);
    UNDEF_REFERENCE2(data);
    //empty
  }
  void normalizer_stage::save_additional_data(  std::string& data)
  {
    //data contains parameters as of what to normalize.
    UNDEF_REFERENCE(data);
    UNDEF_REFERENCE2(data);
    //empty
  }
  //encoder stage
  encoder_stage::encoder_stage()
  {
    name = "encoder_stage";
    //empty
    //this->_encoder = new encoder_base();
    //_encoder->set_encoder_name("encoder_stage");
  }
  //destructor
  encoder_stage::~encoder_stage()
  {
    //empty
  }
  //load additional data
  void encoder_stage::load_additional_data(const std::string& data)
  {
    //data contains parameters as of what to encode.
    UNDEF_REFERENCE(data);
    UNDEF_REFERENCE2(data);
    //empty
  }
  void encoder_stage::save_additional_data(  std::string& data)
  {
    //data contains parameters as of what to encode.
    UNDEF_REFERENCE(data);
    UNDEF_REFERENCE2(data);
    //empty
  }
  //decoder stage
  decoder_stage::decoder_stage()
  {
    name = "decoder_stage";
    //empty
    //this->_decoder = new decoder_base();
    //_decoder->set_decoder_name("decoder_stage");
  }
  //destructor
  decoder_stage::~decoder_stage()
  {
    //empty
  }
  //load additional data
  void decoder_stage::load_additional_data(const std::string& data)
  {
    //data contains parameters as of what to decode.
    UNDEF_REFERENCE(data);
    UNDEF_REFERENCE2(data);
    //empty
  }
  void decoder_stage::save_additional_data(  std::string& data)
  {
    //data contains parameters as of what to decode.
    UNDEF_REFERENCE(data);
    UNDEF_REFERENCE2(data);
    //empty
  }
  //dimentionality reduction stage
  dimentionality_reduction_stage::dimentionality_reduction_stage()
  {
    name = "dimentionality_reduction_stage";
    //empty
    //this->_dimentionality_reduction = new dimentionality_reduction_base();
    //_dimentionality_reduction->set_dimentionality_reduction_name("dimentionality_reduction_stage");
  }
  //destructor
  dimentionality_reduction_stage::~dimentionality_reduction_stage()
  {
    //empty
  }
  //load additional data
  void dimentionality_reduction_stage::load_additional_data(const std::string& data)
  {
    //data contains parameters as of what to reduce.
    UNDEF_REFERENCE(data);
    UNDEF_REFERENCE2(data);
    //empty
  }
  void dimentionality_reduction_stage::save_additional_data(  std::string& data)
  {
    //data contains parameters as of what to reduce.
    UNDEF_REFERENCE(data);
    UNDEF_REFERENCE2(data);
    //empty
  }

  //vectorizer_stage::build
  //dataset_stage constructor
  dataset_stage::dataset_stage():_dataset(nullptr)
  {
    name = "dataset_stage";

    //empty
    //    this->_dataset = new dataset_base();

    //_dataset->set_dataset_name("dataset_stage");

  }

  regressor_stage::regressor_stage() 
  {
    name = "regressor_stage";
    //empty
    //this->_regressor = new regressor_base();
    //_regressor->set_regressor_name("regressor_stage");
  } 
  //destructor
  regressor_stage::~regressor_stage()
  {
    //empty
  }
  //load additional data
  void regressor_stage::load_additional_data(const std::string& data)
  {
    //data contains parameters as of what to regress.
    UNDEF_REFERENCE(data);
    UNDEF_REFERENCE2(data);
    //empty
  }
  void regressor_stage::save_additional_data(  std::string& data)
  {
    //data contains parameters as of what to regress.
    UNDEF_REFERENCE(data);
    UNDEF_REFERENCE2(data);
    //empty
  }
  //dataset_stage destructor
  dataset_stage::~dataset_stage()
  {
    //empty
    if(_dataset)
    delete _dataset;
  }

  //dataset_stage -load additional data loads the file with the stages this dataset contains
  void dataset_stage::load_additional_data(const std::string& file_name)
  {
    //empty 
    //load dataset from file 
    //this is a blind dataset, no attributes or labels are known yet. 
    //we don't know the number of samples/columns/rows yet
     //load dataset from file
    //create dataset
    UNDEF_REFERENCE(file_name);
    UNDEF_REFERENCE2(file_name);

    //load dataset from file

  #if 0
        io::csv_reader reader;
        reader.read(file_name);

        if(!_dataset) _dataset = new dataset_base();
        _dataset->set_dataset_name(file_name);
        _dataset->set_dataset(reader.get_dataset());
        _dataset->set_attributes(reader.get_attributes());
        _dataset->set_labels(reader.get_labels());
        _dataset->set_number_of_samples(reader.get_number_of_samples());
        _dataset->set_number_of_attributes(reader.get_number_of_attributes());
        _dataset->set_number_of_labels(reader.get_number_of_labels());
        _dataset->set_number_of_classes(reader.get_number_of_classes());
        _dataset->set_number_of_clusters(reader.get_number_of_clusters());
        _dataset->set_number_of_rows(reader.get_number_of_rows());
        _dataset->set_number_of_columns(reader.get_number_of_columns());
        _dataset->set_number_of_samples_per_class(reader.get_number_of_samples_per_class());
        _dataset->set_number_of_samples_per_cluster(reader.get_number_of_samples_per_cluster());
        _dataset->set_number_of_samples_per_label(reader.get_number_of_samples_per_label());
        _dataset->set_number_of_samples_per_row(reader.get_number_of_samples_per_row());
        _dataset->set_number_of_samples_per_column(reader.get_number_of_samples_per_column());
        _dataset->set_number_of_samples_per_attribute(reader.get_number_of_samples_per_attribute());
        _dataset->set_number_of_samples_per_class_and_label(reader.get_number_of_samples_per_class_and_label());
        _dataset->set_number_of_samples_per_class_and_attribute(reader.get_number_of_samples_per_class_and_attribute());
  #endif
    //set dataset
    //set attributes
    
 } 
 //vectorizer_stage::vectorizer_stage
  vectorizer_stage::vectorizer_stage() 
  {
    name = "vectorizer_stage";

    //empty
    //this->_vectorizer = new vectorizer_base();
    //_vectorizer->set_vectorizer_name("vectorizer_stage");
  }
  knowledge_transfer_stage::knowledge_transfer_stage()   
  {
    name = "knowledge_transfer_stage";
    //empty
    //this->_knowledge_transfer = new knowledge_transfer_base();
    //_knowledge_transfer->set_knowledge_transfer_name("knowledge_transfer_stage");
  } 
  //destructor
  knowledge_transfer_stage::~knowledge_transfer_stage()
  {
    //empty
  }
  //load additional data
   
     //cluster_stage::birch
  std::vector<size_t> cluster_stage::birch(const matrix<real_t>& data,size_t n_clusters ,size_t threshold,size_t branching_factor,size_t compute_labels,size_t copy)
  {
    //birch
    //parameters n_clusters, size_t threshold, size_t branching_factor, size_t compute_labels, size_t copy
    std::vector<size_t> ret;
    UNDEF_REFERENCE(n_clusters);
    UNDEF_REFERENCE2(threshold);
    UNDEF_REFERENCE2(branching_factor);
    UNDEF_REFERENCE2(compute_labels);
    UNDEF_REFERENCE2(copy);
    //birch

    //Given a set of N d-dimensional data points, the clustering feature C F CF of the set is defined as the triple C F = ( N , L S → , S S ) {\displaystyle CF=(N,{\overrightarrow {LS}},SS)}, where 
    //N is the number of data points in the set,
    //L S → {\displaystyle {\overrightarrow {LS}}} {\displaystyle {\overrightarrow {LS}}} is the linear sum of the data points in the set, and
    //S S {\displaystyle SS} SS is the squared sum of the data points in the set.
    //The clustering feature of a set of data points can be computed recursively from the clustering features of its subsets.

     //L S → = ∑ i = 1 N X i → {\displaystyle {\overrightarrow {LS}}=\sum _{i=1}^{N}{\overrightarrow {X_{i}}}} is the linear sum.
     //S S = ∑ i = 1 N X i → ⋅ X i → {\displaystyle SS=\sum _{i=1}^{N}{\overrightarrow {X_{i}}}\cdot {\overrightarrow {X_{i}}}} is the squared sum.
      //N = ∑ i = 1 N N i {\displaystyle N=\sum _{i=1}^{N}N_{i}} is the number of data points.

    
    
    
    matrix<real_t> linear_sum;
    matrix<real_t> squared_sum;
    matrix<real_t> number_of_data_points;
    matrix<real_t> co_moments;
    matrix<real_t> centroids;
    matrix<real_t> labels;
    matrix<real_t> distances;


    co_moments.resize(data.cols(),data.cols());
    co_moments.fill(0.0);
    //linear sum
    linear_sum.resize(data.cols(),1);
    linear_sum.fill(0.0);
    //squared sum
    squared_sum.resize(data.cols(),1);
    squared_sum.fill(0.0);
    //number of data points
    number_of_data_points.resize(data.cols(),1);
    number_of_data_points.fill(0.0);
    //compute linear sum

    for ( size_t i = 0; i < data.rows(); i++)
    {
      for ( size_t j = 0; j < data.cols(); j++)
      {
        linear_sum(j,0) += data(i,j);
        squared_sum(j,0) += data(i,j)*data(i,j);
        number_of_data_points(j,0) += 1.0;
        co_moments(j,j) += data(i,j)*data(i,j);
        
      }
    }
    //compute linear sum
    //compute squared sum
    for ( size_t j = 0; j < data.cols(); j++)
    {
      squared_sum(j,0) = sqrt(squared_sum(j,0));
    }
    //compute squared sum
    //compute number of data points
    for ( size_t j = 0; j < data.cols(); j++)
    {
      number_of_data_points(j,0) = number_of_data_points(j,0);
    }
    //compute number of data points
    //compute clustering feature
    matrix<real_t> clustering_feature;
    clustering_feature.resize(data.cols(),3);
    clustering_feature.fill(0.0);
    //compute clustering feature
    

    //calculate clustering feature
    for ( size_t j = 0; j < data.cols(); j++)
    {
      clustering_feature(j,0) = number_of_data_points(j,0);
      clustering_feature(j,1) = linear_sum(j,0);
      clustering_feature(j,2) = squared_sum(j,0);
      //co moments

    } 
    //calculate covariance and distances between the matrices 
    matrix<real_t> covariance = data.covariance();
    
    
    //calculate covariance

    //calculate distances using the distance function and fill distance matrix
    distances.resize(data.cols(),data.cols());
    distances.fill(0.0);
    //get the distance function from the map metrics
     //std::map<std::string,metric_t> metrics;
    
    //all the distance functions should be mapped to the metrics map
    auto manhatten_distance = metrics["manhatten_distance"] ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    auto euclidean_distance = metrics["euclidean_distance"] ;
    auto inter_cluster_distance = metrics["inter_cluster_distance"] ; 
    auto intra_cluster_distance = metrics["intra_cluster_distance"] ;
    auto variance_increased = metrics["variance_increased"] ;


    std::vector<metric_t> distance_functions;
    distance_functions.push_back(manhatten_distance);
    distance_functions.push_back(euclidean_distance);
    distance_functions.push_back(inter_cluster_distance);
    distance_functions.push_back(intra_cluster_distance);
    distance_functions.push_back(variance_increased);


    //    calculate_distance_matrix(data,distances,metrics); 
    

    //auto cluster_function = metrics.find(_distance_function)->second;    
     //calculate distances
    distances = covariance;
    for ( auto dist : distance_functions )
    {
      auto ret = ((this->*dist)(covariance));
      for  ( size_t i=0;i*distances.cols()<ret.size();++i)
      {
        for(size_t j=0;j<distances.cols();j++)
        {
          distances(i,j) += ret[i*distances.cols()+j];
        }
      }
    } 
    //normalize distance
    distances =distances  / real_t(distance_functions.size());
    //calculate distances

    //distances are now calculated
    //now we can calculate the centroids
    //we can use the distances to calculate the centroids
    //we can use the distances to calculate the centroids


    //calculate centroids
    centroids.resize(data.cols(),data.cols());
    centroids.fill(0.0);
    //calculate centroids
    

    
    //calculate the centroids
    for ( size_t i = 0; i < data.cols(); i++)
    {
      for ( size_t j = 0; j < data.cols(); j++)
      {
        centroids(i,j) = clustering_feature(i,j);
      }
    }
    //find the labels for the data
    labels.resize(data.rows(),1);
    labels.fill(0.0);
    //find the labels for the data
    for ( size_t i = 0; i < data.rows(); i++)
    {
      for ( size_t j = 0; j < data.cols(); j++)
      {
        labels(i,0) = clustering_feature(j,0);
      }
    } 
    
    //fill ret   labels
    for ( size_t i = 0; i < labels.rows(); i++)
    {
      ret.push_back(labels(i,0));
    }

      this->cluster_centers = centroids;
      //convert labels 
      this->labels.resize(labels.rows(),1);
      for(size_t i=0;i<ret.size();i++)
      {
        this->labels[i] = reinterpret_cast<size_t>( ret[i] );
      } 
      //return labels

    return ret;
  }
  //    void dataset_stage::set_data(const matrix <real_t>& data )      
   //parameters n_clusters, size_t n_init, size_t max_iter, real_t tol, int random_state, int verbose, bool precompute_distances, bool copy_x
   std::vector<size_t> cluster_stage::kmeans(const matrix<real_t>& data,size_t n_clusters ,size_t n_init,size_t max_iter,size_t random_state ,size_t verbose )
   {
      //kmeans
      //parameters n_clusters, size_t n_init, size_t max_iter , int random_state, 
      //kmeans
      //this tolerance means that if the relative error is less than tol, the algorithm is considered to have converged and the iterations are stopped. 
 
      UNDEF_REFERENCE(n_clusters);
      UNDEF_REFERENCE2(n_init);
      UNDEF_REFERENCE2(max_iter);
      UNDEF_REFERENCE2(random_state);
      UNDEF_REFERENCE2(verbose);

      

      
      std::vector<size_t> result(data.rows(),0);
      //initialize the centroids
      matrix<real_t> centroids,distances,labels,clustering_feature;

      matrix<real_t> covariance = data.covariance();


      //calculate distances using the distance function and fill distance matrix
      distances.resize(data.cols(),data.cols());
      distances.fill(0.0);
      //get the distance function from the map metrics
       //std::map<std::string,metric_t> metrics;  
      //all the distance functions should be mapped to the metrics map
      metric_t distance_function =  metrics["squared_sum"] ;
      //calculate distances
      distances = covariance;
      
      //normalize distance
      //distances =distances  / real_t(distance_functions.size());
      //calculate distances
      auto dist_ret = ((this->*distance_function)(covariance));

      for  ( size_t i=0;i*distances.cols()<dist_ret.size();++i)
      {
        for(size_t j=0;j<distances.cols();j++)
        {
          distances(i,j) += dist_ret[i*distances.cols()+j];
        }
      }
      //normalize distance
      distances =distances  / real_t(1);
      
      //distances are now calculated
      //now we can calculate the centroids
      //use n_cluster to initialize clustering_feature matrix 

      clustering_feature.resize(data.cols(),1);
      clustering_feature.fill(0.0);
      //initialize the clustering_feature matrix
      for ( size_t i = 0; i < data.cols(); i++)
      {
        clustering_feature(i,0) = i % n_clusters;
      }
      //initialize the clustering_feature matrix

      //calculate centroids
      centroids.resize(data.cols(),data.cols());
      centroids.fill(0.0);
      //calculate centroids
      //calculate the centroids
      for ( size_t i = 0; i < data.cols(); i++)
      {
        for ( size_t j = 0; j < data.cols(); j++)
        {
          centroids(i,j) = clustering_feature(i,j);
        }
      }
      //find the labels for the data
       labels.resize(data.rows(),1);
      labels.fill(0.0);
      //find the labels for the data
      for ( size_t i = 0; i < data.rows(); i++)
      {
        for ( size_t j = 0; j < data.cols(); j++)
        {
          labels(i,0) = clustering_feature(j,0);
        }
      }
      //fill ret   labels
      for ( size_t i = 0; i < labels.rows(); i++)
      {
        result.push_back(labels(i,0));
      }
      this->cluster_centers = centroids;
      //convert labels 
      this->labels.resize(labels.rows(),1);
      for(size_t i=0;i<result.size();i++)
      {
        this->labels[i] = reinterpret_cast<size_t>( result[i] );
      } 
      //return labels

      return result; //return labels
      
   }  
   
  //dataset_stage::dataset_stage
  //scoring functions implementations 
  //silouhette score       std::vector<real_t> silhouette_score(const matrix<real_t>& data, std::vector<size_t> labels, size_t met, size_t sample_size, size_t random_state, size_t n_jobs);

  std::vector<real_t> cluster_stage::silhouette_score(const matrix<real_t>& data, std::vector<size_t> labels, size_t met, size_t sample_size, size_t random_state, size_t n_jobs)
  { 
    //calculate the goodness of the clustering method, 
    //Silhouette Coefficient or silhouette score is a metric used to calculate the goodness of a clustering technique. Its value ranges from -1 to 1. 1: Means clusters are well apart from each other and clearly distinguished.

    UNDEF_REFERENCE(met);
    UNDEF_REFERENCE2(sample_size);
    UNDEF_REFERENCE2(random_state);
    UNDEF_REFERENCE2(n_jobs);
    std::vector<real_t> ret;


    auto intra_distance = scores["intra_cluster_distance"];
    auto inter_distance = scores["inter_cluster_distance"];

    //calculate the intra cluster distance
    auto intra = ((this->*intra_distance)(data,labels,met,sample_size,random_state,n_jobs));
    //calculate the inter cluster distance
    auto inter = ((this->*inter_distance)(data,labels,met,sample_size,random_state,n_jobs));
    //calculate the silhouette score
    //silhouette score = (inter - intra) / max(inter,intra)

    for ( size_t i = 0; i < intra.size(); i++)
    {
      ret.push_back((inter[i] - intra[i]) / std::max(inter[i],intra[i]));
    } 
    return ret;
  }
  //calinski harabasz score
  std::vector<real_t> cluster_stage::calinski_harabasz_score(const matrix<real_t>& data, std::vector<size_t> labels , size_t met, size_t sample_size, size_t random_state, size_t n_jobs  )
  {
    std::vector<real_t> ret;
    
    UNDEF_REFERENCE(met);
    UNDEF_REFERENCE2(sample_size);
    UNDEF_REFERENCE2(random_state);
    UNDEF_REFERENCE2(n_jobs);

    //Variance Ratio Criterion (Calinski & Harabasz, 1974)
    //The Calinski-Harabasz index (variance ratio criterion) can be used to evaluate the model, where a higher Calinski-Harabasz score relates to a model with better defined clusters.
    //The score is defined as ratio between the within-cluster dispersion and the between-cluster dispersion.
    auto intra_cluster_distance = scores["intra_cluster_distance"];
    auto inter_cluster_distance = scores["inter_cluster_distance"];

    //calculate the intra cluster distance
    auto intra = ((this->*intra_cluster_distance)(data,labels,met,sample_size,random_state,n_jobs));
    //calculate the inter cluster distance
    auto inter = ((this->*inter_cluster_distance)(data,labels,met,sample_size,random_state,n_jobs));
    //(n-cn)*sum(diag(B))/((cn-1)*sum(diag(W))) . B being the between-cluster means, and W being the within-clusters covariance matrix.

    //calculate the centroids
    matrix<real_t> centroids;
    centroids.resize(data.cols(),data.cols());
    centroids.fill(0.0);
    
    //labels for each row of the data matrix represents the cluster number for that row of the data matrix   
    //calculate the covariant matrix for each cluster
    matrix<real_t> clustering_feature;
    clustering_feature.resize(data.cols(),data.cols());
    clustering_feature.fill(0.0);
    //fill the clustering_feature matrix
    for ( size_t i = 0; i < data.rows(); i++)
    {
      for ( size_t j = 0; j < data.cols(); j++)
      {
        clustering_feature(labels[i],j) += data(i,j);
      }
    } 
    //calculate the centroids
    for ( size_t i = 0; i < data.cols(); i++)
    {
      for ( size_t j = 0; j < data.cols(); j++)
      {
        centroids(i,j) = clustering_feature(i,j);
      }
    } 
    //fill the ret vector
    for ( size_t i = 0; i < intra.size(); i++)
    {
      ret.push_back(((data.rows() - intra[i]) * inter[i]) / ((intra[i] - 1) * intra[i]));
    }

    this->cluster_centers = centroids;
    return ret;

  }
  //davies bouldin score
  std::vector<real_t> cluster_stage::davies_bouldin_score(const matrix<real_t>& data, std::vector<size_t> labels, size_t met, size_t sample_size, size_t random_state, size_t n_jobs)
  {
    std::vector<real_t> ret;
    
    UNDEF_REFERENCE(met);
    UNDEF_REFERENCE2(sample_size);
    UNDEF_REFERENCE2(random_state);
    UNDEF_REFERENCE2(n_jobs);

    //Davies-Bouldin index can be used to evaluate the model, where a lower Davies-Bouldin index relates to a model with better separation between the clusters. 
    //The index is calculated by finding the average similarity measure of each cluster with its most similar cluster, where similarity is the ratio of within-cluster distances to between-cluster distances.
    //Thus, clusters which are farther apart and less dispersed will result in a better score.
    //The index is the average similarity measure of each cluster with its most similar cluster, where similarity is the ratio of within-cluster distances to between-cluster distances.
    //Thus, those clusters which are farther apart and less dispersed will result in a better score.
    //The minimum score is zero, with lower values indicating better clustering.

    auto intra_cluster_distance = scores["intra_cluster_distance"];
    auto inter_cluster_distance = scores["inter_cluster_distance"];

    //calculate the intra cluster distance
    auto intra = ((this->*intra_cluster_distance)(data,labels,  met, sample_size, random_state, n_jobs));
    //calculate the inter cluster distance
    auto inter = ((this->*inter_cluster_distance)(data,labels,  met, sample_size, random_state, n_jobs));
    //calculate the centroids
    matrix<real_t> centroids;
    centroids.resize(data.cols(),data.cols());
    centroids.fill(0.0);
    

    //labels for each row of the data matrix represents the cluster number for that row of the data matrix
    //calculate the covariant matrix for each cluster
    matrix<real_t> clustering_feature;
    clustering_feature.resize(data.cols(),data.cols());
    clustering_feature.fill(0.0);
    //fill the clustering_feature matrix
    for ( size_t i = 0; i < data.rows(); i++)
    {
      for ( size_t j = 0; j < data.cols(); j++)
      {
        clustering_feature(labels[i],j) += data(i,j);
      }
    } 
    //calculate the centroids
    for ( size_t i = 0; i < data.cols(); i++)
    {
      for ( size_t j = 0; j < data.cols(); j++)
      {
        centroids(i,j) = clustering_feature(i,j);
      }
    } 


    //fill the ret vector
    for ( size_t i = 0; i < intra.size(); i++)
    {
      ret.push_back((intra[i] + inter[i]) / intra[i]);
    } 
    this->cluster_centers = centroids;

    return ret;
  }// davies_bouldin_score
  //silhouette score

  

  std::vector<real_t> cluster_stage::adjusted_rand_score(const matrix<real_t>& data, std::vector<size_t> labels, size_t met, size_t sample_size, size_t random_state, size_t n_jobs)
  {
    std::vector<real_t> ret; 
    UNDEF_REFERENCE(met);
    UNDEF_REFERENCE2(sample_size);
    UNDEF_REFERENCE2(random_state);
    UNDEF_REFERENCE2(n_jobs);
    //adjusted rand score
    //adjusted_rand_score(labels_true, labels_pred)
    //labels_true : int array, shape = [n_samples]
    //Ground truth class labels to be used as a reference`
    //labels_pred : array, shape = [n_samples]
    //Cluster labels to evaluate
    //adjusted_rand_score : float
    //Rand index adjusted for chance.
    //The Rand Index computes a similarity measure between two clusterings by considering all pairs of samples and counting pairs that are assigned in the same or different clusters in the predicted and true clusterings.
    //The raw RI score is then “adjusted for chance” into the ARI score using the following scheme:
    //ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)
    //The adjusted Rand index is thus ensured to have a value close to 0.0 for random labeling independently of the number of clusters and samples and exactly 1.0 when the clusterings are identical (up to a permutation).
    //ARI is a symmetric measure:
    //ARI(a, b) == ARI(b, a)
    //ARI is furthermore symmetric to label permutations:
    //ARI(a, b) == ARI(a[i], b[i]) for any permutations of the labels a[i] of cluster A and b[i] of cluster B.
    //ARI is furthermore almost symmetric to adding uniformly distributed noise to a clustering:
    //ARI(a, b) == ARI(a, c) for c = add_uniform_noise(a, size=n_samples, n_clusters=n_clusters)
    //ARI is in the range [-1, 1]. Random labelings have an ARI close to 0.0. 1.0 stands for perfect match.
     //calculate the intra cluster distance
    auto intra_cluster_distance = scores["intra_cluster_distance"];
    auto inter_cluster_distance = scores["inter_cluster_distance"];

    auto intra = ((this->*intra_cluster_distance)(data,labels,met,sample_size,random_state,n_jobs));  
    //calculate the inter cluster distance
    auto inter = ((this->*inter_cluster_distance)(data,labels,met,sample_size,random_state,n_jobs));
    //calculate the centroids
    matrix<real_t> centroids;

    centroids.resize(data.cols(),data.cols());
    centroids.fill(0.0);
    
    //labels for each row of the data matrix represents the cluster number for that row of the data matrix
    //calculate the covariant matrix for each cluster
    matrix<real_t> clustering_feature;
    clustering_feature.resize(data.cols(),data.cols());
    clustering_feature.fill(0.0);
    //fill the clustering_feature matrix
    for ( size_t i = 0; i < data.rows(); i++)
    {
      for ( size_t j = 0; j < data.cols(); j++)
      {
        clustering_feature(labels[i],j) += data(i,j);
      }
    } 
    //calculate the centroids
    for ( size_t i = 0; i < data.cols(); i++)
    {
      for ( size_t j = 0; j < data.cols(); j++)
      {
        centroids(i,j) = clustering_feature(i,j);
      }
    } 
    //calculate the adjusted rand score
    for ( size_t i = 0; i < intra.size(); i++)
    { 
      real_t ari = (inter[i] - intra[i]) / (std::max(inter[i],intra[i]) - intra[i]); 
      ret.push_back(ari);
      
    }     
 
    return ret;
  }
  std::vector<real_t> cluster_stage::adjusted_mutual_info_score(const matrix<real_t>& data, std::vector<size_t> labels, size_t met, size_t sample_size, size_t random_state, size_t n_jobs)
  {
    std::vector<real_t> ret;
    
    UNDEF_REFERENCE(met);
    UNDEF_REFERENCE2(sample_size);
    UNDEF_REFERENCE2(random_state);
    UNDEF_REFERENCE2(n_jobs);
    //adjusted_mutual_info_score(labels_true, labels_pred)
    //labels_true : int array, shape = [n_samples]
    //Ground truth class labels to be used as a reference
    //labels_pred : array, shape = [n_samples]
    //Cluster labels to evaluate
    //adjusted_mutual_info_score : float
    //Adjusted Mutual Information between two clusterings.
    //Adjusted Mutual Information (AMI) is an adjustment of the Mutual Information (MI) score to account for chance. It accounts for the fact that the MI is generally higher for two clusterings with a larger number of clusters, regardless of whether there is actually more information shared. For two clusterings U and V, the AMI is given as:
    //where H(U) and H(V) are the entropy of the two clusterings and MI(U, V) is the mutual information between the two clusterings.
    //The AMI returns a value of 1 when the two partitions are identical (ie perfectly matched). Random partitions (independent labellings) have an expected AMI around 0 on average hence can be negative.
    //The AMI is symmetric: switching label_true with label_pred will return the same score value. This is not the case for the Mutual Information.
    //Be mindful that this function is an order of magnitude slower than other metrics, such as the Adjusted Rand Index.
    //Note: The logarithm used is the natural logarithm (base-e).
    //Note: This is an asymmetric measure and thus does not satisfy the mathematical definition of a metric.
    //Note: In the literature, the MI score is sometimes called the variation of information (VI).
    //Note: The algorithm used by this function is an adaptation of the one used in FAST (Francois 2009).
    //Note: The AMI between two clusterings with random labels is 0. The AMI between a clustering and its shuffled version is 1.
    //Note: The AMI is not adjusted for chance.
    //Note: The AMI requires the number of clusters to be equal in both clusterings.
    //Note: The AMI is often used in the literature when the number of clusters are not equal. In this case, the AMI is not well-defined. It is preferable to use the Adjusted Rand Index instead in such application, which is implemented in adjusted_rand_score and also available as adjusted_rand_index.
    //Note: The AMI is symmetric, the order of the label arguments does not matter.
    //Note: The AMI is bounded between -1 and 1. Values close to 0 indicate two label assignments that are largely independent, while values close to 1 indicate significant agreement. Values close to -1 on the other hand indicate significant disagreement.


    //calculate the intra cluster distance
    auto intra_cluster_distance = scores["intra_cluster_distance"];
    auto inter_cluster_distance = scores["inter_cluster_distance"];

    auto intra = ((this->*intra_cluster_distance)(data,labels,met,sample_size,random_state,n_jobs));
    //calculate the inter cluster distance
    auto inter = ((this->*inter_cluster_distance)(data,labels,met,sample_size,random_state,n_jobs));

    //calculate the centroids
    matrix<real_t> centroids;

    centroids.resize(data.cols(),data.cols());
    centroids.fill(0.0);

    //labels for each row of the data matrix represents the cluster number for that row of the data matrix
    //calculate the covariant matrix for each cluster
    matrix<real_t> clustering_feature;
    clustering_feature.resize(data.cols(),data.cols());
    clustering_feature.fill(0.0);
    //fill the clustering_feature matrix
    for ( size_t i = 0; i < data.rows(); i++)
    {
      for ( size_t j = 0; j < data.cols(); j++)
      {
        clustering_feature(labels[i],j) += data(i,j);
      }
    }
    //calculate the centroids
    for ( size_t i = 0; i < data.cols(); i++)
    {
      for ( size_t j = 0; j < data.cols(); j++)
      {
        centroids(i,j) = clustering_feature(i,j);
      }
    }
    //calculate the adjusted mutual info score
    for ( size_t i = 0; i < intra.size(); i++)
    {
      real_t ami = (inter[i] - intra[i]) / (std::max(inter[i],intra[i]) - intra[i]);
      ret.push_back(ami);

    }

    return ret;
  }

  std::vector<real_t> cluster_stage::homogeneity_completeness_v_measure(const matrix<real_t>& data, std::vector<size_t> labels, size_t met, size_t sample_size, size_t random_state, size_t n_jobs)
  {
    //return homogeneity_score(labels_true, labels_pred), completeness_score(labels_true, labels_pred), v_measure_score(labels_true, labels_pred)
    std::vector<real_t> ret;
    UNDEF_REFERENCE(met);
    UNDEF_REFERENCE2(sample_size);
    UNDEF_REFERENCE2(random_state);
    UNDEF_REFERENCE2(n_jobs);

    //calculate the homogeneity score
    auto homogeneity = homogeneity_score(data,labels,met,sample_size,random_state,n_jobs); 
    //calculate the completeness score
    auto completeness = completeness_score(data,labels,met,sample_size,random_state,n_jobs);
    //calculate the v measure score
    auto v_measure = v_measure_score(data,labels,met,sample_size,random_state,n_jobs);


    for ( size_t i = 0; i < homogeneity.size(); i++)
    {
      ret.push_back(homogeneity[i]);
      ret.push_back(completeness[i]);
      ret.push_back(v_measure[i]);
    }
    //return homogeneity_score(labels_true, labels_pred), completeness_score(labels_true, labels_pred), v_measure_score(labels_true, labels_pred)
    return ret;

  }
  std::vector<real_t> cluster_stage::completeness_score(const matrix<real_t>& data, std::vector<size_t> labels, size_t met, size_t sample_size, size_t random_state, size_t n_jobs)
  {
    //return the completeness score
    UNDEF_REFERENCE(met);
    UNDEF_REFERENCE2(sample_size);
    UNDEF_REFERENCE2(random_state);
    UNDEF_REFERENCE2(n_jobs);

    std::vector<real_t> ret;
    //calculate the intra cluster distance
    auto intra_cluster_distance = scores["intra_cluster_distance"];
    auto inter_cluster_distance = scores["inter_cluster_distance"];

    //calculate the intra cluster distance
    auto intra = ((this->*intra_cluster_distance)(data,labels,met,sample_size,random_state,n_jobs));
    //calculate the inter cluster distance
    auto inter = ((this->*inter_cluster_distance)(data,labels,met,sample_size,random_state,n_jobs));

    //calculate the centroids
    matrix<real_t> centroids;
    //calculate the centroids
    centroids.resize(data.cols(),data.cols());
    centroids.fill(0.0);

    //labels for each row of the data matrix represents the cluster number for that row of the data matrix
    //calculate the covariant matrix for each cluster
    matrix<real_t> clustering_feature;
    clustering_feature.resize(data.cols(),data.cols());
    clustering_feature.fill(0.0);
    //fill the clustering_feature matrix
    for ( size_t i = 0; i < data.rows(); i++)
    {
      for ( size_t j = 0; j < data.cols(); j++)
      {
        clustering_feature(labels[i],j) += data(i,j);
      }
    }
    //calculate the centroids
    for ( size_t i = 0; i < data.cols(); i++)
    {
      for ( size_t j = 0; j < data.cols(); j++)
      {
        centroids(i,j) = clustering_feature(i,j);
      }
    }
    //calculate the completeness score
    for ( size_t i = 0; i < intra.size(); i++)
    {
      real_t completeness = intra[i] / inter[i];
      ret.push_back(completeness);
    }
    //return the completeness score
    return ret;

  }
  std::vector<real_t> cluster_stage::v_measure_score(const matrix<real_t>& data, std::vector<size_t> labels, size_t met, size_t sample_size, size_t random_state, size_t n_jobs)
  {
    //return the v measure score
    UNDEF_REFERENCE(met);
    UNDEF_REFERENCE2(sample_size);
    UNDEF_REFERENCE2(random_state);
    UNDEF_REFERENCE2(n_jobs);

    std::vector<real_t> ret;
    //calculate the homogeneity score
    auto homogeneity = homogeneity_score(data,labels,met,sample_size,random_state,n_jobs);
    //calculate the completeness score
    auto completeness = completeness_score(data,labels,met,sample_size,random_state,n_jobs);

    //calculate the v measure score
    for ( size_t i = 0; i < homogeneity.size(); i++)
    {
      real_t v_measure = 2 * (homogeneity[i] * completeness[i]) / (homogeneity[i] + completeness[i]);
      ret.push_back(v_measure);
    }
    //return the v measure score
    return ret; 
  }
  std::vector<real_t> cluster_stage::fowlkes_mallows_score(const matrix<real_t>& data, std::vector<size_t> labels, size_t met, size_t sample_size, size_t random_state, size_t n_jobs)
  {
    //return the fowlkes mallows score
    UNDEF_REFERENCE(met);
    UNDEF_REFERENCE2(sample_size);
    UNDEF_REFERENCE2(random_state);
    UNDEF_REFERENCE2(n_jobs);



    std::vector<real_t> ret;
    //false positive
    real_t fp = 0.0;
    //false negative
    real_t fn = 0.0;
    //true positive
    real_t tp = 0.0;

    //calculate the intra cluster distance
    auto intra_cluster_distance = scores["intra_cluster_distance"];
    auto inter_cluster_distance = scores["inter_cluster_distance"];

    //calculate the intra cluster distance
    auto intra = ((this->*intra_cluster_distance)(data,labels,met,sample_size,random_state,n_jobs));
    //calculate the inter cluster distance
    auto inter = ((this->*inter_cluster_distance)(data,labels,met,sample_size,random_state,n_jobs));
    //calculate the centroids
    matrix<real_t> centroids;
    //calculate the centroids
    centroids.resize(data.cols(),data.cols());
    centroids.fill(0.0);
    //labels for each row of the data matrix represents the cluster number for that row of the data matrix
    //calculate the covariant matrix for each cluster
    matrix<real_t> clustering_feature;
    clustering_feature.resize(data.cols(),data.cols());
    clustering_feature.fill(0.0);
    //fill the clustering_feature matrix
    //fill the clustering_feature matrix
    for ( size_t i = 0; i < data.rows(); i++)
    {
      for ( size_t j = 0; j < data.cols(); j++)
      {
        clustering_feature(labels[i],j) += data(i,j);
      }
    }
    //calculate the centroids
    for ( size_t i = 0; i < data.cols(); i++)
    {
      for ( size_t j = 0; j < data.cols(); j++)
      {
        centroids(i,j) = clustering_feature(i,j);
      }
    }
    //calculate the false positive
    for ( size_t i = 0; i < intra.size(); i++)
    {
      fp += intra[i];
    }
    //calculate the false negative
    for ( size_t i = 0; i < inter.size(); i++)
    {
      fn += inter[i];
    }
    //calculate the true positive
    for ( size_t i = 0; i < centroids.rows(); i++)
    {
      for ( size_t j = 0; j < centroids.cols(); j++)
      {
        tp += centroids(i,j);
      }
    }
    //calculate the fowlkes mallows score
    real_t fowlkes_mallows = sqrt((tp * tp) / ((tp + fp) * (tp + fn)));
    //return the fowlkes mallows score
    ret.push_back(fowlkes_mallows);
    //return the fowlkes mallows score
    return ret;

  }//end of fowlkes mallows score
  std::vector<real_t> cluster_stage::mutual_info_score(const matrix<real_t>& data, std::vector<size_t> labels, size_t met, size_t sample_size, size_t random_state, size_t n_jobs)
  {
        //return the mutual info score
        UNDEF_REFERENCE(met);
        UNDEF_REFERENCE2(sample_size);
        UNDEF_REFERENCE2(random_state);
        UNDEF_REFERENCE2(n_jobs);
        UNDEF_REFERENCE2(labels);
        //return the mutual info score
        std::vector<real_t> ret;
        //calculate the mutual info score
        for ( size_t i = 0; i < data.cols(); i++)
        {
          for ( size_t j = 0; j < data.cols(); j++)
          {
            real_t mutual_info = 0.0;
            //calculate the mutual info score
            for ( size_t k = 0; k < data.rows(); k++)
            {
              mutual_info += data(k,i) * data(k,j) * std::log(data(k,i) / (data(k,i) * data(k,j)));
            }
            //push the mutual info score
            ret.push_back(mutual_info);
          }
        }
        //return the mutual info score
        return ret; 
      }
      std::vector<real_t> cluster_stage::normalized_mutual_info_score(const matrix<real_t>& data, std::vector<size_t> labels, size_t met, size_t sample_size, size_t random_state, size_t n_jobs)
      {
        std::vector<real_t> ret;
        //calculate the mutual info score
        auto mutual_info = mutual_info_score(data,labels,met,sample_size,random_state,n_jobs);
        //calculate the entropy
        auto entrop = data.row_entropy();
        //calculate the mutual info score
        for ( size_t i = 0; i < mutual_info.size(); i++)
        {
          real_t normalized_mutual_info = mutual_info[i] / entrop[i];
          //push the normalized mutual info score
          ret.push_back(normalized_mutual_info);
        }
        //return the normalized mutual info score
        return ret;

      }
      std::vector<real_t> cluster_stage::rand_score(const matrix<real_t>& data, std::vector<size_t> labels, size_t contingency,size_t sample_size, size_t random_state, size_t n_jobs) //ignore parameters 
      {
        //return the rand score
        UNDEF_REFERENCE(contingency);
        UNDEF_REFERENCE2(sample_size);
        UNDEF_REFERENCE2(random_state);
        UNDEF_REFERENCE2(n_jobs);
        UNDEF_REFERENCE2(labels);
        //return the rand score
        std::vector<real_t> ret;
        //calculate the rand score
        for ( size_t i = 0; i < data.cols(); i++)
        {
          for ( size_t j = 0; j < data.cols(); j++)
          {
            real_t rand_score = 0.0;
            //calculate the rand score
            for ( size_t k = 0; k < data.rows(); k++)
            {
              rand_score += data(k,i) * data(k,j);
            }
            //push the rand score
            ret.push_back(rand_score);
          }
        }
        //return the rand score
        return ret;   
  }//end of rand score
  //homogeneity_score
  std::vector<real_t> cluster_stage::homogeneity_score( const matrix<real_t>& data , std::vector<size_t> labels, size_t met, size_t sample_size, size_t random_state, size_t n_jobs)
  {
    //return the homogeneity score
    UNDEF_REFERENCE(met);
    UNDEF_REFERENCE2(sample_size);
    UNDEF_REFERENCE2(random_state);
    UNDEF_REFERENCE2(n_jobs);
    UNDEF_REFERENCE2(labels);
    //return the homogeneity score
    std::vector<real_t> ret;
    //calculate the homogeneity score
    for ( size_t i = 0; i < data.cols(); i++)
    {
      for ( size_t j = 0; j < data.cols(); j++)
      {
        real_t homogeneity_score = 0.0;
        //calculate the homogeneity score
        for ( size_t k = 0; k < data.rows(); k++)
        {
          homogeneity_score += data(k,i) * data(k,j) / data(k,i);
        }
        //push the homogeneity score
        ret.push_back(homogeneity_score);
      }
    }
    //return the homogeneity score
    return ret;   
  }//end of homogeneity score 

   
  //load and save additional data for datasets :
    void dataset_stage::save_additional_data(std::string& data)
    {
        UNDEF_REFERENCE(data);
        UNDEF_REFERENCE2(data);
    }
    //load and save additional data for clusters :
    void cluster_stage::load_additional_data(const std::string& data)
    {
        UNDEF_REFERENCE(data);
        UNDEF_REFERENCE2(data);
    }
    void cluster_stage::save_additional_data(std::string& data)
    {
        UNDEF_REFERENCE(data);
        UNDEF_REFERENCE2(data);
    }
    const std::string dataset_stage::get_additional_data()const
    {
      std::string ret;
      ret = "[";
      for ( size_t i = 0; i < additional_data.size(); i++)
      {
        ret += additional_data[i];
        if ( i != additional_data.size() - 1)
        {
          ret += ",";
        }
      }
      ret += "]";
      return ret;

    }
    void dataset_stage::process_data(  const std::vector<std::vector<real_t>>& data )
    {
        _data.resize(data.size(),data[0].size());
        //process the data
        for ( size_t i = 0; i < data.size(); i++)
        {
          for ( size_t j = 0; j < data[i].size(); j++)
          {
            _data(i,j) = data[i][j];
          }
        }

        process_data(_data);
    } 
    void dataset_stage::set_additional_data(const std::string& data)
    {
        this->additional_data.push_back(data);  
    }
    void dataset_stage::process_data( const std::string& data)
    {
        UNDEF_REFERENCE(data);
        UNDEF_REFERENCE2(data);

        //load from file 'data' 
        //create a matrix of data
        //
        //process the data and if labels exists , then process the labels

        std::ifstream file(data);
        std::string line;
        std::vector<std::vector<real_t>> data_;
        while (std::getline(file, line))
        {
          std::vector<real_t> row;
          std::stringstream iss(line);
          std::string val;
          while (std::getline(iss, val, ','))
          {
            row.push_back(std::stod(val));
          }
          data_.push_back(row);
        }
        //process the data
        process_data(data_);
    }
     void dataset_stage::process_data(  matrix_base& data )
    {
        //matrix   converts to and from matrix_base
        _data = data;
        //process the data
        process_data(_data);
    } 
    
    void dataset_stage::process_data(const matrix<real_t>& data )
    {
        this->_data = data;
        //process the data
        UNDEF_REFERENCE(data);
        //initialize local variables of the data
      this->_number_of_samples = data.rows(); 
      this->_number_of_classes = 0;
      this->_number_of_labels = 0;
      this->_number_of_attributes = 0;
      this->_num_of_clusters = 0;
      this->_num_of_outliers = 0;
      this->_num_of_noise = 0;
      this->_num_of_features = 0;
      this->_num_of_unlabelled = 0;
      this->_num_of_labelled = 0;
      this->_num_of_test = 0;
      this->_num_of_train = 0;
      this->_num_of_validate = 0;
      this->_num_of_optimize = 0;
      this->_num_of_xvalidate = 0;
      this->_num_of_build_train = 0;
      this->_num_of_optimize_train = 0;
      this->_num_of_optimize_test = 0;
      //iterate over the data once to calculate the number of classes
      //and the number of attributes, and the number of labels
      for ( size_t i = 0; i < data.rows(); i++)
      {
        //get the label
        size_t label = data(i,data.cols() - 1);
        //check if the label is not in the labels
        if ( std::find(_labels.begin(),_labels.end(),label) == _labels.end())
        {
          //push the label
          _labels.push_back(label);
          //increment the number of labels
          this->_number_of_labels++;
        }
        //check if the label is not in the classes
        if ( std::find(_classes.begin(),_classes.end(),std::to_string(label)) ==_classes.end())
        {
          //push the label
          _classes.push_back(std::to_string(label));
          //increment the number of classes
          this->_number_of_classes++;
        }
        //iterate over the features
         
      } 
       //second iteration - calculate the number of samples in each class
      for ( size_t i = 0; i < data.rows(); i++)
      {
        //get the label
        size_t label = data(i,data.cols() - 1);
        //get the class
        size_t class_ = data(i,data.cols() - 1);
        //increment the number of samples in the class
        //
        this->number_of_samples_per_class[class_]++;
        //check if the label is not in the classes
        if ( std::find(_classes.begin(),_classes.end(),std::to_string(label)) ==_classes.end())
        {
          //push the label
          _classes.push_back(std::to_string(label));
          //increment the number of classes
          this->_number_of_classes++;
        }
        //iterate over the features
        
      }
      //calculate the number of samples
      this->_number_of_samples = data.rows();
      //calculate the number of features
      this->_num_of_features = data.cols() - 1;
  

      //calculate the number of samples
      this->_number_of_samples = data.rows();
      //calculate the noise
      this->_num_of_noise = 0;
      //noisy samples are those samples that are not in the classes
      //iterate over the samples
      for ( size_t i = 0; i < data.rows(); i++)
      {
        //get the label
        size_t label = data(i,data.cols() - 1);
        //check if the label is not in the classes
        if ( std::find(_classes.begin(),_classes.end(),std::to_string(label)) ==_classes.end())
        {
          //increment the number of noise
          this->_num_of_noise++;
        }
      } 
      //calculate the number of clusters
      this->_num_of_clusters = 0;
      
      //calculate the number of outliers
      this->_num_of_outliers = 0;
      //optimize
      this->_num_of_optimize = 0;
      //validate
      this->_num_of_validate = 0;
      //test
      

      //calculate the number of unlabelled
      this->_num_of_unlabelled = 0;
      //calculate the number of labelled
      this->_num_of_labelled = 0;
      
    }
    //process the data
    void dataset_stage::process_data( const std::vector<real_t>& data )
    {
        //in order to transform the data into a matrix 
        //we need to know the number of features and the number of samples 
        //the number of features is the size of the vector - 1
        //the number of samples is the size of the vector 
        //create temporary matrix m and fill it with the data 
        
        //find the number of samples and features from the sum of attributes 

        size_t number_of_samples = 0;
        size_t number_of_features = 0;
        size_t symmetric = 0;
        size_t asymetric = 0;
        //create a symmetric matrix from the size of the vector and fill it with its value.
        //a symmetric matrix would contain the number of samples and the number of features 
        //the number of samples is the size of the vector 
        //the number of features is the size of the vector - 1

        size_t size_root = std::sqrt(data.size()); 
        //check if the size is a perfect square
        if ( size_root * size_root == data.size())
        {
            //the size is a perfect square
            //the number of samples is the size of the vector 
            //the number of features is the size of the vector - 1
            number_of_samples = size_root;
            number_of_features = size_root - 1;
            symmetric = 1;
        }
        else
        {
            //the size is not a perfect square
            //the number of samples is the size of the vector 
            //the number of features is the size of the vector - 1
            number_of_samples = data.size();
            number_of_features = data.size() - 1;
            asymetric = 1;
        } 

        //create a matrix from the size of the vector and fill it with its value.
        //a matrix would contain the number of samples and the number of features

        matrix<real_t> m(number_of_samples,number_of_features); 
        //fill the matrix with the data
        for ( size_t i = 0; i < number_of_samples; i++)
        {
            for ( size_t j = 0; j < number_of_features; j++)
            {
              //if symmetric then fill the matrix with the data 
              //if asymetric then fill the matrix with the data 
              if ( symmetric == 1)
              {

                //fill the matrix with the data
                m(i,j) = data[i * number_of_features + j];
              }
              else if ( asymetric == 1)
              {
                //fill the matrix with the data
                m(i,j) = data[i * number_of_features + j];
              } 
              else
              {
                //fill the matrix with the data
                m(i,j) = data[i * number_of_features + j];
              }
             }
        } 
        //iterate over the data
        //and try to find more information(labels,classes,samples,features,attributes)
        process_data(_data);
    } //process_data

    //dataset_stage load data helpers :
    /*    
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
    */
    //dataset_stage load data helpers :
    matrix<real_t> dataset_stage::load_txt(const std::string& file_name)
    {
      matrix<real_t> ret;
      std::vector<std::vector<real_t> > collector;
      std::vector<std::string> headers;
      //open the file
      std::ifstream file(file_name);
      //check if the file is open
      if ( file.is_open())
      {
        //read the file
        std::string line;
        //iterate over the file
        while ( std::getline(file,line))
        {
          //check the line: 
          if(!std::none_of(line.begin(),line.begin()+1 ,[](char x ) { return !((x=='#')||(x=='@')||(x=='?')||(x)=='%'||(x)=='$');}) )
          {
            headers.push_back(line);
            //skip the line,save as headers 
            continue;
          }
          //create a vector
          std::vector<real_t> v;
          //split the line
          std::vector<std::string> tokens;
          //split the line
          tokenize(line,tokens," ,\t");
          //iterate over the tokens
          for ( size_t i = 0; i < tokens.size(); i++)
          {
            //convert the token to a real_t
            real_t value = std::stod(tokens[i]);
            //push the value to the vector
            v.push_back(value);
          }
          //push the vector to the matrix
          collector.push_back(v);
        }
      } 
      //close the file
      file.close();
      //check if the collector is not empty
      if ( collector.size() > 0)
      {
        //get the number of samples
        size_t number_of_samples = collector.size();
        //get the number of features
        size_t number_of_features = collector[0].size();
        //create a matrix
        ret.resize(number_of_samples,number_of_features);
        //fill the matrix
        for ( size_t i = 0; i < number_of_samples; i++)
        {
          for ( size_t j = 0; j < number_of_features; j++)
          {
            //fill the matrix
            ret(i,j) = collector[i][j];
          }
        }
      }
      //return the matrix
      return ret;
    }
    matrix<real_t> dataset_stage::load_csv(const std::string& file_name) 
    {
      matrix<real_t> ret;
      std::vector<sparse_ix> labels ;

      std::vector<std::vector<real_t> > collector;
      //open the file
      std::ifstream file(file_name);
      //check if the file is open
      if ( file.is_open())
      {
        //read the file
        std::string line;
        //iterate over the file
        while ( std::getline(file,line))
        {
          //create a vector
          std::vector<real_t> v;
          //split the line
          std::vector<std::string> tokens;
          //split the line
          tokenize(line,tokens,",");
          //iterate over the tokens
          for ( size_t i = 0; i < tokens.size(); i++)
          {
            //convert the token to a real_t
            real_t value = std::stod(tokens[i]);
            //push the value to the vector
            v.push_back(value);
          }
          //push the vector to the matrix
          collector.push_back(v);
        }
      }   
      //close the file
      file.close();
      //check if the collector is not empty
      if ( collector.size() > 0)
      {
        //get the number of samples
        size_t number_of_samples = collector.size();
        //get the number of features
        size_t number_of_features = collector[0].size();
        //create a matrix
        ret.resize(number_of_samples,number_of_features);
        //fill the matrix
        for ( size_t i = 0; i < number_of_samples; i++)
        {
          for ( size_t j = 0; j < number_of_features; j++)
          {
            //fill the matrix
            ret(i,j) = collector[i][j];
          }
        }
      } 
      //return the matrix
      return ret;
      
    }
    matrix<real_t> dataset_stage::load_libsvm (const std::string& file_name)
    {
      //create a matrix
      matrix<real_t> ret;
      std::vector<std::vector<real_t>> collector;
      //open the file
      std::ifstream file(file_name);
      //check if the file is open
      if ( file.is_open())
      {
        //read the file
        std::string line;
        //iterate over the file
        while ( std::getline(file,line))
        {
          //create a vector
          std::vector<real_t> v;
          //split the line
          std::vector<std::string> tokens;
          //split the line
          tokenize(line,tokens," \t");
          //iterate over the tokens
          for ( size_t i = 0; i < tokens.size(); i++)
          {
            //check if the token is not empty
            if ( !tokens[i].empty())
            {
              //push the token
              v.push_back(std::stod(tokens[i]));
            }
          }
          //push the vector
          collector.push_back(v);
         }
         //close the file
          file.close();
          //update the matrix
          ret.resize(collector.size(),collector[0].size());
          //fill the matrix
          for ( size_t i = 0; i < collector.size(); i++)
          {
            for ( size_t j = 0; j < collector[i].size(); j++)
            {
              //fill the matrix
              ret(i,j) = collector[i][j];
            }
          } 
          //return the matrix
      } //end of if
      //return the matrix
     return ret;

    }//end of load_libsvm
  //load the data
   matrix<real_t> dataset_stage::load_mxnet(const std::string& file_name)
   {
     //return csv load for now 
      return load_csv(file_name);
   }

  //torch
  matrix<real_t> dataset_stage::load_torch(const std::string& file_name)
  {
    //return csv load for now 
    return load_csv(file_name);
  }
  //tf-record
  matrix<real_t> dataset_stage::load_tfrecord(const std::string& file_name)
  {
    //return csv load for now 
    return load_csv(file_name);
  }

  //TSV
  matrix<real_t> dataset_stage::load_tsv(const std::string& file_name)
  {
    //return csv load for now 
    return load_csv(file_name);
  }
  
  //CAFFE2
  matrix<real_t> dataset_stage::load_caffe2(const std::string& file_name)
  {
    //return csv load for now 
    return load_csv(file_name);
  } 
  //NPY
  matrix<real_t> dataset_stage::load_npy(const std::string& file_name)
  {
    //return csv load for now 
    return load_csv(file_name);
  } 
  //load_hdf5
  matrix<real_t> dataset_stage::load_hdf5(const std::string& file_name)
  {
    //return csv load for now 
    return load_csv(file_name);
  }

  matrix<real_t>  dataset_stage::load_matrix(const std::string& file_name)
  {
    //load a raw matrix from the file.
    //create a matrix
    matrix<real_t> ret;
    std::vector<std::vector<real_t>> collector;
    //open the file
    std::ifstream file(file_name);
    //check if the file is open
    if ( file.is_open())
    { 
      //read the file
      std::string line;
      //iterate over the file
      while ( std::getline(file,line))
      {
        //create a vector
        std::vector<real_t> v;
        //split the line
        std::vector<std::string> tokens;
        //split the line
        tokenize(line,tokens,",");
        //iterate over the tokens
        for ( size_t i = 0; i < tokens.size(); i++)
        {
          //convert the token to a real_t
          real_t value = std::stod(tokens[i]);
          //push the value to the vector
          v.push_back(value);
        }
        //push the vector to the matrix
        collector.push_back(v);
      } 

      //close the file
      file.close();
      //copy the data to the matrix
      ret.resize(collector.size(),collector[0].size());
      //fill the matrix
      for ( size_t i = 0; i < collector.size(); i++)
      {
        for ( size_t j = 0; j < collector[i].size(); j++)
        {
          //fill the matrix
          ret(i,j) = collector[i][j];
        }
      } 
      
      //return the matrix
    } //end of if   

    return ret;

  }
    matrix<real_t> dataset_stage::load_dataset(const std::string& file_name)
    {
      //load the dataset
      //create a matrix
      matrix<real_t> ret;
      
      //get the extension
      std::string extension = file_name.substr(file_name.find_last_of(".") + 1);  
      //check the extension

      if ( extension == "csv")
      {
        //load the csv
        ret = load_csv(file_name);
      }
      else if ( extension == "libsvm")
      {
        //load the libsvm
        ret = load_libsvm(file_name);
      }
      else if ( extension == "mxnet")
      {
        //load the mxnet
        ret = load_mxnet(file_name);
      }
      else if ( extension == "torch")
      {
        //load the torch
        ret = load_torch(file_name);
      }
      else if ( extension == "tfrecord")
      {
        //load the tfrecord
        ret = load_tfrecord(file_name);
      }
      else if ( extension == "tsv")
      {
        //load the tsv
        ret = load_tsv(file_name);
      }
      else if ( extension == "caffe2")
      {
        //load the caffe2
        ret = load_caffe2(file_name);
      }
      else if ( extension == "npy")
      {
        //load the npy
        ret = load_npy(file_name);
      }
      else if ( extension == "hdf5")
      {
        //load the hdf5
        ret = load_hdf5(file_name);
      }
      else if ( extension == "matrix")
      {
        //load the matrix
        ret = load_matrix(file_name);
      }
      else
      {
        //throw an error
        throw std::runtime_error("Unsupported file format");
      }
      //return the matrix
      return ret; 
    }
    matrix<real_t> dataset_stage::load_data(const std::string& file_name)
    {
      //load the data
      return dataset_stage::load_dataset(file_name);  
    }
    matrix<real_t> dataset_stage::load(const std::string& file_name)
    {
      //load the data
      return dataset_stage::load_dataset(file_name);
    }


    //one-hot helper class implementation
    //constructor:
    one_hot_vectorizer::one_hot_vectorizer():vectorizer<std::string,real_t>(ONE_HOT_VECTORIZER) 
    {
      //initialize the one-hot vectorizer
      this->initialize();
    }
    //destructor:
    one_hot_vectorizer::~one_hot_vectorizer()
    {
      //clear the one-hot vectorizer
      _vocabulary.clear();

      this->clear();
    }
    //initialize the one-hot vectorizer
    void one_hot_vectorizer::initialize()
    {
      //initialize the one-hot vectorizer
      //clear the one-hot vectorizer
      this->clear();
    }
    //clear the one-hot vectorizer
    void one_hot_vectorizer::clear()
    {
       this->_bow.clear(); //clear the bag of words
      
      //clear the one-hot vectorizer
    }
    //fit the one-hot vectorizer to the data
    //transform the one-hot vectorizer
    //fit_transform the one-hot vectorizer

    //fit the one-hot vectorizer to the data
    //add a single document add_document(const std::string& document)

    void one_hot_vectorizer::add_document(const std::string& doc)
    {
      //add a single document to the one-hot vectorizer
      this->_bow.add_document(doc);
    }
    //fit a single document : 
    std::vector<real_t> one_hot_vectorizer::fit(const std::string& single_doc )
    {
        std::vector<real_t> ret;
        //fit the data using the bag of words and update local variables
        this->_bow.process_document(single_doc);
        //copy the vocabulary
        this->_vocabulary = this->_bow.get_vocabulary();
        //get the bag of words
        std::vector<real_t> bow = this->_bow.get_bag_of_words();
        //get the number of documents
        this->num_docs = this->_bow.get_number_of_documents();
        //get the number of words
        this->num_words = this->_bow.get_number_of_words();
        //get the number of unique words
        this->num_unique_tokens = this->_bow.get_number_of_unique_tokens();
        //  fill ret
        ret.resize(this->num_unique_tokens);
        //fill ret
        for ( size_t i = 0; i < this->num_unique_tokens; i++)
        {
          //fill ret
          ret[i] = bow[i];
        }

        _transformed_data = ret;
         //return ret
        return ret;

    }
    //process_documents 
    void one_hot_vectorizer::process_documents (const std::vector<std::string>& documents )
    {
      //process the documents
      this->_bow.process_documents(documents);
      //calculate the number of documents
      this->num_docs = this->_bow.get_number_of_documents();
      //calculate the number of words
      this->num_words = this->_bow.get_number_of_words();
      //calculate the number of unique tokens
      this->num_unique_tokens = this->_bow.get_number_of_unique_tokens();
      //copy the vocabulary
      this->_vocabulary = this->_bow.get_vocabulary();
      //get the bag of words
      _fitted_data = this->_bow.get_bag_of_words();
      //fill the transformed data
      
      //transform the data

      //one-hot the data
      //transform bow to one-hot vector:
      //create a one-hot vector
      std::vector<real_t> ret;
      //loop over the bag of words
      for ( size_t i = 0; i < _fitted_data.size(); i++)
      {
        //check if the word is in the vocabulary
        if ( _fitted_data[i] > 0.0)
        {
          //set the one-hot vector
          ret[i] = 1.0;
          

        }  
        else
        {
          ret[i]=0;
        }
      }   
      //set the transformed data
      _transformed_data = ret;
       
    }

    //inverse transform 
    std::vector<std::string> one_hot_vectorizer::inverse_transform(const matrix<real_t>& values )
    {

        return this->_bow.inverse_transform(values);
    }
    //transform the one-hot vectorizer from vector of strings

    std::vector<std::string> one_hot_vectorizer::inverse_transform(const std::vector<real_t>& values )
    {
         return {this->_bow.inverse_transform(values)};
    }

    std::vector<real_t> one_hot_vectorizer::fit( const std::vector<std::string>& data_)
    {
      std::vector<real_t> ret; 
      //fit the data using the bag of words and update local variables 
      //
      this->_bow.process_documents(data_);

      //copy the vocabulary 
      this->_vocabulary = this->_bow.get_vocabulary(); 
      //get the bag of words
       std::vector<real_t> bow =this->_bow.get_bag_of_words();
      //get the number of documents
      this->num_docs = this->_bow.get_number_of_documents(); 
      //get the number of words
      this->num_words = this->_bow.get_number_of_words();
      //get the number of unique words
      this->num_unique_tokens = this->_bow.get_number_of_unique_tokens();
      //
      //get the one-hot vector
      //transform bow to one-hot vector:
      //create a one-hot vector
      ret.resize(bow.size(),0.0);
      //loop over the bag of words
      for ( size_t i = 0; i < bow.size(); i++)
      {
        //check if the word is in the vocabulary
        if ( bow[i] > 0.0)
        {
          //set the one-hot vector
          ret[i] = 1.0;
          

        }  
        else
        {
          ret[i]=0;
        }
       } 
       _transformed_data = ret;
      //return the one-hot vector
      return ret;
    }


    std::vector<real_t>  one_hot_vectorizer::fit( const provallo::matrix<real_t>&data_ )
    {
      std::vector<real_t> ret; 
      //fit the one hot matrix
      //-
      //get the number of documents
      this->num_docs = data_.rows();
      //get the number of words
      this->num_words = data_.cols();
      //get the number of unique words
      this->num_unique_tokens = data_.cols();
      //get the vocabulary
      this->_vocabulary = this->_bow.get_vocabulary();
      //get the bag of words
      std::vector<real_t> bow = this->_bow.get_bag_of_words();
      //get the one-hot vector
      //transform bow to one-hot vector:
      //create a one-hot vector
      ret.resize(this->num_unique_tokens,0.0);
      //loop over the bag of words
      for ( size_t i = 0; i < bow.size(); i++)
      {
        //check if the word is in the vocabulary
        if ( bow[i] > 0.0)
        {
          //set the one-hot vector
          ret[i] = 1.0;
          

        }  
        else
        {
          ret[i]=0;
        }
      }
       _transformed_data = ret;

      //return the one-hot vector
      return ret;

    }

    std::vector<std::vector<real_t>> one_hot_vectorizer::fit( const std::vector<std::vector<std::string>>& data_)
    {
        std::vector<std::vector<real_t> > ret;

        for ( auto& document  : data_ )
        {
            //  fit the one-hot vectorizer to the data
            std::vector<real_t> one_hot_vector = this->fit(document); 
            //push the one-hot vector to the vector
            ret.push_back(one_hot_vector);
            this->num_docs++;   
        }
        return ret;
    }
    //predict single source 
    std::vector<real_t> one_hot_vectorizer::predict(const std::string& source)
    {
       //predict the source
        if(source.empty())
        {
          std::cout<<"[-] source is empty"<<std::endl;
          return {};
        }
        //predict the source
        std::vector<real_t> ret = this->_bow.predict(source);
        if(ret.size()==0)
        {
          //initialize to token size of source
          std::vector<std::string> tokens;
          tokenize(source,tokens," ,\t");
          ret.resize(tokens.size(),0.0);
          for ( size_t i = 0; i < tokens.size(); i++)
          {
            //set the one-hot vector
            auto token = reduce  (tokens[i],"");

            if( std::find(_vocabulary.begin(),_vocabulary.end(),token) != _vocabulary.end())
            {
              ret[i] = 1.0;
            }
            else
            {

              ret[i] = 0.0;
            }
          }
        }
        return ret;
 
    }//end of predict
    //
    /* fit_transform the one-hot vectorizer
    std::vector<std::vector<real_t>> one_hot_vectorizer::fit_transform( const std::vector<std::string>& data_)
    {
        std::vector<std::vector<real_t> > ret;

        for ( auto& document  : data_ )
        {
            //  fit the one-hot vectorizer to the data
            std::vector<real_t> one_hot_vector = this->fit(document); 
            //push the one-hot vector to the vector
            ret.push_back(one_hot_vector);
            this->num_docs++;   
        }
        return ret;
    } 
    //fit_transform the one-hot vectorizer
    std::vector<std::vector<real_t>> one_hot_vectorizer::fit_transform( const provallo::matrix<real_t>& data_)
    {
        std::vector<std::vector<real_t> > ret;

        for ( auto& document  : data_ )
        {
            //  fit the one-hot vectorizer to the data
            std::vector<real_t> one_hot_vector = this->fit(document); 
            //push the one-hot vector to the vector
            ret.push_back(one_hot_vector);
            this->num_docs++;   
        }
        return ret;
    }
    //fit_transform the one-hot vectorizer
    std::vector<std::vector<real_t>> one_hot_vectorizer::fit_transform( const std::vector<std::vector<std::string>>& data_)
    {
        std::vector<std::vector<real_t> > ret;

        for ( auto& document  : data_ )
        {
            //  fit the one-hot vectorizer to the data
            std::vector<real_t> one_hot_vector = this->fit(document); 
            //push the one-hot vector to the vector
            ret.push_back(one_hot_vector);
            this->num_docs++;   
        }
        return ret;
    }*/
    //fit_transform the one-hot vectorizer
     std::vector<real_t> one_hot_vectorizer::fit_transform( const std::vector<std::string>& data_)
    {
        std::vector<real_t> ret;

        for ( auto& document  : data_ )
        {
            //  fit the one-hot vectorizer to the data
            std::vector<real_t> one_hot_vector = this->fit(document); 
            //push the one-hot vector to the vector
            for ( auto& val : one_hot_vector)
                ret.push_back(val);
             this->num_docs++;   
        }
        //return the one-hot vector
           
        return ret;

    }
    //fit transform matrix<real_t>     virtual  std::vector<real_t> fit_transform(const provallo::matrix<real_t>& data_);//{ DEFAULT_IMPL(data_);}  

std::vector<real_t>  one_hot_vectorizer::fit_transform  (const matrix<real_t>& data)
{
  std::vector<real_t> ret;
  //fit the one hot matrix
  //-
  //get the number of documents
  this->num_docs = data.rows();
  //get the number of words
  this->num_words = data.cols();
  //get the number of unique words
  this->num_unique_tokens = data.cols();
  //get the vocabulary
  this->_vocabulary = this->_bow.get_vocabulary();
  //get the bag of words
  std::vector<real_t> bow = this->_bow.get_bag_of_words();
  //get the one-hot vector
  //transform bow to one-hot vector:
  //create a one-hot vector
  ret.resize(this->num_unique_tokens,0.0);
  //loop over the bag of words
  for ( size_t i = 0; i < bow.size(); i++)
  {
    //check if the word is in the vocabulary
    if ( bow[i] > 0.0)
    {
      //set the one-hot vector
      ret[i] = 1.0;
      

    }  
    else
    {
      ret[i]=0;
    }
  }
  //return the one-hot vector
  return ret;

}
 

    // document processing : 
    void one_hot_vectorizer::process_document(const std::string& document)
    {
      //use bag of words results  to calculate onehot 
      //values :
      this->_bow.process_document (document );
      //
      //get the number of words
      //size_t num_words = this->_bow.num_words();
      //get the number of documents
      //size_t num_docs = this->_bow.num_documents();
      //get the number of classes
      //size_t num_classes = this->_bow.num_classes();
      //get the number of features
      //size_t num_features = this->_bow.num_features();
      //get the number of samples
      //size_t num_samples = this->_bow.num_samples();
      //get the number of tokens
      //size_t num_tokens = this->_bow.num_tokens();

      //begin: 
      //get the number of rows
      const matrix<real_t> & X = _bow.get_matrix()  ;

      size_t rows = X.rows();
      //get the number of columns
      size_t cols = X.cols();
      //create the one-hot matrix
      matrix<real_t> ret(rows,cols * this->num_classes); 
      //iterate over the rows
      for ( size_t i = 0; i < rows; i++)
      {
        //get the class
        size_t cls = X(i,cols-1);
        //iterate over the columns
        for ( size_t j = 0; j < cols; j++)
        {
          //get the value
          real_t value = X(i,j);
          //get the index
          size_t index = j * this->num_classes + cls;
          //set the value
          ret(i,index) = value;
        }
      } 
      //end:
      //set the one-hot matrix
      this->_matrix = ret;
      return;
     } 
    
    //predict the one-hot vectorizer
    std::vector<real_t> one_hot_vectorizer::predict(const std::vector<std::string>& documents)
    {
        //get the one-hot vector
        std::vector<real_t> one_hot_vector = this->fit_transform(documents); 
        //return the one-hot vector
        return one_hot_vector;
    }
    //predict the one-hot vectorizer
    std::vector<real_t> one_hot_vectorizer::predict(const matrix<real_t>& data)
    {
        //get the one-hot vector
        std::vector<real_t> one_hot_vector = this->fit_transform(data); 
        //return the one-hot vector
        return one_hot_vector;
    }
    //transform the one-hot vectorizer 
    std::vector<real_t> one_hot_vectorizer::transform(const std::vector<std::string>& documents)
    {
        //get the one-hot vector
        std::vector<real_t> one_hot_vector = this->fit_transform(documents); 
        //return the one-hot vector
        return one_hot_vector;  
    }
    //transform the one-hot vectorizer
    std::vector<real_t> one_hot_vectorizer::transform(const matrix<real_t>& data)
    {
        //get the one-hot vector
        std::vector<real_t> one_hot_vector = this->fit_transform(data); 
        //return the one-hot vector
        return one_hot_vector;  
    }

    
    //complete the one-hot helper class implementation
    



    void vectorizer_stage::initialize()
    {
     
        //fill in the vectorizers 

        vectorizers.push_back(new tfidf_vectorizer);
        vectorizers.push_back(new one_hot_vectorizer); 
        vectorizers.push_back(new pca_vectorizer); 

        //vectorizers.push_back(std::make_shared<lda_vectorizer>()); 
        //vectorizers.push_back(std::make_shared<word2vec_vectorizer>()); 

    }
    vectorizer_stage::~vectorizer_stage()
    {
    }
    //complete the vectorizer_stage implementation
    void bag_of_words::process_documents (const std::vector<std::string>& documents )
    {
          for ( auto& doc : documents )
            process_document(doc);
        //  this->number_of_documents += documents.size();

    }
    //global ifstream/ofstream vector<real_t> operators 
    std::ifstream& operator >>  (std::ifstream& iff,std::vector<real_t>& in)
    {
      size_t size=0;
      std::istringstream ss;
      std::string line,tmp="size:";

      size_t index=0;
      
      //get line 
      std::getline(iff, line);
      //parse line
      if(line.length()<7) 
        throw std::runtime_error("wrong file format. vectors must declare size");
      
      if (line.find(tmp) != std::string::npos)
      {
        std::string scount = line.substr(tmp.size());
        size = std::stoul(scount);
      }
      in.clear();
      in.resize(size);

      while(index<size)
      {
        iff>>tmp;
        if(tmp=="\n") break;
        real_t value = std::stod(tmp);
        in.push_back(value);
        index++;
      }
      return iff;
    }
    std::ofstream& operator << (std::ofstream& out,const std::vector<real_t>&in)
    {
      size_t size = in.size();
      out<<"size:"<<std::to_string(size).c_str()<<std::endl;
      for ( size_t i=0;i<in.size();++i)
      {
          out<<std::to_string(in[i]).c_str()<<" ";
      }
      out<<std::endl;
      return out;
    }
     //global ifstream/ofstream vector<std::string> operators
    std::ifstream& operator >> (std::ifstream& iff,std::vector<std::string>& in)
    {
      size_t size=0;
      std::istringstream ss;
      std::string line,tmp="size:";

      size_t index=0;
      
      //get line 
      std::getline(iff, line);
      //parse line
      if(line.length()<7) 
        throw std::runtime_error("wrong file format. vectors must declare size");
      
      if (line.find(tmp) != std::string::npos)
      {
        std::string scount = line.substr(tmp.size());
        size = std::stoul(scount);
      }
      in.clear();
      in.resize(size);

      while(index<size)
      {
        std::getline(iff, line);
        if(line=="\n") break;
        in.push_back(line);
        index++;
      }
      return iff;
    }
    std::ofstream& operator << (std::ofstream& out,const std::vector<std::string>&in)
    {
      size_t size = in.size();
      out<<"size:"<<std::to_string(size).c_str()<<std::endl;
      for ( size_t i=0;i<in.size();++i)
      {
          out<<in[i].c_str()<<std::endl;
      }
      out<<std::endl;
      return out;
    }
    //global ifstream/ofstream vector<std::pair<std::string,std::string>> operators
    std::ifstream& operator >> (std::ifstream& iff,std::vector<std::pair<std::string,std::string>>& in)
    {
      size_t size=0;
      std::istringstream ss;
      std::string line,tmp="size:";

      size_t index=0;
      
      //get line 
      std::getline(iff, line);
      //parse line
      if(line.length()<7) 
        throw std::runtime_error("wrong file format. vectors must declare size");
      
      if (line.find(tmp) != std::string::npos)
      {
        std::string scount = line.substr(tmp.size());
        size = std::stoul(scount);
      }
      in.clear();
      in.resize(size);

      while(index<size)
      {
        std::getline(iff, line);
        if(line=="\n") break;
        std::string first,second;
        std::istringstream ss(line);
        std::getline(ss,first,' ');
        std::getline(ss,second,' ');
        in.push_back(std::make_pair(first,second));
        index++;
      }
      return iff;
    }
    std::ofstream& operator << (std::ofstream& out,const std::vector<std::pair<std::string,std::string>>&in)
    {
      size_t size = in.size();
      out<<"size:"<<std::to_string(size).c_str()<<std::endl;
      for ( size_t i=0;i<in.size();++i)
      {
          out<<in[i].first.c_str()<<" "<<in[i].second.c_str()<<std::endl;
      }
      out<<std::endl;
      return out;
    }

    //feature_stage missing methods

    //complete the feature_stage implementation
    //process_data
     void feature_stage::process_data(const std::string& data)
    {
        //process the document
        matrix<real_t> m = extract_features(data);
        //concatenate the matrices
        
        this->data = m;
 
    }
     void feature_stage::process_data(const matrix<real_t>& data)
     {
          extract_features(data);
     }
     void feature_stage::process_data(const std::vector<real_t>& data)
     {
          extract_features(data);
     }
     void feature_stage::process_data(const std::vector<std::vector<real_t> >& data)
     {
          extract_features(data);
     }
      void feature_stage::load_additional_data(const std::string& data)
      {
        UNDEF_REFERENCE(data)
        UNDEF_REFERENCE2(data)
        //load data and labels
        //load the feature engineering functor names
        //std::istringstream iss(data);
        //iss>>_functor_names;
        //load the functors
        //for ( size_t i=0;i<_functor_names.size();++i)
        //{
        //          _engineering_functors.push_back(feature_engineering_functor::get(_functor_names[i]));
        //  }
        
      }
      void feature_stage::save_additional_data(std::string& data)
      {
        UNDEF_REFERENCE(data)
        UNDEF_REFERENCE2(data)
        //save data and labels
        //std::ostringstream oss;
        //oss<<_functor_names;
        //data = oss.str();
      }
      //clasifier stage missing implementations:
  
      //constructor 
      classifier_stage::classifier_stage(const std::string& name): factory(nullptr)
      {
        this->name = name;
 
      } 
      classifier_stage::classifier_stage():factory(nullptr) 
      {
        this->name="classifier_stage";
       }  
      classifier_stage::~classifier_stage()
      {
        if(factory)
          delete factory;

      }

      void classifier_stage::load_additional_data(const std::string& data)
      {
        UNDEF_REFERENCE(data)
        UNDEF_REFERENCE2(data)
        //load data and labels
        //load the feature engineering functor names
        //std::istringstream iss(data);
        //iss>>_functor_names;
        //load the functors
        //for ( size_t i=0;i<_functor_names.size();++i)
        //{
        //          _engineering_functors.push_back(feature_engineering_functor::get(_functor_names[i]));
        //  }
        
      }
      void classifier_stage::save_additional_data(std::string& data)
      {
        UNDEF_REFERENCE(data)
        UNDEF_REFERENCE2(data)
        //save data and labels
        //std::ostringstream oss;
        //oss<<_functor_names;
        //data = oss.str();
      }

     /*
    nice to have later
    void feature_stage::process_data(const std::vector<std::string>& documents)
    {
        //process the documents
        std::vector<matrix<real_t>> m;

        for ( auto& doc : documents )
          m.push_back(extract_features(doc));

        //concatenate the matrices
        if (m.size()>0)
        {
          matrix<real_t> tmp(m[0].rows(),m[0].cols());  
          tmp.fill(1.0);

          for(auto& mat : m)
          {
            tmp = tmp* mat;
          }
        }
        this->_data = m;

    }
    */
    //process_data
  //lda_vectorizer implementation
  //constructor:
  lda_vectorizer::lda_vectorizer():vectorizer<std::string,real_t>(LDA_VECTORIZER)
  {
    //initialize the lda vectorizer
    this->initialize();
  }
  //destructor:
  lda_vectorizer::~lda_vectorizer()
  {
    //clear the lda vectorizer
    this->clear();
  }
  //initialize the lda vectorizer
  void lda_vectorizer::initialize()
  {
    //initialize the lda vectorizer
    //clear the lda vectorizer
    this->clear();
  } 
  //clear the lda vectorizer
  void lda_vectorizer::clear()
  {
    //clear the lda vectorizer
    this->_bow.clear();
  }
  //fit the lda vectorizer to the data
  std::vector<real_t> lda_vectorizer::fit(const std::string& single_doc )
  {
    //fit the data using the bag of words and update local variables
    this->add_document(single_doc);
    this->process_documents();
 
    //
    //  fill ret
    std::vector<real_t> ret = this->_bow.get_bag_of_words();
    //fill ret
    for ( size_t i = 0; i < this->num_unique_tokens&&i<ret.size(); i++)
    {
      //fill ret  
      ret[i] = std::log(ret[i]) + 1.0 / this->num_unique_tokens;

    }
     //return ret
    return ret;
  } 

  //process_documents
  void lda_vectorizer::process_documents (const std::vector<std::string>& documents )
  {
    //process the documents
    this->_bow.process_documents(documents);
     //
    this->process_documents();
  }
  //fit a single document :
  std::vector<real_t> lda_vectorizer::fit( const std::vector<std::string>& data_)
  {
    //fit the data using the bag of words and update local variables
    this->_bow.process_documents(data_);
    //copy the vocabulary
     this->_vocabulary = this->_bow.get_vocabulary();     
    //  fill ret
    std::vector<real_t> ret = this->_bow.get_bag_of_words();
    //fill ret
    for ( size_t i = 0; i < this->num_unique_tokens; i++)
    {
      //fill ret
      ret[i] = std::log(ret[i]) + 1.0 / this->num_unique_tokens;

    }
    //return ret
    return ret;
  }


  std::vector<std::vector<real_t>> lda_vectorizer::fit( const std::vector<std::vector<std::string>>& data_)
  {
    //fit the data using the bag of words and update local variables
    for ( auto doc_coll : data_ ) 
          this->_bow.process_documents(doc_coll); 
    
    this->process_documents();
    

     //  fill ret
    std::vector<std::vector<real_t>> ret; 
    const provallo::matrix<real_t> values= this->_bow.get_matrix();
    ret.resize(values.rows()); 

    for(size_t i=0;i<values.size1();++i)
    {
      ret[i].resize(values.size2());

      for(size_t j=0;j< values.size2();++j)
      {
        ret[i][j] = std::log(values(i,j)) + 1.0 / this->num_unique_tokens;
      }
    }

    //return ret
    return ret; 

  }

  //std::vector<std::vector<real_t>> predict( const std::vector<std::vector<std::string>>& data_); 

  //fit_transform the lda vectorizer
  std::vector<real_t> lda_vectorizer::fit_transform( const std::vector<std::string>& data_)
  {
      this->_bow.process_documents(data_); 
      //copy the vocabulary
      this->_vocabulary = this->_bow.get_vocabulary();
      //get the bag of words
      std::vector<real_t> bow = this->_bow.get_bag_of_words();
      //get the number of documents
      this->_num_docs = this->_bow.get_number_of_documents();
      //get the number of words
      this->_num_words = this->_bow.get_number_of_words();
      //get the number of unique words
      this->num_unique_tokens = this->_bow.get_number_of_unique_tokens();
      //  fill ret
      std::vector<real_t> ret(this->num_unique_tokens,0.0) ;

      //fill ret  
      for ( size_t i = 0; i < this->num_unique_tokens; i++)
      {
        //fill ret
        ret[i] = bow[i];
      }
      return ret;
     
      //return ret
      return ret;

  }

  ///usr/bin/ld: CMakeFiles/provallo_engine.dir/decision_engine/pipelinebuilder.cpp.o:(.data.rel.ro._ZTVN8provallo14lda_vectorizerE[_ZTVN8provallo14lda_vectorizerE]+0x28): undefined reference to `provallo::lda_vectorizer::fit(provallo::matrix<double> const&)'
  //fit(matrix)
  std::vector<real_t> lda_vectorizer::fit(const matrix<real_t>& data_)
  {

    const auto & docs =_bow.inverse_transform(data_);
    //fit the data using the bag of words and update local variables
     this->_bow.process_documents(docs);    
    //return the one-hot vector
    return this->fit(docs);
    //copy the vocabulary

  }

  //usr/bin/ld: CMakeFiles/provallo_engine.dir/decision_engine/pipelinebuilder.cpp.o:(.data.rel.ro._ZTVN8provallo14lda_vectorizerE[_ZTVN8provallo14lda_vectorizerE]+0xa0): undefined reference to `provallo::lda_vectorizer::save(std::basic_ofstream<char, std::char_traits<char> >&)'


  //fit_transform the lda vectorizer
  std::vector<std::vector<real_t>> lda_vectorizer::fit_transform( const provallo::matrix<real_t>& data_)
  {
    //fit the data using the bag of words and update local variables
    //transform the matrix into a vector of strings
    
    UNDEF_REFERENCE(data_)
    UNDEF_REFERENCE2(data_)

    //inverse transform the matrix into a vector of strings 
    std::vector<std::vector<real_t>> ret; 
    std::vector<std::string>  docs;
    for(size_t i=0;i<data_.rows();++i)
    { 
      std::string row_doc;

      for (size_t j = 0; j <data_.cols(); j++)
      {
        //get the value
        real_t value = data_(i,j);
        //get the index
        size_t index = (size_t)value;
        //check if the index is in the vocabulary
        if(index< this->_vocabulary.size())
        {
          //get the word
          std::string word = this->_vocabulary[index]; 
          row_doc+=word+" ";
        }
        else
        {
          row_doc+="UNK ";
        }
        
        
      }//end for j
      
      docs.push_back(row_doc);
      
    }//end for i
     //return ret
    //now process the documents
    ret = this->_bow.fit_transform(docs);
    
    return ret;

  } 
  //lda dump:
  void lda_vectorizer::dump(std::ostream& out) const
  {
    //dump the lda vectorizer
    //dump the bag of words
    this->_bow.dump(out);
    //dump the vocabulary
    out<<this->_vocabulary;
    //dump the number of documents
    out<<this->_num_docs;
    //dump the number of words
    out<<this->_num_words;
    //dump the number of unique tokens
    out<<this->num_unique_tokens;

  }
  //lda load:
  void lda_vectorizer::load(std::ifstream& in)
  {
    //load the lda vectorizer
    //load the bag of words
    this->_bow.load(in);
    //load the vocabulary
    in>>this->_vocabulary;
    //load the number of documents
    in>>this->_num_docs;
    //load the number of words
    in>>this->_num_words;
    //load the number of unique tokens
    in>>this->num_unique_tokens;
  }

  //lda save:
  void lda_vectorizer::save(std::ofstream& out) const
  {
    //save the lda vectorizer
    //save the bag of words
    this->_bow.save(out);
    //save the vocabulary
    out<<this->_vocabulary;
    //save the number of documents
    out<<this->_num_docs;
    //save the number of words
    out<<this->_num_words;
    //save the number of unique tokens
    out<<this->num_unique_tokens;
  } 
  

  //lda_vectorizer process_documents: 
  void lda_vectorizer::process_documents ()
 
  {
        //add data_src if not added already
    if(_data.length()>0)
    {
      _bow.add_document(_data);

    }
     _bow.process_documents();

    const std::vector<double>& bow = _bow.get_bag_of_words();    

    //calculate LDAModel: 
    //lda
    lda::LDA lda(_n_topics,_n_features,_n_samples,_n_components,_n_top_words,_n_iter,_n_jobs,_random_state,_alpha,_beta,_eta,_gamma,_theta,_lambda,_learning_decay,_learning_offset,_max_doc_update_iter,_total_samples,_mean_change_tol,_verbose);
    lda.fit({bow});
    //get the lda data

    _lda_data = lda.transform({bow});
    //  get the lda components
    _lda_components = lda.components();
    //get the lda explained_variance
    _lda_explained_variance = lda.explained_variance();
    //get the   explained_variance_ratio
    _lda_explained_variance_ratio = lda.explained_variance_ratio();
    //get the lda singular values
    _lda_singular_values = lda.singular_values();
    //get the lda noise variance
    _lda_noise_variance = lda.noise_variance();
    // get the lda mean
    _lda_mean = lda.mean();
    //get the lda covariance
    _lda_covariance = lda.covariance();
    _lda_precision = lda.precision();
    _lda_whiten = lda.whiten();
    _n_components = _lda_components.size(); 
    

    //update vectorizer<> members (fitted and transformed)

    //update fitted
    this->_fitted_data=bow ;
    //update transformed
    this->_transformed_data = _lda_data;

      //update the number of samples
    this->_n_samples = _bow.get_number_of_documents();
    //update the number of features
    this->_n_features = _bow.get_number_of_words();
    //update the number of components
    this->_n_components = _bow.get_number_of_unique_tokens();
    //update the number of top words
    this->_n_top_words = _bow.get_number_of_unique_tokens();
    //update the number of iterations
    this->_n_iter = _bow.get_number_of_words();
    
    //update the number of jobs
    this->_n_jobs = _bow.get_number_of_words();
    //no need to update the random state
    //
    //update the alpha
    this->_alpha = 1.0/_n_topics;
    //update the beta
    this->_beta = 1.0/_num_words;
    //update the eta
    this->_eta = 1.0/_n_topics;
    //update the gamma
    this->_gamma = 1.0/_n_topics;
    //update the theta
    this->_theta = 1.0/_n_topics;
    //update the lambda
    this->_lambda = 1.0/_n_topics;
    //update the learning decay
    this->_learning_decay = 1.0/_n_topics;
    //update the learning offset
    this->_learning_offset = 1.0/_n_topics;
    //update the max doc update iter
    this->_max_doc_update_iter = 1.0/_n_topics;
    //update the total samples
    this->_total_samples = 1.0/_n_topics;
    //update the mean change tol
    this->_mean_change_tol = 1.0/_n_topics;
    
    //---
    //update fitted_data
    this->_fitted_data =  _bow.get_bag_of_words();
    //update transformed_data
    this->_transformed_data =  _lda_data;
    //update the number of samples
    this->_n_samples = _bow.get_number_of_documents();
    //update the number of features
    this->_n_features = _bow.get_number_of_words();
    //update the number of components
    this->_n_components = _bow.get_number_of_unique_tokens();
    //update the number of top words
    this->_n_top_words = _bow.get_number_of_unique_tokens();
    //update the number of iterations
    this->_n_iter = _bow.get_number_of_words();

    }//end of process_documents
   
  //lda predict:
  std::vector<real_t> lda_vectorizer::predict(const std::vector<std::string>& documents)
  {
    //predict the lda vectorizer
    //get the one-hot vector
    std::vector<real_t> ret; 
    auto ft  = _bow.predict(documents);
    if ( ft.size() == 0 ) return ret;
    
    ret.resize(ft.size()*this->num_unique_tokens);
    //calculate the lda vectorizer
    //
    //transform the bow prediction to lda prediction :
    size_t i=0,j=0,k=0;
    for(auto& vec: ft)
    {
      j=0;
      for(auto& val: vec)
      {
        //transform the bow prediction to lda prediction
        //get the index
        size_t index = k * this->num_unique_tokens + j;
        //set the value
        ret[index] = val * _lda_data[j%_lda_data.size()];
        
        j++;

      }i++;
      k+=j;
    } 

    return ret;
  } 
  //lda predict:
  std::vector<real_t> lda_vectorizer::predict(const matrix<real_t>& data)
  {
    //predict the lda vectorizer
    //get the one-hot vector
    std::vector<real_t> ret;
    auto ft  = _bow.predict(data); 
    //return the one-hot vector
    for(auto& vec: ft)
    {
      for(auto& val: vec)
      {
        ret.push_back(val);
      }
    }

    //return the one-hot vector
    return ret;
  }
  //lda transform:
  std::vector<real_t> lda_vectorizer::transform(const std::vector<std::string>& documents)
  {
    //transform the lda vectorizer
    //get the one-hot vector
    std::vector<real_t> one_hot_vector = this->fit_transform(documents); 
    //return the one-hot vector
    return one_hot_vector;  
  }
  //lda transform:
  std::vector<real_t> lda_vectorizer::transform(const matrix<real_t>& data)
  {
    //transform the lda vectorizer
    //get the one-hot vector
    std::vector<real_t> ret;
    auto vector_of_vectors = this->fit_transform(data); 
    for ( auto& v:vector_of_vectors)
    {
      for ( auto& val:v)
      {
        ret.push_back(val);
      }
    }
    //return the one-hot vector
    return  ret;  
  }
  //lda transform:
  std::vector<real_t> lda_vectorizer::transform(const std::vector<std::vector<std::string>>& data)
  { 
    std::vector<real_t> one_hot_vector;
    for ( auto& doc : data )
    {
        std::vector<real_t> tmp= this->transform(doc); 
        for ( auto& val : tmp)
          one_hot_vector.push_back(val);
      //push the one-hot vector to the vector
      this->_num_docs++;   
    }
    //return the one-hot vector
    return one_hot_vector;  
  }
  
 } // namespace provallo
