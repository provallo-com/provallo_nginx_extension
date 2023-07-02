/*
 * pipelinebuilder.cpp
 *
 *  Created on: Jun 19, 2023
 *      Author: kardon
 */

#include "pipelinebuilder.h"

namespace provallo
{
 
  //vectorizers implementation:  


  tfidf::tfidf (const tfidf &other):_tf (other._tf),_idf (other._idf)
  {

  }
  tfidf::tfidf (tfidf &&other):_tf (std::move (other._tf)),_idf (std::move (other._idf))
  {

  }
  tfidf&
  tfidf::operator= (const tfidf &other)
  {
    if (this != &other)
      {
            _tf = other._tf;
            _idf = other._idf;
      }
    return *this;
  }
  tfidf&
  tfidf::operator= (tfidf &&other)
  {
    if (this != &other)
      {
            _tf = std::move (other._tf);
            _idf = std::move (other._idf);
      }
    return *this;
  }
  tfidf::tfidf ()
  {

  }
  tfidf::~tfidf ()
  {

  }


  //tfidf implementation: 
  void tfidf::process_documents ( )
  {
      this->_vocabulary.clear ();
      this->_tf.clear ();
      this->_idf.clear ();
      this->_tfidf.clear ();
      //calculate vocabulary
      for (auto &doc:_documents)
      {
          std::vector<std::string> words;
          tokenize (  doc,words," ,;.:-_()[]{}!?\"\'\n\t" );

          for (auto &word:words)
          {
              if (std::find (_vocabulary.begin (), _vocabulary.end (), word) == _vocabulary.end ())
              {
                  _vocabulary.push_back (word);
              }
          }
      }

      //calculate term frequency
      for(auto &word:_vocabulary)
      {
          std::map<std::string, double> tf_doc;
          for (auto &doc:_documents)
          {
              std::vector<std::string> words;
              tokenize (doc,words," ,;.:-_()[]{}!?\"\'\n\t" ) ;
              for (auto &w:words)
              {
                  if (w == word)
                  {
                      if (tf_doc.find (word) == tf_doc.end ())
                      {
                          tf_doc[word] = 1.0;
                      }
                      else
                      {
                          tf_doc[word]+=1.0;
                      }
                  }
              }
          }
          //normalize tf  
          double sum = 0.0;
          for (auto doc_pair : tf_doc)
          {
              sum += doc_pair.second;
          }
          for (auto doc_pair : tf_doc)
          {
              doc_pair.second /= sum;
          }

          for ( auto doc_pair : tf_doc)
            _tf.push_back (doc_pair.second);

      }
      //calculate idf

  
      for (auto &word:_vocabulary)
      {
          double count = 0.0;
          for (auto &doc:_documents)
          {
              std::vector<std::string> words;
              tokenize (doc,words," ,;.:-_()[]{}!?\"\'\n\t" ) ;
              for (auto &w:words)
              {
                  if (w == word)
                  {
                      count++;
                      break;
                  }
              }
          }
          _idf.push_back ( provallo::log<2>( double( _documents.size () )  /double( count) ));
      }   
      //calculate tfidf
      _tfidf.resize (_vocabulary.size ());  
      for (uint32_t i = 0; i < _tf.size () && i<_idf.size(); ++i)
      {
          _tfidf[i].push_back (_tf[i] * _idf[i]);
      }     
      //done
      return ;
  } 
  //tfidf::get_tfidf   
  std::vector<std::vector<double>>
  tfidf::get_tfidf () const
  {
      return _tfidf;
  }

  //tfidf::get_vocabulary
  const std::vector<std::string>&
  tfidf::get_vocabulary () const
  {
      return _vocabulary;
  }
  //tfidf::get_tf
  std::vector<double>
  tfidf::get_tf () const
  {
      return _tf;
  }
  //tfidf::get_idf
  std::vector<double>
  tfidf::get_idf ()  const
  {
      return _idf;
  }
  //tfidf::get_documents
  const std::vector<std::string>&
  tfidf::get_documents () const
  {
      return _documents;
  }
  //tfidf::set_documents
  void
  tfidf::set_documents (const std::vector<std::string> &documents)
  {
    if(_documents.size () != documents.size ())
        _documents.clear ();

      _documents = documents;
      this->process_documents ();
  }

  //tfidf::add_document
  void
  tfidf::add_document (const std::string &doc)
  {
      std::vector<std::string> words;
      tokenize (doc,words," ,;.:-_()[]{}!?\"\'\n\t" ) ;
      for (auto &word:words)
      {

          //push unique words to vocabulary

          if (std::find (_vocabulary.begin (), _vocabulary.end (), word) == _vocabulary.end ())
          {
              _vocabulary.push_back (word);
          }
      }
  }   
  std::vector<double> tfidf::transform( const std::string& doc)
  {
      std::vector<double> result;
      std::vector<std::string> words;
      tokenize (doc,words," ,;.:-_()[]{}!?\"\'\n\t" ) ;
      for (auto &word:_vocabulary)
      {
          double count = 0.0;
          for (auto &w:words)
          {
              if (w == word)
              {
                  count++;
              }
          }
          result.push_back (count);
      }
      for ( uint32_t i = 0; i < result.size () && i<_idf.size(); ++i)
      {
          result[i] *= _idf[i]  ;

      }
      return result;
  }

  tfidf_vectorizer::tfidf_vectorizer (const tfidf_vectorizer &other):vectorizer (other),_tfidf (other._tfidf) 
  {

  }
  tfidf_vectorizer&
  tfidf_vectorizer::operator= (const tfidf_vectorizer &other)
  {
    if (this != &other)
      {
            _tfidf = other._tfidf;
         
      }

    return *this;
  }
  tfidf_vectorizer&
  tfidf_vectorizer::operator= (tfidf_vectorizer &&other)
  {
    if (this != &other)
      {
            _tfidf = std::move (other._tfidf);
       
      }
    return *this;
  }
  tfidf_vectorizer::tfidf_vectorizer (tfidf_vectorizer &&other):vectorizer (std::move (other)),_tfidf (std::move (other._tfidf)) 
  {

  } 
  tfidf_vectorizer::tfidf_vectorizer ():vectorizer (TFIDF)
  {

  }
  tfidf_vectorizer::~tfidf_vectorizer ()
  {

  }

  // tfidf_vectorizer::fit    
  std::vector<double>  
  tfidf_vectorizer::fit (const std::vector<std::string> &corpus)
  {
    
    for(auto &doc:corpus)
      {
          _tfidf.add_document (doc);
      }

    _tfidf.process_documents ();  
    return _tfidf.get_tfidf ();
    
  } 
  // tfidf_vectorizer::fit_transform
  std::vector<double>
  tfidf_vectorizer::fit_transform (const std::vector<std::string> &corpus)
  {
    std::vector<double> ret = fit(corpus);
    //transform : 
    provallo::matrix<double> ret_matrix (ret.size (),1);
    for (uint32_t i = 0; i < ret.size (); ++i)
    {
      ret_matrix (i,0) = ret[i];
    }

    return transform(ret_matrix);


   } 

  //
  //predict 

  std::vector<double> 
  tfidf_vectorizer::predict (const std::string &doc)
  {
    return transform (doc);
  }
  std::vector<std::vector<double>>
  tfidf_vectorizer::predict (const std::vector<std::string> &corpus)
  {
    return transform (corpus);
  }
  // tfidf_vectorizer::tfidf_vectorizer

  tfidf_vectorizer::tfidf_vectorizer (const std::vector<std::string> &corpus)
  {
    fit (corpus);
  } 
  
  std::vector<std::string> tfidf::inverse_transform (const std::vector<std::vector<double>> &corpus)
  {
    std::vector<std::string> result;
    for (auto &doc:corpus)
      {
            result.push_back (inverse_transform (doc));
      }
    return result;
  } 

  // tfidf_vectorizer::transform
  std::vector<double>
  tfidf_vectorizer::transform (const std::string &doc)
  {

    return _tfidf.transform (doc);

  } 

  std::vector<string> reverse_transform ( const std::vector<std::vector<double>>& corpus)
  {
    std::vector<std::string> result;
    for (auto &doc:corpus)
      {
            result.push_back (reverse_transform (doc));
      }
    return result;
  }

  // tfidf_vectorizer::transform
  std::vector<std::vector<double>>
  tfidf_vectorizer::transform (const std::vector<std::string> &corpus)
  {
    std::vector<std::vector<double>> result;
    for (auto &doc:corpus)
      {
            result.push_back (transform (doc));
      }
    return result;
  } 
  tfidf_vectorizer::tfidf_vectorizer ():vectorizer (TFIDF)
  { 



  } 
  tfidf_vectorizer::~tfidf_vectorizer ()
  {

  } 


  standard_scaler_vectorizer::standard_scaler_vectorizer (const standard_scaler_vectorizer &other):vectorizer (other),_scaler (other._scaler) 
  {

  }
  standard_scaler_vectorizer&
  standard_scaler_vectorizer::operator= (const standard_scaler_vectorizer &other)
  {
    if (this != &other)
      {
            _scaler = other._scaler;
         
      }

    return *this;
  } 
  standard_scaler_vectorizer&
  standard_scaler_vectorizer::operator= (standard_scaler_vectorizer &&other)
  {
    if (this != &other)
      {
            _scaler = std::move (other._scaler);
       
      }
    return *this;
  }
  standard_scaler_vectorizer::standard_scaler_vectorizer (standard_scaler_vectorizer &&other):vectorizer (std::move (other)),_scaler (std::move (other._scaler)) 
  {

  } 
  standard_scaler_vectorizer::standard_scaler_vectorizer ():vectorizer (STANDARD_SCALER)
  {

  } 
  standard_scaler_vectorizer::~standard_scaler_vectorizer ()
  {

  }

// standard_scaler_vectorizer::fit
  std::vector<double>
  standard_scaler_vectorizer::fit (const matrix<double> &corpus)
  {
    _scaler.fit (corpus);
    return _scaler.get_mean ();
  } 




  //principal_component_analysis_vectorizer::fit_transform 
  std::vector<double> 
  principal_component_analysis_vectorizer::fit_transform (const matrix<double> &corpus)
  {
    std::vector<double> ret = fit(corpus);
    //transform : 
    provallo::matrix<double> ret_matrix (ret.size (),1);
    for (uint32_t i = 0; i < ret.size (); ++i)
    {
      ret_matrix (i,0) = ret[i];
    }

    return transform(ret_matrix); 
  }
  //principal_component_analysis_vectorizer::fit
  std::vector<double>
  principal_component_analysis_vectorizer::fit (const matrix<double> &corpus)
  {
    
    
      matrix<double> covariance ;
      matrix<double> variance;

      
      matrix<double> b;
      matrix<double> c;
 
  }   


  pca_vectorizer::pca_vectorizer (const pca_vectorizer &other):vectorizer (other),_pca (other._pca) 
  {

  }
  pca_vectorizer&
  pca_vectorizer::operator= (const pca_vectorizer &other)
  {
    if (this != &other)
      {
            _pca = other._pca;
         
      }

    return *this;
  }
  pca_vectorizer&
  pca_vectorizer::operator= (pca_vectorizer &&other)
  {
    if (this != &other)
      {
            _pca = std::move (other._pca);
       
      }
    return *this;
  }
  pca_vectorizer::pca_vectorizer (pca_vectorizer &&other):vectorizer (std::move (other)),_pca (std::move (other._pca)) 
  {

  } 
  pca_vectorizer::pca_vectorizer ():vectorizer (PCA)
  {

  }
  pca_vectorizer::~pca_vectorizer ()
  {

  }
  // pca_vectorizer::fit
  std::vector<double>
  pca_vectorizer::fit (const  provallo::matrix<double>& data)
  { 
    return _pca.fit (data);
 
  } 
  // standard_scaler_vectorizer::fit
  std::vector<double>
  standard_scaler_vectorizer::fit (const std::vector<std::string> &corpus)
  {
    std::vector<double> ret;
    for (auto &doc:corpus)
      {
            ret.push_back (doc.size ());
      }
    _scaler.fit (ret);
    return ret;
  } 
  // standard_scaler_vectorizer::fit_transform
  std::vector<double>
  standard_scaler_vectorizer::fit_transform (const std::vector<std::string> &corpus)
  {
      return transform(matrix<double> ( fit (corpus) ));
  }
  // standard_scaler_vectorizer::predict
  std::vector<double>
  standard_scaler_vectorizer::predict (const std::string &doc)  {
  
  }
  
  //pca_vectorizer::fit 
  std::vector<double> pca_vectorizer::fit (const std::vector<std::string> &corpus)
  {
    std::vector<double> ret;
    for (auto &doc:corpus)
      {
            ret.push_back (doc.size ());
      }
    _scaler.fit (ret);
    return ret;
  }


  pipeline_builder&
  pipeline_builder::operator= (const pipeline_builder &other)
  {

    return *this;
  }

  pipeline_builder&
  pipeline_builder::operator= (pipeline_builder &&other)
  {
    return *this;

  }

  pipeline_builder::pipeline_builder (const pipeline_builder &other)
  {

  }

  pipeline_builder::~pipeline_builder ()
  {

  }

  pipeline_builder::pipeline_builder ()
  {

  }

} /* namespace provallo */
