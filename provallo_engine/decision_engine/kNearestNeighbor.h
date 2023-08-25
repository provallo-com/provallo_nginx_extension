/*
 * kNearestNeighbor.h
 *
 *  Created on: May 11, 2021
 *      Author: kardon
 */

#ifndef DECISION_ENGINE_KNEARESTNEIGHBOR_H_
#define DECISION_ENGINE_KNEARESTNEIGHBOR_H_
#include "dataset.h"
#include "matrix.h"
namespace provallo
{


  //ultra fast K-NN implementation

  class single_result
  {
  public:
    single_result (size_t nExamples, size_t nSuccess, size_t nRejected)
    {
      this->nExamples = nExamples;
      this->nSuccess = nSuccess;
      this->nRejected = nRejected;
    }

    double
    successRate ()
    {
      return ((double) nSuccess) / (nExamples - nRejected);
    }
    double
    rejectionRate ()
    {
      return ((double) nRejected) / nExamples;
    }

    size_t nExamples;
    size_t nSuccess;
    size_t nRejected;
  };
  class kNN_result
  {
  public:
    single_result
    top1Result ();
    single_result
    topXResult (size_t n);
    matrix_ptr
    getConfusionMatrix ();
    matrix_ptr
    getPredictions ();
    dataset_ptr
    getRawResults ()
    {
      return results;
    }
    kNN_result (dataset_ptr results)
    {
      this->results = results;
    }

  private:
    dataset_ptr results;
  };


  class kNN
  {
  public:
    kNN (dataset_ptr train)
    {
      this->data = train;
    }
    template<typename T,typename U>
    kNN(matrix<T>& train ,std::vector<U>& labels) :data(train.rows(),train.cols()+1 ,labels.size())
    {
      for(size_t i=0;i<train.rows();i++)  
      {
        for(size_t j=0;j<train.cols();j++)
        {
          data->pos(i,j)=train(i,j);
        }
        if(labels.size()>i)
          data->label(i)=labels[i];
      }
     }
    kNN_result
    run (size_t k, dataset_ptr target);

  private:
    dataset_ptr data;
  };





} /* namespace provallo */

#endif /* DECISION_ENGINE_KNEARESTNEIGHBOR_H_ */
