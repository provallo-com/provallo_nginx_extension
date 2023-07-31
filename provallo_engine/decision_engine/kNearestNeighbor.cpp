/*
 * kNearestNeighbor.cpp
 *
 *  Created on: May 11, 2021
 *      Author: kardon
 */

#include "kNearestNeighbor.h"

#include <algorithm>

using namespace std;

namespace provallo
{

  single_result
  kNN_result::top1Result ()
  {
    int nSuccess = 0;
    int nRejected = 0;
    matrix_ptr pred = getPredictions ();
    for (size_t currentExample = 0; currentExample < results->rows();
	currentExample++)
      {
	if ((int) pred->pos (currentExample, 0) == -1)
	  nRejected++;
	else if ((int) pred->pos (currentExample, 0)
	    == results->label (currentExample))
	  nSuccess++;
      }
    return single_result (results->rows(), nSuccess, nRejected) ;
  }

  single_result
  kNN_result::topXResult (size_t n)
  {
    int nSuccess = 0;
    int nRejected = 0;

    for (size_t currentExample = 0; currentExample < results->rows();
	currentExample++)
      {
	      std::vector<std::pair<double, int>> resultsForExample;
	      resultsForExample.resize(results->cols());


	for (size_t j = 0; j < results->cols(); j++)
	  resultsForExample[j] = make_pair (results->pos (currentExample, j),
					    j);

	std::sort (resultsForExample.begin(), resultsForExample.end(),
		   greater<pair<double, int> > ());

	for (size_t j = 0; j < n; j++)
	  {
	    if (resultsForExample[j].second == results->label (currentExample))
	      nSuccess++;
	  }
      }

    return single_result (results->rows(), nSuccess, nRejected);
  }

  matrix_ptr
  kNN_result::getPredictions ()
  {

    matrix_ptr predictions (new matrix_base (results->rows(), 1));

    for (size_t currentExample = 0; currentExample < results->rows();
	currentExample++)
      {

	double maxProbability = 0;
	int maxIndex = -1;
	bool rejecting = false;
	for (size_t j = 0; j < results->cols(); j++)
	  {
	    double currentProbability = results->pos (currentExample, j);
	    if (currentProbability > maxProbability)
	      {
			maxIndex = j;
			maxProbability = currentProbability;
			rejecting = false;
	      }
	    else if (currentProbability == maxProbability)
	      {
				rejecting = true;
	      }
	  }

	if (rejecting)
	  maxIndex = -1;

		predictions->pos (currentExample, 0) = maxIndex;

		
      }
    return predictions;
  }

  matrix_ptr
  kNN_result::getConfusionMatrix ()
  {
    matrix_ptr pred = getPredictions ();
    matrix_ptr confusion (new matrix_base (results->cols(), results->cols()));
    confusion->clear ();

    for (size_t currentExample = 0; currentExample < results->rows();
	currentExample++)
      {
	int predicted = (int) pred->pos (currentExample, 0);
	int actual = results->label (currentExample);
	if (predicted != -1 && predicted != actual)
	  confusion->pos (results->label (currentExample),
			  (int) pred->pos (currentExample, 0)) ++;}
    return confusion;
  }

double GetSquaredDistance(dataset_ptr train, size_t trainExample, dataset_ptr target, size_t targetExample)  
{
	if(train->cols() != target->cols()) 
	{
		//
		cerr<<"Error: train and target have different number of columns, train columns:"<<train->cols()<<", target columns:"<<target->cols()<<" , train rows: "<<train->rows()<<" test rows : " << target ->rows()<< std::endl;	
		exit(1);
	}
	double sum = 0;
	double difference;
	for(size_t col = 0; col < train->cols(); col++) {
		difference = train->pos(trainExample, col) - target->pos(targetExample, col);
		sum += difference * difference;
	}
	return sum;
}

kNN_result kNN::run(size_t k, dataset_ptr target) {


	dataset_ptr results(new dataset_base(target->rows(),target->_num_of_labels, target->_num_of_labels));
	results->clear();

	//squaredDistances: first is the distance; second is the trainExample row
	std::vector<std::pair<double, int>> squaredDistances(data->_rows);

	for(size_t targetExample = 0; targetExample < target->rows(); targetExample++) {

#ifdef DEBUG_KNN
		if (targetExample % 100 == 0)
				DEBUGKNN("Target %lu of %lu\n", targetExample, target->rows);
#endif
		//Find distance to all examples in the training set
		for (size_t trainExample = 0; trainExample < data->rows(); trainExample++) {
				squaredDistances[trainExample].first = GetSquaredDistance(data, trainExample, target, targetExample);
				squaredDistances[trainExample].second = trainExample;
		}

		//sort by closest distance
		sort(squaredDistances.begin(), squaredDistances.end());
		
		//count classes of nearest neighbors
		size_t nClasses = target->_num_of_labels;
		std::vector<size_t>  countClosestClasses(nClasses);
		for(size_t i = 0; i< nClasses; i++)
			 countClosestClasses[i] = 0;

		for (size_t i = 0; i < k; i++)
		{

			size_t currentClass = data->label(squaredDistances[i].second);
			countClosestClasses[currentClass]++;
		}

		//result: probability of class K for the example X
		for(size_t i = 0; i < nClasses; i++)
		{
			results->pos(targetExample, i) = ((double)countClosestClasses[i]) / k;
		}
	}

	//copy expected labels:
	for (size_t i = 0; i < target->rows(); i++)
		results->label(i) = target->label(i);

	return kNN_result(results);

}

} /* namespace provallo */
