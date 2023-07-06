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
    for (size_t currentExample = 0; currentExample < results->rows;
	currentExample++)
      {
	if ((int) pred->pos (currentExample, 0) == -1)
	  nRejected++;
	else if ((int) pred->pos (currentExample, 0)
	    == results->label (currentExample))
	  nSuccess++;
      }
    return single_result (results->rows, nSuccess, nRejected);
  }

  single_result
  kNN_result::topXResult (size_t n)
  {
    int nSuccess = 0;
    int nRejected = 0;

    for (size_t currentExample = 0; currentExample < results->rows;
	currentExample++)
      {
	pair<double, int> resultsForExample[results->cols];

	for (size_t j = 0; j < results->cols; j++)
	  resultsForExample[j] = make_pair (results->pos (currentExample, j),
					    j);

	std::sort (resultsForExample, resultsForExample + results->cols,
		   greater<pair<double, int> > ());

	for (size_t j = 0; j < n; j++)
	  {
	    if (resultsForExample[j].second == results->label (currentExample))
	      nSuccess++;
	  }
      }

    return single_result (results->rows, nSuccess, nRejected);
  }

  matrix_ptr
  kNN_result::getPredictions ()
  {

    matrix_ptr predictions (new matrix_base (results->rows, 1));

    for (size_t currentExample = 0; currentExample < results->rows;
	currentExample++)
      {

	double maxProbability = 0;
	int maxIndex = -1;
	bool rejecting = false;
	for (size_t j = 0; j < results->cols; j++)
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
    matrix_ptr confusion (new matrix_base (results->cols, results->cols));
    confusion->clear ();

    for (size_t currentExample = 0; currentExample < results->rows;
	currentExample++)
      {
	int predicted = (int) pred->pos (currentExample, 0);
	int actual = results->label (currentExample);
	if (predicted != -1 && predicted != actual)
	  confusion->pos (results->label (currentExample),
			  (int) pred->pos (currentExample, 0)) ++;}
    return confusion;
  }

} /* namespace provallo */
