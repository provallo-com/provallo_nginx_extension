/*
 * distributedkmeans.cpp
 *
 *  Created on: May 24, 2021
 *      Author: kardon
 */

#include "distributedkmeans.h"
#include "dataset.h"
namespace provallo
{

  void
  test_distributed_kmeans ()
  {
    distributed_kmeans<float, 2> km ("iris.data.csv");
    auto data = km.load_csv ("iris.data.csv");
    if (data.size ())
      {
		clustering_parameters<float> parameters (3);
		auto kmeans = provallo::kmeans_lloyd<float, 2> (data, parameters);
		dataset_base dbb (data.size () - 1, 3, 2);
		dataset_ptr train = nullptr, valid = nullptr;
		double train_percent = 0.5;
		dbb.splitdataset (train, valid, train_percent);	
		if ((train != nullptr) & (valid != nullptr))
		  {
		    std::cout << "[+] dkm split successful" << std::endl;
		    ;
		    auto means = std::get<0> (kmeans);
		    auto clusters = std::get<1> (kmeans);
		  }
	  }
  }

  
  		/* namespace provallo */		
}	
  /* namespace provallo */
