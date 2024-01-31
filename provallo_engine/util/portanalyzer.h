/*
 * portanalyzer.h
 *
 *  Created on: Apr 3, 2021
 *      Author: kardon
 */

#ifndef PORTANALYZER_H_
#define PORTANALYZER_H_

#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>
#include <thread>
#include <mutex>

namespace provallo
{

  template<typename T>
    struct port_distribution
    {
      //static locker for the map.
      
      std::pair<T, T> value_;
      std::default_random_engine e1;
      std::random_device r;
      static std::recursive_mutex locker_;


      T lmean;
      T rmean;
      double stddevr;
      double stddevl;

      std::map<std::pair<T, T>, T> port_histogram_;

      port_distribution<T> () :
	  value_ (std::pair<T, T> (0, 0)), e1 (r ()), lmean(T(0)), rmean (T(0)), stddevr(0.), stddevl(0.)      {
	std::uniform_int_distribution<T> uniform_dist (1, 65355);
	T rand_mean_1 = uniform_dist (e1);
	T rand_mean_2 = uniform_dist (e1);
	value_ = std::pair<T, T> (rand_mean_1, rand_mean_2);

	//add_pair(value_);
      }
      explicit
      port_distribution<T> (std::pair<T&, T&> &v) :
	  value_ (v), e1 (r (v)), lmean (0), rmean (0), stddevl (0.), stddevr (
	      0.)
      {
	//add_pair(value_);
      }

      void
      add_pair (const std::pair<T, T> &ports)
      {
	std::lock_guard<std::recursive_mutex> guard (
	    port_distribution<T>::locker_);
	if (port_distribution::port_histogram_.find (ports)
	    == port_distribution::port_histogram_.end ())
	  port_distribution::port_histogram_.insert (
	      std::pair<std::pair<T, T>, T> (ports, T (1)));
	else
	  port_distribution::port_histogram_[ports] += T (1);
      }

      void
      print_histogram ()
      {
	std::lock_guard<std::recursive_mutex> guard (
	    port_distribution<T>::locker_);
	for (auto port : port_histogram_)
	  {
	    std::cout << "[" << std::dec << port.first.first << ","
		<< port.first.second << "]";
	    for (T i = 0; i < port.second; ++i)
	      std::cout << "*";
	    std::cout << std::endl;

	  }

      }

      double
      get_normal_distribution () const
      {
        //std::normal_distribution<double > d {mean_value,std_dev_value};
        return 0.;
      }
    };

} /* namespace provallo */

#endif /* PORTANALYZER_H_ */
