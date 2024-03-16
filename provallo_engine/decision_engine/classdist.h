/*
 * classdist.h
 *
 *  Created on: May 29, 2023
 *      Author: kardon
 */

#ifndef DECISION_ENGINE_CLASSDIST_H_
#define DECISION_ENGINE_CLASSDIST_H_
#include <vector>
#include <iostream>
#include <numeric>
#include "attribute.h"
#include "utils.h" //real_t and sparse_ix 



using Float = real_t;

namespace provallo
{

  class class_dist
  {
    // Histogram (class attribute tag is the index of the array)
    std::vector<real_t> _histogram;
    // Total sum of the histogram bins
    real_t _sum;
    // Print distribution
    friend std::ostream&
    operator<< (std::ostream &out, const class_dist &q);

  public:
     // Constructor
    class_dist (size_t nbins = 0, real_t  default_value = 0.0) : _histogram (nbins, default_value), _sum (nbins*default_value) 
    {
        
    }
    //copy constructor
    class_dist (const class_dist& other) :
        _histogram (other._histogram), _sum (other._sum)
    {
    } 
    //move constructor
    class_dist (class_dist&& other) :
        _histogram (std::move(other._histogram)), _sum (std::move(other._sum))
    {
    }
    //copy assignment

    const class_dist& operator = (const class_dist& other)
    {
      this->_histogram=other._histogram;
      this->_sum=other._sum;
      
      return *this;

    }
    const class_dist& operator = (class_dist&& other)
    {
      this->_histogram=std::move(other._histogram);
      this->_sum=std::move(other._sum);
      
      return *this;

    }
    bool
    operator!= (const class_dist &other) const
    {
      if (_sum != other._sum)
	        return true;
      for (size_t i = 0; i < _histogram.size (); ++i)
  	      if (_histogram[i] != other._histogram[i])
	        return true;
      
      return false;
    }
    bool
    operator== (const class_dist &other) const
    {
      if (_sum != other._sum)
	        return false;
      for (size_t i = 0; i < _histogram.size (); ++i)
  	      if (_histogram[i] != other._histogram[i])
	        return false;
      
      return true;
    }
    
    real_t* array() {return _histogram.data();}
    //mode and percentage
    std::pair<attribute,real_t> mode_and_percentage () const;
    std::pair<attribute,real_t> mode_and_percentage (const std::vector<discrete_value>& exclude) const;
    std::pair<attribute,real_t> mode_and_percentage (const std::vector<discrete_value>& exclude, const std::vector<discrete_value>& include) const;
 
    // Get size
    size_t
    size () const
    {
      return _histogram.size ();
    }
    // Accumulate a specific tag
    void
    accum (size_t tag, real_t weight = 1.0)
    {
      if(tag<_histogram.size()  ) {
        _histogram[tag] += weight;
      }
      else {
        //resize and add  
          throw std::runtime_error("class_dist::accum tag out of range"); 
        // assert(0);
      }
        _sum += weight;
     }

    // Accumulate a specific tag
    void accum (const class_dist& other)  { 
      for (size_t i = 0; i < _histogram.size (); ++i)
  	      _histogram[i] += other._histogram[i]; 
      _sum += other._sum;
    }
    // Accumulate a specific tag  
    void accum (class_dist&& other)  {    
      for (size_t i = 0; i < _histogram.size (); ++i)
  	      _histogram[i] += other._histogram[i];
      _sum += other._sum;
    }
    // Accumulate a specific tag
    void    
    accum (const std::vector<real_t> &other)
    { 
       for (size_t i = 0; i < _histogram.size (); ++i)
  	      _histogram[i] += other[i];
      _sum += std::accumulate(other.begin(),other.end(),0.0);
    }
      // Set a specific tag
    void
    set (size_t tag, real_t weight)
    {
      _sum -= _histogram[tag];
      _histogram[tag] = weight;
      _sum += weight;
    }
  
    std::vector<real_t>::iterator begin() { return _histogram.begin(); } 
    std::vector<real_t>::iterator end() { return _histogram.end(); } 
    std::vector<real_t>::const_iterator begin() const { return _histogram.begin(); } 
    std::vector<real_t>::const_iterator end() const { return _histogram.end(); } 

    // Get a specific tag
    real_t
    get (size_t tag) const
    {
      return _histogram[tag];
    } 
    // Get a specific tag
    real_t&  
    get (size_t tag) 
    {
      return _histogram[tag];
    }
    
    void add (size_t tag, real_t weight)
    {
      if(tag<_histogram.size()) {
        _histogram[tag] += weight;
        _sum += weight;
      }else
      {
        //resize and add  
        _histogram.resize(tag+1,0.0);
        _histogram[tag] = weight;
        _sum += weight;
       }
      
   }

    // Get sum of the data
    real_t
    sum () const
    {
      return _sum;
    }
    // Get weight of a histogram bins
    real_t
    weight (size_t i) const
    {
      return _histogram[i];
    }
    // Get percentage of a histogram bin
    real_t
    percentage (size_t i) const
    {
      if (_sum != 0.0)
	      return _histogram[i] / _sum;
      return 0.0;
    }

    std::vector<real_t>
    cumulative () const
    {
      std::vector<real_t> values (size (), 0.0);
      for (size_t i = 0; i < size (); ++i)
      {
        auto f= percentage(i);
        // Cumulative probability
        if (i > 0)
          values[i] = f + values[i - 1];
        else
          values[i] = f;
      }
      values[size () - 1] = 1.0;
      return values;
    }
    // Get the probability of a histogram bin
    real_t
    probability (size_t i) const
    {
      if (_sum != 0.0)
        return _histogram[i] / _sum;
      return 0.0;
    }

    // Get the probability of a histogram bin
    real_t
    probability (size_t i, real_t sum) const
    {
      if (sum != 0.0)
        return _histogram[i] / sum;
      return 0.0;
    }
    // Get the probability of a histogram bin
    real_t
    probability (size_t i, real_t sum, real_t weight) const
    {
      if (sum != 0.0)
        return  ( _histogram[i] / sum ) * weight;
      return 0.0;
    } 
    // Get the probability of a histogram bin
    real_t
    probability (size_t i, const std::vector<real_t> &sum) const
    {
      if (sum[i] != 0.0)
        return _histogram[i] / sum[i];
      return 0.0;
    } 
    // Get the probability of a histogram bin
    real_t 
    probability (size_t i, const std::vector<real_t> &sum, real_t weight) const
    {
      if (sum[i] != 0.0)
        return  ( _histogram[i] / sum[i] ) * weight;
      return 0.0;
    }   


    // Get the probability of a histogram bin 
    real_t
    probability (size_t i, const std::vector<real_t> &sum, const std::vector<real_t> &weight) const
    {
      if (sum[i] != 0.0)
        return  ( _histogram[i] / sum[i] ) * weight[i];
      return 0.0;   
    }   
    // Get the probability of a histogram bin
    real_t

    probability (size_t i, const std::vector<real_t> &sum, const std::vector<real_t> &weight, real_t weight_sum) const
    {
      if (sum[i] != 0.0)
        return  ( _histogram[i] / sum[i] ) * weight[i] * weight_sum;
      return 0.0;   
    }   

    void update(size_t tag, real_t weight) {
      _histogram[tag] += weight;
      _sum += weight;
    } 
    void update(size_t tag, real_t weight, real_t old_weight) {
      _histogram[tag] += weight- old_weight;
      _sum += weight - old_weight; 
    } 
 

    void setup(size_t nbins)
    {
      _histogram.clear();
      _histogram.resize(nbins,0.0);
      _sum=0.0; 
    } 
    // Get the mode of the distribution

    attribute
    mode () const;

    // Get the entropy of the distribution
    real_t
    entropy () const;

    // Get the gini index of the distribution
    real_t
    gini () const;

    
    
    virtual
    ~class_dist ()
    {
        //delete _histogram;
        _histogram.clear();
        _sum=0.0;

        
    }
  };

  std::ostream&
  operator<< (std::ostream &out, const class_dist &q);
} /* namespace provallo */

#endif /* DECISION_ENGINE_CLASSDIST_H_ */
