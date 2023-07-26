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
using Float = float;

namespace provallo
{

  class class_dist
  {
    // Histogram (class attribute tag is the index of the array)
    std::vector<Float> _histogram;
    // Total sum of the histogram bins
    Float _sum;
    // Print distribution
    friend std::ostream&
    operator<< (std::ostream &out, const class_dist &q);

  public:
    class_dist () :
	_sum (0.)
    {
    }
    class_dist (uint32_t nbins) :
	_histogram (nbins), _sum (0.0)
    {   
    }

    class_dist (uint32_t nbins, Float weight) :
	_histogram (nbins, (weight / double(nbins))), _sum (weight)
    {
    }
    class_dist ( const class_dist& other) : _histogram(other._histogram),_sum(other._sum)
    {
    }
    class_dist (  class_dist&& other) : _histogram(std::move(other._histogram)),_sum(std::move(other._sum))
    {
    }
 
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
      for (uint32_t i = 0; i < _histogram.size (); ++i)
  	      if (_histogram[i] != other._histogram[i])
	        return true;
      
      return false;
    }
    bool
    operator== (const class_dist &other) const
    {
      if (_sum != other._sum)
	        return false;
      for (uint32_t i = 0; i < _histogram.size (); ++i)
  	      if (_histogram[i] != other._histogram[i])
	        return false;
      
      return true;
    }

    // Get size
    uint32_t
    size () const
    {
      return _histogram.size ();
    }
    // Accumulate a specific tag
    void
    accum (uint32_t tag, Float weight = 1.0)
    {
      if(tag<_histogram.size()  ) {
        _histogram[tag] += weight;
      }
      else {
        std::cout<<"tag "<<tag<<" is out of range, resizing to support  "<<_histogram.size()<<std::endl;
        _histogram.resize(tag+1,0.0);
        _histogram[tag] = weight;
  
       // assert(0);
      }
        _sum += weight;
      
    }

    // Accumulate a specific tag
    void accum (const class_dist& other)  { 
      for (uint32_t i = 0; i < _histogram.size (); ++i)
  	      _histogram[i] += other._histogram[i]; 
      _sum += other._sum;
    }
    // Accumulate a specific tag  
    void accum (class_dist&& other)  {    
      for (uint32_t i = 0; i < _histogram.size (); ++i)
  	      _histogram[i] += other._histogram[i];
      _sum += other._sum;
    }
    // Accumulate a specific tag
    void    
    accum (const std::vector<Float> &other)
    { 

      for (uint32_t i = 0; i < _histogram.size (); ++i)
  	      _histogram[i] += other[i];
      _sum += std::accumulate(other.begin(),other.end(),0.0);
    }
      // Set a specific tag
    void
    set (uint32_t tag, Float weight)
    {
      _sum -= _histogram[tag];
      _histogram[tag] = weight;
      _sum += weight;
    }
  
    std::vector<Float>::iterator begin() { return _histogram.begin(); } 
    std::vector<Float>::iterator end() { return _histogram.end(); } 
    std::vector<Float>::const_iterator begin() const { return _histogram.begin(); } 
    std::vector<Float>::const_iterator end() const { return _histogram.end(); } 

    // Get a specific tag
    Float
    get (uint32_t tag) const
    {
      return _histogram[tag];
    } 
    // Get a specific tag
    Float&  
    get (uint32_t tag) 
    {
      return _histogram[tag];
    }
    
    void add (uint32_t tag, Float weight)
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
    Float
    sum () const
    {
      return _sum;
    }
    // Get weight of a histogram bins
    Float
    weight (uint32_t i) const
    {
      return _histogram[i];
    }
    // Get percentage of a histogram bin
    Float
    percentage (uint32_t i) const
    {
      if (_sum != 0.0)
	      return _histogram[i] / _sum;
      return 0.0;
    }

    std::vector<Float>
    cumulative () const
    {
      std::vector<Float> values (size (), 0.0);
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
    Float
    probability (uint32_t i) const
    {
      if (_sum != 0.0)
        return _histogram[i] / _sum;
      return 0.0;
    }

    // Get the probability of a histogram bin
    Float
    probability (uint32_t i, Float sum) const
    {
      if (sum != 0.0)
        return _histogram[i] / sum;
      return 0.0;
    }
    // Get the probability of a histogram bin
    Float
    probability (uint32_t i, Float sum, Float weight) const
    {
      if (sum != 0.0)
        return  ( _histogram[i] / sum ) * weight;
      return 0.0;
    } 
    // Get the probability of a histogram bin
    Float
    probability (uint32_t i, const std::vector<Float> &sum) const
    {
      if (sum[i] != 0.0)
        return _histogram[i] / sum[i];
      return 0.0;
    } 
    // Get the probability of a histogram bin
    Float 
    probability (uint32_t i, const std::vector<Float> &sum, Float weight) const
    {
      if (sum[i] != 0.0)
        return  ( _histogram[i] / sum[i] ) * weight;
      return 0.0;
    }   


    // Get the probability of a histogram bin 
    Float
    probability (uint32_t i, const std::vector<Float> &sum, const std::vector<Float> &weight) const
    {
      if (sum[i] != 0.0)
        return  ( _histogram[i] / sum[i] ) * weight[i];
      return 0.0;   
    }   
    // Get the probability of a histogram bin
    Float

    probability (uint32_t i, const std::vector<Float> &sum, const std::vector<Float> &weight, Float weight_sum) const
    {
      if (sum[i] != 0.0)
        return  ( _histogram[i] / sum[i] ) * weight[i] * weight_sum;
      return 0.0;   
    }   

    void update(uint32_t tag, Float weight) {
      _histogram[tag] += weight;
      _sum += weight;
    } 
    void update(uint32_t tag, Float weight, Float old_weight) {
      _histogram[tag] += weight- old_weight;
      _sum += weight - old_weight; 
    } 
 

    void setup(uint32_t nbins)
    {
      _histogram.resize(nbins);
      _sum=0.0; 
    } 
    // Get the mode of the distribution

    attribute
    mode () const;

    // Get the entropy of the distribution
    Float
    entropy () const;

    // Get the gini index of the distribution
    Float
    gini () const;

    
    
    virtual
    ~class_dist ()
    {

    }
  };

  std::ostream&
  operator<< (std::ostream &out, const class_dist &q);
} /* namespace provallo */

#endif /* DECISION_ENGINE_CLASSDIST_H_ */
