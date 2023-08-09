/*
 * classdist.cpp
 *
 *  Created on: May 29, 2023
 *      Author: kardon
 */
#include <iostream>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <cmath>
#include <limits>

#include "attribute.h"
#include "classdist.h"

namespace provallo
{

  std::ostream&
  operator<< (std::ostream &out, const class_dist &q)
  {
    size_t i=0;
     for (; i < q.size () - 1; ++i)
      out << i << ":" << 100 * q.percentage (i) << " , ";
    out << i << ":" << 100 * q.percentage (i);
    return out;
  }

  attribute
  class_dist::mode () const
  {
      
      discrete_value distance  = (discrete_value)std::distance (_histogram.begin(),  std::max_element (_histogram.begin (),_histogram.end () ));
      return attribute(distance);
  }
  std::pair<attribute,real_t> class_dist::mode_and_percentage () const
  {
 
      discrete_value distance  = (discrete_value)std::distance (_histogram.begin(),  std::max_element (_histogram.begin (),_histogram.end () ));
      return std::make_pair(attribute(distance),percentage(distance));
  }
  std::pair<attribute,real_t> class_dist::mode_and_percentage (const std::vector<discrete_value>& exclude) const
  {
 
      std::vector<real_t> tmp(_histogram);
      for(auto e:exclude)
      {
        tmp[e]=0.0;
      }
      discrete_value distance  = (discrete_value)std::distance (tmp.begin(),  std::max_element (tmp.begin (),tmp.end () ));
      return std::make_pair(attribute(distance),percentage(distance));
  }
  
} /* namespace provallo */
