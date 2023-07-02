/*
 * classdist.cpp
 *
 *  Created on: May 29, 2023
 *      Author: kardon
 */

#include "classdist.h"
#include <algorithm>
namespace provallo
{

  std::ostream&
  operator<< (std::ostream &out, const class_dist &q)
  {
    uint32_t i (0);
    for (; i < q.size () - 1; ++i)
      out << i << ":" << 100 * q.percentage (i) << " , ";
    out << i << ":" << 100 * q.percentage (i);
    return out;
  }

  attribute
  class_dist::mode () const
  {
    // Get max occurrence
    //    discrete_value max = (discrete_value)  std::max_element (_histogram.begin (),
    //					   _histogram.end ()) ;// - _histogram.begin();
    discrete_value distance  = (discrete_value)  std::distance(_histogram.begin(),std::max_element (_histogram.begin (),_histogram.end ()));// - _histogram.begin();

    return attribute( distance );
  }
} /* namespace provallo */
