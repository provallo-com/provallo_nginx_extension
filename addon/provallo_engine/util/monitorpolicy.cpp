/*
 * monitorpolicy.cpp
 *
 *  Created on: Feb 2, 2022
 *      Author: kardon
 */

#include "monitorpolicy.h"
#include <algorithm>
#include <iostream>
namespace provallo
{

  bool
  monitor_policy::register_policy_visitor (const std::string &resource_value,
					   policy_visitor *visitor)
  {

    auto it = this->visitors.find (resource_value);
    if (it != visitors.end ())
      {

	//ignore same visitor for resource name if already exists.
	if (std::find (it->second.begin (), it->second.end (), visitor)
	    == it->second.end ())
	  it->second.push_back (visitor);

      }
    else
      {
	std::vector<policy_visitor*> v;
	v.push_back (visitor);
	visitors.insert (
	    std::pair<std::string, std::vector<policy_visitor*>> (
		resource_value, v));
      }
    return true;

  }

  void
  monitor_policy::notify_all (const std::string &resource_name,
			      const std::vector<uint8_t> &actions)
  {

    auto it = visitors.find (resource_name);
    if ((it) == visitors.end ())
      {
	std::cerr << "[-]resource not found:";
	std::cerr << resource_name;
	std::cerr << " ";

      }
    else
      {

	for (auto vis : it->second)
	  vis->visit (actions);

      }

  }

} /* namespace provallo */
