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
        std::cerr << "[-]actions:";
        for (auto a : actions)
          std::cerr << a;
        std::cerr << std::endl;

      }
    else
      {

          for (auto vis : it->second)
            vis->visit (actions);


      }

  }

  monitor_policy::monitor_policy (const std::string &policy_id,
          const std::string &policy_description,
          const std::vector<
          std::pair<resource_type, std::string>> &resource_values)
  {
    this->_policy_id = policy_id;
    this->_policy_description = policy_description;
    for (auto r : resource_values)
      {
        this->resource_values.push_back (
            std::pair<resource_type, std::string> (r.first, r.second));
      }
  }  

  monitor_policy::~monitor_policy ()
  {
    
    std::lock_guard<std::mutex> lock (m);
    
    //release visitors

    for (auto v : visitors)
      {
        for (auto vis : v.second)
          delete vis;
      }

  } 


  bool
  policy_manager::register_policy (std::shared_ptr<monitor_policy> policy)
  {
    std::lock_guard<std::mutex> lock (m);
    policies.push_back (policy);
    return true;
  }   

  bool
  policy_manager::unregister_policy (std::shared_ptr<monitor_policy> policy)
  {
    std::lock_guard<std::mutex> lock (m);
    auto it = std::find (policies.begin (), policies.end (), policy);
    if (it != policies.end ())
      {
        policies.erase (it);
        return true;
      }
    return false;
  }

  bool
  policy_manager::start ()
  {
    std::lock_guard<std::mutex> lock (m);
    if (_running)
      return false;
    _running = true;
    t = new std::thread (&policy_manager::run, this);
    return true;
  }

  bool
  policy_manager::stop ()
  {
    std::lock_guard<std::mutex> lock (m);
    if (!_running)
      return false;
    _running = false;
    cv.notify_all ();
    t->join ();
    delete t;
    return true;
  }

  void
  policy_manager::run ()
  {
    while (_running)
      {
        std::unique_lock<std::mutex> lock (m);
        cv.wait (lock);
        for (auto p : policies)
          {
            for (auto r : p->get_resource_values())
              {
                std::vector<uint8_t> actions;
                //TODO: get actions from resource
                actions.push_back (1);
                p->notify_all (r.second, actions);
              }
          }
      }
  } 

} /* namespace provallo */
  