/*
 * DecisionEngine.h
 *
 *  Created on: Jan 19, 2022
 *      Author: kardon
 */

#ifndef DECISION_ENGINE_DECISION_ENGINE_H_
#define DECISION_ENGINE_DECISION_ENGINE_H_

#include <cstdlib>
#include <algorithm>
#include <functional>
#include <map>
#include <thread>
#include <mutex>
#include <chrono>
#include <iostream>

namespace provallo
{
 //decision engine uses collector/sampler and statistics
  template<typename T>
    class decision_engine
    {
      std::recursive_mutex _lock;
    protected:
      decision_engine () :
	  _lock ()
      {
      }
      virtual
      ~decision_engine ()=0;
    public:
      virtual void
      init ()=0;
      virtual void
      good ()=0;
      virtual bool
      clean ()=0;
      virtual bool
      on_decision ()=0;
      bool
      register_detection_class (const T &_class, const std::function<T
      (T, double prob)> &func);
      bool
      unregister_detection_class (const T &_class);
      std::map<T, std::function<T
      (T, double)> > _class_decision_points;
    };

} /* namespace provallo */

template<typename T>
  inline bool
  provallo::decision_engine<T>::register_detection_class (
      const T &_class, const std::function<T
      (T, double prob)> &func)
  {
    std::lock_guard<std::recursive_mutex> lock (_lock);
    if (_class_decision_points.find (_class) == _class_decision_points.end ())
      {
        _class_decision_points.insert (std::make_pair (_class, func));
        return true;
      }
    return false;
  }

template<typename T>
  inline bool
  provallo::decision_engine<T>::unregister_detection_class (const T &_class)
  {

    std::lock_guard<std::recursive_mutex> lock (_lock);
    auto it = _class_decision_points.find (_class);
    if (it != _class_decision_points.end ())
      {
        _class_decision_points.erase (it);
        return true;
      }
    return false;
  }
 

  
#endif /* DECISION_ENGINE_DECISION_ENGINE_H_ */
