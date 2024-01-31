/*
 * objectfactory.h
 *
 *  Created on: Mar 22, 2021
 *      Author: @shokoluv aka Yaniv Karta.
 *
 */

#ifndef UTIL_OBJECTFACTORY_H_
#define UTIL_OBJECTFACTORY_H_
#include <string>
#include <thread>
#include <mutex>
#include <map>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <future>

namespace provallo
{

  template<typename Base>
    struct creator_struct
    {
      template<typename Derived>
      static Base*
      createMethod (Base *pb = NULL)
      {
        static std::recursive_mutex lock;
        std::lock_guard<std::recursive_mutex> a (lock);

        if (pb)
        return new Derived (*pb); //copy constructor
        
        return new Derived;
       }
    };

  template<typename Abstract, typename Identifier,
      typename ObjectCreator = Abstract*
      (*) (Abstract*)>
    class object_factory
    {

      std::recursive_mutex guard_tool;
      typedef object_factory<Abstract, Identifier, ObjectCreator> ThisClass;
      typedef std::map<Identifier, ObjectCreator> FunctionMap;
      FunctionMap _creator_map;
    public:
      Abstract*
      create (const Identifier &id)
      {
	std::lock_guard<std::recursive_mutex> lock (guard_tool);
	typename FunctionMap::iterator i = _creator_map.find (id);
	if (this->_creator_map.end != i)
	  return (i->second) (NULL);
	  return NULL;
     }

      Abstract*
      create (const Identifier &id, Abstract *copy)
      {
	std::lock_guard<std::recursive_mutex> lock (guard_tool);
	typename FunctionMap::iterator i = _creator_map.find (id);
	if (this->_creator_map.end != i)
	  return (i->second) (copy);

	return NULL;
      }

      bool
      subscribe (const Identifier &id, ObjectCreator creator)
      {
	std::lock_guard<std::recursive_mutex> lock (guard_tool);
	return this->_creator_map.insert (
	    std::pair<Identifier, ObjectCreator> (id, creator).second);
      }

    };

  template<typename Abstract, typename Identifier,
      typename ObjectCreator = Abstract*
      (*) (Abstract*)>
    class singleton_object_factory
    {
      typedef object_factory<Abstract, Identifier, ObjectCreator> ThisClass;
      typedef std::map<Identifier, ObjectCreator> FunctionMap;
      FunctionMap _creator_map;
    public:

      Abstract*
      create (const Identifier &id)
      {
                static std::recursive_mutex lock;
                std::lock_guard<std::recursive_mutex> a (lock);

                typename FunctionMap::iterator i = _creator_map.find (id);
                if (this->_creator_map.end != i)
                  return (i->second) (NULL);
                return NULL;
      }

      Abstract*
      create (const Identifier &id, Abstract *copy)
      {
                static std::recursive_mutex lock;
                std::lock_guard<std::recursive_mutex> a (lock);

                typename FunctionMap::iterator i = _creator_map.find (id);
                if (this->_creator_map.end != i)
                  return (i->second) (copy);

                return NULL;
      }
      
      bool
      subscribe (const Identifier &id, ObjectCreator creator)
      {
                static std::recursive_mutex lock;
                std::lock_guard<std::recursive_mutex> a (lock);

                return this->_creator_map.insert (
                    std::pair<Identifier, ObjectCreator> (id, creator).second);
      }

    };
    //std::function factory
  template<typename Abstract, typename Identifier> 
    class std_function_object_factory
    {
      typedef std::function<Abstract* (Abstract*)> ObjectCreator;
      typedef std::map<Identifier, ObjectCreator> FunctionMap;
      FunctionMap _creator_map;
    public:

      Abstract*
      create (const Identifier &id)
      {
                static std::recursive_mutex lock;
                std::lock_guard<std::recursive_mutex> a (lock);

                typename FunctionMap::iterator i = _creator_map.find (id);
                if (this->_creator_map.end != i)
                  return (i->second) (NULL);
                return NULL;
      }

      Abstract*
      create (const Identifier &id, Abstract *copy)
      {
                static std::recursive_mutex lock;
                std::lock_guard<std::recursive_mutex> a (lock);

                typename FunctionMap::iterator i = _creator_map.find (id);
                if (this->_creator_map.end != i)
                  return (i->second) (copy);

                return NULL;
      }
      
      bool
      subscribe (const Identifier &id, ObjectCreator creator)
      {
                static std::recursive_mutex lock;
                std::lock_guard<std::recursive_mutex> a (lock);

                return this->_creator_map.insert (
                    std::pair<Identifier, ObjectCreator> (id, creator).second);
      }

    };
    

} /* namespace provallo */

#endif /* UTIL_OBJECTFACTORY_H_ */
