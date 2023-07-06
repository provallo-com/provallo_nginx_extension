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
	    return new Derived (pb);
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

} /* namespace provallo */

#endif /* UTIL_OBJECTFACTORY_H_ */
