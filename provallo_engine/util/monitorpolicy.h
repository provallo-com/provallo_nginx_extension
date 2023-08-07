/*
 * monitorpolicy.h
 *
 *  Created on: Feb 2, 2022
 *      Author: kardon
 */

#ifndef UTIL_MONITORPOLICY_H_
#define UTIL_MONITORPOLICY_H_
#include <thread>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <vector>
#include <string>
#include <map>
namespace provallo
{

  class policy_visitor
  {

  public:

    policy_visitor ()
    {
    }
    virtual bool
    visit (const std::vector<uint8_t> &actions)=0;

    virtual
    ~policy_visitor ()=0;

  };



  class monitor_policy final
  {
  protected:
    std::condition_variable cv;
    std::mutex m;
    std::string _policy_id;
    std::string _policy_description;

    enum policy_state
    {
      READY = 0x0,
      DISPATCHED = 0xf000,
      APPLIED = 0xff01,
      EXPIRING = 0xff0f,
      EXPIRED = 0xffff,
      REAPPLIED = 0xff02,
      REVOKED = 0x0f00,
      CANCELLED = 0x0f01
    } _state;
    enum resource_type : uint8_t
    { ASN, CIDR, IP, REGION , CLUSTER, NODE, POD ,UPSTREAM,PROXY };
    enum policy_category : uint8_t {
      COMPUTE, STORAGE, NETWORK, DATABASE, CACHE, MESSAGE, QUEUE, TOPIC, SERVICE, FUNCTION, CONTAINER, CONTAINER_REGISTRY, CONTAINER_IMAGE, CONTAINER_IMAGE_TAG, CONTAINER_IMAGE_DIGEST, CONTAINER_IMAGE_LAYER, CONTAINER_IMAGE_LAYER_DIGEST, CONTAINER_IMAGE_LAYER_TAG, CONTAINER_IMAGE_LAYER_TAG_DIGEST 
    };


      

    typedef std::vector<std::pair<resource_type, std::string>> resource_values_container; 

    resource_values_container resource_values;

    //each resource value can register multiple visitors.

    std::map<std::string, std::vector<policy_visitor*>> visitors;

    enum actions : uint8_t
    {
      BLOCK_AND_DROP, SANDBOX_TRAFFIC, ALLOW_AND_MONITOR, ALLOW_PASSTHROUGH
    };
  public:
    //register visitors
    bool
    register_policy_visitor (const std::string &resource_value,
			     policy_visitor *visitor);

    const resource_values_container & get_resource_values() const
    {
      return resource_values;
    }
    resource_values_container & get_resource_values()
    {
      return resource_values;
    }
    
    const std::string & get_policy_id() const
    {
      return _policy_id;
    }
    const std::string & get_policy_description() const
    {
      return _policy_description;
    }




    //notify all visitors
    void
    notify_all (const std::string &resource_name,
		const std::vector<uint8_t> &actions);
    
    
    monitor_policy (const std::string &policy_id,
        const std::string &policy_description,
        const std::vector<std::pair<resource_type, std::string>> &resource_values);
    
    virtual
    ~monitor_policy ();
   
  };
  class log_policy_visitor: public policy_visitor
  {   
  public:
    log_policy_visitor ()
    {
    }
    virtual bool
    visit (const std::vector<uint8_t> &actions)
    {
      std::cerr << "[-]actions:";
      for (auto a : actions)    
            std::cerr << a; 
      std::cerr << std::endl;
      return true;
    }
    virtual
    ~log_policy_visitor ()
    {
    }
  };
  class policy_manager
  {
  protected:
    std::vector<std::shared_ptr<monitor_policy>> policies;
    std::mutex m;
    std::condition_variable cv;
    std::thread *t;
    bool _running;
    void
    run ();
  public:
    policy_manager ();
    virtual
    ~policy_manager ();
    bool
    register_policy (std::shared_ptr<monitor_policy> policy);
    bool
    unregister_policy (std::shared_ptr<monitor_policy> policy);
    bool
    start ();
    bool
    stop ();
  };





} /* namespace provallo */

#endif /* UTIL_MONITORPOLICY_H_ */
