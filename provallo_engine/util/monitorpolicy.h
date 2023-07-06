/*
 * monitorpolicy.h
 *
 *  Created on: Feb 2, 2022
 *      Author: kardon
 */

#ifndef UTIL_MONITORPOLICY_H_
#define UTIL_MONITORPOLICY_H_
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
    {
      ASN, CIDR, IP, REGION
    };

    std::vector<std::pair<resource_type, std::string>> resource_values;

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
    //notify visitors
    void
    notify_all (const std::string &resource_name,
		const std::vector<uint8_t> &actions);

  };

} /* namespace provallo */

#endif /* UTIL_MONITORPOLICY_H_ */
