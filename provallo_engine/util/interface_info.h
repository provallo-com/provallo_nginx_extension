/*
 * interface_info.h
 *
 *  Created on: Nov 21, 2021
 *      Author: kardon
 */

#ifndef INTERFACE_INFO_H_
#define INTERFACE_INFO_H_
#include <string>
#include <map>
namespace provallo
{

//interface info helper for local server interfaces

  struct address_info
  {
    int interface_id;
    int address_type; //AF_INET (IP4)/AF_INET6(IP6)/AF_PACKET(MAC_ADDRESS)
    int flags;
    std::string address;
  
  };
  class interface_info final
  {

    std::map<int, std::string> _interfaces;
    std::map<std::string, int> _indexes;
    std::map<int, address_info> _addresses;

  public:
    interface_info ();
    ~interface_info ();

    void
    collect_interfaces ();

    void
    refresh_addresses ();

    std::string
    get_default_gw (const std::string &dev);
    //returns index<>interface name
    const std::map<int, std::string>
    interfaces () const;
    const std::map<int, struct address_info>
    get_address_info ();

  };

} /* namespace provallo */

#endif /* INTERFACE_INFO_H_ */
