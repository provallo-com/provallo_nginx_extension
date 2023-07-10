/*
 * interface_info.cpp
 *
 *  Created on: Nov 21, 2021
 *      Author: kardon
 */

#include "interface_info.h"
#include <ifaddrs.h>
#include <dirent.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <net/ethernet.h>
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/ioctl.h>
#include <cstring>
#include <linux/if_packet.h>
#include <map>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <net/route.h>
namespace provallo
{

  enum flags : unsigned short
  {
    UNKNWOWN,
    UP = RTF_UP,
    GATEWAY = RTF_GATEWAY,
    HOST = RTF_HOST,
    REINSTATE = RTF_REINSTATE,
    DYNAMIC = RTF_DYNAMIC,
    MODIFIED = RTF_MODIFIED,
    MTU = RTF_MTU,
    MSS = RTF_MSS,
    WINDOW = RTF_WINDOW,
    IRTT = RTF_IRTT,
    REJECT = RTF_REJECT,
    STATIC = RTF_STATIC,
    XRESOLVE = RTF_XRESOLVE,
    NOFORWARD = RTF_NOFORWARD,
    THROW = RTF_THROW,
    NOPMTUDISC = RTF_NOPMTUDISC,
  };

  struct route
  {

    std::string iface;
    std::string destination;
    std::string gateway;
    std::string flag;
    std::string refcnt;
    std::string use;
    std::string metric;
    std::string mask;
    std::string mtu;
    std::string window;
    std::string irtt;

    friend std::ostream&
    operator<< (std::ostream &os, const struct route &rt)
    {

      os << "iface:\t\t" << rt.iface << "\n" << "destination:\t"
	  << rt.destination << "\n" << "gateway:\t" << rt.gateway << "\n"
	  << "mask:\t\t" << rt.mask << "\n" <<

	  "metric:\t\t" << rt.metric << "\n" << "flags:\t\t" << rt.flag << "\n"

	  ;
      os << std::endl;
      return os;

    }
  };

  void
  interface_info::collect_interfaces ()
  {
    struct if_nameindex *if_ni = NULL;
    if_ni = if_nameindex ();
    if (if_ni)
      {
	_interfaces.clear ();
	_indexes.clear ();
	for (struct if_nameindex *i = if_ni;
	    !(i->if_index == 0 && i->if_name == NULL); i++)
	  {
	    _interfaces.insert (
		std::pair<int, std::string> (i->if_index, i->if_name));
	    _indexes.insert (
		std::pair<std::string, int> (std::string (i->if_name),
					     i->if_index));
	  }

	if_freenameindex (if_ni);

      }

  }
  inline std::string
  parse_ip (const struct sockaddr *p)
  {
    char host[NI_MAXHOST];
    int family = p->sa_family;
    int error = getnameinfo (
	p, (family == AF_INET) ? sizeof(sockaddr_in) : sizeof(sockaddr_in6),
	host, NI_MAXHOST, nullptr, 0, NI_NUMERICHOST);
    if (error != 0)
      {
	printf ("%d getnameinfo() failed: %s\n", family, gai_strerror (error));
	strcpy (host, "???");
      }
    return std::string (host);
  }

  inline std::string
  parse_mac (const unsigned char p[8])
  {
    const struct sockaddr_ll *s = reinterpret_cast<const sockaddr_ll*> (p);
    char buffer[4] =
      { };
    std::string ret = "";
    for (int i = 0; i < s->sll_halen; ++i)
      {
	if (i > 0)
	  ret += ":";
	sprintf (buffer, "%02X", s->sll_addr[i]);
	ret += buffer;
      }
    return ret;

  }
  void
  interface_info::refresh_addresses ()
  {

    struct ifaddrs *addresses = NULL, *it = NULL;
    getifaddrs (&addresses);
    it = addresses;
    //size_t index = 1;
    while (it != NULL)
      {
	struct address_info info;
	info.address_type =
	    it->ifa_addr ? it->ifa_addr->sa_family :
	    it->ifa_broadaddr ? it->ifa_broadaddr->sa_family : AF_UNSPEC;
	info.flags = it->ifa_flags;
	info.interface_id = _indexes[std::string (it->ifa_name)]; //reverse lookup the map
	info.address =
	    (info.address_type == AF_PACKET) ?
		parse_mac (
		    reinterpret_cast<const struct sockaddr_ll*> (it->ifa_addr)->sll_addr) :
	    (info.address_type == AF_INET || info.address_type == AF_INET6) ?
		parse_ip (it->ifa_addr) : "?";
	this->_addresses.insert (
	    std::pair<int, struct address_info> (info.interface_id, info));
	it = it->ifa_next;

      }
    if (addresses != NULL)
      freeifaddrs (addresses);
    _indexes.clear ();

  }
  std::string
  interface_info::get_default_gw (const std::string &dev)
  {
    std::string ret = "00:00:00:00:00:00";

    //struct in_addr gateway_ip;
    std::ifstream file;
    file.open ("/proc/net/route");
    std::vector<struct route> result;
    for (std::string line, temp; getline (file, line);)
      {
	    std::stringstream iss (line);
	    getline (iss, temp, '\t');
	    if (temp == "Iface")
	    {
    	    continue;
  	  }
	    struct route rt;
	    rt.iface = temp;
	    getline (iss, rt.destination, '\t');
	    getline (iss, rt.gateway, '\t');
	    getline (iss, rt.flag, '\t');

    	getline (iss, rt.refcnt, '\t');
	    getline (iss, rt.use, '\t');
	    getline (iss, rt.metric, '\t');
    	getline (iss, rt.mask, '\t');
	    getline (iss, rt.mtu, '\t');
	    getline (iss, rt.window, '\t');
	    getline (iss, rt.irtt, '\t');

	// convertToHumanReadableIp(rt.destination);
	// convertToHumanReadableIp(rt.gateway);
	// convertToHumanReadableIp(rt.mask);

	    result.push_back (rt);
    }

    file.open ("/proc/net/arp");

    for (std::string line, temp; getline (file, line);)
      {
	std::string ip_addr_arp, hw_type_arp, flags_arp, hw_addr_arp, mask_arp,
	    device_arp;
	std::stringstream iss (line);
	iss << ip_addr_arp << hw_type_arp << flags_arp << hw_addr_arp
	    << mask_arp << device_arp;
	for (auto& data : result)
	  {
	    std::cout << ip_addr_arp << hw_type_arp << flags_arp << hw_addr_arp
		<< mask_arp << device_arp;
	    std::cout << data.iface << data.destination << data.gateway;
      if (device_arp==dev ) {
         std::cout << "found" << std::endl;
      }
	  }// for line in arp 
      }// for line in route

    return ret;
    
  }
  interface_info::interface_info ()
  {

    collect_interfaces ();
    refresh_addresses ();

  }

  interface_info::~interface_info ()
  {

  }

  const std::map<int, std::string>
  interface_info::interfaces () const
  {
    return this->_interfaces;
  }
  const std::map<int, struct address_info>
  interface_info::get_address_info ()
  {
    return this->_addresses;
  }

} /* namespace provallo */
