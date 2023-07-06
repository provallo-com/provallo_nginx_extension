/*
 * ip_address.cpp
 *
 *  Created on: Mar 20, 2021
 *      Author: kardon
 */

#include "ip_address.h"
#include <netinet/in.h>
#include <netinet/ip.h>
#include <arpa/inet.h>
#define OVERHEAD4 20
#define OVERHEAD6 40

#include <linux/types.h>
namespace provallo
{
  int
  ip_address::overhead () const
  {
    return family_ == AF_INET ? OVERHEAD4 : OVERHEAD6;
  }

  size_t
  ip_address::size () const
  {
    return family_ == AF_INET ? sizeof(in_addr) : sizeof(in6_addr);
  }

  bool
  ip_address::operator== (const ip_address &other) const
  {
    if (family_ != other.family_)
      {
	return false;
      }
    if (family_ == AF_INET)
      {
	return memcmp (&u_.ip4, &other.u_.ip4, sizeof(u_.ip4)) == 0;
      }
    if (family_ == AF_INET6)
      {
	return memcmp (&u_.ip6, &other.u_.ip6, sizeof(u_.ip6)) == 0;
      }

    return family_ == AF_UNSPEC && associated_port == other.associated_port;
  }
  bool
  ip_address::operator!= (const ip_address &other) const
  {
    return !((*this) == other);
  }
  bool
  ip_address::operator> (const ip_address &other) const
  {
    return (*this) != other && !((*this) < other);
  }
  bool
  ip_address::operator< (const ip_address &other) const
  {
    if (family_ != other.family_)
      return
	  (family_ == AF_UNSPEC) ? true :
	  (family_ == AF_INET && other.family_ == AF_INET6) ?
	      (associated_port == other.associated_port) : false;
    //same family:
    return
	associated_port == other.associated_port && (family_ == AF_INET) ?
	    (ntohl (u_.ip4.s_addr) < ntohl (other.u_.ip4.s_addr)) :
	(family_ == AF_INET6) ?
	    memcmp (&u_.ip6.s6_addr, &other.u_.ip6.s6_addr, 16) < 0 : false;

  }
  in6_addr
  ip_address::ipv6_address () const
  {
    return u_.ip6;
  }
  in_addr
  ip_address::ipv4_address () const
  {
    return u_.ip4;
  }
  std::string
  ip_address::to_string () const
  {
    if (family_ != AF_INET && family_ != AF_INET6)
      {
	return std::string ();
      }
    char buf[INET6_ADDRSTRLEN] =
      { 0 };

    const void *src =
	((family_ == AF_INET6) ? (void*) &u_.ip6 : (void*) &u_.ip4);

    if (!inet_ntop (family_, src, buf, sizeof(buf)))
      {
	return std::string ();
      }
    return std::string (buf);
  }

  std::string
  ip_address::get_prefix () const
  {
    return
	(family_ == AF_INET6) ?
	    Prefix<uint128_t> (uint128_t (u_.ip6.s6_addr), 0).to_cidr () :
	    Prefix<uint32_t> (u_.ip4.s_addr, 0).to_cidr ();
  }
  std::string
  ip_address::get_query () const
  {
    return
	(family_ == AF_INET6) ?
	    Prefix<uint128_t> (uint128_t (u_.ip6.s6_addr), 0).to_query () :
	    Prefix<uint32_t> (u_.ip4.s_addr, 0).to_query ();
  }

} /* namespace provallo */
