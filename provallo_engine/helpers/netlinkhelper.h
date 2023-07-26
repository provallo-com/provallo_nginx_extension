/*
 * netlinkhelper.h
 *
 *  Created on: Mar 10, 2021
 *      Author: kardon
 */

#ifndef HELPERS_NETLINKHELPER_H_
#define HELPERS_NETLINKHELPER_H_
#include <memory.h>
#include <iostream>
#include <thread>
#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>
#include <thread>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <linux/rtnetlink.h>
#include <unistd.h>
#include <net/if.h>
#include <linux/lwtunnel.h>
#include <linux/seg6.h>
#include <linux/seg6_local.h>
#include <linux/seg6_iptunnel.h>
#include <cassert>
#include <stdarg.h>


//realtime collector for new provallo_attributes dataset. 



//#include "../v1_handlers/sharkstun.h"
namespace provallo
{

  typedef struct
  {
    uint16_t type;
    std::string name;
  } nl_desc;

  const nl_desc descriptor[] =
    {
      { RTA_UNSPEC, "RTA_UNSPEC" },
      { RTA_DST, "RTA_DST" },
      { RTA_SRC, "RTA_SRC" },
      { RTA_IIF, "RTA_IIF" },
      { RTA_OIF, "RTA_OIF" },
      { RTA_GATEWAY, "RTA_GATEWAY" },
      { RTA_PRIORITY, "RTA_PRIORITY" },
      { RTA_PREFSRC, "RTA_PREFSRC" },
      { RTA_METRICS, "RTA_METRICS" },
      { RTA_MULTIPATH, "RTA_MULTIPATH" },
      { RTA_PROTOINFO, "RTA_PROTOINFO" },
      { RTA_FLOW, "RTA_FLOW" },
      { RTA_CACHEINFO, "RTA_CACHEINFO" },
      { RTA_SESSION, "RTA_SESSION" },
      { RTA_MP_ALGO, "RTA_MP_ALGO" },
      { RTA_TABLE, "RTA_TABLE" },
      { RTA_MARK, "RTA_MARK" },
      { RTA_MFC_STATS, "RTA_MFC_STATS" },
      { RTA_VIA, "RTA_VIA" },
      { RTA_NEWDST, "RTA_NEWDST" },
      { RTA_PREF, "RTA_PREF" },
      { RTA_ENCAP_TYPE, "RTA_ENCAP_TYPE" },
      { RTA_ENCAP, "RTA_ENCAP" },
      { RTA_EXPIRES, "RTA_EXPIRES" },
      { RTA_PAD, "RTA_PAD" },
      { RTA_UID, "RTA_UID" },
      { RTA_TTL_PROPAGATE, "RTA_TTL_PROPAGATE" },
      { RTA_IP_PROTO, "RTA_IP_PROTO" },
      { RTA_SPORT, "RTA_SPORT" },
      { RTA_DPORT, "RTA_DPORT" },
      { RTA_NH_ID, "RTA_NH_ID" },
      { __RTA_MAX, "RTA_MAX" } };

  class netlink_helper
  {
  public:
    bool _valid;
    int _socket;
    bool _running;

  public:
    explicit
    netlink_helper () :
	_valid (false), _running (false)
    {
      struct sockaddr_nl addr;
      _socket = socket (AF_NETLINK, SOCK_RAW, NETLINK_ROUTE);
      if (_socket > 0)
	{
	  ::memset (&addr, 0, sizeof(addr));
	  addr.nl_family = AF_NETLINK;
	  addr.nl_pid = getpid ();
	  addr.nl_groups = RTMGRP_LINK | RTMGRP_IPV4_IFADDR | RTMGRP_IPV6_IFADDR
	      | RTMGRP_IPV4_MROUTE | RTMGRP_IPV6_MROUTE;
	  if (bind (_socket, (struct sockaddr*) &addr, sizeof(addr)) > -1)
	    {
	      _valid = true;

	    }

	}
      if (!_valid)
	{
	  std::cerr << "[-] " << strerror (errno) << " Check permissions... ";
	}
    }

    void*
    net_detect ()
    {

      std::cout << "[+]starting netlink monitoring on the interfaces ";
      void *pret = nullptr;
      if (_valid)
	{
	  static char buf[4096 * 2] =
	    { };
	  struct sockaddr_nl sna;
	  struct iovec iov =
	    { buf, sizeof(buf) };
	  struct msghdr msg = { (void*) &sna, sizeof(sna), &iov, 1, NULL, 0, 0 }; // @suppress("Invalid arguments")
	  _running = true;
	  while (_running)
	    {
	      int status;
	      struct nlmsghdr *h;
	      iov.iov_len = sizeof(buf);
	      status = ::recvmsg (_socket, &msg, 0);
	      for (h = (struct nlmsghdr*) buf;
		  h && NLMSG_OK(h, (unsigned int )status);
		  h = NLMSG_NEXT(h, status))
		{
		  process_netlink_msg (&sna, h);

		} //for
	    } //while

	}
      return pret;
    }

    int
    process_netlink_msg (struct sockaddr_nl *net, struct nlmsghdr *msg)
    {
      if (!msg||!net)
	return 0;
      
      struct ifaddrmsg *ifa = (struct ifaddrmsg*) NLMSG_DATA(msg);
      struct rtattr *rth = IFA_RTA(ifa);
      int rtl = IFA_PAYLOAD(msg);
      int len = rtl;
      static size_t netlink_counter = 0;
      char gateway_address[32];
      char destination_address[32];
      unsigned char route_netmask = 0;

      std::string route_txt;

      //print msgtype for now :

      std::cout << " [+] Netlink messageType:  " << msg->nlmsg_type << ",rtl="
	  << rtl << ",rta_type=" << rth->rta_type;
      ++netlink_counter;

      if ((msg->nlmsg_type == RTM_NEWROUTE)
	  || (msg->nlmsg_type == RTM_DELROUTE))
	{
	  struct rtmsg *route_entry;
	  struct rtattr *route_attribute;
	  int route_attribute_len = 0;

      	  unsigned char route_protocol = 0;
	  route_entry = (struct rtmsg*) NLMSG_DATA(msg);
	  route_netmask = route_entry->rtm_dst_len;
	  route_protocol = route_entry->rtm_protocol;
	  route_attribute = (struct rtattr*) RTM_RTA(route_entry);
	  route_attribute_len = RTM_PAYLOAD(msg);
	  std::string attribute_type =
	      (route_attribute->rta_type < __RTA_MAX) ?
		  descriptor[route_attribute->rta_type].name :
		  descriptor[__RTAX_MAX].name;

	  std::cout << "Attribute_type = > " << attribute_type;
	  std::cout << "Route Protocol = >" << std::to_string(route_protocol) <<std::endl;
	  std::string route_summary = netlink_helper::rtnl_route_summary (msg);
	  std::string route_txt;
	  for (; RTA_OK(route_attribute, route_attribute_len); route_attribute =
	      RTA_NEXT(route_attribute, route_attribute_len))
	    {
	      std::string attribute_type =
		  (route_attribute->rta_type < __RTA_MAX) ?
		      descriptor[route_attribute->rta_type].name :
		      descriptor[__RTAX_MAX].name;

	      std::cout << "Attribute_type = > " << attribute_type;

	      route_txt += rtmsg_rtattr_summary (route_attribute);
	      //destination
	      if (route_attribute->rta_type == RTA_DST)
		{
		  inet_ntop (AF_INET, RTA_DATA(route_attribute),
			     destination_address, sizeof(destination_address));
		}
	      /*gateway */
	      if (route_attribute->rta_type == RTA_GATEWAY)
		{
		  inet_ntop (AF_INET, RTA_DATA(route_attribute),
			     gateway_address, sizeof(gateway_address));
		}
	      std::cout << "  [+] Netlink Route was changed   "
		  << (msg->nlmsg_type == RTM_DELROUTE ? "Deleted" : "Added")
		  << "Destination :" << destination_address << ", " << "proto"
		  << route_netmask << ", gateway :" << gateway_address << "[+]"
		  << route_txt.c_str ();

	    }

	}
      else if (msg->nlmsg_type == RTM_NEWLINK || msg->nlmsg_type == RTM_NEWADDR)
	{
	  std::cout << " [+] Netlink New Address/Link   ";
	  struct rtattr *rta = rth;

	  for (rta = IFLA_RTA(ifa); RTA_OK(rta, len); rta = RTA_NEXT(rta, len))
	    {
	      route_txt += rtmsg_rtattr_summary (rta);

	    }

	}
      else
	{
	  std::cout << " [+] Netlink Counter: " << netlink_counter
	      << " Address/Link    ";
	}
      //link dal.
      std::cout << " [+] Netlink final route txt : " << route_txt << std::endl;

      return 0;
    }

    inline static std::string
    strfmt (const char *fmt, ...)
    {
      char str[1000];
      va_list args;
      va_start(args, fmt);
      vsprintf (str, fmt, args);
      va_end(args);
      return str;
    }

    inline static std::string
    rtmsg_summary (const struct rtmsg *rtm)
    {
      return strfmt ("fmly=%u dl=%u sl=%u tos=%u tab=%u"
		     " pro=%u scope=%u type=%u f=0x%x",
		     rtm->rtm_family, rtm->rtm_dst_len, rtm->rtm_src_len,
		     rtm->rtm_tos, rtm->rtm_table, rtm->rtm_protocol,
		     rtm->rtm_scope, rtm->rtm_type, rtm->rtm_flags);
    }

    inline static std::string
    inetpton (const void *ptr, int afi)
    {
      char buf[256];
      memset (buf, 0, sizeof(buf));
      inet_ntop (afi, ptr, buf, sizeof(buf));
      return buf;
    }

    inline static std::string
    ifindex2str (uint32_t ifindex)
    {
      char str[IF_NAMESIZE];
      char *ret = if_indextoname (ifindex, str);
      return ret ? ret : "none";
    }

    inline static size_t
    rta_payload (const struct rtattr *rta)
    {
      return rta->rta_len - sizeof(struct rtattr);
    }

    inline static uint8_t*
    rtattr_payload_ptr (const struct rtattr *rta)
    {
      return (uint8_t*) (rta + 1);
    }

    inline static const char*
    SEG6_IPTUNNEL_to_str (uint16_t type)
    {
      switch (type)
	{
	case SEG6_IPTUNNEL_UNSPEC:
	  return "SEG6_IPTUNNEL_UNSPEC";
	case SEG6_IPTUNNEL_SRH:
	  return "SEG6_IPTUNNEL_SRH";
	default:
	  return "SEG6_IPTUNNEL_UNKNOWN";
	}
    }
    inline static const char*
    SEG6_IPTUN_MODE_to_str (uint16_t type)
    {
      switch (type)
	{
	case SEG6_IPTUN_MODE_INLINE:
	  return "SEG6_IPTUN_MODE_INLINE";
	case SEG6_IPTUN_MODE_ENCAP:
	  return "SEG6_IPTUN_MODE_ENCAP";
	case SEG6_IPTUN_MODE_L2ENCAP:
	  return "SEG6_IPTUN_MODE_L2ENCAP";
	default:
	  return "SEG6_IPTUN_MODE_UNKNOWN";
	}
    }

    inline static std::string
    rtmsg_rtattr_SEG6_summary (const struct rtattr *rta)
    {
      std::string hdr = strfmt ("0x%04x %-20s :: ", rta->rta_type,
				SEG6_IPTUNNEL_to_str (rta->rta_type));
      switch (rta->rta_type)
	{
	case SEG6_IPTUNNEL_SRH:
	  {
	    const struct seg6_iptunnel_encap *sie;
	    sie = (const struct seg6_iptunnel_encap*) (rta + 1);
	    return hdr
		+ strfmt ("mode=%u (%s) ", sie->mode,
			  SEG6_IPTUN_MODE_to_str (sie->mode))
		+ ipv6_sr_hdr_summary (sie->srh);
	  }
	default:
	  {
	    std::string val;
	    val = strfmt ("unknown-fmt(rta_len=%u,data=", rta->rta_len);
	    const uint8_t *data = (const uint8_t*) (rta + 1);
	    size_t payload_len = size_t (rta->rta_len - sizeof(*rta));
	    size_t n = std::min (size_t (4), payload_len);
	    for (size_t i = 0; i < n; i++)
	      val += strfmt ("%02x", data[i]);
	    if (4 < payload_len)
	      val += "...";
	    val += ")";
	    return hdr + val;
	  }
	}
      return hdr + __func__;
    }

    inline static std::string
    ipv6_sr_hdr_summary (const struct ipv6_sr_hdr *srh)
    {
      std::string str = strfmt ("nh=%u hl=%u t=%u sl=%u [", srh->nexthdr,
				srh->hdrlen, srh->type, srh->segments_left,
				srh->first_segment);
      const size_t n = ((srh->hdrlen * 8 + 8) - 8) / 16;
      for (size_t i = 0; i < n; i++)
	{
	  const struct in6_addr *addr = &srh->segments[i];
	  str += strfmt ("%s%s", inetpton (addr, AF_INET6).c_str (),
			 i + 1 < n ? "," : "]");
	}
      return str;
    }

    inline std::string
    rtmsg_rtattr_summary (const struct rtattr *rta)
    {
      std::string hdr = strfmt (
	  "0x%04x %-16s :: ",
	  rta->rta_type,
	  ((rta->rta_type < __RTA_MAX) ?
	      descriptor[rta->rta_type].name.c_str () :
	      descriptor[__RTAX_MAX].name.c_str ()));

      std::string val;
      val = strfmt (" (rta_len=%u,data=", rta->rta_len);
      const uint8_t *data = (const uint8_t*) (rta + 1);
      size_t payload_len = size_t (rta->rta_len - sizeof(*rta));
      size_t n = std::min (size_t (4), payload_len);
      for (size_t i = 0; i < n; i++)
	val += strfmt ("%02x", data[i]);
      if (4 < payload_len)
	val += "...";
      val += ")";
      return hdr + val;

    }

    inline std::string
    rtnl_route_summary (const struct nlmsghdr *hdr)
    {
      struct rtmsg *rtm = (struct rtmsg*) (hdr + 1);
      std::string str = rtmsg_summary (rtm) + "\n";
      size_t rta_len = IFA_PAYLOAD(hdr);
      for (struct rtattr *rta = RTM_RTA(rtm); RTA_OK(rta, rta_len); rta =
	  RTA_NEXT(rta, rta_len))
	{
	  std::string attr_str = rtmsg_rtattr_summary (rta);
	  if (attr_str != "")
	    str += "  " + attr_str + "\n";
	}
      return str;
    }
    virtual
    ~netlink_helper ()
    {
      if (_socket > 0)
	{
	  shutdown (_socket, 0);
	  close (_socket);
	}
    }
    bool
    is_valid ()
    {
      return _valid;
    }
    inline std::string
    handle_rtattr_Priority (const struct rtattr *rta)
    {
      std::string hdr;

      hdr = strfmt ("0x%04x %-16s :: ", rta->rta_type,
		    descriptor[rta->rta_type % __RTAX_MAX].name);
      uint32_t val = *(uint32_t*) (rta + 1);
      return hdr + std::to_string (val);

    }
    inline static const char*
    rta_type_ROUTE_to_str (uint16_t type)
    {
      switch (type)
	{
	case RTA_UNSPEC:
	  return "RTA_UNSPEC";
	case RTA_DST:
	  return "RTA_DST";
	case RTA_SRC:
	  return "RTA_SRC";
	case RTA_IIF:
	  return "RTA_IIF";
	case RTA_OIF:
	  return "RTA_OIF";
	case RTA_GATEWAY:
	  return "RTA_GATEWAY";
	case RTA_PRIORITY:
	  return "RTA_PRIORITY";
	case RTA_PREFSRC:
	  return "RTA_PREFSRC";
	case RTA_METRICS:
	  return "RTA_METRICS";
	case RTA_MULTIPATH:
	  return "RTA_MULTIPATH";
	case RTA_PROTOINFO:
	  return "RTA_PROTOINFO";
	case RTA_FLOW:
	  return "RTA_FLOW";
	case RTA_CACHEINFO:
	  return "RTA_CACHEINFO";
	case RTA_SESSION:
	  return "RTA_SESSION";
	case RTA_MP_ALGO:
	  return "RTA_MP_ALGO";
	case RTA_TABLE:
	  return "RTA_TABLE";
	case RTA_MARK:
	  return "RTA_MARK";
	case RTA_MFC_STATS:
	  return "RTA_MFC_STATS";
	case RTA_VIA:
	  return "RTA_VIA";
	case RTA_NEWDST:
	  return "RTA_NEWDST";
	case RTA_PREF:
	  return "RTA_PREF";
	case RTA_ENCAP_TYPE:
	  return "RTA_ENCAP_TYPE";
	case RTA_ENCAP:
	  return "RTA_ENCAP";
	case RTA_EXPIRES:
	  return "RTA_EXPIRES";
	case RTA_PAD:
	  return "RTA_PAD";
	case RTA_UID:
	  return "RTA_UID";
	case RTA_TTL_PROPAGATE:
	  return "RTA_TTL_PROPAGATE";
	case RTA_IP_PROTO:
	  return "RTA_IP_PROTO";
	case RTA_SPORT:
	  return "RTA_SPORT";
	case RTA_DPORT:
	  return "RTA_DPORT";
	case RTA_NH_ID:
	  return "RTA_NH_ID";
	default:
	  return "RTA_XXXUNKNOWNXXX";
	}
    }
    inline std::string
    handle_ncap (const struct rtattr *rta)
    {
      uint16_t encap_kind;
      std::string hdr = strfmt ("0x%04x %-16s :: ", rta->rta_type,
				rta_type_ROUTE_to_str (rta->rta_type));

      const uint8_t *ptr = (const uint8_t*) rta;
      ptr += rta->rta_len;
      const struct rtattr *rtan = (const struct rtattr*) ptr;
      assert(rtan->rta_type == RTA_ENCAP_TYPE);
      assert(rtan->rta_len == 6);
      uint16_t val = *(uint16_t*) (rtan + 1);
      encap_kind = val;
      std::string attr_str;

      std::string str = "nested-data\n";
      size_t rta_len = rta->rta_len - sizeof(*rta);
      const uint8_t *data = (const uint8_t*) (rta + 1);
      for (const struct rtattr *rta = (const struct rtattr*) data;
	  RTA_OK(rta, rta_len); rta = RTA_NEXT(rta, rta_len))
	{
	  if (encap_kind == LWTUNNEL_ENCAP_SEG6_LOCAL)
	    attr_str = rtmsg_rtattr_SEG6_LOCAL_summary (rta);
	  else if (encap_kind == LWTUNNEL_ENCAP_SEG6)
	    attr_str = rtmsg_rtattr_SEG6_summary (rta);

	  if (attr_str != "")
	    str += "    " + attr_str + "\n";
	}
      str.pop_back ();
      return hdr + str;

    }
    inline static const char*
    SEG6_LOCAL_ACTION_to_str (uint16_t act)
    {
      switch (act)
	{
	/* define at /usr/include/linux/seg6_local.h */
	case SEG6_LOCAL_ACTION_UNSPEC:
	  return "SEG6_LOCAL_ACTION_UNSPEC ";
	case SEG6_LOCAL_ACTION_END:
	  return "SEG6_LOCAL_ACTION_END";
	case SEG6_LOCAL_ACTION_END_X:
	  return "SEG6_LOCAL_ACTION_END_X";
	case SEG6_LOCAL_ACTION_END_T:
	  return "SEG6_LOCAL_ACTION_END_T";
	case SEG6_LOCAL_ACTION_END_DX2:
	  return "SEG6_LOCAL_ACTION_END_DX2";
	case SEG6_LOCAL_ACTION_END_DX6:
	  return "SEG6_LOCAL_ACTION_END_DX6";
	case SEG6_LOCAL_ACTION_END_DX4:
	  return "SEG6_LOCAL_ACTION_END_DX4";
	case SEG6_LOCAL_ACTION_END_DT6:
	  return "SEG6_LOCAL_ACTION_END_DT6";
	case SEG6_LOCAL_ACTION_END_DT4:
	  return "SEG6_LOCAL_ACTION_END_DT4";
	case SEG6_LOCAL_ACTION_END_B6:
	  return "SEG6_LOCAL_ACTION_END_B6";
	case SEG6_LOCAL_ACTION_END_B6_ENCAP:
	  return "SEG6_LOCAL_ACTION_END_B6_ENCAP";
	case SEG6_LOCAL_ACTION_END_BM:
	  return "SEG6_LOCAL_ACTION_END_BM";
	case SEG6_LOCAL_ACTION_END_S:
	  return "SEG6_LOCAL_ACTION_END_S";
	case SEG6_LOCAL_ACTION_END_AS:
	  return "SEG6_LOCAL_ACTION_END_AS";
	case SEG6_LOCAL_ACTION_END_AM:
	  return "SEG6_LOCAL_ACTION_END_AM";
	default:
	  return "SEG6_LOCAL_ACTION_UNKNOWN";
	}
    }
    inline static const char*
    SEG6_LOCAL_to_str (uint16_t type)
    {
      switch (type)
	{
	case SEG6_LOCAL_UNSPEC:
	  return "SEG6_LOCAL_UNSPEC";
	case SEG6_LOCAL_ACTION:
	  return "SEG6_LOCAL_ACTION";
	case SEG6_LOCAL_SRH:
	  return "SEG6_LOCAL_SRH";
	case SEG6_LOCAL_TABLE:
	  return "SEG6_LOCAL_TABLE ";
	case SEG6_LOCAL_NH4:
	  return "SEG6_LOCAL_NH4";
	case SEG6_LOCAL_NH6:
	  return "SEG6_LOCAL_NH6";
	case SEG6_LOCAL_IIF:
	  return "SEG6_LOCAL_IIF";
	case SEG6_LOCAL_OIF:
	  return "SEG6_LOCAL_OIF";
	default:
	  return "SEG6_LOCAL_UKNOWN";
	}
    }
    inline std::string
    rtmsg_rtattr_SEG6_LOCAL_summary (const struct rtattr *rta)
    {
      std::string hdr = strfmt ("0x%04x %-20s :: ", rta->rta_type,
				SEG6_LOCAL_to_str (rta->rta_type));
      switch (rta->rta_type)
	{

	case SEG6_LOCAL_ACTION:
	  {
	    assert(rta->rta_len == 8);
	    uint32_t num = *(uint32_t*) (rta + 1);
	    return hdr + strfmt ("%u (%s)", num, SEG6_LOCAL_ACTION_to_str (num));
	  }
	case SEG6_LOCAL_NH6:
	  {
	    uint8_t *addr_ptr = (uint8_t*) (rta + 1);
	    size_t addr_len = rta->rta_len - sizeof(*rta);
	    assert(addr_len == 16);
	    return hdr + inetpton (addr_ptr, AF_INET6);
	  }
	case SEG6_LOCAL_NH4:
	  {
	    uint8_t *addr_ptr = (uint8_t*) (rta + 1);
	    size_t addr_len = rta->rta_len - sizeof(*rta);
	    assert(addr_len == 4);
	    return hdr + inetpton (addr_ptr, AF_INET);
	  }
	case SEG6_LOCAL_OIF:
	case SEG6_LOCAL_IIF:
	case SEG6_LOCAL_TABLE:
	  {
	    assert(rta->rta_len == 8);
	    uint32_t num = *(uint32_t*) (rta + 1);
	    return hdr + strfmt ("%u", num);
	  }
	case SEG6_LOCAL_SRH:
	  {
	    const struct ipv6_sr_hdr *srh =
		(const struct ipv6_sr_hdr*) (rta + 1);
	    std::string str = ipv6_sr_hdr_summary (srh);
	    return hdr + str;
	  }
	default:
	  {
	    std::string val;
	    val = strfmt ("unknown-fmt(rta_len=%u,data=", rta->rta_len);
	    const uint8_t *data = (const uint8_t*) (rta + 1);
	    size_t payload_len = size_t (rta->rta_len - sizeof(*rta));
	    size_t n = std::min (size_t (4), payload_len);
	    for (size_t i = 0; i < n; i++)
	      val += strfmt ("%02x", data[i]);
	    if (4 < payload_len)
	      val += "...";
	    val += ")";
	    return hdr + val;
	  }
	}
    }

#if 0

	inline static std::string
	rtmsg_rtattr_summary(const struct rtattr* rta)
	{
	  std::string hdr = strfmt("0x%04x %-16s :: ",
	      rta->rta_type, (rta->rta_type < __RTA_MAX )? descriptor[->rta_type].name:descriptor[__RTAX_MAX].name));


	    { RTA_PRIORITY:
	    { RTA_PREFSRC:
	    { RTA_TABLE:
	    { RTA_IIF:
	    { RTA_OIF:
	    { RTA_NH_ID:
	    {
	      assert(rta->rta_len == 8);
	      uint32_t val = *(uint32_t*)(rta+1);
	      return hdr + strfmt("%u", val);
	    }
	    { RTA_ENCAP_TYPE:
	    {
	      assert(rta->rta_len == 6);
	      uint16_t val = *(uint16_t*)(rta+1);
	      return hdr + strfmt("%u (%s)", val,
	          lwtunnel_encap_types_to_str(val));
	    }
	    { RTA_EXPIRES:
	    {
	      assert(rta->rta_len == 12);
	      uint64_t val = *(uint64_t*)(rta+1);
	      return hdr + strfmt("%lu", val);
	    }
	    { RTA_ENCAP:
	    {
	      uint16_t encap_kind;
	      {
	        const uint8_t* ptr = (const uint8_t*)rta;
	        ptr += rta->rta_len;
	        const struct rtattr* rtan = (const struct rtattr*)ptr;
	        assert(rtan->rta_type == RTA_ENCAP_TYPE);
	        assert(rtan->rta_len == 6);
	        uint16_t val = *(uint16_t*)(rtan+1);
	        encap_kind = val;
	      }

	      std::string str = "nested-data\n";
	      size_t rta_len = rta->rta_len - sizeof(*rta);
	      const uint8_t* data = (const uint8_t*)(rta + 1);
	      for (const struct rtattr* rta = (const struct rtattr*)data;
	           RTA_OK(rta, rta_len); rta = RTA_NEXT(rta, rta_len)) {
	        std::string attr_str;
	        if (encap_kind==LWTUNNEL_ENCAP_SEG6_LOCAL)
	          attr_str = rtmsg_rtattr_SEG6_LOCAL_summary(rta);
	        else if (encap_kind==LWTUNNEL_ENCAP_SEG6)
	          attr_str = rtmsg_rtattr_SEG6_summary(rta);

	        if (attr_str != "")
	          str += "    " + attr_str + "\n";
	      }
	      str.pop_back();
	      return hdr + str;
	    }
	    { RTA_PREF:
	    {
	      assert(rta->rta_len == 5);
	      uint8_t val = *(uint8_t*)(rta+1);
	      return hdr + strfmt("%u", val);
	    }

	    { RTA_DST:
	    { RTA_SRC:
	    { RTA_GATEWAY:
	    {
	      uint8_t* addr_ptr = (uint8_t*)(rta+1);
	      size_t addr_len = rta->rta_len - sizeof(*rta);
	      assert(addr_len==4 || addr_len==16);
	      if (addr_len == 4) return hdr + inetpton(addr_ptr, AF_INET);
	      if (addr_len == 16) return hdr + inetpton(addr_ptr, AF_INET6);
	      else return hdr + "unknown-addr-fmt";
	    }

	    { RTA_MFC_STATS:
	    {
	      assert(rta->rta_len == 28);
	      struct rta_mfc_stats {
	        uint64_t mfcs_packets;
	        uint64_t mfcs_bytes;
	        uint64_t mfcs_wrong_if;
	      };
	      const struct rta_mfc_stats* ms;
	      ms = (const struct rta_mfc_stats*)(rta+1);
	      return hdr + strfmt("pkt=%lu bytes=%lu wrong_if=%lu",
	          ms->mfcs_packets, ms->mfcs_bytes, ms->mfcs_wrong_if);
	    }

	    { RTA_MULTIPATH:
	    {
	      std::string str = strfmt("nested");
	      size_t plen = rta->rta_len - sizeof(struct rtattr);
	      const struct rtnexthop* rtnh = (const struct rtnexthop*)(rta+1);
	      for (; RTNH_OK(rtnh, plen);
	           rtnh = RTNH_NEXT(rtnh)) {
	        str += strfmt("\n%4snexthop flags=0x%x hops=%u ifindex=%d",
	            " ", rtnh->rtnh_flags, rtnh->rtnh_hops, rtnh->rtnh_ifindex);
	        size_t sub_plen = rtnh->rtnh_len - sizeof(struct rtnexthop);
	        for (struct rtattr* rta = RTNH_DATA(rtnh);
	            RTA_OK(rta, sub_plen);
	            rta = RTA_NEXT (rta, sub_plen)) {
	          str += strfmt("\n%6s", " ");
	          str += rtmsg_rtattr_summary(rta);
	        }
	      }
	      return hdr + str;
	    }

	    { RTA_VIA:
	    {
	      struct rtvia *rtvia = (struct rtvia*)(rta+1);
	      size_t addr_len = rta->rta_len - sizeof(*rta);
	      assert((addr_len==6 && rtvia->rtvia_family==AF_INET)
	          || (addr_len==18 && rtvia->rtvia_family==AF_INET6));
	      if (rtvia->rtvia_family == AF_INET) {
	        return hdr + "inet " + inetpton(rtvia->rtvia_addr, AF_INET);
	      } else if (rtvia->rtvia_family == AF_INET6) {
	        return hdr + "inet6 " + inetpton(rtvia->rtvia_addr, AF_INET6);
	      } else
	        return hdr + "unknown-addr-fmt";
	    }

	    // Others
	    { RTA_METRICS:
	    { RTA_PROTOINFO:
	    { RTA_FLOW:
	    { RTA_SESSION:
	    { RTA_MP_ALGO:
	    { RTA_MARK:
	    { RTA_NEWDST:
	    { RTA_PAD:
	    { RTA_UID:
	    { RTA_TTL_PROPAGATE:
	    { RTA_IP_PROTO:
	    { RTA_SPORT:
	    { RTA_DPORT:
	    { RTA_CACHEINFO:
	    default:
	    {
	      std::string val;
	      val = strfmt("unknown-fmt(rta_len=%u,data=", rta->rta_len);
	      const uint8_t* data = (const uint8_t*)(rta+1);
	      size_t payload_len = size_t(rta->rta_len-sizeof(*rta));
	      size_t n = std::min(size_t(4), payload_len);
	      for (size_t i=0; i<n; i++)
	        val += strfmt("%02x", data[i]);
	      if (4 < payload_len) val += "...";
	      val += ")";
	      return hdr + val;
	    }
	  }
	}
#endif //comment

  }; // class NetlinkHelper
// ======================================================================		

} 
/* namespace provallo */


// ======================================================================
// ======================================================================



#endif /* HELPERS_NETLINKHELPER_H_ */
