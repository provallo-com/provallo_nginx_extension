/*
 * netfilterhelper.cpp
 *
 *  Created on: Mar 10, 2021
 *
 * see
 * https://git.netfilter.org/libnetfilter_queue/tree/examples/nf-queue.c
 *
 *      Author: kardon
 */



#ifdef METAMONITOR

#include "netfilterhelper.h"
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <netinet/ip6.h>      // struct ip6_hdr
#include <netinet/udp.h>
#include <netinet/icmp6.h>
#include <netinet/ip_icmp.h>
#include <linux/types.h>
#include <linux/netfilter.h>
#include <libnetfilter_queue/libnetfilter_queue.h>
#include <linux/if_ether.h>   // ETH_P_IP = 0x0800, ETH_P_IPV6 = 0x86DD
#include <linux/if_packet.h>  // struct sockaddr_ll (see man 7 packet)
#include <arpa/inet.h>



#include "../v1_handlers/srtp.h"
#include "../v1_handlers/sharkstun.h"
#include "../v1_handlers/stunparser.h"



//TODO:
//REFACTOR THIS. Add proto_handler, cleanup this code.
//		1) add : protocol_handler<transport>(protocol)
//		   e.g.	 protocol_handler<IP>(icmp), protocol_handler<UDP>(stun)
//				 protocol_handler<SCP>(RTP),protocol_handler<UDP>(ip)
//	    2) separate input validation and client side compatability tests
//
//		3)

#define  IS_ADDRESS_ATT(x)  ((x) == XOR_PEER_ADDRESS\
							|| (x) == XOR_RELAYED_ADDRESS\
							|| (x) == XOR_MAPPED_ADDRESS\
							|| (x) == XOR_RESPONSE_TARGET\
							|| (x) == XOR_REFLECTED_FROM)

#define INSEPECTED_PORT(dest_port,src_port) ((dest_port) == 3478 || (src_port) == 3478 || (src_port) == 19302\
						|| (dest_port) == 19302 || (src_port) == 19305|| (dest_port) == 19305) || (netfilter_helper::port_mapped_addresses.find(src_port)!=port_mapped_addresses.end() )

//TODO: 1-INSPECTED_CHANNEL.
//		2- SIP INVITE

namespace provallo
{

  netfilter_helper::netfilter_helper (mini_dal *p) :
      pMiniDal_ (p), ip6_udp_stun_throughput_counter (0), ip4_udp_stun_throughput_counter (
	  0)
  {

    //can fork here to run execl and waitpid child to finish.
    //
    //int sysret = -1;
    //sysret = execl("/usr/sbin/iptables","iptables","-I","INPUT"," 1","-p","UDP","-j","NFQUEUE","--queue-num","999",NULL);
    //if(sysret<0)
    //	std::cerr<<"[-]execl returned "<<sysret<<",errno="<<errno;
    //else
    //	std::cout<<"[+] successfully added iptables rule "<<std::endl;
    _h = nfq_open ();
    if (!_h)
      {
	std::cerr << "[-]error during nfq_open()";
      }
    std::cout
	<< ("[+]unbinding existing nf_queue handler for AF_INET (if any)\n");
    if (nfq_unbind_pf (_h, AF_INET) < 0)
      {
	std::cerr << "[-]error during nfq_unbind_pf(" << strerror (errno)
	    << ")";
	exit (-1);

      }

    std::cout
	<< ("[+]binding nfnetlink_queue as nf_queue handler for AF_INET\n");
    if (nfq_bind_pf (_h, AF_INET) < 0)
      {
	std::cerr << "[-]error during nfq_bind_pf(" << strerror (errno) << ")";

	exit (-1);

      }

    std::cout << ("[+]binding this socket to queue '999'\n");
    _qh = nfq_create_queue (_h, 999, &netfilter_helper::callback, this);
    if (!_qh)
      {
	std::cerr << "[-]error nfq_create_queue (" << strerror (errno) << ")";
      }

    std::cout << ("[+] setting copy_packet mode\n");

    if (nfq_set_mode (_qh, NFQNL_COPY_PACKET, 0xffff) < 0)
      {
	std::cerr << " [-]can't set packet_copy mode" << strerror (errno)
	    << ")";

      }

    std::cout << "[+]initializing RPC services " << std::endl;
    //load pre-set data , ifnotdef release, we assign default broadcast addresses for notification testings
    this->_rpcserver.init ();
    this->_broadcast_server.init ();

    std::cout << "\033[0m";

  }
  bool
  netfilter_helper::print_stun (struct iphdr *stun_ip_hdr)
  {
    //unsigned short len = ntohs(stun_ip_hdr->); //get udp length
    std::map<stun_attr_hdr, ptrdiff_t> offset_map;
    std::string transaction_id;
    unsigned char *pdata = (unsigned char*) stun_ip_hdr;
    struct udphdr *udp = (struct udphdr*) (
	(stun_ip_hdr->version == 4) ?
	    (pdata + sizeof(iphdr)) : (pdata + sizeof(ip6_hdr)));
    std::cout << "[+]     <--STUN PACKET INFORMATION-->  ";
    bool violated = false;
    unsigned short offset =
	(stun_ip_hdr->version == 4) ?
	    sizeof(iphdr) + sizeof(udphdr) : sizeof(ip6_hdr) + sizeof(udp); //potential bug here.
    //for stun, print all header information to correlate NAT success
    //todo : fix ip6
    unsigned short ipv = stun_ip_hdr->version;

    uint8_t ipttl =
	(uint8_t) stun_ip_hdr->version == 4 ?
	    stun_ip_hdr->ttl :
	    ((ip6_hdr*) stun_ip_hdr)->ip6_ctlun.ip6_un1.ip6_un1_flow;
    uint8_t iptos =
	stun_ip_hdr->version == 4 ?
	    stun_ip_hdr->tos :
	    ((ip6_hdr*) stun_ip_hdr)->ip6_ctlun.ip6_un1.ip6_un1_hlim;
    uint16_t ip_total_len = (
	ipv == 4 ?
	    ntohs (stun_ip_hdr->tot_len) :
	    ntohs (((ip6_hdr*) pdata)->ip6_ctlun.ip6_un1.ip6_un1_plen));
    uint16_t id = ntohs (stun_ip_hdr->id);
    uint16_t protocol = stun_ip_hdr->protocol;
    bool channel = false, bfound = false;
    size_t header_size = (
	(stun_ip_hdr->version == 4) ? sizeof(iphdr) : sizeof(ip6_hdr));
    char tmpbuf[1024];

    //unsigned short len = ntohs(udp->len);
    unsigned short src = ntohs (udp->uh_sport);
    unsigned short dst = ntohs (udp->uh_dport);
    //unsigned short packet_check = ntohs(udp->check);
    std::string src_address, dst_address;
      {
	if (ipv == 6)
	  {
	    ip6_hdr *hdr = (ip6_hdr*) stun_ip_hdr;
	    src_address = std::string (
		inet_ntop (AF_INET6, &hdr->ip6_src, tmpbuf, 1024));
	    dst_address = std::string (
		inet_ntop (AF_INET6, &hdr->ip6_dst, tmpbuf, 1024));

	    _ip6ports.add_pair (std::pair<uint16_t, uint16_t> (src, dst));
	  }
	else
	  {

	    src_address = std::string (
		inet_ntop (AF_INET, &stun_ip_hdr->saddr, tmpbuf, 1024));
	    dst_address = std::string (
		inet_ntop (AF_INET, &stun_ip_hdr->daddr, tmpbuf, 1024));
	    _ip4ports.add_pair (std::pair<uint16_t, uint16_t> (src, dst));
	  }

      }

    unsigned short msg_type = ntohs (*(unsigned short*) &pdata[offset]);
    unsigned short msg_length = ntohs (*(unsigned short*) &pdata[offset + 2]);
    std::cout << "[+ stun ip header : protocol= "
	<< std::string (protocol == 1 ? "IP6UDP" : "IPUDP")
	<< std::string (",ip_total_len = ") << ip_total_len
	<< std::string (",id=") << id << std::string (",version=") << ipv
	<< std::string (",ipttl=") << (uint16_t) ipttl << std::string ("tos=")
	<< (uint16_t) iptos;

    std::cout << "[+] stun msg type:" << msg_type;
    std::cout << "[+] stun msg len:" << msg_length;
    std::cout << "[+] source address: " << src_address;
    std::cout << "[+] dest address: " << dst_address;

    //todo: remove this since we capture ip_total_len, validate later to avoid overflows.
    if (msg_type & 0xC000)
      {
	channel = true;
	std::cout << "[+] stun msg channel msg:" << msg_type;
      }

    if (((msg_length > ip_total_len - (sizeof(udphdr) + STUN_HDR_LEN)
	&& !channel))
	|| ((msg_length > ip_total_len - (sizeof(udphdr) + 4) && channel)))
      {
	//normalize msg_length
	//
	std::cout << "[+] invalid msg_len detected:" << msg_length;
	violated = true;
	msg_length = ip_total_len - (sizeof(udphdr) + STUN_HDR_LEN);

	std::cout << "[+] Sandbox session recommended, limiting to len:"
	    << msg_length;

      }

    uint32_t valid = ntohl ((*(uint32_t*) &pdata[offset + 4]));
    uint32_t *transaction_ptr = (uint32_t*) &pdata[offset + 8];

    if (!channel)
      {
	std::cout << "[+]"
	    << std::string (valid == 0x2112a442 ? "Valid" : "Invalid")
	    << "  STUN packet (" << std::hex << valid << std::dec << ")";

	if (!violated)
	  violated = (valid != 0x2112a442);

      }
    else
      {
	struct rtp_packet::rtp_header *header = nullptr;

	if (rtp_packet::is_rtp ((uint8_t*) &pdata[offset + 4],
				(uint16_t) ip_total_len - (offset + 4)))
	  {
	    header =
		reinterpret_cast<rtp_packet::rtp_header*> ((char*) &pdata[offset
		    + 4]);

	  }
	else if (rtp_packet::is_rtp ((uint8_t*) &pdata[offset],
				     (uint16_t) ip_total_len - (offset)))

	  {
	    header =
		reinterpret_cast<rtp_packet::rtp_header*> ((char*) &pdata[offset]);
	  }
	if (header != nullptr)
	  {
	    std::cout << "[+] RTP HEADER  version :"
		<< (uint8_t) header->version;
	    std::cout << "[+] RTP HEADER  extension :"
		<< (bool) header->extension;
	    std::cout << "[+] RTP HEADER  marker :" << (bool) header->marker;
	    std::cout << "[+] RTP HEADER  padding :" << (bool) header->padding;
	    std::cout << "[+] RTP HEADER  payloadType :"
		<< (uint8_t) header->payloadType;
	    std::cout << "[+] RTP HEADER  sequenceNumber :"
		<< (uint16_t) header->sequenceNumber;
	    std::cout << "[+] RTP HEADER  ssrc :" << (uint32_t) header->ssrc;
	    std::cout << "[+] RTP HEADER  timestamp :"
		<< (uint32_t) header->timestamp;
	  }

      }
    uint16_t msg_type_class = ((msg_type & 0x0010) >> 4)
	| ((msg_type & 0x0100) >> 7);
    uint16_t msg_type_method = (msg_type & 0x000F) | ((msg_type & 0x00E0) >> 1)
	| ((msg_type & 0x3E00) >> 2);

    if (!channel)
      {

	std::cout << "[+]msg_type_class : " << msg_type_class << ", method:"
	    << msg_type_method;

	for (auto class_ : classes)
	  {
	    if (class_.value == msg_type_class && class_.strptr)
	      {
		std::cout << "[+] CLASS :" << class_.strptr << "["
		    << msg_type_class << "]";
	      }
	  }
      }

    //strstream<>::str() is  not null terminated unless null terminated strings were addded.
    //str.write("", 1); is an ugly workaround

    std::strstream str;
    for (size_t i = 0; i < 3; ++i)
      str << std::hex << ntohl (transaction_ptr[i]);
    str.write ("", 1);
    transaction_id = str.str ();
    bool bParserPass = false;
    std::cout << "[+] transaction id  :" << transaction_id;
    if (!channel)
      {
	stun_message *msg = new relay_message ();

	try
	  {
	    if (msg_length > 4 && msg_length < ip_total_len)
	      {

		if (!msg->read (&pdata[header_size + sizeof(udp)], msg_length))
		  {
		    std::cerr
			<< "[-]Warning : Google's webrtc clients may be affected by structure";
		  }
		else
		  {
		    std::cout << "[+]Stun Google ext. parsed successfully";
		    bParserPass = true;
		  }
	      }
	  }
	catch (std::exception &error)
	  {
	    std::cout << "[+] error " << error.what ();
	  }
	catch (...)
	  {
	    std::cout << "[+]libwebrtc potential parsing error ";
	  }
	delete msg;
      }
    else
      {
	struct rtp_packet::rtp_header *header = nullptr;

	if (rtp_packet::is_rtp (
	    (uint8_t*) &pdata[offset + STUN_HDR_LEN],
	    (uint16_t) ip_total_len - (offset + STUN_HDR_LEN)))
	  {
	    header =
		reinterpret_cast<rtp_packet::rtp_header*> ((char*) &pdata[offset
		    + STUN_HDR_LEN]);

	  }
	else if (rtp_packet::is_rtp ((uint8_t*) &pdata[offset],
				     (uint16_t) ip_total_len - (offset)))

	  {
	    header =
		reinterpret_cast<rtp_packet::rtp_header*> ((char*) &pdata[offset]);
	  }
	if (header != nullptr)
	  {
	    std::cout << "[+] RTP HEADER  version :"
		<< (uint8_t) header->version;
	    std::cout << "[+] RTP HEADER  extension :"
		<< (bool) header->extension;
	    std::cout << "[+] RTP HEADER  marker :" << (bool) header->marker;
	    std::cout << "[+] RTP HEADER  padding :" << (bool) header->padding;
	    std::cout << "[+] RTP HEADER  payloadType :"
		<< (uint8_t) header->payloadType;
	    std::cout << "[+] RTP HEADER  sequenceNumber :"
		<< (uint16_t) header->sequenceNumber;
	    std::cout << "[+] RTP HEADER  ssrc :" << (uint32_t) header->ssrc;
	    std::cout << "[+] RTP HEADER  timestamp :"
		<< (uint32_t) header->timestamp;
	  }

      }

    size_t old_offset = offset;
    offset += STUN_HDR_LEN;
    size_t att_count = 0;
    for (auto method : methods)
      {
	if (method.value == msg_type_method)
	  {
	    std::cout << "[+] Method :"
		<< (method.strptr ? method.strptr : "???");
	    bfound = true;
	    //if (msg_type_method == REQUEST)
	    //if (msg_type_method == REQUEST) {
	    if (!channel)
	      while (offset - old_offset < msg_length)
		{ //&&
		  //offset<(ip_total_len -  (header_size+sizeof(udphdr)+STUN_HDR_LEN)))  {

		  stun_attr_hdr hdr;
		  uint16_t att_type = ntohs ((*(uint16_t*) &pdata[offset]));
		  uint16_t att_length = ntohs (
		      (*(uint16_t*) &pdata[offset + 2]));
		  if (att_length <= 0 || att_length > msg_length)
		    break;
		  std::cout << "[+]Attribute (" << att_type << ")";
		  hdr.type = att_type;
		  hdr.length = att_length;
		  offset_map.insert (
		      std::make_pair (hdr, (ptrdiff_t) &pdata[offset + 4]));
		  if (att_type)
		    for (auto at : attributes)
		      {
			if (att_type == at.value)
			  std::cout << at.strptr;
		      }
		  std::cout << ", length = " << att_length;

		  if (IS_ADDRESS_ATT(att_type))
		    {
		      uint16_t port = ntohs ((*(uint16_t*) &pdata[offset + 6]));
		      std::string s_address, r_address;
		      std::cout << "[+] XORED Port : " << (port ^ (valid >> 16))
			  << ",wire port:" << port;
		      if (att_length >= 8 && att_length < 20)
			{
			  //extract ip4
			  uint32_t ip = ntohl (
			      (*(uint32_t*) &pdata[offset + 8]));
			  r_address = std::string (
			      inet_ntop (AF_INET, &ip, tmpbuf, 1024));
			  ip ^= valid;
			  ip = htonl (ip); //reverse order
			  s_address = std::string (
			      inet_ntop (AF_INET, &ip, tmpbuf, 1024));

			}
		      else
			{
			  uint32_t ip6_temp[4];
			  memcpy (&ip6_temp, &pdata[offset + 8],
				  sizeof(in6_addr));
			  r_address = std::string (
			      inet_ntop (AF_INET6, &ip6_temp[0], tmpbuf, 1024));
			  ip6_temp[0] = ip6_temp[0] ^ htonl (valid);
			  ip6_temp[1] = ip6_temp[1]
			      ^ ntohl (transaction_ptr[0]);
			  ip6_temp[2] = ip6_temp[2]
			      ^ ntohl (transaction_ptr[1]);
			  ip6_temp[3] = ip6_temp[3]
			      ^ ntohl (transaction_ptr[2]);
			  s_address = std::string (
			      inet_ntop (AF_INET6, &ip6_temp[0], tmpbuf, 1024));

			}

		      pMiniDal_->insert_address_attribute (
			  src_address, dst_address, s_address,
			  (port ^ (valid >> 16)), ipttl, iptos);
		      std::cout << "[+] XORED IP : " << s_address
			  << ",Wire IP : " << r_address;

		    }
		  else if (att_type == MAPPED_ADDRESS)
		    {
		      //LEGACY :
		      uint16_t port = ntohs ((*(uint16_t*) &pdata[offset + 6]));
		      std::string s_address;

		      if (att_length >= 8 && att_length < 20)
			{
			  uint32_t ip = ntohl (
			      (*(uint32_t*) &pdata[offset + 8]));
			  s_address = std::string (
			      inet_ntop (AF_INET, &ip, tmpbuf, 1024));

			  if (port_mapped_addresses.find (port)
			      == port_mapped_addresses.end ())
			    port_mapped_addresses.insert (
				std::pair<uint16_t, ip_address> (port, ip));

			}
		      else
			{
			  uint32_t ip6_temp[4];
			  memcpy (&ip6_temp, &pdata[offset + 8],
				  sizeof(in6_addr));
			  s_address = std::string (
			      inet_ntop (AF_INET6, &ip6_temp[0], tmpbuf, 1024));

			  if (port_mapped_addresses.find (port)
			      == port_mapped_addresses.end ())
			    port_mapped_addresses.insert (
				std::pair<uint16_t, ip_address> (port,
								 ip6_temp));
			}

		      pMiniDal_->insert_address_attribute (src_address,
							   dst_address,
							   s_address, port,
							   ipttl, iptos);

		    }
		  else if (att_type == REALM || att_type == USERNAME
		      || att_type == PASSWORD || att_type == SOFTWARE)
		    {
		      char tmp[512] =
			{ };
		      memcpy (tmp, &pdata[offset + 4],
			      att_length > 512 ? 511 : att_length);
		      std::cout << "[+] attribute string:" << tmp;

		    }
#ifdef _DEBUG
					{
						if (att_length >= 2)
							std::cout << "[+] Dumping Attribute Hex:";
						for (size_t i = offset + 4;
								att_length && i < offset + att_length; ++i) {

							//std::cout << "\033[1m\033[32m";
							//red:
							unsigned short int val = (uint16_t) pdata[i];
							if (i % 16 == 0)
								std::cout  << "[" << std::dec
										<< i - offset << "]" << " ";
							std::cout << std::hex << val << " ";
						}
					}
#endif
		  if ((att_length % 4) != 0)
		    att_length += (4 - (att_length % 4));

		  std::cout << ", aligned length = " << att_length;
		  offset += 4 + att_length;
		  att_count++;
		  if (offset + 8 > msg_length)
		    break;
		}
	    //} if
	  }					   //if found
      }					   //for
    if (!bfound && !channel)
      {
	std::cout << "[-] Method not found : " << msg_type_method << std::endl;
      }
    std::cout << "[+] parsed :" << att_count << " attributes" << std::endl;

    pMiniDal_->log_stun (msg_type, msg_length, src, dst, msg_type_class,
			 transaction_id, stun_ip_hdr->version, src_address,
			 dst_address, msg_type_class, msg_type_method, ipttl,
			 iptos, channel, bParserPass, ip_total_len);
    if (att_count > 0)
      {
	for (auto it : offset_map)
	  {
	    //
	    //remove this comment to avoid duplicate address attributes. :)
	    //if(!IS_ADDRESS_ATT(it.first.type))
	    pMiniDal_->insert_attribute (it.first.type, it.first.length,
					 (void*) it.second, it.first.length);
	  }
      }

    demo_data_model.push_back (std::array<int32_t, 8> (
      { ipv, ip_total_len, msg_type, msg_length, (int32_t) att_count, src, dst,
	  (int32_t) violated }));
    //enforcement :
    if (violated)
      {
	return true;
      }

    return false;
  }
  void
  netfilter_helper::run ()
  {
    int fd = nfq_fd (_h);
    char buf[4096] __attribute__ ((aligned));
    int rv;
    while ((rv = recv (fd, buf, sizeof(buf), 0)))
      {
	std::cout << "[+]processing   packet";
	nfq_handle_packet (_h, buf, rv);
      }
  }

  u_int32_t
  netfilter_helper::dissect_packet (struct nfq_data *tb, netfilter_helper *nfh)
  {
    int verdict = 0, id = 0;
    struct nfqnl_msg_packet_hdr *ph;
    struct nfqnl_msg_packet_hw *hwph;
    u_int32_t mark, ifi;
    int ret;
    unsigned char *data;
    size_t rpclen;
    char tempbuf[1024];
    static int counter = 0;
    bool in = false;
    bool invalid = false;
    uint32_t gid = 0;
    rpc_msg_header *rpc_prx = nullptr;
    auto start = std::chrono::high_resolution_clock::now ();

    nfq_get_gid (tb, &gid);

    ph = nfq_get_msg_packet_hdr (tb);

    if (ph)
      {
	id = ntohl (ph->packet_id);
	std::cout << "Protocol = " << std::hex << ntohs (ph->hw_protocol)
	    << ",hook=" << std::dec << ph->hook << " gid="
	    << std::to_string (gid) << ",id=" << std::to_string (id);

      }
    hwph = nfq_get_packet_hw (tb);
    if (hwph)
      {
	int i, k, hlen = ntohs (hwph->hw_addrlen);
	std::cout << "hw_src_addr=";
	for (i = 0; i < hlen - 1; k += hwph->hw_addr[i], i++)
	  std::cout << std::hex << (uint16_t) hwph->hw_addr[i];
	std::cout << std::hex << (uint16_t) hwph->hw_addr[hlen - 1] << std::dec;

	if (k == 0)
	  std::cout << "[+] should ignore local packets .... " << std::endl;
      }

    mark = nfq_get_nfmark (tb);
    if (mark)
      std::cout << "mark= " << mark << std::endl;

    ifi = nfq_get_indev (tb);
    if (ifi)
      {
	in = true;
	std::cout << " indev= " << ifi;
      }
    ifi = nfq_get_outdev (tb);
    if (ifi)
      std::cout << " outdev= " << ifi;
    ifi = nfq_get_physindev (tb);
    if (ifi)
      std::cout << " physindev= " << ifi;

    ifi = nfq_get_physoutdev (tb);
    if (ifi)
      std::cout << " physoutdev= " << ifi;

    ret = nfq_get_payload (tb, &data);
    if (ret >= 0)
      {
	iphdr *pIP = (iphdr*) data;
	std::cout << " [+] payload len =" << ret << ", ip version = "
	    << (pIP->version) << std::endl;

	if (pIP->protocol == IPPROTO_UDP || pIP->version != 4)
	  {
	    std::cout << "[+] checking internal RPC message channel"
		<< std::endl;
	    rpc_prx =
		(pIP->version == 4) ?
		    (rpc_msg_header*) &data[sizeof(iphdr) + sizeof(udphdr)] :
		    (rpc_msg_header*) &data[sizeof(ip6_hdr) + sizeof(udphdr)];
	    if (_rpcserver.is_mon_rpc_message (rpc_prx))
	      {

		std::cout << "[+]rpc msg found " << std::endl;

		udphdr *uh = (udphdr*) ((ptrdiff_t) rpc_prx - sizeof(udphdr));

		_rpcserver.process_rpc_message ((void*) rpc_prx,
						ntohs (uh->len));

		return 0xFFFFFFFE;

	      }
	  }
	if (pIP->version == 4 && pIP->protocol == IPPROTO_UDP)
	  {
	    udphdr *udp = (udphdr*) &data[sizeof(iphdr)];

	    std::string src_address = std::string (
		inet_ntop (AF_INET, &pIP->saddr, tempbuf, 1024));
	    std::string dst_address = std::string (
		inet_ntop (AF_INET, &pIP->daddr, tempbuf, 1024));
	    unsigned short dest_port = htons (udp->uh_dport);
	    unsigned short src_port = htons (udp->uh_sport);
	    bool known_stun_port = false;
	    std::cout << "[+] src ip: " << src_address << ", dst ip: "
		<< dst_address;

	    if (_rpcserver.is_mon_rpc_message (rpc_prx) && rpclen > 0)
	      {
		std::cout << "[+]rpc msg found " << std::endl;
		_rpcserver.process_rpc_message ((void*) rpc_prx, rpclen);
		return 0xFFFFFFFE;
	      }

	    if (INSEPECTED_PORT(dest_port, src_port))
	      {

		std::cout << "[+]STUN Packet source port:" << std::dec
		    << src_port << " dest port " << dest_port;
		known_stun_port = true;

	      }

	    else

	      {
		std::cout << "[+]UDP Packet source port:" << std::dec
		    << src_port << " dest port " << dest_port;
	      }

	    if (known_stun_port)
	      {

		nfh->ip4_udp_stun_throughput_counter += ntohs (pIP->tot_len);

		ip_address v4 (in ? pIP->saddr : pIP->daddr);
		std::cout << "[+] CIDR  = " << v4.get_prefix ();

		if (nfh->pMiniDal_)
		  {

		    std::string aso = nfh->pMiniDal_->get_aso (v4);
		    std::string asn = nfh->pMiniDal_->get_asn (v4);
		    std::cout << "[+] ASN  = " << asn << " ASO = " << aso;

		  }

		//check internal RPC :

		invalid = nfh->print_stun (pIP);

		if (invalid)
		  verdict = policy_verdicts::MALFORMED_FIELD;

	      }
#ifdef PRINT_PACKETS
			unsigned char *pPayload = &data[sizeof(iphdr) + sizeof(udphdr)];

			for (size_t i = 0; i < size_t(ret); ++i) {

				if (pPayload == &data[i]) {

					if (known_stun_port)
						std::cout << "\033[1m\033[32m";
					//red:
					else
						std::cout << "\033[31m";
				}
				int val = data[i];
				if (i % 16 == 0)
					std::cout << std::endl << "[" << std::dec << i << "]"
							<< " ";
				std::cout << std::hex << val << " ";
			}
			std::cout << "\033[0m" << std::endl;
			std::cout << std::dec << std::endl;
			std::cout << std::endl;
			#endif //PRINT_PACKETS
	  }	//icmp4
	else if (pIP->protocol == IPPROTO_ICMP)
	  {

	    nfh->unpack4 (((char*) &pIP) + sizeof(iphdr), ret - sizeof(iphdr),
			  pIP->saddr);
	  }
	else if (pIP->version == 6)
	  {
	    struct ip6_hdr *hdr = (ip6_hdr*) data;

	    nfh->ip6_udp_stun_throughput_counter += ntohs (
		hdr->ip6_ctlun.ip6_un1.ip6_un1_plen);
	    if (hdr->ip6_ctlun.ip6_un1.ip6_un1_nxt == IPPROTO_UDP)
	      {
		struct udphdr *udp = (udphdr*) &data[sizeof(ip6_hdr)];
		std::string log;
		uint16_t src_port = ntohs (udp->uh_sport), dest_port = ntohs (
		    udp->uh_dport);
		log += " [+][+]source address "
		    + std::string (
			inet_ntop (AF_INET6, &hdr->ip6_src, tempbuf, 1024));
		log += " [+][+]dest  address "
		    + std::string (
			inet_ntop (AF_INET6, &hdr->ip6_dst, tempbuf, 1024));
		std::cout << "[+][+] ipv6 udp " << log;
		std::cout << "[+][+]  dest port : " << dest_port
		    << ", src port = " << src_port;
		if (rtp_packet::is_rtp (
		    (uint8_t*) &data[sizeof(ip6_hdr) + sizeof(udphdr)],
		    ntohs (udp->len)))
		  {
		    struct rtp_packet::rtp_header *header =
			reinterpret_cast<rtp_packet::rtp_header*> ((char*) &data[sizeof(ip6_hdr)
			    + sizeof(udphdr)]);

		    std::cout << "[+] RTP HEADER  version :" << header->version;
		    std::cout << "[+] RTP HEADER  extension :"
			<< header->extension;
		    std::cout << "[+] RTP HEADER  marker :" << header->marker;
		    std::cout << "[+] RTP HEADER  padding :" << header->padding;
		    std::cout << "[+] RTP HEADER  payloadType :"
			<< header->payloadType;
		    std::cout << "[+] RTP HEADER  sequenceNumber :"
			<< header->sequenceNumber;
		    std::cout << "[+] RTP HEADER  ssrc :" << header->ssrc;
		    std::cout << "[+] RTP HEADER  timestamp :"
			<< header->timestamp;

		  }

		if (INSEPECTED_PORT(dest_port, src_port))
		  {
		    //check ip6 behavior...

		    ip_address v6 (in ? hdr->ip6_src : hdr->ip6_dst);
		    std::cout << "[+] CIDR6  = " << v6.get_prefix ();

		    if (nfh->pMiniDal_)
		      {

			try
			  {
			    std::string aso = nfh->pMiniDal_->get_aso (v6);

			    std::string asn = nfh->pMiniDal_->get_asn (v6);

			    std::cout << "[+] ASN  = " << asn << " ASO = "
				<< aso;
			  }
			catch (io::sqlite::error &err)
			  {
			    std::cerr << err.what () << ", code = "
				<< err.code ();
			  }

		      }

		    invalid = nfh->print_stun (pIP);
		    if (invalid)
		      {
			verdict = policy_verdicts::MALFORMED_FIELD;
		      }

		  }
		//else
		//always check for rtp, it's cheap.

	      }
	    else if (hdr->ip6_ctlun.ip6_un1.ip6_un1_nxt == IPPROTO_ICMPV6)
	      {

		nfh->unpack6 (((char*) hdr) + sizeof(ip6_hdr),
			      ret - sizeof(ip6_hdr), &hdr->ip6_src, 0);
	      }
	    else if (hdr->ip6_ctlun.ip6_un1.ip6_un1_nxt == IPPROTO_ICMP)
	      {
		nfh->unpack6 (((char*) hdr) + sizeof(ip6_hdr),
			      ret - sizeof(ip6_hdr), &hdr->ip6_src, 0);

	      }

	  }

	  {

	    provallo::ip_address ip =
		pIP->protocol == 6 ?
		    (in ?
			provallo::ip_address (
			    reinterpret_cast<ip6_hdr*> (data)->ip6_src) :
			provallo::ip_address (
			    reinterpret_cast<ip6_hdr*> (data)->ip6_dst)) :
		    (in ?
			provallo::ip_address (
			    ntohl (reinterpret_cast<iphdr*> (data)->saddr)) :
			provallo::ip_address (
			    ntohl (reinterpret_cast<iphdr*> (data)->daddr)));
	    if (!ip.is_link_local () && !ip.is_site_local ()
		&& !ip.is_multicast_link_local ())
	      {
		auto it = nfh->ip_throughput.find (ip);
		if (it != nfh->ip_throughput.end ())

		  {
		    //update
		    struct traffic_indice ip_indice = it->second;

		    ip_indice.counter++;
		    ip_indice.duration += std::chrono::duration_cast<
			std::chrono::microseconds> (
			start - ip_indice.last_time_point).count ();
		    ip_indice.last_time_point = start;

		    ip_indice.total_data += ret;

		    if (in)
		      ip_indice.total_inbound += ret;
		    else
		      ip_indice.total_outbound += ret;

		    //insert on first rtt only once
		    if (ip_indice.first_rtt == 0)
		      {
			ip_indice.first_rtt = ip_indice.duration;

			nfh->pMiniDal_->insert_ip_rtt (ip, ip_indice.total_data,
						       ip_indice.total_outbound,
						       ip_indice.total_inbound,
						       ip_indice.duration,
						       ip_indice.first_rtt,
						       ip_indice.counter);

		      }
		    else if (counter % 2 == 0)

		      nfh->pMiniDal_->update_ip_rtt (ip_indice.total_data,
						     ip_indice.total_outbound,
						     ip_indice.total_inbound,
						     ip_indice.duration,
						     ip_indice.counter, ip);

		    nfh->ip_throughput[ip] = ip_indice;

		  }
		else
		  {
		    struct traffic_indice n;
		    n.last_time_point =
			std::chrono::high_resolution_clock::now ();

		    n.duration =
			std::chrono::duration_cast<std::chrono::microseconds> (
			    n.last_time_point - start).count ();
		    ;				//avoid stats zero div
		    n.total_data = ret;
		    n.first_rtt = 0ull;
		    n.counter = 1ull;

		    if (in)
		      {
			n.total_inbound = ret;
			n.total_outbound = 0;
		      }
		    else
		      {
			n.total_outbound = ret;
			n.total_inbound = 0;
		      }
		    nfh->ip_throughput.insert (
			std::pair<provallo::ip_address, traffic_indice> (ip,
									 n));

		  }
	      }

	  }
      }

    return verdict;
  }

  int
  netfilter_helper::callback (struct nfq_q_handle *qh, struct nfgenmsg *nfmsg,
			      struct nfq_data *nfa, void *data)
  {
    netfilter_helper *pThis = (netfilter_helper*) data;
    u_int32_t id;
    u_int32_t verdict = 0;
    struct nfqnl_msg_packet_hdr *ph = nullptr;
    bool indev = false;

    std::cout << "[+]entering netfilter  callback";
    auto start = std::chrono::high_resolution_clock::now ();
    const int MIN_MODEL_SIZE = 20;
    ph = nfq_get_msg_packet_hdr (nfa);
    if (ph)
      id = ntohl (ph->packet_id);
    if (nfq_get_indev (nfa))
      {
	indev = true;
      }
    if (nfa)
      verdict = pThis->dissect_packet (nfa, pThis);

    auto stop = std::chrono::high_resolution_clock::now ();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds> (
	stop - start);
    std::cout << "[+] finished at " << duration.count () << "ns";

    if (verdict == 0xFFFFFFFE)
      {

	std::cout << "[+] dropping RPC message " << std::endl;
	return nfq_set_verdict (qh, id, NF_ACCEPT, 0, NULL);

	//return nfq_set_verdict(qh, id, NF_STOLEN, 0, NULL);//RPC Message ;-)
      }

    if (!verdict) //verdict ==0
      return nfq_set_verdict (qh, id, NF_ACCEPT, 0, NULL);

    if (indev)
      {

	//check policies according to verdict .

	std::cout << "[+] DROPPPED  packet verdict, checking clusters:" << id;
	if (pThis->demo_data_model.size () > MIN_MODEL_SIZE)
	  pThis->demo_check ();
	return nfq_set_verdict (qh, id, NF_DROP, 0, NULL);
      }
    else
      {
	std::cout << "[+] WARNING, ALLOWING packet out : " << id;
	return nfq_set_verdict (qh, id, NF_ACCEPT, 0, NULL);
      }
  }
  std::vector<std::array<int32_t, 8>>
  normalize (const std::vector<std::array<int32_t, 8>> &demo_data_model,
	     size_t size)
  {
    std::vector<std::array<int32_t, 8>> ret_zero;
    std::vector<std::array<int32_t, 8>> ret_one;
    std::vector<std::array<int32_t, 8>> ret;

    std::uniform_int_distribution<int> dist (0, 1);
    std::random_device rd;

    if (demo_data_model.size () <= size)
      return demo_data_model;

    //
    for (auto it : demo_data_model)
      {

	if (it[it.size () - 1] == 0)
	  {
	    if (ret_zero.size () >= size / 2)
	      {
		//coin-flip:dist(rd)
		//maybe override
		if (dist (rd))
		  {
		    //somewhere randomly.
		    std::uniform_int_distribution<int> dist (0,
							     ret_zero.size ());
		    ret_zero[dist (rd) % ret_zero.size ()] = it;
		  }
	      }
	    else
	      {
		ret_zero.push_back (it);
	      }
	  }

	else
	  {
	    if (ret_one.size () >= size / 2)
	      {
		//coin-flip:dist(rd)
		//maybe override
		if (dist (rd))
		  {
		    //somewhere randomly.
		    std::uniform_int_distribution<int> dist (0,
							     ret_zero.size ());
		    ret_one[dist (rd) % ret_one.size ()] = it;
		  }
	      }
	    else
	      ret_one.push_back (it);
	  }

	if (ret_zero.size () + ret_one.size () >= size)
	  break;

      }
    std::merge (ret_zero.begin (), ret_zero.end (), ret_one.begin (),
		ret_one.end (), std::back_inserter (ret));//ret_zero+ret_one;
    return ret;
    //iterate data points and select cases;

  }
  void
  netfilter_helper::demo_check ()
  {
    //separate into 2 classes and check means.
    static bool first_time = true;
    auto start = std::chrono::high_resolution_clock::now ();
    static float fpr = 0.f, tpr = 0.f;
    static int fp = 0, tp = 0;
    //_data_model_copy;
    //std::for_each( demo_data_model.begin() + (demo_data_model.size()/2) , demo_data_model.end(), [demo_data_model_copy]{demo_data_model_copy.push_back(*it);});
    //for (std::vector<std::array<int32_t, 8 >>::iterator it = demo_data_model.begin() + (demo_data_model.size()/2) ;it!= demo_data_model.end();++it)

    //split model and merge:

    std::vector<std::array<int32_t, 8>> dataset = normalize (
	this->demo_data_model, 50);

    if (dataset.size () < 40)
      {
	std::cout << "reduced dataset is unbalanced";
	return;

      }
    static auto cluster_data = provallo::kmeans_lloyd (dataset, 2);
    if (first_time)
      {

	std::cout << "Building KMeans over model first time takes a while...:"
	    << std::endl;
	first_time = false;
	std::cout << "Means:" << std::endl;
	for (const auto &mean : std::get<0> (cluster_data))
	  {
	    std::string out = "(" + std::to_string (mean[0]) + ","
		+ std::to_string (mean[1]) + ")";
	    std::cout << out;

	  }
	std::cout << "Cluster labels:" << std::endl;
	std::cout << "[]Point:";
	for (const auto &point : demo_data_model)
	  {
	    std::stringstream value;
	    value << "(" << point[0] << "," << point[1] << ")";
	    std::cout << std::setw (14) << value.str ();
	  }
	std::cout << std::endl;
	std::cout << "Label:";
	for (const auto &label : std::get<1> (cluster_data))
	  {
	    std::cout << std::setw (14) << label;
	  }

      }
    auto query = demo_data_model.back ();
    auto centroids = std::get<0> (cluster_data);

    auto res = predict (centroids, query);
    if (res != query[query.size () - 1])
      {

	std::cout << "[+]   clusters incorrectly predicted the last sample: "
	    << fp << ",TP:" << tp;
	//update fp:
	fp++;
	//retrain:
	if (tp == 0)
	  {
	    std::cout
		<< "[+]   clusters needs to grow, ignoring and retraining again FP : "
		<< fp << ",TP:" << tp;

	    fp = 0;

	    first_time = true;
	    return;
	  }

      }
    else
      {
	tp++;
	std::cout << "[+]   clusters predictions correct , FP : " << fp
	    << ",TP:" << tp;

      }
    auto stop = std::chrono::high_resolution_clock::now ();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds> (
	stop - start);

    std::cout << "[+] classifier cost :" << duration.count () << "ns"
	<< ",predict=" << res << "verdict was " << query[query.size () - 1];
    std::cout << std::endl;
  }

  inline static const char*
  icmp6_type_name (int id)
  {

    static const std::map<int, std::string> ids =
      {
	{ 0, "Empty" },
	{ ICMP6_DST_UNREACH, "Destination Unreachable" },
	{ ICMP6_PACKET_TOO_BIG, "Packet too big" },
	{ ICMP6_TIME_EXCEEDED, "Time Exceeded" },
	{ ICMP6_PARAM_PROB, "Parameter Problem" },
	{ ICMP6_ECHO_REPLY, "Echo Reply" },
	{ ICMP6_ECHO_REQUEST, "Echo Request" },
	{ MLD_LISTENER_QUERY, "Listener Query" },
	{ MLD_LISTENER_REPORT, "Listener Report" },
	{ MLD_LISTENER_REDUCTION, "Listener Reduction" } };

    for (auto it : ids)
      {
	if (it.first == id)
	  return it.second.c_str ();
      }
    return "unknown ICMP type";

  }

  static const char*
  icmp_type_name (int id)
  {

    static const std::vector<std::pair<int, std::string>> ids =
      {
	{ ICMP_ECHOREPLY, "Echo Reply" },
	{ ICMP_DEST_UNREACH, "Destination Unreachable" },
	{ ICMP_SOURCE_QUENCH, "Source Quench" },
	{ ICMP_REDIRECT, "Redirect (change route)" },
	{ ICMP_ECHO, "Echo Request" },
	{ ICMP_TIME_EXCEEDED, "Time Exceeded" },
	{ ICMP_PARAMETERPROB, "Parameter Problem" },
	{ ICMP_TIMESTAMP, "Timestamp Request" },
	{ ICMP_TIMESTAMPREPLY, "Timestamp Reply" },
	{ ICMP_INFO_REQUEST, "Information Request" },
	{ ICMP_INFO_REPLY, "Information Reply" },
	{ ICMP_ADDRESS, "Address Mask Request" },
	{ ICMP_ADDRESSREPLY, "Address Mask Reply" } };

    for (auto it : ids)
      {
	if (it.first == id)
	  return it.second.c_str ();
      }

    return "unknown ICMP type";

  }

  void
  netfilter_helper::update_icmp_triptimes (int sz, uint32_t *tp,
					   const std::string &ringapp,
					   uint16_t recv_seq, int ttl)
  {
    //
    // P{o}r{t}
    //

  }
  void
  netfilter_helper::unpack4 (char *buf, size_t sz, uint32_t from)
  {
    struct icmp *icmppkt;
    char address_buf[INET_ADDRSTRLEN] =
      { };
    bzero (address_buf, INET_ADDRSTRLEN);
    icmppkt = (struct icmp*) buf;
    std::string icmp_typename = icmp_type_name (icmppkt->icmp_type);
    uint8_t icmp_code = icmppkt->icmp_code;
    uint16_t icmp6_cksum = ntohs (icmppkt->icmp_cksum); /* checksum field */
    uint16_t recv_seq = ntohs (icmppkt->icmp_seq);
    uint32_t *tp = (uint32_t*) icmppkt->icmp_data;
    inet_ntop (AF_INET, &from, address_buf, sizeof(address_buf));

    if (icmppkt->icmp_type != ICMP_ECHOREPLY)
      {
	std::string log = std::string ("[+] ICMP4 Filter :") + icmp_typename
	    + std::string (" (") + std::to_string (icmp_code)
	    + std::string (")") + std::string (",sequence : ")
	    + std::to_string (recv_seq) + std::string (" from:")
	    + std::string ((const char*) address_buf);
	std::cout << log;
      }
    pMiniDal_->report_icmp_stats (icmp_typename, (size_t) icmp_code,
				  (size_t) recv_seq, (size_t) icmp6_cksum,
				  std::string ((const char*) address_buf));

  }
  void
  netfilter_helper::unpack6 (char *packet, size_t sz, struct in6_addr *from,
			     int hoplimit)
  {
    struct icmp6_hdr *icmppkt;
    char addr_buf[INET6_ADDRSTRLEN] =
      { };
    /* discard if too short */
    if (sz < (1 + sizeof(struct icmp6_hdr)))
      return;
    icmppkt = (struct icmp6_hdr*) packet;
    std::string icmp6_typename = icmp6_type_name (icmppkt->icmp6_type);
    uint8_t icmp_code = icmppkt->icmp6_code;
    uint16_t icmp6_cksum = ntohs (icmppkt->icmp6_cksum); /* checksum field */
    uint16_t recv_seq = ntohs (icmppkt->icmp6_seq);
    inet_ntop (AF_INET6, &from, addr_buf, sizeof(addr_buf));

    if (icmppkt->icmp6_type != ICMP6_ECHO_REPLY)
      {
	std::string log = std::string ("[+] ICMP6 Filter :") + icmp6_typename
	    + " (" + std::to_string (icmp_code) + std::string (")")
	    + std::string (",sequence : ") + std::to_string (recv_seq)
	    + " from:" + std::string ((const char*) addr_buf);
	std::cout << log;
      }

    pMiniDal_->report_icmp_stats (icmp6_typename, (size_t) icmp_code,
				  (size_t) recv_seq, (size_t) icmp6_cksum,
				  std::string ((const char*) addr_buf));

  }
  netfilter_helper::~netfilter_helper ()
  {

#ifdef __FLUSH__IPTABLES__

	int sysret = -1;
	sysret = execl("/usr/sbin/iptables", "iptables", "-F", NULL);
	if (sysret < 0)
		std::cerr << "[-]execl returned " << sysret << ",errno=" << errno;
#endif
    nfq_destroy_queue (_qh);
    nfq_close (_h);
    if (pMiniDal_)
      delete pMiniDal_;
    pMiniDal_ = nullptr;
  }
  struct provallo::port_distribution<uint16_t> netfilter_helper::_ip6ports;
  struct provallo::port_distribution<uint16_t> netfilter_helper::_ip4ports;
  std::map<uint16_t, provallo::ip_address> netfilter_helper::port_mapped_addresses;

  void
  netfilter_helper::handle_packet (struct nfq_data *tb)
  {

    for (auto handle : _handlers)
      {
	//parallelize
	provallo::packet_handler handler = handle;
	(*handler) (tb);
      }

  }

  void
  netfilter_helper::register_packet_handler (provallo::packet_handler &handler)
  {

    _handlers.push_back (handler);

  }

  u_int32_t
  dns_filter::dissect_packet (struct nfq_data *tb, netfilter_helper *nfh)
  {
    std::cout<<"[+] dissect dns packet"<<std::endl;
    return 0;
  }

  dns_filter::~dns_filter ()
  {
  }


} /* namespace provallo */



#endif  //metamon
