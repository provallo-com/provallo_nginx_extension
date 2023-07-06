/*
 * netfilterhelper.h
 *
 *  Created on: Mar 10, 2021
 *      Author: kardon
 */

#ifndef HELPERS_NETFILTERHELPER_H_
#define HELPERS_NETFILTERHELPER_H_

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <linux/types.h>
#include <linux/netfilter.h>
#include <libnetfilter_queue/libnetfilter_queue.h>
#include "../util/portanalyzer.h"
#include "../util/ip_address.h"
#include <string>
#include <list>
#include "../third_party/sqlite.h"
#include "../data/minidal.h"
#include "../decision_engine/distributedkmeans.h"

#ifdef _USE_GRPC_
#include "../rpc/rpcserver.h"
#endif


#ifdef _USE_PCAP_
#include "pcap_support/pcap_packet.h"
#endif

#ifdef _USE_EBPF
#include "ebpf_support/ebpf_packet.h"
#endif

namespace provallo
{
//for each ip address we show:
  struct traffic_indice
  {
    uint64_t total_data;
    uint64_t total_outbound;
    uint64_t total_inbound;
    uint64_t duration;
    uint64_t first_rtt;
    uint64_t counter;
    std::chrono::time_point<std::chrono::high_resolution_clock,
	std::chrono::nanoseconds> last_time_point;
  };
  typedef uint32_t transaction_[3];

  typedef void
  (*packet_handler) (struct nfq_data *dfa);

  enum policy_actions : uint32_t
  {
    DROP, DROP_ALL, DROP_AND_SAVE, SANDBOX_IP
  };
  enum policy_verdicts : uint32_t
  {
    VALID, MALFORMED_STRUCTURE, MALFORMED_FIELD, BLOCKED_SOURCE
  };

#ifdef METAMONITOR 
  class netfilter_helper
  {
  protected:



    struct nfq_handle *_h;
    struct nfq_q_handle *_qh;
    std::chrono::high_resolution_clock _clock;
    std::vector<packet_handler> _handlers;
    static struct provallo::port_distribution<uint16_t> _ip6ports;
    static struct provallo::port_distribution<uint16_t> _ip4ports;
    std::map<provallo::ip_address, traffic_indice> ip_throughput;
    std::map<transaction_, std::vector<provallo::candidate>> candidates_per_transaction;
    std::map<uint64_t, uint64_t> channel_bytes;
    mini_dal *pMiniDal_;
    uint64_t ip6_udp_stun_throughput_counter;
    uint64_t ip4_udp_stun_throughput_counter;

    // model:
    // ip_version, total_len,msg_type,msg_len,n_attributes,src_port,dst_port,violated
    std::vector<std::array<int32_t, 8>> demo_data_model;
    static std::map<uint16_t, provallo::ip_address> port_mapped_addresses;
  public:
    netfilter_helper (mini_dal *p = nullptr);
    virtual
    ~netfilter_helper ();
    virtual u_int32_t
    dissect_packet (struct nfq_data *tb, netfilter_helper *nfh);
    static int
    callback (struct nfq_q_handle *qh, struct nfgenmsg *nfmsg,
	      struct nfq_data *nfa, void *data);
    bool
    print_stun (iphdr *pIP);
    void
    handle_packet (struct nfq_data *tb);
    static void
    update_icmp_triptimes (int sz, uint32_t *tp, const std::string &ringapp,
			   uint16_t recv_seq, int ttl);
    void
    unpack4 (char *buf, size_t sz, uint32_t from);
    void
    unpack6 (char *packet, size_t sz, struct in6_addr *from, int hoplimit = 0);
    void
    register_packet_handler (provallo::packet_handler &handler);
    void
    demo_check ();
    void
    run ();
#ifdef _USE_GRPC_ 
    provallo::rpc_server _rpcserver;
    provallo::rpc_broadcaster _broadcast_server;
#endif 

  };
  class dns_filter : virtual public netfilter_helper
  {
  public:
    dns_filter (mini_dal *p = nullptr) :
	netfilter_helper (p)
    {
    }
    virtual u_int32_t
    dissect_packet (struct nfq_data *tb, netfilter_helper *nfh);
    virtual
    ~dns_filter ();
  };
#endif//metamon
} /* namespace provallo */

#endif /* HELPERS_NETFILTERHELPER_H_ */
