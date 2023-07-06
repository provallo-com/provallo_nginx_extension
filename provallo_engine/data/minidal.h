/*
 * minidal.h
 *
 *  Created on: May 11, 2021
 *      Author: kardon

 ================
 */

#ifndef MINIDAL_H_
#define MINIDAL_H_
#include "../util/ip_address.h"
 
#include "linux/if_link.h"
#include "../third_party/sqlite.h"
#include <vector>
#include "../util/interface_info.h"

 
namespace provallo
{

  /*typedef struct
  {
    NATType identified_type;
    double probability;
  } nat_type;
  */
  
  typedef std::map<std::string, std::string> cache_map;
  typedef std::map<std::string, std::string>::iterator cache_map_it;
  typedef std::map<std::string, std::string>::const_iterator cache_map_cit;
  
  
  
  enum last_rowid : size_t
  {
    STUNLOG = 0,
    TRANSACTION_ANALYSIS = 1,
    ICMP_STATS = 2,
    ATTRIBUTE = 3,
    ADDRESS_ATTRIBUTE = 4,
    CANDIDATE = 5,
    MAX
  };
  const std::string row_ids[] =
    { "STUNLOG", "TRANSACTION_ANALYSIS", "ICMP_stats", "stun_attribute",
	"stun_address_attribute", "CandidateInformation" };

  class mini_dal
  {
    io::sqlite::db *pDB;
    cache_map aso_cache;
    cache_map asn_cache;
    std::map<size_t, uint64_t> last_rowid_map;
    void
    init_last_rowids (); // initialize the map set for last_rowid values for the tables

  public:
    explicit
    mini_dal (io::sqlite::db *db) :
	pDB (db)
    {
    }
    const provallo::interface_info&
    update_adapter_addresses ();
    // read global /proc/net/sockstat and  /proc/net/sockstat6.
    void
    report_sock_stats ();
    //
    void
    report_sys_info ();

    //
    void
    report_link_statistics (const std::map<uint16_t, rtnl_link_stats64> &stats);

    void
    report_icmp_stats (const std::string &typename_, size_t code,
		       size_t recv_seq, size_t cksum, const std::string &from);
    //get key values set from DB
    bool
    get_configuration (std::map<std::string, std::string> &fill);
    // report session state
    void
    report_session (uint32_t transaction_id[3], size_t participating_candidates,
		    uint16_t last_method /*= DATA_IND*/, uint16_t last_attribute /*=HANGOUTS*/);

    // log stun requests valid/invalid requests
    void
    log_stun (uint16_t msg_type, uint16_t msg_len, uint16_t src_port,
	      uint16_t dst_port, uint16_t class_,
	      const std::string &transaction_id, uint16_t ip_version,
	      const std::string &src_ip, const std::string &dst_ip,
	      uint16_t msg_type_class, uint16_t method, uint8_t ttl,
	      uint8_t tos, bool is_channel, bool libwebrtc_valid,
	      uint32_t total_ip_len);
    // log stun attributes

    void
    insert_attribute (uint16_t attribute_type, uint16_t att_length,
		      void *att_offset, size_t validated_size);

    // insert address
    void
    insert_address_attribute (const std::string &from, const std::string &to,
			      const std::string &ref_address,
			      uint16_t mapped_port, uint8_t tos, uint8_t ttl);
    //get number i.e. AS4302
    std::string
    get_asn (const ip_address &src);
    // get organization i.e. Microsoft,Inc
    std::string
    get_aso (const ip_address &src);
    // get other alternate ip addresses associated with this specific source address
    std::vector<ip_address>
    alternate_ip (const ip_address &src);
    //get pre-cached Nat type for the ip address, if unknown it will try to identify.
    /*nat_type
    get_nat_type (const ip_address &src); */
    // update candidate information, including generation/foundation hueristics.
    //void
    //update_candidate (candidate &can);
    // report event for unknown ICMPs/reserved flags, and other potential indicators.
    //void report_event (int ifi_index, int event_type,uint32_t event_severity,const std::string& event_info,void* offset,size_t blobsize);
    void
    update_ip_rtt (uint64_t total_data, uint64_t total_outbound,
		   uint64_t total_inbound, uint64_t duration, uint64_t counter,
		   const ip_address &ip);
    void
    insert_ip_rtt (const ip_address &IP, uint64_t total_data,
		   uint64_t total_outbound, uint64_t total_inbound,
		   uint64_t duration, uint64_t first_rtt, uint64_t counter);

    bool
    check_quota_exceeded (uint64_t quota);
    //

    void
    archive_stun_log ();
    //

    void
    archive_icmp ();
    //
    void
    label_detection_category (uint64_t stun_log, size_t action_id,
			      size_t risk_id, size_t threat_id);

    //
    void
    archive_attributes ();
    //
    void
    run_query (const std::string &query);
  };

} /* namespace provallo */

#endif /* MINIDAL_H_ */
