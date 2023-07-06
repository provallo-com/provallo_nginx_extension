/*
 * minidal.cpp
 *
 *  Created on: May 11, 2021
 *      Author: kardon
 */

#include "minidal.h"
#include <strstream>
#include <iomanip>
#include <sys/sysinfo.h>

namespace provallo
{

//TODO:
//1) Add Setup code (adapters)
//2) Add archive trigger
//3)

  void
  mini_dal::report_session (uint32_t transaction_id[3],
			    size_t participating_candidates,
			    uint16_t last_method, uint16_t last_attribute)
  {

    std::strstream str_buf;
    str_buf << ntohl (transaction_id[0]) << ntohl (transaction_id[1])
	<< ntohl (transaction_id[2]);
    std::string id = str_buf.str ();

    try
      {
	io::sqlite::stmt u (
	    *pDB,
	    "INSERT INTO SESSION (TransactionID,Participants) VALUES(?,?)");
	u.bind ().text (1, id).int32 (2, participating_candidates).int32 (
	    3, last_method).int32 (4, last_attribute);
	u.exec ();

      }
    catch (io::sqlite::error &e)
      {
	std::cerr << e.what () << "," << e.code ();
	io::sqlite::db *pOld = pDB;
	pDB = new io::sqlite::db (pOld->file_name ());
	delete pOld;
      }

  }

  std::string
  mini_dal::get_asn (const ip_address &src)
  {

    std::string last_asn = "N/A";
    cache_map_it it = this->asn_cache.find (src.to_string ());
    if ((it) != asn_cache.end ())
      return it->second;
    else
      {

	try
	  {
	    std::string query = std::string (
		"SELECT DISTINCT autonomous_system_number FROM ")
		+ std::string (
		    (src.family () == AF_INET) ?
			" \'ASN-Blocks-IPv4\'" : " \'ASN-Blocks-IPv6\'");
	    query += " WHERE network like ?";
	    std::string prefix = src.get_query ();
	    //prefix[prefix.length() - 1] = '%';
	    std::cout << "[+] Query : " << query << "prefix=" << prefix;
	    io::sqlite::stmt u (*pDB, query.c_str ());
	    //hackish:
	    u.bind ().text (1, prefix);
	    //debug:
	    while (u.step ())
	      {
		last_asn = u.row ().text (0);
		std::cout << "[+] ASN Be LIKE " << last_asn << "?";
	      }
	    asn_cache.insert (
		std::pair<std::string, std::string> (src.to_string (),
						     last_asn));
	  }
	catch (io::sqlite::error &e)
	  {
	    std::cerr << e.what () << "," << e.code ();
	  }
      }
    return last_asn;

  }

  std::string
  mini_dal::get_aso (const ip_address &src)
  {

    std::string last_aso = "N/A";
    cache_map_it it = this->aso_cache.find (src.to_string ());
    if ((it) != aso_cache.end ())
      {
	return it->second;
      }
    else
      {

	try
	  {

	    std::string query = std::string (
		"SELECT DISTINCT autonomous_system_organization FROM ")
		+ std::string (
		    (src.family () == AF_INET) ?
			" \'ASN-Blocks-IPv4\'" : " \'ASN-Blocks-IPv6\'");
	    query += " WHERE network like ?";
	    std::string prefix = src.get_query ();
	    std::cout << "[+] Query : " << query << "prefix=" << prefix;
	    io::sqlite::stmt u (*pDB, query.c_str ());
	    u.bind ().text (1, prefix);
	    while (u.step ())
	      {
		last_aso = u.row ().text (0);
	      }
	    aso_cache.insert (
		std::pair<std::string, std::string> (src.to_string (),
						     last_aso));

	  }
	catch (io::sqlite::error &e)
	  {
	    std::cerr << e.what () << "," << e.code ();
	  }

      }
    std::cout << "[+] ASO Be LIKE " << last_aso << "?";

    return last_aso;

  }
	
  std::vector<ip_address>
  mini_dal::alternate_ip (const ip_address &src)
  {

    std::vector<ip_address> ret_Cpy;
    return ret_Cpy;

  }

  bool
  mini_dal::get_configuration (std::map<std::string, std::string> &fill)
  {

    bool bret = false;
    try
      {
	io::sqlite::stmt s (*pDB, "SELECT KEY, Value FROM pro_settings");
	while (s.step ())
	  {

	    std::string key = s.row ().text (0);
	    std::string value = s.row ().text (1);
	    fill.insert (std::pair<std::string, std::string> (key, value));
	    bret = true;

	    std::cout << "[+] DB SETTINGS : " << key << '=' << value
		<< std::endl;

	  }
      }
    catch (io::sqlite::error &exx)
      {
	std::cerr << "get_configuration" << " Failed " << exx.what () << "code "
	    << exx.code ();
      }
    return bret;
  }

//LOG stun information, use the same transaction_id for attributes table.
//ALTER TABLE TO SAVE TTL/TOS always
  void
  mini_dal::log_stun (uint16_t msg_type, uint16_t msg_len, uint16_t src_port,
		      uint16_t dst_port, uint16_t class_,
		      const std::string &transaction_id, uint16_t ip_version,
		      const std::string &src_ip, const std::string &dst_ip,
		      uint16_t msg_type_class, uint16_t method, uint8_t ttl,
		      uint8_t tos, bool is_channel, bool libwebrtc_valid,
		      uint32_t total_ip_len)
  {

    std::string query =
	"INSERT INTO STUNLOG(msg_type,msg_len,src_port,dst_port,class,TransactionID,ip_version,src_ip,dst_ip,msg_type_class,method,ttl,tos,is_channel,total_ip_len) VALUES ( ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)";
    std::string analysis =
	"INSERT INTO TRANSACTION_ANALYSIS (TransactionID, channel_id,libwebrtc_validation) VALUES(?,?,?)";

    try
      {
	io::sqlite::stmt u (*pDB, query.c_str ());
	u.bind ().int32 (1, msg_type).int32 (2, msg_len).int32 (3, src_port).int32 (
	    4, dst_port).int32 (5, class_).text (6, transaction_id).int32 (
	    7, ip_version).text (8, src_ip).text (9, dst_ip).int32 (
	    10, msg_type_class).int32 (11, method).int32 (12, ttl).int32 (13,
									  tos).int32 (
	    14, is_channel ? 1 : 0).int32 (15, total_ip_len);
	u.exec ();
	this->last_rowid_map[STUNLOG] = pDB->last_row_id ();

      }
    catch (io::sqlite::error &err)
      {
	std::cerr << "[-]mini_dal::log_stun[1] sqlite error : " << err.what ()
	    << ", code " << err.code () << query.c_str ();

      }

    try
      {

	//msg_type is channel id
	io::sqlite::stmt u (*pDB, analysis.c_str ());
	u.bind ().text (1, transaction_id.c_str ()).int32 (2, msg_type).int32 (
	    3, libwebrtc_valid ? 1 : 0);
	u.exec ();
	this->last_rowid_map[TRANSACTION_ANALYSIS] = pDB->last_row_id ();
      }
    catch (io::sqlite::error &err)
      {
	std::cerr << "[-]mini_dal::log_stunp[2] sqlite error : " << err.what ()
	    << ", code " << err.code ();

      }

  }

  /*nat_type
  mini_dal::get_nat_type (const ip_address &src)
  {

    nat_type ret =
      { NAT_NONE, 0. };
    //ret.identified_type = parse_nat_type();
    //ret.probability = probability;
    return ret;

  }
  */

  void
  mini_dal::insert_attribute (uint16_t attribute_type, uint16_t att_length,
			      void *att_offset, size_t validated_size)
  {
    std::string query =
	"INSERT INTO stun_attribute (att_type,att_length,att_value ,log_id) VALUES(?,?,?,?)";
    if (att_offset == nullptr || !validated_size)
      return;
    try
      {

	//[][][][]//
	io::sqlite::stmt u (*pDB, query.c_str ());
	u.bind ().int32 (1, attribute_type).int32 (2, att_length).blob (
	    3, att_offset, validated_size).int64 (
	    4, this->last_rowid_map[STUNLOG]);
	u.exec ();
	this->last_rowid_map[ATTRIBUTE] = pDB->last_row_id ();

      }
    catch (io::sqlite::error &err)
      {
	std::cerr << "[-] sqlite error : " << query << std::string (" , ")
	    << err.what () << std::string (", code ") << err.code ();
      }

  }

  void
  mini_dal::insert_address_attribute (const std::string &from,
				      const std::string &to,
				      const std::string &ref_address,
				      uint16_t mapped_port, uint8_t tos,
				      uint8_t ttl)
  {
    std::string query =
	"INSERT INTO stun_address_attribute (\'from\',\'to\',\'mapped_address\',\'mapped_port\',tos,ttl,log_id) VALUES(?,?,?,?,?,?,?)";
    try
      {
	//[][][][]//
	io::sqlite::stmt u (*pDB, query.c_str ());
	u.bind ().text (1, from).text (2, to).text (3, ref_address).int32 (
	    4, mapped_port).int32 (5, tos).int32 (6, ttl).int64 (
	    7, this->last_rowid_map[STUNLOG] + 1);
	u.exec ();
	this->last_rowid_map[ADDRESS_ATTRIBUTE] = pDB->last_row_id ();

      }
    catch (io::sqlite::error &err)
      {
	std::cerr << "[-] sqlite error:" << query << std::string (",")
	    << err.what () << std::string (",code") << err.code ();
      }
  }

  void
  mini_dal::report_icmp_stats (const std::string &typename_, size_t code,
			       size_t recv_seq, size_t cksum,
			       const std::string &from)
  {

    std::string query =
	"INSERT INTO ICMP_stats (\'typename\',\'code\',\'recv_seq\',\'cksum\',\'from\') VALUES(?,?,?,?,?)";
    try
      {
	io::sqlite::stmt u (*pDB, query.c_str ());
	u.bind ().text (1, typename_).int32 (2, code).int32 (3, recv_seq).int32 (
	    4, cksum).text (5, from);
	u.exec ();
	this->last_rowid_map[ICMP_STATS] = pDB->last_row_id ();

      }
    catch (io::sqlite::error &err)
      {
	std::cerr << "[-] sqlite error : " << query << std::string (" , ")
	    << err.what () << std::string (", code ") << err.code ();
      }
  }
  /* \
   *
   *
   * commented out - for nginx integration. 
   *
   *
   *
   *
  void
  mini_dal::update_candidate (candidate &can)
  {

    std::string query =
	"INSERT INTO CandidateInformation (Component,Protocol,Address,EstimatedPriority,Generation,Foundation, AttributeUser, AttributePassword,NetworkName,NAT_FAMILY,NAT_TYPE) (?,?,?,?,?,?,?,?,?,?,?)";

    try
      {
	io::sqlite::stmt u (*pDB, query.c_str ());
	u.bind ().int32 (1, can.get_component ()).int32 (2, can.get_protocol ()).text (
	    3, can.address ().to_string ()).int32 (4, can.get_priority ()).int32 (
	    5, can.get_generation ()).int32 (6, can.get_foundation ()).text (
	    7, can.get_rncoded_username ()).text (8,
						  can.get_rncoded_password ()).text (
	    9, can.get_network_name ()).int32 (10, can.get_nat_family_type ()).int32 (
	    11, can.get_nat_type ());

	u.exec ();
	this->last_rowid_map[CANDIDATE] = pDB->last_row_id ();

      }
    catch (io::sqlite::error &err)
      {
	std::cerr << "[-] sqlite error : " << query << std::string (" , ")
	    << err.what () << std::string (", code ") << err.code ();
      }

  }
  */
  bool
  mini_dal::check_quota_exceeded (uint64_t quota)
  {

    bool ret = false;
    uint64_t quota_sum = 0;

    std::string sql_query = "select sum(seq) as total from sqlite_sequence";
    io::sqlite::stmt u (*pDB, sql_query.c_str ());

    while (u.step ())
      {
	quota_sum = u.row ().int64 (0);
      }
    if (quota_sum > 0)
      return quota_sum < quota;
    return ret;
  }

  void
  mini_dal::archive_stun_log ()
  {

    std::string sql_query =
	"BEGIN TRANSACTION;"
	    "DROP VIEW attribute_view_full ;ALTER TABLE STUNLOG RENAME TO ?;"
	    "CREATE TABLE STUNLOG("
	    "\"id\"	INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,"
	    "\"msg_type	INTEGER NOT NULL,"
	    "\"msg_len	INTEGER NOT NULL,"
	    "\"src_port	INTEGER,"
	    "\"dst_port	INTEGER,"
	    "\"class	INTEGER,"
	    "\"TransactionID	TEXT,"
	    "\"ip_version	INTEGER NOT NULL,"
	    "\"src_ip	TEXT,"
	    "\"dst_ip	TEXT,"
	    "\"msg_type_class	INTEGER,"
	    "\"method	INTEGER,"
	    "\"ts	TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
	    "\"ttl	INTEGER,"
	    "\"tos	INTEGER); "
	    "CREATE VIEW \'attribute_view_full\' AS SELECT stun_attribute.id,stun_attribute.att_type,attribute_type.description,stun_attribute.\'att_length\',STUNLOG.ip_version,STUNLOG.method,STUNLOG.method,STUNLOG.src_port,STUNLOG.dst_port,STUNLOG.class,hex(stun_attribute.att_value) as attribute_value,stun_attribute.ts FROM stun_attribute   left join attribute_type on attribute_type.id=stun_attribute.att_type"
	    "COMMIT;"
	    "END;";
    try
      {

	auto t = std::time (nullptr);
	auto tm = *std::localtime (&t);
	std::ostringstream oss;
	oss << std::put_time (&tm, "%d-%m-%Y %H-%M-%S");
	std::string time_s = oss.str ();
	io::sqlite::stmt u (*pDB, sql_query.c_str ());
	u.bind ().text (0, std::string ("STUNLOG_") + time_s);
	u.exec ();

      }

    catch (io::sqlite::error &err)
      {
	std::cerr << "[-] sqlite error : query= " << sql_query
	    << std::string (" , error= ") << err.what ();
      }

  }
  void
  mini_dal::archive_icmp ()
  {
    std::string query =
	"BEGIN TRANSACTION;"
	    "DROP VIEW ICMP_View;"
	    "DROP VIEW attribute_view_full;"
	    "ALTER TABLE ICMP_stats RENAME TO ?;"
	    " CREATE TABLE \"ICMP_stats\" ("
	    " \"id\"	INTEGER PRIMARY KEY AUTOINCREMENT,"
	    " \"typename\"	TEXT NOT NULL,"
	    "	\"code\"	INTEGER NOT NULL,"
	    "	\"recv_seq\"	INTEGER NOT NULL,"
	    "	\"cksum\"	INTEGER NOT NULL,"
	    "	\"from\"	TEXT,"
	    "	\"ts\"	TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
	    ");"
	    " CREATE VIEW ICMP_View AS "
	    "Select ICMP_stats.'from',"
	    "ICMP_stats.typename, count(*) as \'total\' ,"
	    "sum(recv_seq) as seq,sum(code) as total_code"
	    "from ICMP_stats group by ICMP_stats.\'from\',ICMP_stats.typename order by total desc;"
	    "CREATE VIEW attribute_view_full AS"
	    "SELECT stun_attribute.id,stun_attribute.att_type,"
	    "attribute_type.description,stun_attribute.att_length,"
	    "STUNLOG.ip_version,STUNLOG.method,STUNLOG.method,STUNLOG.src_port,STUNLOG.dst_port,STUNLOG.class,hex(stun_attribute.att_value) as attribute_value,stun_attribute.ts"
	    "FROM stun_attribute"
	    "left join attribute_type on attribute_type.id=stun_attribute.att_type"

	    "COMMIT;"
	    "END;";

  }

  void
  mini_dal::report_link_statistics (
      const std::map<uint16_t, rtnl_link_stats64> &stats)
  {
    std::string sql_query =
	"INSERT INTO   linkstatistics64 (interface_id,rx_packets,tx_packets,rx_bytes,tx_bytes,"
	    "rx_errors,tx_errors,rx_dropped,tx_dropped,multicast,"
	    "collisions,rx_length_errors,rx_over_errors,rx_crc_errors,rx_frame_errors,"
	    "rx_fifo_errors,rx_missed_errors,tx_aborted_errors,tx_carrier_errors,tx_fifo_errors,"
	    "tx_heartbeat_errors,tx_window_errors,rx_compressed,tx_compressed,rx_nohandler) "
	    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);";

    for (auto &it : stats)
      {
	try
	  {
	    io::sqlite::stmt u (*pDB, sql_query.c_str ());

	    //bind struct values :

		//refactor: 



	    u.bind ().int64 (1, it.first).int64 (2, it.second.rx_packets).int64 (
		3, it.second.tx_packets).int64 (4, it.second.rx_bytes).int64 (
		5, it.second.tx_bytes).int64 (6, it.second.rx_errors).int64 (
		7, it.second.tx_errors).int64 (8, it.second.rx_dropped).int64 (
		9, it.second.tx_dropped).int64 (10, it.second.multicast).int64 (
		11, it.second.collisions).int64 (12, it.second.rx_length_errors).int64 (
		13, it.second.rx_over_errors).int64 (14,
						     it.second.rx_crc_errors).int64 (
		15, it.second.rx_frame_errors).int64 (16,
						      it.second.rx_fifo_errors).int64 (
		17, it.second.rx_missed_errors).int64 (
		18, it.second.tx_aborted_errors).int64 (
		19, it.second.tx_carrier_errors).int64 (
		20, it.second.tx_fifo_errors).int64 (
		21, it.second.tx_heartbeat_errors).int64 (
		22, it.second.tx_window_errors).int64 (23,
						       it.second.rx_compressed).int64 (
		24, it.second.tx_compressed).int64 (25, it.second.rx_nohandler);
	    u.exec ();
		

	  }
	catch (io::sqlite::error &err)
	  {
	    std::cerr << "[-] sqlite error : query= " << sql_query
		<< std::string (" , error= ") << err.what ();
	  }

      }
    this->report_sys_info ();

  }

  void
  mini_dal::report_sys_info ()
  {

    struct sysinfo ss;
    std::string query =
	"INSERT INTO MEM_INFO_STATS (uptime,LOAD_1,LOAD_5,LOAD_15,totalram,freeram,"
	    "sharedram,bufferram,totalswap,freeswap,procs,"
	    "totalhigh,freehigh,mem_unit,total_physical,total_available)VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)";
    if (sysinfo (&ss) >= 0)
      {
	try
	  {
	    io::sqlite::stmt u (*pDB, query.c_str ());

	    //bind struct values :
	    u.bind ().int64 (1, ss.uptime).int64 (2, ss.loads[0]).int64 (
		3, ss.loads[1]).int64 (4, ss.loads[2]).int64 (5, ss.totalram).int64 (
		6, ss.freeram).int64 (7, ss.sharedram).int64 (8, ss.bufferram).int64 (
		9, ss.totalswap).int64 (10, ss.freeswap).int32 (11, ss.procs).int64 (
		12, ss.totalhigh).int64 (13, ss.freehigh).int64 (14,
								 ss.mem_unit).int64 (
		15, get_phys_pages ()).int64 (16, get_avphys_pages ());

	    u.exec ();

	  }
	catch (io::sqlite::error &err)
	  {
	    std::cerr << "[-] sqlite error : query= " << query
		<< std::string (" , error= ") << err.what ();

	  }

      }

  }

  void
  mini_dal::archive_attributes ()
  {

  }

  void
  mini_dal::update_ip_rtt (uint64_t total_data, uint64_t total_outbound,
			   uint64_t total_inbound, uint64_t duration,
			   uint64_t counter, const ip_address &ip)
  {
    std::string query =
	"UPDATE ip_rtt_coll set ts = CURRENT_TIMESTAMP,total_data=?,total_outbound=?,total_inbound=?,duration=?,counter=? where IP=?";

    try
      {
	io::sqlite::stmt u (*pDB, query.c_str ());

	u.bind ().int64 (1, total_data).int64 (2, total_outbound).int64 (
	    3, total_inbound).int64 (4, duration).int64 (5, counter).text (
	    6, ip.to_string ().c_str ());

	u.exec ();

      }
    catch (io::sqlite::error &err)
      {
	std::cerr << "[-] sqlite error : query= " << query
	    << std::string (" , error= ") << err.what ();
      }

  }

  void
  mini_dal::insert_ip_rtt (const ip_address &IP, uint64_t total_data,
			   uint64_t total_outbound, uint64_t total_inbound,
			   uint64_t duration, uint64_t first_rtt,
			   uint64_t counter)
  {

    std::string query =
	"INSERT INTO ip_rtt_coll (IP,total_data,total_outbound,total_inbound,duration,first_rtt,counter)"
	    "VALUES (?,?,?,?,?,?,?);";

    try
      {
	io::sqlite::stmt u (*pDB, query.c_str ());

	u.bind ().text (1, IP.to_string ().c_str ()).int64 (2, total_data).int64 (
	    3, total_outbound).int64 (4, total_inbound).int64 (5, duration).int64 (
	    6, first_rtt).int64 (7, counter);
	u.exec ();

      }
    catch (io::sqlite::error &err)
      {
	std::cerr << "[-] sqlite error : query= " << query
	    << std::string (" , error= ") << err.what ();

      }

  }

  void
  mini_dal::init_last_rowids ()
  {

    std::string q =
	"SELECT name,seq from sqlite_sequence where name in (\"STUNLOG\",\"TRANSACTION_ANALYSIS\",\"ICMP_stats\",\"stun_attribute\",\"stun_address_attribute\",\"CandidateInformation\");) ";

    try
      {
	io::sqlite::stmt s (*pDB, q.c_str ());
	while (s.step ())
	  {

	    std::string key = s.row ().text (0);
	    uint64_t value = s.row ().int64 (1);
	    for (size_t i = 0; i < last_rowid::MAX; ++i)
	      if (key == row_ids[i])
		this->last_rowid_map.insert (
		    std::pair<size_t, uint64_t> (i, value));

	  }

      }

    catch (const io::sqlite::error &err)

      {
	std::cerr << std::string ("[-] sqlite run_query : ") << q
	    << std::string (err.what ());

      }
  }

  const provallo::interface_info&
  mini_dal::update_adapter_addresses ()
  {

    std::string query =
	" INSERT OR IGNORE INTO adapter_info (interface_id,interface_name) VALUES  ";
    std::string query2 =
	" INSERT OR IGNORE INTO adapter_addresses (address_id,interface_id,address_type,flags,address) VALUES  ";

    static provallo::interface_info infs;
    auto interfaces = infs.interfaces ();
    std::string params = "";
    for (auto interface : interfaces)
      {
	if (params.length () > 1)
	  params += ",";

	params += "( ";
	params += std::to_string (interface.first);
	params += ",";
	params += interface.second;
	params += ") ";

      }
    query += params;

    params = "";

    auto addresses = infs.get_address_info ();
    for (auto address : addresses)
      {
	if (params.length () > 1)
	  params += ",";

	params += "( ";
	params += std::to_string (address.first);
	params += ",";
	params += std::to_string (address.second.interface_id);
	params += ",";
	params += std::to_string (address.second.address_type);
	params += ",";
	params += std::to_string (address.second.flags);
	params += ",";
	params += address.second.address;
	params += ") ";

      }
    query2 += params;
    try
      {

	  {
	    io::sqlite::stmt u (*pDB, query.c_str ());
	    u.exec ();

	  }

	  {
	    io::sqlite::stmt u (*pDB, query2.c_str ());
	    u.exec ();

	  }

      }
    catch (const io::sqlite::error &err)
      {
	std::cerr << std::string ("[-] sqlite run_query : ") << query
	    << std::string (";") << query2 << std::string (err.what ());
      }

    return infs;

  }

  void
  mini_dal::run_query (const std::string &query)
  {

    try
      {
	pDB->exec (query.c_str ());

      }
    catch (const io::sqlite::error &err)
      {
	std::cerr << "[-] sqlite run_query : " << query << std::string (" , ")
	    << err.what ();

      }
  }

  void
  mini_dal::label_detection_category (uint64_t stun_log, size_t action_id,
				      size_t risk_id, size_t threat_id)
  {

    std::string query =
	"INSERT INTO local_threat_detection_table (\'log_id\',\'action_id\',\'risk_id\',\'threat_id\') VALUES(?,?,?,?)";
    try
      {
	io::sqlite::stmt u (*pDB, query.c_str ());
	u.bind ().int64 (1, stun_log).int32 (2, action_id).int32 (3, risk_id).int32 (
	    4, threat_id);
	u.exec ();
      }
    catch (io::sqlite::error &err)
      {
	std::cerr << "[-] sqlite error : " << query << std::string (" , ")
	    << err.what () << std::string (", code ") << err.code ();
      }
  }
} /* namespace provallo */
