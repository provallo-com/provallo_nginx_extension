/*
 * statcollector.cpp
 *
 *  Created on: Jun 28, 2021
 *      Author: kardon
 */
#include "../decision_engine/attribute.h"
#include "statcollector.h"
#include <netinet/in.h>
#include <linux/if_packet.h>
#include "../util/ip_address.h"
#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <cstdio>
#include <set>

namespace provallo
{
  const std::string &
  getInterface();

#define REMOVE_LAST_COMMA(x) x[x.length() - 1] = x[x.length() - 1] == ',' ? ' ' : x[x.length() - 1]

  void
  snmp4_collector::collect()
  {
    if (!_init)
      return;

    std::ifstream f_snmp4("/proc/net/snmp");
    std::string head, ip, tcp, icmp, icmp_msg, udp, udplite, header_ip,
        header_tcp, header_udp, header_icmp, header_icmp_msg, header_udplite;

    while (!f_snmp4.eof())
    {
      std::getline(f_snmp4, head);
      if (head.find("Ip:") != std::string::npos)
      {
        header_ip = head;
        std::getline(f_snmp4, ip);
      }

      if (head.find("Icmp:") != std::string::npos)
      {
        header_icmp = head;

        std::getline(f_snmp4, icmp);
      }
      if (head.find("IcmpMsg:") != std::string::npos)
      {
        header_icmp_msg = head;
        std::getline(f_snmp4, icmp_msg);
      }

      else if (head.find("Tcp:") != std::string::npos)
      {
        header_tcp = head;
        std::getline(f_snmp4, tcp);
      }
      else if (head.find("Udp:") != std::string::npos)
      {
        header_udp = head;
        std::getline(f_snmp4, udp);
      }
      else if (head.find("UdpLite:") != std::string::npos)
      {
        header_udplite = head;
        std::getline(f_snmp4, udplite);
      }

      std::vector<std::string> keys;

      stat_collector::tokenize(
          header_ip.substr(header_ip.find_first_of(":") + 1), keys, " ",
          "Ip");
      stat_collector::tokenize(
          header_icmp.substr(header_icmp.find_first_of(":") + 1), keys, " ",
          "Icmp");
      stat_collector::tokenize(
          header_icmp_msg.substr(header_icmp_msg.find_first_of(":") + 1),
          keys, " ", "IcmpMsg");
      stat_collector::tokenize(
          header_tcp.substr(header_tcp.find_first_of(":") + 1), keys, " ",
          "Tcp");
      stat_collector::tokenize(
          header_udp.substr(header_udp.find_first_of(":") + 1), keys, " ",
          "Udp");
      stat_collector::tokenize(
          header_udplite.substr(header_udplite.find_first_of(":") + 1),
          keys, " ", "UdpLite");
#ifdef DEBUG_MAPS
      std::cout << "[+]Collected " << keys.size() << " Keys ";
#endif
      std::vector<std::string> ip_values, values;
      // size_t nde = 0;
      stat_collector::tokenize(ip.substr(ip.find_first_of(":") + 1),
                               ip_values, " ");
      // std::cout<<"[+]Collected "<<ip_values.size()<<" Values ";
      // nde = ip_values.size ();
      // values  += ip_values;
      stat_collector::tokenize(icmp.substr(icmp.find_first_of(":") + 1),
                               ip_values, " ");
      // std::cout<<"Collected "<<ip_values.size()-nde<<" Values ";
      // nde = ip_values.size ();

      stat_collector::tokenize(
          icmp_msg.substr(icmp_msg.find_first_of(":") + 1), ip_values, " ");
      // std::cout<<"Collected "<<ip_values.size()-nde<<" Values ";
      // nde = ip_values.size ();

      stat_collector::tokenize(tcp.substr(tcp.find_first_of(":") + 1),
                               ip_values, " ");
      // std::cout<<"Collected "<<ip_values.size()-nde<<" Values ";
      // nde = ip_values.size ();

      stat_collector::tokenize(udp.substr(udp.find_first_of(":") + 1),
                               ip_values, " ");
      // std::cout<<"Collected "<<ip_values.size()-nde<<" Values ";
      // nde = ip_values.size ();

      stat_collector::tokenize(
          udplite.substr(udplite.find_first_of(":") + 1), ip_values, " ");

      // std::cout<<"Collected "<<ip_values.size()-nde<<" Values ";
#ifdef DEBUG_MAPS
      std::cout << "Collected Total " << keys.size() << " Keys ";
      std::cout << "Collected Total " << ip_values.size() << " Values ";
#endif
      if (keys.size() != ip_values.size())
      {
        std::cerr << "[-] error collecting snmp stats";
        std::cerr << keys.size() << std::string(" keys, ")
                  << ip_values.size() << std::string(" values.");
      }
      else
      {
        for (size_t i = 0; i < keys.size(); ++i)
        {
          // keep it as txt until usage

          bool new_map = _stats.size() == 0 || _stats.find(keys[i]) == _stats.end();

          std::pair<std::string, struct stat_collector::stat_value_cell> p(
              keys[i], ip_values[i]);
          // new map
          if (new_map)
            this->_stats.insert(p);
          else
            this->_stats[p.first] = p.second;
        }
      }

      // remap ;-)
      for (auto inst : instances())
      {
        std::vector<std::string> str_values;

        for (auto field : *inst)
        {
          // std::cout<<std::string("[+] mapping field : ") << field->get_name() <<std::endl;
          str_values.push_back(
              this->_stats[field->get_name()].to_string().empty() ? "0" : this->_stats[field->get_name()].to_string());
        }
        putValuesFromStrings(str_values.begin(), str_values.end());
        break;
      }
    }
#ifdef DEBUG_MAPS

    print_map();
#endif
    f_snmp4.close();
  }

  void
  snmp6_collector::collect()
  {

    std::ifstream f_snmp("/proc/net/snmp6");
    std::vector<std::string> vv(size(), "0");

    while (not f_snmp.eof())
    {
      std::string name, value;
      f_snmp >> name;
      f_snmp >> value;
      if (value.size() == 0)
        value = "0";
      // this->_stats.insert(std::pair<std::string,uint64_t>(name,(uint64_t)std::stoull(value,nullptr,10)));
      bool new_map = _stats.size() == 0 || _stats.find(name) == _stats.end();

      std::pair<std::string, struct stat_collector::stat_value_cell> p(
          name, value);

      // REFERESH COLUMNS, ADD MISSING.
      if (new_map)
        this->_stats.insert(p);
      else
        this->_stats[p.first] = p.second;

      std::map<std::string, size_t>::const_iterator itt =
          _field_indices.find(name);
      if (itt != _field_indices.end())
      {
        size_t idx = itt->second;
        vv[idx] = value;
      }
    }
    std::vector<std::string> str_values;
    for (auto inst : instances())
    {

      for (auto field : *inst)
      {
        // std::cout << std::string ("[+] snmp6_collector mapping field : ")
        //<< field->get_name () << std::endl;

        str_values.push_back(
            this->_stats[field->get_name()].to_string().empty() ? "0" : this->_stats[field->get_name()].to_string());
      }

      putValuesFromStrings(str_values.begin(), str_values.end());
      break;
    }

    // if(not vv.empty())
    //  putValuesFromStrings(vv.begin(), vv.end());

#ifdef DEBUG_MAPS
    print_map();
#endif
    f_snmp.close();
  }

  void
  sockstat_collector::collect()
  {
    std::ifstream f_sock("/proc/net/sockstat");
    std::string header, type, value;
    std::vector<std::string> tokens;
    while (not f_sock.eof())
    {
      std::string line;
      std::getline(f_sock, line);
      this->strip_key_value(line);
    }
    // print_map();
  }

  void
  route_statistics::collect()
  {

    std::ifstream f_rt("/proc/net/stat/rt_cache");
    std::vector<std::string> header_fields;
    std::string header, value;
    std::getline(f_rt, header);
    stat_collector::tokenize(header, header_fields, " ");
    // Iface	Destination	Gateway 	Flags	RefCnt	Use	Metric	Mask		MTU	Window	IRTT
    const std::string suffix = std::to_string(_entry);

    for (auto it = header_fields.begin(); it != header_fields.end(); it++)
    {
      std::string k = std::string("rtcache_") + *it + std::string("_") + suffix;

      if (_stats.find(k) == _stats.end())
        _stats.insert(std::make_pair(k, stat_value_cell("0")));
    }
    while (!f_rt.eof())

    {
      std::vector<std::string> values;

      std::getline(f_rt, value);
      stat_collector::tokenize(value, values, " ");
      size_t index = 0;
      for (auto it = values.begin(); it != values.end(); ++it)
      {
        if (index < header_fields.size())
        {
          std::string k = std::string("rtcache_") + header_fields[index++] + std::string("_") + suffix;
          _stats[k] = stat_value_cell(it->c_str());
        }
      }
      // unconditioned assign to fields:
      {

        std::vector<std::string> str_values;
        for (auto inst : instances())
        {
          for (auto field : *inst)
          {

            str_values.push_back(
                this->_stats[field->get_name()].to_string().empty() ? "0" : this->_stats[field->get_name()].to_string());
          }
          putValuesFromStrings(str_values.begin(), str_values.end());
          break;
        }
      }
    }
    f_rt.close();
  }

  void
  sockstat64_collector::collect()
  {

    std::ifstream f_sock("/proc/net/sockstat6");
    std::string header, type, value;
    std::vector<std::string> tokens;

    while (not f_sock.eof())
    {
      std::string line;

      std::getline(f_sock, line);

      this->strip_key_value(line);
    }
    std::vector<std::string> str_values;
    for (auto inst : instances())
    {
      for (auto field : *inst)
      {

        /*std::cout
      << std::string ("[+] sockstat64_collector mapping field : ")
      << field->get_name () << std::endl;*/

        str_values.push_back(
            this->_stats[field->get_name()].to_string().empty() ? "0" : this->_stats[field->get_name()].to_string());
      }
      break;
    }

    putValuesFromStrings(str_values.begin(), str_values.end());

#ifdef DEBUG_MAPS
    print_map();
#endif // DEBUG
    f_sock.close();
  }

  void
  diskstats::collect()
  {
    // all kernels
    //        ==  ===================================
    //	 1  major number
    //	 2  minor mumber
    //	 3  device name
    //	 4  reads completed successfully
    //	 5  reads merged
    //	 6  sectors read
    //	 7  time spent reading (ms)
    //	 8  writes completed
    //	 9  writes merged
    //	10  sectors written
    //	11  time spent writing (ms)
    //	12  I/Os currently in progress
    //	13  time spent doing I/Os (ms)
    //	14  weighted time spent doing I/Os (ms)
    //	==  ===================================
    //	==  ===================================
    //	15  discards completed successfully
    //	16  discards merged
    ///	17  sectors discarded
    //	18  time spent discarding
    //	==  ===================================

    //	Kernel 5.5+ appends two more fields for flush requests:

    //	==  =====================================
    //	19  flush requests completed successfully
    //	20  time spent flushing
    //	==  =====================================
    //  format in DB : device_name::headers - value.

    std::string headers[] =
        {"major", "minor", "device_name", "read_success", "read_merged",
         "sectors_read", "time_spent_reading", "writes_complete",
         "writes_merged", "sectors_written", "time_spent_writing",
         "io_in_progress", "time_doing_io", "weighted_time_doing_io",
         "discards_completed", "discards_merged", "sectors_discarded",
         "time_spent_discarding", "flush_completed", "time_spent_flushing",
         "unknown"};

    std::ifstream f_disk("/proc/diskstats");
    size_t row = 0;
    std::cerr << "DISK MONITORING IS DISABLED UNTIL v4.";
    return;

    while (!f_disk.eof())
    {
      std::string line;
      std::vector<std::string> tokens;
      std::getline(f_disk, line);
      if (line.length() > 1)
      {
        stat_collector::tokenize(line, tokens, " ");
#ifdef DEBUG_MAPS
        std::cout << std::string(" tokens : ") << tokens.size();
#endif
        // this->_type_exception_pair_map.insert(std::pair<size_t,std::pair<size_t,std::string>> (row,std::pair<size_t,std::string>( (uint16_t)TEXT,tokens[2])));

        for (size_t index = 0; index < tokens.size(); index++)
        {
          if (tokens[index].empty() || index == 2)
            continue;

          // std::cout<< std::string(" collected ")<<headers[index]  << std::string(" value =")<<(uint64_t)std::stoull(tokens[index],nullptr,10) ;
          //	if (this->_stats.find(tokens[2]+std::string("::")+headers[index])== _stats.end())
          //		this->_stats.insert(std::pair<std::string,uint64_t>(tokens[2]+std::string("::")+headers[index],(uint64_t)std::stoull(tokens[index],nullptr,10)));
          ///	else
          //		this->_stats[tokens[2]+std::string("::")+headers[index]] = (uint64_t)std::stoull(tokens[index],nullptr,10);
        }
      }
      row++;
    }
  }
  static unsigned long
  parse_hex(const char *p_str, unsigned char *p_inet)
  {
    unsigned long len = 0;
    char *p = NULL;

    if (NULL == (p = (char *)p_str))
      return 0;
    while (*p)
    {
      unsigned tmp;
      if (p[1] == 0){
        return 0;
      }
     if (sscanf(p, "%02x",&tmp) != 1)
        return 0;
      p_inet[len] = tmp;
      len++;
      p += 2;
    }
    return len;
  }

  void
  netdev_stats::collect()
  {
    std::ifstream is_dev("/proc/net/dev");
    std::vector<std::string> values;
    std::vector<std::string> keys;

    // default GW interface . could be multiple.
    std::string iface = getInterface();
    // Get first line
    std::string line;
    getline(is_dev, line);

    getline(is_dev, line);

    // normalize |
    for (size_t l = 0; l < line.length(); ++l)
    {
      // replace with whitespace.
      if (line[l] == '|')
        line[l] = ' ';
    }
    tokenize(line.substr(line.find_first_of(":") + 1), keys, " ");

    // Token
    // Now parse the dev file
    if (is_dev.is_open())
    {
      while (is_dev.good())
      {
        // Get line
        getline(is_dev, line);
        // Check interface
        if (line.find(iface) != std::string::npos)
        {
          // Parse lines
          tokenize(line.substr(line.find_first_of(":") + 1), values,
                   " ");
          break;
        }
      }
    }
    size_t prefix = 0;

    is_dev.close();

    keys.erase(keys.begin());

    for (auto key : keys)

    {
      std::string pre;
      if (prefix > (keys.size() / 2))
        pre = "tx_";
      else
        pre = "rx_";

      std::cout << std::string("[+]debug key ") << pre << key;
      std::string s = pre + key;

      if (_stats.find(s) == _stats.end())
      {

        if (prefix < values.size())
          _stats.insert(
              std::pair<std::string, stat_value_cell>(
                  s, stat_value_cell(values[prefix])));
        else
          _stats.insert(
              std::pair<std::string, stat_value_cell>(
                  s, stat_value_cell("0")));
      }
      else
      {
        if (prefix < values.size())
          _stats[s] = stat_value_cell(values[prefix]);
        else
          _stats[s] = stat_value_cell("0");
      }
      prefix++;
    }
    // if need fixing:
    if (values.size() != instances().size())
    {
      for (auto inst : instances())
      {
        for (auto field : *inst)
        {
          if (_stats.find(field->get_name()) == _stats.end())
          {
            std::cerr << std::string("[-] field missing : ")
                      << field->get_name() << std::endl;
            _stats.insert(
                std::make_pair(field->get_name(),
                               stat_value_cell("0")));
          }
        }
      }
    }
    // align values
    for (auto inst : instances())
    {
      std::map<std::string, bool> visit;
      std::vector<std::string> str_values;

      for (auto field : *inst)
      {

        if (visit.find(field->get_name()) == visit.end())
        {
          visit.insert(std::make_pair(field->get_name(), true));

          str_values.push_back(
              this->_stats[field->get_name()].to_string().empty() ? stat_value_cell(0).to_string() : this->_stats[field->get_name()].to_string());
        }
      }
      putValuesFromStrings(str_values.begin(), str_values.end());
      break;
    }

    /*
     if(str_values.size() > 0) {
     // Push values into the table
     putValuesFromStrings(str_values.begin(), str_values.end());
     } else {
     // Push default values
     pushDefaultValues();
     }
     */
  }
  void wireless::collect()
  {

    pushDefaultValues();
    return;
    std::ifstream wireless_file("/proc/net/wireless");

    // Parse first line
    std::string line(""), wireless_line("");
    std::string first_headers = "";
    // Parse wireless stuff (    wireless interfaces)
    getline(wireless_file, line); // firest header
    first_headers += line;

    getline(wireless_file, line);
    first_headers += line;

    getline(wireless_file, wireless_line);

    // If no wireless interface
    if (wireless_line.size() == 0)
    {
      pushDefaultValues();
      return;
    }

    // Token
    std::vector<std::string> values;
    tokenize(wireless_line.substr(wireless_line.find_first_of(":") + 1), values, " ");

    // Push data
    putValuesFromStrings(values.begin(), values.end());
    std::vector<std::string>::const_iterator it1 = values.begin();
    // fix map if needed.
    for (auto it = _stats.begin(); it != _stats.end() && it1 != values.end(); ++it, ++it1)
    {
      it->second = stat_value_cell(it1->c_str());
    }
  }
  void
  igmp_stats::collect()
  {
    std::ifstream f_igmp("/proc/net/igmp");
  }

  void
  igmp6_stats::collect()
  {
    std::ifstream f_igmp6("/proc/net/igmp6");
  }
  void
  ifnet6_stats::collect()
  {
    std::ifstream if_inet6("/proc/net/if_inet6");
    while (!if_inet6.eof())
    {

      std::string line;
      char ipv6Addr[INET6_ADDRSTRLEN];
      unsigned int ifIndex;
      unsigned int prefixLen;
      unsigned int addrScope;
      unsigned int ifFlags;
      uint32_t ip_in_host_byte_order[4];

      char ifName_and_the_rest_[INET6_ADDRSTRLEN]; // INET_ADDRSTRLEN would be enough since iface_name is 16bytes max anyway, but we include training spaces , tabs or other characters .

      std::getline(if_inet6, line);

      if (line.length() > 1)
      {
        std::sscanf(line.c_str(), "%s %02x %02x %02x %02x %s", ipv6Addr,
                    &ifIndex, &prefixLen, &addrScope, &ifFlags,
                    ifName_and_the_rest_);

        parse_hex(ipv6Addr, (uint8_t *)&ip_in_host_byte_order[0]);

        provallo::ip_address address(ip_in_host_byte_order);
        if (this->_stats.empty())
        {
        }
      }
      // ipv6Addr, &ifIndex, &prefixLen, &addrScope, &ifFlags, ifName
      // parse :
    }
  }
  void
  stat_collector::validate_entry(const std::string &name, size_t s,
                                 std::string *entry)
  {

    if (!entry)
      return;

    if (!entry->size())
    {
      (*entry) = name + std::string(":");
      for (size_t i = 0; i < s; ++i)
      {
        (*entry) += "0";
      }
    }
  }

  std::vector<std::string>
  stat_collector::parse_map_to_datatable()
  {

    // if param descriptor doesn't exists alter table.

    std::string param_descriptors;
    std::string value_descriptors_sample;
    std::string insert_param_descriptor;
    std::vector<std::string> ret;
    std::string del = "\"";
    for (auto i = _stats.begin(); i != _stats.end(); ++i)
    {
      if (!i->first.empty())
      {
        if (!_parsed)
        {

          param_descriptors += del + i->first + del + std::string(" BIGINT NOT NULL DEFAULT 0, ");
          insert_param_descriptor += del + i->first + del + std::string(",");
        }

        value_descriptors_sample += i->second.to_string().length() == 0 ? "0," : i->second.to_string() + std::string(",");
      }
    }
    if (!_parsed)
    {
      REMOVE_LAST_COMMA(insert_param_descriptor);
      REMOVE_LAST_COMMA(value_descriptors_sample);
      ret.push_back(param_descriptors);
      ret.push_back(value_descriptors_sample);
      ret.push_back(insert_param_descriptor);
      this->_queries = ret;
      _parsed = true;
    }
    else
    {
      // just update values...
      REMOVE_LAST_COMMA(value_descriptors_sample);

      _queries[1] = value_descriptors_sample;
    }
    return _queries;
  }

  std::string
  stat_collector::create_table_query()
  {
    std::string table_name = _name + "_TABLE";
    std::string create_query = "CREATE TABLE IF NOT EXISTS\"";
    auto parse_map = parse_map_to_datatable();
    create_query += table_name;
    create_query += "\" ( \"id\" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,";
    create_query += parse_map[0].c_str();
    create_query +=
        " sample_ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP); ";
    return create_query;
  }

  void
  stat_collector::strip_key_value(const std::string &line, bool key_headers)
  {

    std::vector<std::string> values;
    std::string header = line.substr(0, line.find_first_of(":"));

    stat_collector::tokenize(line.substr(line.find_first_of(":") + 1),
                             values, " ");
    std::string last_key;
    for (size_t i = 0; i < values.size(); i++)
    {
      if (i % 2 == 0)
        last_key = values[i];
      else if (key_headers)
        this->_stats.insert(
            std::pair<std::string, struct stat_collector::stat_value_cell>(
                header + last_key, values[i].c_str()));
      else
        this->_stats.insert(
            std::pair<std::string, struct stat_collector::stat_value_cell>(
                last_key, values[i].c_str()));
    }

    // if(!values.empty())
    // putValuesFromStrings(values.begin(), values.end());
  }

  std::string
  stat_collector::insert_query()
  {
    std::vector<std::string> queries = parse_map_to_datatable();
    std::string table_name = _name + "_TABLE";
    std::string insert_query = " INSERT INTO \"";
    insert_query += table_name;
    insert_query += "\" (";
    insert_query += queries[2];
    insert_query += " ) VALUES (";
    insert_query += queries[1];
    insert_query += ");";
    return insert_query;
  }
  std::string
  stat_collector::runtime_migrate_table(
      std::vector<std::string> new_columns) const
  {
    std::string table_name = _name + "_TABLE";
    std::string alter_query = "BEGIN  ;";
    for (auto col : new_columns)
    {
      std::string query = std::string("ALTER TABLE ") + table_name;
      query += " ADD COLUMN \"";
      query += col;
      query += "\" BIGINT NOT NULL DEFAULT(0);";
      alter_query += query;
    }
    alter_query += "COMMIT;END  ;";
    return alter_query;
  }

  void
  stat_collector::register_field(field_base *field, data_fieldOp *op)
  {
    for (std::vector<std::vector<field_base *> *>::const_iterator it =
             _instances.begin();
         it != _instances.end(); ++it)
      (*it)->push_back(field);
    _operations.push_back(op);
    // Set as active by default
    _active.push_back(true);
  }

  std::ostream &
  operator<<(std::ostream &out, const stat_collector &q)
  {
    for (size_t i = 0; i < q._instances.size(); ++i)
    {
      out << i << " : ";
      for (size_t j = 0; j < q._instances[i]->size(); ++j)
        out << (*q._instances[i])[j]->getValue() << " ";
      out << std::endl;
    }
    // Return stream
    return out;
  }

  const std::string &
  stat_collector::get_normalized() const
  {
    static std::string normalized;
    normalized.clear();
    if (_instances.size() > 1 && _instances[1] != nullptr && _operations.size() > 0 && _instances[0]->size() > 0)
    {

      for (size_t i = 0; i < _instances[0]->size() && i < _operations.size() && i < _instances[1]->size(); ++i)

      {
        fields_instance f1 = *_instances[1], f0 = *_instances[0];
        data_fieldOp *operation = _operations[i];
        std::cout << "[operation] " << i << std::string(":") << std::hex << operation << std::dec << std::endl;
        f1[i]->print();
        std::cout << "[operation] " << i << std::endl;
        f0[i]->print();

        const std::string &add = std::string(operation->operator()(*(f0[i]), (*(f1[i]))));
        //: f1[i]->getValue());

        if (add.length())
          normalized = normalized + add + std::string(",");
        else
          normalized += std::string("0.0,");
      }
    }
    std::cout << "returning normalized " << normalized << std::endl;
    // Return normalized string
    return normalized;
  }

  std::string
  stat_collector::get_normalized_filter_ignored() const
  {
    static std::string normalized;
    normalized = "";
    for (size_t i = 0; i < _instances[0]->size(); ++i)
      if (i < _active.size() && _active[i])
        normalized += (*_operations[i])(*(*_instances[1])[i],
                                        *(*_instances[0])[i]) +
                      ",";
      else
        normalized += "0,";
    // Return normalized string
    return normalized;
  }

  std::string
  stat_collector::getLastSample() const
  {
    size_t last_instance = _instances.size() - 1;
    static std::string normalized("");
    normalized.clear();
    for (size_t i = 0; i < _instances[last_instance]->size(); ++i)
      normalized += (*_instances[last_instance])[i]->getValue() + ",";
    return normalized;
  }

  real_t
  stat_collector::getTimeStamp() const
  {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    // Get time stamp in seconds
    return (real_t)ts.tv_sec + ((real_t)ts.tv_nsec * 1E-9);
  }

  void
  stat_collector::pushDefaultValues()
  {
    std::vector<std::string> values;
    for (size_t i = 0; i < size(); ++i)
      values.push_back("0");
    for (auto stat : _stats)
    {
      stat.second = stat_value_cell(0);
    }
    putValuesFromStrings(values.begin(), values.end());
  }

  std::vector<field_base *> *
  stat_collector::getNewDataPtr()
  {
    std::vector<field_base *> *ret = nullptr;
    // Check if this is the first instance pushed
    if (_instances.size() == 2)
    {
      // First swap container (overlap the last one)
      std::vector<field_base *> *tmp = _instances[0];
      _instances[0] = _instances[1];
      // Update
      _instances[1] = tmp;

      // Reference to the instance container
      ret = _instances[1];
    }
    else if (_instances.size() == 1)
    {
      // prepare new sample
      _instances.push_back(new fields_instance);
      for (size_t i = 0; i < _instances[0]->size(); ++i)
      {
        _instances[1]->push_back((*_instances[0])[i]->clone());
        (*_instances[1])[i]->setTimeStamp(getTimeStamp());
      }
      // return the first sample
      ret = _instances[0];
    }

    return ret;
  }

  void
  stat_collector::printLastSampleValues(std::ostream &out) const
  {
    // Last instance
    size_t last_instance = _instances.size() - 1;
    // Serialize each field on the sample buffer
    for (size_t i = 0; i < size(); ++i)
      out << (*_instances[last_instance])[i]->get_name() << std::string(" ")
          << (*_instances[last_instance])[i]->getValue() << std::endl;
  }

  std::vector<field_base *>::const_iterator
  stat_collector::getField(const std::string &name) const
  {
    // Loop over the fields
    for (auto it = this->begin(); it != this->end(); ++it)
      // Check name
      if (name == (*it)->get_name())
        return it;
    return end();
  }

  std::vector<field_base *>::iterator
  stat_collector::getField(const std::string &name)
  {
    // Loop over the fields
    for (auto it = this->begin(); it != this->end(); ++it)
      // Check name
      if (name == (*it)->get_name())
        return it;
    return end();
  }

  empty_collector::empty_collector(size_t num) : stat_collector("empty_collector")
  {
    for (size_t i = 0; i < num; i++)
      register_field(new data_field<std::string>("dummy", "continuous"),
                     new Dummy());
  }
  void
  ndis_cache::collect()
  {

    std::ifstream ndis_cache_file("/proc/net/stat/ndisc_cache");
    static const size_t linemax = 512;
    // Parses values container
    std::vector<unsigned int> values(size());
    const std::string suffix = std::to_string(_entry);

    const std::string keys[] =
        {
            std::string("ndisc_entries_") + suffix,
            std::string("ndisc_allocs_") + suffix,
            std::string("ndisc_destroys_") + suffix,
            std::string("ndisc_hash_grows_") + suffix,
            std::string("ndisc_lookups_") + suffix,
            std::string("ndisc_hits_") + suffix,
            std::string("ndisc_res_failed_") + suffix,
            std::string("ndisc_rcv_probes_mcast_") + suffix,
            std::string("ndisc_rcv_probes_ucast_") + suffix,
            std::string("ndisc_periodic_gc_runs_") + suffix,
            std::string("ndisc_forced_gc_runs_") + suffix,
            std::string("ndisc_unresolved_discards_") + suffix,
            std::string("ndisc_table_fulls_") + suffix};
    // Parse data
    if (ndis_cache_file.is_open())
    {
      // Ignore first line
      ndis_cache_file.ignore(linemax, '\n');

      // Just return if EOF Is found
      if (ndis_cache_file.eof())
      {
        return;
      }

      // Parse data
      if (ndis_cache_file.is_open())
      {
        // Parse from the line defined
        while (!ndis_cache_file.eof())
        {
          // Parse each column
          for (size_t i = 0; i < size(); ++i)
          {

            size_t entry_suffix = i / sizeof(keys);
            std::string key = keys[(i % sizeof(keys))] + std::to_string(entry_suffix);

            std::string str_value;
            ndis_cache_file >> str_value;
            if (ndis_cache_file.eof())
              break;
            unsigned int value;
            sscanf(str_value.c_str(), "%x", &value);
            values[i] += value;

            std::pair<std::string,
                      struct stat_collector::stat_value_cell>
                p(key, value);

            if (_stats.find(key) != _stats.end())
              _stats[key] = stat_value_cell(value);
            else
              _stats.insert(p);
          }
        }
      }
    }

    putValues(values.begin(), values.end());
    _collected = true;

    ndis_cache_file.close();
  }
  auto init_net_stats_indices = []() -> std::map<std::string, size_t>
  {
    static std::map<std::string, size_t> m;
    m["SyncookiesSent"] = 0;
    m["SyncookiesRecv"] = 1;
    m["SyncookiesFailed"] = 2;
    m["EmbryonicRsts"] = 3;
    m["PruneCalled"] = 4;
    m["RcvPruned"] = 5;
    m["OfoPruned"] = 6;
    m["OutOfWindowIcmps"] = 7;
    m["LockDroppedIcmps"] = 8;
    m["ArpFilter"] = 9;
    m["TW"] = 10;
    m["TWRecycled"] = 11;
    m["TWKilled"] = 12;
    m["PAWSPassive"] = 13;
    m["PAWSActive"] = 14;
    m["PAWSEstab"] = 15;
    m["DelayedACKs"] = 16;
    m["DelayedACKLocked"] = 17;
    m["DelayedACKLost"] = 18;
    m["ListenOverflows"] = 19;
    m["ListenDrops"] = 20;
    m["TCPPrequeued"] = 21;
    m["TCPDirectCopyFromBacklog"] = 22;
    m["TCPDirectCopyFromPrequeue"] = 23;
    m["TCPPrequeueDropped"] = 24;
    m["TCPHPHits"] = 25;
    m["TCPHPHitsToUser"] = 26;
    m["TCPPureAcks"] = 27;
    m["TCPHPAcks"] = 28;
    m["TCPRenoRecovery"] = 29;
    m["TCPSackRecovery"] = 30;
    m["TCPSACKReneging"] = 31;
    m["TCPFACKReorder"] = 32;
    m["TCPSACKReorder"] = 33;
    m["TCPRenoReorder"] = 34;
    m["TCPTSReorder"] = 35;
    m["TCPFullUndo"] = 36;
    m["TCPPartialUndo"] = 37;
    m["TCPDSACKUndo"] = 38;
    m["TCPLossUndo"] = 39;
    m["TCPLostRetransmit"] = 40;
    m["TCPRenoFailures"] = 41;
    m["TCPSackFailures"] = 42;
    m["TCPLossFailures"] = 43;
    m["TCPFastRetrans"] = 44;
    m["TCPForwardRetrans"] = 45;
    m["TCPSlowStartRetrans"] = 46;
    m["TCPTimeouts"] = 47;
    m["TCPRenoRecoveryFail"] = 48;
    m["TCPSackRecoveryFail"] = 49;
    m["TCPSchedulerFailed"] = 50;
    m["TCPRcvCollapsed"] = 51;
    m["TCPDSACKOldSent"] = 52;
    m["TCPDSACKOfoSent"] = 53;
    m["TCPDSACKRecv"] = 54;
    m["TCPDSACKOfoRecv"] = 55;
    m["TCPAbortOnData"] = 56;
    m["TCPAbortOnClose"] = 57;
    m["TCPAbortOnMemory"] = 58;
    m["TCPAbortOnTimeout"] = 59;
    m["TCPAbortOnLinger"] = 60;
    m["TCPAbortFailed"] = 61;
    m["TCPMemoryPressures"] = 62;
    m["TCPSACKDiscard"] = 63;
    m["TCPDSACKIgnoredOld"] = 64;
    m["TCPDSACKIgnoredNoUndo"] = 65;
    m["TCPSpuriousRTOs"] = 66;
    m["TCPMD5NotFound"] = 67;
    m["TCPMD5Unexpected"] = 68;
    m["TCPSackShifted"] = 69;
    m["TCPSackMerged"] = 70;
    m["TCPSackShiftFallback"] = 71;
    m["TCPBacklogDrop"] = 72;
    m["TCPMinTTLDrop"] = 73;
    m["TCPDeferAcceptDrop"] = 74;
    m["IPReversePathFilter"] = 75;
    m["TCPTimeWaitOverflow"] = 76;
    m["TCPReqQFullDoCookies"] = 77;
    m["TCPReqQFullDrop"] = 78;
    m["TCPRetransFail"] = 79;
    m["TCPRcvCoalesce"] = 80;
    m["TCPOFOQueue"] = 81;
    m["TCPOFODrop"] = 82;
    m["TCPOFOMerge"] = 83;
    m["TCPChallengeACK"] = 84;
    m["TCPSYNChallenge"] = 85;
    m["TCPFastOpenActive"] = 86;
    m["TCPFastOpenPassive"] = 87;
    m["TCPFastOpenPassiveFail"] = 88;
    m["TCPFastOpenListenOverflow"] = 89;
    m["TCPFastOpenCookieReqd"] = 90;
    m["TCPLoss"] = 91;
    m["TCPAbortOnSyn"] = 92;
    return m;
  };

  std::map<std::string, size_t> net_stats::_field_indices =
      init_net_stats_indices();

  void
  net_stats::collect()
  {
    std::ifstream is_netstat("/proc/net/netstat");
    std::vector<std::string> values;
    std::vector<std::string> keys;

    // Parse first line
    std::string keys_line, line, tcpext_line, ipext_line;

    getline(is_netstat, keys_line);
    tokenize(keys_line.substr(keys_line.find_first_of(":") + 1), keys, " ");

    // Get TCPEXT line
    getline(is_netstat, tcpext_line);
    getline(is_netstat, line);
    // Get IPEXT line
    getline(is_netstat, ipext_line);

    // Close stream
    is_netstat.close();

    // Container of values (initialize to zeroes)
    // First push IPEXT values
    size_t value_order = 0;
    tokenize(ipext_line.substr(ipext_line.find_first_of(":") + 1), values,
             " ");
    for (auto it = keys.begin(); it != keys.end(); it++, value_order++)
    {

      _stats[*it] =
          (value_order < values.size()) ? stat_value_cell(values[value_order]) : stat_value_cell("0");
    }

    tokenize(line.substr(line.find_first_of(":") + 1), keys, " ");

    // Then , get TCPEXT values
    std::vector<std::string> tcp_values;
    tokenize(tcpext_line.substr(tcpext_line.find_first_of(":") + 1),
             tcp_values, " ");
    // Sanity check
    value_order = 0;
    for (auto it = keys.begin(); it != keys.end(); it++, value_order++)

    {

      _stats[*it] =
          value_order < tcp_values.size() ? stat_value_cell(tcp_values[value_order]) : stat_value_cell("0");
    }

    if (_tcpext_indices.size() != tcp_values.size())
    {
      std::cerr
          << std::string("[-] fix wrong tcpext collection. mapped indices[ ")
          << _tcpext_indices.size() << std::string("] values size : ")
          << std::to_string(tcp_values.size()) << std::endl;
    }

    // Push the TCPEXT default values
    values.resize(size(), "0");

    // Get names and modify values accordingly
    for (size_t i = 0; i < _tcpext_indices.size() && i < tcp_values.size();
         ++i)
    {
      size_t idx = _tcpext_indices[i];
      if (idx != size())
        values[idx] = tcp_values[i];
    }

    // Push values to the table
    putValuesFromStrings(values.begin(), values.end());
    _collected = true;
  }

  ndis_cache::~ndis_cache()
  {
  }
  net_stats::~net_stats()
  {
  }
  void
  arp_stats::collect()
  {
    std::ifstream is_route("/proc/net/route");
    std::string line;

    // Default gateway for each device on the route file
    std::map<std::string, std::string> gw_addr_map;
    // Now parse the route file

    if (is_route.is_open())
    {
      while (is_route.good())
      {
        // Get line
        getline(is_route, line);
        // Parse information of this interface
        char a[16];
        unsigned int b, c;
        int r = sscanf(line.c_str(), "%s %x %x", a, &b, &c);
        // Check if we have a valid gateway
        if (r == 3 && b == 0)
        {
          struct in_addr in;
          in.s_addr = c;
          gw_addr_map.insert(
              std::make_pair(std::string(a),
                             std::string(inet_ntoa(in))));
        }
      }
    }

    // If not default gateway
    if (gw_addr_map.size() == 0)
    {
      pushDefaultValues();
      return;
    }

    // Parse ARP table
    std::ifstream is_arp("/proc/net/arp");
    std::getline(is_arp, line);

    // Create map of interfaces and addresses
    typedef std::map<std::string, std::string> addr_pair_cont;
    std::map<std::string, addr_pair_cont> iface_addr_map;
    std::string ip_addr_arp, hw_addr_arp, device, dummy;

    // Interfaces container
    std::set<std::string> interfaces;
    // MAC addresses in the ARP table
    std::vector<std::string> hw_addrs;

    // Parse the ARP table
    if (is_arp.is_open())
    {
      while (is_arp.good())
      {
        is_arp >> ip_addr_arp;
        is_arp >> dummy;
        is_arp >> dummy;
        is_arp >> hw_addr_arp;
        is_arp >> dummy;
        is_arp >> device;
        // Break if we reach the end of the file
        if (is_arp.eof())
          break;
        // Push interface name
        interfaces.insert(device);
        // Push MAC address
        hw_addrs.push_back(hw_addr_arp);
        // Map addresses with interfaces
        iface_addr_map[device].insert(
            make_pair(ip_addr_arp, hw_addr_arp));
      }
    }

    // If ARP table is empty
    if (iface_addr_map.size() == 0)
    {
      pushDefaultValues();
      return;
    }

    // ---- Put values on the table
    std::vector<std::string> values;
    // Select an interface (for now just get the first one on the set)
    // TODO: What should be done if there is another one?
    std::string iface = *interfaces.begin();
    values.push_back(iface);

    // Get gateway
    std::string gateway = gw_addr_map[iface];

    // Check for collisions (if MAC addresses are not unique on the table, we have a problem)
    std::vector<std::string>::iterator it = std::unique(hw_addrs.begin(),
                                                        hw_addrs.end());
    if (distance(hw_addrs.begin(), it) != distance(hw_addrs.begin(), hw_addrs.end()))
      values.push_back("1");
    else
      values.push_back("0");

    // Push gateway HW address
    const addr_pair_cont &iface_map = iface_addr_map[iface];
    addr_pair_cont::const_iterator it_gw = iface_map.find(gateway);
    if (it_gw != iface_map.end())
      values.push_back((*it_gw).second);
    else
      values.push_back("0");

    if (values.size() > 2)
    {
      _stats["Interface"] = stat_value_cell(values[0]);
      _stats["CollisionDetected"] = stat_value_cell(values[1]);
      _stats["GwChanged"] = stat_value_cell(values[2]);
    }

    putValuesFromStrings(values.begin(), values.end());
    _collected = true;

    // Close streams
    is_route.close();
    is_arp.close();
  }
  bool
  arp_stats::is_intercepted()
  {
    bool iface_changed = attribute_definition::fromString<bool>(getValue(0));
    bool collision = attribute_definition::fromString<bool>(getValue(1));
    bool gw_changed = attribute_definition::fromString<bool>(getValue(2));

    if (collision)
      return true;
    if (gw_changed && not iface_changed)
      return true;
    return false;
  }
  arp_stats::~arp_stats()
  {
  }

  void
  arp_cache_stats::collect()
  {

    std::ifstream arp_cache_file("/proc/net/stat/arp_cache");
	//    constexpr const size_t maxline = 512;
    // Parses values container
    //size_t line_number = 0;
    std::vector<unsigned int> values(size());
    std::string line;
    std::vector<std::string> keys, field_keys;
    // Parse data
    if (arp_cache_file.is_open())
    {

      getline(arp_cache_file, line);
      stat_collector::tokenize(
          line, keys, " ");

      // Just return if EOF Is found
      if (arp_cache_file.eof())
      {
        return;
      }
      // parse keys :
      for (auto key : keys)
      {
        auto prefix = "arpcache_";
        auto suffix = std::to_string(line_collect);
        auto fieldkey = prefix + key + std::string("_") + suffix;

        _stats.insert(std::make_pair(fieldkey, stat_value_cell("0")));
        field_keys.push_back(fieldkey);
      }
      // Parse data
      size_t val_it = 0;
      if (arp_cache_file.is_open())
      {

        // Parse from the line defined
        while (!arp_cache_file.eof())
        {
          // Parse each column
          for (size_t i = 0; i < size(); ++i)
          {
            std::string str_value;
            arp_cache_file >> str_value;
            if (arp_cache_file.eof())
              break;
            unsigned int value;
            sscanf(str_value.c_str(), "%x", &value);
            values[i] += value;
            _stats[field_keys[val_it % field_keys.size()]].u_value._unsigned = value;
          }
        }
      }
    }

    // Put the parsed data into the table
    putValues(values.begin(), values.end());
    _collected = true;

    // Close file
    arp_cache_file.close();
  }
  arp_cache_stats::~arp_cache_stats()
  {
  }
  proc_collector::proc_collector()
  {
  }

  void
  proc_collector::collect() const
  {
    //  collect data for each table
    for (std::vector<stat_collector *>::const_iterator it = _tables.begin();
         it != _tables.end(); ++it)
      (*it)->collect();
  }

  const std::string &
  proc_collector::get_normalized() const
  {
    // Get   normalized string for each table
    static std::string normalized("");

    for (std::vector<stat_collector *>::const_iterator it = _tables.begin();
         it != _tables.end(); ++it)
    {

      std::cout << "[+] adding normalized string from " << (*it)->get_name() << std::endl;

      std::string normal = (*it)->get_normalized().c_str();
      if (normal.length() > 0)
        normalized += normal;
    }
    return normalized;
  }

  /*
  template <typename T>
  std::vector<T> proc_collector::get_real()const
  {
    static std::vector<T> ret;
    ret.clear();
    for (std::vector<stat_collector*>::const_iterator it = _tables.begin ();
  it != _tables.end (); ++it)
  for (auto jit = (*it)->begin();jit!= (*it)->end();++jit )
    {

        ret.push_back((*jit)->getReal());

    }
    return ret;
  }
  */
  std::string
  proc_collector::get_normalized_filter_ignored() const
  {

    std::string normalized("");
    for (std::vector<stat_collector *>::const_iterator it = _tables.begin();
         it != _tables.end(); ++it)
      normalized += (*it)->get_normalized_filter_ignored();
    return normalized;
  }

  std::string
  proc_collector::get_last_sample() const
  {
    // Get and return normalized string of each table
    std::string normalized("");
    for (std::vector<stat_collector *>::const_iterator it = _tables.begin();
         it != _tables.end(); ++it)
      normalized += (*it)->getLastSample();
    return normalized;
  }

  void
  proc_collector::get_attribute_information(
      std::vector<std::pair<std::string, std::string>> *attributes) const
  {
    // must be friend to access _instances
    for (std::vector<stat_collector *>::const_iterator it = _tables.begin();
         it != _tables.end(); ++it)
    {
      // Last instance
      size_t last_instance = (*it)->instances().size() - 1;
      // Serialize each field on the sample buffer
      for (size_t i = 0; i < (*it)->size(); ++i)
      {
        std::string name =
            (*(*it)->instances()[last_instance])[i]->get_name();
        std::string type =
            (*(*it)->instances()[last_instance])[i]->get_type();

        attributes->push_back(std::make_pair(name, type));
      }
    }
  }

  stat_collector *
  proc_collector::get_table(size_t idx) const
  {
    assert(idx < _tables.size());
    return _tables[idx];
  }

  proc_collector::~proc_collector()
  {
    for (std::vector<stat_collector *>::const_iterator it = _tables.begin();
         it != _tables.end(); ++it)
      delete (*it);
  }

  const std::string &
  getInterface()
  {
    // Parse route file
    std::ifstream rt_file("/proc/net/route");
    std::string line;
    static std::string ret;
    // Now parse the route file
    ret = "lo";
    if (rt_file.is_open())
    {
      while (rt_file.good())
      {
        // Get line
        getline(rt_file, line);
        // Parse information of this interface
        char a[16];
        unsigned int b, c;
        if (line.size())
        {
          int r = sscanf(line.c_str(), "%s %x %x", a, &b, &c);
          // Check if we have a valid gateway
          if (r == 3 && b == 0)
          {
            // std::cout << std::string ("[+] returning  interface ")
            //<< std::string (a) << std::endl;
            ret = std::string(a);
          }
        }
      }
    }
    rt_file.close();
    // No default gateway (return lo)
    return ret;
  }

} /* namespace provallo */
