#ifndef __PCAP_HELPER_H_ 
#define __PCAP_HELPER_H_ 
#include <string.h>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <pcap.h>

namespace provallo
{

  class pcap_packet
  {
    pcap_pkthdr _header;
    u_char *_packet;
  public:
    pcap_packet (const pcap_pkthdr *header, const u_char *packet) :
	_packet (0)
    {
      _packet = new u_char[header->caplen];
      ::memcpy (_packet, packet, header->caplen * sizeof(u_char));
      _header = *header;
    }
    pcap_packet (const pcap_packet &pck) :
	_packet (0)
    {
      _packet = new u_char[pck._header.caplen];
      memcpy (_packet, pck._packet, pck._header.caplen * sizeof(u_char));
      _header = pck._header;
    }
    void
    dump (pcap_dumper_t *dumper) const
    {
      u_char *udumper = reinterpret_cast<u_char*> (dumper);
      pcap_dump (udumper, &_header, _packet);
    }
    ~pcap_packet ()
    {
      delete[] _packet;
    }
  };

  class pcap_collector
  {
    // History of packets
    std::vector<std::vector<pcap_packet> > _packets;
    // Handle for the opened PCAP session
    pcap_t *_handle;
    // Flag to tell the sniffer to keep capturing packets
    bool _capture;
    // History row counter
    size_t _row_counter;
    size_t _history;

    // Thread ID
    pthread_t _threadid;

  public:
    // Initialize with number of samples to accumulate in the history
    pcap_collector (size_t history);

    pcap_t*
    get_handle () const
    {
      return _handle;
    }

    void
    start ();

    void
    stop ();

    void
    dump (const std::string &filename) const;

    void
    push_packet (const pcap_packet &packet)
    {
      _packets[_row_counter % _history].push_back (packet);
    }

    bool
    capture () const
    {
      return _capture;
    }

    void
    clean ();

    const size_t
    size ()
    {

      size_t size (0);
      for (auto packet : _packets)
	      size += packet.size ();
      return size;
    }

    ~pcap_collector ();
  };

}

#endif 
