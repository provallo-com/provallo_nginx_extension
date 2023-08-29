#ifndef _WEBSTATS_COLLECTOR_H_
#define _WEBSTATS_COLLECTOR_H_

//webstats collector normalizes requests,response,headers and mime types and sends them to the detector
//it also collects the statistics for the detector
  
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <thread>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <functional>
//stat collector
#include "statcollector.h"
#include "util/safehttp.h"

//webstats collector
namespace provallo {

class web_collector : public stat_collector 
{
    struct context {
    int64_t _id; //context id
    bool _is_request; //is request or response
    std::string _method; //method
    std::string _uri; //uri
    std::string _version; //version
    std::string _status; //status
    std::string _reason; //reason
    std::string _host; //host
    std::string _user_agent; //user agent
    std::string _referer; //referer
    std::string _content_type; //content type
    std::string _content_length; //content length
    std::string _connection; //connection
    std::string _accept_encoding; //accept encoding
    std::string _accept_language; //accept language
    std::string _accept; //accept
    std::string _cookie; //cookie
    std::string _set_cookie; //set cookie
    std::string _location; //location
    std::string _content_encoding; //content encoding
    std::string _transfer_encoding; //transfer encoding
    std::string _cache_control; //cache control
    std::string _pragma; //pragma
    std::string _expires; //expires
    std::string _last_modified; //last modified
    std::string _etag; //etag
    std::string _vary; //vary
    std::string _x_powered_by; //x powered by
    std::string _x_ua_compatible; //x ua compatible
    std::string _x_content_type_options; //x content type options
    std::string _x_frame_options; //x frame options
    std::string _x_xss_protection; //x xss protection
    std::string _server; //server
    std::string _date; //date
    std::map<std::string,std::string> _connection_stats ; //connection stats:
    
    std::map<std::string,std::string> _http_stats ; //http stats:include http2 stats and http3 stats and http3 stats.

     
    std::map<std::string,std::string> _headers;
    std::map<std::string,std::string> _upstreams;
    std::map<std::string,std::string> _requests;
    std::map<std::string,std::string> _responses;
    std::map<std::string,std::string> _mime_types;
    std::map<std::string,std::string> _mime_types_stats; 
    };
 
    public:
    web_collector();//register on the map headers ,upstreams,requests,responses,mime types
    
    virtual ~web_collector() = default;
    web_collector(const web_collector&) = delete;
    web_collector& operator=(const web_collector&) = delete;
    web_collector(web_collector&&) = delete;
    web_collector& operator=(web_collector&&) = delete;
    virtual void collect() override;
    protected:
};
 
} // namespace provallo





#endif 
