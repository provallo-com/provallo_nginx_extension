/*
 * safehttp.cpp
 *
 *  Created on: Jun 10, 2023
 *      Author: kardon
 */

#include "safehttp.h"
#include <map>
#include <iostream>

namespace provallo
{
  const std::map<size_t, std::string> http_status_map =
  {
  { 100, "Continue" },
  { 101, "Switching Protocols" },
  { 102, "Processing" },
  { 200, "OK" },
  { HTTP_STATUS_CREATED, "Created" },
  { HTTP_STATUS_ACCEPTED, "Accepted" },
  { HTTP_STATUS_NON_AUTHORITATIVE, "Non-Authoritative Information" },
  { HTTP_STATUS_NO_CONTENT, "No Content" },
  { HTTP_STATUS_RESET_CONTENT, "Reset Content" },
  { HTTP_STATUS_PARTIAL_CONTENTS, "Partial Content" },
  { HTTP_STATUS_MULTI_WEBDAV, "Multi-Status" },
  { HTTP_STATUS_WD_REPORTED, "Already Reported" },
  { HTTP_STATUS_IM_USED, "IM Used" },
  { HTTP_STATUS_MULTIPLE_CHOICES, "Multiple Choices" },
  { HTTP_STATUS_MOVED_PERMANENTLY, "Moved Permanently" },
  { HTTP_STATUS_MOVED_TEMPORARILY, "Found" },
  { HTTP_STATUS_SEE_OTHER, "See Other" },
  { HTTP_STATUS_NOT_MODIFIED, "Not Modified" },
  { HTTP_STATUS_USE_PROXY, "Use Proxy" },
  { HTTP_STATUS_SWITCH_PROXY, "Switch Proxy" },
  { HTTP_STATUS_TEMPORARY_REDIRECT, "Temporary Redirect" },
  {HTTP_STATUS_PERMANENT_REDIRECT, "Permanent Redirect" },
  { HTTP_STATUS_BAD_REQUEST, "Bad Request" },
  { HTTP_STATUS_UNAUTHORIZED, "Unauthorized" },
  { HTTP_STATUS_PAYMENT_REQUIRED, "Payment Required" },
  { HTTP_STATUS_FORBIDDEN, "Forbidden" },
  { HTTP_STATUS_NOT_FOUND, "Not Found" },
  { HTTP_STATUS_NOT_ALLOWED, "Method Not Allowed" },
  { HTTP_STATUS_NOT_ACCEPTABLE, "Not Acceptable" },
  { HTTP_STATUS_PROXY_AUTH_REQ, "Proxy Authentication Required" },
  { HTTP_STATUS_REQ_TIMEOUT, "Request Timeout" },
  { HTTP_STATUS_CONFLICT, "Conflict" },
  { HTTP_STATUS_GONE, "Gone" },
  { HTTP_STATUS_LENGTH_REQUIRED, "Length Required" },
  { HTTP_STATUS_PRECONDITION_FAILED, "Precondition Failed" },
  { HTTP_STATUS_PAYLOAD_TOO_BIG, "Payload Too Large" },
  { HTTP_STATUS_URI_TOO_BIG, "URI Too Long" },
  { HTTP_STATUS_UNSUPPORTED_MEDIA, "Unsupported Media Type" },
  { HTTP_STATUS_RANGE_UNSATISFIABLE, "Range Not Satisfiable" },
  { HTTP_STATUS_EXP_FAILED, "Expectation Failed" },
  { HTTP_STATUS_IM_A_TEAPOT, "I'm a teapot" },
  { HTTP_STATUS_MISDIRECTED_REQ, "Misdirected Request" },
  { HTTP_STATUS_UNPROCESSED_ENT, "Unprocessable Entity" },
  { HTTP_STATUS_WD_LOCK, "Locked" },
  { HTTP_STATUS_FAILED_DEP, "Failed Dependency" },
  { HTTP_STATUS_UPGRADE_REQUIRED, "Upgrade Required" },
  { HTTP_STATUS_PRECON_REQUIRED, "Precondition Required" }, 
    { HTTP_STATUS_TOO_MANY_REQ, "Too Many Requests" },
    { HTTP_STATUS_HEADER_FIELD_TOO_BIG, "Request Header Fields Too Large" },
    { HTTP_STATUS_LEGAL_BLOCK, "Unavailable For Legal Reasons" },
    { HTTP_STATUS_INTERNAL, "Internal Server Error" },
    { HTTP_STATUS_NOT_IMPLEMENTED, "Not Implemented" },
    { HTTP_STATUS_BAD_GATEWAY, "Bad Gateway" },
    { HTTP_STATUS_SERVICE_UNAVAIL, "Service Unavailable" },
    { HTTP_STATUS_GW_TIMEOUT, "Gateway Timeout" },
    { HTTP_STATUS_VERSION_UNSUP, "HTTP Version Not Supported" },
    { HTTP_VARIANT_NEGOTIATION, "Variant Also Negotiates" },
    { HTTP_INSUFFICIENT_STORAGE, "Insufficient Storage" },
    { HTTP_LOOP_DETECTED, "Loop Detected" },
    { HTTP_STATUS_NOT_EXTENDED, "Not Extended" },
    { HTTP_STATUS_NETWORK_UNAVAILABLE, "Network Authentication Required" } ,
    
    //add cloudflare status codes
    { HTTP_STATUS_CLOUDFLARE_UNKNOWN_ERROR, "Web server is returning an unknown error" },
    { HTTP_STATUS_CLOUDFLARE_WEB_SERVER_IS_DOWN, "Web server is down" },
    { HTTP_STATUS_CLOUDFLARE_CONNECTION_TIMED_OUT, "Connection timed out" },
    { HTTP_STATUS_CLOUDFLARE_ORIGIN_IS_UNREACHABLE, "Origin is unreachable" },
    { HTTP_STATUS_CLOUDFLARE_A_TIMEOUT_OCCURRED, "A timeout occurred" },
    { HTTP_STATUS_CLOUDFLARE_SSL_HANDSHAKE_FAILED, "SSL handshake failed" },
    { HTTP_STATUS_CLOUDFLARE_INVALID_SSL_CERTIFICATE, "Invalid SSL certificate" }, 
    { HTTP_STATUS_CLOUDFLARE_RAILGUN_ERROR, "Railgun error" },
    { HTTP_STATUS_CLOUDFLARE_ORIGIN_DNS_ERROR, "Origin DNS error" }, 
    { HTTP_STATUS_CLOUDFLARE_UNKNOWN_HOST, "Unknown host" },
    { HTTP_STATUS_CLOUDFLARE_PAGE_EXPIRED, "Page Expired" },
    { HTTP_STATUS_CLOUDFLARE_HOST_NOT_FOUND, "Host Not Found" },
    { HTTP_STATUS_CLOUDFLARE_SSL_CERTIFICATE_ERROR, "SSL Certificate Error" },
    { HTTP_STATUS_CLOUDFLARE_BAD_GATEWAY, "Bad Gateway" },
    { HTTP_STATUS_CLOUDFLARE_INVALID_HOSTNAME, "Invalid Hostname" },
    { HTTP_STATUS_CLOUDFLARE_UPSTREAM_SERVER_ERROR, "Upstream Server Error" },
    { HTTP_STATUS_CLOUDFLARE_GATEWAY_TIMEOUT, "Gateway Timeout" },

    //end of cloudflare status codes
    //add custom status codes
    { HTTP_STATUS_CUSTOM_ERROR, "Custom Error" },
    { HTTP_STATUS_INVALID_REQUEST, "Invalid Request" },
    { HTTP_STATUS_INVALID_RESPONSE, "Invalid Response" },
    { HTTP_STATUS_INVALID_HEADER, "Invalid Header" },
    { HTTP_STATUS_INVALID_CONTENT, "Invalid Content" }


    
    
     };


  safe_http_parser::safe_http_parser() :
          _headerStart(0), _bodyStart(0), _parsedTo(0), _state(0), _keyIndex(0), _valueIndex(0), _contentLength(0),
          _contentStart(0), _uriIndex(0), _status(Incomplete)
  {

  }

  safe_http_parser::~safe_http_parser()
  {

  }

  void safe_http_parser::parse_headers()
  {
      // run the fsm.
      const int CR = 13;
      const int LF = 10;
      const int ANY = 256;
      enum Action
      {
          // make lower case
          LOWER = 0x1,

          // convert current character to null.
          NULLIFY = 0x2,

          // set the header index to the current position
          SET_HEADER_START = 0x4,

          // set the key index to the current position
          SET_KEY = 0x8,

          // set value index to the current position.
          SET_VALUE = 0x10,

          // store current key/value pair.
          STORE_KEY_VALUE = 0x20,

          // sets content start to current position + 1
          SET_CONTENT_START = 0x40
      };

      static const struct FSM
      {
              parse_state curState;
              int c;
              parse_state nextState;
              unsigned actions;

      } fsm[] =
      {
      { parse_request_line, CR, parse_request_line_cr, NULLIFY },
      { parse_request_line, ANY, parse_request_line, 0 },
      { parse_request_line_cr, LF, parse_request_line_crlf, 0 },
      { parse_request_line_crlf, CR, parse_request_line_crlfcr, 0 },
      { parse_request_line_crlf, ANY, parse_key_state, SET_HEADER_START | SET_KEY | LOWER },
      { parse_request_line_crlfcr, LF, parse_content, SET_CONTENT_START },
      { parse_key_state, ':', parse_key_colon, NULLIFY },
      { parse_key_state, ANY, parse_key_state, LOWER },
      { parse_key_colon, ' ', parse_key_colon_seperator, 0 },
      { parse_key_colon_seperator, ANY, parse_value, SET_VALUE },
      { parse_value, CR, parse_value_cr, NULLIFY | STORE_KEY_VALUE },
      { parse_value, ANY, parse_value, 0 },
      { parse_value_cr, LF, parse_value_crlf, 0 },
      { parse_value_crlf, CR, parse_value_crlfcr, 0 },
      { parse_value_crlf, ANY, parse_key_state, SET_KEY | LOWER },
      { parse_value_crlfcr, LF, parse_content, SET_CONTENT_START },
      { p_error, ANY, p_error, 0 } };

      for (unsigned i = _parsedTo; i < _data.length(); ++i)
      {
          char c = _data[i];
          parse_state nextState = p_error;

          for (unsigned d = 0; d < sizeof(fsm) / sizeof(FSM); ++d)
          {
              if (fsm[d].curState == _state && (c == fsm[d].c || fsm[d].c == ANY))
              {

                  nextState = fsm[d].nextState;

                  if (fsm[d].actions & LOWER)
                  {
                      _data[i] = tolower(_data[i]);
                  }

                  if (fsm[d].actions & NULLIFY)
                  {
                      _data[i] = 0;
                  }

                  if (fsm[d].actions & SET_HEADER_START)
                  {
                      _headerStart = i;
                  }

                  if (fsm[d].actions & SET_KEY)
                  {
                      _keyIndex = i;
                  }

                  if (fsm[d].actions & SET_VALUE)
                  {
                      _valueIndex = i;
                  }

                  if (fsm[d].actions & SET_CONTENT_START)
                  {
                      _contentStart = i + 1;
                  }

                  if (fsm[d].actions & STORE_KEY_VALUE)
                  {
                      // store position of first character of key.
                      _keys.push_back(_keyIndex);
                  }

                  break;
              }
          }
          _state = nextState;
          if (_state == p_error)
          {
              std::cerr<< "fsm_http_parser : error on index " << i;
          }
          if (_state == parse_content)
          {
              const char* str = get_value("content-length");

              if (str)
              {
                  _contentLength = std::atoi(str);
                  std::cerr<< "fsm_http_parser : found content length " << _contentLength;
              } else {
        	  std::cerr << "fsm_http_parser no content length on header" << _contentLength;
              }
              break;
          }
      }
      _parsedTo = _data.length();
  }

  bool safe_http_parser::parse_requestline()
  {
      size_t sp1;
      size_t sp2;
      sp1 = _data.find(' ', 0);
      if (sp1 == std::string::npos)
          return false;
      sp2 = _data.find(' ', sp1 + 1);
      if (sp2 == std::string::npos)
          return false;

      _data[sp1] = 0;
      _data[sp2] = 0;
      _uriIndex = sp1 + 1;
      return true;
  }

  safe_http_parser::http_parser_status_t safe_http_parser::add_bytes(const char* bytes, unsigned len)
  {
      if (_status != Incomplete)
      {
          return _status;
      }

      // append the bytes to data.
      _data.append(bytes, len);

      if (_state < parse_content)
      {
          parse_headers();
      }

      if (_state == p_error)
      {
          _status = Error;
      }
      else if (_state == parse_content)
      {
          if (_contentLength == 0 || _data.length() - _contentStart >= _contentLength)
          {
              if (parse_requestline())
              {
                  _status = Done;
              }
              else
              {
                  _status = Error;
              }
          }
      }

      return _status;
  }

  const char*
  safe_http_parser::get_method()
  {
      return &_data[0];
  }

  const char*
  safe_http_parser::get_uri()
  {
      return &_data[_uriIndex];
  }

  const char*
  safe_http_parser::get_query_string()
  {
      const char* pos = get_uri();
      while (*pos)
      {
          if (*pos == '?')
          {
              pos++;
              break;
          }
          pos++;
      }
      return pos;
  }

  const char*
  safe_http_parser::get_body()
  {
      if (/*_contentLength > 0*/_contentStart < _data.length() && _contentStart < _contentLength)
      {
           return &_data.c_str()[_contentStart]; ////////;
      }
      else
      {
          return NULL;
      }
  }
  size_t safe_http_parser::get_body_index() const
  {
      return _contentStart;
  }
  // key should be in lower case.
  const char*
  safe_http_parser::get_value(const char* key)
  {
      for (std::vector<unsigned>::iterator iter = _keys.begin(); iter != _keys.end(); ++iter)
      {
          unsigned index = *iter;
          if (strncmp(&_data[index], key, std::max(_data.length() - index, strlen(key))) == 0)
          {
              return &_data[index + strlen(key) + 2]; //maybe tokenize?
          }
      } 

      return NULL;
  }

  unsigned safe_http_parser::get_content_length()
  {
      return _contentLength;
  }
 
  // Helper function to print HTTP header
  static void printHttpHeader(std::ostream& out, const char* header, size_t size) {
      for(size_t i = 0 ; i < size ; i++) {
          if ((unsigned int)header[i] == 0x09)
              out << "\\t";
          else if ((unsigned int)header[i] == 0x0d)
              out << "\\r";
          else if ((unsigned int)header[i] == 0x0a)
              out << "\\n" ;
          else if(isprint(header[i])) {
              out << header[i];
          } else {
              out << "\\x";
              out << std::hex << (unsigned int)header[i];
          }
      }
  }

  static void printHttpHeader(std::ostream& out, std::string& header) {
      printHttpHeader(out, header.c_str(), header.size());
  }

  const std::string user_agent_srting = "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36)";

  query_builder::query_builder(const std::string& url_path, const std::string& host_name) : _url_path(url_path),_host_name(  host_name) ,_user_agent(user_agent_srting){}

  query_builder::query_builder(const std::string& url_path, const std::string& host_name, const std::string& auth_token) :
      _url_path(url_path), _host_name(host_name), _auth_token(auth_token),_user_agent(user_agent_srting) {}

  std::string query_builder::build_query(const std::string& content_type, const char* body, const size_t size) const {
      // Create HTTP header
      std::string query = "POST ";
      query += _url_path;
      query += "  HTTP/1.0\r\nAccept: */* \r\nUser-Agent:";
      query += _user_agent;
      query += "\r\nHost:";
      query += _host_name;
      query += "\r\nContent-Type: " + content_type + "\r\n";

      // Put authentication token on the header
      if(_auth_token.length() > 0) {
          query += "auth-token: " + _auth_token + "\r\n";
      }

      // Body of the header
      if(body) {
          // Get the raw data as a string
          std::string payload(body, size);
          // Create body of the HTTP header
          query += "Content-Length: " + std::to_string(size) + "\r\n\r\n";
          query += payload;
      }

      // Print query
      printHttpHeader(std::cout, query);

      // Return query
      return query;
  }


} /* namespace provallo */
