/*
 * safehttp.cpp
 *
 *  Created on: Jun 10, 2023
 *      Author: kardon
 */

#include "safehttp.h"
#include <iostream>
namespace provallo
{

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

  safe_http_parser::status_t safe_http_parser::add_bytes(const char* bytes, unsigned len)
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

  constexpr const std::string user_agent_srting = "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36)";

  query_builder::query_builder(const std::string& url_path, const std::string& host_name) : _url_path(url_path),_user_agent(user_agent_srting) {}

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
