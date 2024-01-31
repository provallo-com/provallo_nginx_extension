/*
 * safehttp.h
 *
 *  Created on: Jun 10, 2023
 *      Author: kardon
 */

#ifndef UTIL_SAFEHTTP_H_
#define UTIL_SAFEHTTP_H_

//HTTP status headers :

#define HTTP_STATUS_CREATED             201
#define HTTP_STATUS_ACCEPTED            202
#define HTTP_STATUS_NON_AUTHORITATIVE   203 
#define HTTP_STATUS_NO_CONTENT          204
#define HTTP_STATUS_RESET_CONTENT       205 
#define HTTP_STATUS_PARTIAL_CONTENTS    206
#define HTTP_STATUS_MULTI_WEBDAV        207
#define HTTP_STATUS_WD_REPORTED         208
#define HTTP_STATUS_IM_USED 		226
#define HTTP_STATUS_MULTIPLE_CHOICES    300
#define HTTP_STATUS_MOVED_PERMANENTLY   301
#define HTTP_STATUS_MOVED_TEMPORARILY   302
#define HTTP_STATUS_SEE_OTHER           303
#define HTTP_STATUS_NOT_MODIFIED        304
#define HTTP_STATUS_USE_PROXY           305
#define HTTP_STATUS_SWITCH_PROXY        306
#define HTTP_STATUS_TEMPORARY_REDIRECT  307
#define HTTP_STATUS_PERMANENT_REDIRECT  308 
#define HTTP_STATUS_BAD_REQUEST         400
#define HTTP_STATUS_UNAUTHORIZED        401
#define HTTP_STATUS_PAYMENT_REQUIRED	402
#define HTTP_STATUS_FORBIDDEN           403
#define HTTP_STATUS_NOT_FOUND           404
#define HTTP_STATUS_NOT_ALLOWED		405
#define HTTP_STATUS_NOT_ACCEPTABLE	406
#define HTTP_STATUS_PROXY_AUTH_REQ	407
#define HTTP_STATUS_REQ_TIMEOUT		408
#define HTTP_STATUS_CONFLICT		409
#define HTTP_STATUS_GONE	 	410
#define HTTP_STATUS_LENGTH_REQUIRED	411
#define HTTP_STATUS_PRECONDITION_FAILED 412
#define HTTP_STATUS_PAYLOAD_TOO_BIG 	413
#define HTTP_STATUS_URI_TOO_BIG 	414
#define HTTP_STATUS_UNSUPPORTED_MEDIA 	415
#define HTTP_STATUS_RANGE_UNSATISFIABLE	416
 #define HTTP_STATUS_EXP_FAILED		417
#define HTTP_STATUS_IM_A_TEAPOT		418


#define HTTP_STATUS_MISDIRECTED_REQ 	421
#define HTTP_STATUS_UNPROCESSED_ENT 	422
#define HTTP_STATUS_WD_LOCK	 	423
#define HTTP_STATUS_FAILED_DEP	 	424
#define HTTP_STATUS_TOO_EARLY	 	425
#define HTTP_STATUS_UPGRADE_REQUIRED    426
#define HTTP_STATUS_PRECON_REQUIRED     428
#define HTTP_STATUS_TOO_MANY_REQ	429
#define HTTP_STATUS_HEADER_FIELD_TOO_BIG 431
#define HTTP_STATUS_LEGAL_BLOCK		 451

#define HTTP_STATUS_INTERNAL            500
#define HTTP_STATUS_NOT_IMPLEMENTED     501
#define HTTP_STATUS_BAD_GATEWAY         502
#define HTTP_STATUS_SERVICE_UNAVAIL	503
#define HTTP_STATUS_GW_TIMEOUT       	504
#define HTTP_STATUS_VERSION_UNSUP       505
#define HTTP_VARIANT_NEGOTIATION        506
#define HTTP_INSUFFICIENT_STORAGE       507
#define HTTP_LOOP_DETECTED	        508
#define HTTP_STATUS_UNKNOWN_FAIL        509
#define HTTP_STATUS_NOT_EXTENDED        510
#define HTTP_STATUS_NETWORK_UNAVAILABLE 511
//cloudflare errors : 
#define HTTP_STATUS_CLOUDFLARE_UNKNOWN_ERROR 520
#define HTTP_STATUS_CLOUDFLARE_WEB_SERVER_IS_DOWN 521
#define HTTP_STATUS_CLOUDFLARE_CONNECTION_TIMED_OUT 522
#define HTTP_STATUS_CLOUDFLARE_ORIGIN_IS_UNREACHABLE 523
#define HTTP_STATUS_CLOUDFLARE_A_TIMEOUT_OCCURRED 524
#define HTTP_STATUS_CLOUDFLARE_SSL_HANDSHAKE_FAILED 525
#define HTTP_STATUS_CLOUDFLARE_INVALID_SSL_CERTIFICATE 526
#define HTTP_STATUS_CLOUDFLARE_RAILGUN_ERROR 527
#define HTTP_STATUS_CLOUDFLARE_ORIGIN_DNS_ERROR 530
#define HTTP_STATUS_CLOUDFLARE_ERROR_528 528
#define HTTP_STATUS_CLOUDFLARE_ERROR_529 529
#define HTTP_STATUS_CLOUDFLARE_UNKNOWN_HOST 530
#define HTTP_STATUS_CLOUDFLARE_PAGE_EXPIRED 531
#define HTTP_STATUS_CLOUDFLARE_HOST_NOT_FOUND 532 
#define HTTP_STATUS_CLOUDFLARE_SSL_CERTIFICATE_ERROR 533
#define HTTP_STATUS_CLOUDFLARE_SSL_CERTIFICATE_REQUIRED 534
#define HTTP_STATUS_CLOUDFLARE_BAD_GATEWAY 535 
#define HTTP_STATUS_CLOUDFLARE_INVALID_HOSTNAME  536
#define HTTP_STATUS_CLOUDFLARE_REQUEST_TIMED_OUT 537
#define HTTP_STATUS_CLOUDFLARE_UPSTREAM_SERVER_ERROR 538
#define HTTP_STATUS_CLOUDFLARE_GATEWAY_TIMEOUT 540
#define HTTP_STATUS_CLOUDFLARE_NETWORK_READ_TIMEOUT_ERROR 541 
#define HTTP_STATUS_CLOUDFLARE_NETWORK_CONNECT_TIMEOUT_ERROR 542 
#define HTTP_STATUS_CLOUDFLARE_NETWORK_WRITE_TIMEOUT_ERROR 543
#define HTTP_STATUS_CLOUDFLARE_NETWORK_READ_ERROR 544
#define HTTP_STATUS_CLOUDFLARE_NETWORK_CONNECT_ERROR 545
#define HTTP_STATUS_CLOUDFLARE_NETWORK_WRITE_ERROR 546
#define HTTP_STATUS_CLOUDFLARE_ORIGIN_DNS_TIMEOUT_ERROR 547
#define HTTP_STATUS_CLOUDFLARE_ORIGIN_CONNECTION_TIMEOUT_ERROR 548
#define HTTP_STATUS_CLOUDFLARE_ORIGIN_CONNECTION_REFUSED_ERROR 549
#define HTTP_STATUS_CLOUDFLARE_ORIGIN_CONNECTION_CLOSED_ERROR 550
#define HTTP_STATUS_CLOUDFLARE_ORIGIN_SERVER_DOWN_ERROR 551
#define HTTP_STATUS_CLOUDFLARE_ORIGIN_SERVER_UNREACHABLE_ERROR 552
#define HTTP_STATUS_CLOUDFLARE_ORIGIN_SERVER_TIMEOUT_ERROR 553
#define HTTP_STATUS_CLOUDFLARE_ORIGIN_SERVER_UPSTREAM_ERROR 554

//end of HTTP status headers
//custom http errors :
//HTTP_STATUS_CUSTOM_ERROR
#define HTTP_STATUS_CUSTOM_ERROR        10001
#define HTTP_STATUS_INVALID_REQUEST     1000
#define HTTP_STATUS_INVALID_RESPONSE    1001
#define HTTP_STATUS_INVALID_HEADER      1002
#define HTTP_STATUS_INVALID_CONTENT     1003
#define HTTP_STATUS_INVALID_METHOD      1004
//end of custom http errors

#include <stdlib.h>
#include <vector>
#include <string>
#include <string.h>


namespace provallo
{

  class safe_http_parser
  {
      public:
          //TODO:
           safe_http_parser();
          ~safe_http_parser();

          enum http_parser_status_t
          {
              Done, Error, Incomplete
          };

          http_parser_status_t add_bytes(const char* bytes, unsigned len);

          const char* get_method();
          const char* get_uri();
          const char* get_query_string();
          const char* get_body();
          // key should be in lower case when looking up.
          const char* get_value(const char* key);
          unsigned get_content_length();
          size_t get_body_index() const;

      private:
          void parse_headers();
          bool parse_requestline();
        
          std::string _data;
          unsigned _headerStart;
          unsigned _bodyStart;
          unsigned _parsedTo;
          int _state;
          unsigned _keyIndex;
          unsigned _valueIndex;
          unsigned _contentLength;
          unsigned _contentStart;
          unsigned _uriIndex;

          std::vector<unsigned> _keys;

          enum parse_state
          {
              parse_request_line = 0,
              parse_request_line_cr = 1,
              parse_request_line_crlf = 2,
              parse_request_line_crlfcr = 3,
              parse_key_state = 4,
              parse_key_colon = 5,
              parse_key_colon_seperator = 6,
              parse_value = 7,
              parse_value_cr = 8,
              parse_value_crlf = 9,
              parse_value_crlfcr = 10,
              parse_content = 11,
              p_error = 12
          };
          
          //for http query parsing [LB Testing] 


          http_parser_status_t _status;
  };
  //for http query building [LB Testing]
  //------------------------------------

  class query_builder {
      // Path for queries
      std::string _url_path;
      // Host name (without the port and protocol)
      std::string _host_name;
      // Authentication token (extra header if this member is set)
      std::string _auth_token;

      //user agent
      std::string _user_agent;

  public:

      // Constructor without authentication token
      query_builder(const std::string& url_path, const std::string& host_name);
      // Put authentication token
      query_builder(const std::string& url_path, const std::string& host_name, const std::string& auth_token);

      // Build query an return data (get raw data as an argument for the body)
      std::string build_query(const std::string& content_type, const char* body, const size_t size) const;

      const std::string& user_agent()const {return _user_agent;}
      void  set_agent(const std::string& agent) {_user_agent=agent;}


      virtual ~query_builder()=default;
  };
} /* namespace provallo */

#endif /* UTIL_SAFEHTTP_H_ */
