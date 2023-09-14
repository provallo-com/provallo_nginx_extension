
#ifndef _NGINX_HELPER_H_
#define _NGINX_HELPER_H_

#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>



#ifdef __cplusplus
extern "C" {
#endif

#define PROVALLO_MAX_ARGS  32
#define PROVALLO_MAX_HEADERS  32
// 
 
typedef struct
{
  /* match in full body (POST DATA) */
  ngx_flag_t body : 1;
  /* match in [name] var of body */
  ngx_flag_t body_var : 1;
  /* match in all headers */
  ngx_flag_t headers : 1;
  /* match in [name] var of headers */
  ngx_flag_t headers_var : 1;
  /* match in URI */
  ngx_flag_t url : 1;
  /* match in args (bla.php?<ARGS>) */
  ngx_flag_t args : 1;
  /* match in [name] var of args */
  ngx_flag_t args_var : 1;
  /* match on a global flag : weird_request, big_body etc. */
  ngx_flag_t flags : 1;
  /* match on file upload extension */
  ngx_flag_t file_ext : 1;
  /* set if defined "custom" match zone (GET_VAR/POST_VAR/...)  */
  ngx_array_t* ids;
  ngx_str_t*   name;
} ngx_http_whitelist_location_t;

typedef struct
{
  ngx_str_t name;
  ngx_str_t value;
} ngx_http_whitelist_arg_t;

typedef struct
{
  ngx_str_t name;
  ngx_str_t value;
} ngx_http_whitelist_header_t;

typedef enum {
  HEADERS = 0,
  URL,
  ARGS,
  BODY,
  RAW_BODY,
  FILE_EXT,
  STREAM,
  UNKNOWN
}content_filter_type_t;

typedef struct {
typedef struct  {
      ngx_flag_t ingress_filter: 1;
      ngx_flag_t ougress_filter: 1;
        ngx_flag_t headers_filter : 1;
        ngx_flag_t url_filter : 1;
        ngx_flag_t args_filter : 1;
        ngx_flag_t body_filter : 1;
        ngx_flag_t raw_body_filter : 1;
        ngx_flag_t file_ext_filter : 1;
        ngx_flag_t stream_filter : 1;
        ngx_flag_t unknown_filter : 1;
    }ngx_http_provallo_filter_type_t filter;
 
    ngx_str_t name;
    ngx_array_t* ids; 
    ngx_array_t* headers;
    ngx_array_t* args;
    ngx_array_t* body;
    ngx_array_t* raw_body;
    ngx_array_t* file_ext;
    ngx_array_t* stream;
    ngx_array_t* unknown;
 }ngx_http_provallo_filter_t;


//use adapter to fill nginx structures from our policies and rules
 
//create/destroy thread for monitoring requests

void start_monitoring_requests();
void stop_monitoring_requests();

//create/destroy thread for monitoring statistics (cpu, memory, etc)
void start_metric_collection();
void stop_metric_collection();

//create/destroy thread for classification
void start_classification();
void stop_classification();


//finally initialize the engine  
void initialize_engine();

void finalize_engine();
  
#ifdef __cplusplus
}   
#endif


#endif //_NGINX_HELPER_H_