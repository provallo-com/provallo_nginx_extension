
#ifndef _NGINX_HELPER_H_
#define _NGINX_HELPER_H_

#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>



#ifdef __cplusplus
extern "C" {
#endif

 
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