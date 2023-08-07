#include    <iostream>
#include    <fstream>
#include    <sstream>
#include    <string>
#include    <vector>
#include    <map>
#include    <algorithm>
#include    <iterator>
#include    <math.h>
#include    <string.h>
#include    "nginx_helper.h"
//#include    "../util/csv_file.h"
//#include    "../util/safehttp.h"
#include    "../util/interface_info.h"
#include    "../decision_engine/decision_engine.h"
 
//#include    "../decision_engine/decision_engine.h"
//#include    "../decision_engine/kdt.h"
//#include    "../decision_engine/classifier.h"
//#include    "../decision_engine/dataset.h"
//#include    "../decision_engine/classdist.h"

#ifdef __cplusplus
extern "C" {
#endif
 uint64_t provallo_requests_filtered =0 ;
 uint64_t provallo_malformed_sessions =0;
 uint64_t provallo_stability_issues_avoided =0 ;
 uint64_t provallo_classifiers_running =0;

void start_monitoring_requests()
{
    //use nginx to start thread
    //nginx_thread_start(); 
   
}
void stop_monitoring_requests()
{
    //use nginx to stop thread
    //nginx_thread_stop(); 
}

//create/destroy thread for monitoring statistics (cpu, memory, etc)
void start_metric_collection()
{
    //use nginx to start thread
    //nginx_thread_start(); 
   
}
void stop_metric_collection()
{
    //use nginx to stop thread
    //nginx_thread_stop(); 
}

//create/destroy thread for classification
void start_classification()
{
    //use nginx to start thread
    //nginx_thread_start(); 
   
}
void stop_classification()
{
    //use nginx to stop thread
    //nginx_thread_stop(); 
}


//finally initialize the engine  
void initialize_engine()
{
    //use nginx to start thread
    //nginx_thread_start(); 
    ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "initializing engine");
    //initialize decision engine    
    provallo::pipeline_builder* builder = new pipeline_builder();
    builder->add_stage(new classifier_stage());
    builder->add_stage(new classdist_stage());
    builder->add_stage(new kdt_stage());
    builder->add_stage(new dataset_stage());
    builder->add_stage(new decision_engine_stage());
    builder->add_stage(new decision_engine_stage());
    builder->add_stage(new decision_engine_stage());
    builder->add_stage(new decision_engine_stage());
    builder->add_stage(new decision_engine_stage());
    builder->add_stage(new decision_engine_stage());
    builder->add_stage(new decision_engine_stage());
    builder->add_stage(new decision_engine_stage());
    builder->add_stage(new decision_engine_stage());
    builder->add_stage(new decision_engine_stage());

    builder->build();
    provallo_classifiers_running = builder->get_num_classifiers();
    delete builder;
    ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "engine initialized");

}

void finalize_engine();


#ifdef __cplusplus
}   /* extern "C" */
#endif

/*
** Parse a JSON request
*/
void
ngx_http_provallo_json_parse(ngx_http_request_ctx_t* ctx,
                          ngx_http_request_t*     r,
                          u_char*                 src,
                          u_int                   len)
{

    ngx_http_provallo_filter_t* filter = ctx->filter; 
    ngx_flag_t result = 0;
    std::string json((char*)src, len);
     result = filter_request(filter, json);
        if(result == 1)
        {
            provallo_requests_filtered++;
            ngx_http_finalize_request(r, NGX_HTTP_FORBIDDEN);
        }
        else if(result == -1)
        {
            provallo_malformed_sessions++;
            ngx_http_finalize_request(r, NGX_HTTP_BAD_REQUEST);
        }
        else
        {
            ngx_http_finalize_request(r, NGX_OK);
        } 
 }

 
void ngx_http_provallo_body_filter(ngx_http_request_t *r, ngx_chain_t *in)
{
    ngx_http_request_ctx_t* ctx = (ngx_http_request_ctx_t*)ngx_http_get_module_ctx(r, ngx_http_provallo_module);
    ngx_http_provallo_filter_t* filter = ctx->filter; 
    ngx_flag_t result = 0;
    std::string body;
    ngx_chain_t* cl;
    for (cl = in; cl; cl = cl->next) {
        ngx_buf_t* buf = cl->buf;
        if (buf->in_file) {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "body in file");
            return;
        }
        if (buf->flush || buf->last_buf || buf->last_in_chain) {
            body.append((char*)buf->pos, buf->last - buf->pos);
        }
    }
    result = filter_request(filter, body);
    if(result == 1)
    {
        provallo_requests_filtered++;
        ngx_http_finalize_request(r, NGX_HTTP_FORBIDDEN);
    }
    else if(result == -1)
    {
        provallo_malformed_sessions++;
        ngx_http_finalize_request(r, NGX_HTTP_BAD_REQUEST);
    }
    else
    {
        ngx_http_finalize_request(r, NGX_OK);
    } 
}
//proxy/upstream filter
void ngx_http_provallo_upstream_filter(ngx_http_request_t *r, ngx_chain_t *in)
{
    ngx_http_request_ctx_t* ctx = (ngx_http_request_ctx_t*)ngx_http_get_module_ctx(r, ngx_http_provallo_module);
    ngx_http_provallo_filter_t* filter = ctx->filter; 
    ngx_flag_t result = 0;
    std::string body;
    ngx_chain_t* cl;
    for (cl = in; cl; cl = cl->next) {
        ngx_buf_t* buf = cl->buf;
        if (buf->in_file) {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "body in file");
            return;
        }
        if (buf->flush || buf->last_buf || buf->last_in_chain) {
            body.append((char*)buf->pos, buf->last - buf->pos);
        }
    }
    result = filter_request(filter, body);
    if(result == 1)
    {
        provallo_requests_filtered++;
        ngx_http_finalize_request(r, NGX_HTTP_FORBIDDEN);
    }
    else if(result == -1)
    {
        provallo_malformed_sessions++;
        ngx_http_finalize_request(r, NGX_HTTP_BAD_REQUEST);
    }
    else
    {
        ngx_http_finalize_request(r, NGX_OK);
    } 
}

