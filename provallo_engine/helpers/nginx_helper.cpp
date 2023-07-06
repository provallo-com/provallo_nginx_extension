#include    <nginx.h>
#include    <ngx_config.h>
#include    <ngx_core.h>
#include    <ngx_http.h>
#include    <ngx_md5.h>



#include    <string.h>




#include    "nginx_helper.h"
//#include    "../util/csv_file.h"
//#include    "../util/safehttp.h"
#include    "../util/interface_info.h"

//#include    "../decision_engine/decision_engine.h"
//#include    "../decision_engine/kdt.h"
//#include    "../decision_engine/classifier.h"
//#include    "../decision_engine/dataset.h"
//#include    "../decision_engine/classdist.h"


#include    <iostream>
#include    <fstream>
#include    <sstream>
#include    <string>
#include    <vector>
#include    <map>
#include    <algorithm>
#include    <iterator>
#include    <math.h>
#ifdef __cplusplus
extern "C" {
void  *ngx_http_provallo_create_loc_conf(ngx_conf_t *cf);
char  *ngx_http_provallo_merge_loc_conf(ngx_conf_t *cf, void *parent, void *child);
char  *ngx_http_provallo_set_conf(ngx_conf_t *cf, ngx_command_t *cmd, void *conf);
ngx_int_t ngx_http_provallo_handler(ngx_http_request_t *r);
ngx_int_t ngx_http_provallo_init(ngx_conf_t *cf);
ngx_int_t ngx_http_provallo_init_process(ngx_cycle_t *cycle);
void ngx_http_provallo_exit_process(ngx_cycle_t *cycle);
void ngx_http_provallo_exit_master(ngx_cycle_t *cycle);
void ngx_http_provallo_exit(ngx_conf_t *cf);
ngx_int_t ngx_http_provallo_init_module(ngx_cycle_t *cycle);
void ngx_http_provallo_exit_module(ngx_cycle_t *cycle);
ngx_int_t ngx_http_provallo_init_worker(ngx_cycle_t *cycle);
void ngx_http_provallo_exit_worker(ngx_cycle_t *cycle);

}   /* extern "C" */
#endif




 
