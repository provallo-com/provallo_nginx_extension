/**
 * @file   ngx_provallo_interface_module.c
 * @author Yaniv Karta
 * @date       2023-07-01
 *
 * @brief   Provallo's Nginx. wrapper
 */

#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>
#define PROVALLO_HELLO "ENGINE UP\r\n"

#include "provallo_engine/helpers/nginx_helper.h"

static ngx_int_t ngx_http_provallo_nginx_handler(ngx_http_request_t *r);

//ngx_int_t ngx_http_provallo_nginx_handler(ngx_http_request_t *r);

typedef struct {

    ngx_int_t                  status;

    ngx_http_complex_value_t  *text;

} ngx_http_provallo_nginx_loc_conf_t;


typedef struct {
    ngx_array_t  upstrands;
} ngx_http_upstreams_main_conf_t;


typedef struct {
    ngx_array_t  dyn_upstrands;
    ngx_uint_t   upstrand_gw_modules_checked;
} ngx_http_upstreams_loc_conf_t;


static ngx_int_t ngx_http_provallo_nginx_handler(ngx_http_request_t *r);

static void *ngx_http_provallo_nginx_create_loc_conf(ngx_conf_t *cf);

static char *ngx_http_provallo_nginx_merge_loc_conf(ngx_conf_t *cf, void *parent,

    void *child);

static char *ngx_http_provallo_nginx(ngx_conf_t *cf, ngx_command_t *cmd,

    void *conf);



static ngx_command_t  ngx_http_provallo_nginx_commands[] = {


    { ngx_string("provallo_nginx"),

      NGX_HTTP_LOC_CONF|NGX_CONF_NOARGS,

      ngx_http_provallo_nginx,

      0,

      0,

      NULL },


    { ngx_string("provallo_nginx_status"),

      NGX_HTTP_MAIN_CONF|NGX_HTTP_SRV_CONF|NGX_HTTP_LOC_CONF|NGX_CONF_TAKE1,

      ngx_conf_set_num_slot,

      NGX_HTTP_LOC_CONF_OFFSET,

      offsetof(ngx_http_provallo_nginx_loc_conf_t, status),

      NULL },


    { ngx_string("provallo_nginx_text"),

      NGX_HTTP_MAIN_CONF|NGX_HTTP_SRV_CONF|NGX_HTTP_LOC_CONF|NGX_CONF_TAKE1,

      ngx_http_set_complex_value_slot,

      NGX_HTTP_LOC_CONF_OFFSET,

      offsetof(ngx_http_provallo_nginx_loc_conf_t, text),

      NULL },


      ngx_null_command

};



static ngx_http_module_t  ngx_http_provallo_nginx_module_ctx = {

    NULL,                                  /* preconfiguration */

    NULL,                                  /* postconfiguration */


    NULL,                                  /* create main configuration */

    NULL,                                  /* init main configuration */


    NULL,                                  /* create server configuration */

    NULL,                                  /* merge server configuration */


    ngx_http_provallo_nginx_create_loc_conf,  /* create location configuration */

    ngx_http_provallo_nginx_merge_loc_conf    /* merge location configuration */

};



ngx_module_t  ngx_http_provallo_nginx_module = {

    NGX_MODULE_V1,

    &ngx_http_provallo_nginx_module_ctx,      /* module context */

    ngx_http_provallo_nginx_commands,         /* module directives */

    NGX_HTTP_MODULE,                       /* module type */

    NULL,                                  /* init master */

    NULL,                                  /* init module */

    NULL,                                  /* init process */

    NULL,                                  /* init thread */

    NULL,                                  /* exit thread */

    NULL,                                  /* exit process */

    NULL,                                  /* exit master */

    NGX_MODULE_V1_PADDING

} ;


ngx_module_t provallo_nginx_module = {
	    NGX_MODULE_V1,

    &ngx_http_provallo_nginx_module_ctx,      /* module context */

    ngx_http_provallo_nginx_commands,         /* module directives */

    NGX_HTTP_MODULE,                       /* module type */

    NULL,                                  /* init master */

    NULL,                                  /* init module */

    NULL,                                  /* init process */

    NULL,                                  /* init thread */

    NULL,                                  /* exit thread */

    NULL,                                  /* exit process */

    NULL,                                  /* exit master */

    NGX_MODULE_V1_PADDING

};


//counters for the status page
extern uint64_t provallo_requests_filtered;
extern uint64_t provallo_malformed_sessions;
extern uint64_t provallo_stability_issues_avoided;
extern uint64_t provallo_classifiers_running;



//nginx metrics to show :
//
//
//
//upstream metrics: 
//

static const char * response_template = "Provallo Engine status\r\n\r\n#requests filtered: %llu \r\n#malformed sessions : %llu \r\n#stability issues avoided :%llu \r\n#classifiers running %llu \r\n%s\r\n"; 


static const char* noise_response_distribution = "2xx 0.77\r\n3xx 0.12\r\n4xx 0.05\r\n5xx 0.1\r\nTraffic Distribution: Service 0.23% non-service 0.73%  OS internal 0.04%";


static ngx_int_t

ngx_http_provallo_nginx_handler(ngx_http_request_t *r)

{

    ngx_buf_t                        *b;

    ngx_int_t                         rc;

    ngx_str_t                         text;

    ngx_chain_t                       out;

    static char str_buff [2048]={};
    
        

//    ngx_http_provallo_nginx_loc_conf_t  *hlcf;


    ngx_log_debug0(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,

                   "http provallo_nginx handler");


    /* ignore client request body if any */

    /*
    if (ngx_http_discard_request_body(r) != NGX_OK) {

        return NGX_HTTP_INTERNAL_SERVER_ERROR;

    }
    */

    //memset(str_buff,0,2048*sizeof(char));
    ngx_memzero(str_buff,sizeof(char)*2048);
    provallo_classifiers_running =1; 
    ngx_log_debug(NGX_LOG_DEBUG_HTTP,r->connection->log,0, "getting configuration " );

    ssize_t print = snprintf(str_buff,2048,response_template,
             ++provallo_requests_filtered,
             ++provallo_malformed_sessions,
             ++provallo_stability_issues_avoided,
             provallo_classifiers_running,noise_response_distribution);
 
    if( print >0 )
    {
	    ngx_str_set(&text, str_buff); 
    }
    else {
	    ngx_str_set(&text, (const char*)"Error collecting statistics...");
    }
    

    /* hlcf = ngx_http_get_module_loc_conf(r, ngx_http_provallo_nginx_module);




    if (hlcf!=NULL&&hlcf->text) {

	

	    ngx_log_debug(NGX_LOG_DEBUG_HTTP,r->connection->log,0, "ngx_http_complex" );
        
	    if (   ngx_http_complex_value(r, hlcf->text, &text) != NGX_OK) {

            return NGX_HTTP_INTERNAL_SERVER_ERROR;
 	}


        ngx_log_debug1(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,

                       "http provallo_nginx text: \"%V\"", &text);


    } else {

	ngx_log_debug(NGX_LOG_DEBUG_HTTP,r->connection->log,0, "setting simple response string" );



    }

	
    //send header */
    r->headers_out.status = 200;
    r->headers_out.content_length_n = text.len;
    ngx_log_debug(NGX_LOG_DEBUG_HTTP,r->connection->log,0, "send response headers " );
    rc = ngx_http_send_header(r);

    if (rc == NGX_ERROR || rc > NGX_OK || r->header_only) {

        return rc;
    }

    /* send body */

    b = ngx_calloc_buf(r->pool);

    if (b == NULL) {

        return NGX_HTTP_INTERNAL_SERVER_ERROR;

    }


    b->pos = text.data;

    b->last = text.data + text.len;

    b->memory = text.len ? 1 : 0;

    b->last_buf = (r == r->main) ? 1 : 0;

    b->last_in_chain = 1;

    out.buf = b;

    out.next = NULL;

    ngx_log_debug(NGX_LOG_DEBUG_HTTP,r->connection->log,0, "calling output filter " );

    return ngx_http_output_filter(r, &out);

}



static void *

ngx_http_provallo_nginx_create_loc_conf(ngx_conf_t *cf)

{

    ngx_http_provallo_nginx_loc_conf_t  *conf;


    conf = ngx_pcalloc(cf->pool, sizeof(ngx_http_provallo_nginx_loc_conf_t));

    if (conf == NULL) {

        return NULL;

    }


    /*

     * set by ngx_pcalloc():

     *

     *     conf->text = NULL;

     */


    conf->status = NGX_CONF_UNSET;


    return conf;

}

static char *
ngx_http_provallo_nginx_merge_loc_conf(ngx_conf_t *cf, void *parent, void *child)

{

    ngx_http_provallo_nginx_loc_conf_t *prev = parent;

    ngx_http_provallo_nginx_loc_conf_t *conf = child;


    ngx_conf_merge_ptr_value(conf->text, prev->text, NULL);

    ngx_conf_merge_value(conf->status, prev->status, NGX_HTTP_OK);


    return NGX_CONF_OK;

}

static char *
ngx_http_provallo_nginx(ngx_conf_t *cf, ngx_command_t *cmd, void *conf)
{
    ngx_http_core_loc_conf_t  *clcf;
    clcf = ngx_http_conf_get_module_loc_conf(cf, ngx_http_core_module);
    clcf->handler = ngx_http_provallo_nginx_handler;
    return NGX_CONF_OK;
}
#ifdef TEST_UPSTREAM

static ngx_int_t provallo_upstream_handler ( ngx_http_request_t* r )
{

  ngx_int_t rc = NGX_HTTP_INTERNAL_SERVER_ERROR;
  ngx_http_core_loc_conf_t *plcf = ngx_http_get_module_loc_conf(r,ngx_http_proxy_module);
  ngx_http_upstream_t *u;
  
  if(!plcf)
	  return rc;

  u = ngx_pcalloc(r->pool,sizeof(ngx_upstream_t));
  if(u==NULL) 
	  return rc;

  u->peer.log = r->connection->log;
  u->peer.log_error = NGX_ERROR_ERR;
  u->output.tag = (ngx_buf_tag_t) &provallo_nginx_module;
  u->conf=plcf->upstream;

  //setup callbacks

  u->create_request = nullptr;
  u->reinit_request  =nullptr;
  u->process_header  = nullptr;
  u->reinit_request = nullptr;
  u->finalize_request = nullptr;

  //finally set upstream 
  r->upstream = u;

  rc = ngx_http_read_client_request_body(r,provallo_upstream_init);

  return NGX_OK;

}

static ngx_int_t
ngx_http_upstream_init_extend_single_peers(ngx_conf_t *cf,
    ngx_http_upstream_srv_conf_t *us)
{
    ngx_uint_t                               i, n = 0, m = 0;
    ngx_http_combined_upstreams_srv_conf_t  *scf;
    ngx_http_upstream_server_t              *server, *s;
    ngx_addr_t                              *addr = NULL;

    if (us->servers) {
        server = us->servers->elts;
        for (i = 0; i < us->servers->nelts; i++) {
            if (server[i].backup) {
                n += server[i].naddrs;
            } else {
                m += server[i].naddrs;
            }
        }

        if (n == 1) {
            s = ngx_array_push(us->servers);
            addr = ngx_pcalloc(cf->pool, sizeof(ngx_addr_t));
            if (s == NULL || addr == NULL) {
                return NGX_ERROR;
            }
            ngx_memzero(s, sizeof(ngx_http_upstream_server_t));
            s->addrs = addr;
            s->naddrs = 1;
            s->down = 1;
        }
        if (m == 1) {
            s = ngx_array_push(us->servers);
            if (addr == NULL) {
                addr = ngx_pcalloc(cf->pool, sizeof(ngx_addr_t));
            }
            if (s == NULL || addr == NULL) {
                return NGX_ERROR;
            }
            ngx_memzero(s, sizeof(ngx_http_upstream_server_t));
            s->backup = 1;
            s->addrs = addr;
            s->naddrs = 1;
            s->down = 1;
        }
    }

    scf = ngx_http_conf_upstream_srv_conf(us,
                                          ngx_http_combined_upstreams_module);

    if (scf->original_init_upstream(cf, us) != NGX_OK) {
        return NGX_ERROR;
    }

    return NGX_OK;
}

#endif

/*

static ngx_int_t
provallo_upstream_handler(ngx_http_request_t *r)
{
    ngx_int_t                   rc;
    ngx_http_upstream_t        *u;
    ngx_http_proxy_loc_conf_t  *plcf;

    plcf = ngx_http_get_module_loc_conf(r, ngx_http_proxy_module);

    u = ngx_pcalloc(r->pool, sizeof(ngx_http_upstream_t));
    if (u == NULL) {
        return NGX_HTTP_INTERNAL_SERVER_ERROR;
    }

    u->peer.log = r->connection->log;
    u->peer.log_error = NGX_ERROR_ERR;

    u->output.tag = (ngx_buf_tag_t) &ngx_http_proxy_module;

    u->conf = &plcf->upstream;

    u->create_request = ngx_provallo_http_proxy_create_request;
    u->reinit_request = ngx_provallo_http_proxy_reinit_request;
    u->process_header = ngx_provallo_http_proxy_process_status_line;
    u->abort_request = ngx_provallo_http_proxy_abort_request;
    u->finalize_request = ngx_http_proxy_finalize_request;

    r->upstream = u;

    rc = ngx_http_read_client_request_body(r, ngx_http_upstream_init);

    if (rc >= NGX_HTTP_SPECIAL_RESPONSE) {
        return rc;
    }

    return NGX_DONE;
}
*/
