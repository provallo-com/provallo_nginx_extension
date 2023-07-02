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

#include "helpers/nginx_helper.h"



//taken from hello world module sample as a stub:

static char *ngx_provallo_interface(ngx_conf_t *cf, ngx_command_t *cmd, void *conf);
static ngx_int_t ngx_provallo_interface_handler(ngx_http_request_t *r);

//since we're running on the process context, we can use the global variables and struct instances.
//we can also use the ngx_log_error function to log to the error log.
//we can also use the ngx_cycle->log to log to the error log.
#ifdef ANALYZE_VARS

extern  ngx_cycle_t *ngx_cycle;
extern  ngx_module_t ngx_provallo_interface_module;
extern  ngx_http_request_t *ngx_http_request;
extern  ngx_http_core_main_conf_t *ngx_http_core_main_conf;
extern  ngx_http_connection_t *ngx_http_connection;
extern  ngx_http_core_srv_conf_t *ngx_http_core_srv_conf;
extern  ngx_http_core_loc_conf_t *ngx_http_core_loc_conf;
extern  ngx_http_upstream_srv_conf_t *ngx_http_upstream_srv_conf;
extern  ngx_http_upstream_main_conf_t *ngx_http_upstream_main_conf;
extern  ngx_http_upstream_t *ngx_http_upstream;

extern  ngx_http_upstream_conf_t *ngx_http_upstream_conf;
extern  ngx_http_upstream_server_t *ngx_http_upstream_server;
extern  ngx_http_upstream_loc_conf_t *ngx_http_upstream_loc_conf;
extern  ngx_http_upstream_rr_peer_t *ngx_http_upstream_rr_peer;
extern  ngx_http_upstream_rr_peers_t *ngx_http_upstream_rr_peers;
extern  ngx_http_upstream_rr_peer_data_t *ngx_http_upstream_rr_peer_data;
extern  ngx_http_upstream_rr_peer_t *ngx_http_upstream_rr_peer;

#endif
extern ngx_module_t ngx_provallo_interface_module;



void start_provallo_interface()
{
     //init detection engine    

     ngx_log_error(NGX_LOG_EMERG, ngx_cycle->log, 0, "start_provallo_interface");
}

void reload_provallo_interface()
{
    ngx_log_error(NGX_LOG_EMERG, ngx_cycle->log, 0, "reload_provallo_interface");   
}

void stop_provallo_interface()
{
    ngx_log_error(NGX_LOG_EMERG, ngx_cycle->log, 0, "stop_provallo_interface");
}   


/**
 * This module provided directive: hello world.
 *
 */
static ngx_command_t ngx_provallo_interface_commands[] = {

    {ngx_string("mbalancer"),           /* directive */
     NGX_HTTP_LOC_CONF | NGX_CONF_NOARGS, /* location context and takes
                                             no arguments*/
     ngx_provallo_interface,              /* configuration setup function */
     0,                                   /* No offset. Only one context is supported. */
     0,                                   /* No offset when storing the module configuration on struct. */
     NULL},

    ngx_null_command /* command termination */
};

/* The hello world string. */
static u_char ngx_provallo[] = PROVALLO_HELLO;

/* The module context. */
static ngx_http_module_t ngx_provallo_interface_module_ctx = {
    NULL, /* preconfiguration */
    NULL, /* postconfiguration */

    NULL, /* create main configuration */
    NULL, /* init main configuration */

    NULL, /* create server configuration */
    NULL, /* merge server configuration */

    NULL, /* create location configuration */
    NULL  /* merge location configuration */
};

/* Module definition. */
ngx_module_t ngx_provallo_interface_module = {
    NGX_MODULE_V1,
    &ngx_provallo_interface_module_ctx, /* module context */
    ngx_provallo_interface_commands,    /* module directives */
    NGX_HTTP_MODULE,                    /* module type */
    NULL,                               /* init master */
    NULL,                               /* init module */
    NULL,                               /* init process */
    NULL,                               /* init thread */
    NULL,                               /* exit thread */
    NULL,                               /* exit process */
    NULL,                               /* exit master */
    NGX_MODULE_V1_PADDING};

/**
 * Content handler.
 *
 * @param r
 *   Pointer to the request structure. See http_request.h.
 * @return
 *   The status of the response generation.
 */
static ngx_int_t ngx_provallo_interface_handler(ngx_http_request_t *r)
{
    ngx_buf_t *b;
    ngx_chain_t out;

    /* Set the Content-Type header. */
    r->headers_out.content_type.len = sizeof("text/plain") - 1;
    r->headers_out.content_type.data = (u_char *)"text/plain";

    /* Allocate a new buffer for sending out the reply. */
    b = ngx_pcalloc(r->pool, sizeof(ngx_buf_t));

    /* Insertion in the buffer chain. */
    out.buf = b;
    out.next = NULL; /* just one buffer */

    b->pos = ngx_provallo;                             /* first position in memory of the data */
    b->last = ngx_provallo + sizeof(ngx_provallo) - 1; /* last position in memory of the data */
    b->memory = 1;                                     /* content is in read-only memory */
    b->last_buf = 1;                                   /* there will be no more buffers in the request */

    /* Sending the headers for the reply. */
    r->headers_out.status = NGX_HTTP_OK; /* 200 status code */
    /* Get the content length of the body. */
    r->headers_out.content_length_n = sizeof(ngx_provallo) - 1;
    ngx_http_send_header(r); /* Send the headers */

    /* Send the body, and return the status code of the output filter chain. */
    return ngx_http_output_filter(r, &out);
} /* ngx_provallo_interface_handler */

/**
 * Configuration setup function that installs the content handler.
 *
 * @param cf
 *   Module configuration structure pointer.
 * @param cmd
 *   Module directives structure pointer.
 * @param conf
 *   Module configuration structure pointer.
 * @return string
 *   Status of the configuration setup.
 */
static char *ngx_provallo_interface(ngx_conf_t *cf, ngx_command_t *cmd, void *conf)
{
    ngx_http_core_loc_conf_t *clcf; /* pointer to core location configuration */

    /* Install the hello world handler. */
    clcf = ngx_http_conf_get_module_loc_conf(cf, ngx_http_core_module);
    clcf->handler = ngx_provallo_interface_handler;

    return NGX_CONF_OK;
} /* ngx_provallo_interface */
