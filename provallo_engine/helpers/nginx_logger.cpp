#include "nginx_logger.h"
#include <iostream>
#include <streambuf>
#include <string>


namespace provallo {

NginxLogger::NginxLogger(ngx_log_t *log, ngx_uint_t level)
    : m_log(log), m_level(level)
{
}


NginxLogger::~NginxLogger()
{
}


void NginxLogger::setLevel(ngx_uint_t level)
{
    m_level = level;
}


