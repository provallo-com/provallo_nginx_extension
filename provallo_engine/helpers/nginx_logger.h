
#ifndef _NGINX_LOGGER_H_
#define _NGINX_LOGGER_H_

#include <iostream>
#include <streambuf>
#include <string>
#include<memory.h>
#include <cstring>

#include <syslog.h>
#include <iomanip>
#include <mutex>


#include <nginx.h>

namespace provallo {

class NginxLogger : public std::streambuf
{
public:
    NginxLogger(ngx_log_t *log, ngx_uint_t level);
    ~NginxLogger();

    void setLevel(ngx_uint_t level);
    void setLog(ngx_log_t *log);

protected:
    int sync(); 
    int_type overflow(int_type c);
    std::streamsize xsputn(const char *s, std::streamsize n);

private:
    ngx_log_t *m_log;
    ngx_uint_t m_level;
    std::mutex m_mutex;
    std::string m_buf;
};


class NginxLogStream : public std::ostream
{
public:
    NginxLogStream(ngx_log_t *log, ngx_uint_t level);
    ~NginxLogStream();

    void setLevel(ngx_uint_t level);
    void setLog(ngx_log_t *log);

private:
    NginxLogger m_buf;
};


class NginxLog
{
public:
    static NginxLogStream debug(ngx_log_t *log);
    static NginxLogStream info(ngx_log_t *log);
    static NginxLogStream notice(ngx_log_t *log);
    static NginxLogStream warn(ngx_log_t *log);
    static NginxLogStream error(ngx_log_t *log);
    static NginxLogStream crit(ngx_log_t *log);
    static NginxLogStream alert(ngx_log_t *log);
    static NginxLogStream emerg(ngx_log_t *log);
};      

// NginxLogger
inline NginxLogger::NginxLogger(ngx_log_t *log, ngx_uint_t level)
    : m_log(log), m_level(level)
{

}

inline NginxLogger::~NginxLogger()
{
    sync(); 
}
 

}// namespace provallo

#endif // _NGINX_LOGGER_H_