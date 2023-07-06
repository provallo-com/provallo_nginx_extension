/*
 * syslog.h
 *
 *  Created on: May 6, 2021
 *      Author: kardon
 */

#ifndef THIRD_PARTY_SYSLOG_H_
#define THIRD_PARTY_SYSLOG_H_

#include <iostream>
#include <streambuf>
#include <string>
#include<memory.h>
#include <cstring>

#include <syslog.h>
#include <iomanip>
#include <mutex>
namespace syslogger
{
  struct level
  {
    enum pri
    {
      emerg = LOG_EMERG, 	// A panic condition
      alert = LOG_ALERT, 	// A condition that should be corrected
      critical = LOG_CRIT, 	// Critical condition, e.g, hard device error
      error = LOG_ERR, 	// Errors
      warning = LOG_WARNING, 	// Warning messages
      notice = LOG_NOTICE, 	// Possibly be handled specially
      info = LOG_INFO, 	// Informational
      debug = LOG_DEBUG 	// For debugging program
    };
  };

  class streambuf : public std::streambuf
  {
    std::string _buf;
    int _level;
    std::recursive_mutex _locker;
  public:

    streambuf () :
      _buf(""),_level (level::debug)
    {
    }
    void
    level (int level)
    {
      _level = level;
    }

  protected:

    int
    sync ()
    {
      std::lock_guard<std::recursive_mutex> lock (_locker);
      if (_buf.length()>5)
	    {
  	     ::syslog (_level, "%s", _buf.c_str ());
	      _buf.erase ();
	    }
      return 0;
    }
    int_type
    overflow (int_type c)
    {
      std::lock_guard<std::recursive_mutex> lock (_locker);

      if (c == traits_type::eof ())
    	sync ();
      else {


	    _buf += static_cast<char> (c);
      }
      return c;
    }

  };

  /**/
  class ostream : public std::ostream
  {
    streambuf _logbuf;
  public:
    ostream () :
	std::ostream (&_logbuf)
    {
    }   
    ostream&
    operator<< (const level::pri lev)
    {
      _logbuf.level (lev);
      return *this;
    }
  };

  class redirect
  {
    ostream dst;
    std::ostream &src;
    std::streambuf *const sbuf;

  public:

    redirect (std::ostream &src) :
	  src (src), sbuf (src.rdbuf (dst.rdbuf ()))
    {
      dst
	  << (::memcmp (&src, &std::cout, sizeof(src)) == 0 ?
	      level::info : level::error);

    }
    virtual ~redirect ()
    {
      src.rdbuf (sbuf);
    }
  };
}

#endif /* THIRD_PARTY_SYSLOG_H_ */
