/*
 * procwatcher.h
 *
 *  Created on: Jan 17, 2022
 *      Author: kardon
 */

#ifndef UTIL_PROCWATCHER_H_
#define UTIL_PROCWATCHER_H_
#include <string>
#include <vector>
#include <queue>
#include <unordered_map>
#include <iostream>
#include <cerrno>
#include <cstring>
#include <sys/types.h>
#include <sys/inotify.h>
#include <dirent.h>
#include <unistd.h>

namespace provallo
{

#define PROC_PATH "/proc"

  class system_process
  {
  public:

    system_process ()
    {
    }
    virtual bool
    check (const std::string &pid) const =0;
    virtual ~system_process ()=default;

  };

  class str_process : system_process
  {
    std::vector<std::string> _procs;
  public:
    template<typename It>
      str_process (It begin, It end) :
	  _procs (begin, end)
      {
      }
    str_process (const std::string &pname)
    {
      if (pname.length () > 0)
	_procs.push_back (pname);
    }
    bool
    check (const std::string &pid) const;
  };
  class process_desc
  {
    std::string _pid;
    std::string _pathname;

  public:
    process_desc (const std::string &path):_pid(path) {}

//class methods
  public:
    static std::string
    get_last_pid ();
    static int
    get_max_pid ();
    static std::string
    get_cmname (const std::string&);
    static std::string
    get_cmdline (const std::string&);
    const std::string& get_process_id() const
    {
      return _pid;
    }

  };

   template<class observer_policy>
    class proc_watcher : observer_policy
    {

      std::queue<process_desc*> _processes;
      std::string _last;
      static constexpr const int ev_size = sizeof(struct inotify_event);
      static constexpr const int ev_buf_size = (ev_size + 16) * 1024;
    private:
      void
      init ();
    public:
      proc_watcher ()
      {
	init ();
      }
      template<typename arg>
	proc_watcher (const arg &a) :
	    observer_policy (a)
	{
	  init ();
	}
      template<typename arg1, typename arg2>
	proc_watcher (const arg1 &a, const arg2 &b) :
	    observer_policy (a, b)
	{
	  init ();
	}
      std::vector<process_desc*>
      watch ();
      virtual
      ~proc_watcher ();
    };

  template<class observer_policy>
    void
    proc_watcher<observer_policy>::init ()
    {
      DIR *d = opendir ("/proc");
      struct dirent *entry = nullptr;
      while (((entry = readdir (d)) != 0))
	{
	  if (entry && isdigit (entry->d_name[0]))
	    {
	      std::string pid (entry->d_name);
	      if (observer_policy::check (pid))
		_processes.push (new process_desc (pid));

	    }
	}
      if (d)
	closedir (d);
      _last = process_desc::get_last_pid ();
    }

  template<class observer_policy>
    std::vector<process_desc*>
    proc_watcher<observer_policy>::watch ()
    {

    static char buffer[ev_buf_size] =
	{ };
      std::vector<process_desc*> prc_buf;
      while (!_processes.empty ())
	{
	  auto process_desc = _processes.front ();
	  prc_buf.push_back (process_desc);
	  _processes.pop ();
	}
      if (!prc_buf.empty ())
	return prc_buf;

      static int fd = -1;
      if (fd == -1)
	fd = inotify_init ();
      int watch = inotify_add_watch (fd, PROC_PATH, IN_ALL_EVENTS);
      if (watch < 0)
	{
	  std::cerr
	      << std::string ("provallo proc_watcher inofity init error : ");
	  std::cerr << strerror (errno);
	}
      else
	{
	  int length = read (fd, buffer, ev_buf_size), i = 0;
	  if (length >= 0)
	    {

	      do
		{
		  struct inotify_event *event =
		      (struct inotify_event*) &buffer[i];
		  std::string latest (process_desc::get_last_pid ());
		  if (latest != _last)
		    {
		      std::vector<std::string> pids;
		      int lpid = std::atoi (_last.c_str ()), npid = std::atoi (
			  latest.c_str ());
		      _last = latest;
		      int max_pid = process_desc::get_max_pid ();
		      for (int j = lpid + 1; j % max_pid <= npid; ++j)
			{
			  std::string pid = std::to_string ((j % max_pid));
 			}

		    }
		  i += ev_size + event->len;

		}
	      while (i < length);
	    }
	  auto ret = ::close (fd);

	  ret &= inotify_rm_watch (fd, watch);
	  fd = -1;
	}

      return prc_buf;

    }

  template<class observer_policy>
    proc_watcher<observer_policy>::~proc_watcher ()
    {
      while (!_processes.empty ())
	{
	  process_desc *pf = _processes.front ();
	  delete pf;
	  _processes.pop ();
	}
    }
} /* namespace provallo */

#endif /* UTIL_PROCWATCHER_H_ */
