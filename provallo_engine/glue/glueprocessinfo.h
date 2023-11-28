/*
 * glueprocessinfo.h
 *
 *  Created on: Jan 21, 2022
 *      Author: kardon
 */

#ifndef GLUE_GLUEPROCESSINFO_H_
#define GLUE_GLUEPROCESSINFO_H_
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <string>
#include <cstddef>
#include <memory>
#include <iostream>
#include <sys/ptrace.h>
#include <regex>
#include "../decision_engine/utils.h" //real_t

namespace provallo
{
//implements dynamic memory hook :
// based on this sample : https://github.com/dianpeng/dynhook
// without boost.

  class glue_process_info
  {
  public:
    static glue_process_info*
    create (pid_t pid)
    {
      glue_process_info *ret = new glue_process_info (pid);
      if (!ret || !ret->init ())
	  {
	  if (ret)
	    delete ret;
	  return nullptr;

	}
      return ret;
    }
    struct module_info
    {
      uintptr_t start;
      uintptr_t end;
      std::string path;
      module_info () :
	  start (0), end (0), path ()
      {
      }

      module_info (uintptr_t s, uintptr_t e, const std::string &p) :
	  start (s), end (e), path (p)
      {
      }
    };

    struct module_info_less_than
    {
      bool
      operator () (const module_info &l, const module_info &r) const
      {
	      return l.path < r.path;
      }
    };

    typedef std::set<module_info, module_info_less_than> module_list;

    const module_list&
    modules () const
    {
      return m_modules;
    }

    // Find symbol by address
    struct symbol_info
    {
      uintptr_t base; // Base address for this symbol
      std::string name; // Name for this symbol
      size_t size;
      bool weak;

      symbol_info () :
	  base (0), name (), size (0), weak (false)
      {
      }

      symbol_info (uintptr_t b, const std::string &n, size_t sz, bool w) :
	  base (b), name (n), size (sz), weak (w)
      {
      }
    };

    const symbol_info*
    find_symbol (const std::string&) const;

    const symbol_info*
    find_symbol (uintptr_t address) const;
    pid_t
    pid () const
    {
      return _pid;
    }

    const std::string&
    path () const
    {
      return m_entry_info.path;
    }

    const module_info&
    entry_info () const
    {
      return m_entry_info;
    }

  private:
    // For std::lower_bound
    struct symbol_info_less_than
    {
      bool
      operator () (const symbol_info &l, const symbol_info &r) const
      {
	return l.base < r.base;
      }
      bool
      operator () (const symbol_info &l, uintptr_t address) const
      {
	return l.base < address;
      }
      bool
      operator () (uintptr_t address, const symbol_info &r)
      {
	return address < r.base;
      }
    };

    void
    push_symbol_info (const symbol_info &info)
    {
      std::vector<symbol_info>::iterator itr = std::lower_bound (
	    m_symbol_info.begin (), m_symbol_info.end (), info,
	  symbol_info_less_than ());
      std::vector<symbol_info>::iterator pos = m_symbol_info.insert (itr, info);
      m_symbol_name_index.insert (std::make_pair (info.name, *pos));
    }

    // Helper
    bool
    is_abs_path (const std::string &str) const
    {
      return str.size () > 0 && str[0] == '/';
    }

  public: // Task list manipulation
    bool
    attach_all ();
    bool
    stop_all ();
    bool
    resume_all ();
    bool
    resume_and_wait (pid_t, int *status);
    bool
    stop_thread (pid_t);

    struct thread
    {
      pid_t pid;
      enum
      {
	RUNNING, STOPPED
      };
      int state;

      thread (pid_t p, int st) :
	  pid (p), state (st)
      {
      }
    };

    const thread*
    get_thread (pid_t pid) const;

  public:
    // Dump the process information into the output stream
    void
    dump (std::ostream &output) const;
    virtual
    ~glue_process_info ();
  private: // Initialization routines
    bool
    load_process_thread_info (pid_t);
    bool
    load_process_so_list (pid_t);

    bool
    parse_process_module_line (const std::string &line, module_info*);

    bool
    load_symbol_info ();
    bool
    load_symbol_info (const module_info&);

    // Used to do double initialization
    bool
    init ();

    glue_process_info (pid_t);

  private:
    bool
    stop_pid (pid_t);
    bool
    snapshot_thread_list (std::vector<pid_t>*);
    bool
    diff_thread_list (const std::vector<pid_t>&, std::vector<pid_t>*);
    void
    sync_thread_status (const std::vector<pid_t>&);

  private:
    // Module list
    module_list m_modules;

    // Process Id for this process
    pid_t _pid;

    // Process's module
    module_info m_entry_info;

    // Data structure that stores symbol info
    std::vector<symbol_info> m_symbol_info;

    // Index data structure that is used for searching the symbols
    typedef std::multimap<std::string, symbol_info> symbol_index;
    symbol_index m_symbol_name_index;

    // List of threads status
    typedef std::map<pid_t, thread> thread_list;

    thread_list m_thread_list;
    glue_process_info ();

  };

  //tokenize with regex
  template<class container>
    void
    tokenize (const std::string &str, container &tokens,
	      const std::string &delim/*=","*/, const std::string &prefix)
    {

      std::string::size_type lp = str.find_first_not_of (delim, 0);
      std::string::size_type pos = str.find_first_of (delim, lp);
      while (std::string::npos != pos && std::string::npos != lp)
      {

        auto cut = str.substr (lp, pos - lp);
        //remove whitespaces from keys/values :
        //(^[ ]+)|([ ]+$)(^[ ]+)|([ ]+$)
        cut = std::regex_replace(cut, std::regex("\\s+"),"");
        if (cut.length () > 0)
          {

              while (std::isspace (cut[0]))
              {
                cut.erase (cut.begin ());
                if (cut.length () == 1)
                  break;
              }

            if (prefix.length ())
        tokens.push_back (prefix + cut);
            else
        tokens.push_back (cut);
          }
        lp = str.find_first_not_of (delim, pos);
        pos = str.find_first_of (delim, lp);
      }

    }
  //tokenize with with find_first_of
  template<class Container>
  void tokenize(const std::string& str, Container& tokens, const std::string& delimiters = ",") {
      Container copy;
      //use find_first_of to find the first delimiter
      std::string::size_type initial_delim = str.find_first_not_of(delimiters, 0);
      std::string::size_type delim = str.find_first_of(delimiters, initial_delim);
      while (std::string::npos != delim || std::string::npos != initial_delim) {
          // Found a token, add it to the vector.
          // validate the token
          std::string token = str.substr(initial_delim, delim - initial_delim);
          if (token.length() > 0)
          {
              while (std::isspace(token[0]))
              {
                  token.erase(token.begin());
                  if (token.length() == 1)
                      break;
              }
              if (token.length() > 0)
              copy.push_back(token);
          } 
          //just empty? skip it
          // Skip delimiters.  Note the "not_of"
          initial_delim = str.find_first_not_of(delimiters, delim);
          // Find next "non-delimiter"
          delim = str.find_first_of(delimiters, initial_delim);
      }
      
      tokens = copy;
  }
  //specialization for real_t


  template <> 
  void tokenize<std::vector<real_t>>(const std::string& str, std::vector<real_t>& tokens, const std::string& delimiters);

} /* namespace provallo */

#endif /* GLUE_GLUEPROCESSINFO_H_ */
