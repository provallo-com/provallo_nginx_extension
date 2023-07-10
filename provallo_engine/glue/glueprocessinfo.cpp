/*
 * glueprocessinfo.cpp
 *
 *  Created on: Jan 21, 2022
 *      Author: kardon
 */

#include "glueprocessinfo.h"
#include "linuxglue.h"
#include "../statistics/statcollector.h"
#include <unistd.h>

#include <errno.h>
#include <fstream>
#include <iostream>
#include <iomanip>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/types.h>
#include <signal.h>
#include <cstring>
#include <libelf.h>

namespace provallo
{

  glue_process_info::glue_process_info (pid_t pid) :
      m_modules (), _pid (pid), m_entry_info (), m_symbol_info (), m_symbol_name_index (), m_thread_list ()
  {
    // TODO Auto-generated constructor stub

  }
  bool
  glue_process_info::init ()
  {
    if (!load_process_so_list (_pid))
      return false;
    if (!load_symbol_info ())
      return false;
    return true;
  }
  bool
  glue_process_info::load_symbol_info ()
  {

    for (auto mod : m_modules)
      {
	if (!load_symbol_info (mod))
	  {
	    return false;
	  }
      }
    return true;
  }

  bool
  glue_process_info::load_symbol_info (const module_info &mod)
  {
    const bool is_entry = mod.path == path ();
    //const uintptr_t offset = is_entry ? 0 : mod.start;
    int fd = ::open (mod.path.c_str (), O_RDONLY);
    if (fd < 0)
      {
	return false;
      }
    if (elf_version(EV_CURRENT) == EV_NONE)

      {
 	std::cerr<<"[-] elf read will fail!"<<std::endl;
      }
    Elf *elf = elf_begin (fd, ELF_C_READ, NULL);
    if (elf == NULL)
      {
	std::cerr << "Cannot call function elf_begin with error: "
	    << elf_errmsg (elf_errno ());

	::close (fd);

	return false;
      }

    int cnt = 0;
    Elf_Scn *elf_section = NULL;
    Elf64_Shdr *elf_shdr;
    do
      {
	while ((elf_section = elf_nextscn (elf, elf_section)) != NULL)
	  {
	    if ((elf_shdr = elf64_getshdr (elf_section)) != NULL)
	      {


		if (elf_shdr->sh_type == SHT_SYMTAB

		    || elf_shdr->sh_type == SHT_DYNSYM)
		      {
			cnt++;
			break;
		      }
		else
		  {
		    if (elf_shdr->sh_type == SHT_DYNSYM)
		      {
			cnt++;
			break;
		      }
		  }
	      }
	  }
      }
    while (is_entry && cnt < 2);
#if DEBUG__
    if(m_symbol_info.size()==0){
      std::cout<<"[=] mismatch symbols from "<< std::string(m_entry_info.path) << "module path"<<std::string(mod.path.c_str()) <<std::endl;
    }
    else{
    std::cout << "[+]Loaded " <<std::to_string( m_symbol_info.size () ) << " symbols" <<std::endl;
    }
#endif
    ::close (fd);
    return true;
  }

  bool
  glue_process_info::load_process_so_list (pid_t pid)
  {

    static char buf[1024];
    snprintf (buf, 1024, "/proc/%d/maps", pid);
    std::string path = buf;
    std::fstream file (path.c_str (), std::ios::in);
    if (!file)
      {
	std::cerr << "Cannot open file:" << path << " with error :"
	    << std::strerror (errno);
	return false;
      }

    // Now we start to parse the process/maps file
    std::string line;

    while (std::getline (file, line))
      {
	if (!line.empty ())
	  {
	    module_info minfo;
	    if (parse_process_module_line (line, &minfo))
	      {
		m_modules.insert (minfo);
		// Assume very first line is the path of the executable
		if (m_entry_info.path.empty ())
		  {
		    m_entry_info = minfo;
		  }
	      }
	  }
      }

    return true;

  }

  uintptr_t
  address_cast (const std::string &src)
  {
    std::stringstream formatter;
    formatter << std::hex << src;
    uintptr_t ret;
    formatter >> ret;
    return ret;
  }


  bool
  glue_process_info::parse_process_module_line (const std::string &line,
						module_info *output)
  {
    typedef std::vector<std::string> container;
    container words;
    tokenize<container> (line, words, " ", std::string());


    auto pop=line.find_first_of('/');
    if( pop!=std::string::npos)
      {
	output->path=line.substr(pop,line.length()-pop);
	if(words.size()>1)
	  {
	    size_t f=0; while (f<words.size()&&words[f].length()<2) f++;

	    const std::string&range = words[f];
	    std::string::size_type pos = range.find_first_of ("-");
	    if(pos!=std::string::npos)
	    output->start = address_cast (range.substr (0, pos));
	    output->end = address_cast (
		    range.substr (pos + 1, range.size () - pos - 1));

	  }

	    return true;
      }

	return false;

  }
    bool
    glue_process_info::resume_and_wait (pid_t pid, int *status)
      {
	thread_list::iterator itr = m_thread_list.find (pid);
	if (itr == m_thread_list.end ())
	  {
	    std::cerr << "Try to resume and wait on pid:" << pid
	    << "However this pid is not attached or existed!";
	    return false;
	  }
	if (itr->second.state == thread::RUNNING)
	  {
	    std::cerr << "Try to resume and wait on pid:" << pid
	    << "However it is running!";
	    return false;
	  }
	if (!linux_glue::ptrace_cont_and_wait_event (pid, status))
	return false;
	return true;
      }
    const glue_process_info::symbol_info*
    glue_process_info::find_symbol (const std::string &name) const
      {
	// Query address by the symbol name
	typedef symbol_index::const_iterator itr;
	std::pair<itr, itr> ret = m_symbol_name_index.equal_range (name);
	if (ret.first == ret.second)
	  {
	    return NULL;
	  }
	// Try to find a strong symbol
	for (itr beg = ret.first; beg != ret.second; ++beg)
	  {
	    const symbol_info &sinfo = beg->second;
	    if (!sinfo.weak)
	      {
		return &sinfo;
	      }
	  }

	// Just return a weak symbol
	return &ret.first->second;
      }
    const glue_process_info::symbol_info*
    glue_process_info::find_symbol (uintptr_t address) const
      {
	std::vector<symbol_info>::const_iterator itr = std::lower_bound (
	    m_symbol_info.begin (), m_symbol_info.end (), address,
	    symbol_info_less_than ());
	if (itr == m_symbol_info.end ())
	return NULL;
	const symbol_info &sinfo = *itr;
	// ::assert(address >= sinfo.base && address <= sinfo.base + sinfo.size);
	return &sinfo;
      }

    bool
    glue_process_info::snapshot_thread_list (std::vector<pid_t> *output)
      {
		char buff[2048] =
	  	{};
		snprintf (buff, 2048, "/proc/%d/task", this->_pid);
		//parse tids from buffer and add to output
		output->clear();
		
		return true;
	  }
 

    bool
    glue_process_info::attach_all ()
      {
	std::vector<pid_t> tlist;
	std::vector<pid_t> diff;

	do
	  {
	    if (!snapshot_thread_list (&tlist))
	    return false;
	    if (diff_thread_list (tlist, &diff))
	      {
		sync_thread_status (tlist);
		break;
	      }
	    for (pid_t p : diff)
	      {
		int status;
		if (!linux_glue::ptrace_attach_and_wait (p, &status))
		return false;
		m_thread_list.insert (
		    std::make_pair (p, thread (p, thread::STOPPED)));
	      }
	    tlist.clear ();
	    diff.clear ();
	  }
	while (true);
	return true;

      }
    bool glue_process_info::resume_all()
    {
	for (auto & t : m_thread_list)
	  t.second.state = thread::RUNNING;
	return true;

    }
    glue_process_info::~glue_process_info ()
      {
	// TODO Auto-generated destructor stub
      }

  bool
  glue_process_info::diff_thread_list (const std::vector<pid_t> &allocator,
      std::vector<pid_t> *allocator1)
    {

      if (allocator1)
		{
			for (pid_t p : allocator)
				{
				if (m_thread_list.find (p) == m_thread_list.end ())
					{
					allocator1->push_back (p);
					}
				}

		return allocator1->empty ();
		}

      return false;

   }

  void
  glue_process_info::sync_thread_status (const std::vector<pid_t> &allocator)
    {

      for (thread_list::iterator itr = m_thread_list.begin ();
	  itr != m_thread_list.end ();)
	{
	  std::vector<pid_t>::const_iterator ret = std::find (allocator.begin (),
	      allocator.end (),
	      itr->second.pid);
	  if (ret == allocator.end ())
	    {
	      m_thread_list.erase (itr++);
	    }
	  else
	    {
	      ++itr;
	    }
	}

    }

} /* namespace provallo */
