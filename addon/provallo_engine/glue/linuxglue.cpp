/*
 * linuxglue.cpp
 *
 *  Created on: Jan 21, 2022
 *      Author: kardon
 */
#include <sys/ptrace.h>
#include <sys/user.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <errno.h>
#include <assert.h>

#include <cstring>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include "linuxglue.h"
namespace
{
  extern "C"
  {
#include "dynasm/dasm_proto.h"
#include "dynasm/dasm_x86.h"
    enum
    {
      MEM_MAP_GLOBALSstart, MEM_MAP_GLOBALS_MAX
    };

  }

}
namespace provallo
{

  class load_symbol : public stub
  {
  public:
    static load_symbol*
    create (const glue_process_info &proc, const std::string &so,
	    const std::string &name)
    {
      load_symbol *ret = new load_symbol ();
      if (ret && !ret->init (proc, so, name))
	{
	  delete ret;
	  return nullptr;
	}
      return ret;
    }

    virtual void*
    code () const
    {
      return _code;
    }

    virtual size_t
    size () const
    {
      return m_code_size;
    }

    virtual size_t
    rip_offset () const
    {
      return m_data_size;
    }

    const std::string&
    so_name () const
    {
      return m_so;
    }

    const std::string&
    hook_name () const
    {
      return m_hook;
    }

    // Dump the whole freaking code body
    void
    dump (std::ostream&);

  private:
    load_symbol () :
	stub (), _code (nullptr), m_code_size (0), m_data_size (0), m_so (), m_hook ()
    {
    }

    bool
    init (const glue_process_info&, const std::string&, const std::string&);

  private:
    void *_code;
    size_t m_code_size;
    size_t m_data_size;
    std::string m_so;
    std::string m_hook;
  };

  linux_glue::linux_glue (const std::string &process_name)
  {
    // TODO Auto-generated constructor stub
  }

  linux_glue::~linux_glue ()
  {
    // TODO Auto-generated destructor stub
  }
  bool
  linux_glue::hook ()
  {
    bool ret = false;

    return ret;
  }
  bool
  linux_glue::unhook ()
  {
    bool ret = false;

    return ret;

  }

  bool
  linux_glue::ptrace_peek (pid_t pid, uintptr_t address, uintptr_t *ret)
  {
    errno = 0;
    *ret = ::ptrace (PTRACE_PEEKTEXT, pid, address, 0);
    if (errno)
      {
	std::cerr << "ptrace(PTRACE_PEEKTEXT," << pid << "," << address
	    << ") failed with:" << std::strerror (errno);
	return false;
      }
    return true;
  }
  bool
  linux_glue::ptrace_poke (pid_t pid, uintptr_t address, uintptr_t value)
  {
    errno = 0;
    ::ptrace (PTRACE_POKETEXT, pid, address, value);
    if (errno)
      {
	std::cerr << "ptrace(PTRACE_POKETEXT," << pid << "," << address << ","
	    << value << ") failed with:" << std::strerror (errno);
	return false;
      }
    return true;
  }
  bool
  linux_glue::ptrace_getregs (pid_t pid, struct user_regs_struct *output)
  {
    errno = 0;
    ::ptrace (PTRACE_GETREGS, pid, 0, output);
    if (errno)
      {
	std::cerr << "ptrace(PTRACE_GETREGS," << pid << "," << std::hex
	    << output << std::dec << ") failed with:" << std::strerror (errno);
	return false;
      }
    return true;
  }
  bool
  linux_glue::ptrace_setregs (pid_t pid, const struct user_regs_struct &output)
  {
    errno = 0;
    ::ptrace (PTRACE_SETREGS, pid, 0, &output);
    if (errno)
      {
	std::cerr << "ptrace(PTRACE_SETREGS," << pid << "," << ") failed with:"
	    << std::strerror (errno);
	return false;
      }
    return true;
  }
  bool
  linux_glue::ptrace_continue (pid_t pid)
  {
    errno = 0;
    ::ptrace (PTRACE_CONT, pid, 0, 0);
    if (errno)
      {
	std::cerr << "ptrace(PTRACE_CONT," << pid << ") failed with:"
	    << std::strerror (errno);
	return false;
      }
    return true;
  }
  bool
  linux_glue::ptrace_signal (pid_t pid, int sig)
  {
    errno = 0;
    ::ptrace (PTRACE_CONT, pid, 0, sig);
    if (errno)
      {
	std::cerr << "ptrace(PTRACE_CONT," << pid << ") failed with:"
	    << std::strerror (errno);
	return false;
      }
    return true;
  }
  bool
  linux_glue::ptrace_cont_and_wait_event (pid_t pid, int *status)
  {
    if (!ptrace_continue (pid))
      return false;
    // Now blocked for events send by peers
    errno = 0;
    pid_t p = ::waitpid (pid, status, __WALL);
    if (errno)
      {
	std::cerr << "waitpid(" << pid << ") failed with:"
	    << std::strerror (errno);
	return false;
      }
    if (p != pid)
      return false;
    return true;
  }
  bool
  linux_glue::ptrace_attach (pid_t pid)
  {
    errno = 0;
    ::ptrace (PTRACE_ATTACH, pid, 0, 0);
    if (errno)
      {
	std::cerr << "ptrace(PTRACE_ATTACH," << pid << ") failed with:"
	    << std::strerror (errno);
	return false;
      }
    return true;
  }
  bool
  linux_glue::ptrace_attach_and_wait (pid_t pid, int *status)
  {
    if (!ptrace_attach (pid))
      return false;
    errno = 0;
    ::pid_t p = ::waitpid (pid, status, __WALL);
    if (errno)
      {
	std::cerr << "waitpid(" << pid << ") failed with:"
	    << std::strerror (errno);
	return false;
      }
    if (p != pid)
      return false;
    return true;
  }

  bool
  linux_glue::remote_allocator::init ()
  {

    bool r1 = _low_pool->init ();
    bool r2 = _high_pool->init ();
    return r1 || r2;
  }

  uintptr_t
  linux_glue::remote_allocator::allocate (size_t cap, uintptr_t hint)
  {
    if (hint < pool::kHighHint)
      {
	// Try to allocate it from low address pool
	uintptr_t ret = _low_pool->allocate (cap);
	if (ret == 0)
	  {
	    // Try high pool since low pool may not be able to allocate
	    return _high_pool->allocate (cap);
	  }
      }
    else
      {
	return _high_pool->allocate (cap);
      }
    return 0;
  }

  size_t
  linux_glue::remote_allocator::size () const
  {
    return _low_pool->size () + _high_pool->size ();
  }

  size_t
  linux_glue::remote_allocator::capacity () const
  {
    return _low_pool->capacity () + _high_pool->capacity ();
  }

  linux_glue::remote_allocator::remote_allocator (glue_process_info *pinfo) :
      _low_pool (
	  new provallo::linux_glue::remote_allocator::pool (
	      pinfo, provallo::linux_glue::remote_allocator::pool::pool::LOW)), _high_pool (
	  new provallo::linux_glue::remote_allocator::pool (
	      pinfo, provallo::linux_glue::remote_allocator::pool::pool::HIGH))
  {
  }

  linux_glue::remote_allocator::~remote_allocator ()
  {
  }
  inline const provallo::glue_process_info::module_info*
  find_injectable_segment (const provallo::glue_process_info &info)
  {
    const provallo::glue_process_info::module_list &mlist = info.modules ();
    for (auto itr = info.modules ().begin (); itr != info.modules ().end ();
	++itr)
      {
	const glue_process_info::module_info &minfo = *itr;
	if (minfo.path == info.path ())
	  {
	    return &minfo;
	  }
      }
    return NULL;
  }

  class code_copy
  {
  public:
    bool
    init ()
    {

      const size_t len = m_code.size (); // Size of the code that needs to be replaced
      const size_t loops = (len / kWordSize) + 1; // We peek this much of memory

      // Using ptrace to grab *ALL* the required data from the target process
      m_backup_code = new uintptr_t[loops];

      // Using ptrace to peek all the data out in the current process
      // Here ptrace can fail and we have no way to tell the caller that
      // we failed at ptrace without using exceptions , FUCK
      std::cout << "Try to peek the target process :" << m_pid
	  << " from address: " << m_segment.start << " until "
	  << loops * kWordSize << "!";

      for (size_t i = 0; i < loops; ++i)
	{
	  uintptr_t data;
	  if (!linux_glue::ptrace_peek (m_pid, m_segment.start + i * kWordSize,
					&data))
	    return false;
	  m_backup_code[i] = data;
	}

      std::cout << "Finish peek the target process :" << m_pid << "!";

      // Now try to poke the data to the target process
      size_t word_len = len / kWordSize;
      size_t trailer = len - word_len * kWordSize;

      for (size_t i = 0; i < len; i += kWordSize)
	{
	  if (!linux_glue::ptrace_poke (
	      m_pid,
	      m_segment.start + i,
	      *reinterpret_cast<uintptr_t*> (static_cast<char*> (m_code.code ())
		  + i)))
	    return false;
	  ++m_poked_size;
	}

      // Finish the trailer parts
	{
	  uintptr_t buf = 0;
	  assert(trailer < kWordSize);
	  if (!trailer)
	    {
	      memcpy (
		  &buf,
		  static_cast<char*> (m_code.code ()) + word_len * kWordSize,
		  trailer);
	      if (!linux_glue::ptrace_poke (
		  m_pid, m_segment.start + word_len * kWordSize, buf))
		return false;
	      ++m_poked_size;
	    }
	}

      std::cout << "Finish poke the target process :" << m_pid << "!";

      return true;
    }

    // Recovery inside of the destructor
    ~code_copy ()
    {
      for (size_t i = 0; i < m_poked_size; ++i)
	{
	  if (!linux_glue::ptrace_poke (m_pid, m_segment.start + i * kWordSize,
					m_backup_code[i]))
	    return;
	}
      std::cout << "Finish recovery the poked process: " << m_pid
	  << " memory address spaces!";
    }

    code_copy (pid_t pid, const glue_process_info::module_info &segment,
	       const stub &code) :
	m_pid (pid), m_segment (segment), m_code (code), m_backup_code (), m_poked_size (
	    0)
    {     // assert(m_segment.end-m_segment.start >= m_code.size());

    }

  private:
    // PID of the target process
    pid_t m_pid;

    // Which segment my target code will go to
    const glue_process_info::module_info &m_segment;

    // Code needs to be patched
    const stub &m_code;

    // Buffer to store the backup code in the remote process
    uintptr_t *m_backup_code;

    // Length of the code that *HAS BEEN* modified in the remote process
    // This conut is in machine word not byte
    size_t m_poked_size;
  };

  // RAII class for set the register
  class register_setter
  {
  public:
    // We only need to support setting R8,R9,RIP and RAX registers
    enum
    {
      R8, R9, RIP, RAX
    };

    bool
    init ()
    {
      errno = 0;
      ;
      if (!linux_glue::ptrace_getregs (m_pid, &m_old_regs))
	return false;
      m_new_regs = m_old_regs; // Copy to the new regs
      return true;
    }

    void
    set (int reg, uintptr_t val)
    {
#if 0
	   switch(reg) {
       case R8:
         m_new_regs.r8 = val;
         break;
       case R9:
         m_new_regs.r9 = val;
         break;
       case RIP:
         m_new_regs.rip = val;
         break;
       case RAX:
         m_new_regs.rax = val;
         break;
       default:
#else
      //assert(0);//
      return;
      //}}
#endif
    }

    bool
    perform ()
    {
      if (!linux_glue::ptrace_setregs (m_pid, m_new_regs))
	return false;
      m_modify = true;
      return true;
    }

    ~register_setter ()
    {
      if (m_modify)
	{
	  if (!linux_glue::ptrace_setregs (m_pid, m_old_regs))
	    return;
	}
    }

    register_setter (pid_t pid) :
	m_pid (pid), m_old_regs (), m_new_regs (), m_modify (false)
    {
    }

    const struct user_regs_struct&
    old_regs () const
    {
      return m_old_regs;
    }

    const struct user_regs_struct&
    new_regs () const
    {
      return m_new_regs;
    }

  private:
    pid_t m_pid;
    struct user_regs_struct m_old_regs;
    struct user_regs_struct m_new_regs;
    bool m_modify;
  };

  //should be only in stub.
  bool
  invoke (provallo::glue_process_info *pinfo, const stub &code, uintptr_t r9,
	  uintptr_t *ret)
  {
    const provallo::glue_process_info::module_info *minfo =
	find_injectable_segment (*pinfo);

    if (!minfo)
      {
	std::cerr << "Cannot find a correct segment for code injection!";
	return false;
      }

    // 1. Copy the code that user wants to invoke to the remote process
    code_copy cc (pinfo->pid (), *minfo, code);
    if (!cc.init ())
      return false;

    // 2. Set up the registers for doing the job
    register_setter rset (pinfo->pid ());
    if (!rset.init ())
      return false;

    // Set the RIP
    rset.set (register_setter::RIP, minfo->start + code.rip_offset () + 2);

    // Set the R8
    rset.set (register_setter::R8, minfo->start);

    // Set the R9
    rset.set (register_setter::R9, r9);

    if (!rset.perform ())
      return false;

    // 3. Continue the target process
      {
	int status;
	if (!pinfo->resume_and_wait (pinfo->pid (), &status))
	  return false;
	// Check what kind of events/signal got from that thread/process
	if (!WIFSTOPPED(status))
	  {
	    // Fucked up here, unexpected signal and child process events
	    // TODO:: Add more detail logging
	    std::cerr << "Process:" << pinfo->pid ()
		<< " exit unexpected ,we are in the"
		    " middle of executing our remote hook functions !";
	    return false;
	  }

	int sig = WSTOPSIG(status);
	if (sig != SIGTRAP)
	  {
	    // TODO:: Add more detail logging
	    std::cerr << "We wait for the process:" << pinfo->pid ()
		<< " to stop but not for a trap signal , signal:" << sig;

	    // For debugging purpose we fowrad it
	    linux_glue::ptrace_signal (pinfo->pid (), sig);
	    std::cerr << "We forward the signal:" << sig << " to the process:"
		<< pinfo->pid ();
	    return false;
	  }
      }

// 4. Get the return value
      {
	struct user_regs_struct creg;
	if (!linux_glue::ptrace_getregs (pinfo->pid (), &creg))
	  return false;
	// Set the return value
#if 0
 *ret = creg.rax;
#endif
      }

// RAII guarantees us to recover the old register status
    return ret;
  }

  bool
  mem_map::init (const glue_process_info &info, size_t size, uintptr_t addr,
		 int flag)
  {
    // Resolve symbols
    const glue_process_info::symbol_info *mm = info.find_symbol ("mmap");
    if (!mm)
      {
	std::cerr << "Cannot resolve symbol mmap in target process!";
	return false;
      }

    static const unsigned char actions[194] =
      { 248, 10, 255, 235, 255, 0, 255, 144, 144, 255, 72, 141, 61, 244, 10,
	  255, 190, 2, 0, 0, 0, 255, 72, 184, 237, 237, 252, 255, 208, 255, 72,
	  133, 192, 15, 133, 244, 247, 255, 72, 199, 192, 1, 0, 0, 0, 255, 205,
	  3, 255, 248, 1, 255, 72, 137, 199, 255, 72, 141, 53, 244, 10, 72, 129,
	  198, 239, 255, 72, 199, 199, 237, 72, 199, 198, 237, 72, 199, 194, 7,
	  0, 0, 0, 72, 199, 193, 237, 73, 199, 192, 252, 255, 252, 255, 252,
	  255, 252, 255, 73, 199, 193, 0, 0, 0, 0, 72, 184, 237, 237, 252, 255,
	  208, 255, 248, 10, 72, 199, 199, 237, 72, 199, 198, 237, 72, 184, 237,
	  237, 252, 255, 208, 205, 3, 255, 65, 81, 65, 80, 72, 184, 237, 237,
	  252, 255, 208, 65, 88, 65, 89, 255, 65, 80, 65, 81, 255, 72, 184, 237,
	  237, 252, 255, 208, 65, 89, 65, 88, 255, 72, 133, 192, 15, 133, 244,
	  248, 255, 72, 199, 192, 2, 0, 0, 0, 255, 248, 2, 255, 76, 137, 207,
	  252, 255, 208, 255, 72, 49, 192, 255 };

    static void *MEM_MAP_GLOBALS[MEM_MAP_GLOBALS_MAX];
    dasm_State *state;

    dasm_init (&state, 1);
    dasm_setupglobal (&state, MEM_MAP_GLOBALS, MEM_MAP_GLOBALS_MAX);
    dasm_setup (&state, actions);
    m_alloc_size = size;
    m_flag = flag;
    m_addr = addr;

    dasm_put (&state, 7);

    // Prolog
    // Call mmap
    // PROT_READ | PROT_WRITE | PROT_EXEC == 7
    // MAP_ANON | MAP_PRIVATE == 34
    //| mov rdi, addr
    //| mov rsi, size
    //| mov rdx, 7
    //| mov rcx, flag
    //| mov r8 , -1
    //| mov r9 , 0
    //| callq mm->base
    dasm_put (&state, 66, addr, size, flag, (unsigned int) (mm->base),
	      (unsigned int) ((mm->base) >> 32));

    // Return
    //| int 3
    dasm_put (&state, 46);

    int status = dasm_link (&state, &m_code_size);
    if (status != DASM_S_OK)
      {
	std::cerr << "Cannot link generated code!";
	goto fail;
      }

    if (_code)
      delete _code;
    _code = new char[m_code_size];

    dasm_encode (&state, _code);
    dasm_free (&state);

    std::cout << "mem_map code generation finished!";
    return true;

fail: dasm_free (&state);
    return false;
  }

} /* namespace provallo */
;

