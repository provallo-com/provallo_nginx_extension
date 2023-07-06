/*
 * linuxglue.h
 *
 *  Created on: Jan 21, 2022
 *      Author: kardon
 */

#ifndef GLUE_LINUXGLUE_H_
#define GLUE_LINUXGLUE_H_

#include <string>
#include <vector>
#include <array>
#include <sys/mman.h>
#include "glueprocessinfo.h"
namespace provallo
{
  static const size_t kWordSize = sizeof(uintptr_t);

// implements dynamic memory hook :
// based on this code : https://github.com/dianpeng/dynhook
// removed boost and lua dependencies

  class stub
  {
  public:
    virtual void*
    code () const = 0;
    virtual size_t
    size () const = 0;
    virtual size_t
    rip_offset () const = 0;
    virtual
    ~stub ()
    {
    }
  };

  bool
  invoke (glue_process_info*, const stub &code, uintptr_t r9, uintptr_t *ret);

  class mem_map : public stub
  {
  public:
    static mem_map*
    create (const glue_process_info &info, size_t size, uintptr_t addr,
	    int flag)
    {
      mem_map *ptr = new mem_map ();
      if (ptr && !ptr->init (info, size, addr, flag))
	{
	  delete ptr;
	  return nullptr;
	}
      return ptr;
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
      return 0;
    }

    size_t
    alloc_size () const
    {
      return m_alloc_size;
    }

    uintptr_t
    addr () const
    {
      return m_addr;
    }

    int
    flag () const
    {
      return m_flag;
    }

    void
    dump (std::ostream&);

  private:
    bool
    init (const glue_process_info&, size_t size, uintptr_t addr, int flag);

  private:
    mem_map () :
	stub (), _code (nullptr), m_code_size (0), m_alloc_size (0), m_addr (0), m_flag (
	    0)
    {
    }

    void *_code;
    size_t m_code_size;
    size_t m_alloc_size;
    uintptr_t m_addr;
    int m_flag;
  };

  class mem_unmap : public stub
  {
  public:
    static mem_unmap*
    create (const glue_process_info &proc, uintptr_t addr, size_t cap)
    {
      mem_unmap *ret = new mem_unmap ();
      if (ret && !ret->init (proc, addr, cap))
	{
	  delete ret;
	  return NULL;
	}
      return ret;
    }

    virtual void*
    code () const
    {
      return m_code;
    }

    virtual size_t
    size () const
    {
      return m_code_size;
    }

    virtual size_t
    rip_offset () const
    {
      return 0;
    }

    uintptr_t
    mem_addr () const
    {
      return m_addr;
    }

    size_t
    mem_size () const
    {
      return m_size;
    }

    virtual void
    dump (std::ostream&);

  private:
    bool
    init (const glue_process_info&, uintptr_t addr, size_t size);

  private:
    mem_unmap () :
	stub (), m_code (), m_code_size (0), m_addr (0), m_size (0)
    {
    }

    void *m_code;
    size_t m_code_size;
    uintptr_t m_addr;
    size_t m_size;
  };

  //todo: move to utils :
  template<typename T, typename U>
    inline T
    alignment (T value, U target)
    {
      return (value + target - 1) & ~(target - 1);
    }

  class glue
  {
  public:
    glue ()
    {
    }
    virtual bool
    hook ()=0;

    virtual bool
    unhook ()=0;
    virtual
    ~glue ()
    {
    }
  };
  class linux_glue : virtual public glue
  {
  public:
    class remote_allocator
    {
    public:

      remote_allocator (glue_process_info *pinfo);
      ~remote_allocator ();

      bool
      init ();
      uintptr_t
      allocate (size_t addr_size, uintptr_t hint = 0);
      size_t
      size () const;
      size_t
      capacity () const;
    private:
      class pool
      {
      public:
	static const size_t kDefaultCapacity = 4096;
	static const uintptr_t kLowHint = 0x400000;
	static const uintptr_t kHighHint = 0x7f0000000000U;

	enum
	{
	  HIGH, LOW
	};

	pool (glue_process_info *pinfo, int type) :
	    m_pinfo (pinfo), m_size (0), m_capacity (0), m_start (0), m_addr (
		0), m_flag (0)
	{
	  if (type == HIGH)
	    {
	      m_flag = MAP_ANONYMOUS | MAP_PRIVATE;
	      m_addr = kHighHint;
	    }
	  else
	    {
	      m_flag = MAP_ANONYMOUS | MAP_PRIVATE | MAP_32BIT;
	      m_addr = kLowHint;
	    }
	}

	bool
	init ()
	{
	  if (!grow (0))
	    {
	      std::cerr << "Cannot initialize memory pool with hint:"
		  << std::hex << m_addr << std::dec;
	      return false;
	    }
	  return true;
	}

	uintptr_t
	allocate (size_t cap)
	{
	  cap = alignment (cap, 8);
	  if (m_size + cap > m_capacity)
	    {
	      if (!grow (cap))
		return 0;
	    }
	  // assert(m_size + cap < m_capacity);
	  if (m_size + cap < m_capacity)
	    return 0;

	  uintptr_t ret = m_start + m_size;
	  m_size += cap;
	  return ret;
	}

	size_t
	size () const
	{
	  return m_size;
	}

	size_t
	capacity () const
	{
	  return m_capacity;
	}

      private:
	bool
	grow (size_t gaurantee)
	{
	  const size_t cap = (
	      m_capacity == 0 ? kDefaultCapacity : (m_capacity * 2 + gaurantee));

	  stub *mmap (mem_map::create (*m_pinfo, cap, m_addr, m_flag));
	  if (!mmap)
	    return false;
	  uintptr_t ret;
	  if (!invoke (m_pinfo, *mmap, 0, &ret))
	    return false;

	  if (!ret && m_flag & MAP_32BIT)
	    {
	      m_flag &= ~MAP_32BIT;
	      if (mmap)
		delete mmap;
	      mmap = nullptr;

	      mmap = mem_map::create (*m_pinfo, cap, m_addr, m_flag);
	      if (!invoke (m_pinfo, *mmap, 0, &ret))
		return false;
	    }

	  if (ret)
	    {
	      m_size = 0;
	      m_capacity = cap;
	      m_start = ret;
	      return true;
	    }
	  else
	    {
	      std::cout << "Cannot allocate memory from remote process!";
	      return false;
	    }
	}

      private:
	glue_process_info *m_pinfo;
	size_t m_size;
	size_t m_capacity;
	uintptr_t m_start;
	uintptr_t m_addr;
	int m_flag;

	struct segment
	{
	  segment (uintptr_t addr, size_t cap) :
	      address (addr), capacity (cap)
	  {
	  }
	  uintptr_t address;
	  size_t capacity;
	};
      };

      class pool *_low_pool;
      class pool *_high_pool;
    };
    std::string _process_name;
  public:
    explicit
    linux_glue (const std::string &proc_name);
    virtual bool
    hook ();
    std::vector<long>
    pids ();

    static bool
    ptrace_peek (pid_t pid, uintptr_t address, uintptr_t *ret);
    static bool
    ptrace_poke (pid_t pid, uintptr_t address, uintptr_t value);
    static bool
    ptrace_getregs (pid_t pid, struct user_regs_struct *output);
    static bool
    ptrace_setregs (pid_t pid, const struct user_regs_struct &output);
    static bool
    ptrace_continue (pid_t pid);
    static bool
    ptrace_signal (pid_t pid, int sig);
    static bool
    ptrace_cont_and_wait_event (pid_t pid, int *status);
    static bool
    ptrace_attach (pid_t pid);
    static bool
    ptrace_attach_and_wait (pid_t pid, int *status);

    virtual bool
    unhook ();
    virtual
    ~linux_glue ();

  };
  using namespace std;
  class int64_array
  {
  public:
    int64_array (uintptr_t value) :
	m_arr ()
    {
      const char *ch = (const char*) &value;

      for (size_t i = 0; i < sizeof(uintptr_t); ++i)
	{
	  m_arr[i] = ch[i];
	}
    }

    char
    operator [] (int idx) const
    {
      return m_arr[idx];
    }

    char&
    operator [] (int idx)
    {
      return m_arr[idx];
    }

    uintptr_t
    to_int64 () const
    {
      uintptr_t n = 0;
      n = (((uintptr_t) m_arr[7] << 56) & 0xFF00000000000000U)
	  | (((uintptr_t) m_arr[6] << 48) & 0x00FF000000000000U)
	  | (((uintptr_t) m_arr[5] << 40) & 0x0000FF0000000000U)
	  | (((uintptr_t) m_arr[4] << 32) & 0x000000FF00000000U)
	  | ((m_arr[3] << 24) & 0x00000000FF000000U)
	  | ((m_arr[2] << 16) & 0x0000000000FF0000U)
	  | ((m_arr[1] << 8) & 0x000000000000FF00U)
	  | (m_arr[0] & 0x00000000000000FFU);
      return n;
    }

  private:
    std::array<char, sizeof(uintptr_t)> m_arr;
  };

} /* namespace provallo */

#endif /* GLUE_LINUXGLUE_H_ */
