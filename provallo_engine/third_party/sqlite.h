/*
 * sqlite.h
 *
 *
 *  Based on Daniel Beer's sqlite wrapper code
 *  license available here :
 *  https://dlbeer.co.nz/downloads/sqlite.hpp
 *  replaced std::swap with simple swap and added mutex .
 *
 *  Created on: May 6, 2021
 *      Author: kardon
 */

#ifndef THIRD_PARTY_SQLITE_H_
#define THIRD_PARTY_SQLITE_H_

#include <exception>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <string>
#include <sqlite3.h>
//Yaniv : Added thread safety
#include <mutex>
#include <thread>
#include <condition_variable>

namespace io
{
  namespace sqlite
  {

    typedef void
    (*base_func) (sqlite3_context*, int, sqlite3_value**);

    typedef void
    (*step_func) (sqlite3_context*, int, sqlite3_value**);
    typedef void
    (*final_func) (sqlite3_context*);
    typedef void
    (*value_func) (sqlite3_context*);
    typedef void
    (*inverse_func) (sqlite3_context*, int, sqlite3_value**);
    typedef void
    (*destructor_func) (void*);

//sqlite function support for sqlite3_create_window_function/sqlite3_create_function_v2/etc..

    typedef struct func_support_tag
    {
      base_func func;
      step_func step;
      final_func fin;
      value_func val;
      inverse_func inv;
      destructor_func destroy;

    } func_support;

// Exception thrown on error
    class error : public std::exception
    {
    public:
      error (int c) :
	  _code (c)
      {
      }
      virtual
      ~error () throw ();

      const char*
      what () const throw ();

      int
      code () const
      {
	return _code;
      }

    private:
      int _code;
    };

// Check return code, throw error if not ok
    namespace impl
    {

      static inline void
      check (int c)
      {
	if (c != SQLITE_OK)
	  throw error (c);
      }

      void
      destroy_blob (void *blob);
      void
      destroy_text (void *blob);

    }

// Database handle
    class db
    {
		
    public:
      db () :
	  _db (nullptr), _file ("")
      {
      }

      db (const std::string &filename) :
	  _file (filename)
      {
	std::lock_guard<std::recursive_mutex> l_ (_lock);

	impl::check (::sqlite3_open (filename.c_str (), &_db));
      }

      db (const char *filename) :
	  _file (filename)
      {
	std::lock_guard<std::recursive_mutex> l_ (_lock);

	impl::check (::sqlite3_open (filename, &_db));
      }

      const std::string&
      file_name () const
      {
		return _file;
      }


      virtual ~db ()
      {
		std::lock_guard<std::recursive_mutex> l_ (_lock);
		if (_db)
	  ::sqlite3_close (_db);
      }

      db (const db&) = delete;
      db&
      operator= (const db&) = delete;

      void
      swap (db &r)
      {
	std::lock_guard<std::recursive_mutex> l_ (_lock);
	std::lock_guard<std::recursive_mutex> r_ (r._lock);

	sqlite3 *tmp = r._db;
	r._db = _db;
	_db = tmp;

      }

      db (db &&r) :
	  _db (r._db)
      {
	std::lock_guard<std::recursive_mutex> l_ (_lock);
	std::lock_guard<std::recursive_mutex> r_ (r._lock);

	r._db = nullptr;
      }

      db&
      operator= (db &&r)
      {
			std::lock_guard<std::recursive_mutex> l_ (_lock);
			std::lock_guard<std::recursive_mutex> r_ (r._lock);

			db m (std::move (r));
 			swap (m);
			return *this;
      }

      ::sqlite3*
      get ()
      {
		std::lock_guard<std::recursive_mutex> l_ (_lock);

		return _db;
      }

      const ::sqlite3*
      get () const
      {
	std::lock_guard<std::recursive_mutex> l_ (_lock);

	return _db;
      }

      // Number of changes due to the most recent statement.
      unsigned int
      changes () const
      {
	std::lock_guard<std::recursive_mutex> l_ (_lock);

	return ::sqlite3_changes (_db);
      }

      // Execute a simple statement
      void
      exec (const std::string &text)
      {
	std::lock_guard<std::recursive_mutex> l_ (_lock);

	impl::check (
	    ::sqlite3_exec (_db, text.c_str (), nullptr, nullptr, nullptr));
      }

      void
      exec (const char *text)
      {
	std::lock_guard<std::recursive_mutex> l_ (_lock);

	impl::check (::sqlite3_exec (_db, text, nullptr, nullptr, nullptr));
      }
      void
      define_operation (std::string name, int nArg, int eTextRep, void *context,
			func_support &fun)
      {
	std::lock_guard<std::recursive_mutex> l_ (_lock);

		impl::check (
	    sqlite3_create_function (_db, name.c_str (), nArg, eTextRep,
				     context, fun.func, fun.step, fun.fin));
      }
      uint64_t
      last_row_id ()
      {
		std::lock_guard<std::recursive_mutex> l_ (_lock);

		return ::sqlite3_last_insert_rowid (_db);
      }

    private:
      ::sqlite3 *_db;
      std::string _file;
      mutable std::recursive_mutex _lock;
    };

	// Statement
    class stmt
    {
		//
    public:
      stmt () :
	  _stmt (nullptr)
      {
      }

      stmt (db &db, const char *sql)
      {
	impl::check (
	    ::sqlite3_prepare_v2 (db.get (), sql, -1, &_stmt, nullptr));
      }

	  stmt(db* db, const char* sql)
	  {
		  impl::check(
			  ::sqlite3_prepare_v2(db->get(), sql, -1, &_stmt, nullptr));

	  }


      ~stmt ()
      {
	if (_stmt)
	  ::sqlite3_finalize (_stmt);
      }

      stmt (const stmt&) = delete;
      stmt&
      operator= (const stmt&) = delete;

      void
      swap (stmt &r)
      {
	sqlite3_stmt *tmp = _stmt;
	this->_stmt = r._stmt;
	r._stmt = tmp;
      }

      stmt (stmt &&r) :
	  _stmt (r._stmt)
      {
	r._stmt = nullptr;
      }

      stmt&
      operator= (stmt &&r)
      {
	stmt m (std::move (r));

	swap (m);
	return *this;
      }
      std::string
      statement_text ()
      {
	std::string ret = "";
	if (_stmt)
	  {
	    ret = sqlite3_expanded_sql (_stmt);
	  }
	return ret;
      }

      ::sqlite3_stmt*
      get ()
      {
	return _stmt;
      }

      const ::sqlite3_stmt*
      get () const
      {
	return _stmt;
      }

      class binder
      {
      public:
	binder (stmt &s) :
	    _stmt (s._stmt)
	{
	}

	binder&
	blob (unsigned int i, const void *data, size_t len)
	{
	  uint8_t *copy = new uint8_t[len];

	  ::memcpy (copy, data, len);
	  impl::check (
	      ::sqlite3_bind_blob (_stmt, i, copy, len, impl::destroy_blob));
	  return *this;
	}

	binder&
	blob_ref (unsigned int i, const void *data, size_t len)
	{
	  impl::check (::sqlite3_bind_blob (_stmt, i, data, len, nullptr));
	  return *this;
	}

	binder&
	real (unsigned int i, double value)
	{
	  impl::check (::sqlite3_bind_double (_stmt, i, value));
	  return *this;
	}

	binder&
	int32 (unsigned int i, int32_t value)
	{
	  impl::check (::sqlite3_bind_int (_stmt, i, value));
	  return *this;
	}

	binder&
	int64 (unsigned int i, int64_t value)
	{
	  impl::check (::sqlite3_bind_int64 (_stmt, i, value));
	  return *this;
	}

	binder&
	null (unsigned int i)
	{
	  impl::check (::sqlite3_bind_null (_stmt, i));
	  return *this;
	}

	binder&
	text (unsigned int i, const char *orig)
	{
	  const size_t len = ::strlen (orig);
	  char *copy = new char[len];

	  ::memcpy (copy, orig, len);
	  impl::check (
	      ::sqlite3_bind_text (_stmt, i, copy, len, impl::destroy_text));
	  return *this;
	}

	binder&
	text (unsigned int i, const std::string &value)
	{
	  const char *orig = value.c_str ();
	  const size_t len = value.size ();
	  char *copy = new char[len];

	  ::memcpy (copy, orig, len);
	  impl::check (
	      ::sqlite3_bind_text (_stmt, i, copy, len, impl::destroy_text));
	  return *this;
	}

	binder&
	text_ref (unsigned int i, const std::string &value)
	{
	  impl::check (
	      ::sqlite3_bind_text (_stmt, i, value.c_str (), value.size (),
				   nullptr));
	  return *this;
	}

	binder&
	text_ref (unsigned int i, const char *value)
	{
	  impl::check (::sqlite3_bind_text (_stmt, i, value, -1, nullptr));
	  return *this;
	}

	void
	clear ()
	{
	  impl::check (::sqlite3_clear_bindings (_stmt));
	}

      private:
	::sqlite3_stmt *_stmt;
      };

      binder
      bind ()
      {
	return binder (*this);
      }
	  size_t column_count()const
	  {
		  return ::sqlite3_column_count(_stmt);
	  }
      bool
      step ()
      {
	const int c = ::sqlite3_step (_stmt);

	if (c == SQLITE_ROW)
	  return true;

	if (c == SQLITE_DONE)
	  return false;

	throw error (c);
      }

      void
      exec ()
      {
	while (step ())
	  ;
      }

      void
      reset ()
      {
	impl::check (::sqlite3_reset (_stmt));
      }

      class reader
      {
      public:
	reader (stmt &s) :
	    _stmt (s._stmt)
	{
	}

	const void*
	blob (unsigned int i)
	{
	  return ::sqlite3_column_blob (_stmt, i);
	}

	size_t
	size (unsigned int i)
	{
	  return ::sqlite3_column_bytes (_stmt, i);
	}

	double
	real (unsigned int i)
	{
	  return ::sqlite3_column_double (_stmt, i);
	}

	int32_t
	int32 (unsigned int i)
	{
	  return ::sqlite3_column_int (_stmt, i);
	}

	int64_t
	int64 (unsigned int i)
	{
	  return ::sqlite3_column_int64 (_stmt, i);
	}

	const char*
	cstr (unsigned int i)
	{
	  return reinterpret_cast<const char*> (::sqlite3_column_text (_stmt, i));
	}

	std::string
	text (unsigned int i)
	{
	  return std::string (cstr (i), size (i));
	}
      private:
	::sqlite3_stmt *_stmt;
      };

      reader
      row ()
      {
		return reader (*this);
      }


	  
    private:
      ::sqlite3_stmt *_stmt;

      mutable std::recursive_mutex _lock;

    };

  }
}

#endif /* THIRD_PARTY_SQLITE_H_ */
