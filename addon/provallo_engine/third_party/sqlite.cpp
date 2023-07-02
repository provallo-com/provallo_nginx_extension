/*
 * sqlite.cpp
 *
 *  Created on: May 6, 2021
 *      Author: kardon
 */
//UYdJMwrCq+O]qeL
//106454407491
//yaniv select * from stun_attribute att join STUNLOG log on log.ts=att.ts where att.id=
#include "../third_party/sqlite.h"

#ifndef IMPL_ONCE_SIMULATOR
#define IMPL_ONCE_SIMULATOR
namespace io
{

  namespace sqlite
  {
    error::~error () throw ()
    {
    }

    const char*
    error::what () const throw ()
    {

      static std::string err;
      err = "sqlite error ";
      err += ::sqlite3_errstr (this->_code);
      return err.c_str ();

    }

    void
    impl::destroy_blob (void *blob)
    {
      delete[] reinterpret_cast<uint8_t*> (blob);
    }

    void
    impl::destroy_text (void *blob)
    {
      delete[] reinterpret_cast<char*> (blob);
    }
  }
}
;
#endif //IMPL_ONCE_SIMULATOR
