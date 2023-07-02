/*
 * turnserverglue.h
 *
 *  Created on: Jan 20, 2022
 *      Author: kardon
 */

#ifndef GLUE_TURNSERVERGLUE_H_
#define GLUE_TURNSERVERGLUE_H_
#include "linuxglue.h"
#include <vector>
#include <string>
namespace provallo
{

  const std::string run_pid_file = "/run/turnserver/turnserver.pid";

  class turnserver_glue : linux_glue
  {

    typedef void
    (*generate_aes_128_key) (char *filePath, unsigned char *returnedKey);
    typedef int
    (*decodedTextSize) (char *input);
    typedef char*
    (*decryptPassword) (char *in, const unsigned char *mykey);
    typedef unsigned char*
    (*base64decode) (const void *b64_decode_this, int decode_this_many_bytes);
    typedef void
    (*encrypt_aes_128) (unsigned char *in, const unsigned char *mykey);

    typedef struct hook_ptr_tag
    {
      void *origin;
      void *hooked;
    } *hook_ptr;
    std::vector<std::string> glue_names;
  public:
    turnserver_glue ();
    virtual
    ~turnserver_glue ();
    static void
    hooked_generate_aes_128_key (char *file, unsigned char *ret_key);
    static int
    hooked_decodedTextSize (char *in);
    static char*
    hooked_decryptPassword (char *in, unsigned char *ret_key);
    static void
    hooked_encrypt_aes_128 (unsigned char *in, const unsigned char *mykey);
    static unsigned char*
    hooked_base64decode (const void *b64_decode_this,
			 int decode_this_many_bytes);
    std::vector<glue_process_info> hook_processses;
    virtual bool
    hook ();
    static void *context; //
  };

} /* namespace provallo */

#endif /* GLUE_TURNSERVERGLUE_H_ */
