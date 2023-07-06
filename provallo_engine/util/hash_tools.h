/*
 * hash_tools.h
 *
 *  Created on: Mar 26, 2021
 *      Author: kardon
 */

#ifndef HASH_TOOLS_H_
#define HASH_TOOLS_H_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <string>
#include <sys/types.h>
namespace provallo
{
  typedef unsigned int uint;
  typedef unsigned char byte;

//derived from the RSA Data Security, Inc. MD5 Message-Digest Algorithm
  class md5sum
  {

    /*
     *	Rotate amounts used in the algorithm
     */
    enum
    {
      S11 = 7, S12 = 12, S13 = 17, S14 = 22,

      S21 = 5, S22 = 9, S23 = 14, S24 = 20,

      S31 = 4, S32 = 11, S33 = 16, S34 = 23,

      S41 = 6, S42 = 10, S43 = 15, S44 = 21
    };

    typedef struct Table
    {
      uint sin; /* integer part of 4294967296 times abs(sin(i)) */
      byte x; /* index into data block */
      byte rot; /* amount to rotate left by */
    } Table;

    typedef struct MD5state
    {
      uint len;
      uint state[4];
    } MD5state;
    MD5state *nil;

    int debug;
    int hex;

    typedef unsigned long ulong;

    static byte t64d[256];
    static char t64e[64];
    static const Table tab[];

  public:

    void
    encode (byte *output, uint *input, uint len);
    void
    decode (byte *output, uint *input, uint len);
    MD5state*
    md5 (byte *p, uint len, byte *digest, MD5state *s);
    void
    sum (FILE*, char*);
    std::string
    sum (byte *data, size_t nDataLen);

  private:

    class initializor
    {
      static bool init;
    public:
      initializor ()
      {
	if (!init)
	  md5sum::init64 ();
	init = true;
      }
      ~initializor ()
      {
      }

    };
    static void
    init64 (void);
    int
    dec64 (byte *out, byte *in, int n);
    int
    enc64 (byte *out, byte *in, int n);
    static initializor init;
  };

}

#endif /* HASH_TOOLS_H_ */
