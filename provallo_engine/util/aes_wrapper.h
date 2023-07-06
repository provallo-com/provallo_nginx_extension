#ifndef AES_WRAPPER_H_
#define AES_WRAPPER_H_

#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <ctype.h>
#include <cstdlib>
#include <cassert>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <openssl/sha.h>
#include <openssl/evp.h>
#include <openssl/aes.h>

#include <iostream>
#include <streambuf>
#include <iosfwd>
#include <vector>

//AES streambufs

struct auto_evp_cipher_ctx
{
  //EVP CIPHER SILLY PTR
  EVP_CIPHER_CTX * _p;


  auto_evp_cipher_ctx():_p(nullptr){
    _p = EVP_CIPHER_CTX_new();

  }

  EVP_CIPHER_CTX* operator()(){return _p;}
  virtual ~auto_evp_cipher_ctx(){
    EVP_CIPHER_CTX_free(_p);
    _p=nullptr;
  }

};

namespace provallo
{

	class aes_ifile : public std::streambuf {
	public:
		// Initialize from file (all data in the file is read at once)
		explicit aes_ifile(const char *filename, const char *key_data);

		// Check whether state of stream is good
		bool good() {
			return _good;
		}

		static ptrdiff_t decrypt_buffer(ptrdiff_t,ssize_t&,const std::string&);

		~aes_ifile();

	private:
 		aes_ifile(const aes_ifile &);
		aes_ifile &operator= (const aes_ifile &);

		int_type underflow();
		int_type uflow();
		int_type pbackfail(int_type ch);
		std::streamsize showmanyc();

		// data helpers:

		// Flag if file is good or not

		const char * _begin;
		const char * _end;
		const char * _current;

		// Buffer with data
		const char* _data;

		// Encryption data
		auto_evp_cipher_ctx en_, de_;
		std::string _key_data;

		bool _good;
	};

	class aes_ofile : public std::streambuf
	{
	    public:
	        explicit aes_ofile(const char *filename, const char *key_data, std::size_t buff_sz = 256);

 			inline bool good() {
				return _good;
			}


 		static ptrdiff_t encrypt_buffer(ptrdiff_t,ssize_t&,const std::string&);

	    protected:
	        bool encrypt_and_flush();
	    private:
	        int_type overflow(int_type ch);
	        int sync();


	        //private assignment & copy
	        aes_ofile(const aes_ofile &);
	        aes_ofile &operator= (const aes_ofile &);

	        // File name
	        std::string _filename;
 	        bool _good;

			// Buffer with data
			std::vector<char> _data;
			// Buffer size (to allocate new chunks)
			std::size_t _size_of_buffer;
			// Overflow count
			std::size_t _overflow_cnt;

			// Encryption data
			auto_evp_cipher_ctx en_, de_;
			std::string _key_data;
	};


}
#endif
