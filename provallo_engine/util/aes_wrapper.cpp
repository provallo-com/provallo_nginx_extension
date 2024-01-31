#include "../util/aes_wrapper.h"


namespace provallo
{

  //utility :
  int aes_init(const unsigned char *key_data, int key_data_len, unsigned char *salt, EVP_CIPHER_CTX *e_ctx,
		EVP_CIPHER_CTX *d_ctx);
  unsigned char *aes_encrypt(EVP_CIPHER_CTX *e, unsigned char *plaintext, int *len);
  unsigned char *aes_decrypt(EVP_CIPHER_CTX *e, unsigned char *ciphertext, int *len);


  aes_ifile::aes_ifile(const char *filename, const char *key_data)
  : _begin(0), _end(0), _current(0), _data(0), _key_data(key_data),_good(true) {
  	// Try to open the file
  	int fd = -1;
  	if ((fd = open(filename, O_CREAT | O_RDONLY, 0600)) == -1) {
  		_good = false;
  		return;
  	}
  	// Initialize AES
  	unsigned int salt[] = { 0xF9090909, 0x090909F9 };
  	if (aes_init((const unsigned char*)(_key_data.c_str()), _key_data.size(),
  			(unsigned char*)(&salt), en_(), de_())) {
  		_good = false;	
  		return;
  	}
  	// Get the number of bytes inside the file
  	struct stat st;
  	stat(filename, &st);
  	int size = st.st_size;

  	// Allocate data
  	char* ciphertext = new char[size];
  	int bytes_read = read(fd, (void *)(ciphertext), size);
  	if(bytes_read != size) {
  		_good = false;
		//deallocation
		delete [] ciphertext;
  		return;
  	}
   	// Get plain text
  	_data = (const char *)(aes_decrypt(de_(), (unsigned char *)(ciphertext), &size));
  	// Setup the stream pointers 

	if (_data == nullptr)
	{
		_good = false;
		delete [] ciphertext;

		return;
	}
  	_begin = _data;
  	_end = _data + size;
  	_current = _data;

   
	_good = true;
  	// Delete cipher text
  	delete [] ciphertext;
	// Close the file
  	close(fd);
  }

  aes_ifile::int_type aes_ifile::underflow() {
  	if (_current == _end) {
  		return traits_type::eof();
  	}
  	return traits_type::to_int_type(*_current);
  }

  aes_ifile::int_type aes_ifile::uflow() {
  	if (_current == _end) {
  		return traits_type::eof();
  	}
  	return traits_type::to_int_type(*_current++);
  }

  aes_ifile::int_type aes_ifile::pbackfail(int_type ch) {
  	if (_current == _begin || (ch != traits_type::eof() && ch != _current[-1])) {
  		return traits_type::eof();
  	}
  	return traits_type::to_int_type(*--_current);
  }

  ptrdiff_t  aes_ifile::decrypt_buffer(ptrdiff_t buffer,ssize_t& len,const std::string& key)
  {
 	unsigned int salt[] = { 0xF9090909, 0x090909F9 };
	auto_evp_cipher_ctx en, de;

	ptrdiff_t ret = ptrdiff_t(nullptr);
	if (aes_init((const unsigned char*)(key.c_str()), key.size(),
  			(unsigned char*)(&salt), en(), de())) {
   		return ret;
  	}
	ret = ptrdiff_t( aes_decrypt(de(), (unsigned char *)(buffer), (int*)&len));
 	return ret;

  }

  std::streamsize aes_ifile::showmanyc() {
  	return _end - _current;
  }

  aes_ifile::~aes_ifile() {
  	delete [] _data;
  }

  aes_ofile::aes_ofile(const char *filename, const char *key_data, std::size_t buff_sz) :
  		_filename(filename), _good(true), _data(buff_sz), _size_of_buffer(buff_sz),
  		_overflow_cnt(0), _key_data(key_data) {
  	//
  	unsigned int salt[] = { 0xF9090909, 0x090909F9 };
  	if (aes_init((const unsigned char*)(_key_data.c_str()), _key_data.size(),
  			(unsigned char*)(&salt), en_(), de_())) {
  		_good = false;
  		return;
  	}
      char *base = &_data.front();
      setp(base, base + _data.size());
  }
  ptrdiff_t aes_ofile::encrypt_buffer(ptrdiff_t buffer,ssize_t& len,const std::string& key)
  {
	unsigned int salt[] = { 0xF9090909, 0x090909F9 };
	auto_evp_cipher_ctx en, de;

	ptrdiff_t ret = ptrdiff_t (nullptr);
	if (aes_init((const unsigned char*)(key.c_str()), key.size(),
      			(unsigned char*)(&salt), en(), de())) {
       		return ret;
      	}
  	ret = ptrdiff_t(aes_encrypt(en(), (unsigned char *)(buffer), (int*)&len));


  	return ret;
  }

  bool aes_ofile::encrypt_and_flush() {
   	int fd;
  	if ((fd = open(_filename.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600)) == -1) {
  		_good = false;
  		return false;
  	}

  	// encrypt
  	unsigned char* ucdata = (unsigned char* )&_data.front();
  	int nlen = (pptr() - pbase()) + _overflow_cnt * _size_of_buffer;
  	unsigned char *ciphertext = aes_encrypt(en_(), (unsigned char *) ucdata, &nlen);
  	if (ciphertext)
  	{
            
   		ssize_t wr_size = 	write(fd, ciphertext, nlen);
  		struct stat statinfo;
  		if (fstat(fd, &statinfo) == -1) {
  		  _good = false;
  			delete []ciphertext;
  			return false;
  		}
		
		if (wr_size<nlen-1)
		{
			delete []ciphertext;
			return false;
		}

  	}

  	// Close file
  	close(fd);
  	// Delete buffer
  	delete [] ciphertext;

  	// All good...
  	return true;
  }

  aes_ofile::int_type aes_ofile::overflow(int_type ch) {
  	if(ch != traits_type::eof()) {

  	        // Resize the buffer
  		std::size_t current_size = _data.size();
  		_data.resize(current_size + _size_of_buffer);
  		++_overflow_cnt;

  	    char *base = &(*(_data.begin() + current_size));
  	    setp(base, base + _size_of_buffer);
  	    //mark eof
  	    *pptr() = ch;

  	    pbump(1);
   	    return ch;
  	}
  	return traits_type::eof();
  }

  int aes_ofile::sync() {
  	return encrypt_and_flush() ? 0 : -1;
  }




  int aes_init(const unsigned char *key_data, int key_data_len, unsigned char *salt, EVP_CIPHER_CTX *e_ctx,
		EVP_CIPHER_CTX *d_ctx)
{
	int i, nrounds = 5;
	unsigned char key[32], iv[32];

	i = EVP_BytesToKey(EVP_aes_256_cbc(), EVP_sha1(), salt, key_data, key_data_len, nrounds, key, iv);
	if (i != 32)
	{
		return -1;
	}

	EVP_CIPHER_CTX_init(e_ctx);
	EVP_EncryptInit_ex(e_ctx, EVP_aes_256_cbc(), NULL, key, iv);
	EVP_CIPHER_CTX_init(d_ctx);
	EVP_DecryptInit_ex(d_ctx, EVP_aes_256_cbc(), NULL, key, iv);

	return 0;
}

  unsigned char *aes_encrypt(EVP_CIPHER_CTX *e, unsigned char *plaintext, int *len)
{
	// Max cipher text length for a n bytes of plain text is n + AES_BLOCK_SIZE -1 bytes
	int c_len = *len + AES_BLOCK_SIZE, f_len = 0;
	unsigned char *ciphertext = new unsigned char[c_len];

	// Allows reusing of 'e' for multiple encryption cycles
	EVP_EncryptInit_ex(e, NULL, NULL, NULL, NULL);

	// Update cipher text, c_len is filled with the length of cipher text generated,
	// length is the size of plain text in bytes
	EVP_EncryptUpdate(e, ciphertext, &c_len, plaintext, *len);

	// Update cipher text with the final remaining bytes
	EVP_EncryptFinal_ex(e, ciphertext + c_len, &f_len);

	*len = c_len + f_len;
	return ciphertext;
}

  unsigned char *aes_decrypt(EVP_CIPHER_CTX *e, unsigned char *ciphertext, int *len)
{
	// Plain text will always be equal to or lesser than length of cipher text
	int p_len = *len, f_len = 0;
	unsigned char *plaintext = new unsigned char[p_len];

	EVP_DecryptInit_ex(e, NULL, NULL, NULL, NULL);
	EVP_DecryptUpdate(e, plaintext, &p_len, ciphertext, *len);
	EVP_DecryptFinal_ex(e, plaintext + p_len, &f_len);

	*len = p_len + f_len;
	return plaintext;
}


  }
