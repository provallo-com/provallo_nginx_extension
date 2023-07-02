/*
 * basic_encoding.h
 *
 *  Created on: May 8, 2023
 *      Author: kardon
 */

#ifndef UTIL_BASIC_ENCODING_H_
#define UTIL_BASIC_ENCODING_H_
#include <string>
std::string
url_encode (const std::string &text);
std::string
base64_encode (unsigned char const *bytes_to_encode, unsigned int in_len);
std::string
base64_decode (std::string const &encoded_string);
#endif /* UTIL_BASIC_ENCODING_H_ */
