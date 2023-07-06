/*
 * parser.cpp
 *
 *  Created on: Jan 20, 2022
 *      Author: kardon
 */

#include "../parsers/parser.h"

namespace provallo
{

  parser::parser ()
  {
    // TODO Auto-generated constructor stub

  }
  // TODO Auto-generated constructor stub
  // TODO Auto-generated destructor stub
  // TODO Auto-generated destructor stub
  parser::~parser ()
  {
    // TODO Auto-generated destructor stub
  }
  void parser::add_decoder (decoder *decoder, bool override)
  { 
      this->_decoders.push_back (decoder); 
  }
  void parser::add_encoder (encoder *encoder, bool ovberride)
  { 
      this->_encoders.push_back (encoder); 
  }   
  void parser::on_decoded_buffer (decoder *&decoder, char *buffer, size_t len)
  { 
    buffer_can buffer_;
    buffer_.buffer = buffer;
    buffer_.len = len;
    for (auto &decoder : this->_decoders)
      { 
          decoder->visit (*this,buffer_);
      } 

    
  }   
  void parser::on_encoded_buffer (encoder *&encoder, char *buffer, size_t len)
  {
    buffer_can buffer_;
    buffer_.buffer = buffer;
    buffer_.len = len;

    for (auto &encoder : this->_encoders)
      { 
          encoder->visit (*this,buffer_);
      }

  }      


  //
  //encoder / decoder visit methods
  //
  bool encoder::visit (parser &parser, buffer_can &buffer)
  {
    //encode parser's buffer
     return false;
  }  
  bool decoder::visit (parser &parser,  buffer_can &buffer)
  {
    //decode parser's buffer
    return false;
  }  
} /* namespace provallo */
