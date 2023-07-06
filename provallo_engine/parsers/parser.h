/*
 * parser.h
 *
 *  Created on: Jan 20, 2022
 *      Author: kardon
 */

#ifndef PARSERS_PARSER_H_
#define PARSERS_PARSER_H_
#include <map>
#include <variant>
#include <string>
#include <vector>
#include <cstddef>
namespace provallo
{

  class parser;
  struct buffer_can 
  {char *buffer; size_t len;};
  //encoder/decoder visitor pattern
  struct coder
  {
    virtual  uint16_t
    get_id () const =0;
    virtual const std::string
    get_name () const =0;
    virtual
    ~coder ()=0;
  };

  class encoder : virtual public coder
  {
    friend class parser;
  public:
    encoder ();
    bool
    visit (parser &parser,buffer_can &buffer  );
    virtual
    ~encoder ()=0;
  };

  class decoder : virtual public coder
  {
  public:
    decoder ();
    bool
    visit (parser &parser,buffer_can &buffer);
    virtual
    ~decoder ()=0;
  };

  class parser
  {

  public:
    void
    on_decoded_buffer (decoder *&decoder, char *buffer, size_t len);
    void
    on_encoded_buffer (encoder *&decoder, char *buffer, size_t len);

  public:
    std::vector<encoder*> _encoders;
    std::vector<decoder*> _decoders;
    std::vector<ptrdiff_t> _offsets; //data offsets to encode/decode/
    ptrdiff_t _decoded;
    ptrdiff_t _encoded;
    ptrdiff_t _parsed;
    parser ();
    
    parser(const parser &other)
    {
      _encoders = other._encoders;
      _decoders = other._decoders;
      _offsets = other._offsets;
      _decoded = other._decoded;
      _encoded = other._encoded;
      _parsed = other._parsed;
    }
    parser(parser &&other)
    {
      _encoders = std::move(other._encoders);
      _decoders = std::move(other._decoders);
      _offsets = std::move(other._offsets);
      _decoded = std::move(other._decoded);
      _encoded = std::move(other._encoded);
      _parsed = std::move(other._parsed);
    }
    
    void
    add_encoder (encoder *encoder, bool override_ = true);
    void
    add_decoder (decoder *encoder, bool override_ = true);
    virtual void


    parse (const void *block, size_t len, bool direction = false)=0;
    virtual
    ~parser ();
  };

} /* namespace provallo */

#endif /* PARSERS_PARSER_H_ */
