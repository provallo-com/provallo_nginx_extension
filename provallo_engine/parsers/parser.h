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
#include <cstdint>
#include <iostream>
#include <memory>
#include <cstring>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <cmath>
#include <limits>
#include <functional>
#include <type_traits>
#include <utility>
#include <typeinfo>
#include <typeindex>
#include <cxxabi.h>
#include <sstream>
#include <iomanip>

namespace provallo
{

  class parser;
  struct buffer_can 
  {char *buffer; size_t len;};
  //encoder/decoder visitor pattern
  struct coder
  {
    virtual  uint32_t
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
    

    //information: 
    virtual uint32_t get_id() const =0;
    virtual const std::string get_name() const =0;
    virtual const std::string get_description() const =0; 
    virtual const std::string get_version() const =0;


    //print information
    virtual void print(std::ostream&)const =0;
    //serialize/deserialize  
    virtual void load(std::ifstream&) =0; 
    virtual void save(std::ofstream&)const =0; 
    //encoder/decoder management
    void
    add_encoder (encoder *encoder, bool override_ = true);
    void
    add_decoder (decoder *encoder, bool override_ = true);
    virtual void

    on_encoded (const void *block, size_t len, bool direction = true)=0; 
    virtual void
    on_decoded (const void *block, size_t len, bool direction = false)=0;
    virtual void
    on_parse (const void *block, size_t len, bool direction = false)=0;
    virtual
    ~parser ();

  };
  //specific factory for each parser
  class parser_factory
  {
  public:
    parser_factory ();
    virtual parser*
    create ()=0;
    virtual
    ~parser_factory ();
  };
  //manager for parser factories
  class parser_factory_manager
  {
  public:
    parser_factory_manager ();
    void
    add_factory (parser_factory *factory);
    parser*
    create (uint32_t id);
    virtual
    ~parser_factory_manager (); 
  private:

    std::map<uint32_t, parser_factory*> _factories;
  };
  //manager for parsers
  class parser_manager
  {   
    
  public:
    parser_manager ();
    // add_parser enumerates all encoders/decoders and adds them to the parser manager 

    void
    add_parser (parser *parser);
    //returns a parser from the tuple by id 
    parser*
    get_parser (uint32_t id);

    encoder* get_encoder (uint32_t id);
    decoder* get_decoder (uint32_t id);
    //returns a parser from the tuple by name
    parser*
    get_parser (const std::string &name);
    encoder* get_encoder (const std::string &name);
    decoder* get_decoder (const std::string &name);
    //returns a parser from the tuple by type
    template<typename T>
    parser*
    get_parser ()
    {
      return get_parser (typeid(T).hash_code());
    } 
    template<typename T>
    encoder*
    get_encoder ()
    {
      return get_encoder (typeid(T).hash_code());
    } 
    template<typename T>  
    decoder*  
    get_decoder ()
    {
      return get_decoder (typeid(T).hash_code());
    }   
    
   
    virtual
    ~parser_manager ();   
  private:  
    std::map<std::string,uint32_t> _parser_names_to_ids;
    typedef std::tuple<encoder*,decoder*,parser*> parser_tuple; 
    std::map<uint32_t, parser_tuple> _parser_tuples; 
   };  
  


} /* namespace provallo */

#endif /* PARSERS_PARSER_H_ */
