/*
 * interpreter.h
 *
 *  Created on: Jan 20, 2022
 *      Author: kardon
 */

#ifndef PARSERS_INTERPRETER_H_
#define PARSERS_INTERPRETER_H_

#include "../parsers/parser.h"

namespace provallo
{
  //interpreter pattern with visitor pattern for encoders/decoders 
  //and visitor pattern for parsers 
  class interpreter
  {

    parser_manager* _parser_manager; 
    
  public:
    explicit
    interpreter ();

    virtual bool
    interpret (void *offset, size_t len)=0; 
    
    virtual
    ~interpreter ();
    //interpreter pattern with visitor pattern for encoders/decoders 
    //and visitor pattern for parsers
    
  };

} /* namespace provallo */

#endif /* PARSERS_INTERPRETER_H_ */
