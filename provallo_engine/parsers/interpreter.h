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

  class interpreter
  {
  public:
    explicit
    interpreter ();
    bool
    interpret (void *offset, size_t len); 
    virtual
    ~interpreter ();
  };

} /* namespace provallo */

#endif /* PARSERS_INTERPRETER_H_ */
