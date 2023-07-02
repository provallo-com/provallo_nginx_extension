/*
 * turnserverglue.cpp
 *
 *  Created on: Jan 20, 2022
 *      Author: kardon
 */
#include "glueprocessinfo.h"
#include "turnserverglue.h"

namespace provallo
{

  turnserver_glue::turnserver_glue () :
      linux_glue ("turnserver"), glue_names (
	{ "generate_aes_128_key", "decodedTextSize", "decryptPassword",
	    "base64decode", "encrypt_aes_128" })
  {
	// TODO Auto-generated constructor stub	
	//glue_process_info in;
	//in.name = "turnserver";
	//in.pid_file = "/run/turnserver/turnserver.pid";	
	//in.attach_all ();		
	//hook_processses.push_back (in);

  }

  turnserver_glue::~turnserver_glue ()
  {
		// TODO Auto-generated destrctor stub		
		

  }
  bool
  turnserver_glue::hook ()
  { 
	// TODO Auto-generated destructor stub
	//hook_processses[0].attach_all ();
	//hook_processses[0].attach_all ();

  }
} /* namespace provallo */
