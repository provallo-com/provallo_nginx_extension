/*
 * procwatcher.cpp
 *
 *  Created on: Jan 17, 2022
 *      Author: kardon
 */

#include "procwatcher.h"
#include <iosfwd>
#include <fstream>

namespace provallo
{


  bool system_process::check(const std::string& pid) const {
  	// Check if truncated name is inside the command line parameters when the process was launched. In
  	// case is false, means that this is a forked process of another one (such as an AsyncTask in Android)
      std::string cmdline = process_desc::get_cmdline(pid);
      std::string comm = process_desc::get_cmname(pid);
      return (cmdline.size() > 0 && comm.size() > 0 && cmdline.find(comm) != std::string::npos);
  }

  bool str_process::check(const std::string& pid) const {
  	bool is_new = system_process::check(pid);
  	if(not is_new) return false;

  	// Nothing to compare
  	if(_procs.size() == 0) return true;

  	// Get name of the process (truncated)
      std::string proc_name = process_desc::get_cmdline(pid);
  	for(size_t i = 0 ; i < _procs.size() ; ++i) {
  		if(proc_name.size() > 0 && proc_name.find(_procs[i]) != std::string::npos) return true;
  	}

  	return false;
  }

  std::string process_desc::get_last_pid() {
      // Open /proc/loadavg file to look for last PID
  	std::ifstream in("/proc/loadavg");

  	// Praes dummy values
      std::string dummy;
      for(size_t i = 0 ; i < 4 ; ++i) {
      	if(in.good()) in >> dummy;
      	else break;
      }

  	// Read the fifth entry on the file
      std::string new_pid;
      if(in.good()) in >> new_pid;

      // Close the file and return the PID
      in.close();
      return new_pid;
  }

  // Function to get process name
  std::string process_desc::get_cmname(const std::string& pid) {
      // Open COMM file
      std::string comm("/proc/" + pid + "/comm");
      std::ifstream in(comm.c_str());

      // Parse data in the file
      std::string data;
      if(in.good()) {
          in >> data;
      }
      in.close();

      // Return parsed data
      return data;
  }

  // Get CMD line of the process
  std::string process_desc::get_cmdline(const std::string& pid) {
  	// Name of the process
  	char name[1024] = {0};
  	std::string cmdline_file = "/proc/" + pid + "/cmdline";
  	FILE* cmdline = fopen(cmdline_file.c_str(), "r");

  	if(cmdline) {
  		fscanf(cmdline, "%1023s", name);
  		fclose(cmdline);
  	}

      // Return parsed data
      return std::string(name);
  }

  // Get maximum value of the process ID
  int process_desc::get_max_pid() {
      std::ifstream in("/proc/sys/kernel/pid_max");
      int max_pid;
      in >> max_pid;
      in.close();
      return max_pid;
  }

} /* namespace provallo */
