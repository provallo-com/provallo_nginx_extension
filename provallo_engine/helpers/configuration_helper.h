#ifndef __CONFIGURATION_HELPER_H__
#define __CONFIGURATION_HELPER_H__

#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <dirent.h>

// load nginx configuration and store structure of the different sections, subsections and parameters
// in a map of maps of maps
// the structure is as follows:
// map< section, map< subsection, map< parameter, value > > >
// the map is loaded from a file, and the file is loaded from a string
// each file is loaded into a map, and each map is loaded into a vector of maps
// the vector of maps is loaded into a map of maps of maps
//  files : /etc/nginx/nginx.conf, /etc/nginx/sites-available/default, /etc/nginx/sites-available/default-ssl
//  if config.d is present, load all files in config.d
//  if config.d is not present, load all files in sites-enabled
namespace provallo
{
    namespace nginx_config_helper
    {
        class configuration_helper
        {
            std::map<std::string /*file_name*/, std::map<std::string /*section_name = server*/, std::map<std::string /*subsection name*/, std::map<std::string /*key*/, std::string /*value*/>>>> configuration;

        public:
            // load all the configuration files and maps the values to the configuration maps

            configuration_helper()
            {
                // load all the configuration files and maps the values to the configuration maps
                parse_file("/etc/nginx/nginx.conf");
                parse_file("/etc/nginx/sites-available/default");
                parse_file("/etc/nginx/sites-available/default-ssl");

                // check if config.d directory is present
                DIR *dir = opendir("/etc/nginx/conf.d");
                if (dir)
                {
                    // traverse the directory and load all the files in the directory
                    struct dirent *ent;
                    while ((ent = readdir(dir)) != NULL)
                    {
                        // load all the configuration files and maps the values to the configuration maps
                        if (ent->d_name[0] != '.' && ent->d_type == DT_REG)
                            parse_file("/etc/nginx/conf.d/" + std::string(ent->d_name));
                    }
                    /* Directory exists. */
                    closedir(dir);
                    // load all the configuration files and maps the values to the configuration maps
                    // parse_file("/etc/nginx/conf.d");
                }
                else if (ENOENT == errno)
                {
                    /* Directory does not exist. */
                    // load all the configuration files and maps the values to the configuration maps
                    // parse_file("/etc/nginx/sites-enabled");
                }
                else
                {
                    // ignore
                }

                // check if sites-enabled directory is present
                dir = opendir("/etc/nginx/sites-enabled");
                if (dir)
                {
                    // traverse the directory and load all the files in the directory
                    struct dirent *ent;
                    while ((ent = readdir(dir)) != NULL)
                    {
                        // load all the configuration files and maps the values to the configuration maps
                        if (ent->d_name[0] != '.' && ent->d_type == DT_REG)
                            parse_file("/etc/nginx/sites-enabled/" + std::string(ent->d_name));
                    }
                    /* Directory exists. */
                    closedir(dir);
                    // load all the configuration files and maps the values to the configuration maps
                    // parse_file("/etc/nginx/sites-enabled");
                }
                else if (ENOENT == errno)
                {
                    /* Directory does not exist. */
                    // load all the configuration files and maps the values to the configuration maps
                    // parse_file("/etc/nginx/sites-enabled");
                }
                else
                {
                    // ignore
                }
            }
            void parse_file(std::string file_name)
            {
                std::ifstream file(file_name);
                std::string file_content;
                std::string line;
                if (file.is_open())
                {
                    while (std::getline(file, line))
                    {
                        file_content += line + "\n";
                    }
                    file.close();
                }
                else
                {
                    std::cout << "Unable to open file " << file_name << std::endl;
                }
                parse_sections(file_name, file_content);
            }
            void parse_section(std::string file_name, std::string section_name, std::string section_content)
            {
                // parse generic non sectioned parameters , key values as defualt :
                // i.e. user www-data;
                //      worker_processes N;
                //      pid /run/nginx.pid;
                //      include config file;
                // section names event, http, mail, server, upstream , location, if, limit_except , map , split_clients ,    types , variables , geo , server_names_hash_bucket_size

                // parse generic non sectioned parameters , key values as defualt :
                std::string subsection_name = "default";
                std::string parameter_name = "default";
                std::string parameter_value = "default";
                std::string line;
                std::istringstream iss(section_content);
                //if { is found, call parse_section again with the content of the section until the closing } 
                //if another { is found, call parse_section again with the content of the section  until the closing } 
                //each sectin may have subsections and parameters:
                //i.e. server {
                //          listen 80;
                //          server_name example.com;
                //          location / {
                //              root /var/www/example.com;....

                while (std::getline(iss, line))
                {
                    //if line starts with #, ignore 
                    //if line starts with section name and contains { parse section read until closing } and call parse_section with each section
                    if (line.length() || line.find_first_of('#') != std::string::npos)
                    {
                        //ignore
                    }
                    else if (line.find_first_of('{') != std::string::npos)
                    {
                        //parse section
                        subsection_name = line.substr(0, line.find_first_of('{'));
                        parameter_value = line.substr(line.find_first_of('{') + 1);
                        parse_section(file_name, section_name, parameter_value);
                    }
                    else if (line.find_first_of('}') != std::string::npos)
                    {
                        //ignore
                    }
                    else
                    {
                        //parse section
                        parameter_name = line.substr(0, line.find_first_of('{'));
                        parameter_value = line.substr(line.find_first_of('{') + 1);
                        configuration[file_name][section_name][subsection_name][parameter_name] = parameter_value;
                    }
                }              // end while
                // parse section
                

            }             // end parse_section

            void parse_sections(std::string file_name, std::string file_content)
            {
                // parse generic non sectioned parameters , key values as defualt :
                // i.e. user www-data;
                //      worker_processes N;
                //      pid /run/nginx.pid;
                //      include config file;
                // section names event, http, mail, server, upstream , location, if, limit_except , map , split_clients ,    types , variables , geo , server_names_hash_bucket_size

                // parse generic non sectioned parameters , key values as defualt :
                std::string section_name = "default";
                std::string subsection_name = "default";
                std::string parameter_name = "default";
                std::string parameter_value = "default";
                std::string line;
                std::string section_value;
                std::istringstream iss(file_content);
                while (std::getline(iss, line))
                {
                   //if line starts with #, ignore 
                   //if line starts with section name and contains { parse section read until closing } and call parse_section with each section
                   if(line.length()||line.find_first_of('#')!=std::string::npos ) 
                   {
                          //ignore
                     }
                     else if(line.find_first_of('{')!=std::string::npos)
                     {
                          //parse section
                          section_name = line.substr(0,line.find_first_of('{'));
                          section_value = line.substr(line.find_first_of('{')+1);
                          parse_section(file_name,section_name,section_value);
                     }
                     else if(line.find_first_of('}')!=std::string::npos)
                     {
                          //ignore
                     }
                     else
                     {
                          //parse section
                          section_name = line.substr(0,line.find_first_of('{'));
                          section_value = line.substr(line.find_first_of('{')+1);
                          parse_section(file_name,section_name,section_value);

                   }
                }
            }                 // end parse_sections

            // getters
            std::map<std::string /*file_name*/, std::map<std::string /*section_name = server*/, std::map<std::string /*subsection name*/, std::map<std::string /*key*/, std::string /*value*/>>>> get_configuration() { return configuration; }
            std::map<std::string /*section_name = server*/, std::map<std::string /*subsection name*/, std::map<std::string /*key*/, std::string /*value*/>>> get_file(std::string file_name) { return configuration[file_name]; }
            std::map<std::string /*subsection name*/, std::map<std::string /*key*/, std::string /*value*/>> get_section(std::string file_name, std::string section_name) { return configuration[file_name][section_name]; }
            std::map<std::string /*key*/, std::string /*value*/> get_subsection(std::string file_name, std::string section_name, std::string subsection_name) { return configuration[file_name][section_name][subsection_name]; }
            std::string get_parameter(std::string file_name, std::string section_name, std::string subsection_name, std::string parameter_name) { return configuration[file_name][section_name][subsection_name][parameter_name]; }
            void set_configuration(std::map<std::string /*file_name*/, std::map<std::string /*section_name = server*/, std::map<std::string /*subsection name*/, std::map<std::string /*key*/, std::string /*value*/>>>> configuration) { this->configuration = configuration; }

            void dump() const
            {
                // dump the configuration
                std::cout << "[D]umping configuration :" << std::endl;
                for (auto file : configuration)
                {
                    std::cout << "\t[F]ile : " << file.first << std::endl;
                    for (auto section : file.second)
                    {
                        std::cout << "\t[S]ection : " << section.first << std::endl;
                        for (auto subsection : section.second)
                        {
                            std::cout << "\t[s]ubsection : " << subsection.first << std::endl;
                            for (auto parameter : subsection.second)
                            {
                                std::cout << "[P]arameter : " << parameter.first << " " << parameter.second << std::endl;
                            } // end for
                        }     // end for
                    }         // end for
                }             // end for

            } // end dump
        };
    } // namespace nginx_config_helper
} // namespace provallo
// end of file
#endif