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


//load nginx configuration and store structure of the different sections, subsections and parameters 
//in a map of maps of maps  
//the structure is as follows:
//map< section, map< subsection, map< parameter, value > > > 
//the map is loaded from a file, and the file is loaded from a string
//each file is loaded into a map, and each map is loaded into a vector of maps 
//the vector of maps is loaded into a map of maps of maps
// files : /etc/nginx/nginx.conf, /etc/nginx/sites-available/default, /etc/nginx/sites-available/default-ssl 
// if config.d is present, load all files in config.d 
// if config.d is not present, load all files in sites-enabled
namespace provallo {
    namespace nginx_config_helper
    {
        class configuration_helper
        {
            std::map< std::string/*file_name*/, std::map< std::string/*section_name = server*/, std::map< std::string/*subsection name*/, std::map<std::string/*key*/,std::string/*value*/>> > > configuration;
            
        public:
            //load all the configuration files and maps the values to the configuration maps

            configuration_helper()
            {
                //load all the configuration files and maps the values to the configuration maps 
                parse_file("/etc/nginx/nginx.conf");
                parse_file("/etc/nginx/sites-available/default");
                parse_file("/etc/nginx/sites-available/default-ssl");
                
                
                //check if config.d directory is present
                DIR *dir = opendir("/etc/nginx/conf.d");
                if (dir)
                {
                    //traverse the directory and load all the files in the directory 
                    struct dirent *ent;
                    while ((ent = readdir (dir)) != NULL) 
                    {
                        //load all the configuration files and maps the values to the configuration maps 
                        if(ent->d_name[0] != '.' && ent->d_type == DT_REG)
                            parse_file("/etc/nginx/conf.d/" + std::string(ent->d_name));
                    }
                    /* Directory exists. */
                    closedir(dir);
                    //load all the configuration files and maps the values to the configuration maps 
                    //parse_file("/etc/nginx/conf.d");
                }
                else if (ENOENT == errno)
                {
                    /* Directory does not exist. */
                    //load all the configuration files and maps the values to the configuration maps 
                    //parse_file("/etc/nginx/sites-enabled");
                }
                else
                {
                    //ignore    
                }

                //check if sites-enabled directory is present
                dir = opendir("/etc/nginx/sites-enabled");
                if (dir)
                {
                    //traverse the directory and load all the files in the directory 
                    struct dirent *ent;
                    while ((ent = readdir (dir)) != NULL) 
                    {
                        //load all the configuration files and maps the values to the configuration maps 
                        if(ent->d_name[0] != '.' && ent->d_type == DT_REG)
                            parse_file("/etc/nginx/sites-enabled/" + std::string(ent->d_name));
                    }
                    /* Directory exists. */
                    closedir(dir);
                    //load all the configuration files and maps the values to the configuration maps 
                    //parse_file("/etc/nginx/sites-enabled");
                }
                else if (ENOENT == errno)
                {
                    /* Directory does not exist. */
                    //load all the configuration files and maps the values to the configuration maps 
                    //parse_file("/etc/nginx/sites-enabled");
                }
                else
                {
                    //ignore    
                }


             }
            void parse_file(std::string file_name)
            {
                std::ifstream file(file_name);
                std::string file_content;
                std::string line;
                if (file.is_open())
                {
                    while ( std::getline (file,line) )
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
            void parse_sections(std::string file_name, std::string file_content)
            {
               //parse generic non sectioned parameters , key values as defualt : 
               //i.e. user www-data; 
               //     worker_processes N;
                //     pid /run/nginx.pid;
                //     include config file;
                //section names event, http, mail, server, upstream , location, if, limit_except , map , split_clients ,    types , variables , geo , server_names_hash_bucket_size 
                
                //parse generic non sectioned parameters , key values as defualt :
                std::string section_name = "default";
                std::string subsection_name = "default";
                std::string parameter_name = "default";
                std::string parameter_value = "default";
                std::string line;
                std::istringstream iss(file_content);
                while ( std::getline (iss,line) )
                {
                    //std::cout << line << std::endl;
                    //check if line is a comment
                    if(line[0] == '#')
                    {
                        //ignore
                    }
                    else
                    {
                        //check if it's a section:
                        if(line.find('{')!=std::string::npos)
                        {
                            //read until the end of the section
                            //std::cout << "section" << std::endl;
                            //std::cout << line << std::endl;
                            section_name = line.substr(0, line.find("{"));
                            std::string section_data = line.substr(line.find("{")+1, line.length()-line.find("{")-1); 
                            //read until the end of the section
                            while (std::getline(iss,line))
                            {
                                 if (line.find('}')!=std::string::npos)
                                {
                                        break;
                                }
                                
                                section_data += line + "\n";
                               
                            }
                            //parse subsections from section_data 
                            //std::cout << section_data << std::endl;
                            std::istringstream iss2(section_data);
                            //get keys and values from subsections
                            while ( std::getline (iss2,line) )
                            {
                                //std::cout << line << std::endl;
                                //check if line is a comment
                                if(line[0] == '#')
                                {
                                    //ignore
                                }
                                else
                                {
                                    //check if line is a section
                                    if(line[0] == ' ')
                                    {
                                        //check if line is a subsection
                                        if(line[1] == ' ')
                                        {
                                            //check if line is a parameter
                                            if(line[2] == ' ')
                                            {
                                                //parameter
                                                //std::cout << "parameter" << std::endl;
                                                //std::cout << line << std::endl;
                                                //std::cout << line.find(" ") << std::endl;
                                                //std::cout << line.find(";") << std::endl;
                                                parameter_name = line.substr(line.find(" ")+1, line.find(";")-line.find(" ")-1);
                                                parameter_value = line.substr(line.find(";")+1, line.length()-line.find(";")-1);
                                                //std::cout << parameter_name << std::endl;
                                                //std::cout << parameter_value << std::endl;
                                                configuration[file_name][section_name][subsection_name][parameter_name] = parameter_value;
                                            }
                                            else
                                            {
                                                //subsection
                                                //std::cout << "subsection" << std::endl;
                                                //std::cout << line << std::endl;
                                                subsection_name = line.substr(line.find(" ")+1, line.find("{")-line.find(" ")-1);
                                                //std::cout << subsection_name << std::endl;
                                                configuration[file_name][section_name][subsection_name];
                                            }
                                        }
                                        else
                                        {
                                            //section
                                            //std::cout << "section" << std::endl;
                                            //std::cout << line << std::endl;
                                            section_name = line.substr(line.find(" ")+1, line.find("{")-line.find(" ")-1);
                                            //std::cout << section_name << std::endl;
                                            configuration[file_name][section_name];

                                        }
                                    }//end if
                                }//end else
                        }// 
                    }//end if
                }// 
            }//end while
            }//end parse_sections

            //getters
            std::map< std::string/*file_name*/, std::map< std::string/*section_name = server*/, std::map< std::string/*subsection name*/, std::map<std::string/*key*/,std::string/*value*/>> > > get_configuration()    { return configuration; }   
            std::map< std::string/*section_name = server*/, std::map< std::string/*subsection name*/, std::map<std::string/*key*/,std::string/*value*/>> > get_file(std::string file_name) { return configuration[file_name]; }
            std::map< std::string/*subsection name*/, std::map<std::string/*key*/,std::string/*value*/>> get_section(std::string file_name, std::string section_name) { return configuration[file_name][section_name]; }
            std::map<std::string/*key*/,std::string/*value*/> get_subsection(std::string file_name, std::string section_name, std::string subsection_name) { return configuration[file_name][section_name][subsection_name]; }
            std::string get_parameter(std::string file_name, std::string section_name, std::string subsection_name, std::string parameter_name) { return configuration[file_name][section_name][subsection_name][parameter_name]; }
            void set_configuration(std::map< std::string/*file_name*/, std::map< std::string/*section_name = server*/, std::map< std::string/*subsection name*/, std::map<std::string/*key*/,std::string/*value*/>> > > configuration) { this->configuration = configuration; }
            
            void dump()const
            {
                //dump the configuration
                std::cout << "[D]umping configuration :" << std::endl;
                for(auto file : configuration)
                {
                    std::cout << "\t[F]ile : " << file.first << std::endl;
                    for(auto section : file.second)
                    {
                        std::cout << "\t[S]ection : " << section.first << std::endl;
                        for(auto subsection : section.second)
                        {
                            std::cout << "\t[s]ubsection : " << subsection.first << std::endl;
                            for(auto parameter : subsection.second)
                            {
                                std::cout << "[P]arameter : " << parameter.first << " " << parameter.second << std::endl;
                            }// end for
                        }//end for
                    }//end for
                }   //end for

            }//end dump

            
            };
    }   //namespace nginx_config_helper
}   //namespace provallo
//end of file
#endif