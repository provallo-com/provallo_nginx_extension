#ifndef RAG_LOADER_H_ 
#define RAG_LOADER_H_

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <map>
#include <functional>
#include <mutex>
#include <algorithm>
#include <thread>
#include <chrono>
#include <iostream>
//for autoencoders and rnn: 
#include "autoencoder.h"
#include "rnn.h"
//for matrix operations: 
#include "matrix.h" 
//for info helper: 
#include "info_helper.h" 
//for utils:
#include "utils.h"

//for pipeline components:
#include "pipelinebuilder.h"

namespace provallo
{
    //load llm pipeline components from file :
    template <typename T> 
    class rag_loader
    {
        std::mutex _lock; 
        matrix<T> _latent_embedding; 
        matrix<T> _voxel_activity; 
        matrix<T> _covariance_matrix;
        matrix<T> _neural_activity;
        matrix<T> _task_activity;
        matrix<T> _task_embedding;
        matrix<T> _task_covariance;
        matrix<T> _task_neural_activity; 
        matrix<T> _task_voxel_activity;
        std::map<T, std::function<T (T, real_t)> > _class_decision_points;
        //vocabularies for the pipeline components: 
        std::map<T, std::vector<T> > _vocabularies; 
        //pipeline components: 
        std::map<T, pipeline<T> > _pipeline_components; 
        //autoencoders: 
        std::map<T, autoencoder<T> > _autoencoders; 
        //rnn: 
        std::map<T, rnn<T> > _rnn; 

        public:

        rag_loader () : _latent_embedding(0,0), _voxel_activity(0,0), _covariance_matrix(0,0), _neural_activity(0,0), _task_activity(0,0), _task_embedding(0,0), _task_covariance(0,0), _task_neural_activity(0,0), _task_voxel_activity(0,0)
        {
        } 
        //constructor with filename:
        rag_loader (const std::string &filename) : _latent_embedding(0,0), _voxel_activity(0,0), _covariance_matrix(0,0), _neural_activity(0,0), _task_activity(0,0), _task_embedding(0,0), _task_covariance(0,0), _task_neural_activity(0,0), _task_voxel_activity(0,0)
        {
            load (filename); 
        }    
        ~rag_loader () 
        {
        }   
        //load:
        void load(const std::string& filename)
        {
            //lock:
            std::lock_guard<std::mutex> lock(_lock);
            //open file:

            if (filename.empty())
            {
                return; 
            } 

            std::ifstream file(filename); 
            if (!file.is_open())
            {
                return; 
            }
            //parse file format: 
            //if llvm was saved as cpkl: 
            if (filename.find(".cpkl") != std::string::npos)
            {
                //load cpkl: 
                std::map<std::string, matrix<T> > data; 
                //load cpkl: 
                data = utils::load_cpkl<std::string, matrix<T> >(filename); 
                //load data: 
                _latent_embedding = data["latent_embedding"]; 
                _voxel_activity = data["voxel_activity"]; 
                _covariance_matrix = data["covariance_matrix"]; 
                _neural_activity = data["neural_activity"]; 
                _task_activity = data["task_activity"]; 
                _task_embedding = data["task_embedding"]; 
                _task_covariance = data["task_covariance"]; 
                _task_neural_activity = data["task_neural_activity"]; 
                _task_voxel_activity = data["task_voxel_activity"]; 
                _vocabularies = utils::load_cpkl<std::string, std::vector<T> >(filename)["vocabularies"]; 
                _pipeline_components = utils::load_cpkl<std::string, pipeline<T> >(filename)["pipeline_components"]; 
                _autoencoders = utils::load_cpkl<std::string, autoencoder<T> >(filename)["autoencoders"]; 
                _rnn = utils::load_cpkl<std::string, rnn<T> >(filename)["rnn"]; 
            }
            else if (filename.find(".json") != std::string::npos)
            {
                //load json: 
                std::map<std::string, matrix<T> > data; 
                //load json: 
                data = utils::load_json<std::string, matrix<T> >(filename); 
                //load data: 
                _latent_embedding = data["latent_embedding"]; 
                _voxel_activity = data["voxel_activity"]; 
                _covariance_matrix = data["covariance_matrix"]; 
                _neural_activity = data["neural_activity"]; 
                _task_activity = data["task_activity"]; 
                _task_embedding = data["task_embedding"]; 
                _task_covariance = data["task_covariance"]; 
                _task_neural_activity = data["task_neural_activity"]; 
                _task_voxel_activity = data["task_voxel_activity"]; 
                _vocabularies = utils::load_json<std::string, std::vector<T> >(filename)["vocabularies"]; 
                _pipeline_components = utils::load_json<std::string, pipeline<T> >(filename)["pipeline_components"]; 
                _autoencoders = utils::load_json<std::string, autoencoder<T> >(filename)["autoencoders"]; 
                _rnn = utils::load_json<std::string, rnn<T> >(filename)["rnn"]; 
            }
                            
        }
    }; 
  
} /* namespace provallo */



#endif // RAG_LOADER_H_ 