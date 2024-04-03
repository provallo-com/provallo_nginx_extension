// implementation of https://www.biorxiv.org/content/10.1101/2021.03.22.436524v2.full :
// "A theoretical framework for the role of the brain's connectome in the emergence of consciousness" 
// by Michael W. Cole, 2021 
//
#ifndef _BRAIN_KERNEL_H_
#define _BRAIN_KERNEL_H_


#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>

#include "../matrix.h"
#include "../info_helper.h"
#include "../utils.h"
#include "../rnn.h"




//mapping from latent embedding locations to a covariance matrix for neural activity.
//from voxel space to the latent embedding space
//map estimation
//4.4 Predicting activity for new voxels
//4.5 Predicting activity for new tasks 
namespace provallo
{   

template <typename T> 
class brain_kernel
{
    matrix<T> _latent_embedding; 
    matrix<T> _voxel_activity; 
    matrix<T> _covariance_matrix; 
    matrix<T> _neural_activity; 
    matrix<T> _task_activity; 
    matrix<T> _task_embedding; 
    matrix<T> _task_covariance; 
    matrix<T> _task_neural_activity; 
    matrix<T> _task_voxel_activity; 
    //detection classes to register
    std::map<T, std::function<T (T, real_t)> > _class_decision_points; 
    //regression : 
    //neural activity to voxel activity 
    //task activity to neural activity 
    std::mutex _lock; 

    //nonlinearity 
    public:

    brain_kernel () : _latent_embedding(0,0), _voxel_activity(0,0), _covariance_matrix(0,0), _neural_activity(0,0), _task_activity(0,0), _task_embedding(0,0), _task_covariance(0,0), _task_neural_activity(0,0), _task_voxel_activity(0,0)
    {
    } 
    ~brain_kernel () 
    {
    }
    void init ()
    {
    }
    
    bool clean ()
    {
        return true; 
    }
    bool on_decision (T _class, real_t prob)
    {
        auto it = _class_decision_points.find (_class); 
        while (it != _class_decision_points.end () )
        {
            if(it->first == _class )
                it->second (_class, prob); 
            it++;
        }

        return false; 
    } 
    bool register_detection_class (const T &_class, const std::function<T (T, double prob)> &func)
    {
        //lock mutex
        std::lock_guard<std::mutex> lock (_lock); 
        if (_class_decision_points.find (_class) == _class_decision_points.end ())
        {
            _class_decision_points.insert (std::make_pair (_class, func));
            return true; 
        }
        return false;
    }
    bool unregister_detection_class (const T &_class)
    {
        std::lock_guard<std::mutex> lock (_lock); 
        auto it = _class_decision_points.find (_class); 
        if (it != _class_decision_points.end ())
        {
            _class_decision_points.erase (it); 
            return true; 
        }
        return false;   

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
        }
        else if (filename.find(".csv") != std::string::npos)
        {
            //load csv: 
            std::vector<std::vector<std::string> > data; 
            //load csv: 
            data = utils::load_csv<std::string>(filename); 
            //load data: 
            _latent_embedding = matrix<T>(data.size(), data[0].size()); 
            _voxel_activity = matrix<T>(data.size(), data[0].size()); 
            _covariance_matrix = matrix<T>(data.size(), data[0].size()); 
            _neural_activity = matrix<T>(data.size(), data[0].size()); 
            _task_activity = matrix<T>(data.size(), data[0].size()); 
            _task_embedding = matrix<T>(data.size(), data[0].size()); 
            _task_covariance = matrix<T>(data.size(), data[0].size()); 
            _task_neural_activity = matrix<T>(data.size(), data[0].size()); 
            _task_voxel_activity = matrix<T>(data.size(), data[0].size());
            //parse data:
            for (size_t i = 0; i < data.size(); i++)
            {
                for (size_t j = 0; j < data[i].size(); j++)
                {
                    _latent_embedding(i,j) = std::stod(data[i][j]); 
                    _voxel_activity(i,j) = std::stod(data[i][j]); 
                    _covariance_matrix(i,j) = std::stod(data[i][j]); 
                    _neural_activity(i,j) = std::stod(data[i][j]); 
                    _task_activity(i,j) = std::stod(data[i][j]); 
                    _task_embedding(i,j) = std::stod(data[i][j]); 
                    _task_covariance(i,j) = std::stod(data[i][j]); 
                    _task_neural_activity(i,j) = std::stod(data[i][j]); 
                    _task_voxel_activity(i,j) = std::stod(data[i][j]); 
                }
            }   
        }
        //unlock:
        return;
    }
    
};
}
#endif 