#ifndef AI_LIFE_SPLIT_STRATEGY_H 
#define AI_LIFE_SPLIT_STRATEGY_H


#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <algorithm>
#include <iterator>
#include <chrono>
#include <thread>
#include <mutex>
#include "matrix.h"
#include "info_helper.h"
using namespace std;


namespace provallo{

//create ai-life structure with flocking algorithm 
//and split strategy

template <typename T>
class life_split_strategy {
    
    private:
    struct splitter 
    {
        size_t row;
        size_t col;
        size_t split_size;
        size_t split_count;
    };  
    struct flocking
    {
        size_t row;
        size_t col;
        size_t flock_size;
        size_t flock_count;
    };
    struct flocking flock;
    struct splitter split;
    matrix<T> data;
    std::vector<T> labels;
    std::vector<T> predictions;
    std::vector<T> errors;
     
    public:
    split_strategy();
    void init (matrix<T> &m)
    {
        //initialize the split and flock 
        //with matrix m properties split (n-1)/2 and flock (n-1)/2 
        //where n is the number of rows and columns of matrix m 
        

        //split
        split.row = size_t((m.rows()-1) /2.);
        split.col =  size_t((m.cols()-1) /2.);
        split.split_size = split.row * split.col; 
        split.split_count = 0; 

        //flock
        flock.row = size_t((m.rows()-1) /2.);
        flock.col =  size_t((m.cols()-1) /2.);
        flock.flock_size = flock.row * flock.col;
        flock.flock_count = 0;
        this->data = m;
        
    }
    //get tuple of matrices of split data: 
    std::tuple<matrix<T>,matrix<T>,matrix<T>,matrix<T>> get_split(matrix<T> &m)
    {
        //get the split data from matrix m 
        //and return a tuple of matrices 
        //split the matrix into 4 parts :
        //top left, top right, bottom left, bottom right 
        //and return a tuple of matrices
        
        //std::tuple<matrix<T>,matrix<T>,matrix<T>,matrix<T>> split_data; 
        
        //top left

        matrix<T> top_left(split.row,split.col);
        for(size_t i = 0; i < split.row; i++)
        {
            for(size_t j = 0; j < split.col; j++)
            {
                top_left(i,j) = m(i,j);
            }
        } 
        //top right 
        matrix<T> top_right(split.row,split.col); 
        for(size_t i = 0; i < split.row; i++)
        {
            for(size_t j = split.col; j < m.cols(); j++)
            {
                top_right(i,j) = m(i,j);
            }
        }   
        //bottom left
        matrix<T> bottom_left(split.row,split.col);
        for(size_t i = split.row; i < m.rows(); i++)
        {
            for(size_t j = 0; j < split.col; j++)
            {
                bottom_left(i,j) = m(i,j);
            }
        }   
        //bottom right
        matrix<T> bottom_right(split.row,split.col);
        for(size_t i = split.row; i < m.rows(); i++)
        {
            for(size_t j = split.col; j < m.cols(); j++)
            {
                bottom_right(i,j) = m(i,j);
            }
        }
        auto split_data = std::make_tuple(top_left,top_right,bottom_left,bottom_right); 
        return split_data; 
    }
    //get tuple of matrices of flock data: 
    std::tuple<matrix<T>,matrix<T>,matrix<T>,matrix<T>> get_flock(matrix<T> &m)
    {
        //get the flock data from matrix m 
        //and return a tuple of matrices 
        //split the matrix into 4 parts :
        //top left, top right, bottom left, bottom right 
        //and return a tuple of matrices
        
        //std::tuple<matrix<T>,matrix<T>,matrix<T>,matrix<T>> flock_data; 
        
        //top left

        matrix<T> top_left(flock.row,flock.col);
        for(size_t i = 0; i < flock.row; i++)
        {
            for(size_t j = 0; j < flock.col; j++)
            {
                top_left(i,j) = m(i,j);
            }
        } 
        //top right 
        matrix<T> top_right(flock.row,flock.col); 
        for(size_t i = 0; i < flock.row; i++)
        {
            for(size_t j = flock.col; j < m.cols(); j++)
            {
                top_right(i,j) = m(i,j);
            }
        }   
        //bottom left
        matrix<T> bottom_left(flock.row,flock.col);
        for(size_t i = flock.row; i < m.rows(); i++)
        {
            for(size_t j = 0; j < flock.col; j++)
            {
                bottom_left(i,j) = m(i,j);
            }
        }   
        //bottom right
        matrix<T> bottom_right(flock.row,flock.col);
        for(size_t i = flock.row; i < m.rows(); i++)
        {
            for(size_t j = flock.col; j < m.cols(); j++)
            {
                bottom_right(i,j) = m(i,j);
            }
        }
        auto flock_data = std::make_tuple(top_left,top_right,bottom_left,bottom_right); 
        return flock_data; 
    } 
    //flocking algorithm on matrix m: 
    void on_flock()
    {
        matrix<T> m = this->data; 
        //get the flock data from matrix m
        auto flock_data = get_flock(m);
        
        //calculate trajectory of the flock,angle of movement, axis and direction 
        //calculate the flock center of mass 

        double flock_center_of_mass = 0.0; 
        double flock_angle_of_movement = 0.0;
        double flock_axis = 0.0;
        double flock_direction = 0.0;
        double flock_trajectory = 0.0;
        double flock_speed = 0.0;
        double flock_acceleration = 0.0;
        double flock_velocity = 0.0;
        double flock_distance = 0.0;
        double flock_time = 0.0;
        double flock_force = 0.0;
        double flock_energy = 0.0;
        double flock_momentum = 0.0;
        double flock_mass = 0.0;
        double flock_inertia = 0.0;
        double flock_torque = 0.0;

        

        //calculate the flock center of mass from the flock data :
        flock_center_of_mass = (std::get<0>(flock_data).sum() + std::get<1>(flock_data).sum() + std::get<2>(flock_data).sum() + std::get<3>(flock_data).sum()) / (flock.flock_size * 4); 
        //calculate the flock angle of movement from the flock data using tanh and sum:

        flock_angle_of_movement = (std::get<0>(flock_data)).sum() * (std::get<1>(flock_data)).tanh() + (std::get<2>(flock_data)).sum() * (std::get<3>(flock_data)).tanh(); 
        //check if angle of movement is zero 
        //if zero then flock is not moving
        //if not zero then flock is moving
        bool flock_is_moving = false; 
        if(flock_angle_of_movement != 0.0)
        {
            flock_is_moving = true; 
        }
        else
        {
            flock_is_moving = false; 
        }   
        //calculate the flock axis from the flock data using tanh and sum: 
        flock_axis = (std::get<0>(flock_data)).sum() * (std::get<1>(flock_data)).tanh() + (std::get<2>(flock_data)).sum() * (std::get<3>(flock_data)).tanh(); 
        //calculate the flock direction from the flock data using tanh and sum: 
        flock_direction = (std::get<0>(flock_data)).sum() * (std::get<1>(flock_data)).tanh() + (std::get<2>(flock_data)).sum() * (std::get<3>(flock_data)).tanh(); 
        //calculate the flock trajectory from the flock data using tanh and sum: 
        flock_trajectory = (std::get<0>(flock_data)).sum() * (std::get<1>(flock_data)).tanh() + (std::get<2>(flock_data)).sum() * (std::get<3>(flock_data)).tanh(); 
        //calculate the flock speed from the flock data using tanh and sum: 
        flock_speed = (std::get<0>(flock_data)).sum() * (std::get<1>(flock_data)).tanh() + (std::get<2>(flock_data)).sum() * (std::get<3>(flock_data)).tanh(); 
        //calculate the flock acceleration from the flock data using tanh and sum:
        flock_acceleration = (std::get<0>(flock_data)).sum() * (std::get<1>(flock_data)).tanh() + (std::get<2>(flock_data)).sum() * (std::get<3>(flock_data)).tanh();
        //calculate the flock velocity from the flock data using tanh and sum:
        flock_velocity = (std::get<0>(flock_data)).sum() * (std::get<1>(flock_data)).tanh() + (std::get<2>(flock_data)).sum() * (std::get<3>(flock_data)).tanh();
        //calculate the flock distance from the flock data using tanh and sum:
        flock_distance = (std::get<0>(flock_data)).sum() * (std::get<1>(flock_data)).tanh() + (std::get<2>(flock_data)).sum() * (std::get<3>(flock_data)).tanh();
        //calculate the flock time from the flock data using tanh and sum:
        flock_time = (std::get<0>(flock_data)).sum() * (std::get<1>(flock_data)).tanh() + (std::get<2>(flock_data)).sum() * (std::get<3>(flock_data)).tanh();
        //calculate the flock force from the flock data using tanh and sum:
        flock_force = (std::get<0>(flock_data)).sum() * (std::get<1>(flock_data)).tanh() + (std::get<2>(flock_data)).sum() * (std::get<3>(flock_data)).tanh();
        //calculate the flock energy from the flock data using tanh and sum:
        flock_energy = (std::get<0>(flock_data)).sum() * (std::get<1>(flock_data)).tanh() + (std::get<2>(flock_data)).sum() * (std::get<3>(flock_data));
        //calculate the flock momentum from the flock data using tanh and sum:
        flock_momentum = (std::get<0>(flock_data)).sum() * (std::get<1>(flock_data)) + (std::get<2>(flock_data)).sum() * (std::get<3>(flock_data));
        //calculate the flock mass from the flock data using tanh and sum:
        flock_mass = (std::get<0>(flock_data)).sum() * (std::get<1>(flock_data)) + (std::get<2>(flock_data)).sum() * (std::get<3>(flock_data));
        //calculate the flock inertia from the flock data using tanh and sum:
        flock_inertia = (std::get<0>(flock_data)).sum() * (std::get<1>(flock_data)) + (std::get<2>(flock_data)).sum() * (std::get<3>(flock_data));
        //calculate the flock torque from the flock data using tanh and sum:
        flock_torque = (std::get<0>(flock_data)).sum() * (std::get<1>(flock_data)) + (std::get<2>(flock_data)).sum() * (std::get<3>(flock_data));
        
        //calculate the flock center of mass from the flock data :
        flock_center_of_mass = (std::get<0>(flock_data).sum() + std::get<1>(flock_data).sum() + std::get<2>(flock_data).sum() + std::get<3>(flock_data).sum()) / (flock.flock_size * 4); 
        //calculate the flock angle of movement from the flock data using tanh and sum:


    }
    void step( real_t dt)
    {
        //step the flock by time dt 
        //update the flock position and velocity 
        //update the flock center of mass 
        //  update the flock angle of movement 
        //  update the flock axis
    }
    void reset()
    {
        //reset the flock to its initial state 
        //reset the flock position and velocity 
        //reset the flock center of mass 
        //

    }
    void draw_step() 
    {
        //saves .dat file for animation 
        //used by the gnuplot script generated by the save function 
        //to create an animation of the flock
    }
    ~split_strategy();
   

};





}//namespace provallo






#endif 