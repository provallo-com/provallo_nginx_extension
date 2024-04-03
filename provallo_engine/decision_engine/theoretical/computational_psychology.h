#ifndef __COMPUTATIONAL_PSYCHOLOGY_H__
#define __COMPUTATIONAL_PSYCHOLOGY_H__


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
   
#include "../pipeline.h"
#include "../decision_engine.h"
namespace provallo {   
    //COMPUTAIONAL MODEL OF ATTENTION :
    //Nicola De Pisapia,Greg Repovs, and Todd S. Braver 
    //Cognitive, Affective, & Behavioral Neuroscience
    //2006, Vol. 6, No. 2, 99–113
    //2006 Psychonomic Society, Inc.
    //https://www.researchgate.net/publication/7310003_Computational_Model_of_Attention 

    //inhibitory input implements audio-visual attention model 
    //Sherman-Morrison-Woodbury formula:
    //https://en.wikipedia.org/wiki/Woodbury_matrix_identity 
    //https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
    //https://en.wikipedia.org/wiki/Matrix_determinant_lemma

    template <typename T>
    matrix<T> sherman_morrison_woodbury(const matrix<T> &A, const matrix<T> &U, const matrix<T> &V, const matrix<T> &X)
    {
        //A is a square matrix
        //U and V are matrices
        //X is a matrix
        //A = U*V
        //A^-1 = V^-1*U^-1
        matrix<T> A_inv = A.inverse();
        matrix<T> V_inv = V.inverse();
        matrix<T> U_inv = U.inverse();
        return A_inv - A_inv*U*(V_inv*X*U_inv)*V_inv*A_inv; 
    }
    template <typename T>
    matrix<T> woodbury_matrix_identity(const matrix<T> &A, const matrix<T> &U, const matrix<T> &V, const matrix<T> &X)
    {
         //        where A, U, C and V are conformable matrices: A is n×n, C is k×k, U is n×k, and V is k×n. This can be derived using blockwise matrix inversion.

        matrix<T> ret ;
        matrix<T> C_inv = X.inverse();
        matrix<T> A_inv = A.inverse();
        matrix<T> VTA_inv = V*A_inv;
        matrix<T> C_inv_VTA_inv = C_inv*VTA_inv;
        matrix<T> U_C_inv_VTA_inv = U*C_inv_VTA_inv;
        matrix<T> U_C_inv_VTA_inv_A_inv = U_C_inv_VTA_inv*A_inv;
        matrix<T> A_inv_U_C_inv_VTA_inv_A_inv = A_inv*U_C_inv_VTA_inv_A_inv;
        ret = A_inv - A_inv_U_C_inv_VTA_inv_A_inv;
        return ret;
 
    }
    ///Moore–Penrose inverse
    template <typename T>
    matrix<T> moore_penrose_inverse(const matrix<T> &A)
    {
        matrix<T> A_transpose = A.transpose();
        matrix<T> A_A_transpose = A*A_transpose;
        matrix<T> A_transpose_A = A_transpose*A;    
        matrix<T> A_A_transpose_inv = A_A_transpose.inverse();
        matrix<T> A_transpose_A_inv = A_transpose_A.inverse();
        matrix<T> A_moore_penrose = A_transpose_A_inv*A
        return A_moore_penrose;
    }

    template <typename T>
    class inhibitory_input
    {
        size_t _n;
        size_t _m;
        //time points:
        size_t _t;
        //frame rate:
        size_t _frame_rate;
        //audio frequency:
        size_t _audio_frequency;
        //visual frequency:
        size_t _visual_frequency;


        
        //n-dimensional input:

        provallo::matrix<T> _optical_input;
        provallo::matrix<T> _auditory_input;
        provallo::matrix<T> _colors;
        provallo::matrix<T> _saliency;
        provallo::matrix<T> _attention;
        provallo::matrix<T> _task;
        //hypercolumns :
        provallo::matrix<T> _hypercolumns;
        //neural activity :
        provallo::matrix<T> _neural_activity;
        //task activity :
        std::map<T,provallo::matrix<T> > _task_activity; 
        //cortical/spontaenous activity : 
        provallo::matrix<T> _cortical_activity; 
        //exhibitory and inhibitory neurons : 
        provallo::matrix<T> _excitatory_neurons; 
        provallo::matrix<T> _inhibitory_neurons;
        //neural activity to voxel activity
        provallo::matrix<T> _voxel_activity;
        //task activity to neural activity
        provallo::matrix<T> _task_neural_activity;

        std::mutex _lock;
        std::string video_path;//video path
        std::string audio_path;//audio path
        std::string visual_path;//image path
        std::string task_path;//task path

        std::string model_path;//model path
        std::string data_path;//data path
        std::string output_path;//output path
        std::string input_path;//input path
        std::string output_file;//output file
        std::string input_file;//   input file
        std::string model_file;
        std::string data_file;
        
        public:

        //construct with video path : 
        inhibitory_input(size_t n, size_t m, size_t t, size_t frame_rate, size_t audio_frequency, size_t visual_frequency) : _n(n), _m(m), _t(t), _frame_rate(frame_rate), _audio_frequency(audio_frequency), _visual_frequency(visual_frequency)
        {
            init();
            //load the video and audio data
            load_video_data(video_path);
            //parse the video data
            //parse the audio data
            parse_audio_data(audio_path);
            //parse the visual data
            parse_visual_data(visual_path);
            //parse the task data

        }
        //inhibitory input destructor
        ~inhibitory_input()
        {
        }
        //initialize inhibitory input
        void init()
        {
            //fit the generators to the data inputs 
            _optical_input = provallo::matrix<T>(_n,_m); 
            _auditory_input = provallo::matrix<T>(_n,_m);
            _colors = provallo::matrix<T>(_n,_m);
            _saliency = provallo::matrix<T>(_n,_m);
            _attention = provallo::matrix<T>(_n,_m);
            _task = provallo::matrix<T>(_n,_m);
            _hypercolumns = provallo::matrix<T>(_n,_m);
            _neural_activity = provallo::matrix<T>(_n,_m);
            _cortical_activity = provallo::matrix<T>(_n,_m);
            _excitatory_neurons = provallo::matrix<T>(_n,_m);
            _inhibitory_neurons = provallo::matrix<T>(_n,_m);
            _voxel_activity = provallo::matrix<T>(_n,_m);
            _task_neural_activity = provallo::matrix<T>(_n,_m);

            //initialize the mixed gaussian generators
 
            
        }   
        


    };

}



#endif 
