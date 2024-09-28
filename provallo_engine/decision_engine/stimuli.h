#ifndef __STIMULI_H__
#define __STIMULI_H__ 

#include "info_helper.h"
#include "utils.h" 
#include "matrix.h" 

//for linux high resolution timers 
#include <time.h> 
#include <sys/time.h> 
#include <unistd.h> 
#include <linux/types.h> 
#include <linux/unistd.h> 
#include <sys/syscall.h> 
#include <sys/resource.h>
#include <sys/times.h>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random> 
#include <algorithm>
#include <iterator> 

//periodic stimuli: 
namespace provallo 
{
    //periodic stimuli: 
    //parameters : 
    //1. period : the time interval between stimuli 
    //2. duration : the time interval for which the stimuli is active 
    //3. amplitude : the strength of the stimuli 
    //4. phase : the phase of the stimuli 
    //5. frequency : the frequency of the stimuli 
    //6. time : the time at which the stimuli is active 
    //7. stimuli : the stimuli 
    //8. stimuli_type : the type of stimuli 
    //9. stimuli_id : the id of the stimuli 
    enum stimuli_type 
    {
        periodic, 
        aperiodic, 
        random
    };  

    //attention box in the matrices:
    template <typename T> 
    struct attention_box 
    {
        T x; //column 
        T y;  //row
        T width; //in columns 
        T height; //in rows 
    };  

    //spike triggered average stimuli function: 
    template <typename T> 
    struct sta_stimuli 
    {
        T amplitude; 
        T frequency; 
        T phase; 
        attention_box<T> box; 
    };  




    //stimuli class:
    template < typename T>
    struct stimuli_signal 
    {
            T amplitude; 
            T frequency; 
            T phase; 
    };  
            
    template <typename T> 
    class periodic_stimuli 
    {
        T _period; 
        T _duration; 
        T _amplitude; 
        T _phase; 
        T _frequency; 
        T _time; 
        T _stimuli; 
        stimuli_type _stimuli_type; 
        T _stimuli_id; 
        std::mutex _lock; 
        std::condition_variable _cv; 
        std::atomic_bool _active; 
        std::atomic_bool _stop; 
        std::thread _stimuli_thread; 
        
        //each stimuli function recieves a matrix and returns a signal with an amplitude, frequency, and phase parameters 


        //std::function<sta_stimuli<T> (const provallo::matrix<T>&)> _stimuli_function;
        
        //linux timer:
        timer_t _timer; 
        struct sigevent _sev; 
        struct itimerspec _its; 
        struct sigaction _sa; 
        //generative model, use infohelper to generate the stimuli: 
        boltzman_base<T> _boltzman; 
        matrix<T> _data;
        
        //last index for value hits: 

        public:

        

        periodic_stimuli() : _period(0), _duration(0), _amplitude(0), _phase(0), _frequency(0), _time(0), _stimuli(0), _stimuli_type(periodic), _stimuli_id(0), _active(false), _stop(false),_boltzman()
        {
            //setup the linux timer: 
            _sa.sa_flags = SA_SIGINFO; 
            _sa.sa_sigaction = stimuli_handler; 
            sigemptyset(&_sa.sa_mask); 
            sigaction(SIGRTMIN, &_sa, NULL); 
            _sev.sigev_notify = SIGEV_SIGNAL; 
            _sev.sigev_signo = SIGRTMIN;
            _sev.sigev_value.sival_ptr = &_timer;
            //init the stimuli function: 
            _boltzman.set_temperature(0.5); 
            _boltzman.set_learning_rate(0.1); 
            _boltzman.set_momentum(0.9); 
            _boltzman.set_decay(0.9); 
            _boltzman.set_max_iterations(1000);
            _boltzman.set_min_error(0.0001);
            _boltzman.set_max_error(0.1);
            _boltzman.set_max_epochs(1000); 

            //set the size of the boltzman machine: 
            _boltzman.set_size(100,100); 
           
            init_timer()  ;
        }
        //destructor:
        ~periodic_stimuli() 
        {
            //close the timer: 
            close_timer(); 
        }   
        //start the stimuli:
        void start() 
        {
            _active = true; 
            _stop = false; 
            _stimuli_thread = std::thread([this](){
                //start the timer: 
                start_timer(); 
                //wait for the stimuli to stop: 
                std::unique_lock<std::mutex> lock(_lock); 
                _cv.wait(lock, [this](){return !_active;}); 
            }); 
        }   
        //stop the stimuli:
        void stop() 
        {
            _stop = true; 
            _active = false; 
            _cv.notify_all(); 
            _stimuli_thread.join(); 
        }   
        //init timer    

        void update_data(const provallo::matrix<T>& m) 
        {
            //lock the mutex:
            std::lock_guard<std::mutex> lock(_lock); 

            _data = m; 
        }   

        void set_data(const provallo::matrix<T>& m) 
        {
            //lock the mutex:
            std::lock_guard<std::mutex> lock(_lock); 

            _data = m; 
            //set dimensions according to the data: 
            _boltzman.set_size(m.rows(),m.cols()); 
            //set the data: 
            _boltzman.set_data(m); 
            //train the boltzman machine: 
            _boltzman.train(); 

        }   

            
        void init_timer() 
        {
            auto ret = timer_create(CLOCK_REALTIME, &_sev, &_timer); 
            if(ret == -1) 
            {
                std::cerr << "error creating timer" << std::endl; 
            }   

        } 
        void start_timer() 
        {
            _its.it_value.tv_sec = 0; 
            _its.it_value.tv_nsec = 1000000; 
            _its.it_interval.tv_sec = 0; 
            _its.it_interval.tv_nsec = 1000000; 
            auto ret = timer_settime(_timer, 0, &_its, NULL); 
            if(ret == -1) 
            {
                std::cerr << "error setting timer" << std::endl; 
            }   
        } 
        void stop_timer() 
        {
            _its.it_value.tv_sec = 0; 
            _its.it_value.tv_nsec = 0; 
            _its.it_interval.tv_sec = 0; 
            _its.it_interval.tv_nsec = 0; 
            auto ret = timer_settime(_timer, 0, &_its, NULL); 
            if(ret == -1) 
            {
                std::cerr << "error setting timer" << std::endl; 
            }   
        }   
        //close timer 
        void close_timer() 
        {
            auto ret = timer_delete(_timer); 
            if(ret == -1) 
            {
                std::cerr << "error deleting timer" << std::endl; 
            }   
        }   

        //setup the linux timer handle 

        //step function: 
        T step(T t) 
        {
            T s = 0; 
            if(_active) 
            {
                if(_stimuli_type == periodic) 
                {
                    s = stimuli_function(t); 
                }
                else if(_stimuli_type == aperiodic) 
                {
                    s = stimuli_function_aperiodic(t); 
                }
                else if(_stimuli_type == random) 
                {
                    s = stimuli_function_random(t); 
                }
            }
            return s; 
        }   
        //stimuli function for aperiodic stimuli:
        
        //set the stimuli type:
        void set_stimuli_type(stimuli_type t) 
        {
            _stimuli_type = t; 
        }
        //set the stimuli id:
        void set_stimuli_id(T id) 
        {
            _stimuli_id = id; 
        }
        //set the stimuli amplitude:
        void set_amplitude(T a) 
        {
            _amplitude = a; 
        }
        //set the stimuli frequency:
        void set_frequency(T f) 
        {
            _frequency = f; 
        }
        //set the stimuli phase:
        void set_phase(T p) 
        {
            _phase = p; 
        }
        //set the stimuli period:
        void set_period(T p) 
        {
            _period = p; 
        }
        //set the stimuli duration:
        void set_duration(T d) 
        {
            _duration = d; 
        }
        //set the stimuli time:
        void set_time(T t) 
        {
            _time = t; 
        }
        //set the stimuli:
        void set_stimuli(T s) 
        {
            _stimuli = s; 
        }
        //get the stimuli amplitude:
        T get_amplitude() 
        {
            return _amplitude; 
        }
        //get the stimuli frequency:
        T get_frequency() 
        {
            return _frequency; 
        }
        //get the stimuli phase:
        T get_phase() 
        {
            return _phase; 
        }
        //get the stimuli period:
        T get_period() 
        {
            return _period; 
        }
        //get the stimuli duration:
        T get_duration() 
        {
            return _duration; 
        }
        //get the stimuli time:
        T get_time() 
        {
            return _time; 
        }
        
        //stimuli_handler
        static void stimuli_handler(int sig, siginfo_t *si, void *uc) 
        {
            //get the stimuli object: 
            periodic_stimuli<T> *stimuli = (periodic_stimuli<T> *)si->si_value.sival_ptr; 
            //lock the stimuli object: 
            std::lock_guard<std::mutex> lock(stimuli->_lock); 
            if(stimuli->_active) 
            {
                            //get the current time: 
                stimuli->_time = std::chrono::high_resolution_clock::now(); 
                //get the stimuli: 
                stimuli->_stimuli = stimuli->step(stimuli->_time); 
                //notify the stimuli object: 
                stimuli->_cv.notify_all(); 
                //check if the stimuli is active: 
                if(stimuli->_stop) 
                {
                    //stop the stimuli: 
                    stimuli->stop_timer(); 
                    //close the timer: 
                    stimuli->close_timer(); 
                    //set the stimuli to inactive: 
                    stimuli->_active = false; 
                }
                //else start the timer again: 
                else 
                {
                    stimuli->start_timer(); 
                } 
         
            }
        }
        //stimuli functions :
        sta_stimuli<T> stimuli_function(T t) 
        {
            sta_stimuli<T> s; 
            //get the stimuli: 
            s.amplitude = _amplitude; 
            s.frequency = _frequency; 
            s.phase = _phase; 
            s.box.x = 0;
            s.box.y = 0;
            s.box.width = 100;
            s.box.height = 100;
                std::vector<T> v = _boltzman.generate(); 
                //get the stimuli:
                s.amplitude = v[0]; 
                s.frequency = v[1];
                s.phase = v[2];
                s.box.x = v[3]; 
                s.box.y = v[4];
                s.box.width = v[5];
                s.box.height = v[6];
                //return the stimuli:
                return s;
                
        } 
        //stimuli function for aperiodic stimuli: 
        sta_stimuli<T> stimuli_function_aperiodic(T t) 
        {
            sta_stimuli<T> s; 
            //get the stimuli: 
            s.amplitude = _amplitude; 
            s.frequency = _frequency; 
            s.phase = _phase; 
            //return the stimuli: 
            return s; 
        }   
        //stimuli function for random stimuli:  
        sta_stimuli<T> stimuli_function_random(T t) 
        {
            sta_stimuli<T> s; 
            //get the stimuli: 
            s.amplitude = _amplitude; 
            s.frequency = _frequency; 
            s.phase = _phase; 
            //return the stimuli: 
            return s; 
        }   
     };

};
#endif