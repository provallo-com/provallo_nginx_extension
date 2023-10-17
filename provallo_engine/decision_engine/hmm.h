/*
 * hmm.h
 *
 *  Created on: Jan 24, 2022
 *      Author: kardon
 */

#ifndef DECISION_ENGINE_HMM_H_
#define DECISION_ENGINE_HMM_H_
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <map>
#include <stdexcept>
#include <cmath>

#include "utils.h"
#include "matrix.h"

namespace provallo
{

  template<class T = size_t, class container = std::vector<T> >
    class hmm
    {

      typedef container *ptr;
      size_t _nstates;
      size_t _nobs;
      ptr _observation_sampler;
      provallo::matrix<real_t> _observation;
      ptr _initial_sampler;
      void
      sanity () const;
      void
      serialize (std::ostream &out);
      hmm*
      deserialize (std::istream &in);
      void
      print (std::ostream &out) const;
      size_t
      size () const;
      size_t
      observation_size () const;
      real_t
      trans (const size_t i, const size_t j) const;
      hmm<T,container>*
      set_initial (std::vector<real_t> &init);
      size_t
      sample_observation (size_t state) const;
    public:
      hmm<T, container> () :
	  _nstates (0), _nobs (0), _initial_sampler (nullptr), _observation_sampler (
	      nullptr)
      {

      }
      ~hmm<T, container> ()
      {

        if(_observation_sampler)
        for (auto sample : _observation_sampler)
              sample->clear ();
        if(_initial_sampler)
        for (auto sample : _initial_sampler)
              sample->clear ();

        if (_observation_sampler)
            {delete[] _observation_sampler;}
        
        _observation_sampler = nullptr;

        if (_initial_sampler)
            {delete[] _initial_sampler;}
        
        _initial_sampler = nullptr;

      }

    };

  template<class T, class container>  
    void
    hmm<T, container>::sanity () const
    {
      if (_nstates == 0)  throw std::runtime_error ("hmm: no states");  
      if (_nobs == 0)  throw std::runtime_error ("hmm: no observations");
      if (_observation_sampler == nullptr)  throw std::runtime_error ("hmm: no observation sampler");
      if (_initial_sampler == nullptr)  throw std::runtime_error ("hmm: no initial sampler");

    }
  template<class T, class container>
    void
    hmm<T, container>::serialize (std::ostream &out)
    {
      sanity ();
      out << _nstates << " " << _nobs << std::endl;
      for (size_t i = 0; i < _nstates; i++){
      for (size_t j = 0; j < _nstates; j++){
        out << _observation (i, j) << " ";    
      }        
      out << std::endl;
      }
      for (size_t i = 0; i < _nstates; i++){
        out << _initial_sampler[i] << " ";    
      }
      out << std::endl;
    }
  template<class T, class container>
    hmm<T, container>*
    hmm<T, container>::deserialize (std::istream &in)
    {
      sanity ();
      in >> _nstates >> _nobs;
      _observation = provallo::matrix<real_t> (_nstates, _nstates);
      for (size_t i = 0; i < _nstates; i++){
      for (size_t j = 0; j < _nstates; j++){
        in >> _observation (i, j);    
      }        
      }
      for (size_t i = 0; i < _nstates; i++){
        in >> _initial_sampler[i];    
      }
    } 
  template<class T, class container>  
    void
    hmm<T, container>::print (std::ostream &out) const
    {
      sanity ();
      out << "hmm: " << _nstates << " states, " << _nobs << " observations"
    << std::endl;
      out << "initial: ";
      for (size_t i = 0; i < _nstates; i++) out << _initial_sampler[i] << " ";    
      out << std::endl;
      out << "observation: " << std::endl;
      for (size_t i = 0; i < _nstates; i++){
      for (size_t j = 0; j < _nstates; j++){
        out << _observation (i, j) << " ";    
      }   
      out << std::endl;     
      }   


    }

  template<class T, class container>  
    size_t
    hmm<T, container>::size () const
    {
      sanity ();
      return _nstates;
    }   
  template<class T, class container>  
    size_t
    hmm<T, container>::observation_size () const
    {
      sanity ();
      return _nobs;
    }

  template<class T, class container>      
  hmm<T, container>*
    hmm<T, container>::set_initial (std::vector<real_t> &init)
    {
      sanity ();
      if (init.size () != _nstates) throw std::runtime_error ("hmm: invalid initial state");
          for (size_t i = 0; i < _nstates; i++) _initial_sampler[i] = init[i];
           return this;
    } 

  template<class T, class container>
  size_t
    hmm<T, container>::sample_observation (size_t state) const
    {
      sanity ();
      if (state >= _nstates) throw std::runtime_error ("hmm: invalid state");
      return _observation_sampler[state];

    }

  template<class T, class container>
  real_t  hmm<T,container>::trans (const size_t i, const size_t j) const
  {
    sanity ();
    if (i >= _nstates || j >= _nstates) throw std::runtime_error ("hmm: invalid state");
    return _observation (i, j);
    //
  } 
  template<class T, class container>
  hmm<T,container>*
  create_hmm (size_t nstates, size_t nobs)
  {
    if (nstates == 0) throw std::runtime_error ("hmm: no states");
    if (nobs == 0) throw std::runtime_error ("hmm: no observations");
    hmm<T,container> *h = new hmm<T,container> ();
    h->_nstates = nstates;
    h->_nobs = nobs;
    h->_observation_sampler = new container[nstates];
    h->_initial_sampler = new container[nstates];
    for (size_t i = 0; i < nstates; i++)
      {
        h->_observation_sampler[i] = container (nobs);
        h->_initial_sampler[i] = container (nobs);
      }
    return h;
  } 
  template<class T, class container>
  hmm<T,container>*
  create_hmm (std::vector<real_t> &init, std::vector<real_t> &obs)
  {
    if (init.size () == 0) throw std::runtime_error ("hmm: no initial state");
    if (obs.size () == 0) throw std::runtime_error ("hmm: no observations");
    hmm<T,container> *h = new hmm<T,container> ();
    h->_nstates = init.size ();
    h->_nobs = obs.size ();
    h->_observation_sampler = new container[h->_nstates];
    h->_initial_sampler = new container[h->_nstates];
    for (size_t i = 0; i < h->_nstates; i++)
      {
        h->_observation_sampler[i] = container (h->_nobs);
        h->_initial_sampler[i] = container (h->_nobs);
      }
    return h;
  }
  template<class T, class container>
  hmm<T,container>*
  create_hmm (std::vector<real_t> &init, std::vector<std::vector<real_t>> &obs)
  {
    if (init.size () == 0) throw std::runtime_error ("hmm: no initial state");
    if (obs.size () == 0) throw std::runtime_error ("hmm: no observations");
    hmm<T,container> *h = new hmm<T,container> ();
    h->_nstates = init.size ();
    h->_nobs = obs.size ();
    h->_observation_sampler = new container[h->_nstates];
    h->_initial_sampler = new container[h->_nstates];
    for (size_t i = 0; i < h->_nstates; i++)
      {
        h->_observation_sampler[i] = container (h->_nobs);
        h->_initial_sampler[i] = container (h->_nobs);
      }
    return h;
  }
  template<class T, class container>
  hmm<T,container>*
  create_hmm (std::vector<std::vector<real_t>> &init, std::vector<std::vector<real_t>> &obs)
  {
    if (init.size () == 0) throw std::runtime_error ("hmm: no initial state");
    if (obs.size () == 0) throw std::runtime_error ("hmm: no observations");
    hmm<T,container> *h = new hmm<T,container> ();
    h->_nstates = init.size ();
    h->_nobs = obs.size ();
    h->_observation_sampler = new container[h->_nstates];
    h->_initial_sampler = new container[h->_nstates];
    for (size_t i = 0; i < h->_nstates; i++)
      {
        h->_observation_sampler[i] = container (h->_nobs);
        h->_initial_sampler[i] = container (h->_nobs);
      }
    return h;
  } 
  template<class T, class container>
  hmm<T,container>*
  create_hmm (std::vector<std::vector<real_t>> &init, std::vector<real_t> &obs)
  {
    if (init.size () == 0) throw std::runtime_error ("hmm: no initial state");
    if (obs.size () == 0) throw std::runtime_error ("hmm: no observations");
    hmm<T,container> *h = new hmm<T,container> ();
    h->_nstates = init.size ();
    h->_nobs = obs.size ();
    h->_observation_sampler = new container[h->_nstates];
    h->_initial_sampler = new container[h->_nstates];
    for (size_t i = 0; i < h->_nstates; i++)
      {
        h->_observation_sampler[i] = container (h->_nobs);
        h->_initial_sampler[i] = container (h->_nobs);
      }
    return h;
  } 
  

} // namespace provallo


#endif /* DECISION_ENGINE_HMM_H_ */
