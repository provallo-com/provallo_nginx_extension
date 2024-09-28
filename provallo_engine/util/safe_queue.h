#ifndef SAFE_QUEUE_H_
#define SAFE_QUEUE_H_

#include <map>
#include <condition_variable>
#include <thread>
#include <mutex>
#include <queue>

namespace provallo
{
  template<typename lock_type, typename queue_data_type>
    class safe_queue
    {
    private:
      mutable lock_type _locker;
      mutable lock_type _varlock;

      std::queue<queue_data_type*> _queue;
      std::condition_variable _mcond;
    public:
      safe_queue ();
      safe_queue (const safe_queue&) = delete;
      safe_queue& operator=(const safe_queue&) = delete;
      safe_queue (safe_queue&&) = delete;
      safe_queue& operator=(safe_queue&&) = delete;
      virtual ~safe_queue ();
      inline void
      wait ()
      {
	      _mcond.wait (_varlock);
      }
      queue_data_type*
      pop ();
      void
      push (queue_data_type *qdt);
      void
      touch ();
      size_t
      size () const;

      bool empty() const
      {
        std::lock_guard<lock_type> g (_locker);
        return _queue.empty();
      } 
      //front
      queue_data_type*
      front()
      {
        std::lock_guard<lock_type> g (_locker);
        return _queue.front();
      } 
      //back
      queue_data_type*  
      back()
      {
        std::lock_guard<lock_type> g (_locker);
        return _queue.back();
      }
      //push_back
      void
      push_back (queue_data_type *qdt)
      {
        std::lock_guard<lock_type> g (_locker);
        _queue.push_back (qdt);
        _mcond.notify_one ();
      }
      //pop_back
      queue_data_type*
      pop_back ()
      {
        std::lock_guard<lock_type> g (_locker);
        queue_data_type *pData = _queue.back ();
        _queue.pop_back ();
        return pData;
      } 
      //push_front
      void
      push_front (queue_data_type *qdt)
      {
        std::lock_guard<lock_type> g (_locker);
        _queue.push_front (qdt);
        _mcond.notify_one ();
      }
      //pop_front
      queue_data_type*
      pop_front ()
      {
        std::lock_guard<lock_type> g (_locker);
        queue_data_type *pData = _queue.front ();
        _queue.pop_front ();
        return pData;
      } 

    };

  template<typename lock_type, typename queue_data_type>
    safe_queue<lock_type, queue_data_type>::safe_queue ()
    {
      std::lock_guard<lock_type> g (_locker);
    }
  template<typename lock_type, typename queue_data_type>
    safe_queue<lock_type, queue_data_type>::~safe_queue ()
    {
      std::lock_guard<lock_type> g (_locker);
    }
  template<typename lock_type, typename queue_data_type>
    queue_data_type*
    safe_queue<lock_type, queue_data_type>::pop ()
    {
      std::lock_guard<lock_type> g (_locker);
      queue_data_type *pData = _queue.front ();
      _queue.pop ();
      return pData;
    }

  template<typename lock_type, typename queue_data_type>
    void
    safe_queue<lock_type, queue_data_type>::push (queue_data_type *qdt)
    {
      std::lock_guard<lock_type> g (_locker);

      _queue.push (qdt);
      _mcond.notify_one ();
    }

  template<typename lock_type, typename queue_data_type>
    void
    safe_queue<lock_type, queue_data_type>::touch ()
    {
      std::lock_guard<lock_type> g (_locker);
      _mcond.notify_one ();
    }

  template<typename lock_type, typename queue_data_type>
    size_t
    safe_queue<lock_type, queue_data_type>::size () const
    {
      std::lock_guard<lock_type> g (_locker);
      return _queue.size ();
    }

    
}

#endif //SAFE_QUEUE_H_
