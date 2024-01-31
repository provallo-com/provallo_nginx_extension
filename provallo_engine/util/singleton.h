#ifndef __SIGNLETON_H__
#define __SIGNLETON_H__

#include <memory>
#include <atomic>
#include <mutex>
#include <condition_variable>

namespace provallo {

template <typename T>
class singleton
{   
    //TODO: add thread safe instance
    private:
    static std::atomic<T*> instance;
    //
    static std::mutex m;
   // static std::condition_variable cv;
    static std::atomic_bool initialized;
    //static std::atomic_bool destroyed;
    public:
    //TODO: add thread safe get_instance
    static T* get_instance()
    {
        std::lock_guard<std::mutex> lg(m);

        if(!initialized)
        {
                instance.store(new T());
                initialized.store(true);

        }
        return instance.load();

    }
    static void destroy_instance()
    {
        std::lock_guard<std::mutex> lg(m);
        delete instance.load();
        instance.store(nullptr);
        initialized.store(false);
    }
    //TODO: add thread safe destroy_instance

    protected:
    singleton() = default;
    virtual ~singleton() = default;

    singleton(const singleton&) = delete;
    singleton& operator=(const singleton&) = delete;
    singleton(singleton&&) = delete;
    singleton& operator=(singleton&&) = delete;
    
};  

//initialize static members

template <typename T>
std::atomic<T*> singleton<T>::instance(nullptr);

template <typename T>
std::mutex singleton<T>::m;

//template <typename T>
//std::condition_variable singleton<T>::cv;


template <typename T>
std::atomic_bool singleton<T>::initialized(false);

//template <typename T>
//std::atomic_bool singleton<T>::destroyed(false);

 
} // namespace provallo
#endif