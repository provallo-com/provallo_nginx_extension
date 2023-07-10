/*
 * mmap_allocator.h
 *
 *  Created on: Apr 1, 2023
 *      Author: kardon
 */

#ifndef UTIL_MMAP_ALLOCATOR_H_
#define UTIL_MMAP_ALLOCATOR_H_
// some inclusions... 

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cerrno>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <limits>
#include <utility>
#include <iterator>
#include <functional>
#include <chrono>




namespace provallo  
{

    template<typename T>
    class mmap_allocator : public std::allocator<T>
    {   
        public:
        mmap_allocator() = default;
        mmap_allocator(const mmap_allocator&) = default;
        mmap_allocator(mmap_allocator&&) = default;
        mmap_allocator& operator=(const mmap_allocator&) = default;
        mmap_allocator& operator=(mmap_allocator&&) = default;
        ~mmap_allocator() = default;

        

 
 
        typedef T value_type;
        typedef T* pointer;
        typedef const T* const_pointer;
        typedef T& reference;
        typedef const T& const_reference;
        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;
        

        template<typename U>
        struct rebind
        {
            typedef mmap_allocator<U> other;
        };


        pointer allocate(size_type n, const void* hint = 0)
        {
            if (n > std::numeric_limits<size_type>::max() / sizeof(T))
                throw std::bad_alloc();
            
            if( hint != 0 )
                return static_cast<pointer>(mmap(const_cast<void*>(hint), n * sizeof(T), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0));

            if (auto p = static_cast<pointer>(mmap(0, n * sizeof(T), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)))
                return p;
            
            throw std::bad_alloc();
        }

        void deallocate(pointer p, size_type n)
        {
            munmap(p, n * sizeof(T));
        }
        
        template<typename U, typename... Args>
        void construct(U* p, Args&&... args)
        {
            ::new((void*)p) U(std::forward<Args>(args)...);
        }

        void destroy(pointer p)
        {
            p->~T();
        }   

        size_type max_size() const noexcept
        {
            return std::numeric_limits<size_type>::max() / sizeof(T);
        }   

        pointer address(reference x) const noexcept
        {
            return std::addressof(x);
        }       

       

        bool operator==(const mmap_allocator& other) const noexcept
        {
	      if (this == &other)
        	        return true;
        
	      return std::memcmp(this, &other, sizeof(*this)) == 0;    

        }   

        bool operator!=(const mmap_allocator& other) const noexcept
        {
            return !(*this == other);
        }   

        

    };// class mmap_allocator


    template<typename T>
    class safe_mmap_allocator : public mmap_allocator<T> 
    {
        std::recursive_mutex _mutex;
        public: 
        safe_mmap_allocator() = default;
        safe_mmap_allocator(const safe_mmap_allocator&) = default;
        safe_mmap_allocator(safe_mmap_allocator&&) = default;
        safe_mmap_allocator& operator=(const safe_mmap_allocator&) = default;
        safe_mmap_allocator& operator=(safe_mmap_allocator&&) = default;
        ~safe_mmap_allocator() = default;
        

        typedef T value_type;
        typedef T* pointer;

        pointer allocate(std::size_t n)
        {
            std::lock_guard<std::recursive_mutex> lock(_mutex);
            return mmap_allocator<T>::allocate(n);
        }

        void deallocate(pointer p, std::size_t n)
        {
            std::lock_guard<std::recursive_mutex> lock(_mutex);
            mmap_allocator<T>::deallocate(p,n);
        }

        template<typename U, typename... Args>
        void construct(U* p, Args&&... args)
        {
            std::lock_guard<std::recursive_mutex> lock(_mutex);
            mmap_allocator<T>::construct(p, std::forward<Args>(args)...);
        }

        void destroy(pointer p)
        {
            std::lock_guard<std::recursive_mutex> lock(_mutex);
            mmap_allocator<T>::destroy(p);
        }

        

    };// class safe_mmap_allocator



    template<typename T , typename AL = mmap_allocator<T> >
    class mmap_vector : public std::vector<T,AL>
    {
        public:

        typedef std::vector<T, AL> base_type;
        typedef typename base_type::value_type value_type;
        typedef typename base_type::pointer pointer;
        typedef typename base_type::const_pointer const_pointer;
        typedef typename base_type::reference reference;
        typedef typename base_type::const_reference const_reference;
        typedef typename base_type::size_type size_type;
        typedef typename base_type::difference_type difference_type;
        typedef typename base_type::iterator iterator;
        typedef typename base_type::const_iterator const_iterator;
        typedef typename base_type::reverse_iterator reverse_iterator;
        typedef typename base_type::const_reverse_iterator const_reverse_iterator;
        typedef typename base_type::allocator_type allocator_type; 
        static constexpr const size_type reserve_block = 1024;
        typedef typename base_type::allocator_type::template rebind<value_type>::other allocator_type_rebind;   

        // constructors

        mmap_vector() : base_type(reserve_block) {}

        mmap_vector(const mmap_vector& rhs) : base_type(rhs) {}
        
        mmap_vector(const base_type& rhs) : base_type(rhs) {}

        mmap_vector(mmap_vector&& rhs) : base_type(std::move(rhs)) {}
 
        template<typename InputIt>
        mmap_vector(InputIt first, InputIt last) : base_type(first, last) {}
        
        mmap_vector(std::size_t n, const AL& al) : base_type(n, al) {}
        mmap_vector(std::size_t n, const T& value, const AL& al) : base_type(n, value, al) {}

        mmap_vector(std::initializer_list<T> init, const AL& al) : base_type(init, al) {}   

        template<typename InputIt>
        mmap_vector(InputIt first, InputIt last, const AL& al) : base_type(first, last, al) {}


        mmap_vector(std::size_t n, const T& value, const AL& al, const AL& al2) : base_type(n, value, al, al2) {}
        mmap_vector(std::initializer_list<T> init, const AL& al, const AL& al2) : base_type(init, al, al2) {}
        template<typename InputIt>
        mmap_vector(InputIt first, InputIt last, const AL& al, const AL& al2) : base_type(first, last, al, al2) {}

        // assignment operators


        mmap_vector& operator=(const mmap_vector& rhs) {
            

            std::copy(rhs.begin(), rhs.end(), std::back_inserter(*this));   

            return *this;   

        }

        mmap_vector& operator=(const base_type& rhs) {

            std::copy(rhs.begin(), rhs.end(), std::back_inserter(*this));   

            return *this;   

        }   

        mmap_vector& operator=(const value_type& rhs) { 

            std::fill(this->begin(), this->end(), rhs);   

            return *this;

        }
        mmap_vector& operator=(mmap_vector&& rhs) {

            std::move(rhs.begin(), rhs.end(), std::back_inserter(*this));
            return *this;

        }
        ~mmap_vector() = default;
        
        mmap_vector(std::size_t n) : base_type(n) {}
        mmap_vector(std::size_t n, const T& value) : base_type(n, value) {}
        mmap_vector(std::initializer_list<T> init) : base_type(init) {}
        

        base_type& base() { return *this; } 
        const base_type& base() const { return *this; } 

        iterator begin() noexcept { return base_type::begin(); }
        const_iterator begin() const noexcept { return base_type::begin(); }
        const_iterator cbegin() const noexcept { return base_type::cbegin(); }


        iterator end() noexcept { return base_type::end(); }
        const_iterator end() const noexcept { return base_type::end(); }
        const_iterator cend() const noexcept { return base_type::cend(); }
        // reverse iterators    
        

        reverse_iterator rbegin() noexcept { return base_type::rbegin(); }
        const_reverse_iterator rbegin() const noexcept { return base_type::rbegin(); }
        const_reverse_iterator crbegin() const noexcept { return base_type::crbegin(); }


        reverse_iterator rend() noexcept { return base_type::rend(); }
        const_reverse_iterator rend() const noexcept { return base_type::rend(); }
        const_reverse_iterator crend() const noexcept { return base_type::crend(); }


        bool empty() const noexcept { return base_type::empty(); }
        size_type size() const noexcept { return base_type::size(); }
        size_type max_size() const noexcept { return base_type::max_size(); }
        size_type capacity() const noexcept { return base_type::capacity(); }

        template <typename allocator_type_rebind>
        const std::vector<T,allocator_type_rebind>& to_std_vector() const{  

            return std::vector<T,allocator_type_rebind>(this->begin(), this->end());            
        }

        void reserve(std::size_t n)
        {
            base_type::reserve(n);
        }


        mmap_vector& operator=(std::initializer_list<T> ilist)
        {
            base_type::operator=(ilist);
            return *this;
        }

 
        void resize(std::size_t n)
        {
            base_type::resize(n);
        }

        void resize(std::size_t n, const T& value)
        {
            base_type::resize(n, value);
        }

        void shrink_to_fit()
        {
            base_type::shrink_to_fit();
        }

        void clear()
        {
            base_type::clear();
        }

        void push_back(const T& value)
        {
            base_type::push_back(value);
        }

        void push_back(T&& value)
        {
            base_type::push_back(std::move(value));
        }

        template<typename... Args>
        void emplace_back(Args&&... args)
        {
            base_type::emplace_back(std::forward<Args>(args)...);
        }

        void pop_back()
        {
            base_type::pop_back();
        }

        void swap(mmap_vector& other)
        {
            base_type::swap(other);                  
        }


        void assign(std::size_t count, const T& value)
        {
            base_type::assign(count, value);
        }   


        template<typename InputIt>
        void assign(InputIt first, InputIt last)
        {
            base_type::assign(first, last);
        }   

        void assign(std::initializer_list<T> ilist)
        {
            base_type::assign(ilist);
        }       

        void insert(iterator pos, const T& value)
        {
            base_type::insert(pos, value);
        }   


        void insert(iterator pos, T&& value)
        {
            base_type::insert(pos, std::move(value));
        }   


        void insert(iterator pos, std::size_t count, const T& value)
        {
            base_type::insert(pos, count, value);
        }   

        template<typename InputIt>
        void insert(iterator pos, InputIt first, InputIt last)
        {
            base_type::insert(pos, first, last);
        }       


        T& operator [](std::size_t pos)
        {
            return base_type::operator[](pos);
        }  
        const T& operator [](std::size_t pos) const
        {
            return base_type::operator[](pos);
        }

        T& at(std::size_t pos)
        {
            return base_type::at(pos);
        }       

        const T& at(std::size_t pos) const
        {
            return base_type::at(pos);
        }   


        T& front()
        {
            return base_type::front();
        }

        const T& front() const
        {
            return base_type::front();
        }


        T& back()
        {
            return base_type::back();
        }


        const T& back() const
        {
            return base_type::back();
        }
 
        T* data()
        {
            return base_type::data();
        }


        const std::allocator<T>& get_allocator() const noexcept
        {
            return base_type::get_allocator();
        }       

      
        template<typename... Args>
        iterator emplace(const_iterator pos, Args&&... args)
        {
            return base_type::emplace(pos, std::forward<Args>(args)...);
        }   


        iterator erase(const_iterator pos)
        {
            return base_type::erase(pos);
        }


        iterator erase(const_iterator first, const_iterator last)
        {
            return base_type::erase(first, last);
        }

  
        protected:
        //  _M_get_Tp_allocator() const noexcept
 
        ///usr/include/c++/11/bits/stl_vector.h:555:68: error: no matching function for call to ‘std::_Vector_base<provallo::attribute, provallo::safe_mmap_allocator<provallo::attribute> >::_Vector_base(std::vector<provallo::attribute, provallo::safe_mmap_allocator<provallo::attribute> >::size_type, provallo::mmap_allocator<provallo::attribute>)’

        // _Vector_base(size_type __n, const allocator_type& __a = allocator_type())
        // : _M_impl(_M_allocate(__n), __a)
        // { _M_impl._M_finish = std::__uninitialized_default_n_a(_M_impl._M_start, __n, _M_get_Tp_allocator()); }

        // _Vector_base(size_type __n, const allocator_type& __a = allocator_type())
        // : _M_impl(_M_allocate(__n), __a)
        // { _M_impl._M_finish = std::__uninitialized_default_n_a(_M_impl._M_start, __n, _M_get_Tp_allocator()); }


        // _Vector_base(size_type __n, const allocator_type& __a = allocator_type())


        // _Vector_base(size_type __n, const allocator_type& __a = allocator_type())


        // _Vector_base(size_type __n, const allocator_type& __a = allocator_type())


        // _Vector_base(size_type __n, const allocator_type& __a = allocator_type())


        // _Vector_base(size_type __n, const allocator_type& __a = allocator_type())
 

        private:

    };// class mmap_vector

   
     


}// namespace provallo


#endif //PROVALLO_SAFE_MMAP_ALLOCATOR_HPP
