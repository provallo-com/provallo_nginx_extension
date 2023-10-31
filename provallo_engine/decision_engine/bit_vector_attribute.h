#ifndef __BIT_VECTOR_ATTRIBUTE_H__
#define __BIT_VECTOR_ATTRIBUTE_H__



#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <map>

 #include "matrix.h"
#include "optimizers.h"
#include "dataset.h"
#include "../util/csv_file.h"

//provallo namespace
namespace provallo {

    #pragma pack(0)

    template <class T = uint8_t, size_t N = 8>
    class bit_type 
    {
        public :
        typedef T value_type;
        typedef T & reference;
        typedef const T & const_reference;
        typedef T * pointer;
        typedef const T * const_pointer;
        typedef T * iterator;

        static const size_t size = N;
        static const size_t npos = -1;
         

        private:
        value_type _bits; 
        public:
        bit_type() : _bits(0) {}
        bit_type(const bit_type &other) : _bits(other._bits) {}
        bit_type(bit_type &&other) : _bits(other._bits) {}
        bit_type(const T &other) : _bits(other) {}


        bit_type & operator=(const bit_type &other) { 
            if( &other!=this) {
            _bits = other._bits;}
         return *this; 
         }
        bit_type & operator=(bit_type &&other) {
                if( &other!=this)
               { _bits = other._bits;} return *this; 
             }
        bit_type & operator=(const T &other) {
                if( &other!=this){
                _bits = other;} return *this;
             }
        bit_type & operator=(T &&other) { if(&other!=this){
         _bits = other;} return *this;
        }

        bool operator[](size_t i) const { return (_bits & (1 << i)) != 0; }
        bool operator[](size_t i) { return (_bits & (1 << i)) != 0; }

        bool operator[](int i) const { return (_bits & (1 << i)) != 0; }
        bool operator[](int i) { return (_bits & (1 << i)) != 0; }

        bool operator==(const bit_type &other) const { return _bits == other._bits; }
        bool operator!=(const bit_type &other) const { return _bits != other._bits; }
        bool operator<(const bit_type &other) const { return _bits < other._bits; }
        bool operator>(const bit_type &other) const { return _bits > other._bits; }
        bool operator<=(const bit_type &other) const { return _bits <= other._bits; }
        bool operator>=(const bit_type &other) const { return _bits >= other._bits; }
        bit_type operator&(const bit_type &other) const { return bit_type(_bits & other._bits); }
        bit_type operator|(const bit_type &other) const { return bit_type(_bits | other._bits); }
        bit_type operator^(const bit_type &other) const { return bit_type(_bits ^ other._bits); }
        bit_type operator~() const { return bit_type(~_bits); }
        bit_type & operator&=(const bit_type &other) { _bits &= other._bits; return *this; }
        bit_type & operator|=(const bit_type &other) { _bits |= other._bits; return *this; }
        bit_type & operator^=(const bit_type &other) { _bits ^= other._bits; return *this; }
        bit_type & operator<<=(const bit_type &other) { _bits <<= other._bits; return *this; }
        bit_type & operator>>=(const bit_type &other) { _bits >>= other._bits; return *this; }
        bit_type operator<<(const bit_type &other) const { return bit_type(_bits << other._bits); }
        bit_type operator>>(const bit_type &other) const { return bit_type(_bits >> other._bits); }
   //     bit_type operator<<(size_t i) const { return bit_type(_bits << i); }
   //     bit_type operator>>(size_t i) const { return bit_type(_bits >> i); }
        bit_type & operator<<=(size_t i) { _bits <<= i; return *this; }

        bit_type & operator>>=(size_t i) { _bits >>= i; return *this; }
        bit_type operator&(const T &other) const { return bit_type(_bits & other); }
        bit_type operator|(const T &other) const { return bit_type(_bits | other); }
        bit_type operator^(const T &other) const { return bit_type(_bits ^ other); }
        //bit_type operator~() const { return bit_type(~_bits); }
        bit_type & operator&=(const T &other) { _bits &= other; return *this; }
        bit_type & operator|=(const T &other) { _bits |= other; return *this; }
        bit_type & operator^=(const T &other) { _bits ^= other; return *this; }
        bit_type & operator<<=(const T &other) { _bits <<= other; return *this; }
        bit_type & operator>>=(const T &other) { _bits >>= other; return *this; }
        bit_type operator<<(const T &other) const { return bit_type(_bits << other); }
        bit_type operator>>(const T &other) const { return bit_type(_bits >> other); }
       //bit_type operator<<(size_t i) const { return bit_type(_bits << i); }
        //bit_type operator>>(size_t i) const { return bit_type(_bits >> i); }

        //bit_type & operator<<=(size_t i) { _bits <<= i; return *this; }
        //bit_type & operator>>=(size_t i) { _bits >>= i; return *this; }

        bit_type & set() { _bits = ~0; return *this; }
        bit_type & reset() { _bits = 0; return *this; }
        bit_type & flip() { _bits = ~_bits; return *this; }
        bit_type & set(size_t i) { _bits |= (1 << i); return *this; }
        bit_type & reset(size_t i) { _bits &= ~(1 << i); return *this; }
        bit_type & flip(size_t i) { _bits ^= (1 << i); return *this; }
        bit_type & set(size_t i, bool v) { if (v) set(i); else reset(i); return *this; }
        bit_type & reset(size_t i, bool v) { if (v) reset(i); else set(i); return *this; }
        bit_type & flip(size_t i, bool v) { if (v) flip(i); return *this; }
        bit_type & set(size_t i, const T &v) { if (v) set(i); else reset(i); return *this; }
        bit_type & reset(size_t i, const T &v) { if (v) reset(i); else set(i); return *this; }
        bit_type & flip(size_t i, const T &v) { if (v) flip(i); return *this; }
        bit_type & set(size_t i, const bit_type &v) { if (v[i]) set(i); else reset(i); return *this; }
        bit_type & reset(size_t i, const bit_type &v) { if (v[i]) reset(i); else set(i); return *this; }
        bit_type & flip(size_t i, const bit_type &v) { if (v[i]) flip(i); return *this; }
        bit_type & set(size_t i, const bit_type &v, bool b) { if (v[i]) set(i, b); else reset(i, b); return *this; }
        bit_type & reset(size_t i, const bit_type &v, bool b) { if (v[i]) reset(i, b); else set(i, b); return *this; }
        bit_type & flip(size_t i, const bit_type &v, bool b) { if (v[i]) flip(i, b); return *this; }
        bit_type & set(size_t i, const bit_type &v, const T &b) { if (v[i]) set(i, b); else reset(i, b); return *this; }
        bit_type & reset(size_t i, const bit_type &v, const T &b) { if (v[i]) reset(i, b); else set(i, b); return *this; }
        bit_type & flip(size_t i, const bit_type &v, const T &b) { if (v[i]) flip(i, b); return *this; }
        bit_type & set(size_t i, const bit_type &v, const bit_type &b) { if (v[i]) set(i, b); else reset(i, b); return *this; }
        bit_type & reset(size_t i, const bit_type &v, const bit_type &b) { if (v[i]) reset(i, b); else set(i, b); return *this; }
         
        friend bool operator==(const T &a, const bit_type &b) { return a == b._bits; }  
        friend bool operator!=(const T &a, const bit_type &b) { return a != b._bits; }
        friend bool operator<(const T &a, const bit_type &b) { return a < b._bits; }
        friend bool operator>(const T &a, const bit_type &b) { return a > b._bits; }
        friend bool operator<=(const T &a, const bit_type &b) { return a <= b._bits; }
        friend bool operator>=(const T &a, const bit_type &b) { return a >= b._bits; }

        friend bool operator==(const bit_type &a, const T &b) { return a._bits == b; }
        friend bool operator!=(const bit_type &a, const T &b) { return a._bits != b; }
        friend bool operator<(const bit_type &a, const T &b) { return a._bits < b; }
        friend bool operator>(const bit_type &a, const T &b) { return a._bits > b; }

        friend bool operator<=(const bit_type &a, const T &b) { return a._bits <= b; }
        friend bool operator>=(const bit_type &a, const T &b) { return a._bits >= b; }
            
        
     };

    typedef std::vector<bit_type<uint8_t,8>> u_bit_vector;
    typedef std::vector<bit_type<uint16_t,16>> u_bit_vector16;
    typedef std::vector<bit_type<uint32_t,32>> u_bit_vector32;
    typedef std::vector<bit_type<uint64_t,64>> u_bit_vector64;
    typedef std::vector<bit_type<uint8_t,8>> u_bit_vector8;
    typedef std::vector<bit_type<int8_t,8>> s_bit_vector8;
    typedef std::vector<bit_type<int16_t,16>> s_bit_vector16;
    typedef std::vector<bit_type<int32_t,32>> s_bit_vector32;
    typedef std::vector<bit_type<int64_t,64>> s_bit_vector64;
    typedef std::vector<bit_type<int8_t,8>> s_bit_vector;

    typedef std::vector<bit_type<float,32>> f_bit_vector32;
    typedef std::vector<bit_type<double,64>> f_bit_vector64;
    typedef std::vector<bit_type<float,32>> f_bit_vector;
    typedef std::vector<bit_type<bool,1>> b_bit_vector;

    template <class T,size_t N>
    std::ostream & operator<<(std::ostream &out, const bit_type<T,N> &b)
    {
        for (size_t i = 0; i < N; i++)
         out << b[i];
       
        return out;
    }   


    template <class T,size_t N>
    std::istream & operator>>(std::istream &in, bit_type<T,N> &b)
    {
        for (size_t i = 0; i < N; i++)
        {
            char c;
            in >> c;
            b[i] = c == '1';
        }
        return in;
    }   

    template <class T,size_t N>
    std::string to_string(const bit_type<T,N> &b)
    {
        std::stringstream ss;
        ss << b;
        return ss.str();
    }   

    template <class T,size_t N> 
    std::string to_string(const bit_type<T,N> &b, size_t n)
    {
        std::stringstream ss;
        for (size_t i = 0; i < n; i++)
            ss << b[i];
        return ss.str();
    }   
        

    #pragma pack()

}

#endif 