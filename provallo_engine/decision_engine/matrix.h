/*
 * matrix.h
 *
 *  Created on: May 11, 2021
 *      Author: kardon
 */

#ifndef DECISION_ENGINE_MATRIX_H_
#define DECISION_ENGINE_MATRIX_H_

#include <iostream>
#include <memory>
#include <numeric>
#include <algorithm>
#include <cstddef>
#include <cassert>
#include <stdexcept>
#include <vector>
#include <random>
#include <cstdlib>
#include <complex>
#include <cmath>
#include <functional>
#include <type_traits>
#include <iterator>
#include <initializer_list>
#include <utility>
#include <string>
#include <sstream>
#include <fstream>
#include "utils.h" //real_t
// simple matrix and NRC/NRCPP/eigen compat matrix.
  
namespace provallo
{
#if 0
 class pro_random
 {
   std::random_device rd;
   std::mt19937 gen;
   std::uniform_real_distribution<double> _urd;
   std::uniform_int_distribution<uint64_t> _u;

   pro_random():rd,gen(rd) {}
   pro_random(long unsigned int seed):rd,gen(rd(seed)) {}

     pro_random(long unsigned int seed) : rd(), gen(rd()),_urd(0, RAND_MAX - 1) {}
     pro_random(const std::random_device& r) : rd(r), uint(0, RAND_MAX - 1) {}
     pro_random(const std::random_device& r,long unsigned int seed) : rd(r), uint(0, RAND_MAX - 1) {}
     pro_random(const pro_random& other) : rd(other.rd), _u(other._u), _urd(other._urd) {}
     // Uniform sampling
     double uniform() {return 1.0 - _u(rd);}
     // Get integer
     unsigned int integer() {return _u(rd);}
     // Jump on sequence
     void jump(size_t value) {/*rd.jump(value);*/}
     // Split sequence
     void split(size_t size, size_t stream) {/*rd.split(size,stream);*/}
     // Seed the generator
     void seed(size_t s) {/*rd.seed((long unsigned int)s);*/}
     ~pro_random(){}
 };

#endif

  // Array :
  template <class T, std::size_t N>
  class array
  {
  public:
    T elems[N]; // fixed-size array of elements of type T

  public:
    // type definitions
    typedef T value_type;
    typedef T *iterator;
    typedef const T *const_iterator;
    typedef T &reference;
    typedef const T &const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    // iterator support
    iterator
    begin()
    {
      return elems;
    }
    const_iterator
    begin() const
    {
      return elems;
    }
    const_iterator
    cbegin() const
    {
      return elems;
    }

    iterator
    end()
    {
      return elems + N;
    }
    const_iterator
    end() const
    {
      return elems + N;
    }
    const_iterator
    cend() const
    {
      return elems + N;
    }

    // Default constructor
    array()
    {
      std::fill(elems, elems + N, T());
    }

    // reverse iterator support
    typedef std::reverse_iterator<iterator> reverse_iterator;
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

    reverse_iterator
    rbegin()
    {
      return reverse_iterator(end());
    }
    const_reverse_iterator
    rbegin() const
    {
      return const_reverse_iterator(end());
    }
    const_reverse_iterator
    crbegin() const
    {
      return const_reverse_iterator(end());
    }

    reverse_iterator
    rend()
    {
      return reverse_iterator(begin());
    }
    const_reverse_iterator
    rend() const
    {
      return const_reverse_iterator(begin());
    }
    const_reverse_iterator
    crend() const
    {
      return const_reverse_iterator(begin());
    }

    // operator[]
    reference
    operator[](size_type i)
    {
      assert(i < N);
      return elems[i];
    }

    const_reference
    operator[](size_type i) const
    {
      assert(i < N);
      return elems[i];
    }
    const array<T, N> &operator*(const T &value) const;
    const array<T, N> &operator/(const T &value) const;

    // at() with range check
    reference
    at(size_type i)
    {
      rangecheck(i);
      return elems[i];
    }
    const_reference
    at(size_type i) const
    {
      rangecheck(i);
      return elems[i];
    }

    // front() and back()
    reference
    front()
    {
      return elems[0];
    }

    const_reference
    front() const
    {
      return elems[0];
    }

    reference
    back()
    {
      return elems[N - 1];
    }

    const_reference
    back() const
    {
      return elems[N - 1];
    }

    // size is constant
    static size_type
    size()
    {
      return N;
    }
    static bool
    empty()
    {
      return false;
    }
    static size_type
    max_size()
    {
      return N;
    }
    enum
    {
      static_size = N
    };

    // swap (note: linear complexity)
    void
    swap(array<T, N> &y)
    {
      for (size_type i = 0; i < N; ++i)
        std::swap(elems[i], y.elems[i]);
    }

    // direct access to data (read-only)
    const T *
    data() const
    {
      return elems;
    }
    T *
    data()
    {
      return elems;
    }

    // use array as C array (direct read/write access to data)
    T *
    c_array()
    {
      return elems;
    }

    // assignment with type conversion
    template <typename T2>
    array<T, N> &
    operator=(const array<T2, N> &rhs)
    {
      std::copy(rhs.begin(), rhs.end(), begin());
      return *this;
    }

    // assign one value to all elements
    void
    assign(const T &value)
    {
      fill(value); // A synonym for fill
    }

    void
    fill(const T &value)
    {
      std::fill_n(begin(), size(), value);
    }

  private:
    // check range (may be private because it is static)
    static void
    rangecheck(size_type i)
    {
      if (i >= size())
      {
        std::out_of_range e("array<>: index out of range");
        throw e;
      }
    }
  };

  // comparisons
  template <class T, std::size_t N>
  bool
  operator==(const array<T, N> &x, const array<T, N> &y)
  {
    return std::equal(x.begin(), x.end(), y.begin());
  }
  template <class T, std::size_t N>
  bool
  operator<(const array<T, N> &x, const array<T, N> &y)
  {
    return std::lexicographical_compare(x.begin(), x.end(), y.begin(),
                                        y.end());
  }
  template <class T, std::size_t N>
  bool
  operator!=(const array<T, N> &x, const array<T, N> &y)
  {
    return !(x == y);
  }
  template <class T, std::size_t N>
  bool
  operator>(const array<T, N> &x, const array<T, N> &y)
  {
    return y < x;
  }
  template <class T, std::size_t N>
  bool
  operator<=(const array<T, N> &x, const array<T, N> &y)
  {
    return !(y < x);
  }
  template <class T, std::size_t N>
  bool
  operator>=(const array<T, N> &x, const array<T, N> &y)
  {
    return !(x < y);
  }

  // global swap()
  template <class T, std::size_t N>
  inline void
  swap(array<T, N> &x, array<T, N> &y)
  {
    x.swap(y);
  }

  template <class T, std::size_t N>
  array<T, N>
  operator*(const T &scalar, const array<T, N> &b)
  {
    array<T, N> result;
    std::transform(b.begin(), b.end(), result.begin(),
                   std::bind1st(std::multiplies<T>(), scalar));
    return result;
  }

  template <class T, std::size_t N>
  array<T, N>
  operator*(const array<T, N> &a, const array<T, N> &b)
  {
    array<T, N> result;
    std::transform(a.begin(), a.end(), b.begin(), result.begin(),
                   std::multiplies<T>());
    return result;
  }

  template <class T, std::size_t N>
  array<T, N>
  operator+(const array<T, N> &a, const array<T, N> &b)
  {
    array<T, N> result;
    std::transform(a.begin(), a.end(), b.begin(), result.begin(),
                   std::plus<T>());
    return result;
  }

  template <class T, std::size_t N>
  array<T, N>
  operator-(const array<T, N> &a, const array<T, N> &b)
  {
    array<T, N> result;
    std::transform(a.begin(), a.end(), b.begin(), result.begin(),
                   std::minus<T>());
    return result;
  }

  template <class T, std::size_t N>
  T dot(const array<T, N> &a, const array<T, N> &b)
  {
    return std::inner_product(a.begin(), a.end(), b.begin(), T());
  }

  template <class T, std::size_t N>
  std::ostream &
  operator<<(std::ostream &out, const array<T, N> &a)
  {
    std::size_t i = 0;
    out << std::string("(");
    for (; i < N - 1; ++i)
      out << a[i] << ',';
    out << a[i] << std::string(")");
    return out;
  }

  // comparisons
  template <class T>
  bool
  operator==(const std::vector<T> &x, const std::vector<T> &y)
  {
    return std::equal(x.begin(), x.end(), y.begin());
  }
  template <class T>
  bool
  operator<(const std::vector<T> &x, const std::vector<T> &y)
  {
    return std::lexicographical_compare(x.begin(), x.end(), y.begin(),
                                        y.end());
  }
  template <class T>
  bool
  operator!=(const std::vector<T> &x, const std::vector<T> &y)
  {
    return !(x == y);
  }
  template <class T>
  bool
  operator>(const std::vector<T> &x, const std::vector<T> &y)
  {
    return y < x;
  }
  template <class T>
  bool
  operator<=(const std::vector<T> &x, const std::vector<T> &y)
  {
    return !(y < x);
  }
  template <class T>
  bool
  operator>=(const std::vector<T> &x, const std::vector<T> &y)
  {
    return !(x < y);
  }

  template <class T>
  std::vector<T>
  operator*(const T &scalar, const std::vector<T> &b)
  {
    std::vector<T> result(b.size());
    std::transform(b.begin(), b.end(), result.begin(),
                   std::bind1st(std::multiplies<T>(), scalar));
    return result;
  }

  template <class T>
  std::vector<T>
  operator*(const std::vector<T> &a, const std::vector<T> &b)
  {
    std::vector<T> result(b.size());
    std::transform(a.begin(), a.end(), b.begin(), result.begin(),
                   std::multiplies<T>());
    return result;
  }

  template <class T>
  std::vector<T>
  operator+(const std::vector<T> &a, const std::vector<T> &b)
  {
    std::vector<T> result(b.size());
    std::transform(a.begin(), a.end(), b.begin(), result.begin(),
                   std::plus<T>());
    return result;
  }

  template <class T>
  std::vector<T>
  operator-(const std::vector<T> &a, const std::vector<T> &b)
  {
    std::vector<T> result(b.size());
    std::transform(a.begin(), a.end(), b.begin(), result.begin(),
                   std::minus<T>());
    return result;
  }

  template <class T>
  T dot(const std::vector<T> &a, const std::vector<T> &b)
  {
    return std::inner_product(a.begin(), a.end(), b.begin(), T());
  }

  template <class T>
  std::ostream &
  operator<<(std::ostream &out, const std::vector<T> &a)
  {
    std::size_t i = 0;
    out << std::string("(");
    for (; i < a.size() - 1; ++i)
      out << a[i] << std::string(",");
    out << a[i] << std::string(")");
    return out;
  }

  class matrix_base;

  typedef std::shared_ptr<matrix_base> matrix_ptr;

  
  class matrix_base
  {
  private:
    double *data;

  public:
    size_t _rows;
    size_t _cols;
    double &
    pos(size_t row, size_t col)
    {
      return data[row * _cols + col];
    }
    const double & pos(size_t row,size_t col)const 
    {
      return data[row * _cols + col];
    }
    inline  size_t rows()const{return _rows;}
    inline  size_t cols()const{return _cols;}

    inline double & operator()(size_t row, size_t col)
    {
      return data[row * _cols + col];
    }

    inline double & operator()(size_t row, size_t col)const
    {
      return data[row * _cols + col];
    }
     matrix_base(size_t rows, size_t cols);
    virtual ~matrix_base();
    virtual void
    clear();
  };
  //note : matrix is not derived from matrix_base, since matrix_base is  dataset oriented and matrix<T>is comutation oriented.
  // 
  template <class T>
  class matrix
  {
    typedef T *ptr;

  public:
    typedef T value_type;
    typedef T *array_type;
    typedef T *iterator;
    typedef const T *const_iterator;
    typedef T &reference;
    typedef const T &const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;



    matrix() : size1_(0), size2_(0), data_(nullptr)
    {
    }
    matrix(size_type size1, size_type size2) : size1_(size1), size2_(size2), data_(nullptr)
    {

      data_ = new T[size1 * size2];
    }
    matrix(size_type size1, size_type size2, const T *value) : size1_(size1), size2_(size2), data_(nullptr)
    {
      data_ = new T[size1 * size2];
      std::copy(value, value + size1 * size2, data_);
    }
    matrix(matrix_base &m) : size1_(m.rows()), size2_(m.cols()), data_(nullptr)
    {
      data_ = new T[size1_ * size2_];
      for (size_t i = 0; i < size1_; i++)
        for (size_t j = 0; j < size2_; j++)
          element(i, j) = m(i, j);
    }

    matrix(const matrix &m) : size1_(m.size1_), size2_(m.size2_), data_(nullptr)
    {
      size_t a_size = size1_ * size2_;
      data_ = new T[a_size];
      std::copy(m.data_, m.data_ + a_size, data_);
    }
    matrix(const matrix &&move_matrix) noexcept :

                                                  size1_(move_matrix.size1_), size2_(move_matrix.size2_), // explicit move of a member of non-class type
				  
    data_(std::move(move_matrix.data_))                   // explicit move of a member of class type
    {
    }
    matrix(const std::initializer_list<T> &list) : size1_(list.size()), size2_(1), data_(nullptr)
    {
      data_ = new T[size1_];
      std::copy(list.begin(), list.end(), data_);
    }
    matrix(const std::initializer_list<std::initializer_list<T>> &list) : size1_(list.size()), size2_(list.begin()->size()), data_(nullptr)
    {
      data_ = new T[size1_ * size2_];
      size_t i = 0;
      for (auto &row : list)
      {
        std::copy(row.begin(), row.end(), data_ + i * size2_);
        ++i;
      }
    }
    matrix(const std::vector<T> &vec) : size1_(vec.size()), size2_(1), data_(nullptr)
    {
      data_ = new T[size1_];
      std::copy(vec.begin(), vec.end(), data_);
    }
    matrix(const std::vector<std::vector<T>> &vec) : size1_(vec.size()), size2_(vec.begin()->size()), data_(nullptr)
    {
      data_ = new T[size1_ * size2_];
      size_t i = 0;
      for (auto &row : vec)
      {
        std::copy(row.begin(), row.end(), data_ + i * size2_);
        ++i;
      }
    }
    matrix(const std::vector<std::vector<T>> &&vec) : size1_(vec.size()), size2_(vec.begin()->size()), data_(nullptr)
    {
      data_ = new T[size1_ * size2_];
      size_t i = 0;
      for (auto &row : vec)
      {
        std::copy(row.begin(), row.end(), data_ + i * size2_);
        ++i;
      }
    }
    matrix(const std::vector<T> &&vec) : size1_(vec.size()), size2_(1), data_(nullptr)
    {
      data_ = new T[size1_];
      std::copy(vec.begin(), vec.end(), data_);
    }
    matrix(const std::vector<std::initializer_list<T>> &vec) : size1_(vec.size()), size2_(vec.begin()->size()), data_(nullptr)
    {
      data_ = new T[size1_ * size2_];
      size_t i = 0;
      for (auto &row : vec)
      {
        std::copy(row.begin(), row.end(), data_ + i * size2_);
        ++i;
      }
    }
    matrix(const std::vector<std::initializer_list<T>> &&vec) : size1_(vec.size()), size2_(vec.begin()->size()), data_(nullptr)
    {
      data_ = new T[size1_ * size2_];
      size_t i = 0;
      for (auto &row : vec)
      {
        std::copy(row.begin(), row.end(), data_ + i * size2_);
        ++i;
      }
    }

    #ifdef SIM_MATRIX_ITERATOR
     
    class const_iterator : public std::iterator<std::random_access_iterator_tag, T>
    {
      private:

      T *ptr_;
      size_t size1_;
      size_t size2_;
      size_t i_;
      size_t j_;

      public:
      const_iterator(T *ptr, size_t size1, size_t size2, size_t i, size_t j) : ptr_(ptr), size1_(size1), size2_(size2), i_(i), j_(j)
      {
      }
      const_iterator(const const_iterator &it) : ptr_(it.ptr_), size1_(it.size1_), size2_(it.size2_), i_(it.i_), j_(it.j_)
      {
      } 
      const_iterator(const_iterator &&it) : ptr_(it.ptr_), size1_(it.size1_), size2_(it.size2_), i_(it.i_), j_(it.j_)
      {
      }
      const_iterator &operator=(const const_iterator &it)
      {
        ptr_ = it.ptr_;
        size1_ = it.size1_;
        size2_ = it.size2_;
        i_ = it.i_;
        j_ = it.j_;
        return *this;
      }
      const_iterator &operator=(const_iterator &&it)
      {
        ptr_ = it.ptr_;
        size1_ = it.size1_;
        size2_ = it.size2_;
        i_ = it.i_;
        j_ = it.j_;
        return *this;
      }
      const_iterator &operator++()
      {
        if (j_ < size2_ - 1)
          ++j_;
        else
        {
          j_ = 0;
          ++i_;
        }
        return *this;
      }
      const_iterator operator++(int)
      {
        const_iterator tmp(*this);
        operator++();
        return tmp;
      }
      const_iterator &operator--()
      {
        if (j_ > 0)
          --j_;
        else
        {
          j_ = size2_ - 1;
          --i_;
        }
        return *this;
      }
      const_iterator operator--(int)
      {
        const_iterator tmp(*this);
        operator--();
        return tmp;
      }
      const_iterator &operator+=(size_t n)
      {
        size_t k = i_ * size2_ + j_ + n;
        i_ = k / size2_;
        j_ = k % size2_;
        return *this;
      }
      const_iterator &operator-=(size_t n)
      {
        size_t k = i_ * size2_ + j_ - n;
        i_ = k / size2_;
        j_ = k % size2_;
        return *this;
      }
      const_iterator operator+(size_t n) const
      {
        const_iterator tmp(*this);
        tmp += n;
        return tmp;
      }
      const_iterator operator-(size_t n) const
      {
        const_iterator tmp(*this);
        tmp -= n;
        return tmp;
      }
      size_t operator-(const const_iterator &it) const
      {
        return i_ * size2_ + j_ - it.i_ * size2_ - it.j_;
      }
      bool operator==(const const_iterator &it) const
      {
        return ptr_ == it.ptr_ && i_ == it.i_ && j_ == it.j_;
      }
      bool operator!=(const const_iterator &it) const
      {
        return ptr_ != it.ptr_ || i_ != it.i_ || j_ != it.j_;
      }
      bool operator<(const const_iterator &it) const
      {
        return ptr_ == it.ptr_ && i_ == it.i_ && j_ < it.j_;
      }
      bool operator>(const const_iterator &it) const
      {
        return ptr_ == it.ptr_ && i_ == it.i_ && j_ > it.j_;
      }
      bool operator<=(const const_iterator &it) const
      {
        return ptr_ == it.ptr_ && i_ == it.i_ && j_ <= it.j_;
      }
      bool operator>=(const const_iterator &it) const
      {
        return ptr_ == it.ptr_ && i_ == it.i_ && j_ >= it.j_;
      }
      const T &operator*() const
      {
        return *(ptr_ + i_ * size2_ + j_);
      }
      const T *operator->() const
      {
        return ptr_ + i_ * size2_ + j_;
      }
      const T &operator[](size_t n) const
      {
        return *(ptr_ + i_ * size2_ + j_ + n);
      }

    };

    const_iterator begin() const
    {
      return const_iterator(data_, size1_, size2_, 0, 0);
    }
    const_iterator end() const
    {
      return const_iterator(data_, size1_, size2_, size1_, 0);
    }
    const_iterator cbegin() const
    {
      return const_iterator(data_, size1_, size2_, 0, 0);
    }
    const_iterator cend() const
    {
      return const_iterator(data_, size1_, size2_, size1_, 0);
    }
    #endif // ITERATOR


    T row_sum(size_t row)const
    {
      T sum = 0;
      for (size_t i = 0; i < size2_; i++)
        sum += data_[row * size2_ + i];
      return sum;
    }
    T col_sum(size_t col)const
    {
      T sum = 0;
      for (size_t i = 0; i < size1_; i++)
        sum += data_[i * size2_ + col];
      return sum;
    }

    T*& row_begin(size_t row)
    {
      
      return data_ + row * size2_;
    }
    T*& row_end(size_t row)
    {
      return data_ + (row + 1) * size2_;
    }
    T*& col_begin(size_t col)
    {
      return data_ + col;
    }
    T*& col_end(size_t col)
    {
      return data_ + size1_ * size2_ + col;
    }
    void remove_column(size_t col)
    {
      size_t new_col_size = size2_ - 1;

      T* tmp = new T[size1_ * new_col_size];
      for (size_t i = 0; i < size1_; i++)
      {
        for (size_t j = 0; j < col; j++)
          tmp[i * new_col_size + j] = data_[i * size2_ + j];
        for (size_t j = col; j < new_col_size; j++)
          tmp[i * new_col_size + j] = data_[i * size2_ + j + 1];
      }
      delete[] data_;
      data_ = tmp;
      size2_--;

    }
    std::vector<real_t> row_entropy()const 
    {
      //returns the entropy of each column of the matrix as a vector
      std::vector<real_t> entropy(size1_);
      for (size_t i = 0; i < size1_; i++)
      {
        entropy[i] = 0;
        for (size_t j = 0; j < size2_; j++)
          entropy[i] += data_[i * size2_ + j] * std::log(data_[i * size2_ + j]);
        entropy[i] = -entropy[i];
      }
      return entropy;

    }
    std::vector<real_t> col_entropy()const
    {
      //returns the entropy of each column of the matrix as a vector
      std::vector<real_t> entropy(size2_);
      for (size_t j = 0; j < size2_; j++)
      {
        entropy[j] = 0;
        for (size_t i = 0; i < size1_; i++)
          entropy[j] += data_[i * size2_ + j] * log(data_[i * size2_ + j]);
        entropy[j] = -entropy[j];
      }
      return entropy;

    } 
    void remove_row(size_t row)
    {
        
        size_t new_row_count = size1_ - 1;
        T* tmp = new T[new_row_count * size2_];
        for (size_t i = 0; i < row; i++)
          for (size_t j = 0; j < size2_; j++)
            tmp[i * size2_ + j] = data_[i * size2_ + j];
        for (size_t i = row; i < new_row_count; i++)
          for (size_t j = 0; j < size2_; j++)
            tmp[i * size2_ + j] = data_[(i + 1) * size2_ + j];
        delete[] data_;
        data_ = tmp;
        size1_--;
    }
    virtual ~matrix()
    {
      if (data_ != nullptr)
        delete[] data_;
      data_ = nullptr;
    }

    const matrix<T> operator*(const T &value) const
    {
      matrix<T> ret = *this;

      for (size_t i = 0; i < ret.size1_; i++)
        for (size_t j = 0; j < ret.size2_; j++)
          ret(i, j) = ret(i, j) * value;

      return ret;
    }
    const matrix<T> operator/(const T &value) const
    {
      matrix<T> ret = *this;

      for (size_t i = 0; i < ret.size1_; i++)
        for (size_t j = 0; j < ret.size2_; j++)
          ret(i, j) = ret(i, j) / value;

      return ret;
    }
    const matrix<T> operator-(const T &value) const
    {
      matrix<T> ret = *this;

      for (size_t i = 0; i < ret.size1_; i++)
        for (size_t j = 0; j < ret.size2_; ++j)
          ret(i, j) = ret(i, j) - value;

      return ret;
    }

    const matrix<T> operator+(const T &value) const
    {
       matrix<T> ret = *this;
       for (size_t i = 0; i < ret.size1_; i++)
        for (size_t j = 0; j < ret.size2_; j++)
          ret(i, j) = ret(i, j) + value;

      return ret;
    }
    const matrix<T> operator*(const matrix<T> &m) const
    {
      matrix<T> ret = *this;

      for (size_t i = 0; i < ret.size1_; i++)
        for (size_t j = 0; j < ret.size2_; j++)
          ret(i, j) = ret(i, j) * m(i, j);

      return ret;
    }
    const matrix<T> operator/(const matrix<T> &m) const
    {
      matrix<T> ret = *this;

      for (size_t i = 0; i < ret.size1_; i++)
        for (size_t j = 0; j < ret.size2_; j++)
          ret(i, j) = ret(i, j) / m(i, j);

      return ret;
    }
    const_reference
    operator()(size_type i, size_type j) const
    {
      return data_[i * size2_ + j];
    }

    reference
    element(size_type i, size_type j)
    {
      return data_[i * size2_ + j];
    }
    reference
    operator()(size_type i, size_type j)
    {
      return data_[i * size2_ + j];
    }
    reference
    insert_element(size_type i, size_type j, const_reference t)
    {
      return (element(i, j) = t);
    }
    void
    erase_element(size_type i, size_type j)
    {
      element(i, j) = value_type();
    }
    void
    clear()
    {
      std::fill(data_, data_ + size1_ * size2_, value_type());
    }
    matrix &
    operator=(const matrix &m)
    {
      size1_ = m.size1_;
      size2_ = m.size2_;
      if (data_ != nullptr)
        delete[] data_;
      data_ = new T[size1_ * size2_];
      std::copy(m.data_, m.data_ + size1_ * size2_, data_);
      return *this;
    }
    matrix &operator=(const std::vector<T> &vec)
    {
      size1_ = vec.size();
      size2_ = 1;
      if (data_ != nullptr)
        delete[] data_;
      data_ = new T[size1_];
      std::copy(vec.begin(), vec.end(), data_);
      return *this;
    }
    matrix &operator=(const std::vector<std::vector<T>> &vec)
    {
      size1_ = vec.size();
      size2_ = vec.begin()->size();
      if (data_ != nullptr)
        delete[] data_;
      data_ = new T[size1_ * size2_];
      size_t i = 0;
      for (auto &row : vec)
      {
        std::copy(row.begin(), row.end(), data_ + i * size2_);
        ++i;
      }
      return *this;
    }
    matrix &operator=(const std::vector<std::initializer_list<T>> &vec)
    {
      size1_ = vec.size();
      size2_ = vec.begin()->size();
      if (data_ != nullptr)
        delete[] data_;
      data_ = new T[size1_ * size2_];
      size_t i = 0;
      for (auto &row : vec)
      {
        std::copy(row.begin(), row.end(), data_ + i * size2_);
        ++i;
      }
      return *this;
    }
    matrix &operator=(const std::vector<std::initializer_list<T>> &&vec)
    {
      size1_ = vec.size();
      size2_ = vec.begin()->size();
      if (data_ != nullptr)
        delete[] data_;
      data_ = new T[size1_ * size2_];
      size_t i = 0;
      for (auto &row : vec)
      {
        std::copy(row.begin(), row.end(), data_ + i * size2_);
        ++i;
      }
      return *this;
    }
    matrix &operator=(const std::initializer_list<std::initializer_list<T>> &vec)
    {
      size1_ = vec.size();
      size2_ = vec.begin()->size();
      if (data_ != nullptr)
        delete[] data_;
      data_ = new T[size1_ * size2_];
      size_t i = 0;
      for (auto &row : vec)
      {
        std::copy(row.begin(), row.end(), data_ + i * size2_);
        ++i;
      }
      return *this;
    }
    matrix &operator=(const std::initializer_list<std::initializer_list<T>> &&vec)
    {
      size1_ = vec.size();
      size2_ = vec.begin()->size();
      if (data_ != nullptr)
        delete[] data_;
      data_ = new T[size1_ * size2_];
      size_t i = 0;
      for (auto &row : vec)
      {
        std::copy(row.begin(), row.end(), data_ + i * size2_);
        ++i;
      }
      return *this;
    }
    matrix &operator=(const std::initializer_list<T> &vec)
    {
      size1_ = vec.size();
      size2_ = 1;
      if (data_ != nullptr)
        delete[] data_;
      data_ = new T[size1_];
      std::copy(vec.begin(), vec.end(), data_);
      return *this;
    }
    matrix &operator=(const std::initializer_list<T> &&vec)
    {
      size1_ = vec.size();
      size2_ = 1;
      if (data_ != nullptr)
        delete[] data_;
      data_ = new T[size1_];
      std::copy(vec.begin(), vec.end(), data_);
      return *this;
    }
    matrix &operator=(const std::vector<T> &&vec)
    {
      size1_ = vec.size();
      size2_ = 1;
      if (data_ != nullptr)
        delete[] data_;
      data_ = new T[size1_];
      std::copy(vec.begin(), vec.end(), data_);
      return *this;
    }

    void
    resize(size_type dim, size_type dim1)
    {
      if (data_ != nullptr)
        delete[] data_;
      size1_ = dim;
      size2_ = dim1;
      data_ = new T[size1_ * size2_];
    }
    size_type
    size1() const
    {
      return size1_;
    }
    size_type
    size2() const
    {
      return size2_;
    }

    array_type row(size_t i)
    {
      array_type ret  = reinterpret_cast<array_type>( data_ + (i * cols()));    
      return ret;
      
    }

     array_type row(size_t i) const
    {
      array_type ret  = reinterpret_cast<array_type>(
          data() + (i * cols()));

      return ret;
    }
 
 
    template <size_t N> matrix<T> lpNorm() const 
    {
      matrix<T> ret(size1(), size2());
      for (size_t i = 0; i < size1(); i++)
        for (size_t j = 0; j < size2(); j++)
          ret(i, j) = std::pow(std::abs((*this)(i, j)), N);
      return ret;
    }

     matrix<T> lpNorm(double N) const
    {
      matrix<T> ret(size1(), size2());
      for (size_t i = 0; i < size1(); i++)
        for (size_t j = 0; j < size2(); j++)
          ret(i, j) = std::pow(std::abs((*this)(i, j)), N);
      return ret;
    }
    matrix<T> lpNorm(int N) const
    {
      matrix<T> ret(size1(), size2());
      for (size_t i = 0; i < size1(); i++)
        for (size_t j = 0; j < size2(); j++)
          ret(i, j) = std::pow(std::abs((*this)(i, j)), N);
      return ret;
    } 
    matrix<T> lpNorm(float N) const
    {
      matrix<T> ret(size1(), size2());
      for (size_t i = 0; i < size1(); i++)
        for (size_t j = 0; j < size2(); j++)
          ret(i, j) = std::pow(std::abs((*this)(i, j)), N);
      return ret;
    } 
    matrix<T> lpNorm(long double N) const
    {
      matrix<T> ret(size1(), size2());
      for (size_t i = 0; i < size1(); i++)
        for (size_t j = 0; j < size2(); j++)
          ret(i, j) = std::pow(std::abs((*this)(i, j)), N);
      return ret;
    } 
    matrix<T> lpNorm(long long int N) const
    {
      matrix<T> ret(size1(), size2());
      for (size_t i = 0; i < size1(); i++)
        for (size_t j = 0; j < size2(); j++)
          ret(i, j) = std::pow(std::abs((*this)(i, j)), N);
      return ret;
    } 

    size_type
    rows() const
    {
      return size1();
    }
    size_type
    cols() const
    {
      return size2();
    }
    array_type &
    data()
    {

      return data_;
    }


    const array_type &
    data()const
    {

      return data_;
    }
    void fill(const T &val)
    {
      for (size_t i = 0; i < size1_; i++)
        for (size_t j = 0; j < size2_; j++)
          (*this)(i, j) = val;
    }
    inline matrix<real_t> covariance()  const
    { 
      matrix<real_t> ret(size2(), size2());
      for(size_t i=0;i<size2();i++)
        for(size_t j=0;j<size2();j++)
          ret(i,j) = covariance(i,j); 
      return ret;
    }
    real_t covariance(size_t i,size_t j) const
    {
      return covariance(row(i),row(j));
    }
    real_t covariance(array_type a,array_type b) const
    {
      real_t ret = 0;
      for(size_t i=0;i<size1();i++)
        ret += (a[i]-mean(a))*(b[i]-mean(b));
      return ret/(size1()-1);
    }
    real_t mean(const array_type &a) const
    {
      real_t ret = 0;
      for(size_t i=0;i<size1();i++)
        ret += a[i];
      return ret/size1();
    } 
    T mean() const
    {
      T ret = 0;
      for(size_t i=0;i<size1();i++)
        for(size_t j=0;j<size2();j++)
          ret += data_[i*size2()+j];
      
      return ret/(size1()*size2()); 

    }
    T median()const 
    {
      std::vector<T> tmp(size1()*size2());
      for(size_t i=0;i<size1();i++)
        for(size_t j=0;j<size2();j++)
          tmp[i*size2()+j] = data_[i*size2()+j];
      std::sort(tmp.begin(),tmp.end());
      return tmp[tmp.size()/2]; 

    }
    inline matrix<real_t> correlation() const
    {
      matrix<real_t> ret(size2(), size2());
      for(size_t i=0;i<size2();i++)
        for(size_t j=0;j<size2();j++)
          ret(i,j) = correlation(i,j); 
      return ret;
    } 
    real_t correlation(size_t i,size_t j) const
    {
      //returm correlation(row(i),row(j)); 
      return correlation(data_+i*size2(),data_+j*size2());

    }
    real_t correlation(const array_type &a,const  array_type &b) const
    {
      real_t ret = 0;
      for(size_t i=0;i<size1();i++)
        ret += (a[i]-mean(a))*(b[i]-mean(b));
      return ret/(size1()-1);
    }
    //correlation_coefficient (x(i)-mean(x))*(y(i)-mean(y)) / ((x(i)-mean(x))2 * (y(i)-mean(y))2.
    inline
     matrix<real_t> correlation_coefficient()const
    {
      matrix<real_t> ret(size2(), size2());
      for(size_t i=0;i<size2();i++)
        for(size_t j=0;j<size2();j++)
         {
            ret(i,j) =std::sqrt( variance(i,j)/variance(i,i)*variance(j,j));  

         } 

      return ret;
    }
 
    inline matrix<real_t> variance()const 
    {
      matrix<real_t> ret(size2(), size2());
      for(size_t i=0;i<size2();i++)
        for(size_t j=0;j<size2();j++)
          ret(i,j) = variance(i,j); 
      return ret;
    }
    real_t variance(size_t i,size_t j)
    {
      return variance(row(i),row(j));
    }
    real_t variance(array_type &a,array_type &b) const
    {
      real_t ret = 0;
      for(size_t i=0;i<size1();i++)
        ret += (a[i]-mean(a))*(b[i]-mean(b));
      return ret/(size1()-1);
    } 
    inline matrix<real_t> std()   const
    {
      matrix<real_t> ret(size2(), size2());
      for(size_t i=0;i<size2();i++)
        for(size_t j=0;j<size2();j++)
          ret(i,j) = std(i,j); 
      return ret;
    } 
    real_t std(size_t i,size_t j) const
    {
      return std(row(i),row(j));
    } 

    real_t std(const array_type &a,const array_type &b) const
    {
      real_t ret = 0;
      for(size_t i=0;i<size1();i++)
        ret += (a[i]-mean(a))*(b[i]-mean(b));
      return ret/(size1()-1);
    }
    real_t col_mean(size_t i) const
    {
      real_t ret = 0.;
      for(size_t j=0;j<size1();j++)
        ret += data_[j*size2()+i ]; // (j,i);
      return ret/size1();
    }
    real_t col_std(size_t i) const
    {
      real_t ret = 0.;
      real_t mean = col_mean(i);
      for(size_t j=0;j<size1();j++)
        ret += (data_[j * size2_ + i]-mean)*(data_[j * size2_ + i]-mean);   // element(j,i)--> data_[j*size2()+i ] data_[j * size2_ + i]
      return std::sqrt(ret/(size1()-1));
    } 
    real_t col_variance(size_t col) const
    {
      real_t ret = 0.;
      real_t mean = col_mean(col);
      for(size_t i=0;i<size1();i++)
        ret += (data_[i * size2_ + col]-mean)*(data_[i * size2_ + col]-mean); // element(i,col)--> data_[i*size2()+col ] data_[i * size2_ + col]
      return ret/(size1()-1);
    }
 

    inline matrix<real_t> eigenvalues() const
    {
      matrix<real_t> ret(size2(), size2());
      for(size_t i=0;i<size2();i++)
        for(size_t j=0;j<size2();j++)
          ret(i,j) = eigen_values(i,j); 
      return ret;
    }
    real_t eigen_values(size_t i,size_t j) const
    {
        real_t ret = 0;
        array_type a = row(i);
        array_type b = row(j);
        for(size_t i=0;i<size1();i++)
          ret += (a[i]-mean(a))*(b[i]-mean(b));
        return ret/(size1()-1);
        
      //return eigen_values(row(i),row(j));
    }
    real_t eigen_values(const array_type &a,const array_type &b)  const
    {
      real_t ret = 0;
      for(size_t i=0;i<size1();i++)
        ret += (a[i]-mean(a))*(b[i]-mean(b));
      return ret/(size1()-1);
    }



    inline matrix<real_t> eigenvectors() const 
    {
      matrix<real_t> ret(size2(), size2());
      for(size_t i=0;i<size2();i++)
        for(size_t j=0;j<size2();j++)
          ret(i,j) = eigen_vectors(i,j); 
      return ret;
    }
    real_t eigen_vectors(size_t i,size_t j) const
    {
      return eigen_vectors(row(i),row(j));
    }
    real_t eigen_vectors(const array_type &a,const array_type &b) const
    {
      real_t ret = 0;
      for(size_t i=0;i<size1();i++)
        ret += (a[i]-mean(a))*(b[i]-mean(b));
      return ret/(size1()-1);
    }
    inline matrix<real_t> eigen_values_and_vectors() const
    {
      matrix<real_t> ret(size2(), size2());
      for(size_t i=0;i<size2();i++)
        for(size_t j=0;j<size2();j++)
          ret(i,j) = eigen_values_and_vectors(i,j); 
      return ret;
    } 
    real_t eigen_values_and_vectors(size_t i,size_t j) const
    {
      return eigen_values_and_vectors(row(i),row(j));
    }
    real_t eigen_values_and_vectors(array_type &a,array_type &b)  const
    {
      real_t ret = 0;
      for(size_t i=0;i<size1();i++)
        ret += (a[i]-mean(a))*(b[i]-mean(b));
      return ret/(size1()-1);
    }
 
    void get_eigen_values_and_vectors(std::vector<T> &eigen_values, matrix<T> &eigen_vectors) const
    {
      //calculate eigen values and vectors

      //eigen values
      eigen_values.resize(size1());
      eigen_vectors.resize(size1(), size1());

      //copy data
      std::copy(data_, data_+(size1_*size2_), eigen_vectors.data().begin());
      //calculate eigen values and vectors
      eigen(eigen_vectors, eigen_values);
      //sort eigen values and vectors
      std::vector<std::pair<T, size_t> > eigen_values_index(size1());
      for (size_t i = 0; i < size1(); ++i)
        eigen_values_index[i] = std::make_pair(eigen_values[i], i);
      std::sort(eigen_values_index.begin(), eigen_values_index.end(), std::greater<std::pair<T, size_t> >());
      //copy sorted eigen values and vectors
      for (size_t i = 0; i < size1(); ++i)
      {
        eigen_values[i] = eigen_values_index[i].first;
        for (size_t j = 0; j < size1(); ++j)
          eigen_vectors(j, i) = eigen_vectors(j, eigen_values_index[i].second);
      }
      //normalize eigen vectors
      for (size_t i = 0; i < size1(); ++i)
      {
        T norm = 0;
        for (size_t j = 0; j < size1(); ++j)
          norm += eigen_vectors(j, i) * eigen_vectors(j, i);
        norm = std::sqrt(norm);
        for (size_t j = 0; j < size1(); ++j)
          eigen_vectors(j, i) /= norm;
      } 
      
      //done  
      return;

    }      
    
    const array_type &
    as_diagonal()
    {

      static array_type diagonal_instance = nullptr;

      if (diagonal_instance)
        delete diagonal_instance;

      diagonal_instance = new T[size1_];

      for (size_t i = 0, j = 0; i < size1_ && j < size2_; ++i, ++j)
        diagonal_instance[i] = data_[i + j];

      return diagonal_instance;
    }

    matrix<T> sqrt()
    {

      matrix<T> sqrt_prod(size2(), size1());
      for (typename matrix<T>::size_type i = 0; i < size1(); i++)
        for (typename matrix<T>::size_type j = 0; j < size2(); j++)
          sqrt_prod(i, j) = std::sqrt(element(i, j));

      return sqrt_prod;
    }
    matrix<T>
    transpose()
    {

      matrix<T> transp(size2(), size1());
      for (typename matrix<T>::size_type i = 0; i < size2(); ++i)
        for (typename matrix<T>::size_type j = 0; j < size1(); ++j)
          transp(i, j) = element(j, i);
      return transp;
    }
    matrix<T> reverse()
    {
      matrix<T> reverse(size1(), size2());
      for (typename matrix<T>::size_type i = 0; i < size2(); ++i)
        for (typename matrix<T>::size_type j = 0; j < size1(); ++j)
          reverse(i, j) = element(size1() - i, size2() - j);
      return reverse;
    }

    iterator
    begin()
    {
      return data_;
    }
    const_iterator
    begin() const
    {
      return data_;
    }
    const_iterator
    cbegin() const
    {
      return data_;
    }

    iterator
    end()
    {
      return data_ + size1_ * size2_;
    }

    const_iterator
    end() const
    {
      return data_ + size1_ * size2_;
    }

    const_iterator
    cend() const
    {
      return data_ + size1_ * size2_;
    }
    T squaredNorm()
    {

      T ret = 0.;
      for (typename matrix<T>::size_type i = 0; i < size1(); ++i)
        for (typename matrix<T>::size_type j = 0; j < size2(); ++j)
          ret += element(i, j) * element(i, j);

      return ret;
    }

    T norm()
    {
      return std::sqrt(squaredNorm());
    }

    T sum() const
    {
      T ret(0.);
      for (typename matrix<T>::size_type i = 0; i < size1(); ++i)
        for (typename matrix<T>::size_type j = 0; j < size2(); ++j)
          ret += this->operator() (i, j);

      return ret;
    }

    T sum_row(size_t row) const
    {
      T ret(0.);
      for (size_t j = 0; j < size2_; j++)
        ret += data_[row * size2_ + j];
      return ret;
    }
    T dot_row(size_t row, const matrix<T> &other) const
    {
      T ret(0.);
      for (size_t j = 0; j < size2_; j++)
        ret += data_[row * size2_ + j] * other(row, j);
      return ret;
    }
    T dot_row(size_t row, const std::vector<T> &other) const
    {
      T ret(0.);
      for (size_t j = 0; j < size2_; j++)
        ret += data_[row * size2_ + j] * other[j];
      return ret;
    }
    // sets zero on all values
    void
    set_zero() { this->clear(); }
    const array_type &
    array() const
    {
      return data_;
    }
    // return 0 matrix

    static matrix
    Zero(const size_t &a, const size_t &b)
    {
      matrix ret(a, b);
      std::fill(ret.begin(), ret.end(), 0.0);
      return ret;
    }
    // fills a matrix with constant 1

    static matrix
    One(const size_t &a, const size_t &b)
    {
      matrix ret(a, b);
      std::fill(ret.begin(), ret.end(), 1.0);
      return ret;
    }

    // fills a matrix with constant value

    static matrix
    Constant(const size_t &a, const size_t &b, float pt)
    {

      matrix ret(a, b);
      for (auto &x : ret)
        x = pt;

      return ret;
    }

    // generates random values between -1 and 1

    static matrix
    Random(const size_t &a, const size_t &b)
    {
      std::random_device dev;
      std::mt19937 gen(dev());
      std::uniform_real_distribution<float> uniform_dist(-1.0, 1.0);

      matrix ret(a, b);
      for (auto &x : ret)
        x = (T)1.0 - uniform_dist(gen);

      return ret;
    }

    inline T *
    operator[](const int i) // subscripting
    {
      return &data_[i * size2_];
    }
    inline const T *
    operator[](const int i) const // readonly
    {
      return &data_[i * size2_];
    }

  private:
    size_type size1_;
    size_type size2_;
    array_type data_;
  };
  template <class T>
  bool
  operator==(const matrix<T> &x, const matrix<T> &y)
  {
    return std::equal(x.begin(), x.end(), y.begin());
  }

  template <class T>
  bool
  operator!=(const matrix<T> &x, const matrix<T> &y)
  {

    return !(x == y);
  }

  template <class T>
  bool
  operator<(const matrix<T> &x, const matrix<T> &y)
  {
    return std::lexicographical_compare(x.begin(), x.end(), y.begin(),
                                        y.end());
  }

  template <class T>
  bool
  operator>(const matrix<T> &x, const matrix<T> &y)
  {
    return y < x;
  }
  template <class T>
  bool
  operator<=(const matrix<T> &x, const matrix<T> &y)
  {
    return !(y < x);
  }
  template <class T>
  bool
  operator>=(const matrix<T> &x, const matrix<T> &y)
  {
    return !(x < y);
  }

  template <class T>
  matrix<T>
  operator*(const T &scalar, const matrix<T> &b)
  {
    matrix<T> result(b);
    std::transform(b.begin(), b.end(), result.begin(),
                   std::bind1st(std::multiplies<T>(), scalar));
    return result;
  }
  template <class T>
  matrix<T>
  operator*(const matrix<T> &a, const std::vector<T> &b)
  {
    assert(a.size2() == b.size());
    std::vector<T> result(a.size1());
    for (size_t i = 0; i < result.size(); ++i)
    {
      T sum(0);
      for (size_t k = 0; k < a.size2(); ++k)
        sum += a(i, k) * b[k];
      result[i] = sum;
    }
    return result;
  }
  template <class T>
  matrix<T>
  operator*(const matrix<T> &a, const matrix<T> &b)
  {
    assert(a.size2() == b.size2());
    matrix<T> result(b.size1(), b.size2());
    for (size_t i = 0; i < b.size1(); ++i)
    {
      for (size_t k = 0; k < b.size2(); ++k)
        result(i, k) = b(i, k) * a(0, k);
    }
    return result;
  }

  template <class T, std::size_t N>
  const array<T, N> &
  array<T, N>::operator*(const T &a) const
  {
    static array<T, N> result = *this;

    for (size_t i = 0; i < result.size(); ++i)
      result[i] = result[i] * a;

    return *result;
  }

  template <class T, std::size_t N>
  const array<T, N> &
  array<T, N>::operator/(const T &a) const
  {
    static array<T, N> result = *this;

    for (size_t i = 0; i < result.size(); ++i)
      result[i] = result[i] / a;

    return *result;
  }

  template <class T, std::size_t N>
  array<T, N>
  operator*(const matrix<T> &a, const array<T, N> &b)
  {
    assert(a.size2() == N);
    array<T, N> result;
    for (size_t i = 0; i < result.size(); ++i)
    {
      T sum(0);
      for (size_t k = 0; k < a.size2(); ++k)
        sum += a(i, k) * b[k];
      result[i] = sum;
    }
    return result;
  }

  template <class T>
  matrix<T>
  operator+(const matrix<T> &a, const matrix<T> &b)
  {
    assert(a.size1() == b.size1() && a.size2() == b.size2());

    matrix<T> result(a);
    std::transform(a.begin(), a.end(), b.begin(), result.begin(),
                   std::plus<T>());
    return result;
  }

  template <class T>
  matrix<T>
  operator-(const matrix<T> &a, const matrix<T> &b)
  {
    assert(a.size1() == b.size1() && a.size2() == b.size2());
    matrix<T> result(a);
    std::transform(a.begin(), a.end(), b.begin(), result.begin(),
                   std::minus<T>());
    return result;
  }

  //for x-matrix  x = 1-x

  template <class T>
    const matrix<T> operator-(const T &lval,const matrix<T>& rval) 
    {
      const size_t size1 = rval.size1(), size2 = rval.size2();
      matrix<T> ret(rval);
      for (size_t i = 0; i < size1; i++)
        for (size_t j = 0; j <  size2 ; ++j)
          ret(i, j) = lval - ret(i, j) ;

      return ret;
    }

  template <class T>
  std::ostream &
  operator<<(std::ostream &out, const matrix<T> &a)
  {
    for (size_t j = 0; j < a.size1(); ++j)
    {
      size_t i = 0;
      out << std::string("(");
      for (; i < a.size2(); ++i)
        out << a(j, i) << "," << a(j, i) << ")";
    }
    return out;
  }
  // Dense matrix implementation with static size
  template <class T, std::size_t N, std::size_t M>
  class bounded_matrix
  {
    typedef T *pointer;

  public:
    // Type definitions
    typedef T value_type;
    typedef T *array_type;
    typedef T *iterator;
    typedef const T *const_iterator;
    typedef T &reference;
    typedef const T &const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    // ---- Construction and destruction

    // Default dense matrix constructor. Make a dense matrix of size (0,0)
    bounded_matrix()
    {
    }

    // Dense matrix constructor with defined size a initial value for all the matrix elements
    bounded_matrix(const value_type &init)
    {
      std::fill(data_, data_ + N * M, init);
    }

    // Dense matrix constructor with defined size and an initial data array
    bounded_matrix(const array_type &data)
    {
      std::copy(data, data + N * M, data_);
    }

    // Copy-constructor of a dense matrix
    bounded_matrix(const bounded_matrix<T, N, M> &m)
    {
      std::copy(m.data_, m.data_ + N * M, data_);
    }

    // iterator support
    iterator
    begin()
    {
      return data_;
    }
    const_iterator
    begin() const
    {
      return data_;
    }
    const_iterator
    cbegin() const
    {
      return data_;
    }

    iterator
    end()
    {
      return data_ + N * M;
    }
    const_iterator
    end() const
    {
      return data_ + N * M;
    }
    const_iterator
    cend() const
    {
      return data_ + N * M;
    }

    // reverse iterator support
    typedef std::reverse_iterator<iterator> reverse_iterator;
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

    reverse_iterator
    rbegin()
    {
      return reverse_iterator(end());
    }
    const_reverse_iterator
    rbegin() const
    {
      return const_reverse_iterator(end());
    }
    const_reverse_iterator
    crbegin() const
    {
      return const_reverse_iterator(end());
    }

    reverse_iterator
    rend()
    {
      return reverse_iterator(begin());
    }
    const_reverse_iterator
    rend() const
    {
      return const_reverse_iterator(begin());
    }
    const_reverse_iterator
    crend() const
    {
      return const_reverse_iterator(begin());
    }

    // ---- Accessors

    // Return the number of rows of the matrix
    static size_type
    size1()
    {
      return N;
    }

    // Return the number of colums of the matrix
    static size_type
    size2()
    {
      return M;
    }
    // Return a constant reference to the internal storage of a dense matrix, i.e. the raw data
    const array_type &
    data() const
    {
      return data_;
    }

    // Return a reference to the internal storage of a dense matrix, i.e. the raw data
    array_type &
    data()
    {
      return data_;
    }

    // Element access
    const_reference
    operator()(size_type i, size_type j) const
    {
      return data_[i * M + j];
    }

    reference
    at_element(size_type i, size_type j)
    {
      return data_[i * M + j];
    }

    reference
    operator()(size_type i, size_type j)
    {
      return data_[i * M + j];
    }

    // Element assignment
    reference
    insert_element(size_type i, size_type j, const_reference t)
    {
      return (at_element(i, j) = t);
    }

    void
    erase_element(size_type i, size_type j)
    {
      at_element(i, j) = value_type();
    }

    // Zeroing
    void
    clear()
    {
      std::fill(data_, data_ + N * M, value_type());
    }

    bounded_matrix<T, N, M> &
    operator=(const bounded_matrix<T, N, M> &m)
    {
      std::copy(m.data_, m.data_ + N * M, data_);
      return *this;
    }

  private:
    T data_[N * M];
  };

  // comparisons
  template <class T, std::size_t N, std::size_t M>
  bool
  operator==(const bounded_matrix<T, N, M> &x,
             const bounded_matrix<T, N, M> &y)
  {
    return std::equal(x.begin(), x.end(), y.begin());
  }
  template <class T, std::size_t N, std::size_t M>
  bool
  operator<(const bounded_matrix<T, N, M> &x,
            const bounded_matrix<T, N, M> &y)
  {
    return std::lexicographical_compare(x.begin(), x.end(), y.begin(),
                                        y.end());
  }
  template <class T, std::size_t N, std::size_t M>
  bool
  operator!=(const bounded_matrix<T, N, M> &x,
             const bounded_matrix<T, N, M> &y)
  {
    return !(x == y);
  }
  template <class T, std::size_t N, std::size_t M>
  bool
  operator>(const bounded_matrix<T, N, M> &x,
            const bounded_matrix<T, N, M> &y)
  {
    return y < x;
  }
  template <class T, std::size_t N, std::size_t M>
  bool
  operator<=(const bounded_matrix<T, N, M> &x,
             const bounded_matrix<T, N, M> &y)
  {
    return !(y < x);
  }
  template <class T, std::size_t N, std::size_t M>
  bool
  operator>=(const bounded_matrix<T, N, M> &x,
             const bounded_matrix<T, N, M> &y)
  {
    return !(x < y);
  }

  template <class T, std::size_t N, std::size_t M>
  bounded_matrix<T, N, M>
  operator*(const T &scalar, const bounded_matrix<T, N, M> &b)
  {
    bounded_matrix<T, N, M> result(b);
    std::transform(b.begin(), b.end(), result.begin(),
                   std::bind1st(std::multiplies<T>(), scalar));
    return result;
  }

  // Naive matrix multiplication
  template <class T, std::size_t N1, std::size_t M1, std::size_t M2>
  bounded_matrix<T, N1, M2>
  operator*(const bounded_matrix<T, N1, M1> &a,
            const bounded_matrix<T, M1, M2> &b)
  {
    bounded_matrix<T, N1, M2> result;
    for (size_t i = 0; i < N1; ++i)
    {
      for (size_t j = 0; j < M2; ++j)
      {
        typename bounded_matrix<T, N1, M2>::value_type sum(0);
        for (size_t k = 0; k < M1; ++k)
          sum += a(i, k) * b(k, j);
        result(i, j) = sum;
      }
    }
    return result;
  }

  // Naive matrix multiplication
  template <class T, std::size_t N, std::size_t M>
  std::vector<T>
  operator*(const bounded_matrix<T, N, M> &a, const std::vector<T> &b)
  {
    // Sanity check (will only work in debug compilation)
    assert(M == b.size());
    std::vector<T> result(N);
    for (size_t i = 0; i < N; ++i)
    {
      T sum(0);
      for (size_t k = 0; k < M; ++k)
        sum += a(i, k) * b[k];
      result[i] = sum;
    }
    return result;
  }

  // Naive matrix multiplication
  template <class T, std::size_t N, std::size_t M>
  array<T, N>
  operator*(const bounded_matrix<T, N, M> &a, const array<T, M> &b)
  {
    array<T, N> result;
    for (size_t i = 0; i < N; ++i)
    {
      T sum(0);
      for (size_t k = 0; k < M; ++k)
        sum += a(i, k) * b[k];
      result[i] = sum;
    }
    return result;
  }

  template <class T, std::size_t N, std::size_t M>
  bounded_matrix<T, N, M>
  operator+(const bounded_matrix<T, N, M> &a,
            const bounded_matrix<T, N, M> &b)
  {
    // Sanity check (will only work in debug compilation)
    assert(a.size1() == b.size1());
    assert(a.size2() == b.size2());
    bounded_matrix<T, N, M> result(a);
    std::transform(a.begin(), a.end(), b.begin(), result.begin(),
                   std::plus<T>());
    return result;
  }

  template <class T, std::size_t N, std::size_t M>
  bounded_matrix<T, N, M>
  operator-(const bounded_matrix<T, N, M> &a,
            const bounded_matrix<T, N, M> &b)
  {
    // Sanity check (will only work in debug compilation)
    assert(a.size1() == b.size1());
    assert(a.size2() == b.size2());
    bounded_matrix<T, N, M> result(a);
    std::transform(a.begin(), a.end(), b.begin(), result.begin(),
                   std::minus<T>());
    return result;
  }

  // Initialize array
  template <class T, std::size_t N>
  void
  resizeArray(array<T, N> *values, size_t dim)
  {
  }

  template <class T>
  void
  resizeArray(std::vector<T> *values, size_t dim)
  {
    values->resize(dim);
  }

  template <typename Function, typename Array>
  static typename Array::value_type
  central_deriv(Function function, Array &point,
                const typename Array::value_type &h,
                typename Array::size_type i,
                typename Array::value_type *abserr_round,
                typename Array::value_type *abserr_trunc)
  {
    typedef typename Array::value_type value_type;

    // Save original value
    value_type orig = point[i];
    value_type eps = std::numeric_limits<value_type>::epsilon();

    // Compute the derivative using the 5-point rule (x-h, x-h/2, x,
    // x+h/2, x+h). Note that the central point is not used.
    // Compute the error using the difference between the 5-point and
    // the 3-point rule (x-h,x,x+h).
    point[i] = orig - h;
    value_type fm1 = function(point);
    point[i] = orig + h;
    value_type fp1 = function(point);

    point[i] = orig - h / 2;
    value_type fmh = function(point);
    point[i] = orig + h / 2;
    value_type fph = function(point);

    value_type r3 = 0.5 * (fp1 - fm1);
    value_type r5 = (4.0 / 3.0) * (fph - fmh) - (1.0 / 3.0) * r3;

    value_type e3 = (fabs(fp1) + fabs(fm1)) * eps;
    value_type e5 = 2.0 * (fabs(fph) + fabs(fmh)) * eps + e3;

    value_type dy = std::max(fabs(r3 / h), fabs(r5 / h)) * (fabs(orig) / h) * eps;

    // The truncation error in the r5 approximation itself is O(h^4).
    // However, for safety, we estimate the error from r5-r3, which is
    // O(h^2).  By scaling h we will minimize this estimated error, not
    // the actual truncation error in r5.
    *abserr_trunc = fabs((r5 - r3) / h);
    *abserr_round = fabs(e5 / h) + dy;

    // Before leave put the original value into the point
    point[i] = orig;

    return r5 / h;
  }

  template <typename Function, typename Array>
  typename Array::value_type
  partial_derivative(Function function, Array &point,
                     const typename Array::value_type &h,
                     typename Array::size_type i)
  {
    typedef typename Array::value_type value_type;
    value_type round, trunc;
    value_type r_0 = central_deriv(function, point, h, i, &round, &trunc);
    value_type error = round + trunc;

    if (round < trunc && (round > 0 && trunc > 0))
    {
      value_type round_opt, trunc_opt;
      // Compute an optimized step-size to minimize the total error,
      // using the scaling of the truncation error (O(h^2)) and
      // rounding error (O(1/h)). */
      value_type h_opt = h * pow(round / (2.0 * trunc), 1.0 / 3.0);
      value_type r_opt = central_deriv(function, point, h_opt, i,
                                       &round_opt, &trunc_opt);
      value_type error_opt = round_opt + trunc_opt;

      // Check that the new error is smaller, and that the new derivative
      // is consistent with the error bounds of the original estimate.
      if (error_opt < error && fabs(r_opt - r_0) < 4.0 * error)
      {
        r_0 = r_opt;
        error = error_opt;
      }
    }

    return r_0;
  }
  // Gradient operator (partial derivative in all components)
  template <typename Function, typename Array>
  Array
  gradient(Function function, Array &point,
           const typename Array::value_type &h)
  {
    
    //typedef typename Array::value_type value_type;
    typedef typename Array::size_type size_type;
    // Initialize and copy array
    Array result(point);
    for (size_type i = 0; i < result.size(); ++i)
      result[i] = partial_derivative(function, point, h, i);
    return result;
  }

  // Partial derivative functor
  template <typename Function, typename Array>
  class PartialDeriv
  {
    typedef typename Array::value_type value_type;
    typedef typename Array::size_type size_type;

    // Internal function
    Function _function;
    // Step
    value_type _h;
    // Variable
    size_type _i;

  public:
    PartialDeriv(const Function &function, const value_type &h,
                 const size_type &i) : _function(function), _h(h), _i(i)
    {
    }

    // Calculate partial derivative
    value_type
    operator()(Array &point) const
    {
      return partial_derivative(_function, point, _h, _i);
    }

    ~PartialDeriv()
    {
    }
  };

  template <typename Function, typename Array>
  class Gradient
  {
    typedef typename Array::value_type value_type;
    typedef typename Array::size_type size_type;

    // Internal function
    Function _function;
    // Step
    value_type _h;

  public:
    Gradient(const Function &function, const value_type &h) : _function(function), _h(h)
    {
    }

    // Calculate partial derivative
    Array
    operator()(Array &point) const
    {
      return gradient(_function, point, _h);
    }

    ~Gradient()
    {
    }
  };

  // Calculate hessian matrix
  template <typename Function, typename Array>
  matrix<typename Array::value_type>
  hessian(Function function, Array &point,
          const typename Array::value_type &h)
  {
    // Hessian matrix
    matrix<typename Array::value_type> mhessian(point.size(),
                                                point.size());
    // Loop over each element of the matrix
    for (typename Array::size_type i = 0; i < point.size(); ++i)
    {
      PartialDeriv<Function, Array> first_deriv(function, h, i);
      for (typename Array::size_type j = 0; j < point.size(); ++j)
      {
        // Calculate second derivative (i,j)
        PartialDeriv<PartialDeriv<Function, Array>, Array> second_deriv(
            first_deriv, h, j);
        mhessian(i, j) = second_deriv(point);
      }
    }
    // Return result
    return mhessian;
  }

  // Calculate hessian matrix
  template <typename Function, typename FloatType, std::size_t N>
  bounded_matrix<FloatType, N, N>
  bounded_hessian(Function function, array<FloatType, N> &point,
                  const typename array<FloatType, N>::value_type &h)
  {
    // Hessian matrix
    bounded_matrix<FloatType, N, N> mhessian;
    // Loop over each element of the matrix
    for (typename array<FloatType, N>::size_type i = 0; i < point.size();
         ++i)
    {
      PartialDeriv<Function, array<FloatType, N>> first_deriv(function, h,
                                                              i);
      for (typename array<FloatType, N>::size_type j = 0; j < point.size();
           ++j)
      {
        // Calculate second derivative (i,j)
        PartialDeriv<PartialDeriv<Function, array<FloatType, N>>,
                     array<FloatType, N>>
            second_deriv(first_deriv, h, j);
        mhessian(i, j) = second_deriv(point);
      }
    }
    // Return result
    return mhessian;
  }

  // 1) calculate cblas zgemm :
  // 2) sort results
  // 3) find max
  // 4) return unary expression labmda of max value with 0.
  // 5) to eliminate complex factor of the number
  template <typename T>
  struct Quaternion;
  template <typename T>
  struct RollPitchYaw;
  template <typename T>
  struct AxisAngle;
  template <typename T>
  struct Vec3;

  template <typename T>
  struct RollPitchYaw
  {
    // Angles in radians
    T roll;
    T pitch;
    T yaw;

    // Coordinate system:
    // x forward, roll around x, positive rotation clockwise
    // y left, pitch around y, positive rotation down
    // z up, yaw around z, positive rotation to the left

    RollPitchYaw(const T roll_, const T pitch_, const T yaw_);
    RollPitchYaw();

    matrix<T> toRotationMatrix() const;
    Quaternion<T> toQuaternion() const;
    AxisAngle<T> toAxisAngle() const;
  };

  template <typename T>
  struct AxisAngle
  {
    T phi;
    T x;
    T y;
    T z;

    AxisAngle(const T phi_, const T x_, const T y_, const T z_);
    AxisAngle(const T x_, const T y_, const T z_);
    AxisAngle(const Vec3<T> &v);
    AxisAngle();

    AxisAngle<T> normalized() const;

    matrix<T> toRotationMatrix() const;
    Quaternion<T> toQuaternion() const;
    RollPitchYaw<T> toRollPitchYaw() const;
  };

  template <typename T>
  struct Quaternion
  {
    T w; // real
    T x; // imaginary
    T y; // imaginary
    T z; // imaginary

    Quaternion(const T w_, const T x_, const T y_, const T z_) : w(w_), x(x_), y(y_), z(z) {}
    Quaternion() = default;

    matrix<T> toRotationMatrix() const;
    AxisAngle<T> toAxisAngle() const;
    RollPitchYaw<T> toRollPitchYaw() const;

    T norm() const;
    T squaredNorm() const;
    Quaternion<T> normalized() const;
  };

  template <typename T>
  RollPitchYaw<T> Quaternion<T>::toRollPitchYaw() const
  {
    RollPitchYaw<T> rpy;
    // Roll
    T sinr_cosp = 2.0 * (w * x + y * z);
    T cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
    rpy.roll = std::atan2(sinr_cosp, cosr_cosp);

    // Pitch
    T sinp = 2.0 * (w * y - z * x);
    if (std::fabs(sinp) >= 1)
    {
      rpy.pitch = std::copysign(M_PI / 2.0, sinp); // Use 90 degrees if out of range
    }
    else
    {
      rpy.pitch = std::asin(sinp);
    }

    // Yaw
    T siny_cosp = 2.0 * (w * z + x * y);
    T cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
    rpy.yaw = std::atan2(siny_cosp, cosy_cosp);
    return rpy;
  }

  template <typename T>
  matrix<T> Quaternion<T>::toRotationMatrix() const
  {
    matrix<T> m(3, 3);

    const Quaternion<T> qn = this->normalized();
    const T qr = qn.w;
    const T qi = qn.x;
    const T qj = qn.y;
    const T qk = qn.z;

    m(0, 0) = 1.0 - 2.0 * (qj * qj + qk * qk);
    m(0, 1) = 2.0 * (qi * qj - qk * qr);
    m(0, 2) = 2.0 * (qi * qk + qj * qr);
    m(1, 0) = 2.0 * (qi * qj + qk * qr);
    m(1, 1) = 1.0 - 2.0 * (qi * qi + qk * qk);
    m(1, 2) = 2.0 * (qj * qk - qi * qr);
    m(2, 0) = 2.0 * (qi * qk - qj * qr);
    m(2, 1) = 2.0 * (qj * qk + qi * qr);
    m(2, 2) = 1.0 - 2.0 * (qi * qi + qj * qj);

    return m;
  }
  template <typename T>
  T Quaternion<T>::norm() const
  {
    return std::sqrt(w * w + x * x + y * y + z * z);
  }
  template <typename T>
  T Quaternion<T>::squaredNorm() const
  {
    return w * w + x * x + y * y + z * z;
  }

  template <typename T>
  Quaternion<T> Quaternion<T>::normalized() const
  {
    const T d = this->norm();
    return Quaternion<T>(w / d, x / d, y / d, z / d);
  }

  template <typename T>
  Quaternion<T> operator*(const Quaternion<T> &q, const Quaternion<T> &p)
  {
    Vec3<T> qv = Vec3<T>(q.x, q.y, q.z);
    Vec3<T> pv = Vec3<T>(p.x, p.y, p.z);
    Vec3<T> intermediate_vector = qv.crossProduct(pv) + q.w * pv + p.w * qv;
    return Quaternion<T>(q.w * p.w - pv * qv, intermediate_vector.x, intermediate_vector.y, intermediate_vector.z);
  }

  template <typename T>
  Quaternion<T> rotationMatrixToQuaternion(const matrix<T> &m)
  {
    // Reference:
    // http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    Quaternion<T> q;
    q.w = std::sqrt<T>(1.0 + m(0, 0) + m(1, 1) + m(2, 2)) / 2.0;
    q.x = (m(2, 1) - m(1, 2)) / (4.0 * q.w);
    q.y = (m(0, 2) - m(2, 0)) / (4.0 * q.w);
    q.z = (m(1, 0) - m(0, 1)) / (4.0 * q.w);

    return q;
  }

  template <typename T>
  Quaternion<T> RollPitchYaw<T>::toQuaternion() const
  {
    T cy = std::cos(yaw * 0.5);
    T sy = std::sin(yaw * 0.5);
    T cp = std::cos(pitch * 0.5);
    T sp = std::sin(pitch * 0.5);
    T cr = std::cos(roll * 0.5);
    T sr = std::sin(roll * 0.5);

    Quaternion<T> q;

    q.w = cy * cp * cr + sy * sp * sr;
    q.x = cy * cp * sr - sy * sp * cr;
    q.y = sy * cp * sr + cy * sp * cr;
    q.z = sy * cp * cr - cy * sp * sr;

    return q;
  }
  template <typename T>
  AxisAngle<T> RollPitchYaw<T>::toAxisAngle() const
  {
    Quaternion<T> q = toQuaternion();
    return q.toAxisAngle();
  }
  template <typename T>
  matrix<T> rotationMatrixFromYaw(const T yaw)
  {
    matrix<T> m(3, 3);
    const T ca = std::cos(yaw);
    const T sa = std::sin(yaw);

    m(0, 0) = ca;
    m(0, 1) = -sa;
    m(0, 2) = 0.0;

    m(1, 0) = sa;
    m(1, 1) = ca;
    m(1, 2) = 0.0;

    m(2, 0) = 0.0;
    m(2, 1) = 0.0;
    m(2, 2) = 1.0;

    return m;
  }

  template <typename T>
  matrix<T> rotationMatrixFromRoll(const T roll)
  {
    matrix<T> m(3, 3);
    const T ca = std::cos(roll);
    const T sa = std::sin(roll);

    m(0, 0) = 1.0;
    m(0, 1) = 0.0;
    m(0, 2) = 0.0;

    m(1, 0) = 0.0;
    m(1, 1) = ca;
    m(1, 2) = -sa;

    m(2, 0) = 0.0;
    m(2, 1) = sa;
    m(2, 2) = ca;

    return m;
  }
  template <typename T>
  matrix<T> rotationMatrixFromPitch(const T pitch)
  {
    matrix<T> m(3, 3);
    const T ca = std::cos(pitch);
    const T sa = std::sin(pitch);

    m(0, 0) = ca;
    m(0, 1) = 0.0;
    m(0, 2) = sa;

    m(1, 0) = 0.0;
    m(1, 1) = 1.0;
    m(1, 2) = 0.0;

    m(2, 0) = -sa;
    m(2, 1) = 0.0;
    m(2, 2) = ca;

    return m;
  }
  template <typename T>
  matrix<T> RollPitchYaw<T>::toRotationMatrix() const
  {
    return rotationMatrixFromYaw(yaw) * rotationMatrixFromPitch(pitch) * rotationMatrixFromRoll(roll);
  }

  template <typename T>
  RollPitchYaw<T> rotationMatrixToRollPitchYaw(const matrix<T> &m)
  {
    return RollPitchYaw<T>(std::atan2<T>(m(2, 1), m(2, 2)), std::asin<T>(-m(2, 0)), std::atan2<T>(m(1, 0), m(0, 0)));
  }

  template <typename std_type, typename value_type>
  struct blas_multiplier
  {

    size_t max_matrix_count;

  public:
    const std_type &operator()(const std_type &matrices, const matrix<value_type> &transpose)
    {
      const std::complex<value_type> alpha = 1.0;
      const std::complex<value_type> beta = 0.0;
      size_t npoints = transpose.rows();

      size_t count = matrices.size();
      static std_type vResult(count);
      for (size_t k = 0; k < count; ++k)
      {
        size_t CblasColMajor = 0, CblasNoTrans = 0;
        matrix<value_type> a(npoints, npoints);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, npoints, npoints, npoints, &alpha,
                    matrices[k].data(), npoints, transpose.data(), npoints, &beta, a.data(), npoints);

#ifdef _WIN32
        vResult[k] = MatrixXd(Npts, Npts);
        for (int i = 0; i < Npts; ++i)
          for (int j = 0; j < Npts; ++j)
            vResult[k](i, j) = max(a(i, j).real(), 0.0);
#else
        vResult[k] = a.unaryExpr([](std::complex<value_type> x)
                                 { return max(x.real(), 0.0); });
#endif
      }
      return vResult;
    }
  };

  // General quadratic form
  template <typename FloatType, std::size_t N>
  struct Quadratic
  {
    // Quadratic term
    bounded_matrix<FloatType, N, N> _a;
    // Linear term
    array<FloatType, N> _b;
    // Constant
    FloatType _c;

  public:
    Quadratic(const bounded_matrix<FloatType, N, N> &a,
              array<FloatType, N> b /*= array<FloatType,N>()*/,
              const FloatType &c = FloatType()) : _a(a), _b(b), _c(c)
    {
    }

    FloatType
    operator()(const array<FloatType, N> &point) const
    {
      FloatType quad = dot(point, (_a * point));
      return (0.5 * quad - dot(_b, point) + _c);
    }
    ~Quadratic()
    {
    }
  };

  // Transpose matrix
  template <typename T>
  matrix<T>
  transpose(const matrix<T> &mat)
  {
    matrix<T> transp(mat.size2(), mat.size1());
    for (typename matrix<T>::size_type i = 0; i < mat.size2(); ++i)
      for (typename matrix<T>::size_type j = 0; j < mat.size1(); ++j)
        transp(i, j) = mat(j, i);
    return transp;
  }

  // Transpose matrix
  template <typename T, std::size_t N, std::size_t M>
  bounded_matrix<T, M, N>
  transpose(const bounded_matrix<T, N, M> &mat)
  {
    bounded_matrix<T, M, N> transp;
    for (typename bounded_matrix<T, N, M>::size_type i = 0; i < M; ++i)
      for (typename bounded_matrix<T, N, M>::size_type j = 0; j < N; ++j)
        transp(i, j) = mat(j, i);
    return transp;
  }

  // Transpose a vector
  template <typename T>
  matrix<T>
  transpose(const std::vector<T> &vec)
  {
    matrix<T> transp(1, vec.size());
    for (typename matrix<T>::size_type i = 0; i < vec.size(); ++i)
      transp(0, i) = vec[i];
    return transp;
  }

  // Transpose a vector
  template <typename T, std::size_t N>
  matrix<T>
  transpose(const array<T, N> &vec)
  {
    matrix<T> transp(1, vec.size());
    for (typename matrix<T>::size_type i = 0; i < N; ++i)
      transp(0, i) = vec[i];
    return transp;
  }

  // Get identity matrix
  template <typename FloatType>
  matrix<FloatType>
  identity(std::size_t n)
  {
    matrix<FloatType> ident(n, n, FloatType());
    for (size_t i = 0; i < n; ++i)
      ident(i, i) = 1;
    return ident;
  }

  // Check if a vector (array) has nan values
  template <typename T>
  bool
  isNan(const std::vector<T> &vec)
  {
    for (typename std::vector<T>::size_type i = 0; i < vec.size(); ++i)
      if (isnan(vec[i]))
        return true;

    return false;
  }

  template <typename T, std::size_t N>
  bool
  isNan(const array<T, N> &vec)
  {
    for (typename array<T, N>::size_type i = 0; i < N; ++i)
      if (isnan(vec[i]))
        return true;
    return false;
  }

  // utilities with matrices :

  // rot matrix
  template <typename T>
  inline void rot(matrix<T> &a, T s, T tau, const size_t i, const size_t j, const size_t k, const size_t l)
  {
    T g = a(i, j), h = a(k, l);
    a(i, j) = g - s * (h + g * tau);
    a(k, l) = h + s * (g - h * tau);
  }
  //
  template <typename T>
  void slvsml(matrix<T> &out, matrix<T> &rhs)
  {
    // size_t i,j;
    T h(0.5);
    for (size_t i = 0; i < 3; i++)
      for (size_t j = 0; j < 3; ++j)
        out(i, j) = T(0.);

    out(1, 1) = -h * h * rhs(1, 1) / 4.;
  }
  //
  template <typename T>
  void rstrct(matrix<T> &uc, matrix<T> &uf)
  {
    // coarse grid:
    size_t ic, iif, jc, jf, ncc;
    size_t nc = uc.rows();
    ncc = 2 * nc - 2;
    for (jf = 2, jc = 1; jc < nc - 1; jc++, jf += 2)
    {
      for (iif = 2, ic = 1; ic < nc - 1; ic++, iif += 2)
      {
        uc(ic, jc) = T(0.5) * uf(iif, jf - 1) + T(0.125) * uf(iif + 1, jf) + uf(iif - 1, jf) + uf(iif, jf + 1) + uf(iif, jf - 1);
      }
    }
    for (jc = 0, ic = 0; jc < nc; ic++, jc += 2)
    {
      uc(ic, 0) = uf(jc, 0);
      uc(ic, nc - 1) = uf(jc, ncc);
    }

    for (jc = 0, ic = 0; ic < nc; ic++, jc += 2)
    {
      uc(0, ic) = uf(0, jc);
      uc(nc - 1, ic) = uf(ncc, jc);
    }
  }
  //  interpolate from coarse to fine grid  :
  //  uc : coarse grid
  //  uf : fine grid

  template <typename T>
  void interp(matrix<T> &uf, matrix<T> &uc)
  {

    size_t ic, iif, jc, jf, nc;
    const size_t nf = uf.rows();
    nc = size_t(T(nf) / 2. + 1.);
    for (jc = 0; jc < nc; jc++)
      for (ic = 0; ic < nc; ic++)
        uf(2 * ic, 2 * jc) = uc(ic, jc);
    for (jf = 0; jf < nf; jf += 2)
      for (iif = 1; iif < nf - 1; iif++)
        uf(iif, jf) = T(.5) * (uf(iif + 1, jf) + uf(iif - 1, jf));
    for (jf = 1; jf < nf; jf += 2)
      for (iif = 0; iif < nf; iif++)
        uf(iif, jf) = T(.5) * (uf(iif, jf + 1) + uf(iif, jf - 1));
  }
  //    add interpolation to uf from uc
  template <typename T>
  void addint(matrix<T> &uf, matrix<T> &uc, matrix<T> &res)
  {
    size_t i, j;
    size_t nf = uf.rows();
    interp(res, uc);
    for (j = 0; j < nf; j++)
      for (i = 0; i < nf; i++)
        uf(i, j) += res(i, j);
  }
  //
  // compute residual r = b - A*x
  // A is the matrix, x is the solution, b is the right hand side
  // r is the residual
  //
  template <typename T>
  void mg(int j, matrix<T> &u, matrix<T> &rhs)
  {
    const size_t PRE = 1, NPOST = 1;
    size_t npost, jpre, nc, nf;
    nf = u.rows();
    nc = reinterpret_cast<size_t>(T((nf) + 1.0) / 2.);
    if (j == 0)
      slvsml(u, rhs);
    else
    {
      matrix<T> res(nc, nc), v(matrix<T>::Constant(nc, nc, T(0.))), temp(nf, nf);
      // pre relaxation
      for (jpre = 0; jpre < PRE; jpre++)
        relax(u, rhs);
      // residual computation
      resid(temp, u, rhs);
      // restriction
      rstrct(res, temp);
      // recursive call
      mg(j - 1, v, res);
      // interpolation
      addint(u, v, temp);
      // post relaxation
      for (size_t jpost = 0; jpost < NPOST; jpost++)
        relax(u, rhs);
    }
  }

  // Jacobi method
  // a1 : matrix
  // d : diagonal
  // v : eigenvectors
  // nrot : number of rotations
  template <typename T>
  void jacobi(const matrix<T> &a1, std::vector<T> &d, matrix<T> &v, size_t &nrot)
  {
    size_t i, j, ip, iq;
    T thresh, theta, tau, t, sm, s, h, g, c;
    size_t n = d.size();
    matrix<T> a(a1);
    std::vector<T> b(n), z(n);
    for (ip = 0; ip < n; ++ip)
    {

      for (iq = 0; iq < n; ++iq)
        v[ip][iq] = T(0.);

      v[ip][ip] = 1.;
    }
    for (ip = 0; ip < n; ip++)
    {
      b[ip] = d[ip] = a[ip][ip];
      z[ip] = T(0.);
    }
    nrot = 0;
    for (i = 0; i <= 50; ++i)
    {
      sm = 0.;
      for (ip = 0; ip < n - 1; ip++)
      {
        for (iq = ip + 1; iq < n; iq++)
          sm += fabs(a(ip, iq));
      }
      if (sm == 0.0)
        return;

      if (i < 4)
        thresh = .2 * sm / T(n * n);
      else
        thresh = 0.;
      for (ip = 0; ip < n; ip++)
      {
        for (iq = ip + 1; iq < n; iq++)
        {
          g = 100. * fabs(a(ip, iq));
          if (i > 4 && fabs(d[ip]) + g == fabs(d[ip]) && (fabs(d[iq]) + g == fabs(d[iq])))
          {
            a(ip, iq) = 0.;
          }
          else if (fabs(a(ip, iq) > thresh))
          {
            h = d[iq] - d[ip];
            if ((fabs(h) + g) == fabs(h))
            {
              t = (a(ip, iq) / h); // t = 1/phi
            }
            else
            {
              theta = .5 * h / a(ip, iq);
              t = 1. / fabs(theta) / sqrt(1. + (theta * theta));

              if (theta < 0.)
                t = -t;

            } // end else
          }
          c = 1. / sqrt(1 + (t * t));
          s = t * c;
          tau = s / (1. + c);
          h = t * a(ip, iq);
          z[ip] -= h;
          z[iq] += h;
          d[ip] -= h;
          d[iq] += h;
          a(ip, iq) = 0.;
          for (j = 0; j < ip; j++)
            rot(a, s, tau, j, ip, j, iq);
          for (j = ip + 1; j < iq; j++)
            rot(a, s, tau, ip, j, j, iq);
          for (j = iq + 1; j < n; j++)
            rot(a, s, tau, ip, j, iq, j);

          for (j = 0; j < n; j++)
            rot(v, s, tau, j, ip, j, iq);
          ++nrot;

        } // end for iq
      }   // end for ip
      // update d with z
      for (ip = 0; ip < n; ip++)
      {
        b[ip] += z[ip];
        d[ip] += b[ip];
        z[ip] = 0.;
      }

      // sort eigenvalues
      for (ip = 0; ip < n - 1; ip++)
      {
        size_t k = ip;
        T p = d[ip];
        for (j = ip + 1; j < n; j++)
          if (d[j] >= p)
          {
            k = j;
            p = d[j];
          }
        if (k != ip)
        {
          d[k] = d[ip];
          d[ip] = p;
          for (j = 0; j < n; j++)
          {
            p = v[j][ip];
            v[j][ip] = v[j][k];
            v[j][k] = p;
          }
        }
      }
    }
  }

  // Jacobi method (return eigenvalues)
  template <typename T>
  matrix<T> jacobi(const matrix<T> &a, size_t &nrot)
  {
    std::vector<T> diagonal(a.rows());
    for (size_t i = 0; i < a.rows(); ++i)
      diagonal[i] = a(i, i);

    matrix<T> v(a.size1(), a.size2());
    jacobi<T>(a, diagonal, v, nrot);
    return v;
  }

  // Class to create gaussian points using Box-Muller
  template <typename Float>
  class Gaussian
  {
    // Mean
    Float _mean;
    // Standard deviation
    Float _stdev;

  public:
    Gaussian(Float mean, Float stdev) : _mean(mean), _stdev(stdev)
    {
    }

    // Generate point
    Float
    operator()(std::random_device &rd) const
    {
      // Auxiliary parameters
      std::mt19937 gen(rd());
      std::uniform_int_distribution<> uniform(0, RAND_MAX);

      Float theta = 2.0 * M_PI * uniform(gen);
      Float rho = std::sqrt(-2.0 * std::log(1.0 - uniform(gen)));
      // Sample point (using Box-Muller)
      Float x = _mean + _stdev * rho * cos(theta);
      Float y = _mean + _stdev * rho * sin(theta);
      return std::sqrt(x * x + y * y);
    }

    ~Gaussian()
    {
    }
  };
   // Spherical distribution
  template <class Array>
  class SphericalPoint
  {
    typedef typename Array::value_type float_type;
    // Gaussian functor
    Gaussian<float_type> _gauss;
    // Radius
    float_type _radius;
    // Dimension
    size_t _dim;

  public:
    SphericalPoint(size_t dim, float_type radius = 1.0) : _gauss(0, 1), _radius(radius), _dim(dim)
    {
    }

    Array
    operator()(std::random_device &r) const
    {
      // Buffer
      std::mt19937 gen(r());
      std::uniform_int_distribution<> uniform(0, RAND_MAX);

      Array _buffer;
      resizeArray(&_buffer, _dim);
      for (size_t i = 0; i < _buffer.size(); ++i)
        _buffer[i] = _gauss(uniform(gen));
      _buffer = (_radius / sqrt(dot(_buffer, _buffer))) * _buffer;
      return _buffer;
    }

    ~SphericalPoint()
    {
    }
  };

  // global operators for X/matrix
  template <typename T>
  matrix<T> operator/(const T &b, const matrix<T> &a)
  {
    matrix<T> ret;
    for (typename matrix<T>::size_type i = 0; i < a.size1(); i++)
    {
      for (typename matrix<T>::size_type j = 0; j < a.size2(); j++)
      {
        ret(i, j) = a(i, j) / b;
      }
    }

    return ret;
  }
  //    
  template <typename T>
  matrix<T> operator/(const matrix<T> &a, const T &b)
  {
    matrix<T> ret;
    for (typename matrix<T>::size_type i = 0; i < a.size1(); i++)
    {
      for (typename matrix<T>::size_type j = 0; j < a.size2(); j++)
      {
        ret(i, j) = a(i, j) / b;
      }
    }

    return ret;
  } // end operator/

  template <typename T>
  matrix<T> operator/(const matrix<T> &a, const matrix<T> &b)
  {
    matrix<T> ret;
    for (typename matrix<T>::size_type i = 0; i < a.size1(); i++)
    {
      for (typename matrix<T>::size_type j = 0; j < a.size2(); j++)
      {
        ret(i, j) = a(i, j) / b(i, j);
      }
    }

    return ret;
  } // end operator/    
 
} /* namespace provallo */ ;
#endif /* DECISION_ENGINE_MATRIX_H_ */
