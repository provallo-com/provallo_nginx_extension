/*********************************
 * dataset.h
 *
 *  Created on: May 11, 2021
 *      Author: Kardon

 ***********************************/
#ifndef dataset_H_
#define dataset_H_

#include <memory>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include "matrix.h"
#include "attribute.h"
#include "classdist.h"
#include <thread>
#include <mutex>
#include <atomic>
#include "../util/mmap_allocator.h"

// #include "split_utils.hpp"
//
//
//
namespace provallo
{
  // fwd declr for dataset functors
  //
  class dataset_base;
  //
  typedef std::shared_ptr<dataset_base> dataset_ptr;
  //
  class split_method;

  class dataset_base : public matrix_base
  {
  private:
    int *_labels;
    //
  public:
    size_t _num_of_labels;

    int &
    label(size_t row)
    {
      return _labels[row];
    }
    int
    label(size_t row) const
    {
      return _labels[row];
    }
    //
    
    double get(const size_t row,const size_t col) const
    {
      return matrix_base::pos(row,col);
    }
    dataset_base(size_t rows, size_t cols, size_t numLabels);
    virtual ~dataset_base();
    // use splitters from
    void
    splitdataset(dataset_ptr &train, dataset_ptr &valid, double train_percent);
    
  };
  // fwd declr for dataset functors

  class dataset; // fwd:
  // Dataset statistics Debugging:
  Float entropy(const dataset &data);

  Float gini(const dataset &data);
  Float variance(const dataset &data);
  Float mean(const dataset &data);
  Float stddev(const dataset &data);
  Float skewness(const dataset &data);
  Float kurtosis(const dataset &data);
  Float median(const dataset &data);
  Float mode(const dataset &data);
  Float min(const dataset &data);
  Float max(const dataset &data);
  Float sum_of_squares(const dataset &data);
  Float median_absolute_deviation(const dataset &data);
  Float max(const dataset &data);
  Float min(const dataset &data);
  Float sum(const dataset &data);

  // Class to hold data
  class dataset
  {

    
  protected:
    clock_t last_sort;
    // Information about the attributes on this data set
    attribute_information _attributes_info;
    // Holds sorted indices of each attribute (useful when dealing with
    // numeric attributes)
    

    // Class or target distribution of this set

    // Filters to apply to the data set
    // Last time the data set was sorted




    //fix mmap_vector<> 

    //typedef mmap_vector<uint32_t> sorted_index;
    //typedef mmap_vector<sorted_index> sorted_indices;
    typedef std::vector<uint32_t> sorted_index;
    typedef std::vector<sorted_index> sorted_indices;


    sorted_indices   _sorted_indices;
    
    class_dist _distribution;

    mutable bool _dirty; // push makes it dirty. sort cleans.
    mutable std::recursive_mutex _mutex;

  public:
    typedef void (*filter_function)(const provallo::attribute_information &attribute_info, const std::vector<std::string> csv, provallo::dataset *set);

    virtual void add_filter(filter_function filter)
    {
      this->_filters.push_back(filter);
    }
    
    // Push the n-th attribute of an instance
    virtual void
    pushattribute(uint32_t n, const attribute &att) = 0;
    
    inline  void
    pushattribute(uint32_t n,const discrete_value& r) 
    {
      pushattribute(n, attribute(r));
    }
    inline  void
    pushattribute(uint32_t n,const cont_value& r) 
    { 
      pushattribute(n, (attribute(r)));
    }
    
    
    // Return new pointer
    virtual dataset *
    getNew() const = 0;
    // Print attribute given the information
    void
    printattribute(std::ostream &out, const attribute_tag &tag,
                   const attribute &att,
                   const attribute_information &info) const;
    // Print data set
    void
    print(std::ostream &out) const;

    // Sort an attribute (i.e. set the sorted_indices array for this attribute)
    void
    sortattribute(const attribute_tag &tag);
    // Sort all attributes
    void
    sortattributes();
    // Sort attributes of a specified type
    void
    sortattributes(attribute_type type);
    // Get sorted indices of an attribute
    const sorted_index &
    getSortedIndices(const attribute_tag &tag) const
    {
      assert(_sorted_indices[tag].size() > 0);
      return _sorted_indices[tag];
    }
    // Get sorted indices of an attribute (non-const version)
    sorted_index &
    getSortedIndices(const attribute_tag &tag)
    {
      assert(_sorted_indices[tag].size() > 0);
      return _sorted_indices[tag];
    }
    // Get sorted indices of all attributes
    const sorted_indices &get_sorted_indices() const
    {
      return _sorted_indices;
    }
    bool
    isSorted(const attribute_tag &tag) const
    {
      return tag < _sorted_indices.size() && _sorted_indices[tag].size() > 0;
    }
    virtual const attribute *getattributeptr(uint32_t i, attribute_tag tag, bool *found) const = 0;

    // Set distribution
    void
    setDistribution(const class_dist &dist)
    {
      _distribution = dist;
    }

    // Printer
    friend std::ostream &
    operator<<(std::ostream &out, const dataset &q);

  public:
    // Base iterator class
    class iterator_base
    {
    protected:
      iterator_base() : _data(0), _instance(0), _tag(0)
      {
      }
  
    public:
      typedef attribute value_type;
      typedef attribute *pointer;
      typedef attribute &reference;
      typedef size_t size_type;
      typedef size_t difference_type;
      typedef std::bidirectional_iterator_tag iterator_category;
      iterator_base(const iterator_base &r) : _data(r._data), _instance(r._instance), _tag(r._tag) {}

      iterator_base(const dataset *data, size_type instance, attribute_tag tag) : _data(const_cast<provallo::dataset *>(data)), _instance(instance), _tag(tag)
      {
      }
      iterator_base(iterator_base &&move) : _data(std::move(move._data)), _instance(std::move(move._instance)), _tag(std::move(move._tag)) {}

      bool
      operator==(const iterator_base &other) const
      {
        return ((other._data == _data) && (other._instance == _instance) && (other._tag == _tag));
      }

      bool
      operator!=(const iterator_base &other) const
      {
        return ((other._data != _data) || (other._instance != _instance) || (other._tag != _tag));
      }

      iterator_base &operator=(const iterator_base &other)
      {
        _data = other._data;
        _instance = other._instance;
        _tag = other._tag;
        return *this;
      }
      iterator_base &operator=(iterator_base &&other)
      {
        _data = std::move(other._data);
        _instance = std::move(other._instance);
        _tag = std::move(other._tag);
        return *this;
      }

      // Internal data
      dataset *_data;

      // Location of the attribute on the data
      size_type _instance; // Instance
      attribute_tag _tag;  // Tag of the attribute

      // Get the attribute
      attribute
      getattribute() const
      {
        return _data->getattribute(_instance, _tag);
      }
      virtual ~iterator_base() {}

    };

  public:
    class attribute_iterator;
    class instance_iterator;
    class sorted_iterator;
    class sorted_instance_iterator;
    class sorted_attribute_iterator;

  public:
    // Given an attribute, iterate over all the instances
    class instance_iterator : public iterator_base
    {
    public:
      instance_iterator()
      {
      }
      instance_iterator(const instance_iterator &other) : iterator_base(other._data, other._instance, other._tag)
      {
      }

      instance_iterator(instance_iterator &&other) : iterator_base(other) {}

      instance_iterator(const dataset *data, size_type instance) : iterator_base(data, instance, 0)
      {
      }

      attribute_iterator
      begin() const
      {
        return _data->begin(_instance);
      }
      attribute_iterator
      end() const
      {
        return _data->end(_instance);
      }

      instance_iterator &
      operator++()
      {
        _instance++;
        return *this;
      }
      instance_iterator &
      operator--()
      {
        _instance--;
        return *this;
      }
      instance_iterator
      operator++(int)
      {
        instance_iterator t(*this);
        ++(*this);
        return t;
      }
      instance_iterator
      operator--(int)
      {
        instance_iterator t(*this);
        --(*this);
        return t;
      }

      instance_iterator
      operator+(difference_type n)
      {
        return instance_iterator(_data, _instance + n);
      }
      instance_iterator
      operator-(difference_type n)
      {
        return instance_iterator(_data, _instance - n);
      }
      bool
      operator<(const instance_iterator &other) const
      {
        return _instance < other._instance;
      }
      bool
      operator>(const instance_iterator &other) const
      {
        return _instance > other._instance;
      }
      difference_type
      operator-(const instance_iterator &other) const
      {
        return _instance - other._instance;
      }
    };

    // Given an attribute, iterate over all the instances, in increasing order
    class sorted_iterator : public iterator_base
    {
    public:
      sorted_iterator(sorted_iterator &&move) : iterator_base(move) {}

      sorted_iterator(const sorted_iterator &other) : iterator_base(other._data, other._instance, other._tag)
      {
      }
      sorted_iterator(const dataset *data, size_type instance,
                      attribute_tag tag) : iterator_base(data, instance, tag)
      {
      }

      attribute_iterator
      begin() const
      {
        uint32_t index(_data->getSortedIndices(_tag)[_instance]);
        return _data->begin(index);
      }
      attribute_iterator
      end() const
      {
        uint32_t index(_data->getSortedIndices(_tag)[_instance]);
        return _data->end(index);
      }

      const sorted_iterator &operator=(const sorted_iterator &ref)
      {
        if (this->_data && this->_data != ref._data)
          delete _data;
        this->_data = ref._data;
        this->_instance = ref._instance;
        this->_tag = ref._tag;
        return *this;
      }

      sorted_iterator &
      operator++()
      {
        _instance++;
        return *this;
      }
      sorted_iterator &
      operator--()
      {
        _instance--;
        return *this;
      }
      sorted_iterator
      operator++(int)
      {
        sorted_iterator t(*this);
        ++(*this);
        return t;
      }
      sorted_iterator
      operator--(int)
      {
        sorted_iterator t(*this);
        --(*this);
        return t;
      }
      sorted_iterator
      operator+(difference_type n)
      {
        return sorted_iterator(_data, _instance + n, _tag);
      }
      sorted_iterator
      operator-(difference_type n)
      {
        return sorted_iterator(_data, _instance - n, _tag);
      }
      // comparison operators
      bool
      operator<(const sorted_iterator &other) const
      {
        return _instance < other._instance;
      }

      bool
      operator>(const sorted_iterator &other) const
      {
        return _instance > other._instance;
      }
      bool operator==(const sorted_iterator &other) const
      {
        return _instance == other._instance;
      }
      bool operator!=(const sorted_iterator &other) const
      {
        return _instance != other._instance;
      }
      difference_type
      position(size_type __sub = 0) const
      { return std::distance( begin(), begin()+__sub); }

      difference_type
      operator-(const sorted_iterator &other) const
      {
        return _instance - other._instance;
      }

      // Get the attribute
      const attribute &
      getattribute() const
      {
        return _data->getattribute(_instance, _tag);
      } 

      // Get the attribute  
      const attribute &
      operator*() const
      {
        return getattribute();
      } 

    };
    class iterator : public iterator_base
    {
    public:
      iterator(const iterator &other) : iterator_base(other._data, other._instance, other._tag)
      {
      }
      iterator(const dataset *data, size_type instance, attribute_tag tag) : iterator_base(data, instance, tag)
      {
      }
      iterator(iterator &&other) : iterator_base(other)
      {
      }
      iterator &operator=(const iterator &ref)
      {
        if (this->_data && this->_data != ref._data)
          delete _data;
        this->_data = ref._data;
        this->_instance = ref._instance;
        this->_tag = ref._tag;
        return *this;
      }
      iterator &
      operator++()
      {
        _instance++;
        return *this;
      }
      iterator &
      operator--()
      {
        _instance--;
        return *this;
      }
      iterator
      operator++(int)
      {
        iterator t(*this);
        ++(*this);
        return t;
      }
      iterator
      operator--(int)
      {
        iterator t(*this);
        --(*this);
        return t;
      }
      iterator
      operator+(difference_type n)
      {
        return iterator(_data, _instance + n, _tag);
      }

      iterator
      operator-(difference_type n)
      {
        return iterator(_data, _instance - n, _tag);
      }
      bool
      operator<(const iterator &other) const
      {
        return _instance < other._instance;
      }
      bool
      operator>(const iterator &other) const
      {
        return _instance > other._instance;
      }
      difference_type
      operator-(const iterator &other) const
      {
        return _instance - other._instance;
      }
      // const operators
      const iterator operator+(difference_type n) const
      {
        return iterator(_data, _instance + n, _tag);
      }
      const iterator operator-(difference_type n) const
      {
        return iterator(_data, _instance - n, _tag);
      }
    };

    class const_iterator : public iterator_base
    {
    public:
      const_iterator(const const_iterator &other) : iterator_base(other._data, other._instance, other._tag)
      {
      }
      const_iterator(const dataset *data, size_type instance,
                     attribute_tag tag) : iterator_base(data, instance, tag)
      {
      }
      const_iterator(const iterator &other) : iterator_base(other._data, other._instance, other._tag)
      {
      }
      const_iterator(const_iterator &&other) : iterator_base(other)
      {
      }
      const_iterator &operator=(const const_iterator &ref)
      {
        if (this->_data && this->_data != ref._data)
          delete _data;
        this->_data = ref._data;
        this->_instance = ref._instance;
        this->_tag = ref._tag;
        return *this;
      }
      const_iterator &
      operator++()
      {
        _instance++;
        return *this;
      }
      const_iterator &
      operator--()
      {
        _instance--;
        return *this;
      }
      const_iterator
      operator++(int)
      {
        const_iterator t(*this);
        ++(*this);
        return t;
      }
      const_iterator
      operator--(int)
      {
        const_iterator t(*this);
        --(*this);
        return t;
      }
      const_iterator
      operator+(difference_type n)
      {
        return const_iterator(_data, _instance + n, _tag);
      }
      const_iterator
      operator-(difference_type n)
      {
        return const_iterator(_data, _instance - n, _tag);
      }
      bool
      operator<(const const_iterator &other) const
      {
        return _instance < other._instance;
      }
      bool
      operator>(const const_iterator &other) const
      {
        return _instance > other._instance;
      }
      difference_type
      operator-(const const_iterator &other) const
      {
        return _instance - other._instance;
      }
    };

    class attribute_iterator : public iterator_base
    {

    private:
      static std::atomic_uint64_t _instance_counter;

    public:
      attribute_iterator(const attribute_iterator &other) : iterator_base(other._data, other._instance, other._tag)
      {
        attribute_iterator::_instance_counter++;
      }
      attribute_iterator(const dataset *data, size_type instance,
                         attribute_tag tag) : iterator_base(data, instance, tag)
      {
        attribute_iterator::_instance_counter++;
      }
      attribute_iterator(const attribute_iterator &&other) : iterator_base(other)
      {
        attribute_iterator::_instance_counter++;
      }

      attribute_iterator &operator=(const attribute_iterator &ref)
      {
        if (this->_data && this->_data != ref._data)
          delete _data;
        this->_data = ref._data;
        this->_instance = ref._instance;
        this->_tag = ref._tag;
        return *this;
      }
      attribute_iterator &
      operator++()
      {
        _tag++;

        return *this;
      }
      attribute_iterator &
      operator--()
      {
        _tag--;

        return *this;
      }
      attribute_iterator
      operator++(int)
      {
        attribute_iterator t(*this);
        ++(*this);
        return t;
      }
      attribute_iterator
      operator--(int)
      {
        attribute_iterator t(*this);
        --(*this);
        return t;
      }
      attribute_iterator
      operator+(difference_type n) const
      {
        return attribute_iterator(_data, _instance, _tag + n);
      }
      attribute_iterator
      operator-(difference_type n) const
      {
        return attribute_iterator(_data, _instance, _tag - n);
      }
      attribute_iterator
      operator+(difference_type n)
      {
        return attribute_iterator(_data, _instance, _tag + n);
      }
      attribute_iterator
      operator-(difference_type n)
      {
        return attribute_iterator(_data, _instance, _tag - n);
      }
      bool operator==(const attribute_iterator &other) const
      {
        return _tag == other._tag && _instance == other._instance;
      }
      bool operator!=(const attribute_iterator &other) const
      {
        return _tag != other._tag || _instance != other._instance;
      } 
      bool operator > (const attribute_iterator &other) const
      {
        return  _instance > other._instance || _tag > other._tag;
      } 
      bool
      operator<(const attribute_iterator &other) const
      {
        return  _instance < other._instance || _tag < other._tag;
      }
 
       difference_type
      operator-(const attribute_iterator &other) const
      {
        return _tag - other._tag - _instance + other._instance;
      }

      difference_type operator+(const attribute_iterator &other) const
      {
        return _tag + other._tag + _instance + other._instance;
      }

      const attribute &operator*() const
      {
        return _data->getattribute(_instance, _tag);
      }

      attribute operator*()
      {
        return _data->getattribute(_instance, _tag);
      }

      const attribute *operator->() const
      {
        return &_data->getattribute(_instance, _tag);
      }

      const attribute *operator->()
      {
        // return &_data->getattribute (_instance, _tag)
        // return &_data->getattribute (_instance, _tag);
        bool bfound = false;
        const attribute *ret = _data->getattributeptr(_instance, _tag, &bfound);

        if (!bfound)
        {
          std::cout << "attribute_iterator::operator->() attribute not found" << std::endl;
          
          // throw std::runtime_error("attribute_iterator::operator->() attribute not found");
          return nullptr;
        }
        return ret;
      }
      virtual ~attribute_iterator()
      {

        attribute_iterator::_instance_counter--;
        // std::cout << "[=] attribute_iterator instances: " <<std::to_string( (uint64_t) _instance_counter ) << std::endl;
      }
    };

  public: // Iterators
    // Ite

    inline iterator
    begin()
    {
      return iterator(this, 0, 0);
    }
    inline iterator
    end()
    {
      return iterator(this, _attributes_info.getSize(), 0);
    }
    inline const_iterator
    begin() const
    {
      return const_iterator(this, 0, 0);
    }
    inline const_iterator
    end() const
    {
      return const_iterator(this, _attributes_info.getSize(), 0);
    }

    // Iterate over attributes of an instance
    inline attribute_iterator
    begin(const uint32_t &instance) const
    {
      return attribute_iterator(this, instance, 0);
    }
    inline attribute_iterator
    end(const uint32_t &instance) const
    {
      return attribute_iterator(this, instance, _attributes_info.getSize());
    }

    // Iterate over instances of an attribute
    inline instance_iterator
    begin_instance() const
    {
      return instance_iterator(this, 0);
    }
    inline instance_iterator
    end_instance() const
    {
      return instance_iterator(this, size());
    }

    // Iterate over sorted instances of an attribute
    inline sorted_iterator
    begin_sorted(const attribute_tag &tag) const
    {
      return sorted_iterator(this, 0, tag);
    }
    inline sorted_iterator
    end_sorted(const attribute_tag &tag) const
    {
      return sorted_iterator(this, size(), tag);
    }

    // Construction

    // dataset () = default;

    dataset(const attribute_information &attributes_info) : last_sort(clock_t(0)), _attributes_info(attributes_info), _sorted_indices(
                                                                                                                          _attributes_info.getSize()),
                                                            _distribution(
                                                                attributes_info.getCount(attributes_info.get_target_tag())),
                                                            _dirty(true)
    {
      id_counter += 1;
      _id = size_t(id_counter);
      dest_counter += 1;
    }

    // Copy constructor
    dataset(const dataset &right) : last_sort(clock_t(0)), _attributes_info(right._attributes_info), _sorted_indices(
                                                                                                         right._sorted_indices),
                                    _distribution(right._distribution), _dirty(false)
    {

      _id = size_t(++id_counter);
      dest_counter += 1;
    }

    // Copy constructor but replacing the attributes information class
    dataset(const dataset &right, const attribute_information &attributes_info) : last_sort(clock_t(0)), _attributes_info(attributes_info), _sorted_indices(
                                                                                                                                                right._sorted_indices),
                                                                                  _distribution(right._distribution), _dirty(false)
    {

      _id = size_t(++id_counter);
      dest_counter += 1;
    }

    dataset(dataset &&move) : last_sort(std::move(move.last_sort)),
                              _attributes_info(std::move(move._attributes_info)),
                              _sorted_indices(std::move(move._sorted_indices)),
                              _distribution(std::move(move._distribution)), _dirty(move._dirty)

    {

      _id = size_t(++id_counter);
      dest_counter += 1;
    }

    // Assignment operator
    dataset &operator=(const dataset &ref)
    {
      if (this != &ref)
      {
        _attributes_info = ref._attributes_info;
        _sorted_indices = ref._sorted_indices;
        _distribution = ref._distribution;
        _dirty = ref._dirty;
        last_sort = ref.last_sort;
      }
      return *this;
    }
    // Assignment operator
    dataset &operator=(dataset &&ref)
    {
      if (this != &ref)
      {
        _attributes_info = std::move(ref._attributes_info);
        _sorted_indices = std::move(ref._sorted_indices);
        _distribution = std::move(ref._distribution);
        _dirty = ref._dirty;
        last_sort = std::move(ref.last_sort);
      }
      return *this;
    }
    // Assignment operator
    dataset &operator=(const dataset *ref)
    {
      if (this != ref)
      {
        _attributes_info = ref->_attributes_info;
        _sorted_indices = ref->_sorted_indices;
        _distribution = ref->_distribution;
        _dirty = ref->_dirty;
        last_sort = ref->last_sort;
      }

      return *this;
    }
    sorted_index & operator[](const attribute_tag &tag)
    {
      return _sorted_indices[tag];
    } 

    attribute_tag get_target_tag() const
    {
      return _attributes_info.get_target_tag();
    } 

    // Return new pointer with a reference data set

    virtual dataset *
    getNewReference(std::vector<size_t> &indices) const = 0;

    
     // Push a case (this include the class / target attribute)
    template <class InputIterator>
    void
    pushData(InputIterator begin, InputIterator end);

    //push data as a vector of tokens
    void pushData(const std::vector<std::string> &tokens);

    // Get attributes information
    const attribute_information &
    getattributes() const
    {
      return _attributes_info;
    }

    // Number of attributes
    uint32_t
    getattributesNumber() const
    {
      return _attributes_info.getSize();
    }

    // Get distribution
    const class_dist &
    getDistribution() const
    {
      return _distribution;
    }

    // Get subset of samples in a branch
    dataset *
    subset(const split_method &split_method, uint32_t nbranch) const;
    const dataset *
    subsetReference(const split_method &split_method, uint32_t nbranch) const;

    // Get subset of samples with a particular instances of an attribute using a test
    // functor to select the instances on the data set
    // Caller is responsible for deleting the returned pointer
    template <class Tester>
    dataset *
    subset(const Tester &tester) const;

    // Get new set of data randomly sampling using the internal weights.
    // Caller is responsible for deleting the returned pointer
    dataset *
    randomSubset(std::random_device &random,
                 const class_dist &distribution) const;

    // Get new set of data randomly sampling using the internal weights, also
    // the out of bag data is returned. Caller is responsible for deleting both
    // pointers. Pair returned = <data_set, oob_set>. This method also sets the OOB
    // indices from the original data set (so the caller knows which cases were OOB
    // after the sampling process)
    std::pair<dataset *, dataset *>
    randomSubsetOob(std::random_device &, const class_dist &distribution,
                    std::vector<uint32_t> *oob_indices) const;

    // ---- Non-constant operations

    // Craft data (i.e. signal the object that the data is setup). This will initialize
    // internal variables on the object.
    virtual void
    setup();

    // Randomly permute instance of an attribute value and get the original values
    // on a container (should be used to restore values on the future)
    void
    permute(std::random_device &random, attribute_tag tag,
            std::vector<attribute> *prev_values);

    // Set values of instance of an attribute. Usually called when restoring
    // values of attributes after random permutation
    void
    restore(attribute_tag tag, const std::vector<attribute> &prev_values);

    // Get number of cases on the set
    virtual uint32_t
    size() const = 0;

    // Get a particular instance's attribute  (this is used to permute attributes)
    virtual const attribute &
    getattribute(uint32_t i, attribute_tag tag) const = 0;

    // Set an instance's attribute  (this is used to permute attributes)
    virtual void
    setattribute(uint32_t i, attribute_tag tag, const attribute &value) = 0;

    inline void setupclassdist()
    {
      _distribution.setup(_attributes_info.getCount(_attributes_info.get_target_tag()));
    }
    inline void update_classdist(const class_dist &distribution)
    {
      _distribution.accum(distribution);
    }
    virtual ~dataset()
    {
      dest_counter -= 1;
      size_t remaining = dest_counter;
      std::cout << "[=] dataset" << std::to_string(_id) << " destroyed , remaining ( " << std::to_string(remaining) << ")" << std::endl;

    } 


    Float
    gini() const
    {
      return provallo::gini(*this);
    }
    Float entropy() const
    {
      return provallo::entropy(*this);
    }
    // variance of the data  set
    Float variance() const
    {
      return provallo::variance(*this);
    }
    // mean of the data  set
    Float mean() const
    {
      return provallo::mean(*this);
    }
    // median of the data  set
    Float median() const
    {
      return provallo::median(*this);
    }
    // mode of the data  set
    Float mode() const
    {
      return provallo::mode(*this);
    }
    // min of the data  set
    Float min() const
    {
      return provallo::min(*this);
    }
    // max of the data  set
    Float max() const
    {
      return provallo::max(*this);
    }
    // sum of the data  set
    Float sum() const
    {
      return provallo::sum(*this);
    }
    // std of the data  set
    Float stddev() const
    {
      return provallo::stddev(*this);
    }
    // skewness of the data  set
    Float skewness() const
    {
      return provallo::skewness(*this);
    }
    // kurtosis of the data  set
    Float kurtosis() const
    {
      return provallo::kurtosis(*this);
    }

  private:
    std::vector<filter_function> _filters;

  protected:
    static std::atomic_int id_counter;
    static std::atomic_int dest_counter;

    size_t _id;
  };

  std::ostream &
  operator<<(std::ostream &out, const dataset &q);

  template <class InputIterator>
  void
  dataset::pushData(InputIterator begin, InputIterator end)
  {
    // Get target tag
    uint32_t target_tag(_attributes_info.get_target_tag());
    // Loop over each sample
    while ( begin!=end && end-begin > 0 ) {
      // Loop over each attribute
      

      for (uint32_t i = 0; i < getattributesNumber(); ++i)
      {
        // Get attribute value
         assert(begin != end);
        
        //  attribute value);
        attribute_type att_type = _attributes_info.getType(i);

        attribute_value value_string = *begin;

        attribute   value ( value_string, att_type);
        // Push attribute into the set

        if (i==target_tag) {
                  //make sure the target value is less than the number of classes
                if(value.discrete() < _attributes_info.getTargetClassCount())
                {
                  _distribution.accum(value.discrete());
                }
                else {
                            std::cout<< "[!] target tag : " << std::to_string(target_tag) << std::string(" number of classes ")<< std::to_string(_attributes_info.getTargetClassCount()) <<" target value discrete : " << std::to_string(value.discrete()) << " , continous : "<<value.continous() << std::endl;
                            //normalize to 0
                            value = attribute(discrete_value(0));
                 }
        }
        
          pushattribute(i, value);
          begin++;
      }
    }
    
    
     
  }

  // The tester functor should implement a test method grabbing iterators to attributes for a
  // specific instance
  template <class Tester>
  dataset *
  dataset::subset(const Tester &tester) const
  {
    // Get target tag
    uint32_t target_tag(_attributes_info.get_target_tag());
    // Crate new data set
    dataset *new_set = getNew();
    // Loop over each sample
    for (uint32_t i = 0; i < size(); ++i)
    {
      // Check the value of the attribute
      if (tester.test(this->begin(i), this->end(i)))
      {
        // Push this sample into the new set
        for (uint32_t j = 0; j < getattributesNumber(); ++j)
        {
          attribute value(getattribute(i, j));
          // Push back attribute
          new_set->pushattribute(j, getattribute(i, j));
          // Sum contribution into the distribution
          if (j == target_tag)
            new_set->_distribution.accum(value.discrete());
        } // end for
        // Sanity check
        //assert(new_set->size() == i + 1);
        //
      } // end if

    } // end for

    // Setup the new set
    new_set->setup();
    // Return new data set
    return new_set;
  }

  // Training or Testing Set classes just differ in the schemes that attributes are saved in the
  // memory. training_set memory access is optimized for training a classifier and TestingTest
  // to test a classifier. But since both classes are a dataset is possible to test a
  // classifier with training_set or train a classifier with TestingData
    typedef mmap_vector<attribute, safe_mmap_allocator<attribute> > safe_mmap_vector ; 
    typedef std::vector<std::vector<attribute> > samples_container;

  // Class to hold the training data
  class training_set : public dataset
  {
    // Container of samples. The access is done using attributes as rows and
    // instances as columns [attribute][instance]. Better memory access scheme
    // for training
    
  public:
    // typedef std::vector<attribute> safe_mmap_vector ;         
  protected:
    // Container of samples. The access is done using attributes as rows and
    
    samples_container  _samples;

    // Container of samples. The access is done using attributes as rows and
  

    //  std::vector<std::vector<attribute>> _samples;

    void
    pushattribute(uint32_t n, const attribute &att)
    {
      static std::atomic_uint64_t counter(0); 
      ++counter;
      // Push sample into the set of samples
      // if n==target_tag then verify the value is discrete.
      if(_samples.size() <= n)  
        _samples.resize(n+1);
      _samples[n].push_back(att); 

      if(counter % 1000000 == 0 && counter/1000000 > 0)
          std::cout << "[+] " << std::to_string(counter) << " samples pushed, for  " << std::to_string(id_counter)<< "datasets"<< std::endl;
      // Set dirty flag
      _dirty = true;
    }

    dataset *
    getNew() const
    {
      dataset *n_ret = nullptr;
      n_ret = new training_set(*this);
      // Return new data set
      return n_ret;
    }

    // Return a reference
    friend class training_setReference;

  public:
    training_set(const attribute_information &attributes_info) : dataset(attributes_info), _samples(attributes_info.getSize())
    {
      const size_t n = attributes_info.getSize();
      for ( uint32_t i = 0; i <n; ++i)
      {   
          _samples[i].reserve(n);
      }
    }
    training_set(training_set &&right) : dataset(std::move(right)), _samples(std::move(right._samples))
    {
    }
    training_set(const training_set &right) : dataset(right), _samples(right._samples.size())
    {
      
      std::copy(right._samples.begin(), right._samples.end(), _samples.begin());  

    } // end training_set
 
    training_set(const dataset &right,
                 const attribute_information &attributes_info);

    dataset *
    getNewReference(std::vector<size_t> &indices) const;


    dataset * 
    getNewReference(std::vector<size_t,safe_mmap_allocator<size_t>> &indices) const; 
    uint32_t
    size() const
    {
      assert(_samples.size() != 0&&_attributes_info.get_target_tag() < _samples.size()) ;
      return _samples[0].size(); //return number of samples 
    }

    unsigned int
    getattributesNumber() const
    {
      return _attributes_info.getSize();
    } 
    

    const attribute &
    getattribute(uint32_t i, attribute_tag tag) const
    {
      static const attribute empty;
      if (_samples.size() > tag && _samples[tag].size() > i)
        return _samples[tag][i];
      return empty;
    }
    attribute &getattribute(uint32_t i, attribute_tag tag)
    {
      static attribute empty;
      if (_samples.size() > tag && _samples[tag].size() > i)
        return _samples[tag][i];
      return empty;
    }
    const attribute *getattributeptr(uint32_t i, attribute_tag tag, bool *found) const
    {
      static const attribute empty(NA_VAL);

      if (  _samples[tag].size() > i )
      {
        *found = true;
        return &_samples.at(tag).at(i);
      }
      *found = false;
      return &empty;
    }
    attribute *getattributeptr(uint32_t i, attribute_tag tag, bool *found)
    {
      static attribute empty;
      //size_t index = tag;
      if (_samples[tag].size() > i)
      {
        *found = true;
        return &_samples[tag][i];
      }
      *found = false;
      return &empty;
    }

    void
    setattribute(uint32_t i, attribute_tag tag, const attribute &value)
    {

      if (_samples.size() > tag)
      {
        // mark dirty

        if (i < _samples[tag].size())
        {
          _dirty = true;

          // set value
          _samples[tag][i] = value;
        }

        else
        {

          // resize
          _dirty = true;

          size_t size = _samples[tag].size();
          size_t delta = abs(int(i) - int(size));

          _samples[tag].resize(delta);
          _samples[tag][i] = value;
        }
      }
    }
    inline const samples_container & get_samples() const
    {

      return _samples;
    }
    // std::vector<safe_mmap_vector >  &get_samples()
    // {

    //   return _samples; 
    // }
    matrix<attribute> get_matrix() const 
    {
      matrix<attribute> ret(_samples.size(), _samples[0].size());
      for (size_t i = 0; i < _samples.size(); ++i)
      {
        for (size_t j = 0; j < _samples[i].size(); ++j)
        {
          ret[i][j] = _samples[i][j];
        }
      }
      return ret;
    }



     virtual ~training_set()
    {
      //_samples.clear();
      
    }
  };

  // Emulate a constant reference to some indices on a data set. Can't modify value from here.
  class training_setReference : public dataset
  {
    // Internal reference to the parent data set
    const training_set *_dataset;
    // Indices of original data set
    std::vector<size_t  > _indices;

    void
    pushattribute(uint32_t n, const attribute &att)
    {
      throw std::runtime_error(
          std::string("Can't push an attribute on a data set reference") + std::to_string(n) + std::to_string(att.continous()) );
    }

    dataset *
    getNew() const
    {
      //copy new dataset
      return new training_setReference(*this->_dataset, this->_indices);
    }

  public:
    training_setReference(const training_set &dataset,
                          const std::vector<size_t,safe_mmap_allocator<size_t>> &indices) : dataset::dataset(dataset.getattributes()), _dataset(&dataset), _indices(indices.size()) 
    {
      //_indices.resize(dataset.size());
      for (size_t i = 0; i < _indices.size(); ++i)
        _indices[i] = indices[i];
    }
        training_setReference(const training_set &dataset,
                          const std::vector<size_t> &indices) : dataset::dataset(dataset.getattributes()), _dataset(&dataset), _indices(indices.size()) 
    {
      //_indices.resize(dataset.size());
      for (size_t i = 0; i < _indices.size(); ++i)
        _indices[i] = indices[i];
    }


    //  Copy constructor
    training_setReference(const training_set &dataset) : dataset::dataset(dataset.getattributes()), _dataset(&dataset)
    {
 
       
    }
        
    // Move constructor
    training_setReference(training_set &&dataset) : dataset::dataset(dataset.getattributes()), _dataset(&dataset)
    {
 
 
    }

    training_setReference(const training_setReference &right) : dataset::dataset(right.getattributes()), _dataset(right._dataset), _indices(right._indices.size() )  
    {
      //copied indices 
      for (size_t i = 0; i < _indices.size(); ++i)
        _indices[i] = right._indices[i];
    }
    training_setReference(training_setReference &&right) : dataset::dataset(right.getattributes()), _dataset(right._dataset), _indices(std::move(right._indices))
    {
      //moved indices 
    }

    // Copy assignment
    training_setReference &operator=(const training_set &dataset)
    {
      _dataset = &dataset;
      
       return *this;
    }
    // Move assignment
    training_setReference &operator=(training_set &&dataset)
    {
      _dataset = &dataset;
       return *this;
    }

    dataset *
    getNewReference(std::vector<size_t> &indices) const
    {
     std::vector<size_t> orig_indices(indices.size());
      for (size_t i = 0; i < orig_indices.size(); ++i)
        orig_indices[i] = _indices[indices[i]];
      return new training_setReference(*this->_dataset, orig_indices);
    }

    uint32_t
    size() const
    {
       return _indices.size();
    }

    unsigned int
    getattributesNumber() const
    {
      return _dataset->getattributesNumber();
    }

    const attribute &
    getattribute(uint32_t i, attribute_tag tag) const
    {
      if( _indices.size() < i && tag<_dataset->_samples[0].size()  )
      return _dataset->getattribute(_indices[i],tag);
      else
      return _dataset->getattribute(i,tag);

    }
    void
    setattribute(uint32_t i, attribute_tag tag, const attribute &value)
    {
      throw std::runtime_error(std::string("Can't modify a data set through a reference") + std::to_string(i) + std::to_string(tag) + std::to_string(value.continous())  );
    }
    inline const  samples_container &get_samples() const
    {
      return _dataset->_samples;
    }
    const attribute *getattributeptr(uint32_t i, attribute_tag tag, bool *found) const
    { 
      if(_indices.size() >= i)
      return _dataset->getattributeptr(_indices[i], tag, found);
      else
      return _dataset->getattributeptr(i, tag, found);
    }

    virtual ~training_setReference()
    {
      // Do nothing, we don't own the data
    }
    // Return the indices of the original data set
    const std::vector<size_t> &
    getindices() const
    {
      return _indices;
    }
  };
  typedef std::vector<attribute> testing_samples;
  // Class to hold the test data
  class testing_set : public dataset
  {
    // Container of samples. attributes are saved in contiguous memory locations.
    // Better memory access scheme for testing


    
    testing_samples  _samples;
    // Number of attributes to break, should be equal to the number of attributes on attribute_information

    uint32_t _nattr;
    //mmap_allocator<dataset> _allocator;


    void
    pushattribute(uint32_t n, const attribute &att)
    {
      // Ignore n, attributes are pushed sequentially

      ++n;//avoid warning
      _samples.push_back(att);
    }

    dataset *
    getNew() const
    {
      return new  testing_set(this->getattributes());
    }

    friend class testing_set_ref;
    dataset *
    getNewReference(std::vector<size_t> &indices) const;

  public:
    testing_set(const attribute_information &attributes_info) : dataset(attributes_info), _nattr(attributes_info.getSize())
    {
      
    }

    testing_set(const dataset &right);
    testing_set(const dataset &right,
                const attribute_information &attributes_info);

    testing_set(const testing_set &right) : dataset(right.getattributes()), _samples(right._samples), _nattr(
                                                                                                          right._nattr)
    {
    }

    testing_set(testing_set &&right) : dataset(right.getattributes()), _samples(std::move(right._samples)), _nattr(
                                                                                                                right._nattr)
    {
    }

    testing_set &
    operator=(const testing_set &right)
    {
      dataset::operator=(right);
      _samples = right._samples;
      _nattr = right._nattr;
      return *this;
    }

    testing_set &
    operator=(testing_set &&right)
    {
      dataset::operator=(right);
      _samples = std::move(right._samples);
      _nattr = right._nattr;
      return *this;
    }

    uint32_t
    size() const
    {
       return _samples.size() / _nattr;
    }

    const attribute &
    getattribute(uint32_t i, attribute_tag tag) const
    {
      return _samples[tag + ( i * _nattr)];
    }

    void
    setattribute(uint32_t i, attribute_tag tag, const attribute &value)
    {
      _samples[tag + (i * _nattr) ] = value;
    }
    inline const testing_samples &get_samples() const
    {
      // cache copying
       return   this->_samples;
    }
    const attribute *getattributeptr(uint32_t i, attribute_tag tag, bool *found) const
    {
      static attribute empty;
      size_t index = tag + (i * _nattr);

      if (index < _samples.size())
      {
        *found = true;
        return &_samples[index];
      }
      *found = false;
      return &empty;
    }
    virtual ~testing_set()
    {
      _samples.clear();
    }
  };

  // Class to hold the test data
  class testing_set_ref : public dataset
  {
    // Internal reference to the parent data set
    const testing_set *_dataset;

    // Indices of original data set
    std::vector<size_t> _indices;

    void
    pushattribute(uint32_t n, const attribute &att)
    {
      throw std::runtime_error(
          std::string("Can't push an attribute on a data set reference") + std::to_string(n) + std::to_string(att.continous())     );
    }

    dataset *
    getNew() const
    {
      return new testing_set_ref(*this->_dataset, this->_indices);
    }

  public:
    explicit testing_set_ref(const testing_set &dataset,
                             const std::vector<size_t> &indices) : dataset::dataset(dataset.getattributes()), _dataset(&dataset), _indices(indices)
    {
    }
    testing_set_ref(const testing_set_ref &right) : dataset::dataset(right.getattributes()), _dataset(right._dataset), _indices(right._indices)
    {
    }
    testing_set_ref(const testing_set_ref &right, const attribute_information &attributes_info) : dataset::dataset(attributes_info), _dataset(right._dataset), _indices(right._indices)
    {
 
    }
    testing_set_ref(const testing_set_ref &right, const attribute_information &attributes_info, const std::vector<size_t> &indices) : dataset::dataset(attributes_info), _dataset(right._dataset), _indices(indices)
    {
 
    }
    testing_set_ref(const testing_set &right) : dataset::dataset(right.getattributes()), _dataset(&right),_indices(_dataset->size())
    {

      for (size_t i = 0; i < _indices.size(); i++)
      {
        _indices[i] = i;
      }
    } 
    testing_set_ref(testing_set_ref &right, const std::vector<size_t> &indices) : dataset::dataset(right.getattributes()), _dataset(right._dataset), _indices(indices)
    {
    }
    testing_set_ref(testing_set_ref &&right) : dataset::dataset(right.getattributes()), _dataset(right._dataset), _indices(right._indices)
    {
    }
    testing_set_ref(testing_set_ref &&right, const attribute_information &attributes_info) : dataset::dataset(attributes_info), _dataset(right._dataset), _indices(right._indices)
    {
    }
    testing_set_ref(testing_set_ref &&right, const attribute_information &attributes_info, const std::vector<size_t> &indices) : dataset::dataset(attributes_info), _dataset(right._dataset), _indices(indices)
    {
    }
    testing_set_ref(testing_set_ref &&right, const attribute_information &attributes_info, std::vector<size_t> &&indices) : dataset::dataset(attributes_info), _dataset(right._dataset), _indices(indices)
    {
    }
    testing_set_ref(testing_set_ref &&right, const attribute_information &attributes_info, std::vector<size_t> &indices) : dataset::dataset(attributes_info), _dataset(right._dataset), _indices(indices)
    {
    }
    testing_set_ref(testing_set_ref &&right, const attribute_information &attributes_info, std::vector<size_t> &indices, bool) : dataset::dataset(attributes_info), _dataset(right._dataset), _indices(std::move(indices))
    {
    }
    testing_set_ref &operator=(testing_set_ref &&right)
    {
      _dataset = right._dataset;
      _indices = std::move(right._indices);
      return *this;
    }
    testing_set_ref &operator=(const testing_set_ref &right)
    {
      _dataset = right._dataset;
      _indices = right._indices;
      return *this;
    }
    testing_set_ref &operator=(testing_set_ref &right)
    {
      _dataset = right._dataset;
      _indices = right._indices;
      return *this;
    }
    testing_set_ref &operator=(const testing_set &right)
    {
      _dataset = &right;
      _indices.resize(right.size());
      for (size_t i = 0; i < _indices.size(); ++i)
        _indices[i] = i;
      return *this;
    }
    testing_set_ref &operator=(testing_set &right)
    {
      _dataset = &right;
      _indices.resize(right.size());
      for (size_t i = 0; i < _indices.size(); ++i)
        _indices[i] = i;
      return *this;
    }
    dataset *
    getNewReference(std::vector<size_t> &indices) const
    {
      std::vector<size_t> orig_indices(indices.size());
      for (size_t i = 0; i < orig_indices.size(); ++i)
        orig_indices[i] = _indices[indices[i]];
      return new testing_set_ref(*this->_dataset, orig_indices);
    }

    uint32_t
    size() const
    {
      return _indices.size();
    }

    const attribute &
    getattribute(uint32_t i, attribute_tag tag) const
    {
      return _dataset->_samples[tag + _indices[i] * _dataset->_nattr];
    }

    const attribute *
    getattributeptr(uint32_t i, attribute_tag tag, bool *found) const
    {
      static attribute empty;

      size_t index = ( i<_indices.size() )? (  tag + _indices[i] * _dataset->_nattr ) : (tag + (i*_dataset->_nattr) ) ; 

      if (index < _dataset->_samples.size())
      {
        *found = true;
        return &_dataset->_samples[index];
      }
      *found = false;
      return &empty;
    }

    void
    setattribute(uint32_t i, attribute_tag tag, const attribute &value)
    {
      throw std::runtime_error( std::string("Can't modify a data set through a reference ")  + std::to_string(i) +  std::to_string(tag) +std::to_string(value.continous()) +std::string(   __FILE__ ) +  std::string(":") + std::to_string(__LINE__));  
    }
    inline const  testing_samples &get_samples() const
    {

      return std::ref(_dataset->get_samples());
    }

    virtual ~testing_set_ref()
    {
    }
  };

  // Base class to collect data that defines a sample set and a collection
  // of attributes. The product of a DataCollector is a DataSet and an AttributesInformation
  // class.
  class data_collector
  {

  public:
    data_collector() = default;

    // Get attributes information (returns a reference)
    virtual const attribute_information &
    getAttributes() const = 0;

    // Push training cases on the data set
    virtual void
    pushTrainData(dataset *data) const = 0;

    // Push testing data into the data set
    virtual void
    pushTestData(dataset *data) const = 0;

    virtual ~data_collector() = default;
  };

  class files_collector : public data_collector
  {
    std::string _filestem;
    attribute_information _attributes_info;

  public:
    files_collector(const std::string &filestem);
    files_collector(const std::string &filestem,
                    const attribute_information &attrs);
    files_collector(
        const std::string &filestem, const std::string &target,
        const std::vector<std::pair<std::string, std::string>> &attrs);

    // Push CSV data from file
    static void
    pushFileData(const std::string &filename, dataset *data);

    // Parse the name file and return the attributes information
    static attribute_information
    parseNames(const std::string &names_file);

    const attribute_information &
    getAttributes() const
    {
      return _attributes_info;
    }

    void
    pushTrainData(dataset *data) const
    {
      pushFileData(_filestem + ".data", data);
    }

    void
    pushTestData(dataset *data) const
    {
      pushFileData(_filestem + ".test", data);
    }
 
    virtual ~files_collector()
    {
    }
  };

  class afiles_collector : public data_collector
  {
    // File steam
    std::string _filestem;
    // Target name
    std::string _target;
    // Save a map of original definitions
    std::vector<std::pair<std::string, std::string>> _orig_attributes;
    // Attributes information
    attribute_information _attributes_info;
    // Random number engine
    std::random_device _random;

    // Push CSV data from file
    void
    pushFileData(const std::string &filename, dataset *data) const;

    // Parse the name file and return the attributes information
    attribute_information
    parseNames(const std::string &names_file);

    // Add artificial attributes (no need to parse any file)
    attribute_information
    addArtificial(
        const std::string &target,
        const std::vector<std::pair<std::string, std::string>> &attrs);

  public:
    // Construct from a filename
    afiles_collector(const std::string &filestem,
                     const std::random_device &random);

    // Construct from attributes names and definitions
    afiles_collector(
        const std::string &filestem, const std::string &target,
        const std::vector<std::pair<std::string, std::string>> &attrs,
        const std::random_device &random);

    // This will include the attribute information ready to construct a classifier WITH the
    // artificial attributes
    const attribute_information &
    getAttributes() const
    {
      return _attributes_info;
    }

    void
    pushTrainData(dataset *data) const
    {
      pushFileData(_filestem + ".data", data);
    }

    void
    pushTestData(dataset *data) const
    {
      pushFileData(_filestem + ".test", data);
    }

    // Get attributes definition
    std::vector<std::pair<std::string, std::string>>
    getAttributesDefinition() const
    {
      // Return original attributes (name and definition)
      return _orig_attributes;
    }

    // Get target name
    std::string
    getTargetName() const
    {
      return _target;
    }

    virtual ~afiles_collector()
    {
    }
  };
 
  
  // Read importance / weight map from a file (and also check if the attributes are defined on the Attribute information
  // class)
  std::map<std::string, Float>
  getWeightMap(const attribute_information &attrs, std::ifstream &file);

  // Reorder attribute map and return a container with the attributes ordered from higher importance to lower
  std::vector<std::pair<std::string, Float>>
  sortImportances(const std::map<std::string, Float> &imp_map);

  // Print importance map
  void
  printImportanceMap(std::ostream &out,
                     const std::vector<std::pair<std::string, Float>> &imps);

} /* namespace provallo */

#endif /* dataset_H_ */
