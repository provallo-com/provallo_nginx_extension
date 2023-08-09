/*
 * attribute.h
 * generic attribute for ML Serialization
 *
 *
 *  Created on: May 12, 2021
 *      Author: kardon
 */

#ifndef ATTRIBUTE_H_
#define ATTRIBUTE_H_
#include <unistd.h>
#include <ctype.h>
#include <string>
#include <iostream>
#include <vector>
#include <map>
#include <typeinfo>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <strstream>
#include <sstream>
#include <atomic>
#include "utils.h"
#include "../glue/glueprocessinfo.h"


namespace provallo
{

  // types:
  typedef unsigned int tag;
  typedef size_t count;
  typedef tag attribute_tag;
  typedef std::string attribute_name;
  typedef std::string attribute_value;
  typedef attribute_tag discrete_value;
  typedef real_t cont_value;
  enum attribute_type
  {
    DISCRETE = 0,
    CONTINUOUS = 1,
    IGNORED = 2
  };
  enum split_type
  {
    DISC = 0,
    CONE_BINARY = 1,
    CONE_MULTI = 2,
    CONE_MDLP = 3,
    CONE_RANDOM = 4,
    HISTOGRAM = 5,
    IGNOREDSPLIT = 6
  };

  constexpr const cont_value UNKNOWN = 0160000000;
  constexpr const cont_value NA_VAL = 01;
  // bins for histogram
  template <typename T = double>
  class bin
  {
    std::vector<T> values;
    T min_, max_, delta_;
    T total;
    bin(const T &min, const T &max, const size_t &nbins) : values(nbins, 0.0), min_(min), max_(max), delta_(
                                                                                                         max - min / reinterpret_cast<T>(nbins))
    {
    }
    inline void
    accumulate(const T &value)
    {

      if (value > min_ && value < max_)
      {
        /* Accumulate value on histogram bin */
        double diff = (value - min_);
        /* Calculate bin position */
        size_t pos = diff / delta_;
        /* Accumulate */
        values[pos]++;
        /* Increment counter */
        total++;
      }
    }
    inline void
    normalize()
    {
      size_t i = 0;
      for (auto value : values)
      {
        if (value < 1E-20)
        {
          values.at(i) = value / total;
        }
        ++i;
      }
    }
    void
    debug();
    void
    print(const std::ostream &out) const;

    virtual ~bin()
    {
    }
  };
  // histogram
  template <class accum>
  class histogram : public accum
  {
  public:
    /* Print operator */
    template <class acc>
    friend std::ostream &
    operator<<(std::ostream &out, const histogram<acc> &q);

    /* Constructor (linear) */
    histogram(double min, double max, size_t nbins) : accum(min, max, nbins){
                                                          /* */
                                                      };

    /* Accumulate value */
    void
    operator()(double value)
    {
      accum::accumulate(value);
    }

    void
    print(std::ostream &out) const
    {
      accum::print(out);
    }

    virtual ~histogram()
    { /* */
    }
  };

  template <class accum>
  std::ostream &
  operator<<(std::ostream &out, const histogram<accum> &q)
  {
    q.print(out);
    return out;
  }
// attribute class  replace by cell type.
#pragma pack(0)
  class attribute
  {

  private:
    static std::atomic_uint64_t _id_counter;
    union value
    {
      discrete_value _disc;
      cont_value _cont;
    } _value;
    attribute_type _type;

  public:
    bool operator==(const attribute &other) const
    {
      if (other.NA() || other.unknown())
        return false;

      return discrete() == other.discrete() || continous() == other.continous();
    }

    explicit attribute(const discrete_value &v): _type(DISCRETE)
    {
      _value._disc = v; 
      _id_counter++;
    }
    explicit attribute(const cont_value &v) : _type(CONTINUOUS)
    {
       _value._cont = v;
      _id_counter++;
     
    }
    
    explicit attribute(const attribute_value &v , attribute_type t):_type(t)
    {
      _id_counter++;
      if(_type==DISCRETE)
      {
        std::stringstream ss(v);
        discrete_value c;
        ss >> c;
        _value._disc = c;
      }
      else
      {
        std::stringstream ss(v);
        cont_value c;
        ss >> c;
        _value._cont = c;
      }
    }

    explicit attribute(const attribute_value &v)
    {
      _id_counter++;
      if (v == "NA")
      {
        _value._cont = NA_VAL;
      }
      else if (v == "?")
      {
        _value._disc = UNKNOWN;
      }
      else
      {
        if(v.find(".") != std::string::npos)
        {
          std::stringstream ss(v);
          cont_value c;
          ss >> c;
          _value._cont = c;
          _type = CONTINUOUS;
        }
        else
        {
          std::stringstream ss(v);
          discrete_value c;
          ss >> c;
          _value._disc = c;
          _type = DISCRETE;
        }
      }
    }

    attribute() : _value()
    {
      _id_counter++;
      _value._cont = NA_VAL;
    }


    attribute(const attribute &other) : _value(other._value), _type(other._type)
    {
      _id_counter++;
     }
    attribute(const attribute &&other) : _value(other._value), _type(other._type)
    {
      _id_counter++;
    } 

    attribute &operator=(const attribute &&other)
    {
      _value = other._value;
      _type = other._type;
      return *this;
    }
    attribute &operator=(const attribute &other)
    {
      _value = other._value;
      _type = other._type;
      return *this;
    }
    attribute &operator=(const discrete_value &other)
    {
      _value._disc = other;
      _type = DISCRETE;
      return *this;
    }
    attribute &operator=(const cont_value &other)
    {
      _value._cont = other;
      _type = CONTINUOUS;
      return *this;
    }
    operator discrete_value() const
    { 
      if( _type == DISCRETE)
          return _value._disc;
      else
      return (discrete_value)_value._cont;

    }
    operator cont_value() const
    {
      if( _type == CONTINUOUS)
          return _value._cont;
      else 
      return (cont_value) _value._disc;
    }
    attribute &operator=(const attribute_value &other)
    {
      if (other == "NA")
      {
        _value._cont = NA_VAL;
      }
      else if (other == "?")
      {
        _value._disc = UNKNOWN;
      }
      else
      { 
        if(other.find(".") != std::string::npos)
        {
          std::stringstream ss(other);
          cont_value c;
          ss >> c;
          _value._cont = c;
          _type = CONTINUOUS;
        }
        else
        {
          std::stringstream ss(other);
          discrete_value c;
          ss >> c;
          _value._disc = c;
          _type = DISCRETE;
        }  
      }
      return *this;
    }
    inline discrete_value
    discrete() const
    {
      return _value._disc;
    }
    inline cont_value
    continous() const
    {
      return _value._cont;
    }
    bool
    unknown() const
    {
      return _value._disc == UNKNOWN;
    }
    bool
    NA() const
    {
      return (_value._cont == NA_VAL);
    }
    bool is_discrete() const
    {
      return _type == DISCRETE;
    }
    bool is_continous() const
    {
      return _type == CONTINUOUS;
    }
    attribute_type type() const
    {
      return _type;
    }
    std::string 
    to_string() const
    {
      if(is_discrete())
      {
        return std::to_string(_value._disc);
      }
      else
      {
        return std::to_string(_value._cont);
      }
    } 

    virtual ~attribute()
    {
// debug attributes allocation
#ifdef DEBUG_ATTRIBUTES_DESTRUCTOR

      static uint64_t destructor_count = 0;
      constexpr const uint64_t million = 1000000;
      constexpr const uint64_t ten_millions = 10000000;
      constexpr const uint64_t hundred_millions = 100000000;
      constexpr const uint64_t billion = 1000000000;
      constexpr const double million_d = 1000000.;
      destructor_count++;
      if (destructor_count % ten_millions == 0 && destructor_count / million_d > 0.0)
      {

        std::cout << "[attribute] destructor_count  : " << std::to_string(double(destructor_count / million_d)) << " allocated : " << std::to_string(double(uint64_t(attribute::_id_counter) / million_d)) << std::endl;
      }
#endif
    }
  };
  // Base class that defines an attribute. Given a concrete definition,
  // the object should return an Attribute object from a human readable
  // string. This class is mostly used in when parsing .names or .data
  class attribute_definition
  {
    // Name of the attribute
    attribute_name _name;
    // Attribute tag
    attribute_tag _tag;

    size_t _index;
    // Virtual method implemented on the derived class that returns an attribute from its value
    virtual attribute
    _getAttribute(const attribute_value &value) const = 0;
    // Virtual method to get a value from the data
    virtual attribute_value
    _getValue(const attribute &value) const = 0;
    // Virtual method to compare attributes definitions
    virtual bool
    _compare(const attribute_definition &other) const
    {
      return other.getTag() == getTag() && other.getName() == getName()
             && other.getIndex() == getIndex();

    }
    // Friendly printer
    friend std::ostream &
    operator<<(std::ostream &out, const attribute_definition &q);

  public:
    attribute_definition(const attribute_name &name, const attribute_tag &tag) : _name(name), _tag(tag), _index(0)
    {
       static size_t definition_index = 0;
         _index = definition_index++;
    }
    attribute_definition(const attribute_definition *deserial);

    // Get name of the attribute
    const attribute_name &
    getName() const
    {
      return _name;
    }

    size_t getIndex() const
    {
      return _index;
    }

    // Get tag of this attribute
    attribute_tag
    getTag() const
    {
      return _tag;
    }

    // Get attribute
    attribute
    getAttribute(const attribute_value &value) const
    {
      if (value == "?")
        return attribute(UNKNOWN);
      else
        return _getAttribute(value);
    }

    // Create attribute from a floating point number
    attribute
    getAttribute(const float &value) const
    {
      if (value == NA_VAL)
        return attribute(NA_VAL);
         return attribute((cont_value)value);
    }

    attribute
    getAttribute(const attribute &value) const
    {
      return value;
    }

    template <typename T>
    static inline T
    fromString(const std::string &str)
    {

      std::istringstream s(str);
      T t;
      s >> t;
      return t;
    }
    
    // Comparison operator
    bool
    operator==(const attribute_definition &other) const
    {
      if (_name != other._name || _tag != other._tag || typeid(*this) != typeid(other))
        return false;
      return _compare(other);
    }

    // Create attribute definitions
    static attribute_definition *
    getDefinition(const std::string &name, const attribute_tag &tag,
                  const std::string &str);
    static attribute_definition *
    getDefinition(const attribute_definition *deserial);

    virtual attribute_definition *
    clone() const = 0;

    virtual attribute_type
    get_type() const = 0;

    virtual size_t
    getCount() const = 0;

    // Get value given an internal number
    attribute_value
    getValue(const attribute &value) const
    {
      if (value.unknown())
        return "?";
      else
        return _getValue(value);
    }
    // Virtual method to write internal data on buffer
    virtual void
    serialize(attribute_definition *serial) const;

    // Print definition
    virtual void
    print(std::ostream &out) const = 0;

    virtual ~attribute_definition()
    {
    }
    // return index from const attribute_value &value
    virtual size_t getValueIndex(const attribute_value &value) const
    { 
      //ignore value
       if ( value.size() == 0)
        return 0;

       return this->_index;

    }

    // Get attribute definition from a string
    static attribute_definition *
    getDefinition(const std::string &str);
  };
  //specialize it
  template <>
       inline float attribute_definition::fromString<float>(const std::string &str)
    {
      if (str == "NA")
        return NA_VAL;
      else
        return std::stof(str);
    }
    template <>   inline double  attribute_definition::fromString<double>(const std::string &str)
    {
      if (str == "NA")
        return NA_VAL;
      else
        return std::stod(str);
    } 
    template <>   inline size_t  attribute_definition::fromString<size_t>(const std::string &str)
    {
      if (str == "NA")
        return NA_VAL;
      else
        return std::stoul(str);
    } 
  // Continuous attributes
  class continous_attribute : public attribute_definition
  {

    attribute
    _getAttribute(const attribute_value &value) const
    {
      // Convert string to continuous number
      cont_value att_value = fromString<cont_value>(value);
      // Return new attribute
      return attribute(att_value);
    }

    attribute_value
    _getValue(const attribute &value) const
    {
      if(value.is_discrete()  ) 
        return std::to_string(value.discrete());
      else
      {
        if(value.unknown())
        return "?";
      else if (value.continous() == NA_VAL)
        return "NA";
      else 
        return std::to_string(value.continous());
      }
            
    }

  public:
    continous_attribute(const attribute_name &name, const attribute_tag &tag) : attribute_definition(name, tag)
    {
    }

    continous_attribute(const attribute_definition *deserial) : attribute_definition(deserial)
    {
    }

    attribute_definition *
    clone() const
    {
      return new continous_attribute(getName(), getTag());
    }

    void
    print(std::ostream &out) const;

    static attribute_type
    _type()
    {
      return CONTINUOUS;
    }

    attribute_type
    get_type() const
    {
      return _type();
    }

    size_t
    getCount() const
    {
      return 0;
    }
     ~continous_attribute() = default;
  };

  // Ignored attributes
  class ignored_attribute : public attribute_definition
  {

    attribute
    _getAttribute(const attribute_value &value) const
    {
      return attribute(value);
    }

    attribute_value
    _getValue(const attribute &value) const
    {
      return std::to_string(value.discrete()) + " (ignored)"  ;
    }

  public:
    ignored_attribute(const attribute_name &name, const attribute_tag &tag) : attribute_definition(name, tag)
    {
    }

    ignored_attribute(attribute_definition *deserial) : attribute_definition(deserial)
    {
    }
    ignored_attribute(const attribute_definition &attr) : attribute_definition(&attr)
    {
    }
    attribute_definition *
    clone() const
    {
      return new ignored_attribute(getName(), getTag());
    }

    void
    print(std::ostream &out) const;

    static attribute_type
    _type()
    {
      return IGNORED;
    }

    attribute_type
    get_type() const
    {
      return _type();
    }

    size_t
    getCount() const
    {
      return 0;
    }

    virtual ~ignored_attribute()
    {
    }
  };

  // Discrete attributes
  class discrete_attribute : public attribute_definition
  {
    //   names to values
    std::map<attribute_value, discrete_value> _name_map;
    //   values to discrete names (nominal values is represented by the vector index)
    std::vector<attribute_value> _values_map;
    // Private constructor from internal data
    discrete_attribute(
        const attribute_name &name, const attribute_tag &tag,
        const std::map<attribute_value, discrete_value> &name_map,
        const std::vector<attribute_value> &values_map) : attribute_definition(name, tag), _name_map(name_map), _values_map(values_map)
    {
    }

    attribute
    _getAttribute(const attribute_value &value) const
    {
      // Check if the attribute is valid
      std::map<attribute_value, discrete_value>::const_iterator att =
          _name_map.find(value);
      // Return attribute
      return attribute((*att).second);
    };

    virtual attribute_value
    _getValue(const attribute &value) const
    {
      if ( value.discrete() < _values_map.size()  ) 
        return _values_map[value.discrete()];
      else if (value.discrete()==_values_map.size())
      {
        return  _values_map[value.discrete() -1];
      } 
      else{
        std::cerr<<"[-]"<<"discrete_attribute::_getValue: value out of range : "<<value.discrete()<<" "<<_values_map.size()<<std::endl; 
        std::cerr<<"[-] values map : "<<std::endl;
        for(auto v : _values_map)
        {
          std::cerr<<v<<std::endl;
        }
        std::cerr<<"[-] "<<std::endl;
        return "0";
        //throw std::runtime_error(std::string("discrete_attribute::_getValue: value out of range : ")+std::to_string(value.discrete())+" "+std::to_string(_values_map.size()))  ;  
      }
     }

    bool
    _compare(const attribute_definition &other) const;

    void
    serialize(attribute_definition *serial) const;

  public:
    // Construct from a set of attributes values (alphabetic order)
    discrete_attribute(const attribute_name &name, const attribute_tag &tag,
                       const std::vector<attribute_value> &attribute_values);
    discrete_attribute(const attribute_definition *deserial);

    attribute_definition *
    clone() const
    {
      return new discrete_attribute(getName(), getTag(), _name_map,
                                    _values_map);
    }

    void
    print(std::ostream &out) const;

    static attribute_type
    _type()
    {
      return DISCRETE;
    }

    attribute_type
    get_type() const
    {
      return _type();
    }

    size_t
    getCount() const
    {
      return _name_map.size();
    }

    virtual ~discrete_attribute()
    {
      static uint64_t count = 0;
      count++;
      if (count % 100000 == 0)
      {
        std::cout << "discrete_attribute destructor called " << count << std::endl;
      }
    }
  };

  // Print definition
  std::ostream &
  operator<<(std::ostream &out, const attribute_definition &q);

  class attribute_descriptor
  {
    attribute_name _name;
    attribute_tag _tag;

  public:
    attribute_descriptor(const attribute_name &n, const attribute_tag &t) : _name(n), _tag(t)
    {
    }

    virtual attribute_value
    get_value(const attribute &val) const = 0;
    virtual bool
    compare(const attribute_descriptor &other)
    {
      return other._tag == _tag&& other._name == _name; 
    }

    friend std::ostream &
    operator<<(std::string &out, const attribute_descriptor &attr);

    virtual ~attribute_descriptor()
    {
    }
  };
  // Class to handle a group of attributes
  class attribute_groups
  {
  public:
    attribute_groups()
    {
    }
    attribute_groups(const attribute_groups *deserial);
    attribute_groups(const std::vector<std::vector<attribute_tag>> &groups,
                     const std::vector<split_type> &split_type) : _groups(groups), _split_type(split_type), _type(groups.size(), CONTINUOUS )
    {
    }
    attribute_groups(const std::vector<std::vector<attribute_tag>> &groups,
                     const std::vector<split_type> &split_type,
                     const std::vector<attribute_type> &type) : _groups(groups), _split_type(split_type), _type(type)
    {
    }

    attribute_groups(const attribute_groups &other) : _groups(other._groups),
                                                      _split_type(other._split_type),
                                                      _type(other._type)
    {
    }
    attribute_groups(attribute_groups &other) : _groups(other._groups),
                                                _split_type(other._split_type),
                                                _type(other._type)
    { 
    }
    attribute_groups(attribute_groups *other) : _groups(other->_groups),
                                                _split_type(other->_split_type),
                                                _type(other->_type)
    {
    }

    attribute_groups(attribute_groups &&other)
    {
      _groups = std::move(other._groups);
      _split_type = std::move(other._split_type);
      _type = std::move(other._type);
    }
    attribute_groups &
    operator=(attribute_groups &&other)
    {
      _groups = std::move(other._groups);
      _split_type = std::move(other._split_type);
      _type = std::move(other._type);
      return *this;
    }
    attribute_groups &operator=(const attribute_groups &other)
    {
      _groups = other._groups;
      _split_type = other._split_type;
      _type = other._type;
      return *this;
    }

    virtual ~attribute_groups()
    {
#ifdef DEBUG_ATTRIBUTE_DESTRUCTOR_CALLS

      static std::atomic_uint64_t count = 0;
      count++;

      std::cout << "attribute_groups destructor called " << count << std::endl;
#endif
    }

    // Access to group of attributes
    const std::vector<attribute_tag> &
    getGroup(uint32_t i) const
    {
      assert(i < _groups.size());
      return _groups[i];
    }

    // Get split type of a group
    split_type
    getsplit_type(uint32_t i) const
    {
      assert(i < _split_type.size());
      return _split_type[i];
    }

    // Get type of a group (i.e. type of an effective attribute)
    attribute_type
    getattribute_type(uint32_t i) const
    {
      assert(i < _type.size());
      return _type[i];
    }

    // Set split type of a group
    void
    setsplit_type(uint32_t i, split_type type)
    {
      assert(i < _split_type.size());

      _split_type[i] = type;
      _type[i] = (type == DISC) ? DISCRETE : CONTINUOUS;
    }

    // Get size (total number of groups, i.e. the effective number of attributes)
    // The total number of attributes could be less or more than the original number
    size_t
    size() const
    {
      return _groups.size();
    }

    // Push attribute group with the split type
    void
    push(const std::vector<attribute_tag> &group, split_type type,
         attribute_type attr_type)
    {
      _groups.push_back(group);
      _split_type.push_back(type);
      _type.push_back(attr_type);
    }

    // Comparison operator
    bool
    operator==(const attribute_groups &other) const
    {

      return (other._groups == _groups && other._type == _type && other._split_type == _split_type);
    }
    bool operator!=(const attribute_groups &other) const
    {
      return other._groups != _groups || other._type != _type || other._split_type != _split_type;
    }

    //
    void clear()
    {
      _groups.clear();
      _split_type.clear();
      _type.clear();
    }

    // Serialize attribute information into the buffer
    void
    serialize(attribute_groups *serial) const;

  private:
    // Array of grouped attributes
    std::vector<std::vector<attribute_tag>> _groups;
    // Split methods for each group
    std::vector<split_type> _split_type;
    // Type of each *effective* attribute
    std::vector<attribute_type> _type;
  };

  // This class defines a set of attributes on a sample set. Once an object
  // is instantiated, is responsible for mapping tags into human readable names (and
  // vice versa) and return Attributes objects given a string (AttributeValue). This class
  // is also responsible to provide information about each attribute and the interaction between
  // them.
  class attribute_information
  {

    // Map of tag to names (vector index is the tag)
    std::vector<attribute_name> _tag_map;
    // Map of names to tags
    std::map<attribute_name, attribute_tag> _name_map;
    // Target attribute position
    attribute_tag _target_pos;
    // Map of definitions of each attribute (vector index is the tag)
    std::vector<attribute_definition *> _definition_map;
    // Number of values of each attribute
    std::vector<uint32_t> _count;
    // Type of each attribute (continuous or discrete)
    std::vector<attribute_type> _type;
    // Group of attributes, the target attribute does not appear here
    attribute_groups _groups;

    // Friendly printer
    friend std::ostream &
    operator<<(std::ostream &out, const attribute_information &q);

  public:
    // Construct from attributes list
    attribute_information(
        const attribute_name &target_name,
        const std::vector<std::pair<attribute_name, std::string>> &attributes);
    //copy constructor
    attribute_information(const attribute_information &right);
    //move constructor
    attribute_information(attribute_information &&right);
    //copy assignment
    attribute_information &operator=(const attribute_information &right);
    //move assignment
    attribute_information &operator=(attribute_information &&right);

    attribute_information(const attribute_information *deserial);

    // Get attribute by value
    template <class Value>
    attribute
    getAttribute(const attribute_tag &tag, const Value &value) const
    {
      return _definition_map[tag]->getAttribute(value);
    }

    // Get attribute name from the tag
    attribute_name
    getName(const attribute_tag &tag) const
    {
      return _tag_map[tag];
    }

    // Get attribute value
    attribute_value
    getValue(const attribute_tag &tag, const attribute &value) const
    {
      return _definition_map[tag]->getValue(value);
    }

    std::vector<attribute> get_values (const attribute_tag &tag) const
    {
      std::vector<attribute> ret;
      attribute_definition *def = _definition_map[tag];
      if(def)
      {
        for(size_t n = 0;n<def->getCount();n++)
        { 
          attribute val(def->getValue( attribute(std::to_string(n)) ));
          
          ret.push_back(val);
        }

      }
      return ret;
    }
    // Get definition of an attribute ("right hand" side on the .names file)
    std::string
    getDefinition(const attribute_tag &tag) const;

    // Get definition map
    std::vector<std::pair<attribute_name, std::string>>
    getDefinitionMap() const;


    //get the number of classes in the target attribute
    uint32_t
    getTargetClassCount() const
    {
      const constexpr uint32_t MAX_INT32 = 0x7FFFFFFF;

      if ( _target_pos > size_t(MAX_INT32) || _target_pos >= _count.size() )
        return 1;

      return _count[_target_pos];
    } 
     
    // Get number of attributes (including the target attribute)
    uint32_t
    getSize() const
    {
      return _tag_map.size();
    }

    // Get count of attribute values
    uint32_t
    getCount(const attribute_tag &tag) const
    {
      return _count[tag];
    }

    // Get type of an attribute
    attribute_type
    getType(const attribute_tag &tag) const
    {
      return _type[tag];
    }

    // Get target attribute tag
    attribute_tag
    get_target_tag() const
    {
      return _target_pos;
    }

    // Given a name returns a tag
    attribute_tag
    getTag(const attribute_name &name) const
    {
      std::map<attribute_name, attribute_tag>::const_iterator it =
          _name_map.find(name);
      if (it == _name_map.end())
      {
        
        std::cout << "Available attributes: ";
        for(auto &i : _name_map)
        {
          std::cout << i.first << " ";
        }
        std::cout << std::endl;
        throw std::runtime_error(std::string("Attribute name ")+name+" not found");
      }
      return (*it).second;
    }


    bool validate() const
    {
        bool ret = true;
        if (_target_pos >= _tag_map.size())
        {
          std::cout << "Target attribute position " << _target_pos << " is out of range" << std::endl;
          ret = false;
        }
        if (ret && _definition_map.size() != _tag_map.size())
        {
          std::cout << "Definition map size " << _definition_map.size() << " does not match tag map size " << _tag_map.size() << std::endl;
          ret = false;

        }
        if (ret &&  _count.size() != _tag_map.size())
        {
          std::cout << "Count map size " << _count.size() << " does not match tag map size " << _tag_map.size() << std::endl;
          ret = false;
        }
        if (ret &&  _type.size() != _tag_map.size())
        {
          std::cout << "Type map size " << _type.size() << " does not match tag map size " << _tag_map.size() << std::endl;
          ret = false;
        }
        
        if(ret && getCount(_target_pos) == 0)
        {
          std::cout << "Target attribute has no values" << std::endl;
          ret = false;
        } 
        if(ret && _groups.size() == 0)
        {
          std::cout << "No attribute groups defined" << std::endl;
          ret = false;
        }
        
        for (uint32_t i = 0; ret && i < _definition_map.size(); ++i)
        {
          if (_definition_map[i] == nullptr)
          {
            std::cout << "Attribute " << _tag_map[i] << " has no definition" << std::endl;
            ret = false;
          }
        }
        

        return ret;

    }

    // Get constant reference to the attribute groups
    const attribute_groups &
    getGroups() const
    {
      return _groups;
    }

    // Get mutable reference to the attribute groups
    attribute_groups &
    getGroups()
    {
      return _groups;
    }
    bool operator!=(const attribute_information &other)
    {
      if ((_target_pos != other._target_pos) || (_definition_map.size() != other._definition_map.size()))
      {
        return true;
      }
      if (_groups != other._groups)
      {
        return true;
      }
      for (uint32_t i = 0; i < _definition_map.size(); ++i)
      {
        if (!(*_definition_map[i] == *other._definition_map[i]))
        {
          return true;
        }
      }
      return false;
    }
    // Comparison operator
    bool
    operator==(const attribute_information &other) const
    {
      if (_target_pos != other._target_pos || _definition_map.size() != other._definition_map.size())
        return false;
      // Check groups
      if (not(_groups == other._groups))
        return false;
      // Check definitions
      for (uint32_t i = 0; i < _definition_map.size(); ++i)
        if (not(*_definition_map[i] == *other._definition_map[i]))
          return false;
      return true;
    }

    // Serialize attribute information into the buffer
    void
    serialize(attribute_information *serial) const;

     // get value index
    uint32_t getValueIndex(const attribute_tag &tag, const attribute &value) const
    {
      if ( tag < _definition_map.size())
        return _definition_map[tag]->getValueIndex(value.to_string());
      else
        return value.discrete();
    }
    // get value index
    uint32_t getValueIndex(const attribute_tag &tag, const attribute_value &value) const
    {
      if ( tag < _definition_map.size())
      return _definition_map[tag]->getValueIndex(value);
      else
      return attribute(value).discrete();
    }

    //destruct
    ~attribute_information()
    {
      for (uint32_t i = 0; i < _definition_map.size(); ++i)
        delete _definition_map[i];
    } 
    
  private:
  };
  // Print attributes information
  std::ostream &
  operator<<(std::ostream &out, const attribute_information &q);
  template <class Container>
  void
  parseNormalizedString(const attribute_information &info,
                        const std::string &normalized, Container *container)
  {
    std::vector<std::string> sample;
    tokenize(normalized, sample, ",");
    std::vector<std::string>::const_iterator begin(sample.begin());
    // Sanity check
    // assert(sample.size () == info.getSize () - 1);
    // Target attribute
    uint32_t target_tag(info.get_target_tag());
    // Loop over attributes
    for (uint32_t i = 0; i < info.getSize(); ++i)
    {
      if (i == target_tag)
        container->push_back(attribute(discrete_value(0)));
      else
        container->push_back(info.getAttribute(i, (*begin++)));
    }
  }

} /* namespace provallo */

#endif /* ATTRIBUTE_H_ */
