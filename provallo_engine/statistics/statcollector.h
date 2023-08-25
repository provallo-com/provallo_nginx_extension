/*
 * statcollector.h
 *
 *  Created on: Jun 28, 2021
 *      Author: kardon
 */

#ifndef STATISTICS_STATCOLLECTOR_H_
#define STATISTICS_STATCOLLECTOR_H_
// abstract collector
#include <string>
#include <map>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <regex>
#include <variant>
#include <assert.h>

//patch to_string for std::string to avoid if  
namespace std
{
  inline const std::string &to_string(const std::string &s)
  {
    return s;
  }
}
// #define DEBUG_MAPS

namespace provallo
{
  //==========================================
  // stat_collector
  //==========================================
  //
  // collect realtime stats from procfs / netlink /etc...
  // re-order and apply realtime operators on the data
  //

  class field_base
  {

  protected:
    // data_field have a name
    std::string _name;
    // An absolute time stamp (measure from a monotonic clock source)
    real_t _timestamp;
    // Type of field (i.e continuous, discrete or ignore)
    std::string _type;

  public:
    field_base(const std::string &name, const std::string &type) : _name(name), _timestamp(0.0), _type(type){};

    // Get name of this field
    std::string
    get_name() const
    {
      return _name;
    }

    // Set time stamp of this field (this value should be updated each time the field value is collected)
    void
    setTimeStamp(const real_t &timestamp)
    {
      _timestamp = timestamp;
    }

    // Get time stamp
    real_t
    getTimeStamp() const
    {
      return _timestamp;
    }

    // Get type
    std::string
    get_type()
    {
      return _type;
    }

    // Get internal values as a string
    virtual const std::string &
    getValue() const = 0;

    // Set field from string (generic conversion from string)
    virtual void
    setValue(const std::string &value) = 0;

    // Clone field
    virtual field_base *
    clone() const = 0;

    virtual ~field_base()
    {
    }
    virtual void print()
    {
      std::cout << "[+] field " << get_name() << std::string(" : [") << getValue() << std::string(" ]") << std::endl;
    }
  };

  template <class InternalClass>
  class data_field : public field_base
  {

  protected:
    // Internal value of the field
    InternalClass _value;

  public:
    data_field(const std::string &name, const std::string &type) : field_base(name, type)
    {
    }

    // Set field
    void
    setdata_field(const InternalClass &value)
    {
      _value = value;
    }

    // Get field
    const InternalClass &
    getdata_field() const
    {
      return _value;
    }

    virtual void
    setValue(const std::string &value);

    virtual const std::string &getValue() const
    {
      static std::string s_buffer;
      s_buffer.clear();
      s_buffer = std::to_string(getdata_field());
      return s_buffer;
    }

    // Clone field
    field_base *
    clone() const
    {
      data_field<InternalClass> *new_ptr = new data_field<InternalClass>(
          _name, _type);
      new_ptr->setdata_field(_value);
      return new_ptr;
    }

    virtual ~data_field(){};
  };

  template <class InternalClass>
  void
  data_field<InternalClass>::setValue(const std::string &value)
  {
    // Only works if the Internal class has operator>> defined
    std::stringstream ss(value);
    ss >> _value;
  }

  struct data_fieldOp
  {
    // Virtual operator (returns a string with the result)
    virtual const std::string &
    operator()(const field_base &l, const field_base &r) const = 0;
    virtual ~data_fieldOp(){

    };
  };

  // Different types of operations
  template <class InternalClass>
  struct Sum : public data_fieldOp
  {
    const std::string &
    operator()(const field_base &l, const field_base &r) const;
    virtual ~Sum(){};
  };

  template <class InternalClass>
  const std::string &
  Sum<InternalClass>::operator()(const field_base &l,
                                 const field_base &r) const
  {
    std::stringstream ss;
    InternalClass sum =
        static_cast<const data_field<InternalClass> *>(&l)->getdata_field() + static_cast<const data_field<InternalClass> *>(&r)->getdata_field();

    return std::to_string(sum);
  }

  // Different types of operations
  template <class InternalClass>
  struct Less : public data_fieldOp
  {
    const std::string &
    operator()(const field_base &l, const field_base &r) const;
    virtual ~Less(){};
  };

  template <class InternalClass>
  const std::string &
  Less<InternalClass>::operator()(const field_base &l,
                                  const field_base &r) const
  {
    static std::string ret_;
    std::stringstream ss;
    ret_.clear();
    InternalClass less =
        static_cast<const data_field<InternalClass> *>(&l)->getdata_field() - static_cast<const data_field<InternalClass> *>(&r)->getdata_field();
    ss << less;
    ret_ = ss.str();
    return ret_;
  }

  // Different types of operations
  template <class InternalClass>
  struct Changed : public data_fieldOp
  {
    const std::string &
    operator()(const field_base &l, const field_base &r) const;
    virtual ~Changed(){

    };
  };

  template <class InternalClass>
  const std::string &
  Changed<InternalClass>::operator()(const field_base &l,
                                     const field_base &r) const
  {
    static const std::string one = "1", zero = "0";
    if (static_cast<const data_field<InternalClass> *>(&l)->getdata_field() == static_cast<const data_field<InternalClass> *>(&r)->getdata_field())
      return zero;
    return one;
  }

  // Keep last value
  struct LastValue : public data_fieldOp
  {
    const std::string &
    operator()(const field_base &l, const field_base &r) const
    {
      //always return the last value, reference new value for pedantic reasons
      if(r.getValue()!=l.getValue() && r.getValue()!="0")
            return l.getValue();
      return l.getValue();
    }
  };

  // Dummy operation
  struct Dummy : public data_fieldOp
  {
    const std::string &
    operator()(const field_base &l, const field_base &r) const
    {
      
      static const std::string zero = "0";
      if(r.getValue()==l.getValue())  
            return zero;
      return zero;

    }
  };

  // First order differentiation
  template <class InternalClass>
  struct FirstDeriv : public data_fieldOp
  {
    const std::string &
    operator()(const field_base &l, const field_base &r) const
    {
      static std::string ret_;

      real_t deriv(0.0);
      // Sanity check
      if ((l.getTimeStamp() - r.getTimeStamp()) < 1E-06)
        deriv = 0.0;
      else
        deriv =
            ((real_t)static_cast<const data_field<InternalClass> *>(&l)->getdata_field() - (real_t)static_cast<const data_field<InternalClass> *>(&r)->getdata_field()) / (l.getTimeStamp() - r.getTimeStamp());

      ret_ = std::to_string(deriv);

      return ret_;
    }
    virtual ~FirstDeriv()
    {
    }
  };

  // Calculate the delta in time
  struct DeltaTime : public data_fieldOp
  {
    const std::string &
    operator()(const field_base &l, const field_base &r) const
    {
      static std::string ret_;

      ret_ = std::to_string(l.getTimeStamp() - r.getTimeStamp());
      return ret_;
    }
    virtual ~DeltaTime()
    {
    }
  };

  // Compare to a value specified in the constructor (returns 1 if the field value is
  // equal to the internal value and 0 otherwise)
  template <class InternalClass>
  struct Comparison : public data_fieldOp
  {
    Comparison(const InternalClass &value) : _value(value)
    {
    }
    const std::string &
    operator()(const field_base &l, const field_base &r) const
    {

      static const std::string one = "1", zero = "0";

      if (static_cast<const data_field<InternalClass> *>(&l)->getdata_field() == _value)
        return one;
      return zero;
    }
    virtual ~Comparison(){};

  private:
    InternalClass _value;
  };
  class stat_collector;

  class proc_collector
  {
    // Container of ProcTable's
    friend class stat_collector;
    std::vector<stat_collector *> _tables;

  public:
    proc_collector();

    // Push table
    void
    push_table(stat_collector *table)
    {
      _tables.push_back(table);
    }

    // Parse data (using the parse data from each table)
    void
    collect() const;

    // Get normalized string obtained from each table copy string.
    const std::string &
    get_normalized() const;

    std::string
    get_normalized_filter_ignored() const;
    void
    get_attribute_information(
        std::vector<std::pair<std::string, std::string>> *attributes) const;

    template <class Iterator>
    void
    setIgnored(Iterator begin, Iterator end);

    // template <typename T>
    // const std::vector<typename T>& get_real()const;

    // Get last sample as a normalized string
    std::string
    get_last_sample() const;

    stat_collector *
    get_table(size_t idx) const;

    size_t
    size() const
    {
      return _tables.size();
    }
    virtual ~proc_collector();
  };

  class stat_collector
  {

  protected:
    std::map<std::string, size_t> _field_indices;
    std::string _name;

    enum types : uint16_t
    {
      BYTE,
      INT64,
      INT32,
      UINT32,
      UINT64,
      TEXT,
      REAL
    };

    typedef std::vector<field_base *> fields_instance;
    std::vector<fields_instance *> _instances;
    
    std::vector<data_fieldOp *> _operations;
   
    std::vector<bool> _active;

    
    std::vector<std::string> _queries;
    
    bool _collected;
    bool _parsed;
    
    // normalize original data column name, type to collect,  db type to insert
    std::map<size_t, std::pair<uint16_t, std::string>> _type_exception_pair_map; // map indice to type :
    friend std::ostream &
    operator<<(std::ostream &out, const stat_collector &q);
    real_t
    getTimeStamp() const;

  public:
    // maybe replace with std::variant

    struct stat_value_cell
    {

      union values
      {
        uint64_t _ull;
        double _real;
        uint8_t _bvalue;
        uint32_t _unsigned;
        uint32_t _signed;
        int64_t _long;

      } u_value;
      std::string _txt;
      enum stat_collector::types _type;
      stat_value_cell() : _type(TEXT)
      {
        u_value._ull = 0;
        _txt = "";
      }

      explicit stat_value_cell(const std::string &value) : _type(TEXT)
      {
        u_value._ull = 0;
        _txt = value;
      }

      explicit stat_value_cell(long long unsigned int &value) : _type(UINT64)
      {
        u_value._ull = value;
      }

      explicit stat_value_cell(uint64_t &value) : _type(UINT64)
      {
        u_value._ull = value;
      }
      explicit stat_value_cell(const double &value) : _type(REAL)
      {
        u_value._real = value;
      }
      explicit stat_value_cell(const uint8_t &value) : _type(BYTE)
      {
        u_value._bvalue = value;
      }
      explicit stat_value_cell(const uint32_t &value) : _type(UINT32)
      {
        u_value._unsigned = value;
      }

      explicit stat_value_cell(const int32_t &value) : _type(INT32)
      {
        u_value._signed = value;
      }
      explicit stat_value_cell(const int64_t &value) : _type(INT64)
      {
        u_value._long = value;
      }

      inline void
      remap_to_string()
      {
        this->_txt = this->to_string();
        this->_type = TEXT;
      }
      inline void
      remap_to_int32()
      {
        if (_txt.size() > 1)
          this->u_value._signed = std::atoi(_txt.c_str());
        else
          this->u_value._signed = 0;
        _type = INT32;
      }
      inline void
      remap_to_uint32()
      {
        if (_txt.size() > 1)
          this->u_value._unsigned = std::strtoul(_txt.c_str(), nullptr, 10);
        else
          this->u_value._unsigned = 0;
        _type = UINT32;
      }
      void
      remap_to_int64()
      {
        if (_txt.size() > 1)
          this->u_value._long = std::strtoll(_txt.c_str(), nullptr, 10);
        else
          this->u_value._long = 0;
        _type = INT64;
      }
      inline void
      remap_to_uint64()
      {
        if (_txt.size() > 1)
          this->u_value._ull = std::strtoull(_txt.c_str(), nullptr, 10);
        else
          this->u_value._ull = 0;
        _type = UINT64;
      }
      inline void
      remap_to_real()
      {
        if (_txt.size() > 1)
          this->u_value._real = std::strtold(_txt.c_str(), nullptr);
        else
          this->u_value._real = 0.;
        _type = REAL;
      }

      inline std::string
      to_string()
      {
        return (_type == BYTE) ? std::to_string(u_value._bvalue) : (_type == INT64) ? std::to_string(u_value._long)
                                                               : (_type == INT32)   ? std::to_string(u_value._signed)
                                                               : (_type == UINT32)  ? std::to_string(u_value._unsigned)
                                                               : (_type == UINT64)  ? std::to_string(u_value._ull)
                                                               : (_type == REAL)    ? std::to_string(u_value._real)
                                                                                    : _txt;
      }
    };

    std::map<std::string, struct stat_collector::stat_value_cell> _stats;

    stat_collector(const std::string &n) : _name(n), _instances(1), _collected(false),_parsed(false)
    {
      _instances[0] = new fields_instance;
    }

    virtual void
    collect() = 0;

    const std::string &get_name() const { return _name; }
    const std::string &drop_table_query() const
    {

      static const std::string drop = std::string("DROP TABLE  ") + std::string(_name) + "_TABLE;";
      return drop;
    }

    template <class container>
    static void
    tokenize(const std::string &str, container &tokens,
             const std::string &delim = ",", const std::string &prefix = std::string());

    void
    print_map()
    {
      for (auto it : _stats)
      {
        std::cout << it.first << std::string("=") << it.second.to_string();
      }
    }

    const std::vector<fields_instance *> &
    instances() const
    {
      return _instances;
    }

    virtual void
    strip_key_value(const std::string &line, bool key_header = true);
    virtual void
    validate_entry(const std::string &name, size_t s, std::string *entry);

    const std::map<std::string, struct stat_collector::stat_value_cell> &
    get_stats() const
    {
      return _stats;
    }
    // return insertion query data, column headers for create table and sample data data
    // ret[0] -> param_descriptors i.e. $paramname BIGINT NOT NULL DEFAULT 0,...,
    // ret[1] -> value_descriptors_sample  i.e. %d,%d...,
    // ret[2] ->
    virtual std::vector<std::string>
    parse_map_to_datatable();

    virtual std::string
    create_table_query();

    virtual std::string
    insert_query();

    // when new values appear, alter table to reflect new attributes.

    virtual std::string
    runtime_migrate_table(std::vector<std::string> new_columns) const; // returns migration for the table from N to M columns

    const std::string &
    get_normalized() const;
    std::string
    get_normalized_filter_ignored() const;
    size_t
    size() const
    {
      return _operations.size();
    }

    size_t
    instancesNumber() const
    {
      return _instances.size();
    }

    std::string
    getLastSample() const;

    std::vector<field_base *>::const_iterator
    getField(const std::string &name) const;

    std::vector<field_base *>::iterator
    getField(const std::string &name);

    std::string
    getValue(size_t idx) const
    {
      return (*_operations[idx])(*(*_instances[1])[idx],
                                 *(*_instances[0])[idx]);
    }

    std::string
    getLastValue(size_t idx) const
    {
      size_t last_instance = _instances.size() - 1;
      return (*_instances[last_instance])[idx]->getValue();
    }

    void
    printLastSampleValues(std::ostream &out) const;

    std::vector<field_base *>::iterator
    begin()
    {
      size_t last_instance = _instances.size() - 1;
      return _instances[last_instance]->begin();
    }
    std::vector<field_base *>::const_iterator
    begin() const
    {
      size_t last_instance = _instances.size() - 1;
      return _instances[last_instance]->begin();
    }
    std::vector<field_base *>::iterator
    end()
    {
      size_t last_instance = _instances.size() - 1;
      return _instances[last_instance]->end();
    }
    std::vector<field_base *>::const_iterator
    end() const
    {
      size_t last_instance = _instances.size() - 1;
      return _instances[last_instance]->end();
    }
    // Put field using a string container
    template <class InputIterator>
    void
    putValuesFromStrings(InputIterator begin, InputIterator end);

    // Put field using a container of the correct type
    template <class InputIterator>
    void
    putValues(InputIterator begin, InputIterator end);

    // Put field using a container of <string value, time stamp>
    template <class InputIterator>
    void
    putValuesAndTimestamps(InputIterator begin, InputIterator end);

    // Register field on this table - not virtual to be called from constructor 
    void
    register_field(field_base *field, data_fieldOp *op);
    // Get pointer to put new data
    std::vector<field_base *> *
    getNewDataPtr();

    // Push default values (in case something is wrong, just push all zeroes to the table)
    void
    pushDefaultValues();

    // Set active flag
    void
    set_active(std::vector<bool>::const_iterator it, bool flag)
    {

      size_t del = it - _active.begin();
      set_active(del, flag);
    }

    // Set active flag
    void
    set_active(size_t idx, bool flag)
    {
      _active[idx % _active.size()] = flag;
    }
    virtual ~stat_collector()
    { 
      auto instance_ = _instances[0 ] ;
      for (auto item : *instance_)
           delete item;
      if (_instances.size() > 1 )
      {
        for (auto in : _instances)
          delete in;
      }

      for (auto item : _operations)
        delete item;

     _instances.clear();
     _operations.clear();
     _active.clear();
      _stats.clear();
     }
   
  protected:        

      // protected constructor for empty collector
    stat_collector(const std::string &name, size_t instances) : _name(name), _instances(), _operations(), _active(), _stats() 
    {
      _instances.reserve(instances);
    } 
    
     
  };
  class empty_collector : public stat_collector
  {
  public:
    explicit empty_collector(size_t);
    virtual void
    collect()
    {
      // do nothing.
    } 
    virtual ~empty_collector() 
    {


    }
  };

  // snmp4_collector

  class snmp4_collector : virtual public stat_collector
  {
    bool _init;

  public:
    snmp4_collector() :stat_collector("SNMPv4"),  _init(false), _ip_size(0), _tcp_size(0), _icmp_size(0), _udp_size(
                                                                                    0),
                        _udplite_size(0)
    {

      register_field(new data_field<int>("Forwarding", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("DefaultTTL", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("InReceives", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("InHdrErrors", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("InAddrErrors", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("ForwDatagrams", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("InUnknownProtos", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("InDiscards", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("InDelivers", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("OutRequests", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("OutDiscards", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("OutNoRoutes", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("ReasmTimeout", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("ReasmReqds", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("ReasmOKs", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("ReasmFails", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("FragOKs", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("FragFails", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("FragCreates", "continuous"),
                     new FirstDeriv<int>());
      _ip_size = size();
      // TCP columns
      register_field(new data_field<int>("RtoAlgorithm", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("RtoMin", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("RtoMax", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("MaxConn", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("ActiveOpens", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("PassiveOpens", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("AttemptFails", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("EstabResets", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("CurrEstab", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("InSegs", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("OutSegs", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("RetransSegs", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("InErrs", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("OutRsts", "continuous"),
                     new FirstDeriv<int>());
      _tcp_size = size() - _ip_size;
      // ICMP columns
      register_field(new data_field<int>("InMsgs", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("InErrors", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("InDestUnreachs", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("InTimeExcds", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("InParmProbs", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("InSrcQuenchs", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("InRedirects", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("InEchos", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("InEchoReps", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("InTimestamps", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("InTimestampReps", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("InAddrMasks", "ignore"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("InAddrMaskReps", "ignore"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("OutMsgs", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("OutErrors", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("OutDestUnreachs", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("OutTimeExcds", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("OutParmProbs", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("OutSrcQuenchs", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("OutRedirects", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("OutEchos", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("OutEchoReps", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("OutTimestamps", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("OutTimestampReps", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("OutAddrMasks", "ignore"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("OutAddrMaskReps", "ignore"),
                     new FirstDeriv<int>());
      _icmp_size = size() - _tcp_size - _ip_size;
      // Udp columns
      register_field(new data_field<int>("UdpInDatagrams", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("UdpNoPorts", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("UdpInErrors", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("UdpOutDatagrams", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("UdpRcvbufErrors", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("UdpSndbufErrors", "continuous"),
                     new FirstDeriv<int>());
      _udp_size = size() - _tcp_size - _ip_size - _icmp_size;
      // Udp Lite columns
      register_field(new data_field<int>("UdpLiteInDatagrams", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("UdpLiteNoPorts", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("UdpLiteInErrors", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("UdpLiteOutDatagrams", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("UdpLiteRcvbufErrors", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("UdpLiteSndbufErrors", "continuous"),
                     new FirstDeriv<int>());
      _udplite_size = size() - _tcp_size - _ip_size - _icmp_size - _udp_size;
      _init = true;
    }
    virtual void
    collect();
    size_t _ip_size, _tcp_size, _icmp_size, _udp_size, _udplite_size;
  };

  // snmp6_collector
  class snmp6_collector : virtual public stat_collector
  {
  public:
    snmp6_collector() : stat_collector("SNMPv6")
    {

      register_field(new data_field<int>("Ip6InReceives", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Ip6InHdrErrors", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Ip6InTooBigErrors", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Ip6InNoRoutes", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Ip6InAddrErrors", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Ip6InUnknownProtos", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Ip6InTruncatedPkts", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Ip6InDiscards", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Ip6InDelivers", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Ip6OutForwDatagrams", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Ip6OutRequests", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Ip6OutDiscards", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Ip6OutNoRoutes", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Ip6ReasmTimeout", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Ip6ReasmReqds", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Ip6ReasmOKs", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Ip6ReasmFails", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Ip6FragOKs", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Ip6FragFails", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Ip6FragCreates", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Ip6InMcastPkts", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Ip6OutMcastPkts", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Ip6InOctets", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Ip6OutOctets", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Ip6InMcastOctets", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Ip6OutMcastOctets", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Ip6InBcastOctets", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Ip6OutBcastOctets", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Icmp6InMsgs", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Icmp6InErrors", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Icmp6OutMsgs", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Icmp6OutErrors", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Icmp6InDestUnreachs", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Icmp6InPktTooBigs", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Icmp6InTimeExcds", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Icmp6InParmProblems", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Icmp6InEchos", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Icmp6InEchoReplies", "continuous"),
                     new FirstDeriv<int>());
      register_field(
          new data_field<int>("Icmp6InGroupMembQueries", "continuous"),
          new FirstDeriv<int>());
      register_field(
          new data_field<int>("Icmp6InGroupMembResponses", "continuous"),
          new FirstDeriv<int>());
      register_field(
          new data_field<int>("Icmp6InGroupMembReductions", "continuous"),
          new FirstDeriv<int>());
      register_field(
          new data_field<int>("Icmp6InRouterSolicits", "continuous"),
          new FirstDeriv<int>());
      register_field(
          new data_field<int>("Icmp6InRouterAdvertisements", "continuous"),
          new FirstDeriv<int>());
      register_field(
          new data_field<int>("Icmp6InNeighborSolicits", "continuous"),
          new FirstDeriv<int>());
      register_field(
          new data_field<int>("Icmp6InNeighborAdvertisements", "continuous"),
          new FirstDeriv<int>());
      register_field(new data_field<int>("Icmp6InRedirects", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Icmp6InMLDv2Reports", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Icmp6OutDestUnreachs", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Icmp6OutPktTooBigs", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Icmp6OutTimeExcds", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Icmp6OutParmProblems", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Icmp6OutEchos", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Icmp6OutEchoReplies", "continuous"),
                     new FirstDeriv<int>());
      register_field(
          new data_field<int>("Icmp6OutGroupMembQueries", "continuous"),
          new FirstDeriv<int>());
      register_field(
          new data_field<int>("Icmp6OutGroupMembResponses", "continuous"),
          new FirstDeriv<int>());
      register_field(
          new data_field<int>("Icmp6OutGroupMembReductions", "continuous"),
          new FirstDeriv<int>());
      register_field(
          new data_field<int>("Icmp6OutRouterSolicits", "continuous"),
          new FirstDeriv<int>());
      register_field(
          new data_field<int>("Icmp6OutRouterAdvertisements", "continuous"),
          new FirstDeriv<int>());
      register_field(
          new data_field<int>("Icmp6OutNeighborSolicits", "continuous"),
          new FirstDeriv<int>());
      register_field(
          new data_field<int>("Icmp6OutNeighborAdvertisements", "continuous"),
          new FirstDeriv<int>());
      register_field(new data_field<int>("Icmp6OutRedirects", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Icmp6OutMLDv2Reports", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Udp6InDatagrams", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Udp6NoPorts", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Udp6InErrors", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Udp6OutDatagrams", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Udp6RcvbufErrors", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("Udp6SndbufErrors", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("UdpLite6InDatagrams", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("UdpLite6NoPorts", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("UdpLite6InErrors", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("UdpLite6OutDatagrams", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("UdpLite6RcvbufErrors", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("UdpLite6SndbufErrors", "continuous"),
                     new FirstDeriv<int>());
    }
    virtual void
    collect();
  };

  // sockstat_collector
  class sockstat_collector : virtual public stat_collector
  {
  public:
    sockstat_collector() : stat_collector("sockstats")
    {
    }
    virtual void
    collect();
  };

  // sockstat64_collector
  class sockstat64_collector : virtual public stat_collector
  {
  public:
    sockstat64_collector() : stat_collector("sockstats6")
    {
    }
    virtual void
    collect();
  };

  //
  // sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
  // 0: 3500007F:0035 00000000:0000 0A 00000000:00000000 00:00000000 00000000   101        0 34345 1 0000000000000000 100 0 0 10 0
  // 1: 0100007F:0277 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 30545 1 0000000000000000 100 0 0 10 0

  class tcp_stats : virtual public stat_collector
  {
    tcp_stats() : stat_collector("tcp_stats")
    {
    }
    virtual void
    collect();
  };

  class udp_stats : virtual public stat_collector
  {
    udp_stats() : stat_collector("udp_stats")
    {
    }
    virtual void
    collect();
  };

  class raw_stats : virtual public stat_collector
  {
    raw_stats() : stat_collector("raw_ip_stats")
    {
    }
    virtual void
    collect();
  };

  class raw6_stats : virtual public stat_collector
  {
    raw6_stats() : stat_collector("raw_ip6_stats")
    {
    }
    virtual void
    collect();
  };

  class net_stats : virtual public stat_collector
  {
  public:
    net_stats() : stat_collector("net_stats")
    {
      register_field(new data_field<int64_t>("InNoRoutes", "continuous"),
                     new FirstDeriv<int64_t>());
      register_field(new data_field<int64_t>("InTruncatedPkts", "continuous"),
                     new FirstDeriv<int64_t>());
      register_field(new data_field<int64_t>("InMcastPkts", "continuous"),
                     new FirstDeriv<int64_t>());
      register_field(new data_field<int64_t>("OutMcastPkts", "continuous"),
                     new FirstDeriv<int64_t>());
      register_field(new data_field<int64_t>("InBcastPkts", "continuous"),
                     new FirstDeriv<int64_t>());
      register_field(new data_field<int64_t>("OutBcastPkts", "continuous"),
                     new FirstDeriv<int64_t>());
      register_field(new data_field<int64_t>("InOctets", "continuous"),
                     new FirstDeriv<int64_t>());
      register_field(new data_field<int64_t>("OutOctets", "continuous"),
                     new FirstDeriv<int64_t>());
      register_field(new data_field<int64_t>("InMcastOctets", "continuous"),
                     new FirstDeriv<int64_t>());
      register_field(new data_field<int64_t>("OutMcastOctets", "continuous"),
                     new FirstDeriv<int64_t>());
      register_field(new data_field<int64_t>("InBcastOctets", "continuous"),
                     new FirstDeriv<int64_t>());
      register_field(new data_field<int64_t>("OutBcastOctets", "continuous"),
                     new FirstDeriv<int64_t>());
      _ipext_size = size();

      // Field names TCPEXT
      register_field(new data_field<int>("SyncookiesSent", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("SyncookiesRecv", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("SyncookiesFailed", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("EmbryonicRsts", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("PruneCalled", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("RcvPruned", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("OfoPruned", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("OutOfWindowIcmps", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("LockDroppedIcmps", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("ArpFilter", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TW", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TWRecycled", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TWKilled", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("PAWSPassive", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("PAWSActive", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("PAWSEstab", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("DelayedACKs", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("DelayedACKLocked", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("DelayedACKLost", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("ListenOverflows", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("ListenDrops", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPPrequeued", "continuous"),
                     new FirstDeriv<int>());
      register_field(
          new data_field<int>("TCPDirectCopyFromBacklog", "continuous"),
          new FirstDeriv<int>());
      register_field(
          new data_field<int>("TCPDirectCopyFromPrequeue", "continuous"),
          new FirstDeriv<int>());
      register_field(new data_field<int>("TCPPrequeueDropped", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPHPHits", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPHPHitsToUser", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPPureAcks", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPHPAcks", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPRenoRecovery", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPSackRecovery", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPSACKReneging", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPFACKReorder", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPSACKReorder", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPRenoReorder", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPTSReorder", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPFullUndo", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPPartialUndo", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPDSACKUndo", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPLossUndo", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPLostRetransmit", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPRenoFailures", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPSackFailures", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPLossFailures", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPFastRetrans", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPForwardRetrans", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPSlowStartRetrans", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPTimeouts", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPRenoRecoveryFail", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPSackRecoveryFail", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPSchedulerFailed", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPRcvCollapsed", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPDSACKOldSent", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPDSACKOfoSent", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPDSACKRecv", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPDSACKOfoRecv", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPAbortOnData", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPAbortOnClose", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPAbortOnMemory", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPAbortOnTimeout", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPAbortOnLinger", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPAbortFailed", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPMemoryPressures", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPSACKDiscard", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPDSACKIgnoredOld", "continuous"),
                     new FirstDeriv<int>());
      register_field(
          new data_field<int>("TCPDSACKIgnoredNoUndo", "continuous"),
          new FirstDeriv<int>());
      register_field(new data_field<int>("TCPSpuriousRTOs", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPMD5NotFound", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPMD5Unexpected", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPSackShifted", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPSackMerged", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPSackShiftFallback", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPBacklogDrop", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPMinTTLDrop", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPDeferAcceptDrop", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("IPReversePathFilter", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPTimeWaitOverflow", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPReqQFullDoCookies", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPReqQFullDrop", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPRetransFail", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPRcvCoalesce", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPOFOQueue", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPOFODrop", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPOFOMerge", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPChallengeACK", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPSYNChallenge", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPFastOpenActive", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPFastOpenPassive", "continuous"),
                     new FirstDeriv<int>());
      register_field(
          new data_field<int>("TCPFastOpenPassiveFail", "continuous"),
          new FirstDeriv<int>());
      register_field(
          new data_field<int>("TCPFastOpenListenOverflow", "continuous"),
          new FirstDeriv<int>());
      register_field(
          new data_field<int>("TCPFastOpenCookieReqd", "continuous"),
          new FirstDeriv<int>());
      register_field(new data_field<int>("TCPLoss", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("TCPAbortOnSyn", "continuous"),
                     new FirstDeriv<int>());
      _tcpext_size = size() - _ipext_size;

      std::ifstream is_netstat("/proc/net/netstat");

      std::cout << "[+] netstat  lambda indices size : "
                << std::to_string(_field_indices.size()) << std::endl;
      // Parse first line
      std::string tcpext_names;
      std::getline(is_netstat, tcpext_names);
      // Close stream

      // Then , get TCPEXT values
      std::vector<std::string> tcp_names;
      tokenize(tcpext_names.substr(tcpext_names.find_first_of(":") + 1),
               tcp_names, " ");
      // Get names and set indices accordingly
      for (std::vector<std::string>::const_iterator it = tcp_names.begin();
           it != tcp_names.end(); ++it)
      {
        // Check if the parsed name is on the list
        std::map<std::string, size_t>::const_iterator itt =
            _field_indices.find(*it);
        if (itt != _field_indices.end())
        {
          // If so, get the index
          size_t idx = itt->second;
          // Modify the values
          _tcpext_indices.push_back(_ipext_size + idx);
        }
        else
        {
          std::cerr << "[-] Warning: Field name " << (*it) << " not found."
                    << std::endl;
          // Push incorrect index
          _tcpext_indices.push_back(size());
        }
      }
      is_netstat.close();
    }
    virtual void
    collect();
    virtual ~net_stats();

  protected:
    static std::map<std::string, size_t> _field_indices;

    size_t _ipext_size;

    size_t _tcpext_size;

    std::vector<size_t> _tcpext_indices;
  };

  class arp_stats : virtual public stat_collector
  {
  public:
    arp_stats() : stat_collector("arp_stats")
    {
      register_field(new data_field<std::string>("Interface", "0,1"),
                     new Changed<std::string>());
      register_field(
          new data_field<std::string>("CollisionDetected", "ignore"),
          new LastValue());
      register_field(new data_field<std::string>("GwChanged", "0,1"),
                     new Changed<std::string>());

      _stats.insert(std::make_pair("Interface", stat_value_cell(0)));
      _stats.insert(std::make_pair("CollisionDetected", stat_value_cell(0)));
      _stats.insert(std::make_pair("GwChanged", stat_value_cell(0)));
    }
    virtual void
    collect();
    virtual ~arp_stats();
    bool
    is_intercepted();
  };

  class arp_cache_stats : virtual public stat_collector
  {
  public:
    size_t line_collect;
    arp_cache_stats(const size_t entry = 0) : stat_collector("arp_cache_stats"), line_collect(entry)
    {
      register_field(
          new data_field<int>("arpcache_entries_" + std::to_string(entry),
                              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>("arpcache_allocs_" + std::to_string(entry),
                              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>("arpcache_destroys_" + std::to_string(entry),
                              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>("arpcache_hash_grows_" + std::to_string(entry),
                              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>("arpcache_lookups_" + std::to_string(entry),
                              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>("arpcache_hits_" + std::to_string(entry),
                              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>("arpcache_res_failed_" + std::to_string(entry),
                              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>(
              "arpcache_rcv_probes_mcast_" + std::to_string(entry),
              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>(
              "arpcache_rcv_probes_ucast_" + std::to_string(entry),
              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>(
              "arpcache_periodic_gc_runs_" + std::to_string(entry),
              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>(
              "arpcache_forced_gc_runs_" + std::to_string(entry),
              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>(
              "arpcache_unresolved_discards_" + std::to_string(entry),
              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>(
              "arpcache_table_fulls_" + std::to_string(entry),
              "continuous"),
          new Less<int>());
    }
    virtual void
    collect();
    ~arp_cache_stats();
  };

  //
  // Example:
  //----------------------------------------------------------------------
  // Idx	Device    : Count Querier	Group    Users Timer	Reporter
  // 1	lo        :     2      V3\n
  //									FB0000E0     1 0:00000000		0
  //									010000E0     1 0:00000000		0
  // 2	eno2      :     1      V3
  //				010000E0     1 0:00000000		0
  //-----------------------------------------------------------------------

  class igmp_stats : virtual public stat_collector
  {
  public:
    igmp_stats() : stat_collector("igmp_stats")
    {
    }
    virtual void
    collect();
  };

  // ifindex ,dev->name,  NIP6(im->mca_addr)-> ip6 address bytes encoded ,users,flags
  //+		   im->mca_users, im->mca_flags,  if flags has TIMER_RUNNING - jiffies.

  class igmp6_stats : virtual public stat_collector
  {
  public:
    igmp6_stats() : stat_collector("igmp6_stats")
    {
    }
    virtual void
    collect();
  };
  class ifnet6_stats : virtual public stat_collector
  {
    ifnet6_stats() : stat_collector("ifnet6_stats")
    {
    }
    virtual void
    collect();
  };
  /// proc/net/stat/rt_cache
  class route_statistics : virtual public stat_collector
  {
  public:
    explicit route_statistics(size_t entry) : stat_collector("route_statistics"), _entry(entry)
    {
      const std::string suffix = std::to_string(entry);
      register_field(
          new data_field<int>("rtcache_entries_" + suffix,
                              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>("rtcache_in_hit_" + suffix,
                              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>("rtcache_in_slow_tot_" + suffix,
                              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>("rtcache_in_slow_mc_" + suffix,
                              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>("rtcache_in_no_route_" + suffix,
                              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>("rtcache_in_brd_" + suffix,
                              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>(
              "rtcache_in_martian_dst_" + suffix, "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>(
              "rtcache_in_martian_src_" + suffix, "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>("rtcache_out_hit_" + suffix,
                              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>("rtcache_out_slow_tot_" + suffix,
                              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>("rtcache_out_slow_mc_" + suffix,
                              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>("rtcache_gc_total_" + suffix,
                              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>("rtcache_gc_ignored_" + suffix,
                              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>("rtcache_gc_goal_miss_" + suffix,
                              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>(
              "rtcache_gc_dst_overflow_" + suffix,
              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>(
              "rtcache_in_hlist_search_" + suffix,
              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>(
              "rtcache_out_hlist_search_" + suffix,
              "continuous"),
          new Less<int>());
    }
    virtual void
    collect();

  protected:
    size_t _entry;
  };

  class netdev_stats : virtual public stat_collector
  {
  public:
    netdev_stats() : stat_collector("netdev_statistics")
    {
      register_field(new data_field<int64_t>("tx_bytes", "continuous"),
                     new FirstDeriv<int64_t>());
      register_field(new data_field<int64_t>("tx_packets", "continuous"),
                     new FirstDeriv<int64_t>());
      register_field(new data_field<int>("tx_errs", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("tx_drop", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("tx_fifo", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("tx_frame", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("tx_compressed", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("tx_multicast", "continuous"),
                     new FirstDeriv<int>());

      // Receive
      register_field(new data_field<int64_t>("rx_bytes", "continuous"),
                     new FirstDeriv<int64_t>());
      register_field(new data_field<int64_t>("rx_packets", "continuous"),
                     new FirstDeriv<int64_t>());
      register_field(new data_field<int>("rx_errs", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("rx_drop", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("rx_fifo", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("rx_colls", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("rx_carrier", "continuous"),
                     new FirstDeriv<int>());
      register_field(new data_field<int>("rx_compressed", "continuous"),
                     new FirstDeriv<int>());
    }
    virtual void
    collect();
  };

  class ndis_cache : virtual public stat_collector
  {
    size_t _entry;

  public:
    ndis_cache(const size_t entry = 0) : stat_collector("ndis_cache"), _entry(entry)
    {
      const std::string suffix = std::to_string(entry);
      register_field(
          new data_field<int>("ndisc_entries_" + suffix,
                              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>("ndisc_allocs_" + suffix,
                              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>("ndisc_destroys_" + suffix,
                              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>("ndisc_hash_grows_" + suffix,
                              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>("ndisc_lookups_" + suffix,
                              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>("ndisc_hits_" + suffix,
                              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>("ndisc_res_failed_" + suffix,
                              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>(
              "ndisc_rcv_probes_mcast_" + suffix,
              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>(
              "ndisc_rcv_probes_ucast_" + suffix,
              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>(
              "ndisc_periodic_gc_runs_" + suffix,
              "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>(
              "ndisc_forced_gc_runs_" + suffix, "continuous"),
          new Less<int>());
      register_field(
          new data_field<int>(
              "ndisc_unresolved_discards_" + suffix,
              "continuous"),
          new Less<int>());
    }

    virtual void
    collect();
    virtual ~ndis_cache();
  };

  /// proc/diskstats
  class diskstats : virtual public stat_collector
  {
    std::map<size_t, size_t> type_exception_indice; // map indice to type :
  public:
    diskstats() : stat_collector("diskstats")
    {
    }
    virtual void
    collect();
  };

  template <class container>
  void
  stat_collector::tokenize(const std::string &str, container &tokens,
                           const std::string &delim /*=","*/,
                           const std::string &prefix)
  {

    std::string::size_type lp = str.find_first_not_of(delim, 0);
    std::string::size_type pos = str.find_first_of(delim, lp);
    while (std::string::npos != pos && std::string::npos != lp)
    {

      auto cut = str.substr(lp, pos - lp);
      // remove whitespaces from keys/values :
      // cut = std::regex_replace(cut, std::regex("(^[ ]+)|([ ]+$)"),"");
      if (cut.length() > 0)
      {

        if (std::isspace(cut[0]))
        {
          cut.erase(cut.begin());
        }

        if (prefix.length())
          tokens.push_back(prefix + cut);
        else
          tokens.push_back(cut);
      }
      lp = str.find_first_not_of(delim, pos);
      pos = str.find_first_of(delim, lp);
    }
  }

  template <class Iterator>
  void
  proc_collector::setIgnored(Iterator begin, Iterator end)
  {
    // Save field indices on each table
    std::vector<std::vector<size_t>> indices(_tables.size());
    while (begin < end)
    {
      // Name of the field
      std::string field_name((*begin++));
      // Loop over all the tables
      for (size_t i = 0; i < _tables.size(); ++i)
      {
        std::vector<field_base *>::const_iterator it =
            _tables[i]->getField(field_name);
        if (it != _tables[i]->end())
          // data_field is on this table
          indices[i].push_back(it - _tables[i]->begin());
      }
    }

    // Check for completely ignored tables
    for (size_t i = 0; i < indices.size(); ++i)
    {
      if (indices[i].size() == _tables[i]->size())
      {
        // Replace table by a dummy one
        delete _tables[i];
        _tables[i] = new empty_collector(indices[i].size());
      }
      else
      {
        // Set as ignored
        for (size_t j = 0; j < indices[i].size(); ++j)
        {
          _tables[i]->set_active(indices[i][j], false);
        }
      }
    }
  }

  class wireless : virtual public stat_collector
  {
  public:
    wireless() : stat_collector("wireless")
    {

      register_field(new data_field<int>("tus", "ignore"), new Less<int>());
      register_field(new data_field<int>("link", "ignore"), new Less<int>());
      register_field(new data_field<int>("level", "ignore"), new Less<int>());
      register_field(new data_field<int>("noise", "ignore"), new Less<int>());
      register_field(new data_field<int>("nwid", "ignore"), new Less<int>());
      register_field(new data_field<int>("crypt", "ignore"), new Less<int>());
      register_field(new data_field<int>("frag", "ignore"), new Less<int>());
      register_field(new data_field<int>("retry", "ignore"), new Less<int>());
      register_field(new data_field<int>("misc", "ignore"), new Less<int>());
      register_field(new data_field<int>("beacon", "ignore"), new Less<int>());

      _stats.insert(std::make_pair("tus", stat_value_cell(0)));
      _stats.insert(std::make_pair("link", stat_value_cell(0)));
      _stats.insert(std::make_pair("level", stat_value_cell(0)));
      _stats.insert(std::make_pair("noise", stat_value_cell(0)));
      _stats.insert(std::make_pair("nwid", stat_value_cell(0)));
      _stats.insert(std::make_pair("crypt", stat_value_cell(0)));
      _stats.insert(std::make_pair("frag", stat_value_cell(0)));
      _stats.insert(std::make_pair("retry", stat_value_cell(0)));
      _stats.insert(std::make_pair("misc", stat_value_cell(0)));
      _stats.insert(std::make_pair("beacon", stat_value_cell(0)));
    }
    virtual void
    collect();
  };
  template <class InputIterator>
  void
  stat_collector::putValuesFromStrings(InputIterator begin,
                                       InputIterator end)
  {
    // Sanity check
    const InputIterator end_minus_one = end - 1;

    size_t value_size = size_t(end - begin);
    size_t delta_size = value_size / size_t(end - end_minus_one);
    if (value_size < size())
    {
      size_t count = 0;
      for (field_base *field : *_instances[0])
      {
        if (count > value_size)
        {
          std::cerr << "[-] Collector " << this->_name << " failed to find field :" << field->get_name() << "," << field->getValue() << ", (" << field->get_type() << ")" << std::endl;
        }
        count++;
      }
    }
    if (_instances[0]->size() != delta_size)
    {
      std::cerr << this->_name
                << std::string(" [-] instance size  expected :")
                << std::to_string(_instances[0]->size())
                << std::string(" , got ")
                << std::to_string(size_t(end - begin)) << std::endl;
      std::cerr << this->_name
                << std::string(" [-] internal size() return : ")
                << std::to_string(size()) << std::endl;

      print_map();

      if (_instances[0]->size() == delta_size)
      {
        std::cout << "[+] iterator delta bug ! " << std::endl;
      }

      for (auto it = begin; it != end; ++it)
        std::cout << *it << (it != end_minus_one ? std::string(",") : "");

      if (value_size > _instances[0]->size())
      {
        // split
        putValuesFromStrings<InputIterator>(begin, begin + size());
        putValuesFromStrings<InputIterator>(begin + size(), end);
        return;
      }
      else
      {
        std::cerr << this->_name << "[-] missing values : " << std::endl;
        return;
      }
    }

    fields_instance *ptr(getNewDataPtr());

    // Field counter
    size_t cnt(0);
    while (cnt < value_size)
    {
      if ((*ptr).size() < this->size())
        ptr->resize(this->size());

      (*ptr)[cnt]->setValue(*begin);
      (*ptr)[cnt]->setTimeStamp(getTimeStamp());
      ++begin;
      ++cnt;
    }
  }

  template <class InputIterator>
  void
  stat_collector::putValues(InputIterator begin, InputIterator end)
  {
    typedef typename std::iterator_traits<InputIterator>::value_type ValueType;
    // Sanity check
    assert(_instances[0]->size() == size_t( std::distance(begin, end)) ); //(end - begin));
    fields_instance *ptr(getNewDataPtr());

    size_t cnt(0);
    while (begin != end)
    {
      static_cast<data_field<ValueType> *>((*ptr)[cnt])->setdata_field(*begin);
      (*ptr)[cnt]->setTimeStamp(getTimeStamp());
      ++begin;
      ++cnt;
    }
  }

  template <class InputIterator>
  void
  stat_collector::putValuesAndTimestamps(InputIterator begin,
                                         InputIterator end)
  {
    // Sanity check
    InputIterator end_minus_one = end--;

    size_t value_size = size_t(end - begin);
    size_t delta_size = value_size / size_t(abs(end - end_minus_one));
    assert(_instances[0]->size() == (end - begin));
    fields_instance *ptr(getNewDataPtr());
    size_t cnt(0);
    while (begin != end)
    {
      (*ptr)[cnt]->setValue((*begin).first);
      (*ptr)[cnt]->setTimeStamp((*begin).second);
      ++begin;
      ++cnt;
    }
  }

} /* namespace provallo */

#endif /* STATISTICS_STATCOLLECTOR_H_ */
