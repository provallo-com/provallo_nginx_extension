/*
 * attribute.cpp
 *
 *  Created on: May 12, 2021
 *      Author: kardon
 */

#include "../glue/glueprocessinfo.h"
#include "attribute.h"
#include <streambuf>
#include <strstream>
#include <fstream>
using namespace std;
namespace provallo
{

  void
  print_error(const std::string &error_message)
  {
    std::cerr << error_message << std::endl;
  }

  void
  attribute_definition::serialize(attribute_definition *serial) const
  {

    if (serial == NULL)
      return;
    else
    {
      serial->_index = this->_index;
      serial->_name = this->_name;
      serial->_tag = this->_tag;
    }
  }

  attribute_definition::attribute_definition(
      const attribute_definition *deserial)
  {
    // Set internal data
    //_name = deserial->name();
    //_tag = deserial->tag();
    if (deserial == NULL)
      return;
    else
    {
      this->_index = deserial->_index;
      this->_name = deserial->_name;
      this->_tag = deserial->_tag;
    }
  }

  discrete_attribute::discrete_attribute(
      const attribute_name &name, const attribute_tag &tag,
      const vector<attribute_value> &attribute_values) : attribute_definition(name, tag), _values_map(attribute_values)
  {
    vector<attribute_value>::const_iterator att = attribute_values.begin();
    discrete_value value(0);

    // Iterate over the set and create the map
    for (; att != attribute_values.end(); ++att)
    {
      // Insert data on the map
      _name_map.insert(
          pair<attribute_value, discrete_value>((*att), value));
      ++value;
    }
  }

  discrete_attribute::discrete_attribute(const attribute_definition *deserial) : attribute_definition(deserial)
  {
    // Check size
    // const discrete_attribute& discrete_buffer = deserial->GetExtension(serialize9::discrete_attribute::child);
    // assert(discrete_buffer.names_size() == discrete_buffer.values_size());
    // int size(discrete_buffer.values_size());
    // Push values
    // for(int i = 0 ; i < size ; ++i) {
    //        _name_map.insert(make_pair(discrete_buffer.names(i), discrete_buffer.values(i)));
    //       _values_map.push_back(discrete_buffer.names(i));
    //  }
  }

  bool
  discrete_attribute::_compare(const attribute_definition &other) const
  {
    const discrete_attribute dother =
        *static_cast<const discrete_attribute *>(&other);
    return dother._values_map == _values_map && dother._name_map == _name_map;
  }

  void
  discrete_attribute::serialize(attribute_definition *serial) const
  {

    if (serial == NULL)
      return;
    else
    {

      if (serial == this)
        return;
    }

    // discrete_attribute* ensemble_buffer(serial->MutableExtension(discrete_attribute::child));
    // for(map<attribute_value, DiscValue>::const_iterator it = _name_map.begin() ; it != _name_map.end() ; ++it) {
    //     ensemble_buffer->add_names((*it).first);
    // ensemble_buffer->add_values((*it).second);
    // }
  }

  void
  discrete_attribute::print(ostream &out) const
  {
    // Print discrete values
    map<attribute_value, discrete_value>::const_iterator it =
        _name_map.begin();
    size_t cnt(0);
    for (; it != _name_map.end(); ++it)
    {
      if (cnt == _name_map.size() - 1)
        out << (*it).first;
      else
        out << (*it).first << ",";
      ++cnt;
    }
  }

  void
  continous_attribute::print(ostream &out) const
  {
    // Continuous data
    out << "continuous";
  }

  void
  ignored_attribute::print(std::ostream &out) const
  {
    // Ignore data
    out << "ignore";
  }

  std::ostream &
  operator<<(std::ostream &out, const attribute_definition &q)
  {
    // Call virtual function
    q.print(out);
    return out;
  }

  attribute_groups::attribute_groups(const attribute_groups *deserial)
  {
    // Sanity check
    /*  assert(deserial->groups_size() == deserial->type_size());
     // Resize groups
     _groups.resize(deserial->groups_size());
     for(int i = 0 ; i < deserial->groups_size() ; ++i) {
     const tag_group& tag_buffer = deserial->groups(i);
     for(int j = 0 ; j < tag_buffer.tags_size() ; ++j)
     _groups[i].push_back(tag_buffer.tags(j));
     }
     // Set split types
     for(int i = 0 ; i < deserial->split_type_size() ; ++i)
     _split_type.push_back(deserial->split_type(i));
     for(int i = 0 ; i < deserial->type_size() ; ++i)
     _type.push_back(deserial->type(i));
     */
    if (!deserial || !deserial->size())
      return;
  }

  void
  attribute_groups::serialize(attribute_groups *serial) const
  {
    if (!serial)
      return;

    // Set groups of attributes
    /*
     for(uint32_t i = 0 ; i < _groups.size() ; ++i) {
     const vector<attribute_tag>& tags = _groups[i];
     tag_group* serial_tags = serial->add_groups();
     // Add each tag
     for(uint32_t j = 0 ; j < tags.size() ; ++j)
     serial_tags->add_tags(tags[j]);
     }
     // Add types
     for(uint32_t i = 0 ; i < _split_type.size() ; ++i)
     serial->add_split_type(_split_type[i]);
     for(uint32_t i = 0 ; i < _type.size() ; ++i)
     serial->add_type(_type[i]);
     */
  }

  // Get new attribute definition
  attribute_definition *
  attribute_definition::getDefinition(const std::string &name,
                                      const attribute_tag &tag,
                                      const std::string &str)
  {
    // Discrete
    if (str.find(",") != string::npos)
    {
      // Values are defined in a CSV format
      vector<attribute_value> values;
      tokenize(str, values);
      // Create discrete definition
      return new discrete_attribute(name, tag, values);
    }
    // Continuous
    else if (str.find("continuous") != string::npos)
    {
      return new continous_attribute(name, tag);
    }
    else if (str.find("ignore") != string::npos)
    {
      return new ignored_attribute(name, tag);
    }
    // Not defined
    return nullptr;
  }

  attribute_definition *
  attribute_definition::getDefinition(const attribute_definition *deserial)
  {
    attribute_type type(deserial->get_type());
    if (type == discrete_attribute::_type())
      return new discrete_attribute(deserial);
    else if (type == continous_attribute::_type())
      return new continous_attribute(deserial);
    else if (type == ignored_attribute::_type())
      return new ignored_attribute(*deserial);
    return 0;

    return nullptr;
  }
//    // Map of tag to names (vector index is the tag)
//  std::vector<attribute_name> _tag_map;
    // Map of names to tags
  //std::map<attribute_name, attribute_tag> _name_map;
    // Target attribute position
   // attribute_tag _target_pos;
    // Map of definitions of each attribute (vector index is the tag)
   // std::vector<attribute_definition *> _definition_map;
    // Number of values of each attribute
   //. std::vector<uint32_t> _count;
    // Type of each attribute (continuous or discrete)
    //td::vector<attribute_type> _type;
    // Group of attributes, the target attribute does not appear here
//    attribute_groups _groups;

  attribute_information::attribute_information(
      const attribute_name &target_name,
      const vector<pair<attribute_name, std::string>> &attributes) :_tag_map(attributes.size()), _name_map(),_target_pos(0),_definition_map(attributes.size()) ,_count(attributes.size()), _type( attributes.size()), _groups()
  {
    // Create attribute definitions
    vector<pair<attribute_name, std::string>>::const_iterator att =
        attributes.begin();

    // Attribute tag
    attribute_tag tag(0);

    // Resize containers
    _tag_map.resize(attributes.size());
    _definition_map.resize(attributes.size(),nullptr);

    // Definition of the target attribute
    attribute_name target_str("");

    // Check the type of the attribute
    for (; att != attributes.end(); ++att)
    {
      // Name of this attribute
      attribute_name att_name = (*att).first;

      // Check if this is the target attribute
      if (att_name.compare(target_name) == 0)
      {
        // Save definition  
        target_str = (*att).second;
        // Set target attribute position
        _target_pos = tag;
      }

      // Create definition object

      _definition_map[tag] = attribute_definition::getDefinition(
          att_name, tag, (*att).second);
      // Get count
      _count[tag] = _definition_map[tag]->getCount();
      // Get type
      _type[tag] = _definition_map[tag]->get_type();
      // Set tag map
      _tag_map[tag] = att_name;
      // Set name map
      _name_map.insert(
          std::pair<attribute_name, attribute_tag>(att_name, tag));

      // Increment tag
      ++tag;
    }

    //  Sanity check
    // Check if the target attribute exists


    if (target_str.size() == 0)
      {
        print_error("Target attribute [" + target_name + "] does not exist");
      }


    // Check if the target attribute is not continuous
    if (_definition_map[_target_pos]->get_type() == continous_attribute::_type())
    {print_error(
          "Target attribute " + target_name + " can't be a continuous value");
    }
    // By default there aren't groups and continuous attributes have the random split method
    for (uint32_t i = 0; i < getSize(); ++i)
    {
      // Do not group the target attribute or ignored attributes
      if (i != _target_pos && getType(i) != ignored_attribute::_type())
      {
        // Push group (only one attribute)
        if (getType(i) == continous_attribute::_type())
          _groups.push(std::vector<attribute_tag>(1, i), CONE_RANDOM,
                       CONTINUOUS);
        else if (getType(i) == discrete_attribute::_type())
          _groups.push(std::vector<attribute_tag>(1, i), DISC, DISCRETE); 
    
       }
    }
  }

  attribute_information::attribute_information(
      const attribute_information &right) : _tag_map(right._tag_map), _name_map(right._name_map), _target_pos(right._target_pos), _definition_map(right._definition_map.size()), _count(right._count), _type(right._type), _groups(right._groups)
  {

    // Clone definitions
    for (uint32_t i = 0; i < _definition_map.size(); ++i)
      _definition_map[i] = right._definition_map[i]->clone();
  }

  attribute_information::attribute_information(
      const attribute_information *deserial) : _tag_map(deserial->_tag_map), _name_map(deserial->_name_map), _target_pos(deserial->_target_pos), _definition_map(deserial->_definition_map.size()), _count(deserial->_count), _type(deserial->_type), _groups(deserial->_groups)

  {
  }

  std::string
  attribute_information::getDefinition(const attribute_tag &tag) const
  {
    std::stringstream ss;
    std::string ret;
    assert(tag < _definition_map.size());
    ss << *(_definition_map[tag]);
    ret = ss.str();
    return ret;
  }

  vector<pair<attribute_name, std::string>>
  attribute_information::getDefinitionMap() const
  {
    vector<pair<attribute_name, std::string>> def_map;
    for (uint32_t i = 0; i < _definition_map.size(); ++i)
      def_map.push_back(make_pair(_tag_map[i], getDefinition(i)));
    return def_map;
  }

  std::ostream &
  operator<<(std::ostream &out, const attribute_information &q)
  {
    for (uint32_t i = 0; i < q._definition_map.size(); ++i)
      out << q._tag_map[i] << " : " << *(q._definition_map[i]) << endl;
    return out;
  }

  void
  attribute_information::serialize(attribute_information *serial) const
  {
    if (!serial || serial == this)
      return;
      // Put information on buffer
#if 0
  for(uint32_t i = 0 ; i < _tag_map.size() ; ++i) {
        serial->add_names(_tag_map[i]);
        serial->add_count(_count[i]);
        serial->add_types(_type[i]);
        attribute_definition* definition_buffer = serial->add_definitions();
        _definition_map[i]->serialize(definition_buffer);
    }
    serial->set_target(_target_pos);
    // Serialize groups
    attribute_groups* groups_buffer = serial->mutable_groups();
    _groups.serialize(groups_buffer);

#endif
  }

  attribute_information::~attribute_information()
  {
    // Delete definitions

    for (vector<attribute_definition *>::const_iterator it =
             _definition_map.begin();
         it != _definition_map.end(); ++it){
          attribute_definition* def =  *it;
          if(def!=nullptr )
            delete def;
      }

    _definition_map.clear();
    _groups.clear();
  }
  attribute_information &attribute_information::operator=(const attribute_information &right)
  {
    if (this != &right)
    {
      // Delete definitions
      for (vector<attribute_definition *>::const_iterator it =
               _definition_map.begin();
           it != _definition_map.end(); ++it)
        delete (*it);

      // Clone definitions
      if(_definition_map.size()>0) 
        for(auto it = _definition_map.begin();it!=_definition_map.end();++it)
            delete (*it);
      _definition_map.clear();

      _definition_map.resize(right._definition_map.size(),nullptr);
      for (uint32_t i = 0; i < _definition_map.size(); ++i)
        _definition_map[i] = right._definition_map[i]->clone();

      _tag_map = right._tag_map;
      _name_map = right._name_map;
      _target_pos = right._target_pos;
      _count = right._count;
      _type = right._type;
      _groups = right._groups;
    }
    return *this;
  }

  std::atomic_uint64_t attribute::_id_counter(0);

} /* namespace provallo */
