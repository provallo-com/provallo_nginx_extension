/*
 * dataset.cpp
 *
 *  Created on: May 11, 2021
 *      Author: kardon
 */
#include "../glue/glueprocessinfo.h"
#include "dataset.h"
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>
#include <algorithm>
#include "attribute.h"
#include "classdist.h"
#include "split_utils.hpp"
namespace provallo
{

  dataset_base::dataset_base(size_t rows, size_t cols, size_t numLabels) : matrix_base(rows, cols)
  {
    _labels = new int[rows];
    this->_num_of_labels = numLabels;

    // std::cout<<"dataset: allocated int: "<< std::to_string(rows) <<" at "<< hex << ptrdiff_t(labels ) << std::endl;
  }

  dataset_base::~dataset_base()
  {
    if (_labels)
      delete[] _labels;
  }

  void
  dataset_base::splitdataset(dataset_ptr &train, dataset_ptr &valid,
                             double train_percent)
  {
    
    std::vector<int> randomIndices;
    randomIndices.resize(rows);


    for (size_t i = 0; i < rows; i++)
      randomIndices[i] = i;

    std::random_shuffle(randomIndices.begin(), randomIndices.end());

    size_t threshold = (size_t)(train_percent * rows);

    train = dataset_ptr(new dataset_base(threshold, cols, _num_of_labels));
    valid = dataset_ptr(
        new dataset_base(rows - threshold, cols, _num_of_labels));

    size_t row_train = 0;
    size_t row_validate = 0;
    for (size_t i = 0; i < rows; i++)
    {
      if (i < threshold)
      {
        for (size_t j = 0; j < cols; j++)
          train->pos(row_train, j) = pos(randomIndices[i], j);
        train->label(row_train) = label(randomIndices[i]);
        row_train++;
      }
      else
      {
        for (size_t j = 0; j < cols; j++)
          valid->pos(row_validate, j) = pos(randomIndices[i], j);
        valid->label(row_validate) = label(randomIndices[i]);
        row_validate++;
      }
    }
  }

  std::ostream &
  operator<<(std::ostream &out, const dataset &q)
  {
    q.print(out);
    return out;
  }

  void
  dataset::print(std::ostream &out) const
  {
    // Print each sample in a CSV format
    for (uint32_t i = 0; i < size(); ++i)
    {
      // Loop over attributes
      for (uint32_t j = 0; j < getattributesNumber(); ++j)
      {
        // Get attribute
        attribute att = getattribute(i, j);
        // Print class
        printattribute(out, j, att, getattributes());
        if (j < getattributesNumber() - 1)
          out << ",";
      }

      out << std::endl;
    }
  }

  void
  dataset::setup()
  {
    auto c_start = clock();
    // First initialize sorted indices
    sortattributes();

    auto c_end = clock();
    std::cout << "[+]sorting dataset[ " << std::to_string(_id) << "] attributes CPU time elapsed in s: "
              << (double)(c_end - c_start) / CLOCKS_PER_SEC << std::endl;

    // Now setup class distribution
    c_start = c_end;
    uint32_t target_tag(_attributes_info.get_target_tag());

    _distribution.setup(getattributesNumber());
    for (uint32_t i = 1; i < getattributesNumber(); ++i)
    {
      // Get attribute
      attribute att = getattribute(0, i);
      // Get class
      attribute class_att = getattribute(0, target_tag);
      // Get class value
      attribute_value class_value = std::to_string(class_att.discrete());
      // Get attribute value
      attribute_value value = std::to_string(att.discrete());
      // Get class index
      uint32_t class_index = _attributes_info.getValueIndex(target_tag, class_value);
      // Get attribute index
      uint32_t index = _attributes_info.getValueIndex(i, value);
      // Update distribution
      _distribution.update(i, index, class_index);
    }
    c_end = clock();
    std::cout << "[+]setup attribute distribution CPU time elapsed in s: " << (double)(c_end - c_start) / CLOCKS_PER_SEC << std::endl;
  }

  void
  dataset::printattribute(std::ostream &out, const attribute_tag &tag,
                          const attribute &att,
                          const attribute_information &info) const
  {
    out << info.getValue(tag, att);
  }

  // Compare to attributes
  struct attribute_comparator
  {
    typedef std::pair<const attribute *, uint32_t> AttrPair;
    attribute_type _type;
    attribute_comparator(const attribute_type &type) : _type(type)
    {
    }
    inline bool
    operator()(const AttrPair &i, const AttrPair &j)
    {

      return (_type == continous_attribute::_type()) ? bool(i.first->continous() < j.first->continous()) : bool(i.first->discrete() < j.first->discrete());
    }
    ~attribute_comparator()
    {
    }
  };

  void
  dataset::sortattribute(const attribute_tag &tag)
  {
    // make sure tag is in range
    if (_sorted_indices.size() < tag)
      _sorted_indices.resize(tag + 1);
    // make sure sorted indices array is allocated
    if (_sorted_indices[tag].size() != size())
      _sorted_indices[tag].resize(size());

    std::vector<std::pair<const attribute *, uint32_t>> pairs(size());
    // Create pairs

    for (uint32_t i = 0; i < size(); ++i)
    {
      bool bfound = false;
      const attribute *attr_ptr = getattributeptr(i, tag, &bfound);
      pairs[i] = std::pair<const attribute *, uint32_t>(attr_ptr, i);
    }
    // Sort pairs
    std::sort(pairs.begin(), pairs.end(),
              attribute_comparator(_attributes_info.getType(tag)));
    // Update indices array
    for (uint32_t i = 0; i < pairs.size(); ++i)
    {
      _sorted_indices[tag][i] = pairs[i].second;
    }
    // Clear pairs
    pairs.clear();
  }

  void
  dataset::sortattributes()
  {
    auto c_start = clock();
    if (last_sort == clock_t(0))
    {
      last_sort = c_start;
    }

    static std::map<dataset *, size_t> sort_count;

    // Sort each attribute
    for (uint32_t j = 0; j < getattributesNumber(); ++j)
      if (!isSorted(j) || _dirty)
        sortattribute(j);

    auto c_end = clock();

    // guard sort count
    {

      std::lock_guard<std::recursive_mutex> guard(_mutex);
      _dirty = false;

      if (sort_count.find(this) != sort_count.end())
      {
        sort_count[this]++;
      }
      else
      {
        sort_count.insert(std::make_pair(this, 1));
      }
      std::cout << "[+] dataset  " << std::hex << ptrdiff_t(this) << std::dec << " ( " << std::to_string(_id) << " ) sorted " << getattributesNumber() << " attributes ( " << sort_count[this] << " ) times " << std::endl;

      std::cout << "[+]dataset::sortattributes  CPU time elapsed in s: "
                << (double)(c_end - c_start) / CLOCKS_PER_SEC << "started after last sort :"
                << double(last_sort == clock_t(0) ? double(0.) : (double)(c_end - last_sort) / CLOCKS_PER_SEC) << std::endl;
      c_end = clock();
    }

    last_sort = c_end;
  }

  void
  dataset::permute(std::random_device &rd, attribute_tag tag,
                   std::vector<attribute> *prev_values)
  {
    // First save current values
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> uniform(0, size());

    for (uint32_t i = 0; i < size(); i++)
      prev_values->push_back(getattribute(i, tag));
    // Permute values
    for (uint32_t i = 0; i < size(); i++)
    {
      // Randomly select an index
      uint32_t idx = uniform(gen);

      // Swap attributes values
      attribute tmp = getattribute(i, tag);
      setattribute(i, tag, getattribute(idx, tag));
      setattribute(idx, tag, tmp);
    }
  }

  void
  dataset::restore(attribute_tag tag,
                   const std::vector<attribute> &prev_values)
  {
    // Sanity check
    std::cout << std::string("[+] restoring [") << prev_values.size() << std::string("] values to dataset of [") << size() << std::string("]") << std::endl;
    if (prev_values.size() < size())
    {
      for (uint32_t i = 0; i < prev_values.size(); ++i)
        setattribute(i, tag, prev_values[i]);
    }

    else
    {
      for (uint32_t i = 0; i < size(); ++i)
        setattribute(i, tag, prev_values[i]);
      // dont resize.
    }

    // Restore attributes
  }

  dataset *
  dataset::randomSubset(std::random_device &rd,
                        const class_dist &distribution) const
  {
    // Sanity check
    assert(distribution.size() == size());

    // Create cumulative probabilities
    std::vector<float> cumulative(distribution.cumulative());

    // Get target tag
    uint32_t target_tag(_attributes_info.get_target_tag());

    // Create new data set
    dataset *new_set(getNew());

    // Create a random set with equal size than the original
    for (uint32_t sample = 0; sample < size(); ++sample)
    {

      // Sample random number
      std::mt19937 gen(rd());
      std::uniform_real_distribution<> uniform(0.0, 1.0);

      Float rho(uniform(gen));

      // Check numerics
      if (rho < 0.0)
        rho = 0.0;
      if (rho > 1.0)
        rho = 1.0;

      // Get sampled index
      uint32_t idx = std::lower_bound(cumulative.begin(), cumulative.end(),
                                      rho) -
                     cumulative.begin();

      // Push this sample into the new set
      for (uint32_t j = 0; j < getattributesNumber(); ++j)
      {
        attribute value(getattribute(idx, j));
        // Push back attribute
        new_set->pushattribute(j, value);
        // Sum contribution into the distribution
        if (j == target_tag)
          new_set->_distribution.accum(value.discrete());
      }
    }

    // Setup the new set
    new_set->setup();

    return new_set;
  }

  std::pair<dataset *, dataset *>
  dataset::randomSubsetOob(std::random_device &rd,
                           const class_dist &distribution,
                           std::vector<uint32_t> *oob_indices) const
  {
    // Sanity check
    assert(distribution.size() == size());

    // Create cumulative probabilities
    std::vector<Float> cumulative(distribution.cumulative());

    // Get target tag
    uint32_t target_tag(_attributes_info.get_target_tag());

    // Create new data set and OOB set
    dataset *new_set = getNew();
    dataset *oob_set = getNew();
    // Number of attributes
    uint32_t nattrs(getattributesNumber());

    // Container of flags (true if it was selected, and false otherwise)
    std::vector<bool> oob_samples(size(), false);

    // Create a random set with equal size than the original
    for (uint32_t sample = 0; sample < size(); ++sample)
    {

      // Sample random number
      // Sample random number
      std::mt19937 gen(rd());
      std::uniform_real_distribution<> uniform(0.0, 1.0);

      Float rho(uniform(gen));

      // Check numerics
      if (rho < 0.0)
        rho = 0.0;
      if (rho > 1.0)
        rho = 1.0;

      // Get sampled index
      uint32_t idx = std::lower_bound(cumulative.begin(), cumulative.end(),
                                      rho) -
                     cumulative.begin();

      // Toggle flag of OOB samples
      oob_samples[idx] = true;

      // Push this sample into the new set
      for (uint32_t j = 0; j < nattrs; ++j)
      {
          attribute value(getattribute(idx, j));

        
          if (j == target_tag) {
          //make sure it's discrete 
            if ( value.discrete() > _distribution.size() ){
                if(discrete_value(value.continous() < _distribution.size()))
                    value = attribute(discrete_value(value.continous()));
                else
                    value = attribute(discrete_value(0)); 
	    }

           

          }
            

        
          // Push back attribute
          new_set->pushattribute(j, value);
        // Sum contribution into the distribution

         if (j == target_tag)
          new_set->_distribution.accum(value.discrete());
        
      }//end for
    }//end for


    // Setup the new set
    new_set->setup();

    // Loop over the OOB flags
    for (uint32_t i = 0; i < oob_samples.size(); ++i)
    {
      if (not oob_samples[i])
      {
        for (uint32_t j = 0; j < nattrs; ++j)
          // Push OOB attribute
          oob_set->pushattribute(j, getattribute(i, j));
        // Set index
        oob_indices->push_back(i);
      }
    }

    // Setup the out of bag set
    oob_set->setup();

    // Return pair of sets
    return std::make_pair(new_set, oob_set);
  }

  dataset *
  dataset::subset(const split_method &_method, uint32_t nbranch) const
  {
    // Get  the target tag
    uint32_t target_tag(_attributes_info.get_target_tag());
    // Crate new data set
    dataset *new_set = getNew();
    // Loop over each sample
    for (uint32_t i = 0; i < size(); ++i)
    {
      // Check the value of the attribute
      if (_method.isInBranch(this->begin(i), nbranch))
      {
        // Push this sample into the new set
        for (uint32_t j = 0; j < getattributesNumber(); ++j)
        {
          attribute value(getattribute(i, j));
          // Push back attribute
          new_set->pushattribute(j, value);
          // Sum contribution  into the distribution
          if (j == target_tag)
            new_set->_distribution.accum(value.discrete());
        }
      }
    }
    // Setup the new set
    new_set->setup();
    // Return new data set
    return new_set;
  }

  const dataset *
  dataset::subsetReference(const split_method &s,
                           uint32_t nbranch) const
  {
    // Get target tag
    uint32_t target_tag(_attributes_info.get_target_tag());
    // Distribution of new set
    class_dist distribution(_attributes_info.getCount(target_tag));
    // Indices of the subset
    std::vector<size_t> indices;

    // Loop over each sample
    for (uint32_t i = 0; i < size(); ++i)
    {
      // Check the value of the attribute
      if (s.isInBranch(this->begin(i), nbranch))
      {
        attribute value(getattribute(i, target_tag));
        distribution.accum(value.discrete());
        indices.push_back(i);
      }
    }

    // Crate new data set
    dataset *new_set = getNewReference(indices);
    new_set->setDistribution(distribution);
    // Setup the new set
    new_set->setup();
    // Return new data set
    return new_set;
  }
  /*
    training_set::training_set (const dataset &right) :
        dataset (right), _samples (right.getattributes ().getSize ())
    {
      // Copy the data
      static size_t count=0;
      count++;
      std::cout<<"[+] Training set copy : ["<<count<<"]"<<std::endl;

      for (uint32_t i = 0; i < right.size (); ++i)
        {
          // Push this sample into the new set
           for (uint32_t j = 0; j < right.getattributesNumber (); ++j)
                //pushattribute (j, right.getattribute (i, j));
                _samples[j].push_back(right.getattribute (i, j));

        }
    }
  */
  training_set::training_set(const dataset &right,
                             const attribute_information &attributes_info) : dataset(right, attributes_info), _samples(right.getattributes().getSize())
  {
    // Copy the data
    for (uint32_t i = 0; i < right.size(); ++i)
    {
      // Push this sample into the new set
      for (uint32_t j = 0; j < right.getattributesNumber(); ++j)
        // pushattribute (j, right.getattribute (i, j));
        _samples[j].push_back(right.getattribute(i, j));
    }
  }

  dataset *
  training_set::getNewReference(std::vector<size_t> &indices) const
  {
    // const  &indice
     std::vector<size_t, safe_mmap_allocator<size_t>> in;
     std::copy(indices.begin(), indices.end(), std::back_inserter(in));

    return getNewReference(in);
  }

  dataset *
  training_set::getNewReference(std::vector<size_t, safe_mmap_allocator<size_t>> &indices) const
  {
    // const  &indices 
     return new training_setReference(*this, indices);
  }

  testing_set::testing_set(const dataset &right) : dataset(right), _nattr(right.getattributes().getSize())
  {
    // Copy the data
    for (uint32_t i = 0; i < right.size(); i++)
    {
      // Push this sample into the new set

      for (uint32_t j = 0; j < right.getattributesNumber(); ++j)

        _samples.push_back(right.getattribute(i, j));
    }
  }

  dataset *
  testing_set::getNewReference(std::vector<size_t> &indices) const
  {
    return new testing_set_ref(*this, indices);
  }

  testing_set::testing_set(const dataset &right,
                           const attribute_information &attributes_info) : dataset(right, attributes_info), _samples(right.getattributes().getSize())
  {
    // Copy the data
    for (uint32_t i = 0; i < right.size(); ++i)
    {
      // Push this sample into the new set
      for (uint32_t j = 0; j < right.getattributesNumber(); ++j)
        _samples.push_back(right.getattribute(i, j));
    }
  }

  files_collector::files_collector(const std::string &filestem) : _filestem(filestem), _attributes_info(parseNames(_filestem + ".names"))
  {
  }

  files_collector::files_collector(const std::string &filestem,
                                   const attribute_information &attrs) : _filestem(filestem), _attributes_info(attrs)
  {
  }

  files_collector::files_collector(
      const std::string &filestem, const std::string &target,
      const std::vector<std::pair<std::string, std::string>> &attrs) : _filestem(filestem), _attributes_info(target, attrs)
  {
  }

  attribute_information
  files_collector::parseNames(const std::string &names_file)
  {
    std::string line;
    std::ifstream file(names_file.c_str());
    std::vector<std::pair<attribute_name, std::string>> attributes;
    std::string target("");
    size_t nline(1);
    if (file.is_open())
    {
      while (file.good())
      {
        std::getline(file, line);
        // Check line
        if (line.find(":") != std::string::npos)
        {
          // Attribute is defined on this line
          std::vector<std::string> definition;
          tokenize(line, definition, ":");

          if (definition.size() != 2)
          {
            throw std::runtime_error(
                std::string("Bad attribute definition in line ") + std::to_string(nline) + std::string(" in file ") + names_file);
          }

          attribute_name _name(trim(reduce(definition[0], " "), " "));
          std::string attribute_values(
              trim(reduce(definition[1], " "), " "));

          if (_name.compare("target") == 0 || _name == std::string("target") || _name == "target")
          {

            // Target definition
            target = trim(attribute_values, " ");
            std::cout << std::string("found target:") << target
                      << std::endl;
          }
          else
          // Push attribute definition
          {
            std::cout << std::string("pushing [")
                      << _name + std::string("]") << std::endl;

            attributes.push_back(
                std::make_pair(trim(_name, " "),
                               trim(attribute_values, " ")));
          }
        }
        // Increment line
        ++nline;
      }
      file.close();
    }
    else
    {
      throw(std::runtime_error(
          std::string("Can't open the file : ") + names_file));
    }

    // Check if the target was defined on the file

    if (target.size() == 0)
    {
      throw(std::runtime_error(
          std::string("The target attribute is not defined on the file : ") + names_file));
    }

    // Return information
    return attribute_information(target, attributes);
  }

  void
  files_collector::pushFileData(const std::string &filename, dataset *data)
  {
    std::string line;
    std::ifstream file(filename.c_str());
    size_t nline(1);
    if (file.is_open())
    {
      while (file.good())
      {
        getline(file, line);
        // Check line
        if (line.length() > 0 && line.find(",") != std::string::npos)
        {
          // Sample is defined on this line
          std::vector<std::string> sample;
          tokenize(reduce(line, ""), sample, ",");

          // Check number of samples
          if (sample.size() != data->getattributes().getSize())
          {

            throw(std::runtime_error(
                std::string("Bad number of attributes in line ") + std::to_string(nline) + " in file " + filename));
          }
          else

            data->pushData(sample.begin(), sample.end());

          // Increment line
          ++nline;
        }
      }

      file.close();
      // Setup data
      data->setup();
    }
    else
    {
      throw(std::runtime_error("Can't open the file : " + filename));
    }
  }

  afiles_collector::afiles_collector(const std::string &filestem,
                                     const std::random_device &rm) : _filestem(filestem), _target(""), _orig_attributes(0), _attributes_info(parseNames(_filestem + ".names"))
  {
	  //undefined reference to rm
	  //
	  //std::mt19937 generator(rm); 
	  //generator();
	  ptrdiff_t last_random= ptrdiff_t(&rm);
	  if ( last_random == ptrdiff_t(0) )
		  throw std::runtime_error( "should never reach random null" );

	  
  }

  afiles_collector::afiles_collector(
      const std::string &filestem, const std::string &target,
      const std::vector<std::pair<std::string, std::string>> &attrs,
      const std::random_device &rm) : _filestem(filestem), _target(target), _orig_attributes(attrs), _attributes_info(addArtificial(target, attrs))
  {

	  ptrdiff_t last_random= ptrdiff_t(&rm);
          if ( last_random == ptrdiff_t(0) )
                  throw std::runtime_error( "should never reach random null" );

  }

  attribute_information
  afiles_collector::addArtificial(
      const std::string &target,
      const std::vector<std::pair<std::string, std::string>> &attrs)
  {
    //
    // New map for attributes (will add artificial attributes)
    //
    
    std::vector<std::pair<std::string, std::string>> attributes(
        _orig_attributes);
    
    for(auto  & it : attrs ) 
    {
	attributes.push_back(it);
    }

    for (std::vector<std::pair<std::string, std::string>>::const_iterator it =
             _orig_attributes.begin();
         it != _orig_attributes.end(); ++it)
    {
      std::pair<std::string, std::string> pair_name(*it);
      // Create artificial attribute
      if (pair_name.first != target)
      {
        pair_name.first = "@metamon_artifical_" + pair_name.first;
        attributes.push_back(pair_name);
      }
    }

    // Return information
    return attribute_information(target, attributes);
  }

  attribute_information
  afiles_collector::parseNames(const std::string &names_file)
  {
    std::string line;
    std::ifstream file(names_file.c_str());
    size_t nline(1);
    if (file.is_open())
    {
      while (file.good())
      {
        std::getline(file, line);
        // Check line
        if (line.length() > 0 && line.find(":") != std::string::npos)
        {
          // Attribute is defined on this line
          std::vector<std::string> definition;
          tokenize(line, definition, ":");

          if (definition.size() != 2)
          {
            throw(std::runtime_error(
                std::string("Bad attribute definition in line ") + std::to_string(nline) + std::string(" in file ") + names_file));
          }

          attribute_name name(reduce(definition[0], ""));
          std::string attribute_values(reduce(definition[1], ""));

          if (name == "target")
            // Target definition
            _target = attribute_values;
          else
            // Push attribute definition
            _orig_attributes.push_back(
                make_pair(name, attribute_values));
        }
        // Increment line
        ++nline;
      }
      file.close();
    }
    else
    {
      throw(std::runtime_error("Can't open the file : " + names_file));
    }

    // Check if the target was defined on the file
    if (_target.size() == 0)
    {
      throw(std::runtime_error(
          std::string("The target attribute is not defined on the file : ") + names_file));
    }

    // New map for attributes (will add artificial attributes)
    std::vector<std::pair<std::string, std::string>> attributes(
        _orig_attributes);

    for (std::vector<std::pair<std::string, std::string>>::const_iterator it =
             _orig_attributes.begin();
         it != _orig_attributes.end(); ++it)
    {
      std::pair<std::string, std::string> pair_name(*it);
      // Create artificial attribute
      if (pair_name.first != _target)
      {
        pair_name.first = "@metamon_artifical_" + pair_name.first;
        attributes.push_back(pair_name);
      }
    }

    // Return information
    return attribute_information(_target, attributes);
  }

  void
  afiles_collector::pushFileData(const std::string &filename,
                                 dataset *data) const
  {
    std::string line;
    std::ifstream file(filename.c_str());
    size_t nline(1);
    if (file.is_open())
    {
      while (file.good())
      {
        std::getline(file, line);
        // Check line
        if (line.length() > 0 && line.find(",") != std::string::npos)
        {
          // Sample is defined on this line
          std::vector<std::string> sample;
          tokenize(reduce(line, ""), sample, ",");

          // Get target TAG
          uint32_t target_offset =
              data->getattributes().get_target_tag();
          uint32_t current_size(sample.size());
          // Push back artificial attribute
          for (size_t i = 0; i < current_size; ++i)
            if (i != target_offset)
              sample.push_back(sample[i]);

          // Check number of samples
          if (sample.size() != data->getattributes().getSize())
          {
            throw(std::runtime_error(
                "Bad number of attributes in line " + std::to_string(nline) + " in file " + filename));
          }

          data->pushData(sample.begin(), sample.end());
        }
        // Increment line
        ++nline;
      }
      // Close file
      file.close();

      // Setup data
      data->setup();

      // Now randomly permute each artificial variable
      std::vector<attribute> dummy(data->size());
      std::random_device d;
      // Permute instance of each attribute associated with this method
      for (size_t i = 0; i < data->getattributes().getSize(); ++i)
        if (data->getattributes().getName(i).find("@metamon_artifical") != std::string::npos)
          data->permute(d, i, &dummy);
    }
    else
    {

      throw(std::runtime_error("Can't open the file : " + filename));
    }
  }

  // Get most frequent class (target attribute) on the data set
  attribute
  getBestClass(const dataset &data)
  {
    // Get target attribute tag
    attribute_tag target = data.getattributes().get_target_tag();
    // Get number of different outcomes on the data set
    uint32_t count = data.getattributes().getCount(target);
    // Hold frequency of each class on the set
    std::vector<uint32_t> class_count(count, 0);
    // Accumulate counters
    for (uint32_t i = 0; i < data.size(); ++i)
      class_count[data.getattribute(i, target).discrete()]++;
    // Get max occurrence

    discrete_value max = discrete_value(std::distance(class_count.begin(), std::max_element(class_count.begin(), class_count.end())));
    ///- class_count.begin ();
    // Return attribute with this discrete value
    return attribute(max);
  }

  Float
  entropy(const dataset &data)
  {
    attribute_tag tag = data.getattributes().get_target_tag();
    std::vector<Float> probs(data.getattributes().getCount(tag), 0.);
    for (uint32_t i = 0; i < data.size(); ++i)
    {
      cont_value v = data.getattribute(i, tag).continous();
      discrete_value d = data.getattribute(i, tag).discrete();

      if (d >= probs.size())
        if (v < probs.size())
          probs[discrete_value(v)]++;
        else
          d = 0;
      else
        probs[d]++;
    }

    // Entropy
    Float entropy = 0.0;
    for (uint32_t i = 0; i < probs.size(); ++i)
    {
      Float prob = probs[i] / (Float)data.size();
      if (prob != 0.0)
        entropy += -prob * log<2>(prob);
    }
    // Return entropy
    return entropy;
  }
  // Gini index
  Float gini(const dataset &data)
  {
    attribute_tag tag = data.getattributes().get_target_tag();
    std::vector<Float> probs(data.getattributes().getCount(tag), 0.0);
    for (uint32_t i = 0; i < data.size(); ++i)
    {
      cont_value v = data.getattribute(i, tag).continous();
      discrete_value d = data.getattribute(i, tag).discrete();

      if (d >= probs.size())
        if (v < probs.size())
          probs[discrete_value(v)]++;
        else
          d = 0;
      else
        probs[d]++;

      /// probs[data.getattribute (i,tag).discrete ()]++;
    }
    // gini
    Float gini = 1.0;
    for (uint32_t i = 0; i < probs.size(); ++i)
    {

      Float c = (Float)std::count(probs.begin(), probs.end(), probs[i]);

      // weight gini index against other tags

      Float prob = c * probs[i] / (Float)data.size();

      gini -= prob * prob;
    }
    // Return gini
    return gini;
  }
  Float variance(const dataset &data)
  {
    attribute_tag tag = data.getattributes().get_target_tag();
    std::vector<Float> probs(data.getattributes().getCount(tag), 0);
    for (uint32_t i = 0; i < data.size(); ++i)
    {
      cont_value v = data.getattribute(i, tag).continous();
      discrete_value d = data.getattribute(i, tag).discrete();

      if (d >= probs.size())
        if (v < probs.size())
          probs[discrete_value(v)]++;
        else
          d = 0;
      else
        probs[d]++;
    } // variance
    Float variance = 0.0;
    for (uint32_t i = 0; i < probs.size(); ++i)
    {

      Float prob = probs[i] / (Float)data.size();
      variance += prob * prob;
    }
    // Return variance
    return variance;
  }
  Float mean(const dataset &data)
  {
    attribute_tag tag = data.getattributes().get_target_tag();
    std::vector<Float> probs(data.getattributes().getCount(tag), 0);
    for (uint32_t i = 0; i < data.size(); ++i)
    {
      cont_value v = data.getattribute(i, tag).continous();
      discrete_value d = data.getattribute(i, tag).discrete();

      if (d >= probs.size())
        if (v < probs.size())
          probs[discrete_value(v)]++;
        else
          d = 0;
      else
        probs[d]++;
    }
    // mean
    Float mean = 0.0;
    Float sum = std::accumulate(probs.begin(), probs.end(), 0.0);
    mean = sum / Float(probs.size());
    // Return mean
    return mean;
  }
  Float stddev(const dataset &data)
  {
    attribute_tag tag = data.getattributes().get_target_tag();
    std::vector<Float> probs(data.getattributes().getCount(tag), 0);
    for (uint32_t i = 0; i < data.size(); ++i)
    {
      cont_value v = data.getattribute(i, tag).continous();
      discrete_value d = data.getattribute(i, tag).discrete();

      if (d >= probs.size())
        if (v < probs.size())
          probs[discrete_value(v)]++;
        else
          d = 0;
      else
        probs[d]++;
    } // stddev
    Float stddev = 0.0;
    Float sum = std::accumulate(probs.begin(), probs.end(), Float(0.0));
    Float mean = sum / Float(probs.size());
    Float accum = 0.0;
    std::for_each(std::begin(probs), std::end(probs), [&](const Float d)
                  { accum += (d - mean) * (d - mean); });

    stddev = sqrt(accum / (Float(probs.size())));

    // Return stddev
    return stddev;
  }
  Float skewness(const dataset &data)
  {
    attribute_tag tag = data.getattributes().get_target_tag();
    std::vector<Float> probs(data.getattributes().getCount(tag), 0);
    for (uint32_t i = 0; i < data.size(); ++i)
    {
      cont_value v = data.getattribute(i, tag).continous();
      discrete_value d = data.getattribute(i, tag).discrete();

      if (d >= probs.size())
        if (v < probs.size())
          probs[discrete_value(v)]++;
        else
          d = 0;
      else
        probs[d]++;

      // probs[data.getattribute (i,tag).discrete ()]++;
    }
    // skewness
    //Float kurtosis = 0., kurt = 0.0;
    Float delta, delta_n, delta_n2, term1;
    size_t n = 0;
    Float M1, M2, M3, M4;
    M1 = M2 = M3 = M4 = 0.0;
    Float skewn = 0.0, skewness = 0.0;
    for (uint32_t i = 0; i < probs.size(); ++i)
    {

      Float prob = probs[i] / (Float)data.size();
      n++;
      delta = prob - M1;
      delta_n = delta / n;
      delta_n2 = delta_n * delta_n;
      term1 = delta * delta_n * (n - 1);
      M1 += delta_n;
      M4 += term1 * delta_n2 * (n * n - 3 * n + 3) + 6 * delta_n2 * M2 - 4 * delta_n * M3;
      M3 += term1 * delta_n * (n - 2) - 3 * delta_n * M2;
      M2 += term1;
      skewn = sqrt(n) * M3 / pow(M2, 1.5);
      skewness += skewn / (Float)data.size();
    }
    // Return skewness
    return skewness;
  }
  // kurtosis :
  Float kurtosis(const dataset &data)
  {
    attribute_tag tag = data.getattributes().get_target_tag();
    std::vector<Float> probs(data.getattributes().getCount(tag), 0);
    for (uint32_t i = 0; i < data.size(); ++i)

    {

      // fix mislabeling discrete and continous target labels
      discrete_value d(data.getattribute(i, tag).discrete());
      cont_value c(data.getattribute(i, tag).continous());

      if (d > probs.size()){
        if (c < probs.size())
          d = (discrete_value)c;
        else
          d = probs.size() - 1;
      }
      probs[d]++;
    }
    // kurtosis
    Float kurtosis = 0., kurt = 0.0;
    Float delta, delta_n, delta_n2, term1;
    size_t n = 0;
    Float M1, M2, M3, M4;
    M1 = M2 = M3 = M4 = 0.0;
    for (uint32_t i = 0; i < probs.size(); ++i)
    {

      Float prob = probs[i] / (Float)data.size();
      n++;
      delta = prob - M1;
      delta_n = delta / n;
      delta_n2 = delta_n * delta_n;
      term1 = delta * delta_n * (n - 1);
      M1 += delta_n;
      M4 += term1 * delta_n2 * (n * n - 3 * n + 3) + 6 * delta_n2 * M2 - 4 * delta_n * M3;
      M3 += term1 * delta_n * (n - 2) - 3 * delta_n * M2;
      M2 += term1;
      kurt = (n * M4) / (M2 * M2) - 3;
      kurtosis += kurt * kurt;
    }
    // Return kurtosis
    return kurtosis;
  }
  // return the median
  Float median(const dataset &data)
  {
    attribute_tag tag = data.getattributes().get_target_tag();
    std::vector<Float> probs(data.getattributes().getCount(tag), 0);
    for (uint32_t i = 0; i < data.size(); ++i)
    {
      // fix mislabeling discrete and continous target labels
      discrete_value d(data.getattribute(i, tag).discrete());
      cont_value c(data.getattribute(i, tag).continous());

      if (d > probs.size()){
        if (c < probs.size())
          d = (discrete_value)c;
        else
          d = probs.size() - 1;
      }
      probs[d]++;
    }
    // median
    Float median = 0.0;
    for (uint32_t i = 0; i < probs.size(); ++i)
    {

      Float prob = probs[i] / (Float)data.size();

      median += prob * prob;
    }
    // Return median
    return median;
  }
  // return the mode
  Float mode(const dataset &data)
  {
    attribute_tag tag = data.getattributes().get_target_tag();
    std::vector<Float> probs(data.getattributes().getCount(tag), 0);
    for (uint32_t i = 0; i < data.size(); ++i)
    {

      // fix mislabeling discrete and continous target labels
      discrete_value d(data.getattribute(i, tag).discrete());
      cont_value c(data.getattribute(i, tag).continous());

      if (d > probs.size()) {

        if (c < probs.size())
          d = (discrete_value)c;
        else
          d = probs.size() - 1;
      }
      probs[d]++;

      // probs[data.getattribute (i,tag).discrete ()]++;
    }
    // mode
    Float mode = 0.0;
    for (uint32_t i = 0; i < probs.size(); ++i)
    {

      Float prob = probs[i] / (Float)data.size();

      mode += prob * prob;
    }
    // Return mode
    return mode;
  }
  Float min(const dataset &data)
  {
    attribute_tag tag = data.getattributes().get_target_tag();
    std::vector<Float> probs(data.getattributes().getCount(tag), 0);
    Float min = 0.0;
    for (uint32_t i = 0; i < data.size(); ++i)
      probs[data.getattribute(tag, i).discrete()]++;
    // min
    for (uint32_t i = 0; i < probs.size(); ++i)
    {

      Float prob = probs[i] / (Float)data.size();

      min += prob * prob;
    }
    // Return min
    return min;
  }
  Float max(const dataset &data)
  {

    attribute_tag tag = data.getattributes().get_target_tag();
    std::vector<Float> probs(data.getattributes().getCount(tag), 0);
    Float max = 0.0;
    for (uint32_t i = 0; i < data.size(); ++i)
    {
      cont_value v = data.getattribute(i, tag).continous();
      discrete_value d = data.getattribute(i, tag).discrete();

      if (d >= probs.size())
        if (v < probs.size())
          probs[discrete_value(v)]++;
        else
          d = 0;
      else
        probs[d]++;

    } // max
    for (uint32_t i = 0; i < probs.size(); ++i)
    {

      Float prob = probs[i] / (Float)data.size();

      max = prob > max ? prob : max; // prob*prob;
    }
    // Return max
    return max;
  }
  Float sum_of_squares(const dataset &data)
  {

    attribute_tag tag = data.getattributes().get_target_tag();
    std::vector<Float> probs(data.getattributes().getCount(tag), 0);
    Float sum_of_squares = 0.0;
    for (uint32_t i = 0; i < data.size(); ++i)
    {
      cont_value v = data.getattribute(i, tag).continous();
      discrete_value d = data.getattribute(i, tag).discrete();

      if (d >= probs.size())
        if (v < probs.size())
          probs[discrete_value(v)]++;
        else
          d = 0;
      else
        probs[d]++;
    }
    // probs[data.getattribute ( tag , i ).discrete ()]++;
    // sum_of_squares
    for (uint32_t i = 0; i < probs.size(); ++i)
    {

      Float prob = probs[i] / (Float)data.size();

      sum_of_squares += prob * prob;
    }
    // Return sum_of_squares
    return sum_of_squares;
  }
  // return the median from absolute deviation
  Float median_absolute_deviation(const dataset &data)
  {
    attribute_tag tag = data.getattributes().get_target_tag();
    std::vector<Float> probs(data.getattributes().getCount(tag), 0);
    Float median_absolute_deviation = 0.0;
    for (uint32_t i = 0; i < data.size(); ++i)
    {
      cont_value v = data.getattribute(i, tag).continous();
      discrete_value d = data.getattribute(i, tag).discrete();

      if (d >= probs.size())
        if (v < probs.size())
          probs[discrete_value(v)]++;
        else
          d = 0;
      else
        probs[d]++;
      //
      // probs[data.getattribute (i, tag).discrete ()]++;
    }
    // median_absolute_deviation
    for (uint32_t i = 0; i < probs.size(); ++i)
    {

      Float prob = probs[i] / (Float)data.size();

      median_absolute_deviation += prob * prob;
    }
    // Return median_absolute_deviation
    return median_absolute_deviation;
  }

  std::map<std::string, Float>
  getWeightMap(const attribute_information &attrs, std::ifstream &file)
  {
    std::string line;
    std::map<std::string, Float> weights;
    std::string target("");
    size_t nline(1);
    if (file.is_open())
    {
      while (file.good())
      {
        std::getline(file, line);
        // Attribute is defined on this line
        std::vector<std::string> definition;
        tokenize(line, definition, " ");

        // Check for line
        if (line.size() == 0)
          break;
        if (definition.size() != 2)
        {
          throw(std::runtime_error(
              "Bad importance definition in line " + std::to_string(nline)));
        }

        attribute_name _name(reduce(definition[0], ""));
        attribute_tag tag = attrs.getTag(_name);
        Float value(
            attribute_definition::fromString<Float>(definition[1]));
        std::cout << "[#] Reading attribute (tag = " << tag << ") " << _name
                  << " with weight " << value << std::endl;
        weights.insert(std::make_pair(_name, value));

        // Increment line
        ++nline;
      }
      file.close();
    }
    // Return information
    return weights;
  }

  void
  printImportanceMap(std::ostream &out,
                     const std::vector<std::pair<std::string, Float>> &imps)
  {
    for (std::vector<std::pair<std::string, Float>>::const_iterator it =
             imps.begin();
         it != imps.end(); ++it)
      out << (*it).first << " " << (*it).second << std::endl;
  }

  // static debug counters:
  std::atomic_int dataset::id_counter(0);
  std::atomic_int dataset::dest_counter(0);
  std::atomic_uint64_t dataset::attribute_iterator::_instance_counter(0);

} /* namespace provallo */
