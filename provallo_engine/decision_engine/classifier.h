/*
 * classifier.h
 *
 *  Created on: Jan 19, 2022
 *      Author: kardon
 */

#ifndef DECISION_ENGINE_CLASSIFIER_H_
#define DECISION_ENGINE_CLASSIFIER_H_
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <unordered_set>
#include <unordered_map>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <csignal>
#include <thread>
#include <mutex>
typedef void (*sig_t_)(int);
using std::raise;
using std::signal;

#include "utils.h"

#include "dataset.h"
#include "attribute.h"
#include "parameters.h"
#include  "kdt.h"
#include "split_utils.hpp"
#define _FOR_R 1

namespace provallo
{

  /*  improved  isoforest classifier
   *  hyper parameters tuning and configuration is done by GAN
   *  parameters and data sets are loaded at runtime.
   *
   *  dataset adapter
   *
   *
   * */

  typedef struct PredictionData
  {
    real_t *numeric_data;
    int *categ_data;
    size_t nrows;
    bool is_col_major;
    size_t ncols_numeric; /* only required for row-major data */
    size_t ncols_categ;   /* only required for row-major data */
    real_t *Xc;           /* only for sparse matrices */
    sparse_ix *Xc_ind;    /* only for sparse matrices */
    sparse_ix *Xc_indptr; /* only for sparse matrices */
    real_t *Xr;           /* only for sparse matrices */
    sparse_ix *Xr_ind;    /* only for sparse matrices */
    sparse_ix *Xr_indptr; /* only for sparse matrices */
  } prediction_data;

  typedef struct iso_tree_struct
  {
    ColType col_type = NotUsed;
    size_t col_num = 0;
    real_t num_split = 0.;
    std::vector<char> cat_split;
    int chosen_cat = 0;
    size_t tree_left = 0;
    size_t tree_right = 0;
    real_t pct_tree_left = 0.;
    real_t score = 0.00000001; /* will not be integer when there are weights or early stop */
    real_t range_low = -HUGE_VAL;
    real_t range_high = HUGE_VAL;
    real_t remainder; /* only used for distance/similarity */

    iso_tree_struct() = default;

  } IsoTree;

  typedef struct iso_forest
  {
    std::vector<std::vector<iso_tree_struct>> trees;
    NewCategAction new_cat_action;
    CategSplit cat_split_type;
    MissingAction missing_action;
    ScoringMetric scoring_metric;

    real_t exp_avg_depth;
    real_t exp_avg_sep;
    size_t orig_sample_size;
    bool has_range_penalty;

    size_t
    ntrees() const
    {
      return trees.size();
    }
  } IsoForest;

  typedef struct ext_iso_forest
  {
    std::vector<std::vector<iso_hplane>> hplanes;
    NewCategAction new_cat_action;
    CategSplit cat_split_type;
    MissingAction missing_action;
    ScoringMetric scoring_metric;

    real_t exp_avg_depth;
    real_t exp_avg_sep;
    size_t orig_sample_size;
    bool has_range_penalty;

    size_t
    ntrees() const
    {
      return hplanes.size();
    }
  } ExtIsoForest;

  typedef struct Imputer
  {
    size_t ncols_numeric;
    size_t ncols_categ;
    std::vector<int> ncat;
    std::vector<std::vector<ImputeNode>> imputer_tree;
    std::vector<real_t> col_means;
    std::vector<int> col_modes;
    void init(size_t ncols_numeric, size_t ncols_categ, const std::vector<int> &ncat)
    {
      this->ncols_numeric = ncols_numeric;
      this->ncols_categ = ncols_categ;
      this->ncat = ncat;
      this->imputer_tree.resize(ncols_numeric + ncols_categ);
      
      if(ncols_numeric > 0)
        this->col_means.resize(ncols_numeric);
      if(ncols_categ > 0)
        this->col_modes.resize(ncols_categ);

      // Initialize imputer tree
      for (auto& it : this->imputer_tree)
        it.resize(ncols_numeric+ncols_categ);
      // Initialize column means and modes
      for (size_t col = 0; col < ncols_numeric; col++)
        this->col_means[col] = NAN;
      for (size_t col = 0; col < ncols_categ; col++)
        this->col_modes[col] = -1;
      // Initialize imputer tree

    } /* init */
    //default constructor
    Imputer() = default;
    //copy constructor
    Imputer(const Imputer& other) = default;
    //move constructor
    Imputer(Imputer&& other) = default;
    //copy assignment
    Imputer& operator=(const Imputer& other) = default;
    //move assignment
    Imputer& operator=(Imputer&& other) = default;
    //destructor
    ~Imputer() = default;

  } Imputer;

  typedef struct SingleTreeIndex
  {
    std::vector<size_t> terminal_node_mappings;
    std::vector<real_t> node_distances;
    std::vector<real_t> node_depths;
    std::vector<size_t> reference_points;
    std::vector<size_t> reference_indptr;
    std::vector<size_t> reference_mapping;
    size_t n_terminal;
  } TreeNodeIndex;

  typedef struct TreesIndexer
  {
    std::vector<SingleTreeIndex> indices;
    TreesIndexer() = default;
  } TreesIndexer;
  class classifier;


  class signal_switcher
  {
    sig_t_ old_sig;
    bool is_active;

  public:
    static bool handle_is_locked;
    static bool interrupt_switch;

    static void
    set_interrup_global_variable(int s);
    signal_switcher();
    ~signal_switcher();
    void
    restore_handle();
  };

  class classifier_factory
  {

    // deserialize/serialize classifiers
    friend class classifier; // friend class classifier

  public:
    typedef classifier *(*create_classifier_func)(std::istream &input); // create_classifier_func
    typedef std::map<std::string, create_classifier_func> map_type;     // map_type
  private:
    map_type _map;
    classifier_factory()
    {
    }

  public:
    static classifier_factory &
    get_factory()
    {
      static classifier_factory factory;

      return factory;
    }
    void
    register_creator(const std::string &name, create_classifier_func func)
    {
      _map[name] = func;
    }
    create_classifier_func
    get_creator(const std::string &name)
    {
      map_type::iterator it = _map.find(name);
      if (it == _map.end()) // end ()
      {
        return nullptr;
      }
      else
      {
        return it->second;
      }
    };
  }; // end class classifier_factory

  class classifier
  {

  protected:

    // Attributes information
    attribute_information _attributes_info;

    split_method_factory *_split_factory;
    std::random_device _rand;
    std::vector<size_t> _offsets;
    // classifier types :
    // semi auto-labeled, fully-supervised,generative, sparse/binary/multi-class
    std::string _name;
    const dataset& _data;
    const parameter_base& _parameters;
    
  public:
  protected:
    const std::random_device &
    getRandom() const
    {
      return _rand;
    }
    std::random_device &
    getRandom()
    {
      return _rand;
    }

    // Print classifier information
    virtual void
    print(std::ostream &out) const = 0;
    // Printer
    friend std::ostream &
    operator<<(std::ostream &out, const classifier &q);
    // Internal function to serialize data into the buffer
    virtual void
    serialize(classifier *serial) const = 0;

    // Get access to the split method factory
    const split_method_factory &
    splitFactory() const
    {
      if(!_split_factory)
      {
        throw std::runtime_error("split factory is null");
      }
      return *_split_factory;
    }

   public:
    
    //order    : 
    //1. attribute_information _attributes_info;
    //2. split_method_factory *_split_factory;
    //3. std::random_device _rand;
    //4. mmap_vector<size_t> _offsets;
    //5. name
    //6. dataset
    //7. parameters
    //
    // classifier types :
    // semi auto-labeled, fully-supervised,generative, sparse/binary/multi-class

    
    // A classifier should be constructed from a data set and a parameter object
    classifier(const dataset &data, const parameter_base &parameters = none(),
               const std::random_device &ra = std::random_device(), split_method_factory *factory = nullptr) : _attributes_info(data.getattributes()), _split_factory(factory == nullptr ? new split_method_factory(data, ra) : factory), _rand(),_offsets(data.get_sorted_indices().size()), _name("classifier"), _data(data), _parameters(parameters)   
    {
      // Initialize the offsets
      //set offset size
      _offsets.resize(data.get_sorted_indices().size());
      //fill offsets iota of size data.get_sorted_indices().size()
      std::iota(_offsets.begin(), _offsets.end(), 0 );
      //set the factory


      if(_split_factory == nullptr)
      {
       _split_factory = new split_method_factory(data, ra);
       this->_factory_allocated = true;

      }
      else _factory_allocated = false;

    }

    // A classifier should be constructed from a data set and a parameter object
    classifier( const classifier& other) : _attributes_info(other._attributes_info), _split_factory(other._split_factory),_rand(),_offsets(other._offsets),_name(other._name), _data(other._data),_parameters(other._parameters)  {
      this->_factory_allocated = false;
    } 
    classifier ( classifier&& other): _attributes_info(other._attributes_info), _split_factory(other._split_factory),_rand(),_offsets(other._offsets),_name(other._name), _data(other._data),_parameters(other._parameters)  {
      this->_factory_allocated = false;
    } 

    // Construct classifier from buffer
    classifier(const classifier *deserial) : _attributes_info(deserial->_attributes_info), _split_factory(deserial->_split_factory),_rand(), _offsets(deserial->_offsets), _name(deserial->_name), _data(deserial->_data), _parameters(deserial->_parameters)
    {
    }   
    

    // Build classifier from an input stream
    static classifier *
    buildClassifier(std::istream &input)
    {
      std::string name;
      input >> name;
      classifier_factory::create_classifier_func func = classifier_factory::get_factory().get_creator(name); // get_creator (name); //get_creator (name);
      if (func == nullptr)
      {
        throw std::runtime_error("Unknown classifier type: " + name);
      }
      else
      {
        return func(input);
      }
    }

    // Build classifier from information on the buffer
    static classifier *
    buildClassifier(const classifier *deserial);
    // Return classifier from file name
    static classifier *
    buildClassifier(const std::string &filename)
    {
      std::ifstream input(filename.c_str());
      return buildClassifier(input);
    }

    // Classify a set of attributes
    virtual attribute
    classify(dataset::attribute_iterator begin,
             dataset::attribute_iterator end) const = 0;
    virtual attribute
    classify(std::vector<attribute>::const_iterator begin,
             std::vector<attribute>::const_iterator end) const = 0;

    // Get distribution of outcomes
    virtual class_dist
    posterior(dataset::attribute_iterator begin,
              dataset::attribute_iterator end) const = 0;
    virtual class_dist
    posterior(std::vector<attribute>::const_iterator begin,
              std::vector<attribute>::const_iterator end) const = 0;

    // Get type of classifier
    virtual classifier_type
    get_type() const = 0;

    size_t get_nclasses() const
    {
      size_t ret = 0;
      if (_split_factory && _split_factory->getTargetMethod())
      {
        ret = _split_factory->getTargetMethod()->size();
      }
      return ret;
    }

    // Classify normalized string (CSV of attributes)
    attribute
    classify(const std::string &normalized) const
    {
      std::vector<attribute> cases;
      // Push cases from normalized string
      parseNormalizedString(_attributes_info, normalized, &cases);
      // Classify
      return classify(cases.begin(), cases.end());
    }
    class_dist
    posterior(const std::string &normalized) const
    {
      std::vector<attribute> cases;
      // Push cases from normalized string
      parseNormalizedString(_attributes_info, normalized, &cases);
      // Classify
      return posterior(cases.begin(), cases.end());
    }

    // Get class name from attribute
    std::string
    getClassName(const attribute &attribute) const
    {
      attribute_tag target = _attributes_info.get_target_tag();
      return _attributes_info.getValue(target, attribute);
    }
    const dataset & get_data() const
    {
      return _data;
    }

    const std::string
    get_name();

    virtual ~classifier();
    bool delete_factory()
    {
      if (_split_factory && _factory_allocated)
      {
        delete _split_factory;
        _split_factory = nullptr;
        _factory_allocated = false;
        return true;
      }
      return false;
    }
    bool _factory_allocated = false;
  };
  
  template <class itype>
  void
  inspect_serialized_object(itype &serialized_bytes, bool &is_isotree_model,
                            bool &is_compatible, bool &has_combined_objects,
                            bool &has_IsoForest, bool &has_ExtIsoForest,
                            bool &has_Imputer, bool &has_Indexer,
                            bool &has_metadata, size_t &size_metadata,
                            bool &has_same_int_size,
                            bool &has_same_size_t_size,
                            bool &has_same_endianness,
                            bool &lacks_range_penalty,
                            bool &lacks_scoring_metric)
  {
   // auto saved_position = set_return_position(serialized_bytes);
    //bool has_same_int_size = false;
    //bool has_same_size_t_size = false;
    //bool has_same_endianness = false;
    //bool lacks_range_penalty = false;
    if (sizeof (serialized_bytes)!=sizeof (itype))
    {
      is_compatible = false;
      return;
    }

    is_compatible = false;
    is_isotree_model = false;
    has_combined_objects = false;
    has_IsoForest = false;
    has_ExtIsoForest = false;
    has_Imputer = false;
    has_Indexer = false;
    has_metadata = false;
    size_metadata = 0;
    has_same_int_size = false;

    has_same_size_t_size = false;
    has_same_endianness = false;
    lacks_range_penalty = false;
    lacks_scoring_metric = false;
    //serialized_bytes  = serialized_bytes + 1; 
    //bool has_same_int_size = false;
    //bool lacks_indexer = false;

    //bool has_same_real_t = false;
    //bool has_incomplete_watermark = false;
    // TODO -> rewrite serialization layer : s
    //return_to_position(serialized_bytes, saved_position); 

  }

  inline void
  inspect_serialized_object(const char *serialized_bytes,
                            bool &is_isotree_model, bool &is_compatible,
                            bool &has_combined_objects, bool &has_IsoForest,
                            bool &has_ExtIsoForest, bool &has_Imputer,
                            bool &has_Indexer, bool &has_metadata,
                            size_t &size_metadata)
  {

    bool has_same_int_size, has_same_size_t_size, has_same_endianness,
        lacks_range_penalty, lacks_scoring_metric;

    const char *in = serialized_bytes;
    inspect_serialized_object<const char *>(in, is_isotree_model, is_compatible,
                                            has_combined_objects, has_IsoForest,
                                            has_ExtIsoForest, has_Imputer,
                                            has_Indexer, has_metadata,
                                            size_metadata, has_same_int_size,
                                            has_same_size_t_size,
                                            has_same_endianness,
                                            lacks_range_penalty,
                                            lacks_scoring_metric);
  }
  inline void
  inspect_serialized_object(std::FILE *serialized_bytes, bool &is_isotree_model,
                            bool &is_compatible, bool &has_combined_objects,
                            bool &has_IsoForest, bool &has_ExtIsoForest,
                            bool &has_Imputer, bool &has_Indexer,
                            bool &has_metadata, size_t &size_metadata)
                            {
                                bool has_same_int_size, has_same_size_t_size, has_same_endianness,
                                lacks_range_penalty, lacks_scoring_metric;
                                inspect_serialized_object<std::FILE*>(serialized_bytes, is_isotree_model, is_compatible,
                                has_combined_objects, has_IsoForest, has_ExtIsoForest, has_Imputer,
                                has_Indexer, has_metadata, size_metadata, has_same_int_size,
                                has_same_size_t_size, has_same_endianness, lacks_range_penalty,
                                lacks_scoring_metric);

                            }
  inline void
  inspect_serialized_object(std::istream &serialized_bytes,
                            bool &is_isotree_model, bool &is_compatible,
                            bool &has_combined_objects, bool &has_IsoForest,
                            bool &has_ExtIsoForest, bool &has_Imputer,
                            bool &has_Indexer, bool &has_metadata,
                            size_t &size_metadata)
  {
    bool has_same_int_size, has_same_size_t_size, has_same_endianness,
        lacks_range_penalty, lacks_scoring_metric;

    inspect_serialized_object<std::istream>(serialized_bytes, is_isotree_model,
                                            is_compatible,
                                            has_combined_objects,
                                            has_IsoForest, has_ExtIsoForest,
                                            has_Imputer, has_Indexer,
                                            has_metadata, size_metadata,
                                            has_same_int_size,
                                            has_same_size_t_size,
                                            has_same_endianness,
                                            lacks_range_penalty,
                                            lacks_scoring_metric

    );
  }

  inline void
  inspect_serialized_object(const std::string &serialized_bytes,
                            bool &is_isotree_model, bool &is_compatible,
                            bool &has_combined_objects, bool &has_IsoForest,
                            bool &has_ExtIsoForest, bool &has_Imputer,
                            bool &has_Indexer, bool &has_metadata,
                            size_t &size_metadata);
  size_t
  determine_serialized_size(const iso_forest &model) noexcept;
  size_t
  determine_serialized_size(const ExtIsoForest &model) noexcept;
  size_t
  determine_serialized_size(const Imputer &model) noexcept;
  size_t
  determine_serialized_size(const TreesIndexer &model) noexcept;
  void
  serialize_IsoForest(const iso_forest &model, char *out);
  void
  serialize_IsoForest(const iso_forest &model, std::FILE *out);
  void
  serialize_IsoForest(const iso_forest &model, std::ostream &out);
  std::string
  serialize_IsoForest(const iso_forest &model);
  void
  deserialize_IsoForest(iso_forest &model, const char *in);
  void
  deserialize_IsoForest(iso_forest &model,std::FILE *in);
  void
  deserialize_IsoForest(iso_forest &model, std::istream &in);
  void
  deserialize_IsoForest(iso_forest &model, const std::string &in);
  void
  serialize_ExtIsoForest(const ExtIsoForest &model, char *out);

  void
  serialize_ExtIsoForest(const ExtIsoForest &model, std::FILE *out);
  void
  serialize_ExtIsoForest(const ExtIsoForest &model, std::ostream &out);
  std::string
  serialize_ExtIsoForest(const ExtIsoForest &model);
  void
  deserialize_ExtIsoForest(ExtIsoForest &model, const char *in);
  void
  deserialize_ExtIsoForest(ExtIsoForest &model, std::FILE *in);
  void
  deserialize_ExtIsoForest(ExtIsoForest &model, std::istream &in);
  void
  deserialize_ExtIsoForest(ExtIsoForest &model, const std::string &in);
  void
  serialize_Imputer(const Imputer &model, char *out);
  void
  serialize_Imputer(const Imputer &model, std::FILE *out);
  void
  serialize_Imputer(const Imputer &model, std::ostream &out);
  std::string
  serialize_Imputer(const Imputer &model);
  void
  deserialize_Imputer(Imputer &model, const char *in);
  void
  deserialize_Imputer(Imputer &model, std::FILE *in);

  void
  deserialize_Imputer(Imputer &model, std::istream &in);

  void
  deserialize_Imputer(Imputer &model, const std::string &in);

  void
  serialize_Indexer(const TreesIndexer &model, char *out);

  void
  serialize_Indexer(const TreesIndexer &model, std::FILE *out);

  void
  serialize_Indexer(const TreesIndexer &model, std::ostream &out);

  std::string
  serialize_Indexer(const TreesIndexer &model);

  void
  deserialize_Indexer(TreesIndexer &model, const char *in);
  size_t
  determine_serialized_size_combined(
      const iso_forest *model, const ExtIsoForest *model_ext,
      const Imputer *imputer, const TreesIndexer *indexer,
      const size_t size_optional_metadata) noexcept;

  size_t
  determine_serialized_size_combined(
      const char *serialized_model, const char *serialized_model_ext,
      const char *serialized_imputer, const char *serialized_indexer,
      const size_t size_optional_metadata) noexcept;
 
  template <class otype>
  void
  serialize_combined(const IsoForest *model, const ExtIsoForest *model_ext,
                     const Imputer *imputer, const TreesIndexer *indexer,
                     const char *optional_metadata,
                     const size_t size_optional_metadata, otype &out)
  {
    signal_switcher ss;

    auto pos_watermark = set_return_position(out);
  }

  inline void
  serialize_combined(const iso_forest *model, const ExtIsoForest *model_ext,
                     const Imputer *imputer, const TreesIndexer *indexer,
                     const char *optional_metadata,
                     const size_t size_optional_metadata, char *out)
  {
    //validate parameters and check if they are compatible
    bool is_isotree_model, is_compatible, has_combined_objects, has_IsoForest,
        has_ExtIsoForest, has_Imputer, has_Indexer, has_metadata;
    size_t size_metadata;
    inspect_serialized_object(out, is_isotree_model, is_compatible,
                              has_combined_objects, has_IsoForest,
                              has_ExtIsoForest, has_Imputer, has_Indexer,
                              has_metadata, size_metadata);
                              
    if (is_isotree_model && is_compatible && has_combined_objects &&
        has_IsoForest && has_ExtIsoForest && has_Imputer && has_Indexer)
    {
      serialize_IsoForest(*model, out);
      serialize_ExtIsoForest(*model_ext, out);
      serialize_Imputer(*imputer, out);
      serialize_Indexer(*indexer, out);
    }
    else
    {
      throw std::runtime_error("Serialized object is not compatible with "
                               "this version of IsoTree.");
    }

    if (has_metadata)
    {
      std::memcpy(out + size_metadata, optional_metadata,
                  size_optional_metadata);
    }

    return;
  }
  inline void
  serialize_combined(const iso_forest *model, const ExtIsoForest *model_ext,
                     const Imputer *imputer, const TreesIndexer *indexer,
                     const char *optional_metadata,
                     const size_t size_optional_metadata, std::FILE *out)
  {
      //validate parameters and check if they are compatible
    bool is_isotree_model, is_compatible, has_combined_objects, has_IsoForest,
        has_ExtIsoForest, has_Imputer, has_Indexer, has_metadata; 
    size_t size_metadata;
    inspect_serialized_object(out, is_isotree_model, is_compatible,
                              has_combined_objects, has_IsoForest,
                              has_ExtIsoForest, has_Imputer, has_Indexer,
                              has_metadata, size_metadata);
    //validate parameters and check if they are compatible
    if (is_isotree_model && is_compatible && has_combined_objects &&
        has_IsoForest && has_ExtIsoForest && has_Imputer && has_Indexer)
    {
      serialize_IsoForest(*model, out);
      serialize_ExtIsoForest(*model_ext, out);
      serialize_Imputer(*imputer, out);
      serialize_Indexer(*indexer, out);
    }
    else
    {
      throw std::runtime_error("Serialized object is not compatible with "
                               "this version of IsoTree.");
    }   
    if (has_metadata)
    {
      std::memcpy(out + size_metadata, optional_metadata,
                  size_optional_metadata);
    }

    return;
    
  }
  inline void
  serialize_combined(const iso_forest *model, const ExtIsoForest *model_ext,
                     const Imputer *imputer, const TreesIndexer *indexer,
                     const char *optional_metadata,
                     const size_t size_optional_metadata, std::ostream &out)
  {

      if (model != nullptr && model_ext != nullptr && imputer != nullptr &&
          indexer != nullptr)
      {
        serialize_IsoForest(*model, out);
        serialize_ExtIsoForest(*model_ext, out);
        serialize_Imputer(*imputer, out);
        serialize_Indexer(*indexer, out);
      }
      else
      {
        throw std::runtime_error(
            "Error: all four models must be non-null to serialize them "
            "together.");
      } 
      if ( optional_metadata != nullptr && size_optional_metadata > 0)
      {
        out.write(optional_metadata, size_optional_metadata);
      } 
      return;

  }
  inline std::string
  serialize_combined(const iso_forest *model, const ExtIsoForest *model_ext,
                     const Imputer *imputer, const TreesIndexer *indexer,
                     const char *optional_metadata,
                     const size_t size_optional_metadata)
  {
    std::string out;
    if(model != nullptr && model_ext != nullptr && imputer != nullptr && indexer != nullptr)    
    {
      out+=serialize_IsoForest(*model);
      out+=serialize_ExtIsoForest(*model_ext);
      out+=serialize_Imputer(*imputer);
      out+=serialize_Indexer(*indexer);
    }
    else
    {
      throw std::runtime_error("Error: all four models must be non-null to serialize them together.");
    }

    if(optional_metadata != nullptr && size_optional_metadata > 0)
    {
      out.append(optional_metadata, size_optional_metadata);
    }
    return out;

  } 

  
  //c-style
  inline void
  serialize_combined(const char *serialized_model,
                     const char *serialized_model_ext,
                     const char *serialized_imputer,
                     const char *serialized_indexer,
                     const char *optional_metadata,
                     const size_t size_optional_metadata, std::FILE *out)
  {

      fwrite(serialized_model, sizeof(char), strlen(serialized_model), out);    
      fwrite(serialized_model_ext, sizeof(char), strlen(serialized_model_ext), out);
      fwrite(serialized_imputer, sizeof(char), strlen(serialized_imputer), out);
      fwrite(serialized_indexer, sizeof(char), strlen(serialized_indexer), out);
      if (optional_metadata != nullptr && size_optional_metadata > 0)
      {
        fwrite(optional_metadata, sizeof(char), size_optional_metadata, out);
      }
      return;
  }
  inline void
  serialize_combined(const char *serialized_model,
                     const char *serialized_model_ext,
                     const char *serialized_imputer,
                     const char *serialized_indexer,
                     const char *optional_metadata,
                     const size_t size_optional_metadata, std::ostream &out)
  {
      
        out.write(serialized_model, strlen(serialized_model));
        out.write(serialized_model_ext, strlen(serialized_model_ext));
        out.write(serialized_imputer, strlen(serialized_imputer));
        out.write(serialized_indexer, strlen(serialized_indexer));
        if (optional_metadata != nullptr && size_optional_metadata > 0)
        {
          out.write(optional_metadata, size_optional_metadata);
        }
        return; 
  }
  inline std::string
  serialize_combined(const char *serialized_model = nullptr,
                     const char *serialized_model_ext = nullptr,
                     const char *serialized_imputer = nullptr,
                     const char *serialized_indexer = nullptr,
                     const char *optional_metadata = nullptr,
                     const size_t size_optional_metadata = 0)
  {

    std::string ret = "";

    if(serialized_model != nullptr)
      ret += std::string(serialized_model);
    if(serialized_model_ext != nullptr)
      ret += std::string(serialized_model_ext);
    if(serialized_imputer != nullptr)
      ret += std::string(serialized_imputer);
    if(serialized_indexer != nullptr)
      ret += std::string(serialized_indexer);
    if(optional_metadata != nullptr && size_optional_metadata > 0   )
      ret += std::string(optional_metadata);


    

    return ret;
  }

  inline void
  deserialize_combined(std::istream &in, iso_forest *model,
                       ExtIsoForest *model_ext, Imputer *imputer,
                       TreesIndexer *indexer, char *optional_metadata)
  {
    signal_switcher ss;
    //read the watermark
    //auto pos_watermark = get_return_position(in); 

    //read the model
    if ( model != nullptr && model_ext != nullptr && imputer != nullptr &&
         indexer != nullptr )
      { 
        

        //read the optional metadata
        if ( optional_metadata != nullptr  && in.good()  )
        {
          in.read(optional_metadata, sizeof(char) * 4);
        }
         
        

      }
    else
      throw std::runtime_error(
          "One of the model, model_ext, imputer or indexer is null" +         
          std::string(__FILE__) + ":" + std::to_string(__LINE__));    

  }

  inline void
  deserialize_combined(std::FILE *in, iso_forest *model, ExtIsoForest *model_ext,
                       Imputer *imputer, TreesIndexer *indexer,
                       char *optional_metadata)
  {

    //    std::cout << "deserializing combined" << std::endl;
    if ( in == nullptr )
      throw std::runtime_error("Input stream is null");
    if ( model == nullptr )
      throw std::runtime_error("Model pointer is null");  
    if ( model_ext == nullptr )
      throw std::runtime_error("Model ext pointer is null");  
    if ( imputer == nullptr )
      throw std::runtime_error("Imputer pointer is null");
    if ( indexer == nullptr )
      throw std::runtime_error("Indexer pointer is null");
      
    signal_switcher ss;
    //read the watermark
    //auto pos_watermark = get_return_position(in);
    //std::cout << "pos watermark: " << pos_watermark << std::endl;
    //read the model
    deserialize_IsoForest(*model, in);
    //std::cout << "pos watermark: " << pos_watermark << std::endl;
    //read the model ext
    deserialize_ExtIsoForest(*model_ext, in);
    //std::cout << "pos watermark: " << pos_watermark << std::endl;
    //read the imputer
    deserialize_Imputer(*imputer, in);
    //std::cout << "pos watermark: " << pos_watermark << std::endl;
    //read the indexer
    deserialize_Indexer(*indexer, "");
    //std::cout << "pos watermark: " << pos_watermark << std::endl;
    //read the optional metadata
    if ( optional_metadata != nullptr )
      {
        //std::cout << "reading optional metadata" << std::endl;
        //std::cout << "pos watermark: " << pos_watermark << std::endl;
        //std::cout << "pos: " << pos << std::endl;
        //std::cout << "size: " << size << std::endl;


        }

  }

  inline void
  deserialize_combined(const char *in, iso_forest *model = nullptr,
                       ExtIsoForest *model_ext = nullptr, Imputer *imputer = nullptr,
                       TreesIndexer *indexer = nullptr,
                       char *optional_metadata = nullptr);



  
  //serialize_Imputer
  inline void serialize_Imputer(const Imputer &imputer, std::ostream &out)
  {
    provallo::signal_switcher ss;
    //write the watermark
    //auto pos_watermark = get_return_position(out);
    //write the model

    out<<"Imputer tree" <<std::to_string(imputer.imputer_tree.size())<<std::endl;

    for ( auto it = imputer.imputer_tree.begin(); it != imputer.imputer_tree.end();
          ++it )
      {
          for(auto  it2=it->begin();it2!=it->end();++it2)
            {
              for ( auto it3 = it2->cat_sum.begin(); it3 != it2->cat_sum.end(); ++it3 )
                out<<(*it3) <<std::endl;  
              for ( auto it3 = it2->cat_weight.begin(); it3 != it2->cat_weight.end(); ++it3 )
                out<<(*it3) <<std::endl;
              for ( auto it3 = it2->num_sum.begin(); it3 != it2->num_sum.end(); ++it3 )
                out<<(*it3) <<std::endl;
              for ( auto it3 = it2->num_weight.begin(); it3 != it2->num_weight.end(); ++it3 )
                out<<(*it3) <<std::endl;
            }//it2
       }//it
  }//serialize_Imputer

  //serialize Indexer

  inline void serialize_Indexer(const TreesIndexer &indexer, std::ostream &out)
  {
    signal_switcher ss;
    //write the watermark
    //auto pos_watermark = get_return_position(out);
    //write the model
    out<<"indices"<<" : " <<indexer.indices.size() << std::endl;
    size_t index=0,index2=0;
    for ( auto it =  indexer.indices.begin(); it != indexer.indices.end(); it++)
    {
          out <<std::string("indexer : ") + std::to_string(index++) << std::endl;
          //terminal nodes
          out <<std::string("n_terminal_nodes : ") + std::to_string((*it).n_terminal) << std::endl;

          //node_distances
          out <<std::string("node_distances : ")<< std::endl; ;

          for ( auto itt = (*it).node_distances.begin(); itt != (*it).node_distances.end(); itt++)
          {
              out << std::to_string(++index2) << ":" << std::to_string(*itt) << std::endl;
          } 

          out<<std::string("node_depths : ")<<std::endl;
          index2=0;
          //node_depths
          for ( auto itt = (*it).node_depths.begin(); itt != (*it).node_depths.end(); itt++)
          {
             out << std::to_string(++index2) << ":" << std::to_string(*itt) << std::endl;
          } 
          //reference_points
          out<<std::string("reference_points : ")<<std::endl ;

          for (  auto itt = (*it).reference_points.begin(); itt != (*it).reference_points.end(); ++itt)
          {
            out << std::to_string(++index2) << ":" << std::to_string(*itt) << std::endl;          
          }
    
          //reference_indptr:
          for ( auto itt = (*it).reference_indptr.begin(); itt != (*it).reference_indptr.end(); ++itt)
          {
            out <<std::string("reference_indptr : ") + std::to_string(*itt) << std::endl;
          }
          //reference_mapping
          for ( auto itt = (*it).reference_mapping.begin(); itt != (*it).reference_mapping.end(); ++itt)
          {
            out <<std::string("reference_mapping : ") + std::to_string(*itt) << std::endl;
          }
          //terminal_node_mappings
          for ( auto itt = (*it).terminal_node_mappings.begin(); itt != (*it).terminal_node_mappings.end(); ++itt)
          {
            out <<std::string("terminal_node_mappings : ") + std::to_string(*itt) << std::endl;
          }

    }
    out<<std::endl;
    //write the optional metadata
    //write the watermark

    //write the optional metadata
    //write the watermark
    //auto pos = get_return_position(out);
    //auto size = pos - pos_watermark;
    //out.seekp(pos_watermark);
    //out.write(reinterpret_cast<const char *>(&size), sizeof(size));
    //out.seekp(pos);
    return;
  }

  inline void serialize_Imputer(const Imputer &imputer, std::FILE *out)
  {
    signal_switcher ss;
    //write the watermark
    //auto pos_watermark = get_return_position(out);
    //write the model
 
    fwrite(&imputer, sizeof(Imputer), 1, out);

     //write the optional metadata
    //write the watermark
    //auto pos = get_return_position(out);
    //auto size = pos - pos_watermark;
    //out.seekp(pos_watermark);
    //out.write(reinterpret_cast<const char *>(&size), sizeof(size));
    //out.seekp(pos);
    return;
  } 
  
  class isolation_forest /*: public classifier*/
  {
  public:
    
    //isolation_forest() = default;

    ~isolation_forest() {
     }
    isolation_forest(size_t ndim, size_t ntry, provallo::CoefType coef_type,
                     bool coef_by_prop, bool with_replacement,
                     bool weight_as_sample, size_t sample_size, size_t ntrees,
                     size_t max_depth, size_t ncols_per_tree, bool limit_depth,
                     bool penalize_range, bool standardize_datam,
                     provallo::ScoringMetric scoring_metric, bool fast_bratio,
                     bool weigh_by_kurt, real_t prob_pick_by_gain_pl,
                     real_t prob_pick_by_gain_avg,
                     real_t prob_pick_by_full_gain, real_t prob_pick_by_dens,
                     real_t prob_pick_col_by_range,
                     real_t prob_pick_col_by_var, real_t prob_pick_col_by_kurt,
                     real_t min_gain, provallo::MissingAction missing_action,
                     provallo::CategSplit cat_split_type,
                     provallo::NewCategAction new_cat_action, bool all_perm,
                     bool build_imputer, size_t min_imp_obs,
                     provallo::UseDepthImp depth_imp,
                     provallo::WeighImpRows weigh_imp_rows,
                     uint64_t random_seed, int nthreads);

    static void
    predict_iforest(real_t numeric_data[], int categ_data[], bool is_col_major,
                    size_t ld_numeric, size_t ld_categ,
                    real_t Xc[],
                    sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                    real_t Xr[],
                    sparse_ix Xr_ind[], sparse_ix Xr_indptr[], size_t nrows,
                    int nthreads, bool standardize,
                    provallo::iso_forest *model_outputs,
                    provallo::ext_iso_forest *model_outputs_ext,
                    real_t output_depths[], sparse_ix tree_num[],
                    real_t per_tree_depths[], provallo::TreesIndexer *indexer);

    static int
    fit_iforest(provallo::iso_forest *model_outputs,
                provallo::ExtIsoForest *model_outputs_ext,
                real_t numeric_data[],
                size_t ncols_numeric, int categ_data[], size_t ncols_categ,
                int ncat[],
                real_t Xc[],
                sparse_ix Xc_ind[], sparse_ix Xc_indptr[], size_t ndim,
                size_t ntry, provallo::CoefType coef_type, bool coef_by_prop,
                real_t sample_weights[],
                bool with_replacement, bool weight_as_sample, size_t nrows,
                size_t sample_size, size_t ntrees, size_t max_depth,
                size_t ncols_per_tree, bool limit_depth, bool penalize_range,
                bool standardize_data, provallo::ScoringMetric scoring_metric,
                bool fast_bratio, bool standardize_dist, real_t tmat[],
                real_t output_depths[], bool standardize_depth,
                real_t col_weights[],
                bool weigh_by_kurt, real_t prob_pick_by_gain_pl,
                real_t prob_pick_by_gain_avg, real_t prob_pick_by_full_gain,
                real_t prob_pick_by_dens, real_t prob_pick_col_by_range,
                real_t prob_pick_col_by_var, real_t prob_pick_col_by_kurt,
                real_t min_gain, provallo::MissingAction missing_action,
                provallo::CategSplit cat_split_type,
                provallo::NewCategAction new_cat_action, bool all_perm,
                Imputer *imputer, size_t min_imp_obs,
                provallo::UseDepthImp depth_imp,
                provallo::WeighImpRows weigh_imp_rows, bool impute_at_fit,
                uint64_t random_seed, bool use_long_real_t, int nthreads);
    
    
    isolation_forest(size_t ndim =3, size_t ntrees = 100,
                     bool build_imputer = true ,int nthreads=-1);


    bool fitted() const{
      return is_fitted;
    }
    
    void
    fit(real_t X[], size_t nrows, size_t ncols);


    inline void fit ( matrix<real_t> &X  )
    {
      fit(X.data(), X.size1(), X.size2());
    }

     

    // set numeric and categorical data and weights for fitting the model
    // using attribute_information and attribute_weights:
    // attribute_information: 0 for numeric, 1 for categorical
    // attribute_weights: 0 for uniform, 1 for weighted
    /*
    void fit (matrix<real_t> numeric , matrix<int> categorical,
              matrix<real_t> weights)
              {

                real_t *numeric_data = &numeric.data()[0];
                int *categorical_data = &categorical.data()[0];
                real_t *weights_data = &weights.data()[0];

                size_t nrows = numeric.rows();

                size_t ncols_numeric = numeric.cols();
                size_t ncols_categ = categorical.cols();

                int *ncat = new int[ncols_categ];
                for (size_t i = 0; i < ncols_categ; i++)
                  ncat[i] =categorical (0, i);
                
                fit(numeric_data, ncols_numeric, nrows, categorical_data,
                    ncols_categ, ncat, weights_data, NULL); // no col_weights
                delete[] ncat;
              }
    */
                

    void
    fit(real_t numeric_data[], size_t ncols_numeric, size_t nrows,
        int categ_data[], size_t ncols_categ, int ncat[],
        real_t sample_weights[], real_t col_weights[]);

    void
    fit(real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
        size_t ncols_numeric, size_t nrows, int categ_data[],
        size_t ncols_categ, int ncat[], real_t sample_weights[],
        real_t col_weights[]);

    std::vector<real_t>
    predict(real_t X[], size_t nrows, bool standardize);

    std::vector<real_t>
    predict(const std::string &); // comma delimited<--><real_t>

    void
    predict(real_t numeric_data[], int categ_data[], bool is_col_major,
            size_t nrows, size_t ld_numeric, size_t ld_categ, bool standardize,
            real_t output_depths[], sparse_ix tree_num[],
            real_t per_tree_depths[]);

    void
    predict(real_t X_sparse[], sparse_ix X_ind[], sparse_ix X_indptr[],
            bool is_csc, int categ_data[], bool is_col_major, size_t ld_categ,
            size_t nrows, bool standardize, real_t output_depths[],
            sparse_ix tree_num[],
            real_t per_tree_depths[]);


    void predict (matrix<real_t> &X, std::vector<real_t> &out)
    {
      out = predict(X.data(), X.rows(), X.cols());
    }

    std::vector<real_t> predict (matrix<real_t> &X)
    {
      return predict(X.data(), X.rows(), X.cols());
    } 
    std::vector<real_t> predict ( matrix<real_t>& X, matrix<uint32_t>& cat)
    {
        // TODO: check if cat is categorical
        //if (cat.cols() == 0)
          //return predict(X);
        //else
           //return predict( (real_t*)X.data(), X.rows(), X.cols(), cat.data(), cat.rows(), cat.cols()); 
          //return predict( X.data(), X.rows(), X.cols(), cat.data(), cat.rows(), cat.cols());    }
        
        //for now ignore:
        matrix<real_t> y(cat.rows(), cat.cols());

        for ( size_t i = 0; i < cat.rows(); i++) {
          for ( size_t j = 0; j < cat.cols(); j++)
            y(i,j) = (real_t) cat(i,j);
        }
        y =  X+y;

        return predict(y.array(), y.rows(),true );
    }

    std::vector<real_t>
    predict_distance(real_t X[], size_t nrows, bool as_kernel,
                     bool assume_full_distr, bool standardize,
                     bool triangular);

    void
    predict_distance(real_t numeric_data[], int categ_data[], size_t nrows,
                     bool as_kernel, bool assume_full_distr, bool standardize,
                     bool triangular, real_t dist_matrix[]);
    //sparse_ix version. 
    //TODO: check if it works insead of int Xc_indptr[] and Xc_ind[]
    void
    predict_distance(real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                     int categ_data[], size_t nrows, bool as_kernel,
                     bool assume_full_distr, bool standardize, bool triangular,
                     real_t dist_matrix[]);

    void
    impute(real_t X[], size_t nrows);

    void
    impute(real_t numeric_data[], int categ_data[], bool is_col_major,
           size_t nrows);

    void
    impute(real_t Xr[], sparse_ix Xr_ind[], sparse_ix Xr_indptr[],
           int categ_data[], bool is_col_major, size_t nrows);

    void
    build_indexer(const bool with_distances);

    void
    set_as_reference_points(real_t Xc[], sparse_ix Xc_ind[],
                            sparse_ix Xc_indptr[],
                            int categ_data[], size_t nrows,
                            const bool with_distances);

    void
    set_as_reference_points(real_t numeric_data[], int categ_data[],
                            bool is_col_major, size_t nrows, size_t ld_numeric,
                            size_t ld_categ, const bool with_distances);

    size_t
    get_num_reference_points() const noexcept;

    void
    predict_distance_to_ref_points(real_t numeric_data[], int categ_data[],
                                   real_t Xc[], sparse_ix Xc_ind[], sparse_ix Xc_indptr[],
                                   size_t nrows, bool is_col_major,
                                   size_t ld_numeric, size_t ld_categ,
                                   bool as_kernel, bool standardize,
                                   real_t dist_matrix[]);

    void
    serialize(std::FILE *out) const;

    void
    serialize(std::ostream &out) const;

    static isolation_forest
    deserialize(std::FILE *inp, int nthreads);

    static isolation_forest
    deserialize(std::istream &inp, int nthreads);

    friend std::ostream &
    operator<<(std::ostream &ost, const isolation_forest &model);

    friend std::istream &
    operator>>(std::istream &ist, isolation_forest &model);

    provallo::iso_forest &
    get_model();

    provallo::ExtIsoForest &
    get_model_ext();

    provallo::Imputer &
    get_imputer();

    provallo::TreesIndexer &
    get_indexer();

    void
    check_nthreads();

    size_t
    get_ntrees() const;

    bool
    check_can_predict_per_tree() const;

  public:



    size_t ndim = 3;
    size_t ntry = 1;

    provallo::CoefType coef_type = Uniform;
    bool coef_by_prop = false;
    
    bool with_replacement = false;
    bool weight_as_sample = true;
    size_t sample_size = 0;
    size_t ntrees = 500;
    size_t max_depth = 0;
    size_t ncols_per_tree = 0;
    bool limit_depth = true;
    bool penalize_range = false;
    bool standardize_data = true;
    enum provallo::ScoringMetric scoring_metric = Depth;
    bool fast_bratio = true;
    bool weigh_by_kurt = false;
    real_t prob_pick_by_gain_pl = 0.;
    real_t prob_pick_by_gain_avg = 0.;
    real_t prob_pick_by_full_gain = 0.;
    real_t prob_pick_by_dens = 0.;
    real_t prob_pick_col_by_range = 0.;
    real_t prob_pick_col_by_var = 0.;
    real_t prob_pick_col_by_kurt = 0.;
    real_t min_gain = 0.;
    enum provallo::MissingAction missing_action = Impute;

    enum provallo::CategSplit cat_split_type = SubSet;
    enum provallo::NewCategAction new_cat_action = Weighted;

    bool all_perm = false;
    bool build_imputer = false;
    size_t min_imp_obs = 3;


    enum provallo::UseDepthImp depth_imp = Higher;

    enum provallo::WeighImpRows weigh_imp_rows = Inverse;
    
    uint64_t random_seed = 1;
    
    int nthreads = -1;
    provallo::iso_forest model;
    provallo::ExtIsoForest model_ext;
    provallo::Imputer imputer;
    provallo::TreesIndexer indexer;

 


  private:
    bool is_fitted = false;

    void
    override_previous_fit();
    void
    check_params();
    void
    check_is_fitted() const;
   
    template <class otype>
    void
    serialize_template(otype &out) const;
    template <class itype>
    static isolation_forest
    deserialize_template(itype &inp, int nthreads);
    static void
    remap_terminal_trees(IsoForest *model_outputs,
                         ExtIsoForest *model_outputs_ext,
                         prediction_data &data, sparse_ix *tree_num,
                         int nthreads);




  
  };

  template <typename lreal_t_safe = real_t>
  class column_sampler
  {
  public:
    std::vector<size_t> col_indices;
    std::vector<real_t> tree_weights;
    size_t curr_pos;
    size_t curr_col;
    size_t last_given;
    size_t n_cols;
    size_t tree_levels;
    size_t offset;
    size_t n_dropped;
    void
    initialize(real_t weights[], size_t n_cols);
    void
    initialize(size_t n_cols);
    void
    drop_weights();
    void
    leave_m_cols(size_t m, RNG_engine &rnd_generator);
    bool
    sample_col(size_t &col, RNG_engine &rnd_generator);
    void
    prepare_full_pass(); /* when passing through all columns */
    bool
    sample_col(size_t &col); /* when passing through all columns */
    void
    drop_col(size_t col, size_t nobs_left);
    void
    drop_col(size_t col);
    void
    drop_from_tail(size_t col);
    void
    shuffle_remainder(RNG_engine &rnd_generator);
    bool
    has_weights();
    size_t
    get_remaining_cols();
    void
    get_array_remaining_cols(std::vector<size_t> &);
    template <class other_t>
    column_sampler &
    operator=(const column_sampler<other_t> &other);
    column_sampler() = default;
  };

  std::ostream &
  operator<<(std::ostream &ost, const isolation_forest &model);

  std::istream &
  operator>>(std::istream &ist, isolation_forest &model);

  template <class lreal_t_safe = long real_t, class real_t_ = real_t>
  class SingleNodeColumnSampler
  {
  public:
    real_t *weights_orig;
    std::vector<bool> inifinite_weights;
    lreal_t_safe cumw;
    size_t n_inf;
    size_t *col_indices;
    size_t curr_pos;
    bool using_tree;

    bool backup_weights;
    std::vector<real_t> weights_own;
    size_t n_left;

    std::vector<real_t> tree_weights;
    size_t offset;
    size_t tree_levels;
    std::vector<real_t> used_weights;
    std::vector<size_t> mapped_indices;
    std::vector<size_t> mapped_inf_indices;

    bool
    initialize(real_t *weights, std::vector<size_t> *col_indices,
               size_t curr_pos, size_t n_sample, bool backup_weights);

    bool
    sample_col(size_t &col_chosen, RNG_engine &rnd_generator);

    void
    backup(SingleNodeColumnSampler<lreal_t_safe, real_t> &other,
           size_t ncols_tot);

    void
    restore(const SingleNodeColumnSampler<lreal_t_safe, real_t> &other);
  };
  template <class lreal_t_safe, class real_t_>
  class density_estimator
  {
  public:
    std::vector<lreal_t_safe> multipliers;
    real_t xmin;
    real_t xmax;
    std::vector<size_t> counts;
    int n_present;
    int n_left;
    std::vector<real_t> box_low;
    std::vector<real_t> box_high;
    std::vector<real_t> queue_box;
    bool fast_bratio;
    std::vector<lreal_t_safe> ranges;
    std::vector<int> ncat;
    std::vector<int> queue_ncat;
    std::vector<int> ncat_orig;
    std::vector<real_t> vals_ext_box;
    std::vector<real_t> queue_ext_box;

    void
    initialize(size_t max_depth, int max_categ, bool reserve_counts,
               ScoringMetric scoring_metric);
    template <class InputData>
#ifndef _FOR_R
    [[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
#endif
    void
    initialize_bdens(const InputData &input_data,
                     const ModelParams &model_params,
                     std::vector<size_t> &ix_arr,
                     column_sampler<lreal_t_safe> &col_sampler);
    template <class InputData>
    void
    initialize_bdens_ext(const InputData &input_data,
                         const ModelParams &model_params,
                         std::vector<size_t> &ix_arr,
                         column_sampler<lreal_t_safe> &col_sampler,
                         bool col_sampler_is_fresh);
#ifndef _FOR_R
    [[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
#endif
    void
    push_density(real_t xmin, real_t xmax, real_t split_point);
    void
    push_density(size_t counts[], int ncat);
    void
    push_density(int n_left, int n_present);
    void
    push_density(int n_present);
    void
    push_density();
    void
    push_adj(real_t xmin, real_t xmax, real_t split_point,
             real_t pct_tree_left, ScoringMetric scoring_metric);
    void
    push_adj(signed char *categ_present, size_t *counts, int ncat,
             ScoringMetric scoring_metric);
    void
    push_adj(size_t *counts, int ncat, int chosen_cat,
             ScoringMetric scoring_metric);
    void
    push_adj(real_t pct_tree_left, ScoringMetric scoring_metric);
    void
    push_bdens(real_t split_point, size_t col);
    void
    push_bdens(int ncat_branch_left, size_t col);
    void
    push_bdens(const std::vector<signed char> &cat_split, size_t col);
    void
    push_bdens(const std::vector<char> &cat_split, size_t col); // convert from trees

#ifndef _FOR_R
    [[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
#endif
    void
    push_bdens_fast_route(real_t split_point, size_t col);
    void
    push_bdens_internal(real_t split_point, size_t col);
#ifndef _FOR_R
    [[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
#endif
    void
    push_bdens_fast_route(int ncat_branch_left, size_t col);
    void
    push_bdens_internal(int ncat_branch_left, size_t col);
#ifndef _FOR_R
    [[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
#endif
    void
    push_bdens_fast_route(const std::vector<signed char> &cat_split,
                          size_t col);
    void
    push_bdens_internal(const std::vector<signed char> &cat_split,
                        size_t col);
#ifndef _FOR_R
    [[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
#endif
    void
    push_bdens_ext(const IsoHPlane &hplane, const ModelParams &model_params);
    void
    pop();
    void
    pop_right();
    void
    pop_bdens(size_t col);
    void
    pop_bdens_right(size_t col);
    void
    pop_bdens_cat(size_t col);
    void
    pop_bdens_cat_right(size_t col);
    void
    pop_bdens_fast_route(size_t col);
    void
    pop_bdens_internal(size_t col);
    void
    pop_bdens_right_fast_route(size_t col);
    void
    pop_bdens_right_internal(size_t col);
    void
    pop_bdens_cat_fast_route(size_t col);
    void
    pop_bdens_cat_internal(size_t col);
    void
    pop_bdens_cat_right_fast_route(size_t col);
    void
    pop_bdens_cat_right_internal(size_t col);
    void
    pop_bdens_ext();
    void
    pop_bdens_ext_right();
#ifndef _FOR_R
    [[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
#endif
    real_t
    calc_density(lreal_t_safe remainder, size_t sample_size);
    lreal_t_safe
    calc_adj_depth();
    real_t
    calc_adj_density();
#ifndef _FOR_R
    [[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
#endif
    lreal_t_safe
    calc_bratio_log();
#ifndef _FOR_R
    [[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
#endif
    lreal_t_safe
    calc_bratio_inv_log();
#ifndef _FOR_R
    [[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
#endif
    real_t
    calc_bratio();
#ifndef _FOR_R
    [[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
#endif
    real_t
    calc_bdens(lreal_t_safe remainder, size_t sample_size);
#ifndef _FOR_R
    [[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
#endif
    real_t
    calc_bdens2(lreal_t_safe remainder, size_t sample_size);
#ifndef _FOR_R
    [[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
#endif
    lreal_t_safe
    calc_bratio_log_ext();
#ifndef _FOR_R
    [[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
#endif
    real_t
    calc_bratio_ext();
#ifndef _FOR_R
    [[gnu::optimize("no-trapping-math"), gnu::optimize("no-math-errno")]]
#endif
    real_t
    calc_bdens_ext(lreal_t_safe remainder, size_t sample_size);
    void
    save_range(real_t xmin, real_t xmax);
    void
    restore_range(real_t &xmin, real_t &xmax);
    void
    save_counts(size_t *cat_counts, int ncat);
    void
    save_n_present_and_left(signed char *split_left, int ncat);
    void
    save_n_present(size_t *cat_counts, int ncat);
  };

  template <class ImputedData, class lreal_t_safe = long real_t,
            class real_t_ =real_t>
  struct WorkerMemory
  {
    std::vector<size_t> ix_arr;
    std::vector<size_t> ix_all;
    RNG_engine rnd_generator;
    UniformUnitInterval rbin;
    size_t st;
    size_t end;
    size_t st_NA;
    size_t end_NA;
    size_t split_ix;
    hashed_map<size_t, real_t> weights_map;
    std::vector<real_t> weights_arr; /* when not ignoring NAs and when using weights as dty */
    bool changed_weights;            /* when using 'missing_action'='Divide' or density weights */
    real_t xmin;
    real_t xmax;
    size_t npresent; /* 'npresent' and 'ncols_tried' are used interchangeable and for unrelated things */
    bool unsplittable;
    std::vector<bool> is_repeated;
    std::vector<signed char> categs;
    size_t ncols_tried; /* 'npresent' and 'ncols_tried' are used interchangeable and for unrelated things */
    int ncat_tried;
    std::vector<real_t> btree_weights;        /* only when using weights for sampling */
    column_sampler<lreal_t_safe> col_sampler; /* columns can get eliminated, keep a copy for each thread */
    SingleNodeColumnSampler<lreal_t_safe, real_t_> node_col_sampler;
    SingleNodeColumnSampler<lreal_t_safe, real_t_> node_col_sampler_backup;

    /* for split criterion */
    std::vector<real_t> buffer_dbl;
    std::vector<size_t> buffer_szt;
    std::vector<signed char> buffer_chr;
    real_t prob_split_type;
    ColCriterion col_criterion;
    GainCriterion criterion;
    real_t this_gain;
    real_t this_split_point;
    int this_categ;
    std::vector<signed char> this_split_categ;
    bool determine_split;
    std::vector<real_t> imputed_x_buffer;
    real_t saved_xmedian;
    real_t best_xmedian;
    int saved_cat_mode;
    int best_cat_mode;
    std::vector<size_t> col_indices; /* only for full gain calculation */

    /* for weighted column choices */
    std::vector<real_t> node_col_weights;
    std::vector<real_t> saved_stat1;
    std::vector<real_t> saved_stat2;
    bool has_saved_stats;
    real_t *tree_kurtoses; /* only when mixing 'weight_by_kurt' with 'prob_pick_col*' */

    /* for the extended model */
    size_t ntry;
    size_t ntaken;
    size_t ntaken_best;
    size_t ntried;
    bool try_all;
    size_t col_chosen; /* also used as placeholder in the single-variable model */
    ColType col_type;
    real_t ext_sd;
    std::vector<real_t> comb_val;
    std::vector<size_t> col_take;
    std::vector<ColType> col_take_type;
    std::vector<real_t> ext_offset;
    std::vector<real_t> ext_coef;
    std::vector<real_t> ext_mean;
    std::vector<real_t> ext_fill_val;
    std::vector<real_t> ext_fill_new;
    std::vector<int> chosen_cat;
    std::vector<std::vector<real_t>> ext_cat_coef;
    std::uniform_real_distribution<real_t> coef_unif;
    std::normal_distribution<real_t> coef_norm;
    std::vector<real_t> sample_weights; /* when using weights and split criterion */

    /* for similarity/distance calculations */
    std::vector<real_t> tmat_sep;

    /* when calculating average depth on-the-fly */
    std::vector<real_t> row_depths;

    /* when imputing NAs on-the-fly */
    std::vector<ImputedData> impute_vec;
    hashed_map<size_t, ImputedData> impute_map;

    /* for non-depth scoring metric */
    density_estimator<lreal_t_safe, real_t> density_calculator;
  
  };
  void
  check_interrupt_switch(signal_switcher &ss);
  inline bool
  has_long_real_t()
  {
    return sizeof(real_t) < sizeof(long real_t);
  }
  int return_EXIT_SUCCESS();
  int return_EXIT_FAILURE();

  template <typename lreal_t_safe>
  inline void
  provallo::column_sampler<lreal_t_safe>::initialize(real_t weights[],
                                                     size_t n_cols)
  {

    this->n_cols = n_cols;
    this->tree_levels = log2ceil(n_cols);
    if (this->tree_weights.empty())
      this->tree_weights.resize(pow2(this->tree_levels + 1), 0);
    else
    {
      if (this->tree_weights.size() != pow2(this->tree_levels + 1))
        this->tree_weights.resize(this->tree_levels);
      std::fill(this->tree_weights.begin(), this->tree_weights.end(),
                0.);
    }

    /* compute sums for the tree leaves at each node */
    this->offset = pow2(this->tree_levels) - 1;
    for (size_t ix = 0; ix < this->n_cols; ix++)
      this->tree_weights[ix + this->offset] = std::fmax(0., weights[ix]);
    for (size_t ix = this->tree_weights.size() - 1; ix > 0; ix--)
      this->tree_weights[ix_parent(ix)] += this->tree_weights[ix];

    if (unlikely(
            std::isnan(this->tree_weights[0]) || this->tree_weights[0] <= 0))
    {
      this->drop_weights();
    }

    this->n_dropped = 0;
  }

  template <typename lreal_t_safe>
  inline void
  provallo::column_sampler<lreal_t_safe>::initialize(size_t n_cols)
  {
    if (!this->has_weights())
    {
      this->n_cols = n_cols;
      this->curr_pos = n_cols;
      this->col_indices.resize(n_cols);
      std::iota(this->col_indices.begin(), this->col_indices.end(),
                (size_t)0);
    }
  }

  template <typename lreal_t_safe>
  inline void
  provallo::column_sampler<lreal_t_safe>::drop_weights()
  {

    this->tree_weights.clear();
    this->tree_weights.shrink_to_fit();
    this->initialize(n_cols);
    this->n_dropped = 0;
  }

  template <typename lreal_t_safe>
  inline void
  provallo::column_sampler<lreal_t_safe>::leave_m_cols(size_t m,
                                                       RNG_engine &rnd_generator)
  {
    if (m == 0 || m >= this->n_cols)
      return;

    if (!this->has_weights())
    {
      size_t chosen;
      if (m <= this->n_cols / 4)
      {
        for (this->curr_pos = 0; this->curr_pos < m; this->curr_pos++)
        {
          chosen = std::uniform_int_distribution<size_t>(
              0, this->n_cols - this->curr_pos - 1)(rnd_generator);
          std::swap(this->col_indices[this->curr_pos + chosen],
                    this->col_indices[this->curr_pos]);
        }
      }

      else if ((lreal_t_safe)m >= (lreal_t_safe)(3. / 4.) * (lreal_t_safe)this->n_cols)
      {
        for (this->curr_pos = this->n_cols - 1;
             this->curr_pos > this->n_cols - m; this->curr_pos--)
        {
          chosen = std::uniform_int_distribution<size_t>(
              0, this->curr_pos)(rnd_generator);
          std::swap(this->col_indices[chosen],
                    this->col_indices[this->curr_pos]);
        }
        this->curr_pos = m;
      }

      else
      {
        std::shuffle(this->col_indices.begin(),
                     this->col_indices.end(), rnd_generator);
        this->curr_pos = m;
      }
    }

    else
    {
      std::vector<real_t> curr_weights = this->tree_weights;
      std::fill(this->tree_weights.begin(), this->tree_weights.end(),
                0.);
      real_t rnd_subrange, w_left;
      real_t curr_subrange;
      size_t curr_ix;

      for (size_t col = 0; col < m; col++)
      {
        curr_ix = 0;
        curr_subrange = curr_weights[0];
        if (curr_subrange <= 0)
        {
          if (col == 0)
          {
            this->drop_weights();
            return;
          }

          else
          {
            m = col;
            goto rebuild_tree;
          }
        }

        for (size_t lev = 0; lev < this->tree_levels; lev++)
        {
          rnd_subrange = std::uniform_real_distribution<real_t>(
              0., curr_subrange)(rnd_generator);
          w_left = curr_weights[ix_child(curr_ix)];
          curr_ix = ix_child(curr_ix) + (rnd_subrange >= w_left);
          curr_subrange = curr_weights[curr_ix];
        }

        this->tree_weights[curr_ix] = curr_weights[curr_ix];

        /* now remove the weight of the chosen element */
        curr_weights[curr_ix] = 0;
        for (size_t lev = 0; lev < this->tree_levels; lev++)
        {
          curr_ix = ix_parent(curr_ix);
          curr_weights[curr_ix] = curr_weights[ix_child(curr_ix)] + curr_weights[ix_child(curr_ix) + 1];
        }
      }

      /* rebuild the tree after getting new weights */
    rebuild_tree:
      for (size_t ix = this->tree_weights.size() - 1; ix > 0; ix--)
        this->tree_weights[ix_parent(ix)] += this->tree_weights[ix];

      this->n_dropped = this->n_cols - m;
    }
  }

  template <typename lreal_t_safe>
  inline bool
  provallo::column_sampler<lreal_t_safe>::sample_col(size_t &col,
                                                     RNG_engine &rnd_generator)
  {
    if (!this->has_weights())
    {
      switch (this->curr_pos)
      {
      case 0:
        return false;
      case 1:
      {
        this->last_given = 0;
        col = this->col_indices[0];
        return true;
      }
      default:
      {
        this->last_given = std::uniform_int_distribution<size_t>(
            0, this->curr_pos - 1)(rnd_generator);
        col = this->col_indices[this->last_given];
        return true;
      }
      }
    }

    else
    {
      /* TODO: here could instead generate only 1 random number from zero to the full weight,
       and then subtract from it as it goes down every level. Would have less precision
       but should still work fine. */
      size_t curr_ix = 0;
      real_t rnd_subrange, w_left;
      real_t curr_subrange = this->tree_weights[0];
      if (curr_subrange <= 0)
        return false;

      for (size_t lev = 0; lev < tree_levels; lev++)
      {
        rnd_subrange = std::uniform_real_distribution<real_t>(
            0., curr_subrange)(rnd_generator);
        w_left = this->tree_weights[ix_child(curr_ix)];
        curr_ix = ix_child(curr_ix) + (rnd_subrange >= w_left);
        curr_subrange = this->tree_weights[curr_ix];
      }

      col = curr_ix - this->offset;
      return true;
    }
  }

  template <typename lreal_t_safe>
  inline void
  provallo::column_sampler<lreal_t_safe>::prepare_full_pass()
  {
    this->curr_col = 0;

    if (this->has_weights())
    {
      if (this->col_indices.size() < this->n_cols)
        this->col_indices.resize(this->n_cols);
      this->curr_pos = 0;
      for (size_t col = 0; col < this->n_cols; col++)
      {
        if (this->tree_weights[col + this->offset] > 0)
          this->col_indices[this->curr_pos++] = col;
      }
    }
  }

  template <typename lreal_t_safe>
  inline bool
  provallo::column_sampler<lreal_t_safe>::sample_col(size_t &col)
  {
    if (this->curr_pos == this->curr_col || this->curr_pos == 0)
      return false;
    this->last_given = this->curr_col;
    col = this->col_indices[this->curr_col++];
    return true;
  }

  template <typename lreal_t_safe>
  inline void
  provallo::column_sampler<lreal_t_safe>::drop_col(size_t col,
                                                   size_t nobs_left)
  {

    if (!this->has_weights())
    {
      if (this->col_indices[this->last_given] == col)
      {
        std::swap(this->col_indices[this->last_given],
                  this->col_indices[--this->curr_pos]);
      }

      else if (this->curr_pos > 4 * nobs_left)
      {
        return;
      }

      else
      {
        for (size_t ix = 0; ix < this->curr_pos; ix++)
        {
          if (this->col_indices[ix] == col)
          {
            std::swap(this->col_indices[ix],
                      this->col_indices[--this->curr_pos]);
            break;
          }
        }
      }

      if (this->curr_col)
        this->curr_col--;
    }

    else
    {
      this->n_dropped++;
      size_t curr_ix = col + this->offset;
      this->tree_weights[curr_ix] = 0.;
      for (size_t lev = 0; lev < this->tree_levels; lev++)
      {
        curr_ix = ix_parent(curr_ix);
        this->tree_weights[curr_ix] =
            this->tree_weights[ix_child(curr_ix)] + this->tree_weights[ix_child(curr_ix) + 1];
      }
    }
  }

  template <typename lreal_t_safe>
  inline void
  provallo::column_sampler<lreal_t_safe>::drop_col(size_t col)
  {
    this->drop_col(col, SIZE_MAX);
  }

  template <typename lreal_t_safe>
  inline void
  provallo::column_sampler<lreal_t_safe>::drop_from_tail(size_t col)
  {
    std::swap(this->col_indices[col], this->col_indices[--this->curr_pos]);
  }

  template <typename lreal_t_safe>
  inline void
  provallo::column_sampler<lreal_t_safe>::shuffle_remainder(
      RNG_engine &rnd_generator)
  {
    if (!this->has_weights())
    {
      this->prepare_full_pass();
      std::shuffle(this->col_indices.begin(),
                   this->col_indices.begin() + this->curr_pos,
                   rnd_generator);
    }
    else
    {
      if (this->tree_weights[0] <= 0)
        return;
      std::vector<real_t> curr_weights = this->tree_weights;
      this->curr_pos = 0;
      this->curr_col = 0;

      if (this->col_indices.size() < this->n_cols)
        this->col_indices.resize(this->n_cols);

      real_t rnd_subrange, w_left;
      real_t curr_subrange;
      size_t curr_ix;

      for (this->curr_pos = 0; this->curr_pos < this->n_cols;
           this->curr_pos++)
      {
        curr_ix = 0;
        curr_subrange = curr_weights[0];
        if (curr_subrange <= 0)
          return;

        for (size_t lev = 0; lev < this->tree_levels; lev++)
        {
          rnd_subrange = std::uniform_real_distribution<real_t>(
              0., curr_subrange)(rnd_generator);
          w_left = curr_weights[ix_child(curr_ix)];
          curr_ix = ix_child(curr_ix) + (rnd_subrange >= w_left);
          curr_subrange = curr_weights[curr_ix];
        }
        /* finally, add element from this iteration */
        this->col_indices[this->curr_pos] = curr_ix - this->offset;

        /* now remove the weight of the chosen element */
        curr_weights[curr_ix] = 0;
        for (size_t lev = 0; lev < this->tree_levels; lev++)
        {
          curr_ix = ix_parent(curr_ix);
          curr_weights[curr_ix] = curr_weights[ix_child(curr_ix)] + curr_weights[ix_child(curr_ix) + 1];
        }
      }
    }
  }

  template <typename lreal_t_safe>
  inline bool
  provallo::column_sampler<lreal_t_safe>::has_weights()
  {
    return !this->tree_weights.empty();
  }

  template <typename lreal_t_safe>
  inline size_t
  provallo::column_sampler<lreal_t_safe>::get_remaining_cols()
  {

    if (!this->has_weights())
      return this->curr_pos;
    else
      return this->n_cols - this->n_dropped;
  }

  template <typename lreal_t_safe>
  inline void
  provallo::column_sampler<lreal_t_safe>::get_array_remaining_cols(
      std::vector<size_t> &cols)
  {
    if (!this->has_weights())
    {
      cols.assign(this->col_indices.begin(),
                  this->col_indices.begin() + this->curr_pos);
      std::sort(cols.begin(), cols.begin() + this->curr_pos);
    }

    else
    {
      size_t n_rem = 0;
      for (size_t col = 0; col < this->n_cols; col++)
      {
        if (this->tree_weights[col + this->offset] > 0)
        {
          cols[n_rem++] = col;
        }
      }
    }
  }
  template <class lreal_t_safe>
  template <class other_t>
  inline provallo::column_sampler<lreal_t_safe> &
  provallo::column_sampler<lreal_t_safe>::operator=(
      const column_sampler<other_t> &other)
  {

    this->col_indices = other.col_indices;
    this->tree_weights = other.tree_weights;
    this->curr_pos = other.curr_pos;
    this->curr_col = other.curr_col;
    this->last_given = other.last_given;
    this->n_cols = other.n_cols;
    this->tree_levels = other.tree_levels;
    this->offset = other.offset;
    this->n_dropped = other.n_dropped;
    return *this;
  }

  template <class lreal_t_safe, class real_t_>
  inline bool
  provallo::SingleNodeColumnSampler<lreal_t_safe, real_t_>::initialize(
      real_t *weights, std::vector<size_t> *col_indices, size_t curr_pos,
      size_t n_sample, bool backup_weights)
  {

    if (!curr_pos)
      return false;

    this->col_indices = col_indices->data();
    this->curr_pos = curr_pos;
    this->n_left = this->curr_pos;
    this->weights_orig = weights;
    if (n_sample > std::max(log2ceil(this->curr_pos), (size_t)3))
    {
      this->using_tree = true;
      this->backup_weights = false;

      if (this->used_weights.empty())
      {
        this->used_weights.reserve(col_indices->size());
        this->mapped_indices.reserve(col_indices->size());
        this->tree_weights.reserve(2 * col_indices->size());
      }

      this->used_weights.resize(this->curr_pos);
      this->mapped_indices.resize(this->curr_pos);

      for (size_t col = 0; col < this->curr_pos; col++)
      {
        this->mapped_indices[col] = this->col_indices[col];
        this->used_weights[col] = weights[this->col_indices[col]];
        if (!weights[this->col_indices[col]])
          this->n_left--;
      }
      this->tree_weights.resize(0);
      build_btree_sampler(this->tree_weights, this->used_weights.data(),
                          this->curr_pos, this->tree_levels, this->offset);

      this->n_inf = 0;
      if (std::isinf(this->tree_weights[0]))
      {
        if (this->mapped_inf_indices.empty())
          this->mapped_inf_indices.resize(this->curr_pos);

        for (size_t col = 0; col < this->curr_pos; col++)
        {
          if (std::isinf(weights[this->col_indices[col]]))
          {
            this->mapped_inf_indices[this->n_inf++] =
                this->col_indices[col];
            weights[this->col_indices[col]] = 0;
          }

          else
          {
            this->mapped_indices[col - this->n_inf] =
                this->col_indices[col];
            this->used_weights[col - this->n_inf] =
                weights[this->col_indices[col]];
          }
        }
        this->tree_weights.resize(0);
        build_btree_sampler(this->tree_weights,
                            this->used_weights.data(),
                            this->curr_pos - this->n_inf,
                            this->tree_levels, this->offset);
      }

      this->used_weights.resize(0);

      if (this->tree_weights[0] <= 0 && !this->n_inf)
        return false;
    }

    else
    {
      this->using_tree = false;
      this->backup_weights = backup_weights;

      if (this->backup_weights)
      {
        if (this->weights_own.empty())
          this->weights_own.resize(col_indices->size());
        this->weights_own.assign(weights, weights + this->curr_pos);
      }

      this->cumw = 0;
      for (size_t col = 0; col < this->curr_pos; col++)
      {
        this->cumw += weights[this->col_indices[col]];
        if (!weights[this->col_indices[col]])
          this->n_left--;
      }

      if (std::isnan(this->cumw))
        throw std::runtime_error(
            "NAs encountered. Try using a different value for 'missing_action'.\n");

      /* if it's infinite, will choose among columns with infinite weight first */
      this->n_inf = 0;
      if (std::isinf(this->cumw))
      {
        if (this->inifinite_weights.empty())
          this->inifinite_weights.resize(col_indices->size());
        else
          this->inifinite_weights.assign(col_indices->size(), false);

        this->cumw = 0;
        for (size_t col = 0; col < this->curr_pos; col++)
        {
          if (std::isinf(weights[this->col_indices[col]]))
          {
            this->n_inf++;
            this->inifinite_weights[this->col_indices[col]] = true;
            weights[this->col_indices[col]] = 0;
          }

          else
          {
            this->cumw += weights[this->col_indices[col]];
          }
        }
      }

      if (!this->cumw && !this->n_inf)
        return false;
    }

    return true;
  }
  size_t
  get_number_of_reference_points(const TreesIndexer &indexer) noexcept;
  template <class lreal_t_safe, class real_t_>
  inline bool
  provallo::SingleNodeColumnSampler<lreal_t_safe, real_t_>::sample_col(
      size_t &col_chosen, RNG_engine &rnd_generator)
  {
    if (!this->using_tree)
    {
      if (this->backup_weights)
        this->weights_orig = this->weights_own.data();

      /* if there's infinites, choose uniformly at random from them */
      if (this->n_inf)
      {
        size_t chosen = std::uniform_int_distribution<size_t>(
            0, this->n_inf - 1)(rnd_generator);
        size_t curr = 0;
        for (size_t col = 0; col < this->curr_pos; col++)
        {
          curr += inifinite_weights[this->col_indices[col]];
          if (curr == chosen)
          {
            col_chosen = this->col_indices[col];
            this->n_inf--;
            this->inifinite_weights[col_chosen] = false;
            this->n_left--;
            return true;
          }
        }
        assert(0);
      }
      if (!this->n_left)
        return false;

      /* due to the way this is calculated, there can be large roundoff errors and even negatives */
      if (this->cumw <= 0)
      {
        this->cumw = 0;
        for (size_t col = 0; col < this->curr_pos; col++)
          this->cumw += this->weights_orig[this->col_indices[col]];
        if (unlikely(this->cumw <= 0))
          provallo::unexpected_error();
      }

      /* if there are no infinites, choose a column according to weight */
      lreal_t_safe chosen = std::uniform_real_distribution<lreal_t_safe>(
          (lreal_t_safe)0, this->cumw)(rnd_generator);
      lreal_t_safe cumw_ = 0;
      for (size_t col = 0; col < this->curr_pos; col++)
      {
        cumw_ += this->weights_orig[this->col_indices[col]];
        if (cumw_ >= chosen)
        {
          col_chosen = this->col_indices[col];
          this->cumw -= this->weights_orig[col_chosen];
          this->weights_orig[col_chosen] = 0;
          this->n_left--;
          return true;
        }
      }
      col_chosen = this->col_indices[this->curr_pos - 1];
      this->cumw -= this->weights_orig[col_chosen];
      this->weights_orig[col_chosen] = 0;
      this->n_left--;
      return true;
    }

    else
    {
      /* if there's infinites, choose uniformly at random from them */
      if (this->n_inf)
      {
        size_t chosen = std::uniform_int_distribution<size_t>(
            0, this->n_inf - 1)(rnd_generator);
        col_chosen = this->mapped_inf_indices[chosen];
        std::swap(this->mapped_inf_indices[chosen],
                  this->mapped_inf_indices[--this->n_inf]);
        this->n_left--;
        return true;
      }

      else
      {
        /* TODO: should standardize all these tree traversals into one.
         This one in particular could do with sampling only a single
         random number as it will not typically require exhausting all
         options like the usual column sampler. */
        if (!this->n_left)
          return false;
        size_t curr_ix = 0;
        real_t rnd_subrange, w_left;
        real_t curr_subrange = this->tree_weights[0];
        if (curr_subrange <= 0)
          return false;

        for (size_t lev = 0; lev < tree_levels; lev++)
        {
          rnd_subrange = std::uniform_real_distribution<real_t>(
              0., curr_subrange)(rnd_generator);
          w_left = this->tree_weights[ix_child(curr_ix)];
          curr_ix = ix_child(curr_ix) + (rnd_subrange >= w_left);
          curr_subrange = this->tree_weights[curr_ix];
        }
        col_chosen = this->mapped_indices[curr_ix - this->offset];

        this->tree_weights[curr_ix] = 0.;
        for (size_t lev = 0; lev < this->tree_levels; lev++)
        {
          curr_ix = ix_parent(curr_ix);
          this->tree_weights[curr_ix] = this->tree_weights[ix_child(
                                            curr_ix)] +
                                        this->tree_weights[ix_child(curr_ix) + 1];
        }
        this->n_left--;
        return true;
      }
    }
  }

  template <class lreal_t_safe, class real_t_>
  inline void
  provallo::SingleNodeColumnSampler<lreal_t_safe, real_t_>::backup(
      SingleNodeColumnSampler<lreal_t_safe, real_t> &other, size_t ncols_tot)
  {

    other.n_inf = this->n_inf;
    other.n_left = this->n_left;
    other.using_tree = this->using_tree;

    if (this->using_tree)
    {
      if (other.tree_weights.empty())
      {
        other.tree_weights.reserve(ncols_tot);
        other.mapped_inf_indices.reserve(ncols_tot);
      }
      other.tree_weights.assign(this->tree_weights.begin(),
                                this->tree_weights.end());
      other.mapped_inf_indices.assign(this->mapped_inf_indices.begin(),
                                      this->mapped_inf_indices.end());
    }
    else
    {
      other.cumw = this->cumw;
      if (this->backup_weights)
      {
        if (other.weights_own.empty())
          other.weights_own.reserve(ncols_tot);

        other.weights_own.resize(this->n_left);
        for (size_t col = 0; col < this->n_left; col++)
          other.weights_own[col] =
              this->weights_own[this->col_indices[col]];
      }

      if (this->inifinite_weights.size())
      {
        if (other.inifinite_weights.empty())
          other.inifinite_weights.reserve(ncols_tot);

        other.inifinite_weights.resize(this->n_left);
        for (size_t col = 0; col < this->n_left; col++)
          other.inifinite_weights[col] =
              this->inifinite_weights[this->col_indices[col]];
      }
    }
  }

  template <class lreal_t_safe, class real_t_>
  inline void
  provallo::SingleNodeColumnSampler<lreal_t_safe, real_t_>::restore(
      const SingleNodeColumnSampler<lreal_t_safe, real_t> &other)
  {
    this->n_inf = other.n_inf;
    this->n_left = other.n_left;
    this->using_tree = other.using_tree;
    if (this->using_tree)
    {
      this->tree_weights.assign(other.tree_weights.begin(),
                                other.tree_weights.end());
      this->mapped_inf_indices.assign(other.mapped_inf_indices.begin(),
                                      other.mapped_inf_indices.end());
    }

    else

    {
      this->cumw = other.cumw;
      if (this->backup_weights)
      {
        for (size_t col = 0; col < this->n_left; col++)
          this->weights_own[this->col_indices[col]] =
              other.weights_own[col];
      }
      if (this->inifinite_weights.size())
      {
        for (size_t col = 0; col < this->n_left; col++)
          this->inifinite_weights[this->col_indices[col]] =
              other.inifinite_weights[col];
      }
    }
  }
  enum NodeType
  {
    NODE = 0,
    LEAF = 1,
    LIGHT_NODE = 2,
    LIGHT_LEAF = 3
  };

  class TreeNodeBase
  {

  protected:
    // Parent node
    const TreeNodeBase *_parent;
    // Children node (number of attributes or zero if this is a leaf)
    std::vector<TreeNodeBase *> _children;
    // Attribute tag
    const split_method *_split_method;
    class_dist _distribution;
    friend class Tree;

  public:
    // Construct from parent
    TreeNodeBase(const TreeNodeBase *parent) : _parent(parent), _split_method(0) 
    {
    }
    // Node constructor (from a data set)
    TreeNodeBase(const split_method *split_method, const TreeNodeBase *parent =
                                                       0) : _parent(parent), _children(0), _split_method(split_method?split_method->clone():nullptr)
    {
    }
    TreeNodeBase(const split_method *split_method, const TreeNodeBase *parent , const class_dist& dist) : _parent(parent), _children(0), _split_method(split_method?split_method->clone():nullptr),_distribution(dist)
    {
    } 
    // Function to print node information
    void
    print(std::ostream &out, const attribute_information &information,
          int count = 0) const;

    // Get type of this node
    NodeType
    getType() const
    {
      return get_type();
    }
    //dangerous when used with light nodes  
    // Get split method
    const split_method &
    get_split_method() const
    {
      return *_split_method;
    }
    // Internal virtual function to print node information
    virtual void
    _print(std::ostream &out,
           const attribute_information &information) const = 0;
    // Virtual comparison function
    virtual bool
    _compare(const TreeNodeBase &other) const = 0;
    // Get type of node
    virtual NodeType
    get_type() const = 0;
    // Internal serialization into buffer

    // Given a sample, returns next node
    template <class InputIterator>
    const TreeNodeBase *
    getNode(InputIterator begin, InputIterator end) const
    {
      if (_children.size() > 0)
      {
        // Return node in the correct branch
        if(_split_method) {
        size_t branch = _split_method->getBranch(
            begin);

        if (branch < _children.size())
          return _children[branch]->getNode(begin, end); 
        }
        else
        {
          return _children[0]->getNode(begin, end); 
        } 

      }
      // Return leaf
      return this;
    }

    // Comparison operator
    bool
    operator==(const TreeNodeBase &other) const
    {
      // Check type
      if (typeid(*this) != typeid(other))
        return false;
      // Check split method
      if (not(*_split_method == *other._split_method))
        return false;
      // Recursively check each child
      if (_children.size() != other._children.size())
        return false;
      for (uint32_t i = 0; i < _children.size(); ++i)
        if (not(*_children[i] == *other._children[i]))
          return false;
      return _compare(other);
    }
    const class_dist&  get_distribution() const
    {
      return _distribution;
    } 
    void set_distribution(const class_dist& dist)
    {
      _distribution = dist;
    } 
    void set_parent(const TreeNodeBase* parent)
    {
      _parent = parent;
    } 
    const TreeNodeBase* get_parent() const
    {
      return _parent;
    } 
    // Get number of children
    size_t
    getNumChildren() const
    {
      return _children.size();
    }

    void set_children(const std::vector<TreeNodeBase*>& children)
    {
      _children = children;
    } 

    const std::vector<TreeNodeBase*>& get_children() const
    {
      return _children;
    } 
    // Get child
    const TreeNodeBase *
    getChild(size_t index) const
    {
      if(index < _children.size()  )
         return _children[index];
      
      return nullptr;
    } 

    // Node builder
    static TreeNodeBase *
    nodeBuilder(const TreeNodeBase *parent, const TreeNodeBase *node);

    // Recursive serialization into buffer
    virtual void
    serialize(TreeNodeBase *node) const;

    // Recursively get data from buffer
    virtual void
    deserialize(const TreeNodeBase *node);

    virtual ~TreeNodeBase();
  };

  // A light-weight node in the tree (this is used on classifiers with big trees such as RF)
  class TreeLightNode : public TreeNodeBase
  {
    friend class Tree;
    friend class TreeNodeBase;

  public:
    // Construct from parent
    TreeLightNode(const TreeNodeBase *parent) : TreeNodeBase(parent)
    {
    }
    // Node constructor (from a data set)
    TreeLightNode(const split_method *split_method,
                  const class_dist &distribution, const TreeNodeBase *parent = 0) : 
	    TreeNodeBase(split_method, parent,distribution)
    {
    }
    NodeType
    get_type() const
    {
      return LIGHT_NODE;
    }
    virtual ~TreeLightNode(){}

  protected:
    void
    _print(std::ostream &out, const attribute_information &information) const;

     bool
    _compare(const TreeNodeBase &other) const
    {
      return (other.getType() == LIGHT_NODE &&
             get_distribution() == other.get_distribution()&& this->get_split_method().get_type() == other.get_split_method().get_type() );
    }

    void
    serialize(TreeNodeBase *node) const
    {
      if(node->getType() == LIGHT_NODE && _split_method->get_type() == node->get_split_method().get_type()   )  
        {
          //no need to change split method if it's the same.

          node->set_distribution(this->_distribution);
          node->set_parent(this->_parent);
          node->set_children(_children) ;

        }
    }

    void
    deserialize(const TreeNodeBase *node)
    {
      if(node->getType() == LIGHT_NODE && _split_method->get_type() == node->get_split_method().get_type() )
        {
          this->_children = node->get_children();
          this->_parent = node->get_parent();
          this->_distribution = node->get_distribution();
       

        } 
        
    }

  };

  // A leaf in the tree
  class TreeLightLeaf : public TreeNodeBase
  {
    friend class Tree;

  public:
    // Construct from parent
    TreeLightLeaf(const TreeNodeBase *parent) : TreeNodeBase(parent), _target_tag(0)
    {
    }
    // Node constructor (from a data set)
    TreeLightLeaf(const split_method *split_method,
                  const class_dist &distribution, const TreeNodeBase *parent = 0) : TreeNodeBase(split_method, parent), _target_tag(distribution.mode().discrete())
    {
    }

    NodeType
    get_type() const
    {
      return LIGHT_LEAF;
    }

  protected:
    // Target tag at this leaf
    attribute_tag _target_tag;

    void
    _print(std::ostream &out, const attribute_information &information) const;

    bool
    _compare(const TreeNodeBase &other) const
    {
      const TreeLightLeaf &dother = *static_cast<const TreeLightLeaf *>(&other);
      // Check internal data
      if (_target_tag != dother._target_tag)
        return false;
      return true;
    }

    const attribute_tag &get_target_tag() const
    {
      return _target_tag;
    }
    void
    serialize(TreeNodeBase *node) const;

    void
    deserialize(const TreeNodeBase *node);

    virtual ~TreeLightLeaf(){};
  };

  // A normal node in the tree (contains information of the data distribution
  // and an unique instance of the split method)
  class TreeNode : public TreeNodeBase
  {
    friend class Tree;

  public:
    // Construct from parent
    TreeNode(const TreeNodeBase *parent) : TreeNodeBase(parent)
    {
    }
    // Node constructor (from a data set)
    TreeNode(const split_method *split_method, const class_dist &distribution,
             const TreeNodeBase *parent = 0) : TreeNodeBase(split_method, parent), _distribution(distribution)
    {
    }
    NodeType
    get_type() const
    {
      return NODE;
    }
    // Get distribution
    const class_dist &get_distribution() const
    {
      return _distribution;
    }

  protected:
    // Class to hold distribution of data
    class_dist _distribution;

    void
    _print(std::ostream &out, const attribute_information &information) const;

    bool
    _compare(const TreeNodeBase &other) const
    {
      const TreeNode &dother = *static_cast<const TreeNode *>(&other);
      return (_distribution == dother._distribution);
    }

    void
    serialize(TreeNodeBase *node) const;

    void
    deserialize(const TreeNodeBase *node);

    virtual ~TreeNode(){};
  };

  // A leaf in the tree (contains information of the data distribution
  // and an unique instance of the split method)
  class TreeLeaf : public TreeNode
  {
    friend class Tree;

  public:
    // Construct from parent
    TreeLeaf(const TreeNodeBase *parent) : TreeNode(parent)
    {
    }
    // Node constructor (from a data set)
    TreeLeaf(const split_method *split_method, const class_dist &distribution,
             const TreeNodeBase *parent = 0) : TreeNode(split_method, distribution, parent)
    {
    }
    const class_dist &get_distribution() const
    {
      return TreeNode::get_distribution();
    }

  protected:
    void
    _print(std::ostream &out, const attribute_information &information) const;

    NodeType
    get_type() const
    {
      return LEAF;
    }

    virtual ~TreeLeaf(){};
  };

  class Tree
  {
  public:
    // Default constructor (empty tree)
    Tree() : _root(0)
    {
    }
    // Tree iterator
    typedef TreeNodeBase *iterator;
    typedef const TreeNodeBase *const_iterator;

    // Return iterator to the beginning of the tree.
    iterator
    begin()
    {
      return _root;
    }
    const_iterator
    begin() const
    {
      return _root;
    }

    // Classify sample
    template <class InputIterator>
    attribute
    classify(InputIterator begin, InputIterator end) const
    {
      const TreeNodeBase *node = _root->getNode(begin, end);
      if (node->getType() == LIGHT_LEAF)
        return attribute(
            discrete_value(
                static_cast<const TreeLightLeaf *>(_root->getNode(begin, end))->get_target_tag()));

      return (static_cast<const TreeNode *>(_root->getNode(begin, end)))->get_distribution().mode();
    }

    template <class InputIterator>
    class_dist
    posterior(InputIterator begin, InputIterator end) const
    {
      const TreeNodeBase *node = _root->getNode(begin, end);
      if (node->getType() == LIGHT_LEAF)
      {
        // Create "artificial distribution"
        class_dist dist(node->_split_method->size());
        dist.accum(
            static_cast<const TreeLightLeaf *>(_root->getNode(begin, end))->_target_tag);
        return dist;
      }
      return (static_cast<const TreeNode *>(_root->getNode(begin, end)))->_distribution;
    }

    // Insert new node on the tree
    template <class NodeType>
    iterator
    insert(const split_method *split_method, const class_dist &distribution,
           iterator position)
    {
      if (position)
      {
        // Normal node
        iterator child = new NodeType(split_method, distribution,
                                      position);
        position->_children.push_back(child);
        return child;
      }
      // Root node
      _root = new NodeType(split_method, distribution);
      return _root;
    }

    // Print tree
    void
    print(std::ostream &out, const attribute_information &information) const
    {
      _root->print(out, information);
    }

    // Put tree into buffer
    void
    serialize(TreeNodeBase *node) const
    {
      _root->serialize(node);
    }

    // Get tree from buffer
    void
    deserialize(const TreeNodeBase *node)
    {
      _root = TreeNodeBase::nodeBuilder(0, node);
    }

    // Comparison operator
    bool
    operator==(const Tree &other) const
    {
      return (*_root == *other._root);
    }

    virtual ~Tree()
    {
      if (_root)
        delete _root;
      _root = nullptr;
    }

  protected:
    // Root node
    TreeNodeBase *_root;
  };


  // A classifier based on a tree
  class tree_classifier : public classifier
  {
  protected:
    // Tree structure
    Tree _tree;

    // Print classifier information
    void
    print(std::ostream &out) const
    {
      _tree.print(out, this->_attributes_info);
      out << std::endl; // End line
    }

    // Internal function to serialize data into the buffer
    void
    serialize(classifier *serial) const
    {
      // Serialize tree
      if(serial->get_type()==get_type())
      {
        tree_classifier *tree_serial = static_cast<tree_classifier *>(serial);
        tree_serial->_tree= _tree;
        
      }
      //_tree.serialize (serial->_tree._root);

      // Serialize classifier
      // classifier::serialize (serial);
    }
    tree_classifier(const dataset &data, const parameter_base &parameters,
                    const std::random_device &random, split_method_factory *factory) : classifier(data, parameters, random, factory)
    {
      _name = "Tree";
    }


    tree_classifier(const dataset &data, const parameter_base &parameters,
                    const std::random_device &random, std::ostream& out, split_method_factory *factory) : classifier(data, parameters, random, factory)
    {
      _name = "Tree";
        out.flush();
    }
	
    tree_classifier(const classifier *deserial);
    Tree &get_tree()
    {
      return _tree;
    }
    const Tree &get_tree() const
    {
      return _tree;
    }

    template <class InputIterator>
    class_dist
    posterior(InputIterator begin, InputIterator end) const
    {
      return _tree.posterior(begin, end);
    }

    template <class InputIterator>
    attribute
    classify(InputIterator begin, InputIterator end) const
    {
      return _tree.classify(begin, end);
    }

    attribute
    classify(dataset::attribute_iterator begin,
             dataset::attribute_iterator end) const
    {
      return classify<dataset::attribute_iterator>(begin, end);
    }
    class_dist
    posterior(dataset::attribute_iterator begin,
              dataset::attribute_iterator end) const
    {
      return posterior<dataset::attribute_iterator>(begin, end);
    }

    attribute
    classify(std::vector<attribute>::const_iterator begin,
             std::vector<attribute>::const_iterator end) const
    {
      return classify<std::vector<attribute>::const_iterator>(begin, end);
    }

    class_dist
    posterior(std::vector<attribute>::const_iterator begin,
              std::vector<attribute>::const_iterator end) const
    {
      return posterior<std::vector<attribute>::const_iterator>(begin, end);
    }

    attribute classify(const std::vector<attribute> &sample) const
    {
      return classify(sample.begin(), sample.end());
    }
    class_dist posterior(const std::vector<attribute> &sample) const
    {
      return posterior(sample.begin(), sample.end());
    }

    classifier_type
    get_type() const
    {
      return TREE;
    }
    virtual ~tree_classifier()
    {
    }
  };

  class ensemble_classifier : public classifier
  {
  protected:
    // Vector of base classifiers
    std::vector<classifier *> _classifiers;
    // Error factors for each classifier
    std::vector<real_t> _error;

    // Print classifier information
    void
    print(std::ostream &out) const;

    // Get beta of classifier
    real_t
    getBeta(uint32_t i) const
    {
      return _error[i] / (1 - _error[i]);
    }

    static constexpr const real_t lowError = 1E-6;

    // Get weight of the classifier
    real_t
    getWeight(uint32_t i) const
    {
      real_t error(_error[i]);
      if (error < lowError)
        return 1.0 / lowError; // If error is small, return a big enough number
      else
        return log<10>(1.0 / getBeta(i));
    }

    void
    serialize(classifier *serial) const;

  public:
    ensemble_classifier(const dataset &data, const parameter_base &parameters = none(),
                        const std::random_device &random =
                            std::random_device(),
                        split_method_factory *factory = nullptr) : classifier(data, parameters, random, factory)
    {

      _name = "Ensemble";
    }
    ensemble_classifier(const classifier *deserial);

    template <class InputIterator>
    class_dist
    posterior(InputIterator begin, InputIterator end) const;

    template <class InputIterator>
    attribute
    classify(InputIterator begin, InputIterator end) const;

    attribute
    classify(dataset::attribute_iterator begin,
             dataset::attribute_iterator end) const
    {
      return classify<dataset::attribute_iterator>(begin, end);
    }

    attribute
    classify(std::vector<attribute>::const_iterator begin,
             std::vector<attribute>::const_iterator end) const
    {
      return posterior<std::vector<attribute>::const_iterator>(begin, end).mode();
    }

    classifier_type
    get_type() const
    {
      return ENSEMBLE;
    }

    class_dist
    posterior(dataset::attribute_iterator begin,
              dataset::attribute_iterator end) const
    {
      return posterior<dataset::attribute_iterator>(begin, end);
    }

    class_dist
    posterior(std::vector<attribute>::const_iterator begin,
              std::vector<attribute>::const_iterator end) const
    {
      return posterior<std::vector<attribute>::const_iterator>(begin, end);
    }

    ~ensemble_classifier();
  };

  template <class InputIterator>
  class_dist
  ensemble_classifier::posterior(InputIterator begin,
                                 InputIterator end) const
  {
    // Create container of class votes
    class_dist votes(_attributes_info.getTargetClassCount());

    // Classify with each tree
    for (uint32_t i = 0; i < _classifiers.size(); ++i)
    {
      // Get posterior
      class_dist posterior(_classifiers[i]->posterior(begin, end));
      // Sum both distributions
      for (uint32_t j = 0; j < votes.size(); ++j)
        votes.accum(
            j,
            votes.percentage(j) + getWeight(i) * posterior.percentage(j));
    }
    return votes;
  }

  template <class InputIterator>
  attribute
  ensemble_classifier::classify(InputIterator begin, InputIterator end) const
  {
    //removed.mode()
    //make sure begin-end is row size of dataset

    size_t distance = std::distance(begin, end);
    if (distance != _attributes_info.getSize())
    {
      std::cerr << "Error: input size does not match dataset size" << std::endl;
      return attribute();
    }
      
    return posterior(begin, end).mode();
  }

  // Very similar to a decision tree but with some randomness injected
  template <typename GainPolicy>
  class random_tree : public tree_classifier, public GainPolicy
  {
    // Use gain function from the Gain policy
    using GainPolicy::gain;

    // Number of attributes to select on a split node
    uint32_t _rho;
    // Maximum level of the tree
    uint32_t _level;
    // Minimum gain
    real_t _min_gain;

    // Method that construct the tree using a list of attributes
    // to choose the best split and the data set
    void
    createTree(Tree::iterator parent, const dataset &data,
               std::vector<attribute_tag> prev_attributes, uint32_t level);

    // Select attributes (using rho)
    void
    selectAttributes(std::vector<attribute_tag> *new_attributes,
                     const std::vector<attribute_tag> &prev_attributes);

  public:
    // Constructor parameters
    // data : Training data
    // rho : Number of attributes / features selected when the node splitting is done
    // level : Maximum level of the growing tree (zero means no limit)
    // min_gain : Minimum gain to stop growing the tree
    
    //provallo::dataset&, const provallo::parameter_base&, std::random_device, std::ostream&, const provallo::split_method_factory&

    //
    random_tree(const dataset &data, const parameter_base &parameters,
                const std::random_device &random, split_method_factory *factory = nullptr);

    random_tree(const dataset &data, const parameter_base &parameters,
                const std::random_device &random, std::ostream& out = std::cout , split_method_factory *factory = nullptr);
                
    virtual classifier_type
    get_type() const
    {
      return RTREE;
    }

    // Get distribution of outcomes
    template <class InputIterator>
    class_dist
    posterior(InputIterator begin, InputIterator end) const
    {
      return tree_classifier::posterior(begin, end);
    }

    template <class InputIterator>
    attribute
    classify(InputIterator begin, InputIterator end) const
    {
      return tree_classifier::classify(begin, end).mode();
    }

    ~random_tree()
    {
    }
  };

  template <typename GainPolicy>
  random_tree<GainPolicy>::random_tree(const dataset &data,
                                       const parameter_base &parameters,
                                       const std::random_device &rm, split_method_factory *factory) : tree_classifier(data, parameters, rm, factory), _rho(0), _level(0), _min_gain(0)
  {
    _name = "Random Tree";
    if(_split_factory==nullptr||(_split_factory&&(_split_factory->getSize()==0)))
    {
      _split_factory = new split_method_factory(data, rm);
    }

     // Sanity check
    if (parameters.getType() != random_tree_param::_type())
      throw(std::runtime_error(
          "Bad parameter type " + std::to_string(parameters.getType()) + " in random forest"));
    
    // Cast to correct type
    const random_tree_param *local_parameters =
        static_cast<const random_tree_param *>(&parameters);

    // Setup parameters
    _rho = local_parameters->getRho();
    _level = local_parameters->getLevel();
    _min_gain = local_parameters->getMinGain();
    // Push all attributes tags

    // Push all attributes tags
    std::vector<attribute_tag> attributes;
    // Push each attribute
    for (uint32_t i = 0; i <_split_factory->getSize(); ++i)
      attributes.push_back(i);

    //setup factories for each attribute
     // Create tree
    createTree(_tree.begin(), data, attributes, 0);
  }


  template <typename GainPolicy>
  random_tree<GainPolicy>::random_tree ( const dataset& data,const parameter_base& parameters,const std::random_device& rm ,std::ostream& os,split_method_factory* factory) : 
  tree_classifier(data, parameters, rm, factory), _rho(0), _level(0), _min_gain(0)
  {
    _name = "Random Tree";
    UNDEF_REFERENCE(os);
    UNDEF_REFERENCE2 (os);

    if(_split_factory==nullptr||(_split_factory&&(_split_factory->getSize()==0)))
    {
      _split_factory = new split_method_factory(data, rm);
    }
    // Sanity check
    if (parameters.getType() != random_tree_param::_type())
      throw(std::runtime_error(
          "Bad parameter type " + std::to_string(parameters.getType()) + " in random forest"));

    // Cast to correct type
    const random_tree_param *local_parameters =
        static_cast<const random_tree_param *>(&parameters);

    // Setup parameters
    _rho = local_parameters->getRho();
    _level = local_parameters->getLevel();
    _min_gain = local_parameters->getMinGain();

    // Push all attributes tags
    std::vector<attribute_tag> attributes;
    // Push each attribute
    for (uint32_t i = 0; i < _split_factory->getSize(); ++i)
      attributes.push_back(i);

    // Create tree
    createTree(_tree.begin(), data, attributes, 0);
  }         

  template <typename GainPolicy>
  void
  random_tree<GainPolicy>::selectAttributes(
      std::vector<attribute_tag> *new_attributes,
      const std::vector<attribute_tag> &prev_attributes)
  {
    uint32_t n(prev_attributes.size());
    if (_rho < n)
    {
      uint32_t pop = n;
      for (uint32_t i = _rho; i > 0; --i)
      {
        real_t prob = 1.0;

        // std::mt19937 gen();
        std::uniform_real_distribution<real_t> uniform(0.0, 1.0);
        real_t x = uniform(getRandom());
        for (; x < prob; pop--)
          prob -= prob * i / pop;
        new_attributes->push_back(prev_attributes[n - pop - 1]);
      }
    }
    else
    {
      for (uint32_t i = 0; i < n; i++)
        new_attributes->push_back(prev_attributes[i]);
    }
  }

  struct isEqual
  {
    attribute_tag _tag;
    isEqual(const attribute_tag &tag) : _tag(tag)
    {
    }
    bool
    operator()(const attribute_tag &tag)
    {
      return tag == _tag;
    }
  };

  template <typename GainPolicy>
  void
  random_tree<GainPolicy>::createTree(
      Tree::iterator parent, const dataset &data,
      std::vector<attribute_tag> prev_attributes, uint32_t level)
  {
    // Create split factory
    // split_method_factory local_factory (data, getRandom ());

    // Attributes
    std::vector<attribute_tag> attributes;
    // Add randomness in the attribute selection
    selectAttributes(&attributes, prev_attributes);

    // Split method for target tag
    const split_method *target_split = _split_factory->getTargetMethod();
    assert(target_split);

    // Check if there are attributes to split the data
    if (attributes.size() == 0 || (_level && level == _level))
    {
      // Add a leaf node with the best class
      _tree.insert<TreeLightLeaf>(target_split, data.getDistribution(),
                                  parent);
    }
    else
    {
      // Calculate the gain of each attribute
      std::vector<real_t> gain_values(attributes.size());

      // Calculate the gain for each attribute
      for (uint32_t i = 0; i < gain_values.size(); ++i)
      {
        // Get tag
        attribute_tag tag(attributes[i]);
        // Calculate the gain
        const split_method* m=_split_factory->getMethod(tag);
        if(m!=nullptr)    
          gain_values[i] = gain(data, std::ref(*m) );
        else
          gain_values[i] = 0;

        //skip if no split.

      }

      // Get the higher gain
      uint32_t idx = max_element(gain_values.begin(), gain_values.end()) - gain_values.begin();

      // Check the gain minimum
      if (gain_values[idx] <= _min_gain)
      {
        // Insert a leaf node with the only target attribute on the data set
        _tree.insert<TreeLightLeaf>(target_split,
                                    data.getDistribution(), parent);
      }
      else
      {
        // Select split attribute
        attribute_tag split_tag = attributes[idx];
        // Get split method
        const split_method *split_method(
            _split_factory->getMethod(split_tag));

        // Add node to the tree
        uint32_t split_count = split_method->size();
        parent = _tree.insert<TreeLightNode>(split_method,
                                             data.getDistribution(),
                                             parent);

        std::vector<attribute_tag>::iterator pend(
            prev_attributes.end());
        if (split_method->get_type() == DISC)
          pend = std::remove_if(prev_attributes.begin(),
                                prev_attributes.end(),
                                isEqual(attributes[idx]));

        // Increment level counter
        ++level;
        // Apply the same process on child nodes
        for (uint32_t i = 0; i < split_count; ++i)
        {

          // Create a subset of the current data
          const dataset *subset = data.subsetReference(*split_method,
                                                       i);

          if (subset->size() == 0)
          {
            // Uniform distribution
            class_dist uniform(data.getDistribution().size(),
                               data.getDistribution().sum());
            // If attribute instance don't exist on the data, add a leaf
            _tree.insert<TreeLightLeaf>(target_split, uniform,
                                        parent);
          }
          else
          {
            // Create subtree for each child node
            createTree(
                parent,
                *subset,
                std::vector<attribute_tag>(prev_attributes.begin(),
                                           pend),
                level);
          }

          // Delete subset
          delete subset;
        }
      }
    }
  }

  class bayesian : public classifier
  {
    // ---- Probability matrix and arrays
    std::vector<real_t> _prior;

    std::vector<std::vector<std::vector<real_t>>> _likelihood;

    // Print classifier information
    void
    print(std::ostream &out) const;
    // Internal function to serialize data into the buffer
    void
    serialize(classifier *serial) const;

    // Encapsulate access to likelihood matrix (for future improvement on memory access)
    real_t &
    getLikelihood(uint32_t attr, uint32_t class_value, uint32_t branch);

    const real_t &
    getLikelihood(uint32_t attr, uint32_t class_value, uint32_t branch) const;

    // Encapsulate access to prior array (for future improvement on memory access)
    real_t &
    getPrior(uint32_t class_value);

    const real_t &
    getPrior(uint32_t class_value) const;


    void init();// called from constructor. 

  public:
    // No parameters for now
    bayesian(const dataset &data, const parameter_base &parameters = none(),
             const std::random_device &random = std::random_device(),split_method_factory *factory = nullptr); 
    
    bayesian(const dataset &data, const parameter_base &parameters = none(),
             const std::random_device &random = std::random_device(),const std::ostream& out = std::cout,split_method_factory *factory = nullptr);//:bayesian(data,parameters,random,factory);
    
    
    // Construct from buffer
    bayesian(const classifier *deserial);

    template <class InputIterator>
    class_dist
    posterior(InputIterator begin, InputIterator end) const;

    template <class InputIterator>
    attribute
    classify(InputIterator begin, InputIterator end) const;

    attribute
    classify(dataset::attribute_iterator begin,
             dataset::attribute_iterator end) const
    {
      return classify<dataset::attribute_iterator>(begin, end);
    }
    class_dist
    posterior(dataset::attribute_iterator begin,
              dataset::attribute_iterator end) const
    {
      return posterior<dataset::attribute_iterator>(begin, end);
    }

    attribute
    classify(std::vector<attribute>::const_iterator begin,
             std::vector<attribute>::const_iterator end) const
    {
      return classify<std::vector<attribute>::const_iterator>(begin, end);
    }

    class_dist
    posterior(std::vector<attribute>::const_iterator begin,
              std::vector<attribute>::const_iterator end) const
    {
      return posterior<std::vector<attribute>::const_iterator>(begin, end);
    }
    // Get the probability of a class value
    real_t
    getProbability(uint32_t class_value) const;
    //get likelihood of a class value
    
    	real_t &
	  getLikelihood(size_t attr, size_t class_value, size_t branch)
	{
		return _likelihood[attr][class_value][branch];
	}

	const real_t &
	getLikelihood(size_t attr, size_t class_value,
							size_t branch) const
	{
		return _likelihood[attr][class_value][branch];
	}

    //get prior of a class value
    real_t&
    getPrior(size_t class_value) const;



    // Get type of classifier
    classifier_type
    get_type() const;

    virtual ~bayesian();
  };

  template <class InputIterator>
  class_dist
  bayesian::posterior(InputIterator begin, InputIterator end) const
  {
    // Split method for target tag
    const split_method *target_split = splitFactory().getTargetMethod();
    // Get some counts
    uint32_t class_number(target_split->size());
    uint32_t attrs_number(_split_factory->getSize());
    // Create container of class votes
    class_dist votes(class_number);
    ssize_t distance = std::distance(begin, end);
    if( distance ==0 ) 
        {
		      return votes;

  	    }
    // Loop over each class value
    for (uint32_t i = 0; i < class_number; ++i)
    {
      real_t prior(_prior[i]);
      real_t likelihood(1.0);

      // Loop over each attributes
      for (uint32_t j = 0; j < attrs_number; ++j)
      {
        // Get split method for this attribute
        const split_method *attr_split = splitFactory().getMethod(j);
        if ( attr_split == nullptr ) continue;
        uint32_t attr_branch(
            attr_split->getBranch(begin));

        // Probability that Aj attribute takes branch <attr_branch> in a sample with class Ci
        if( attr_branch>_likelihood[j][i].size() ) continue;

        likelihood *= _likelihood[j][i][attr_branch];
      }

      // Calculate posterior probability
      real_t posterior = prior * likelihood;
      // Accumulate in distribution for this class
      votes.accum(i, posterior);
    }

    return votes;
  }

  template <class InputIterator>
  attribute
  bayesian::classify(InputIterator begin, InputIterator end) const
  {
    // Get the higher count
    return posterior(begin, end).mode();
  }

  //-- KMEANS/METRIC

  template <class MetricPolicy>
  class metric_classifier : public classifier, public MetricPolicy
  {

    // Training data
    std::vector<std::vector<attribute> *> _data;
    // Classes of each training samples saved internally
    std::vector<uint32_t> _class;
    // Attribute names
    std::vector<std::string> _names;
    // Offsets (to take into account the ignores attributes on the sample)
    // Weights of each attribute
    std::vector<real_t> _weights;
    // Type of each attribute
    std::vector<attribute_type> _types;
    // Number of neighbors used to classify a sample
    uint32_t _neighbors;
    // We should standardize the samples before classifying them
    std::vector<real_t> _mean;
    std::vector<real_t> _stdev;

    // Cast to the correct parameter type
    const metric_classifier_param *
    castParams(const parameter_base &parameters)
    {
      // Sanity check
      return static_cast<const metric_classifier_param *>(&parameters);
    }

  protected:
    // ---- Methods to safely access internal data from child class

    // Set the size of the internal storage of points
    void
    setPointsNumber(size_t points);

    // Get number of points
    size_t
    getPointsNumber() const;

    // Set data values
    template <class InputIterator>
    void
    setData(size_t idx, InputIterator begin, InputIterator end);

    // Set class
    void
    setClass(size_t idx, uint32_t vclass);

    // Get constant reference to data at some index (return NULL of data is not available at that position)
    std::vector<attribute> *
    getData(size_t idx) const;

    // Get class of data at some index
    uint32_t
    getClass(size_t idx) const;

    // ---- Auxiliary functions to deal with internal points and sample standardization

    // Tolerance for convergence (relative error)
    static constexpr const real_t _tolerance = 10E-10;

    // Push samples from data set iterator (using internal information)
    template <class InputIterator>
    void
    pushSample(std::vector<attribute> *sample, InputIterator begin,
               InputIterator end) const;
    
  // Push samples from data set iterator to memory mapped file vector
  template <class InputIterator>
    void
    pushSample(mmap_vector<attribute> *sample, InputIterator begin,
               InputIterator end) const;


    // Get closest point index
    template <class InputIterator>
    size_t
    getClosestPoint(uint32_t target, InputIterator begin,
                    InputIterator end) const;

    template <class InputIterator>
    bool
    checkConvergence(InputIterator lbegin, InputIterator lend,
                     InputIterator nbegin, InputIterator nend) const;

    // Check if the point is equal (with some tolerance) to the internal ones
    template <class InputIterator>
    bool
    checkPoint(InputIterator lbegin, InputIterator lend) const;

    // Get random sample from data set (will return a standardized sample)
    std::pair<std::vector<attribute>, uint32_t>
    getRandomSample(const dataset &dataset);

    // Print classifier information
    void
    print(std::ostream &out) const;

    // Internal function to serialize data into the buffer
    void
    serialize(classifier *serial) const;

  public:
    // Constructor
    metric_classifier(const dataset &data, const parameter_base &parameters,
                      const std::random_device &random =
                          std::random_device(),
                      split_method_factory *factory = nullptr);
    // Construct from buffer
    metric_classifier(const classifier *deserial);

    template <class InputIterator>
    class_dist
    posterior(InputIterator begin, InputIterator end) const;

    template <class InputIterator>
    attribute
    classify(InputIterator begin, InputIterator end) const;

    attribute
    classify(dataset::attribute_iterator begin,
             dataset::attribute_iterator end) const
    {
      return classify<dataset::attribute_iterator>(begin, end);
    }
    class_dist
    posterior(dataset::attribute_iterator begin,
              dataset::attribute_iterator end) const
    {
      return posterior<dataset::attribute_iterator>(begin, end);
    }

    attribute
    classify(std::vector<attribute>::const_iterator begin,
             std::vector<attribute>::const_iterator end) const
    {
      return classify<std::vector<attribute>::const_iterator>(begin, end);
    }

    class_dist
    posterior(std::vector<attribute>::const_iterator begin,
              std::vector<attribute>::const_iterator end) const
    {
      return posterior<std::vector<attribute>::const_iterator>(begin, end);
    }

    classifier_type
    get_type() const
    {
      return METRIC;
    }

    // Get size of the test data
    uint32_t
    size() const
    {
      return _data.size();
    }

    // Remove a point
    void
    remove(uint32_t idx);

    // Add a point (the "length" of the container should be equal to a sample)
    template <class Iterator>
    void
    add(Iterator begin, Iterator end);

    // Test a point
    bool
    test(uint32_t idx) const;

    // Print points
    void
    printPoints(std::ostream &out) const;

    virtual ~metric_classifier();
  };

  template <class MetricPolicy>
  metric_classifier<MetricPolicy>::metric_classifier(
      const dataset &data, const parameter_base &parameters,
      const std::random_device &random, split_method_factory *factory) : classifier(data, parameters, random, factory), _neighbors(castParams(parameters)->getNeighboursNumber())
  {

    // Attributes information
    const attribute_information &info(data.getattributes());
    const std::map<std::string, real_t> &weight_map =
        castParams(parameters)->getWeights();
    // Loop over attributes and check the not ignored ones
    for (uint32_t i = 0; i < info.getSize(); ++i)
    {
      if (info.getType(i) != ignored_attribute::_type() && i != info.get_target_tag())
      {
        // Push offset
        _offsets.push_back(i);
        // Push type of this attribute
        _types.push_back(info.getType(i));
        // Push name
        std::string name(info.getName(i));
        _names.push_back(name);
        // Check if we have weights defined
        if (weight_map.size())
        {
          std::map<std::string, real_t>::const_iterator it =
              weight_map.find(name);
          // Push weight
          if (it != weight_map.end())
            _weights.push_back(it->second);
          // If attribute is not on list, put a NULL weight
          else
            _weights.push_back(0.0);
        }
        else
        {
          // Push the same weight for all attributes
          _weights.push_back(1.0);
        }
      }
    }

    // Normalize weights
    real_t weight_sum(
        std::accumulate(_weights.begin(), _weights.end(), 0));
    // Sanity check
    // This is like an assert, it shouldn't happen... If the user controls the attribute's name on the weight map before
    // using it on the classifier everything should be fine.
    if (weight_sum == 0.0)
      throw(std::runtime_error("No weights were assigned to attributes"));

    std::transform(
        _weights.begin(), _weights.end(), _weights.begin(),
        std::bind1st(std::multiplies<real_t>(), 1.0 / weight_sum));

    // Resize containers
    _mean.resize(_offsets.size(), 0.0);
    _stdev.resize(_offsets.size(), 0.0);

    for (uint32_t i = 0; i < data.size(); ++i)
    {
      // Only standardize continuous attributes
      dataset::attribute_iterator it = data.begin(i);
      for (uint32_t j = 0; j < _offsets.size() && j<_types.size(); ++j)
      {
        if (_types[j] == continous_attribute::_type())
        {
          real_t value = (*(it + _offsets[j])).continous();
          // Increment mean
          _mean[j] += value;
          // Increment standard deviation
          _stdev[j] += value * value;
        }
      }
    }

    for (uint32_t j = 0; j < _mean.size(); ++j)
    {
      _mean[j] /= (real_t)data.size();
      _stdev[j] = sqrt(_stdev[j] / data.size() - _mean[j] * _mean[j]);
    }
  }

  template <class MetricPolicy>
  metric_classifier<MetricPolicy>::metric_classifier(
      const classifier *deserial) : classifier(deserial), _neighbors(0)
  {
    metric_classifier &buffer =
        reinterpret_cast<metric_classifier>(*deserial);
    // First get number attributes related data
    for (int i = 0; i < buffer.names_size(); ++i)
    {
      _names.push_back(buffer.names(i));
      _offsets.push_back(buffer.offsets(i));
      _weights.push_back(buffer.weights(i));
      _mean.push_back(buffer.mean(i));
      _stdev.push_back(buffer.stdev(i));
    }

    // Number of attributes
    size_t nattrs(buffer.names_size());
    // Total size of data
    size_t total_size(buffer.data_size());

    // Sanity check
    assert(total_size % nattrs == 0);

    for (size_t i = 0; i < total_size / nattrs; ++i)
    {
      // Create data
      std::vector<attribute> *data = new std::vector<attribute>();

      // Loop over attributes
      for (size_t j = 0; j < nattrs; ++j)
      {
        // Get value
        const attribute_value &value = buffer.data(i * nattrs + j);
      }

      // Set class
      _class.push_back(buffer.class_(i));
      // Set data
      _data.push_back(data);
    }

    // Set number of neighbors
    _neighbors = buffer.neighbors();
  }

  template <class MetricPolicy>
  template <class InputIterator>
  void
  metric_classifier<MetricPolicy>::pushSample(
      mmap_vector<attribute> *sample, InputIterator begin,
      InputIterator end) const
  {
    if(begin == end)
      return;
    // Loop over attributes and check the not ignored ones
    for (uint32_t j = 0; j < _offsets.size(); ++j)
    {
      if (_types[j] == continous_attribute::_type())
      {
        real_t value = (*(begin + _offsets[j])).continous();
        real_t standard = (value - _mean[j]);
        // Check for NULL standard deviation (in case all values are equal)
        if (_stdev[j] > _tolerance)
          standard /= _stdev[j];
        sample->push_back(attribute(standard));
      }
      else
      {
        // Just push attribute
        sample->push_back(*(begin + _offsets[j]));
      }
    }
  }

  template <class MetricPolicy>
  template <class InputIterator>
  void
  metric_classifier<MetricPolicy>::pushSample(
      std::vector<attribute> *sample, InputIterator begin,
      InputIterator end) const
  {
    if(begin == end)
      return;
    // Loop over attributes and check the not ignored ones
    for (uint32_t j = 0; j < _offsets.size(); ++j)
    {
      if (_types[j] == continous_attribute::_type())
      {
        real_t value = (*(begin + _offsets[j])).continous();
        real_t standard = (value - _mean[j]);
        // Check for NULL standard deviation (in case all values are equal)
        if (_stdev[j] > _tolerance)
          standard /= _stdev[j];
        sample->push_back(attribute(cont_value(standard)));  
      }
      else
      {
        // Just push attribute
        sample->push_back(*(begin + _offsets[j]));
      }
    }
  }

  template <class MetricPolicy>
  template <class InputIterator>
  size_t
  metric_classifier<MetricPolicy>::getClosestPoint(uint32_t target,
                                                   InputIterator begin,
                                                   InputIterator end) const
  {
    // Initial distance
    std::pair<real_t, size_t> minor_distance(
        std::numeric_limits<real_t>::infinity(), 0);

    // Loop over each point and found the closest one
    for (size_t i = 0; i < _data.size(); ++i)
    {
      // Not applicable
      if (_class[i] != target || not _data[i])
        continue;
      // Calculate distance
      real_t distance = MetricPolicy::distance(begin, end,
                                              _data[i]->begin(),
                                              _data[i]->end(),
                                              _types.begin(),
                                              _weights.begin());
      if (distance < minor_distance.first)
      {
        minor_distance.first = distance;
        minor_distance.second = i;
      }
    }

    // Return index
    return minor_distance.second;
  }

  template <class MetricPolicy>
  template <class InputIterator>
  bool
  metric_classifier<MetricPolicy>::checkConvergence(
      InputIterator lbegin, InputIterator lend, InputIterator nbegin,
      InputIterator nend) const
  {
    real_t distance = MetricPolicy::distance(lbegin, lend, nbegin, nend,
                                            _types.begin(),
                                            _weights.begin());
    return (distance <= _tolerance);
  }

  template <class MetricPolicy>
  template <class InputIterator>
  bool
  metric_classifier<MetricPolicy>::checkPoint(InputIterator lbegin,
                                              InputIterator lend) const
  {
    bool is_equal(false);
    for (size_t i = 0; i < _data.size(); ++i)
    {
      if (_data[i])
      {
        if (checkConvergence(lbegin, lend, _data[i]->begin(),
                             _data[i]->end()))
        {
          is_equal = true;
          break;
        }
      }
    }
    return is_equal;
  }

  template <class MetricPolicy>
  std::pair<std::vector<attribute>, uint32_t>
  metric_classifier<MetricPolicy>::getRandomSample(const dataset &dataset)
  {
    // Select instance
    std::uniform_int_distribution<uint32_t> uniform(0, dataset.size());

    size_t idx = uniform(getRandom());

    // PUsh random sample
    std::vector<attribute> sample;
    pushSample(&sample, dataset.begin(idx), (dataset.begin(idx) + dataset.getattributes().get_target_tag()));

    // Push class of this sample
    size_t target_class = (*(dataset.begin(idx) + dataset.getattributes().get_target_tag())).discrete();

    // Return random point
    return std::make_pair(sample, target_class);
  }

  template <class MetricPolicy>
  template <class InputIterator>
  class_dist
  metric_classifier<MetricPolicy>::posterior(InputIterator begin,
                                             InputIterator end) const
  {
    // TODO : Do some research on how to this more efficiently (this is just a POC)
    // Sorted vector

    std::vector<std::pair<real_t, uint32_t>> sorted(_data.size());
    // Sample to classify
    std::vector<attribute> sample;
    pushSample(&sample, begin, end);

    // Calculate distance
    for (uint32_t i = 0; i < sorted.size(); ++i)
    {
      if (_data[i])
      {
        real_t dist = MetricPolicy::distance(sample.begin(),
                                            sample.end(),
                                            _data[i]->begin(),
                                            _data[i]->end(),
                                            _types.begin(),
                                            _weights.begin());
        sorted[i] = std::make_pair(dist, i);
      }
      else
      {
        sorted[i] = std::make_pair(
            std::numeric_limits<real_t>::infinity(), i);
      }
    }
    // Sort values
    std::cout<<"[+] Sorting distances"<<std::endl;
    sort(sorted.begin(), sorted.end());

    // Split method for target tag
    const split_method *target_split = splitFactory().getTargetMethod();
    // Get some counts
    uint32_t class_number(target_split->size());
    // Create container of class votes
    class_dist votes(class_number);
    std::cout<<"[+] searching concensus..."<<std::endl;

    // Check if the test point is the same one that a training point
    if (sorted[0].first < 10e-10)
    {
      votes.accum(_class[sorted[0].second]);
      return votes;
    }
    std::cout<<"[+] Nearest neighbor distance: "<<sorted[0].first<<std::endl;
    // Weight each neighbor with the inverse of the distance to the testing point
    for (uint32_t i = 0; i < sorted.size() && i < _neighbors; ++i)
    {
      uint32_t idx(sorted[i].second);
      if (_data[idx])
      {
        real_t dist(sorted[i].first);
        votes.accum(_class[idx], 1.0 / dist);
      }
    }

    return votes;
  }

  template <class MetricPolicy>
  template <class InputIterator>
  attribute
  metric_classifier<MetricPolicy>::classify(InputIterator begin,
                                            InputIterator end) const
  {
    // Get the higher count
    return posterior(begin, end).mode();
  }

  template <class MetricPolicy>
  void
  metric_classifier<MetricPolicy>::printPoints(std::ostream &out) const
  {
    // Print points for each class
    out << "% Internal points : " << std::endl;

    for (size_t i = 0; i < _data.size(); ++i)
    {
      if (not _data[i])
        continue;

      for (size_t j = 0; j < _data[i]->size(); ++j)
      {
        if (_types[j] == continous_attribute::_type())
        {
          real_t not_standard((*_data[i])[j].continous());
          if (_stdev[j] > _tolerance)
            not_standard *= _stdev[j];
          not_standard += _mean[j];
          out << not_standard << ",";
        }
        else // Discrete attribute
        {
          out << std::setw(10) << (*_data[i])[j].discrete() << " ";
        }
      }

      out << _class[i] << std::endl;
    }
  }

  template <class MetricPolicy>
  void
  metric_classifier<MetricPolicy>::print(std::ostream &out) const
  {
    // Print some information about the classifier
    out << " ----- Metric Classifier " << std::endl;
    out << " [+] Number of attributes = " << _names.size() << std::endl;
    out << " [+] Mean, standard deviation and weight of each attribute : "
        << std::endl;
    out << " " << std::left << std::setw(40) << "Name" << std::setw(15)
        << "Mean" << std::setw(15) << "StdDeviation" << std::setw(15)
        << "Weight" << std::endl;
    for (size_t i = 0; i < _mean.size(); ++i)
      out << " " << std::left << std::setw(40) << _names[i] << std::fixed
          << std::setw(15) << _mean[i] << std::setw(15) << _stdev[i]
          << std::setw(15) << _weights[i] << std::endl;
  }

  template <class MetricPolicy>
  template <class Iterator>
  void
  metric_classifier<MetricPolicy>::add(Iterator begin, Iterator end)
  {
    // Add new vector
    _data.push_back(new std::vector<attribute>());
    // Loop over attributes and check the not ignored ones
    for (uint32_t j = 0; j < _offsets.size(); ++j)
    {
      if (_types[j] == continous_attribute::_type())
      {
        real_t value = (*(begin + _offsets[j])).continous();
        real_t standard = (value - _mean[j]);
        // Check for NULL standard deviation (in case all values are equal)
        if (_stdev[j] > 10e-10)
          standard /= _stdev[j];
        _data[_data.size() - 1]->push_back(attribute(standard));
      }
      else
      {
        // Just push attribute
        _data[_data.size() - 1]->push_back(*(begin + _offsets[j]));
      }
    }
  }

  template <class MetricPolicy>
  bool
  metric_classifier<MetricPolicy>::test(uint32_t idx) const
  {
    // Don't check removed points (return true)
    if (not _data[idx])
      return true;

    // Sorted vector
    std::vector<std::pair<real_t, uint32_t>> sorted(_data.size());

    // Calculate distance
    for (uint32_t i = 0; i < sorted.size(); ++i)
    {
      if (_data[i])
      {
        real_t dist = MetricPolicy::distance(_data[idx]->begin(),
                                            _data[idx]->end(),
                                            _data[i]->begin(),
                                            _data[i]->end(),
                                            _types.begin(),
                                            _weights.begin());
        sorted[i] = std::make_pair(dist, i);
      }
      else
      {
        sorted[i] = std::make_pair(
            std::numeric_limits<real_t>::infinity(), i);
      }
    }
    // Sort values
    sort(sorted.begin(), sorted.end());

    // Split method for target tag
    const split_method *target_split = splitFactory().getTargetMethod();
    // Get some counts
    uint32_t class_number(target_split->size());
    // Create container of class votes
    class_dist votes(class_number);

    // Get the first index different from zero (we can have the same point more than one time)
    uint32_t first = 0;
    for (; first < sorted.size(); ++first)
    {
      if (sorted[first].first > 10e-10)
      {
        break;
      }
    }

    // Sanity exit... All points are equal!
    if (first == sorted.size())
      return true;

    // Weight each neighbor with the inverse of the distance to the testing point
    for (uint32_t i = first; i < sorted.size() && (i < (_neighbors + first));
         ++i)
    {
      uint32_t index(sorted[i].second);
      if (_data[index])
      {
        real_t dist(sorted[i].first);
        votes.accum(_class[index], 1.0 / dist);
      }
    }

    return (votes.mode().discrete() == _class[idx]);
  }

  // Remove a point
  template <class MetricPolicy>
  void
  metric_classifier<MetricPolicy>::remove(uint32_t idx)
  {
    // Delete data
    delete *(_data.begin() + idx);
    _data[idx] = 0;
  }

  // Set the size of the internal storage of points
  template <class MetricPolicy>
  void
  metric_classifier<MetricPolicy>::setPointsNumber(size_t points)
  {
    _data.resize(points, 0);
    _class.resize(points, 0);
  }

  // Get number of points
  template <class MetricPolicy>
  size_t
  metric_classifier<MetricPolicy>::getPointsNumber() const
  {
    return _data.size();
  }

  // Set data values
  template <class MetricPolicy>
  template <class InputIterator>
  void
  metric_classifier<MetricPolicy>::setData(size_t idx, InputIterator begin,
                                           InputIterator end)
  {
    assert(idx < _data.size());
    if( _data[idx] != nullptr)
          delete _data[idx];
    _data[idx] = new std::vector<attribute>(begin, end);
  }

  // Set class
  template <class MetricPolicy>
  void
  metric_classifier<MetricPolicy>::setClass(size_t idx, uint32_t vclass)
  {
    assert(idx < _data.size());
    _class[idx] = vclass;
  }

  // Get constant reference to data at some index (return NULL of data is not available at that position)
  template <class MetricPolicy>
  std::vector<attribute> *
  metric_classifier<MetricPolicy>::getData(size_t idx) const
  {
    assert(idx < _data.size());
    return _data[idx];
  }

  // Get class of data at some index
  template <class MetricPolicy>
  uint32_t
  metric_classifier<MetricPolicy>::getClass(size_t idx) const
  {
    assert(idx < _data.size());
    return _class[idx];
  }

  template <class MetricPolicy>
  void
  metric_classifier<MetricPolicy>::serialize(classifier *serial) const
  { 
	 UNDEF_REFERENCE(serial)
         UNDEF_REFERENCE2(serial)
    // Get buffer for metric classifier
    // metric_classifier* buffer(serial->MutableExtension(metric_classifier::child));
#if 0
      // Loop over data points
      for(size_t i = 0 ; i < _data.size() ; ++i) {
          for(size_t j = 0 ; j < _types.size() ; ++j) {
              // Add data
              attribute_value* serial_value = buffer->add_data();
              // Check the type of attribute
              if(_types[j] == continous_attribute::_type()) {
                  // Set continuous value for this attribute
       //           serial_value->set_type(ContinuousAttribute::type());
               } else if(_types[j] == discrete_attribute::_type()) {
                  // Set discrete value for this attribute
          //        serial_value->set_type(DiscreteAttribute::type());
               }
          }
          // Add class
          buffer->add_class_(_class[i]);
      }

      // Now put attributes related information
      for(size_t i = 0 ; i < _names.size() ; ++i) {
          buffer->add_names(_names[i]);
          buffer->add_offsets(_offsets[i]);
          buffer->add_weights(_weights[i]);
          buffer->add_mean(_mean[i]);
          buffer->add_stdev(_stdev[i]);
      }

      // Put number of neighbors
      buffer->set_neighbors(_neighbors);
#endif
  }

  template <class MetricPolicy>
  metric_classifier<MetricPolicy>::~metric_classifier()
  {
    this->_data.clear();
    //nothing to do
  }

  template <class MetricPolicy>
  class Kmeans : public metric_classifier<MetricPolicy>
  {

    using metric_classifier<MetricPolicy>::pushSample;
    using metric_classifier<MetricPolicy>::setData;
    using metric_classifier<MetricPolicy>::getData;
    using metric_classifier<MetricPolicy>::setClass;
    using metric_classifier<MetricPolicy>::setPointsNumber;
    using metric_classifier<MetricPolicy>::getPointsNumber;
    using metric_classifier<MetricPolicy>::getClosestPoint;
    using metric_classifier<MetricPolicy>::checkConvergence;
    using metric_classifier<MetricPolicy>::checkPoint;
    using metric_classifier<MetricPolicy>::getRandomSample;

    // Cast to the correct parameter type
    const kmeans_param *
    castParams(const parameter_base &parameters)
    {
      return static_cast<const kmeans_param *>(&parameters);
    }

  public:
    // Data set subset selector (discrimination over target class)
    class selector_class
    {
      uint32_t _target_class;
      attribute_tag _target_tag;

    public:
      selector_class(const dataset &d, uint32_t target_class) : _target_class(target_class), _target_tag(
                                                                                                 d.getattributes().get_target_tag())
      {
      }

      // Test
      template <class Iterator>
      bool
      test(Iterator begin, Iterator end) const
      {
        if ((Iterator(begin + _target_tag ) > (end)) || (Iterator(begin + _target_tag ) ==end)  )  
          return false;
        return ((*(begin + _target_tag)).discrete() == _target_class);
      }

      ~selector_class()
      {
      }
    };

    // Auxiliary class to accumulate values of different points
    class Accumulator
    {
    public:
      Accumulator()
      {
      }

      // Accumulate attribute's value
      virtual void
      accum(const attribute &attr) = 0;

      // Get mean attribute
      virtual attribute
      mean() const = 0;

      // Reset accumulator (should clear internal information to count all over again)
      virtual void
      reset() = 0;

      virtual ~Accumulator()
      {
      }
    };

    // Accumulator for discrete attributes (will count the frequency of each class)
    class DiscreteAccumulator : public Accumulator
    {
      class_dist _distribution;

    public:
      DiscreteAccumulator(size_t count) : _distribution(count)
      {
      }

      // Accumulate attribute's value
      void
      accum(const attribute &attr)
      {
        _distribution.accum(attr.discrete());
      }

      // Get mean attribute
      attribute
      mean() const
      {
        return _distribution.mode();
      }

      // Reset accumulator (should clear internal information to count all over again)
      void
      reset()
      {
        _distribution = class_dist(_distribution.size());
      }

      virtual ~DiscreteAccumulator()
      {
      }
    };

    // Accumulator for continuous attributes
    class ContinuousAccumulator : public Accumulator
    {
      real_t _accum;
      size_t _count;

    public:
      ContinuousAccumulator() : _accum(0.0), _count(0)
      {
      }

      // Accumulate attribute's value
      void
      accum(const attribute &attr)
      {
        _accum += attr.continous();
        _count++;
      }

      // Get mean attribute
      attribute
      mean() const
      {
        return attribute(cont_value(_accum / (cont_value)_count));
      }

      // Reset accumulator (should clear internal information to count all over again)
      void
      reset()
      {
        _accum = 0.0;
        _count = 0;
      }

      virtual ~ContinuousAccumulator()
      {
      }
    };

    // Create a container of accumulator from attribute's information
    std::vector<Accumulator *>
    getAccumulators(const attribute_information &info) const;

  public:
    // Constructor
    Kmeans(const dataset &data, const parameter_base &parameters,
           const std::random_device &random = std::random_device(),split_method_factory* fac=nullptr) ;
    // Construct from buffer
    Kmeans(const classifier *deserial);

    classifier_type
    get_type() const
    {
      return KMEANS;
    }

    virtual ~Kmeans();
  };

  template <class MetricPolicy>
  Kmeans<MetricPolicy>::Kmeans(const dataset &data,
                               const parameter_base &parameters,
                               const std::random_device &rnd,split_method_factory* fac) : metric_classifier<MetricPolicy>(data,
                                                                                                   metric_classifier_param(1, castParams(parameters)->getWeights()),
                                                                                                   rnd,fac)   
  {

    // Number of point for k-means
    size_t points(castParams(parameters)->getPointsNumber());

    // Get information of attributes
    const attribute_information &attrs = data.getattributes();
    attribute_tag target_tag(attrs.get_target_tag());
    size_t target_count(attrs.getCount(target_tag));
    std::vector<dataset *> sets(target_count);

    // Resize containers
    setPointsNumber(target_count * points);

    // Create subsets
    for (size_t i = 0; i < sets.size(); ++i)
      sets[i] = data.subset(selector_class(data, i));

    for (size_t i = 0; i < sets.size(); ++i)
    {
      // Offset to select the first points
      size_t data_size(sets[i]->size());

      // Check if there is data on the set
      if (not data_size)
        continue;

      if (data_size <= points)
        throw(std::runtime_error(
            "Not enough points in data set for k-means"));
  
      for (size_t k = 0; k < points; ++k)
      {
        // Get random sample
        std::pair<std::vector<attribute>, uint32_t> random_sample =
            getRandomSample(*sets[i]);

        // Check that the new point is not on the internal data
        size_t max_count(0);
        while (checkPoint(random_sample.first.begin(),
                          random_sample.first.end()) &&
               max_count < attrs.getSize()*attrs.getSize()*attrs.getSize()  ) 
        {
          random_sample = getRandomSample(*sets[i]);
          setData(i * points + k, random_sample.first.begin(),
                random_sample.first.end());
          // Push class of this sample
          setClass(i * points + k, random_sample.second);

          ++max_count;
        }

        //throw(std::runtime_error(
        //      "Not enough points in data set for k-means"));
        // Create vector
        
      }
    }

    // Create a container of accumulators for each point
    std::vector<std::vector<Accumulator *>> accums(getPointsNumber());
    for (size_t i = 0; i < accums.size(); ++i)
      accums[i] = getAccumulators(attrs);

    // Main iteration loop
    bool finish = false;
    while (!finish)
    {
      // Set finish flag to true
      finish = true;

      for (size_t i = 0; i < sets.size(); ++i)
      {
        // Loop over points in this set
        for (size_t j = 0; j < sets[i]->size(); ++j)
        {
          // Get standardized samples
          std::vector<attribute> sample;
          pushSample(&sample, sets[i]->begin(j), sets[i]->begin(j)+accums[i].size() );



          // Get closest point index
          size_t idx = getClosestPoint(i, sample.begin(),
                                       sample.end());

          // Accumulate

            for (size_t k = 0; k < sample.size(); ++k)
              accums[idx%accums.size()][k%accums[idx].size()]->accum(sample[k]);
             

        }
      }

      // Update point values
      for (size_t i = 0; i < accums.size(); ++i)
      {
        std::vector<attribute> *data_ptr(getData(i));
        if (not data_ptr)
          continue;
        std::vector<attribute> old_points = (*data_ptr);
        for (size_t j = 0; j < accums[i].size(); ++j)
        {
          (*data_ptr)[j] = accums[i][j]->mean();
          accums[i][j]->reset();
        }
        bool converged = checkConvergence(data_ptr->begin(),
                                          data_ptr->end(),
                                          old_points.begin(),
                                          old_points.end());
        if (not converged)
          finish = false;
      }
    }

    // Delete accumulators
    for (size_t i = 0; i < accums.size(); ++i)
      for (size_t j = 0; j < accums[i].size(); ++j)
        delete accums[i][j];

    // Delete subsets
    for (size_t i = 0; i < sets.size(); ++i)
      delete sets[i];
  }

  template <class MetricPolicy>
  Kmeans<MetricPolicy>::Kmeans(const classifier *deserial) : metric_classifier<MetricPolicy>(deserial)
  {
  }

  template <class MetricPolicy>
  std::vector<typename Kmeans<MetricPolicy>::Accumulator *>
  Kmeans<MetricPolicy>::getAccumulators(
      const attribute_information &info) const
  {
    // Container of accumulators
    std::vector<Accumulator *> accums;
    for (uint32_t i = 0; i < info.getSize(); ++i)
    {
      if (info.getType(i) == continous_attribute::_type() && i != info.get_target_tag())
        accums.push_back(new ContinuousAccumulator());
      else if (info.getType(i) == discrete_attribute::_type() && i != info.get_target_tag())
        accums.push_back(new DiscreteAccumulator(info.getCount(i)));
    }
    return accums;
  }

  template <class MetricPolicy>
  Kmeans<MetricPolicy>::~Kmeans()
  {
  } 

  template <class BoostedClassifier>
  class adaboost : public ensemble_classifier
  {
    // Set target attribute
    attribute_tag _target;
    // Max random numbers per classifier
    static const size_t _max_rng_per_class = 1000000;
    adaboost_param _parameters;

  public:

    // Constructor

    adaboost(const dataset& data,const parameter_base& p = provallo::none(), const std::random_device& rand = std::random_device(),const std::ostream& out=std::cout,split_method_factory* split_factory = nullptr) : ensemble_classifier(data, p, rand,split_factory), _target(data.getattributes().get_target_tag()), _parameters(  data.getDistribution().size(),p)    
             {
                init_boost();
             }
    adaboost ( const dataset& data,const parameter_base& param , const std::random_device& dev, split_method_factory* factory) :adaboost(data,param,dev,std::cout,factory)
    {

    }   
    adaboost(const provallo::dataset& data, const provallo::parameter_base& param , const std::random_device& rand  ) ;
        
 
    //copy constructor
    adaboost(const adaboost& other) : ensemble_classifier(other), _target(other._target), _parameters(other._parameters)
    {

    }  
    //move constructor
    adaboost(adaboost&& other) : ensemble_classifier(std::move(other)), _target(std::move(other._target)), _parameters(std::move(other._parameters))
    {

    } 

 

    classifier_type
    get_type() const
    {
      return BOOST;
    }
    void init_boost();
    // Get distribution of outcomes
    template <class InputIterator>
    class_dist
    posterior(InputIterator begin, InputIterator end) const
    {
      return ensemble_classifier::posterior(begin, end);
    }

    template <class InputIterator>
    attribute
    classify(InputIterator begin, InputIterator end) const
    {
      return ensemble_classifier::classify(begin, end);
    }


    const adaboost_param& get_parameters() const
    {
      return _parameters;
    }

    virtual ~adaboost(){};
  };

  template <class BoostedClassifier>
  adaboost<BoostedClassifier>::adaboost(const dataset &data,
                                        const parameter_base &parameters,
                                        const std::random_device &random) : ensemble_classifier(data, parameters, random), _target(data.getattributes().get_target_tag()),_parameters(data.getDistribution().size(),parameters) 
  {
     if (parameters.getType() != adaboost_param::_type())
      throw(std::runtime_error("Bad parameter type in adaptive boost"));
 
     init_boost();
  }


  template <class BoostedClassifier>
  void adaboost<BoostedClassifier>::init_boost()
  {
    // Create threads
    std::vector<std::thread> threads;
    
    // Create base learners
    for (uint32_t i = 0; i < _parameters.getClassifiersNumber(); i++)
    { 


       static std::recursive_mutex mtx; 
       static adaboost<BoostedClassifier>* pThis=this;
    
       threads.push_back(
          std::thread([i]( ) 
          {
    
      // Get random number engine
      // local_random(getRandom());

      // Seed for sampling new set
      // local_random.jump(i * _max_rng_per_class);
      // Randomly draw training set
      // Initialize distribution of samples (uniform distribution)
      std::random_device dev;

      size_t datasize = pThis->_data.size() ;
      
      class_dist distribution(datasize, 1.0);

       const adaboost_param *local_parameters  = &pThis->get_parameters();
  
      //dev.seed(i * _max_rng_per_class);

       dataset *sampled_data = pThis->_data.randomSubset(dev, distribution);

      // New classifier
      BoostedClassifier *new_classifier = new BoostedClassifier(
          std::ref(*sampled_data), local_parameters->getParameters(), dev ,std::cout, const_cast<split_method_factory*>( &pThis->splitFactory()) );

      // Incorrect classified samples
      std::vector<uint32_t> correct_idx;
      // Error accumulator
      real_t error(0.0);

      // Test each sample , run only on the sample data 
      datasize = sampled_data->size();

      //size_t datasize = pThis->_data.size();
      for (uint32_t j = 0; j < datasize; ++j)
      {
        // Get target value on the test data
        attribute test_attr(*(sampled_data->begin(j) +pThis->_target));
        // Classify the data
        attribute class_attr(
            new_classifier->classify(sampled_data->begin(j), sampled_data->end(j)));
        // Count error
        if (test_attr.discrete() != class_attr.discrete())
        {
          // Count error
          error += distribution.percentage(j);
        }
        else
        {
          // Save index
          correct_idx.push_back(j);
        }
      }

      // Check error rate (because we need weak classifiers, with error rate less than 1/2)
      if (error > 0.5)
      {
        // Delete allocated stuff and don't push anything
        delete new_classifier;
        delete sampled_data;
        std::cout << "Error rate too high: " << error << std::endl;

        // Break main loop
        return;
      }

      // Lock mutex
      {
        std::lock_guard<std::recursive_mutex> lock(mtx);

        // Push classifier
        pThis->_classifiers.push_back(new_classifier);
        // Push error
        pThis->_error.push_back(error);

      }
      // Break main loop if the error rate is too low
      if (error < lowError)
        return;

      // Decrease probabilities if classification is correct
      for (uint32_t j = 0; j < correct_idx.size(); ++j)
      {
        //assume no Nan values in the distribution
        // Get index of sample correctly classified
        uint32_t idx(correct_idx[j]);
        real_t dist= distribution.percentage(idx) * pThis->getBeta(i);
        if(dist!=dist) dist=0.0;//if Nan
         // Set weight of sample
        distribution.set(idx,
                         pThis->getBeta(i) * distribution.percentage(idx));
      }

      // Finally delete the last set of samples
      delete sampled_data;
    } )//end lambda
    );//end thread
    }//end for

    // Wait for all threads to finish 
    // Wait for all threads to finish
       
   try {
    std::for_each(threads.begin(), threads.end(), [](std::thread &t)  
                  { t.join(); });
    
    } catch ( std::runtime_error & e ) {
      
      std::cout << "[!] Exception: " << e.what() << std::endl;
    }
    catch( ... ) {
      
      std::cout << "[!] Unknown exception!" << std::endl;
    
    } 

    std::cout << "[+] All classifiers finished." << std::endl;


   }

  template <typename GainPolicy>
  class decision_tree : public tree_classifier, public GainPolicy
  {
    // Use gain function from the Gain policy
    using GainPolicy::gain;

    // Method that construct the tree using a list of attributes
    // to choose the best split and the data set
    void
    createTree(Tree::iterator parent, const dataset &data,
               const std::vector<attribute_tag> &attributes);

  public:
    decision_tree(const dataset &data, const parameter_base &parameters = none(),
                 const std::random_device &random = std::random_device(), split_method_factory *factory = nullptr);

    classifier_type
    get_type() const
    {
      return DTREE;
    }

    template <class InputIterator>
    attribute
    classify(InputIterator begin, InputIterator end) const
    {
      return tree_classifier::classify(begin, end);
    }

    // Get distribution of outcomes
    template <class InputIterator>
    class_dist
    posterior(InputIterator begin, InputIterator end) const
    {
      return tree_classifier::posterior(begin, end);
    }

    ~decision_tree()
    {
    }
  };

  template <typename GainPolicy>
  decision_tree<GainPolicy>::decision_tree(const dataset &data,
                                         const parameter_base &parameters,
                                         const std::random_device &random, split_method_factory *factory) : tree_classifier(data, parameters, random, factory)
  {
    if(_split_factory == nullptr)
      _split_factory = new split_method_factory(data, getRandom() );
    // Push all attributes tags
    std::vector<attribute_tag> attributes;
    // Push each attribute
    for (uint32_t i = 0; i < _split_factory->getSize(); ++i)
      attributes.push_back(i);

    // Create tree
    createTree(_tree.begin(), data, attributes);
  }

  template <typename GainPolicy>
  void
  decision_tree<GainPolicy>::createTree(Tree::iterator parent,
                                       const dataset &data,
                                       const std::vector<attribute_tag> &attributes)
  {
    // Create split factory

    if(_split_factory == nullptr)
      _split_factory = new split_method_factory(data, getRandom() );  

    attribute_tag target_tag(data.getattributes().get_target_tag()); 

    // split_method_factory local_factory (data, getRandom ());
    std::vector<attribute_tag> att(data.getattributesNumber());
    // Split method for target tag
    const split_method *target_split = splitFactory().getTargetMethod();
    assert (target_split != nullptr);
    
    // local_factory.getTargetMethod ();

    // Check if there are attributes to split the data and the number of samples at this node
    if (attributes.size() == 0)
    {
      // Add a leaf node with the best class
      _tree.insert<TreeLeaf>(target_split, data.getDistribution(),
                             parent);
    }
    else
    {
      // Calculate the gain of each attribute
      std::vector<real_t> gain_values(attributes.size());
      // Get attributes
       // Calculate the gain for each attribute
      for (uint32_t i = 0; i < gain_values.size(); ++i)
      {
        // Get tag
        if (attributes[i] == target_tag)
          continue;
        attribute_tag tag(attributes[i]);
        // Calculate the gain
        
        const split_method *xsplit_method = _split_factory->getMethod(tag);
        if(xsplit_method != nullptr && xsplit_method->size() > 0 )
          gain_values[i] = gain(data, *xsplit_method );
        else
          gain_values[i] = 0.0;
      }

      // Get the higher gain
      uint32_t idx = max_element(gain_values.begin(), gain_values.end()) -  gain_values.begin();

      // Check if the gain is null (splitting won't gain any information, so
      // this should be a leaf node)
      if (gain_values[idx] == 0.0)
      {
        // Insert a leaf node with the only target attribute on the data set
        _tree.insert<TreeLeaf>(target_split, data.getDistribution(),
                               parent);
      }
      else
      {
        // Select split attribute
        attribute_tag split_tag = attributes[idx];

        // Get split method
        const split_method *split_method(
            splitFactory().getMethod(split_tag));
        
        // Add node to the tree
        uint32_t split_count = split_method->size();
        parent = _tree.insert<TreeNode>(split_method,
                                        data.getDistribution(), parent);

        // Remove the current attribute from the list
        if (split_method->get_type() == DISC)
        {
          
          att.resize(attributes.size() );
          std::copy(attributes.begin() , attributes.end(), att.begin());
          att.erase(att.begin() + idx);
          // Remove the current attribute from the list

        }
        else
        {
          att.resize(attributes.size());
          std::copy(attributes.begin(), attributes.end(), att.begin());

        }

        // Apply the same process on child nodes
        for (uint32_t i = 0; i < split_count; ++i)
        {

          // Create a subset of the current data
          const dataset *subset = data.subsetReference(*split_method,
                                                       i);
          if (subset)
          {
            if (subset->size() == 0 && target_split)
            {
              // Uniform distribution
              class_dist uniform(data.getDistribution().size(),
                                 data.getDistribution().sum());
              // If attribute instance don't exist on the data, add a leaf
              _tree.insert<TreeLeaf>(target_split, uniform, parent);
            }
            else
            {
              // Create subtree for each child node
              createTree(
                  parent,
                  std::ref(*subset),
                  std::ref(att));
            }

            // Delete subset
            delete subset;
          }
        }
      }
    }
  }
  template <class MetricPolicy>
  class nearest_neighbor : public metric_classifier<MetricPolicy>
  {

    using metric_classifier<MetricPolicy>::pushSample;
    using metric_classifier<MetricPolicy>::setData;
    using metric_classifier<MetricPolicy>::setClass;
    using metric_classifier<MetricPolicy>::setPointsNumber;
    using metric_classifier<MetricPolicy>::getPointsNumber;

    // Cast to the correct parameter type
    const nearest_neighbor_param *castParams(const parameter_base &parameters)
    {
      // Sanity check
      if (parameters.getType() != nearest_neighbor_param::_type())
        throw(std::runtime_error(std::string("Bad parameter type ") + std::to_string(parameters.getType()) + std::string(" in nearest neighbour")));
      return static_cast<const nearest_neighbor_param *>(&parameters);
    }

  public:
    // Constructor
    nearest_neighbor(const dataset &data, const parameter_base &parameters = none(), const std::random_device &random = std::random_device(),
                    const std::ostream& out=std::cout,
                    split_method_factory *factory = nullptr);
    // Construct from buffer
    nearest_neighbor(const classifier *deserial);

    classifier_type getType() const
    {
      return NNEIG;
    }

    virtual ~nearest_neighbor();
  };

  template <class MetricPolicy>
  nearest_neighbor<MetricPolicy>::nearest_neighbor(const dataset &data, const parameter_base &parameters, const std::random_device &rand,const std::ostream& out, split_method_factory *factory) : metric_classifier<MetricPolicy>(data, parameters, rand, factory)
  {

    // Attributes information
    const attribute_information &info(data.getattributes());

    // Resize containers
    setPointsNumber(data.size());

    // Now copy each attribute
    for (uint32_t i = 0; i < getPointsNumber(); ++i)
    {
      dataset::attribute_iterator it = data.begin(i);
      std::vector<attribute> sample;
      pushSample(&sample, data.begin(i), data.end(i));
      setData(i, sample.begin(), sample.end());
      // Push class of this sample
      setClass(i, (*(it + info.get_target_tag())).discrete());
    }
    UNDEF_REFERENCE(out)
    UNDEF_REFERENCE2(out)
   }

  template <class MetricPolicy>
  nearest_neighbor<MetricPolicy>::nearest_neighbor(const classifier *deserial) : metric_classifier<MetricPolicy>(deserial) {}

  template <class MetricPolicy>
  nearest_neighbor<MetricPolicy>::~nearest_neighbor() {}
  // Random Forests :
  template <class RandomClassifier>
  class random_forest : public ensemble_classifier
  {
    // Raw importance of each variable
    std::vector<real_t> _raw_importance;
    // Out of Bag (OOB) error estimation
    real_t _oob_error;

    // Max random numbers per tree
    static const size_t _max_rng_per_tree = 1000000;

    void
    print(std::ostream &out) const;

  public:
    // Constructor parameters
    random_forest(const dataset &data, const parameter_base &parameters = none(),
                  const std::random_device &dev = std::random_device(),
                  std::ostream &out = std::cout, split_method_factory *factory = nullptr);

    virtual classifier_type
    get_type() const
    {
      return RFORE;
    }

    // Get distribution of outcomes
    template <class InputIterator>
    class_dist
    posterior(InputIterator begin, InputIterator end) const
    {
      return ensemble_classifier::posterior(begin, end);
    }

    template <class InputIterator>
    attribute
    classify(InputIterator begin, InputIterator end) const
    {
      return ensemble_classifier::classify(begin, end);
    }

    // Get importance map
    std::map<std::string, real_t>
    getImportanceMap() const;

    // Get OOB error
    real_t
    getOobError() const
    {
      return _oob_error;
    }

    virtual ~random_forest()
    {
      _raw_importance.clear(); 
    }
    // for each template instantiation, initialize statics with defaults.

    static std::recursive_mutex _mutex;
    static attribute_tag last_target;
    static split_method_factory *_last_factory;   //_split_factory;
    static random_forest_param *_last_parameters; // local_parameters;
    static dataset *_last_dataset;
    static std::vector<classifier *> *gclassifiers; //= std::ref(_classifiers) ;
    static std::vector<class_dist> *gdist;          // =  std::ref(global_oob_predictions) ;
    static class_dist *distibution_ptr;             // distribution (data.size (), 1.0);
    static std::vector<real_t> *error_copy;
    static std::vector<real_t> *last_raw;
    static real_t *_last_oob;
  };
  // MT support for parallel training..

  template <class RandomClassifier>
  std::recursive_mutex random_forest<RandomClassifier>::_mutex;

  template <class RandomClassifier>
  attribute_tag random_forest<RandomClassifier>::last_target;

  template <class RandomClassifier>
  split_method_factory *random_forest<RandomClassifier>::_last_factory = nullptr;

  template <class RandomClassifier>
  random_forest_param *random_forest<RandomClassifier>::_last_parameters = nullptr;

  template <class RandomClassifier>
  dataset *random_forest<RandomClassifier>::_last_dataset = nullptr;

  template <class RandomClassifier>
  std::vector<classifier *> *random_forest<RandomClassifier>::gclassifiers = nullptr;

  template <class RandomClassifier>
  std::vector<class_dist> *random_forest<RandomClassifier>::gdist = nullptr;

  template <class RandomClassifier>
  class_dist *random_forest<RandomClassifier>::distibution_ptr = nullptr;

  template <class RandomClassifier>
  std::vector<real_t> *random_forest<RandomClassifier>::error_copy = nullptr;

  template <class RandomClassifier>
  std::vector<real_t> *random_forest<RandomClassifier>::last_raw = nullptr;

  template <class RandomClassifier>
  real_t *random_forest<RandomClassifier>::_last_oob = nullptr;

  template <class RandomClassifier>
  random_forest<RandomClassifier>::random_forest(
      const dataset &data, const parameter_base &parameters,
      const std::random_device &random, std::ostream &out, split_method_factory *factory) : ensemble_classifier(data, parameters, random, factory), _oob_error(0.0)
  {

    clock_t c = clock();
    if (parameters.getType() != random_forest_param::_type())
      throw(std::runtime_error(
          "Bad parameter type " + std::to_string(parameters.getType()) + std::string(" in random forest")));
    
    // Cast to correct type
    const random_forest_param *local_parameters =
        static_cast<const random_forest_param *>(&parameters);

    if (_split_factory == nullptr)
      _split_factory = new split_method_factory(data, getRandom());

    // Get target tag
    attribute_tag target_tag = data.getattributes().get_target_tag();

    // Resize importance
    _raw_importance.resize(_split_factory->getSize(), 0.0);

    // Number of classifiers
    uint32_t nclass = local_parameters->getClassifiersNumber();

    // Resize container before the loop
    _classifiers.resize(nclass, 0);
    _error.resize(nclass, 0.0);
     _oob_error = 0.0;

    _last_oob =&_oob_error;
    // Create uniform distribution
    static class_dist distribution(_attributes_info.getTargetClassCount());

    distribution.setup( _attributes_info.getTargetClassCount());

    // Create OBB prediction distribution
    uint32_t nclasses(splitFactory().getTargetMethod()->size());
    // This container has the distribution of predictions of each sample
    // each time is OOB
    static std::vector<class_dist> global_oob_predictions;
    global_oob_predictions.clear();
    global_oob_predictions.resize(data.size(), class_dist(nclasses, 0.0));

    std::vector<std::thread> train_loadthreads;
    size_t n = local_parameters->getClassifiersNumber();

    // static std::recursive_mutex _mutex; 
    //push the parameters to the statics for the threads to use.
    last_target = target_tag;
    _last_factory = _split_factory;
    _last_parameters = const_cast<random_forest_param *>(local_parameters);
    _last_dataset = const_cast<dataset *>(&data);
    gclassifiers = &_classifiers;
    gdist = &global_oob_predictions;
    distibution_ptr = &distribution;
    error_copy = &_error;
    last_raw = &_raw_importance;
 
    static std::ostream &out2 = out;

    for (uint32_t i = 0; i < n; ++i)
    {
       // Get random number engine
      // MathUtils::Random local_random(getRandom());
      // classifier* local_classifier = new RandomClassifier(local_data, local_parameters->getClassifierParameters(), local_random, out, _split_factory);
      train_loadthreads.push_back(
          std::thread([i]()
                      {
                  		  std::vector<uint32_t> oob_indices;
 		                    random_forest_param* local_parameters (_last_parameters);
		                    const dataset& data (std::ref(*_last_dataset));
		                    std::vector<classifier*>& _classifiers (std::ref(*gclassifiers ));

		                    class_dist distribution (*distibution_ptr);
		                    std::vector<real_t>& _error(std::ref(*error_copy));
		                    std::vector<real_t>& _raw_importance (std::ref(*last_raw));
		                    std::vector<class_dist>& global_oob_predictions(std::ref(*gdist));
		                    real_t& _oob_error = *_last_oob;
                        //real_t oob_error = 0.0;//local Out of bag error
                        std::ostream& out=out2;
		                    clock_t c =clock();
		                    std::random_device d;
				                attribute_tag target_tag = last_target;
                        const size_t ratio =  ((real_t)data.size () * (real_t)_classifiers.size ());


		  std::pair<dataset*, dataset*> random_set = data.randomSubsetOob (
		      d, distribution, &oob_indices);

      //it doesn't have to be the same size, indices are empty at fisirst

      //		  assert(oob_indices.size () == random_set.second->size ());
      if(oob_indices.size () != random_set.second->size ())
      oob_indices.resize(random_set.second->size ());
      
		  out<<"[+] creating [" << std::to_string(i) <<"] classifier ["<<std::to_string(local_parameters->getClassifiersNumber())<<"]"<<std::endl;

		  out << "[+] threaded load "<<::gettid()<<std::endl;
      //duplicate split method factory
      split_method_factory* sf = new split_method_factory (*_last_factory);
      
      		  classifier* new_classifier = new RandomClassifier (
	  	      *random_set.first, local_parameters->getParameters (),
		      std::random_device(),   out,sf);     

	  	  clock_t vla = clock();
	  	  out << "[+] Total Tree construction CPU time elapsed in s: "
	  				  << (real_t) ( vla-c) / CLOCKS_PER_SEC << std::endl;

  
		  // Push error (so all weights are equal to one)
	  	  {
	  	    std::lock_guard<std::recursive_mutex> guard(_mutex);

	  	     _classifiers[i] = new_classifier;
	  	    _error[i] = 0.0000001;

	  	  }
		  std::vector<real_t> class_importance (sf->getSize (), 0);  
      // Get OOB set
		  // OOB set
		  dataset &oob_set (*random_set.second);
		  // Get OOB error
		  size_t oob_error (0); //local thread error
		  size_t n(oob_set.size());
      //bool translate_values = false;
		  // Test OOB data
		  for (uint32_t k = 0; k < n; k++){

		      out << "[+] Testing : ["<<k <<"/"<<n<<"]"<< std::endl;
  	      // Get target value on the test data

        attribute test_attr(oob_set.getattribute (k, target_tag));


        // Classify the data

        std::vector<attribute> set_clas;
        
        //push the attributes to the vector
        for (uint32_t j = 0; j < oob_set.getattribute_info().getSize(); ++j)
        {
          if (j != target_tag)
          {
            set_clas.push_back (oob_set.getattribute (k, j));
          }
        }
        
        attribute class_attr (
            _classifiers[i]->classify (set_clas.begin(), set_clas.end () ));

        
 

          // Check if the classification is correct 

		      if (  test_attr.discrete() != class_attr.discrete ()) {
			    { // Accumulate OOB error (for importance estimation)
		      	++oob_error;
            out<<"[+] misclassification : "<<pthread_self()<<" expected : "<<std::to_string(test_attr.discrete()) << ", got "<<class_attr.to_string()<< std::endl; 
          }
		      // Set the result in OOB distribution
           out <<  "[+] classifier running on thread : "<<pthread_self()<<" Classified: ["<<k <<"/"<<n<<"], with :"<<std::to_string(oob_error) +std::string(" errors.")<< std::endl;
		      {

      			std::lock_guard<std::recursive_mutex> guard(_mutex);
      			// uint32_t index (oob_indices[k]);  // Get index of sample
             // Get the distribution of predictions for this sample

            if ( global_oob_predictions[i].size()==0)
              global_oob_predictions[i].setup (oob_set.getattribute_info().getTargetClassCount()); 

            if (class_attr.discrete () < global_oob_predictions[i].size()) 
              global_oob_predictions[i].accum ( class_attr.discrete (),  1.0f); 
            else 
              global_oob_predictions[i].accum ( 0,  1.0f);
             
		      }//lock
 		    }//if
      }//for
      // Get importance of each variable

		  out << "[+] importance labels for classifier "<<i<< std::endl;

		  {
   			  std::lock_guard<std::recursive_mutex> guard(_mutex);

		      size_t im_size=_raw_importance.size();
		      for (uint32_t j = 0; j < im_size ; j++)
		    {

		      _raw_importance[j] += class_importance[j] / (real_t)_classifiers.size ();
		    }
		  }

		       // Once everything is done, check OOB error

		  out << "[+] importance labels for classifier "<<i<< std::endl;
 
      //last lock: oob error
      {

      	std::lock_guard<std::recursive_mutex> guard(_mutex);

		    size_t gn = global_oob_predictions.size();
		    for (uint32_t i = 0; i < gn; ++i)
          {
            // Get target value on the test data
            attribute test_attr (data.getattribute(i,target_tag));
            // Check against the OOB prediction
            if (test_attr.discrete ()
                != global_oob_predictions[i].mode ().discrete ())
              ++oob_error;
          }
      _oob_error += oob_error/real_t(ratio);
      std::cout << "[+]Global OOB error = % " << 100.0 * _oob_error << std::endl; 
          

		  // Print some feedback to the caller
			  out << "[#] Classifier number " << i << " finished." << std::endl;
			  out << "[#]   OOB samples =  " << oob_set.size () << " / "
			      << data.size () << " (%" << 100 * real_t(oob_set.size ()-data.size()) /real_t( data.size ())
			      << ")" << std::endl;
			  out << "[#]   OOB error = % "
			      <<( 100.0 * (real_t) oob_error / (real_t) oob_set.size ())
			      << std::endl;
        out << "[#]   CPU time elapsed in s: "  
            << (real_t) (clock() - c) / CLOCKS_PER_SEC << std::endl;  
                      // Delete data

        
        delete random_set.first;
        delete random_set.second;
        
      
      }//end lock
        //cleanup data
		 })

        );//end lambda 

    }
    try {
    std::for_each(train_loadthreads.begin(), train_loadthreads.end(), [](std::thread &t)
                  { t.join(); });
    }catch ( std::runtime_error & e ) {
      std::cout << "[!] Exception: " << e.what() << std::endl;
    }
    catch( ... ) {
      std::cout << "[!] Unknown exception!" << std::endl;
    } 

    std::cout << "[+] All classifiers finished." << std::endl;
    std::cout << "[+] Total CPU time elapsed in s: "  
            << (real_t) (clock() - c) / CLOCKS_PER_SEC << std::endl;


  }

  // Print random forest (just the importance of each variable)
  template <class RandomClassifier>
  void
  random_forest<RandomClassifier>::print(std::ostream &out) const
  {
    out << std::setw(40) << std::left << "Name" << std::setw(15)
        << std::left << "Raw" << std::setw(15) << std::left << "Normalized"
        << std::endl;
    for (uint32_t i = 0; i < splitFactory().getSize(); ++i)
    {
      out << std::setw(40) << std::left;
      splitFactory().getMethod(i)->printName(out, _attributes_info);
      out << std::setw(15) << std::left << _raw_importance[i]
          << std::setw(15) << std::endl;
    }
    out << std::endl
        << "Out of Bag error =  % " << 100 * _oob_error
        << std::endl;
  }

  // Get map of variable name and raw importance
  template <class RandomClassifier>
  std::map<std::string, real_t>
  random_forest<RandomClassifier>::getImportanceMap() const
  {
    // Loop over split methods
    std::map<std::string, real_t> importance_map;
    for (uint32_t i = 0; i < splitFactory().getSize(); ++i)
    {
      // Get method
      const split_method *method = splitFactory().getMethod(i);
      // Method's name
      std::ostringstream oss;
      method->printName(oss, _attributes_info);
      // Get raw importance
      real_t raw_value = _raw_importance[i];
      importance_map.insert(std::make_pair(oss.str(), raw_value));
    }
    // Return map
    return importance_map;
  }

  class confusion_matrix
  { 
    //mmap_vector<mmap_vector<size_t>> _matrix;
    matrix<attribute_tag> _matrix;
    // Information about attributes
    std::vector<attribute_value> _class_values;
    // Number of errors
    size_t _fp;
    size_t _fn;
    size_t _tp;
    size_t _tn;

    size_t _total;
    size_t _error;

    real_t _accuracy, _precision, _recall, _f1;


  
    // Friendly printer
    friend std::ostream &
    operator<<(std::ostream &out, const confusion_matrix &q);

  public:
    confusion_matrix(const dataset &data, const classifier &_classifier);


    //copy constructor
    confusion_matrix(const confusion_matrix &other);
    //move constructor
    confusion_matrix(confusion_matrix &&other);
    //copy assignment
    confusion_matrix &operator=(const confusion_matrix &other);
    //move assignment
    confusion_matrix &operator=(confusion_matrix &&other);

    //comparison operator
    inline bool operator==(const confusion_matrix &other) const
    {
      if (_fp!=other._fp||_fn!=other._fn||_tp!=other._tp||_tn!=other._tn||_precision!=other._precision||_recall!=other._recall||_error!=other._error || _total!=other._total )
			 return false;
		if(_matrix.size1()!=other._matrix.size1() || _matrix.size2()!=other._matrix.size2())
			return false;
		
		for(size_t i=0;i<_matrix.size1();++i)
			for(size_t j=0;j<_matrix.size2();++j)
				if(_matrix(i,j)!=other._matrix(i,j))
					return false;
		return true;		
    }

    // Get total number of errors
    size_t
    getError() const;

 
    size_t getMatrixDim() const
    {
      return _matrix.size1();
    }
    size_t get_false_positive() const
    {
      return _fp;
    }
    size_t get_false_negative() const
    {
      return _fn;
    }
    size_t get_true_positive() const
    {
      return _tp;
    }
    size_t get_true_negative() const
    {
      return _tn;
    }
    size_t get_total() const
    {
      return _total;
    }
    size_t get_error() const
    {
      return _error;
    }

    virtual ~confusion_matrix()
    {
      
      this->_class_values.clear();
      this->_matrix.clear();

    }
  };


  class roc_curve  
  {
    // Area under the curve
    real_t _auc;  
    typedef std::pair<real_t,real_t> roc_point;
    // ROC points [dataset row size]
    std::vector<std::pair<real_t, real_t>> _roc_points;
    matrix<discrete_value> _matrix;
    friend std::ostream &
    
    operator<<(std::ostream &out, const roc_curve &q);

  public:
    roc_curve(const dataset &data, const classifier &_classifier);

    // Get area under the curve
    real_t
    getAUC() const
    {
      return _auc;
    }
     virtual ~roc_curve()
    {
      _matrix.clear();  
    }


    std::vector<std::pair<real_t, real_t>> get_points()const 
    {
      return _roc_points;
    }


  };

  class lightgbm_classifier: public ensemble_classifier 
  {
    protected:
    //Liu et. al 2017 LightGBM: A Highly Efficient Gradient Boosting Decision Tree 
    // Number of classes
    uint32_t _classes;
    // Number of attributes
    uint32_t _attributes;
    // Number of samples
    uint32_t _samples;
    // Number of trees
    uint32_t _trees;
    // Number of threads
    uint32_t _threads;
    // Number of leaves
    uint32_t _leaves;
    // Number of bins
    uint32_t _bins;
    // Number of iterations
    uint32_t _iterations;
    // Number of early stopping rounds
    uint32_t _early_stopping_rounds;
    // Learning rate
    real_t _learning_rate;
    // Number of boosting iterations
    uint32_t _boosting_iterations;
    // Bagging fraction
    real_t _bagging_fraction;
    // Bagging frequency
    uint32_t _bagging_freq;
    // Bagging seed
    uint32_t _bagging_seed;
    // Feature fraction
    real_t _feature_fraction;
    // Feature fraction seed
    uint32_t _feature_fraction_seed;
     // Minimum sum of instance weight in one leaf
    real_t _min_sum_hessian_in_leaf;
    // L1 regularization
    real_t _lambda_l1;
    // L2 regularization
    real_t _lambda_l2;
    // Minimum gain to perform split
    real_t _min_gain_to_split;
    // Maximum number of leaves
    uint32_t _max_leaves;
    // Maximum depth
    uint32_t _max_depth;
    // Maximum bin
    uint32_t _max_bin;
    // Number of data points to construct feature histogram
    uint32_t _bin_construct_sample_cnt;
    // Number of data points to use for finding splits
    uint32_t _num_leaves;
    // Number of data points to use for finding splits
    uint32_t _min_data_in_bin;
    // Number of data points to use for finding splits
    uint32_t _min_data_in_leaf;
    // Minimum number of data points in one node
    uint32_t _min_data_per_group;
    // Minimum number of data points in one node
    uint32_t _max_cat_threshold;
    // Minimum number of data points in one node
    uint32_t _cat_l2;
    // Minimum number of data points in one node
    uint32_t _cat_smooth;
    // Minimum number of data points in one node
    uint32_t _max_cat_to_onehot;
    // Minimum number of data points in one node
    uint32_t _top_k;
    // Minimum number of data points in one node
    uint32_t _monotone_constraints;
    // Minimum number of data points in one node
    uint32_t _feature_contri;
    // Minimum number of data points in one node
    uint32_t _forcedsplits_filename;
    // Minimum number of data points in one node
    uint32_t _verbosity;

    
   
    public:
      
     lightgbm_classifier(const dataset &data, const parameter_base &parameters,
      const std::random_device &random, std::ostream &out=std::cout, split_method_factory *factory=nullptr);  

      virtual ~lightgbm_classifier();
      

    //pure virtual functions:
        // Classify a set of attributes
    virtual attribute
    classify(dataset::attribute_iterator begin,
             dataset::attribute_iterator end) const ;
    virtual attribute
    classify(std::vector<attribute>::const_iterator begin,
             std::vector<attribute>::const_iterator end) const;

    // Get distribution of outcomes
    virtual class_dist
    posterior(dataset::attribute_iterator begin,
              dataset::attribute_iterator end) const;
    virtual class_dist
    posterior(std::vector<attribute>::const_iterator begin,
              std::vector<attribute>::const_iterator end) const;


    //helpers for the above:
    virtual class_dist  classify(const std::vector<attribute> &sample) const;
      
    virtual class_dist posterior (const std::vector<attribute> &sample) const ; 

    virtual void
    
    print(std::ostream &out) const;     
      
 

  };
  class xgboost_classifier: public ensemble_classifier 
  {
    protected:
    //Chen et. al 2016 XGBoost: A Scalable Tree Boosting System 
    //xgboost_params :
    // Number of classes
    uint32_t _classes;
    // Number of attributes
    uint32_t _attributes;
    // Number of samples
    uint32_t _samples;
    // Number of trees
    uint32_t _trees;
    // Number of threads
    uint32_t _threads;
    // Number of leaves
    uint32_t _leaves;
    // Number of bins
    uint32_t _bins;
    // Number of iterations
    uint32_t _iterations;
    public:
      //xgboost_classifier(const dataset &data, const std::string &params);

      xgboost_classifier(const dataset &data);
      
      xgboost_classifier(const dataset &data,const parameter_base &params   = none());


      
      virtual ~xgboost_classifier();
      
      virtual class_dist  classify(const std::vector<attribute> &sample) const;
      //posteriors
      //posterior probabilities
      virtual class_dist posterior (const std::vector<attribute> &sample) const ; 



      virtual void
      print(std::ostream &out) const;     
      
      virtual void  print(std::ostream &out, const dataset &data) const;    
      
      virtual void  print(std::ostream &out, const dataset &data, const std::vector<attribute> &predictions) const;
  };  



  class iso_classifier : public classifier 
  {
    

    const dataset &_data;
    isoforest_param _params;
    
    //iso_forest_params :
    // Number of classes
    isolation_forest* _isoforest = nullptr;
    //class distribution
    class_dist _class_dist; // Class distribution
    
    public:

      iso_classifier(const dataset &data,
      const parameter_base& isoforest_params ,
      std::random_device& rd  ,
      split_method_factory* factory ):classifier(data,isoforest_params, rd, factory ),

      _data(data),
      _params(static_cast<const isoforest_param&>(isoforest_params)),
      _isoforest(nullptr),
      _class_dist(2)
      {
        //resize class dist to number of classes .
        const attribute_information& attributes = data.getattributes();
        attribute_tag target = attributes.get_target_tag();
        _class_dist.setup(attributes.getCount(target));
 
        //create isolation forest 
        _isoforest = new isolation_forest();
        //set parameters
        matrix<real_t> data_matrix = transform_data(); 
        _isoforest->fit(data_matrix) ; 

      }

      //constructor

      iso_classifier(const dataset &data);
      //copy constructor
      iso_classifier(const iso_classifier &other);
      //move constructor
      iso_classifier(iso_classifier &&other);
      //copy assignment
      iso_classifier &operator=(const iso_classifier &other);
      //move assignment
      iso_classifier &operator=(iso_classifier &&other);

      //destructor

      virtual ~iso_classifier();
      //classify

      //validate data
      void validate_data();
      //print
      virtual void
      print(std::ostream &out) const;     

      //transform current dataset to matrix<real_t> 
      matrix<real_t> transform_data() const;


      //print
      matrix<real_t> get_importance() const;
      //print
      virtual void  print(std::ostream &out, const dataset &data) const;    
      //print
      virtual void  print(std::ostream &out, const dataset &data, const std::vector<attribute> &predictions) const; 

      //empty print 
      virtual void  print() const;
      //
      void set_class_dist(const class_dist &class_dist) { _class_dist = class_dist; }
      class_dist get_class_dist() const { return _class_dist; }
       //get dataset    
      const dataset &get_data() const { return _data; }
      //get isolation forest
      isolation_forest* get_isoforest() const { return _isoforest; }
      //set isolation forest
      void set_isoforest(isolation_forest* isoforest) {
           if (_isoforest &&_isoforest != isoforest ){
            delete _isoforest;            
           }
          _isoforest = isoforest; 
         }       
         //implement pure virtual function : classify

  
      //override classify
      virtual class_dist classify (const std::vector<attribute> &sample) const ;
      //posterior probabilities
      virtual class_dist posterior (const std::vector<attribute> &sample) const ; 

      //override pure virtual function : classify
      virtual class_dist classify (dataset::attribute_iterator begin, dataset::attribute_iterator end )
      {
        std::vector<attribute> sample_copy; 
        for (auto it = begin; it != end; ++it)
        {
          sample_copy.push_back(*it);
          
        }
        return classify(sample_copy);
      }
      //override attribute classify 
      virtual attribute classify (dataset::attribute_iterator begin,dataset::attribute_iterator end) const 
      {
        std::vector<attribute> sample_copy; 
        for (auto it = begin; it != end; ++it)
        {
          sample_copy.push_back(*it);
          
        }
        return classify(sample_copy).mode();
      }
      //override pure virtual function : get_type
      virtual classifier_type
      get_type() const{
        return classifier_type::ISO_FOREST;
      }



      //override pure virtual function : serialize  
      virtual void serialize( classifier* serial) const
      {
          UNDEF_REFERENCE(serial);
          UNDEF_REFERENCE2(serial);
      }
      // Classify a case
     virtual attribute
    classify(std::vector<attribute>::const_iterator begin,
             std::vector<attribute>::const_iterator end) const
             {

              std::vector<attribute> sample(begin, end);

              return classify(sample).mode();
             }

    // Get distribution of outcomes
    virtual class_dist
    posterior(dataset::attribute_iterator begin,
              dataset::attribute_iterator end) const
              {
                std::vector<attribute> copy;
                for (auto it = begin; it != end; ++it)
                {
                  copy.push_back(*it);
                }
                return posterior(copy);
              }
    virtual class_dist
    posterior(std::vector<attribute>::const_iterator begin,
              std::vector<attribute>::const_iterator end) const
              {
                return posterior(std::vector<attribute>(begin, end)); 
              }

      };


    //kdtree classifier  - wraps kdt template from kdt.h
    template <uint32_t N>
    class kdtree_classifier : public classifier, public kd_tree<N,real_t> 
    {
      //kdtree_params :
      // Number of classes
      uint32_t _classes;
      // Number of attributes
      uint32_t _attributes;
      // Number of samples
      uint32_t _samples;
      // Number of trees
      uint32_t _trees;
      // Number of threads
      uint32_t _threads;
      // Number of leaves
      uint32_t _leaves;
      // Number of bins
      uint32_t _bins;
      // Number of iterations
      uint32_t _iterations;
      // Number of early stopping rounds
      uint32_t _early_stopping_rounds;
      // Learning rate
      real_t _learning_rate;
      // Number of boosting iterations

      uint32_t _boosting_iterations;
      // Bagging fraction
      real_t _bagging_fraction;
      // Bagging frequency
      uint32_t _bagging_freq;
      // Bagging seed
      uint32_t _bagging_seed;
      // Feature fraction
      real_t _feature_fraction;
      // Feature fraction seed
      uint32_t _feature_fraction_seed;
       // Minimum sum of instance weight in one leaf
      real_t _min_sum_hessian_in_leaf;
      // L1 regularization
      real_t _lambda_l1;
      // L2 regularization
      real_t _lambda_l2;
      // Minimum gain to perform split
      real_t _min_gain_to_split;
      // Maximum number of leaves
      uint32_t _max_leaves;
      // Maximum depth
      uint32_t _max_depth;
      // Maximum bin
      uint32_t _max_bin;
      // Number of data points to construct feature histogram
      // Number of data points to use for finding splits
      uint32_t _bin_construct_sample_cnt;
      // Number of data points to use for finding splits
      uint32_t _num_leaves;
      // Number of data points to use for finding splits
      uint32_t _min_data_in_bin;
      // Number of data points to use for finding splits
      uint32_t _min_data_in_leaf;
      // Minimum number of data points in one node
      uint32_t _min_data_per_group;

      // Minimum number of data points in one node
      uint32_t _max_cat_threshold;
      // Minimum number of data points in one node
      uint32_t _cat_l2;
      // Minimum number of data points in one node
      uint32_t _cat_smooth;
      // Minimum number of data points in one node
      uint32_t _max_cat_to_onehot;
      // Minimum number of data points in one node
      uint32_t _top_k;
      // Minimum number of data points in one node
      uint32_t _monotone_constraints;

      //classifiers : 
      std::vector<kd_tree<N,real_t> *> _classifiers; 
      //class distribution
      class_dist _class_dist; // Class distribution
      //global oob predictions
      std::vector<class_dist> global_oob_predictions;
      //raw importance
      std::vector<real_t> _raw_importance;
      //oob error
      real_t _oob_error;
      //mutex
      mutable std::recursive_mutex _mutex;
      //split method factory
      split_method_factory* _factory;
      
      //constructor
      public:
      kdtree_classifier(const dataset &data, const parameter_base &parameters,
      const std::random_device &random, std::ostream &out=std::cout, split_method_factory *factory=nullptr);
      //copy constructor
      kdtree_classifier(const kdtree_classifier &other);
      //move constructor
      kdtree_classifier(kdtree_classifier &&other);
      //copy assignment
      kdtree_classifier &operator=(const kdtree_classifier &other);
      //move assignment
      kdtree_classifier &operator=(kdtree_classifier &&other);

      //destructor
      virtual ~kdtree_classifier();
      //classify
      virtual class_dist  classify(const std::vector<attribute> &sample) const;
      //posterior probabilities
      virtual class_dist posterior (const std::vector<attribute> &sample) const ;
      //print
      virtual void
      print(std::ostream &out) const;
      //print
      virtual void  print(std::ostream &out, const dataset &data) const;
      //print
      virtual void  print(std::ostream &out, const dataset &data, const std::vector<attribute> &predictions) const;
      //empty print
      virtual void  print() const;
      //get classifiers
      std::vector<kd_tree<N,real_t>*> get_classifiers() const { return _classifiers; }
      //set classifiers

      void set_classifiers(std::vector<kd_tree<N,real_t>*> classifiers) { _classifiers = classifiers; }
      //get classes
      uint32_t get_classes() const { return _classes; }
      //set classes
      void set_classes(uint32_t classes) { _classes = classes; }
      //get attributes
      uint32_t get_attributes() const { return _attributes; }
      //set attributes
      void set_attributes(uint32_t attributes) { _attributes = attributes; }
      //get samples
      uint32_t get_samples() const { return _samples; }
      //set samples
      void set_samples(uint32_t samples) { _samples = samples; }
      //get trees
      uint32_t get_trees() const { return _trees; }
      //set trees
      void set_trees(uint32_t trees) { _trees = trees; }
      //get threads
      uint32_t get_threads() const { return _threads; }
      //set threads
      void set_threads(uint32_t threads) { _threads = threads; }
      //get leaves

      uint32_t get_leaves() const { return _leaves; }
      //set leaves
      void set_leaves(uint32_t leaves) { _leaves = leaves; }
      //get bins
      uint32_t get_bins() const { return _bins; }
      //set bins
      void set_bins(uint32_t bins) { _bins = bins; }
      //get iterations
      uint32_t get_iterations() const { return _iterations; }
      //set iterations

      void set_iterations(uint32_t iterations) { _iterations = iterations; }


      //get early stopping rounds
      uint32_t get_early_stopping_rounds() const { return _early_stopping_rounds; }
      //set early stopping rounds

      void set_early_stopping_rounds(uint32_t early_stopping_rounds) { _early_stopping_rounds = early_stopping_rounds; }
      //get learning rate
      real_t get_learning_rate() const { return _learning_rate; }
      //set learning rate
      void set_learning_rate(real_t learning_rate) { _learning_rate = learning_rate; }
      //get boosting iterations
      uint32_t get_boosting_iterations() const { return _boosting_iterations; }
      //set boosting iterations
      void set_boosting_iterations(uint32_t boosting_iterations) { _boosting_iterations = boosting_iterations; }
      //get bagging fraction
      real_t get_bagging_fraction() const { return _bagging_fraction; }
      //set bagging fraction
      void set_bagging_fraction(real_t bagging_fraction) { _bagging_fraction = bagging_fraction; }
      //get bagging frequency
      uint32_t get_bagging_freq() const { return _bagging_freq; }
      //set bagging frequency

      void set_bagging_freq(uint32_t bagging_freq) { _bagging_freq = bagging_freq; }

      //get bagging seed

      uint32_t get_bagging_seed() const { return _bagging_seed; }
      //set bagging seed
      void set_bagging_seed(uint32_t bagging_seed) { _bagging_seed = bagging_seed; }
      //get feature fraction
      real_t get_feature_fraction() const { return _feature_fraction; }
      //set feature fraction
      void set_feature_fraction(real_t feature_fraction) { _feature_fraction = feature_fraction; }
      //get feature fraction seed
      uint32_t get_feature_fraction_seed() const { return _feature_fraction_seed; }
      //set feature fraction seed

      void set_feature_fraction_seed(uint32_t feature_fraction_seed) { _feature_fraction_seed = feature_fraction_seed; }
      //get minimum sum of instance weight in one leaf
      real_t get_min_sum_hessian_in_leaf() const { return _min_sum_hessian_in_leaf; }
      //set minimum sum of instance weight in one leaf
      void set_min_sum_hessian_in_leaf(real_t min_sum_hessian_in_leaf) { _min_sum_hessian_in_leaf = min_sum_hessian_in_leaf; }
      //get l1 regularization
      real_t get_lambda_l1() const { return _lambda_l1; }
      //set l1 regularization
      void set_lambda_l1(real_t lambda_l1) { _lambda_l1 = lambda_l1; }
      //get l2 regularization
      real_t get_lambda_l2() const { return _lambda_l2; }
      //set l2 regularization
      void set_lambda_l2(real_t lambda_l2) { _lambda_l2 = lambda_l2; }
      //get minimum gain to perform split
      real_t get_min_gain_to_split() const { return _min_gain_to_split; }
      //set minimum gain to perform split
      void set_min_gain_to_split(real_t min_gain_to_split) { _min_gain_to_split = min_gain_to_split; }
      //get maximum number of leaves
      uint32_t get_max_leaves() const { return _max_leaves; }
      //set maximum number of leaves
      void set_max_leaves(uint32_t max_leaves) { _max_leaves = max_leaves; }
      //get maximum depth
      uint32_t get_max_depth() const { return _max_depth; }
      //set maximum depth
      void set_max_depth(uint32_t max_depth) { _max_depth = max_depth; }
      //get maximum bin
      uint32_t get_max_bin() const { return _max_bin; }
      //set maximum bin
      void set_max_bin(uint32_t max_bin) { _max_bin = max_bin; }
      //get number of data points to construct feature histogram
      uint32_t get_bin_construct_sample_cnt() const { return _bin_construct_sample_cnt; }
      //set number of data points to construct feature histogram
      void set_bin_construct_sample_cnt(uint32_t bin_construct_sample_cnt) { _bin_construct_sample_cnt = bin_construct_sample_cnt; }
      //get number of data points to use for finding splits
      uint32_t get_num_leaves() const { return _num_leaves; }
      //set number of data points to use for finding splits
      void set_num_leaves(uint32_t num_leaves) { _num_leaves = num_leaves; }
      //get number of data points to use for finding splits
      uint32_t get_min_data_in_bin() const { return _min_data_in_bin; }
      //set number of data points to use for finding splits
      void set_min_data_in_bin(uint32_t min_data_in_bin) { _min_data_in_bin = min_data_in_bin; }
      //get number of data points to use for finding splits
      uint32_t get_min_data_in_leaf() const { return _min_data_in_leaf; }
      //set number of data points to use for finding splits
      void set_min_data_in_leaf(uint32_t min_data_in_leaf) { _min_data_in_leaf = min_data_in_leaf; }
      //get minimum number of data points in one node
      uint32_t get_min_data_per_group() const { return _min_data_per_group; }
      //set minimum number of data points in one node
      void set_min_data_per_group(uint32_t min_data_per_group) { _min_data_per_group = min_data_per_group; }
      //get minimum number of data points in one node
      uint32_t get_max_cat_threshold() const { return _max_cat_threshold; }
      //set minimum number of data points in one node
      void set_max_cat_threshold(uint32_t max_cat_threshold) { _max_cat_threshold = max_cat_threshold; }
      //get minimum number of data points in one node
      uint32_t get_cat_l2() const { return _cat_l2; }
      //set minimum number of data points in one node
      void set_cat_l2(uint32_t cat_l2) { _cat_l2 = cat_l2; }
      //get minimum number of data points in one node
      uint32_t get_cat_smooth() const { return _cat_smooth; }
      //set minimum number of data points in one node

      void set_cat_smooth(uint32_t cat_smooth) { _cat_smooth = cat_smooth; }
      //get minimum number of data points in one node
      uint32_t get_max_cat_to_onehot() const { return _max_cat_to_onehot; }
      //set minimum number of data points in one node
      void set_max_cat_to_onehot(uint32_t max_cat_to_onehot) { _max_cat_to_onehot = max_cat_to_onehot; }
      //get minimum number of data points in one node
      uint32_t get_top_k() const { return _top_k; }
      //set minimum number of data points in one node
      void set_top_k(uint32_t top_k) { _top_k = top_k; }
      //get minimum number of data points in one node
      uint32_t get_monotone_constraints() const { return _monotone_constraints; }
      //set minimum number of data points in one node
      void set_monotone_constraints(uint32_t monotone_constraints) { _monotone_constraints = monotone_constraints; }
      //get minimum number of data points in one node
      //uint32_t get_feature_contri() const { return _feature_contri; }
      //set minimum number of data points in one node
      //void set_feature_contri(uint32_t feature_contri) { _feature_contri = feature_contri; }
      
      uint32_t get_classifiers_size() const { return _classifiers.size(); }
      //set minimum number of data points in one node
      void set_classifiers_size(uint32_t classifiers_size) { _classifiers.resize(classifiers_size); }
      //get minimum number of data points in one node
      uint32_t get_classifiers_capacity() const { return _classifiers.capacity(); }   
      //set minimum number of data points in one node
      void set_classifiers_capacity(uint32_t classifiers_capacity) { _classifiers.reserve(classifiers_capacity); }
      //get minimum number of data points in one node
      uint32_t get_classifiers_max_size() const { return _classifiers.max_size(); }
      //set minimum number of data points in one node
      void set_classifiers_max_size(uint32_t classifiers_max_size) { _classifiers.resize(classifiers_max_size); }

  //helper functions : 
  //get classifier
  kd_tree<N,real_t>* get_classifier(uint32_t index) const { return _classifiers[index]; }
  //set classifier
  void set_classifier(uint32_t index, kd_tree<N,real_t>* classifier) { _classifiers[index] = classifier; }
  //add classifier
  void add_classifier(kd_tree<N,real_t>* classifier) { _classifiers.push_back(classifier); }
  //remove classifier
  void remove_classifier(uint32_t index) { _classifiers.erase(_classifiers.begin() + index); }
  //clear classifiers
  void clear_classifiers() { _classifiers.clear(); }
  //get classifiers begin
  typename std::vector<kd_tree<N,real_t>*>::iterator get_classifiers_begin() { return _classifiers.begin(); }
  //get classifiers end
  typename std::vector<kd_tree<N,real_t>*>::iterator get_classifiers_end() { return _classifiers.end(); }
    
};//end of kdtree classifier class

// utility: print classifier summary()
  void
  print_classifier_summary(const std::string &data_set_name,
                           const dataset &data, const classifier &_classifier);
// utility: train and test classifier and print summary

  template <class Classifier>
  inline void
  train_and_test(const std::string &class_name, dataset &train_data,
                 const dataset &test_data , const parameter_base &params  = provallo::none() )
  {
    // measure :
    auto c_start = clock();
    auto t_start = std::chrono::high_resolution_clock::now();
  
     // create classifier and train it
    Classifier _classifier(train_data, params );
    // print classifier output
    std::cout << " -- Classifier " << class_name << " : " << std::endl;
    std::cout << train_data.size() << " samples, " << train_data.getattributesNumber()
              << " attributes" << std::endl;  
    std::cout << test_data.size() <<  " samples, " << test_data.getattributesNumber()
              << " attributes" << std::endl;  
    
    
    // measure :
    // ========================================

    auto c_end = clock();
    auto t_end = std::chrono::high_resolution_clock::now();
    
              
    // Error counter
    uint32_t error(0);
    // Get attribute tag
    attribute_tag target_tag(test_data.getattributes().get_target_tag());
    // Test each sample on the training set
    print_classifier_summary("train", train_data, _classifier);

    //std::cout << _classifier << std::endl;

    // Test each sample on the testing set
    for (uint32_t i = 0; i < test_data.size(); ++i)
    {

      try{
      // Get target value on the test data
      //attribute test_attr(*(test_data.begin(i) + target_tag));
      attribute test_attr(test_data.getattribute(i,target_tag));

      attribute test_value(test_data.getattributes().getValue(target_tag,test_attr) );

      // Classify the data
      attribute class_attr(
          _classifier.classify(test_data.begin(i), test_data.end(i)));
      // Count error
      if (test_attr.discrete() != class_attr.discrete())
        ++error;
      }
      catch (std::exception &e)
      {
        std::cout << "[=] Exception: " << e.what() << std::endl;
        ++error;
      }
      
    }
    // Print summary

    print_classifier_summary("test", test_data, _classifier);

 
      c_end = clock();
      t_end = std::chrono::high_resolution_clock::now();

    std::cout << "[+] CPU time elapsed in s: "
              << (real_t)(c_end - c_start) / CLOCKS_PER_SEC << std::endl;
    std::cout << "[+] Wall time elapsed in s: "
              << std::chrono::duration<real_t>(t_end - t_start).count()
              << std::endl;
     std::cout << "[+]Number of errors = " << error << std::endl;
    std::cout << "[+]Number of test samples = " << test_data.size()
              << std::endl;
    std::cout << "[+]Error rate = " << 100 * error / test_data.size()  
              << " %" << std::endl; 
  }
    
} // namespace provallo

#endif /* DECISION_ENGINE_CLASSIFIER_H_ */
