/*
 * parameters.h
 *
 *  Created on: May 30, 2023
 *      Author: kardon
 * 
 *  This file is part of the Provallo project and is licensed under the
 */

#ifndef DECISION_ENGINE_PARAMETERS_H_
#define DECISION_ENGINE_PARAMETERS_H_
#include <iostream>
#include <map>

#include "utils.h" //real_t
namespace provallo
{
  // Enum for classifier type
  enum classifier_type
  {
    CNONE = 0,
    TREE = 1,
    DTREE = 2,
    RTREE = 3,
    ENSEMBLE = 4,
    RFORE = 5,
    BOOST = 6,
    BAYES = 7,
    METRIC = 8,
    NNEIG = 9,
    KMEANS = 10,
    ISO_TREE = 11,
    ISO_FOREST = 12,
    KD_TREE = 13,
    LIGHT_GBM = 14,
    XGBOOST = 15,
    CATBOOST = 16,
    SVM = 17,
    LOG_REG = 18,
    KNN = 19,
    MLP = 20,
    LINEAR_REG = 21,
    PERCEPTRON = 22,
    SGD = 23,
    GA = 24,
    QDA = 25,
    LDA = 26,
    GNB = 27,
    GMM = 28,
    HMM = 29,
    DBSCAN = 30,
    OPTICS = 31,
    KMEANS_PLUS_PLUS = 32,
    KMEANS_PARALLEL = 33,
    KMEANS_LLOYD = 34,
    KMEANS_LLOYD_PARALLEL = 35,
    CLASSIFIER_MAX = 36
  };
  // Base class for parameters
  class parameter_base
  {
  public:
    parameter_base()
    {
    }
    parameter_base(const parameter_base &other)
    {
      if (this == &other)
      {
        throw std::runtime_error(" parameter_base - Self assignment");  
      } 
    }
    parameter_base(parameter_base &&other)=default; 
    const parameter_base &
    operator=(const parameter_base &other) 
    {
      if (this == &other)
      {
        throw std::runtime_error(" parameter_base - Self assignment");  
      } 
      return *this;
    } 

    const parameter_base &
    operator=(parameter_base &&other)
    {
      //nothing to move from other  
     
       if(this==&other)
        {
          throw std::runtime_error(" parameter_base - Self assignment");  
        } 
       return *this;
    }

    // Virtual function to print parameters
    virtual void
    print(std::ostream &out) const = 0;
    // Get type of parameter
    virtual classifier_type
    getType() const = 0;
    virtual ~parameter_base(){};
  };

  class none : virtual public parameter_base
  {
  public:
    none(){};
    void
    print(std::ostream &out) const
    {
      out << "No parameters" << std::endl;
    }

    static classifier_type
    _type()
    {
      return CNONE;
    }

    classifier_type
    getType() const
    {
      return _type();
    }

    ~none(){};
  };

  // Parameters for Random Tree
  class random_tree_param : public parameter_base
  {
    // Number of attributes to select on a split node
    uint32_t _rho;
    // Maximum level of the tree
    uint32_t _level;
    // Minimum gain
    real_t _min_gain;

  public:
    random_tree_param(uint32_t rho = 1, uint32_t level = 0, real_t min_gain = 0.0) : _rho(rho), _level(level), _min_gain(min_gain)
    {      
    }

    void
    print(std::ostream &out) const 
    {

      out << "Number of attributes to select on a split node: " << _rho << std::endl;
      out << "Maximum level of the tree: " << _level << std::endl;
      out << "Minimum gain: " << _min_gain << std::endl;
      out << std::endl;
    }

    static classifier_type
    _type()
    {
      return RTREE;
    }

    classifier_type
    getType() const
    {
      return _type();
    }

    uint32_t
    getLevel() const
    {
      return _level;
    }

    real_t
    getMinGain() const
    {
      return _min_gain;
    }

    uint32_t
    getRho() const
    {
      return _rho;
    }

    ~random_tree_param(){};
  };


  // Parameters for light gbm
  class lightgbm_param : public parameter_base
  {
    // Number of attributes to select on a split node
    uint32_t _rho;
    // Maximum level of the tree
    uint32_t _level;
    // Minimum gain
    real_t _min_gain; 
    // Number of leaves
    uint32_t _num_leaves;
    // Number of iterations
    uint32_t _num_iterations;
    // Learning rate
    real_t _learning_rate;
    // Number of threads
    uint32_t _num_threads;
    // Number of classes
    uint32_t _num_class;
    // Number of bins
    uint32_t _num_bins;
    // Number of trees
    uint32_t _num_trees;
     // Number of features
    uint32_t _num_features;
    // Number of data
    uint32_t _num_data;
    // Number of data per leaf
    uint32_t _num_data_per_leaf;
    // Number of data per group
    uint32_t _num_data_per_group;
    // Number of groups
    uint32_t _num_groups;
    // Number of splits
    uint32_t _num_splits;

    //
    
  public:
    //constructor 
    lightgbm_param(uint32_t rho = 1, uint32_t level = 0, real_t min_gain = 0.0, uint32_t num_leaves = 31, uint32_t num_iterations = 100, real_t learning_rate = 0.1, uint32_t num_threads = 0, uint32_t num_class = 1, uint32_t num_bins = 255, uint32_t num_trees = 1,  uint32_t num_features = 0, uint32_t num_data = 0, uint32_t num_data_per_leaf = 0, uint32_t num_data_per_group = 0, uint32_t num_groups = 0, uint32_t num_splits = 0) : _rho(rho), _level(level), _min_gain(min_gain), _num_leaves(num_leaves), _num_iterations(num_iterations), _learning_rate(learning_rate), _num_threads(num_threads), _num_class(num_class), _num_bins(num_bins), _num_trees(num_trees),  _num_features(num_features), _num_data(num_data), _num_data_per_leaf(num_data_per_leaf), _num_data_per_group(num_data_per_group), _num_groups(num_groups), _num_splits(num_splits)
    {

    }
    static classifier_type
    _type()
    {
      return LIGHT_GBM;
    }

  //print 
    void print(std::ostream& out)const
    {
      out<<"Number of attributes to select on a split node: "<<_rho<<std::endl;
      out<<"Maximum level of the tree: "<<_level<<std::endl;
      out<<"Minimum gain: "<<_min_gain<<std::endl;
      out<<"Number of leaves: "<<_num_leaves<<std::endl;
      out<<"Number of iterations: "<<_num_iterations<<std::endl;
      out<<"Learning rate: "<<_learning_rate<<std::endl;
      out<<"Number of threads: "<<_num_threads<<std::endl;
      out<<"Number of classes: "<<_num_class<<std::endl;
      out<<"Number of bins: "<<_num_bins<<std::endl;
      out<<"Number of trees: "<<_num_trees<<std::endl;
      out<<"Number of features: "<<_num_features<<std::endl;
      out<<"Number of data: "<<_num_data<<std::endl;
      out<<"Number of data per leaf: "<<_num_data_per_leaf<<std::endl;
      out<<"Number of data per group: "<<_num_data_per_group<<std::endl;
      out<<"Number of groups: "<<_num_groups<<std::endl;
      out<<"Number of splits: "<<_num_splits<<std::endl;
      out<<"Number of leaves: "<<_num_leaves<<std::endl;
      out<<std::endl;

    }
    //get type
    classifier_type getType() const
    {
      return _type();
    }
    //get level
    uint32_t getLevel() const
    {
      return _level;
    }
    //get min gain
    real_t getMinGain() const
    {
      return _min_gain;
    }
    //get rho
    uint32_t getRho() const
    {
      return _rho;
    }
    //get num leaves
    uint32_t getNumLeaves() const
    {
      return _num_leaves;
    }
    //get num iterations
    uint32_t getNumIterations() const
    {
      return _num_iterations;
    }
    //get learning rate
    real_t getLearningRate() const
    {
      return _learning_rate;
    }
    //get num threads
    uint32_t getNumThreads() const
    {
      return _num_threads;
    }
    //get num class
    
    virtual   ~lightgbm_param(){};
  //  
  };

  // Parameters for XGBoost
  class xgboost_param : public parameter_base 
  {
    // Number of attributes to select on a split node
    uint32_t _rho;
    // Maximum level of the tree
    uint32_t _level;
    // Minimum gain
    real_t _min_gain;
    // Number of leaves
    uint32_t _num_leaves;
    // Number of iterations
    uint32_t _num_iterations;
    // Learning rate
    real_t _learning_rate;
    // Number of threads
    uint32_t _num_threads;
    // Number of classes
    uint32_t _num_class;
    // Number of bins
    uint32_t _num_bins;
    // Number of trees
    uint32_t _num_trees;
    // Number of features
    uint32_t _num_features;
    // Number of data
    uint32_t _num_data;
    // Number of data per leaf
    uint32_t _num_data_per_leaf;
    // Number of data per group
    uint32_t _num_data_per_group;
    // Number of groups
    uint32_t _num_groups;
    // Number of splits
    uint32_t _num_splits;

  public:

    //constructor
    xgboost_param(uint32_t rho = 1, uint32_t level = 0, real_t min_gain = 0.0, uint32_t num_leaves = 31, uint32_t num_iterations = 100, real_t learning_rate = 0.1, uint32_t num_threads = 0, uint32_t num_class = 1, uint32_t num_bins = 255, uint32_t num_trees = 1, uint32_t num_features = 0, uint32_t num_data = 0, uint32_t num_data_per_leaf = 0, uint32_t num_data_per_group = 0, uint32_t num_groups = 0, uint32_t num_splits = 0) : _rho(rho), _level(level), _min_gain(min_gain), _num_leaves(num_leaves), _num_iterations(num_iterations), _learning_rate(learning_rate), _num_threads(num_threads), _num_class(num_class), _num_bins(num_bins), _num_trees(num_trees), _num_features(num_features), _num_data(num_data), _num_data_per_leaf(num_data_per_leaf), _num_data_per_group(num_data_per_group), _num_groups(num_groups), _num_splits(num_splits)
    {
      //check
      if(_rho == 0)
      {
        throw std::invalid_argument("xgboost_param: rho must be greater than 0");
      }
      if(_level == 0)
      {
        throw std::invalid_argument("xgboost_param: level must be greater than 0");
      }
      if(_min_gain < 0.0)
      {
        throw std::invalid_argument("xgboost_param: min_gain must be greater than or equal to 0.0");
      }
      if(_num_leaves == 0)
      {
        throw std::invalid_argument("xgboost_param: num_leaves must be greater than 0");
      }
      if(_num_iterations == 0)
      {
        throw std::invalid_argument("xgboost_param: num_iterations must be greater than 0");
      }

      if(_learning_rate <= 0.0)
      {
        throw std::invalid_argument("xgboost_param: learning_rate must be greater than 0.0");
      }


    }
    static classifier_type
    _type()
    {
      return XGBOOST;
    }
    //print

    void print(std::ostream& out)const
    {
      out<<"Number of attributes to select on a split node: "<<_rho<<std::endl;
      out<<"Maximum level of the tree: "<<_level<<std::endl;
      out<<"Minimum gain: "<<_min_gain<<std::endl;
      out<<"Number of leaves: "<<_num_leaves<<std::endl;
      out<<"Number of iterations: "<<_num_iterations<<std::endl;
      out<<"Learning rate: "<<_learning_rate<<std::endl;
      out<<"Number of threads: "<<_num_threads<<std::endl;
      out<<"Number of classes: "<<_num_class<<std::endl;
      out<<"Number of bins: "<<_num_bins<<std::endl;
      out<<"Number of trees: "<<_num_trees<<std::endl;
      out<<"Number of features: "<<_num_features<<std::endl;
      out<<"Number of data: "<<_num_data<<std::endl;
      out<<"Number of data per leaf: "<<_num_data_per_leaf<<std::endl;
      out<<"Number of data per group: "<<_num_data_per_group<<std::endl;
      out<<"Number of groups: "<<_num_groups<<std::endl;
      out<<"Number of splits: "<<_num_splits<<std::endl;
      out<<"Number of leaves: "<<_num_leaves<<std::endl;
      out<<std::endl;

    } 
    virtual   ~xgboost_param(){};
  //
    //get type
    classifier_type
    getType() const
    {
      return _type();
    }

    uint32_t getRho() const
    {
      return _rho;
    }

    uint32_t getLevel() const
    {
      return _level;
    } 

    real_t getMinGain() const
    {
      return _min_gain;
    }

    
    uint32_t getNumIterations() const
    {
      return _num_iterations;
    }

    real_t getLearningRate() const
    {
      return _learning_rate;
    }

    uint32_t getNumThreads() const
    {
      return _num_threads;
    }

    // Number of classes
    
    // Number of bins
    uint32_t getNumBins() const
    {
      return _num_bins;
    }
    // Number of trees
    uint32_t getNumTrees() const
    {
      return _num_trees;
    }
    // Number of features
    uint32_t getNumFeatures() const
    {
      return _num_features;
    }
    // Number of data
    uint32_t getNumData() const
    {
      return _num_data;
    }
    // Number of data per leaf
    uint32_t getNumDataPerLeaf() const
    {
      return _num_data_per_leaf;
    }
    // Number of data per group
    
    uint32_t getNumDataPerGroup() const
    {
      return _num_data_per_group;
    }
    // Number of groups
    uint32_t getNumGroups() const
    {
      return _num_groups;
    }
    // Number of splits
    uint32_t getNumSplits() const
    {
      return _num_splits;
    }
    // Number of leaves
    uint32_t getNumLeaves() const
    {
      return _num_leaves;
    }
    
    //use matrix<real_t> inside xgboost/lightgbm algorith to transform the data instead of actually holding trees in memory. 
    //this is a matrix of probabilities that can be used to classify data points
    //the matrix is a transition matrix from a tree structure to a matrix of probabilities 
 
  };
  // Parameters for Random Forest
  class random_forest_param : public parameter_base
  {
    // Reference to parameters of the classifier
    const parameter_base &_parameters;
    // Number of classifiers

    uint32_t _nclass;

  public:
    random_forest_param(uint32_t nclass, const parameter_base &parameters) : _parameters(parameters), _nclass(nclass)
    {
    }

    void
    print(std::ostream &out) const
    {

      out << "Number of classifiers: " << _nclass << std::endl;
      out << "Parameters: " << std::endl;
      _parameters.print(out);
      out << std::endl;
    }

    static classifier_type
    _type()
    {
      return RFORE;
    }

    classifier_type
    getType() const
    {
      return _type();
    }

    uint32_t
    getClassifiersNumber() const
    {
      return _nclass;
    }

    const parameter_base &
    getParameters() const
    {
      return _parameters;
    }

    virtual ~random_forest_param()
    {
    }
  };
  // Parameters for adaboost
  class adaboost_param : public parameter_base
  {
    // Reference to parameters of the classifier
    const parameter_base &_parameters;
    // Maximum number of classifiers
    uint32_t _nclass;

  public:
    adaboost_param(uint32_t nclass, const parameter_base &parameters) : _parameters(parameters), _nclass(nclass)
    {

    }


    void
    print(std::ostream &out) const
    {
      out << "Number of classifiers: " << _nclass << std::endl;
      out << "Parameters for the classifiers : " << std::endl;
      _parameters.print(out);
    }

    static classifier_type
    _type()
    {
      return BOOST;
    }

    classifier_type
    getType() const
    {
      return _type();
    }

    uint32_t
    getClassifiersNumber() const
    {
      return _nclass;
    }

    const parameter_base &
    getParameters() const
    {
      return _parameters;
    }
    void set_nclass(uint32_t nclass)
    {
      _nclass = nclass;
    }

    virtual ~adaboost_param()
    {
    } 
  };

  // Parameters for metric classifier
  class metric_classifier_param : public parameter_base
  {
    // Number of neighbors
    uint32_t _neighbours;
    // Map of attribute's weights
    std::map<std::string, real_t> _weights;

  public:
    metric_classifier_param(uint32_t neighbours,
                            const std::map<std::string, real_t> &weights) : _neighbours(neighbours), _weights(weights)
    {

    }

    void set_weights(const std::map<std::string, real_t> &weights)
    {
      this->_weights.clear();
      _weights = weights;
    }
    void
    print(std::ostream &out) const
    {
      out << "Neighbours: " << _neighbours << std::endl;
      out << "Weights: " << std::endl;
      for (std::map<std::string, real_t>::const_iterator it = _weights.begin();
           it != _weights.end(); ++it)
      {
        out << it->first << " " << it->second << std::endl;
      }
    }

    static classifier_type
    _type()
    {
      return METRIC;
    }

    classifier_type
    getType() const
    {
      return _type();
    }

    uint32_t
    getNeighboursNumber() const
    {
      return _neighbours;
    }

    const std::map<std::string, real_t> &
    getWeights() const
    {
      return _weights;
    }

    ~metric_classifier_param()
    {
    }
  };

  // Parameters for Isolation Forest
  class isoforest_param : public parameter_base
  {
    // Number of trees
    uint32_t _ntrees;
    // Number of samples
    uint32_t _nsamples;
    // Maximum depth
    uint32_t _max_depth;
    // Subsampling size
    uint32_t _subsample_size;
    // Number of attributes to select on a split node
    uint32_t _rho;
    // Maximum level of the tree
    uint32_t _level;
    // Minimum gain
    real_t _min_gain;
    // Number of threads
    uint32_t _nthreads;
    // Seed
    uint32_t _seed;

  public:
    isoforest_param(uint32_t ntrees, uint32_t nsamples, uint32_t max_depth,
                    uint32_t subsample_size, uint32_t rho, uint32_t level,
                    real_t min_gain, uint32_t nthreads, uint32_t seed ) : _ntrees(ntrees), _nsamples(nsamples), _max_depth(max_depth),
                                                                        _subsample_size(subsample_size), _rho(rho), _level(level),
                                                                        _min_gain(min_gain), _nthreads(nthreads), _seed(seed)
    {
      check_parameters();
    }

    isoforest_param(const isoforest_param &cpy_) :parameter_base(), _ntrees(cpy_._ntrees), _nsamples(cpy_._nsamples), _max_depth(cpy_._max_depth),
                                                   _subsample_size(cpy_._subsample_size), _rho(cpy_._rho), _level(cpy_._level),
                                                   _min_gain(cpy_._min_gain), _nthreads(cpy_._nthreads), _seed(cpy_._seed)
    {
      check_parameters();
    }

    void set_ntrees(uint32_t ntrees)
    {
      _ntrees = ntrees;
      //  check_parameters();
    }
    void set_nsamples(uint32_t nsamples)
    {
      _nsamples = nsamples;
    }
    void set_max_depth(uint32_t max_depth)
    {
      _max_depth = max_depth;
    }
    void set_subsample_size(uint32_t subsample_size)
    {
      _subsample_size = subsample_size;
    }
    void set_rho(uint32_t rho)
    {
      _rho = rho;
    }
    void set_level(uint32_t level)
    {
      _level = level;
    }
    void set_min_gain(real_t min_gain)
    {
      _min_gain = min_gain;
    }
    void set_nthreads(uint32_t nthreads)
    {
      _nthreads = nthreads;
    }
    void set_seed(uint32_t seed)
    {
      _seed = seed;
    }

    void check_parameters()
    {

      // Check the number of trees
      if (_ntrees < 1)
      {
        throw std::runtime_error("The number of trees must be greater than 0");
      }
      // Check the number of samples
      if (_nsamples < 1)
      {
        throw std::runtime_error("The number of samples must be greater than 0");
      }
      // Check the maximum depth
      if (_max_depth < 1)
      {
        throw std::runtime_error("The maximum depth must be greater than 0");
      }
      // Check the subsample size
      if (_subsample_size < 1)
      {
        throw std::runtime_error("The subsample size must be greater than 0");
      }
      // Check the number of attributes to select on a split node
      if (_rho < 1)
      {
        throw std::runtime_error("The number of attributes to select on a split node must be greater than 0");
      }
      // Check the maximum level of the tree
      if (_level < 1)
      {
        throw std::runtime_error("The maximum level of the tree must be greater than 0");
      }
      // Check the minimum gain
      if (_min_gain < 0)
      {
        throw std::runtime_error("The minimum gain must be greater than 0");
      }
      // Check the number of threads
      if (_nthreads < 1)
      {
        throw std::runtime_error("The number of threads must be greater than 0");
      }
    }
    static classifier_type
    _type()
    {
      return ISO_FOREST;
    }

    classifier_type
    getType() const
    {
      return _type();
    }

    uint32_t get_ntrees() const
    {
      return _ntrees;
    }
    uint32_t get_nsamples() const
    {
      return _nsamples;
    }
    uint32_t get_max_depth() const
    {
      return _max_depth;
    }
    uint32_t get_subsample_size() const
    {
      return _subsample_size;
    }
    uint32_t get_rho() const
    {
      return _rho;
    }
    uint32_t get_level() const
    {
      return _level;
    }
    real_t get_min_gain() const
    {
      return _min_gain;
    }
    uint32_t get_nthreads() const
    {
      return _nthreads;
    }
    uint32_t get_seed() const
    {
      return _seed;
    }
    void print(std::ostream &out) const
    {
      out<<"Isolation Forest Parameters: "<<std::endl;
      out<<"Number of Trees: "<<_ntrees<<std::endl;
      out<<"Number of Samples: "<<_nsamples<<std::endl;
      out<<"Maximum Depth: "<<_max_depth<<std::endl;
      out<<"Subsample Size: "<<_subsample_size<<std::endl;
      out<<"Number of Attributes to Select on a Split Node: "<<_rho<<std::endl;
      out<<"Maximum Level of the Tree: "<<_level<<std::endl;
      out<<"Minimum Gain: "<<_min_gain<<std::endl;
      out<<"Number of Threads: "<<_nthreads<<std::endl;

    }

  }; // end of class isoforest_param
  
  // Nearest Neighbor Parameters

  class nearest_neighbor_param : public parameter_base
  {
    uint32_t _neighbours;
    std::map<std::string, real_t> _weights;

  public:
    nearest_neighbor_param(uint32_t neighbours) : _neighbours(neighbours)
    {
    }

    nearest_neighbor_param(uint32_t neighbours,
                           const std::map<std::string, real_t> &weights) : _neighbours(neighbours), _weights(weights)
    {

    }

    void
    print(std::ostream &out) const
    {
      out << "Nearest Neighbor Parameters: " << std::endl;
      out << "Number of Neighbors: " << _neighbours << std::endl;
      out << "Weights: " << std::endl;
      out << "Attribute"
          << " "
          << "Weight" << std::endl;
      out << "---------"
          << " "
          << "------" << std::endl;
      out << "---------"
          << " "
          << "------" << std::endl;
      for (std::map<std::string, real_t>::const_iterator it =
               _weights.begin();
           it != _weights.end(); ++it)
      {
        out << it->first << " " << it->second << std::endl;
      }
    }
    static classifier_type
    _type()
    {
      return NNEIG;
    }

    classifier_type
    getType() const
    {
      return _type();
    }

    uint32_t
    getNeighboursNumber() const
    {
      return _neighbours;
    }

    const std::map<std::string, real_t> &
    getWeights() const
    {
      return _weights;
    }

    ~nearest_neighbor_param()
    {
    }
  }; 

  // Parameters for k-means classifier
  class kmeans_param : public parameter_base
  {
    // Number of internal points
    uint32_t _points;
    // Map of attribute's weights
    std::map<std::string, real_t> _weights;

  public:

    kmeans_param(size_t points) : _points(points)
    {

    }

    kmeans_param(size_t points, const std::map<std::string, real_t> &weights) : _points(points), _weights(weights)
    {

    }
 
    void
    print(std::ostream &out) const
    {
      out << "points: " << _points << std::endl;
      out << "weights: " << std::endl;
      for (std::map<std::string, real_t>::const_iterator it = _weights.begin(); it != _weights.end(); ++it)
      {
        out << "  " << it->first << ": " << it->second << std::endl;
      }
    } 
    static classifier_type
    _type()
    {
      return KMEANS;
    }

    classifier_type
    getType() const
    {
      return _type();
    }

    uint32_t
    getPointsNumber() const
    {
      return _points;
    }

    const std::map<std::string, real_t> &
    getWeights() const
    {
      return _weights;
    }

    virtual ~kmeans_param()
    {
    }



  };

} /* namespace provallo */

#endif /* DECISION_ENGINE_PARAMETERS_H_ */
