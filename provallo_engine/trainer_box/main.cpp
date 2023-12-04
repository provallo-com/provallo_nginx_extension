#include "../decision_engine/split_utils.hpp"
#include "../decision_engine/classifier.h"
#include "../decision_engine/dataset.h"
#include "../decision_engine/neuralhelper.h"
#include "../decision_engine/kNearestNeighbor.h"
#include "../decision_engine/kdt.h"
#include "../decision_engine/parameters.h"
#include "../decision_engine/pipelinebuilder.h"
#include "../decision_engine/autoencoder.h"
#include "../decision_engine/bit_vector_attribute.h"
#include "../decision_engine/tree_classifiers.h"
#include "../decision_engine/info_helper.h"
#include "../third_party/sqlite.h"

#include <dirent.h>
#include <filesystem>

#include <iostream>
#include <string>
#include <fstream>
#include <chrono>

#include <vector>
#include <map>
#include <algorithm>
#include <iterator>
// std::filesystem::path
#include <sys/stat.h>

#include <math.h>
#include "../helpers/configuration_helper.h"
provallo::classifier *_lastclassifier = nullptr;
provallo::isolation_forest *iso_last_fit = nullptr;

const int kNumThreads = 8; // number of threads to use for kNN classification
std::atomic_uint64_t numQueriesProcessed(0);
std::atomic_uint64_t correctCount(0);
std::atomic_uint64_t errorsCount(0);

std::mutex queryLock; // lock for global counters

// fwd :
provallo::learning_task::task_configuration build_task_from_source(provallo::names_source &src);
const char *vectorizer_types[] = {
    "TFIDF",
    "STANDARD_SCALER",
    "MIN_MAX_SCALER",
    "PCA", "ONE_HOT_VECTORIZER",
    "NEURAL_TRANSFORMER",
    "AERONATIC_QARTERION",
    "SVD_OPERATOR",
    "NGRAM_HMM_TRANSFORMER",
    "HPLANE_TRANSFORMER",
    "HUFFMAN_TRANSFORMER",
    "HMM_TRANSFORMER", "REGRESSION_TRANSFORMER",
    "UMAP_VECTORIZER", "TSNE_VECTORIZER", "AUTOENCODER_VECTORIZER",
    "LDA_VECTORIZER", "UNKNOWN_VECTORIZER", "ERROR_INDEX"};

bool test_fit_iso_forest();
bool test_dataset_load();
void test_spike_train_generator();

template <size_t N>
std::vector<provallo::point<N>, int> load_dataset_as_points_and_lables(const std::string &filename)
{
  // dynamically create point<N> and labels vector from file:

  std::vector<provallo::point<N>, int> points;
  std::ifstream file(filename, std::ios::in);
  std::string line;
  while (std::getline(file, line))
  {
    std::istringstream iss(line);
    std::string token;
    provallo::point<N> p;
    int i = 0;
    int label = 0;
    while (std::getline(iss, token, ','))
    {
      if (i < N)
      {
        p[i] = std::stod(token);
      }
      else
      {
        label = std::stoi(token);
      }
      i++;
    }
    points.push_back(std::make_pair(p, label));
  }
  return points;
}

void test_matrix()
{
  provallo::matrix<real_t> m(3, 3), m2(3, 3);

  m(0, 0) = 1;
  m(0, 1) = 2;
  m(1, 0) = 3;
  m(1, 1) = 4;

  m(2, 0) = 5;
  m(2, 1) = 6;
  m(2, 2) = 7;
  m(0, 2) = 8;
  m(1, 2) = 9;
  m(2, 2) = 10;

  m2(0, 0) = 10;
  m2(0, 1) = 9;
  m2(1, 0) = 8;
  m2(1, 1) = 7;

  m2(2, 0) = 6;
  m2(2, 1) = 5;
  m2(2, 2) = 4;
  m2(0, 2) = 3;
  m2(1, 2) = 2;
  m2(2, 2) = 1;

  // test matrix multiplication
  std::cout << "m1 : " << std::endl;
  std::cout << m << std::endl;
  std::cout << "m2 : " << std::endl;
  std::cout << m2 << std::endl;
  std::cout << "m1 * m2 : " << std::endl;
  std::cout << provallo::matrix<real_t>(m * m2) << std::endl;
  std::cout << "m1 + m2 : " << std::endl;
  std::cout << m + m2 << std::endl;
  std::cout << "m1 - m2 : " << std::endl;
  std::cout << m - m2 << std::endl;
  std::cout << "m1 / m2 : " << std::endl;
  std::cout << m / m2 << std::endl;
  std::cout << "m1 * 2 : " << std::endl;
  std::cout << m * 2. << std::endl;
  std::cout << "m1 + 2 : " << std::endl;
  std::cout << m + 2. << std::endl;
  std::cout << "m1 - 2 : " << std::endl;
  std::cout << m - 2. << std::endl;
  std::cout << "m1 / 2 : " << std::endl;
  std::cout << m / 2. << std::endl;
  std::cout << "m1 * m2 : " << std::endl;
  std::cout << m * m2 << std::endl;

  provallo::matrix<real_t> m3 = m * m2;
  std::cout << "m3 : " << std::endl;
  std::cout << m3 << std::endl;

  // transpose
  std::cout << "m1 : " << std::endl;
  std::cout << m << std::endl;
  std::cout << "m1 transpose : " << std::endl;
  std::cout << m.transpose() << std::endl;
  std::cout << "m1 transpose transpose : " << std::endl;
  std::cout << m.transpose().transpose() << std::endl;
  std::cout << "m1 transpose transpose transpose : " << std::endl;
  std::cout << m.transpose().transpose().transpose() << std::endl;
  std::cout << "m1 transpose transpose transpose transpose : " << std::endl;
  std::cout << m.transpose().transpose().transpose().transpose() << std::endl;

  std::vector<real_t> v = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  provallo::matrix<real_t> m4 = m * v;
  std::cout << "m4 : " << std::endl;
  std::cout << m4 << std::endl;


  //unit test for save/load
  //save m4 to file
  std::string filename = "m4.tmp";
  std::ofstream tmp(filename, std::ios::out | std::ios::binary);
  tmp<<m4;
  tmp.close();
  //load m5 from  m4.tmp file
  provallo::matrix<real_t> m5;
  std::ifstream intmp(filename, std::ios::in | std::ios::binary);
  intmp>>m5;
  intmp.close();
  std::cout << "m5 : " << std::endl;
  std::cout << m5 << std::endl;
  //delete tmp file
  std::remove(filename.c_str());
 
}

// alternative to dataset load
provallo::matrix<real_t> read_data_file(const std::string &filepath)
{

  std::cout << "[+] reading data file : " << filepath << std::endl;

  // use bag of words to transform strings to numbers
  try
  {

    std::ifstream file(filepath, std::ios::in | std::ios::binary | std::ios::ate);

    std::string line;
    // for each feature(column) we have a map of values and counts
    // for each row we have a map of features and values
    std::map<size_t /*index*/, std::map<std::string /*value*/, size_t /*count*/>> BoW;
    std::map<size_t /*index*/, std::map<std::string /*value*/, size_t /*count*/>> BoW_labels;
    std::map<size_t /*index*/, std::map<std::string /*value*/, size_t /*count*/>> BoW_features;

    std::vector<provallo::bit_type<uint32_t, 17>> types;

    std::vector<std::string> labels;
    std::vector<std::string> features;
    std::vector<std::vector<std::string>> data;
    std::vector<std::string> row;
    // 17 default types than can be mapped to each cell
    /*
    bool is_label = false;
    bool is_feature = false;
    bool is_discrete = false;
    bool is_continous = false;
    bool is_vector = false;
    bool is_matrix = false;
    bool is_string = false;
    bool is_bool = false;
    bool is_date = false;
    bool is_time = false;
    bool is_datetime = false;
    bool is_categorical = false;
    bool is_ordinal = false;
    bool is_nominal = false;
    bool is_interval = false;
    bool is_ratio = false;
    bool is_count = false;
    */

    while (std::getline(file, line))
    {
      std::stringstream lineStream(line);

      std::string cell;
      size_t column_index = 0;

      while (std::getline(lineStream, cell, ','))
      {
        // check if cell is empty
        // trim cell
        column_index++;
        cell.erase(std::remove_if(cell.begin(), cell.end(), [](char c)
                                  { return std::isspace(c); }),
                   cell.end());
        if (cell.empty())
        {
          continue;
        }
        // check if cell is a label / discrete value or a feature / continous value
        // start :

        // bit vector of 18 bits:
        // label,feature,discrete,continous,vector,matrix,string,bool,date,time,datetime,categorical,ordinal,nominal,interval,ratio,count:

        std::string continous_s = "^(( )*|([0-9]+(\\.[0-9]+)?))$";
        std::string discrete_s = "^(( )*|([0-9]+))$";
        std::string label_s = "^(( )*|([0-9]+))$";
        std::string feature_s = "^(( )*|([0-9]+(\\.[0-9]+)?))$";
        std::string vector_s = "^(( )*|([0-9]+(\\.[0-9]+)?))$";
        std::string matrix_s = "^(( )*|([0-9]+(\\.[0-9]+)?))$";
        std::string string_s = "^$";
        // TRUE or true or FALSE or false or 0 or 1 :
        std::string bool_s = " ^(( )*|([0-1]+))$";
        // date format : YYYY-MM-DD
        std::string date_s = "\\d{4}-[01]\\d-[0-3]\\dT[0-2]\\d:[0-5]\\d:[0-5]\\d(?:\\.\\d+)?Z?";
        // time format : HH:MM:SS.mmmmmm
        std::string time_s = "\\d{2}:\\d{2}:\\d{2}\\.\\d{6}";
        // datetime format : YYYY-MM-DDTHH:MM:SS.mmmmmm
        std::string datetime_s = "\\d{4}-[01]\\d-[0-3]\\dT[0-2]\\d:[0-5]\\d:[0-5]\\d(?:\\.\\d+)?Z?";
        // categorical,ordinal,nominal,interval,ratio,count
        std::string categorical_s = "^(( )*|([0-1]+))$";
        std::string ordinal_s = "^(( )*|([0-1]+))$";
        std::string nominal_s = "^(( )*|([0-1]+))$";
        std::string interval_s = "^(( )*|([0-1]+))$";
        std::string ratio_s = "^(( )*|([0-9]+))$/^(( )*|([0-9]+(\\.[0-9]+)?))$";
        std::string count_s = "^(( )*|([0-9]+))$";

        std::regex label_regex(label_s.c_str(), std::regex_constants::icase);
        std::regex feature_regex(feature_s.c_str(), std::regex_constants::icase);
        std::regex discrete_regex(discrete_s.c_str(), std::regex_constants::icase);
        std::regex continous_regex(continous_s.c_str(), std::regex_constants::icase);
        std::regex vector_regex(vector_s.c_str(), std::regex_constants::icase);
        std::regex matrix_regex(matrix_s.c_str(), std::regex_constants::icase);
        std::regex string_regex(string_s.c_str(), std::regex_constants::icase);
        std::regex bool_regex(bool_s, std::regex_constants::icase);
        std::regex date_regex(date_s, std::regex_constants::icase);
        std::regex time_regex("time", std::regex_constants::icase);
        std::regex datetime_regex("datetime", std::regex_constants::icase);
        std::regex categorical_regex("categorical", std::regex_constants::icase);
        std::regex ordinal_regex("ordinal", std::regex_constants::icase);
        std::regex nominal_regex("nominal", std::regex_constants::icase);
        std::regex interval_regex("interval", std::regex_constants::icase);
        std::regex ratio_regex("ratio", std::regex_constants::icase);
        std::regex count_regex("count", std::regex_constants::icase);

        // check and update column information

        provallo::bit_type<uint32_t, 17> type;
        if (types.size() < column_index)
        {
          types.push_back(type);
        }
        else
        {
          type = types[column_index - 1];
        }

        // end
        // check and update column information
        if (std::regex_search(cell, label_regex))
        {
          //        is_label = true;
          type.flip(0);
        }
        if (std::regex_search(cell, feature_regex))
        {
          type.flip(1);
          //     is_feature = true;
        }
        if (std::regex_search(cell, discrete_regex))
        {
          type.flip(2);

          //   is_discrete = true;
          //    is_continous=false;
        }
        if (std::regex_search(cell, continous_regex))
        {
          type.flip(3);
          //      is_continous = true;
          //       is_discrete=false;
        }
        if (std::regex_search(cell, vector_regex))
        {
          type.flip(4);
          //       is_vector = true;
        }
        if (std::regex_search(cell, matrix_regex))
        {
          type.flip(5);
          //                is_matrix = true;
        }
        if (std::regex_search(cell, string_regex))
        {
          type.flip(6);
          //     is_string = true;
        }
        if (std::regex_search(cell, bool_regex))
        {
          type.flip(7);
          //      is_bool = true;
        }
        if (std::regex_search(cell, date_regex))
        {
          type.flip(8);
          //   is_date = true;
        }
        if (std::regex_search(cell, time_regex))
        {
          type.flip(9);
          //           is_time = true;
        }
        if (std::regex_search(cell, datetime_regex))
        {
          type.flip(10);
          //       is_datetime = true;
        }
        if (std::regex_search(cell, categorical_regex))
        {
          type.flip(11);
          //     is_categorical = true;
        }
        if (std::regex_search(cell, ordinal_regex))
        {
          type.flip(12);
          //  is_ordinal = true;
        }
        if (std::regex_search(cell, nominal_regex))
        {
          type.flip(13);
          // is_nominal = true;
        }
        if (std::regex_search(cell, interval_regex))
        {
          type.flip(14);
          //  is_interval = true;
        }
        if (std::regex_search(cell, ratio_regex))
        {
          type.flip(15);
          // is_ratio = true;
        }
        if (std::regex_search(cell, count_regex))
        {
          type.flip(16);
          // is_count = true;
        }
        // end
        if (type.value() == 0)
        // set continous as default
        {
          type.flip(3);
          //  is_continous = true;
        }
        types[column_index - 1] = type;

        // end
        row.push_back(cell);
      }
      data.push_back(row);
      row.clear();
    }

    // now we can transform the data
    provallo::matrix<real_t> X(data.size(), data[0].size());
    size_t i = 0;
    for (const auto &row : data)
    {
      size_t j = 0;
      for (const auto &cell : row)
      {
        X(i, j) = std::stod(cell);
        j++;
      }
      i++;
    }
    return X;
  }
  catch (const std::exception &e)
  {
    std::cout << e.what() << std::endl;
    return provallo::matrix<real_t>(1, 1);
  }
  return provallo::matrix<real_t>(1, 1);
}

provallo::isolation_forest *isoforest_single(const provallo::attribute_information &attributes);

bool benchmark_classifiers(const std::string benchmark_folder = "./db/benchmarks");

bool fit_fuzzsb(const std::string &benchmark_folder = "/home/kardon/eclipse-workspace/fuzzdb");

ssize_t
which_max(std::vector<double> &v)
{
  auto loc_max_el = std::max_element(v.begin(), v.end());
  return std::distance(v.begin(), loc_max_el);
}

bool test_fit_iso_forest()
{
  /* Random data from a standard normal distribution
   (100 points generated randomly, plus 1 outlier added manually)
   Library assumes it is passed as a single-dimensional pointer,
   following column-major order (like Fortran) */
  int nrow = 101;
  int ncol = 2;
  std::vector<double> X(nrow * ncol);
  std::default_random_engine rng;

  std::normal_distribution<double> rnorm(0., 1.);

#define get_ix(row, col) (row + col * nrow)
  for (int col = 0; col < ncol; col++)
    for (int row = 0; row < 100; row++)
    {
      do
      {
        auto x = rnorm(rng);
        if (x == x && x != std::numeric_limits<double>::infinity() && x != -std::numeric_limits<double>::infinity())
        {
          X[get_ix(row, col)] = x;
          break;
        }
      } while (true);
    }

  /* Now add obvious outlier point (-3,-3) */
  X[get_ix(100, 0)] = -3;
  X[get_ix(100, 1)] = -3;
  provallo::isolation_forest iso = provallo::isolation_forest();
  iso.fit(X.data(), nrow, ncol);
  // check if the outlier is detected

  std::vector<double> outlier_scores = iso.predict(X.data(), nrow, false);
  int row_highest = which_max(outlier_scores);
  double x = X[get_ix(row_highest, 0)];
  double y = X[get_ix(row_highest, 1)];
  if (x != y || x != -3.0)
  {
    std::cout << "[-] isoforest results :" << std::to_string(x) << "," << std::to_string(y) << std::endl;
    // maybe max element is not the outlier
    for (size_t i = 0; i < outlier_scores.size(); i++)
    {
      std::cout << "[-] outlier score " << std::to_string(i) << " : " << std::to_string(outlier_scores[i]) << std::endl;
    }
  }

  std::cout << "[+]Provallo IsoForest test" << std::endl;
  std::cout << std::string("Point with highest outlier score: [")
            << x << std::string(", ")
            << y << std::string("]") << std::endl;

  std::cout.flush();
  std::getchar();
  return true;
}
// fitting ISO FORESTS

provallo::isolation_forest *isoforest_single(const provallo::attribute_information &attributes)
{

  static const provallo::attribute_information &att_stat = attributes;
  // fitting ISO FORESTS
  /*
         size_t ndim = attributes.getGroups ().size ();
                     size_t ntry =10;
                     provallo::CoefType coef_type = provallo::CoefType::Uniform;
                     bool coef_by_prop = false;
                     bool with_replacement = false;
                     bool weight_as_sample = true;
                     size_t sample_size = attributes.getSize();
                     size_t ntrees = 100;
                     size_t max_depth = 50;
                     size_t ncols_per_tree = 50;
                     bool limit_depth = false;
                     bool penalize_range = true;
                     bool standardize_datam = true;

                     provallo::ScoringMetric scoring_metric = provallo::ScoringMetric::BoxedDensity2;
                     bool fast_bratio = false;

                     bool weigh_by_kurt = true;
                     double prob_pick_by_gain_pl =false;
                     double prob_pick_by_gain_avg = false;
                     double prob_pick_by_full_gain = true;
                     double prob_pick_by_dens = false;
                     double prob_pick_col_by_range = false;
                     double prob_pick_col_by_var = true;
                     double prob_pick_col_by_kurt = true;
                     double min_gain =0.;

                     provallo::MissingAction missing_action = provallo::MissingAction::Impute;

                     provallo::CategSplit cat_split_type = provallo::CategSplit::SubSet;

               provallo::NewCategAction new_cat_action = provallo::NewCategAction::Weighted;
               bool all_perm = false;
               bool build_imputer = true;
               size_t min_imp_obs =0;
               provallo::UseDepthImp depth_imp = provallo::UseDepthImp::Lower;
               provallo::WeighImpRows weigh_imp_rows =provallo::WeighImpRows::Flat;
               uint64_t random_seed = 0ull;
               int nthreads = 50;

  */
  static provallo::isolation_forest *forest = new provallo::isolation_forest();

  /*
                 ndim,ntry,coef_type ,coef_by_prop ,with_replacement ,weight_as_sample ,sample_size ,ntrees ,max_depth,
                                 ncols_per_tree,
                                 limit_depth,
                                 penalize_range ,
                                 standardize_datam ,
                                 scoring_metric,
                                 fast_bratio ,
                                 weigh_by_kurt ,
                                 prob_pick_by_gain_pl,
                                 prob_pick_by_gain_avg ,prob_pick_by_full_gain,
         prob_pick_by_dens ,prob_pick_col_by_range ,
         prob_pick_col_by_var ,prob_pick_col_by_kurt ,min_gain,
         missing_action,
         cat_split_type,
         new_cat_action,
                           all_perm ,build_imputer ,min_imp_obs,
                           depth_imp,
                           weigh_imp_rows ,
                           random_seed ,
                                   nthreads );

  */
  // ignore forest hyper parameters for now
  if (attributes == att_stat)
  {
    return forest;
  }
  else
  {
    return new provallo::isolation_forest();
  }
}
std::vector<std::string> getFilesInFolder(const std::string &benchmark_folder)
{
  std::vector<std::string> files;
  // only c++2a has std::filesystem
  // std directory iterator
  // use c-style dirent

  DIR *dir;
  struct dirent *ent;

  if ((dir = opendir(benchmark_folder.c_str())) != NULL)
  {
    /* print all the files and directories within directory */
    while ((ent = readdir(dir)) != NULL)
    {
      std::string file_name = ent->d_name;
      if (file_name.find(".names") != std::string::npos)
      {
        files.push_back(file_name);
      }
    }
    closedir(dir);
  }
  else
  {
    /* could not open directory */
    perror("");
    return files;
  }

  return files;
}

template <size_t N>
static void
kNNQueryThread(uint64_t start, uint64_t end,
               const provallo::kd_tree<N, unsigned int> &kd, size_t k,
               const std::vector<std::pair<provallo::point<N>, unsigned int>> &data)
{
  for (uint64_t i = start; i < end; i++)
  {
    const auto &p = data[i];
    unsigned int pred = kd.kNNValue(p.first, k);
    queryLock.lock();
    ++numQueriesProcessed;
    if (pred == p.second)
      ++correctCount;
    else
    {
      ++errorsCount;
      std::cout << "Error: " << p.second << " " << pred << std::endl;
    }
    if (numQueriesProcessed % 500 == 0)
      std::cout << numQueriesProcessed << std::endl;
    queryLock.unlock();
  }
}

bool test_fast_knn(const std::string benchmark_folder = "./db/benchmarks");

bool test_fast_knn(const std::string benchmark_folder)
{

  /**
    using datarecord14 = std::vector<std::pair<provallo::point<14>, unsigned int>>;
    using datarecord17 = std::vector<std::pair<provallo::point<17>, unsigned int>>;
    using datarecord20 = std::vector<std::pair<provallo::point<20>, unsigned int>>;
    using datarecord22 = std::vector<std::pair<provallo::point<22>, unsigned int>>;
    using datarecord61 = std::vector<std::pair<provallo::point<61>, unsigned int>>;
    using datarecord65 = std::vector<std::pair<provallo::point<65>, unsigned int>>;
    using datarecord37 = std::vector<std::pair<provallo::point<37>, unsigned int>>;
    using datarecord19 = std::vector<std::pair<provallo::point<19>, unsigned int>>;
    using datarecord58 = std::vector<std::pair<provallo::point<58>, unsigned int>>;
    using datarecord17 = std::vector<std::pair<provallo::point<17>, unsigned int>>;
    using datarecord362 = std::vector<std::pair<provallo::point<362>, unsigned int>>;
    */

  bool ret = false;

  std::vector<std::string> files = getFilesInFolder(benchmark_folder);
  std::vector<std::string> description_files;
  for (auto file : files)
  {
    if (file.find(".names") != std::string::npos)
    {
      description_files.push_back(file);

      std::cout << "-- found description file : " << file << std::endl;
    }
  }
  // iterate all the description files, build 'collector' and 'classifiers' for each of them

  for (auto description_file : description_files)
  {
    std::string file_stem = description_file.substr(0, description_file.find(".names"));
    std::string data_file = benchmark_folder + "/" + file_stem + ".data";
    // std::string weights_file_name = benchmark_folder+"/"+file_stem+".weights";
    // std::ifstream weights_ (weights_file_name);
    // std::map<std::string,Float> weights;
    std::cout << "-- building dataset for : " << file_stem << std::endl;
    provallo::files_collector collector = provallo::files_collector(benchmark_folder + "/" + file_stem);
    const provallo::attribute_information &attributes = collector.getAttributes(), attributes_copy = collector.getAttributes();
    // checking zero knowledge distributed kmeans
    std::cout << "-- Attributes information : " << std::endl
              << std::endl;
    std::cout << description_file << std::endl;
    std::cout << collector.getAttributes() << std::endl;
    std::cout << "-- checking for weights .... " << std::endl
              << std::endl;
    // std::cout<<"weights size : "<<std::to_string(weights.size())<<std::endl;
    std::cout << "attributes size : " << std::to_string(attributes.getSize()) << std::endl;
    /*std::cout<<"-- weights : "<<std::endl<<std::endl;
    for (const auto& weight : weights)
    {
      std::cout<<weight.first<<" : "<<std::to_string(weight.second)<<std::endl;
    } */

    std::cout << std::endl
              << "------- attributes --------" << std::endl
              << std::endl;
    provallo::training_set set(attributes);
    // add default weight map  to dataset itself, not classifier
    // params.

    collector.pushTrainData(&set);
    std::cout << "-- training set size : " << std::to_string(set.size()) << std::endl;
    provallo::matrix<provallo::attribute> data = set.get_matrix();

    std::cout << "-- training set matrix size : " << std::to_string(data.size1() * data.size2()) << std::endl;

    if (false && attributes.getSize() == 14)
    {
      // will crash.
      // wine ?
      using datarecord14 = std::vector<std::pair<provallo::point<14>, unsigned int>>;
      datarecord14 record, testData;
      for (size_t i = 0; i < data.size1(); ++i)
      {
        provallo::point<14> point;
        for (size_t j = 0; j < data.size2(); ++j)
        {
          point[j] = data(i, j).continous();
        }
        record.push_back(std::make_pair(point, data(i, set.get_target_tag()).discrete()));
      }
      provallo::kd_tree<14, unsigned int> kd(record);

      std::cout << "[+]Finished building KD-Tree!" << std::endl;

      bool sanityPass = true;
      for (size_t i = 0; i < record.size(); i++)
      {
        if (!kd.contains(record[i].first) || kd.kNNValue(record[i].first, 1) != record[i].second)
        {
          sanityPass = false;
          break;
        }
      }

      if (sanityPass)
        std::cout << "[+]Sanity check PASSED!" << std::endl;
      else
        std::cout << "[=]Sanity check FAILED!" << std::endl;

      std::cout << "[+]Starting testing!" << std::endl;

      errorsCount = correctCount = 0;
      //
      provallo::testing_set testset(attributes_copy);
      collector.pushTestData(&testset);
      const provallo::testing_samples &samples = testset.get_samples();
      provallo::attribute_tag target_tag = testset.get_target_tag();
      if (target_tag != set.get_target_tag())
      {

        throw std::runtime_error("target tag mismatch");
      }
      size_t j = 0;
      for (const auto &sample : samples)
      {
        provallo::point<14> point;
        const size_t point_size = 14;
        {
          if ((j == 0) || (j != 0 && j != target_tag && (j % target_tag != 0)))
          {
            point[j % point_size] = sample.continous();
          }
          else
          {
            uint32_t target = sample.discrete();

            if (target == 1 || target == 2 || target == 3)
            {
              std::cout << "[+] target is in range" << std::endl;
            }
            else
            {
              std::cout << "[+] target is not in range corrupt testing_set load" << std::endl;
              throw std::runtime_error(std::string("target") + std::to_string(target) + " is not in range corrupt testing_set load");
            }
            testData.push_back(std::make_pair(point, target));
            std::cout << "[+] adding test data for label: " << std::to_string(target) << std::endl;
          }
        }
        j++;
      } // for each sample

      size_t testCnt = testData.size();
      std::cout << "[+]Loaded " << testCnt << " test records." << std::endl;
      std::cout << "[+]Starting benchmarking..." << std::endl;
      size_t k = sqrt(attributes.getSize()); // Number of nearest neighbors

      int queriesPerThread = (int)double(testCnt) / double(kNumThreads);
      std::vector<std::thread> threads;
      std::cout << "[+]Start evaluating kNN performance on the test set ("
                << "k = "
                << k << ")" << std::endl;
      auto c_start = clock();
      auto t_start = std::chrono::high_resolution_clock::now();

      for (int i = 0; i < kNumThreads; i++)
      {
        uint64_t start = i * queriesPerThread;
        uint64_t end = (i == kNumThreads - 1) ? testCnt : start + queriesPerThread;
        threads.push_back(
            std::thread(kNNQueryThread<14>, start, end, (kd), k, ref(testData)));

      } // for each thread
      for (std::thread &t : threads)
        t.join();
      clock_t c_end = clock();
      auto t_end = std::chrono::high_resolution_clock::now();
      std::cout << "[+] Test rate: " << correctCount - errorsCount * 100.0 / double(testCnt) << "%"
                << std::endl;
      std::cout << "[+] CPU time elapsed in s: "
                << (double)(c_end - c_start) / CLOCKS_PER_SEC << std::endl;
      std::cout << "[+] Wall time elapsed in s: "
                << std::chrono::duration<double>(t_end - t_start).count() << std::endl;
      std::cout << "[+] Finished benchmarking!" << std::endl;
    } // if attributes size == 14
  }

  // we have a set of benchmarking datasets for classification.
  // let's test with fast_knn
  // for each dataset, we need to build a classifier
  // for each classifier, we need to test it with fast_knn

  ret = true;
  return ret;

} // test_fast_knn

bool test_dataset_load_msource()
{
  try
  {
    provallo::names_source source("provallo_core/provallo_attributes");
    // use the source to build the attributes
    // use the matrices and labels to train and benchmark classifiers
    provallo::matrix<real_t> training(source.trainingSample());
    provallo::matrix<real_t> testing(source.testingSample());
    std::vector<size_t> labels = source.trainingLabels();
    size_t nclasses = source.n_classes();
    real_t auc = 0.;

    std::cout << "[+] training set size : " << std::to_string(training.size1()) << std::endl;
    std::cout << "[+] training set attributes : " << std::to_string(training.size2()) << std::endl;
    std::cout << "[+] training set labels : " << std::to_string(labels.size()) << std::endl;
    std::cout << "[+] testing set size : " << std::to_string(testing.size1()) << std::endl;
    std::cout << "[+] testing set attributes : " << std::to_string(testing.size2()) << std::endl;

    // initialize a softmax classifier with the training set and thelabels :

    provallo::softmax_classifier<double, double> softlassifier(/*nclasses*/ nclasses, (size_t) /*dimensions*/ source.n_features(), /*alpha*/ (real_t)1., (real_t)0.05 /*lambda*/);

    // fit the classifier with the training set and the labels :
    softlassifier.train(training, labels);
    // test the classifier with the testing set :
    std::vector<real_t> target(labels.size());
    // copy labels to target vector :
    for (size_t i = 0; i < std::min(labels.size(), target.size()); i++)
    {
      target[i] = size_t(labels[i]);
    }

    // copy the   labels to the target vector :
    for (size_t i = 0; i < std::min(labels.size(), target.size()); i++)
    {
      target[i] = labels[i];
    }
    std::vector<real_t> predictions(target);
    softlassifier.test(testing, target);
    // print the predictions errrors vs target :

    // calculate the accuracy, confusion matrix and auc :
    provallo::matrix<real_t> confusion_matrix(nclasses, nclasses);
    std::vector<std::pair<real_t,real_t>> roc_curve(target.size(),std::make_pair<real_t>(0.0,0.0) );
    real_t accuracy = 0., error = 0., fpr = 0., tpr = 0.;
    auc = 0.;
    for (size_t i = 0; i < target.size(); i++)
    {
      // calculate the confusion matrix :
      size_t target_class = predictions[i];
      size_t predicted_class = target[i];

      // update confusion matrix :
      accuracy += (target_class == predicted_class) ? 1. : 0.;

      confusion_matrix(target_class, predicted_class)++;

      // update fpr and tpr :
      if (target_class == predicted_class)
      {
        tpr += 1.;
      }
      else
      {
        fpr += 1.;
      }
      // update tn and tp :

      // update roc curve :
      real_t roc_point = 0.;
      // calculate the roc point from the fpr and tpr :
      if (fpr != 0. && tpr != 0.)
      {
        if (fpr != 0. && fpr == fpr)
        {
          roc_point = (tpr / fpr);
        }
        else
        {
          roc_point = 0.;
        }
      }
      else
      {
        roc_point = 0.;
      }
      // add the roc point to the roc curve :

      roc_curve[i].first = fpr/target.size();
      roc_curve[i].second = tpr/target.size();

      // update auc :

      auc += roc_point;
      // update fpr and tpr :

      error += std::abs(target[i] - predictions[i]);
    }
    // calculate the fpr and tpr :
    fpr /= target.size();
    tpr /= target.size();

    // auc :
    auc /= target.size();

    // F1 score :
    real_t f1_score = 2. * (tpr * fpr) / (tpr + fpr);
    // Precision :
    real_t precision = tpr / (tpr + fpr);
    // Recall :
    real_t recall = tpr / (tpr + fpr);

    // calculate the accuracy :
    accuracy /= target.size();
    error /= target.size();

    std::cout << "[+] accuracy : " << std::to_string(accuracy) << std::endl;
    // precision -recall :
    std::cout << "[+] precision : " << std::to_string(precision) << std::endl;
    std::cout << "[+] recall : " << std::to_string(recall) << std::endl;
    // F1 score :
    std::cout << "[+] f1 score : " << std::to_string(f1_score) << std::endl;

    // calculate the error :
    std::cout << "[+] error : " << std::to_string(error) << std::endl;

    // calculate the confusion matrix :

    std::cout << "[+] confusion matrix : " << std::endl;
    // print the labels :
    for (size_t i = 0; i < source.trainingLabels().size(); i++)
    {
      std::cout << source.trainingLabels()[i] << " ";
    }
    std::cout << std::endl;
    // print the label vertically  and the matrix
    for (size_t i = 0; i < confusion_matrix.size1(); i++)
    {
      std::cout << source.trainingLabels()[i] << " ";
      for (size_t j = 0; j < confusion_matrix.size2(); j++)
      {
        std::cout << std::to_string(confusion_matrix(i, j)) << " ";
      }
      std::cout << std::endl;
    }
    // calculate the auc :

    std::cout << "[+] auc : " << std::to_string(auc) << std::endl;

    // save ROC for plotting :
    std::ofstream roc_file("provallo_net_roc_curve.dat");
    for (size_t i = 0; i < roc_curve.size(); i++)
    {
      roc_file << std::to_string(roc_curve[i].first) << " " << std::to_string(roc_curve[i].second) << std::endl;
    }
    roc_file.close();

    // create gnuplot for ROC curve :
    std::ofstream roc_gnuplot("provallo_net_roc_curve.gnuplot");
    roc_gnuplot << "set terminal png" << std::endl;
    roc_gnuplot << "set output \"provallo_net_roc_curve.png\"" << std::endl;
    roc_gnuplot << "set title \"ROC curve\"" << std::endl;
    roc_gnuplot << "set xlabel \"False Positive Rate\"" << std::endl;
    roc_gnuplot << "set ylabel \"True Positive Rate\"" << std::endl;
    roc_gnuplot << "set xrange [0:1]" << std::endl;
    roc_gnuplot << "set yrange [0:1]" << std::endl;
    roc_gnuplot << "plot \"provallo_net_roc_curve.dat\" with lines" << std::endl;
    roc_gnuplot.close();

    // save confusion matrix for plotting :
    std::ofstream confusion_file("provallo_net_confusion_matrix.dat");
    for (size_t i = 0; i < source.trainingLabels().size(); i++)
    {
      // print the labels:
      confusion_file << source.trainingLabels()[i] + " ";
    }
    confusion_file << std::endl;
    // print the label vertically  and the matrix

    std::cout << std::endl;

    for (auto &label : labels)
      std::cout << label + " ";
    std::cout << std::endl;

    for (size_t i = 0; i < confusion_matrix.size1(); i++)
    {

      for (size_t j = 0; j < confusion_matrix.size2(); j++)
      {
        if (j == 0)
        {
          if (labels.size() > i)
            std::cout << labels[i] + " ";
        }
        confusion_file << std::to_string(confusion_matrix(i, j)) << " ";
      }
      // done printing the row
      confusion_file << std::endl;
      std::cout << std::endl;
    }
    confusion_file.close();

    // create gnuplot for confusion matrix :
    std::ofstream confusion_gnuplot("provallo_net_confusion_matrix.gnuplot");
    confusion_gnuplot << "set terminal png" << std::endl;
    confusion_gnuplot << "set output \"provallo_net_confusion_matrix.png\"" << std::endl;
    confusion_gnuplot << "set title \"Confusion matrix\"" << std::endl;

    confusion_gnuplot << "set xlabel \"Predicted label\"" << std::endl;
    confusion_gnuplot << "set ylabel \"True label\"" << std::endl;
    confusion_gnuplot << "set xrange [0:" << std::to_string(confusion_matrix.size1()) << "]" << std::endl;
    confusion_gnuplot << "set yrange [0:" << std::to_string(confusion_matrix.size2()) << "]" << std::endl;
    // plot the labels :
    for (size_t i = 0; i < source.trainingLabels().size(); i++)
    {
      confusion_gnuplot << "set label \"" << source.trainingLabels()[i] << "\" at " << std::to_string(i) << ",0" << std::endl;
      confusion_gnuplot << "set label \"" << source.trainingLabels()[i] << "\" at 0," << std::to_string(i) << std::endl;
    }

    confusion_gnuplot << "plot \"provallo_net_confusion_matrix.dat\" matrix with image" << std::endl;
    confusion_gnuplot.close();
    roc_curve.clear();
    predictions.clear();
    target.clear();
  }

  catch (const std::exception &e)
  {
    std::cout << e.what() << std::endl;
    return false;
  }
  catch (...)
  {
    std::exception_ptr p = std::current_exception();
    try
    {
      std::rethrow_exception(p);
    }
    catch (const std::exception &e)
    {
      std::cout << e.what() << std::endl;
      return false;
    }
    return false;
  }
  return true;
}
bool test_dataset_load()
{
  try
  {
    provallo::files_collector collector("provallo_core/provallo_attributes");
    const provallo::attribute_information &attributes(
        collector.getAttributes());

    std::ifstream weights_file("provallo_core/provallo_attributes.weights");
    std::map<std::string, Float> weights = provallo::getWeightMap(attributes, weights_file);

    std::cout << "-- Attributes information : " << std::endl
              << std::endl;
    std::cout << collector.getAttributes() << std::endl;
    std::cout << "-- checking for weights .... " << std::endl
              << std::endl;
    std::cout << "weights size : " << std::to_string(weights.size()) << std::endl;
    std::cout << "attributes size : " << std::to_string(attributes.getSize()) << std::endl;
    std::cout << "attributes groups size : " << std::to_string(attributes.getGroups().size()) << std::endl;

    auto c_start = clock();
    auto t_start = std::chrono::high_resolution_clock::now();
    provallo::training_set train_data(collector.getAttributes());
    collector.pushTrainData(&train_data);

    auto c_end = clock();
    auto t_end = std::chrono::high_resolution_clock::now();

    std::cout << "[+] training set size : " << std::to_string(train_data.size()) << std::endl;

    std::cout << "[+] training set attributes : " << std::to_string(train_data.getattributesNumber()) << std::endl;
    std::cout << "[+] training set samples : " << std::to_string(train_data.get_samples().size()) << std::endl;
    std::cout << "[+] training set entropy : " << std::to_string(train_data.entropy()) << std::endl;
    std::cout << "[+] training set skewness : " << std::to_string(train_data.skewness()) << std::endl;
    std::cout << "[+] training set kurtosis : " << std::to_string(train_data.kurtosis()) << std::endl;
    std::cout << "[+] training set standard deviation : " << std::to_string(train_data.stddev()) << std::endl;
    std::cout << "[+] training set mean : " << std::to_string(train_data.mean()) << std::endl;
    std::cout << "[+] training set median : " << std::to_string(train_data.median()) << std::endl;
    std::cout << "[+] training set variance : " << std::to_string(train_data.variance()) << std::endl;

    std::cout << "[+] CPU time elapsed in s: "
              << (double)(c_end - c_start) / CLOCKS_PER_SEC << std::endl;

    // std::cout << "Fitting iso-forest: " << std::endl;
    // size_t sample_size = train_data.getattributesNumber();
    // size_t column_size = train_data.getattributes().getSize();
    /* provallo::isolation_forest* single = isoforest_single( collector.getAttributes());
    provallo::matrix<double> data_matrix(sample_size,column_size);
    size_t nrow=0;
     //init "training" set ...
     //float* X = nullptr;
     std::cout<<"[+] fitting sample vector , no vectorization... "<<std::endl;
     for(const auto& samples_vector : train_data.get_samples())
      {
          size_t col =0;
          for(auto sample : samples_vector)
          {
            //take continous value even if it garbages discrete sets.
            data_matrix(nrow,col) = (double) sample.continous();
          }

          nrow++;
      }
      size_t rot =  sqrt(data_matrix.size1()*data_matrix.size2()) ;
      provallo::matrix<double> reduced = provallo::jacobi(data_matrix , rot);
      single->fit(reduced.data(), reduced.rows(),reduced.cols() );
      std::cout<<"[+] finished fitting single tree"<<std::endl;


    std::cout << "[+] Iso CPU time elapsed in s: "
        << (double) (c_end - c_start) / CLOCKS_PER_SEC << std::endl;


     c_end = clock ();
     t_end = std::chrono::high_resolution_clock::now ();
    */
    // Load the testing data

    c_start = clock();

    // setDiscretizationType(DiscretizationType::EqualFrequency);
    // setDiscretizationType(DiscretizationType::EqualWidth);

    provallo::testing_set test_data(collector.getAttributes());
    collector.pushTestData(&test_data);

    std::cout << "Testing DTGR Classifier: " << std::endl;

    std::cout << "[+] Test CPU time elapsed in s: "
              << (double)(c_end - c_start) / CLOCKS_PER_SEC << std::endl;
    std::cout << "[+] Test Wall time elapsed in s: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << std::endl;

    c_start = c_end;
    t_start = t_end;
    const provallo::testing_samples &attributes_vector(test_data.get_samples());
    std::cout << "[+] attributes vector size " << attributes_vector.size() << std::endl;

    // test nn

    std::cout << "Testing KMB Classifier: " << std::endl;

    typedef provallo::metric<provallo::Overlap, provallo::Euclidean> over_euc;
    typedef provallo::Kmeans<over_euc> km_classifier;
    provallo::kmeans_param km_param(5, weights);
    provallo::train_and_test<km_classifier>(
        "kmeans-binary-split", train_data, test_data, km_param);

    c_end = clock();
    t_end = std::chrono::high_resolution_clock::now();
    std::cout << "[+] Test CPU time elapsed in s: "
              << (double)(c_end - c_start) / CLOCKS_PER_SEC << std::endl;

    std::cout << "Training RF Classifier with 50 parallel trees " << std::endl;

    // train and test DTGR
    typedef provallo::decision_tree<provallo::GainRatio> dtgr_classifier;
    provallo::none dtgr_param;
    provallo::train_and_test<dtgr_classifier>(
        "dtgr", train_data, test_data, dtgr_param);

    c_end = clock();
    std::cout << "[+] Test CPU time elapsed in s: "
              << (double)(c_end - c_start) / CLOCKS_PER_SEC << std::endl;
    std::cout << "[+] Test Wall time elapsed in s: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << std::endl;
    typedef provallo::random_tree<provallo::GainRatio> egain_tree;
    typedef provallo::random_forest<egain_tree> rf_classifier;
    provallo::random_tree_param egain_param(.5, 10, 0.0);

    provallo::random_forest_param rf_param(50, egain_param);

    c_start = clock();
    provallo::train_and_test<rf_classifier>(
        "rf-rf", train_data, test_data, rf_param);
    c_end = clock();
    t_end = std::chrono::high_resolution_clock::now();

    std::cout << "[+] Test CPU time elapsed in s: "
              << (double)(c_end - c_start) / CLOCKS_PER_SEC << std::endl;
    std::cout << "[+] Test Wall time elapsed in s: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << std::endl;

    // std::cout << "Training Kmeans bayesian Classifier: " <<std::endl;
    // provallo::train_and_test<provallo::Kmeans<provallo::bayesian> >("kmeans_bayesian", train_data, test_data);
    // std::cout << "Training ABB Classifier: " <<std::endl;

    // provallo::train_and_test<provallo::adaboost <provallo::bayesian>>("gain ratio split, binary splitting", train_data, test_data);

    // c_end = clock();
    // t_end = std::chrono::high_resolution_clock::now();

    // std::cout << "[+] Test CPU time elapsed in s: " << (double)(c_end - c_start) / CLOCKS_PER_SEC << std::endl;
    // std::cout << "[+] Test Wall time elapsed in s: " << std::chrono::duration<double>(t_end - t_start).count() << std::endl;
  }
  catch (std::exception &e)
  {
    std::cerr << std::string("exception raised : ") + e.what() << std::endl;
    return false;
  }
  return true;
}
void train_web_requests_patterns();
void test_web_requests_patterns();
void validate_simple_softmax();
bool benchmark_neural_network(const std::string banchmark_folder);
int main(int argc, char *argv[])
{

  if (argc > 1)
  {
    std::cout << argv[0] << " running...." << std::endl;
  }
  //to run the tests :
  //validat matrix operations for debugging:
  //test_matrix();
  //validate softmax classifier 
  validate_simple_softmax();
  // validate section readnig of  nginx configuration file
  // 
  provallo::nginx_config_helper::configuration_helper config_helper;
  config_helper.dump();

  std::getchar();

  test_spike_train_generator();

  if (test_fit_iso_forest())
  {
    std::cout << "Test fit iso forest OK" << std::endl;
  }


  train_web_requests_patterns();
  
  //load pre-trained model and test with a diffrent dataset
  //test_web_requests_patterns();
  if (benchmark_classifiers("./db/benchmarks"))
  {
    std::cout << "Classifiers benchmark OK" << std::endl;
  }
  else
  {
    std::cout << "Classifiers benchmark FAILED" << std::endl;
    exit(-1);
  }

  if (test_dataset_load_msource())
  {
    std::cout << "Test dataset load OK" << std::endl;
  }
  else
  {
    std::cout << "Test dataset load FAILED" << std::endl;
  }

  // benchmark fuzzdb dataset
  if (fit_fuzzsb())
  {
    std::cout << "Fuzzdb test OK" << std::endl;
  }
  else
  {
    std::cout << "Fuzzdb test FAILED" << std::endl;
    exit(-1);
  }

  if (test_fast_knn("./db/benchmarks"))
  {
    std::cout << "Test fast knn OK" << std::endl;
  }
  else
  {
    std::cout << "Test fast knn FAILED" << std::endl;
    exit(-1);
  }

  if(benchmark_neural_network("./db/benchmarks"))
  {
    std::cout<<"Benchmark neural networks OK"<<std::endl;
  }
  return 0;
}

bool test_vectorizers()
{
  // normalize the attributes

  bool ret = false;
  std::vector<std::string> vectorize_set_attributes({"ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN"});
  std::vector<std::string> vectorize_set_classes({"CLASS1", "CLASS2", "CLASS3", "CLASS4", "CLASS5", "CLASS6", "CLASS7", "CLASS8", "CLASS9", "CLASS10"});

  std::vector<std::vector<std::string>> vectorize_set_data;
  for (size_t i = 0; i < vectorize_set_attributes.size(); i++)
  {
    std::vector<std::string> row;
    for (size_t j = 0; j < 10; j++)
    {
      if (j == i)
        row.push_back(vectorize_set_attributes[i]);
      else
        row.push_back(std::to_string(i * j));
    }
    vectorize_set_data.push_back(row);
  }

  std::cout << "Testing vectorizers" << std::endl;

  std::cout << "Testing vectorizer 1" << std::endl;

  provallo::tfidf_vectorizer vectorizer1;
  provallo::pca_vectorizer vectorizer2;
  provallo::lda_vectorizer vectorizer3;
  provallo::pca_vectorizer vectorizer4;
  // provallo::nmf_vectorizer vectorizer4;
  // provallo::svd_vectorizer vectorizer5;
  // provallo::kmeans_vectorizer vectorizer6;
  // provallo::knn_vectorizer vectorizer7;
  // provallo::random_projection_vectorizer vectorizer8;
  // provallo::fast_knn_vectorizer vectorizer9;

  vectorizer1.fit(vectorize_set_data);
  vectorizer2.fit(vectorize_set_data);
  vectorizer3.fit(vectorize_set_data);
  vectorizer4.fit(vectorize_set_data);

  std::vector<provallo::vectorizer<std::string, real_t> *> vectorizers;
  vectorizers.push_back(&vectorizer1);
  vectorizers.push_back(&vectorizer2);
  vectorizers.push_back(&vectorizer3);
  // vectorizers.push_back(&vectorizer3);
  vectorizers.push_back(&vectorizer4);

  // vectorizers.push_back(&vectorizer4);

  for (size_t i = 0; i < vectorize_set_data.size(); i++)
  {
    size_t j = 0;
    std::cout << "Testing vectorizer pass 1 transform" << std::endl;
    for (auto v : vectorizers)
    {

      // print vectorizer name
      std::cout << " [+] vectorizer :  " << vectorizer_types[v->get_type()] << std::endl;

      std::vector<double> vectorized_data = v->transform(vectorize_set_data[i]);
      j = 0;
      for (auto f : vectorized_data)
      {
        std::cout << " [+] transform :  " << std::to_string(f) << " " << (vectorize_set_data[i][j++ % vectorize_set_data[i].size()]) << std::endl;

        ret = true;
      }
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;
  char x = std::getchar();
  x--;
  for (size_t i = 0; i < vectorize_set_data.size(); i++)
  {

    std::cout << "Testing vectorizer pass 2 transform" << std::endl;
    for (auto v : vectorizers)
    {
      std::vector<double> vectorized_data = v->transform(vectorize_set_data[i]);
      size_t j = 0;

      std::cout << " [+] vectorizer :  " << vectorizer_types[v->get_type()] << std::endl;

      for (auto f : vectorized_data)
      {
        std::cout << " [+] predict :  " << std::to_string(f) << " " << vectorize_set_data[i][j++ % vectorize_set_data[i].size()] << std::endl;
        ret = true;
      }
      std::cout << std::endl;
    }
  }

  return ret;
}
bool benchmark_neural_network(const std::string banchmark_folder)
{
  bool ret = false;
  // bool buse_random_forest = false;
  // bool test_ultra_fast_knn = false;
  // iterate all the files in the folder, if it's a descrition file, build dataset and classifiers for it
  std::vector<std::string> files = getFilesInFolder(banchmark_folder);
  std::vector<std::string> description_files;
  for (auto file : files)
  {
    if (file.find(".names") != std::string::npos)
    {
      description_files.push_back(file);

      std::cout << "-- found description file : " << file << std::endl;
    }
  }

  // iterate all the description files, build 'collector' and 'classifiers' for each of them
  // bool sanity_check = false;
  for (auto description_file : description_files)
  {
    std::string file_stem = description_file.substr(0, description_file.find(".names"));
    std::string data_file = banchmark_folder + "/" + file_stem + ".data";
    // std::string weights_file_name = banchmark_folder+"/"+file_stem+".weights";
    // std::ifstream weights_ (weights_file_name);
    std::map<std::string, Float> weights;
    std::cout << "-- building dataset for : " << file_stem << std::endl;
    //use provallo::name_source and neural network to build a classifier 
    //use the classifier to test the dataset

 
    provallo::names_source source(banchmark_folder + std::string("/") + file_stem);
    //provallo::learning_task::task_configuration learn(build_task_from_source(source));
    //provallo::learning_task task(learn);
 
    //build a neural network for the discriminator. initialize vectors for the neural_network use kl for gen function 
   provallo::neural_net *generator =nullptr, *discriminator =nullptr;
   std::vector<size_t> layerTypes;
   std::vector<size_t> layerSizes;
   std::vector<size_t> layerChannels;
   std::vector<std::vector<size_t>> layerArg;
   std::vector<std::function<real_t(real_t)>> activations;

   size_t nclasses = source.n_classes(); 
    size_t nfeatures = source.n_features();
    size_t nhidden =  nfeatures*2;
    size_t noutput = nclasses;
    size_t ninput = nfeatures;
    size_t nchannels = 1;
    size_t nfilters = 3;
    size_t nkernel = 3;
    size_t nstride = 1;
    size_t npadding = 1;
    size_t ngroups = 1;
    size_t nbatch = 1;
    size_t nepochs = 100;
    size_t nbatch_size = 1;
    size_t nlearning_rate = 0.01;
    size_t nweight_decay = 0.01;
    size_t nmomentum = 0.9;
    size_t nstep_size = 1;
    size_t ngamma = 0.1;
    
    //build a discriminator neural network :
    //layer types : 0 - fully connected, 1 - convolutional, 2 - max pooling, 3 - average pooling, 4 - dropout, 5 - batch normalization, 6 - activation 
    //layer sizes : number of neurons in the layer
    //layer channels : number of channels in the layer
    //layer args : arguments for the layer
    //activation funs : activation functions for the layer
    
    //configure the layers,size,channels,arguments and activation functions based on the source inputs  
    //input layers : 
    layerTypes.push_back(0); //fully connected layer
    layerSizes.push_back(ninput);
    layerChannels.push_back(nchannels);
    layerArg.push_back(std::vector<size_t>());
    //input activation function: 
    activations.push_back(provallo::sigmoid(0.1) );
    //hidden layers :
    layerTypes.push_back(0); //fully connected layer
    layerSizes.push_back(nhidden);
    layerChannels.push_back(nchannels);
    layerArg.push_back(std::vector<size_t>());
    //hidden activation function:
    activations.push_back(provallo::sigmoid(0.1) );

    //convolutional layer :
    layerTypes.push_back(1); //convolutional layer
    layerSizes.push_back(nhidden);
    layerChannels.push_back(nchannels);
    layerArg.push_back(std::vector<size_t>({nfilters,nkernel,nstride,npadding,ngroups}));
    //convolutional activation function:
    activations.push_back(provallo::sigmoid(0.1) );
    
    //max pooling layer :
    layerTypes.push_back(2); //max pooling layer
    layerSizes.push_back(nhidden);
    layerChannels.push_back(nchannels);
    layerArg.push_back(std::vector<size_t>({nfilters,nkernel,nstride,npadding,ngroups}));
    //max pooling activation function:
    activations.push_back(provallo::sigmoid(0.1) );

    //average pooling layer :
    layerTypes.push_back(3); //average pooling layer
    layerSizes.push_back(nhidden);
    layerChannels.push_back(nchannels);
    layerArg.push_back(std::vector<size_t>({nfilters,nkernel,nstride,npadding,ngroups}));
    //average pooling activation function:
    activations.push_back(provallo::sigmoid(0.1) );

    //dropout layer :
    layerTypes.push_back(4); //dropout layer
    layerSizes.push_back(nhidden);
    layerChannels.push_back(nchannels);
    layerArg.push_back(std::vector<size_t>({nfilters,nkernel,nstride,npadding,ngroups}));
    //dropout activation function:
    activations.push_back(provallo::sigmoid(0.1));
    activations.push_back(provallo::sigmoid(0.1) );

    discriminator = new provallo::neural_net(layerTypes,layerSizes,layerChannels,layerArg,activations);
    generator = new provallo::neural_net(layerTypes,layerSizes,layerChannels,layerArg,activations);
    //build a neural helper with discriminator and generator and KL-divergence loss function :
    provallo::neural_helper helper(generator,discriminator,2);
    
    std::cout<<"-- building neural network for : "<<file_stem<<std::endl;
    //build a neural network with the helper
    std::cout<<"[+] features: "<<std::to_string(nfeatures)<<std::endl;
    std::cout<<"[+] hidden: "<<std::to_string(nhidden)<<std::endl;
    std::cout<<"[+] output: "<<std::to_string(noutput)<<std::endl;
    std::cout<<"[+] classes: "<<std::to_string(nclasses)<<std::endl;
    std::cout<<"[+] channels: "<<std::to_string(nchannels)<<std::endl;
    std::cout<<"[+] momentum: "<<std::to_string( nmomentum)<<std::endl;
        std::cout<<"[+] decay: "<<std::to_string( nweight_decay)<<std::endl;
            std::cout<<"[+] epochs: "<<std::to_string( nepochs)<<std::endl;
                std::cout<<"[+] batch size: "<<std::to_string( nbatch_size)<<std::endl;
                    std::cout<<"[+] learning rate: "<<std::to_string( nlearning_rate)<<std::endl;
                        std::cout<<"[+] step size: "<<std::to_string( nstep_size)<<std::endl;
                            std::cout<<"[+] gamma: "<<std::to_string( ngamma)<<std::endl;
                                                        std::cout<<"[+] batch size: "<<std::to_string( nbatch)<<std::endl;


    //train the neural network with the training set :

    // std::cout<< attributes <<std::endl<<std::endl;
    //delete the neural network
    delete generator;
    delete discriminator;

    ret = true;
    // x+= std::getchar();
    // checking zero knowledge distributed kmeans
  } // for description files
    // std::cout<<"-- Attributes information : "<<std::endl<<
    // std::cout<< description_file<<std::endl;

  return ret;
}
bool benchmark_classifiers(const std::string benchmark_folder)
{
  bool ret = false;
  // bool buse_random_forest = false;
  // bool test_ultra_fast_knn = false;
  // iterate all the files in the folder, if it's a descrition file, build dataset and classifiers for it
  std::vector<std::string> files = getFilesInFolder(benchmark_folder);
  std::vector<std::string> description_files;
  for (auto file : files)
  {
    if (file.find(".names") != std::string::npos)
    {
      description_files.push_back(file);

      std::cout << "-- found description file : " << file << std::endl;
    }
  }

  // iterate all the description files, build 'collector' and 'classifiers' for each of them
  // bool sanity_check = false;
  for (auto description_file : description_files)
  {
    std::string file_stem = description_file.substr(0, description_file.find(".names"));
    std::string data_file = benchmark_folder + "/" + file_stem + ".data";
    // std::string weights_file_name = benchmark_folder+"/"+file_stem+".weights";
    // std::ifstream weights_ (weights_file_name);
    std::map<std::string, Float> weights;
    std::cout << "-- building dataset for : " << file_stem << std::endl;
    std::vector<provallo::classifier *> classifiers;

    // provallo::files_collector collector = provallo::files_collector (benchmark_folder+"/"+file_stem);

    // provallo::attribute_information attributes = collector.getAttributes ();

    // size_t nclasses = attributes.getTargetClassCount();
    // checking zero knowledge distributed kmeans

    // provallo::auto_encoder<double,double> encoder( collector.getAttributes().getSize(),collector.getAttributes().getSize()*collector.getAttributes().getSize(),collector.getAttributes().getTargetClassCount() );

    // std::cout<<"-- attribute info : "<<std::endl<<std::endl;
    // std::cout<< attributes <<std::endl<<std::endl;
    // char x =      std::getchar();

    // std::cout<< attributes <<std::endl<<std::endl;

    // x+= std::getchar();
    // checking zero knowledge distributed kmeans

    // std::cout<<"-- Attributes information : "<<std::endl<<std::endl;
    // std::cout<< description_file<<std::endl;
    // std::cout<<collector.getAttributes ()<<std::endl;
    /* std::cout<<"-- checking for weights .... "<<std::endl<<std::endl;
    if(weights_.is_open()&&weights_.good())
      {
        weights = provallo::getWeightMap(attributes,weights_);
        std::cout<<"-- weights : "<<std::endl<<std::endl;
        for (const auto& weight : weights)
        {
          std::cout<<weight.first<<" : "<<std::to_string(weight.second)<<std::endl;
        }

      }
     else
      {
        std::cout<<"-- no weights found "<<std::endl<<std::endl;
      }

    */
    // std::cout<<"attributes size : "<<std::to_string(attributes.getSize())<<std::endl;
    /*std::cout<<"-- weights : "<<std::endl<<std::endl;
    for (const auto& weight : weights)
    {
      std::cout<<weight.first<<" : "<<std::to_string(weight.second)<<std::endl;
    } */

    // std::cout<<std::endl<<"------- attributes --------"<<std::endl<<std::endl;
    // provallo::training_set set(attributes);

    // add default weight map  to dataset itself, not classifier
    // params.

    // collector.pushTrainData (&set);
    // std::cout<<"-- training set size : "<<std::to_string(set.size() )<<std::endl;

    provallo::names_source source(benchmark_folder + std::string("/") + file_stem);
    // provallo::learning_task learn(build_task_from_source(source)  );
    // build a neural helper with learning task :
    // build a neural network with the helper

    // create a batch of inputs and desired outputs
    provallo::matrix<real_t> inputs = source.trainingSample();
     provallo::matrix<real_t> outputs(inputs.size1(), 1);
     // train the generator and the discriminator with the inputs and outputs

    source.print();
    std::getchar();

    provallo::matrix<real_t> data = source.trainingSample(false);
    size_t nclasses = source.n_classes();

    provallo::softmax_classifier<double, double> softmax(/*nclasses*/ nclasses, (size_t) /*dimensions*/ source.n_features(), /*alpha*/ (real_t)1., (real_t)0.05 /*lambda*/);

    // provallo::matrix<real_t> data = read_data_file(data_file);

    std::cout << "-- training set matrix size : " << std::to_string(data.size1() * data.size2()) << std::endl;
    std::cout << "-- training set matrix size1 : " << std::to_string(data.size1()) << std::endl;
    std::cout << "-- attribute info : " << std::endl
              << std::endl;
    // size_t nclasses = set.getattribute_info().getTargetClassCount();
    // size_t ndimention = set.getattribute_info().getSize();
    // std::cout<<"-- press enter to continue .... "<<std::endl<<std::endl;
    // std::getchar();
    // train encoders
    // provallo::class_dist ds(set.getattribute_info().getTargetClassCount());
    provallo::matrix<double> mdata(data);
    const auto &labels = source.trainingLabels();
    // update output data with the labels :
    std::vector<size_t> losses(labels); //(labels.size(),0);

    // std::cout<<"-- training encoder "<<std::endl;

    // encoder.train(mdata,ds);
    softmax.train(mdata, losses);

    // update correct/error rates
    size_t correct = 0;
    size_t errors = 0;
    size_t fp = 0, fn = 0, tp = 0, tn = 0;
    // size_t target_column = source.target_column();
    std::vector<real_t> errors_per_class(nclasses, 0.0);
    provallo::matrix<size_t> training_confusion(nclasses, nclasses);

    // compare the predictions (losses) with the labels (labels) :
    for (size_t i = 0; i < std::min(losses.size(), data.size1()); i++)
    {
      size_t predicted_class = losses[i];
      size_t target_class = labels[i];

      // update confusion matrix :
      training_confusion(target_class, predicted_class)++;
      // update correct/error rates
      correct += target_class == predicted_class;
      errors += target_class != predicted_class;
      // update fp,fn,tp,tn
      if (target_class == predicted_class)
      {
        tp += 1.;
      }
      else
      {
        // check if it's a false negative or a false positive
        if (target_class == 0)
        {
          fn += 1.;
        }
        else if (predicted_class == 0)
        {
          fp += 1.;
        }
        else
        {
          tn += 1.;
        }
      }
      // update errors per class :
      errors_per_class[target_class] += target_class != predicted_class;
    }

    // calculate prediction accuracy
    real_t accuracy = correct * 100.0 / (data.size1());
    real_t fpr = fp * 100.0 / data.size1();
    real_t tpr = tp * 100.0 / data.size1();
    real_t auc = fpr == 0.0 ? 1. : (tpr / fpr);

    real_t F1 = 2. * (tpr * fpr) / (tpr + fpr);
    // Precision :
    real_t precision = tpr / (tpr + fp);
    // Recall :
    real_t recall = tpr / (tpr + fn);

    // calculate auc
    // calculate confusion matrix
    std::cout << "-- softmax train correct : " << std::to_string(correct) << std::endl;
    std::cout << "-- softmax train errors : " << std::to_string(errors) << std::endl;
    std::cout << "-- softmax train false positive rate : " << std::to_string(fpr) << std::endl;
    std::cout << "-- softmax train true positive rate : " << std::to_string(tpr) << std::endl;
    std::cout << "-- softmax train auc : " << std::to_string(auc) << std::endl;
    std::cout << "-- softmax train F1 : " << std::to_string(F1) << std::endl;
    std::cout << "-- softmax train precision : " << std::to_string(precision) << std::endl;
    std::cout << "-- softmax train recall : " << std::to_string(recall) << std::endl;
    std::cout << "-- softmax train confusion matrix : " << std::endl;

    // for ( auto& label : labels )
    //   std::cout<<label+ " ";

    std::cout << std::endl;
    for (size_t i = 0; i < training_confusion.size1(); i++)
    {

      for (size_t j = 0; j < training_confusion.size2(); j++)
      {
        if (j == 0)
        {
          if (labels.size() > i)
            std::cout << labels[i] + " ";
        }
        std::cout << std::to_string(training_confusion(i, j)) << " ";
      }
      std::cout << std::endl;
    }

    std::cout << "-- softmax train error per class : " << std::endl;
    for (size_t i = 0; i < errors_per_class.size(); i++)
    {
      std::cout << "-- class " << std::to_string(i) << " : " << std::to_string(errors_per_class[i]) << std::endl;
    }

    std::cout << "-- softmax train accuracy : " << std::to_string(accuracy) << std::endl;
    // gnuplot::plot(softmax.getLosses(),"softmax_loss");
    // std::cout<<"-- training encoder "<<std::endl;

    // std::cout<<"-- encoder trained, classdist size : "<<std::to_string(ds.size())<<std::endl;

    // std::cout<<"-- encoder trained, classdist size : "<<std::to_string(ds.size())<<std::endl;
    // std::cout<<"-- classdist: "<<ds<<std::endl;

    // encoder.save("encoder_"+file_stem+".json");

    // encoder.load("encoder_"+file_stem+".json");
    std::cout << "-- encoder loaded " << std::endl;
    std::cout << "-- encoder input size : " << std::to_string(softmax.getInputDim()) << std::endl;
    std::cout << "-- encoder output size : " << std::to_string(softmax.getOutputDim()) << std::endl;
    std::cout << "-- encoder hidden size : " << std::to_string(softmax.getHiddenDim()) << std::endl;

    std::cout << "-- building classifiers " << std::endl
              << std::endl;

#if 0 
      typedef provallo::metric<provallo::Euclidean,provallo::Overlap> over_euc;
      typedef provallo::Kmeans<over_euc> km_classifier;
      typedef provallo::random_tree<provallo::GainRatio> egain_tree;
      typedef provallo::random_forest<egain_tree> rf_classifier;
      provallo::random_tree_param egain_param(.5,10,0.0) ; //rho,level,min-gain
      provallo::random_forest_param rf_param(50,egain_param); //tree-count,tree-param
      provallo::nearest_neighbor_param km_param(sqrt(attributes.getSize()),weights); 
      provallo::metric_classifier_param metric_param(sqrt(attributes.getSize()),weights); 
      provallo::adaboost_param boost_param(attributes.getTargetClassCount(),provallo::none());
     // provallo::kNN_param knn_param(sqrt(attributes.getSize()),weights);

      std::random_device rd;
      //std::cout <<"-- allocating factory .... "<<std::endl<<std::endl;
      //provallo::split_method_factory* factory = new provallo::split_method_factory(set,rd ) ; 
      //factory->set_override_split_method(provallo::split_type::CONE_RANDOM);
      provallo::split_method_factory* random_factory = buse_random_forest? new provallo::split_method_factory(set,rd):nullptr;
      //random_factory->set_override_split_method(provallo::split_type::CONE_BINARY);
      
      provallo::isoforest_param isoparam(500/*ntrees*/,  1000 ,10,
                    256,129,10,
                    0.,50, std::chrono::system_clock::now().time_since_epoch().count()   );
 
      provallo::split_method_factory* factory = nullptr;
      provallo::dataset_ptr dataset_ptr1(nullptr);
      std::cout<<"-- training classifiers .... "<<std::endl<<std::endl; 
 
      if(buse_random_forest) 
         classifiers.push_back(new rf_classifier(set,rf_param,rd,std::cout,random_factory ));
      else
         classifiers.push_back(new provallo::iso_classifier(set,isoparam,rd,factory));   
       //push nullptr factories to initialize separately. 

     // classifiers.push_back(new provallo::decision_tree<provallo::GainRatio> (set,provallo::none(),rd,factory ));
     // classifiers.push_back(new provallo::decision_tree<provallo::EntropyGain> (set,provallo::none(),rd,factory ));
     // classifiers.push_back(new provallo::decision_tree<provallo::ChiSquare> (set,provallo::none(),rd,factory ));

      std::cout<<"-- training UF classifiers  .... "<<std::endl<<std::endl; 
      for(auto & class_ : classifiers)
      { 
          std::cout<<set.getattribute_info()<<std::endl<<std::endl;
          print_classifier_summary(file_stem,set,*class_);
      }//
#endif // 0
    std::getchar();

    std::cout << "-- building neural network .... " << std::endl
              << std::endl;
    std::cout << "-- training classifiers .... " << std::endl
              << std::endl;
    std::cout << "-- testing classifiers .... " << std::endl
              << std::endl;
    // read test data

    std::cout << "-- reading test data .... " << std::endl
              << std::endl;
    // provallo::testing_set test_set( set.getattribute_info());
    // collector.pushTestData (&test_set);
    mdata = source.testingSample(false);
    std::vector<size_t> testing_labels = source.testingLabels();
    // test autoencoder
    std::cout << "-- testing autoencoder .... " << std::endl
              << std::endl;
    // go over the testing set and test the autoencoder
    std::vector<size_t> olabels = testing_labels;

    // test softmax
    std::cout << "-- testing softmax .... " << std::endl
              << std::endl;
    // go over the testing set and test the autoencoder

    softmax.test(mdata, testing_labels);
    // update correct/error rates

    std::cout << "-- softmax test finished " << std::endl
              << std::endl;
    std::cout << "-- softmax test results ---" << std::endl
              << std::endl;

    // compare results with test set  output confusion matrix with labels from names_source
    std::cout << "-- softmax test results : " << std::endl
              << std::endl;
    provallo::matrix<real_t> confusion_(nclasses, nclasses);
    fp = fn = tp = tn = 0.0;
    // init confusion matrix
    size_t target_class, predicted_class;

    // calculate confusion matrix:
    for (size_t i = 0; i < testing_labels.size(); i++)
    {
      // get target class
      target_class = testing_labels[i];
      // get predicted class
      predicted_class = olabels[i];
      // update confusion matrix :
      confusion_(target_class, predicted_class)++;
      // update correct/error rates
      if (target_class == predicted_class)
      {
        tp++;
      }
      else
      {
        // false positive or true negative?
        if (predicted_class == 0 && target_class != 0)
        {
          tn++;
        }
        else if (predicted_class != 0 && target_class == 0)
        {
          fp++;
        }
        else if (predicted_class == 0 && target_class != 0)
        {
          tn++;
        }
        else
        {
          fn++;
        }
      }

      // update correct/error rates
    }

    // calculate prediction accuracy
    accuracy = tp * 100.0 / (testing_labels.size() * nclasses);
    precision = tp / (tp + fp) == 0 ? 1 : (tp + fp);
    recall = tp / (tp + fn) == 0 ? 1 : (tp + fn);
    real_t f1 = 2.0 * (precision * recall) / (precision + recall);

    std::cout << "-- softmax test precision : " << std::to_string(precision) << std::endl;
    std::cout << "-- softmax test recall : " << std::to_string(recall) << std::endl;
    std::cout << "-- softmax test f1 : " << std::to_string(f1) << std::endl;

    std::cout << "-- softmax test accuracy : " << std::to_string(accuracy) << std::endl;
    std::cout << "-- softmax test confusion matrix : " << std::endl
              << std::endl;

    // print confusion matrix
    std::cout << "-- softmax test results : " << std::endl
              << std::endl;
    // save confusion matrix
    std::ofstream confusion_file("softmax_test_confusion_matrix_" + file_stem + ".dat");

    // save the confusion_ data to file
    for (size_t i = 0; i < confusion_.size1(); i++)
    {
      for (size_t j = 0; j < confusion_.size2(); j++)
      {
        confusion_file << confusion_(i, j) << " ";
        std::cout << confusion_(i, j) << " ";
      }
      std::cout << std::endl;
      confusion_file << std::endl;
    }
    // close confusion file

    confusion_file.close();
    std::cout << std::endl;
    // save gnuplot for confusion matrix :
    std::ofstream confusion_gnuplot("softmax_test_confusion_matrix_" + file_stem + ".gnuplot");
    confusion_gnuplot << "set yrange [0:" << std::to_string(confusion_.size2()) << "]" << std::endl;
    confusion_gnuplot << "set xrange [0:" << std::to_string(confusion_.size1()) << "]" << std::endl;
    confusion_gnuplot << "set xlabel \"predicted class\"" << std::endl;
    confusion_gnuplot << "set ylabel \"actual class\"" << std::endl;
    confusion_gnuplot << "set title \"softmax test confusion matrix\"" << std::endl;
    confusion_gnuplot << "set palette rgbformulae 33,13,10" << std::endl;
    // remove uneeded grids
    confusion_gnuplot << "unset xtics" << std::endl;
    confusion_gnuplot << "unset ytics" << std::endl;
    confusion_gnuplot << "unset border" << std::endl;
    confusion_gnuplot << "unset colorbox" << std::endl;
    confusion_gnuplot << "set grid" << std::endl;
    confusion_gnuplot << "set key off" << std::endl;
    confusion_gnuplot << "set size square" << std::endl;
    confusion_gnuplot << "set style fill solid" << std::endl;
    confusion_gnuplot << "set pm3d map" << std::endl;
    confusion_gnuplot << "set palette defined (0 \"white\", 1 \"blue\")" << std::endl;

    // plot confusion matrix to png
    confusion_gnuplot << "set terminal png" << std::endl;
    confusion_gnuplot << "set output \"softmax_test_confusion_matrix_" << file_stem << ".png\"" << std::endl;
    auto &labelnames = source.trainingLabelNames();
    // set the labels for the confusion matrix
    confusion_gnuplot << "set xtics (";
    if (labelnames.size() > nclasses)
    {
      for (size_t i = 0; i < nclasses; i++)
      {
        confusion_gnuplot << "\"" << labelnames[i] << "\" " << i << ",";
      }
    }
    else
    {
      for (size_t i = 0; i < nclasses; i++)
      {
        confusion_gnuplot << "\"" << i << "\" " << i << ",";
      }
    }
    confusion_gnuplot << ")" << std::endl;
    confusion_gnuplot << "set ytics (";
    if (labelnames.size() > nclasses)
    {
      for (size_t i = 0; i < nclasses; i++)
      {
        confusion_gnuplot << "\"" << labelnames[i] << "\" " << i << ",";
      }
    }
    else
    {
      for (size_t i = 0; i < nclasses; i++)
      {
        confusion_gnuplot << "\"" << i << "\" " << i << ",";
      }
    }

    confusion_gnuplot << ")" << std::endl;

    confusion_gnuplot << "plot \"softmax_test_confusion_matrix_" << file_stem << ".dat\" matrix with image" << std::endl;

    // close gnuplot file
    confusion_gnuplot.close();

    // std::cout<<confusion_<<std::endl<<std::endl;
    std::cout << "-- softmax test results ---" << std::endl
              << std::endl;

    // save softmax results

    // save softmax classifier
    softmax.save("softmax_" + file_stem + ".json");
    // std::getchar();

    // test kmeans

    // print confusion matrix of test data

    /*for(auto & class_ : classifiers)
    {
         print_classifier_summary(file_stem,test_set,*class_);
    }*/
    // test ultra fast knn :

    // weights_.close();
    std::cout << "-- deleting description file .... " << std::endl
              << std::endl;

    description_file.clear();
    mdata.resize(1, 1);
    ret = true;

    // clean up
    // delete factory
    // std::cout<<"-- deleting random factory .... "<<std::endl<<std::endl;
    // if(random_factory)
    //  delete random_factory;
    // std::cout<<"-- deleting factory .... "<<std::endl<<std::endl;
    // if(factory)
    //    delete factory;

    /*
          std::cout<<"-- deleting factory .... "<<std::endl<<std::endl;
          if(factory)
            delete factory;
          std::cout<<"-- deleting random factory .... "<<std::endl<<std::endl;
          if(random_factory)
            delete random_factory;
            */

    // avoid double free
    // if(random_factory)
    //  delete random_factory;
    // delete factory
    // delete classifiers
    // delete weights
    // delete weights file

  } // end for description_files
  return ret;
} // end benchmark_classifiers

provallo::learning_task::task_configuration build_task_from_source(provallo::names_source &src)
{
  provallo::learning_task::task_configuration task;
  // configure task according to names data inputs
  task.step = 0.01;
  task.dx = 0.01;
  task.sigmoidParameter = 1.0;
  task.networkAreImported = false;
  task.useAverageForBatchlearning = false;

  task.descent = 0;
  task.descentTypeGen = 0;
  task.descentTypeDis = 0;
  task._Experiments = 1;
  task._LoopsPerExperiment = 1;
  task._TeachingsPerLoop = 1;
  task._DisTeach = 1;
  task._GenTeach = 1;
  task._DisTest = 1;
  task._TeachingsPerLoop = src.n_classes();
  task._GenTest = 1;
  task.labelTrainSize = src.n_train();
  task.labelTestSize = src.n_test();
  task.intervalleImg = 0;
  task._ImgParIntervalleImg = 0;
  task.minibatchSize = 1;
  task.imageSizeSide = 1;

  task.imageSizeSide = 1;
  task.generatorPath = "./gen/";
  task.discriminatorPath = "./disc/dest";
  task.generatorDest = "./gen/";
  task.typeOfExperiment = "GAN";
  task.CSVFileNameImage = "./gen/Img.csv";
  task.CSVFileNameResult = "./gen/Result.csv";
  task.databaseToUse = "MNIST";

  return task;
}
std::vector<std::string> get_files_recursively(const std::string folder)
{
  std::vector<std::string> files;
  std::vector<std::string> subfolders;
  std::vector<std::string> subfiles;
  std::vector<std::string> ret;
  DIR *dir;
  struct dirent *ent;

  if ((dir = opendir(folder.c_str())) != NULL)
  {
    /* print all the files and directories within directory */
    while ((ent = readdir(dir)) != NULL)
    {
      // check if entry is a file or a subfolder
      if (ent->d_type == DT_DIR)
      {
        std::string subfolder = ent->d_name;
        if (subfolder != "." && subfolder != "..")
        {
          if (subfolder[subfolder.length() - 1] == '/')
            subfolders.push_back(folder + subfolder);
          else
            subfolders.push_back(folder + "/" + subfolder);
        }
      }
      else if (ent->d_type == DT_REG)
      {
        std::string file_name = ent->d_name;
        if (folder[folder.length() - 1] == '/')
          files.push_back(folder + file_name);
        else
          files.push_back(folder + "/" + file_name);
      }
      else
      {
        std::cout << "[-] skipping  " << std::string(ent->d_name) << std::endl;
      }
    }
    closedir(dir);
  }
  else
  {
    /* could not open directory */
    perror("could not open directory");
    return ret;
  }
  std::cout << "\t[+] total sub-folders : " << subfolders.size() << std::endl;
  for (auto subfolder : subfolders)
  {
    subfiles = get_files_recursively(subfolder);
    files.insert(files.end(), subfiles.begin(), subfiles.end());
  }
  return files;
}
bool fit_fuzzsb(const std::string &fit_fuzzsb_folder)
{

  bool ret = true;
  std::vector<std::string> files = get_files_recursively(fit_fuzzsb_folder);
  std::vector<std::string> string_files, string_files_attacks, string_files_normal;

  // std::vector<provallo::vectorizer<std::string,real_t>*> vectorizers;

  std::vector<provallo::vectorizer<std::string, real_t> *> vectorizers_attacks;
  std::vector<provallo::vectorizer<std::string, real_t> *> vectorizers_normal;
  std::vector<provallo::auto_encoder<real_t, real_t> *> autoencoders;

  std::vector<std::string> exclude_list;
  exclude_list.push_back("COPYRIGHT");
  exclude_list.push_back("LICENSE");
  exclude_list.push_back("README");
  exclude_list.push_back("README.md");
  exclude_list.push_back("README.txt");
  exclude_list.push_back("README.rst");
  exclude_list.push_back("README.html");
  exclude_list.push_back("README.pdf");
  exclude_list.push_back(".JPEG");
  exclude_list.push_back(".png");
  exclude_list.push_back(".jpg");
  exclude_list.push_back(".jpeg");
  exclude_list.push_back(".gif");
  exclude_list.push_back("_copyright.txt");
  exclude_list.push_back(".gitignore");
  exclude_list.push_back(".git");

  // first , validate vectorizers mechanism works:
  if (!test_vectorizers())
  {
    std::cout << "[-] vectorizers test failed " << std::endl;
    return false;
  }

  for (auto file : files)
  {

    bool skip = false;
    for (auto exclude : exclude_list)
    {
      std::string exclude_me = file.substr(file.find_last_of("/") + 1);
      if (exclude_me.find(exclude) == std::string::npos)
      {
        skip = false;
      }
      else
      {
        skip = true;
        break;
      }
    }
    if (skip)
      continue;

    // check if attack or normal file
    if (file.find("attack") != std::string::npos || file.find("door") != std::string::npos || file.find("discovery") != std::string::npos)
    {
      string_files_attacks.push_back(file);
    }
    else
    {
      string_files_normal.push_back(file);
    }

    string_files.push_back(file);
  }
  std::cout << "[+] found " << string_files.size() << " files in folder " << fit_fuzzsb_folder << std::endl;
  std::cout << "[+] found " << files.size() << " files in folder " << fit_fuzzsb_folder << std::endl;
  std::cout << "[+] found " << string_files_attacks.size() << " attack files in folder " << fit_fuzzsb_folder << std::endl;
  std::cout << "[+] found " << string_files_normal.size() << " normal files in folder " << fit_fuzzsb_folder << std::endl;

  // feed the files to the pipeline,
  // the pipeline will load the files, vectorize them, train the autoencoder,

  // first we want to verify the vectorizers and autoencoders work properly
  vectorizers_attacks.push_back(new provallo::lda_vectorizer);
  vectorizers_attacks.push_back(new provallo::one_hot_vectorizer);
  vectorizers_attacks.push_back(new provallo::pca_vectorizer);

  vectorizers_normal.push_back(new provallo::lda_vectorizer);
  vectorizers_normal.push_back(new provallo::one_hot_vectorizer);
  vectorizers_normal.push_back(new provallo::pca_vectorizer);

  // vectorizers_normal.push_back(new provallo::auto_encoder_vectorizer<real_t,real_t>(autoencoders));
  // create vectorizers:
  //  vectorizers.push_back(new provallo::pca_vectorizer);
  // vectorizers.push_back(new provallo::word2vec_vectorizer);
  clock_t c_start, c_end, c_point;
  std::chrono::high_resolution_clock::time_point t_start, t_end, t_point;
  c_start = clock();
  t_start = std::chrono::high_resolution_clock::now();
  t_point = t_start;
  c_point = c_start;

  size_t total = string_files.size();
  size_t cur = 0;

  for (auto &fuzz_file : string_files_attacks)
  {
    std::ifstream fuzz(fuzz_file);
    std::string fuzz_string((std::istreambuf_iterator<char>(fuzz)),
                            std::istreambuf_iterator<char>());

    if (fuzz_string.size() == 0)
    {
      std::cout << "[-] skipping empty attack file " << fuzz_file << std::endl;
      continue;
    }

    // std::cout<<"[+] fuzzing "<<fuzz_file<< "("<<std::to_string(cur)+"/"+std::to_string(total)<<")"<<std::endl;
    for (auto &vectorizer : vectorizers_attacks)
    {

      std::string vectorizer_type = vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER]; //= vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER];//unknown vectorizer->get_type();
      if (vectorizer->get_type() < provallo::vectorizer_type::UNKNOWN_VECTORIZER)
      {
        vectorizer_type = vectorizer_types[vectorizer->get_type()];
      }
      std::cout << "[+] vectorizing attack with " << vectorizer_type << std::endl;
      vectorizer->add_document(fuzz_string);
      std::cout << "[+] vectorizing attack  with " << vectorizer_type << " done" << std::endl;
      std::cout << "[+] finished " << fuzz_file << std::to_string(cur) + "/" + std::to_string(total) << std::endl;
    }

    cur++;
  }
  for (auto &fuzz_file : string_files_normal)
  {
    std::ifstream fuzz(fuzz_file);
    std::string fuzz_string((std::istreambuf_iterator<char>(fuzz)),
                            std::istreambuf_iterator<char>());

    if (fuzz_string.size() == 0)
    {
      std::cout << "[-] skipping empty file " << fuzz_file << std::endl;
      continue;
    }
    for (auto &vectorizer : vectorizers_normal)
    {
      std::string vectorizer_type = vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER]; //= vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER];//unknown vectorizer->get_type();
      if (vectorizer->get_type() < provallo::vectorizer_type::UNKNOWN_VECTORIZER)
      {
        vectorizer_type = vectorizer_types[vectorizer->get_type()];
      }
      std::cout << "[+] vectorizing normal with " << vectorizer_type << std::endl;
      vectorizer->add_document(fuzz_string);
      std::cout << "[+] vectorizing normal  with " << vectorizer_type << " done" << std::endl;
      std::cout << "[+] finished " << fuzz_file << std::to_string(cur) + "/" + std::to_string(total) << std::endl;
    }
  }

  std::cout << "[+] finished vectorizing " << std::endl;
  c_end = clock();
  t_end = std::chrono::high_resolution_clock::now();
  std::cout << "[+] Test CPU time elapsed in s: "
            << (double)(c_end - c_point) / CLOCKS_PER_SEC << std::endl;
  std::cout << "[+] Test Wall time elapsed in s: "
            << std::chrono::duration<double>(t_end - t_point).count()
            << std::endl;
  std::cout << "[+] fitting vectorizers " << std::endl;

  // now that we have the vectorizers, we can train the autoencoders

  for (auto &v : vectorizers_attacks)
  {
    std::string vectorizer_type = vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER]; //= vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER];//unknown vectorizer->get_type();
    c_start = clock();
    t_start = std::chrono::high_resolution_clock::now();

    if (v->get_type() < provallo::vectorizer_type::UNKNOWN_VECTORIZER)
    {
      vectorizer_type = vectorizer_types[v->get_type()];
    }
    std::cout << "[+] fitting vectorizer " << vectorizer_type << std::endl;
    v->process_documents();
    std::cout << "[+] fitting vectorizer " << vectorizer_type << " done" << std::endl;

    c_end = clock();
    t_end = std::chrono::high_resolution_clock::now();

    std::cout << "[+] vectorizer CPU time elapsed in s: "
              << (double)(c_end - c_start) / CLOCKS_PER_SEC << std::endl;
    std::cout << "[+] vectorizer Wall time elapsed in s: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << std::endl;
    // print output size for each vectorizer :
    std::cout << "[+] vectorizer expected output size : " << std::to_string(v->get_output_size()) << std::endl;
  }

  for (auto &v : vectorizers_normal)
  {
    std::string vectorizer_type = vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER]; //= vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER];//unknown vectorizer->get_type();
    c_start = clock();
    t_start = std::chrono::high_resolution_clock::now();

    if (v->get_type() < provallo::vectorizer_type::UNKNOWN_VECTORIZER)
    {
      vectorizer_type = vectorizer_types[v->get_type()];
    }
    std::cout << "[+] fitting normal vectorizer " << vectorizer_type << std::endl;
    v->process_documents();
    std::cout << "[+] fitting normal  vectorizer " << vectorizer_type << " done" << std::endl;

    c_end = clock();
    t_end = std::chrono::high_resolution_clock::now();

    std::cout << "[+] vectorizer CPU time elapsed in s: "
              << (double)(c_end - c_start) / CLOCKS_PER_SEC << std::endl;
    std::cout << "[+] vectorizer Wall time elapsed in s: "
              << std::chrono::duration<double>(t_end - t_start).count()
              << std::endl;
    // print output size for each vectorizer :
    std::cout << "[+] vectorizer expected output size : " << std::to_string(v->get_output_size()) << std::endl;
  }

  // now that we have the autoencoders, we can train them
  // reiterate on the files and train the autoencoders
  // print total time
  c_end = clock();
  t_end = std::chrono::high_resolution_clock::now();
  std::cout << "[+] Test CPU time elapsed in s: "
            << (double)(c_end - c_point) / CLOCKS_PER_SEC << std::endl;
  std::cout << "[+] Test Wall time elapsed in s: "
            << std::chrono::duration<double>(t_end - t_point).count()
            << std::endl;
  // save vectorizers
  size_t vectr = 1;
  for (auto &vect : vectorizers_attacks)
  {
    // print vectorizer type
    std::string vectorizer_type = vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER]; // unknown vectorizer->get_type();
    if (vect->get_type() < provallo::vectorizer_type::UNKNOWN_VECTORIZER)
    {
      vectorizer_type = vectorizer_types[vect->get_type()];
    }
    std::cout << "[+] saving attack vectorizer " << vectorizer_type << std::endl;
    std::ofstream vectorizer_out("vectorizer_fuzzdb_attacks" + vectorizer_type + std::to_string(vectr) + ".json");

    vect->save(vectorizer_out);
    vect->gnuplot("vectorizer_fuzzdb_attacks" + vectorizer_type + std::to_string(vectr) + ".gnuplot");

    vectr++;
  }
  for (auto &vect : vectorizers_normal)
  {
    // print vectorizer type
    std::string vectorizer_type = vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER]; // unknown vectorizer->get_type();
    if (vect->get_type() < provallo::vectorizer_type::UNKNOWN_VECTORIZER)
    {
      vectorizer_type = vectorizer_types[vect->get_type()];
    }
    std::cout << "[+] saving benign vectorizer " << vectorizer_type << std::endl;
    std::ofstream vectorizer_out("vectorizer_fuzzdb_benign" + vectorizer_type + std::to_string(vectr) + ".json");

    vect->save(vectorizer_out);
    vect->gnuplot("vectorizer_fuzzdb_benign" + vectorizer_type + std::to_string(vectr) + ".gnuplot");
    vectr++;
  }

  std::cout << "[+] creating autoencoders " << std::endl;
  for (auto &vectorizer : vectorizers_attacks)
  {
    // get random file from normal files
    std::vector<real_t> input;
    size_t input_size = 0;
    for (;;)
    {
      std::string random_file = string_files_attacks[rand() % string_files_attacks.size()];

      std::cout << "[+] loading random attack file : " << random_file << std::endl;

      std::ifstream ifrandom(random_file, std::ios::binary | std::ios::ate);
      std::string fit_file; //((std::istreambuf_iterator<char>(ifrandom)),    std::istreambuf_iterator<char>());

      // print vectorizer type
      std::string vectorizer_type = vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER]; // unknown vectorizer->get_type();
      if (vectorizer->get_type() < provallo::vectorizer_type::UNKNOWN_VECTORIZER)
      {
        vectorizer_type = vectorizer_types[vectorizer->get_type()];
      }
      // read file:
      size_t nlines = 0;
      while (ifrandom.is_open() && ifrandom.good())
      {
        std::string line;
        ifrandom >> line;

        fit_file += line + "\n";
        nlines++;
      }
      ifrandom.close();
      std::cout << "[+] loaded random attack file : " << random_file << " , lines : " << std::to_string(nlines) << std::endl;

      std::cout << "[+] creating autoencoder for vectorizer:" << vectorizer_types[vectorizer->get_type()] << " , input size:" << std::to_string(input_size) << std::endl;

      c_start = clock();
      t_start = std::chrono::high_resolution_clock::now();
      input = vectorizer->predict(fit_file.c_str());
      input_size = input.size();
      if (input_size == 0) //< vectorizer->get_output_size()  )
      {
        input_size = sqrt(vectorizer->get_output_size());

        if (input_size <= 1 || fit_file.length() < 1)
        {
          std::cout << "[+] skipping autoencoder creation for vectorizer:" << vectorizer_types[vectorizer->get_type()] << " , input size:" << std::to_string(input_size) << ", file : " << fit_file << std::endl;
          continue;
        }
        for (size_t i = input.size(); i < input_size; i++)
        {
          input.push_back(real_t(i) / vectorizer->get_output_size());
        }
      }

      // push softmax classifiers as autoencoders
      provallo::auto_encoder<real_t, real_t> *ae = new provallo::softmax_classifier<real_t, real_t>(2, input_size, /*alpha*/ 0.1, /*lambda*/ 0.05); // input,hidden,output
      autoencoders.push_back(ae);

      // autoencoders.push_back(new provallo::auto_encoder<real_t,real_t>(input_size,input_size*(2*std::log2(input_size)),1));//input,hidden,output
      // c_end = clock ();
      c_end = clock();
      t_end = std::chrono::high_resolution_clock::now();
      std::cout << "[+] auto encoder init Test CPU time elapsed in s: " << (double)(c_end - c_start) / CLOCKS_PER_SEC << std::endl;

      std::cout << "[+] auto encoder init Test Wall time elapsed in s: " << std::chrono::duration<double>(t_end - t_start).count() << std::endl;
      std::cout << "[+] autoencoder created " << std::endl;
      break;
    } // end for vectorizers

  } // end for vectorizers
  // now that we have the autoencoders, we can train them
  for (auto &fuzz_file : string_files_attacks)
  {
    std::ifstream fuzz(fuzz_file, std::ios::binary | std::ios::ate);
    std::string data_string; //((std::istreambuf_iterator<char>(fuzz)),std::istreambuf_iterator<char>());
    size_t nlines = 0;
    while (fuzz.is_open() && fuzz.good())
    {
      std::string line;
      fuzz >> line;
      data_string += line + "\n";
      nlines++;
    }
    fuzz.close();
    std::cout << "[+] loaded attack file : " << fuzz_file << " , lines : " << std::to_string(nlines) << std::endl;
    // print vectorizer type
    size_t vectr = 0;
    for (auto &enc : autoencoders)
    {
      auto &vectorizer = vectorizers_attacks[vectr++];
      // print vectorizer type
      std::string vectorizer_type = vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER]; // unknown vectorizer->get_type();
      if (vectorizer->get_type() < provallo::vectorizer_type::UNKNOWN_VECTORIZER)
      {
        vectorizer_type = vectorizer_types[vectorizer->get_type()];
      }
      std::cout << "[+] vectorizing with " << vectorizer_type << std::endl;
      std::vector<real_t> input = vectorizer->predict(data_string);
      std::cout << "[+] vectorizing with " << vectorizer_type << " done" << std::endl;
      std::cout << "[+] training / loading autoencoder" << std::endl;
      // print vectorize output
      if (input.size() < 2)
      {
        std::cout << "[+] vectorizer output size : " << std::to_string(input.size()) << "/" << std::to_string(vectorizer->get_output_size()) << std::endl;
        size_t old_size = input.size();
        input.resize(enc->getInputDim());
        for (size_t i = old_size; i < input.size(); i++)
        {
          input[i] = i / enc->getInputDim();
        }
      }

      std::cout << "[+] vectorizer output size : " << std::to_string(input.size()) << std::endl;
      std::cout << "[+] autoencoder input size : " << std::to_string(enc->getInputDim()) << std::endl;

      // attack output

      double output[2] = {1, 1};
      // do
      //{
      // get dictionary and weight and train the autoencoder
      // train with vectorized data:
      auto t = input.size() / enc->getInputDim();
      for (size_t inputs = 0; inputs < enc->getInputDim() * t; inputs += enc->getInputDim())
      {
        enc->train((real_t *)input.data() + inputs, output, 1);
        std::cout << "[+] train output  : " << std::to_string(output[0]) << std::endl;
      }

      // test with vectorized data

      //      }while(output[0]+output[1]<0.99);
      std::cout << "[+] training autoencoder done" << std::endl;

      // evaluate the autoencoder
      std::vector<real_t> output_vector(output, output + 2);
      std::cout << "[+] evaluating autoencoder" << std::endl;
      enc->test(input.data(), output, 1);

      std::cout << "[+] evaluating autoencoder done" << std::endl;
      std::cout << "[+] output[0] : " << std::to_string(output[0]) << std::endl;
      // std::cout<<"[+] loss = "<<std::to_string(enc->get_loss())<<std::endl;

      std::cout << "[+] training autoencoder done" << std::endl;
      std::cout << "[+] training autoencoder done" << std::endl;
      enc->save("encoder_fuzzdb_" + vectorizer_type + ".json");
      // enc->load ("encoder_fuzzdb_"+vectorizer_type+".json");
    } // end for autoencoders

    c_start = clock();
    t_start = std::chrono::high_resolution_clock::now();
    std::cout << "[+] Test CPU time elapsed in s: "
              << (double)(c_end - c_point) / CLOCKS_PER_SEC << std::endl;
    std::cout << "[+] Test Wall time elapsed in s: "
              << std::chrono::duration<double>(t_end - t_point).count()
              << std::endl;
    std::cout << "[+] finished " << fuzz_file << std::endl;

  } // end for fuzz files

  // let's try without retraining softmax classfiers since the autoencoders are already trained
  // create softmax classifier
  // size_t n_classes =2 ,n_dimensions  = sqrt(autoencoders[0]->getInputDim());
  // real_t alpha = 1. , lambda = 0.15;

  provallo::softmax_classifier<real_t, real_t> softmax(*(autoencoders[0])); //(n_dimensions,n_classes,alpha,lambda);

  auto &selected_vectorizer = vectorizers_attacks[0];
  auto &selected_normal_vectorizer = vectorizers_normal[0];

  std::cout << "[+] attack vectorizer output size = " << std::to_string(selected_vectorizer->get_output_size()) << std::endl;
  std::cout << "[+] selected normal vectorizer output size = " << std::to_string(selected_normal_vectorizer->get_output_size()) << std::endl;
  char x = std::getchar();

  // create softmax classifier
  // size_t n_classes =2 ,n_dimensions  = sqrt(selected_vectorizer->get_output_size());
  // real_t alpha = 1. , lambda = 0.15;

  // provallo::softmax_classifier<real_t,real_t> softmax(n_dimensions,n_classes,alpha,lambda);
  // train softmax classifier
  std::cout << "[+] training softmax classifier " << std::endl;
  size_t n_classes = 2;
  provallo::matrix<real_t> out_mat(1, n_classes);
  size_t total_cases = 0; // string_files_attacks.size()+string_files_normal.size();
  size_t correct_classifications = 0;
  size_t error_classifications = 0;
  size_t true_positives = 0;
  size_t true_negatives = 0;
  size_t false_positives = 0;
  size_t false_negatives = 0;

  // first train with normal data
  for (auto &fuzz_file : string_files_normal)
  {
    try
    {
      std::ifstream fuzz(fuzz_file, std::ios::binary | std::ios::ate);

      if (!fuzz.is_open() || !fuzz.good())
      {
        std::cout << "[+] error opening file " << fuzz_file << std::endl;
        continue;
      }
      else
      {

        std::string data_string; //((std::istreambuf_iterator<char>(fuzz)),std::istreambuf_iterator<char>());
        size_t nlines = 0;
        while (fuzz.is_open() && fuzz.good())
        {
          std::string line;
          fuzz >> line;
          data_string += line + "\n";
          nlines++;
        }
        fuzz.close();
        std::cout << "[+] loaded normal file : " << fuzz_file << " , lines : " << std::to_string(nlines) << std::endl;
        std::cout << "[+] vectorizing with " << vectorizer_types[selected_normal_vectorizer->get_type()] << std::endl;
        std::vector<real_t> input = selected_normal_vectorizer->predict(data_string);
        std::cout << "[+] vectorizing with " << vectorizer_types[selected_normal_vectorizer->get_type()] << " done" << std::endl;
        std::cout << "[+] training softmax classifier" << std::endl;
        if (input.size() < softmax.getInputDim())
        {
          std::cout << "[+] vectorizer output size : " << std::to_string(input.size()) << "/" << std::to_string(selected_normal_vectorizer->get_output_size()) << std::endl;
          size_t old_size = input.size();
          input.resize(softmax.getInputDim());
          for (size_t i = old_size; i < input.size(); i++)
          {
            input[i] = i / real_t(softmax.getInputDim());
          }
        }
        out_mat.resize(1, n_classes);
        out_mat.fill(0.0);

        provallo::matrix<real_t> input_mat = provallo::matrix<real_t>(input.data(), 1, input.size());
        softmax.train(input, out_mat);
        std::cout << "[+] training softmax benign case done" << std::endl;
        std::cout << "[+] output = " << out_mat << std::endl;
        // update confusion

        for (size_t i = 0; i < out_mat.cols(); i++)
        {
          for (size_t j = 0; j < out_mat.rows(); j++)
          {
            total_cases++;
            if (out_mat(j, i) < 0.5)
            {
              // true positive
              true_positives++;
              correct_classifications++;
              std::cout << "[+] TP benign case detected" << std::endl;
            }
            else if (out_mat(j, i) == 0.5)
            {
              // error
              error_classifications++;
              true_negatives++;
              std::cout << "[+] TN error : softmax output = 0.5" << std::endl;
            }
            else if (out_mat(j, i) > 0.5)
            {
              // false positive
              false_positives++;
              error_classifications++;
              std::cout << "[+] FP attack case detected" << std::endl;
            } // else
            else
            {
              // false negative
              false_negatives++;
              error_classifications++;
              std::cout << "[+] FN attack case detected" << std::endl;
            } // else
          }   // end for softmax output
        }     // end for softmax output
      }       // and else
      // end try
    }
    catch (std::exception &e)
    {
      std::cout << "[+] exception : " << e.what() << std::endl;
    }

  } // end for normal files

  std::cout << "[+] training softmax classifier on normal cases done" << std::endl;
  // print confusion matrix
  std::cout << "[+] softmax training : " << std::endl;
  std::cout << "[+] total training samples:" << std::to_string(total_cases) << std::endl;
  std::cout << "[+] false positives : " << std::to_string(false_positives) << std::endl;
  std::cout << "[+] false negatives : " << std::to_string(false_negatives) << std::endl;
  std::cout << "[+] true positives : " << std::to_string(true_positives) << std::endl;
  std::cout << "[+] true negatives : " << std::to_string(true_negatives) << std::endl;
  std::cout << "[+] correct classifications : " << std::to_string(correct_classifications) << std::endl;
  std::cout << "[+] error classifications : " << std::to_string(error_classifications) << std::endl;
  std::cout << "[+] softmax benign training done" << std::endl;

  x += std::getchar();
  x++;
  // now do the same and train with the attack data a few times.
  for (size_t i = 0; i < 5; i++)
    for (auto &fuzz_file : string_files_attacks)
    {
      try
      {

        std::ifstream fuzz(fuzz_file, std::ios::binary | std::ios::ate);

        if (!fuzz.is_open() || !fuzz.good())
        {
          std::cout << "[+] error opening file " << fuzz_file << std::endl;
          continue;
        }
        // when out_mat is allocated to 1,n_classes no allocation is done
        out_mat = (provallo::matrix<real_t>::One(1, n_classes));

        std::string data_string; //((std::istreambuf_iterator<char>(fuzz)),std::istreambuf_iterator<char>());
        size_t nlines = 0;
        while (fuzz.is_open() && fuzz.good())
        {
          std::string line;
          fuzz >> line;
          data_string += line + "\n";
          nlines++;
        }
        fuzz.close();

        std::cout << "[+] fuzzing " << fuzz_file << std::endl;
        std::cout << "[+] vectorizing with " << vectorizer_types[selected_vectorizer->get_type()] << std::endl;
        std::vector<real_t> input = selected_vectorizer->predict(data_string);
        std::cout << "[+] vectorizing with " << vectorizer_types[selected_vectorizer->get_type()] << " done with size:" << std::to_string(input.size()) << std::endl;
        std::cout << "[+] training softmax classifier" << std::endl;
        if (input.size() < softmax.getInputDim())
        {
          std::cout << "[+] vectorizer output size : " << std::to_string(input.size()) << "/" << std::to_string(selected_vectorizer->get_output_size()) << std::endl;
          size_t old_size = input.size();
          input.resize(softmax.getInputDim());
          for (size_t i = old_size; i < input.size(); i++)
          {
            input[i] = i / real_t(softmax.getInputDim());
          }
        }
        out_mat.resize(1, n_classes);
        out_mat.fill(1.0);

        provallo::matrix<real_t> input_mat = provallo::matrix<real_t>(input.data(), 1, input.size());
        softmax.train(input_mat, out_mat);
        std::cout << "[+] training softmax attack cases done" << std::endl;
        std::cout << "[+] output = " << out_mat << std::endl;

        for (size_t i = 0; i < out_mat.cols(); i++)
        {
          for (size_t j = 0; j < out_mat.rows(); ++j)
          {
            // update confusion matrix
            total_cases++;
            if (out_mat(i, j) < 0.5)
            {
              // true negative
              false_positives++;
              error_classifications++;
              std::cout << "[+] benign case detected" << std::endl;
            }
            else if (out_mat(i, j) > 0.5)
            {
              // false positive
              true_positives++;
              correct_classifications++;
              std::cout << "[+] attack case detected" << std::endl;
            }
            else if (out_mat(i, j) == 0.5)
            {
              error_classifications++;
              true_negatives++;
              std::cout << "[+] error : softmax output = 0.5" << std::endl;
            }
            else
            {
              false_negatives++;
              error_classifications++;
              std::cout << "[+] error : softmax output = " << std::to_string(out_mat(i, j)) << std::endl;
            }

          } // end for softmax output
        }   // end for softmax output
      }     // end try
      catch (std::exception &e)
      {
        std::cout << "[+] exception : " << e.what() << std::endl;
      }
    } // end for attack files
  std::cout << "[+] training softmax classifier done" << std::endl;
  // print confusion matrix
  std::cout << "[+] softmax training : " << std::endl;
  std::cout << "[+] false positives : " << std::to_string(false_positives) << std::endl;
  std::cout << "[+] false negatives : " << std::to_string(false_negatives) << std::endl;
  std::cout << "[+] true positives : " << std::to_string(true_positives) << std::endl;
  std::cout << "[+] true negatives : " << std::to_string(true_negatives) << std::endl;
  std::cout << "[+] correct classifications : " << std::to_string(correct_classifications) << std::endl;
  std::cout << "[+] error classifications : " << std::to_string(error_classifications) << std::endl;
  std::cout << "[+] softmax training done" << std::endl;
  // save softmax classifier
  std::cout << "[+] saving softmax classifier" << std::endl;
  softmax.save("softmax_fuzzdb_attacks_trained.json");

  // now test the softmax classifier
  std::cout << "[+] testing softmax classifier  " << std::endl;

  // reset for test
  total_cases = 0;
  correct_classifications = 0;
  error_classifications = 0;
  true_positives = 0;
  true_negatives = 0;
  false_positives = 0;
  false_negatives = 0;

  for (auto &fuzz_file : string_files_normal)
  {
    std::ifstream fuzz(fuzz_file, std::ios::binary | std::ios::ate);
    provallo::matrix<real_t> out_mat(1, n_classes);
    std::string data_string;
    size_t nlines = 0;
    while (fuzz.is_open() && fuzz.good())
    {
      std::string line;
      fuzz >> line;
      data_string += line + "\n";
      nlines++;
    }
    fuzz.close();
    std::cout << "[+] loaded normal file : " << fuzz_file << " , lines : " << std::to_string(nlines) << std::endl;
    //((std::istreambuf_iterator<char>(fuzz)),std::istreambuf_iterator<char>());
    std::cout << "[+] fuzzing " << fuzz_file << std::endl;
    std::cout << "[+] vectorizing with " << vectorizer_types[selected_normal_vectorizer->get_type()] << std::endl;
    std::vector<real_t> input = selected_normal_vectorizer->predict(data_string);
    std::cout << "[+] vectorizing with " << vectorizer_types[selected_normal_vectorizer->get_type()] << " done" << std::endl;
    std::cout << "[+] testing softmax classifier" << std::endl;
    if (input.size() < softmax.getInputDim())
    {
      std::cout << "[+] vectorizer output size : " << std::to_string(input.size()) << "/" << std::to_string(selected_normal_vectorizer->get_output_size()) << std::endl;
      size_t old_size = input.size();
      input.resize(sqrt(sqrt(selected_normal_vectorizer->get_output_size())));
      for (size_t i = old_size; i < input.size(); i++)
      {
        input[i] = 0.0;
      }
    }
    provallo::matrix<real_t> input_mat = provallo::matrix<real_t>(input.data(), 1, input.size());
    softmax.predict(input_mat, out_mat);
    std::cout << "[+] testing softmax benign case done" << std::endl;
    std::cout << "[+] output = " << out_mat << std::endl;
    // update confusion matrix
    for (size_t i = 0; i < out_mat.cols(); i++)
    {
      for (size_t j = 0; j < out_mat.rows(); ++j)
      {
        total_cases++;
        if (out_mat(j, i) < 0.5)
        {
          // true negative
          true_negatives++;
          correct_classifications++;
          std::cout << "[+] benign case detected" << std::endl;
        }
        else if (out_mat(j, i) > 0.5)
        {
          // false positive
          false_positives++;
          error_classifications++;
          std::cout << "[+] attack case detected" << std::endl;
        }
        else if (out_mat(j, i) == 0.5)
        {
          // error
          error_classifications++;
          true_positives++;
          std::cout << "[+] error : softmax output = 0.5" << std::endl;
        }
        else
        {
          // error
          error_classifications++;
          false_negatives++;
          std::cout << "[+] error : softmax output = " << std::to_string(out_mat(j, i)) << std::endl;
        }

      } // end for softmax output
    }   // end for softmax output
  }     // end for normal files

  std::cout << "[+] testing softmax classifier done" << std::endl;

  // test with attack data
  for (auto &fuzz_file : string_files_attacks)
  {
    std::ifstream fuzz(fuzz_file, std::ios::binary | std::ios::ate);
    provallo::matrix<real_t> out_mat(provallo::matrix<real_t>::Zero(1, n_classes));
    std::string data_string; //((std::istreambuf_iterator<char>(fuzz)),std::istreambuf_iterator<char>());
    size_t nlines = 0;
    while (fuzz.is_open() && fuzz.good())
    {
      std::string line;
      fuzz >> line;
      data_string += line + "\n";
      nlines++;
    }
    fuzz.close();
    std::cout << "[+] loaded attack file : " << fuzz_file << " , lines : " << std::to_string(nlines) << std::endl;

    std::cout << "[+] vectorizing with " << vectorizer_types[selected_vectorizer->get_type()] << std::endl;
    std::vector<real_t> input = selected_vectorizer->predict(data_string);
    std::cout << "[+] vectorizing with " << vectorizer_types[selected_vectorizer->get_type()] << " done" << std::endl;
    std::cout << "[+] testing softmax classifier" << std::endl;
    if (input.size() < softmax.getInputDim())
    {
      std::cout << "[+] vectorizer output size : " << std::to_string(input.size()) << "/" << std::to_string(selected_normal_vectorizer->get_output_size()) << std::endl;
      size_t old_size = input.size();
      input.resize(sqrt(sqrt(selected_vectorizer->get_output_size())));
      for (size_t i = old_size; i < input.size(); i++)
      {
        input[i] = 0.0;
      }
    }
    provallo::matrix<real_t> input_mat = provallo::matrix<real_t>(input.data(), 1, input.size());
    softmax.predict(input_mat, out_mat);
    std::cout << "[+] testing softmax attack case done" << std::endl;
    std::cout << "[+] output = " << out_mat << std::endl;
    // update confusion matrix
    for (size_t i = 0; i < out_mat.cols(); i++)
    {
      for (size_t j = 0; j < out_mat.rows(); ++j)
      {
        total_cases++;
        if (out_mat(j, i) < 0.5)
        {
          // true negative
          false_positives++;
          error_classifications++;
          std::cout << "[+] benign case detected" << std::endl;
        }
        else if (out_mat(j, i) > 0.5)
        {
          // false positive
          true_positives++;
          correct_classifications++;
          std::cout << "[+] attack case detected" << std::endl;
        }
        else if (out_mat(j, i) == 0.5)
        {
          // error
          error_classifications++;
          true_negatives++;
          std::cout << "[+] error : softmax output = 0.5" << std::endl;
        }
        else
        {
          // error
          error_classifications++;
          false_negatives++;
          std::cout << "[+] error : softmax output = " << std::to_string(out_mat(j, i)) << std::endl;
        }

      } // end for softmax output
    }   // end for softmax output

  } // end for attack files
  std::cout << "[+] testing softmax classifier done" << std::endl;
  std::cout << "[+] softmax test results: " << std::endl;
  std::cout << "[+] total test samples:" << std::to_string(total_cases) << std::endl;
  std::cout << "[+] false positives : " << std::to_string(false_positives) << std::endl;
  std::cout << "[+] false negatives : " << std::to_string(false_negatives) << std::endl;
  std::cout << "[+] true positives : " << std::to_string(true_positives) << std::endl;
  std::cout << "[+] true negatives : " << std::to_string(true_negatives) << std::endl;
  std::cout << "[+] correct classifications : " << std::to_string(correct_classifications) << std::endl;
  std::cout << "[+] error classifications : " << std::to_string(error_classifications) << std::endl;
  std::cout << "[+] softmax training done" << std::endl;

  // save softmax classifier
  // save softmax classifier --> softmax_softmax_fuzzdb_test.json
  softmax.save("softmax_fuzzdb_test.json");
  std::cout << "[+] saving softmax classifier done" << std::endl;

  // save autoencoders
  size_t enc_index = 1;
  for (auto &enc : autoencoders)
  {
    std::string file_name = "encoder_fuzzdb" + std::to_string(enc_index++) + ".json";
    enc->save("encoder_fuzzdb.json");

    //plot autoencoder
    enc->gnuplot(file_name + ".gnuplot");

    //  delete enc;
  }
  // delete vectorizers :
  // delete autoencoders

  /*
  for(auto& vectorizer : vectorizers_attacks)
  {
    delete vectorizer;
  }
  for(auto& vectorizer : vectorizers_normal)
  {
    delete vectorizer;
  }
  */
  return ret;

} //  end of fit_fuzzdb
  //-----------------------------------------------------------------------------

// simplified version of fit_fuzzdb, for testing purposes:
bool fit_fuzzdb_test(const std::string &ff)
{
  std::vector<std::string> files = get_files_recursively(ff);
  // separate the files into attack and normal files
  std::vector<std::string> string_files, string_files_attacks, string_files_normal;
  // create vectorizers for attacks and normal files
  std::string vectorizer_type = vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER]; // unknown vectorizer->get_type();

  // std::vector<provallo::vectorizer<std::string,real_t>*> vectorizers;

  std::vector<provallo::vectorizer<std::string, real_t> *> vectorizers_attacks;
  std::vector<provallo::vectorizer<std::string, real_t> *> vectorizers_normal;
  std::vector<provallo::softmax_classifier<real_t, real_t> *> soft_classifiers;
  // exclude files:
  std::vector<std::string> exclude_list;
  exclude_list.push_back("COPYRIGHT");
  exclude_list.push_back("LICENSE");
  exclude_list.push_back("README");
  exclude_list.push_back("README.md");
  exclude_list.push_back("README.txt");
  exclude_list.push_back("README.rst");
  exclude_list.push_back("README.html");
  exclude_list.push_back("README.pdf");
  exclude_list.push_back(".JPEG");
  exclude_list.push_back(".png");
  exclude_list.push_back(".jpg");
  exclude_list.push_back(".jpeg");
  exclude_list.push_back(".gif");
  exclude_list.push_back("_copyright.txt");
  exclude_list.push_back(".gitignore");
  exclude_list.push_back(".git");

  // first , validate vectorizers mechanism works:
  if (!test_vectorizers())
  {
    std::cout << "[-] vectorizers test failed " << std::endl;
    return false;
  }

  for (auto file : files)
  {

    bool skip = false;
    for (auto exclude : exclude_list)
    {
      std::string exclude_me = file.substr(file.find_last_of("/") + 1);
      if (exclude_me.find(exclude) == std::string::npos)
      {
        skip = false;
      }
      else
      {
        skip = true;
        break;
      }
    }
    if (skip)
      continue;

    // check if attack or normal file
    if (file.find("attack") != std::string::npos || file.find("door") != std::string::npos || file.find("discovery") != std::string::npos)
    {
      string_files_attacks.push_back(file);
    }
    else
    {
      string_files_normal.push_back(file);
    }

    string_files.push_back(file);
  }
  std::cout << "[+] found " << string_files.size() << " files in folder " << ff << std::endl;
  std::cout << "[+] found " << files.size() << " files in folder " << ff << std::endl;
  std::cout << "[+] found " << string_files_attacks.size() << " attack files in folder " << ff << std::endl;
  std::cout << "[+] found " << string_files_normal.size() << " normal files in folder " << ff << std::endl;

  // feed the files to the pipeline,
  // the pipeline will load the files, vectorize them, train the autoencoder,

  // first we want to verify the vectorizers and autoencoders work properly
  vectorizers_attacks.push_back(new provallo::lda_vectorizer);
  vectorizers_attacks.push_back(new provallo::one_hot_vectorizer);
  vectorizers_attacks.push_back(new provallo::pca_vectorizer);

  vectorizers_normal.push_back(new provallo::lda_vectorizer);
  vectorizers_normal.push_back(new provallo::one_hot_vectorizer);
  vectorizers_normal.push_back(new provallo::pca_vectorizer);

  // fit the vectorizers with the data files :
  // first fit the vectorizers with the normal files :
  for (auto &fuzz_file : string_files_normal)
  {
    std::ifstream fuzz(fuzz_file, std::ios::binary | std::ios::ate);
    std::string data_string; //((std::istreambuf_iterator<char>(fuzz)),std::istreambuf_iterator<char>());
    size_t nlines = 0;
    while (fuzz.is_open() && fuzz.good())
    {
      std::string line;
      fuzz >> line;
      data_string += line + "\n";
      nlines++;
    }
    fuzz.close();
    std::cout << "[+] loaded normal file : " << fuzz_file << " , lines : " << std::to_string(nlines) << std::endl;
    // print vectorizer type
    // size_t vectr=0;
    vectorizer_type = vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER]; // unknown vectorizer->get_type();

    for (auto &vectorizer : vectorizers_normal)
    {
      // print vectorizer type
      if (vectorizer->get_type() < provallo::vectorizer_type::UNKNOWN_VECTORIZER)
      {
        vectorizer_type = vectorizer_types[vectorizer->get_type()];
      }
      std::cout << "[+] vectorizing with " << vectorizer_type << std::endl;
      vectorizer->add_document(data_string);
      std::cout << "[+] vectorizing with " << vectorizer_type << " done" << std::endl;
      std::cout << "[+] training softmax classifier" << std::endl;
      // print vectorize output
    }
    for (auto &vectorizer : vectorizers_normal)
    {
      vectorizer->process_documents();
      vectorizer_type = vectorizer_types[vectorizer->get_type()];

      std::cout << "[+] vectorizer  " << vectorizer_type << "output size = " << std::to_string(vectorizer->get_output_size()) << std::endl;
    }

  } // end for normal files
  // now fit the vectorizers with the attack files :
  for (auto &fuzz_file : string_files_attacks)
  {
    std::ifstream fuzz(fuzz_file, std::ios::binary | std::ios::ate);
    std::string data_string; //((std::istreambuf_iterator<char>(fuzz)),std::istreambuf_iterator<char>());
    size_t nlines = 0;
    while (fuzz.is_open() && fuzz.good())
    {
      std::string line;
      fuzz >> line;
      data_string += line + "\n";
      nlines++;
    }
    fuzz.close();
    std::cout << "[+] loaded attack file : " << fuzz_file << " , lines : " << std::to_string(nlines) << std::endl;
    // print vectorizer type
    // size_t vectr=0;
    vectorizer_type = vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER]; // unknown vectorizer->get_type();

    for (auto &vectorizer : vectorizers_attacks)
    {
      // print vectorizer type
      if (vectorizer->get_type() < provallo::vectorizer_type::UNKNOWN_VECTORIZER)
      {
        vectorizer_type = vectorizer_types[vectorizer->get_type()];
      }
      std::cout << "[+] vectorizing with " << vectorizer_type << std::endl;
      vectorizer->add_document(data_string);
      std::cout << "[+] vectorizing with " << vectorizer_type << " done" << std::endl;
      std::cout << "[+] training softmax classifier" << std::endl;
      // print vectorize output
    }
    // end for attack files
  }
  // process documents
  for (auto &vectorizer : vectorizers_attacks)
  {
    vectorizer->process_documents();
    vectorizer_type = vectorizer_types[vectorizer->get_type()];

    std::cout << "[+]attack vectorizer  " << vectorizer_type << "output size = " << std::to_string(vectorizer->get_output_size()) << std::endl;
  }
  // process documents
  for (auto &vectorizer : vectorizers_normal)
  {
    vectorizer->process_documents();
    vectorizer_type = vectorizer_types[vectorizer->get_type()];

    std::cout << "[+]normal vectorizer  " << vectorizer_type << "output size = " << std::to_string(vectorizer->get_output_size()) << std::endl;
  }
  // now train the softmax classfiers  with normal and attacks data :
  // create softmax classifier
  size_t n_classes = 2, n_dimensions = 2;
  real_t alpha = 1., lambda = 0.15;
  provallo::softmax_classifier<real_t, real_t> softmax(n_dimensions, n_classes, alpha, lambda);
  // train softmax classifier
  std::cout << "[+] training softmax classifier " << std::endl;
  size_t total_cases = 0;
  size_t correct_classifications = 0;
  size_t error_classifications = 0;
  size_t true_positives = 0;
  size_t true_negatives = 0;
  size_t false_positives = 0;
  size_t false_negatives = 0;
  // first train with normal data
  // open normal files and pass to both attack and normal vectorizers and sum the outputs of each:
  for (auto &fuzz_file : string_files_normal)
  {
    std::ifstream fuzz(fuzz_file, std::ios::binary | std::ios::ate);
    std::string data_string; //((std::istreambuf_iterator<char>(fuzz)),std::istreambuf_iterator<char>());
    size_t nlines = 0;
    while (fuzz.is_open() && fuzz.good())
    {
      std::string line;
      fuzz >> line;
      data_string += line + "\n";
      nlines++;
    }
    fuzz.close();
    std::cout << "[+] loaded normal file : " << fuzz_file << " , lines : " << std::to_string(nlines) << std::endl;
    // print vectorizer type
    // size_t vectr=0;
    vectorizer_type = vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER]; // unknown vectorizer->get_type();

    for (size_t i = 0; i < std::min(vectorizers_attacks.size(), vectorizers_normal.size()); i++)
    {
      // create input from the sum of the outputs of each vectorizer with the normal data.
      std::vector<real_t> input1 = vectorizers_attacks[i]->predict(data_string);
      std::vector<real_t> input2 = vectorizers_normal[i]->predict(data_string);
      real_t input1_sum = std::accumulate(input1.begin(), input1.end(), 0.0);
      real_t input2_sum = std::accumulate(input2.begin(), input2.end(), 0.0);
      std::vector<real_t> input = {input1_sum, input2_sum};
      std::cout << "[+] vectorizing with " << vectorizer_type << std::endl;
      std::cout << "[+] vectorizing with " << vectorizer_type << " done" << std::endl;
      std::cout << "[+] training softmax classifier" << std::endl;
      // print vectorize output
      std::cout << "[+] vectorizer  " << vectorizer_type << "output size = " << std::to_string(vectorizers_attacks[i]->get_output_size()) << std::endl;
      std::cout << "[+] vectorizer  " << vectorizer_type << "output size = " << std::to_string(vectorizers_normal[i]->get_output_size()) << std::endl;
      std::cout << "[+] softmax input size = " << std::to_string(softmax.getInputDim()) << std::endl;
      if (input.size() < softmax.getInputDim())
      {
        std::cout << "[+] vectorizer output size : " << std::to_string(input.size()) << "/" << std::to_string(softmax.getInputDim()) << std::endl;
        size_t old_size = input.size();
        input.resize(softmax.getInputDim());
        for (size_t i = old_size; i < input.size(); i++)
        {
          input[i] = 0.0;
        }
      }
      provallo::matrix<real_t> input_mat(input.data(), 1, input.size());
      provallo::matrix<real_t> out_mat(1, n_classes);
      // out_mat.fill(0.0); no need
      softmax.train(input_mat, out_mat);
    }
  } // end for normal files
  // now train with attack data
  for (auto &fuzz_file : string_files_attacks)
  {
    std::ifstream fuzz(fuzz_file, std::ios::binary | std::ios::ate);
    std::string data_string; //((std::istreambuf_iterator<char>(fuzz)),std::istreambuf_iterator<char>());
    size_t nlines = 0;
    while (fuzz.is_open() && fuzz.good())
    {
      std::string line;
      fuzz >> line;
      data_string += line + "\n";
      nlines++;
    }
    fuzz.close();
    std::cout << "[+] loaded attack file : " << fuzz_file << " , lines : " << std::to_string(nlines) << std::endl;
    // print vectorizer type
    // size_t vectr=0;
    vectorizer_type = vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER]; // unknown vectorizer->get_type();

    for (size_t i = 0; i < std::min(vectorizers_attacks.size(), vectorizers_normal.size()); i++)
    {
      // create input from the sum of the outputs of each vectorizer with the
      // attack data.
      std::vector<real_t> input1 = vectorizers_attacks[i]->predict(data_string);
      std::vector<real_t> input2 = vectorizers_normal[i]->predict(data_string);
      real_t input1_sum = std::accumulate(input1.begin(), input1.end(), 0.0);
      real_t input2_sum = std::accumulate(input2.begin(), input2.end(), 0.0);
      std::vector<real_t> input = {input1_sum, input2_sum};
      std::cout << "[+] vectorizing with " << vectorizer_type << std::endl;
      std::cout << "[+] vectorizing with " << vectorizer_type << " done" << std::endl;
      std::cout << "[+] training softmax classifier" << std::endl;
      // print vectorize output
      std::cout << "[+]attack vectorizer  " << vectorizer_type << "output size = " << std::to_string(vectorizers_attacks[i]->get_output_size()) << std::endl;
      std::cout << "[+]benign vectorizer  " << vectorizer_type << "output size = " << std::to_string(vectorizers_normal[i]->get_output_size()) << std::endl;
      std::cout << "[+] softmax input size = " << std::to_string(softmax.getInputDim()) << std::endl;
      if (input.size() < softmax.getInputDim())
      {
        std::cout << "[+] vectorizer output size : " << std::to_string(input.size()) << "/" << std::to_string(softmax.getInputDim()) << std::endl;
        size_t old_size = input.size();
        input.resize(softmax.getInputDim());
        for (size_t i = old_size; i < input.size(); i++)
        {
          input[i] = 0.0;
        }
      }
      provallo::matrix<real_t> input_mat(input.data(), 1, input.size());
      provallo::matrix<real_t> out_mat(1, n_classes);
      out_mat.fill(1.0);
      softmax.train(input_mat, out_mat);
    }
  } // end for attack files
  std::cout << "[+] training softmax classifier done" << std::endl;
  // print confusion matrix
  std::cout << "[+] softmax training : " << std::endl;
  std::cout << "[+] false positives : " << std::to_string(false_positives) << std::endl;
  std::cout << "[+] false negatives : " << std::to_string(false_negatives) << std::endl;
  std::cout << "[+] true positives : " << std::to_string(true_positives) << std::endl;
  std::cout << "[+] true negatives : " << std::to_string(true_negatives) << std::endl;
  std::cout << "[+] correct classifications : " << std::to_string(correct_classifications) << std::endl;

  std::cout << "[+] error classifications : " << std::to_string(error_classifications) << std::endl;
  std::cout << "[+] softmax training done" << std::endl;
  // save softmax classifier
  std::cout << "[+] saving softmax classifier" << std::endl;
  softmax.save(std::string("softmax_fuzzdb_attacks_") + vectorizer_type + std::string("trained.json"));
  std::cout << "[+] saving softmax classifier done" << std::endl;
  // now test the softmax classifier
  std::cout << "[+] testing softmax classifier  " << std::endl;
  // reset for test
  total_cases = 0;
  correct_classifications = 0;
  error_classifications = 0;

  true_positives = 0;
  true_negatives = 0;
  false_positives = 0;
  false_negatives = 0;

  // first test with normal data
  // open normal files and pass to both attack and normal vectorizers and sum the outputs of each:

  for (auto &fuzz_file : string_files_normal)
  {
    std::ifstream fuzz(fuzz_file, std::ios::binary | std::ios::ate);
    std::string data_string; //((std::istreambuf_iterator<char>(fuzz)),std::istreambuf_iterator<char>());
    size_t nlines = 0;
    while (fuzz.is_open() && fuzz.good())
    {
      std::string line;
      fuzz >> line;
      data_string += line + "\n";
      nlines++;
    }
    fuzz.close();
    std::cout << "[+] loaded normal file : " << fuzz_file << " , lines : " << std::to_string(nlines) << std::endl;
    // print vectorizer type
    // size_t vectr=0;
    vectorizer_type = vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER]; // unknown vectorizer->get_type();

    for (size_t i = 0; i < std::min(vectorizers_attacks.size(), vectorizers_normal.size()); i++)
    {
      // create input from the sum of the outputs of each vectorizer with the normal data.
      std::vector<real_t> input1 = vectorizers_attacks[i]->predict(data_string);
      std::vector<real_t> input2 = vectorizers_normal[i]->predict(data_string);
      real_t input1_sum = std::accumulate(input1.begin(), input1.end(), 0.0);
      real_t input2_sum = std::accumulate(input2.begin(), input2.end(), 0.0);
      std::vector<real_t> input = {input1_sum, input2_sum};
      std::cout << "[+] vectorizing attacks with " << vectorizer_type << std::endl;
      std::cout << "[+] vectorizing benign with " << vectorizer_type << " done" << std::endl;
      std::cout << "[+] testing softmax classifier" << std::endl;
      // print vectorize output
      std::cout << "[+] vectorizer  " << vectorizer_type << "output size = " << std::to_string(vectorizers_attacks[i]->get_output_size()) << std::endl;
      // std::cout<<"[+] vectorizer  "<<vectorizer_type << "output size = "<<std::to_string(vectorizers_normal[i]->get_output_size())<<std::endl;
      std::cout << "[+] softmax input size = " << std::to_string(softmax.getInputDim()) << std::endl;
      if (input.size() < softmax.getInputDim())
      {
        std::cout << "[+] vectorizer output size : " << std::to_string(input.size()) << "/" << std::to_string(softmax.getInputDim()) << std::endl;
        size_t old_size = input.size();
        input.resize(softmax.getInputDim());
        for (size_t i = old_size; i < input.size(); i++)
        {
          input[i] = 0.0;
        }
      }
      provallo::matrix<real_t> input_mat(input.data(), 1, input.size());
      provallo::matrix<real_t> out_mat(1, n_classes);
      softmax.predict(input_mat, out_mat);
      std::cout << "[+] testing softmax benign case done" << std::endl;
      std::cout << "[+] output = " << out_mat << std::endl;
      // update confusion matrix
      for (size_t i = 0; i < out_mat.cols(); i++)
      {
        for (size_t j = 0; j < out_mat.rows(); ++j)
        {
          total_cases++;
          if (out_mat(j, i) < 0.5)
          {
            // true negative
            true_negatives++;
            correct_classifications++;
            std::cout << "[+] benign case detected" << std::endl;
          }
          else if (out_mat(j, i) > 0.5)
          {
            // false positive
            false_positives++;
            error_classifications++;
            std::cout << "[+] attack case detected" << std::endl;
          }
          else if (out_mat(j, i) == 0.5)
          {
            // error
            error_classifications++;
            true_positives++;
            std::cout << "[+] error : softmax output = 0.5" << std::endl;
          }
          else
          {
            // error
            error_classifications++;
            false_negatives++;
            std::cout << "[+] error : softmax output = " << std::to_string(out_mat(j, i)) << std::endl;
          }

        } // end for softmax output
      }   // end for softmax output
    }     // end for vectorizers
  }       // end for normal files
  // now test with attack data
  for (auto &fuzz_file : string_files_attacks)
  {
    std::ifstream fuzz(fuzz_file, std::ios::binary | std::ios::ate);
    std::string data_string; //((std::istreambuf_iterator<char>(fuzz)),std::istreambuf_iterator<char>());
    size_t nlines = 0;
    while (fuzz.is_open() && fuzz.good())
    {
      std::string line;
      fuzz >> line;
      data_string += line + "\n";
      nlines++;
    }
    fuzz.close();
    std::cout << "[+] loaded attack file : " << fuzz_file << " , lines : " << std::to_string(nlines) << std::endl;
    // print vectorizer type
    // size_t vectr=0;
    vectorizer_type = vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER]; // unknown vectorizer->get_type();

    for (size_t i = 0; i < std::min(vectorizers_attacks.size(), vectorizers_normal.size()); i++)
    {
      // create input from the sum of the outputs of each vectorizer with the
      // attack data.
      std::vector<real_t> input1 = vectorizers_attacks[i]->predict(data_string);
      std::vector<real_t> input2 = vectorizers_normal[i]->predict(data_string);
      real_t input1_sum = std::accumulate(input1.begin(), input1.end(), 0.0);
      real_t input2_sum = std::accumulate(input2.begin(), input2.end(), 0.0);
      std::vector<real_t> input = {input1_sum, input2_sum};
      std::cout << "[+] vectorizing attacks with " << vectorizer_type << std::endl;
      std::cout << "[+] vectorizing benign with " << vectorizer_type << " done" << std::endl;
      std::cout << "[+] testing softmax classifier" << std::endl;
      // print vectorize output
      std::cout << "[+]attack vectorizer  " << vectorizer_type << "output size = " << std::to_string(vectorizers_attacks[i]->get_output_size()) << std::endl;
      // std::cout<<"[+]benign vectorizer  "<<vectorizer_type << "output size = "<<std::to_string(vectorizers_normal[i]->get_output_size())<<std::endl;
      std::cout << "[+] softmax input size = " << std::to_string(softmax.getInputDim()) << std::endl;
      if (input.size() < softmax.getInputDim())
      {
        std::cout << "[+] vectorizer output size : " << std::to_string(input.size()) << "/" << std::to_string(softmax.getInputDim()) << std::endl;
        size_t old_size = input.size();
        input.resize(softmax.getInputDim());
        for (size_t i = old_size; i < input.size(); i++)
        {
          input[i] = 0.0;
        }
      }
      provallo::matrix<real_t> input_mat(input.data(), 1, input.size());
      provallo::matrix<real_t> out_mat(1, n_classes);
      out_mat.fill(1.0);
      softmax.predict(input_mat, out_mat);
      std::cout << "[+] testing softmax attack case done" << std::endl;
      std::cout << "[+] output = " << out_mat << std::endl;
      // update confusion matrix
      for (size_t i = 0; i < out_mat.cols(); i++)
      {
        for (size_t j = 0; j < out_mat.rows(); ++j)
        {
          total_cases++;
          if (out_mat(j, i) < 0.5)
          {
            // true negative
            false_positives++;
            error_classifications++;
            std::cout << "[+] benign case detected" << std::endl;
          }
          else if (out_mat(j, i) > 0.5)
          {
            // false positive
            true_positives++;
            correct_classifications++;
            std::cout << "[+] attack case detected" << std::endl;
          }
          else if (out_mat(j, i) == 0.5)
          {
            // error
            error_classifications++;
            true_negatives++;
            std::cout << "[+] error : softmax output = 0.5" << std::endl;
          }
          else
          {
            // error
            error_classifications++;
            false_negatives++;
            std::cout << "[+] error : softmax output = " << std::to_string(out_mat(j, i)) << std::endl;
          }

        } // end for softmax output
      }   // end for softmax output
    }     // end for vectorizers
  }       // end for attack files
  std::cout << "[+] testing softmax classifier done" << std::endl;
  std::cout << "[+] softmax test results: " << std::endl;
  std::cout << "[+] total test samples:" << std::to_string(total_cases) << std::endl;
  std::cout << "[+] false positives : " << std::to_string(false_positives) << std::endl;
  std::cout << "[+] false negatives : " << std::to_string(false_negatives) << std::endl;
  std::cout << "[+] true positives : " << std::to_string(true_positives) << std::endl;
  std::cout << "[+] true negatives : " << std::to_string(true_negatives) << std::endl;
  std::cout << "[+] correct classifications : " << std::to_string(correct_classifications) << std::endl;
  std::cout << "[+] error classifications : " << std::to_string(error_classifications) << std::endl;
  std::cout << "[+] softmax test done" << std::endl;
  // save softmax classifier
  // save softmax classifier --> softmax_softmax_fuzzdb_test.json
  softmax.save(std::string("softmax_fuzzdb_attacks_") + vectorizer_type + std::string("test.json"));
  std::cout << "[+] saving softmax classifier done" << std::endl;
  // save vectorizers
  // save vectorizers --> vectorizers_fuzzdb_test.json
  std::cout << "[+] saving vectorizers" << std::endl;
  for (auto &vectorizer : vectorizers_attacks)
  {
    vectorizer->save(std::string("vectorizer_fuzzdb_attacks_") + vectorizer_type + std::string("test.json"));
  }
  for (auto &vectorizer : vectorizers_normal)
  {
    vectorizer->save(std::string("vectorizer_fuzzdb_normal_") + vectorizer_type + std::string("test.json"));
  }
  std::cout << "[+] saving vectorizers done" << std::endl;
  // delete vectorizers
  for (auto &vectorizer : vectorizers_attacks)
  {
    delete vectorizer;
  }
  for (auto &vectorizer : vectorizers_normal)
  {
    delete vectorizer;
  }
  return true;

} // end fit_fuzzdb_test
void test_spike_train_generator()
{
 
  real_t sigma = 1.0, mu = 0.1, dt = 0.1, t_min = 0.01, t_max = 1000.0;
  // create spike train generator
  provallo::gaussian_spike_train_generator<real_t> test_spike_train_generator(sigma, mu, dt, t_min, t_max);
  // generate spike train
  std::vector<real_t> train = test_spike_train_generator.generate();
  while (test_spike_train_generator.refine(train) <= 0.0)
  {
    std::cout << "[+] spike not  refined" << std::endl; 
    mu+=0.01;
    sigma+=0.01;
    test_spike_train_generator.set_mu(mu);

    train = test_spike_train_generator.generate();
  }
  std::cout << "[+] spike train refined" << std::endl;
  // print spike train
  std::cout << "[+] spike train  " << std::endl;
  std::cout << "[+] saving spike train size : " << std::to_string(train.size()) << std::endl;
  std::ofstream spike_train_file("spike_train.DAT", std::ios::binary | std::ios::ate);
  if (spike_train_file.is_open() && spike_train_file.good())
  {
    size_t nlines = 0;
    auto refined = train;
    const auto &output_ = test_spike_train_generator.get_output();
    for (auto &spike : train)
    {
      refined.push_back(spike);
      spike_train_file << std::to_string(nlines) << " " << spike << " " << std::to_string(output_[nlines]) << std::endl;
      nlines++;
    }
    spike_train_file.close();

    std::string gnuplot_script = "set terminal png\n"
                                 "set output \"spikes.png\"\n"
                                 "set title \"spike train\"\n"
                                 "set xlabel \"time\"\n"
                                 "set ylabel \"spike\"\n"
                                 "set zlabel \"sum\"\n"
                                 "set xrange [0:10000]\n"
                                 "set yrange [0:1]\n"
                                 "set cblabel \"spike\"\n"
                                 "set cbtics ( 0, 1 )\n"
                                 "set cbrange [0:1]\n"
                                 "set pm3d map\n"
                                 "set palette defined ( 0 \"white\", 1 \"red\" )\n"
                                 "set view map\n"
                                 "set grid\n"
                                 "set key off\n"
                                 "set ticslevel 0\n"
                                 "set tics out nomirror\n"
                                 "set tics scale 0.5\n"
                                 // splot:
                                 //   Warning: Single isoline (scan) is not enough for a pm3d plot.
                                 //   Hint: Missing blank lines in the data file? See 'help pm3d' and FAQ.
                                 //   Hint: If your data is all-zeros, you may need to use the 'set dgrid3d' command.
                                 "set dgrid3d 500,500,500\n"
                                 "set hidden3d\n"
                                 "set pm3d depthorder\n"
                                 "set pm3d interpolate 1,1\n"
                                 "set pm3d lighting primary 0.5 specular 0.5\n"
                                 // splot:

                                 "splot \"spike_train.DAT\"  with pm3d\n";

    // plot:
    //"plot \"spike_train.DAT\" using 1:2 with lines\n"

    std::ofstream gnuplot_script_file("spikes.gp", std::ios::binary | std::ios::ate);

    // save gnuplot script
    if (gnuplot_script_file.is_open() && gnuplot_script_file.good())
    {
      gnuplot_script_file << gnuplot_script;
      gnuplot_script_file.close();
      std::cout << "[+] spiketrain script saved" << std::endl;
    }
    else
    {
      std::cout << "[+] error opening spiketrain script file" << std::endl;
    }
  }
  else
  {
    std::cout << "[+] error opening spike train file" << std::endl;
  }
  std::getchar();
}


void validate_simple_softmax ()
{
    //multiclass problem.
    //create softmax classifier
    size_t n_classes=4,n_dimensions=5;
    real_t alpha=1.,lambda=0.15;
    provallo::softmax_classifier<real_t,real_t> softmax(n_dimensions,n_classes,alpha,lambda); 
    //create input matrix
    provallo::matrix<real_t> input_mat(n_dimensions*n_classes,n_dimensions); 
    //create output matrix
    provallo::matrix<real_t> out_mat(n_dimensions*n_classes,n_classes);

    //create spike generator
    real_t sigma=1.0,mu=0.1,dt=0.1,t_min=0.01,t_max=1000.0;
    provallo::gaussian_spike_train_generator<real_t> test_spike_train_generator(sigma,mu,dt,t_min,t_max);
    //generate spike train
    std::vector<real_t> train=test_spike_train_generator.generate();
    while(test_spike_train_generator.refine(train)<=0.0)
    {
        std::cout<<"[+] spike not  refined"<<std::endl;
        train=test_spike_train_generator.generate();
    }
    //fill input and output matrices  
    for(size_t i=0;i<n_dimensions*n_classes;i++)
    {
        for(size_t j=0;j<n_dimensions;j++)
        {
            test_spike_train_generator.refine(train);
            input_mat(i,j)=train[i]  ;
          
        }
    } 

    //label each row of the input matrix with the corresponding class 
    for(size_t i=0;i<n_dimensions*n_classes;i++)
    {
        for(size_t j=0;j<n_classes;j++)
        {
            if(i/n_dimensions==j)
            {
                out_mat(i,j)=1.0;
            }
            else
            {
                out_mat(i,j)=0.0;
            }
        }
    }
    provallo::matrix<real_t> oout = out_mat;
    //train softmax classifier
    softmax.train(input_mat,out_mat);
    //test softmax classifier

   if((oout-out_mat).sum()==0.0)
   {
       std::cout<<"[+] softmax classifier test passed"<<std::endl;
   }
   else
   {
        //print the errors: 
       std::cout<<"[-] expected: "<<std::to_string(oout.maxCoeff())<<", got :"<<std::to_string( out_mat.maxCoeff())<<std::endl;
       
   }  

}


// simplified version of fit_fuzzdb for testing purposes:

void train_web_requests_patterns()
{

  const std::string db_path = "./db/web_requests.db";

  io::sqlite::db *database = new io::sqlite::db(db_path.c_str());
  std::ofstream gnuplot_script_file("web_requests.gp" /*, std::ios::binary | std::ios::ate*/),gnuplot_confusion_matrix_file("web_requests.confusion_matrix.gp" /*, std::ios::binary | std::ios::ate*/),web_requests_file("web_requests.DAT" /*, std::ios::binary | std::ios::ate*/),roc_curve_script_file("web_requests.roc.gp" /*, std::ios::binary | std::ios::ate*/); 

  std::ofstream web_requests("web_requests.DAT" /*, std::ios::binary | std::ios::ate*/);  

  std::ofstream roc_curve("web_requests_roc.DAT" /*, std::ios::binary | std::ios::ate*/);  
  std::ofstream confusion_matrix("web_requests.confusion_matrix.DAT" /*, std::ios::binary | std::ios::ate*/); 
  std::ofstream model_data("web_requests.model_data.DAT" /*, std::ios::binary | std::ios::ate*/);
   // load all the fields from the database
  provallo::matrix<size_t> confusion_matrix_mat(5,5);

  // load all the fields from the database
  // create vectorizer
  // Dynamically build dataset from sqlite database of web reqest logs
  // basic headers :
  // virtual host(or service)|remote_addr(IP)|time_local(TS)|method(TXT)|url(TXT)|protocol(TXT)|status(DISCRETE)|body_bytes_sent(real)|referer|user_agent

  std::vector<std::vector<std::string>> results;
  // run query with cursor :

  // create vectorizer
  // Dynamically build dataset from sqlite database of web reqest logs
  // basic headers :
  // virtual host(or service)|remote_addr(IP)|time_local(TS)|method(TXT)|url(TXT)|protocol(TXT)|status(DISCRETE)|body_bytes_sent(real)|referer|user_agent
  //---
  //
  // extended headers :
  //
  //  referer can indicate admin page access requests
  //  user_agent can indicate bot access requests,injection attacks,etc
  //  url can indicate injection attacks over paramerters,etc
  //  method can indicate injection attacks over method,etc
  //  protocol can indicate injection attacks over protocol,etc

  // status is the classifier's output
  // body_bytes_sent is ignored for now
  // time_local is parsed as a timestamp
  // remote_addr is vectorized as real number
  // method is vectorized as a real number
  // url is vectorized as a real number
  // protocol is vectorized as a real number
  // status is vectorized as a real number
  // referer is vectorized as a real number
  // user_agent is vectorized as a real number

  // create dataset:
  // classes / labels are response codes : 200,404,500,etc
  // features are :
  //   remote_addr
  //   method
  //   url
  //   protocol
  //   status
  //   referer
  //   user_agent
  //   time_local
  //   body_bytes_sent
  //   virtual_host
  //   service
  //   time_local

  // additional fields :
  //   longtitude
  //   latitude
  //   country
  //   city
  //   region
  //   postal_code
  //   metro_code
  //   check if method is get,post,put,delete,etc, anything not mapped to a known method is mapped to unknown method discrete value
  //   check if url is has been mapped to a 200 status code before , if yes then it is a known url, if not then it is an unknown url
  //   check if protocol is http 1.0,1.1,2.0,etc, anything not mapped to a known protocol is mapped to unknown protocol discrete value
  //   transform time_local to timestamp
  //   check if referer is a known url, if yes then it is a known referer, if not then it is an unknown referer
  //   check if user_agent is a known user_agent, if yes then it is a known user_agent, if not then it is an unknown user_agent, add index to unknown user_agent
  //   check if virtual_host is a known virtual_host, if yes then it is a known virtual_host, if not then it is an unknown virtual_host, add index to unknown virtual_host
  //   check if service is a known service, if yes then it is a known service, if not then it is an unknown service, add index to unknown service
  // execute query :
  std::string query = "SELECT * FROM logs";
  try
  {
    size_t nrow = 0;
    io::sqlite::stmt u(
        *database,
        query.c_str());
    // u.bind ().text (1, id).int32 (2, participating_candidates).int32 (
    //  3, last_method).int32 (4, last_attribute);
    while (u.step())
    {

      // std::cout<<"[+] query results : "<<std::endl;
      std::vector<std::string> row;
      size_t ncolumns = u.column_count();
      for (size_t i = 0; i < ncolumns; ++i)
      {
        std::string va = u.row().text(i);

        /* std::cout<<"[+]  [ "<<std::to_string(nrow)<<","<<std::to_string(i)
         <<va<<std::endl; */
        row.push_back(va);
      }
      results.push_back(row);

      nrow++;
    }
  }
  catch (io::sqlite::error &e)
  {
    std::cerr << e.what() << "," << e.code();
    io::sqlite::db *pOld = database;
    database = new io::sqlite::db(pOld->file_name());
    delete pOld;
  }
  if (results.size() == 0)
  {
    std::cout << "[+] no results found" << std::endl;
    return;
  }
  // create vectorizer
  // create vector of labels from results.
  // create vectorizer

  provallo::hashed_bag_of_words vectorizer;
  // add labels to vectorizer
  // add documents to vectorizer
  //  size_t docs = results.size(),doc_index=0;
  for (auto &row : results)
  {
    // add url,method,protocol, referer,user_agent to vectorizer
    // std::string document = row[3]+" "+row[4]+" "+row[5]+" "+" "+row[8]+" "+row[9];

    vectorizer.add_document(row[3]); // already fits the document
    vectorizer.add_document(row[4]); // already fits the document
    vectorizer.add_document(row[5]); // already fits the document
    vectorizer.add_document(row[8]); // already fits the document
    vectorizer.add_document(row[9]); // already fits the document

    // doc_index++;
    // std::cout<<"[+] added document : "<<std::to_string(doc_index)<<"/"<<std::to_string(docs)<<std::endl;
  }
  // process documents
  vectorizer.process_documents();

  // calculate the number of input dimensions for the softmax classifier :
  // input dimensions = average of the number of words in each document

  // print vectorizer output
  std::cout << "[+] vectorizer output size = " << std::to_string(vectorizer.get_bag_of_words().size()) << std::endl;
  // create dataset

  auto mixed = results;
  results.clear();

  // predict http status code from url,method,protocol, referer,user_agent
  // create softmax classifier
  size_t n_classes = 5 /*normal,redir,client,server,unknown*/, n_dimensions = 5 /*url,method,protocol,referer,user_agent*/;
  real_t alpha = 1., lambda = 0.15;
  provallo::softmax_classifier<real_t, real_t> softmax(n_dimensions, n_classes, alpha, lambda);
  // train softmax classifier
  std::cout << "[+] training softmax classifier " << std::endl;
  size_t total_cases = 0;
  size_t correct_classifications = 0;
  size_t error_classifications = 0;
  size_t true_positives = 0;
  size_t true_negatives = 0;
  size_t false_positives = 0;
  size_t false_negatives = 0;

  // classes :
  // 100,200 - normal
  // 300 - redirect
  // 400 - client error
  // 500 - server error
  //>600 - unknown error

  // first train with normal data  , select all rows with status code 100,200

  // execute query :
  std::string query_normal = "SELECT * FROM logs order by status asc ";
  try
  {
    io::sqlite::stmt u(
        *database, query_normal.c_str());

    while (u.step())
    {
      std::vector<std::string> row;
      size_t ncolumns = u.column_count();
      for (size_t i = 0; i < ncolumns; ++i)
      {
        row.push_back(u.row().text(i));
      }
      results.push_back(row);
    }
  }
  catch (io::sqlite::error &e)
  {
    std::cerr << e.what() << "," << e.code();
    io::sqlite::db *pOld = database;
    database = new io::sqlite::db(pOld->file_name());
    delete pOld;
  }

  // first train with  vectorized the status codes

  // add documents to vectorizer
  // create input from the sum of the outputs of each vectorizer with the normal data.
  size_t label_index = 0; // label index
  provallo::matrix<real_t> out_mat(1, n_classes);
  provallo::matrix<real_t> input_mat(1, n_dimensions);

  for (auto &row : results)
  {

    // make sure we have a label for this row

    // add url,method,protocol, referer,user_agent to vectorizer
    //  std::string document = row[3]+" "+row[4]+" "+row[5]+" "+" "+row[8]+" "+row[9];

    // refactor , setup input matrix of 5 dimensions

    std::vector<real_t> input;
    for (size_t i = 0; i < row.size(); i++)
    {
      if (i == 3 || i == 4 || i == 5 || i == 8 || i == 9)
      {
        // vectorize
        std::vector<real_t> vectorized = vectorizer.predict(row[i]);
        // sum
        if (vectorized.size() == 0)
        {
          input.push_back(0.0);
        }
        else
        {
          real_t sum = std::accumulate(vectorized.begin(), vectorized.end(), 0.0);
          // DEBUG:
          // std::cout<<"[+] vectorizer added sum = "<<std::to_string(sum)<< ",value = "<<std::to_string(sum/vectorized.size())<<std::endl;
          web_requests << std::to_string(sum) << " " << std::to_string(sum / vectorized.size()) << std::endl; 

          input.push_back(sum / vectorized.size());
        }
      }
    }
    // = vectorizer.predict(document);
    input_mat.resize(1, input.size());
    // copy input to input_mat
    for (size_t i = 0; i < input.size(); i++)
    {
      input_mat(0, i) = input[i];
    }

    std::cout << "[+] training softmax classifier" << std::endl;
    // print vectorize output
    std::cout << "[+] vectorizer  output size = " << std::to_string( input.size()) << std::endl;
    std::cout << "[+] softmax input size = " << std::to_string(softmax.getInputDim()) << std::endl;

    // out_mat.fill(0.0); no need
    // fill output with label from index

    size_t status_code = row[6].length() > 1 ? std::stoi(row[6]) : 0;

    if (status_code > 100 && status_code < 210)
    {
      out_mat(0, 0) = 1.0; // normal
    }
    else if (status_code >= 300 && status_code < 400)
    {
      out_mat(0, 1) = 1.0; // redirect
    }
    else if (status_code >= 400 && status_code < 500)
    {
      out_mat(0, 2) = 1.0; // client error
    }
    else if (status_code >= 500)
    {
      out_mat(0, 3) = 1.0; // server error
    }
    else if (status_code > 600)
    {
      out_mat(0, 4) = 1.0; // unknown error
    }
    else
    {
      std::cout << "[+] error : unknown status code : " << std::to_string(status_code) << std::endl;
    }
    // train single input
    softmax.train(input_mat.data(), out_mat.data());
    // check if softmax output is correct
    // update confusion matrix
    total_cases++;
    size_t max_index = 0;
    real_t max_value = 0.0;
    for (size_t i = 0; i < out_mat.cols(); i++)
    {
      for (size_t j = 0; j < out_mat.rows(); ++j)
      {
        if (out_mat(j, i) > max_value)
        {
          max_value = out_mat(j, i);
          max_index = i;
        }
      }
    }
    // translate label to softmax output index

    size_t label_target = status_code; // labels[label_index].length()?std::stoi(labels[label_index].c_str()):0;
    // translate status code to softmax output index
    if (label_target > 100 && label_target < 210)
    {
      label_target = 0;
    }
    else if (label_target >= 300 && label_target < 400)
    {
      label_target = 1;
    }
    else if (label_target >= 400 && label_target < 500)
    {
      label_target = 2;
    }
    else if (label_target >= 500)
    {
      label_target = 3;
    }
    else if (label_target > 600)
    {
      label_target = 4;
    }
    else
    {
      std::cout << "[+] error : unknown status code : " << std::to_string(label_target) << std::endl;
    }
    confusion_matrix_mat(label_target,max_index)++; 
    // check if softmax output is correct
    if (max_index == label_target)
    {
      correct_classifications++;
      
        true_positives++;
      
    }
    else
    {
      error_classifications++;
      if (label_target == 0)
      {
        false_positives++;
      }
      else if (max_index == 0)
      {
        false_negatives++;
      }
      else{
        true_negatives++;
      }
    }
     label_index++;

  } // end for normal rows
  // now train with abnormal data  , select all rows with status code 300,400,500,600,700,800,900,1000,etc
  // execute query :
  // print training results :
  std::cout << "[+] softmax dynamic training with lda vectorizer : " << std::endl;
  std::cout << "[+] total training samples:" << std::to_string(total_cases) << std::endl;
  std::cout << "[+] false positives : " << std::to_string(false_positives) << std::endl;
  std::cout << "[+] false negatives : " << std::to_string(false_negatives) << std::endl;
  std::cout << "[+] true positives : " << std::to_string(true_positives) << std::endl;
  std::cout << "[+] true negatives : " << std::to_string(true_negatives) << std::endl;
  std::cout << "[+] correct classifications : " << std::to_string(correct_classifications) << std::endl;
  std::cout << "[+] error classifications : " << std::to_string(error_classifications) << std::endl;
  //accuracy
  real_t accuracy = (real_t)correct_classifications / (real_t)total_cases;
  std::cout << "[+] accuracy : " << std::to_string(accuracy) << std::endl;
  // precision
  real_t precision = (real_t)true_positives / (real_t)(true_positives + false_positives);
  std::cout << "[+] precision : " << std::to_string(precision) << std::endl;
  // recall
  real_t recall = (real_t)true_positives / (real_t)(true_positives + false_negatives);
  std::cout << "[+] recall : " << std::to_string(recall) << std::endl;
  // f1 score
  real_t f1_score = 2.0 * (precision * recall) / (precision + recall);
  std::cout << "[+] f1 score : " << std::to_string(f1_score) << std::endl;
  // save softmax classifier
  std::cout << "[+] softmax training done" << std::endl; 


  //print confusion matrix 
  std::cout<<"[+] confusion matrix : "<<std::endl;
  std::cout <<"--------------------------------------------------------------------------"<<std::endl; 
  std::cout <<"\t100-200  300-400  400-500  500-600  OTHER "<<std::endl;
  std::cout <<"\t100-200  "<<confusion_matrix_mat(0,0)<<"  "<<confusion_matrix_mat(0,1)<<"  "<<confusion_matrix_mat(0,2)<<"  "<<confusion_matrix_mat(0,3)<<"  "<<confusion_matrix_mat(0,4)<<std::endl; 
  std::cout <<"\t300-400  "<<confusion_matrix_mat(1,0)<<"  "<<confusion_matrix_mat(1,1)<<"  "<<confusion_matrix_mat(1,2)<<"  "<<confusion_matrix_mat(1,3)<<"  "<<confusion_matrix_mat(1,4)<<std::endl;
  std::cout <<"\t400-500  "<<confusion_matrix_mat(2,0)<<"  "<<confusion_matrix_mat(2,1)<<"  "<<confusion_matrix_mat(2,2)<<"  "<<confusion_matrix_mat(2,3)<<"  "<<confusion_matrix_mat(2,4)<<std::endl;
  std::cout <<"\t500-600  "<<confusion_matrix_mat(3,0)<<"  "<<confusion_matrix_mat(3,1)<<"  "<<confusion_matrix_mat(3,2)<<"  "<<confusion_matrix_mat(3,3)<<"  "<<confusion_matrix_mat(3,4)<<std::endl;
  std::cout <<"\tOTHER    "<<confusion_matrix_mat(4,0)<<"  "<<confusion_matrix_mat(4,1)<<"  "<<confusion_matrix_mat(4,2)<<"  "<<confusion_matrix_mat(4,3)<<"  "<<confusion_matrix_mat(4,4)<<std::endl;
  std::cout <<"--------------------------------------------------------------------------"<<std::endl; 



  // save softmax classifier
  std::cout << "[+] testing softmax classifier  " << std::endl;
  // reset for test
  total_cases = 0;
  correct_classifications = 0;
  error_classifications = 0;
  true_positives = 0;
  true_negatives = 0;
  false_positives = 0;
  false_negatives = 0;
  confusion_matrix_mat.fill(0);

  // test again with same data, different order on SELECT query
  // execute query :
  // clear results :
  results.clear();
  // execute query :

  std::string query_normal2 = "SELECT * FROM logs order by status desc ";
  try
  {
    io::sqlite::stmt u(
        *database, query_normal2.c_str());
    u.exec();
    while (u.step())
    {
      std::vector<std::string> row;
      size_t ncolumns = u.column_count();
      for (size_t i = 0; i < ncolumns; ++i)
      {
        row.push_back(u.row().text(i));
      }
      results.push_back(row);
    }
  }
  catch (io::sqlite::error &e)
  {
    std::cerr << e.what() << "," << e.code();
    io::sqlite::db *pOld = database;
    database = new io::sqlite::db(pOld->file_name());
    delete pOld;
  }
  // test the data from results
  // resize matrices :
  // create input from the sum of the outputs of each vectorizer with the normal data.
  label_index = 0; // label index

  for (auto &row : results)
  {
    // add url,method,protocol, referer,user_agent to vectorizer
    // refactor: setup input matrix of 5 dimensions
    // std::string document = row[3]+" "+row[4]+" "+row[5]+" "+" "+row[8]+" "+row[9];
    std::vector<real_t> input;
    for (size_t i = 0; i < row.size(); i++)
    {
      if (i == 3 || i == 4 || i == 5 || i == 8 || i == 9)
      {
        // vectorize
        std::vector<real_t> vectorized = vectorizer.predict(row[i]);
        // sum
        if (vectorized.size() == 0)
        {
          input.push_back(0.0);
        }
        else
        {
          real_t sum = std::accumulate(vectorized.begin(), vectorized.end(), 0.0);
          // DEBUG:
          //std::cout << "[+] vectorizer added sum = " << std::to_string(sum) << ",value = " << std::to_string(sum / vectorized.size()) << std::endl;
          web_requests << std::to_string(sum) << " " << std::to_string(sum / vectorized.size()) << std::endl; 
          
          input.push_back(sum / vectorized.size());
        }
      }
    }
    // std::vector<real_t> input = vectorizer.predict(document);
    input_mat.resize(1, input.size());
    // copy input to input_mat
    for (size_t i = 0; i < input.size(); i++)
    {
      input_mat(0, i) = input[i];
    }

    std::cout << "[+] testing softmax classifier" << std::endl;
    // print vectorize output
    //std::cout << "[+] vectorizer  output size = " << std::to_string(input.size()) << std::endl;
    //std::cout << "[+] softmax input size = " << std::to_string(softmax.getInputDim()) << std::endl;

    // out_mat.fill(0.0); no need
    // fill output with label from index

    size_t status_code = std::stoi(row[6]);
    if (status_code > 100 && status_code < 210)
    {
      out_mat(0, 0) = 1.0;
    }
    else if (status_code >= 300 && status_code < 400)
    {
      out_mat(0, 1) = 1.0;
    }
    else if (status_code >= 400 && status_code < 500)
    {
      out_mat(0, 2) = 1.0;
    }
    else if (status_code >= 500)
    {
      out_mat(0, 3) = 1.0;
    }
    else if (status_code > 600)
    {
      out_mat(0, 4) = 1.0;
    }
    else
    {
      std::cout << "[+] error : unknown status code : " << std::to_string(status_code) << std::endl;
    }
    softmax.test(input_mat, out_mat);
    // check if softmax output is correct
    // update confusion matrix
    total_cases++;
    size_t max_index = 0;
    real_t max_value = 0.0;
    for (size_t i = 0; i < out_mat.cols(); i++)
    {
      for (size_t j = 0; j < out_mat.rows(); ++j)
      {
        if (out_mat(j, i) > max_value)
        {
          max_value = out_mat(j, i);
          max_index = i;
        }
      }
    }
    size_t target_label = status_code; // std::atoi(labels[label_index].c_str());

    // translate target label to softmax label
    if (target_label > 100 && target_label < 210)
    {
      target_label = 0;
    }
    else if (target_label >= 300 && target_label < 400)
    {
      target_label = 1;
    }
    else if (target_label >= 400 && target_label < 500)
    {
      target_label = 2;
    }
    else if (target_label >= 500)
    {
      target_label = 3;
    }
    else if (target_label > 600)
    {
      target_label = 4;
    }
    else
    {
      std::cout << "[+] error : unknown status code : " << std::to_string(target_label) << std::endl;
    }
    confusion_matrix_mat(target_label,max_index)++;
    if (max_index == target_label)
    {
      correct_classifications++;
      
        true_positives++;
       
    }
    else
    {
      error_classifications++;
      if (target_label == 0)
      {
        false_positives++;
      }
      else if (max_index == 0)
      {
        false_negatives++;
      }
      else
      {
        true_negatives++;
      }
    }
    input_mat.clear();
    out_mat.clear();
    label_index++;
    //update roc_curve file 
    if(roc_curve.is_open() && roc_curve.good())
      roc_curve<<std::to_string(max_index)<<" "<<std::to_string(target_label)<<std::endl; 
    // vectorize and add to testing matrix with labels
  }
  // print testing results :
  std::cout << "[+] softmax dynamic testing with lda vectorizer : " << std::endl;
  std::cout << "[+] total testing samples:" << std::to_string(total_cases) << std::endl;
  std::cout << "[+] false positives : " << std::to_string(false_positives) << std::endl;
  std::cout << "[+] false negatives : " << std::to_string(false_negatives) << std::endl;
  std::cout << "[+] true positives : " << std::to_string(true_positives) << std::endl;
  std::cout << "[+] true negatives : " << std::to_string(true_negatives) << std::endl;
  std::cout << "[+] correct classifications : " << std::to_string(correct_classifications) << std::endl;
  std::cout << "[+] error classifications : " << std::to_string(error_classifications) << std::endl;
  std::cout << "[+] softmax testing done" << std::endl;

  //accuracy
  accuracy = (real_t)correct_classifications / (real_t)total_cases;
  std::cout << "[+] accuracy : " << std::to_string(accuracy) << std::endl;
  // precision
  precision = (real_t)true_positives / (real_t)(true_positives + false_positives);
  std::cout << "[+] precision : " << std::to_string(precision) << std::endl;
  // recall
  recall = (real_t)true_positives / (real_t)(true_positives + false_negatives);
  std::cout << "[+] recall : " << std::to_string(recall) << std::endl;
  // f1 score
  f1_score = 2.0 * (precision * recall) / (precision + recall);
  std::cout << "[+] f1 score : " << std::to_string(f1_score) << std::endl;
  
  //print confusion matrix
  std::cout<<"[+] confusion matrix : "<<std::endl;
  std::cout <<"--------------------------------------------------------------------------"<<std::endl;
  std::cout <<"\t100-200  300-400  400-500  500-600  OTHER "<<std::endl;
  std::cout <<"\t100-200  "<<confusion_matrix_mat(0,0)<<"  "<<confusion_matrix_mat(0,1)<<"  "<<confusion_matrix_mat(0,2)<<"  "<<confusion_matrix_mat(0,3)<<"  "<<confusion_matrix_mat(0,4)<<std::endl;
  std::cout <<"\t300-400  "<<confusion_matrix_mat(1,0)<<"  "<<confusion_matrix_mat(1,1)<<"  "<<confusion_matrix_mat(1,2)<<"  "<<confusion_matrix_mat(1,3)<<"  "<<confusion_matrix_mat(1,4)<<std::endl;
  std::cout <<"\t400-500  "<<confusion_matrix_mat(2,0)<<"  "<<confusion_matrix_mat(2,1)<<"  "<<confusion_matrix_mat(2,2)<<"  "<<confusion_matrix_mat(2,3)<<"  "<<confusion_matrix_mat(2,4)<<std::endl;
  std::cout <<"\t500-600  "<<confusion_matrix_mat(3,0)<<"  "<<confusion_matrix_mat(3,1)<<"  "<<confusion_matrix_mat(3,2)<<"  "<<confusion_matrix_mat(3,3)<<"  "<<confusion_matrix_mat(3,4)<<std::endl;
  std::cout <<"\tOTHER    "<<confusion_matrix_mat(4,0)<<"  "<<confusion_matrix_mat(4,1)<<"  "<<confusion_matrix_mat(4,2)<<"  "<<confusion_matrix_mat(4,3)<<"  "<<confusion_matrix_mat(4,4)<<std::endl;
  std::cout <<"--------------------------------------------------------------------------"<<std::endl;
  

  // save softmax classifier
  // save softmax classifier --> softmax_softmax_fuzzdb_test.json
  softmax.save(std::string("softmax_web_requests_") + std::to_string(n_classes) + std::string("_test.json"));
  std::cout << "[+] saving softmax classifier done" << std::endl;
  // save vectorizer
  // save vectorizer --> vectorizers_fuzzdb_test.json
  std::cout << "[+] saving vectorizer" << std::endl;
  std::ofstream vectorizer_file(std::string("vectorizer_web_requests_") + std::to_string(n_classes) + std::string("_test.json"), std::ios::binary | std::ios::ate);
  if (vectorizer_file.is_open() && vectorizer_file.good())
  {

    vectorizer.save(vectorizer_file);
    vectorizer_file.close();
    std::cout << "[+] vectorizer saved" << std::endl;
  }
  else
  {
    std::cout << "[+] error opening vectorizer file" << std::endl;
  }

  //save roc curve
  roc_curve.close();
  //save roc curve script
  if(roc_curve_script_file.is_open() && roc_curve_script_file.good())
  {
    std::string roc_curve_script = "set terminal png\n"
                                   "set output \"web_requests_roc_curve.png\"\n"
                                   "set title \"web requests roc curve\"\n"
                                   "set xlabel \"Predicted class\"\n"
                                   "set ylabel \"Target Class\"\n"
                                   "set xrange [0:5]\n"
                                   "set yrange [0:5]\n"
                                   "set grid\n"
                                   "set key off\n"
                                   "set ticslevel 0\n"
                                   "set tics out nomirror\n"
                                   "set tics scale 0.5\n"
                                   "set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 7 ps 1.5\n"
                                   "set style line 2 lc rgb '#dd181f' lt 1 lw 2 pt 5 ps 1.5\n"
                                   "set style line 3 lc rgb '#dd181f' lt 1 lw 2 pt 9 ps 1.5\n"
                                   "set style line 4 lc rgb '#dd181f' lt 1 lw 2 pt 13 ps 1.5\n"
                                   "set style line 5 lc rgb '#dd181f' lt 1 lw 2 pt 15 ps 1.5\n"
                                   "set style line 6 lc rgb '#dd181f' lt 1 lw 2 pt 17 ps 1.5\n"
                                   "set style line 7 lc rgb '#dd181f' lt 1 lw 2 pt 19 ps 1.5\n"
                                   "set style line 8 lc rgb '#dd181f' lt 1 lw 2 pt 21 ps 1.5\n"
                                   "set style line 9 lc rgb '#dd181f' lt 1 lw 2 pt 23 ps 1.5\n"
                                   "set style line 10 lc rgb '#dd181f' lt 1 lw 2 pt 25 ps 1.5\n"
                                   "set style line 11 lc rgb '#dd181f' lt 1 lw 2 pt 27 ps 1.5\n"
                                   "set style line 12 lc rgb '#dd181f' lt 1 lw 2 pt 29 ps 1.5\n"
                                    "set style line 13 lc rgb '#dd181f' lt 1 lw 2 pt 31 ps 1.5\n"
                                    "set style line 14 lc rgb '#dd181f' lt 1 lw 2 pt 33 ps 1.5\n"
                                    // plot:
                                    //   Warning: Single isoline (scan) is not enough for a pm3d plot.
                                    //   Hint: Missing blank lines in the data file? See 'help pm3d' and FAQ.
                                    //   Hint: If your data is all-zeros, you may need to use the 'set dgrid3d' command.
                                    //plot predicted class vs target class
                                    "plot \"web_requests_roc_curve.DAT\" using 1:2 with linespoints ls 1\n"; 
      // save roc curve script        
      
        roc_curve_script_file<<roc_curve_script; 
        roc_curve_script_file.close(); 
        std::cout<<"[+] roc curve script saved"<<std::endl; 
    
  }
  // save gnuplot script
  if (gnuplot_script_file.is_open() && gnuplot_script_file.good())
  {

    std::string gnuplot_script = " set terminal gif animate delay 100\n"
                                  "set output \"web_requests.gif\"\n" 
                                  "set title \"web requests\"\n"
                                  "set xlabel \"Predicted class\"\n"
                                  "set ylabel \"Target Class\"\n"
                                  "set hidden3d\n"
                                  "set xrange [0:5]\n"
                                  "set yrange [0:5]\n"
                                  "set grid\n"
                                  "set multiplot\n"
                                  "set key off\n"
                                  "set ticslevel 0\n"
                                  "set tics out nomirror\n"
                                  "set tics scale 0.5\n"
                                  "set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 7 ps 1.5\n"
                                  "set style line 2 lc rgb '#dd181f' lt 1 lw 2 pt 5 ps 1.5\n"
                                  "set style line 3 lc rgb '#dd181f' lt 1 lw 2 pt 9 ps 1.5\n"
                                  "set style line 4 lc rgb '#dd181f' lt 1 lw 2 pt 13 ps 1.5\n"
                                  "set style line 5 lc rgb '#dd181f' lt 1 lw 2 pt 15 ps 1.5\n"

                                  "set style line 6 lc rgb '#dd181f' lt 1 lw 2 pt 17 ps 1.5\n"
                                  "set style line 7 lc rgb '#dd181f' lt 1 lw 2 pt 19 ps 1.5\n"
                                  "set style line 8 lc rgb '#dd181f' lt 1 lw 2 pt 21 ps 1.5\n"
                                  "set style line 9 lc rgb '#dd181f' lt 1 lw 2 pt 23 ps 1.5\n"
                                  "set style line 10 lc rgb '#dd181f' lt 1 lw 2 pt 25 ps 1.5\n"
                                  "do for [i=0:100] {\n"
                                  "set view i,30,1,1\n"
                                  "plot \"web_requests.DAT\" every ::::i using 1:2 with linespoints ls 1\n"
                                 
                                  
                                  "pause 0.1\n"
                                  "}\n"

                                  "unset multiplot\n";
                                  

    // plot:
    //"plot \"spike_train.DAT\" using 1:2 with lines\n"

    // save gnuplot script
    if (gnuplot_script_file.is_open() && gnuplot_script_file.good())
    {
      gnuplot_script_file << gnuplot_script;
      gnuplot_script_file.close();
      std::cout << "[+] web requests script saved" << std::endl;
    }
    else
    {
      std::cout << "[+] error opening web requests  script file" << std::endl;  
    }
  }
  else
  {
    std::cout << "[+] error opening web requests  script file" << std::endl;
  }
  // save roc curve
  
  // save confusion matrix
  if (confusion_matrix.is_open() && confusion_matrix.good())
  {
        for ( size_t i = 0; i < confusion_matrix_mat.rows(); i++)
        {
          for ( size_t j = 0; j < confusion_matrix_mat.cols(); j++)
          {
            confusion_matrix << confusion_matrix_mat(i,j)<<" ";
          }
          confusion_matrix << std::endl;
        }
      confusion_matrix.close();
  }
  else
  {
    std::cout << "[+] error opening confusion matrix file" << std::endl;
  } 

  // save model data
  if (model_data.is_open() && model_data.good())
  {
    //save the weights and biases of the softmax classifier 
     model_data.close();
  }
  else
  {
    std::cout << "[+] error opening model data file" << std::endl;
  } 

  //save confusion matrix script 
  std::ofstream confusion_matrix_script("web_requests_confusion_matrix.gp", std::ios::binary | std::ios::ate);  
  if (confusion_matrix_script.is_open() && confusion_matrix_script.good())
  {
    std::string confusion_matrix_script_string = "set terminal png\n"
                                                 "set output \"web_requests_confusion_matrix.png\"\n"
                                                 "set title \"web requests confusion matrix\"\n"
                                                 "set xlabel \"Predicted class\"\n"
                                                 "set ylabel \"Target Class\"\n"
                                                 "set xrange [0:5]\n"
                                                 "set yrange [0:5]\n"
                                                 "set grid\n"
                                                 "set key off\n"
                                                 "set ticslevel 0\n"
                                                 "set tics out nomirror\n"
                                                 "set tics scale 0.5\n"
                                                 "set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 7 ps 1.5\n"
                                                 "set style line 2 lc rgb '#dd181f' lt 1 lw 2 pt 5 ps 1.5\n"
                                                 "set style line 3 lc rgb '#dd181f' lt 1 lw 2 pt 9 ps 1.5\n"
                                                 "set style line 4 lc rgb '#dd181f' lt 1 lw 2 pt 13 ps 1.5\n"
                                                 "set style line 5 lc rgb '#dd181f' lt 1 lw 2 pt 15 ps 1.5\n"
                                                 "set style line 6 lc rgb '#dd181f' lt 1 lw 2 pt 17 ps 1.5\n"
                                                 "set style line 7 lc rgb '#dd181f' lt 1 lw 2 pt 19 ps 1.5\n"
                                                 "set style line 8 lc rgb '#dd181f' lt 1 lw 2 pt 21 ps 1.5\n"
                                                 "set style line 9 lc rgb '#dd181f' lt 1 lw 2 pt 23 ps 1.5\n"
                                                 "set style line 10 lc rgb '#dd181f' lt 1 lw 2 pt 25 ps 1.5\n"
                                                 "set style line 11 lc rgb '#dd181f' lt 1 lw 2 pt 27 ps 1.5\n"
                                                 "set style line 12 lc rgb '#dd181f' lt 1 lw 2 pt  29 ps 1.5\n" 
                                                 //set xtics and ytics
                                                  "set xtics (\"100-200\" 0, \"300-400\" 1, \"400-500\" 2, \"500-600\" 3, \"OTHER\" 4)\n" 
                                                  "set ytics (\"100-200\" 0, \"300-400\" 1, \"400-500\" 2, \"500-600\" 3, \"OTHER\" 4)\n"
                                                  //plot matrix with image 
                                                  "plot \"web_requests_confusion_matrix.DAT\" matrix with image\n";
    confusion_matrix_script << confusion_matrix_script_string;
    confusion_matrix_script.close();
  }

  std::cout << "[+] saving vectorizer done" << std::endl;
  // delete vectorizer
  delete database;
  return;
  // return true;

 } // end train_web_requests_patterns
 void test_web_requests_patterns()
 {
   //load the pretrained vectorizer and softmax classifier 
    // load vectorizer
  
  //database:
  io::sqlite::db *database = new io::sqlite::db("./db/provallo_web.db");
  std::cout << "[+] loading vectorizer" << std::endl;

  std::ifstream vectorizer_file(std::string("vectorizer_web_requests_") + std::to_string(5) + std::string("_test.json"), std::ios::binary | std::ios::ate);
  provallo::hashed_bag_of_words vectorizer; 
  if (vectorizer_file.is_open() && vectorizer_file.good())
  {
    vectorizer.load(vectorizer_file);
    vectorizer_file.close();
    std::cout << "[+] vectorizer loaded" << std::endl;
  }
  else
  {
    std::cout << "[-] error opening vectorizer file" << std::endl;
    return;
  } 
  // load softmax classifier
  std::cout << "[+] loading softmax classifier" << std::endl;
  provallo::softmax_classifier<real_t, real_t> softmax(5, 5, 1.0, 0.15);  
  softmax.load(std::string("softmax_web_requests_") + std::to_string(5) + std::string("_test.json"));
  std::cout << "[+] softmax classifier loaded" << std::endl;
  // load test data
  std::cout << "[+] loading test data" << std::endl;
  std::vector<std::vector<std::string>> results;
  std::string query_normal = "SELECT * FROM logs order by status desc ";
  try
  {
    io::sqlite::stmt u(
        *database, query_normal.c_str());
    u.exec();
    while (u.step())
    {
      std::vector<std::string> row;
      size_t ncolumns = u.column_count();
      for (size_t i = 0; i < ncolumns; ++i)
      {
        row.push_back(u.row().text(i));
      }
      results.push_back(row);
    }
  }
  catch (io::sqlite::error &e)
  {
    std::cerr << e.what() << "," << e.code();
    io::sqlite::db *pOld = database;
    database = new io::sqlite::db(pOld->file_name());
    delete pOld;
  } 

  // test the data from results
  // resize matrices :
  // create input from the sum of the outputs of each vectorizer with the normal data.
  size_t label_index = 0; // label index
  size_t total_cases = 0;
  size_t correct_classifications = 0;
  size_t error_classifications = 0;
  size_t true_positives = 0;
  size_t true_negatives = 0;
  size_t false_positives = 0;
  size_t false_negatives = 0;
  provallo::matrix<real_t> out_mat(1, 5);
  provallo::matrix<real_t> input_mat(1, 5);
  provallo::matrix<real_t> confusion_matrix_mat(5,5);
  confusion_matrix_mat.fill(0);
  std::ofstream roc_curve("provallo_web_requests_roc_curve.DAT", std::ios::binary | std::ios::ate);
  
  std::ofstream gnuplot_script_file("provallo_web_requests.gnuplot", std::ios::binary | std::ios::ate);
  std::ofstream confusion_matrix_script( "provallo_web_requests_test_confusion_matrix.gnuplot", std::ios::binary | std::ios::ate );
  std::ofstream roc_curve_script_file("provallo_web_requests_roc_curve.gnuplot", std::ios::binary | std::ios::ate);
  std::ofstream confusion_matrix("provallo_web_requests_confusion_matrix.DAT", std::ios::binary | std::ios::ate);
  std::ofstream model_data("provallo_web_requests_model_data.DAT", std::ios::binary | std::ios::ate);
  //validate vectorizer:
  std::cout<<"[+] validating vectorizer : "<<std::endl;
  if ( vectorizer.get_bag_of_words().size() > 2)
  {
    std::cout<<"[+] vectorizer validated"<<std::endl;
  }
  else
  {
    std::cout<<"[-] error validating vectorizer"<<std::endl;
    return;
  }
  //validate softmax classifier:
  std::cout<<"[+] validating softmax classifier : "<<std::endl;
  if ( softmax.getInputDim() > 2)
  {
    std::cout<<"[+] softmax classifier validated"<<std::endl;
  }
  else
  {
    std::cout<<"[-] error validating softmax classifier"<<std::endl;
    return;
  } 

  for (auto &row : results)
  {
    // add url,method,protocol, referer,user_agent to vectorizer
    // refactor: setup input matrix of 5 dimensions
    // std::string document = row[3]+" "+row[4]+" "+row[5]+" "+" "+row[8]+" "+row[9];
    std::vector<real_t> input;
    for (size_t i = 0; i < row.size(); i++)
    {
      if (i == 3 || i == 4 || i == 5 || i == 8 || i == 9)
      {
        // vectorize
        std::vector<real_t> vectorized = vectorizer.predict(row[i]);
        // sum
        if (vectorized.size() == 0)
        {
          input.push_back(0.0);
        }
        else
        {
          real_t sum = std::accumulate(vectorized.begin(), vectorized.end(), 0.0);
          // DEBUG:
          //std::cout << "[+] vectorizer added sum = " << std::to_string(sum) << ",value = " << std::to_string(sum / vectorized.size()) << std::endl;
          //roc_curve << std::to_string(sum) << " " << std::to_string(sum / vectorized.size()) << std::endl; 
          
          input.push_back(sum / real_t(vectorized.size()));
        }
      }
    }
    // std::vector<real_t> input = vectorizer.predict(document);
    input_mat.resize(1, input.size());
    // copy input to input_mat
    for (size_t i = 0; i < input.size(); i++)
    {
      input_mat(0, i) = input[i];
    }

    std::cout << "[+] testing softmax classifier" << std::endl;
    // print vectorize output
    //std::cout << "[+] vectorizer  output size = " << std::to_string(input.size()) << std::endl;
    //std::cout << "[+] softmax input size = " << std::to_string(softmax.getInputDim()) << std::endl;

    // out_mat.fill(0.0); no need
    // fill output with label from index

    size_t status_code = std::stoi(row[6]);
    size_t status_label =0;
    if (status_code > 100 && status_code < 210)
    {
      out_mat(0, 0) = 1.0;
    }
    else if (status_code >= 300 && status_code < 400)
    {
      out_mat(0, 1) = 1.0;
      status_label = 1;
    }
    else if (status_code >= 400 && status_code < 500)
    {
      out_mat(0, 2) = 1.0;
      status_label = 2;
    }
    else if (status_code >= 500)
    {
      out_mat(0, 3) = 1.0;
      status_label = 3;
    }
    else if (status_code > 600)
    {

      out_mat(0, 4) = 1.0;
      status_label = 4;
    }
    else
    {
      std::cout << "[+] error : unknown status code : " << std::to_string(status_code) << std::endl;
      status_label = 4;
    }
    softmax.test(input_mat, out_mat);
    // check if softmax output is correct
    // update confusion matrix
    total_cases++;
    size_t max_index = 0;
    real_t max_value = 0.0;
    for (size_t i = 0; i < out_mat.cols(); i++)
    {
      for (size_t j = 0; j < out_mat.rows(); ++j)
      {
        if (out_mat(j, i) > max_value)
        {
          max_value = out_mat(j, i);
          max_index = i;
        }
      }
    } 
     // status_label is the target label softmax label,max_index is the predicted label 
    confusion_matrix_mat(status_label,max_index)++;
    if (max_index == status_label)
    {
      correct_classifications++;
      true_positives++;
       
    }
    else
    {
      error_classifications++;
      if (status_label == 0)
      {
        false_positives++;
      }
      else if (max_index == 0)
      {
        false_negatives++;
      }
      else
      {
        true_negatives++;
      }
    }   
    input_mat.clear();
    out_mat.clear();
    label_index++;
    //update roc_curve file
    if(roc_curve.is_open() && roc_curve.good())
      roc_curve<<std::to_string(status_label)<<" "<<std::to_string(max_index)<<std::endl;  
  }
  // print testing results :
  std::cout << "[+] softmax dynamic testing with lda vectorizer : " << std::endl;
  std::cout << "[+] total testing samples:" << std::to_string(total_cases) << std::endl;
  std::cout << "[+] false positives : " << std::to_string(false_positives) << std::endl;
  std::cout << "[+] false negatives : " << std::to_string(false_negatives) << std::endl;
  std::cout << "[+] true positives : " << std::to_string(true_positives) << std::endl;
  std::cout << "[+] true negatives : " << std::to_string(true_negatives) << std::endl;
  std::cout << "[+] correct classifications : " << std::to_string(correct_classifications) << std::endl;
  std::cout << "[+] error classifications : " << std::to_string(error_classifications) << std::endl;
  std::cout << "[+] softmax testing done" << std::endl;

  //accuracy
  real_t accuracy = (real_t)correct_classifications / (real_t)total_cases;
  std::cout << "[+] accuracy : " << std::to_string(accuracy) << std::endl;
  // precision
  real_t precision = (real_t)true_positives / (real_t)(true_positives + false_positives);
  std::cout << "[+] precision : " << std::to_string(precision) << std::endl;
  // recall
  real_t recall = (real_t)true_positives / (real_t)(true_positives + false_negatives);
  std::cout << "[+] recall : " << std::to_string(recall) << std::endl;
  // f1 score
  real_t f1_score = 2.0 * (precision * recall) / (precision + recall);

  std::cout << "[+] f1 score : " << std::to_string(f1_score) << std::endl;
  //print confusion matrix
  std::cout<<"[+] confusion matrix : "<<std::endl;
  std::cout <<"--------------------------------------------------------------------------"<<std::endl;
  std::cout <<"\t100-200  300-400  400-500  500-600  OTHER "<<std::endl;
  std::cout <<"\t100-200  "<<confusion_matrix_mat(0,0)<<"  "<<confusion_matrix_mat(0,1)<<"  "<<confusion_matrix_mat(0,2)<<"  "<<confusion_matrix_mat(0,3)<<"  "<<confusion_matrix_mat(0,4)<<std::endl;
  std::cout <<"\t300-400  "<<confusion_matrix_mat(1,0)<<"  "<<confusion_matrix_mat(1,1)<<"  "<<confusion_matrix_mat(1,2)<<"  "<<confusion_matrix_mat(1,3)<<"  "<<confusion_matrix_mat(1,4)<<std::endl;
  std::cout <<"\t400-500  "<<confusion_matrix_mat(2,0)<<"  "<<confusion_matrix_mat(2,1)<<"  "<<confusion_matrix_mat(2,2)<<"  "<<confusion_matrix_mat(2,3)<<"  "<<confusion_matrix_mat(2,4)<<std::endl;
  std::cout <<"\t500-600  "<<confusion_matrix_mat(3,0)<<"  "<<confusion_matrix_mat(3,1)<<"  "<<confusion_matrix_mat(3,2)<<"  "<<confusion_matrix_mat(3,3)<<"  "<<confusion_matrix_mat(3,4)<<std::endl;
  std::cout <<"\tOTHER    "<<confusion_matrix_mat(4,0)<<"  "<<confusion_matrix_mat(4,1)<<"  "<<confusion_matrix_mat(4,2)<<"  "<<confusion_matrix_mat(4,3)<<"  "<<confusion_matrix_mat(4,4)<<std::endl;
  std::cout <<"--------------------------------------------------------------------------"<<std::endl;
  //save confusion matrix
  if (confusion_matrix.is_open() && confusion_matrix.good())
  {
        for ( size_t i = 0; i < confusion_matrix_mat.rows(); i++)
        {
          for ( size_t j = 0; j < confusion_matrix_mat.cols(); j++)
          {
            confusion_matrix << confusion_matrix_mat(i,j)<<" ";
          }
          confusion_matrix << std::endl;
        }
      confusion_matrix.close();
  }
  else
  {
    std::cout << "[+] error opening confusion matrix file" << std::endl;
  } 
  // save roc curve
  roc_curve.close();
  // save roc curve script
  if(roc_curve_script_file.is_open() && roc_curve_script_file.good())
  {
    std::string roc_curve_script = "set terminal png\n"
                                   "set output \"web_requests_test_conf.png\"\n"
                                   "set title \"web requests confusion_matrix\"\n"
                                   "set xlabel \"Predicted labels\"\n"
                                   "set ylabel \"Target labels\"\n"
                                   "set grid\n"
                                   "set key off\n"
                                   "set ticslevel 0\n"
                                   "set tics out nomirror\n"
                                   "set tics scale 0.5\n"
                                   //set the 5 classes 
                                    "set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 7 ps 1.5\n"
                                    "set style line 2 lc rgb '#dd181f' lt 1 lw 2 pt 5 ps 1.5\n"
                                    "set style line 3 lc rgb '#dd181f' lt 1 lw 2 pt 9 ps 1.5\n"
                                    "set style line 4 lc rgb '#dd181f' lt 1 lw 2 pt 13 ps 1.5\n"
                                    "set style line 5 lc rgb '#dd181f' lt 1 lw 2 pt 15 ps 1.5\n"
                                    "set style line 6 lc rgb '#dd181f' lt 1 lw 2 pt 17 ps 1.5\n"
                                    "set style line 7 lc rgb '#dd181f' lt 1 lw 2 pt 19 ps 1.5\n"
                                    "set style line 8 lc rgb '#dd181f' lt 1 lw 2 pt 21 ps 1.5\n"
                                    //describe labels 
                                    "set label 1 \"100-200\" at graph 0.1,0.9\n"
                                    "set label 2 \"300-400\" at graph 0.1,0.8\n"
                                    "set label 3 \"400-500\" at graph 0.1,0.7\n"
                                    "set label 4 \"500-600\" at graph 0.1,0.6\n"
                                    "set label 5 \"OTHER\" at graph 0.1,0.5\n"

                                    //set matrix size
                                    "set xrange [0:5]\n"
                                    "set yrange [0:5]\n"
                                    //set the palette
                                    "set palette defined ( 0 '#0060ad', 1 '#dd181f', 2 '#dd181f', 3 '#dd181f', 4 '#dd181f', 5 '#dd181f', 6 '#dd181f', 7 '#dd181f', 8 '#dd181f', 9 '#dd181f', 10 '#dd181f', 11 '#dd181f', 12 '#dd181f', 13 '#dd181f', 14 '#dd181f', 15 '#dd181f', 16 '#dd181f', 17 '#dd181f', 18 '#dd181f', 19 '#dd181f', 20 '#dd181f', 21 '#dd181f', 22 '#dd181f', 23 '#dd181f', 24 '#dd181f', 25 '#dd181f', 26 '#dd181f', 27 '#dd181f', 28 '#dd181f', 29 '#dd181f', 30 '#dd181f', 31 '#dd181f', 32 '#dd181f', 33 '#dd181f', 34 '#dd181f', 35 '#dd181f', 36 '#dd181f', 37 '#dd181f', 38 '#dd181f', 39 '#dd181f', 40 '#dd181f', 41 '#dd181f', 42 '#dd181f', 43 '#dd181f', 44 '#dd181f', 45 '#dd181f', 46 '#dd181f', 47 '#dd181f', 48 '#dd181f', 49 '#dd181f', 50 '#dd181f', 51 '#dd181f', 52 '#dd181f', 53 '#dd181f', 54 '#dd181f', 55 '#dd181f', 56 '#dd181f', 57 '#dd181f', 58 '#dd181f', 59 '#dd181f', 60 '#dd181f', 61 '#dd181f', 62 '#dd181f', 63 '#dd181f', 64 '#dd181f', 65 '#dd181f', 66 '#dd181f', 67 '#dd181f', 68 '#dd181f', 69 '#dd181f')\n"
                                    //choose heatmap style
                                    "set pm3d map\n"

                                    //plot the confusion matrix
                                    "plot \"web_requests_confusion_matrix.DAT\" matrix with image\n";
                                    
      // save roc curve script
      roc_curve_script_file<<roc_curve_script;
      roc_curve_script_file.close();
      std::cout<<"[+] roc curve script saved"<<std::endl;
  }
      //save confusion matrix script
      std::string dv = "set terminal png\n"
                                            "set output \"web_requests_confusion_matrix.png\"\n"
                                            "set title \"web requests confusion matrix\"\n"
                                            "set xlabel \"Predicted class\"\n"
                                            "set ylabel \"Target Class\"\n"
                                            "set xrange [0:5]\n"
                                            "set yrange [0:5]\n"
                                            "set grid\n"
                                            "set key off\n"
                                            "set ticslevel 0\n"
                                            "set tics out nomirror\n"
                                            "set tics scale 0.5\n"
                                            "set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 7 ps 1.5\n"
                                            "set style line 2 lc rgb '#dd181f' lt 1 lw 2 pt 5 ps 1.5\n"
                                            "set style line 3 lc rgb '#dd181f' lt 1 lw 2 pt 9 ps 1.5\n"
                                            "set style line 4 lc rgb '#dd181f' lt 1 lw 2 pt 13 ps 1.5\n"
                                            "set style line 5 lc rgb '#dd181f' lt 1 lw 2 pt 15 ps 1.5\n"
                                            "set style line 6 lc rgb '#dd181f' lt 1 lw 2 pt 17 ps 1.5\n"
                                            "set style line 7 lc rgb '#dd181f' lt 1 lw 2 pt 19 ps 1.5\n"
                                            "set style line 8 lc rgb '#dd181f' lt 1 lw 2 pt 21 ps 1.5\n"
                                            "set style line 9 lc rgb '#dd181f' lt 1 lw 2 pt 23 ps 1.5\n"
                                            "set style line 10 lc rgb '#dd181f' lt 1 lw 2 pt 25 ps 1.5\n"
                                            "set style line 11 lc rgb '#dd181f' lt 1 lw 2 pt 27 ps 1.5\n"
                                            "set style line 12 lc rgb '#dd181f' lt 1 lw 2 pt 29 ps 1.5\n"
                                            "set style line 13 lc rgb '#dd181f' lt 1 lw 2 pt 31 ps 1.5\n"
                                            "set style line 14 lc rgb '#dd181f' lt 1 lw 2 pt 33 ps 1.5\n"
                                            "set xtics ('100-200' 0, '300-400' 1, '400-500' 2, '500-600' 3, 'OTHER' 4)\n" 
                                            "set ytics ('100-200' 0, '300-400' 1, '400-500' 2, '500-600' 3, 'OTHER' 4)\n"

                                            // plot:
                                            //   Warning: Single isoline (scan) is not enough for a pm3d plot.
                                            //   Hint: Missing blank lines in the data file? See 'help pm3d' and FAQ.
                                            //   Hint: If your data is all-zeros, you may need to use the 'set dgrid3d' command.
                                            //plot predicted class vs target class
                                            "plot \"web_requests_confusion_matrix.DAT\" matrix with image\n";
      // save confusion matrix script 
      confusion_matrix_script<<dv;
      confusion_matrix_script.close();
      std::cout<<"[+] confusion matrix script saved"<<std::endl;
      roc_curve.clear();
      // save gnuplot script
    softmax.gnuplot("provallo_softmax_model_web_requests_test.gnuplot"); 
    //save softmax
    softmax.save(std::string("provallo_softmax_model_web_requests_test.json"));

    //save vectorizer
    vectorizer.save(std::string("provallo_vectorizer_model_web_requests_test.json"));

    //save model data 
    if (model_data.is_open() && model_data.good())
    {
      //save the weights and biases of the softmax classifier 
      
      //go over hidden*output weights and save them 
      for ( size_t i = 0; i < softmax.getHiddenDim(); i++)
      {
        for ( size_t j = 0; j < softmax.getOutputDim(); j++)
        {
          model_data<< softmax.getWeight1()[i*softmax.getOutputDim()+j]<<" ";
          model_data<< softmax.getBias1()[i*softmax.getOutputDim()+j]<<" ";
          model_data<< softmax.getWeight2()[i*softmax.getOutputDim()+j]<<" ";
          model_data<< softmax.getBias2()[i*softmax.getOutputDim()+j]<<" ";
          model_data<< softmax.getWeight1Inc()[i*softmax.getOutputDim()+j]<<" ";
          model_data<< softmax.getBias1Inc()[i*softmax.getOutputDim()+j]<<" ";
          model_data<< softmax.getWeight2Inc()[i*softmax.getOutputDim()+j]<<" ";
          model_data<< softmax.getBias2Inc()[i*softmax.getOutputDim()+j]<<" ";
          model_data<< softmax.getWeight1Grad()[i*softmax.getOutputDim()+j]<<" ";
          model_data<< softmax.getBias1Grad()[i*softmax.getOutputDim()+j]<<" ";
          model_data<< softmax.getWeight2Grad()[i*softmax.getOutputDim()+j]<<" ";
          model_data<< softmax.getBias2Grad()[i*softmax.getOutputDim()+j]<<" ";

          
          
        }
        model_data<<std::endl;
      }

      model_data.close();
    }
    else
    {
      std::cout << "[+] error opening model data file" << std::endl;
    }
    std::cout << "[+] saving softmax classifier done" << std::endl;


 } // end test_web_requests_patterns