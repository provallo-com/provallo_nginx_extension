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
//std::filesystem::path
#include <sys/stat.h>

#include <math.h>

provallo::classifier *_lastclassifier = nullptr;
provallo::isolation_forest* iso_last_fit = nullptr;

const int kNumThreads = 8; // number of threads to use for kNN classification
std::atomic_uint64_t  numQueriesProcessed(0);
std::atomic_uint64_t  correctCount(0);
std::atomic_uint64_t  errorsCount(0);

std::mutex queryLock; // lock for global counters

 const char* vectorizer_types[] ={ 
          "TFIDF",
          "STANDARD_SCALER",
          "MIN_MAX_SCALER",
          "PCA","ONE_HOT_VECTORIZER",
          "NEURAL_TRANSFORMER",
          "AERONATIC_QARTERION", 
          "SVD_OPERATOR",
          "NGRAM_HMM_TRANSFORMER",
          "HPLANE_TRANSFORMER",
          "HUFFMAN_TRANSFORMER",
          "HMM_TRANSFORMER","REGRESSION_TRANSFORMER",
          "UMAP_VECTORIZER","TSNE_VECTORIZER","AUTOENCODER_VECTORIZER",
          "LDA_VECTORIZER","UNKNOWN_VECTORIZER","ERROR_INDEX"
          };

bool test_fit_iso_forest();
bool test_dataset_load();

template<size_t N>
std::vector<provallo::point<N>,int>  load_dataset_as_points_and_lables(const std::string& filename ) 
{
  //dynamically create point<N> and labels vector from file:

  std::vector<provallo::point<N>,int> points;
  std::ifstream file(filename);
  std::string line;
  while (std::getline(file, line))
  {
    std::istringstream iss(line);
    std::string token;
    provallo::point<N> p ;
    int i = 0;
    int label=0;
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
    points.push_back(std::make_pair(p,label));
  }
  return points;
}

// alternative to dataset load
provallo::matrix<real_t> read_data_file(const std::string& filepath ) 
{

  std::cout<<"[+] reading data file : "<<filepath<<std::endl;

  //use bag of words to transform strings to numbers
  try
  {
  
 
  std::ifstream file(filepath);

  std::string line;
  //for each feature(column) we have a map of values and counts
  //for each row we have a map of features and values
  std::map< size_t/*index*/, std::map<std::string/*value*/, size_t/*count*/>> BoW; 
  std::map< size_t/*index*/, std::map<std::string/*value*/, size_t/*count*/>> BoW_labels;  
  std::map< size_t/*index*/, std::map<std::string/*value*/, size_t/*count*/>> BoW_features;
  
  std::vector<provallo::bit_type<uint32_t,17>> types;

  
  std::vector<std::string> labels;
  std::vector<std::string> features;
  std::vector<std::vector<std::string>> data;
  std::vector<std::string> row;
  //17 default types than can be mapped to each cell 
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
      //check if cell is empty
       //trim cell
      column_index++;
      cell.erase(std::remove_if(cell.begin(), cell.end(), [](char c) { return std::isspace(c); }), cell.end());  
      if (cell.empty())
      {
        continue; 
      }
      //check if cell is a label / discrete value or a feature / continous value
      //start :

      //bit vector of 18 bits:
      //label,feature,discrete,continous,vector,matrix,string,bool,date,time,datetime,categorical,ordinal,nominal,interval,ratio,count:

      
      std::string continous_s = "^(( )*|([0-9]+(\\.[0-9]+)?))$";
      std::string discrete_s = "^(( )*|([0-9]+))$";
      std::string label_s = "^(( )*|([0-9]+))$";
      std::string feature_s = "^(( )*|([0-9]+(\\.[0-9]+)?))$";
      std::string vector_s = "^(( )*|([0-9]+(\\.[0-9]+)?))$";
      std::string matrix_s = "^(( )*|([0-9]+(\\.[0-9]+)?))$";
      std::string string_s = "^$";
      //TRUE or true or FALSE or false or 0 or 1 :
      std::string bool_s = " ^(( )*|([0-1]+))$";
      //date format : YYYY-MM-DD
      std::string date_s = "\\d{4}-[01]\\d-[0-3]\\dT[0-2]\\d:[0-5]\\d:[0-5]\\d(?:\\.\\d+)?Z?"; 
      //time format : HH:MM:SS.mmmmmm
      std::string time_s = "\\d{2}:\\d{2}:\\d{2}\\.\\d{6}";
      //datetime format : YYYY-MM-DDTHH:MM:SS.mmmmmm
      std::string datetime_s = "\\d{4}-[01]\\d-[0-3]\\dT[0-2]\\d:[0-5]\\d:[0-5]\\d(?:\\.\\d+)?Z?";
      //categorical,ordinal,nominal,interval,ratio,count
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

      //check and update column information

 
      provallo::bit_type<uint32_t,17> type(0);
      if(types.size()<column_index)
      {
        types.push_back(type);
      }
      else
      {
        type = types[column_index-1];
      }
      
      //end
      //check and update column information
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
        //is_ratio = true;
      }
        if (std::regex_search(cell, count_regex))
      {
        type.flip(16);
       // is_count = true;
      }
      //end 
      if( type==0)
      //set continous as default
      {
        type.flip(3);
      //  is_continous = true;
      } 
      
      types[column_index-1]=type;
      //end
      row.push_back(cell);

    }
    data.push_back(row);
    row.clear();
 
  } 
  
  //now we can transform the data
  provallo::matrix<real_t> X( data.size(),data[0].size());
  size_t i = 0;
  for (const auto& row : data)
  {
    size_t j = 0;
    for (const auto& cell : row)
    {
      X(i, j) = std::stod(cell);
      j++;
    }
    i++;
  }
  return X; 

  }
  catch (const std::exception& e)
  {
    std::cout << e.what() << std::endl;
    return provallo::matrix<real_t>(1,1);
  }


}

provallo::isolation_forest* isoforest_single(const provallo::attribute_information& attributes);

bool benchmark_classifiers (const std::string benchmark_folder = "./db/benchmarks");

bool fit_fuzzsb(const std::string& benchmark_folder = "/home/kardon/eclipse-workspace/fuzzdb");
 
ssize_t
which_max (std::vector<double> &v)
{
  auto loc_max_el = std::max_element (v.begin (), v.end ());
  return std::distance (v.begin (), loc_max_el);
}

bool
test_fit_iso_forest ()
{
  /* Random data from a standard normal distribution
   (100 points generated randomly, plus 1 outlier added manually)
   Library assumes it is passed as a single-dimensional pointer,
   following column-major order (like Fortran) */
  int nrow = 101;
  int ncol = 2;
  std::vector<double> X (nrow * ncol);
  std::default_random_engine rng (1);
  std::normal_distribution<double> rnorm (0, 1);

#define get_ix(row, col) (row + col*nrow)
  for (int col = 0; col < ncol; col++)
    for (int row = 0; row < 100; row++)
      X[get_ix(row, col)] = rnorm (rng);

  /* Now add obvious outlier point (3,3) */
  X[get_ix(100, 0)] = 3.;
  X[get_ix(100, 1)] = 3.;
  provallo::isolation_forest iso = provallo::isolation_forest ();
  iso.fit (X.data (), nrow, ncol);
  std::vector<double> outlier_scores = iso.predict (X.data (), nrow, true);
  int row_highest = which_max (outlier_scores);
  std::cout << "[+]Provallo IsoForest test" << std::endl;
  std::cout << std::string ("Point with highest outlier score: [")
      << X[get_ix(row_highest, 0)] << std::string (", ")
      << X[get_ix(row_highest, 1)] << std::string ("]") << std::endl;
  return true;
}
     // fitting ISO FORESTS

provallo::isolation_forest* isoforest_single(const provallo::attribute_information& attributes)
{
 
  static const provallo::attribute_information& att_stat = attributes ;
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
  static provallo::isolation_forest* forest = new provallo::isolation_forest ();
  
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
 //ignore forest hyper parameters for now 
  if( attributes==att_stat){
    return forest;
    }
  else
  {
    return new provallo::isolation_forest ();    
  }
}
std::vector<std::string>   getFilesInFolder(const std::string & benchmark_folder)
{
   std::vector<std::string> files;
  //only c++2a has std::filesystem
   //std directory iterator
  // use c-style dirent

  DIR *dir;
  struct dirent *ent;

  if ((dir = opendir (benchmark_folder.c_str())) != NULL) {
    /* print all the files and directories within directory */
    while ((ent = readdir (dir)) != NULL) {
      std::string file_name = ent->d_name;
      if (file_name.find(".names") != std::string::npos)
      {
        files.push_back(file_name);
      }
    }
    closedir (dir);
  } else {
    /* could not open directory */
    perror ("");
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
      else {
      ++errorsCount;
         std::cout << "Error: " << p.second << " " << pred << std::endl;
       }
    if (numQueriesProcessed % 500 == 0)
      std::cout << numQueriesProcessed << std::endl;
    queryLock.unlock();
  }
}


bool test_fast_knn(const std::string benchmark_folder = "./db/benchmarks" );

bool test_fast_knn(const std::string benchmark_folder )
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

        std::cout<<"-- found description file : "<<file<<std::endl;

      }
    }
    //iterate all the description files, build 'collector' and 'classifiers' for each of them

    for(auto description_file : description_files)
    { 
        std::string file_stem = description_file.substr(0,description_file.find(".names"));
        std::string data_file = benchmark_folder+"/"+file_stem+".data";
        //std::string weights_file_name = benchmark_folder+"/"+file_stem+".weights";
        //std::ifstream weights_ (weights_file_name);
        //std::map<std::string,Float> weights;
        std::cout<<"-- building dataset for : "<<file_stem<<std::endl;
        provallo::files_collector collector = provallo::files_collector (benchmark_folder+"/"+file_stem);
        const provallo::attribute_information& attributes = collector.getAttributes (),attributes_copy=collector.getAttributes ();
        //checking zero knowledge distributed kmeans 
        std::cout<<"-- Attributes information : "<<std::endl<<std::endl;
        std::cout<< description_file<<std::endl;
        std::cout<<collector.getAttributes ()<<std::endl;
        std::cout<<"-- checking for weights .... "<<std::endl<<std::endl;
       // std::cout<<"weights size : "<<std::to_string(weights.size())<<std::endl;
        std::cout<<"attributes size : "<<std::to_string(attributes.getSize())<<std::endl; 
        /*std::cout<<"-- weights : "<<std::endl<<std::endl;
        for (const auto& weight : weights)
        {
          std::cout<<weight.first<<" : "<<std::to_string(weight.second)<<std::endl;
        } */
        
        std::cout<<std::endl<<"------- attributes --------"<<std::endl<<std::endl;
        provallo::training_set set(attributes);
        // add default weight map  to dataset itself, not classifier 
        // params. 

        collector.pushTrainData (&set);   
        std::cout<<"-- training set size : "<<std::to_string(set.size())<<std::endl;
        provallo::matrix<provallo::attribute> data = set.get_matrix();



        std::cout<<"-- training set matrix size : "<<std::to_string(data.size1()*data.size2())<<std::endl; 
         
        if( false && attributes.getSize()==14)
        {
          //will crash.
          //wine ?
          using  datarecord14 = std::vector<std::pair<provallo::point<14>, unsigned int>>; 
          datarecord14 record, testData;
          for (size_t i = 0; i < data.size1(); ++i)
          {
            provallo::point<14> point;
            for (size_t j = 0; j < data.size2(); ++j)
            {
              point[j] = data(i,j).continous();
            }
            record.push_back(std::make_pair(point,data(i,set.get_target_tag()).discrete()));

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

          errorsCount=correctCount=0;
          //
          provallo::testing_set testset(attributes_copy); 
          collector.pushTestData (&testset);
          const provallo::testing_samples& samples = testset.get_samples();
          provallo::attribute_tag target_tag = testset.get_target_tag(); 
          if(target_tag!=set.get_target_tag()) 
          { 

            throw std::runtime_error("target tag mismatch");
          }
          size_t j=0;
          for (const auto& sample : samples)
          {
              provallo::point<14> point;
              const size_t point_size = 14;
             {
                if(( j==0) || ( j!=0&& j != target_tag &&(j%target_tag!=0) )  )
                {
                  point[j%point_size] = sample.continous();
                }
                else
                {
                    uint32_t target = sample.discrete();

                    if (target==1||target==2||target==3 )
                    {
                        std::cout<<"[+] target is in range" <<std::endl;

                    }
                    else
                    {
                        std::cout<<"[+] target is not in range corrupt testing_set load" <<std::endl;
                        throw std::runtime_error(std::string("target") + std::to_string(target)+" is not in range corrupt testing_set load");
                    }
                    testData.push_back(std::make_pair(point,target));
                    std::cout<<"[+] adding test data for label: "<<std::to_string(target)<<std::endl;
                     
                }
            }
            j++;
          }//for each sample



          size_t testCnt = testData.size();
          std::cout << "[+]Loaded " << testCnt << " test records." << std::endl;  
          std::cout << "[+]Starting benchmarking..." << std::endl;  
            size_t k = sqrt( attributes.getSize() ); // Number of nearest neighbors
              
              int queriesPerThread =(int) double(testCnt) / double( kNumThreads );
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


              }//for each thread
              for (std::thread &t : threads)
              t.join();
              clock_t c_end = clock();
              auto t_end = std::chrono::high_resolution_clock::now();
              std::cout << "[+] Test rate: " << correctCount-errorsCount  * 100.0 / double(testCnt) << "%"  
                        << std::endl;
              std::cout << "[+] CPU time elapsed in s: "
                        << (double)(c_end - c_start) / CLOCKS_PER_SEC << std::endl;
              std::cout << "[+] Wall time elapsed in s: "
                        << std::chrono::duration<double>(t_end - t_start).count() << std::endl;
              std::cout << "[+] Finished benchmarking!" << std::endl; 
          }//if attributes size == 14


       }
      
    //we have a set of benchmarking datasets for classification. 
    //let's test with fast_knn
    //for each dataset, we need to build a classifier
    //for each classifier, we need to test it with fast_knn
    

    ret = true;
    return ret;
  
    }//test_fast_knn

   
   
bool
test_dataset_load ()
{
  try
    {
      provallo::files_collector collector ("provallo_core/provallo_attributes");
      const provallo::attribute_information& attributes (
	    collector.getAttributes ());
      
      std::ifstream weights_file("provallo_core/provallo_attributes.weights");
      std::map<std::string,Float> weights = provallo::getWeightMap(attributes,weights_file);
 

      std::cout << "-- Attributes information : " << std::endl << std::endl;
      std::cout << collector.getAttributes () << std::endl;
      std::cout << "-- checking for weights .... " << std::endl << std::endl;
      std::cout << "weights size : " << std::to_string(weights.size()) << std::endl;          
      std::cout << "attributes size : " << std::to_string(attributes.getSize()) << std::endl;
      std::cout << "attributes groups size : " << std::to_string(attributes.getGroups().size()) << std::endl;

      auto c_start = clock ();
      auto t_start = std::chrono::high_resolution_clock::now ();
      provallo::training_set train_data (collector.getAttributes ());
      collector.pushTrainData (&train_data);
      
      auto c_end = clock ();  
      auto t_end = std::chrono::high_resolution_clock::now ();

      std::cout<<"[+] training set size : "<< std::to_string (  train_data.size() )<<std::endl;

      
      std::cout<<"[+] training set attributes : "<< std::to_string (train_data.getattributesNumber())<<std::endl;
      std::cout<<"[+] training set samples : "<< std::to_string (train_data.get_samples().size())<<std::endl;
      std::cout<<"[+] training set entropy : "<< std::to_string (train_data.entropy())<<std::endl;
      std::cout<<"[+] training set skewness : "<< std::to_string (train_data.skewness())<<std::endl;
      std::cout<<"[+] training set kurtosis : "<< std::to_string (train_data.kurtosis())<<std::endl;
      std::cout<<"[+] training set standard deviation : "<< std::to_string (train_data.stddev())<<std::endl;
      std::cout<<"[+] training set mean : "<< std::to_string (train_data.mean())<<std::endl;
      std::cout<<"[+] training set median : "<< std::to_string (train_data.median())<<std::endl;
      std::cout<<"[+] training set variance : "<< std::to_string (train_data.variance())<<std::endl;

      std::cout << "[+] CPU time elapsed in s: "
                << (double) (c_end - c_start) / CLOCKS_PER_SEC << std::endl;


      //std::cout << "Fitting iso-forest: " << std::endl;
      //size_t sample_size = train_data.getattributesNumber();
      //size_t column_size = train_data.getattributes().getSize();
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

      c_start = clock ();
      
      //setDiscretizationType(DiscretizationType::EqualFrequency);
      //setDiscretizationType(DiscretizationType::EqualWidth);
      
      provallo::testing_set test_data (collector.getAttributes ());
      collector.pushTestData (&test_data);

      std::cout << "Testing DTGR Classifier: " << std::endl;

      std::cout << "[+] Test CPU time elapsed in s: "
	  << (double) (c_end - c_start) / CLOCKS_PER_SEC << std::endl;
      std::cout << "[+] Test Wall time elapsed in s: "
	  << std::chrono::duration<double> (t_end - t_start).count ()
	  << std::endl;

      c_start = c_end;
      t_start = t_end;
      const provallo::testing_samples & attributes_vector (  test_data.get_samples());
      std::cout<<"[+] attributes vector size "<<attributes_vector.size()<<std::endl;
    
      //test nn
      
      
      std::cout << "Testing KMB Classifier: " << std::endl;
     
     typedef provallo::metric<provallo::Overlap,provallo::Euclidean> over_euc;
     typedef provallo::Kmeans<over_euc> km_classifier;
      provallo::kmeans_param km_param(5,weights);
      provallo::train_and_test<km_classifier>  (
        "kmeans-binary-split", train_data, test_data ,km_param  );

      
      c_end = clock ();
      t_end = std::chrono::high_resolution_clock::now ();
      std::cout << "[+] Test CPU time elapsed in s: "
              << (double) (c_end - c_start) / CLOCKS_PER_SEC << std::endl;
    
      std::cout << "Training RF Classifier with 50 parallel trees " << std::endl;

      //train and test DTGR
      typedef provallo::decision_tree<provallo::GainRatio> dtgr_classifier;
      provallo::none dtgr_param;
       provallo::train_and_test<dtgr_classifier>  (
        "dtgr", train_data, test_data ,dtgr_param  );

      c_end = clock ();
      std::cout << "[+] Test CPU time elapsed in s: "
	  << (double) (c_end - c_start) / CLOCKS_PER_SEC << std::endl;
      std::cout << "[+] Test Wall time elapsed in s: "
	  << std::chrono::duration<double> (t_end - t_start).count ()
	  << std::endl;
      typedef provallo::random_tree<provallo::GainRatio> egain_tree;
      typedef provallo::random_forest<egain_tree> rf_classifier;
      provallo::random_tree_param egain_param(.5,10,0.0) ;
      
      provallo::random_forest_param rf_param(50,egain_param);
 

      c_start = clock ();
      provallo::train_and_test<rf_classifier>  (
	      "rf-rf", train_data, test_data ,rf_param  );
      c_end = clock ();
      t_end = std::chrono::high_resolution_clock::now ();

      std::cout << "[+] Test CPU time elapsed in s: "
	  << (double) (c_end - c_start) / CLOCKS_PER_SEC << std::endl;
      std::cout << "[+] Test Wall time elapsed in s: "
	  << std::chrono::duration<double> (t_end - t_start).count ()
	  << std::endl;


      //std::cout << "Training Kmeans bayesian Classifier: " <<std::endl;
      //provallo::train_and_test<provallo::Kmeans<provallo::bayesian> >("kmeans_bayesian", train_data, test_data);
      //std::cout << "Training ABB Classifier: " <<std::endl;

      // provallo::train_and_test<provallo::adaboost <provallo::bayesian>>("gain ratio split, binary splitting", train_data, test_data);

      // c_end = clock();
      //t_end = std::chrono::high_resolution_clock::now();

      //std::cout << "[+] Test CPU time elapsed in s: " << (double)(c_end - c_start) / CLOCKS_PER_SEC << std::endl;
      //std::cout << "[+] Test Wall time elapsed in s: " << std::chrono::duration<double>(t_end - t_start).count() << std::endl;

    }
  catch (std::exception &e)
    {
      std::cerr << std::string ("exception raised : ") + e.what () << std::endl;
      return false;
    }
  return true;

}

int main ( int argc, char* argv[]  ) {    

  if ( argc>1) 
  {
    std::cout <<argv[0] << " running...." << std::endl;
    
  }
  if (test_fit_iso_forest())
  {
    std::cout << "Test fit iso forest OK" << std::endl;
  }

  //benchmark fuzzdb dataset
  if(  fit_fuzzsb( ) ) 
  {
    std::cout << "Fuzzdb test OK" << std::endl;

  }


  if(benchmark_classifiers("./db/benchmarks"))
  {
    std::cout << "Classifiers benchmark OK" << std::endl;
  }
  else
  {
    std::cout << "Classifiers benchmark FAILED" << std::endl;
    exit(-1);
  }

  if(test_fast_knn("./db/benchmarks"))
  {
    std::cout << "Test fast knn OK" << std::endl;
  }
  else
  {
    std::cout << "Test fast knn FAILED" << std::endl;
    exit(-1);
  }

  if ( test_dataset_load() ) 
  {
      std::cout << "Test dataset load OK" << std::endl;
  }
  else
  {
      std::cout << "Test dataset load FAILED" << std::endl;
  }         
  return 0;
}

bool test_vectorizers ( ) 
{
  //normalize the attributes 

 
  bool ret = false;
  std::vector<std::string> vectorize_set_attributes({"ONE","TWO","THREE","FOUR","FIVE","SIX","SEVEN","EIGHT","NINE","TEN"}); 
  std::vector<std::string> vectorize_set_classes({"CLASS1","CLASS2","CLASS3","CLASS4","CLASS5","CLASS6","CLASS7","CLASS8","CLASS9","CLASS10"}); 

  std::vector<std::vector<std::string>> vectorize_set_data;
  for(size_t  i=0;i < vectorize_set_attributes.size() ; i++)
  {
    std::vector<std::string> row;
    for(size_t j=0; j < 10; j++)
    {
      if(j==i)
        row.push_back(vectorize_set_attributes[i]);
      else
       row.push_back(std::to_string(i*j));
    }
    vectorize_set_data.push_back(row);
  }
   
  std::cout << "Testing vectorizers" << std::endl;

  std::cout << "Testing vectorizer 1" << std::endl;
  
  provallo::tfidf_vectorizer vectorizer1;
  //provallo::pca_vectorizer vectorizer2;
  provallo::lda_vectorizer vectorizer3;
  provallo::pca_vectorizer vectorizer4;
   //provallo::nmf_vectorizer vectorizer4;
  //provallo::svd_vectorizer vectorizer5;
  //provallo::kmeans_vectorizer vectorizer6;
  //provallo::knn_vectorizer vectorizer7;
  //provallo::random_projection_vectorizer vectorizer8;
  //provallo::fast_knn_vectorizer vectorizer9;
  
  
  vectorizer1.fit(vectorize_set_data);
  //vectorizer2.fit(vectorize_set_data);
  vectorizer3.fit(vectorize_set_data);
  vectorizer4.fit(vectorize_set_data);

  std::vector<provallo::vectorizer<std::string,real_t>*> vectorizers;
  vectorizers.push_back(&vectorizer1);
  //vectorizers.push_back(&vectorizer2);
  vectorizers.push_back(&vectorizer3);
  //vectorizers.push_back(&vectorizer3);
  vectorizers.push_back(&vectorizer4);

  //vectorizers.push_back(&vectorizer4);

  
  for ( size_t i=0;i<vectorize_set_data.size();i++)
  {
    size_t j=0;
    std::cout << "Testing vectorizer pass 1 transform" << std::endl;
    for ( auto v : vectorizers   )
    {

      //print vectorizer name
      std::cout << " [+] vectorizer :  "<<vectorizer_types[v->get_type()] << std::endl; 

      std::vector<double> vectorized_data = v->transform(vectorize_set_data[i]);
      j=0;
      for(auto f : vectorized_data)
      {
        std::cout << " [+] transform :  "<<std::to_string(f) << " "<< (vectorize_set_data[i][j++ %vectorize_set_data[i].size()])<<std::endl;

        ret = true;
      } 
      std::cout << std::endl;
    }
  } 
  std::cout << std::endl;
  char x = std::getchar(); 
  x--;
  for ( size_t i=0;i<vectorize_set_data.size();i++)
  {
     
     std::cout << "Testing vectorizer pass 2 transform" << std::endl;
    for ( auto v : vectorizers   )
    {
      std::vector<double> vectorized_data = v->transform(vectorize_set_data[i]);
      size_t j=0;

      std::cout << " [+] vectorizer :  "<<vectorizer_types[v->get_type()] << std::endl; 
 
      for(auto f : vectorized_data)
      {
        std::cout << " [+] predict :  "<<std::to_string(f) << " "<<vectorize_set_data[i][j++%vectorize_set_data[i].size()]<<std::endl;
        ret = true;
      } 
      std::cout << std::endl;
    }
  } 


 
  return ret;

}


bool benchmark_classifiers (const std::string benchmark_folder )
{
    bool ret = false;
    //bool buse_random_forest = false;
    //bool test_ultra_fast_knn = false;
    //iterate all the files in the folder, if it's a descrition file, build dataset and classifiers for it
    std::vector<std::string> files = getFilesInFolder(benchmark_folder);
    std::vector<std::string> description_files;
    for (auto file : files)
    {
      if (file.find(".names") != std::string::npos)
      {
        description_files.push_back(file);

        std::cout<<"-- found description file : "<<file<<std::endl;

      }
    }
  
    //iterate all the description files, build 'collector' and 'classifiers' for each of them
    //bool sanity_check = false;
    for(auto description_file : description_files)
    { 
        std::string file_stem = description_file.substr(0,description_file.find(".names"));
        std::string data_file = benchmark_folder+"/"+file_stem+".data";
        //std::string weights_file_name = benchmark_folder+"/"+file_stem+".weights";
        //std::ifstream weights_ (weights_file_name); 
        std::map<std::string,Float> weights;
        std::cout<<"-- building dataset for : "<<file_stem<<std::endl;
        std::vector<provallo::classifier*> classifiers ;
 
        provallo::files_collector collector = provallo::files_collector (benchmark_folder+"/"+file_stem);
 
        provallo::attribute_information attributes = collector.getAttributes ();

        size_t nclasses = attributes.getTargetClassCount();
        //checking zero knowledge distributed kmeans

        //provallo::auto_encoder<double,double> encoder( collector.getAttributes().getSize(),collector.getAttributes().getSize()*collector.getAttributes().getSize(),collector.getAttributes().getTargetClassCount() );
        provallo::softmax_classifier<double,double> softmax(/*nclasses*/nclasses, (size_t)/*dimensions*/attributes.getSize(),/*alpha*/(real_t)1.,(real_t)0.05/*lambda*/);  

        std::cout<<"-- attribute info : "<<std::endl<<std::endl;
        std::cout<< attributes <<std::endl<<std::endl;
        char x =      std::getchar();
      
        std::cout<< attributes <<std::endl<<std::endl;

       // x+= std::getchar();
        //checking zero knowledge distributed kmeans 


        std::cout<<"-- Attributes information : "<<std::endl<<std::endl;
        std::cout<< description_file<<std::endl;
        std::cout<<collector.getAttributes ()<<std::endl;
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
        std::cout<<"attributes size : "<<std::to_string(attributes.getSize())<<std::endl; 
        /*std::cout<<"-- weights : "<<std::endl<<std::endl;
        for (const auto& weight : weights)
        {
          std::cout<<weight.first<<" : "<<std::to_string(weight.second)<<std::endl;
        } */
        
        //std::cout<<std::endl<<"------- attributes --------"<<std::endl<<std::endl;
        //provallo::training_set set(attributes);
        
        
        // add default weight map  to dataset itself, not classifier 
        // params. 

        //collector.pushTrainData (&set);   
        //std::cout<<"-- training set size : "<<std::to_string(set.size() )<<std::endl;
        provallo::matrix<real_t> data = read_data_file(data_file);

        std::cout<<"-- training set matrix size : "<<std::to_string(data.size1()*data.size2())<<std::endl; 
        std::cout<<"-- training set matrix size1 : "<<std::to_string(data.size1())<<std::endl;
        std::cout<<"-- attribute info : "<<std::endl<<std::endl;
         //size_t nclasses = set.getattribute_info().getTargetClassCount();
        //size_t ndimention = set.getattribute_info().getSize();
        std::cout<<"-- press enter to continue .... "<<std::endl<<std::endl;

        x+=      std::getchar();
        //train encoders 
        //provallo::class_dist ds(set.getattribute_info().getTargetClassCount());
        provallo::matrix<double> mdata(data),odata(data.size1(),nclasses);
        for(size_t i=0;i<data.size1();i++)
        {
          for(size_t j=0;j<data.size2();j++)
          {
            mdata(i,j) = data(i,j);
          }
          odata(i,data(i,attributes.getTargetClassCount())) = 1.0;
        }
        //std::cout<<"-- training encoder "<<std::endl;


         

        //encoder.train(mdata,ds);
        
        softmax.train(mdata ,odata);    
        //get prediction accuracy
        softmax.predict(mdata,odata); 
        size_t correct = 0;
        for(size_t i=0;i<data.size1();i++)
        {
          size_t max = 0;
          for(size_t j=0;j<nclasses;j++)
          {
            if(odata(i,j)>odata(i,max))
              max = j;
          }
          if(max==data(i,attributes.getTargetClassCount()))
            correct++;
            else
            {
              std::cout<<"-- prediction error : "<<std::to_string(max)<<" "<<std::to_string(data(i,attributes.getTargetClassCount()))<<std::endl;
              std::cout<<"-- backpropagating error "<<std::endl; 
              softmax.backpropogate_error (mdata,odata);


            }

        } 
        //
        //calculate prediction accuracy
        real_t accuracy = correct*100.0/data.size1(); 
        std::cout<<"-- softmax accuracy : "<<std::to_string(accuracy)<<std::endl;

      //calculate prediction accuracy
      
 
        //std::cout<<"-- encoder trained, classdist size : "<<std::to_string(ds.size())<<std::endl;

        //std::cout<<"-- encoder trained, classdist size : "<<std::to_string(ds.size())<<std::endl; 
        //std::cout<<"-- classdist: "<<ds<<std::endl;

        //encoder.save("encoder_"+file_stem+".json");
 
        //encoder.load("encoder_"+file_stem+".json");
        std::cout<<"-- encoder loaded "<<std::endl;
        std::cout<<"-- encoder input size : "<<std::to_string(softmax.getInputDim())<<std::endl;
        std::cout<<"-- encoder output size : "<<std::to_string(softmax.getOutputDim())<<std::endl;
        std::cout<<"-- encoder hidden size : "<<std::to_string(softmax.getHiddenDim())<<std::endl;  
      
      std::cout<<"-- building classifiers "<<std::endl<<std::endl;


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
      #endif //0 
      x += std::getchar();
  
      std::cout<<"-- building neural network .... "<<std::endl<<std::endl;
      std::cout<<"-- training classifiers .... "<<std::endl<<std::endl;
      std::cout<<"-- testing classifiers .... "<<std::endl<<std::endl; 
      //read test data

      std::cout<<"-- reading test data .... "<<std::endl<<std::endl;
      //provallo::testing_set test_set( set.getattribute_info());
      //collector.pushTestData (&test_set);
      std::string test_data_file = benchmark_folder+"/"+file_stem+".test";
      mdata = read_data_file(test_data_file);
        //test autoencoder
      std::cout<<"-- testing autoencoder .... "<<std::endl<<std::endl;
      //go over the testing set and test the autoencoder
      
      //test softmax
      std::cout<<"-- testing softmax .... "<<std::endl<<std::endl;
      //go over the testing set and test the autoencoder
      odata.resize(mdata.size1(),mdata.size2());
      softmax.predict(mdata,odata);
       
      std::cout<<"-- softmax test finished "<<std::endl<<std::endl;
      std::cout<<"-- softmax test results : "<<std::endl<<odata<<std::endl<<std::endl;
      std::cout<<"-- softmax test results ---"<<std::endl<<std::endl;

      //save softmax results
      std::ofstream softmax_results("softmax_results_"+file_stem+".csv");
      softmax_results<<odata<<std::endl;
      softmax_results.close();
      //save softmax classifier 
      softmax.save("softmax_"+file_stem+".json");

      //
      std::cout<<odata<<std::endl<<std::endl;
      //test kmeans

      //print confusion matrix of test data
      
      /*for(auto & class_ : classifiers)
      {
           print_classifier_summary(file_stem,test_set,*class_);          
      }*/
      //test ultra fast knn : 
      
       


      //weights_.close();
      std::cout<<"-- deleting description file .... "<<std::endl<<std::endl;

      description_file.clear();
      mdata.resize(1,1);
      odata.resize(1,1);
      ret = true;

      //clean up
      //delete factory
      //std::cout<<"-- deleting random factory .... "<<std::endl<<std::endl;
      //if(random_factory)
       // delete random_factory;
      //std::cout<<"-- deleting factory .... "<<std::endl<<std::endl;
      //if(factory)
        //   delete factory;

/*
      std::cout<<"-- deleting factory .... "<<std::endl<<std::endl;
      if(factory)
        delete factory;
      std::cout<<"-- deleting random factory .... "<<std::endl<<std::endl;
      if(random_factory)
        delete random_factory;
        */
   

      //avoid double free 
      //if(random_factory)
      // delete random_factory;
      //delete factory
      //delete classifiers
      //delete weights
      //delete weights file


    } //end for description_files
    return ret;
}// end benchmark_classifiers
 
      
std::vector<std::string> get_files_recursively (const std::string folder)
{
  std::vector<std::string> files;
  std::vector<std::string> subfolders;
  std::vector<std::string> subfiles;
  std::vector<std::string> ret;


  DIR *dir;
  struct dirent *ent;

  if ((dir = opendir (folder.c_str())) != NULL) {
    /* print all the files and directories within directory */
    while ((ent = readdir (dir)) != NULL) {
      //check if entry is a file or a subfolder
      if ( ent->d_type == DT_DIR)
      {
        std::string subfolder = ent->d_name;
        if (subfolder != "." && subfolder != ".." )
        {
          if(subfolder[subfolder.length()-1]=='/')
          subfolders.push_back(folder+subfolder);
          else
          subfolders.push_back(folder+"/"+subfolder);
        }
      }
      else if (ent->d_type == DT_REG)
      {
        std::string file_name = ent->d_name;
          if(folder[folder.length()-1]=='/')
          files.push_back(folder+file_name);
          else
           files.push_back(folder+"/"+file_name);
       } 
      else
      {
        std::cout<<"[-] skipping  "<< std::string(ent->d_name)<< std::endl;
      }
    }
    closedir (dir);
  } else {
    /* could not open directory */
    perror ("could not open directory");
    return ret;
  }
  std::cout<<"\t[+] total sub-folders : "<<subfolders.size()<<std::endl;
  for(auto subfolder : subfolders)
  {
    subfiles = get_files_recursively(subfolder);
    files.insert(files.end(),subfiles.begin(),subfiles.end());
  }
  return files;
} 
bool fit_fuzzsb ( const std::string& fit_fuzzsb_folder)
{
 
  bool ret = true;
  std::vector<std::string> files = get_files_recursively(fit_fuzzsb_folder);
  std::vector<std::string> string_files, string_files_attacks, string_files_normal; 
  
  //std::vector<provallo::vectorizer<std::string,real_t>*> vectorizers;
 
  std::vector<provallo::vectorizer<std::string,real_t>*> vectorizers_attacks;
  std::vector<provallo::vectorizer<std::string,real_t>*> vectorizers_normal;
  std::vector<provallo::auto_encoder<real_t,real_t>*> autoencoders;

  std::vector<std::string>  exclude_list;
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

  //first , validate vectorizers mechanism works: 
  if ( !test_vectorizers() ) 
  {
    std::cout<<"[-] vectorizers test failed "<<std::endl;
    return false;
  } 

  for (auto file : files)
  {
 
      bool skip = false;
      for (auto exclude : exclude_list)
      {
        std::string exclude_me = file.substr(file.find_last_of("/")+1);
        if ( exclude_me.find(exclude) == std::string::npos)
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



      //check if attack or normal file
      if(file.find("attack")!=std::string::npos||file.find("door")!=std::string::npos||file.find("discovery")!=std::string::npos )
      {
        string_files_attacks.push_back(file);
      }
      else
      {
        string_files_normal.push_back(file);
      }

      string_files.push_back(file);

  } 
  std::cout<<"[+] found "<<string_files.size()<<" files in folder "<<fit_fuzzsb_folder<<std::endl;
  std::cout<<"[+] found "<<files.size()<<" files in folder "<<fit_fuzzsb_folder<<std::endl;
  std::cout<<"[+] found "<<string_files_attacks.size()<<" attack files in folder "<<fit_fuzzsb_folder<<std::endl; 
  std::cout<<"[+] found "<<string_files_normal.size()<<" normal files in folder "<<fit_fuzzsb_folder<<std::endl;  

  //feed the files to the pipeline, 
  //the pipeline will load the files, vectorize them, train the autoencoder, 

  //first we want to verify the vectorizers and autoencoders work properly 
  vectorizers_attacks.push_back(new provallo::lda_vectorizer);
  vectorizers_attacks.push_back(new provallo::one_hot_vectorizer);
  vectorizers_attacks.push_back(new provallo::pca_vectorizer);
  
  vectorizers_normal.push_back(new provallo::lda_vectorizer);
  vectorizers_normal.push_back(new provallo::one_hot_vectorizer);
  vectorizers_normal.push_back(new provallo::pca_vectorizer);
  
  //vectorizers_normal.push_back(new provallo::auto_encoder_vectorizer<real_t,real_t>(autoencoders)); 
  //create vectorizers:
//  vectorizers.push_back(new provallo::pca_vectorizer);
    //vectorizers.push_back(new provallo::word2vec_vectorizer);
  clock_t c_start, c_end,c_point;
  std::chrono::high_resolution_clock::time_point t_start, t_end,t_point;
  c_start = clock ();
  t_start = std::chrono::high_resolution_clock::now ();
  t_point = t_start;
  c_point = c_start;

  size_t total =string_files.size();
  size_t cur =  0;

  
  for (auto & fuzz_file : string_files_attacks )
  {
    std::ifstream fuzz(fuzz_file);
    std::string fuzz_string((std::istreambuf_iterator<char>(fuzz)),
    std::istreambuf_iterator<char>());
     
   // std::cout<<"[+] fuzzing "<<fuzz_file<< "("<<std::to_string(cur)+"/"+std::to_string(total)<<")"<<std::endl;
    for (auto & vectorizer : vectorizers_attacks )
    {
      
      std::string vectorizer_type = vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER];//= vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER];//unknown vectorizer->get_type();
      if ( vectorizer->get_type() < provallo::vectorizer_type::UNKNOWN_VECTORIZER)
      {
        vectorizer_type = vectorizer_types[vectorizer->get_type()];
      } 
      std::cout<<"[+] vectorizing attack with "<<vectorizer_type<<std::endl;
      vectorizer->add_document(fuzz_string);
      std::cout<<"[+] vectorizing attack  with "<<vectorizer_type<<" done"<<std::endl;
      std::cout<<"[+] finished "<<fuzz_file<<std::to_string(cur)+"/"+std::to_string(total)<<std::endl; 

    }
  
    cur++;

   }
  for (auto & fuzz_file : string_files_normal )
  {
    std::ifstream fuzz(fuzz_file);
    std::string fuzz_string((std::istreambuf_iterator<char>(fuzz)),
    std::istreambuf_iterator<char>());


    for (auto & vectorizer : vectorizers_normal )
    { 
      std::string vectorizer_type = vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER];//= vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER];//unknown vectorizer->get_type();
      if ( vectorizer->get_type() < provallo::vectorizer_type::UNKNOWN_VECTORIZER)
      {
        vectorizer_type = vectorizer_types[vectorizer->get_type()];
      } 
      std::cout<<"[+] vectorizing normal with "<<vectorizer_type<<std::endl;
      vectorizer->add_document(fuzz_string);
      std::cout<<"[+] vectorizing normal  with "<<vectorizer_type<<" done"<<std::endl;
      std::cout<<"[+] finished "<<fuzz_file<<std::to_string(cur)+"/"+std::to_string(total)<<std::endl;

    }
    
  }



  std::cout<<"[+] finished vectorizing "<<std::endl;
  c_end = clock ();
  t_end = std::chrono::high_resolution_clock::now ();
  std::cout << "[+] Test CPU time elapsed in s: "
    << (double) (c_end - c_point) / CLOCKS_PER_SEC << std::endl;
  std::cout << "[+] Test Wall time elapsed in s: "
    << std::chrono::duration<double> (t_end - t_point).count ()
    << std::endl;
  std::cout<<"[+] fitting vectorizers "<<std::endl;

  //now that we have the vectorizers, we can train the autoencoders
  
  for ( auto& v : vectorizers_attacks) 
  {
      std::string vectorizer_type = vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER];//= vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER];//unknown vectorizer->get_type();
      c_start = clock ();
      t_start = std::chrono::high_resolution_clock::now ();

      if ( v->get_type() < provallo::vectorizer_type::UNKNOWN_VECTORIZER)
      {
        vectorizer_type = vectorizer_types[v->get_type()];
      } 
      std::cout<<"[+] fitting vectorizer "<<vectorizer_type<<std::endl;
      v->process_documents();
      std::cout<<"[+] fitting vectorizer "<<vectorizer_type<<" done"<<std::endl;
      
      c_end = clock ();
      t_end = std::chrono::high_resolution_clock::now ();

      std::cout << "[+] vectorizer CPU time elapsed in s: "
        << (double) (c_end - c_start) / CLOCKS_PER_SEC << std::endl;
      std::cout << "[+] vectorizer Wall time elapsed in s: "
        << std::chrono::duration<double> (t_end - t_start).count ()
        << std::endl;
      //print output size for each vectorizer :
      std::cout<<"[+] vectorizer expected output size : "<<std::to_string(v->get_output_size())<<std::endl;  
  }
 
  for ( auto& v : vectorizers_normal) 
  {
      std::string vectorizer_type = vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER];//= vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER];//unknown vectorizer->get_type();
      c_start = clock ();
      t_start = std::chrono::high_resolution_clock::now ();

      if ( v->get_type() < provallo::vectorizer_type::UNKNOWN_VECTORIZER)
      {
        vectorizer_type = vectorizer_types[v->get_type()];
      } 
      std::cout<<"[+] fitting normal vectorizer "<<vectorizer_type<<std::endl;
      v->process_documents();
      std::cout<<"[+] fitting normal  vectorizer "<<vectorizer_type<<" done"<<std::endl;
      
      c_end = clock ();
      t_end = std::chrono::high_resolution_clock::now ();

      std::cout << "[+] vectorizer CPU time elapsed in s: "
        << (double) (c_end - c_start) / CLOCKS_PER_SEC << std::endl;
      std::cout << "[+] vectorizer Wall time elapsed in s: "
        << std::chrono::duration<double> (t_end - t_start).count ()
        << std::endl;
      //print output size for each vectorizer :
      std::cout<<"[+] vectorizer expected output size : "<<std::to_string(v->get_output_size())<<std::endl;  
  }
 
  //now that we have the autoencoders, we can train them
  //reiterate on the files and train the autoencoders 
  //print total time 
  c_end = clock ();
  t_end = std::chrono::high_resolution_clock::now ();
  std::cout << "[+] Test CPU time elapsed in s: "
    << (double) (c_end - c_point) / CLOCKS_PER_SEC << std::endl;  
  std::cout << "[+] Test Wall time elapsed in s: "
    << std::chrono::duration<double> (t_end - t_point).count ()
    << std::endl; 
    //save vectorizers
    size_t vectr=1;
    for (auto & vect : vectorizers_attacks )    
    {
      //print vectorizer type
      std::string vectorizer_type = vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER];//unknown vectorizer->get_type();
      if ( vect->get_type() < provallo::vectorizer_type::UNKNOWN_VECTORIZER)
      {
        vectorizer_type = vectorizer_types[vect->get_type()];
      }
      std::cout<<"[+] saving attack vectorizer "<<vectorizer_type<<std::endl;  
      std::ofstream vectorizer_out("vectorizer_fuzzdb_attacks"+vectorizer_type+std::to_string(vectr)+".json");

      vect->save(vectorizer_out);
      vect->gnuplot("vectorizer_fuzzdb_attacks"+vectorizer_type+std::to_string(vectr)+".gnuplot");

      vectr++;

    }
    for (auto & vect : vectorizers_normal )    
    {
      //print vectorizer type
      std::string vectorizer_type = vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER];//unknown vectorizer->get_type();
      if ( vect->get_type() < provallo::vectorizer_type::UNKNOWN_VECTORIZER)
      {
        vectorizer_type = vectorizer_types[vect->get_type()];
      }
      std::cout<<"[+] saving benign vectorizer "<<vectorizer_type<<std::endl;  
      std::ofstream vectorizer_out("vectorizer_fuzzdb_benign"+vectorizer_type+std::to_string(vectr)+".json");

      vect->save(vectorizer_out);
      vect->gnuplot("vectorizer_fuzzdb_benign"+vectorizer_type+std::to_string(vectr)+".gnuplot");
      vectr++;
    }

  std::cout<<"[+] creating autoencoders "<<std::endl;
  for ( auto & vectorizer : vectorizers_attacks)
  {
    //get random file from normal files
    std::vector<real_t> input;
    size_t input_size=0;
    for(;;)
    {
    std::string random_file = string_files_attacks[rand()%string_files_attacks.size()]; 
  
    std::cout<<"[+] loading random attack file : "<<random_file<<std::endl;
      

    std::ifstream ifrandom(random_file);
       std::string fit_file((std::istreambuf_iterator<char>(ifrandom)),
    std::istreambuf_iterator<char>());
 
    

      //print vectorizer type
      std::string vectorizer_type = vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER];//unknown vectorizer->get_type(); 
      if ( vectorizer->get_type() < provallo::vectorizer_type::UNKNOWN_VECTORIZER)
      {
        vectorizer_type = vectorizer_types[vectorizer->get_type()];
      } 
  
   
    std::cout<<"[+] creating autoencoder for vectorizer:"<< vectorizer_types[vectorizer->get_type()] << " , input size:"<<std::to_string(input_size)<<std::endl;

    c_start = clock ();
    t_start = std::chrono::high_resolution_clock::now ();
      input = vectorizer->predict(fit_file);
      input_size = input.size();
      if( input_size==0)//< vectorizer->get_output_size()  )
    {
      input_size =sqrt(vectorizer->get_output_size());

      for(size_t i=input.size();i<input_size;i++)
      {
        input.push_back(real_t(i)/vectorizer->get_output_size());
      }
    } 
    if(input_size<=1){ 
      std::cout<<"[+] skipping autoencoder creation for vectorizer:"<< vectorizer_types[vectorizer->get_type()] << " , input size:"<<std::to_string(input_size)<<std::endl;
      continue; }
    //push softmax classifiers as autoencoders
    provallo::auto_encoder<real_t,real_t>* ae = new provallo::softmax_classifier<real_t,real_t>(2,input_size,/*alpha*/0.1,/*lambda*/0.05);//input,hidden,output 
    autoencoders.push_back(ae);

    //autoencoders.push_back(new provallo::auto_encoder<real_t,real_t>(input_size,input_size*(2*std::log2(input_size)),1));//input,hidden,output   
    //c_end = clock ();
    c_end = clock ();
    t_end = std::chrono::high_resolution_clock::now ();
    std::cout << "[+] auto encoder init Test CPU time elapsed in s: " << (double) (c_end - c_start) / CLOCKS_PER_SEC << std::endl;    
 
    std::cout << "[+] auto encoder init Test Wall time elapsed in s: " << std::chrono::duration<double> (t_end - t_start).count ()<< std::endl; 
    std::cout<<"[+] autoencoder created "<<std::endl;
    break;
    } //end for vectorizers

  }//end for vectorizers
  //now that we have the autoencoders, we can train them 
  for (auto & fuzz_file : string_files_attacks )
  {
    std::ifstream fuzz(fuzz_file);
    std::string data_string((std::istreambuf_iterator<char>(fuzz)),std::istreambuf_iterator<char>());
    std::cout<<"[+] fuzzing "<<fuzz_file<<std::endl;
    size_t vectr=0;
    for (auto & enc : autoencoders   )
    {
      auto & vectorizer = vectorizers_attacks[vectr++];
       //print vectorizer type
      std::string vectorizer_type = vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER];//unknown vectorizer->get_type(); 
      if ( vectorizer->get_type() < provallo::vectorizer_type::UNKNOWN_VECTORIZER)
      {
        vectorizer_type = vectorizer_types[vectorizer->get_type()];
      }
      std::cout<<"[+] vectorizing with "<<vectorizer_type<<std::endl;
      std::vector<real_t> input =vectorizer->predict(data_string);
      std::cout<<"[+] vectorizing with "<<vectorizer_type<<" done"<<std::endl;
      std::cout<<"[+] training / loading autoencoder"<<std::endl;
      //print vectorize output
      if(input.size()<2)
      {
        std::cout<<"[+] vectorizer output size : "<<std::to_string(input.size())<<"/"<<std::to_string(vectorizer->get_output_size())<<std::endl; 
        size_t old_size = input.size();
        input.resize(enc->getInputDim());
        for(size_t i=old_size;i<input.size();i++)
        {
          input[i] = i/enc->getInputDim();
        } 

      } 

      std::cout<<"[+] vectorizer output size : "<<std::to_string(input.size())<<std::endl; 
      std::cout<<"[+] autoencoder input size : "<<std::to_string(enc->getInputDim())<<std::endl;
     
      //attack output
     
      double output[2]={1,1};
      //do
      //{
      //get dictionary and weight and train the autoencoder 
      //train with vectorized data:
      auto t =input.size()/enc->getInputDim();
      for ( size_t inputs = 0 ; inputs < enc->getInputDim()*t ; inputs+=enc->getInputDim() )
      {
        enc->train((real_t*)input.data()+inputs,output,1); 
        std::cout<<"[+] train output  : "<<std::to_string(output[0])<<std::endl;
        
       }
      
      //test with vectorized data  
      
       
//      }while(output[0]+output[1]<0.99);
      std::cout<<"[+] training autoencoder done"<<std::endl;

      //evaluate the autoencoder
      std::vector<real_t> output_vector(output,output+2);
      std::cout<<"[+] evaluating autoencoder"<<std::endl;
      enc->test(input.data(),output,1);

      std::cout<<"[+] evaluating autoencoder done"<<std::endl;
      std::cout<<"[+] output[0] : "<<std::to_string(output[0])<<std::endl;
      //std::cout<<"[+] loss = "<<std::to_string(enc->get_loss())<<std::endl; 

      std::cout<<"[+] training autoencoder done"<<std::endl;
      std::cout<<"[+] training autoencoder done"<<std::endl;
      enc->save("encoder_fuzzdb_"+vectorizer_type+".json");
      //enc->load ("encoder_fuzzdb_"+vectorizer_type+".json");
     }//end for autoencoders
  
    c_start = clock ();
    t_start = std::chrono::high_resolution_clock::now ();
    std::cout << "[+] Test CPU time elapsed in s: "
    << (double) (c_end - c_point) / CLOCKS_PER_SEC << std::endl;
    std::cout << "[+] Test Wall time elapsed in s: "
    << std::chrono::duration<double> (t_end - t_point).count ()
    << std::endl;
    std::cout<<"[+] finished "<<fuzz_file<<std::endl;

  } //end for fuzz files
  
  //let's try without retraining softmax classfiers since the autoencoders are already trained 
  //create softmax classifier
  //size_t n_classes =2 ,n_dimensions  = sqrt(autoencoders[0]->getInputDim()); 
  //real_t alpha = 1. , lambda = 0.15;

  provallo::softmax_classifier<real_t,real_t> softmax(*(autoencoders[0])); //(n_dimensions,n_classes,alpha,lambda);

  auto& selected_vectorizer = vectorizers_attacks[0];
  auto& selected_normal_vectorizer = vectorizers_normal[0];

  std::cout<<"[+] attack vectorizer output size = "<<std::to_string(selected_vectorizer->get_output_size())<<std::endl; 
  std::cout<<"[+] selected normal vectorizer output size = "<<std::to_string(selected_normal_vectorizer->get_output_size())<<std::endl;
  char x=std::getchar();


  //create softmax classifier
  //size_t n_classes =2 ,n_dimensions  = sqrt(selected_vectorizer->get_output_size());
  //real_t alpha = 1. , lambda = 0.15;

  //provallo::softmax_classifier<real_t,real_t> softmax(n_dimensions,n_classes,alpha,lambda);
  //train softmax classifier
  std::cout<<"[+] training softmax classifier "<<std::endl;
  size_t n_classes = 2;
  provallo::matrix<real_t> out_mat(1,n_classes ); 
  size_t total_cases = 0; // string_files_attacks.size()+string_files_normal.size(); 
  size_t correct_classifications = 0;
  size_t error_classifications = 0;
  size_t true_positives = 0;
  size_t true_negatives = 0;
  size_t false_positives = 0;
  size_t false_negatives = 0;


  //first train with normal data
  for (auto & fuzz_file : string_files_normal )
  {
    try {
    std::ifstream fuzz(fuzz_file);
  
    if(!fuzz.is_open()|| !fuzz.good())
    {
      std::cout<<"[+] error opening file "<<fuzz_file<<std::endl;
      continue;
    } else {

      
    std::string data_string((std::istreambuf_iterator<char>(fuzz)),std::istreambuf_iterator<char>());
    std::cout<<"[+] fuzzing "<<fuzz_file<<std::endl;
    std::cout<<"[+] vectorizing with "<<vectorizer_types[selected_normal_vectorizer->get_type()]<<std::endl;
    std::vector<real_t> input = selected_normal_vectorizer->predict(data_string);
    std::cout<<"[+] vectorizing with "<<vectorizer_types[selected_normal_vectorizer->get_type()]<<" done"<<std::endl;
    std::cout<<"[+] training softmax classifier"<<std::endl;
    if(input.size()<softmax.getInputDim() )
    {
      std::cout<<"[+] vectorizer output size : "<<std::to_string(input.size())<<"/"<<std::to_string(selected_normal_vectorizer->get_output_size())<<std::endl; 
      size_t old_size = input.size();
      input.resize(softmax.getInputDim());
      for(size_t i=old_size;i<input.size();i++)
      {
           input[i] = i/real_t(softmax.getInputDim());
        
      } 
    
    } 
    out_mat.resize(1,n_classes);
    out_mat.fill(0.0);

    provallo::matrix<real_t> input_mat = provallo::matrix<real_t>(input.data(),1,input.size()); 
      softmax.train(input,out_mat);
      std::cout<<"[+] training softmax benign case done"<<std::endl;
      std::cout<<"[+] output = "<<out_mat<<std::endl;
       //update confusion 
    
    for ( size_t i=0;i<out_mat.cols();i++)  {
    for (size_t j=0;j<out_mat.rows();j++) 
    {
     total_cases++;
      if (out_mat(j,i) <0.5 )
      {
        //true positive
         true_positives++;
         correct_classifications++;
        std::cout<<"[+] TP benign case detected"<<std::endl;
      }
      else if (out_mat(j,i)==0.5)
      {
        //error
        error_classifications++;
        true_negatives++;
        std::cout<<"[+] TN error : softmax output = 0.5"<<std::endl;
      }
      else if (out_mat(j,i) > 0.5 )
      {
        //false positive
        false_positives++;
        error_classifications++;
        std::cout<<"[+] FP attack case detected"<<std::endl;
      }     //else
      else 
      {
        //false negative
        false_negatives++;
        error_classifications++;
        std::cout<<"[+] FN attack case detected"<<std::endl;
      }//else
     } //end for softmax output
    } //end for softmax output
    }//and else
    //end try
    }catch(std::exception& e)
    {
      std::cout<<"[+] exception : "<<e.what()<<std::endl; 
    }
    

  } //end for normal files
   

  std::cout<<"[+] training softmax classifier on normal cases done"<<std::endl;
  //print confusion matrix
  std::cout<<"[+] softmax training : " <<std::endl;
  std::cout<<"[+] total training samples:"<<std::to_string(total_cases)<<std::endl;
  std::cout<<"[+] false positives : "<<std::to_string(false_positives)<<std::endl;
  std::cout<<"[+] false negatives : "<<std::to_string(false_negatives)<<std::endl;
  std::cout<<"[+] true positives : "<<std::to_string(true_positives)<<std::endl;
  std::cout<<"[+] true negatives : "<<std::to_string(true_negatives)<<std::endl;
  std::cout<<"[+] correct classifications : "<<std::to_string(correct_classifications)<<std::endl;
  std::cout<<"[+] error classifications : "<<std::to_string(error_classifications)<<std::endl;
  std::cout<<"[+] softmax benign training done"<<std::endl;

  x+= std::getchar();
  x++;
  //now do the same and train with the attack data a few times. 
  for(size_t i=0;i<5;i++)
  for(auto & fuzz_file : string_files_attacks)
  {
    try {

    std::ifstream fuzz(fuzz_file);

    if(!fuzz.is_open()|| !fuzz.good())
    {
      std::cout<<"[+] error opening file "<<fuzz_file<<std::endl;
      continue;
    }
    //when out_mat is allocated to 1,n_classes no allocation is done
    out_mat = (provallo::matrix<real_t>::One(1,n_classes)); 

    std::string data_string((std::istreambuf_iterator<char>(fuzz)),std::istreambuf_iterator<char>());
    std::cout<<"[+] fuzzing "<<fuzz_file<<std::endl;
    std::cout<<"[+] vectorizing with "<<vectorizer_types[selected_vectorizer->get_type()]<<std::endl;
    std::vector<real_t> input = selected_vectorizer->predict(data_string);
    std::cout<<"[+] vectorizing with "<<vectorizer_types[selected_vectorizer->get_type()]<<" done with size:"<<std::to_string(input.size())<<std::endl;
    std::cout<<"[+] training softmax classifier"<<std::endl;
    if(input.size()<softmax.getInputDim() )
    {
      std::cout<<"[+] vectorizer output size : "<<std::to_string(input.size())<<"/"<<std::to_string(selected_vectorizer->get_output_size())<<std::endl; 
      size_t old_size = input.size();
      input.resize( softmax.getInputDim());
      for(size_t i=old_size;i<input.size();i++)
      {
        input[i] = i/real_t(softmax.getInputDim());
      } 

    } 
    out_mat.resize(1,n_classes);
    out_mat.fill(1.0);


       provallo::matrix<real_t> input_mat = provallo::matrix<real_t>(input.data(),1,input.size()); 
      softmax.train(input_mat,out_mat);
      std::cout<<"[+] training softmax attack cases done"<<std::endl;
      std::cout<<"[+] output = "<<out_mat<<std::endl;

      for ( size_t i=0;i<out_mat.cols();i++)  {
        for ( size_t j=0;j<out_mat.rows();++j)
        {
      //update confusion matrix
      total_cases++;
          if( out_mat(i,j) < 0.5 )
          {
            //true negative
            false_positives++;
            error_classifications++;
            std::cout<<"[+] benign case detected"<<std::endl;
          }
          else if ( out_mat(i,j) > 0.5 )
          {
            //false positive
            true_positives++;
            correct_classifications++;
            std::cout<<"[+] attack case detected"<<std::endl;
          }     
          else if ( out_mat(i,j) == 0.5 )
          {
            error_classifications++;
            true_negatives++;
            std::cout<<"[+] error : softmax output = 0.5"<<std::endl;
          }
          else 
          {
            false_negatives++;
            error_classifications++;
            std::cout<<"[+] error : softmax output = "<<std::to_string(out_mat(i,j))<<std::endl;
          }
          
      }//end for softmax output
      }//end for softmax output
    }//end try
    catch(std::exception& e)
    {
      std::cout<<"[+] exception : "<<e.what()<<std::endl;   
    } 
    }//end for attack files
    std::cout<<"[+] training softmax classifier done"<<std::endl;
    //print confusion matrix 
    std::cout<<"[+] softmax training : " <<std::endl;
    std::cout<<"[+] false positives : "<<std::to_string(false_positives)<<std::endl; 
    std::cout<<"[+] false negatives : "<<std::to_string(false_negatives)<<std::endl;
    std::cout<<"[+] true positives : "<<std::to_string(true_positives)<<std::endl;
    std::cout<<"[+] true negatives : "<<std::to_string(true_negatives)<<std::endl;
    std::cout<<"[+] correct classifications : "<<std::to_string(correct_classifications)<<std::endl;
    std::cout<<"[+] error classifications : "<<std::to_string(error_classifications)<<std::endl;
    std::cout<<"[+] softmax training done"<<std::endl;
    //save softmax classifier
    std::cout<<"[+] saving softmax classifier"<<std::endl;
    softmax.save("softmax_fuzzdb_attacks_trained.json");

  //now test the softmax classifier
  std::cout<<"[+] testing softmax classifier  "<<std::endl;
 
  //reset for test
   total_cases = 0;
   correct_classifications = 0;
   error_classifications = 0;
   true_positives = 0;
   true_negatives = 0;
   false_positives = 0;
   false_negatives = 0;

    for (auto & fuzz_file : string_files_normal )
    {
    std::ifstream fuzz(fuzz_file);
    provallo::matrix<real_t> out_mat(provallo::matrix<real_t>::Zero(1,n_classes));
    std::string data_string((std::istreambuf_iterator<char>(fuzz)),std::istreambuf_iterator<char>());
    std::cout<<"[+] fuzzing "<<fuzz_file<<std::endl;
    std::cout<<"[+] vectorizing with "<<vectorizer_types[selected_normal_vectorizer->get_type()]<<std::endl;
    std::vector<real_t> input = selected_normal_vectorizer->predict(data_string);
    std::cout<<"[+] vectorizing with "<<vectorizer_types[selected_normal_vectorizer->get_type()]<<" done"<<std::endl;
    std::cout<<"[+] testing softmax classifier"<<std::endl;
    if(input.size()<softmax.getInputDim() )
    {
      std::cout<<"[+] vectorizer output size : "<<std::to_string(input.size())<<"/"<<std::to_string(selected_normal_vectorizer->get_output_size())<<std::endl;
      size_t old_size = input.size();
      input.resize(sqrt(sqrt(selected_normal_vectorizer->get_output_size() )));
      for(size_t i=old_size;i<input.size();i++)
      {
        input[i] = 0.0;
      }
      
    }
    provallo::matrix<real_t> input_mat = provallo::matrix<real_t>(input.data(),1,input.size());
    softmax.predict(input_mat,out_mat);
    std::cout<<"[+] testing softmax benign case done"<<std::endl;
    std::cout<<"[+] output = "<<out_mat<<std::endl;
    //update confusion matrix
    for ( size_t i=0;i<out_mat.cols();i++)  {
      for ( size_t j=0;j<out_mat.rows();++j) {
         total_cases++;  
        if( out_mat(j,i) < 0.5 )
        {
          //true negative
          true_negatives++;
          correct_classifications++;
          std::cout<<"[+] benign case detected"<<std::endl;
        }
        else if (out_mat(j,i) > 0.5)
        {
          //false positive
          false_positives++;
          error_classifications++;
          std::cout<<"[+] attack case detected"<<std::endl;
        }
        else if (out_mat(j,i)==0.5)
        {
          //error
          error_classifications++;
          true_positives++;
          std::cout<<"[+] error : softmax output = 0.5"<<std::endl;
        }
        else 
        {
          //error
          error_classifications++;
          false_negatives++;
          std::cout<<"[+] error : softmax output = "<<std::to_string(out_mat(j,i))<<std::endl;    
        }
        
    }//end for softmax output
    }//end for softmax output
    } //end for normal files

    std::cout<<"[+] testing softmax classifier done"<<std::endl;

    //test with attack data
    for (auto & fuzz_file : string_files_attacks )
    {
      std::ifstream fuzz(fuzz_file);
      provallo::matrix<real_t> out_mat(provallo::matrix<real_t>::Zero(1,n_classes));
      std::string data_string((std::istreambuf_iterator<char>(fuzz)),std::istreambuf_iterator<char>());
      std::cout<<"[+] fuzzing "<<fuzz_file<<std::endl;
      std::cout<<"[+] vectorizing with "<<vectorizer_types[selected_vectorizer->get_type()]<<std::endl;
      std::vector<real_t> input = selected_vectorizer->predict(data_string);
      std::cout<<"[+] vectorizing with "<<vectorizer_types[selected_vectorizer->get_type()]<<" done"<<std::endl;
      std::cout<<"[+] testing softmax classifier"<<std::endl;
      if(input.size()<softmax.getInputDim() )
      {
        std::cout<<"[+] vectorizer output size : "<<std::to_string(input.size())<<"/"<<std::to_string(selected_normal_vectorizer->get_output_size())<<std::endl;
        size_t old_size = input.size();
        input.resize(sqrt(sqrt(selected_vectorizer->get_output_size() )));
        for(size_t i=old_size;i<input.size();i++)
        {
          input[i] =  0.0;
        }

      }
      provallo::matrix<real_t> input_mat = provallo::matrix<real_t>(input.data(),1,input.size());
      softmax.predict(input_mat,out_mat);
      std::cout<<"[+] testing softmax attack case done"<<std::endl;
      std::cout<<"[+] output = "<<out_mat<<std::endl;
      //update confusion matrix
      for ( size_t i=0;i<out_mat.cols();i++)  {
        for ( size_t j=0;j<out_mat.rows();++j) {
           total_cases++;  
          if( out_mat(j,i) < 0.5 )
          {
            //true negative
            false_positives++;
            error_classifications++;
            std::cout<<"[+] benign case detected"<<std::endl;
          }
          else if (out_mat(j,i) > 0.5)
          {
            //false positive
            true_positives++;
            correct_classifications++;
            std::cout<<"[+] attack case detected"<<std::endl;
          }
          else if (out_mat(j,i)==0.5)
          {
            //error
            error_classifications++;
            true_negatives++;
            std::cout<<"[+] error : softmax output = 0.5"<<std::endl;
          }
          else 
          {
            //error
            error_classifications++;
            false_negatives++;
            std::cout<<"[+] error : softmax output = "<<std::to_string(out_mat(j,i))<<std::endl;    
          }
          
      }//end for softmax output
      }//end for softmax output   


      }  //end for attack files
      std::cout<<"[+] testing softmax classifier done"<<std::endl;
      std::cout<<"[+] softmax test results: " <<std::endl;
      std::cout<<"[+] total test samples:"<<std::to_string(total_cases)<<std::endl;
      std::cout<<"[+] false positives : "<<std::to_string(false_positives )<<std::endl; 
      std::cout<<"[+] false negatives : "<<std::to_string(false_negatives )<<std::endl;
      std::cout<<"[+] true positives : "<<std::to_string(true_positives)<<std::endl;
      std::cout<<"[+] true negatives : "<<std::to_string(true_negatives )<<std::endl;
      std::cout<<"[+] correct classifications : "<<std::to_string(correct_classifications)<<std::endl;
      std::cout<<"[+] error classifications : "<<std::to_string(error_classifications)<<std::endl;
      std::cout<<"[+] softmax training done"<<std::endl;

      //save softmax classifier
      //save softmax classifier --> softmax_softmax_fuzzdb_test.json 
      softmax.save("softmax_fuzzdb_test.json");
      std::cout<<"[+] saving softmax classifier done"<<std::endl;

      //save autoencoders
      size_t enc_index=1;
      for (auto & enc : autoencoders   )
      {
          std::string file_name = "encoder_fuzzdb" +std::to_string(enc_index++) + ".json"; 
          enc->save("encoder_fuzzdb.json");
        //  delete enc;
      }
        //delete vectorizers : 
        //delete autoencoders

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
  
 }//  end of fit_fuzzdb
 //-----------------------------------------------------------------------------


 