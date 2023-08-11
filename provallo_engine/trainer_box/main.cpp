#include "../decision_engine/split_utils.hpp"
#include "../decision_engine/classifier.h"
#include "../decision_engine/dataset.h" 
#include "../decision_engine/neuralhelper.h"
#include "../decision_engine/kNearestNeighbor.h"
#include "../decision_engine/kdt.h"
#include "../decision_engine/parameters.h"
#include "../decision_engine/pipelinebuilder.h"
#include "../decision_engine/autoencoder.h"
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

bool test_fit_iso_forest();
bool test_dataset_load();

provallo::isolation_forest* isoforest_single(const provallo::attribute_information& attributes);

bool benchmark_classifiers (const std::string benchmark_folder = "./db/benchmarks");
bool fit_fuzzsb_folder(const std::string benchmark_folder = "/home/kardon/eclipse-workspace/fuzzdb/fuzzdb/attack/");


 
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
  std::cout << "Provallo IsoForest test" << std::endl;
  std::cout << std::string ("Point with highest outlier score: [")
      << X[get_ix(row_highest, 0)] << std::string (", ")
      << X[get_ix(row_highest, 1)] << std::string ("]") << std::endl;

  return true;
}

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
 
        const provallo::attribute_information& attributes = collector.getAttributes ();

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
          provallo::testing_set testset(attributes); 
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
          }


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

bool test_vectorizers ( provallo::dataset& vectorize_set   ) 
{
  //normalize the attributes 

  UNDEF_REFERENCE(vectorize_set);
  UNDEF_REFERENCE2(vectorize_set);

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
  vectorizer1.fit(vectorize_set_data);

  for ( size_t i=0;i<vectorize_set_data.size();i++)
  {
    size_t j=0;
    std::cout << "Testing vectorizer 1 transform" << std::endl;
    std::vector<double> vectorized_data1 = vectorizer1.transform(vectorize_set_data[i]);
    for(auto f : vectorized_data1)
    {
      std::cout << " [+] fit :  "<<std::to_string(f) << " "<<vectorize_set_data[i][j++]<<std::endl;

      ret = true;
    } 
    std::cout << std::endl;
  }

  char x = std::getchar();
  
   UNDEF_REFERENCE2(x);


  return ret;

}

bool fit_fuzzsb_folder ( const std::string& fit_fuzzsb_folder)
{
  //get the files in the folder
  //iterate over them and 
  // 1) load the string file 
  // 2) vectorize it
  // 3) train autoencoder to select the best features
  // 4) create a matrix of the best features and save it to disk
  // 5) create a dataset from the matrix and save it to disk
  // 6) create a classifier and train it
  // 7) save the classifier to disk
  // 8) test the classifier
  // 9) save the results to disk
  // 10) repeat for all the files in the folder
  // 11) save the results to disk
  // 12) return true if all the tests are passed, false otherwise

  bool ret = false;
  std::vector<std::string> files = getFilesInFolder(fit_fuzzsb_folder);
  std::vector<std::string> string_files;
  for (auto file : files)
  {
    if (file.find(".txt") != std::string::npos)
    {
      string_files.push_back(file);
    }
  } 
  //iterate over the txt files,
  // load them, vectorize them, train autoencoder,
  // create matrix, create dataset, create classifier, train classifier, test classifier, save results to disk 
  // repeat for all the files in the folder
  //autoencoder has 2 outputs, one for the features, one for the classes 
  //autoencoder has multiple inputs for the features, one input for the classes
  //the features are used to train the classifier, the classes are used to test the classifier
  //the classifier is trained on the features, and tested on the classes
  //the results are saved to disk
  //the results are compared to the expected results
  //the results are saved to disk

  //load or generate a pipeline: 
  // 1) load the pipeline from disk
   //provallo::meta_builder meta("fuzzdb_meta",fit_fuzzsb_folder+"/fuzzdb_meta.json");
  // 
  // 2) if the pipeline is not found, create a new one
    provallo::pipeline_builder builder("fuzzdb_meta",fit_fuzzsb_folder+"/fuzzdb_meta.json"); 
  
    if(builder.get_number_of_stages()<2 ||builder.get_number_of_pipelines()<1)
    {
        builder.add_pipeline("fuzzdb_pipeline",false);
        provallo::pipeline* new_pipeline =  builder.get_pipeline("fuzzdb_pipeline");
        // data set load may or may not parse columns or split the data set into train and test
        // if the data set is not split into train and test, the pipeline will do it on the training load 
        // if the stages contain a classifier, the pipeline will train the classifier on the training data set 
        if(new_pipeline!=nullptr)
        {

          //let's make the pipeline a pipeline of pipelines
          //get each pipeline by name from the new pipeline to set the stages
          new_pipeline->add_stage("pipeline","data_pipeline");
          new_pipeline->add_stage("pipeline","feature_pipeline");
          new_pipeline->add_stage("pipeline","autoencoder_pipeline");
          new_pipeline->add_stage("pipeline","classifier_pipeline");
          new_pipeline->add_stage("pipeline","train_pipeline");
          new_pipeline->add_stage("pipeline","evaluation_pipeline");
          new_pipeline->add_stage("pipeline","test_pipeline");
          new_pipeline->add_stage("pipeline","save_pipeline");
          new_pipeline->add_stage("pipeline","load_pipeline");
          new_pipeline->add_stage("pipeline","compare_pipeline");
          new_pipeline->add_stage("pipeline","save_results_pipeline");

          provallo::pipeline* data_pipeline =  new_pipeline->get_pipeline("data_pipeline");
          provallo::pipeline* feature_pipeline =  new_pipeline->get_pipeline("feature_pipeline");
          provallo::pipeline* autoencoder_pipeline =  new_pipeline->get_pipeline("autoencoder_pipeline");
          provallo::pipeline* classifier_pipeline =  new_pipeline->get_pipeline("classifier_pipeline");
          provallo::pipeline* train_pipeline =  new_pipeline->get_pipeline("train_pipeline");
          provallo::pipeline* evaluation_pipeline =  new_pipeline->get_pipeline("evaluation_pipeline");
          provallo::pipeline* test_pipeline =  new_pipeline->get_pipeline("test_pipeline");
          provallo::pipeline* save_pipeline =  new_pipeline->get_pipeline("save_pipeline");
          provallo::pipeline* load_pipeline =  new_pipeline->get_pipeline("load_pipeline");
          provallo::pipeline* compare_pipeline =  new_pipeline->get_pipeline("compare_pipeline");
          provallo::pipeline* save_results_pipeline =  new_pipeline->get_pipeline("save_results_pipeline");
            
            //push stages into pipelines : 
            data_pipeline->add_stage("dataset","load_train_dataset");
            data_pipeline->add_stage("dataset","load_test_dataset");
            feature_pipeline->add_stage("feature_engineering","tfidf_vectorizer");
            feature_pipeline->add_stage("feature_engineering","bow_vectorizer");
            feature_pipeline->add_stage("feature_engineering","pca_vectorizer");
            feature_pipeline->add_stage("feature_engineering","autoencoder");
            feature_pipeline->add_stage("feature_engineering","select_k_best");
            feature_pipeline->add_stage("feature_engineering","tsne");
            feature_pipeline->add_stage("feature_engineering","pca");
            feature_pipeline->add_stage("feature_engineering","ica");
            feature_pipeline->add_stage("feature_engineering","nmf");
            feature_pipeline->add_stage("feature_engineering","lda");
            autoencoder_pipeline->add_stage("autoencoder","autoencoder");
            classifier_pipeline->add_stage("classifier","classifier");
            train_pipeline->add_stage("train","train");
            evaluation_pipeline->add_stage("evaluation","roc_auc");
            evaluation_pipeline->add_stage("evaluation","pr_auc");

            test_pipeline->add_stage("test","test");
            save_pipeline->add_stage("save","save");
            load_pipeline->add_stage("load","load");
            compare_pipeline->add_stage("compare","compare");
            save_results_pipeline->add_stage("save_results","save_results");
            
    }
    else  //if the pipeline is found, load it
    {
      //build the pipeline from disk
      builder.build();


    }
    //iterate all the files in the folder, if it's a txt file, vectorize it and add to the dataset matrix 
    std::vector<std::string> files = getFilesInFolder(fit_fuzzsb_folder);
    std::vector<std::string> txt_files;

    for (auto file : files)
    {
      if (file.find(".txt") != std::string::npos)
      {
        txt_files.push_back(file);

        std::cout<<"-- found txt file : "<<file<<std::endl;

      }
    }
    //TODO: add the files to the dataset
    //TODO: vectorize the documents
    //extract number of features from the pipeline
    //extract number of classes from the pipeline
    //activate the pipeline 


     
      ret = true;
  }//if 
  else
  {
    std::cout<<"-- benchmark folder not found : "<<fit_fuzzsb_folder<<std::endl;
  }
  return ret;
}


bool benchmark_classifiers (const std::string benchmark_folder )
{
    bool ret = false;
    bool buse_random_forest = true;
    bool test_ultra_fast_knn = false;
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
    bool sanity_check = false;
    for(auto description_file : description_files)
    { 
        std::string file_stem = description_file.substr(0,description_file.find(".names"));




        std::string data_file = benchmark_folder+"/"+file_stem+".data";
        std::string weights_file_name = benchmark_folder+"/"+file_stem+".weights";
        std::ifstream weights_ (weights_file_name);
        std::map<std::string,Float> weights;
        std::cout<<"-- building dataset for : "<<file_stem<<std::endl;
        std::vector<provallo::classifier*> classifiers ;
 
        provallo::files_collector collector = provallo::files_collector (benchmark_folder+"/"+file_stem);
 
        const provallo::attribute_information& attributes = collector.getAttributes ();
        provallo::auto_encoder<double,double> encoder( collector.getAttributes().getSize(),collector.getAttributes().getSize()*collector.getAttributes().getSize(),collector.getAttributes().getTargetClassCount() );



        //checking zero knowledge distributed kmeans 


        std::cout<<"-- Attributes information : "<<std::endl<<std::endl;
        std::cout<< description_file<<std::endl;
        std::cout<<collector.getAttributes ()<<std::endl;
        std::cout<<"-- checking for weights .... "<<std::endl<<std::endl;
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
        std::cout<<"-- training set size : "<<std::to_string(set.size() )<<std::endl;
        provallo::matrix<provallo::attribute> data = set.get_matrix();
        std::cout<<"-- training set matrix size : "<<std::to_string(data.size1()*data.size2())<<std::endl; 
        std::cout<<"-- training set matrix size1 : "<<std::to_string(data.size1())<<std::endl;
        //train encoders 
        provallo::class_dist ds(set.getattribute_info().getTargetClassCount());
        provallo::matrix<double> mdata(data.size1(),data.size2());
        for(size_t i=0;i<data.size1();i++)
        {
          for(size_t j=0;j<data.size2();j++)
          {
            mdata(i,j) = data(i,j).continous();
          }
        }

        encoder.train(mdata,ds);

        std::cout<<"-- encoder trained, classdist size : "<<std::to_string(ds.size())<<std::endl; 
        std::cout<<"-- classdist: "<<ds<<std::endl;

        encoder.save("encoder_"+file_stem+".json");
        encoder.load("encoder_"+file_stem+".json");
        std::cout<<"-- encoder loaded "<<std::endl;
        std::cout<<"-- encoder input size : "<<std::to_string(encoder.getInputDim())<<std::endl;
        std::cout<<"-- encoder output size : "<<std::to_string(encoder.getOutputDim())<<std::endl;
        std::cout<<"-- encoder hidden size : "<<std::to_string(encoder.getHiddenDim())<<std::endl;  
      
      if(!sanity_check )
      {
          std::cout<<"[+] sanity check for vectorizers"<<std::endl;
 
        if(  test_vectorizers(set) )
          {
             std::cout<<"[+] sanity check for vectorizers OK"<<std::endl;
             sanity_check = true;
          }
          else
          {
            std::cout<<"[+] sanity check for vectorizers FAILED"<<std::endl;
            exit(-1);
          }

      }
      typedef provallo::metric<provallo::Euclidean,provallo::Overlap> over_euc;
      typedef provallo::Kmeans<over_euc> km_classifier;
      typedef provallo::random_tree<provallo::GainRatio> egain_tree;
      typedef provallo::random_forest<egain_tree> rf_classifier;
      provallo::random_tree_param egain_param(.5,10,0.0) ; //rho,level,min-gain
      provallo::random_forest_param rf_param(50,egain_param); //tree-count,tree-param
      provallo::nearest_neighbor_param km_param(sqrt(attributes.getSize()),weights); 
      provallo::metric_classifier_param metric_param(sqrt(attributes.getSize()),weights); 
       
      provallo::adaboost_param boost_param(attributes.getTargetClassCount(),provallo::none());
      std::random_device rd;
      //std::cout <<"-- allocating factory .... "<<std::endl<<std::endl;
      //provallo::split_method_factory* factory = new provallo::split_method_factory(set,rd ) ; 
      //factory->set_override_split_method(provallo::split_type::CONE_RANDOM);
      provallo::split_method_factory* random_factory = new provallo::split_method_factory(set,rd);    
      random_factory->set_override_split_method(provallo::split_type::CONE_RANDOM);
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

      classifiers.push_back(new provallo::decision_tree<provallo::GainRatio> (set,provallo::none(),rd,factory ));
      classifiers.push_back(new provallo::decision_tree<provallo::EntropyGain> (set,provallo::none(),rd,factory ));
      classifiers.push_back(new provallo::decision_tree<provallo::ChiSquare> (set,provallo::none(),rd,factory ));


      std::cout<<"-- training UF classifiers  .... "<<std::endl<<std::endl; 
    

      for(auto & class_ : classifiers)
      { 
          print_classifier_summary(file_stem,set,*class_);
      }
     

      std::cout<<"-- press enter to continue .... "<<std::endl<<std::endl;
      std::getchar();
  
      std::cout<<"-- building neural network .... "<<std::endl<<std::endl;
      std::cout<<"-- training classifiers .... "<<std::endl<<std::endl;
      std::cout<<"-- testing classifiers .... "<<std::endl<<std::endl; 
      provallo::training_set test_set(attributes);
      collector.pushTestData (&test_set);


      //test autoencoder
      std::cout<<"-- testing autoencoder .... "<<std::endl<<std::endl;
      //go over the testing set and test the autoencoder
      size_t test_size = test_set.size()/test_set.getattributesNumber();
      mdata.resize(test_size,test_set.getattributesNumber());
      for(size_t i=0;i<test_size;i++)
      {
        for(size_t j=0;j<test_set.getattributesNumber();j++)
        {
          mdata(i,j) = test_set.getattribute(i,j).continous();
        }
      }
      encoder.test(mdata,ds);
      std::cout<<"-- autoencoder test finished "<<std::endl<<std::endl;
      std::cout<<"-- autoencoder test results : "<<ds<<std::endl<<std::endl;
      
      //print confusion matrix of test data
      
      for(auto & class_ : classifiers)
      {
           print_classifier_summary(file_stem,test_set,*class_);          
      }
      //test ultra fast knn : 
      if(test_ultra_fast_knn)  {
        try 
        {
        provallo::dataset_ptr dataset_ptr1( new provallo::dataset_base( set.copy_to_base()));
        provallo::kNN k(dataset_ptr1); 
        provallo::dataset_ptr   dataset_ptr2(new provallo::dataset_base(test_set.copy_to_base()));
        //normalize it :
         provallo::kNN_result  res =k.run(sqrt(attributes.getSize()),  dataset_ptr2 );
         std::cout<<"[+] ultra fast knn result : "<<std::endl;
        provallo::matrix_ptr confusion = res.getConfusionMatrix();
        //print confusion matrix :
        for(size_t i=0;i<confusion->rows();i++)
        { 
          for(size_t j=0;j<confusion->cols();j++)
          {
            if(i==j)
              std::cout<<"["<<std::to_string((*confusion)(i,j))<<"] ";
            else
            std::cout<<std::to_string((*confusion)(i,j))<<" ";
          }
          std::cout<<std::endl;
         }
          
        } //try
        catch ( std::exception& e) 
        {
          std::cerr<<"[-] error "<<e.what()<<std::endl;
        }
        catch ( ... ) 
        {
          std::cerr<<"[-] error "<<std::endl;

        }
      }//if _test_ultra_fast_knn

      std::cout<<"-- deleting classifiers .... "<<std::endl<<std::endl;
      for( auto class_ =  classifiers.begin(); class_ != classifiers.end(); ++class_ )
      {
        if(*class_)
        delete *class_;
      }
      classifiers.clear();

       std::cout<<"-- deleting weights .... "<<std::endl<<std::endl;
      weights.clear();
      std::cout<<"-- deleting weights file .... "<<std::endl<<std::endl;
      weights_.close();
      std::cout<<"-- deleting description file .... "<<std::endl<<std::endl;
      description_file.clear();
      ret = true;
      std::cout<<"-- deleting dataset .... "<<std::endl<<std::endl;
      if(dataset_ptr1)
      delete dataset_ptr1.get();

    

      std::cout<<"-- deleting random factory .... "<<std::endl<<std::endl;
      if(random_factory)
        delete random_factory;
      std::cout<<"-- deleting factory .... "<<std::endl<<std::endl;
      if(factory)
        delete factory;

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

