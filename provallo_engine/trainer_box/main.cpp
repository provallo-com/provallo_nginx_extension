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
  std::cout << "Provallo IsoForest test" << std::endl;
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
    //bool sanity_check = false;
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

        std::cout<<"-- attribute info : "<<std::endl<<std::endl;
        std::cout<< attributes <<std::endl<<std::endl;
        char x =      std::getchar();
      
        std::cout<< attributes <<std::endl<<std::endl;

       // x+= std::getchar();
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
        std::cout<<"-- attribute info : "<<std::endl<<std::endl;
        std::cout<<set.getattribute_info()<<std::endl<<std::endl;
        x+=      std::getchar();
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
        //encoder.load("encoder_"+file_stem+".json");
        std::cout<<"-- encoder loaded "<<std::endl;
        std::cout<<"-- encoder input size : "<<std::to_string(encoder.getInputDim())<<std::endl;
        std::cout<<"-- encoder output size : "<<std::to_string(encoder.getOutputDim())<<std::endl;
        std::cout<<"-- encoder hidden size : "<<std::to_string(encoder.getHiddenDim())<<std::endl;  
      
      std::cout<<"-- building classifiers "<<std::endl<<std::endl;
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
          std::cout<<set.getattribute_info()<<std::endl<<std::endl;

          print_classifier_summary(file_stem,set,*class_);
      }
      
      std::cout<<"-- attribute info : "<<std::endl<<std::endl;
      std::cout<<set.getattribute_info()<<std::endl<<std::endl;
      std::cout<<"-- press enter to continue .... "<<std::endl<<std::endl;
      x += std::getchar();
  
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
      std::cout<<"-- clearing training set .... "<<std::endl<<std::endl;

      set.clear();

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
  std::vector<std::string> string_files;
  
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
      string_files.push_back(file);

  } 
  std::cout<<"[+] found "<<string_files.size()<<" files in folder "<<fit_fuzzsb_folder<<std::endl;
  std::cout<<"[+] found "<<files.size()<<" files in folder "<<fit_fuzzsb_folder<<std::endl;
  
  //feed the files to the pipeline, 
  //the pipeline will load the files, vectorize them, train the autoencoder, 

  //first we want to verify the vectorizers and autoencoders work properly 
  vectorizers_attacks.push_back(new provallo::lda_vectorizer);
  vectorizers_attacks.push_back(new provallo::one_hot_vectorizer);
  
  vectorizers_normal.push_back(new provallo::lda_vectorizer);
  vectorizers_normal.push_back(new provallo::one_hot_vectorizer);
  //vectorizers_normal.push_back(new provallo::auto_encoder_vectorizer<real_t,real_t>(autoencoders)); 
  //create vectorizers:
//  vectorizers.push_back(new provallo::pca_vectorizer);
    //vectorizers.push_back(new provallo::word2vec_vectorizer);
  clock_t c_start, c_end;
  std::chrono::high_resolution_clock::time_point t_start, t_end;
  c_start = clock ();
  t_start = std::chrono::high_resolution_clock::now ();

  size_t total =string_files.size();
  size_t cur =  0;
  for (auto & fuzz_file : string_files )
  {
    std::ifstream fuzz(fuzz_file);
    std::string fuzz_string((std::istreambuf_iterator<char>(fuzz)),
    std::istreambuf_iterator<char>());
    bool is_attack = false;
    if(fuzz_file.find("attack")!=std::string::npos||fuzz_file.find("door")!=std::string::npos )
      is_attack = true;
    
   // std::cout<<"[+] fuzzing "<<fuzz_file<< "("<<std::to_string(cur)+"/"+std::to_string(total)<<")"<<std::endl;
    if ( is_attack ){
    for (auto & vectorizer : vectorizers_attacks )
    {
      
      std::string vectorizer_type = vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER];//= vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER];//unknown vectorizer->get_type();
      if ( vectorizer->get_type() < provallo::vectorizer_type::UNKNOWN_VECTORIZER)
      {
        vectorizer_type = vectorizer_types[vectorizer->get_type()];
      } 
      std::cout<<"[+] vectorizing with "<<vectorizer_type<<std::endl;
      vectorizer->add_document(fuzz_string);
      std::cout<<"[+] vectorizing with "<<vectorizer_type<<" done"<<std::endl;
      std::cout<<"[+] finished "<<fuzz_file<<std::to_string(cur)+"/"+std::to_string(total)<<std::endl; 

    }}//if is_attack
    else
    {
      for(auto& vectorizer : vectorizers_normal)
      {
        std::string vectorizer_type = vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER];//= vectorizer_types[provallo::vectorizer_type::UNKNOWN_VECTORIZER];//unknown vectorizer->get_type();
        if ( vectorizer->get_type() < provallo::vectorizer_type::UNKNOWN_VECTORIZER)
        {
          vectorizer_type = vectorizer_types[vectorizer->get_type()];
        } 
        std::cout<<"[+] vectorizing with "<<vectorizer_type<<std::endl;
        vectorizer->add_document(fuzz_string);
        std::cout<<"[+] vectorizing with "<<vectorizer_type<<" done"<<std::endl;
        std::cout<<"[+] finished "<<fuzz_file<<std::to_string(cur)+"/"+std::to_string(total)<<std::endl;
      }
    }//normal case
  
    cur++;

  }//for fuzz_file
  std::cout<<"[+] finished vectorizing "<<std::endl;
  c_end = clock ();
  t_end = std::chrono::high_resolution_clock::now ();
  std::cout << "[+] Test CPU time elapsed in s: "
    << (double) (c_end - c_start) / CLOCKS_PER_SEC << std::endl;
  std::cout << "[+] Test Wall time elapsed in s: "
    << std::chrono::duration<double> (t_end - t_start).count ()
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
       v->fit();
      std::cout<<"[+] fitting vectorizer "<<vectorizer_type<<" done"<<std::endl;
      c_end = clock ();
      t_end = std::chrono::high_resolution_clock::now ();
      std::cout << "[+] vectorizer CPU time elapsed in s: "
        << (double) (c_end - c_start) / CLOCKS_PER_SEC << std::endl;
      std::cout << "[+] vectorizer Wall time elapsed in s: "
        << std::chrono::duration<double> (t_end - t_start).count ()
        << std::endl;

  }
 
          
  //now that we have the autoencoders, we can train them
  //reiterate on the files and train the autoencoders 
  //print total time 
  c_end = clock ();
  t_end = std::chrono::high_resolution_clock::now ();
  std::cout << "[+] Test CPU time elapsed in s: "
    << (double) (c_end - c_start) / CLOCKS_PER_SEC << std::endl;  
  std::cout << "[+] Test Wall time elapsed in s: "
    << std::chrono::duration<double> (t_end - t_start).count ()
    << std::endl; 
  std::cout<<"[+] creating autoencoders "<<std::endl;
  for ( auto & vectorizer : vectorizers_attacks)
  {
 
    bool is_attack = false;
    std::string random_file = string_files[t_end.time_since_epoch().count()%string_files.size()]; 
    while(!is_attack&&(random_file.find("attack")==std::string::npos||random_file.find("door")==std::string::npos) )  
    {
      std::cout<<"[+] skipping "<<random_file<<std::endl; 
      t_end = std::chrono::high_resolution_clock::now ();
      random_file = string_files[t_end.time_since_epoch().count()%string_files.size()]; 
      if(random_file.find("attack")!=std::string::npos||random_file.find("door")!=std::string::npos )
        break;
    }
    is_attack = true;

    std::cout<<"[+] loading random file : "<<random_file<<std::endl;
    
    std::ifstream ifrandom(random_file);
       std::string fit_file((std::istreambuf_iterator<char>(ifrandom)),
    std::istreambuf_iterator<char>());
 
    std::vector<real_t> input =vectorizer->predict(fit_file);
    size_t input_size = input.size();

     std::cout<<"[+] creating autoencoder for vectorizer:"<< vectorizer_types[vectorizer->get_type()] << "input size:"<<std::to_string(input_size)<<std::endl;
    if( input_size==0)
      input_size=1; //avoid zero size input
     

    c_start = clock ();
    t_start = std::chrono::high_resolution_clock::now ();
  
    autoencoders.push_back(new provallo::auto_encoder<real_t,real_t>(input_size,input_size*(2*std::log2(input_size)),1));//input,hidden,output 
    c_end = clock ();
    t_end = std::chrono::high_resolution_clock::now ();
    std::cout << "[+] auto encoder init Test CPU time elapsed in s: " << (double) (c_end - c_start) / CLOCKS_PER_SEC << std::endl;    

  }

  for (auto & fuzz_file : string_files )
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
      if(input.size()==0)
      {
        std::cout<<"[+] vectorizer output size : "<<std::to_string(input.size())<<"/"<<std::to_string(vectorizer->get_output_size())<<std::endl; 
        
        input.resize(vectorizer->get_output_size());
        for(size_t i=0;i<input.size();i++)
        {
          input[i] = 0.0;
        } 

      } 

      std::cout<<"[+] vectorizer output size : "<<std::to_string(input.size())<<std::endl; 

      double output[2]={1,1};
      do
      {
      //get dictionary and weight and train the autoencoder 
      //train with vectorized data:
      enc->train((real_t*)input.data(),output,1); 
      //test with vectorized data 
      std::cout<<"[+] train output[0] : "<<std::to_string(output[0])<<std::endl;
      
      enc->test(input.data(), output,1);
       
      }while(output[0]+output[1]<0.99);
      std::cout<<"[+] training autoencoder done"<<std::endl;

      //evaluate the autoencoder
      std::vector<real_t> output_vector(output,output+2);
      std::cout<<"[+] evaluating autoencoder"<<std::endl;
      enc->test(input.data(),output,1);

      std::cout<<"[+] evaluating autoencoder done"<<std::endl;
      std::cout<<"[+] output[0] : "<<std::to_string(output[0])<<std::endl;
       
      std::cout<<"[+] training autoencoder done"<<std::endl;
      std::cout<<"[+] training autoencoder done"<<std::endl;
      enc->save("encoder_fuzzdb_"+vectorizer_type+".json");
      //enc->load ("encoder_fuzzdb_"+vectorizer_type+".json");
     }
  
    c_start = clock ();
    t_start = std::chrono::high_resolution_clock::now ();
    std::cout << "[+] Test CPU time elapsed in s: "
    << (double) (c_end - c_start) / CLOCKS_PER_SEC << std::endl;
    std::cout << "[+] Test Wall time elapsed in s: "
    << std::chrono::duration<double> (t_end - t_start).count ()
    << std::endl;
    std::cout<<"[+] finished "<<fuzz_file<<std::endl;

  } 
    
  //then we want to fit the documents into the vectorizers
  //for each vectorizer, we want to train an autoencoder
  //for each autoencoder, we want to train a weak classifier
  //first we iterate the files and fill the dictionaries of the vectorizers :

  //create autoencoders:
    //autoencoders.push_back(new provallo::autoencoder);
    //train auto encoders on the output of the vectorizers
    //autoencoders[0]->train(vectorizers[0]->get_output());
    //autoencoders[1]->train(vectorizers[1]->get_output());
    //autoencoders[2]->train(vectorizers[2]->get_output());
  
  for(auto& vectorizer : vectorizers_attacks)
  {
    delete vectorizer;
  }
  for(auto& vectorizer : vectorizers_normal)
  {
    delete vectorizer;
  } 
  for ( auto& ec : autoencoders)
  {
    delete ec;
  }
  return ret;
 
 }//  end of fit_fuzzdb
 //-----------------------------------------------------------------------------


 