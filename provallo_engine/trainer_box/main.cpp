#include "../decision_engine/classifier.h"
#include "../decision_engine/dataset.h" 
#include "../decision_engine/neuralhelper.h"

#include <iostream>
#include <string>
#include <fstream> 
#include <chrono>
provallo::classifier *_lastclassifier = nullptr;
provallo::isolation_forest* iso_last_fit = nullptr;


bool test_fit_iso_forest();
bool test_dataset_load();

provallo::isolation_forest* isoforest_single(const provallo::attribute_information& attributes);


 
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
  if( attributes==att_stat)
    return forest;
  else
  {
    return new provallo::isolation_forest ();    
  }
}


bool
test_dataset_load ()
{
  try
    {
      provallo::files_collector collector ("provallo_core/provallo_attributes");
      const provallo::attribute_information& attributes (
	    collector.getAttributes ());

      std::ifstream weights_file("provallo_core/provallo_attributes.weights");

      std::cout << "-- Attributes information : " << std::endl << std::endl;
      std::cout << collector.getAttributes () << std::endl;

      std::cout<< "[ " << std::to_string(attributes.getSize()) << "]"<<std::endl;
      auto c_start = clock ();
      auto t_start = std::chrono::high_resolution_clock::now ();
      provallo::training_set train_data (collector.getAttributes ());
      collector.pushTrainData (&train_data);

      auto c_end = clock ();  
      auto t_end = std::chrono::high_resolution_clock::now ();

		
      
      std::cout<<"[+] training set size : "<< std::to_string (train_data.size())<<std::endl;
      std::cout<<"[+] training set attributes : "<< std::to_string (train_data.getattributesNumber())<<std::endl;
      std::cout<<"[+] training set samples : "<< std::to_string (train_data.get_samples().size())<<std::endl;
      std::cout<<"[+] training set entropy : "<< std::to_string (train_data.entropy())<<std::endl;
      std::cout<<"[+] training set skewness : "<< std::to_string (train_data.skewness())<<std::endl;
      std::cout<<"[+] training set kurtosis : "<< std::to_string (train_data.kurtosis())<<std::endl;
      std::cout<<"[+] training set standard deviation : "<< std::to_string (train_data.stddev())<<std::endl;
      std::cout<<"[+] training set mean : "<< std::to_string (train_data.mean())<<std::endl;
      std::cout<<"[+] training set median : "<< std::to_string (train_data.median())<<std::endl;
      std::cout<<"[+] training set variance : "<< std::to_string (train_data.variance())<<std::endl;


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
      const provallo::mmap_vector<provallo::attribute> & attributes_vector =  test_data.get_samples();
      std::cout<<"[+] attributes vector size "<<attributes_vector.size()<<std::endl;
      /*
      provallo::mmap_vector<double> predictions(attributes_vector.size());
      //size_t column_size_test = test_data.getattributes().getSize();

      size_t nrow = 0;
      size_t col=0;
      size_t rows = test_data.get_samples().size() / test_data.getattributesNumber();
      size_t columns = test_data.getattributesNumber();

      std::cout<<"[+] test data rows "<<rows<<" columns "<<columns<<std::endl;
      data_matrix.clear();
      data_matrix.resize(rows,columns);
      for ( auto& sample : test_data.get_samples() )
      {

          col++;
          col %= columns;
          data_matrix(nrow,col) = (double) sample.continous();

          if(col == columns-1)
          {
            nrow++;
          }

      }

      //rot= sqrt(data_matrix.size1()*data_matrix.size2() );
        */
      //reduced = provallo::jacobi(data_matrix,rot) ;
      //todo: create bayesian parameter set


      typedef provallo::adaboost<provallo::bayesian> bayesian_boost;

      /*std::vector<double> outlier_scores = single->predict (data_matrix.data(), data_matrix.rows(), true);

      std::cout<<"[+] predict outlier scores size "<<outlier_scores.size()<<std::endl;
      col =0;
      for(  auto score : outlier_scores)
      {
        std::cout<<"[+]predict outlier score for index "<< std::to_string(col) <<":"<< std::to_string(  score) <<std::endl;
        col++;
      }
      std::cout<<"[+] finished predicting outlier scores"<<std::endl;
      */
//      typedef provallo::nearest_neighbor<provallo::metric<provallo::Overlap,provallo::Euclidean>>  nn ;
 //     std::map<std::string, Float> weights = getWeightMap(attributes, weights_file);

  //    provallo::nearest_neighbor_param nnp (attributes.getGroups().size(),weights);

      typedef provallo::random_tree<provallo::EntropyGain> egain_tree;
      typedef provallo::random_forest<egain_tree> rf_classifier;
      provallo::random_tree_param egain_param(.5,10,0.0) ;
      
      provallo::random_forest_param rf_param(100,egain_param);
      /*
      rf_param.rho = 0.5;
      rf_param.max_depth = 10;
      rf_param.min_samples_split = 2;
      rf_param.min_samples_leaf = 1;
      */


      std::cout << "Training RF Classifier with 100 parallel trees " << std::endl;
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
