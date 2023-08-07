#ifndef _META_LEARNER_H_
#define _META_LEARNER_H_


#include <vector>
#include <string>
#include <map>
#include <set>
#include <memory>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <atomic>
#include "neuralhelper.h"
#include "autoencoder.h"
#include "matrix.h"
#include "pipelinebuilder.h"

namespace provallo {


    class pipeline_evaluator {
        
        //====================================================================================================
        //evaluators test the pipeline's accuracy and performance cost  
        //they are used to determine the best pipeline
        //the best pipeline is the one with the highest accuracy and lowest computational cost
        //in-addition to the accuracy and cost, the pipeline's complexity is also considered
        //the pipeline's complexity is the number of elements  in the pipeline , number of attributes/samples associated with each element, and the number of connections between elements
        //the complexity is used to determine the computational cost of the pipeline 
        //the computational cost is the number of floating point operations required to process a single sample
        //the accuracy is the number of correct predictions divided by the total number of predictions
        

private:
         
         std::vector<std::pair<std::string,std::function<matrix<real_t> (const matrix<real_t>&)>>> evaluator_desc;
                
public:
        pipeline_evaluator() {
            //register default evaluators
            add_evaluator("accuracy",[](const matrix<real_t>& data) -> matrix<real_t> {
                return matrix<real_t>(1,1);
            });
            add_evaluator("cost",[](const matrix<real_t>& data) -> matrix<real_t> {
                return matrix<real_t>(1,1);
            });
            add_evaluator("complexity",[](const matrix<real_t>& data) -> matrix<real_t> {
                return matrix<real_t>(1,1);
            });
            
        }
        void evaluate(const matrix<double>& data) {
            for(auto& e: evaluator_desc) {
                 e.second(data);
            }
        }   
        
        void add_evaluator(std::string name, std::function< matrix<real_t> (const matrix<double>&)> evaluator) {

            evaluator_desc.push_back(std::make_pair(name,evaluator));
        }

        void remove_evaluator(std::string name) {
            for(auto it = evaluator_desc.begin(); it != evaluator_desc.end(); ++it) {
                if(it->first == name) {
                    evaluator_desc.erase(it);
                    break;
                }
            }
        }

        void clear() {
            evaluator_desc.clear();
        }

        void print(const matrix<double>& data) {
            for(auto& e: evaluator_desc) {
                //use matrix print function
                std::cout << e.first <<  e.second(data)<<std::endl;
            }
        }
        //evaluator functions for each stage in the pipeline
        //the evaluator functions are used to determine the accuracy and cost of each stage in the pipeline
        //accuracy of vectorizer  = number of correct predictions / total number of predictions 
        //cost of vectorizer = number of floating point operations required to process a single sample
        //complexity of vectorizer = number of elements in the vectorizer + number of attributes/samples associated with each element + number of connections between elements

        //acciuracy of neural network = number of correct predictions / total number of predictions
        //cost of neural network = number of floating point operations required to process a single sample
        //complexity of neural network = number of elements in the neural network + number of attributes/samples associated with each element + number of connections between elements

        //accuracy of autoencoder = number of correct predictions / total number of predictions
        //cost of autoencoder = number of floating point operations required to process a single sample
        //complexity of autoencoder = number of elements in the autoencoder + number of attributes/samples associated with each element + number of connections between elements

        //accuracy of classifier = number of correct predictions / total number of predictions
        //cost of classifier = number of floating point operations required to process a single sample
        //complexity of classifier = number of elements in the classifier + number of attributes/samples associated with each element + number of connections between elements

        //accuracy of regressor = number of correct predictions / total number of predictions
        //cost of regressor = number of floating point operations required to process a single sample
        //complexity of regressor = number of elements in the regressor + number of attributes/samples associated with each element + number of connections between elements

        //accuracy of normalizer = number of correct predictions / total number of predictions
        //cost of normalizer = number of floating point operations required to process a single sample
        //complexity of normalizer = number of elements in the normalizer + number of attributes/samples associated with each element + number of connections between elements  

        //accuracy of denormalizer = number of correct predictions / total number of predictions
        //cost of denormalizer = number of floating point operations required to process a single sample
        //complexity of denormalizer = number of elements in the denormalizer + number of attributes/samples associated with each element + number of connections between elements

        //metrics for the entire pipeline 
        //accuracy of pipeline = number of correct predictions / total number of predictions
        //cost of pipeline = number of floating point operations required to process a single sample
        //complexity of pipeline = number of elements in the pipeline + number of attributes/samples associated with each element + number of connections between elements

        
        
    };

    class meta_learner {
        uint64_t id;
        static std::atomic_uint64_t urn_count;
        meta_builder builder;
        
        std::vector< provallo::neural_net::ptr>  neural_networks;
        typedef std::vector<provallo::auto_encoder<double,double> > auto_encoders;
        std::vector<std::pair<provallo::neural_net::ptr,provallo::auto_encoder<double,double> > > connections;
        std::mutex connections_lock;
        std::condition_variable connections_cv;

    
        
    public: 
        meta_learner():id(urn_count++) {}
        virtual ~meta_learner() {}
        virtual void setup(const matrix<double>& data) = 0; 
        virtual void train() = 0;
        virtual void test(const matrix<double>& data) = 0;
        virtual void add_neural_network(provallo::neural_net::ptr nn) = 0; 
        virtual void add_auto_encoder(provallo::auto_encoder<double,double> ae) = 0;
        virtual void remove_neural_network(provallo::neural_net::ptr nn) = 0;
        virtual void remove_auto_encoder(provallo::auto_encoder<double,double> ae) = 0;
        virtual void clear() = 0;
    };

}//namespace provallo














#endif 