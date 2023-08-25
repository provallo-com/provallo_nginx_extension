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
        meta_learner(): builder(std::string("meta_learner")+std::to_string(urn_count+1),"./db/meta_learner") ,id(urn_count++) {}
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