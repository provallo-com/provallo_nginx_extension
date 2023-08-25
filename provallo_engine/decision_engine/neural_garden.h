#ifndef __NEURAL_GARDEN_H_
#define __NEURAL_GARDEN_H_


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

namespace provallo {
namespace decision_engine {

    //neural urns are the basic unit of the neural garden 
    //they are the containers for the neural networks
    class base_neural_urn {
        uint64_t id;
        static std::atomic_uint64_t urn_count;
    public:
        base_neural_urn():id(urn_count++) {}
      
        virtual ~base_neural_urn() {}

        //setup connects the rows in the matrix to the neural networks 
        //and auto encoders
        //each auto encoder is connected to a neural network 
        //each neural network is connected to a row in the matrix 
        //the matrix is the data set


        virtual void setup(const matrix<double>& data) = 0; 
        

        virtual void train() = 0;
        virtual void test(const matrix<double>& data) = 0;

        virtual void add_neural_network(provallo::neural_net::ptr nn) = 0; 
        virtual void add_auto_encoder(provallo::auto_encoder<double,double> ae) = 0;
        virtual void remove_neural_network(provallo::neural_net::ptr nn) = 0;
        virtual void remove_auto_encoder(provallo::auto_encoder<double,double> ae) = 0;
        virtual void clear() = 0;
        
    protected:
        virtual void connect_auto_encoder(provallo::neural_net::ptr nn, provallo::auto_encoder<double,double> ae) = 0; 
        virtual void disconnect_auto_encoder(provallo::neural_net::ptr nn, provallo::auto_encoder<double,double> ae) = 0;


         protected:

            std::vector< provallo::neural_net::ptr>  neural_networks;   
            typedef std::vector<provallo::auto_encoder<double,double> > auto_encoders; 
            std::vector<std::pair<provallo::neural_net::ptr,provallo::auto_encoder<double,double> > > connections; 
            std::mutex connections_lock;
            std::condition_variable connections_cv;
            std::thread connections_thread;
            bool connections_thread_running;
            void connections_thread_loop();
            void build_connections(const matrix<double>& data); 
            void add_connection(provallo::neural_net::ptr nn, provallo::auto_encoder<double,double> ae);
            void remove_connection(provallo::neural_net::ptr nn, provallo::auto_encoder<double,double> ae);
            void clear_connections();
        
    };      
        
    //neural garden is the container for the neural urns
    //it is the main class of the decision engine
    class neural_garden {
        private: 
            std::vector<base_neural_urn*> urns;
            std::mutex urns_lock;
            std::condition_variable urns_cv;
            std::thread urns_thread;
            bool urns_thread_running;
            void urns_thread_loop(); 
        public:
            neural_garden(provallo::learning_task task);
            ~neural_garden();
            void add_urn(base_neural_urn* urn);
            void remove_urn(base_neural_urn* urn);
            void start();
            void stop();
            void wait();
            void clear();

    };
    //neural urns are the basic unit of the neural garden
    //they are the containers for the neural networks

    template<typename T>
    class neural_urn : public base_neural_urn {
        public:
            neural_urn() {}
            
            virtual ~neural_urn() {}
            
            void setup(const matrix<double>& data) override {
                for(size_t i=0;i<data.rows();i++) {
                    auto& row = data.row(i);
                    for(auto& ae : auto_encoders) {
                        ae.add_data(row);
                    }
                    for(auto& nn : neural_networks) {
                        nn->add_data(row);
                    }
                }
            }
           
            void train() override;
            void test() override;
            void add_neural_network(provallo::neural_net::ptr nn) override; 
            void add_auto_encoder(provallo::auto_encoder<double,double> ae) override;
            void remove_neural_network(provallo::neural_net::ptr nn) override;
            void remove_auto_encoder(provallo::auto_encoder<double,double> ae) override;
            void clear() override;
        private:
            std::vector< provallo::neural_net::ptr>  neural_networks;   
            std::vector<provallo::auto_encoder<double,double> > auto_encoders; 
    };
 
    template<typename T>
    void neural_urn<T>::train() {
        for(auto& nn : neural_networks) {
            
            nn->train();

        }
    }
    template<typename T>
    void neural_urn<T>::test() {
        for(auto& nn : neural_networks) {
            nn->test();
        }
    }
    template<typename T>
    void neural_urn<T>::add_neural_network(provallo::neural_net::ptr nn) {
        neural_networks.push_back(nn);
    }
    template<typename T>
    void neural_urn<T>::add_auto_encoder(provallo::auto_encoder<double,double> ae) {
        auto_encoders.push_back(ae);
    }
    template<typename T>
    void neural_urn<T>::remove_neural_network(provallo::neural_net::ptr nn) {
        neural_networks.erase(std::remove(neural_networks.begin(), neural_networks.end(), nn), neural_networks.end());
    }
    template<typename T>
    void neural_urn<T>::remove_auto_encoder(provallo::auto_encoder<double,double> ae) {
        auto_encoders.erase(std::remove(auto_encoders.begin(), auto_encoders.end(), ae), auto_encoders.end());
    }
    template<typename T>
    void neural_urn<T>::clear() {
        neural_networks.clear();
        auto_encoders.clear();
    }

} // namespace decision_engine
} // namespace provallo 

#endif // __NEURAL_GARDEN_H_