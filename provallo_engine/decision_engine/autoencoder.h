#ifndef PROVALLO_AUTO_ENCODER_H_
#define PROVALLO_AUTO_ENCODER_H_

#include "neuralhelper.h"
#include "matrix.h"

#include <string>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <iomanip>
#include <assert.h>
#include <vector>
#include "classdist.h"
namespace provallo
{
    //simple shokoder autoencoder
    //
    // see
    //http://www.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf
    //http://www.stanford.edu/class/cs294a/sparseAutoencoder_notes.pdf
    //vae : http://arxiv.org/pdf/1312.6114v10.pdf
    //variational autoencoder : http://arxiv.org/pdf/1312.6114v10.pdf
    //http://www.cs.toronto.edu/~fritz/absps/transauto6.pdf
    
    

    template <typename T, typename real_x = real_t>
    class auto_encoder
    {
    private:
        size_t inputDim;
        size_t hiddenDim;
        size_t outputDim;
        T *input;
        T *hidden;
        T *output;
        T *weight1;
        T *weight2;
        T *bias1;
        T *bias2;
        T *weight1Grad;
        T *weight2Grad;
        T *bias1Grad;
        T *bias2Grad;
        T *weight1Momentum;
        T *weight2Momentum;
        T *bias1Momentum;
        T *bias2Momentum;
        T *weight1Update;
        T *weight2Update;
        T *bias1Update;
        T *bias2Update;
        T *weight1Decay;
        T *weight2Decay;
        T *bias1Decay;
        T *bias2Decay;
        T *weight1Sparsity;
        T *weight2Sparsity;
        T *bias1Sparsity;
        T *bias2Sparsity;
        T *weight1SparsityHat;
        T *weight2SparsityHat;
        T *bias1SparsityHat;
        T *bias2SparsityHat;
        T *weight1SparsityGrad;
        T *weight2SparsityGrad;
        T *bias1SparsityGrad;
        T *bias2SparsityGrad;
        T *weight1SparsityGradHat;
        T *weight2SparsityGradHat;

        T* weight1Inc;
        T* weight2Inc;
        T* weight1GradPrev;
        T* weight2GradPrev;

        //prevprev
        T* weight1GradPrevPrev;
        T* weight2GradPrevPrev;
        T* bias1GradPrevPrev;
        T* bias2GradPrevPrev;


        T* bias1Inc;
        T* bias2Inc;
        T* bias1GradPrev;
        T* bias2GradPrev;
        T* weight1Prev;
        T* weight2Prev;



        real_x learningRate;
        real_x momentum;
        real_x weightDecay;
        real_x sparsityParam;
        real_x beta;
        real_x sparsityParamHat;
        real_x sparsityPenalty;
        real_x sparsityGradient;
        real_x sparsityGradientHat;

        T *bias1SparsityGradHat;
        T *bias2SparsityGradHat;
        
        T* bias1Prev;
        T* bias2Prev;



        void initializeWeight();
        void initializeBias();
        void initializeWeight(T *weight, size_t size);
        void initializeBias(T *bias, size_t size);
        void initializeWeight(T *weight, size_t row, size_t col);
        void conjugateGradient();

        virtual void initializeWeightGrad()
        {
            //initialize the gradients 
            initializeWeight1Grad();
            initializeWeight2Grad();
            //initialize bias grads
            initializeBias1Grad();
            initializeBias2Grad();

            //initialize weight prev
            initializeWeight1Prev();
            initializeWeight2Prev();
            //initialize bias prev
            initializeBias1Prev();
            initializeBias2Prev();

            //initialize weight momentum
            initializeWeight1Momentum();

            initializeWeight2Momentum();
            //initialize bias momentum
            initializeBias1Momentum();
            initializeBias2Momentum();


            //initialize weight update
            initializeWeight1Update();
            initializeWeight2Update();

            //initialize bias update
            initializeBias1Update();
            initializeBias2Update();


            //initialize weight decay
            initializeWeight1Decay();
            initializeWeight2Decay();


            //initialize bias decay 
            initializeBias1Decay();
            initializeBias2Decay();


            //initialize weight sparsity

            initializeWeight1Sparsity();
            initializeWeight2Sparsity();


            //initialize bias sparsity
            initializeBias1Sparsity();
            initializeBias2Sparsity();


            //initialize weight sparsity hat
            initializeWeight1SparsityHat();
            initializeWeight2SparsityHat();

            
            //initialize bias sparsity hat
            initializeBias1SparsityHat();
            initializeBias2SparsityHat();


            //initialize weight sparsity grad
            initializeWeight1SparsityGrad();
            initializeWeight2SparsityGrad();


            //initialize bias sparsity grad
            initializeBias1SparsityGrad();
            initializeBias2SparsityGrad();


            //initialize weight sparsity grad hat
            initializeWeight1SparsityGradHat();
            initializeWeight2SparsityGradHat();

            //initialize bias sparsity grad hat

            initializeBias1SparsityGradHat();
            initializeBias2SparsityGradHat();

            //initialize weight grad prev prev

            initializeWeight1GradPrevPrev();
            initializeWeight2GradPrevPrev();

            //initialize bias grad prev prev
            //
            initializeBias1GradPrevPrev();
            initializeBias2GradPrevPrev();

        }
       
        typedef T (auto_encoder<T,real_x>::*XactivationFunctionPtr)(T);

        


         XactivationFunctionPtr activationFunctionPtr; 
         XactivationFunctionPtr activationGradientFunctionPtr;
         XactivationFunctionPtr activationPrimeFunctionPtr;
         XactivationFunctionPtr activationPrimeGradientFunctionPtr;
         XactivationFunctionPtr activationPrimeGradientHatFunctionPtr;

        void initializeActivationFunction();    
        void forward();
        void backward();
        void update();
        
    public:
        //constructor
        auto_encoder(size_t inputDim, size_t hiddenDim, size_t outputDim);
        virtual ~auto_encoder(); 
        T sigmoid(T x);
        T sigmoidPrime(T x);
        T sigmoidGradient(T x);
        T relu(T x);
        T reluGradient(T x);
        T reLuLeaky(T x);
        T reLuLeakyGradient(T x);
        T leakyReluPrime(T x);
        T leakyRelu(T x);
        T leakyReluGradient(T x);

        T identityGradient(T x);
        T tanhPrime(T x);
        T reluPrime(T x);
        T softplusPrime(T x);
        T linearPrime(T x);
        T softmaxPrime(T x);
        T identity(T x);

        T tanh(T x);
        T tanhGradient(T x);
        T softplus(T x);
        T softplusGradient(T x);
        T linear(T x);
        T linearGradient(T x);
        T softmax(T x);
        T softmaxGradient(T x);

        T gaussian(T x);
        T gaussianGradient(T x);
        T gaussianPrime(T x);

        T sinusoid(T x);
        T sinusoidGradient(T x);
        T sinusoidPrime(T x);

        T softsign(T x);
        T softsignGradient(T x);
        T softsignPrime(T x);

        T sinc(T x);
        T sincGradient(T x);
        T sincPrime(T x);

        T bentIdentity(T x);
        T bentIdentityGradient(T x);
        T bentIdentityPrime(T x);

        T softExponentialPrime(T x);
        T softExponential(T x);
        T softExponentialGradient(T x);

        T cost(T *input, T *output, size_t size);
        void train(T *input, T *output, size_t size);
        void train (matrix<T>& input,  class_dist& output);

        void test(T *input, T *output, size_t size);
        void test (matrix<T>& input,  class_dist& output);


        void dump(  std::ostream &out = std::cout) const;  
        
        void save(std::string filename);
        void save_as_pt(std::string filename);

        void load(std::string filename);
        void load(std::istream &in);

        void feedforward(T *input, T *output, size_t size);
        
        void backprop(T *input, T *output, size_t size);

        T *getWeight1();
        T *getWeight2();
        void setInput(T *input);
        void setHidden(T *hidden);
        void setOutput(T *output);
        //set dimension
        void setInputDim(size_t inputDim);
        void setHiddenDim(size_t hiddenDim);
        void setOutputDim(size_t outputDim);


        T *getInput() const;
        T *getHidden() const;
        T *getOutput() const;
        size_t getInputDim() const;
        size_t getHiddenDim() const;
        size_t getOutputDim() const;
        void setLearningRate(real_x learningRate);
        void setMomentum(real_x momentum);
        void setWeightDecay(real_x weightDecay);
        void setSparsityParam(real_x sparsityParam);
        void setBeta(real_x beta);

        real_x getLearningRate() const;
        real_x getMomentum() const;
        real_x getWeightDecay() const;
        real_x getSparsityParam() const;
        real_x getBeta() const;

        T *getWeight1() const;
        T *getWeight2() const;
        void setWeight1(T *weight1);
        void setWeight2(T *weight2);
        void setBias1(T *bias1);
        void setBias2(T *bias2);

        T *getBias1() const;
        T *getBias2() const;


        void setSparsityParamHat(real_x sparsityParamHat);
        void setSparsityPenalty(real_x sparsityPenalty);
        void setSparsityGradient(real_x sparsityGradient);
        void setSparsityGradientHat(real_x sparsityGradientHat);
        
        real_x getSparsityParamHat() const;
        real_x getSparsityPenalty() const;


 

        void setWeight1Inc(T* weight1Inc);
        void setWeight2Inc(T* weight2Inc);
        void setBias1Inc(T* bias1Inc);
        void setBias2Inc(T* bias2Inc);

        T *getWeight1Inc() const;
        T *getWeight2Inc() const;
        T *getBias1Inc() const;
        T *getBias2Inc() const;


        void setWeight1Grad(T* setWeight1Grad); 
        void setWeight2Grad(T* setWeight2Grad);
        void setBias1Grad(T* setBias1Grad);
        void setBias2Grad(T* setBias2Grad);

        T *getWeight1Grad() const;
        T *getWeight2Grad() const;
        T *getBias1Grad() const;
        T *getBias2Grad() const;


        //previous values :
        void setWeight1Prev(T* weight1Prev);
        void setWeight2Prev(T* weight2Prev);
        void setBias1Prev(T* bias1Prev);
        void setBias2Prev(T* bias2Prev);

        T *getWeight1Prev() const;
        T *getWeight2Prev() const;
        T *getBias1Prev() const;
        T *getBias2Prev() const;  


        void updateWeight1GradPrev();
        void updateWeight2GradPrev();
        void updateBias1GradPrev();
        void updateBias2GradPrev();
        
        void updateWeight1Inc();
        void updateWeight2Inc();
        void updateBias1Inc();
        void updateBias2Inc();

        void updateWeight1();
        void updateWeight2();
        void updateBias1();
        void updateBias2();


        void updateWeight1Decay();
        void updateWeight2Decay();
        void updateBias1Decay();
        void updateBias2Decay();


        void updateWeight1Sparsity();
        void updateWeight2Sparsity();
        void updateBias1Sparsity();
        void updateBias2Sparsity();

        void updateWeight1SparsityHat();
        void updateWeight2SparsityHat();
        void updateBias1SparsityHat();
        void updateBias2SparsityHat();


        void updateBias1GradPrevPrev();
        void updateBias2GradPrevPrev();
        void updateWeight1GradPrevPrev();
        void updateWeight2GradPrevPrev();

        void updateWeight1SparsityGrad();
        void updateWeight2SparsityGrad();
        void updateBias1SparsityGrad();
        void updateBias2SparsityGrad();

        void updateWeight1SparsityGradHat();
        void updateWeight2SparsityGradHat();
        void updateBias1SparsityGradHat();
        void updateBias2SparsityGradHat();

        
        void initializeWeight1Grad()
        {
            initializeWeight(weight1Grad, inputDim, hiddenDim);
        }
        void initializeWeight2Grad()
        {
            initializeWeight(weight2Grad, hiddenDim, outputDim);
        }
        void initializeBias1Grad()
        {
            initializeBias(bias1Grad, hiddenDim);
        }
        void initializeBias2Grad()
        {
            initializeBias(bias2Grad, outputDim);
        }

        void initializeWeight1GradPrev()
        {
            initializeWeight(weight1GradPrev, inputDim, hiddenDim);
        }
        void initializeWeight2GradPrev()
        {
            initializeWeight(weight2GradPrev, hiddenDim, outputDim);
        }
        void initializeBias1GradPrev()  
        {
            initializeBias(bias1GradPrev, hiddenDim);
        }

        void initializeBias2GradPrev()
        {
            initializeBias(bias2GradPrev, outputDim);
        }
        void initializeWeight1Inc()
        {
            initializeWeight(weight1Inc, inputDim, hiddenDim);
        }
        void initializeWeight2Inc()
        {
            initializeWeight(weight2Inc, hiddenDim, outputDim);
        }
        void initializeBias1Inc()
        {
            initializeBias(bias1Inc, hiddenDim);
        }
        void initializeBias2Inc()
        {
            initializeBias(bias2Inc, outputDim);
        }
        void initializeWeight1SparsityGrad()
        {
            initializeWeight(weight1SparsityGrad, inputDim, hiddenDim);
        }
        void initializeWeight2SparsityGrad()
        {
            initializeWeight(weight2SparsityGrad, hiddenDim, outputDim);
        }
        void initializeBias1SparsityGrad()
        {
            initializeBias(bias1SparsityGrad, hiddenDim);
        }
        void initializeBias2SparsityGrad()
        {
            initializeBias(bias2SparsityGrad, outputDim);
        }
        void initializeWeight1SparsityGradHat()
        {
            initializeWeight(weight1SparsityGradHat, inputDim, hiddenDim);
        }
        void initializeWeight2SparsityGradHat()
        {
            initializeWeight(weight2SparsityGradHat, hiddenDim, outputDim);
        }
        void initializeBias1SparsityGradHat()
        {
            initializeBias(bias1SparsityGradHat, hiddenDim);
        }
        void initializeBias2SparsityGradHat()
        {
            initializeBias(bias2SparsityGradHat, outputDim);
        }
        void initializeWeight1SparsityHat()
        {
            initializeWeight(weight1SparsityHat, inputDim, hiddenDim);
        }
        void initializeWeight2SparsityHat()
        {
            initializeWeight(weight2SparsityHat, hiddenDim, outputDim);
        }
        void initializeBias1SparsityHat()
        {
            initializeBias(bias1SparsityHat, hiddenDim);
        }
        void initializeBias2SparsityHat()
        {
            initializeBias(bias2SparsityHat, outputDim);
        }
        void initializeWeight1Sparsity()
        {
            initializeWeight(weight1Sparsity, inputDim, hiddenDim);
        }
        void initializeWeight2Sparsity()
        {
            initializeWeight(weight2Sparsity, hiddenDim, outputDim);
        }
        void initializeBias1Sparsity()
        {
            initializeBias(bias1Sparsity, hiddenDim);
        }
        void initializeBias2Sparsity()
        {
            initializeBias(bias2Sparsity, outputDim);
        }
        void initializeWeight1Decay()
        {
            initializeWeight(weight1Decay, inputDim, hiddenDim);
        }
        void initializeWeight2Decay()
        {
            initializeWeight(weight2Decay, hiddenDim, outputDim);
        }
        void initializeBias1Decay()
        {
            initializeBias(bias1Decay, hiddenDim);
        }
        void initializeBias2Decay()
        {
            initializeBias(bias2Decay, outputDim);
        }
        void initializeWeight1Prev()
        {
            initializeWeight(weight1Prev, inputDim, hiddenDim);

        }
        void initializeWeight2Prev()
        {
            initializeWeight(weight2Prev, hiddenDim, outputDim);
        }
        void initializeBias1Prev()
        {   
            initializeBias(bias1Prev, hiddenDim);

        }
        void initializeBias2Prev()
        {
            initializeBias(bias2Prev, outputDim);
        }
        void initializeWeight1Momentum()
        {
            initializeWeight(weight1Momentum, inputDim, hiddenDim);
        }
        void initializeWeight2Momentum()
        {
            initializeWeight(weight2Momentum, hiddenDim, outputDim);
        }
        void initializeBias1Momentum()
        {
            initializeBias(bias1Momentum, hiddenDim);
        }
        void initializeBias2Momentum()
        {
            initializeBias(bias2Momentum, outputDim);
        }
        void initializeWeight1Update()
        {
            initializeWeight(weight1Update, inputDim, hiddenDim);
        }
        void initializeWeight2Update()
        {
            initializeWeight(weight2Update, hiddenDim, outputDim);
        }
        void initializeBias1Update()
        {
            initializeBias(bias1Update, hiddenDim);
        }
        void initializeBias2Update()
        {
            initializeBias(bias2Update, outputDim);
        }
        void initializeWeight1()
        {
            initializeWeight(weight1, inputDim, hiddenDim);
        }
        void initializeWeight2(){
            initializeWeight(weight2, hiddenDim, outputDim);
        }

        void initializeBias1()
        {
            initializeBias(bias1, hiddenDim);
        }
        void initializeBias2()
        {
            initializeBias(bias2, outputDim);
        }


        void initializeWeight1GradPrevPrev()
        {
            initializeWeight(weight1GradPrevPrev, inputDim, hiddenDim);
        }
        void initializeWeight2GradPrevPrev()
        {
            initializeWeight(weight2GradPrevPrev, hiddenDim, outputDim);
        }
        void initializeBias1GradPrevPrev()
        {
            initializeBias(bias1GradPrevPrev, hiddenDim);
        }
        void initializeBias2GradPrevPrev()
        {
            initializeBias(bias2GradPrevPrev, outputDim);
        }






        void updateWeight1Grad();   
        void updateWeight2Grad();
        void updateBias1Grad();
        void updateBias2Grad();
        //void updateWeight1();
        //void updateWeight2();
        //void updateBias1();
       // void updateBias2();
        void updateWeight1Prev();
        void updateWeight2Prev();
        void updateBias1Prev();
        void updateBias2Prev();
        void updateWeight1Momentum();
        void updateWeight2Momentum();
        void updateBias1Momentum();
        void updateBias2Momentum();
        void updateWeight1Update();
        void updateWeight2Update();

         
     };
    //variational auto encoder
    template <typename T, typename real_x = real_t>
    class variational_auto_encoder : public auto_encoder<T, real_x>
    {
        //variational auto encoder
        //variational auto encoder is a type of auto encoder that uses a variational bayesian approach to learning
        //additional variables for the variational auto encoder:
        //latentDim : the dimension of the latent space
        //latent : the latent space
        //latentMean : the mean of the latent space
        //latentLogVar : the log variance of the latent space
        //latentMeanGrad : the gradient of the latent mean
        //latentLogVarGrad : the gradient of the latent log variance

        //variational auto encoder uses the reparameterization trick to sample from the latent space
        //the reparameterization trick is used to sample from a distribution with a reparameterization of the distribution
         //variational auto encoder uses the kullback leibler divergence to measure the difference between the latent space and the prior distribution   
        protected:



        size_t latentDim;
        T *latent;
        T *latentMean;
        T *latentLogVar;
        T *latentMeanGrad;
        T *latentLogVarGrad;
        T *latentMeanGradPrev;
        T *latentLogVarGradPrev;
        T *latentMeanMomentum;
        T *latentLogVarMomentum;
        T *latentMeanUpdate;
        T *latentLogVarUpdate;
        T *latentMeanDecay;
        T *latentLogVarDecay;
        T *latentMeanSparsity;
        T *latentLogVarSparsity;
        T *latentMeanSparsityHat;
        T *latentLogVarSparsityHat;
        T *latentMeanSparsityGrad;
        T *latentLogVarSparsityGrad;
        T *latentMeanSparsityGradHat;
        T *latentLogVarSparsityGradHat;
        T *latentMeanGradPrevPrev;
        T *latentLogVarGradPrevPrev;
        T *latentMeanInc;
        T* latentMeanSparsityGradPrev;
        T* latentLogVarSparsityGradPrev;
        T* latentMeanSparsityGradHatPrev;
        T* latentLogVarSparsityGradHatPrev;
        T* latentMeanPrev;
        T* latentMeanSparsityGradPrevPrev;
        T* latentLogVarSparsityGradPrevPrev;
        T* latentMeanSparsityGradHatPrevPrev;
        //init helpers for the variational auto encoder
        void initializeLatent()
        {
            //reallocate the latent with the desired size:
            initialize(latent, latentDim);    

            //initialize the latent mean
            initializeLatentMean();
            //initialize the latent log var
            initializeLatentLogVar();
            //initialize the latent mean grad
            initializeLatentMeanGrad();
            //initialize the latent log var grad
            initializeLatentLogVarGrad();
            //initialize the latent mean grad prev
            initializeLatentMeanGradPrev();
            //initialize the latent log var grad prev
            initializeLatentLogVarGradPrev();
            //initialize the latent mean momentum
            initializeLatentMeanMomentum();
            //initialize the latent log var momentum
            initializeLatentLogVarMomentum();
            //initialize the latent mean update
            initializeLatentMeanUpdate();
            //initialize the latent log var update
            initializeLatentLogVarUpdate();
            //initialize the latent mean decay
            initializeLatentMeanDecay();
            //initialize the latent log var decay
            initializeLatentLogVarDecay();
            //initialize the latent mean sparsity
            initializeLatentMeanSparsity();
            //initialize the latent log var sparsity
            initializeLatentLogVarSparsity();
            //initialize the latent mean sparsity hat
            initializeLatentMeanSparsityHat();
            //initialize the latent log var sparsity hat
            initializeLatentLogVarSparsityHat();
            //initialize the latent mean grad prev
            initializeLatentMeanGradPrev();



        }
        void initializeLatentMean()
        {
            initialize(latentMean, latentDim, T(0));
            //initialize the latent mean grad
            initializeLatentMeanGrad();
            //initialize the latent mean grad prev
            initializeLatentMeanGradPrev();
            //initialize the latent mean momentum
            initializeLatentMeanMomentum();
            //initialize the latent mean update
            initializeLatentMeanUpdate();
            //initialize the latent mean decay
            initializeLatentMeanDecay();
            //initialize the latent mean sparsity
            initializeLatentMeanSparsity();
            //initialize the latent mean sparsity hat
            initializeLatentMeanSparsityHat();

             //done

        }
        void initializeLatentLogVar()
        {
            //reallocate the latent log var with the desired size:
            initialize(latentLogVar, latentDim, T(0));


            //initialize the latent log var grad
            initializeLatentLogVarGrad();
            //initialize the latent log var grad prev
            initializeLatentLogVarGradPrev();
            //initialize the latent log var momentum
            initializeLatentLogVarMomentum();
            //initialize the latent log var update
            initializeLatentLogVarUpdate();
            //initialize the latent log var decay
            initializeLatentLogVarDecay();
            //initialize the latent log var sparsity
            initializeLatentLogVarSparsity();
            //initialize the latent log var sparsity hat
            initializeLatentLogVarSparsityHat();

            //done
        }
        void initializeLatentMeanGrad()
        {
            //reallocate the latent mean grad with the desired size:
            initialize(latentMeanGrad, latentDim, T(0));    

            //initialize the latent mean grad prev
            initializeLatentMeanGradPrev();
            //initialize the latent mean momentum
            initializeLatentMeanMomentum();
            //initialize the latent mean update
            initializeLatentMeanUpdate();
            //initialize the latent mean decay
            initializeLatentMeanDecay();
            //initialize the latent mean sparsity
            initializeLatentMeanSparsity();
            //initialize the latent mean sparsity hat
            initializeLatentMeanSparsityHat();

            //done
        }
        void initializeLatentLogVarGrad()
        {
            //reallocate the latent log var grad with the desired size:
          
            initialize(latentLogVarGrad, latentDim, T(0));
            //initialize the latent log var grad prev
            initializeLatentLogVarGradPrev();
            //initialize the latent log var momentum
            initializeLatentLogVarMomentum();
            //initialize the latent log var update
            initializeLatentLogVarUpdate();
            //initialize the latent log var decay
            initializeLatentLogVarDecay();
            //initialize the latent log var sparsity
            initializeLatentLogVarSparsity();
            //initialize the latent log var sparsity hat
            initializeLatentLogVarSparsityHat();

            //done

        }
        void initializeLatentMeanGradPrev()
        {
            initialize(latentMeanGradPrev, latentDim, T(0));

            //done
        }
        void initializeLatentLogVarGradPrev()
        {
            initialize(latentLogVarGradPrev, latentDim, T(0));

            //done
        }
        void initializeLatentMeanMomentum()
        {
            initialize(latentMeanMomentum, latentDim, T(0));    
            
            //done      
        }
        void initializeLatentLogVarMomentum()
        {
            initialize(latentLogVarMomentum, latentDim, T(0));
            //done
        }
        void initialize ( T* _init_member,size_t size, const T value_)
        {
            //reallocate the member with the desired size:
            if(_init_member != nullptr)
            {
                delete [] _init_member;
                _init_member=nullptr;

            }
            _init_member = new T[size];
            //initialize the member
            for (size_t i = 0; i < size; i++)
            {
                _init_member[i] = value_;
            }
        }
        void initializeLatentMeanUpdate()
        {
            //reallocate the latent mean update with the desired size:
            initialize(latentMeanUpdate, latentDim, T(0));  
        }
        void initializeLatentLogVarUpdate()
        {
            //reallocate the latent log var update with the desired size:
            initialize(latentLogVarUpdate, latentDim, T(0));  
        }
        void initializeLatentMeanDecay()
        {
            //reallocate the latent mean decay with the desired size:
            initialize(latentMeanDecay, latentDim, T(0));  
        }
        void initializeLatentLogVarDecay()
        {
            //reallocate the latent log var decay with the desired size:
            initialize(latentLogVarDecay, latentDim, T(0));  
        }
        void initializeLatentMeanSparsity()
        {
            //reallocate the latent mean sparsity with the desired size:
            initialize(latentMeanSparsity, latentDim, T(0));  
        }
        void initializeLatentLogVarSparsity()
        {
            //reallocate the latent log var sparsity with the desired size:
            initialize(latentLogVarSparsity, latentDim, T(0));  
        }
        void initializeLatentMeanSparsityHat()
        {
            //reallocate the latent mean sparsity hat with the desired size:
            initialize(latentMeanSparsityHat, latentDim, T(0));  
        }
        void initializeLatentLogVarSparsityHat()
        {
            //reallocate the latent log var sparsity hat with the desired size:
            initialize(latentLogVarSparsityHat, latentDim, T(0));  
        }
        //update helpers for the variational auto encoder
        void updateLatentMeanGrad()
        {
            //update the latent mean grad
            for (size_t i = 0; i < latentDim; i++)
            {
                latentMeanGrad[i] = latentMeanGradPrev[i] + latentMeanGrad[i];
            }
        }
        void updateLatentLogVarGrad()
        {
            //update the latent log var grad
            for (size_t i = 0; i < latentDim; i++)
            {
                latentLogVarGrad[i] = latentLogVarGradPrev[i] + latentLogVarGrad[i];
            }
        }
        void updateLatentMeanGradPrev()
        {
            //update the latent mean grad prev
            for (size_t i = 0; i < latentDim; i++)
            {
                latentMeanGradPrev[i] = latentMeanGrad[i];
            }
        }
        void updateLatentLogVarGradPrev()
        {
            //update the latent log var grad prev
            for (size_t i = 0; i < latentDim; i++)
            {
                latentLogVarGradPrev[i] = latentLogVarGrad[i];
            }
        }
        void updateLatentMeanMomentum()
        {
            //update the latent mean momentum
            for (size_t i = 0; i < latentDim; i++)
            {
                latentMeanMomentum[i] = latentMeanMomentum[i] * this->momentum + this->learningRate * latentMeanGrad[i];
            }
        }
        void updateLatentLogVarMomentum()
        {
            //update the latent log var momentum
            for (size_t i = 0; i < latentDim; i++)
            {
                latentLogVarMomentum[i] = latentLogVarMomentum[i] * this->momentum + this->learningRate * latentLogVarGrad[i];
            }
        }
        void updateLatentMeanUpdate()
        {
            //update the latent mean update
            for (size_t i = 0; i < latentDim; i++)
            {
                latentMeanUpdate[i] = latentMeanUpdate[i] * this->momentum + this->learningRate * latentMeanGrad[i];
            }
        }
        void updateLatentLogVarUpdate()
        {
            //update the latent log var update
            for (size_t i = 0; i < latentDim; i++)
            {
                latentLogVarUpdate[i] = latentLogVarUpdate[i] * this->momentum + this->learningRate * latentLogVarGrad[i];
            }
        }
        void updateLatentMeanDecay()
        {
            //update the latent mean decay
            for (size_t i = 0; i < latentDim; i++)
            {
                latentMeanDecay[i] = latentMeanDecay[i] * this->weightDecay + this->learningRate * latentMeanGrad[i];
            }
        }
        void updateLatentLogVarDecay()
        {
            //update the latent log var decay
            for (size_t i = 0; i < latentDim; i++)
            {
                latentLogVarDecay[i] = latentLogVarDecay[i] * this->weightDecay + this->learningRate * latentLogVarGrad[i];
            }
        }
        void updateLatentMeanSparsity()
        {
            //update the latent mean sparsity
            for (size_t i = 0; i < latentDim; i++)
            {
                latentMeanSparsity[i] = latentMeanSparsity[i] * this->sparsityParam + this->learningRate * latentMeanGrad[i];
            }
        }
        void updateLatentLogVarSparsity()
        {
            //update the latent log var sparsity
            for (size_t i = 0; i < latentDim; i++)
            {
                latentLogVarSparsity[i] = latentLogVarSparsity[i] * this->sparsityParam + this->learningRate * latentLogVarGrad[i];
            }
        }   
        void updateLatentMeanSparsityHat()
        {
            //update the latent mean sparsity hat
            for (size_t i = 0; i < latentDim; i++)
            {
                latentMeanSparsityHat[i] = latentMeanSparsityHat[i] * this->sparsityParamHat + this->learningRate * latentMeanGrad[i];
            }
        }
        void updateLatentLogVarSparsityHat()
        {
            //update the latent log var sparsity hat
            for (size_t i = 0; i < latentDim; i++)
            {
                latentLogVarSparsityHat[i] = latentLogVarSparsityHat[i] * this->sparsityParamHat + this->learningRate * latentLogVarGrad[i];
            }
        }
        void updateLatentMean()
        {
            //update the latent mean
            for (size_t i = 0; i < latentDim; i++)
            {
                latentMean[i] = latentMean[i] + latentMeanUpdate[i] + latentMeanMomentum[i] + latentMeanDecay[i] + latentMeanSparsity[i] + latentMeanSparsityHat[i];
            }
        }
        void updateLatentLogVar()
        {
            //update the latent log var
            for (size_t i = 0; i < latentDim; i++)
            {
                latentLogVar[i] = latentLogVar[i] + latentLogVarUpdate[i] + latentLogVarMomentum[i] + latentLogVarDecay[i] + latentLogVarSparsity[i] + latentLogVarSparsityHat[i];
            }
        }
        void updateLatentMeanGradPrevPrev()
        {
            //update the latent mean grad prev prev
            for (size_t i = 0; i < latentDim; i++)
            {
                latentMeanGradPrevPrev[i] = latentMeanGradPrev[i];
            }
        }
        void updateLatentLogVarGradPrevPrev()
        {
            //update the latent log var grad prev prev
            for (size_t i = 0; i < latentDim; i++)
            {
                latentLogVarGradPrevPrev[i] = latentLogVarGradPrev[i];
            }
        }
        void updateLatentMeanSparsityGrad()
        {
            //update the latent mean sparsity grad
            for (size_t i = 0; i < latentDim; i++)
            {
                latentMeanSparsityGrad[i] = latentMeanSparsityGrad[i] * this->sparsityGradient + this->learningRate * latentMeanGrad[i];
            }
        }
        void updateLatentLogVarSparsityGrad()
        {
            //update the latent log var sparsity grad
            for (size_t i = 0; i < latentDim; i++)
            {
                latentLogVarSparsityGrad[i] = latentLogVarSparsityGrad[i] * this->sparsityGradient + this->learningRate * latentLogVarGrad[i];
            }
        }
        void updateLatentMeanSparsityGradHat()
        {
            //update the latent mean sparsity grad hat
            for (size_t i = 0; i < latentDim; i++)
            {
                latentMeanSparsityGradHat[i] = latentMeanSparsityGradHat[i] * this->sparsityGradientHat + this->learningRate * latentMeanGrad[i];
            }
        }
        void updateLatentLogVarSparsityGradHat()
        {
            //update the latent log var sparsity grad hat
            for (size_t i = 0; i < latentDim; i++)
            {
                latentLogVarSparsityGradHat[i] = latentLogVarSparsityGradHat[i] * this->sparsityGradientHat + this->learningRate * latentLogVarGrad[i];
            }
        }
        void updateLatentMeanSparsityGradPrev()
        {
            //update the latent mean sparsity grad prev
            for (size_t i = 0; i < latentDim; i++)
            {
                latentMeanSparsityGradPrev[i] = latentMeanSparsityGrad[i];
            }
        }
        void updateLatentLogVarSparsityGradPrev()
        {
            //update the latent log var sparsity grad prev
            for (size_t i = 0; i < latentDim; i++)
            {
                latentLogVarSparsityGradPrev[i] = latentLogVarSparsityGrad[i];
            }
        }
        void updateLatentMeanSparsityGradHatPrev()
        {
            //update the latent mean sparsity grad hat prev
            for (size_t i = 0; i < latentDim; i++)
            {
                latentMeanSparsityGradHatPrev[i] = latentMeanSparsityGradHat[i];
            }
        }   
        void updateLatentLogVarSparsityGradHatPrev()
        {
            //update the latent log var sparsity grad hat prev
            for (size_t i = 0; i < latentDim; i++)
            {
                latentLogVarSparsityGradHatPrev[i] = latentLogVarSparsityGradHat[i];
            }
        }
        void updateLatentMeanSparsityGradPrevPrev()
        {
            //update the latent mean sparsity grad prev prev
            for (size_t i = 0; i < latentDim; i++)
            {
                latentMeanSparsityGradPrevPrev[i] = latentMeanSparsityGradPrev[i];
            }
        }
        void updateLatentLogVarSparsityGradPrevPrev()
        {
            //update the latent log var sparsity grad prev prev
            for (size_t i = 0; i < latentDim; i++)
            {
                latentLogVarSparsityGradPrevPrev[i] = latentLogVarSparsityGradPrev[i];
            }
        }
        void updateLatentMeanSparsityGradHatPrevPrev()
        {
            //update the latent mean sparsity grad hat prev prev
            for (size_t i = 0; i < latentDim; i++)
            {
                latentMeanSparsityGradHatPrevPrev[i] = latentMeanSparsityGradHatPrev[i];
            }
        }
        //

        public:

        
        variational_auto_encoder(size_t inputDim, size_t hiddenDim, size_t outputDim, size_t lDim) :
        auto_encoder<T,real_x>(inputDim,hiddenDim,outputDim) , latentDim(lDim)
        {
            //initialize the latent dim
            setLatentDim(latentDim);
            //initialize the latent space
            initializeLatent();
            //initialize the latent mean
            initializeLatentMean();
            //initialize the latent log var
            initializeLatentLogVar();
            //initialize the latent mean grad
            initializeLatentMeanGrad();
            //initialize the latent log var grad
            initializeLatentLogVarGrad();
            //initialize the latent mean grad prev
            initializeLatentMeanGradPrev();
            //call the initialize weight grad function
            initializeWeightGrad();

        } 

        //destructor
        virtual ~variational_auto_encoder();
        //copy constructor
        variational_auto_encoder(const variational_auto_encoder &vae);
        //copy assignment operator
        variational_auto_encoder& operator=(const variational_auto_encoder &vae);
        //move constructor
        variational_auto_encoder(variational_auto_encoder &&vae);
        //move assignment operator
        variational_auto_encoder& operator=(variational_auto_encoder &&vae);


        //get the latent dim
        void setLatentDim(size_t latentDim);

        size_t getLatentDim() const;
        
        //override auto encoder functions
        void initializeWeightGrad() override
        {
            auto_encoder<T,real_x>::initializeWeightGrad();
            //initialize the gradients
            initializeWeight1Grad();
            initializeWeight2Grad();
            //initialize bias grads
            initializeBias1Grad();
            initializeBias2Grad();
            //initialize latent mean grad
            initializeLatentMeanGrad();
            //initialize latent log var grad
            initializeLatentLogVarGrad();
            //initialize latent mean grad prev
            initializeLatentMeanGradPrev();
            //initialize latent log var grad prev
            initializeLatentLogVarGradPrev();
            //initialize latent mean momentum
            initializeLatentMeanMomentum();
            //initialize latent log var momentum
            initializeLatentLogVarMomentum();
            //initialize latent mean update
            initializeLatentMeanUpdate();
            //initialize latent log var update
            initializeLatentLogVarUpdate();
            //initialize latent mean decay
            initializeLatentMeanDecay();
            //initialize latent log var decay
            initializeLatentLogVarDecay();
            //initialize latent mean sparsity
            initializeLatentMeanSparsity();
            //initialize latent log var sparsity
            initializeLatentLogVarSparsity();
            //initialize latent mean sparsity hat

            initializeLatentMeanSparsityHat();
            //initialize latent log var sparsity hat
            initializeLatentLogVarSparsityHat();

            //initialize weight prev
            auto_encoder<T,real_x>::initializeWeight1Prev();
            auto_encoder<T,real_x>::initializeWeight2Prev();
            //initialize bias prev
            auto_encoder<T,real_x>::initializeBias1Prev();
            auto_encoder<T,real_x>::initializeBias2Prev();
            //initialize latent mean grad prev
            initializeLatentMeanGradPrev();
            //initialize latent log var grad prev
            initializeLatentLogVarGradPrev();
            //initialize latent mean momentum
            initializeLatentMeanMomentum();
            //initialize latent log var momentum
            initializeLatentLogVarMomentum();


            //initialize latent mean update
            initializeLatentMeanUpdate();

        }
        void initializeWeightGrad(T *weightGrad, size_t size)
        {
                initialize(weightGrad, size, T(0));

        }
        void initializeWeightGrad(T *weightGrad, size_t row, size_t col)
        {
            initialize(weightGrad, row*col,T(0));

        }
        void initializeWeightGrad(T *weightGrad, size_t row, size_t col, size_t depth)
        {
            initialize(weightGrad, row*col*depth,T(0));

        }
        void initializeWeightGrad(T *weightGrad, size_t row, size_t col, size_t depth, size_t height)
        {
            initialize(weightGrad, row*col*depth*height,T(0));
        }
        void initializeWeightGrad(T *weightGrad, size_t row, size_t col, size_t depth, size_t height, size_t width)
        {
            initialize(weightGrad, row*col*depth*height*width,T(0));
        }
        void initializeWeightGrad(T *weightGrad, size_t row, size_t col, size_t depth, size_t height, size_t width, size_t length)
        {
            initialize(weightGrad, row*col*depth*height*width*length,T(0));
        }
        void initializeWeightGrad(T *weightGrad, size_t row, size_t col, size_t depth, size_t height, size_t width, size_t length, size_t dimension)
        {
            initialize(weightGrad, row*col*depth*height*width*length*dimension,T(0));
        }
        //fit with latent space

        void fit(T *input, size_t inputSize, size_t batchSize, size_t epoch)
        {
            //fit the model
            for (size_t i = 0; i < epoch; i++)
            {
                //train the model
                train(input, inputSize, batchSize);
            }
        }
        class_dist predict(T *input, size_t inputSize)
        {
            //predict the class distribution
            class_dist dist;
            //get the output
            predict(input, inputSize, dist);
            //return the class distribution
            return dist;
        }
        void predict(T *input, size_t inputSize, T *output, size_t outputSize)
        {
            //predict the output
            //get the output
            predict(input, inputSize, output, outputSize);
        }
        virtual void train(T *input, size_t inputSize, size_t batchSize)
        {
            //Train the model
            //get the batch
            T *batch = getBatch(input, inputSize, batchSize);
            //train the model
            train(batch, batchSize);

            //delete the batch ?
            delete [] batch;
            
        }
        virtual void train(T *input, size_t inputSize, size_t batchSize, size_t epoch)
        {
            //train the model
            for (size_t i = 0; i < epoch; i++)
            {
                //train the model
                train(input, inputSize, batchSize);
            }
        }
        virtual class_dist test(T *input, size_t inputSize)
        {
            //test the model
            class_dist dist;
            dist.setup(this->outputDim);
            //get the output
            //test(input, inputSize, dist);
            T* output = new T[this->outputDim]; 
            if (output != nullptr) {
            test(input, inputSize, output, this->outputDim);
            //get the class distribution
                for(size_t i = 0; i < dist.size(); i++)
                {
                    dist.accum(i,output[i]);
                }
                delete [] output;
            }
            //return the class distribution
            return dist;
        }

        virtual void test(T *input, size_t inputSize, T *output, size_t outputSize)
        {
            //test the model
            //get the output
            predict(input, inputSize, output, outputSize);

            
        }
        virtual void initializeWeight();
        virtual void initializeBias();
        virtual void initializeActivationFunction();
        virtual void initializeWeight1Grad();
        virtual void initializeWeight2Grad();
        virtual void initializeBias1Grad();
        virtual void initializeBias2Grad();

 
    };   


    template <typename T, typename real_x>
    inline auto_encoder<T, real_x>::auto_encoder(size_t inputDim, size_t hiddenDim, size_t outputDim) : inputDim(inputDim), hiddenDim(hiddenDim), outputDim(outputDim), 
    learningRate(0.1), momentum(0.9), weightDecay(0.0001), sparsityParam(0.01), beta(3), sparsityParamHat(0.01), sparsityPenalty(0), 
    sparsityGradient(0), sparsityGradientHat(0), activationFunctionPtr(nullptr), activationGradientFunctionPtr(nullptr), activationPrimeFunctionPtr(nullptr), activationPrimeGradientFunctionPtr(nullptr), activationPrimeGradientHatFunctionPtr(nullptr)      

    {
        //std::cout << "auto_encoder constructor" << std::endl;
        input = new T[inputDim];
        hidden = new T[hiddenDim];
        output = new T[outputDim];
        weight1 = new T[inputDim * hiddenDim];
        weight2 = new T[hiddenDim * outputDim];
        bias1 = new T[hiddenDim];
        bias2 = new T[outputDim];
        weight1Grad = new T[inputDim * hiddenDim];
        weight2Grad = new T[hiddenDim * outputDim];
        bias1Grad = new T[hiddenDim];
        bias2Grad = new T[outputDim];
        weight1Momentum = new T[inputDim * hiddenDim];
        weight2Momentum = new T[hiddenDim * outputDim];
        bias1Momentum = new T[hiddenDim];
        bias2Momentum = new T[outputDim];
        weight1Update = new T[inputDim * hiddenDim];
        weight2Update = new T[hiddenDim * outputDim];
        bias1Update = new T[hiddenDim];
        bias2Update = new T[outputDim];
        weight1Decay = new T[inputDim * hiddenDim];
        weight2Decay = new T[hiddenDim * outputDim];
        bias1Decay = new T[hiddenDim];
        bias2Decay = new T[outputDim];
        weight1Sparsity = new T[inputDim * hiddenDim];
        weight2Sparsity = new T[hiddenDim * outputDim];
        bias1Sparsity = new T[hiddenDim];
        bias2Sparsity = new T[outputDim];
        weight1SparsityHat = new T[inputDim * hiddenDim];
        weight2SparsityHat = new T[hiddenDim * outputDim];
        bias1SparsityHat = new T[hiddenDim];
        bias2SparsityHat = new T[outputDim];
        weight1SparsityGrad = new T[inputDim * hiddenDim];
        weight2SparsityGrad = new T[hiddenDim * outputDim];
        bias1SparsityGrad = new T[hiddenDim];
        bias2SparsityGrad = new T[outputDim];
        weight1SparsityGradHat = new T[inputDim * hiddenDim];
        weight2SparsityGradHat = new T[hiddenDim * outputDim];
        bias1SparsityGradHat = new T[hiddenDim];
        bias2SparsityGradHat = new T[outputDim];
        weight1Inc = new T[inputDim * hiddenDim];
        weight2Inc = new T[hiddenDim * outputDim];
        weight1GradPrev = new T[inputDim * hiddenDim];
        weight2GradPrev = new T[hiddenDim * outputDim];
        bias1GradPrev = new T[hiddenDim];
        bias2GradPrev = new T[outputDim];
        bias1Inc = new T[hiddenDim];
        bias2Inc = new T[outputDim];

        weight1Prev = new T[inputDim * hiddenDim];
        weight2Prev = new T[hiddenDim * outputDim];

        bias1Prev = new T[hiddenDim];
        bias2Prev = new T[outputDim];
        weight1GradPrevPrev = new T[inputDim * hiddenDim];
        weight2GradPrevPrev = new T[hiddenDim * outputDim];
        bias1GradPrevPrev = new T[hiddenDim];
        bias2GradPrevPrev = new T[outputDim];
        
        //initialize the biases.
        memset(bias1, 0, sizeof(T) * hiddenDim);
        memset(bias2, 0, sizeof(T) * outputDim);
        memset(bias1Grad, 0, sizeof(T) * hiddenDim);
        memset(bias2Grad, 0, sizeof(T) * outputDim);
        memset(bias1Momentum, 0, sizeof(T) * hiddenDim);
        memset(bias2Momentum, 0, sizeof(T) * outputDim);
        memset(bias1Update, 0, sizeof(T) * hiddenDim);
        memset(bias2Update, 0, sizeof(T) * outputDim);
        memset(bias1Decay, 0, sizeof(T) * hiddenDim);
        memset(bias2Decay, 0, sizeof(T) * outputDim);

        initializeWeight();
        initializeBias();
        initializeActivationFunction();
        initializeWeightGrad();
        initializeWeight1Inc();
        initializeWeight2Inc();
        initializeBias1Inc();
        initializeBias2Inc();
        initializeWeight1GradPrev();
        initializeWeight2GradPrev();

        initializeBias1GradPrev();
        initializeBias2GradPrev();
        initializeWeight1Decay();
        initializeWeight2Decay();

        initializeBias1Decay();
        initializeBias2Decay();
        initializeWeight1Sparsity();
        initializeWeight2Sparsity();

        initializeBias1Sparsity();
        initializeBias2Sparsity();

        initializeWeight1SparsityHat();
        initializeWeight2SparsityHat();

        initializeBias1SparsityHat();
        initializeBias2SparsityHat();


    }   
    
    template <typename T, typename real_x>
    auto_encoder<T, real_x>::~auto_encoder()
    {
        if (input != nullptr)
        {
            delete[] input;
            input = nullptr;
        }
        if (hidden != nullptr)
        {
            delete[] hidden;
            hidden = nullptr;
        }
        if (output != nullptr)
        {
            delete[] output;
            output = nullptr;
        }
        if (weight1 != nullptr)
        {
            delete[] weight1;
            weight1 = nullptr;
        }
        if (weight2 != nullptr)
        {
            delete[] weight2;
            weight2 = nullptr;
        }
        if (bias1 != nullptr)
        {
            delete[] bias1;
            bias1 = nullptr;
        }
        if (bias2 != nullptr)
        {
            delete[] bias2;
            bias2 = nullptr;
        }
        if (weight1Grad != nullptr)
        {
            delete[] weight1Grad;
            weight1Grad = nullptr;
        }
        if (weight2Grad != nullptr)
        {
            delete[] weight2Grad;
            weight2Grad = nullptr;
        }
        if (bias1Grad != nullptr)
        {
            delete[] bias1Grad;
            bias1Grad = nullptr;
        }
        if (bias2Grad != nullptr)
        {
            delete[] bias2Grad;
            bias2Grad = nullptr;
        }
        if (weight1Momentum != nullptr)
        {
            delete[] weight1Momentum;
            weight1Momentum = nullptr;
        }
        if (weight2Momentum != nullptr)
        {
            delete[] weight2Momentum;
            weight2Momentum = nullptr;
        }

        if (bias1Momentum != nullptr)
        {
            delete[] bias1Momentum;
            bias1Momentum = nullptr;
        }   

        if (bias2Momentum != nullptr)
        {
            delete[] bias2Momentum;
            bias2Momentum = nullptr;
        }   

        if (weight1Update != nullptr)
        {
            delete[] weight1Update;
            weight1Update = nullptr;
        }   

        if (weight2Update != nullptr)
        {
            delete[] weight2Update;
            weight2Update = nullptr;
        }   

        if (bias1Update != nullptr)
        {
            delete[] bias1Update;
            bias1Update = nullptr;
        }   
        //bias2Update
        if (bias2Update != nullptr)
        {
            delete[] bias2Update;
            bias2Update = nullptr;
        }
        if (weight1Decay != nullptr)
        {
            delete[] weight1Decay;
            weight1Decay = nullptr;
        }
        if (weight2Decay != nullptr)
        {
            delete[] weight2Decay;
            weight2Decay = nullptr;
        }
        //bias1Decay
        if (bias1Decay != nullptr)
        {
            delete[] bias1Decay;
            bias1Decay = nullptr;
        }
        //bias2Decay
        if (bias2Decay != nullptr)
        {
            delete[] bias2Decay;
            bias2Decay = nullptr;
        }
        //weight1Sparsity
        if (weight1Sparsity != nullptr)
        {
            delete[] weight1Sparsity;
            weight1Sparsity = nullptr;
        }
        //weight2Sparsity
        if (weight2Sparsity != nullptr)
        {
            delete[] weight2Sparsity;
            weight2Sparsity = nullptr;
        }
        //bias1Sparsity
        if (bias1Sparsity != nullptr)
        {
            delete[] bias1Sparsity;
            bias1Sparsity = nullptr;
        }
        //bias2Sparsity
        if (bias2Sparsity != nullptr)
        {
            delete[] bias2Sparsity;
            bias2Sparsity = nullptr;
        }
        //weight1SparsityHat
        if (weight1SparsityHat != nullptr)
        {
            delete[] weight1SparsityHat;
            weight1SparsityHat = nullptr;
        }
        //weight2SparsityHat
        if (weight2SparsityHat != nullptr)
        {
            delete[] weight2SparsityHat;
            weight2SparsityHat = nullptr;
        }
        //bias1SparsityHat
        if (bias1SparsityHat != nullptr)
        {
            delete[] bias1SparsityHat;
            bias1SparsityHat = nullptr;
        }
        //bias2SparsityHat
        if (bias2SparsityHat != nullptr)
        {
            delete[] bias2SparsityHat;
            bias2SparsityHat = nullptr;
        }
        //weight1SparsityGrad
        if (weight1SparsityGrad != nullptr)
        {
            delete[] weight1SparsityGrad;
            weight1SparsityGrad = nullptr;
        }
        //weight2SparsityGrad
        if (weight2SparsityGrad != nullptr)
        {
            delete[] weight2SparsityGrad;
            weight2SparsityGrad = nullptr;
        }
        //bias1SparsityGrad
        if (bias1SparsityGrad != nullptr)
        {
            delete[] bias1SparsityGrad;
            bias1SparsityGrad = nullptr;
        }
        //bias2SparsityGrad
        if (bias2SparsityGrad != nullptr)
        {
            delete[] bias2SparsityGrad;
            bias2SparsityGrad = nullptr;
        }
        //weight1SparsityGradHat
        if (weight1SparsityGradHat != nullptr)
        {
            delete[] weight1SparsityGradHat;
            weight1SparsityGradHat = nullptr;
        }
        //weight2SparsityGradHat
        if (weight2SparsityGradHat != nullptr)
        {
            delete[] weight2SparsityGradHat;
            weight2SparsityGradHat = nullptr;
        }
        //weight1Inc
        if (weight1Inc != nullptr)
        {
            delete[] weight1Inc;
            weight1Inc = nullptr;
        }
        //weight2Inc
        if (weight2Inc != nullptr)
        {
            delete[] weight2Inc;
            weight2Inc = nullptr;
        }
        //weight1GradPrev
        if (weight1GradPrev != nullptr)
        {
            delete[] weight1GradPrev;
            weight1GradPrev = nullptr;
        }
        //weight2GradPrev
        if (weight2GradPrev != nullptr)
        {
            delete[] weight2GradPrev;
            weight2GradPrev = nullptr;
        }
        //bias1Inc
        if (bias1Inc != nullptr)
        {
            delete[] bias1Inc;
            bias1Inc = nullptr;
        }
        //bias2Inc
        if (bias2Inc != nullptr)
        {
            delete[] bias2Inc;
            bias2Inc = nullptr;
        }
        //bias1GradPrev
        if (bias1GradPrev != nullptr)
        {
            delete[] bias1GradPrev;
            bias1GradPrev = nullptr;
        }
        //bias2GradPrev
        if (bias2GradPrev != nullptr)
        {
            delete[] bias2GradPrev;
            bias2GradPrev = nullptr;
        }

        //prevprev
        if (weight1GradPrevPrev != nullptr)
        {
            delete[] weight1GradPrevPrev;
            weight1GradPrevPrev = nullptr;
        }
        //prevprev
        if (weight2GradPrevPrev != nullptr)
        {
            delete[] weight2GradPrevPrev;
            weight2GradPrevPrev = nullptr;
        }
        
        //done
    }

    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::initializeWeight()
    {

        initializeWeight(weight1, inputDim, hiddenDim);
        initializeWeight(weight2, hiddenDim, outputDim);
        initializeWeight(weight1Grad, inputDim, hiddenDim);
        initializeWeight(weight2Grad, hiddenDim, outputDim);
        initializeWeight1Momentum();
        initializeWeight2Momentum();
        initializeWeight1Update();
        initializeWeight2Update();
        initializeWeight1Decay();
        initializeWeight2Decay();
        initializeWeight1Sparsity();
        initializeWeight2Sparsity();
        initializeWeight1SparsityHat();
        initializeWeight2SparsityHat();
        initializeWeight1SparsityGrad();
        initializeWeight2SparsityGrad();
        initializeWeight1SparsityGradHat();
        initializeWeight2SparsityGradHat();
        initializeWeight1Prev();
        initializeWeight2Prev();

        initializeWeight1Grad();
        initializeWeight2Grad();
        initializeWeight1GradPrev();
        initializeWeight2GradPrev();
        initializeWeight1GradPrevPrev();
        initializeWeight2GradPrevPrev();
        initializeWeight1Inc();
        initializeWeight2Inc();
        //done

    }

    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::initializeBias()
    {
        initializeBias(input, inputDim);
        initializeBias(hidden, hiddenDim);
        initializeBias(output, outputDim);
        //done

    }

    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::initializeWeight(T *weight, size_t size)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 1);
        for (size_t i = 0; i < size; i++)
        {
            weight[i] = dis(gen);
        }
        //done
    }

    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::initializeBias(T *bias, size_t size)
    {
        for (size_t i = 0; i < size; i++)
        {
            bias[i] = 0;
        }
        //done

    }
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::initializeWeight(T *weight, size_t row, size_t col)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 1);
        for (size_t i = 0; i < row; i++)
        {
            for (size_t j = 0; j < col; j++)
            {
                weight[i * col + j] = dis(gen);
            }
        
        }//for
        //done
    }

    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::train(T *input, T *output, size_t size)
    {
        if ( input==nullptr || output==nullptr)
        {
            std::cout << "[-] autoencoder - error in training - no input." << std::endl;
            return;
        }
        feedforward(input, output, size);
        backprop(input, output, size);
        //done
    }

    //train with matrix and fill classdist 
    template <typename T, typename real_x> 
    void auto_encoder<T,real_x>::train( matrix<T>& input , class_dist& output)
    {
        output.setup(outputDim);//setup output

        T* inputarray  =  new T[input.cols()];
        T* outputarray =  new T[outputDim];
        if ( inputarray==nullptr || outputarray==nullptr)
        {
            std::cout << "error in train matrix" << std::endl;
            return;
        }

        for (size_t i = 0; i < input.rows(); i++)
        {
            for (size_t j = 0; j < input.cols(); j++)
            {
                inputarray[j]=input(i,j);


            }

            train(inputarray,outputarray,1);
         }
        
        for (size_t j = 0; j < outputDim && j<output.size(); j++)
        {
                    output.set(j,outputarray[j] );
        }

        if (inputarray!=nullptr)
        {
            delete[] inputarray;
            inputarray=nullptr;
        }   
        if (outputarray!=nullptr)
        {
            delete[] outputarray;
            outputarray=nullptr;
        }
 
        //done
    }

    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::test(T *input, T *output, size_t size)
    {
        
        feedforward(input, output, size);

        //done
    }

    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::test(matrix<T>& input, class_dist& output)
    {
        T* inputarray  =  new T[input.cols()];
        T* outputarray =  new T[outputDim];
        
        if ( inputarray==nullptr || outputarray==nullptr)
        {
            std::cout << "error in test matrix" << std::endl;
            return;
        }

        bzero(outputarray,sizeof(T)*outputDim);
        bzero(inputarray,sizeof(T)*input.cols());
        
        for (size_t i = 0; i < input.rows(); i++)
        {
            for (size_t j = 0; j < input.cols(); j++)
            {
                inputarray[j]=input(i,j);       
            }
            test(inputarray,outputarray,1);
            for (size_t j = 0; j < outputDim && j<output.size(); j++)
            {
                output.accum(j,outputarray[j] / outputDim);
            }   
        }
        if (inputarray!=nullptr)
        {
            delete[] inputarray;
            inputarray=nullptr;
        }
        if (outputarray!=nullptr)
        {
            delete[] outputarray;
            outputarray=nullptr;
        }
        //done
    }

    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::feedforward(T *input, T *output, size_t size)
    {
        //std::cout << "feedforward" << std::endl;

        for (size_t i = 0; i < size; i++)
        {
            for (size_t j = 0; j < hiddenDim; j++)
            {
                hidden[j] = 0;
                for (size_t k = 0; k < inputDim; k++)
                {
                    hidden[j] += input[i * inputDim + k] * weight1[k * hiddenDim + j];
                }
                hidden[j] = sigmoid(hidden[j]);
                //hidden[j] = relu(hidden[j]); 

            }
            for (size_t j = 0; j < outputDim; j++)
            {
                output[j] = 0;
                for (size_t k = 0; k < hiddenDim; k++)
                {
                    output[j] += hidden[k] * weight2[k * outputDim + j];
                }
                output[j] = sigmoid(output[j]);
            }
             
            //std::cout << "feedforward" << std::endl;
            
        }
        //done

    }

    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::backprop(T *input, T *output, size_t size)
    {

        T *delta1 = new T[inputDim];
        T *delta2 = new T[hiddenDim];
        T *delta3 = new T[outputDim];
        T *weight1Grad = new T[inputDim * hiddenDim];
        T *weight2Grad = new T[hiddenDim * outputDim];
        T *bias1Grad = new T[hiddenDim];
        T *bias2Grad = new T[outputDim];
        T *weight1GradPrev = new T[inputDim * hiddenDim];
        T *weight2GradPrev = new T[hiddenDim * outputDim];
        T *bias1GradPrev = new T[hiddenDim];
        T *bias2GradPrev = new T[outputDim];
        T *weight1Inc = new T[inputDim * hiddenDim];
        T *weight2Inc = new T[hiddenDim * outputDim];
        T *bias1Inc = new T[hiddenDim];
        T *bias2Inc = new T[outputDim];

        if(!delta1 || !delta2 || !delta3 || !weight1Grad || !weight2Grad || !bias1Grad || !bias2Grad || !weight1GradPrev || !weight2GradPrev || !bias1GradPrev || !bias2GradPrev || !weight1Inc || !weight2Inc || !bias1Inc || !bias2Inc)
        {
            throw std::runtime_error("error in backprop");
        }
        initializeWeight(delta1, inputDim);
        initializeWeight(delta2, hiddenDim);
        initializeWeight(delta3, outputDim);

        initializeWeight(weight1Grad, inputDim * hiddenDim);
        initializeWeight(weight2Grad, hiddenDim * outputDim);
        initializeWeight(bias1Grad, hiddenDim);
        initializeWeight(bias2Grad, outputDim);
        initializeWeight(weight1GradPrev, inputDim * hiddenDim);
        initializeWeight(weight2GradPrev, hiddenDim * outputDim);
        initializeWeight(bias1GradPrev, hiddenDim);
        initializeWeight(bias2GradPrev, outputDim);
        initializeWeight(weight1Inc, inputDim * hiddenDim);
        initializeWeight(weight2Inc, hiddenDim * outputDim);
        initializeWeight(bias1Inc, hiddenDim);
        initializeWeight(bias2Inc, outputDim);
        for (size_t i = 0; i < size; i++)
        {
            for (size_t j = 0; j < outputDim; j++)
            {
                delta3[j] = output[i * outputDim + j] * (1 - output[i * outputDim + j]) * (output[i * outputDim + j] - input[i * outputDim + j]);
                for (size_t k = 0; k < hiddenDim; k++)
                {
                    weight2Grad[k * outputDim + j] += hidden[k] * delta3[j];
                }
                bias2Grad[j] += delta3[j];
            }
            for (size_t j = 0; j < hiddenDim; j++)
            {
                delta2[j] = hidden[i * hiddenDim + j] * (1 - hidden[i * hiddenDim + j]);
                for (size_t k = 0; k < outputDim; k++)
                {
                    delta2[j] += delta3[k] * weight2[j * outputDim + k];
                }
                for (size_t k = 0; k < inputDim; k++)
                {
                    weight1Grad[k * hiddenDim + j] += input[i * inputDim + k] * delta2[j];
                }
                bias1Grad[j] += delta2[j];
            }
        }

        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            weight1Inc[i] = momentum * weight1Inc[i] + learningRate * weight1Grad[i];
            weight1[i] -= weight1Inc[i];
        }
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            weight2Inc[i] = momentum * weight2Inc[i] + learningRate * weight2Grad[i];
            weight2[i] -= weight2Inc[i];
        }
        for (size_t i = 0; i < hiddenDim; i++)
        {
            bias1Inc[i] = momentum * bias1Inc[i] + learningRate * bias1Grad[i];
            bias1[i] -= bias1Inc[i];
        }
        for (size_t i = 0; i < outputDim; i++)
        {
            bias2Inc[i] = momentum * bias2Inc[i] + learningRate * bias2Grad[i];
            bias2[i] -= bias2Inc[i];
        }
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            weight1GradPrev[i] = weight1Grad[i];
        }
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            weight2GradPrev[i] = weight2Grad[i];
        }
        for (size_t i = 0; i < hiddenDim; i++)
        {
            bias1GradPrev[i] = bias1Grad[i];
        }
        for (size_t i = 0; i < outputDim; i++)
        {
            bias2GradPrev[i] = bias2Grad[i];
        }
        //std::cout << "backprop" << std::endl;
        //std::cout << "backprop" << std::endl;
        

        delete[] delta1;
        delete[] delta2;
        delete[] delta3;
        delete[] weight1Grad;
        delete[] weight2Grad;
        delete[] bias1Grad;
        delete[] bias2Grad;
        delete[] weight1GradPrev;
        delete[] weight2GradPrev;
        delete[] bias1GradPrev;
        delete[] bias2GradPrev;
        delete[] weight1Inc;
        delete[] weight2Inc;
        delete[] bias1Inc;
        delete[] bias2Inc;
 
        //done

    }

    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::sigmoid(T x)
    {
        return T(1.) / (T(1.) + exp(-x));
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::sigmoidPrime(T x)
    {
        return sigmoid(x) * (1 - sigmoid(x));
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::tanh(T x)
    {
        return T(exp(x) - exp(-x)) / (exp(x) + exp(-x));
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::tanhPrime(T x)
    {
        return T(1 - tanh(x) * tanh(x));
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::relu(T x)
    {
        return T(x > 0 ? x : 0);
    }

    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::reluPrime(T x)
    {
        return T(x > 0 ? 1 : 0);
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::leakyRelu(T x)
    {
        return x > 0 ? x : 0.01 * x;
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::leakyReluPrime(T x)
    {
        return x > 0 ? 1 : 0.01;
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::softplus(T x)
    {
        return log(1 + exp(x));
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::softplusPrime(T x)
    {
        return sigmoid(x);
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::linear(T x)
    {
        return x;
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::linearGradient(T x)
    {
        return 1;
    }

    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::gaussian(T x)
    {
        return exp(-x * x);
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::gaussianPrime(T x)
    {
        return -2 * x * exp(-x * x);
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::sinusoid(T x)
    {
        return sin(x);
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::sinusoidPrime(T x)
    {
        return cos(x);
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::sinc(T x)
    {
        return x == 0 ? 1 : cos(x) / x;
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::sincPrime(T x)
    {
        return x == 0 ? 0 : (x * sin(x) - cos(x)) / (x * x);
    }

    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::bentIdentity(T x)
    {
        return (sqrt(x * x + 1) - 1) / 2 + x;
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::bentIdentityPrime(T x)
    {
        return x / (2 * sqrt(x * x + 1)) + 1;
    }

    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::softExponential(T x)
    {
        return x < 0 ? log(1 + exp(x)) : x;
    }
    template <typename T, typename real_x>
    T auto_encoder<T,real_x>::softExponentialPrime(T x)
    {
        return x < 0 ? exp(x) / (1 + exp(x)) : 1;
    }

    template <typename T, typename real_x>
    size_t auto_encoder<T,real_x>::getInputDim() const
    {
        return inputDim;
    }
    template <typename T, typename real_x>
    size_t auto_encoder<T,real_x>::getHiddenDim() const
    {
        return hiddenDim;
    }
    template <typename T, typename real_x>
    size_t auto_encoder<T,real_x>::getOutputDim() const
    {
        return outputDim;
    }

    template <typename T, typename real_x>
    real_x auto_encoder<T,real_x>::getLearningRate() const
    {
        return learningRate;
    }

    template <typename T, typename real_x>
    real_x auto_encoder<T,real_x>::getMomentum() const
    {
        return momentum;
    }

    template <typename T, typename real_x>
    T *auto_encoder<T,real_x>::getWeight1() const
    {
        return weight1;
    }
    template <typename T, typename real_x>
    T *auto_encoder<T,real_x>::getWeight2() const
    {
        return weight2;
    }
    template <typename T, typename real_x>
    T *auto_encoder<T,real_x>::getBias1() const
    {
        return bias1;
    }
    template <typename T, typename real_x>
    T *auto_encoder<T,real_x>::getBias2() const
    {
        return bias2;
    }
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::setInputDim(size_t inputDim)
    {
        this->inputDim = inputDim;
    }
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::setHiddenDim(size_t hiddenDim)
    {
        this->hiddenDim = hiddenDim;
    }
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::setOutputDim(size_t outputDim)
    {
        this->outputDim = outputDim;
    }
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::setLearningRate(real_x learningRate)
    {
        this->learningRate = learningRate;
    }
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::setMomentum(real_x momentum)
    {
        this->momentum = momentum;
    }
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::setWeight1(T *weight1)
    {
        if (this->weight1 != nullptr)
        {
            delete[] this->weight1;
        }
        this->weight1 = weight1;

        // initialize weight1Inc
        if (weight1Inc != nullptr)
        {
            delete[] weight1Inc;
        }
        weight1Inc = new T[inputDim * hiddenDim];
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            weight1Inc[i] = T(0.);
        }

        // initialize weight1Grad
        if (weight1Grad != nullptr)
        {
            delete[] weight1Grad;
        }
        weight1Grad = new T[inputDim * hiddenDim];
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            weight1Grad[i] = T(0.);
        }

        // initialize weight1GradPrev
        if (weight1GradPrev != nullptr)
        {
            delete[] weight1GradPrev;
        }
        weight1GradPrev = new T[inputDim * hiddenDim];
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            weight1GradPrev[i] = T(0.);
        }
    }

    template <typename T, typename real_x>

    void auto_encoder<T,real_x>::setWeight2(T *weight2)
    {

        if (this->weight2 != nullptr)
        {
            delete[] this->weight2;
        }

        this->weight2 = weight2;

        // initialize weight2Inc
        if (weight2Inc != nullptr)
        {
            delete[] weight2Inc;
        }
        weight2Inc = new T[hiddenDim * outputDim];
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            weight2Inc[i] = T(0.);
        }

        // initialize weight2Grad
        if (weight2Grad != nullptr)
        {
            delete[] weight2Grad;
        }
        weight2Grad = new T[hiddenDim * outputDim];
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            weight2Grad[i] = T(0.);
        }

        // initialize weight2GradPrev
        if (weight2GradPrev != nullptr)
        {
            delete[] weight2GradPrev;
        }
        weight2GradPrev = new T[hiddenDim * outputDim];
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            weight2GradPrev[i] = T(0.);
        }
    }

    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::setBias1(T *bias1)
    {
        if (this->bias1 != nullptr)
        {
            delete[] this->bias1;
        }
        this->bias1 = bias1;

        // initialize bias1Inc
        if (bias1Inc != nullptr)
        {
            delete[] bias1Inc;
        }
        bias1Inc = new T[hiddenDim];
        for (size_t i = 0; i < hiddenDim; i++)
        {
            bias1Inc[i] = T(0.);
        }

        // initialize bias1Grad
        if (bias1Grad != nullptr)
        {
            delete[] bias1Grad;
        }
        bias1Grad = new T[hiddenDim];
        for (size_t i = 0; i < hiddenDim; i++)
        {
            bias1Grad[i] = T(0.);
        }

        // initialize bias1GradPrev
        if (bias1GradPrev != nullptr)
        {
            delete[] bias1GradPrev;
        }
        bias1GradPrev = new T[hiddenDim];
        for (size_t i = 0; i < hiddenDim; i++)
        {
            bias1GradPrev[i] = T(0.);
        }
    }
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::setBias2(T *bias2)
    {
        if (this->bias2 != nullptr)
        {
            delete[] this->bias2;
        }
        this->bias2 = bias2;

        // initialize bias2Inc
        if (bias2Inc != nullptr)
        {
            delete[] bias2Inc;
        }
        bias2Inc = new T[outputDim];
        for (size_t i = 0; i < outputDim; i++)
        {
            bias2Inc[i] = T(0.);
        }

        // initialize bias2Grad
        if (bias2Grad != nullptr)
        {
            delete[] bias2Grad;
        }
        bias2Grad = new T[outputDim];
        for (size_t i = 0; i < outputDim; i++)
        {
            bias2Grad[i] = T(0.);
        }

        // initialize bias2GradPrev
        if (bias2GradPrev != nullptr)
        {
            delete[] bias2GradPrev;
        }
        bias2GradPrev = new T[outputDim];
        for (size_t i = 0; i < outputDim; i++)
        {
            bias2GradPrev[i] = T(0.);
        }
    }

    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::setWeight1Inc(T *weight1Inc)
    {
        if (this->weight1Inc != nullptr)
        {
            delete[] this->weight1Inc;
        }
        this->weight1Inc = weight1Inc;
    }
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::setWeight2Inc(T *weight2Inc)
    {
        if (this->weight2Inc != nullptr)
        {
            delete[] this->weight2Inc;
        }
        this->weight2Inc = weight2Inc;
    }
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::setBias1Inc(T *bias1Inc)
    {
        if (this->bias1Inc != nullptr)
        {
            delete[] this->bias1Inc;
        }
        this->bias1Inc = bias1Inc;
    }
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::setBias2Inc(T *bias2Inc)
    {
        if (this->bias2Inc != nullptr)
        {
            delete[] this->bias2Inc;
        }
        this->bias2Inc = bias2Inc;
    }
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::setWeight1Grad(T *weight1Grad)
    {
        if (this->weight1Grad != nullptr)
        {
            delete[] this->weight1Grad;
        }
        this->weight1Grad = weight1Grad;
    }

    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::setWeight2Grad(T *weight2Grad)
    {
        if (this->weight2Grad != nullptr)
        {
            delete[] this->weight2Grad;
        }
        this->weight2Grad = weight2Grad;
    }

    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::setBias1Grad(T *bias1Grad)
    {
        if (this->bias1Grad != nullptr)
        {
            delete[] this->bias1Grad;
        }
        this->bias1Grad = bias1Grad;
    }

    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::setBias2Grad(T *bias2Grad)
    {
        if (this->bias2Grad != nullptr)
        {
            delete[] this->bias2Grad;
        }
        this->bias2Grad = bias2Grad;
    }  


    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::dump(  std::ostream &out) const

    {
        out << std::string("inputDim: ") << std::to_string ( inputDim ) << std::endl;
        out << std::string("hiddenDim: ")<<std::to_string ( hiddenDim )  << std::endl;
        out << std::string("outputDim: ") << std::to_string(outputDim)<< std::endl;
        out << std::string("weight1: ") << std::endl;
 

        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            out << std::to_string( weight1[i] ) << char(' ');
        }
        out << std::endl;
        out << std::string("weight2: ") << std::endl;
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            out << std::to_string( weight2[i] ) << char(' ')    ;

        }
        out << std::endl;
        out << "bias1: " << std::endl;
        for (size_t i = 0; i < hiddenDim; i++)
        {
            out<< std::to_string( bias1[i] ) << char(' ')    ;
        }
        out<< std::endl;
        
        out<< std::string("bias2: ") << std::endl;
        
        for (size_t i = 0; i < outputDim; i++)
        {
            out <<std::to_string( bias2[i] )<< char(' ')    ;

        }
        out << std::endl;

        out << std::string("weight1Inc: ") << std::endl;
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {

            out << std::to_string(weight1Inc[i]) << char(' ')    ;  
        }
        out << std::endl;
        out << std::string("weight2Inc: ") << std::endl;
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            out << std::to_string(weight2Inc[i] )<< char(' ')    ;
        }
        out << std::endl;
        out << "bias1Inc: " << std::endl;
        for (size_t i = 0; i < hiddenDim; i++)
        {
            out << std::to_string(bias1Inc[i]) << char(' ')    ;
        }
        out << std::endl;
        out << "bias2Inc: " << std::endl;
        for (size_t i = 0; i < outputDim; i++)
        {
            out << bias2Inc[i] << " ";
        }
        out << std::endl;



        out << "weight1Grad: " << std::endl;

        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            out << weight1Grad[i] << " ";
        }
        out << std::endl;
        out << "weight2Grad: " << std::endl;
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            out << weight2Grad[i] << " ";
        }
        out << std::endl;
        out << "bias1Grad: " << std::endl;
        for (size_t i = 0; i < hiddenDim; i++)
        {
            out << bias1Grad[i] << " ";
        }
        out << std::endl;
        out << "bias2Grad: " << std::endl;
        for (size_t i = 0; i < outputDim; i++)
        {
            out << bias2Grad[i] << " ";
        }
        out << std::endl;
        
        out << "weight1GradPrev: " << std::endl;
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            out << weight1GradPrev[i] << " ";
        }
        out << std::endl;
        out << "weight2GradPrev: " << std::endl;
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            out << weight2GradPrev[i] << " ";
        }
        out << std::endl;
        out << "bias1GradPrev: " << std::endl;
        for (size_t i = 0; i < hiddenDim; i++)
        {
            out << bias1GradPrev[i] << " ";
        }
        out << std::endl;
        out << "bias2GradPrev: " << std::endl;
        for (size_t i = 0; i < outputDim; i++)
        {
            out << bias2GradPrev[i] << " ";
        }
        out << std::endl;



        out << "weight1GradPrevPrev: " << std::endl;    


        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            out << weight1GradPrevPrev[i] << " ";
        }   
        out << std::endl;

        out << "weight2GradPrevPrev: " << std::endl;            


        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            out << weight2GradPrevPrev[i] << " ";
        }
        out << std::endl;
        out << "bias1GradPrevPrev: " << std::endl;
        for (size_t i = 0; i < hiddenDim; i++)
        {
            out << bias1GradPrevPrev[i] << " ";
        }


        out << std::endl;
        out << "bias2GradPrevPrev: " << std::endl;
        for (size_t i = 0; i < outputDim; i++)
        {
            out << bias2GradPrevPrev[i] << " ";
        }
        out << std::endl;

        //SAVE PARAMETERS (learning rate, momentum, etc.)
        out << "learningRate: " << learningRate << std::endl;
        out << "momentum: " << momentum << std::endl;
        out << "weightDecay: " << weightDecay << std::endl;
        out << "sparsityParam: " << sparsityParam << std::endl;
        out << "beta: " << beta << std::endl;
         
        
        //all members are saved

        
    }
    //load & save 
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::load(std::istream &in)
    {
        std::string line;
        std::getline(in, line);
        inputDim = std::stoi(line.substr(line.find(':') + 1));
        std::getline(in, line);
        hiddenDim = std::stoi(line.substr(line.find(':') + 1));
        std::getline(in, line);
        outputDim = std::stoi(line.substr(line.find(':') + 1));
        std::getline(in, line);
        std::getline(in, line);
        std::stringstream ss(line);
        if (weight1 != nullptr)
        {
            delete[] weight1;
        }
        weight1 = new T[inputDim * hiddenDim];
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            ss >> weight1[i];
        }
        std::getline(in, line);
        std::getline(in, line);
        ss = std::stringstream(line);
        if (weight2 != nullptr)
        {
            delete[] weight2;
        }
        weight2 = new T[hiddenDim * outputDim];
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            ss >> weight2[i];
        }
        std::getline(in, line);
        std::getline(in, line);
        ss = std::stringstream(line);
        if (bias1 != nullptr)
        {
            delete[] bias1;
        }
        bias1 = new T[hiddenDim];
        for (size_t i = 0; i < hiddenDim; i++)
        {
            ss >> bias1[i];
        }
        std::getline(in, line);
        std::getline(in, line);
        ss = std::stringstream(line);
        if (bias2 != nullptr)
        {
            delete[] bias2;
        }
        bias2 = new T[outputDim];
        for (size_t i = 0; i < outputDim; i++)
        {
            ss >> bias2[i];
        }
        std::getline(in, line);
        std::getline(in, line);
        ss = std::stringstream(line);
        if (weight1Inc != nullptr)
        {
            delete[] weight1Inc;
        }
        weight1Inc = new T[inputDim * hiddenDim];
        for (size_t i = 0; i < inputDim ; i++)
        {
            ss >> weight1Inc[i];
        }   
        std::getline(in, line);
        std::getline(in, line);
        ss = std::stringstream(line);
        if (weight2Inc != nullptr)
        {
            delete[] weight2Inc;
        }
        weight2Inc = new T[hiddenDim * outputDim];
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            ss >> weight2Inc[i];
        }
        std::getline(in, line);
        std::getline(in, line);
        ss = std::stringstream(line);
        if (bias1Inc != nullptr)
        {
            delete[] bias1Inc;
        }   
        bias1Inc = new T[hiddenDim];
        for (size_t i = 0; i < hiddenDim; i++)
        {
            ss >> bias1Inc[i];
        }

        std::getline(in, line);
        std::getline(in, line);
        ss = std::stringstream(line);


        if (bias2Inc != nullptr)
        {
            delete[] bias2Inc;
        }       

        bias2Inc = new T[outputDim];    
        for (size_t i = 0; i < outputDim; i++)
        {
            ss >> bias2Inc[i];
        }
        std::getline(in, line);
        std::getline(in, line);
        ss = std::stringstream(line);
        if (weight1Grad != nullptr)
        {
            delete[] weight1Grad;
        }
        weight1Grad = new T[inputDim * hiddenDim];
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            ss >> weight1Grad[i];
        }
        std::getline(in, line);
        std::getline(in, line);
        ss = std::stringstream(line);
        if (weight2Grad != nullptr)
        {
            delete[] weight2Grad;
        }
        weight2Grad = new T[hiddenDim * outputDim];
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            ss >> weight2Grad[i];
        }
        std::getline(in, line);
        std::getline(in, line);
        ss = std::stringstream(line);
        if (bias1Grad != nullptr)
        {
            delete[] bias1Grad;
        }
        bias1Grad = new T[hiddenDim];
        for (size_t i = 0; i < hiddenDim; i++)
        {
            ss >> bias1Grad[i];
        }
        //std::cout << "bias1Grad: " << std::endl;

        std::getline(in, line);
        std::getline(in, line);
        ss = std::stringstream(line);
        if (bias2Grad != nullptr)
        {
            delete[] bias2Grad;
        }
        bias2Grad = new T[outputDim];
        for (size_t i = 0; i < outputDim; i++)
        {
            ss >> bias2Grad[i];
        }
        std::getline(in, line);
        std::getline(in, line);
        ss = std::stringstream(line);
        if (weight1GradPrev != nullptr)
        {
            delete[] weight1GradPrev;
        }
        weight1GradPrev = new T[inputDim * hiddenDim];
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            ss >> weight1GradPrev[i];
        }
        std::getline(in, line);
        std::getline(in, line);
        ss = std::stringstream(line);
        if (weight2GradPrev != nullptr)
        {
            delete[] weight2GradPrev;
        }
        weight2GradPrev = new T[hiddenDim * outputDim];
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            ss >> weight2GradPrev[i];
        }
        std::getline(in, line);
        std::getline(in, line);
        ss = std::stringstream(line);
        if (bias1GradPrev != nullptr)
        {
            delete[] bias1GradPrev;
        }
        bias1GradPrev = new T[hiddenDim];
        for (size_t i = 0; i < hiddenDim; i++)
        {
            ss >> bias1GradPrev[i];
        }
        std::getline(in, line);
        std::getline(in, line);
        ss = std::stringstream(line);
        if (bias2GradPrev != nullptr)
        {
            delete[] bias2GradPrev;
        }
        bias2GradPrev = new T[outputDim];
        for (size_t i = 0; i < outputDim; i++)
        {
            ss >> bias2GradPrev[i];
        }
        std::getline(in, line);
        std::getline(in, line);
        ss = std::stringstream(line);

        if (weight1GradPrevPrev != nullptr)
        {
            delete[] weight1GradPrevPrev;
        }
        weight1GradPrevPrev = new T[inputDim * hiddenDim];
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            ss >> weight1GradPrevPrev[i];
        }
        std::getline(in, line);
        std::getline(in, line);
        ss = std::stringstream(line);

        if (weight2GradPrevPrev != nullptr)
        {
            delete[] weight2GradPrevPrev;
        }
        weight2GradPrevPrev = new T[hiddenDim * outputDim];
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            ss >> weight2GradPrevPrev[i];
        }
        std::getline(in, line);
        std::getline(in, line);

        ss = std::stringstream(line);
        if (bias1GradPrevPrev != nullptr)
        {
            delete[] bias1GradPrevPrev;
        }
        bias1GradPrevPrev = new T[hiddenDim];
        for (size_t i = 0; i < hiddenDim; i++)
        {
            ss >> bias1GradPrevPrev[i];
        }
        std::getline(in, line);
        std::getline(in, line);
        ss = std::stringstream(line);
        if (bias2GradPrevPrev != nullptr)
        {
            delete[] bias2GradPrevPrev;
        }
        bias2GradPrevPrev = new T[outputDim];
        for (size_t i = 0; i < outputDim; i++)
        {
            ss >> bias2GradPrevPrev[i];
        }   
        //std::cout << "bias2GradPrevPrev: " << std::endl;
        //done
    }





    template<typename T, typename real_x>
    void auto_encoder<T,real_x>::load(std::string filename)
    {
        std::ifstream in(filename);
        load(in);
        in.close();

    }
    template<typename T, typename real_x>
    void  auto_encoder<T,real_x>::save(std::string filename)
    {
        std::ofstream out(filename);
        dump(out);
        out.close();
    }



    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::initializeActivationFunction()
    {
        XactivationFunctionPtr f(&auto_encoder<T,real_x>::sigmoid);
        XactivationFunctionPtr fPrime(&auto_encoder<T,real_x>::sigmoidPrime);

        activationFunctionPtr = f;
        activationPrimeFunctionPtr = fPrime;
        activationGradientFunctionPtr = fPrime;
        activationPrimeGradientFunctionPtr = f;
        activationPrimeGradientHatFunctionPtr = fPrime;


    }
    
    //forward: 
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::forward()
    {
        //std::cout << "auto_encoder forward" << std::endl;
        //update hidden layers
        for (size_t i = 0; i < hiddenDim; i++)
        {
            hidden[i] = 0;
            for (size_t j = 0; j < inputDim; j++)
            {
                hidden[i] += input[j] * weight1[j * hiddenDim + i];
            }
            hidden[i] = (this->*activationFunctionPtr)(hidden[i]);
        }
        //update output dimensions
        for (size_t i = 0; i < outputDim; i++)
        {
            output[i] = 0;
            for (size_t j = 0; j < hiddenDim; j++)
            {
                output[i] += hidden[j] * weight2[j * outputDim + i];
            }
            output[i] = (this->*activationFunctionPtr)(output[i]);
        }
        //std::cout << "auto_encoder forward end" << std::endl;



    }   
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::backward()
    {
        //std::cout << "auto_encoder backward" << std::endl;
        //update weight2Grad
        for (size_t i = 0; i < outputDim; i++)
        {
            for (size_t j = 0; j < hiddenDim; j++)
            {
                weight2Grad[j * outputDim + i] += hidden[j] * (output[i] - input[i]) * (this->*activationPrimeFunctionPtr)(output[i]);
            }
        }   
        //update bias2Grad
        for (size_t i = 0; i < outputDim; i++)
        {
            bias2Grad[i] += (output[i] - input[i]) * (this->*activationPrimeFunctionPtr)(output[i]);
        }
        //update weight1Grad
        for (size_t i = 0; i < hiddenDim; i++)
        {
            for (size_t j = 0; j < inputDim; j++)
            {
                T sum = 0;
                for (size_t k = 0; k < outputDim; k++)
                {
                    sum += (output[k] - input[k]) * (this->*activationPrimeFunctionPtr)(output[k]) * weight2[i * outputDim + k];
                }
                weight1Grad[j * hiddenDim + i] += input[j] * sum * (this->*activationPrimeFunctionPtr)(hidden[i]);
            }
        }
        //update bias1Grad
        for (size_t i = 0; i < hiddenDim; i++)
        {
            T sum = 0;
            for (size_t j = 0; j < outputDim; j++)
            {
                sum += (output[j] - input[j]) * (this->*activationPrimeFunctionPtr)(output[j]) * weight2[i * outputDim + j];
            }
            bias1Grad[i] += sum * (this->*activationPrimeFunctionPtr)(hidden[i]);
        }
        //update everything else :

        //std::cout << "auto_encoder backward end" << std::endl;        

    }

    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::update()
    {
        //std::cout << "auto_encoder update" << std::endl;
        //update weight1
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            weight1[i] -= learningRate * weight1Grad[i];
        }
        //update weight2
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            weight2[i] -= learningRate * weight2Grad[i];
        }
        //update bias1
        for (size_t i = 0; i < hiddenDim; i++)
        {
            bias1[i] -= learningRate * bias1Grad[i];
        }
        //update bias2
        for (size_t i = 0; i < outputDim; i++)
        {
            bias2[i] -= learningRate * bias2Grad[i];
        }

        //update weight1GradPrevPrev
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            weight1GradPrevPrev[i] = weight1GradPrev[i];
        }
        //update weight2GradPrevPrev
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            weight2GradPrevPrev[i] = weight2GradPrev[i];
        }
        //update bias1GradPrevPrev
        for (size_t i = 0; i < hiddenDim; i++)
        {
            bias1GradPrevPrev[i] = bias1GradPrev[i];
        }
        //update bias2GradPrevPrev
        for (size_t i = 0; i < outputDim; i++)
        {
            bias2GradPrevPrev[i] = bias2GradPrev[i];
        }
        //update weight1GradPrev
        updateWeight1GradPrev();
        //update weight2GradPrev
        updateWeight2GradPrev();
        
        //update bias1GradPrev
        updateBias1GradPrev();
        
        //update bias2GradPrev
        updateBias2GradPrev();
        
        //update weight1Inc

        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            weight1Inc[i] = momentum * weight1Inc[i] + learningRate * weight1Grad[i];
        }
        //update weight2Inc
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            weight2Inc[i] = momentum * weight2Inc[i] + learningRate * weight2Grad[i];
        }
        //update bias1Inc
        for (size_t i = 0; i < hiddenDim; i++)
        {
            bias1Inc[i] = momentum * bias1Inc[i] + learningRate * bias1Grad[i];
        }
        //update bias2Inc
        for (size_t i = 0; i < outputDim; i++)
        {
            bias2Inc[i] = momentum * bias2Inc[i] + learningRate * bias2Grad[i];
        }

        //std::cout << "auto_encoder update end" << std::endl;
        //std::cout << "auto_encoder update end" << std::endl;

    }
    //update weight1GradPrev
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateWeight1GradPrev()
    {
        //std::cout << "auto_encoder updateWeight1GradPrev" << std::endl;
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            weight1GradPrev[i] = weight1Grad[i];
        }
        //std::cout << "auto_encoder updateWeight1GradPrev end" << std::endl;

    }
    //update weight2GradPrev
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateWeight2GradPrev()
    {
        //std::cout << "auto_encoder updateWeight2GradPrev" << std::endl;
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            weight2GradPrev[i] = weight2Grad[i];
        }
        //std::cout << "auto_encoder updateWeight2GradPrev end" << std::endl;

    }
    //update bias1GradPrev
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateBias1GradPrev()
    {
        //std::cout << "auto_encoder updateBias1GradPrev" << std::endl;
        for (size_t i = 0; i < hiddenDim; i++)
        {
            bias1GradPrev[i] = bias1Grad[i];
        }
        //std::cout << "auto_encoder updateBias1GradPrev end" << std::endl;

    }
    //update bias2GradPrev
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateBias2GradPrev()
    {
        //std::cout << "auto_encoder updateBias2GradPrev" << std::endl;
        for (size_t i = 0; i < outputDim; i++)
        {
            bias2GradPrev[i] = bias2Grad[i];
        }
        //std::cout << "auto_encoder updateBias2GradPrev end" << std::endl;

    } 


    //update weight1Inc
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateWeight1Inc()
    {
        //std::cout << "auto_encoder updateWeight1Inc" << std::endl;
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            weight1Inc[i] = momentum * weight1Inc[i] + learningRate * weight1Grad[i];
        }
        //std::cout << "auto_encoder updateWeight1Inc end" << std::endl;

    }
    //update weight2Inc
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateWeight2Inc()
    {
        //std::cout << "auto_encoder updateWeight2Inc" << std::endl;
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            weight2Inc[i] = momentum * weight2Inc[i] + learningRate * weight2Grad[i];
        }
        //std::cout << "auto_encoder updateWeight2Inc end" << std::endl;

    }
    //update bias1Inc
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateBias1Inc()
    {
        //std::cout << "auto_encoder updateBias1Inc" << std::endl;
        for (size_t i = 0; i < hiddenDim; i++)
        {
            bias1Inc[i] = momentum * bias1Inc[i] + learningRate * bias1Grad[i];
        }
        //std::cout << "auto_encoder updateBias1Inc end" << std::endl;

    }
    //update bias2Inc
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateBias2Inc()
    {
        //std::cout << "auto_encoder updateBias2Inc" << std::endl;
        for (size_t i = 0; i < outputDim; i++)
        {
            bias2Inc[i] = momentum * bias2Inc[i] + learningRate * bias2Grad[i];
        }
        //std::cout << "auto_encoder updateBias2Inc end" << std::endl;

    }

    //update weight1GradPrevPrev
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateWeight1GradPrevPrev()
    {
        //std::cout << "auto_encoder updateWeight1GradPrevPrev" << std::endl;
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            weight1GradPrevPrev[i] = weight1GradPrev[i];
        }
        //std::cout << "auto_encoder updateWeight1GradPrevPrev end" << std::endl;

    }

    //update weight2GradPrevPrev
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateWeight2GradPrevPrev()
    {
        //std::cout << "auto_encoder updateWeight2GradPrevPrev" << std::endl;
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            weight2GradPrevPrev[i] = weight2GradPrev[i];
        }
        //std::cout << "auto_encoder updateWeight2GradPrevPrev end" << std::endl;

    }
    //update bias1GradPrevPrev
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateBias1GradPrevPrev()
    {
        //std::cout << "auto_encoder updateBias1GradPrevPrev" << std::endl;
        for (size_t i = 0; i < hiddenDim; i++)
        {
            bias1GradPrevPrev[i] = bias1GradPrev[i];
        }
        //std::cout << "auto_encoder updateBias1GradPrevPrev end" << std::endl;

    }
    //update bias2GradPrevPrev
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateBias2GradPrevPrev()
    {
        //std::cout << "auto_encoder updateBias2GradPrevPrev" << std::endl;
        for (size_t i = 0; i < outputDim; i++)
        {
            bias2GradPrevPrev[i] = bias2GradPrev[i];
        }
        //std::cout << "auto_encoder updateBias2GradPrevPrev end" << std::endl;

    }

    //update weight1Grad
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateWeight1Grad()
    {
        //std::cout << "auto_encoder updateWeight1Grad" << std::endl;
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            weight1Grad[i] = weight1GradPrev[i] + momentum * weight1GradPrevPrev[i];
        }
        //std::cout << "auto_encoder updateWeight1Grad end" << std::endl;

    }
    //update weight2Grad
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateWeight2Grad()
    {
        //std::cout << "auto_encoder updateWeight2Grad" << std::endl;
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            weight2Grad[i] = weight2GradPrev[i] + momentum * weight2GradPrevPrev[i];
        }
        //std::cout << "auto_encoder updateWeight2Grad end" << std::endl;

    }
    //update bias1Grad
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateBias1Grad()
    {
        //std::cout << "auto_encoder updateBias1Grad" << std::endl;
        for (size_t i = 0; i < hiddenDim; i++)
        {
            bias1Grad[i] = bias1GradPrev[i] + momentum * bias1GradPrevPrev[i];
        }
        //std::cout << "auto_encoder updateBias1Grad end" << std::endl;

    }
    //update bias2Grad
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateBias2Grad()
    {
        //std::cout << "auto_encoder updateBias2Grad" << std::endl;
        for (size_t i = 0; i < outputDim; i++)
        {
            bias2Grad[i] = bias2GradPrev[i] + momentum * bias2GradPrevPrev[i];
        }
        //std::cout << "auto_encoder updateBias2Grad end" << std::endl;

    }
    
    //update weight1
    template <typename T, typename real_x>
    void auto_encoder<T,real_x>::updateWeight1()
    {
        //std::cout << "auto_encoder updateWeight1" << std::endl;
        for (size_t i = 0; i < inputDim * hiddenDim; i++)
        {
            weight1[i] -= learningRate * weight1Grad[i];
        }
        //std::cout << "auto_encoder updateWeight1 end" << std::endl;

    }
    //update weight2
    template <typename T, typename real_x >
    void auto_encoder<T,real_x>::updateWeight2()
    {
        //std::cout << "auto_encoder updateWeight2" << std::endl;
        for (size_t i = 0; i < hiddenDim * outputDim; i++)
        {
            weight2[i] -= learningRate * weight2Grad[i];
        }
        //std::cout << "auto_encoder updateWeight2 end" << std::endl;

    }
    //update bias1
    template <typename T, typename real_x >
    void auto_encoder<T,real_x>::updateBias1()
    {
        //std::cout << "auto_encoder updateBias1" << std::endl;
        for (size_t i = 0; i < hiddenDim; i++)
        {
            bias1[i] -= learningRate * bias1Grad[i];
        }
        //std::cout << "auto_encoder updateBias1 end" << std::endl;

    }
    //update bias2
    template <typename T, typename real_x >
    void auto_encoder<T,real_x>::updateBias2()
    {
        //std::cout << "auto_encoder updateBias2" << std::endl;
        for (size_t i = 0; i < outputDim; i++)
        {
            bias2[i] -= learningRate * bias2Grad[i];
        }
        //std::cout << "auto_encoder updateBias2 end" << std::endl;

    } 
    //conjugate gradient  
    template <typename T, typename real_x >
    void auto_encoder<T,real_x>::conjugateGradient()
    {
        //std::cout << "auto_encoder conjugateGradient " << std::endl;
        //update weight1Grad
        updateWeight1Grad();
        //update weight2Grad
        updateWeight2Grad();
        //update bias1Grad
        updateBias1Grad();
        //update bias2Grad
        updateBias2Grad();
        //update weight1
        updateWeight1();
        //update weight2
        updateWeight2();
        //update bias1
        updateBias1();
        //update bias2
        updateBias2();
        //std::cout << "auto_encoder conjugateGradient  end" << std::endl;

    }


    struct tf_auto_encoder
    {
        template <typename T, typename real_x >
        void operator()(const T* const input, T* const output)
        {
            auto_encoder<T,real_x> ae;
            ae.setInput(input);
            ae.forward();
            ae.backward();
            ae.conjugateGradient();
            ae.getOutput(output);
        }
    };  

    struct  tf_auto_encoder_grad
    {
        template <typename T, typename real_x >
        void operator()(const T* const input, T* const output)
        {
            auto_encoder<T,real_x> ae;
            ae.setInput(input);
            ae.forward();
            ae.backward();
            ae.getWeight1Grad(output);
        }
    };  

    struct  tf_auto_encoder_grad2
    {
        template <typename T, typename real_x >
        void operator()(const T* const input, T* const output)
        {
            auto_encoder<T,real_x> ae;
            ae.setInput(input);
            ae.forward();
            ae.backward();
            ae.getWeight2Grad(output);
        }
    };     

    struct  tf_auto_encoder_grad3
    {
        template <typename T, typename real_x >
        void operator()(const T* const input, T* const output)
        {
            auto_encoder<T,real_x> ae;
            ae.setInput(input);
            ae.forward();
            ae.backward();
            ae.getBias1Grad(output);
        }
    };

    struct  tf_auto_encoder_grad4
    {
        template <typename T, typename real_x >
        void operator()(const T* const input, T* const output)
        {
            auto_encoder<T,real_x> ae;
            ae.setInput(input);
            ae.forward();
            ae.backward();
            ae.getBias2Grad(output);
        }
    };

    struct  tf_auto_encoder_grad5
    {
        template <typename T, typename real_x >
        void operator()(const T* const input, T* const output)
        {
            auto_encoder<T,real_x> ae;
            ae.setInput(input);
            ae.forward();
            ae.backward();
            ae.getWeight1(output);
        }
    };

    struct  tf_auto_encoder_grad6
    {
        template <typename T, typename real_x >
        void operator()(const T* const input, T* const output)
        {
            auto_encoder<T,real_x> ae;
            ae.setInput(input);
            ae.forward();
            ae.backward();
            ae.getWeight2(output);
        }
    };

    struct  tf_auto_encoder_grad7
    {
        template <typename T, typename real_x >
        void operator()(const T* const input, T* const output)
        {
            auto_encoder<T,real_x> ae;
            ae.setInput(input);
            ae.forward();
            ae.backward();
            ae.getBias1(output);
        }
    };
     
    struct  tf_auto_encoder_grad8
    {
        template <typename T, typename real_x >
        void operator()(const T* const input, T* const output)
        {
            auto_encoder<T,real_x> ae;
            ae.setInput(input);
            ae.forward();
            ae.backward();
            ae.getBias2(output);
        }
    };

    struct  tf_auto_encoder_grad9
    {
        template <typename T, typename real_x >
        void operator()(const T* const input, T* const output)
        {
            output[0] = 0;
        }
    };

    //autoencoder<>::save_as_pt - save autoencoder as tensorflow pre trained file
    //
    template <typename T, typename real_x >
    void auto_encoder<T,real_x>::save_as_pt ( const std::string filename ) {

        std::ofstream out(filename, std::ios::binary); 
        if (!out.is_open()) {
            std::cout << "Cannot open file to write: " << filename << std::endl;
            return;
        }
        //dont use tensorflow namespace and dependencies, just save weights and biases as binary file,no python
        //save weights and biases
        out.write((char*)weight1, sizeof(real_x) * inputDim * hiddenDim);
        out.write((char*)weight2, sizeof(real_x) * hiddenDim * outputDim);
        out.write((char*)bias1, sizeof(real_x) * hiddenDim);
        out.write((char*)bias2, sizeof(real_x) * outputDim);
        out.close();
    }

    //convert autoencoder json to tensorflow pt file:
    //src: autoencoder json file
    //dst: tensorflow pt file
    template <typename T, typename real_x >
    struct file_converter_pt 
    {
        void operator()(const std::string src,const std::string dst)
        {

            auto_encoder<T,real_x> ae;
            ae.load(src);
            ae.save_as_pt(dst); 

        }
    };
 
    template <class T,class real_x>
    class softmax_classifier
    {
       size_t n_classes;
       size_t n_dimensions;

       matrix<real_t> weight;
       real_t alpha;
       real_t lambda;


        softmax_classifier(size_t n_classes,size_t n_dimensions,real_t alpha,real_t lambda)
        {
            this->n_classes = n_classes;
            this->n_dimensions = n_dimensions;
            this->alpha = alpha;
            this->lambda = lambda;
            //init random weight
            weight = matrix<real_t>::Random(n_classes,n_dimensions);
            
        } 
        //forward
        void forward(const matrix<real_t>& input,matrix<real_t>& output)
        {
            //softmax
            output = input * weight.transpose();
            output = output.unaryExpr([](real_t x) { return std::exp(x); });
            output = output.rowwise([](real_t x) {return x;}) / output.sum();
        }
        //backward
        void backward(const matrix<real_t>& input,const matrix<real_t>& output,const matrix<real_t>& target,matrix<real_t>& grad)
        {
            //softmax
            grad = output - target;
            //weight
            grad = grad.transpose() * input;
            //regularization
            grad = grad + lambda * weight;
        }   
        //update
        void update(const matrix<real_t>& grad)
        {
            weight = weight - alpha * grad;
        }
        //train
        void train(const matrix<real_t>& input,const matrix<real_t>& target)
        {
            matrix<real_t> output;
            matrix<real_t> grad;
            forward(input,output);
            backward(input,output,target,grad);
            update(grad);
        }
        //predict
        void predict(const matrix<real_t>& input,matrix<real_t>& output)
        {
            forward(input,output);
        }

        //save
        void save(const std::string filename)
        {
            std::ofstream out(filename, std::ios::binary); 
            if (!out.is_open()) {
                std::cout << "Cannot open file to write: " << filename << std::endl;
                return;
            }
            //dont use tensorflow namespace and dependencies, just save weights and biases as binary file,no python
            //save weights and biases
            out.write((char*)weight.data(), sizeof(real_x) * n_classes * n_dimensions);
            out.close();
        }
        //load
        void load(const std::string filename)
        {
            std::ifstream in(filename, std::ios::binary); 
            if (!in.is_open()) {
                std::cout << "Cannot open file to read: " << filename << std::endl;
                return;
            }
            //dont use tensorflow namespace and dependencies, just save weights and biases as binary file,no python
            //save weights and biases
            in.read((char*)weight.data(), sizeof(real_x) * n_classes * n_dimensions);
            in.close();
        }
    }; 



} // namespace provallo

#endif /* PROVALLO_AUTO_ENCODER_H_ */