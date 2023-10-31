#ifndef __SAMPLING_HELPER_H_
#define __SAMPLING_HELPER_H_



#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <numeric>
#include <complex>
#include <valarray>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <ctime>

#include "matrix.h"

//implement FFT for sampling


namespace provallo
{
    class sampling_helper
    {
        public:
            sampling_helper() = default;
            ~sampling_helper() = default;

            static void
            fft(std::valarray<std::complex<double>> &x)
            {
                const size_t N = x.size();
                if (N <= 1) return;

                // divide
                std::valarray<std::complex<double>> even = x[std::slice(0, N/2, 2)];
                std::valarray<std::complex<double>>  odd = x[std::slice(1, N/2, 2)];

                // conquer
                fft(even);
                fft(odd);

                // combine
                for (size_t k = 0; k < N/2; ++k)
                {
                    std::complex<double> t = std::polar(1.0, -2 * M_PI * k / N) * odd[k];
                    x[k    ] = even[k] + t;
                    x[k+N/2] = even[k] - t;
                }
            }
            static void  fft ( std::complex<double> *x , size_t N)
            {
                if (N <= 1) return;

                // divide
                std::valarray<std::complex<double>> even =  std::valarray<std::complex<double>>(x,N)[std::slice(0, N/2, 2)];
                std::valarray<std::complex<double>>  odd =  std::valarray<std::complex<double>>(x,N)[std::slice(1, N/2, 2)];

                // conquer
                fft(even);
                fft(odd);

                // combine
                for (size_t k = 0; k < N/2; ++k)
                {
                    std::complex<double> t = std::polar(1.0, -2 * M_PI * k / N) * odd[k];
                    x[k    ] = even[k] + t;
                    x[k+N/2] = even[k] - t;
                }
            }
            static void
            ifft(std::valarray<std::complex<double>> &x)
            {
                // conjugate the complex numbers
                x = x.apply(std::conj);

                // forward fft
                fft( x );

                // conjugate the complex numbers again
                x = x.apply(std::conj);

                // scale the numbers
                x /= x.size();
            }

            static  std::valarray<std::complex<double>>
            fft(const std::valarray<std::complex<double>> &x)
            {
                std::valarray<std::complex<double>> y = x;
                fft(y);
                return y;
            }

            static  std::valarray<std::complex<double>>
            ifft(const std::valarray<std::complex<double>> &x)
            {
                std::valarray<std::complex<double>> y = x;
                ifft(y);
                return y;
            }
            //fft of matrix 
            static matrix<std::complex<double>>
            fft(const matrix<double> &x)
            {
                matrix<std::complex<double>> y(x.rows(),x.cols());
                for(size_t i = 0 ; i < x.rows() ; ++i)
                {
                    for(size_t j = 0 ; j < x.cols() ; ++j)
                    {
                        y(i,j) = x(i,j);

                    }
                    //fft of row
                    fft(y[i] , y.cols());

                }
                return y;
            }   
            //ifft of matrix
            static matrix<std::complex<double>>
            ifft(const matrix<std::complex<double>> &x)
            {
                matrix<std::complex<double>> y(x.rows(),x.cols());
                for(size_t i = 0 ; i < x.rows() ; ++i)
                {
                    for(size_t j = 0 ; j < x.cols() ; ++j)
                    {
                        y(i,j) = x(i,j);
                    }
                }
                return y;
            }
    };
}


#endif//__SAMPLING_HELPER_H_