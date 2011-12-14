// Fast version of roots.py...
#include <complex>
#include <algorithm>
#include <limits>
#include <iostream>

#if 1

#include "dcomplexSSE.h"

typedef dcomplexSSE complex;

double evaluate(complex poly, complex* zpows, int n)
{
    // Unrolled a few times for speed.
    if(n < 3)
    {
        __m128d w = _mm_min_sd(
            _mm_min_sd(_mm_min_sd(norm2sse(poly + zpows[0] + zpows[1] + zpows[2]),
                                  norm2sse(poly + zpows[0] + zpows[1] - zpows[2])),
                       _mm_min_sd(norm2sse(poly + zpows[0] - zpows[1] + zpows[2]),
                                  norm2sse(poly + zpows[0] - zpows[1] - zpows[2]))),
            _mm_min_sd(_mm_min_sd(norm2sse(poly - zpows[0] + zpows[1] + zpows[2]),
                                  norm2sse(poly - zpows[0] + zpows[1] - zpows[2])),
                       _mm_min_sd(norm2sse(poly - zpows[0] - zpows[1] + zpows[2]),
                                  norm2sse(poly - zpows[0] - zpows[1] - zpows[2])))
        );
        double res[2];
        _mm_storeu_pd(res, w);
        return res[0];
    }
    else
        return std::min(evaluate(poly + *zpows, zpows+1, n-1),
                        evaluate(poly - *zpows, zpows+1, n-1));
}
#else

typedef std::complex<double> complex;

inline double norm2(std::complex<double> z)
{
    double x = real(z), y = imag(z);
    return x*x + y*y;
}

double evaluate(complex poly, complex* zpows, int n)
{
//    if(n < 0)
//        return norm2(poly);
    // Unrolled a few times for speed.
    if(n < 3)
    {
        return std::min(
            std::min(std::min(norm2(poly + zpows[0] + zpows[1] + zpows[2]),
                              norm2(poly + zpows[0] + zpows[1] - zpows[2])),
                     std::min(norm2(poly + zpows[0] - zpows[1] + zpows[2]),
                              norm2(poly + zpows[0] - zpows[1] - zpows[2]))),
            std::min(std::min(norm2(poly - zpows[0] + zpows[1] + zpows[2]),
                              norm2(poly - zpows[0] + zpows[1] - zpows[2])),
                     std::min(norm2(poly - zpows[0] - zpows[1] + zpows[2]),
                              norm2(poly - zpows[0] - zpows[1] - zpows[2])))
        );
    }
    else
        return std::min(evaluate(poly + *zpows, zpows+1, n-1),
                        evaluate(poly - *zpows, zpows+1, n-1));
}

#endif


int main()
{
    const int degree = 16;
    const double R = 1.6;
    const double R0 = 0;
//    const double R = 0.2;
//    const double R0 = 0.4;
    const int N = 4000;
    double* result = new double[N*N];

#   pragma omp parallel for
    for(int j = 0; j < N; ++j)
    {
        for(int i = 0; i < N; ++i)
        {
            // Calculate powers of z for current point
            double x = R0 + R * (i + 0.5)/N;
            double y = R0 + R * (j + 0.5)/N;
            complex z(x,y);
            complex zpowers[degree+1];
            zpowers[0] = 1.0;
            for(int k = 1; k <= degree; ++k)
                zpowers[k] = z*zpowers[k-1];
            // Accumulate all terms; there are 2^degree of them!
            result[N*j + i] = sqrt(evaluate(0, zpowers, degree));
        }
    }

    // Output numbers to stdout with a single thread.
    for(int j = 0; j < N; ++j)
    {
        for(int i = 0; i < N; ++i)
            std::cout << result[N*j + i] << " ";
        std::cout << "\n";
    }

    delete[] result;
    return 0;
}
