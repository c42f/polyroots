// Fast version of roots.py...
#include <complex>
#include <algorithm>
#include <limits>
#include <iostream>
#include <fstream>

#include "float4.h"

//typedef std::complex<double> complex;
typedef std::complex<float4> complex;

template<typename T>
inline T abs2(std::complex<T> z)
{
    T x = real(z), y = imag(z);
    return x*x + y*y;
}

inline float4 abs(std::complex<float4> z)
{
    return sqrt(abs2(z));
}

float4 evaluate(complex poly, complex* zpows, int n)
{
    using std::min;
//    if(n < 0)
//        return abs2(poly);
    // Unrolled a few times for speed.
    if(n < 3)
    {
        return min(min(min(abs2(poly + zpows[0] + zpows[1] + zpows[2]),
                           abs2(poly + zpows[0] + zpows[1] - zpows[2])),
                       min(abs2(poly + zpows[0] - zpows[1] + zpows[2]),
                           abs2(poly + zpows[0] - zpows[1] - zpows[2]))),
                   min(min(abs2(poly - zpows[0] + zpows[1] + zpows[2]),
                           abs2(poly - zpows[0] + zpows[1] - zpows[2])),
                       min(abs2(poly - zpows[0] - zpows[1] + zpows[2]),
                           abs2(poly - zpows[0] - zpows[1] - zpows[2]))));
    }
    else
        return min(evaluate(poly + *zpows, zpows+1, n-1),
                   evaluate(poly - *zpows, zpows+1, n-1));
}

void writeFile(const char* name, int N, int M, const float* data)
{
    std::ofstream outFile(name, std::ios::out | std::ios::binary);
    outFile << N << " " << M << "\n";
    outFile.write(reinterpret_cast<const char*>(data), sizeof(float)*N*M);
}

int main(int argc, char* argv[])
{
    bool usePolar = false;

    int degree = 14;
    if(argc > 1)
        degree = atoi(argv[1]);
    const double R = 0.18;
    const double x0 = 0.504 - R/2, y0 = 0.87 - R/2;
//    const double R = 0.001;
//    const double x0 = 1.08-R/2, y0 = x0;
//    const int N = 400;
//    const double R = 1.7;
//    const double x0 = 0, y0 = 0;
//    const double R = 0.7;
//    const double x0 = 1.0, y0 = 0;
//    const double R = 0.02;
//    const double x0 = 0.707 - R/2, y0 = x0;
    const int N = 2000;
    int M = N;
    if(usePolar)
        M = int(N * (R-1)/M_PI_2);

    float* result = new float[N*M];

    int linesdone = 0;
    // schedule(static, 1)
#   pragma omp parallel for
    for(int j = 0; j < M; ++j)
    {
        for(int i = 0; i < N; i+=4)
        {
            // Calculate powers of z for current point
            float4 x = x0 + R * (float4(i,i+1,i+2,i+3) + 0.5)/N;
            float4 y = y0 + R * (j + 0.5)/N;
            complex z(x,y);
            complex zpowers[degree];
            for(int k = 0; k < degree; ++k)
                zpowers[k] = pow(z, k+1);
            // Find minimum absolute value of all terms; there are 2^(degree+1)
            // polynomials, but we reduce that by a factor of two by making use
            // of symmetry [for every P(z) there is a -P(z) in the set]
            float4 minpoly = evaluate(float4(1), zpowers, degree-1);
            // Weight of 1/|z|^degree implies z <--> 1/z symmetry
            float4 res = sqrt(minpoly / pow(abs(z),degree));
            result[N*j + i+0] = res[0];
            result[N*j + i+1] = res[1];
            result[N*j + i+2] = res[2];
            result[N*j + i+3] = res[3];
        }
#       pragma omp critical
        {
            ++linesdone;
            std::cout << 100.0*linesdone/M << "%   \r" << std::flush;
        }
    }

    writeFile("minpoly.dat", N, M, result);

    delete[] result;
    return 0;
}
