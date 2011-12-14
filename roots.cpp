// Fast version of roots.py...
#include <complex>
#include <algorithm>
#include <limits>
#include <iostream>
#include <fstream>

typedef std::complex<double> complex;

inline double norm2(std::complex<double> z)
{
    double x = real(z), y = imag(z);
    return x*x + y*y;
}

double evaluate(complex poly, complex* zpows, int n)
{
    using std::min;
//    if(n < 0)
//        return norm2(poly);
    // Unrolled a few times for speed.
    if(n < 3)
    {
        return min(min(min(norm2(poly + zpows[0] + zpows[1] + zpows[2]),
                           norm2(poly + zpows[0] + zpows[1] - zpows[2])),
                       min(norm2(poly + zpows[0] - zpows[1] + zpows[2]),
                           norm2(poly + zpows[0] - zpows[1] - zpows[2]))),
                   min(min(norm2(poly - zpows[0] + zpows[1] + zpows[2]),
                           norm2(poly - zpows[0] + zpows[1] - zpows[2])),
                       min(norm2(poly - zpows[0] - zpows[1] + zpows[2]),
                           norm2(poly - zpows[0] - zpows[1] - zpows[2]))));
    }
    else
        return min(evaluate(poly + *zpows, zpows+1, n-1),
                   evaluate(poly - *zpows, zpows+1, n-1));
}


int main()
{
    bool usePolar = false;

    const int degree = 13;
    const double R = 1.7;
    const double R0 = 0;
//    const double R = 0.2;
//    const double R0 = 0.4;
    const int N = 1000;
    int M = N;
    if(usePolar)
        M = int(N * (R-1)/M_PI_2);

    double* result = new double[N*M];

#   pragma omp parallel for
    for(int j = 0; j < M; ++j)
    {
        for(int i = 0; i < N; ++i)
        {
            // Calculate powers of z for current point
            complex z;
            if(usePolar)
            {
                double t = M_PI_2 * (i + 0.5)/N;
                double r = 1 + (R-1) * (j + 0.5)/M;
                z = complex(r*cos(t), r*sin(t));
            }
            else
                z = complex(R0 + R * (i + 0.5)/N, R0 + R * (j + 0.5)/M);
            complex zpowers[degree+1];
            for(int k = 0; k <= degree; ++k)
                zpowers[k] = pow(z, k);
            // Find minimum absolute value of all terms; there are 2^degree of them!
            double minpoly = evaluate(0, zpowers, degree);
            // Weight of 1/|z|^degree implies z <--> 1/z symmetry
            result[N*j + i] = sqrt(minpoly / pow(abs(z),degree));
        }
    }

    // Output numbers to stdout with a single thread.
    std::ofstream outFile("data.txt");
    for(int j = 0; j < M; ++j)
    {
        for(int i = 0; i < N; ++i)
            outFile << result[N*j + i] << " ";
        outFile << "\n";
    }

    delete[] result;
    return 0;
}
