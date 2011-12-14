#include <complex>
#include <emmintrin.h>

struct dcomplexSSE
{
    __m128d data;

    dcomplexSSE() { }
    dcomplexSSE(__m128d data) : data(data) {}
    dcomplexSSE(double realpart) : data(_mm_set_pd(realpart,0)) {}
    dcomplexSSE(double x, double y) : data(_mm_set_pd(x,y)) { }

    inline dcomplexSSE& operator=(const dcomplexSSE& rhs)
    {
        data = rhs.data;
        return *this;
    }
    inline dcomplexSSE& operator=(double rhs)
    {
        data = _mm_set_pd(rhs, 0);
        return *this;
    }
};

inline dcomplexSSE operator+(dcomplexSSE a, dcomplexSSE b)
{
    return _mm_add_pd(a.data,b.data);
}
inline dcomplexSSE operator-(dcomplexSSE a, dcomplexSSE b)
{
    return _mm_sub_pd(a.data,b.data);
}
inline double real(dcomplexSSE z)
{
    double res[2];
    _mm_storeu_pd(res, z.data);
    return res[1];
}
inline double imag(dcomplexSSE z)
{
    double res[2];
    _mm_storeu_pd(res, z.data);
    return res[0];
}
inline dcomplexSSE operator*(dcomplexSSE a, dcomplexSSE b)
{
    // not performance-critical, cheat using std::complex
    std::complex<double> z1(real(a), imag(a));
    std::complex<double> z2(real(b), imag(b));
    std::complex<double> z = z1*z2;
    return dcomplexSSE(real(z), imag(z));
}

inline double norm2(dcomplexSSE z)
{
    __m128d z2 = _mm_mul_pd(z.data, z.data);
    __m128d z2swap = _mm_shuffle_pd(z2, z2, _MM_SHUFFLE2(0,1));
    __m128d z3 = _mm_add_sd(z2, z2swap);
    double res[2];
    _mm_storeu_pd(res, z3);
    return res[0];
}

inline __m128d norm2sse(dcomplexSSE z)
{
    __m128d z2 = _mm_mul_pd(z.data, z.data);
    __m128d z2swap = _mm_shuffle_pd(z2, z2, _MM_SHUFFLE2(0,1));
    return _mm_add_sd(z2, z2swap);
}

