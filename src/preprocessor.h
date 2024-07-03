#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <numeric>
#include <vector>

#include "options.h"


namespace preprocessor {

    template<typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
    inline void fillNanMean(T *v, size_t n) {
        int count = 0;
        T sum = std::accumulate(v, v + n, 0.0,
                                     [&count](T a, T b) {
                                         if (!std::isnan(b)) {
                                             ++count;
                                             return a + b;
                                         }
                                         return a;
                                     });
        T mean = sum / count;

        std::replace_if(v, v + n, [](T a) { return std::isnan(a); }, mean);
    }

    template<typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
    inline void fillNanMedian(T *v, size_t n) {
        std::vector<T> tmp;
        std::copy_if(v, v + n, std::back_inserter(tmp), [](T a) { return !std::isnan(a); });

        size_t half = tmp.size() / 2;
        std::nth_element(tmp.begin(), tmp.begin() + half, tmp.end());

        T median = tmp[half];
        if (tmp.size() % 2 == 0) {
            median = (median + tmp[half - 1]) / 2;
        }

        std::replace_if(v, v + n, [](T a) { return std::isnan(a); }, median);
    }

    template<typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
    inline void ignoreNan(T *v, size_t n) {
        std::replace_if(v, v + n, [](T a) { return std::isnan(a); }, 0);
    }

    template<typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
    inline bool hasNan(const T *v, size_t n) {
        return std::any_of(v, v + n, [](T x) { return std::isnan(x); });
    }

    template <typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
    void parallelProcessNan(T* X, size_t nrows, size_t ncols, const NAMethod &nanMethod, int nthreads) {
#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
        for (int64_t i = 0; i < nrows; ++i) {
            if (hasNan(X + i * ncols, ncols)) {
                switch (nanMethod) {
                    case NAMethod::FillMean:
                        fillNanMean(X + i * ncols, ncols);
                        break;
                    case NAMethod::FillMedian:
                        fillNanMedian(X + i * ncols, ncols);
                        break;
                    case NAMethod::Ignore:
                        ignoreNan(X + i * ncols, ncols);
                        break;
                }
            }
        }
    };

    template <typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
    inline T calcTopThreeProp(const T *v, size_t n) {
        std::vector<T> tmp(v, v + n);
        std::sort(tmp.begin(), tmp.end(), std::greater<>());
        T sum = std::accumulate(tmp.begin(), tmp.begin() + 3, 0.0);
        return sum / std::accumulate(tmp.begin(), tmp.end(), 0.0);
    }

} // namespace preprocessor

#endif // PREPROCESSOR_H
