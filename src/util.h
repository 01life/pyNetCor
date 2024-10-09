#ifndef UTIL_H
#define UTIL_H

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <vector>

#include "matrix.h"

namespace util {

    // Returns the rank of each element in [begin, end), ignoring NaN.
    void nanRank(const double* v, size_t n, double* result);

    // Returns the rank of each row in matrix X, ignoring NaN.
    Matrix<double> parallelNanRank(const Matrix<double> &X, int nthreads);

    double nanMean(const double* v, size_t n);

    std::vector<size_t> argSort(const double* v, size_t n, bool decreasing = false);

    // Symmetrize matrix X, where X[i, j] = X[j, i]
    void symm_matrix(double *X, int n, int nthreads);

    inline size_t transFullMatIndex(size_t i, size_t j, size_t colNum) {
        return i * colNum + j - (i + 1) * (i + 2) / 2;
    }

    inline size_t transFullMatIndex(size_t i, size_t j, size_t colNum, size_t iStart) {
        return i * colNum + j - iStart * colNum - (iStart + i + 2) * (i - iStart + 1) / 2;
    }

    template<typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
    T nanQuantile(T* v, size_t n, T quantileMultiplier) {
        std::sort(v, v+n, [](T a, T b) {
            if (std::isnan(a)) return false;
            if (std::isnan(b)) return true;
            return a < b;
        });

        // quantile is calculated by linear interpolation between two nearest values
        T targetIndex = quantileMultiplier * (n - 1);
        size_t lowerIndex = std::floor(targetIndex);
        size_t upperIndex = std::ceil(targetIndex);
        return v[lowerIndex] + (targetIndex - lowerIndex) * (v[upperIndex] - v[lowerIndex]);
    };

} // namespace util

#endif // UTIL_H