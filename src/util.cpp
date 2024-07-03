#include "util.h"

#include <cblas.h>
#include <iterator>
#include <omp.h>

namespace util {

    void nanRank(const double *v, size_t n, double *result) {
        std::vector<size_t> sort_index(n);
        std::iota(sort_index.begin(), sort_index.end(), 0);
        std::sort(sort_index.begin(), sort_index.end(), [&v](size_t i, size_t j) {
            if (std::isnan(v[i])) return false;
            if (std::isnan(v[j])) return true;
            return v[i] < v[j];
        });

        for (size_t nties, i = 0; i < n; i += nties) {
            nties = 1;
            while (i + nties < n && v[sort_index[i]] == v[sort_index[i + nties]]) ++nties;
            for (size_t k = 0; k < nties; ++k) {
                result[sort_index[i + k]] = i + (nties + 1) / 2.0; // average rank of n tied values
                // r[sort_index[i+k]] = i + 1;     // min
                // r[sort_index[i+k]] = i + nties;     // max
                // r[sort_index[i+k]] = i + k + 1; // random order
            }
        }
    }

    Matrix<double> parallelNanRank(const Matrix<double> &X, int nthreads) {
        Matrix<double> Y(X.rows(), X.cols());

#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
        for (int64_t i = 0; i < X.rows(); ++i) {
            nanRank(X.row(i), X.cols(), Y.row(i));
        }

        return Y;
    }

    double nanMean(const double *v, size_t n) {
        int count = 0;
        double sum = std::accumulate(v, v + n, 0.0,
                                     [&count](double a, double b) {
                                         if (!std::isnan(b)) {
                                             ++count;
                                             return a + b;
                                         }
                                         return a;
                                     });
        return sum / count;
    }

    std::vector<size_t> argSort(const double *v, size_t n, bool decreasing) {
        std::vector<size_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        if (decreasing) {
            std::sort(indices.begin(), indices.end(), [&v](size_t i, size_t j) { return v[i] > v[j]; });
        } else {
            std::sort(indices.begin(), indices.end(), [&v](size_t i, size_t j) { return v[i] < v[j]; });
        }
        return indices;
    }

} // namespace util