#include "cor.h"

#include <cmath>
#include <omp.h>
#include <random>
#include <cstdint>
#include <cblas.h>

#include "preprocessor.h"


void CorPearson::parallelCalcCor(Matrix<double> &X, Matrix<double> &Y, double *result, int nthreads) {
    parallelPreprocessNormalize(X, nthreads);

    size_t m = X.rows();
    size_t k = X.cols();
    size_t n, otherRows;
    if (Y.isEmpty()) {
        n = X.rows();
        otherRows = X.cols();
    } else {
        parallelPreprocessNormalize(Y, nthreads);
        n = Y.rows();
        otherRows = Y.cols();
    }

    if (X.cols() != otherRows) {
        throw std::invalid_argument("Incompatible matrix dimensions for multiplication.");
    }

    openblas_set_num_threads(nthreads);
    if (Y.isEmpty()) {
        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, m, k, 1.0, X.data(),
                    k, 0.0, result, m);
        util::symm_matrix(result, m, nthreads);
    } else {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0, X.data(), X.cols(),
                    Y.data(), Y.cols(), 0.0, result, n);
    }
}

double CorPearson::calcCor(double *x, double *y, size_t n) {
    preprocessNormalize(x, n);
    preprocessNormalize(y, n);

    return std::inner_product(x, x + n, y, 0.0);
}

double CorPearson::calcPvalue(double r, double df, const PTable &ptable) {
    if (std::isnan(r)) {
        return std::numeric_limits<double>::quiet_NaN();
    } else if (abs(abs(r) - 1) < 1e-8) {
        return MIN_PVALUE;
    }

    double q = r * std::sqrt(df / (1 - r * r));
    return ptable.getPvalue(q);
}

double CorPearson::commonCalcPvalue(double r, double df, const boost::math::students_t &dist) {
    if (std::isnan(r)) {
        return std::numeric_limits<double>::quiet_NaN();
    } else if (abs(abs(r) - 1) < 1e-8) {
        return MIN_PVALUE;
    }

    double q = r * std::sqrt(df / (1 - r * r));

    if (std::isnan(q)) {
        return std::numeric_limits<double>::quiet_NaN();
    } else {
        double ptmp = boost::math::cdf(dist, q);
        return 2 * std::min(ptmp, 1 - ptmp);
    }

//    try {
//        double ptmp = boost::math::cdf(dist, q);
//        return 2 * std::min(ptmp, 1 - ptmp);
//    } catch (const std::domain_error &e) {
//        return std::numeric_limits<double>::quiet_NaN();
//    }

}

void CorPearson::parallelPreprocessNormalize(Matrix<double> &X, int nthreads) {
#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
    for (int64_t i = 0; i < X.rows(); ++i) {
        preprocessNormalize(X.row(i), X.cols());
    }
}

void CorPearson::preprocessNormalize(double *v, size_t n) {
    double mean = util::nanMean(v, n);

    double sqrSum = std::accumulate(v, v + n, 0.0, [mean](double a, double b) {
        return std::isnan(b) ? a : a + (b - mean) * (b - mean);
    });
    double stdDev = std::sqrt(sqrSum);

    std::transform(v, v + n, v, [mean, stdDev](double a) {
        // If all the values in x are very close to the mean,
        // the loss of precision that occurs in the subtraction xm = x - xmean
        // might result in large errors in r. Reference from scipy.stats.pearsonr
        if (stdDev < 1e-13 * std::abs(mean)) {
            return std::numeric_limits<double>::quiet_NaN();
        } else if (std::isnan(a)) { // process nan with ignoreNan
            return 0.0;
        } else {
            return (a - mean) / stdDev;
        }
    });
}

void CorSpearman::parallelCalcCor(Matrix<double> &X, Matrix<double> &Y, double *result, int nthreads) {
    auto X_ranked = util::parallelNanRank(X, nthreads);

    Matrix<double> Y_ranked;
    if (Y.isEmpty()) {
        Y_ranked = Y;
    } else {
        Y_ranked = util::parallelNanRank(Y, nthreads);
    }

    CorPearson::parallelCalcCor(X_ranked, Y_ranked, result, nthreads);
}

double CorSpearman::calcCor(const double *x, const double *y, size_t n) {
    std::unique_ptr<double[]> x_ranked(new double[n]);
    std::unique_ptr<double[]> y_ranked(new double[n]);

    util::nanRank(x, n, x_ranked.get());
    util::nanRank(y, n, y_ranked.get());
    return CorPearson::calcCor(x_ranked.get(), y_ranked.get(), n);
}

//void CorSpearman::parallelCalcPvalue(const double *X, size_t n, double *P, double df, const PTable &ptable,
//                                     int nthreads) {
//    CorPearson::parallelCalcPvalue(X, n, P, df, ptable, nthreads);
//}

void CorKendall::parallelCalcCor(const Matrix<double> &X, const Matrix<double> &Y, double *result, int nthreads) {
    size_t nrows = X.rows();
    size_t ncols = Y.isEmpty() ? X.rows() : Y.rows();

#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
    for (int64_t i = 0; i < nrows; ++i) {
        if (Y.isEmpty()) {
            for (size_t j = i + 1; j < ncols; ++j) {
                auto pair = calcCor(X.row(i), X.row(j), X.cols());
                result[i * ncols + j] = pair.first;
            }
            for (size_t j = 0; j <= i; ++j) {
                if (i == j) { // diagonal, correlation with itself is set to 1.0
                    result[i * ncols + j] = 1.0;
                } else {
                    result[i * ncols + j] = result[j * ncols + i];
                }
            }
        } else {
            for (size_t j = 0; j < ncols; ++j) {
                auto pair = calcCor(X.row(i), Y.row(j), X.cols());
                result[i * ncols + j] = pair.first;
            }
        }
    }
}

std::pair<double, double> CorKendall::calcCor(const double *x, const double *y, size_t n) {
    // copy data for const reference
    std::unique_ptr<double[]> xcopy(new double[n]);
    std::unique_ptr<double[]> ycopy(new double[n]);
    std::memcpy(xcopy.get(), x, sizeof(double) * n);
    std::memcpy(ycopy.get(), y, sizeof(double) * n);
    auto xcopyPtr = xcopy.get();
    auto ycopyPtr = ycopy.get();

    uint64_t m1 = 0, m2 = 0;
    zipSort(xcopyPtr, ycopyPtr, n);
    auto npairs = n * (n - 1) / 2;
    auto s = static_cast<int64_t>(npairs);

    uint64_t nties = 0;
    size_t i;
    for (i = 1; i < n; ++i) {
        if (xcopyPtr[i] == xcopyPtr[i - 1]) {
            ++nties;
        } else if (nties > 0) {
            std::sort(ycopyPtr + i - nties - 1, ycopyPtr + i);
            m1 += nties * (nties + 1) / 2;
            s += static_cast<int64_t>(getMs(ycopyPtr + i - nties - 1, ycopyPtr + i));
            nties = 0;
        }
    }
    if (nties > 0) {
        std::sort(ycopyPtr + i - nties - 1, ycopyPtr + i);
        m1 += nties * (nties + 1) / 2;
        s += static_cast<int64_t>(getMs(ycopyPtr + i - nties - 1, ycopyPtr + i));
    }

    uint64_t swapCount = mergeSort(ycopyPtr, ycopyPtr + n);
    m2 = getMs(ycopyPtr, ycopyPtr + n);
    s -= (m1 + m2 + 2 * swapCount);

    if (m1 == npairs || m2 == npairs) { // all x or all y the same
        return std::make_pair(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
    }

    double denominator1 = npairs - m1;
    double denominator2 = npairs - m2;
    double tau = s / std::sqrt(denominator1 * denominator2);

    return std::make_pair(tau, s);
}

double CorKendall::calcPvalue(double s, const KendallStat &xstat, const KendallStat &ystat, const PTable &ptable) {
    if (std::isnan(s)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double v0 = xstat.v0;
    double vt = xstat.vt;
    double vu = ystat.vt;
    double v1 = xstat.v1 * ystat.v1;
    double v2 = xstat.v2 * ystat.v2;

    double var_s = (v0 - vt - vu) / 18 + v1 / xstat.n1 + v2 / xstat.n2;
    double z = s / std::sqrt(var_s);

    return ptable.getPvalue(z);
}

double CorKendall::commonCalcPvalue(double s, const KendallStat &xstat, const KendallStat &ystat,
                                    const boost::math::normal_distribution<> &dist) {
    if (std::isnan(s)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double v0 = xstat.v0;
    double vt = xstat.vt;
    double vu = ystat.vt;
    double v1 = xstat.v1 * ystat.v1;
    double v2 = xstat.v2 * ystat.v2;

    double var_s = (v0 - vt - vu) / 18 + v1 / xstat.n1 + v2 / xstat.n2;
    double z = s / std::sqrt(var_s);

    try {
        double ptmp = boost::math::cdf(dist, z);
        return 2 * std::min(ptmp, 1 - ptmp);
    } catch (const std::domain_error &e) {
        return std::numeric_limits<double>::quiet_NaN();
    }

}

std::vector<KendallStat> CorKendall::parallelGetKendallStat(const Matrix<double> &X, int nthreads) {
    std::vector<KendallStat> xStatsVec(X.rows());

#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
    for (int64_t i = 0; i < X.rows(); ++i) {
        auto ties = getTies(X.row(i), X.cols());
        xStatsVec[i] = getKendallStat(ties, X.cols());
    }

    return xStatsVec;
}

KendallStat CorKendall::getKendallStat(const std::vector<uint64_t> &ties, size_t n) {
    double v0 = n * (n - 1) * (2 * n + 5);
    double n1 = 2 * n * (n - 1);
    double n2 = 9 * n * (n - 1) * (n - 2);

    double vt = 0, v1 = 0, v2 = 0;
    for (unsigned long tie: ties) {
        double tmp = tie * (tie - 1);
        vt += (tmp * (2 * tie + 5));
        v1 += tmp;
        v2 += (tmp * (tie - 2));
    }

    return {v0, vt, v1, v2, n1, n2};
}

std::vector<uint64_t> CorKendall::getTies(const double *x, size_t n) {
    std::map<double, uint64_t> ties_map;

    for (size_t i = 0; i < n; ++i) {
        ++ties_map[*(x + i)];
    }

    std::vector<uint64_t> ties;
    for (auto &pair: ties_map) {
        if (pair.second > 1) {
            ties.push_back(pair.second);
        }
    }

    if (ties.size() == 0) {
        ties = {0};
    }

    return ties;
}

uint64_t CorKendall::insertionSort(double *v, size_t n) {
    uint64_t swapCount = 0;

    if (n < 2) {
        return 0;
    }

    for (size_t i = 1; i < n; ++i) {
        size_t j = i;
        double val = v[i];
        while (j > 0 && v[j - 1] > val) {
            v[j] = v[j - 1];
        }

        v[j] = val;
        swapCount += (i - j);
    }
    return swapCount;
}

uint64_t CorKendall::merge(double *begin, double *mid, double *end) {
    uint64_t swapCount = 0;
    double *left = begin;
    double *right = mid;
    std::unique_ptr<double[]> buf(new double[std::distance(begin, end)]);
    double *bufIter = buf.get();

    while (left != mid && right != end) {
        if (*right < *left) {
            *bufIter = *right;
            ++right;
            swapCount += std::distance(left, mid);
        } else {
            *bufIter = *left;
            ++left;
        }
        ++bufIter;
    }

    if (left != mid) {
        bufIter = std::copy(left, mid, bufIter);
    } else if (right != end) {
        bufIter = std::copy(right, end, bufIter);
    }
    std::copy(buf.get(), buf.get() + std::distance(begin, end), begin);
    return swapCount;
}

uint64_t CorKendall::mergeSort(double *begin, double *end) {
    uint64_t swapCount = 0;
    size_t len = end - begin;

    if (len < 2) {
        return 0;
    }

    double *mid = begin + len / 2;
    swapCount += mergeSort(begin, mid);
    swapCount += mergeSort(mid, end);
    swapCount += merge(begin, mid, end);
    return swapCount;
}

uint64_t CorKendall::getMs(double *begin, double *end) {
    uint64_t Ms = 0, tieCount = 0;

    for (auto i = (begin + 1); i != end; ++i) {
        if (*i == *(i - 1)) {
            ++tieCount;
        } else {
            Ms += tieCount * (tieCount + 1) / 2;
            tieCount = 0;
        }
    }

    if (tieCount > 0) {
        Ms += tieCount * (tieCount + 1) / 2;
    }
    return Ms;
}

void CorKendall::zipSort(double *x, double *y, size_t n) {
    std::vector<std::pair<double, double>> pairs(n);
    for (size_t i = 0; i < n; ++i) {
        pairs[i] = std::make_pair(x[i], y[i]);
    }

    std::sort(pairs.begin(), pairs.end(), [](const std::pair<double, double> &a, const std::pair<double, double> &b) {
        return a.first < b.first;
    });

    for (size_t i = 0; i < n; ++i) {
        x[i] = pairs[i].first;
        y[i] = pairs[i].second;
    }
}