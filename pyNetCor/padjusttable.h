#ifndef PADJUSTTABLE_H
#define PADJUSTTABLE_H

#include <random>

#include "matrix.h"
#include "options.h"

const size_t PADJUSTTABLE_SIZE = 500000;

// calculate approximate adjusted P values, based on linear interpolation.
class PAdjustTable {
public:
    PAdjustTable() = default;

    // P-values used for construction are not allowed to have nan.
    PAdjustTable(const Matrix<double> &X, const Matrix<double> &Y, size_t n, CorrelationMethod corMethod,
                 PAdjustMethod pAdjustMethod = PAdjustMethod::BH, int nthreads=1);

    ~PAdjustTable();

    double getQvalue(double p) const;

    void parallelCalcPAdjust(const double *P, double *pAdjusted, size_t count, int num_threads) const;

    static std::vector<double> commonPAdjust(const double *P, size_t pLen,
                                             const PAdjustMethod &pAdjustMethod = PAdjustMethod::BH, size_t n = 0);

//    static void commonPAdjust(const double *P, double *pAdjusted, size_t pLen,
//                              const PAdjustMethod &pAdjustMethod = PAdjustMethod::BH);

private:
    CorrelationMethod corMethod_;
    PAdjustMethod pAdjustMethod_;
    std::pair<std::vector<double>, std::vector<double>> table_;
    double pMax_;
    double pMin_;
    double qMax_;
    double qMin_;
    size_t n_;
    // The first element in each row of the upper triangular matrix (excluding the diagonal) in terms of its index.
    std::vector<size_t> indexs_;

    // random generator for sampling pair data from X and Y
    std::mt19937 gen_ = std::mt19937(42);
    std::uniform_int_distribution<> dist_;

    void initTable(const std::vector<double> &P);

    void randomSamplePair(const Matrix<double> &X, const Matrix<double> &Y, double *x, double *y);

    double calcCor(double *x, double*y, size_t n) const;

    static void adjustBonferroni(std::vector<double> &P, size_t n);

    static void adjustHolm(std::vector<double> &P, size_t n);

    static void adjustHochberg(std::vector<double> &P, size_t n);

    static void adjustBH(std::vector<double> &P);

    static void adjustBY(std::vector<double> &P, size_t n);
};

#endif // PADJUSTTABLE_H
