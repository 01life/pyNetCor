#include "cor.h"
#include "padjusttable.h"
#include "ptable.h"
#include "util.h"

PAdjustTable::PAdjustTable(const Matrix<double> &X, const Matrix<double> &Y, size_t n, CorrelationMethod corMethod,
                           PAdjustMethod pAdjustMethod, int nthreads) :
        corMethod_(corMethod), pAdjustMethod_(pAdjustMethod), n_(n) {
    // Bonferroni adjustment can be measured with no need for linear interpolation
    if (pAdjustMethod_ == PAdjustMethod::Bonferroni) {
        return;
    }

    dist_ = std::uniform_int_distribution<>(0, n_ - 1);
    if (Y.isEmpty()) {
        indexs_.resize(X.rows());
        size_t index = 0;
        for (size_t i = 0; i < X.rows() - 1; ++i) {
            indexs_[i] = index;
            index += X.rows() - i - 1;
        }
    }

    std::vector<double> P(PADJUSTTABLE_SIZE);
    std::unique_ptr<double[]> x(new double[X.cols()]);
    std::unique_ptr<double[]> y(new double[X.cols()]);
    std::unique_ptr<double[]> cors(new double[PADJUSTTABLE_SIZE]);

    // generate random sample pairs and calculate P-values
    double df = X.cols() - 2;
    std::vector<KendallStat> xStatVec, yStatVec;
    PTable ptable;
    if (corMethod == CorrelationMethod::Pearson || corMethod == CorrelationMethod::Spearman) {
        ptable = PTable(DistributionType::T, df);
    } else {
        ptable = PTable(DistributionType::Normal);
        xStatVec.resize(PADJUSTTABLE_SIZE);
        yStatVec.resize(PADJUSTTABLE_SIZE);
    }

    for (size_t i = 0; i < PADJUSTTABLE_SIZE; ++i) {
        do {
            randomSamplePair(X, Y, x.get(), y.get());
            if (corMethod_ == CorrelationMethod::Pearson) {
                cors[i] = CorPearson::calcCor(x.get(), y.get(), X.cols());
            } else if (corMethod_ == CorrelationMethod::Spearman) {
                cors[i] = CorSpearman::calcCor(x.get(), y.get(), X.cols());
            } else if (corMethod_ == CorrelationMethod::Kendall) {
                auto corPair = CorKendall::calcCor(x.get(), y.get(), X.cols());
                cors[i] = corPair.second;
                xStatVec[i] = CorKendall::getKendallStat(CorKendall::getTies(x.get(), X.cols()), X.cols());
                yStatVec[i] = CorKendall::getKendallStat(CorKendall::getTies(y.get(), X.cols()), X.cols());
            }
        } while (std::isnan(cors[i]));
    }

    // calculate P-values in parallel
#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
    for (int64_t i = 0; i < PADJUSTTABLE_SIZE; ++i) {
        if (corMethod_ == CorrelationMethod::Pearson || corMethod_ == CorrelationMethod::Spearman) {
            P[i] = CorPearson::calcPvalue(cors[i], df, ptable);
        } else if (corMethod_ == CorrelationMethod::Kendall) {
            P[i] = CorKendall::calcPvalue(cors[i], xStatVec[i], yStatVec[i], ptable);
        }
    }

    // initialize table of q-values with P-values
    initTable(P);
}

PAdjustTable::~PAdjustTable() {}

double PAdjustTable::getQvalue(double p) const {
    if (std::isnan(p)) {
        return NAN;
    }

    if (pAdjustMethod_ == PAdjustMethod::Bonferroni) {
        return std::min(p * n_, 1.0);
    }

    if (p >= pMax_) {
        return qMax_;
    } else if (p <= pMin_) {
        return qMin_;
    } else {
        auto it = std::lower_bound(table_.first.begin(), table_.first.end(), p);
        size_t index = it - table_.first.begin();
        double q = table_.second[index - 1] +
                   (table_.second[index] - table_.second[index - 1]) /
                   (table_.first[index] - table_.first[index - 1]) *
                   (p - table_.first[index - 1]);
        return q;
    }
}

void PAdjustTable::parallelCalcPAdjust(const double *P, double *pAdjusted, size_t count, int num_threads) const {
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (int64_t i = 0; i < count; ++i) {
        pAdjusted[i] = getQvalue(*(P + i));
    }
}

void PAdjustTable::initTable(const std::vector<double> &P) {
    std::vector<double> pVals(P);
    std::sort(pVals.begin(), pVals.end());
    auto qVals = commonPAdjust(pVals.data(), pVals.size(), pAdjustMethod_, n_);
    pMax_ = pVals.back();
    pMin_ = pVals.front();
    qMax_ = qVals.back();
    qMin_ = qVals.front();
    table_ = std::make_pair(pVals, qVals);
}

void PAdjustTable::randomSamplePair(const Matrix<double> &X, const Matrix<double> &Y, double *x, double *y) {
    size_t index = dist_(gen_);
    size_t xIndex, yIndex;

    if (Y.isEmpty()) {
        auto it = std::lower_bound(indexs_.begin(), indexs_.end(), index);
        xIndex = std::distance(indexs_.begin(), it);
        if (*it > index) {
            xIndex -= 1;
        }
        yIndex = index + (xIndex + 1) * (xIndex + 2) / 2 - xIndex * X.rows();
        std::memcpy(y, X.row(yIndex), X.cols() * sizeof(double));
    } else {
        xIndex = index / Y.rows();
        yIndex = index % Y.rows();
        std::memcpy(y, Y.row(yIndex), Y.cols() * sizeof(double));
    }
    std::memcpy(x, X.row(xIndex), X.cols() * sizeof(double));
}

std::vector<double> PAdjustTable::commonPAdjust(const double *P, size_t pLen, const PAdjustMethod &pAdjustMethod,
                                                size_t n) {
    if (n == 0) n = pLen;
    if (pLen <= 1) {
        std::vector<double> pAdjusted(pLen);
        std::copy(P, P + pLen, pAdjusted.begin());
        return pAdjusted;
    }

    size_t nanNum = 0;
    std::vector<double> P1;
    for (size_t i = 0; i < pLen; ++i) {
        if (std::isnan(P[i])) {
            ++nanNum;
        } else {
            P1.push_back(P[i]);
        }
    }

    switch (pAdjustMethod) {
        case PAdjustMethod::Bonferroni:
            adjustBonferroni(P1, n);
            break;
        case PAdjustMethod::Holm:
            adjustHolm(P1, n);
            break;
        case PAdjustMethod::Hochberg:
            adjustHochberg(P1, n);
            break;
        case PAdjustMethod::BH:
            adjustBH(P1);
            break;
        case PAdjustMethod::BY:
            adjustBY(P1, n);
            break;
        default:
            throw std::runtime_error("unknown pAdjustMethod: " + toString(pAdjustMethod));
    }

    std::vector<double> pAdjusted(pLen);
    if (nanNum > 0) {
        size_t j = 0;
        for (size_t i = 0; i < pLen; ++i) {
            if (std::isnan(P[i])) {
                pAdjusted[i] = NAN;
            } else {
                pAdjusted[i] = P1[j++];
            }
        }
    } else {
        // std::copy(P1.cbegin(), P1.cend(), pAdjusted);
        pAdjusted = std::move(P1);
    }
    return pAdjusted;
}

// Bonferroni
void PAdjustTable::adjustBonferroni(std::vector<double> &P, size_t n) {
    std::vector<double> pAdjusted(P.size());
    std::transform(P.begin(), P.end(), P.begin(), [n](double x) {
        return std::min(1.0, x * n);
    });
}

// Holm-Bonferroni
void PAdjustTable::adjustHolm(std::vector<double> &P, size_t n) {
    auto pLen = P.size();
    std::vector<size_t> order = util::argSort(P.data(), pLen);

    std::vector<double> tmp(pLen);
    double factor = n / pLen;
    double pMax = pLen * factor * P.at(order.at(0));
    for (size_t i = 0; i < pLen; ++i) {
        auto p = (pLen - i) * factor * P.at(order.at(i));
        if (p > pMax) pMax = p;
        tmp.at(i) = std::min(1.0, pMax);
    }

    for (size_t j = 0; j < pLen; ++j) {
        P.at(order.at(j)) = tmp.at(j);
    }
}

// Hochberg
void PAdjustTable::adjustHochberg(std::vector<double> &P, size_t n) {
    auto pLen = P.size();
    std::vector<size_t> order = util::argSort(P.data(), pLen, true);

    std::vector<double> tmp(pLen);
    double factor = n / pLen;
    double pMin = factor * P.at(order.at(0));
    for (size_t i = 0; i < pLen; ++i) {
        auto p = (i + 1) * factor * P.at(order.at(i));
        if (p < pMin) pMin = p;
        tmp.at(i) = std::min(1.0, pMin);
    }

    for (size_t i = 0; i < pLen; ++i) {
        P.at(order.at(i)) = tmp.at(i);
    }
}

// Benjamini-Hochberg
void PAdjustTable::adjustBH(std::vector<double> &P) {
    auto pLen = P.size();
    std::vector<size_t> order = util::argSort(P.data(), pLen, true);

    std::vector<double> tmp(pLen);
    double pMin = P.at(order.at(0));
    for (size_t i = 0; i < pLen; ++i) {
        auto p = pLen * 1.0 / (pLen - i) * P.at(order.at(i));
        if (p < pMin) pMin = p;
        tmp.at(i) = std::min(1.0, pMin);
    }

    for (size_t i = 0; i < pLen; ++i) {
        P.at(order.at(i)) = tmp.at(i);
    }
}

// Benjamini-Yekutieli
void PAdjustTable::adjustBY(std::vector<double> &P, size_t n) {
    auto pLen = P.size();
    std::vector<size_t> order = util::argSort(P.data(), pLen, true);

    double q = 0;
    for (size_t i = 1; i < n + 1; ++i) {
        q += 1.0 / i;
    }

    std::vector<double> tmp(pLen);
    double pMin = q * P.at(order.at(0));
    for (size_t i = 0; i < pLen; ++i) {
        auto p = q * pLen / (pLen - i) * P.at(order.at(i));
        if (p < pMin) pMin = p;
        tmp.at(i) = std::min(1.0, pMin);
    }

    for (size_t i = 0; i < pLen; ++i) {
        P.at(order.at(i)) = tmp.at(i);
    }
}