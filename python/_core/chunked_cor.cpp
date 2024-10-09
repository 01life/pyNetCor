#include <queue>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <cor.h>
#include <preprocessor.h>
#include <ptable.h>

namespace py = pybind11;

using py_cdarray_t = py::array_t<double, py::array::c_style | py::array::forcecast>;

Matrix<double> convert_to_matrix(const py::object &obj);

void preprocess(Matrix<double> &X, CorrelationMethod method, int nthreads) {
    if (method == CorrelationMethod::Pearson) {
        CorPearson::parallelPreprocessNormalize(X, nthreads);
    } else if (method == CorrelationMethod::Spearman) {
        X = util::parallelNanRank(X, nthreads);
        CorPearson::parallelPreprocessNormalize(X, nthreads);
    }
}

class CorrcoefIter {
public:
    CorrcoefIter(Matrix<double> &X, Matrix<double> &Y, CorrelationMethod method, size_t chunkSize, int nthreads)
            : X_(std::move(X)), Y_(std::move(Y)), method_(method), chunkSize_(chunkSize), nthreads_(nthreads) {
        preprocess(X_, method_, nthreads_);

        if (!Y_.isEmpty()) {
            preprocess(Y_, method_, nthreads_);
        }
    }

    py_cdarray_t next() {
        if (nextIndex_ >= X_.rows()) {
            throw py::stop_iteration();
        }

        //  adjust chunk size to fit within X
        size_t currentChunkSize = std::min(chunkSize_, X_.rows() - nextIndex_);
        size_t resultCols = Y_.isEmpty() ? X_.rows() : Y_.rows();

        // initialize resultIter with chunkSize_ rows
        auto resultIter = py_cdarray_t(
                {currentChunkSize, resultCols},
                {resultCols * sizeof(double), sizeof(double)}
        );

        if (method_ == CorrelationMethod::Pearson || method_ == CorrelationMethod::Spearman) {
            size_t m = currentChunkSize;
            size_t k = X_.cols();

            openblas_set_num_threads(nthreads_);
            if (Y_.isEmpty()) {
                auto result = resultIter.mutable_unchecked<2>();

                Matrix<double> syrkMatrix(currentChunkSize, currentChunkSize);
                cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, m, k, 1.0, X_.row(nextIndex_), X_.cols(),
                            0.0, syrkMatrix.data(), m);

                Matrix<double> symmMatrix;
                if (nextIndex_ + currentChunkSize < X_.rows()) {
                    symmMatrix = Matrix<double>(currentChunkSize, X_.rows() - nextIndex_ - currentChunkSize);
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, X_.rows() - nextIndex_ - currentChunkSize,
                                k, 1.0, X_.row(nextIndex_), X_.cols(), X_.row(nextIndex_ + currentChunkSize),
                                X_.cols(), 0.0, symmMatrix.data(), X_.rows() - nextIndex_ - currentChunkSize);
                }

#pragma omp parallel for schedule(dynamic) num_threads(nthreads_)
                for (int64_t i = 0; i < currentChunkSize; ++i) {
                    for (size_t j = 0; j < X_.rows(); ++j) {
                        if (j < i + nextIndex_) {
                            result(i, j) = std::numeric_limits<double>::quiet_NaN();
                        } else if (j < nextIndex_ + currentChunkSize) {
                            result(i, j) = syrkMatrix(i, j - nextIndex_);
                        } else {
                            result(i, j) = symmMatrix(i, j - nextIndex_ - currentChunkSize);
                        }
                    }
                }
            } else {
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, Y_.rows(), k, 1.0, X_.row(nextIndex_), X_.cols(),
                            Y_.data(), Y_.cols(), 0.0, resultIter.mutable_data(), Y_.rows());
            }
        } else if (method_ == CorrelationMethod::Kendall) {
            auto result = resultIter.mutable_unchecked<2>();

#pragma omp parallel for schedule(dynamic) num_threads(nthreads_)
            for (int64_t i = nextIndex_; i < nextIndex_ + currentChunkSize; ++i) {
                for (size_t j = 0; j < Y_.rows(); ++j) {
                    auto pair = CorKendall::calcCor(X_.row(i), Y_.row(j), X_.cols());
                    result(i - nextIndex_, j) = pair.first;
                }
            }
        }

        nextIndex_ += currentChunkSize;
        return resultIter;
    }

private:
    Matrix<double> X_;
    Matrix<double> Y_;
    CorrelationMethod method_;
    size_t chunkSize_;
    int nthreads_;
    size_t nextIndex_ = 0;
    bool isYEmpty_ = false;
};

CorrcoefIter chunkedCorrcoef(const py::object &xobj, const std::optional<py::object> &yobj, const std::string &method,
                             const std::string &naAction, size_t chunkSize, int nthreads) {
    auto corMethod = stringToCorrelationMethod(method);
    auto naMethod = stringToNAMethod(naAction);

    // 'ignore' is not supported for Kendall
    if (corMethod == CorrelationMethod::Kendall && naMethod == NAMethod::Ignore) {
        throw std::invalid_argument("The 'ignore' method is not supported for Kendall");
    }

    // Convert python object(numpy array or list) to matrix
    Matrix<double> X = convert_to_matrix(xobj);
    if (X.cols() < 2) {
        throw std::invalid_argument("Input array must have length at least 2");
    }
    if (naMethod == NAMethod::FillMean || naMethod == NAMethod::FillMedian) {
        preprocessor::parallelProcessNan(X.data(), X.rows(), X.cols(), naMethod, nthreads);
    }

    Matrix<double> Y;
    // If yobj is not None, then we need to convert python object(numpy array or list) to matrix.
    // And preprocess the input matrix with nan filling.
    if (yobj.has_value()) {
        Y = convert_to_matrix(yobj.value());
        if (X.cols() != Y.cols()) {
            throw std::invalid_argument("Input arrays x and y must have the same number of columns");
        }

        if (naMethod == NAMethod::FillMean || naMethod == NAMethod::FillMedian) {
            preprocessor::parallelProcessNan(X.data(), X.rows(), X.cols(), naMethod, nthreads);
        }
    }

    return CorrcoefIter(X, Y, corMethod, chunkSize, nthreads);
}

class CortestIter {
public:
    CortestIter(Matrix<double> &X, Matrix<double> &Y, CorrelationMethod corMethod, bool isPvalueApprox,
                bool isMultipletest, PAdjustMethod pAdjustMethod, size_t chunkSize, int nthreads)
            : X_(std::move(X)), Y_(std::move(Y)), corMethod_(corMethod), isPvalueApprox_(isPvalueApprox),
              isMultipletest_(isMultipletest), chunkSize_(chunkSize), nthreads_(nthreads) {
        preprocess(X_, corMethod_, nthreads_);

        size_t resultCorSize = X_.rows() * (X_.rows() - 1) / 2;
        if (Y_.isEmpty()) {
            isYEmpty_ = true;
            Y_ = X_;
        } else {
            preprocess(Y_, corMethod_, nthreads_);
            resultCorSize = X_.rows() * Y_.rows();
        }

        if (corMethod_ == CorrelationMethod::Pearson || corMethod_ == CorrelationMethod::Spearman) {
            if (isPvalueApprox_) {
                ptable_ = PTable(DistributionType::T, X_.cols() - 2);
            } else {
                distT_ = boost::math::students_t(X_.cols() - 2);
            }
        } else if (corMethod_ == CorrelationMethod::Kendall) {
            if (isPvalueApprox_) {
                ptable_ = PTable(DistributionType::Normal);
            }

            xStatsVec_ = CorKendall::parallelGetKendallStat(X_, nthreads_);
            if (isYEmpty_) {
                yStatsVec_ = xStatsVec_;
            } else {
                yStatsVec_ = CorKendall::parallelGetKendallStat(Y_, nthreads_);
            }
        }

        PAdjustTable qtable;
        if (isMultipletest_) {
            qtable_ = PAdjustTable(X_, Y_, resultCorSize, corMethod, pAdjustMethod, nthreads);
        }
        //  result matrix with 4 columns: [index1, index2, cor, pvalue] or 5 columns: [index1, index2, cor, pvalue, qvalue]
        resultCols_ = isMultipletest_ ? 5 : 4;
    }

    py_cdarray_t next() {
        if (nextIndex_ >= X_.rows()) {
            throw py::stop_iteration();
        }

        //  adjust chunk size to fit within X
        size_t currentChunkSize = std::min(chunkSize_, X_.rows() - nextIndex_);

        // initialize resultIter with chunkSize_ rows
        size_t resultIterRows = currentChunkSize * Y_.rows();
        if (isYEmpty_) {
            resultIterRows -= (nextIndex_ + 1 + nextIndex_ + currentChunkSize) * currentChunkSize / 2;
        }
        auto resultIter = py_cdarray_t({resultIterRows, resultCols_}, {resultCols_ * sizeof(double), sizeof(double)});
        auto result = resultIter.mutable_unchecked<2>();

        if (corMethod_ == CorrelationMethod::Pearson || corMethod_ == CorrelationMethod::Spearman) {
            Matrix<double> resultCor(currentChunkSize, Y_.rows());
            size_t m = currentChunkSize;
            size_t k = X_.cols();
            size_t n = Y_.rows();

            openblas_set_num_threads(nthreads_);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0, X_.row(nextIndex_), X_.cols(),
                        Y_.data(), Y_.cols(), 0.0, resultCor.data(), n);

            double df = X_.cols() - 2;
#pragma omp parallel for schedule(dynamic) num_threads(nthreads_)
            for (int64_t i = nextIndex_; i < nextIndex_ + currentChunkSize; ++i) {
                size_t j = isYEmpty_ ? i + 1 : 0;
                for (; j < Y_.rows(); ++j) {
                    size_t index1 = isYEmpty_ ? util::transFullMatIndex(i, j, Y_.rows(), nextIndex_) :
                                    (i - nextIndex_) * Y_.rows() + j;
                    result(index1, 0) = i;
                    result(index1, 1) = j;
                    result(index1, 2) = resultCor(i - nextIndex_, j);
                    if (isPvalueApprox_) {
                        result(index1, 3) = CorPearson::calcPvalue(resultCor(i - nextIndex_, j), df,
                                                                   ptable_);
                    } else {
                        result(index1, 3) = CorPearson::commonCalcPvalue(resultCor(i - nextIndex_, j),
                                                                         df, distT_);
                    }
                    if (isMultipletest_) {
                        result(index1, 4) = qtable_.getQvalue(result(index1, 3));
                    }
                }
            }
        } else if (corMethod_ == CorrelationMethod::Kendall) {
#pragma omp parallel for schedule(dynamic) num_threads(nthreads_)
            for (int64_t i = nextIndex_; i < nextIndex_ + currentChunkSize; ++i) {
                size_t j = isYEmpty_ ? i + 1 : 0;
                for (; j < Y_.rows(); ++j) {
                    size_t index1 = isYEmpty_ ? util::transFullMatIndex(i, j, Y_.rows(), nextIndex_) :
                                    (i - nextIndex_) * Y_.rows() + j;
                    result(index1, 0) = i;
                    result(index1, 1) = j;
                    auto pair = CorKendall::calcCor(X_.row(i), Y_.row(j), X_.cols());
                    result(index1, 2) = pair.first;
                    if (isPvalueApprox_) {
                        result(index1, 3) = CorKendall::calcPvalue(pair.second, xStatsVec_[i],
                                                                   yStatsVec_[j], ptable_);
                    } else {
                        result(index1, 3) = CorKendall::commonCalcPvalue(pair.second, xStatsVec_[i],
                                                                         yStatsVec_[j], distNorm_);
                    }
                    if (isMultipletest_) {
                        result(index1, 4) = qtable_.getQvalue(result(index1, 3));
                    }
                }
            }
        }

        nextIndex_ += currentChunkSize;
        return resultIter;
    }


private:
    Matrix<double> X_;
    Matrix<double> Y_;
    CorrelationMethod corMethod_;
    bool isPvalueApprox_;
    size_t chunkSize_;
    size_t resultCols_;
    int nthreads_;
    size_t nextIndex_ = 0;
    bool isYEmpty_ = false;

    // object for calculating p-value
    PTable ptable_;
    boost::math::students_t distT_ = boost::math::students_t(1);
    boost::math::normal_distribution<> distNorm_ = boost::math::normal_distribution<>(0, 1);

    // object for correction p-value for multiple hypothesis testing
    PAdjustTable qtable_;
    bool isMultipletest_ = false;

    // statistics of Kendall correlation which is used for p-value calculation
    std::vector<KendallStat> xStatsVec_;
    std::vector<KendallStat> yStatsVec_;
};

CortestIter chunkedCortest(const py::object &xobj, const std::optional<py::object> &yobj, const std::string &method,
                           const std::string &naAction, bool isPvalueApprox, bool isMutipletest,
                           const std::string &multipletestMethod, size_t chunkSize, int nthreads) {
    auto corMethod = stringToCorrelationMethod(method);
    auto naMethod = stringToNAMethod(naAction);
    auto pAdjustMethod = stringToPAdjustMethod(multipletestMethod);

    // 'ignore' is not supported for Kendall
    if (corMethod == CorrelationMethod::Kendall && naMethod == NAMethod::Ignore) {
        throw std::invalid_argument("The 'ignore' method is not supported for Kendall");
    }

    // Convert python object(numpy array or list) to matrix
    Matrix<double> X = convert_to_matrix(xobj);
    if (X.cols() < 2) {
        throw std::invalid_argument("Input array must have length at least 2");
    }
    if (naMethod == NAMethod::FillMean || naMethod == NAMethod::FillMedian) {
        preprocessor::parallelProcessNan(X.data(), X.rows(), X.cols(), naMethod, nthreads);
    }

    size_t resultCorSize = X.rows() * (X.rows() - 1) / 2;
    // If yobj is not None, then we need to convert python object(numpy array or list) to matrix.
    // And preprocess the input matrix with nan filling.
    Matrix<double> Y;
    if (yobj.has_value()) {
        Y = convert_to_matrix(yobj.value());
        if (X.cols() != Y.cols()) {
            throw std::invalid_argument("Input arrays x and y must have the same number of columns");
        }

        resultCorSize = X.rows() * Y.rows();
        if (naMethod == NAMethod::FillMean || naMethod == NAMethod::FillMedian) {
            preprocessor::parallelProcessNan(X.data(), X.rows(), X.cols(), naMethod, nthreads);
        }
    }

    // If user specify isPvalueApprox and resultCorSize > (2 * PTABLE_SIZE),
    // then we need to initialize PTable used to calculate approximate p-value
    if (isPvalueApprox && resultCorSize < (2 * PTABLE_SIZE)) {
        isPvalueApprox = false;
    }

    return CortestIter(X, Y, corMethod, isPvalueApprox, isMutipletest, pAdjustMethod, chunkSize, nthreads);
}

struct CorPvalue {
    double key;
    double i;
    double j;
    double cor;
    double pvalue;

    bool operator<(const CorPvalue &other) const {
        return key > other.key;
    }
};

py_cdarray_t corTopk(const py::object &xobj, const std::optional<py::object> &yobj, const std::string &method,
                     double k, const std::string &naAction, const std::string &corMode, bool computePvalue,
                     bool isPvalueApprox, size_t chunkSize, int nthreads) {
//    const CHUNK_SIZE = 1024;

    auto corMethod = stringToCorrelationMethod(method);
    auto naMethod = stringToNAMethod(naAction);
    auto topkCorMode = stringToTopkCorrelationMode(corMode);

    // 'ignore' is not supported for Kendall
    if (corMethod == CorrelationMethod::Kendall && naMethod == NAMethod::Ignore) {
        throw std::invalid_argument("The 'ignore' method is not supported for Kendall");
    }

    // Convert python object(numpy array or list) to matrix
    Matrix<double> X = convert_to_matrix(xobj);
    if (X.cols() < 2) {
        throw std::invalid_argument("Input array must have length at least 2");
    }
    if (naMethod == NAMethod::FillMean || naMethod == NAMethod::FillMedian) {
        preprocessor::parallelProcessNan(X.data(), X.rows(), X.cols(), naMethod, nthreads);
    }

    Matrix<double> Y;
    size_t totalCors = X.rows() * (X.rows() - 1) / 2;
    size_t topkCorsSize = k > 1 ? static_cast<size_t>(k) : std::floor(totalCors * k);
    // If yobj is not None, then we need to convert python object(numpy array or list) to matrix.
    // And preprocess the input matrix with nan filling.
    if (yobj.has_value()) {
        Y = convert_to_matrix(yobj.value());
        if (X.cols() != Y.cols()) {
            throw std::invalid_argument("Input arrays x and y must have the same number of columns");
        }

        if (naMethod == NAMethod::FillMean || naMethod == NAMethod::FillMedian) {
            preprocessor::parallelProcessNan(X.data(), X.rows(), X.cols(), naMethod, nthreads);
        }

        totalCors = X.rows() * Y.rows();
        topkCorsSize = k > 1 ? static_cast<size_t>(k) : std::floor(totalCors * k);
    } else {
        Y = X;
    }

    // declare a priority_queue to store top-k correlations in ascending order
    std::priority_queue<CorPvalue> topkCorQueue;

    // compute p-value by CortestIter
    if (computePvalue) {
        // Iterating through the CortestIter and insert correlations into topkCorMap that are greater than the
        // beginning of the container. The iterator is parallelized with nthreads. But the comparison and insert
        // operations are not parallelized.
        auto iterator = CortestIter(X, Y, corMethod, isPvalueApprox, false, PAdjustMethod::BH, chunkSize, nthreads);
        while (true) {
            py_cdarray_t resultIter;
            try {
                resultIter = iterator.next();
            } catch (py::stop_iteration &) {
                break;
            }

            auto result = resultIter.unchecked<2>();
            for (size_t i = 0; i < resultIter.shape()[0]; i++) {
                // acquire correlation for key of topkCorMap
                auto key = result(i, 2);
                if (topkCorMode == TopkCorrelationMode::Negative) {
                    key = -key;
                } else if (topkCorMode == TopkCorrelationMode::Both) {
                    key = std::abs(key);
                }

                if (topkCorQueue.size() < topkCorsSize) {
                    topkCorQueue.push({key, result(i, 0), result(i, 1), result(i, 2), result(i, 3)});
                } else if (key > topkCorQueue.top().key) {
                    topkCorQueue.pop();
                    topkCorQueue.push({key, result(i, 0), result(i, 1), result(i, 2), result(i, 3)});
                }
            }
        }
    } else {
        auto iterator = CorrcoefIter(X, Y, corMethod, chunkSize, nthreads);
        size_t start_index = 0;

        while (true) {
            py_cdarray_t resultIter;
            try {
                resultIter = iterator.next();
            } catch (py::stop_iteration &) {
                break;
            }

            auto result = resultIter.unchecked<2>();

            for (size_t i = start_index; i < start_index + resultIter.shape()[0]; i++) {
                size_t j = yobj.has_value() ? 0 : i + 1;
                for (; j < resultIter.shape()[1]; j++) {
                    // find correlation from matrix
                    auto cor = result(i - start_index, j);

                    // acquire correlation for key of topkCorMap
                    auto key = cor;
                    if (topkCorMode == TopkCorrelationMode::Negative) {
                        key = -key;
                    } else if (topkCorMode == TopkCorrelationMode::Both) {
                        key = std::abs(key);
                    }

                    if (topkCorQueue.size() < topkCorsSize) {
                        topkCorQueue.push({key, static_cast<double>(i), static_cast<double>(j), cor, 0});
                    } else if (key > topkCorQueue.top().key) {
                        topkCorQueue.pop();
                        topkCorQueue.push({key, static_cast<double>(i), static_cast<double>(j), cor, 0});
                    }
                }
            }

            start_index += resultIter.shape()[0];
        }
    }

    // convert topkCorMap to topkCorsArray
    size_t topkCorsArrayCols = computePvalue ? 4 : 3;
    py_cdarray_t topkCorsArray({topkCorsSize, topkCorsArrayCols}, {topkCorsArrayCols * sizeof(double), sizeof(double)});
    auto topkCors = topkCorsArray.mutable_unchecked<2>();

    size_t ir = topkCorQueue.size() - 1;
    while (!topkCorQueue.empty()) {
        auto topkCor = topkCorQueue.top();
        topkCorQueue.pop();
        topkCors(ir, 0) = topkCor.i;
        topkCors(ir, 1) = topkCor.j;
        topkCors(ir, 2) = topkCor.cor;
        if (computePvalue) {
            topkCors(ir, 3) = topkCor.pvalue;
        }
        ir--;
    }

    return topkCorsArray;
}

struct CorDiff {
    double key;
    double i;
    double j;
    double diffCor;
    double cor1;
    double cor2;

    bool operator<(const CorDiff &other) const {
        return key > other.key;
    }
};

py_cdarray_t corTopkDiff(const py::object &xobj1, const py::object &yobj1,
                        const std::optional<py::object> &xobj2, const std::optional<py::object> &yobj2,
                        const std::string &method, double k, const std::string &naAction,
                        size_t chunkSize, int nthreads) {
    auto corMethod = stringToCorrelationMethod(method);
    auto naMethod = stringToNAMethod(naAction);

    // 'ignore' is not supported for Kendall
    if (corMethod == CorrelationMethod::Kendall && naMethod == NAMethod::Ignore) {
        throw std::invalid_argument("The 'ignore' method is not supported for Kendall");
    }

    // Convert python object(numpy array or list) to matrix
    Matrix<double> X1 = convert_to_matrix(xobj1);
    Matrix<double> Y1 = convert_to_matrix(yobj1);
    if ((X1.cols() < 2) || (Y1.cols() < 2)) {
        throw std::invalid_argument("Input array must have length at least 2");
    }
    if (X1.rows() != Y1.rows()) {
        throw std::invalid_argument("Input arrays x1 and y1 must have the same number of features");
    }

    if (naMethod == NAMethod::FillMean || naMethod == NAMethod::FillMedian) {
        preprocessor::parallelProcessNan(X1.data(), X1.rows(), X1.cols(), naMethod, nthreads);
        preprocessor::parallelProcessNan(Y1.data(), Y1.rows(), Y1.cols(), naMethod, nthreads);
    }

    Matrix<double> X2, Y2;
    size_t totalCors = X1.rows() * (X1.rows() - 1) / 2;
    size_t topkDiffCorsSize = k >= 1 ? static_cast<size_t>(k) : std::floor(totalCors * k);
    // If xobj2 and yobj2 are not None, then we need to convert python object(numpy array or list) to matrix.
    // And preprocess the input matrix with nan filling.
    bool symmetric = true;
    if (xobj2.has_value() && yobj2.has_value()) {
        symmetric = false;

        X2 = convert_to_matrix(xobj2.value());
        Y2 = convert_to_matrix(yobj2.value());

        if (X1.cols() != X2.cols()) {
            throw std::invalid_argument("Input arrays x1 and x2 must have the same number of columns");
        }
        if (Y1.cols() != Y2.cols()) {
            throw std::invalid_argument("Input arrays y1 and y2 must have the same number of columns");
        }

        if (naMethod == NAMethod::FillMean || naMethod == NAMethod::FillMedian) {
            preprocessor::parallelProcessNan(X2.data(), X2.rows(), X2.cols(), naMethod, nthreads);
            preprocessor::parallelProcessNan(Y2.data(), Y2.rows(), Y2.cols(), naMethod, nthreads);
        }

        totalCors = X1.rows() * X2.rows();
        topkDiffCorsSize = k > 1 ? static_cast<size_t>(k) : std::floor(totalCors * k);
    }

    // declare a multimap to store top-k differencial correlations in ascending order
    std::priority_queue<CorDiff> topkDiffCorQueue;

    auto iterator1 = CorrcoefIter(X1, X2, corMethod, chunkSize, nthreads);
    auto iterator2 = CorrcoefIter(Y1, Y2, corMethod, chunkSize, nthreads);
    size_t start_index = 0;

    while (true) {
        py_cdarray_t resultIter1, resultIter2;
        try {
            resultIter1 = iterator1.next();
            resultIter2 = iterator2.next();
        } catch (py::stop_iteration &) {
            break;
        }

        auto result1 = resultIter1.unchecked<2>();
        auto result2 = resultIter2.unchecked<2>();

        for (size_t i = start_index; i < start_index + resultIter1.shape()[0]; i++) {
            size_t j = symmetric ? i + 1 : 0;
            for (; j < resultIter1.shape()[1]; j++) {
                // find correlation from matrix
                auto cor1 = result1(i - start_index, j);
                auto cor2 = result2(i - start_index, j);
                auto diffCor = cor1 - cor2;
                // acquire correlation for key of topkCorMap
                auto key = std::abs(diffCor);

                if (topkDiffCorQueue.size() < topkDiffCorsSize) {
                    topkDiffCorQueue.push({key, static_cast<double>(i), static_cast<double>(j), diffCor, cor1, cor2});
                } else if (key > topkDiffCorQueue.top().key) {
                    topkDiffCorQueue.pop();
                    topkDiffCorQueue.push({key, static_cast<double>(i), static_cast<double>(j), diffCor, cor1, cor2});
                }
            }
        }

        start_index += resultIter1.shape()[0];
    }

    // convert topkDiffCorMap to topkDiffCorsArray
    size_t topkDiffCorsArrayCols = 5;
    py_cdarray_t topkDiffCorsArray({topkDiffCorsSize, topkDiffCorsArrayCols},
                                   {topkDiffCorsArrayCols * sizeof(double), sizeof(double)});
    auto topkDiffCors = topkDiffCorsArray.mutable_unchecked<2>();

    size_t ir = topkDiffCorQueue.size() - 1;
    while (!topkDiffCorQueue.empty()) {
        auto topkDiffCor = topkDiffCorQueue.top();
        topkDiffCors(ir, 0) = topkDiffCor.i;
        topkDiffCors(ir, 1) = topkDiffCor.j;
        topkDiffCors(ir, 2) = topkDiffCor.diffCor;
        topkDiffCors(ir, 3) = topkDiffCor.cor1;
        topkDiffCors(ir, 4) = topkDiffCor.cor2;
        topkDiffCorQueue.pop();
        ir--;
    }

    return topkDiffCorsArray;
}

void bind_chunked_cor(py::module &m) {
    py::class_<CorrcoefIter>(m, "CorrcoefIter")
            .def("__iter__", [](CorrcoefIter &self) { return &self; })
            .def("__next__", &CorrcoefIter::next);

    m.def("chunkedCorrcoef", &chunkedCorrcoef,
          "Calculate chunked correlation matrix, support pearson, spearman and kendall", py::arg("x"),
          py::arg("y"), py::arg("method"), py::arg("naAction"), py::arg("chunkSize"), py::arg("nthreads"));

    py::class_<CortestIter>(m, "CortestIter")
            .def("__iter__", [](CortestIter &self) { return &self; })
            .def("__next__", &CortestIter::next);

    m.def("chunkedCortest", &chunkedCortest,
          "Calculate chunked correlation matrix, support pearson, spearman and kendall", py::arg("x"), py::arg("y"),
          py::arg("method"), py::arg("naAction"), py::arg("isPvalueApprox"), py::arg("isMultipletest"),
          py::arg("multipletestMethod"), py::arg("chunkSize"), py::arg("nthreads"));

    m.def("corTopk", &corTopk,
          "Calculate top-k correlation matrix, support pearson, spearman and kendall", py::arg("x"), py::arg("y"),
          py::arg("method"), py::arg("k"), py::arg("naAction"), py::arg("corMode"), py::arg("computePvalue"),
          py::arg("isPvalueApprox"), py::arg("chunkSize"), py::arg("nthreads"));

    m.def("corTopkDiff", &corTopkDiff,
          "Calculate top-k differencial correlation matrix, support pearson, spearman and kendall",
          py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"),
          py::arg("method"), py::arg("k"), py::arg("naAction"), py::arg("chunkSize"), py::arg("nthreads"));
}
