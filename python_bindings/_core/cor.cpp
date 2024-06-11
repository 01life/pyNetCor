#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <cor.h>
#include <preprocessor.h>
#include <padjusttable.h>
#include <ptable.h>

namespace py = pybind11;

using py_cdarray_t = py::array_t<double, py::array::c_style | py::array::forcecast>;

Matrix<double> convert_to_matrix(const py::object &obj) {
    auto arr = obj.cast<py_cdarray_t>();
//    std::cerr << "Input array shape: ";
//    for (size_t i = 0; i < arr.ndim() - 1; ++i) {
//        std::cerr << arr.shape()[i] << " x ";
//    }
//    std::cerr << arr.shape()[arr.ndim() - 1] << std::endl;

    Matrix<double> mat;
    if (arr.ndim() == 1) {
        mat.resize(1, arr.shape()[0]);
    } else if (arr.ndim() == 2) {
        mat.resize(arr.shape()[0], arr.shape()[1]);
    } else {
        throw std::invalid_argument("Input must be 1D or 2D array");
    }

    std::memcpy(mat.data(), arr.data(), arr.size() * sizeof(double));
    //    std::copy(arr.data(), arr.data() + arr.size(), mat.data());
    return mat;
}

py_cdarray_t corrcoef(const py::object &xobj, const std::optional<py::object> &yobj, const std::string &method,
                      const std::string &naAction, int nthreads) {
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

    size_t resultRows = X.rows();
    size_t resultCols = X.rows();
    Matrix<double> Y;
    // If yobj is not None, then we need to convert python object(numpy array or list) to matrix.
    // And preprocess the input matrix with nan filling.
    if (yobj.has_value()) {
        Y = convert_to_matrix(yobj.value());
        if (X.cols() != Y.cols()) {
            throw std::invalid_argument("Input arrays x and y must have the same number of columns");
        }

        resultCols = Y.rows();
        if (naMethod == NAMethod::FillMean || naMethod == NAMethod::FillMedian) {
            preprocessor::parallelProcessNan(X.data(), X.rows(), X.cols(), naMethod, nthreads);
        }
    }

    auto result = py_cdarray_t(
            {resultRows, resultCols},
            {resultCols * sizeof(double), sizeof(double)});

    switch (corMethod) {
        case CorrelationMethod::Pearson:
            CorPearson::parallelCalcCor(X, Y, result.mutable_data(), nthreads);
            break;
        case CorrelationMethod::Spearman:
            CorSpearman::parallelCalcCor(X, Y, result.mutable_data(), nthreads);
            break;
        case CorrelationMethod::Kendall:
            CorKendall::parallelCalcCor(X, Y, result.mutable_data(), nthreads);
            break;
    }

    return result;
}

py_cdarray_t pvalueStudentT(const py::object &obj, double df, bool isApproximate, int nthreads) {
    auto arr = obj.cast<py_cdarray_t>();

    // Biuld pvalue array that has same shape as input
    std::vector<size_t> shape;
    for (size_t i = 0; i < arr.ndim(); ++i) {
        shape.push_back(arr.shape()[i]);
    }

    py_cdarray_t result(shape);
    auto mutableResult = result.mutable_data();

    if (isApproximate && arr.size() > (2 * PTABLE_SIZE)) {
        PTable ptable(DistributionType::T, df);
#pragma omp parallel for schedule(guided) num_threads(nthreads)
        for (int64_t i = 0; i < arr.size(); ++i) {
            mutableResult[i] = CorPearson::calcPvalue(arr.data()[i], df, ptable);
        }
    } else {
        boost::math::students_t dist(df);
#pragma omp parallel for schedule(guided) num_threads(nthreads)
        for (int64_t i = 0; i < arr.size(); ++i) {
            mutableResult[i] = CorPearson::commonCalcPvalue(arr.data()[i], df, dist);
        }
    }

    return result;
}

py_cdarray_t cortest(const py::object &xobj, const std::optional<py::object> &yobj, const std::string &method,
                     const std::string &naAction, bool isPvalueApprox, bool isMultipletest,
                     const std::string &multipletestMethod, bool isQvalueApprox, int nthreads) {
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

    size_t resultCorRows = X.rows();
    size_t resultCorCols = X.rows();
    size_t resultCorSize = resultCorRows * (resultCorRows - 1) / 2;
    // If yobj is not None, then we need to convert python object(numpy array or list) to matrix.
    // And preprocess the input matrix with nan filling.
    Matrix<double> Y;
    if (yobj.has_value()) {
        Y = convert_to_matrix(yobj.value());
        if (X.cols() != Y.cols()) {
            throw std::invalid_argument("Input X and Y must have same number of columns");
        }

        resultCorCols = Y.rows();
        resultCorSize = resultCorRows * resultCorCols;
        if (naMethod == NAMethod::FillMean || naMethod == NAMethod::FillMedian) {
            preprocessor::parallelProcessNan(X.data(), X.rows(), X.cols(), naMethod, nthreads);
        }
    }

    // If user specify isPvalueApprox and resultCorSize > (2 * PTABLE_SIZE),
    // then we need to initialize PTable used to calculate approximate p-value
    if (isPvalueApprox && resultCorSize < (2 * PTABLE_SIZE)) {
        isPvalueApprox = false;
    }
    // isQvalueApprox is the same as above
    if (isQvalueApprox && resultCorSize < (2 * PADJUSTTABLE_SIZE)) {
        isQvalueApprox = false;
    }

    //  result matrix with 4 columns: [index1, index2, cor, pvalue] or 5 columns: [index1, index2, cor, pvalue, qvalue]
    size_t resultCols = isMultipletest ? 5 : 4;
    py_cdarray_t result({resultCorSize, resultCols}, {resultCols * sizeof(double), sizeof(double)});
    auto mutableResult = result.mutable_unchecked<2>();

    PAdjustTable qtable;
    if (isMultipletest && isQvalueApprox) {
        qtable = PAdjustTable(X, Y, resultCorSize, corMethod, pAdjustMethod, nthreads);
    }

    bool isYEmpty = Y.isEmpty();
    if (corMethod == CorrelationMethod::Pearson || corMethod == CorrelationMethod::Spearman) {
        Matrix<double> resultCor(resultCorRows, resultCorCols);

        if (corMethod == CorrelationMethod::Pearson) {
            CorPearson::parallelCalcCor(X, Y, resultCor.data(), nthreads);
        } else {
            CorSpearman::parallelCalcCor(X, Y, resultCor.data(), nthreads);
        }

        // Initialize PTable for student-t distribution if isPvalueApprox is true
        PTable ptable;
        boost::math::students_t dist(1);
        if (isPvalueApprox) {
            ptable = PTable(DistributionType::T, X.cols() - 2);
        } else {
            dist = boost::math::students_t(X.cols() - 2);
        }

        double df = X.cols() - 2;
#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
        for (int64_t i = 0; i < resultCor.rows(); ++i) {
            size_t j = isYEmpty ? i + 1 : 0;
            for (; j < resultCorCols; ++j) {
                size_t index1 = isYEmpty ? util::transFullMatIndex(i, j, resultCorCols) : i * resultCorCols + j;
                mutableResult(index1, 0) = i;
                mutableResult(index1, 1) = j;
                mutableResult(index1, 2) = resultCor(i, j);
                if (isPvalueApprox) {
                    mutableResult(index1, 3) = CorPearson::calcPvalue(resultCor(i, j), df, ptable);
                } else {
                    mutableResult(index1, 3) = CorPearson::commonCalcPvalue(resultCor(i, j), df, dist);
                }
            }
        }
    } else {
        std::unique_ptr<double[]> resultCor(new double[resultCorSize]);
        std::vector<KendallStat> xStatsVec = CorKendall::parallelGetKendallStat(X, nthreads);
        std::vector<KendallStat> yStatsVec;
        if (yobj.has_value()) {
            yStatsVec = CorKendall::parallelGetKendallStat(Y, nthreads);
        }

        // Initialize PTable for student-t distribution if isPvalueApprox is true
        PTable ptable;
        boost::math::normal_distribution<> dist;
        if (isPvalueApprox) {
            ptable = PTable(DistributionType::Normal);
        } else {
            dist = boost::math::normal_distribution<>(0, 1);
        }

#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
        for (int64_t i = 0; i < resultCorRows; ++i) {
            if (isYEmpty) {
                for (size_t j = i + 1; j < resultCorCols; ++j) {
                    size_t index1 = util::transFullMatIndex(i, j, X.rows());
                    mutableResult(index1, 0) = i;
                    mutableResult(index1, 1) = j;
                    auto pair = CorKendall::calcCor(X.row(i), X.row(j), X.cols());
                    mutableResult(index1, 2) = pair.first;
                    if (isPvalueApprox) {
                        mutableResult(index1, 3) = CorKendall::calcPvalue(pair.second, xStatsVec[i],
                                                                          xStatsVec[j], ptable);
                    } else {
                        mutableResult(index1, 3) = CorKendall::commonCalcPvalue(pair.second, xStatsVec[i],
                                                                                xStatsVec[j], dist);
                    }
                }
            } else {
                for (size_t j = 0; j < resultCorCols; ++j) {
                    size_t index1 = i * resultCorCols + j;
                    mutableResult(index1, 0) = i;
                    mutableResult(index1, 1) = j;
                    auto pair = CorKendall::calcCor(X.row(i), Y.row(j), X.cols());
                    mutableResult(index1, 2) = pair.first;
                    if (isPvalueApprox) {
                        mutableResult(index1, 3) = CorKendall::calcPvalue(pair.second, xStatsVec[i],
                                                                          yStatsVec[j], ptable);
                    } else {
                        mutableResult(index1, 3) = CorKendall::commonCalcPvalue(pair.second, xStatsVec[i],
                                                                                yStatsVec[j], dist);
                    }
                }
            }
        }
    }

    // Multiple test adjustment if isMultipletest is true.
    if (isMultipletest) {
        if (isQvalueApprox) {
#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
            for (int64_t i = 0; i < resultCorSize; ++i) {
                mutableResult(i, 4) = qtable.getQvalue(mutableResult(i, 3));
            }
        } else {
            auto temp = result[py::make_tuple(py::ellipsis(), 3)].cast<py_cdarray_t>();
            auto qvalues = PAdjustTable::commonPAdjust(temp.data(), resultCorSize, pAdjustMethod);
            for (size_t i = 0; i < resultCorSize; ++i) {
                mutableResult(i, 4) = qvalues[i];
            }
        }
    }

    return result;
}

void bind_cor(py::module &m) {
    m.def("corrCoef", &corrcoef, "Calculate correlation matrix, support pearson, spearman and kendall", py::arg("x"),
          py::arg("y"), py::arg("method"), py::arg("naAction"), py::arg("nthreads"));
    m.def("pvalueStudentT", &pvalueStudentT, "Calculate p-value using student-t distribution", py::arg("x"),
          py::arg("df"), py::arg("isApproximate"), py::arg("nthreads"));
    m.def("corTest", &cortest, "Test for correlation matrix, support pearson, spearman and kendall",
          py::arg("x"), py::arg("y"), py::arg("method"), py::arg("naAction"), py::arg("isPvalueApprox"),
          py::arg("isMultipletest"), py::arg("multipletestMethod"), py::arg("isQvalueApprox"), py::arg("nthreads"));
}
