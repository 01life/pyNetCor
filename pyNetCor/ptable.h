#ifndef PTABLE_H
#define PTABLE_H

#include <string>
#include <vector>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/distributions/normal.hpp>

#include "options.h"

const size_t PTABLE_SIZE = 10000;
const double MIN_PVALUE = 2.2e-16;

class PTable {
public:
    PTable() = default;

    PTable(const DistributionType &dist, size_t df = 0);

    ~PTable();

    double getPvalue(double q) const;

    void parallelCalcPvalue(double *q, size_t n, double *p, int nthreads) const;

private:
    DistributionType distName;
    size_t df;
    std::pair<std::vector<double>, std::vector<double>> table;
    double qMax;
    double qMin;

    void initTable();

    std::vector<double> linspace(double start, double end, size_t num);
};

#endif // PTABLE_H
