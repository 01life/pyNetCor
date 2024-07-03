#include "ptable.h"

#include <algorithm>
#include <cmath>
#include <vector>

PTable::PTable(const DistributionType &dist, size_t df) : distName(dist), df(df) {
    initTable();
}

PTable::~PTable() {}

double PTable::getPvalue(double q) const {
    if (std::isnan(q)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    q = std::abs(q);
    if (q >= qMax){
        return MIN_PVALUE;
    } else if (q <= qMin) {
        return 1;
    } else {
        auto it = std::lower_bound(table.first.begin(), table.first.end(), q);
        size_t index = it - table.first.begin();
        double p = table.second[index - 1] +
                   (table.second[index] - table.second[index - 1]) /
                   (table.first[index] - table.first[index - 1]) *
                   (q - table.first[index - 1]);
        return p * 2;
    }
}

void PTable::parallelCalcPvalue(double *q, size_t n, double *p, int nthreads) const {
#pragma omp parallel for schedule(guided) num_threads(nthreads)
    for (int64_t i = 0; i < n; ++i) {
        p[i] = getPvalue(q[i]);
    }
}

void PTable::initTable() {
    std::vector<double> p0 = linspace(std::pow(0.5, 1.0 / 4), std::pow(1.1e-16, 1.0 / 4), PTABLE_SIZE);
    std::for_each(p0.begin(), p0.end(), [](double &x) { x = std::pow(x, 4); });
    std::vector<double> q0(p0.size());

    switch (distName) {
        case DistributionType::Normal: {
            boost::math::normal_distribution<> norm(0, 1);
            std::transform(p0.begin(), p0.end(), q0.begin(), [&norm](double p) {
                return std::abs(boost::math::quantile(norm, p));
            });
            break;
        }
        case DistributionType::T: {
            boost::math::students_t_distribution<> t(df);
            std::transform(p0.begin(), p0.end(), q0.begin(), [&t](double p) {
                return std::abs(boost::math::quantile(t, p));
            });
            break;
        }
        default:
            break;
    }

    table = std::make_pair(q0, p0);
    qMax = q0.back();
    qMin = q0.front();
}


std::vector<double> PTable::linspace(double start, double end, size_t num) {
    std::vector<double> v(num);
    double step = (end - start) / (num - 1);
    for (size_t i = 0; i < num; ++i) {
        v[i] = start + i * step;
    }
    return v;
}