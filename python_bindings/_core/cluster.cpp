#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <cluster.h>

namespace py = pybind11;

using py_cdarray_t = py::array_t<double, py::array::c_style | py::array::forcecast>;
using py_ciarray_t = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

class CanopyCluster {
public:
    CanopyCluster(const py::object &obj, double maxCanopyDist, double maxCloseDist, double maxMergeDist,
                  double minStepDist, size_t maxCanopyWalkNum, size_t stopAfterNumSeedsProcessed,
                  const std::string &distanceMethod, std::optional<size_t> randomSeed, int nthreads)
            : maxCanopyDist_(maxCanopyDist), maxCloseDist_(maxCloseDist), maxMergeDist_(maxMergeDist),
              minStepDist_(minStepDist), maxCanopyWalkNum_(maxCanopyWalkNum),
              stopAfterNumSeedsProcessed_(stopAfterNumSeedsProcessed), nthreads_(nthreads) {
        distanceMethod_ = stringToDistanceMethodType(distanceMethod);

        auto arr = obj.cast<py_cdarray_t>();
        if (arr.ndim() != 2) {
            throw std::runtime_error("Canopy: input array must be 2D");
        }
        if (arr.shape()[1] < 2) {
            throw std::runtime_error("Canopy: input array must have length at least 2");
        }

        // initialize points
        points_.resize(arr.shape()[0]);
#pragma omp parallel for schedule(dynamic) num_threads(nthreads_)
        for (int64_t i = 0; i < arr.shape()[0]; ++i) {
            points_[i] = new Point<double>(arr.data() + i * arr.shape()[1], i, arr.shape()[1], distanceMethod_);
        }

        // Run canopy clustering
        std::mt19937 gen = randomSeed.has_value() ? std::mt19937(randomSeed.value()) : std::mt19937(
                std::random_device()());
        canopies_ = cluster::canopyClustering(points_, maxCanopyDist, maxCloseDist, maxMergeDist, minStepDist,
                                              maxCanopyWalkNum, gen, stopAfterNumSeedsProcessed, nthreads_);

        // Obtain labels, where a single point can belong to multiple canopies in the Canopy clustering
        // algorithm (a form of soft clustering)
        std::unordered_multimap<size_t, Canopy<double> *> pointToCanopyMap;
        std::unordered_map<Canopy<double> *, int64_t> canopyToLabelMap;
        for (size_t i = 0; i < canopies_.size(); ++i) {
            canopyToLabelMap.insert(std::make_pair(canopies_[i].get(), i));
            for (Point<double> *p: canopies_[i]->neighbours()) {
                pointToCanopyMap.insert(std::make_pair(p->getIndex(), canopies_[i].get()));
            }
        }

        labels_.resize(points_.size());
        // Best labels: for each point, a set of canopies that it belongs to (soft clustering), the best
        // label is the one that has the minimum distance to the centroid.
        bestLabels_ = py_ciarray_t({points_.size()}, {sizeof(int64_t)});
        auto bestLabelsData = bestLabels_.mutable_unchecked<1>();

#pragma omp parallel for schedule(dynamic) num_threads(nthreads_)
        for (int64_t i = 0; i < points_.size(); ++i) {
            auto count = pointToCanopyMap.count(i);
            if (count == 0) {
                labels_[i] = {-1};
                bestLabelsData(i) = -1;
            } else if (count == 1) {
                labels_[i] = {canopyToLabelMap.at(pointToCanopyMap.find(i)->second)};
                bestLabelsData(i) = labels_[i][0];
            } else {
                auto pos = pointToCanopyMap.equal_range(i);
                auto resultCanopy = pos.first->second;
                auto minDist = cluster::calcPointsDistance(points_[i], resultCanopy->centroid());

                for (auto it = pos.first; it != pos.second; ++it) {
                    labels_[i].push_back(canopyToLabelMap.at(it->second));

                    auto dist = cluster::calcPointsDistance(points_[i], it->second->centroid());
                    if (dist < minDist) {
                        minDist = dist;
                        resultCanopy = it->second;
                    }
                }

                bestLabelsData(i) = canopyToLabelMap.at(resultCanopy);
            }
        }

        // Obtain cluster centers
        clusterCenters_ = py_cdarray_t({canopies_.size(), canopies_[0]->centroid()->getSampleNum()});
        size_t threads = clusterCenters_.shape()[0] > 1000 ? nthreads_ : 1;

#pragma omp parallel for schedule(static) num_threads(threads)
        for (int64_t i = 0; i < canopies_.size(); ++i) {
            std::memcpy(clusterCenters_.mutable_data() + i * clusterCenters_.shape()[1],
                        canopies_[i]->centroid()->data(), clusterCenters_.shape()[1] * sizeof(double));
        }
    }

    ~CanopyCluster() {
        for (auto p: points_) {
            delete p;
        }
    }

    std::vector<std::vector<int64_t>> predict(const py::object &obj) {
        auto arr = obj.cast<py_cdarray_t>();
        if (arr.ndim() != 2) {
            throw std::runtime_error("Canopy: input array must be 2D");
        }
        if (arr.shape()[1] != clusterCenters_.shape()[1]) {
            throw std::runtime_error("Canopy: input array must have same number of features as centers");
        }

        // Calculate the distance matrix between the newly added data points and the cluster centroids.
        auto distMatrixSize = arr.shape()[0] * clusterCenters_.shape()[0];
        std::unique_ptr<double[]> distMatrix(new double[distMatrixSize]);

        cluster::calcDistMatrix(arr.data(), clusterCenters_.data(), arr.shape()[0], clusterCenters_.shape()[0],
                                arr.shape()[1], distMatrix.get(), nthreads_);

        // Obtain labels for the newly added data points based on the distance matrix
        std::vector<std::vector<int64_t>> labels(arr.shape()[0]);
        for (size_t i = 0; i < arr.shape()[0]; ++i) {
            for (size_t j = 0; j < clusterCenters_.shape()[0]; ++j) {
                if (distMatrix[i * clusterCenters_.shape()[0] + j] < maxCanopyDist_) {
                    labels[i].push_back(j);
                }
            }
            if (labels[i].empty()) {
                labels[i] = {-1};
            }
        }

        return labels;
    }

    py_ciarray_t getBestLabels() {
        return bestLabels_;
    }

    std::vector<std::vector<int64_t>> getLabels() {
        return labels_;
    }

    py_cdarray_t getClusterCenters() {
        return clusterCenters_;
    }

    py_cdarray_t getNeighbours(size_t canopyIndex) {
        py_cdarray_t result(
                {canopies_[canopyIndex]->neighbourSize(), canopies_[canopyIndex]->centroid()->getSampleNum()},
                {canopies_[canopyIndex]->centroid()->getSampleNum() * sizeof(double), sizeof(double)}
        );
        size_t threads = canopies_[canopyIndex]->neighbourSize() > 1000 ? nthreads_ : 1;

#pragma omp parallel for schedule(dynamic) num_threads(threads)
        for (int64_t i = 0; i < canopies_[canopyIndex]->neighbourSize(); ++i) {
            std::memcpy(result.mutable_data() + i * result.shape()[1],
                        canopies_[canopyIndex]->neighbours()[i]->data(),
                        result.shape()[1] * sizeof(double));
        }

        return result;
    }


private:
    double maxCanopyDist_;
    double maxCloseDist_;
    double maxMergeDist_;
    double minStepDist_;
    size_t maxCanopyWalkNum_;
    size_t stopAfterNumSeedsProcessed_;
    DistanceMethodType distanceMethod_;
    int nthreads_;

    std::vector<Point<double> *> points_;
    std::vector<std::unique_ptr<Canopy<double>>> canopies_;
    std::vector<std::vector<int64_t>> labels_;
    py_ciarray_t bestLabels_;
    py_cdarray_t clusterCenters_;
};

void bind_cluster(py::module &m) {
    py::class_<CanopyCluster>(m, "CanopyCluster")
            .def(py::init<const py::object &, double, double, double, double, size_t, size_t, const std::string &,
                 std::optional<size_t>, int>(),
                 py::arg("obj"), py::arg("maxCanopyDist"), py::arg("maxCloseDist"), py::arg("maxMergeDist"),
                 py::arg("minStepDist"), py::arg("maxCanopyWalkNum"), py::arg("stopAfterNumSeedsProcessed"),
                 py::arg("distanceMethod"), py::arg("randomSeed"), py::arg("nthreads"))
            .def("predict", &CanopyCluster::predict)
            .def("get_best_labels", &CanopyCluster::getBestLabels)
            .def("get_labels", &CanopyCluster::getLabels)
            .def("get_cluster_centers", &CanopyCluster::getClusterCenters)
            .def("get_neighbours", &CanopyCluster::getNeighbours);
}
