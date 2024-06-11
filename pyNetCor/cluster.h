#ifndef CLUSTER_H
#define CLUSTER_H

#include <unordered_set>

#include "canopy.h"


namespace cluster {
    /**
     * @brief The distance matrix between two arrays of points is computed for each row.
     *
     * @param X The first dynamic array.
     * @param Y The second dynamic array.
     * @param m The number of rows in X.
     * @param n The number of rows in Y.
     * @param k The number of columns in X and Y.
     * @param distMatrix The distance matrix.
     * @param nthreads The number of threads to use for parallel computation.
     */
    void calcDistMatrix(const double *X, const double *Y, size_t m, size_t n, size_t k, double *distMatrix, int nthreads);
    
    double calcPointsDistance(const Point<double> *p1, const Point<double> *p2);

    std::unique_ptr<Canopy<double>> createCanopy(const Point<double> *origin, const std::vector<Point<double> *> &points,
                                            std::vector<Point<double> *> &closePoints, double maxCanopyDist, double maxCloseDist,
                                            bool setClosePoints);

    std::unique_ptr<Canopy<double>> canopyWalk(const Point<double> *origin, const std::vector<Point<double> *> &points,
                                          std::vector<Point<double> *> &closePoints, double maxCanopyDist, double maxCloseDist,
                                          double minStepDist, size_t maxCanopyWalkNum);

    std::vector<std::unique_ptr<Canopy<double>>> canopyClustering(std::vector<Point<double> *> points, double maxCanopyDist,
                                                             double maxCloseDist, double maxMergeDist, double minStepDist,
                                                             size_t maxCanopyWalkNum, std::mt19937 &gen,
                                                             size_t stopAfterNumSeedsProcessed, int nthreads);
} // namespace cluster


#endif //CLUSTER_H
