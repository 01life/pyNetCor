#include "cluster.h"


namespace cluster {
    void calcDistMatrix(const double *X, const double *Y, size_t m, size_t n, size_t k, double *distMatrix, int nthreads) {
        std::fill_n(distMatrix, m * n, static_cast<double>(1.0));

        openblas_set_num_threads(nthreads);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, -1.0, X, k, Y, k, 1.0, distMatrix, n);
    }

    double calcPointsDistance(const Point<double> *p1, const Point<double> *p2) {
        return 1 - cblas_ddot(p1->getSampleNum(), p1->precomputedData(), 1, p2->precomputedData(), 1);
    };

    std::unique_ptr<Canopy<double>> createCanopy(const Point<double> *origin, const std::vector<Point<double> *> &points,
                                                 std::vector<Point<double> *> &closePoints, double maxCanopyDist,
                                                 double maxCloseDist, bool setClosePoints) {
        std::vector<Point<double> *> neighbours;

        if (setClosePoints) {
            // Iterate over all points, updating the collection of close points to include those that fall within the
            // defined “close” distance.
            closePoints.clear();

//            std::unique_ptr<double[]> pointsPrecomputedMatrix = std::make_unique<double[]>(
//                    points.size() * points[0]->getSampleNum());
//            for (size_t i = 0; i < points.size(); ++i) {
//                std::memcpy(pointsPrecomputedMatrix.get() + i * points[0]->getSampleNum(), points[i]->precomputedData(),
//                            points[0]->getSampleNum() * sizeof(double));
//            }
//
//            auto distMatrix = std::make_unique<double[]>(points.size());
//            calcDistMatrix(origin->precomputedData(), pointsPrecomputedMatrix.get(), 1, points.size(),
//                           origin->getSampleNum(), distMatrix.get(), 1);

            for (size_t i = 0; i < points.size(); ++i) {
//                auto distance = distMatrix[i];
                auto distance = calcPointsDistance(origin, points[i]);
                if (distance < maxCloseDist) {
                    closePoints.push_back(points[i]);
                    if (distance < maxCanopyDist) {
                        neighbours.push_back(points[i]);
                    }
                }
            }
        } else if (!closePoints.empty()) {
//            std::unique_ptr<double[]> closePointsPrecomputedMatrix = std::make_unique<double[]>(
//                    closePoints.size() * closePoints[0]->getSampleNum());
//            for (size_t i = 0; i < closePoints.size(); ++i) {
//                std::memcpy(closePointsPrecomputedMatrix.get() + i * closePoints[0]->getSampleNum(),
//                            closePoints[i]->precomputedData(),
//                            closePoints[0]->getSampleNum() * sizeof(double));
//            }
//
//            auto distMatrix = std::make_unique<double[]>(closePoints.size());
//            calcDistMatrix(origin->precomputedData(), closePointsPrecomputedMatrix.get(), 1, closePoints.size(),
//                           origin->getSampleNum(), distMatrix.get(), 1);

            for (size_t i = 0; i < closePoints.size(); ++i) {
//                auto distance = distMatrix[i];
                auto distance = calcPointsDistance(origin, closePoints[i]);
                if (distance < maxCanopyDist) {
                    neighbours.push_back(closePoints[i]);
                }
            }
        }

        if (!neighbours.empty()) {
            return std::make_unique<Canopy<double>>(neighbours);
        } else {
            return std::make_unique<Canopy<double>>(*origin);
        }
    };

    std::unique_ptr<Canopy<double>> canopyWalk(const Point<double> *origin, const std::vector<Point<double> *> &points,
                                               std::vector<Point<double> *> &closePoints, double maxCanopyDist,
                                               double maxCloseDist, double minStepDist, size_t maxCanopyWalkNum) {
        auto c1 = createCanopy(origin, points, closePoints, maxCanopyDist, maxCloseDist, true);

        // If there is a special case where no walking is required, return the canopy immediately.
        if (maxCanopyWalkNum == 0) {
            return c1;
        }

        auto c2 = createCanopy(c1->centroid(), points, closePoints, maxCanopyDist, maxCloseDist, false);

        auto distance = calcPointsDistance(c1->centroid(), c2->centroid());

        size_t localCanopyWalkNum = 0;
        while ((distance > minStepDist) && (localCanopyWalkNum <= maxCanopyWalkNum)) {
            c1.reset(c2.release());

            ++localCanopyWalkNum;
            c2 = createCanopy(c1->centroid(), points, closePoints, maxCanopyDist, maxCloseDist, false);
            distance = calcPointsDistance(c1->centroid(), c2->centroid());
        }

        // Now that cluster c1 and c2 are sufficiently similar, we should select the one with more neighboring data
        // points for further analysis.
        if (c1->neighbourSize() > c2->neighbourSize()) {
            return c1;
        } else {
            return c2;
        }
    };

    std::vector<std::unique_ptr<Canopy<double>>> canopyClustering(std::vector<Point<double> *> points,
                                                                  double maxCanopyDist, double maxCloseDist,
                                                                  double maxMergeDist, double minStepDist,
                                                                  size_t maxCanopyWalkNum, std::mt19937 &gen,
                                                                  size_t stopAfterNumSeedsProcessed, int nthreads) {
        // shuffle the points in the list before starting
        std::shuffle(points.begin(), points.end(), gen);

        std::unordered_set<Point<double> *> markedPoints; // Points excluded from consideration as origins.
        std::vector<std::unique_ptr<Canopy<double>>> canopies;

        std::vector<Point<double> *> closePoints;
        closePoints.reserve(points.size());

        //
        // Create canopies
        //
        size_t seedsProcessedNum = 0;

#pragma omp parallel for shared(points, markedPoints, canopies, seedsProcessedNum) firstprivate(closePoints, maxCanopyDist, maxCloseDist, maxMergeDist, minStepDist) schedule(dynamic, 100) num_threads(nthreads)
        for (int64_t originIndex = 0; originIndex < points.size(); ++originIndex) {
            // Early stopping after reaching a certain number of seeds
            if (seedsProcessedNum >= stopAfterNumSeedsProcessed) {
                continue;
            }

            Point<double> *origin = points[originIndex];
            if (markedPoints.find(origin) != markedPoints.end()) {
                continue;
            }

            auto finalCanopy = canopyWalk(origin, points, closePoints, maxCanopyDist, maxCloseDist, minStepDist,
                                          maxCanopyWalkNum);

#pragma omp critical
            {
                // The current origin should not be considered if it has been marked by another thread.
                if (markedPoints.find(origin) == markedPoints.end()) {
                    markedPoints.insert(origin);
                    for (Point<double> *p: finalCanopy->neighbours()) {
                        markedPoints.insert(p);
                    }

                    canopies.push_back(std::move(finalCanopy));
                }

                ++seedsProcessedNum;
            }
        }

        //
        // Eliminating canopies of size 1 to optimize merging efficiency
        //
        for (size_t i = 0; i < canopies.size(); ++i) {
            if (canopies[i]->neighbourSize() == 1) {
                canopies.erase(canopies.begin() + i);
                --i;
            }
        }

        //
        // Merge canopies
        //
        std::vector<std::unique_ptr<Canopy<double>>> mergedCanopies;
        std::vector<Canopy<double> *> canopiesToMerge;

        while (!canopies.empty()) {
            // The final canopies should be popped and identify all the canopies that are mergeable with it.
            std::unique_ptr<Canopy<double>> c = std::move(canopies.back());
            canopies.pop_back();

            canopiesToMerge.clear();
            canopiesToMerge.push_back(c.get());

            // TODO: Parallel
            for (size_t i = 0; i < canopies.size(); ++i) {
                double distance = calcPointsDistance(c->centroid(), canopies[i]->centroid());
                if (distance < maxMergeDist) {
                    canopiesToMerge.push_back(canopies[i].get());
                }
            }

            // The canopies need to be merged, please do it.
            if (canopiesToMerge.size() > 1) {
                std::vector<Point<double> *> allPointsFromMergedCanopies;
                for (auto &canopy: canopiesToMerge) {
                    for (auto &p: canopy->neighbours()) {
                        if (std::find(allPointsFromMergedCanopies.begin(), allPointsFromMergedCanopies.end(), p) ==
                            allPointsFromMergedCanopies.end()) {
                            allPointsFromMergedCanopies.push_back(p);
                        }
                    }
                }

                auto tempCentroid = Canopy<double>::getCentroidOfPoints(allPointsFromMergedCanopies,
                                                                        c->profileMethod());
                std::unique_ptr<Canopy<double>> mergedCanopy = canopyWalk(&tempCentroid, allPointsFromMergedCanopies,
                                                                          closePoints, maxCanopyDist, maxCloseDist,
                                                                          minStepDist,
                                                                          maxCanopyWalkNum);

                canopies.push_back(std::move(mergedCanopy));

                // Remove the merged canopies.
                for (auto iter = canopies.begin(); iter != canopies.end();) {
                    if (iter->get() == canopiesToMerge[1]) {
                        iter = canopies.erase(iter);
                        canopiesToMerge.erase(canopiesToMerge.begin() + 1);
                    } else {
                        ++iter;
                    }
                }
            } else {
                mergedCanopies.push_back(std::move(c));
            }
        }

        return mergedCanopies;
    };
}