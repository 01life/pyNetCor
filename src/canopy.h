#ifndef CANOPY_H
#define CANOPY_H

#include "point.h"


template<typename T>
class Canopy {
public:
    Canopy() = default;

    ~Canopy() = default;

    Canopy(const Point<T> &centroid) : centroid_(centroid) {};

    Canopy(std::vector<Point<T> *> neighbours) : neighbours_(neighbours) {
        centroid_ = getCentroidOfPoints(neighbours_, profileMethod_);
    };

    Canopy(const Canopy<T> &other) : centroid_(other.centroid_), neighbours_(other.neighbours_),
                                     profileMethod_(other.profileMethod_) {};

    Canopy(Canopy<T> &&other) : neighbours_(other.neighbours_), profileMethod_(other.profileMethod_) {
        centroid_.reset(other.centroid_.release());
    }

    Canopy<T> &operator=(Canopy<T> other) {
        swap(*this, other);
        return *this;
    }

    inline ProfileMethodType profileMethod() const { return profileMethod_; }

    inline size_t neighbourSize() const { return neighbours_.size(); }

    inline Point<T> *centroid() { return &centroid_; }

    inline const Point<T> *centroid() const { return &centroid_; }

    inline std::vector<Point<T> *> neighbours() { return neighbours_; }

    static Point<T> getCentroidOfPoints(std::vector<Point<T> *> &points, ProfileMethodType profileMethod) {
        assert (points.size() > 0);

        auto centroid = Point<T>(points[0]->getSampleNum(), points[0]->getDistanceMethod());
        if (profileMethod == ProfileMethodType::Mean) {
            calcColumnMeans(points, centroid.data());
        } else {
            T quantileMultiplier;
            switch (profileMethod) {
                case ProfileMethodType::Median:
                    quantileMultiplier = 0.5;
                    break;
                case ProfileMethodType::Percentile_75:
                    quantileMultiplier = 0.75;
                    break;
                case ProfileMethodType::Percentile_80:
                    quantileMultiplier = 0.8;
                    break;
                case ProfileMethodType::Percentile_85:
                    quantileMultiplier = 0.85;
                    break;
                case ProfileMethodType::Percentile_90:
                    quantileMultiplier = 0.9;
                    break;
                case ProfileMethodType::Percentile_95:
                    quantileMultiplier = 0.95;
                    break;
                default:
                    quantileMultiplier = 0.5;
                    break;
            }

            calcColumnQuantiles(points, centroid.data(), quantileMultiplier);
        }

        centroid.precomputeCorrelation();
        return centroid;
    }

    friend void swap(Canopy<T> &a, Canopy<T> &b) {
        using std::swap;
        swap(a.centroid_, b.centroid_);
        swap(a.neighbours_, b.neighbours_);
        swap(a.profileMethod_, b.profileMethod_);
    }

private:
    Point<T> centroid_;
    std::vector<Point<T> *> neighbours_;

    // Profile measure method. May be optional parameter in the future.
    ProfileMethodType profileMethod_ = ProfileMethodType::Percentile_75;

    static void calcColumnQuantiles(std::vector<Point<T> *> &points, T *quantiles, T quantileMultiplier) {
        std::vector<T> colData(points.size());
        for (size_t col = 0; col < points[0]->getSampleNum(); ++col) {
            for (size_t row = 0; row < points.size(); ++row) {
                colData[row] = points[row]->data()[col];
            }

            quantiles[col] = util::nanQuantile(colData.data(), colData.size(), quantileMultiplier);
        }
    };

    static void calcColumnMeans(std::vector<Point<T> *> &points, T *means) {
        for (size_t col = 0; col < points[0]->getSampleNum(); ++col) {
            size_t count = 0;
            for (size_t row = 0; row < points.size(); row++) {
                if (!std::isnan(points[row]->data()[col])) {
                    means[col] += points[row]->data()[col];
                    ++count;
                }
            }

            means[col] /= count;
        }
    }
};

#endif //CANOPY_H
