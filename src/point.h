#ifndef POINT_H
#define POINT_H

#include <memory>

#include "cor.h"


template<typename T>
class Point {
public:
    Point() = default;

    ~Point() = default;

    Point(size_t n, DistanceMethodType distanceMethod) : nsamples_(n), distanceMethod_(distanceMethod) {
        data_ = std::make_unique<T[]>(n);
        precomputedData_ = std::make_unique<T[]>(n);
    };

    Point(const T *data, size_t index, size_t n, DistanceMethodType distanceMethod) :
            index_(index), nsamples_(n), distanceMethod_(distanceMethod) {
        data_ = std::make_unique<T[]>(n);
        std::memcpy(data_.get(), data, n * sizeof(T));
        precomputeCorrelation();
    };

    Point(const Point<T> &p) : index_(p.index_), nsamples_(p.nsamples_), distanceMethod_(p.distanceMethod_) {
        data_ = std::make_unique<T[]>(nsamples_);
        std::memcpy(data_.get(), p.data_.get(), nsamples_ * sizeof(T));

        precomputedData_ = std::make_unique<T[]>(nsamples_);
        std::memcpy(precomputedData_.get(), p.precomputedData_.get(), nsamples_ * sizeof(T));
    };

    Point(Point<T> &&p) : index_(p.index_), nsamples_(p.nsamples_), distanceMethod_(p.distanceMethod_) {
        data_.reset(p.data_.release());
        precomputedData_.reset(p.precomputedData_.release());
    };

    Point &operator=(Point<T> other) {
        swap(*this, other);
        return *this;
    };

    inline size_t getIndex() const { return index_; }

    inline size_t getSampleNum() const { return nsamples_; }

    inline DistanceMethodType getDistanceMethod() const { return distanceMethod_; }

    inline T *data() { return data_.get(); }

    inline const T *data() const { return data_.get(); }

    inline T *precomputedData() { return precomputedData_.get(); }

    inline const T *precomputedData() const { return precomputedData_.get(); }

    void precomputeCorrelation() {
        if (data_ == nullptr) {
            return;
        }

        switch (distanceMethod_) {
            case DistanceMethodType::Pearson:
                precomputedData_ = std::make_unique<T[]>(nsamples_);
                std::memcpy(precomputedData_.get(), data_.get(), nsamples_ * sizeof(T));
                CorPearson::preprocessNormalize(precomputedData_.get(), nsamples_);
                break;
            case DistanceMethodType::Spearman:
                util::nanRank(data_.get(), nsamples_, precomputedData_.get());
                CorPearson::preprocessNormalize(precomputedData_.get(), nsamples_);
                break;
            default:
                break;
        }
    };

    friend void swap(Point<T> &p1, Point<T> &p2) {
        using std::swap;
        swap(p1.index_, p2.index_);
        swap(p1.nsamples_, p2.nsamples_);
        swap(p1.distanceMethod_, p2.distanceMethod_);
        swap(p1.data_, p2.data_);
        swap(p1.precomputedData_, p2.precomputedData_);
    };

private:
    size_t index_;
    size_t nsamples_;
    DistanceMethodType distanceMethod_;
    std::unique_ptr<T[]> data_ = nullptr;
    std::unique_ptr<T[]> precomputedData_ = nullptr;
};


#endif //POINT_H
