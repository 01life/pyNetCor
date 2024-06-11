#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cblas.h>
#include <memory>
#include <cstring>


template<typename T>
class Matrix {
public:
    Matrix() = default;
    ~Matrix() {
        delete[] data_;
    };

    Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols) {
        data_ = new T[rows * cols];
    };

    /**
     * @brief Array initialization of the matrix.
     *
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @param array Dynamic array of size rows * cols.
     *
     * @note This constructor preforms a shallow copy. This means that data_ and array
     * point to the same area of memory. If the array is modified elsewhere, the changes
     * will be reflected in data_.
     */
    Matrix(size_t rows, size_t cols, T* array) : rows_(rows), cols_(cols), data_(array) {};

    /**
     * @brief Array initialization of the matrix.
     *
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @param array The array used to initialize the matrix.
     *
     * @note This constructor performs a deep copy. This means that data_ points to an
     * independent copy of the array. If the array is modified elsewhere, the changes
     * will not be reflected in data_.
     */
    Matrix(size_t rows, size_t cols, const T* array) : rows_(rows), cols_(cols) {
        data_ = new T[rows * cols];
        std::memcpy(data_, array, rows * cols * sizeof(T));
    };

    Matrix(size_t rows, size_t cols, const std::vector<T> &vec) : rows_(rows), cols_(cols) {
        if (vec.size() != rows * cols) {
            throw std::invalid_argument("Data size does not match matrix size.");
        }

        data_ = new T[rows * cols];
        std::memcpy(data_, vec.data(), rows * cols * sizeof(T));
    };

    Matrix(const Matrix<T> &other) : rows_(other.rows_), cols_(other.cols_) {
        data_ = new T[rows_ * cols_];
        std::memcpy(data_, other.data(), rows_ * cols_ * sizeof(T));
    };

    Matrix(Matrix<T> &&other)  noexcept : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {
        other.data_ = nullptr;
    };

    Matrix<T> &operator=(Matrix<T> other) {
        swap(*this, other);
        return *this;
    };

    inline T &operator()(size_t row, size_t col) {
        return data_[row * cols_ + col];
    };

    inline const T &operator()(size_t row, size_t col) const {
        return data_[row * cols_ + col];
    };

    inline size_t rows() const {
        return rows_;
    };

    inline size_t cols() const {
        return cols_;
    };

    inline size_t size() const {
        return rows_ * cols_;
    }

    inline bool isEmpty() const {
        return rows_ == 0 || cols_ == 0;
    }

    inline T* row(size_t index) {
        if (index >= rows_) {
            throw std::invalid_argument("Index out of bounds.");
        }
        return data_ + index * cols_;
    }

    inline const T* row(size_t index) const {
        if (index >= rows_) {
            throw std::invalid_argument("Index out of bounds.");
        }
        return data_ + index * cols_;
    }

    inline T* data() {
        return data_;
    }

    inline const T* data() const {
        return data_;
    }

    void resize(size_t rows, size_t cols) {
        rows_ = rows;
        cols_ = cols;
        delete [] data_;
        data_ = new T[rows_ * cols_];
    }

    void print(std::ostream &os) const {
        for (size_t i = 0; i < rows_; i++) {
            for (size_t j = 0; j < cols_; j++) {
                os << data_[i * cols_ + j] << " ";
            }
            os << std::endl;
        }
    }

    friend void swap(Matrix<T> &a, Matrix<T> &b) {
        using std::swap;
        swap(a.rows_, b.rows_);
        swap(a.cols_, b.cols_);
        swap(a.data_, b.data_);
    }

private:
    size_t rows_ = 0;
    size_t cols_ = 0;
    T* data_ = nullptr;
};


#endif //MATRIX_H
