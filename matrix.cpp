//
// Created by xixuan on 10/11/16.
//

#include "matrix.h"

#include <iomanip>
#include <limits>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <cassert>
#include <tgmath.h>

// explicit instantiation of templated classes
namespace dbm {

    template
    class Matrix<double>;

    template
    class Matrix<float>;

}

// constructors, destructor and IO tools
namespace dbm {

    template<typename T>
    Matrix<T>::Matrix(int height, int width) : height(height), width(width) {

        std::srand((unsigned int)(std::time(NULL)));
        data = new T *[height];
        for (int i = 0; i < height; ++i) {
            data[i] = new T[width];
            for (int j = 0; j < width; ++j) {
//                data[i][j] = T(std::floor(std::rand() % 10000 / 100.0)) / 100.0;
                data[i][j] = T(std::rand()) / RAND_MAX * 2 - 1;
            }
        }

#if _DEBUG_MATRIX
        col_labels = new int[width];
        for (int i = 0; i < width; ++i) {
            col_labels[i] = i;
        }


        row_labels = new int[height];
        for (int i = 0; i < height; ++i)
            row_labels[i] = i;
#endif

#if _CD_INDICATOR
        std::cout << "Matrix at " << this << " is instantiated." << std::endl;
#endif
    }

    template<typename T>
    Matrix<T>::Matrix(int height, int width, const T &value) : height(height), width(width) {

        data = new T *[height];
        for (int i = 0; i < height; ++i) {
            data[i] = new T[width];
            for (int j = 0; j < width; ++j) {
                data[i][j] = value;
            }
        }

#if _DEBUG_MATRIX
        col_labels = new int[width];
        for (int i = 0; i < width; ++i) {
            col_labels[i] = i;
        }


        row_labels = new int[height];
        for (int i = 0; i < height; ++i)
            row_labels[i] = i;
#endif

#if _CD_INDICATOR
        std::cout << "Matrix at " << this << " is instantiated." << std::endl;
#endif
    }

    /*  the data must have the form as below
     * s_f  f_0 f_1 f_2 ....
     * s_0  xx  xx  xx  ....
     * s_1  xx  xx  xx  ....
     * s_2  xx  xx  xx  ....
     */
    template<typename T>
    Matrix<T>::Matrix(int height, int width, const std::string file_name, const char &delimiter) : height(height),
                                                                                                   width(width) {

        data = new T *[height];
        for (int i = 0; i < height; ++i) {
            data[i] = new T[width];
        }
#if _DEBUG_MATRIX
        col_labels = new int[width];
        row_labels = new int[height];
#endif

        std::ifstream file(file_name.c_str());
        std::string line;
        int line_number = 0, col_number = 0;

        // read feature labels
        unsigned long prev = 0, next = 0;
        std::string temp_storage;
#if _DEBUG_MATRIX
        std::getline(file, line);
        next = line.find_first_of(delimiter, prev);
        prev = next + 1;
        while ((next = line.find_first_of(delimiter, prev)) != std::string::npos) {
            temp_storage = line.substr(prev, next - prev);
            col_labels[col_number] = std::atoi(temp_storage.c_str());
            col_number += 1;
            prev = next + 1;
        }
#if _DEBUG_MATRIX
        assert(col_number == width);
#endif
        line_number += 1;
#endif

        // read row labels and samples
        while (std::getline(file, line)) {
            col_number = 0;
            prev = 0, next = 0;
#if _DEBUG_MATRIX
            next = line.find_first_of(delimiter, prev);
            temp_storage = line.substr(prev, next - prev);
            row_labels[line_number - 1] = std::atoi(temp_storage.c_str());
            col_number += 1;
            prev = next + 1;
#endif
            while ((next = line.find_first_of(delimiter, prev)) != std::string::npos) {
                temp_storage = line.substr(prev, next - prev);
                data[line_number - 1][col_number - 1] = std::atof(temp_storage.c_str());
                col_number += 1;
                prev = next + 1;
            }
            line_number += 1;
        }
#if _DEBUG_MATRIX
        assert(line_number - 1 == height);
#endif

#if _CD_INDICATOR
        std::cout << "Matrix at " << this << " is instantiated." << std::endl;
#endif
    }

    template<typename T>
    Matrix<T>::~Matrix() {

        for (int i = 0; i < height; ++i) {
            delete[] data[i];
        }

        delete[] data;

#if _DEBUG_MATRIX
        delete[] col_labels;
        delete[] row_labels;
#endif

#if _CD_INDICATOR
        std::cout << "Matrix at " << this << " is deleted." << std::endl;
#endif

    };

    template<typename T>
    void Matrix<T>::print() const {

        #if _DEBUG_MATRIX
            std::cout << "s_f\t";

            for (int i = 0; i < width; ++i)
                std::cout << col_labels[i] << "\t";
            std::cout << std::endl;
        #endif

        for (int i = 0; i < height; ++i) {
#if _DEBUG_MATRIX
            std::cout << row_labels[i] << "\t";
#endif
            for (int j = 0; j < width; ++j)
                std::cout << std::fixed << std::setprecision(2) << data[i][j] << "\t";
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    template<typename T>
    void Matrix<T>::print_to_file(const std::string &file_name) const {
        std::ofstream file(file_name.c_str());

#if _DEBUG_MATRIX
        file << "s_f\t";

        for (int i = 0; i < width; ++i)
            file << col_labels[i] << "\t";
        file << std::endl;
#endif

        for (int i = 0; i < height; ++i) {
#if _DEBUG_MATRIX
            file << row_labels[i] << "\t";
#endif

            for (int j = 0; j < width; ++j)
                file << std::fixed << std::setprecision(2) << data[i][j] << "\t";
            file << std::endl;
        }

        file.close();
    }

}

// dimensions and ranges
namespace dbm {

    template<typename T>
    T Matrix<T>::get_col_max(int col_index, const int *row_inds, int n_rows) const {
        if (row_inds == NULL) {
            T result = std::numeric_limits<T>::min();
            for (int i = 0; i < height; ++i) {
                if (data[i][col_index] >= result)
                    result = data[i][col_index];
            }
            return result;
        } else {
#if _DEBUG_MATRIX
            assert(n_rows > 0);
#endif
            T result = std::numeric_limits<T>::min();
            for (int i = 0; i < n_rows; ++i) {
                if (data[row_inds[i]][col_index] >= result)
                    result = data[row_inds[i]][col_index];
            }
            return result;
        }
    }

    template<typename T>
    T Matrix<T>::get_col_min(int col_index, const int *row_inds, int n_rows) const {
        if (row_inds == NULL) {
            T result = std::numeric_limits<T>::max();
            for (int i = 0; i < height; ++i) {
                if (data[i][col_index] <= result)
                    result = data[i][col_index];
            }
            return result;
        } else {
#if _DEBUG_MATRIX
            assert(n_rows > 0);
#endif
            T result = std::numeric_limits<T>::max();
            for (int i = 0; i < n_rows; ++i) {
                if (data[row_inds[i]][col_index] <= result)
                    result = data[row_inds[i]][col_index];
            }
            return result;
        }
    }

}

// unique values
namespace dbm {

    // returns the number of unique values
    // sort and put unique values in the beginning of values
    template<typename T>
    inline int Matrix<T>::unique_vals_col(int col_index, T *values, const int *row_inds, int n_rows) const {
        // usage:
        //    double uniques[b.get_height()];
        //    int end = b.unique_vals_col(1, uniques);
        //    cout << b.get_height() << ' ' << end << endl;
        //    for(int i = 0; i < end; ++i) cout << uniques[i] << ' ';
        //    cout << endl;
#if _DEBUG_MATRIX
        assert(col_index < width);
#endif
        if (row_inds == NULL) {
            for (int i = 0; i < height; ++i)
                values[i] = data[i][col_index];
            std::sort(values, values + height);
            T *end = std::unique(values, values + height);
            return (int) std::distance(values, end);
        } else {
#if _DEBUG_MATRIX
            assert(n_rows != 0);
#endif
            for (int i = 0; i < n_rows; ++i)
                values[i] = data[row_inds[i]][col_index];
            std::sort(values, values + n_rows);
            T *end = std::unique(values, values + n_rows);
            return (int) std::distance(values, end);
        }
    }

}

// shuffle rows
namespace dbm {

    // cannot shuffle row labels
    template<typename T>
    void Matrix<T>::row_shuffle() {
        std::random_shuffle(data, data + height);
    };

    // both rows and row labels are shuffled, and return a new Matrix<T>
    template<typename T>
    Matrix<T> Matrix<T>::row_shuffled_to() const {
        int r_inds[height];
        for (int i = 0; i < height; ++i) r_inds[i] = i;
        std::random_shuffle(r_inds, r_inds + height);
        return rows(r_inds, height);
    };

}

// assignment
namespace dbm {

    template<typename T>
    void Matrix<T>::assign(int i, int j, const T &value) {
#if _DEBUG_MATRIX
        assert(i < height && j < width);
#endif
        data[i][j] = value;
    }

    // carefully check if the length of column is equal to height
    template<typename T>
    void Matrix<T>::assign_col(int j, T *column) {
#if _DEBUG_MATRIX
        assert(j < width);
#endif
        for (int i = 0; i < height; ++i) {
            data[i][j] = column[i];
        }
    }

    // carefully check if the length of row is equal to width
    template<typename T>
    void Matrix<T>::assign_row(int i, T *row) {
#if _DEBUG_MATRIX
        assert(i < height);
#endif
        std::copy(row, row + width, data[i]);
    }

#if _DEBUG_MATRIX

    template<typename T>
    void Matrix<T>::assign_row_label(int i, const int &label) {
#if _DEBUG_MATRIX
        assert(i < height);
#endif
        row_labels[i] = label;
    }

    template<typename T>
    void Matrix<T>::assign_col_label(int j, const int &label) {
#if _DEBUG_MATRIX
        assert(j < width);
#endif
        col_labels[j] = label;
    }

#endif
}

// []
namespace dbm {

    // matrix[i][j] returns a reference to [(i+1), (j+1)]'th element
    // matrix[i] returns a pointer to (i + 1)'th row
    template<typename T>
    T *Matrix<T>::operator[](int k) {
#if _DEBUG_MATRIX
        assert(k < height);
#endif
        return data[k];
    }

}

// get element, rows, columns, submatrices
namespace dbm {

    template<typename T>
    T Matrix<T>::get(int i, int j) const {
#if _DEBUG_MATRIX
        assert(i < height && j < width);
#endif
        return data[i][j];
    }

    template<typename T>
    Matrix<T> Matrix<T>::col(int col_index) const {
#if _DEBUG_MATRIX
        assert(col_index < height);
#endif
        Matrix<T> result(height, 1, 0);
        for (int i = 0; i < height; ++i) {
            result.data[i][0] = data[i][col_index];
#if _DEBUG_MATRIX
            result.row_labels[i] = row_labels[i];
#endif
        }
#if _DEBUG_MATRIX
        result.col_labels[0] = col_labels[col_index];
#endif
        return result;
    }

    template<typename T>
    Matrix<T> Matrix<T>::row(int row_index) const {
#if _DEBUG_MATRIX
        assert(row_index < height);
#endif
        Matrix<T> result(1, width, 0);
        std::copy(data[row_index], data[row_index] + width, result.data[0]);
#if _DEBUG_MATRIX
        std::copy(col_labels, col_labels + width, result.col_labels);
        result.row_labels[0] = row_labels[row_index];
#endif

        return result;
    }

    template<typename T>
    Matrix<T> Matrix<T>::cols(const int *col_indices, int n_cols) const {
        Matrix<T> result(height, n_cols, 0);
        for (int j = 0; j < n_cols; ++j) {
#if _DEBUG_MATRIX
            assert(col_indices[j] < width);
#endif
            for (int i = 0; i < height; ++i)
                result.data[i][j] = data[i][col_indices[j]];
#if _DEBUG_MATRIX
            result.col_labels[j] = col_labels[col_indices[j]];
#endif
        }
#if _DEBUG_MATRIX
        for (int i = 0; i < height; ++i)
            result.row_labels[i] = row_labels[i];
#endif

        return result;
    }

    template<typename T>
    Matrix<T> Matrix<T>::rows(const int *row_indices, int n_rows) const {
        Matrix<T> result(n_rows, width, 0);
        for (int i = 0; i < n_rows; ++i) {
#if _DEBUG_MATRIX
            assert(row_indices[i] < height);
#endif
            std::copy(data[row_indices[i]], data[row_indices[i]] + width, result.data[i]);
#if _DEBUG_MATRIX
            result.row_labels[i] = row_labels[row_indices[i]];
#endif

        }
#if _DEBUG_MATRIX
        std::copy(col_labels, col_labels + width, result.col_labels);
#endif

        return result;
    }

    template<typename T>
    Matrix<T> Matrix<T>::submatrix(const int *row_indices, int n_rows,
                                   const int *col_indices, int n_cols) const {
        return rows(row_indices, n_rows).cols(col_indices, n_cols);
    }

}

// split into two Matrix<T> according to a col and a threshold
namespace dbm {

    template<typename T>
    int Matrix<T>::n_larger_in_col(int col_index, const T &threshold, const int *row_inds, int n_rows) const {
#if _DEBUG_MATRIX
        assert(col_index < width);
#endif
        if (row_inds == NULL) {
            int result = 0;
            for (int i = 0; i < height; ++i)
                result += data[i][col_index] > threshold;
            return result;
        } else {
#if _DEBUG_MATRIX
            assert(n_rows > 0);
#endif
            int result = 0;
            for (int i = 0; i < n_rows; ++i)
                result += data[row_inds[i]][col_index] > threshold;
            return result;
        }
    }

    template<typename T>
    int Matrix<T>::n_smaller_or_eq_in_col(int col_index, const T &threshold, const int *row_inds, int n_rows) const {
        if (row_inds == NULL) {
            return height - n_larger_in_col(col_index, threshold);
        } else {
#if _DEBUG_MATRIX
            assert(n_rows > 0);
#endif
            return n_rows - n_larger_in_col(col_index, threshold, row_inds, n_rows);
        }
    }

    template<typename T>
    int Matrix<T>::inds_larger_in_col(int col_index, const T &threshold, int *indices, const int *row_inds,
                                      int n_rows) const {
        if (row_inds == NULL) {
            int k = 0;
            for (int i = 0; i < height; ++i) {
                if (data[i][col_index] > threshold) {
                    indices[k] = i;
                    k += 1;
                }
            }
#if _DEBUG_MATRIX
            assert(k == n_larger_in_col(col_index, threshold));
#endif
            return k;
        } else {
#if _DEBUG_MATRIX
            assert(n_rows > 0);
#endif
            int k = 0;
            for (int i = 0; i < n_rows; ++i) {
                if (data[row_inds[i]][col_index] > threshold) {
                    indices[k] = row_inds[i];
                    k += 1;
                }
            }
#if _DEBUG_MATRIX
            assert(k == n_larger_in_col(col_index, threshold, row_inds, n_rows));
#endif
            return k;
        }
    }

    template<typename T>
    int Matrix<T>::inds_smaller_or_eq_in_col(int col_index, const T &threshold, int *indices, const int *row_inds,
                                             int n_rows) const {
        if (row_inds == NULL) {
            int k = 0;
            for (int i = 0; i < height; ++i) {
                if (data[i][col_index] <= threshold) {
                    indices[k] = i;
                    k += 1;
                }
            }
#if _DEBUG_MATRIX
            assert(k == n_larger_in_col(col_index, threshold));
#endif
            return k;
        } else {
#if _DEBUG_MATRIX
            assert(n_rows > 0);
#endif
            int k = 0;
            for (int i = 0; i < n_rows; ++i) {
                if (data[row_inds[i]][col_index] <= threshold) {
                    indices[k] = row_inds[i];
                    k += 1;
                }
            }
#if _DEBUG_MATRIX
            assert(k == n_smaller_or_eq_in_col(col_index, threshold, row_inds, n_rows));
#endif
            return k;
        }
    }

    template<typename T>
    Matrix<T> Matrix<T>::vert_split_l(int col_index, const T &threshold) const {
        int n_larger = n_larger_in_col(col_index, threshold);
#if _DEBUG_MATRIX
        assert(n_larger > 0);
#endif
        int larger_indices[n_larger];
        inds_larger_in_col(col_index, threshold, larger_indices);
        return rows(larger_indices, n_larger);
    }

    template<typename T>
    Matrix<T> Matrix<T>::vert_split_s(int col_index, const T &threshold) const {
        int n_smaller = n_smaller_or_eq_in_col(col_index, threshold);
#if _DEBUG_MATRIX
        assert(n_smaller > 0);
#endif
        int smaller_indices[n_smaller];
        inds_smaller_or_eq_in_col(col_index, threshold, smaller_indices);
        return rows(smaller_indices, n_smaller);
    }

    template<typename T>
    inline void Matrix<T>::inds_split(int col_index, const T &threshold, int *larger_inds,
                                      int *smaller_inds, int *n_two_inds,
                                      const int *row_inds, int n_rows) const {

        if (row_inds == NULL) {
            int k = 0, j = 0;
            for (int i = 0; i < height; ++i) {
                if (data[i][col_index] > threshold) {
                    larger_inds[k] = i;
                    k += 1;
                } else {
                    smaller_inds[j] = i;
                    j += 1;
                }
            }
#if _DEBUG_MATRIX
            assert(k == n_larger_in_col(col_index, threshold));
#endif
            n_two_inds[0] = k;
            n_two_inds[1] = j;

        } else {
#if _DEBUG_MATRIX
            assert(n_rows > 0);
#endif
            int k = 0, j = 0;
            for (int i = 0; i < n_rows; ++i) {
                if (data[row_inds[i]][col_index] > threshold) {
                    larger_inds[k] = row_inds[i];
                    k += 1;
                } else {
                    smaller_inds[j] = row_inds[i];
                    j += 1;
                }
            }
#if _DEBUG_MATRIX
            assert(k == n_larger_in_col(col_index, threshold, row_inds, n_rows));
#endif
            n_two_inds[0] = k;
            n_two_inds[1] = j;
        }

    }

}

// average in a col for certain rows
namespace dbm {

    template<typename T>
    T Matrix<T>::average_col_for_rows(int col_index, const int *row_inds, int n_rows) const {
        if (row_inds == NULL) {
            T result = 0;
            for (int i = 0; i < height; ++i) {
                result += data[i][col_index];
            }
            return result / T(height);
        } else {
#if _DEBUG_MATRIX
            assert(n_rows > 0);
#endif
            T result = 0;
            for (int i = 0; i < n_rows; ++i) {
                result += data[row_inds[i]][col_index];
            }
            return result / T(n_rows);
        }
    }

    template<typename T>
    void Matrix<T>::ul_average_col_for_rows(int col_index, const T &threshold, T *two_average, const int *row_inds,
                                            int n_rows) const {
        two_average[0] = 0, two_average[1] = 0;
        if (row_inds == NULL) {
            for (int i = 0; i < height; ++i) {
                if (data[i][col_index] > threshold)
                    two_average[0] += data[i][col_index];
                else
                    two_average[1] += data[i][col_index];
            }
            two_average[0] /= T(height), two_average[1] /= T(height);
        } else {
#if _DEBUG_MATRIX
            assert(n_rows > 0);
#endif
            for (int i = 0; i < n_rows; ++i) {
                if (data[row_inds[i]][col_index] > threshold)
                    two_average[0] += data[row_inds[i]][col_index];
                else
                    two_average[1] += data[row_inds[i]][col_index];
            }
            two_average[0] /= T(n_rows), two_average[1] /= T(n_rows);
        }
    }

}

// math operations
namespace dbm {

    template <typename T>
    Matrix<T> transpose(const Matrix<T> &matrix) {
        Matrix<T> result(matrix.width, matrix.height, 0);
        for(int i = 0; i < matrix.height; ++i)
            for(int j = 0; j < matrix.width; ++j)
                result.data[j][i] = matrix.data[i][j];
        return result;
    }

    template <typename T>
    Matrix<T> plus(const Matrix<T> &left, const Matrix<T> &right) {
        #if _DEBUG_MATRIX
            assert(left.width == right.width && left.height == right.height);
        #endif
        Matrix<T> result(left.height, left.width, 0);
        for(int i = 0; i < left.height; ++i)
            for(int j = 0; j < left.width; ++j)
                result.data[i][j] = left.data[i][j] + right.data[i][j];
        return result;
    }

    template <typename T>
    Matrix<T> substract(const Matrix<T> &left, const Matrix<T> &right) {
        #if _DEBUG_MATRIX
            assert(left.width == right.width && left.height == right.height);
        #endif
        Matrix<T> result(left.height, left.width, 0);
        for(int i = 0; i < left.height; ++i)
            for(int j = 0; j < left.width; ++j)
                result.data[i][j] = left.data[i][j] - right.data[i][j];
        return result;
    }

    template <typename T>
    Matrix<T> inner_product(const Matrix<T> &left, const Matrix<T> &right) {
        #if _DEBUG_MATRIX
            assert(left.width == right.height);
        #endif
        Matrix<T> result(left.height, right.width, 0);
        for(int i = 0; i < left.height; ++i)
            for(int j = 0; j < right.width; ++j)
                for(int k = 0; k < left.width; ++k)
                    result.data[i][j] += left.data[i][k] * right.data[k][j];
        return result;
    }

    template <typename T>
    T determinant(const Matrix<T> &matrix) {
        #if _DEBUG_MATRIX
            assert(matrix.width == matrix.height);
        #endif
        if(matrix.width == 1)
            return matrix.data[0][0];
        else if(matrix.width == 2)
            return matrix.data[0][0] * matrix.data[1][1] - matrix.data[1][0] * matrix.data[0][1];
        else {
            Matrix<T> temp = copy(matrix);
            T ratio, result = 1;
            int i = 0;
            for(; i < temp.height - 1; ++i) {
                for(int j = i + 1; j < temp.height; ++j) {
                    ratio = temp.data[j][i] / temp.data[i][i];
                    for(int k = i; k < temp.width; ++k)
                        temp.data[j][k] = temp.data[j][k] - ratio * temp.data[i][k];
                }
                result *= temp.data[i][i];
            }
            return result * temp.data[i][i];
        }
    }

    template <typename T>
    Matrix<T> inverse(const Matrix<T> &matrix) {
        #if _DEBUG_MATRIX
            assert(matrix.width > 0 && matrix.width == matrix.height &&
                           std::abs(determinant(matrix)) > std::numeric_limits<T>::min() * 1e5);
        #endif
        Matrix<T> result(matrix.height, matrix.width, 0);
        if(matrix.width == 1) {
            result.data[0][0] = 1 / matrix.data[0][0];
            return result;
        }
        else if(matrix.width == 2) {
            T denominator = determinant(matrix);
            result.data[0][0] = matrix.data[1][1] / denominator;
            result.data[0][1] = - matrix.data[0][1] / denominator;
            result.data[1][0] = - matrix.data[1][0] / denominator;
            result.data[1][1] = matrix.data[0][0] / denominator;
            return result;
        }
        else {
            Matrix<T> temp(matrix.height, matrix.width * 2, 0);
            for(int i = 0; i < matrix.height; ++i) {
                std::copy(matrix.data[i], matrix.data[i] + matrix.width, temp.data[i]);
                temp.data[i][i + matrix.width] = 1;
            }
            T ratio, rescaling_coef;
            int i = 0;
            for(; i < temp.height - 1; ++i) {
                for(int j = i + 1; j < temp.height; ++j) {
                    ratio = temp.data[j][i] / temp.data[i][i];
                    for(int k = i; k < matrix.width * 2; ++k)
                        temp.data[j][k] = temp.data[j][k] - ratio * temp.data[i][k];
                }
                rescaling_coef = temp.data[i][i];
                for(int k = i; k < matrix.width * 2; ++k)
                    temp.data[i][k] /= rescaling_coef;
            }

            rescaling_coef = temp.data[i][i];
            for(int k = i; k < matrix.width * 2; ++k)
                temp.data[i][k] /= rescaling_coef;

            for(i = temp.height - 1; i > 0; --i) {
                for(int j = i - 1; j > -1; --j) {
                    ratio = temp.data[j][i] / temp.data[i][i];
                    for(int k = i; k < matrix.width * 2; ++k)
                        temp.data[j][k] = temp.data[j][k] - ratio * temp.data[i][k];
                }
            }
            for(i = 0; i < matrix.height; ++i) {
                std::copy(temp.data[i] + matrix.width, temp.data[i] + 2 * matrix.width, result.data[i]);
            }
            return result;
        }

    }

    template <typename T>
    void Matrix<T>::inplace_elewise_prod_mat_with_row_vec(const Matrix<T> &row) {
        #if _DEBUG_MATRIX
            assert(width == row.width && row.height == 1);
        #endif
        for(int i = 0; i < height; ++i)
            for(int j = 0; j < width; ++j)
                data[i][j] *= row.data[0][j];
    }
}

// merge horizontally, merge horizontally and deep copy
namespace dbm {

    template<typename T>
    Matrix<T> vert_merge(const Matrix<T> &upper, const Matrix<T> &lower) {
#if _DEBUG_MATRIX
        assert(upper.width == lower.width);
#endif
        Matrix<T> result(upper.height + lower.height, upper.width, 0);
        for (int i = 0; i < upper.height; ++i) {
            std::copy(upper.data[i], upper.data[i] + upper.width, result.data[i]);
#if _DEBUG_MATRIX
            result.row_labels[i] = upper.row_labels[i];
#endif
        }
        for (int i = 0; i < lower.height; ++i) {
            std::copy(lower.data[i], lower.data[i] + lower.width, result.data[upper.height + i]);
#if _DEBUG_MATRIX
            result.row_labels[upper.height + i] = lower.row_labels[i];
#endif
        }
#if _DEBUG_MATRIX
        for (int j = 0; j < upper.width; ++j)
            result.col_labels[j] = upper.col_labels[j];
#endif
        return result;
    }

    template<typename T>
    Matrix<T> hori_merge(const Matrix<T> &left, const Matrix<T> &right) {
#if _DEBUG_MATRIX
        assert(left.height == right.height);
#endif
        Matrix<T> result(left.height, left.width + right.width, 0);
        for (int i = 0; i < left.height; ++i) {
            std::copy(left.data[i], left.data[i] + left.width, result.data[i]);
            std::copy(right.data[i], right.data[i] + right.width, result.data[i] + left.width);
#if _DEBUG_MATRIX
            result.row_labels[i] = left.row_labels[i];
#endif
        }
#if _DEBUG_MATRIX
        for (int j = 0; j < left.width; ++j)
            result.col_labels[j] = left.col_labels[j];
        for (int j = left.width; j < left.width + right.width; ++j)
            result.col_labels[j] = right.col_labels[j - left.width];
#endif
        return result;
    }

    template<typename T>
    Matrix<T> copy(const Matrix<T> &target) {
        Matrix<T> result(target.height, target.width);
        for (int i = 0; i < target.height; ++i) {
            std::copy(target.data[i], target.data[i] + target.width, result.data[i]);
        }
        #if _DEBUG_MATRIX
            std::copy(target.row_labels, target.row_labels + target.height, result.row_labels);
            std::copy(target.col_labels, target.col_labels + target.width, result.col_labels);
        #endif
        return result;
    }

    template<typename T>
    void copy(const Matrix<T> &target, Matrix<T> &to) {

#if _DEBUG_MATRIX
        assert(target.height == to.height && target.width == to.width);
#endif

        for (int i = 0; i < target.height; ++i) {
            std::copy(target.data[i], target.data[i] + target.width, to.data[i]);
        }
#if _DEBUG_MATRIX
        std::copy(target.row_labels, target.row_labels + target.height, to.row_labels);
        std::copy(target.col_labels, target.col_labels + target.width, to.col_labels);
#endif

    }

}

// explicit instantiation of templated friend functions
namespace dbm {

    template Matrix<double> transpose<double>(const Matrix<double> &matrix);

    template Matrix<float> transpose<float>(const Matrix<float> &matrix);

    template Matrix<double> plus<double>(const Matrix<double> &left, const Matrix<double> &right);

    template Matrix<float> plus<float>(const Matrix<float> &left, const Matrix<float> &right);

    template Matrix<double> substract<double>(const Matrix<double> &left, const Matrix<double> &right);

    template Matrix<float> substract<float>(const Matrix<float> &left, const Matrix<float> &right);

    template Matrix<double> inner_product<double>(const Matrix<double> &left, const Matrix<double> &right);

    template Matrix<float> inner_product<float>(const Matrix<float> &left, const Matrix<float> &right);

    template double determinant<double>(const Matrix<double> &matrix);

    template float determinant<float>(const Matrix<float> &matrix);

    template Matrix<double> inverse<double>(const Matrix<double> &matrix);

    template Matrix<float> inverse<float>(const Matrix<float> &matrix);

    template Matrix<double> vert_merge<double>(const Matrix<double> &upper, const Matrix<double> &lower);

    template Matrix<float> vert_merge<float>(const Matrix<float> &upper, const Matrix<float> &lower);

    template Matrix<double> hori_merge<double>(const Matrix<double> &left, const Matrix<double> &right);

    template Matrix<float> hori_merge<float>(const Matrix<float> &left, const Matrix<float> &right);

    template Matrix<double> copy<double>(const Matrix<double> &target);

    template Matrix<float> copy<float>(const Matrix<float> &target);

    template void copy<double>(const Matrix<double> &target, Matrix<double> &to);

    template void copy<float>(const Matrix<float> &target, Matrix<float> &to);

}



