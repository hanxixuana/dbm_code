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

    template <typename T>
    Matrix<T>::Matrix() : height(0), width(0) {}

    template<typename T>
    Matrix<T>::Matrix(int height, int width) :
            height(height), width(width) {

        #ifdef _CD_INDICATOR
            std::cout << "Instantiating Matrix at " << this << "." << std::endl;
        #endif

        std::srand((unsigned int)(std::time(nullptr)));
        data = new T *[height];
        for (int i = 0; i < height; ++i) {
            data[i] = new T[width];
            for (int j = 0; j < width; ++j) {
                data[i][j] = T(std::rand()) / RAND_MAX * 2 - 1;
            }
        }

        #ifdef _DEBUG_MATRIX
        col_labels = new int[width];
        for (int i = 0; i < width; ++i) {
            col_labels[i] = i;
        }

        row_labels = new int[height];
        for (int i = 0; i < height; ++i)
            row_labels[i] = i;
        #endif

        #ifdef _CD_INDICATOR
            std::cout << "Matrix at " << this << " is instantiated." << std::endl;
        #endif
    }

    template<typename T>
    Matrix<T>::Matrix(int height, int width, const T &value) :
            height(height), width(width) {

        #ifdef _CD_INDICATOR
        std::cout << "Instantiating Matrix at " << this << "." << std::endl;
        #endif

        data = new T *[height];
        for (int i = 0; i < height; ++i) {
            data[i] = new T[width];
            for (int j = 0; j < width; ++j) {
                data[i][j] = value;
            }
        }

        #ifdef _DEBUG_MATRIX
        col_labels = new int[width];
        for (int i = 0; i < width; ++i) {
            col_labels[i] = i;
        }


        row_labels = new int[height];
        for (int i = 0; i < height; ++i)
            row_labels[i] = i;
        #endif

        #ifdef _CD_INDICATOR
        std::cout << "Matrix at " << this << " is instantiated." << std::endl;
        #endif
    }

    template<typename T>
    Matrix<T>::Matrix(const Matrix<T>& rhs) {

        #ifdef _CD_INDICATOR
            std::cout << "Copying Matrix at " << this << "." << std::endl;
        #endif

		if (data != nullptr) {
			for (int i = 0; i < height; i ++) {
				delete[] data[i];
			} // i
			delete[] data;
		}
        #ifdef _DEBUG_MATRIX
        if (col_labels != nullptr) delete[] col_labels;
        if (row_labels != nullptr) delete[] row_labels;
        #endif

		height = rhs.height;
		width = rhs.width;
		data = new T*[height];
		for (int i = 0; i < height; i ++) {
			data[i] = new T[width];
			for (int j = 0; j < width; j ++) {
				data[i][j] = rhs.data[i][j];
			} // j	
		} // i
		
        #ifdef _DEBUG_MATRIX
        col_labels = new int[width];
        for (int i = 0; i < width; ++i) {
            col_labels[i] = rhs.col_labels[i];
        }

        row_labels = new int[height];
        for (int i = 0; i < height; ++i)
            row_labels[i] = rhs.row_labels[i];
        #endif

        #ifdef _CD_INDICATOR
            std::cout << "Matrix at " << this << " is instantiated." << std::endl;
        #endif
    } // copy constructor

    template<typename T>
	Matrix<T>& Matrix<T>::operator=(const Matrix<T>& rhs) {

		#ifdef _CD_INDICATOR
        	std::cout << "Assigning Matrix at " << this << "." << std::endl;
		#endif

		if (data != nullptr) {
			for (int i = 0; i < height; i ++) {
				delete[] data[i];
			} // i
			delete[] data;
		}
        #ifdef _DEBUG_MATRIX
        if (col_labels != nullptr) delete[] col_labels;
        if (row_labels != nullptr) delete[] row_labels;
        #endif
		
		height = rhs.height;
		width = rhs.width;
		data = new T*[height];
		for (int i = 0; i < height; i ++) {
			data[i] = new T[width];
			for (int j = 0; j < width; j ++) {
				data[i][j] = rhs.data[i][j];
			} // j	
		} // i
		
        #ifdef _DEBUG_MATRIX
        col_labels = new int[width];
        for (int i = 0; i < width; ++i) {
            col_labels[i] = rhs.col_labels[i];
        }

        row_labels = new int[height];
        for (int i = 0; i < height; ++i)
            row_labels[i] = rhs.row_labels[i];
        #endif

        #ifdef _CD_INDICATOR
            std::cout << "Matrix at " << this << " is instantiated." << std::endl;
        #endif
		
		return *this;
    } // operator=

    /*  the data must have the form as below
     * s_f  f_0 f_1 f_2 ....
     * s_0  xx  xx  xx  ....
     * s_1  xx  xx  xx  ....
     * s_2  xx  xx  xx  ....
     */
    template<typename T>
    Matrix<T>::Matrix(int height, int width,
                      const std::string file_name, 
                      const char &delimiter) :
            height(height), width(width) {

        #ifdef _CD_INDICATOR
            std::cout << "Instantiating Matrix at " << this << "." << std::endl;
        #endif

        data = new T *[height];
        for (int i = 0; i < height; ++i) {
            data[i] = new T[width];
        }
        #ifdef _DEBUG_MATRIX
        col_labels = new int[width];
        row_labels = new int[height];
        #endif

        std::ifstream file(file_name.c_str());
        std::string line;
        int line_number = 0, col_number = 0;

        // read feature labels
        unsigned long prev = 0, next = 0;
        std::string temp_storage;
        #ifdef _DEBUG_MATRIX
        std::getline(file, line);
        next = line.find_first_of(delimiter, prev);
        prev = next + 1;
        while ((next = line.find_first_of(delimiter, prev)) != std::string::npos) {
            temp_storage = line.substr(prev, next - prev);
            col_labels[col_number] = std::atoi(temp_storage.c_str());
            col_number++;
            prev = next + 1;
        }
        if (prev < line.size()) {
            temp_storage = line.substr(prev);
            col_labels[col_number] = std::atoi(temp_storage.c_str());
            col_number++;
        }
        line_number++;

        assert(col_number == width);
        #endif

        // read row labels and samples
        while (std::getline(file, line)) {
            col_number = 0;
            prev = 0, next = 0;
            #ifdef _DEBUG_MATRIX
            next = line.find_first_of(delimiter, prev);
            temp_storage = line.substr(prev, next - prev);
            row_labels[line_number - 1] = std::atoi(temp_storage.c_str());
            col_number++;
            prev = next + 1;
            #endif
            while ((next = line.find_first_of(delimiter, prev)) != std::string::npos) {
                temp_storage = line.substr(prev, next - prev);
                data[line_number - 1][col_number - 1] = std::atof(temp_storage.c_str());
                col_number++;
                prev = next + 1;
            }
            if (prev < line.size()) {
                temp_storage = line.substr(prev);
                data[line_number - 1][col_number - 1] = std::atof(temp_storage.c_str());
                col_number++;
                prev = next + 1;
            }
            line_number++;
        }
        #ifdef _DEBUG_MATRIX
        assert(line_number - 1 == height);
        #endif

        #ifdef _CD_INDICATOR
        std::cout << "Matrix at " << this << " is instantiated." << std::endl;
        #endif
    }

    template<typename T>
    Matrix<T>::~Matrix() {

        #ifdef _CD_INDICATOR
            std::cout << "Deleting Matrix at " << this << "." << std::endl;
        #endif

        for (int i = 0; i < height; ++i) {
            delete[] data[i];
        }

        delete[] data;

        #ifdef _DEBUG_MATRIX
        delete[] col_labels;
        delete[] row_labels;
        #endif

        #ifdef _CD_INDICATOR
        std::cout << "Matrix at " << this << " is deleted." << std::endl;
        #endif

    };

    template<typename T>
    void Matrix<T>::print() const {

        #ifdef _DEBUG_MATRIX
            printf("s_f\t");
//            std::cout << "s_f\t";

            for (int i = 0; i < width; ++i)
                printf("%d\t", col_labels[i]);
//                std::cout << col_labels[i] << "\t";
            printf("\n");
//            std::cout << std::endl;
        #endif

        for (int i = 0; i < height; ++i) {
            #ifdef _DEBUG_MATRIX
                printf("%d\t", row_labels[i]);
//            std::cout << row_labels[i] << "\t";
            #endif
            for (int j = 0; j < width; ++j)
                printf("%.5lf\t", data[i][j]);
//                std::cout << std::fixed << std::setprecision(4) << data[i][j] << "\t";
            printf("\n");
//            std::cout << std::endl;
        }
        printf("\n");
//        std::cout << std::endl;
    }

//    template <typename T>
//    std::string Matrix<T>::print_to_string() const {
//
//        std::ostringstream output;
//
//        #ifdef _DEBUG_MATRIX
//            output << "s_f\t";
//            int i = 0;
//            for (; i < width - 1; ++i)
//                output << col_labels[i] << "\t";
//            output << col_labels[i] << std::endl;
//        #endif
//
//        for (i = 0; i < height; ++i) {
//            #ifdef _DEBUG_MATRIX
//                output << row_labels[i] << "\t";
//            #endif
//            int j = 0;
//            for (; j < width - 1; ++j)
//                output << std::fixed << std::setprecision(5) << data[i][j] << "\t";
//            output << std::fixed << std::setprecision(5) << data[i][j] << std::endl;
//        }
//
//        return output.str();
//
//    }

    template<typename T>
    void Matrix<T>::print_to_file(const std::string &file_name, const char &delimiter) const {
        std::ofstream file(file_name.c_str());

        #ifdef _DEBUG_MATRIX
            file << "s_f" << delimiter;
            int i = 0;
            for (; i < width - 1; ++i)
                file << col_labels[i] << delimiter;
            file << col_labels[i] << std::endl;
        #else
            int i = 0;
        #endif

        for (i = 0; i < height; ++i) {
            #ifdef _DEBUG_MATRIX
            file << row_labels[i] << delimiter;
            #endif
            int j = 0;
            for (; j < width - 1; ++j)
                file << std::fixed << std::setprecision(5) << data[i][j] << delimiter;
            file << std::fixed << std::setprecision(5) << data[i][j] << std::endl;
        }

        file.close();
    }

}

// dimensions and ranges
namespace dbm {

    template<typename T>
    T Matrix<T>::get_col_max(int col_index, const int *row_inds, int no_rows) const {
        if (row_inds == nullptr) {
            T result = std::numeric_limits<T>::lowest();
            for (int i = 0; i < height; ++i) {
                if (data[i][col_index] >= result)
                    result = data[i][col_index];
            }
            return result;
        } else {
            #ifdef _DEBUG_MATRIX
            assert(no_rows > 0);
            #endif
            T result = std::numeric_limits<T>::lowest();
            for (int i = 0; i < no_rows; ++i) {
                if (data[row_inds[i]][col_index] >= result)
                    result = data[row_inds[i]][col_index];
            }
            return result;
        }
    }

    template<typename T>
    T Matrix<T>::get_col_min(int col_index, const int *row_inds, int no_rows) const {
        if (row_inds == nullptr) {
            T result = std::numeric_limits<T>::max();
            for (int i = 0; i < height; ++i) {
                if (data[i][col_index] <= result)
                    result = data[i][col_index];
            }
            return result;
        } else {
            #ifdef _DEBUG_MATRIX
            assert(no_rows > 0);
            #endif
            T result = std::numeric_limits<T>::max();
            for (int i = 0; i < no_rows; ++i) {
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
    inline int Matrix<T>::unique_vals_col(int col_index,
                                          T *values,
                                          const int *row_inds,
                                          int no_rows) const {
        // usage:
        //    double uniques[b.get_height()];
        //    int end = b.unique_vals_col(1, uniques);
        //    cout << b.get_height() << ' ' << end << endl;
        //    for(int i = 0; i < end; ++i) cout << uniques[i] << ' ';
        //    cout << endl;
        #ifdef _DEBUG_MATRIX
        assert(col_index < width);
        #endif
        if (row_inds == nullptr) {
            for (int i = 0; i < height; ++i)
                values[i] = data[i][col_index];
            std::sort(values, values + height);
            T *end = std::unique(values, values + height);
            return (int) std::distance(values, end);
        } else {
            #ifdef _DEBUG_MATRIX
            assert(no_rows > 0);
            #endif
            for (int i = 0; i < no_rows; ++i)
                values[i] = data[row_inds[i]][col_index];
            std::sort(values, values + no_rows);
            T *end = std::unique(values, values + no_rows);
            return (int) std::distance(values, end);
        }
    }
}

// clear
namespace dbm {

    template <typename T>
    void Matrix<T>::clear() {
        for(int i = 0; i < height; ++i)
            for(int j = 0; j < width; ++j)
                data[i][j] = (T) 0;
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
        for (int i = 0; i < height; ++i) 
            r_inds[i] = i;
        std::random_shuffle(r_inds, r_inds + height);
        return rows(r_inds, height);
    };

}

// assignment
namespace dbm {

    template<typename T>
    void Matrix<T>::assign(int i, int j, const T &value) {
        #ifdef _DEBUG_MATRIX
        assert(i < height && j < width);
        #endif
        data[i][j] = value;
    }

    // carefully check if the length of column is equal to height
    template<typename T>
    void Matrix<T>::assign_col(int j, T *column) {
        #ifdef _DEBUG_MATRIX
        assert(j < width);
        #endif
        for (int i = 0; i < height; ++i) {
            data[i][j] = column[i];
        }
    }

    // carefully check if the length of row is equal to width
    template<typename T>
    void Matrix<T>::assign_row(int i, T *row) {
        #ifdef _DEBUG_MATRIX
        assert(i < height);
        #endif
        std::copy(row, row + width, data[i]);
    }

    #ifdef _DEBUG_MATRIX

    template<typename T>
    void Matrix<T>::assign_row_label(int i, const int &label) {
        assert(i < height);
        row_labels[i] = label;
    }

    template<typename T>
    void Matrix<T>::assign_col_label(int j, const int &label) {
        assert(j < width);
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
        #ifdef _DEBUG_MATRIX
        assert(k < height);
        #endif
        return data[k];
    }

}

// get element, rows, columns, submatrices
namespace dbm {

    template<typename T>
    T Matrix<T>::get(int i, int j) const {
        #ifdef _DEBUG_MATRIX
        assert(i < height && j < width);
        #endif
        return data[i][j];
    }

    template<typename T>
    Matrix<T> Matrix<T>::col(int col_index) const {
        #ifdef _DEBUG_MATRIX
        assert(col_index < height);
        #endif
        Matrix<T> result(height, 1, 0);
        for (int i = 0; i < height; ++i) {
            result.data[i][0] = data[i][col_index];
            #ifdef _DEBUG_MATRIX
            result.row_labels[i] = row_labels[i];
            #endif
        }
        #ifdef _DEBUG_MATRIX
        result.col_labels[0] = col_labels[col_index];
        #endif
        return result;
    }

    template<typename T>
    Matrix<T> Matrix<T>::row(int row_index) const {
        #ifdef _DEBUG_MATRIX
        assert(row_index < height);
        #endif
        Matrix<T> result(1, width, 0);
        std::copy(data[row_index], data[row_index] + width, result.data[0]);
        #ifdef _DEBUG_MATRIX
        std::copy(col_labels, col_labels + width, result.col_labels);
        result.row_labels[0] = row_labels[row_index];
        #endif

        return result;
    }

    template<typename T>
    Matrix<T> Matrix<T>::cols(const int *col_indices, int no_cols) const {
        Matrix<T> result(height, no_cols, 0);
        for (int j = 0; j < no_cols; ++j) {
            #ifdef _DEBUG_MATRIX
            assert(col_indices[j] < width);
            #endif
            for (int i = 0; i < height; ++i)
                result.data[i][j] = data[i][col_indices[j]];
            #ifdef _DEBUG_MATRIX
            result.col_labels[j] = col_labels[col_indices[j]];
            #endif
        }
        #ifdef _DEBUG_MATRIX
        for (int i = 0; i < height; ++i)
            result.row_labels[i] = row_labels[i];
        #endif

        return result;
    }

    template<typename T>
    Matrix<T> Matrix<T>::rows(const int *row_indices, int no_rows) const {
        Matrix<T> result(no_rows, width, 0);
        for (int i = 0; i < no_rows; ++i) {
            #ifdef _DEBUG_MATRIX
            assert(row_indices[i] < height);
            #endif
            std::copy(data[row_indices[i]], data[row_indices[i]] + width, result.data[i]);
            #ifdef _DEBUG_MATRIX
            result.row_labels[i] = row_labels[row_indices[i]];
            #endif

        }
        #ifdef _DEBUG_MATRIX
        std::copy(col_labels, col_labels + width, result.col_labels);
        #endif

        return result;
    }

    template<typename T>
    Matrix<T> Matrix<T>::submatrix(const int *row_indices, int no_rows,
                                   const int *col_indices, int no_cols) const {
        return rows(row_indices, no_rows).cols(col_indices, no_cols);
    }

}

// split into two Matrix<T> according to a col and a threshold
namespace dbm {

    template<typename T>
    int Matrix<T>::n_larger_in_col(int col_index,
                                   const T &threshold,
                                   const int *row_inds,
                                   int no_rows) const {
        #ifdef _DEBUG_MATRIX
        assert(col_index < width);
        #endif
        if (row_inds == nullptr) {
            int result = 0;
            for (int i = 0; i < height; ++i)
                result += data[i][col_index] > threshold;
            return result;
        } else {
            #ifdef _DEBUG_MATRIX
            assert(no_rows > 0);
            #endif
            int result = 0;
            for (int i = 0; i < no_rows; ++i)
                result += data[row_inds[i]][col_index] > threshold;
            return result;
        }
    }

    template<typename T>
    int Matrix<T>::n_smaller_or_eq_in_col(int col_index,
                                          const T &threshold,
                                          const int *row_inds,
                                          int no_rows) const {
        if (row_inds == nullptr) {
            return height - n_larger_in_col(col_index, threshold);
        } else {
            #ifdef _DEBUG_MATRIX
            assert(no_rows > 0);
            #endif
            return no_rows - n_larger_in_col(col_index, threshold, row_inds, no_rows);
        }
    }

    template<typename T>
    int Matrix<T>::inds_larger_in_col(int col_index,
                                      const T &threshold,
                                      int *indices,
                                      const int *row_inds,
                                      int no_rows) const {
        if (row_inds == nullptr) {
            int k = 0;
            for (int i = 0; i < height; ++i) {
                if (data[i][col_index] > threshold) {
                    indices[k] = i;
                    k++;
                }
            }
            #ifdef _DEBUG_MATRIX
            assert(k == n_larger_in_col(col_index, threshold));
            #endif
            return k;
        } else {
            #ifdef _DEBUG_MATRIX
            assert(no_rows > 0);
            #endif
            int k = 0;
            for (int i = 0; i < no_rows; ++i) {
                if (data[row_inds[i]][col_index] > threshold) {
                    indices[k] = row_inds[i];
                    k++;
                }
            }
            #ifdef _DEBUG_MATRIX
            assert(k == n_larger_in_col(col_index, threshold, row_inds, no_rows));
            #endif
            return k;
        }
    }

    template<typename T>
    int Matrix<T>::inds_smaller_or_eq_in_col(int col_index,
                                             const T &threshold,
                                             int *indices,
                                             const int *row_inds,
                                             int no_rows) const {
        if (row_inds == nullptr) {
            int k = 0;
            for (int i = 0; i < height; ++i) {
                if (data[i][col_index] <= threshold) {
                    indices[k] = i;
                    k++;
                }
            }
            #ifdef _DEBUG_MATRIX
            assert(k == n_smaller_or_eq_in_col(col_index, threshold));
            #endif
            return k;
        } else {
            #ifdef _DEBUG_MATRIX
            assert(no_rows > 0);
            #endif
            int k = 0;
            for (int i = 0; i < no_rows; ++i) {
                if (data[row_inds[i]][col_index] <= threshold) {
                    indices[k] = row_inds[i];
                    k++;
                }
            }
            #ifdef _DEBUG_MATRIX
            assert(k == n_smaller_or_eq_in_col(col_index, threshold, row_inds, no_rows));
            #endif
            return k;
        }
    }

    template<typename T>
    Matrix<T> Matrix<T>::vert_split_l(int col_index, const T &threshold) const {
        int n_larger = n_larger_in_col(col_index, threshold);
        #ifdef _DEBUG_MATRIX
        assert(n_larger > 0);
        #endif
        int larger_indices[n_larger];
        inds_larger_in_col(col_index, threshold, larger_indices);
        return rows(larger_indices, n_larger);
    }

    template<typename T>
    Matrix<T> Matrix<T>::vert_split_s(int col_index, const T &threshold) const {
        int n_smaller = n_smaller_or_eq_in_col(col_index, threshold);
        #ifdef _DEBUG_MATRIX
        assert(n_smaller > 0);
        #endif
        int smaller_indices[n_smaller];
        inds_smaller_or_eq_in_col(col_index, threshold, smaller_indices);
        return rows(smaller_indices, n_smaller);
    }

    template<typename T>
    inline void Matrix<T>::inds_split(int col_index, const T &threshold, int *larger_inds,
                                      int *smaller_inds, int *n_two_inds,
                                      const int *row_inds, int no_rows) const {

        if (row_inds == nullptr) {
            int k = 0, j = 0;
            for (int i = 0; i < height; ++i) {
                if (data[i][col_index] > threshold) {
                    larger_inds[k] = i;
                    k++;
                } else {
                    smaller_inds[j] = i;
                    j++;
                }
            }
            #ifdef _DEBUG_MATRIX
            assert(k == n_larger_in_col(col_index, threshold));
            #endif
            n_two_inds[0] = k;
            n_two_inds[1] = j;

        } else {
            #ifdef _DEBUG_MATRIX
            assert(no_rows > 0);
            #endif
            int k = 0, j = 0;
            for (int i = 0; i < no_rows; ++i) {
                if (data[row_inds[i]][col_index] > threshold) {
                    larger_inds[k] = row_inds[i];
                    k++;
                } else {
                    smaller_inds[j] = row_inds[i];
                    j++;
                }
            }
            #ifdef _DEBUG_MATRIX
            assert(k == n_larger_in_col(col_index, threshold, row_inds, no_rows));
            #endif
            n_two_inds[0] = k;
            n_two_inds[1] = j;
        }

    }

}

// average in a col for certain rows
namespace dbm {

    template<typename T>
    T Matrix<T>::average_col_for_rows(int col_index, const int *row_inds, int no_rows) const {
        if (row_inds == nullptr) {
            T result = 0;
            for (int i = 0; i < height; ++i) {
                result += data[i][col_index];
            }
            return result / height;
        } else {
            #ifdef _DEBUG_MATRIX
            assert(no_rows > 0);
            #endif
            T result = 0;
            for (int i = 0; i < no_rows; ++i) {
                result += data[row_inds[i]][col_index];
            }
            return result / no_rows;
        }
    }

    template<typename T>
    void Matrix<T>::ul_average_col_for_rows(int col_index,
                                            const T &threshold,
                                            T *two_average,
                                            const int *row_inds,
                                            int no_rows) const {
        two_average[0] = 0, two_average[1] = 0;
        int j = 0, k = 0;
        if (row_inds == nullptr) {
            for (int i = 0; i < height; ++i) {
                if (data[i][col_index] > threshold) {
                    two_average[0] += data[i][col_index];
                    j++;
                }
                else {
                    two_average[1] += data[i][col_index];
                    k++;
                }
            }
            two_average[0] /= j, two_average[1] /= k;
        } else {
            #ifdef _DEBUG_MATRIX
            assert(no_rows > 0);
            #endif
            j = 0;
            k = 0;
            for (int i = 0; i < no_rows; ++i) {
                if (data[row_inds[i]][col_index] > threshold) {
                    two_average[0] += data[row_inds[i]][col_index];
                    j++;
                }
                else {
                    two_average[1] += data[row_inds[i]][col_index];
                    k++;
                }
            }
            two_average[0] /= j;
            two_average[1] /= k;
        }
    }

}

// math operations
namespace dbm {

    template <typename T>
    T Matrix<T>::row_sum(const int &row_ind) const {

        T result = 0;
        for(int i = 0; i < width; ++i)
            result += data[row_ind][i];

        return result;

    }

    template <typename T>
    T Matrix<T>::col_sum(const int &col_ind) const {

        T result = 0;
        for(int i = 0; i < height; ++i)
            result += data[i][col_ind];

        return result;

    }

    template <typename T>
    T Matrix<T>::row_average(const int &row_ind) const {

        return row_sum(row_ind) / width;

    }

    template <typename T>
    T Matrix<T>::col_average(const int &col_ind) const {

        return col_sum(col_ind) / height;

    }

    template <typename T>
    T Matrix<T>::row_std(const int &row_ind) const {

        T average = row_average(row_ind),
                result = 0;
        for(int i = 0; i < width; ++i)
            result += (data[row_ind][i] - average) * (data[row_ind][i] - average);

        return std::sqrt(result / (width - 1));

    }

    template <typename T>
    T Matrix<T>::col_std(const int &col_ind) const {

        T average = col_average(col_ind),
                result = 0;
        for(int i = 0; i < height; ++i)
            result += (data[i][col_ind] - average) * (data[i][col_ind] - average);

        return std::sqrt(result / (height - 1));

    }

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

        if(!(left.width == right.width && left.height == right.height)) {
            left.print_to_file("left.txt");
            right.print_to_file("right.txt");
            std::cout << left.height << ' ' << right.height << std::endl;
            std::cout << left.width << ' ' << right.width << std::endl;
        }

        #ifdef _DEBUG_MATRIX
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
        #ifdef _DEBUG_MATRIX
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
        #ifdef _DEBUG_MATRIX
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
        #ifdef _DEBUG_MATRIX
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

        T abs_det = std::abs(determinant(matrix));

//        if(std::isnan(abs_det) || std::isinf(abs_det) || abs_det < std::numeric_limits<T>::min() * 1e2) {
//            std::cout << "The matrix has problems and is saved!"
//                      << std::endl;
//            matrix.print_to_file("matrix_fed_to_inverse.txt");
//        }

        #ifdef _DEBUG_MATRIX
            assert(matrix.width > 0 && matrix.width == matrix.height &&
                           abs_det > std::numeric_limits<T>::min() * 1e2);
        #endif
        Matrix<T> result(matrix.height, matrix.width, 0);
        if(matrix.width == 1) {
            result.data[0][0] = 1. / matrix.data[0][0];
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
                std::copy(temp.data[i] + matrix.width,
                          temp.data[i] + 2 * matrix.width,
                          result.data[i]);
            }
            return result;
        }

    }

    template <typename T>
    void Matrix<T>::inplace_elewise_prod_mat_with_row_vec(const Matrix<T> &row) {
        #ifdef _DEBUG_MATRIX
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
        #ifdef _DEBUG_MATRIX
        assert(upper.width == lower.width);
        #endif
        Matrix<T> result(upper.height + lower.height, upper.width, 0);
        for (int i = 0; i < upper.height; ++i) {
            std::copy(upper.data[i], upper.data[i] + upper.width, result.data[i]);
            #ifdef _DEBUG_MATRIX
            result.row_labels[i] = upper.row_labels[i];
            #endif
        }
        for (int i = 0; i < lower.height; ++i) {
            std::copy(lower.data[i], lower.data[i] + lower.width, result.data[upper.height + i]);
            #ifdef _DEBUG_MATRIX
            result.row_labels[upper.height + i] = lower.row_labels[i];
            #endif
        }
        #ifdef _DEBUG_MATRIX
        for (int j = 0; j < upper.width; ++j)
            result.col_labels[j] = upper.col_labels[j];
        #endif
        return result;
    }

    template<typename T>
    Matrix<T> hori_merge(const Matrix<T> &left, const Matrix<T> &right) {
        #ifdef _DEBUG_MATRIX
        assert(left.height == right.height);
        #endif
        Matrix<T> result(left.height, left.width + right.width, 0);
        for (int i = 0; i < left.height; ++i) {
            std::copy(left.data[i], left.data[i] + left.width, result.data[i]);
            std::copy(right.data[i], right.data[i] + right.width, result.data[i] + left.width);
            #ifdef _DEBUG_MATRIX
            result.row_labels[i] = left.row_labels[i];
            #endif
        }
        #ifdef _DEBUG_MATRIX
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
        #ifdef _DEBUG_MATRIX
            std::copy(target.row_labels, target.row_labels + target.height, result.row_labels);
            std::copy(target.col_labels, target.col_labels + target.width, result.col_labels);
        #endif
        return result;
    }

    template<typename T>
    void copy(const Matrix<T> &target, Matrix<T> &to) {

        #ifdef _DEBUG_MATRIX
        assert(target.height == to.height && target.width == to.width);
        #endif

        for (int i = 0; i < target.height; ++i) {
            std::copy(target.data[i], target.data[i] + target.width, to.data[i]);
        }
        #ifdef _DEBUG_MATRIX
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



