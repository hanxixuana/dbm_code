//
// Created by xixuan on 10/10/16.
//

#include "base_learner.h"

#include <cassert>
#include <limits>
#include <cmath>
#include <stdexcept>

namespace dbm {

    template
    class Tree_node<float>;

    template
    class Tree_node<double>;

    template
    class Linear_regression<double>;

    template
    class Linear_regression<float>;

    template
    class Global_mean<float>;

    template
    class Global_mean<double>;

}

namespace dbm {

    template <typename T>
    Global_mean<T>::Global_mean() : Base_learner<T>('m') {}

    template <typename T>
    Global_mean<T>::~Global_mean() {};

    template <typename T>
    T Global_mean<T>::predict_for_row(const Matrix<T> &data, int row_ind) {
        return mean;
    }

    template <typename T>
    void Global_mean<T>::predict(const Matrix<T> &data_x, Matrix<T> &prediction, const T shrinkage, const int *row_inds, int n_rows) {

        if (row_inds == NULL) {
            int data_height = data_x.get_height();
            #if _DEBUG_BASE_LEARNER
                assert(data_height == prediction.get_height() && prediction.get_width() == 1);
            #endif
            T predicted_value;
            for (int i = 0; i < data_height; ++i) {
                predicted_value = prediction.get(i, 0) + shrinkage * predict_for_row(data_x, i);
                prediction.assign(i, 0, predicted_value);
            }
        } else {
            #if _DEBUG_BASE_LEARNER
                assert(n_rows > 0 && prediction.get_height() == 1);
            #endif
            T predicted_value;
            for (int i = 0; i < n_rows; ++i) {
                predicted_value = prediction.get(row_inds[i], 0) + shrinkage * predict_for_row(data_x, row_inds[i]);
                prediction.assign(row_inds[i], 0, predicted_value);
            }
        }

    }

}

namespace dbm {

    template <typename T>
    Linear_regression<T>::Linear_regression(int n_predictor, char loss_type) :
            n_predictor(n_predictor), loss_type(loss_type), Base_learner<T>('l') {
        col_inds = new int[n_predictor];
        coefs_no_intercept = new T[n_predictor];
    }

    template <typename T>
    Linear_regression<T>::~Linear_regression() {
        delete col_inds;
        delete coefs_no_intercept;
    };

    template <typename T>
    T Linear_regression<T>::predict_for_row(const Matrix<T> &data, int row_ind) {
        T result = 0;
        for(int i = 0; i < n_predictor; ++i) {
            result += data.get(row_ind, col_inds[i]) * coefs_no_intercept[i];
        }
        result += intercept;
        switch (loss_type) {
            case 'n':
                return result;
            case 'p':
                return std::log(result <= 0.0001 ? 0.0001 : result);
            case 'b':
                return result;
            case 't':
                return std::log(result <= 0.0001 ? 0.0001 : result);
            default:
                throw std::invalid_argument("Specified distribution does not exist.");
        }
    }

    template <typename T>
    void Linear_regression<T>::predict(const Matrix<T> &data_x, Matrix<T> &prediction, const T shrinkage, const int *row_inds, int n_rows) {

        if (row_inds == NULL) {
            int data_height = data_x.get_height();
            #if _DEBUG_BASE_LEARNER
                assert(data_height == prediction.get_height() && prediction.get_width() == 1);
            #endif
            T predicted_value;
            for (int i = 0; i < data_height; ++i) {
                predicted_value = prediction.get(i, 0) + shrinkage * predict_for_row(data_x, i);
                prediction.assign(i, 0, predicted_value);
            }
        } else {
            #if _DEBUG_BASE_LEARNER
                assert(n_rows > 0 && prediction.get_height() == 1);
            #endif
            T predicted_value;
            for (int i = 0; i < n_rows; ++i) {
                predicted_value = prediction.get(row_inds[i], 0) + shrinkage * predict_for_row(data_x, row_inds[i]);
                prediction.assign(row_inds[i], 0, predicted_value);
            }
        }

    }

}

namespace dbm {

    template<typename T>
    Tree_node<T>::Tree_node(int depth) : larger(nullptr), smaller(nullptr), column(-1),
                                         split_value(0), loss(std::numeric_limits<T>::max()),
                                         depth(depth), last_node(false), prediction(0),
                                         no_training_samples(0), Base_learner<T>('t') {}

    template<typename T>
    Tree_node<T>::Tree_node(int depth, int column, bool last_node,
                            T split_value, T loss, T prediction,
                            int no_tr_samples) : larger(nullptr), smaller(nullptr), column(column),
                                                 split_value(split_value), loss(loss),
                                                 depth(depth), last_node(last_node), prediction(prediction),
                                                 no_training_samples(no_tr_samples), Base_learner<T>('t') {}

    template<typename T>
    Tree_node<T>::~Tree_node() {

        if (this == nullptr) return;

        delete larger;
        larger = nullptr;

        delete smaller;
        smaller = nullptr;

    }

    template<typename T>
    T Tree_node<T>::predict_for_row(const Matrix<T> &data_x, int row_ind) {
        if (last_node)
            return prediction;
        if (data_x.get(row_ind, column) > split_value) {
            #if _DEBUG_BASE_LEARNER
                assert(larger != NULL);
            #endif
            return larger->predict_for_row(data_x, row_ind);
        } else {
            #if _DEBUG_BASE_LEARNER
                assert(larger != NULL);
            #endif
            return smaller->predict_for_row(data_x, row_ind);
        }
    }

    template<typename T>
    void Tree_node<T>::predict(const Matrix<T> &data_x, Matrix<T> &prediction, const T shrinkage,
                               const int *row_inds, int n_rows) {

        if (row_inds == NULL) {
            int data_height = data_x.get_height();
            #if _DEBUG_BASE_LEARNER
                assert(data_height == prediction.get_height() && prediction.get_width() == 1);
            #endif
            T predicted_value;
            for (int i = 0; i < data_height; ++i) {
                predicted_value = prediction.get(i, 0) + shrinkage * predict_for_row(data_x, i);
                prediction.assign(i, 0, predicted_value);
            }
        } else {
            #if _DEBUG_BASE_LEARNER
                assert(n_rows > 0 && prediction.get_height() == 1);
            #endif
            T predicted_value;
            for (int i = 0; i < n_rows; ++i) {
                predicted_value = prediction.get(row_inds[i], 0) + shrinkage * predict_for_row(data_x, row_inds[i]);
                prediction.assign(row_inds[i], 0, predicted_value);
            }
        }

    }
}


