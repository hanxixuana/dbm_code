//
// Created by xixuan on 10/10/16.
//

#include "base_learner.h"

#include <cassert>
#include <limits>

namespace dbm {

    template
    class Tree_node<float>;

    template
    class Tree_node<double>;

}

namespace dbm {

    template<typename T>
    Tree_node<T>::Tree_node(int depth) : larger(nullptr), smaller(nullptr), column(-1),
                                         split_value(0), loss(std::numeric_limits<T>::max()),
                                         depth(depth), last_node(false), prediction(0),
                                         no_training_samples(0), Base_learner<T>('t') {};

    template<typename T>
    Tree_node<T>::Tree_node(int depth, int column, bool last_node,
                            T split_value, T loss, T prediction,
                            int no_tr_samples) : larger(nullptr), smaller(nullptr), column(column),
                                                 split_value(split_value), loss(loss),
                                                 depth(depth), last_node(last_node), prediction(prediction),
                                                 no_training_samples(no_tr_samples), Base_learner<T>('t') {};

    template<typename T>
    Tree_node<T>::~Tree_node() {

        if (this == nullptr) return;

        delete larger;
        larger = nullptr;

        delete smaller;
        smaller = nullptr;

    };

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
    void Tree_node<T>::predict(const Matrix<T> &data_x, Matrix<T> &prediction, const int *row_inds, int n_rows) {

        if (row_inds == NULL) {
            int data_height = data_x.get_height(), data_width = data_x.get_width();
#if _DEBUG_BASE_LEARNER
            assert(data_height == prediction.get_height() && prediction.get_width() == 1);
#endif
            T predicted_value;
            for (int i = 0; i < data_height; ++i) {
                predicted_value = prediction.get(i, 0) + predict_for_row(data_x, i);
                prediction.assign(i, 0, predicted_value);
            }
        } else {
#if _DEBUG_BASE_LEARNER
            assert(n_rows > 0);
#endif
            int data_width = data_x.get_width();
#if _DEBUG_BASE_LEARNER
            assert(prediction.get_height() == 1);
#endif
            T predicted_value;
            for (int i = 0; i < n_rows; ++i) {
                predicted_value = prediction.get(row_inds[i], 0) + predict_for_row(data_x, row_inds[i]);
                prediction.assign(row_inds[i], 0, predicted_value);
            }
        }

    }
}


