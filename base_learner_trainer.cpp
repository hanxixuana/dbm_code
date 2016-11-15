//
// Created by xixuan on 10/10/16.
//

#include "base_learner_trainer.h"

#include <cassert>
#include <limits>
#include <iostream>
#include <cmath>

namespace dbm {

    template
    class Mean_trainer<double>;

    template
    class Mean_trainer<float>;

    template
    class Linear_regression_trainer<double>;

    template
    class Linear_regression_trainer<float>;

    template
    class Tree_trainer<double>;

    template
    class Tree_trainer<float>;

}

namespace dbm {

    template <typename T>
    Mean_trainer<T>::Mean_trainer(const Params &params) :
            loss_function(Loss_function<T>(params)),
            display_training_progress(params.display_training_progress) {};

    template <typename T>
    Mean_trainer<T>::~Mean_trainer() {}

    template <typename T>
    void Mean_trainer<T>::train(Global_mean<T> *mean, const Matrix<T> &train_x,
                                const Matrix<T> &ind_delta, const Matrix<T> &prediction,
                                char loss_function_type,
                                const int *row_inds, int n_rows) {
        if(display_training_progress)
            std::cout << "Training Global Mean at " << mean
                      << " number of samples: " << (n_rows != 0 ? std::to_string(n_rows) : "all")
                      << " ... " << std::endl;

        if(row_inds == nullptr) {
            mean->mean = loss_function.estimate_mean(ind_delta, prediction, loss_function_type);
        }
        else {
            #if _DEBUG_BASE_LEARNER_TRAINER
                assert(n_rows > 0);
            #endif
            mean->mean = loss_function.estimate_mean(ind_delta, prediction, loss_function_type, row_inds, n_rows);
        }

    }

}

namespace dbm {

    template <typename T>
    Linear_regression_trainer<T>::Linear_regression_trainer(const Params &params) :
            display_training_progress(params.display_training_progress) {}

    template <typename T>
    Linear_regression_trainer<T>::~Linear_regression_trainer() {}

    template <typename T>
    void Linear_regression_trainer<T>::train(Linear_regression<T> *linear_regression, const Matrix<T> &train_x,
                                             const Matrix<T> &ind_delta,
                                             const int *row_inds, int n_rows,
                                             const int *col_inds, int n_cols) {

        if(row_inds == nullptr) {
            int height = train_x.get_height(), width = train_x.get_width();
            Matrix<T> w(1, height, 0);
            for(int i = 0; i < height; ++i)
                w.assign(0, i, ind_delta.get(i, 1));
            #if _DEBUG_BASE_LEARNER_TRAINER
                assert(train_x.get_width() == linear_regression->n_predictor);
            #endif
            for(int j = 0; j < width; ++j)
                linear_regression->col_inds[j] = j;

            Matrix<T> intercept(height, 1, 1);
            Matrix<T> x = hori_merge(intercept, train_x);

            Matrix<T> left_in_left = transpose(x);
            left_in_left.inplace_elewise_prod_mat_with_row_vec(w);
            Matrix<T> left = inner_product(left_in_left, x);

            Matrix<T> y = ind_delta.col(0);

            Matrix<T> left_inversed = inverse(left);

            Matrix<T> right = inner_product(left_in_left, y);

            Matrix<T> coefs = inner_product(left_inversed, right);

            #if _DEBUG_BASE_LEARNER_TRAINER
                assert(coefs.get_width() == 1 && coefs.get_height() == width + 1);
            #endif
            linear_regression->intercept = coefs.get(0, 0);
            for(int j = 1; j < width + 1; ++j)
                linear_regression->coefs_no_intercept[j - 1] = coefs.get(j, 0);
        }
        else {
            Matrix<T> w(1, n_rows, 0);
            for(int i = 0; i < n_rows; ++i)
                w.assign(0, i, ind_delta.get(row_inds[i], 1));
            #if _DEBUG_BASE_LEARNER_TRAINER
                assert(n_rows > 0 && n_cols == linear_regression->n_predictor);
            #endif
            for(int j = 0; j < n_cols; ++j)
                linear_regression->col_inds[j] = col_inds[j];

            Matrix<T> intercept(n_rows, 1, 1);
            Matrix<T> x = hori_merge(intercept, train_x.submatrix(row_inds, n_rows, col_inds, n_cols));

            Matrix<T> left_in_left = transpose(x);
            left_in_left.inplace_elewise_prod_mat_with_row_vec(w);
            Matrix<T> left = inner_product(left_in_left, x);

            int y_ind[] = {0}, n_y_ind = 1;
            Matrix<T> y = ind_delta.submatrix(row_inds, n_rows, y_ind, n_y_ind);

            Matrix<T> left_inversed = inverse(left);

            Matrix<T> right = inner_product(left_in_left, y);

            Matrix<T> coefs = inner_product(left_inversed, right);

            #if _DEBUG_BASE_LEARNER_TRAINER
                assert(coefs.get_width() == 1 && coefs.get_height() == n_cols + 1);
            #endif
            linear_regression->intercept = coefs.get(0, 0);
            for(int j = 1; j < n_cols + 1; ++j)
                linear_regression->coefs_no_intercept[j - 1] = coefs.get(j, 0);
        }
    }

}

namespace dbm {

    template<typename T>
    Tree_trainer<T>::Tree_trainer(const Params &params) :
            max_depth(params.max_depth), no_candidate_split_point(params.no_candidate_split_point),
            display_training_progress(params.display_training_progress),
            loss_function(Loss_function<T>(params)) {};

    template<typename T>
    Tree_trainer<T>::~Tree_trainer() {};

    template<typename T>
    void Tree_trainer<T>::train(Tree_node<T> *tree, const Matrix<T> &train_x, const Matrix<T> &train_y,
                                const Matrix<T> &ind_delta, const Matrix<T> &prediction,
                                const int *monotonic_constraints, char loss_function_type,
                                const int *row_inds, int n_rows,
                                const int *col_inds, int n_cols) {
        /*
         * default case has not been implemented
        */

        tree->no_training_samples = n_rows;

        #if _DEBUG_BASE_LEARNER_TRAINER
            assert(n_rows > 0 && n_cols > 0);
        #endif

        tree->prediction = loss_function.estimate_mean(ind_delta, prediction, loss_function_type, row_inds, n_rows);
        if (tree->depth == max_depth || n_rows < no_candidate_split_point * 4) {
            tree->last_node = true;
            return;
        }

        tree->loss = std::numeric_limits<T>::max();

        int larger_inds[n_rows], smaller_inds[n_rows];
        int larger_smaller_n[2] = {0, 0};
        T larger_beta, smaller_beta, loss = tree->loss;
        T uniques[n_rows];

        for (int i = 0; i < n_cols; ++i) {

            int no_uniques = train_x.unique_vals_col(col_inds[i], uniques, row_inds, n_rows);
            no_uniques = middles(uniques, no_uniques);
            shuffle(uniques, no_uniques);
            no_uniques = std::min(no_uniques, no_candidate_split_point);

            for (int j = 0; j < no_uniques; ++j) {
                train_x.inds_split(col_inds[i], uniques[j], larger_inds,
                                   smaller_inds, larger_smaller_n, row_inds, n_rows);

                larger_beta = loss_function.estimate_mean(ind_delta, prediction, loss_function_type,
                                                          larger_inds, larger_smaller_n[0]);
                smaller_beta = loss_function.estimate_mean(ind_delta, prediction, loss_function_type,
                                                           smaller_inds, larger_smaller_n[1]);

                if ( (larger_beta - smaller_beta) * monotonic_constraints[col_inds[j]] < 0 )
                    continue;

                loss = loss_function.loss(train_y, prediction, loss_function_type, larger_beta,
                                          larger_inds, larger_smaller_n[0]) +
                       loss_function.loss(train_y, prediction, loss_function_type, smaller_beta,
                                          smaller_inds, larger_smaller_n[1]);

                if (loss < tree->loss) {
                    tree->loss = loss;
                    tree->column = col_inds[i];
                    tree->split_value = uniques[j];
                }

            }

        }

        if(tree->loss < std::numeric_limits<T>::max()) {
            train_x.inds_split(tree->column, tree->split_value, larger_inds,
                               smaller_inds, larger_smaller_n, row_inds, n_rows);

            if (tree->larger != nullptr)
                delete tree->larger;
            if (tree->smaller != nullptr)
                delete tree->smaller;

            tree->larger = new Tree_node<T>(tree->depth + 1);
            tree->smaller = new Tree_node<T>(tree->depth + 1);

            train(tree->larger, train_x, train_y, ind_delta, prediction, monotonic_constraints, loss_function_type,
                  larger_inds, larger_smaller_n[0], col_inds, n_cols);
            train(tree->smaller, train_x, train_y, ind_delta, prediction, monotonic_constraints, loss_function_type,
                  smaller_inds, larger_smaller_n[1], col_inds, n_cols);
        }
        else {
            tree->last_node = true;
            return;
        }
    }

    template<typename T>
    void Tree_trainer<T>::prune(Tree_node<T> *tree) {

        if (tree->last_node) return;
        #if _DEBUG_BASE_LEARNER_TRAINER
            assert(tree->larger != NULL && tree->smaller != NULL);
        #endif
        if (tree->larger->loss > tree->smaller->loss) {
            tree->larger->last_node = true;

            delete_tree(tree->larger->larger);
            delete_tree(tree->larger->smaller);

            tree->larger->larger = nullptr, tree->larger->smaller = nullptr;

            prune(tree->smaller);
        } else {
            tree->smaller->last_node = true;

            delete_tree(tree->smaller->larger);
            delete_tree(tree->smaller->smaller);

            tree->smaller->larger = nullptr, tree->smaller->smaller = nullptr;

            prune(tree->larger);
        }
    }

}



