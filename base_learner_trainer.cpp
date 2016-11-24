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
    class Neural_network_trainer<float>;

    template
    class Neural_network_trainer<double>;

    template
    class Linear_regression_trainer<double>;

    template
    class Linear_regression_trainer<float>;

    template
    class Splines_trainer<double>;

    template
    class Splines_trainer<float>;

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
    void Mean_trainer<T>::train(Global_mean<T> *mean,
                                const Matrix<T> &train_x,
                                const Matrix<T> &ind_delta,
                                const Matrix<T> &prediction,
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
    Neural_network_trainer<T>::Neural_network_trainer(const Params &params) :
            display_training_progress(params.display_training_progress),
            step_size(params.step_size),
            batch_size(params.batch_size),
            max_iteration(params.max_iteration),
            validate_portion(params.validate_portion),
            shrinkage(params.shrinkage),
            n_rise_of_loss_on_validate(params.n_rise_of_loss_on_validate),
            loss_function(Loss_function<T>(params)) {

        input_delta = new Matrix<T>(params.n_hidden_neuron, params.no_candidate_feature + 1, 0);
        hidden_delta = new Matrix<T>(1, params.n_hidden_neuron + 1, 0);

    }

    template <typename T>
    Neural_network_trainer<T>::~Neural_network_trainer() {

        delete input_delta, hidden_delta;
        input_delta = nullptr, hidden_delta = nullptr;

    }

    template <typename T>
    inline T Neural_network_trainer<T>::activation_derivative(const T &input) {
        return input * (1 - input);
    }

    template <typename T>
    void Neural_network_trainer<T>::backward(Neural_network<T> *neural_network,
                                             T ind_delta,
                                             T weight) {
        T temp = step_size * weight * (neural_network->output_output - ind_delta);

        for(int i = 0; i < neural_network->n_hidden_neuron + 1; ++i)
            hidden_delta->assign(0, i,
                                 hidden_delta->get(0, i) - temp * neural_network->hidden_output->get(i, 0));

        for(int i = 0; i < neural_network->n_hidden_neuron; ++i)
            for(int j = 0; j < neural_network->n_predictor + 1; ++j)
                input_delta->assign(i, j,
                                    input_delta->get(i, j) -
                                            temp * neural_network->hidden_weight->get(0, i) *
                                                    activation_derivative(neural_network->hidden_output->get(i, 0))*
                                                    neural_network->input_output->get(j, 0));
    }

    template <typename T>
    void Neural_network_trainer<T>::train(Neural_network<T> *neural_network,
                                          const Matrix<T> &train_x,
                                          const Matrix<T> &ind_delta,
                                          const int *row_inds, int n_rows,
                                          const int *col_inds, int n_cols) {

        if(row_inds == nullptr) {

            int data_height = train_x.get_height(), data_width = train_x.get_width();

            for(int i = 0; i < data_width; ++i)
                neural_network->col_inds[i] = i;

            int n_validate = int(data_height * validate_portion), n_train = data_height - n_validate;
            int *validate_row_inds = new int[n_validate],
                    *train_row_inds = new int[n_train];
            unsigned int *seeds = new unsigned int[max_iteration];
            for(int i = 0; i < max_iteration; ++i)
                seeds[i] = (unsigned int)(std::rand() / (RAND_MAX / 1e5));
            T weight_sum_in_batch, validate_weight_sum = 0;

            for(int i = 0; i < n_validate; ++i) {
                validate_row_inds[i] = i;
                validate_weight_sum += ind_delta.get(i, 1);
            }
            for(int i = n_validate; i < data_height; ++i)
                train_row_inds[i] = i;

            int n_batch = n_train / batch_size, row_index, count_to_break = 0;
            T mse, last_mse = std::numeric_limits<T>::max();

            for(int i = 0; i < max_iteration; ++i) {
                shuffle(train_row_inds, n_train, seeds[i]);
                for(int j = 0; j < n_batch / 3; ++j) {
                    input_delta->clear();
                    hidden_delta->clear();
                    weight_sum_in_batch = 0;
                    for(int k = 0; k < batch_size; ++k) {
                        weight_sum_in_batch += ind_delta.get(train_row_inds[batch_size * j + k], 1);
                    }
                    for(int k = 0; k < batch_size; ++k) {
                        row_index = train_row_inds[batch_size * j + k];
                        for(int l = 0; l < neural_network->n_predictor; ++l)
                            neural_network->input_output->assign(l, 0,
                                                                 train_x.get(row_index, neural_network->col_inds[l]));
                        neural_network->input_output->assign(neural_network->n_predictor, 0, 1);
                        neural_network->forward();
                        backward(neural_network, ind_delta.get(row_index, 0),
                                 ind_delta.get(row_index, 1) / weight_sum_in_batch);
                    }
                    Matrix<T> temp_input_delta = plus(*neural_network->input_weight, *input_delta);
                    Matrix<T> temp_hidden_delta = plus(*neural_network->hidden_weight, *hidden_delta);
                    copy(temp_input_delta, *neural_network->input_weight);
                    copy(temp_hidden_delta, *neural_network->hidden_weight);
                }

                mse = 0;
                for(int j = 0; j < n_validate; ++j) {
                    for(int k = 0; k < neural_network->n_predictor; ++k)
                        neural_network->input_output->assign(k, 0, train_x.get(validate_row_inds[j],
                                                                               neural_network->col_inds[k]));
                    neural_network->input_output->assign(neural_network->n_predictor, 0, 1);
                    neural_network->forward();
                    mse += std::pow(ind_delta.get(validate_row_inds[j], 0) - neural_network->output_output, 2.0) *
                           ind_delta.get(validate_row_inds[j], 1) / validate_weight_sum;
                }

                if(mse > last_mse) {
                    count_to_break += 1;
                    if(count_to_break > n_rise_of_loss_on_validate) {
                        std::cout << "Training of " << neural_network << " was early stoped!" << std::endl;
                        break;
                    }
                }
            }

            delete validate_row_inds, train_row_inds, seeds;
            validate_row_inds = nullptr, train_row_inds = nullptr, seeds = nullptr;

        }
        else {

            #if _DEBUG_BASE_LEARNER_TRAINER
            assert(n_rows > 0 && neural_network->n_predictor == n_cols);
            #endif

            for(int i = 0; i < n_cols; ++i)
                neural_network->col_inds[i] = col_inds[i];

            int n_validate = int(n_rows * validate_portion), n_train = n_rows - n_validate;
            int *validate_row_inds = new int[n_validate],
                    *train_row_inds = new int[n_train];
            unsigned int *seeds = new unsigned int[max_iteration];
            for(int i = 0; i < max_iteration; ++i)
                seeds[i] = (unsigned int)(std::rand() / (RAND_MAX / 1e5));
            T weight_sum_in_batch = 0, validate_weight_sum = 0;

            for(int i = 0; i < n_validate; ++i) {
                validate_row_inds[i] = row_inds[i];
                validate_weight_sum += ind_delta.get(validate_row_inds[i], 1);
            }
            for(int i = n_validate; i < n_rows; ++i)
                train_row_inds[i - n_validate] = row_inds[i];

            int n_batch = n_train / batch_size, row_index, count_to_break = 0;
            T mse, first_mse, last_mse = std::numeric_limits<T>::max();

            for(int i = 0; i < max_iteration; ++i) {
                shuffle(train_row_inds, n_train, seeds[i]);
                for(int j = 0; j < n_batch / 3; ++j) {
                    input_delta->clear();
                    hidden_delta->clear();
                    weight_sum_in_batch = 0;
                    for(int k = 0; k < batch_size; ++k) {
                        weight_sum_in_batch += ind_delta.get(train_row_inds[batch_size * j + k], 1);
                    }
                    for(int k = 0; k < batch_size; ++k) {
                        row_index = train_row_inds[batch_size * j + k];
                        for(int l = 0; l < neural_network->n_predictor; ++l)
                            neural_network->input_output->assign(l, 0,
                                                                 train_x.get(row_index, neural_network->col_inds[l]));
                        neural_network->input_output->assign(neural_network->n_predictor, 0, 1);
                        neural_network->forward();
                        backward(neural_network, ind_delta.get(row_index, 0),
                                 ind_delta.get(row_index, 1) / weight_sum_in_batch);
                    }
                    Matrix<T> temp_input_delta = plus(*neural_network->input_weight, *input_delta);
                    Matrix<T> temp_hidden_delta = plus(*neural_network->hidden_weight, *hidden_delta);
                    copy(temp_input_delta, *neural_network->input_weight);
                    copy(temp_hidden_delta, *neural_network->hidden_weight);
                }

                mse = 0;
                for(int j = 0; j < n_validate; ++j) {
                    for(int k = 0; k < neural_network->n_predictor; ++k)
                        neural_network->input_output->assign(k, 0, train_x.get(validate_row_inds[j],
                                                                               neural_network->col_inds[k]));
                    neural_network->input_output->assign(neural_network->n_predictor, 0, 1);
                    neural_network->forward();
                    mse += std::pow(ind_delta.get(validate_row_inds[j], 0) - neural_network->output_output, 2.0) *
                           ind_delta.get(validate_row_inds[j], 1) / validate_weight_sum;
                }

//                std::cout << "( " << i << " ) MSE: "<< mse << std::endl;

                if(i == 0)
                    first_mse = mse;

                if(mse > last_mse) {
                    count_to_break += 1;
                    if(count_to_break > n_rise_of_loss_on_validate) {
                        std::cout << "Training of Neural Network at " << neural_network
                                  << " was early stoped after "
                                  << i << "/" << max_iteration
                                  << " iterations with starting MSE: " << first_mse
                                  << " and ending MSE: " << mse
                                  << " !" << std::endl;
                        break;
                    }
                }
                last_mse = mse;
            }

            delete validate_row_inds, train_row_inds, seeds;
            validate_row_inds = nullptr, train_row_inds = nullptr, seeds = nullptr;

        }

    }

}

namespace dbm {

    template <typename T>
    Splines_trainer<T>::Splines_trainer(const Params &params) :
            display_training_progress(params.display_training_progress) {

        n_pairs = params.no_candidate_feature * (params.no_candidate_feature - 1) / 2;

        predictor_pairs_inds = new int *[n_pairs];
        for(int i = 0; i < n_pairs; ++i) {
            predictor_pairs_inds[i] = new int[2];
        }

        for(int i = 0, count = 0; i < params.no_candidate_feature - 1; ++i)
            for(int j = i + 1; j < params.no_candidate_feature; ++j) {
                predictor_pairs_inds[count][0] = i;
                predictor_pairs_inds[count][1] = j;
                count += 1;
            }

    }

    template <typename T>
    Splines_trainer<T>::~Splines_trainer() {

        for(int i = 0; i < n_pairs; ++i) {
            delete predictor_pairs_inds[i];
        }
        delete[] predictor_pairs_inds;

    }

    template <typename T>
    void Splines_trainer<T>::train(Splines<T> *splines,
                                   const Matrix<T> &train_x,
                                   const Matrix<T> &ind_delta,
                                   const int *row_inds, int n_rows,
                                   const int *col_inds, int n_cols) {

        if(row_inds == nullptr) {



        }
        else {
            Matrix<T> w(1, n_rows, 0);
            for(int i = 0; i < n_rows; ++i)
                w.assign(0, i, ind_delta.get(row_inds[i], 1));

            #if _DEBUG_BASE_LEARNER_TRAINER
                assert(n_rows > 0 && n_cols >= splines->n_predictor);
            #endif

            T scaling = 0.7;
            T **knots, min, max;
            knots = new T*[n_cols];
            for(int i = 0; i < n_cols; ++i) {
                knots[i] = new T[splines->n_knot];
                min = train_x.get_col_min(col_inds[i], row_inds, n_rows);
                max = train_x.get_col_max(col_inds[i], row_inds, n_rows);
                range(min, max, splines->n_knot, knots[i], scaling);
            }

            T mse, lowest_mse = std::numeric_limits<T>::max();
            T knot;

            int lowest_mse_pair_ind = 0;

            T x, y, prediction;

            int n_splines = splines->n_knot * 4;

            Matrix<T> lowest_mse_coefs(n_splines, 1, 0);
            Matrix<T> design_matrix(n_rows, n_splines, 0);
            Matrix<T> left_in_left(n_splines, n_rows, 0);
            Matrix<T> eye(n_splines, n_splines, 0);

            for(int i = 0; i < n_splines; ++i)
                eye.assign(i, i, 0.00001);

            for(int i = 0; i < n_pairs / 3; ++i) {

                splines->col_inds[0] = col_inds[ predictor_pairs_inds[i][0] ];
                splines->col_inds[1] = col_inds[ predictor_pairs_inds[i][1] ];

                design_matrix.clear();
                for(int j = 0; j < n_rows; ++j) {

                    x = train_x.get(row_inds[j], splines->col_inds[0]);
                    y = train_x.get(row_inds[j], splines->col_inds[1]);

                    for(int k = 0; k < splines->n_knot; ++k) {

                        knot = knots[ predictor_pairs_inds[i][0] ][k];
                        design_matrix.assign(j, 4 * k, splines->x_left_hinge(x, y, knot) );
                        design_matrix.assign(j, 4 * k + 1, splines->x_right_hinge(x, y, knot) );

                        knot = knots[ predictor_pairs_inds[i][1] ][k];
                        design_matrix.assign(j, 4 * k + 2, splines->y_left_hinge(x, y, knot) );
                        design_matrix.assign(j, 4 * k + 3, splines->y_right_hinge(x, y, knot) );

                    }
                }

                copy(transpose(design_matrix), left_in_left);
                left_in_left.inplace_elewise_prod_mat_with_row_vec(w);
                Matrix<T> left = inner_product(left_in_left, design_matrix);
                Matrix<T> left_inversed = inverse(plus(left, eye));

                int y_ind[] = {0}, n_y_ind = 1;
                Matrix<T> Y = ind_delta.submatrix(row_inds, n_rows, y_ind, n_y_ind);
                Matrix<T> right = inner_product(left_in_left, Y);

                Matrix<T> coefs = inner_product(left_inversed, right);


                // check this part, should I use predict_for_row?
                mse = 0;
                for(int j = 0; j < n_rows; ++j) {

                    prediction = 0;

                    x = train_x.get(row_inds[j], splines->col_inds[0]);
                    y = train_x.get(row_inds[j], splines->col_inds[1]);

                    for(int k = 0; k < splines->n_knot; ++k) {

                        knot = knots[ predictor_pairs_inds[i][0] ][k];
                        prediction += splines->x_left_hinge(x, y, knot) * coefs.get(4 * k, 0);
                        prediction += splines->x_right_hinge(x, y, knot) * coefs.get(4 * k + 1, 0);

                        knot = knots[ predictor_pairs_inds[i][1] ][k];
                        prediction += splines->y_left_hinge(x, y, knot) * coefs.get(4 * k + 2, 0);
                        prediction += splines->y_right_hinge(x, y, knot) * coefs.get(4 * k + 3, 0);

                    }
                    mse += std::pow(ind_delta.get(row_inds[j], 0) - prediction, 2.0);
                }

                mse /= n_rows;

                if(mse < lowest_mse) {
                    lowest_mse = mse;
                    lowest_mse_pair_ind = i;
                    copy(coefs, lowest_mse_coefs);
                }

            }

            splines->col_inds[0] = col_inds[ predictor_pairs_inds[lowest_mse_pair_ind][0] ];
            splines->col_inds[1] = col_inds[ predictor_pairs_inds[lowest_mse_pair_ind][1] ];

            for(int j = 0; j < splines->n_knot; ++j) {

                splines->x_knots[j] = knots[ predictor_pairs_inds[lowest_mse_pair_ind][0] ][j];

                splines->y_knots[j] = knots[ predictor_pairs_inds[lowest_mse_pair_ind][1] ][j];

            }

            for(int j = 0; j < splines->n_knot; ++j) {

                splines->x_left_coefs[j] = lowest_mse_coefs.get(4 * j, 0);
                splines->x_right_coefs[j] = lowest_mse_coefs.get(4 * j + 1, 0);

                splines->y_left_coefs[j] = lowest_mse_coefs.get(4 * j + 2, 0);
                splines->y_right_coefs[j] = lowest_mse_coefs.get(4 * j + 3, 0);

            }


            for(int i = 0; i < n_cols; ++i)
                delete[] knots[i];
            delete[] knots;
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
    void Linear_regression_trainer<T>::train(Linear_regression<T> *linear_regression,
                                             const Matrix<T> &train_x,
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
    void Tree_trainer<T>::train(Tree_node<T> *tree,
                                const Matrix<T> &train_x,
                                const Matrix<T> &train_y,
                                const Matrix<T> &ind_delta,
                                const Matrix<T> &prediction,
                                const int *monotonic_constraints,
                                char loss_function_type,
                                const int *row_inds, int n_rows,
                                const int *col_inds, int n_cols) {

        if(row_inds == nullptr) {

            int data_height = train_x.get_height(), data_width = train_x.get_width();

            tree->no_training_samples = data_height;

            tree->prediction = loss_function.estimate_mean(ind_delta, prediction, loss_function_type);
            if (tree->depth == max_depth || n_rows < no_candidate_split_point * 4) {
                tree->last_node = true;
                return;
            }

            tree->loss = std::numeric_limits<T>::max();

            int *larger_inds = new int[data_height], *smaller_inds = new int[data_height];
            int larger_smaller_n[2] = {0, 0};
            T larger_beta, smaller_beta, loss = tree->loss;
            T *uniques = new T[data_height];
            for(int i = 0; i < data_height; ++i)
                uniques[i] = 0;

            for (int i = 0; i < data_width; ++i) {

                int no_uniques = train_x.unique_vals_col(i, uniques);
                no_uniques = middles(uniques, no_uniques);
                shuffle(uniques, no_uniques);
                no_uniques = std::min(no_uniques, no_candidate_split_point);

                for (int j = 0; j < no_uniques; ++j) {
                    train_x.inds_split(i, uniques[j], larger_inds,
                                       smaller_inds, larger_smaller_n);

                    larger_beta = loss_function.estimate_mean(ind_delta, prediction, loss_function_type,
                                                              larger_inds, larger_smaller_n[0]);
                    smaller_beta = loss_function.estimate_mean(ind_delta, prediction, loss_function_type,
                                                               smaller_inds, larger_smaller_n[1]);

                    if ( (larger_beta - smaller_beta) * monotonic_constraints[i] < 0 )
                        continue;

                    loss = loss_function.loss(train_y, prediction, loss_function_type, larger_beta,
                                              larger_inds, larger_smaller_n[0]) +
                           loss_function.loss(train_y, prediction, loss_function_type, smaller_beta,
                                              smaller_inds, larger_smaller_n[1]);

                    if (loss < tree->loss) {
                        tree->loss = loss;
                        tree->column = i;
                        tree->split_value = uniques[j];
                    }

                }

            }

            if(tree->loss < std::numeric_limits<T>::max()) {
                train_x.inds_split(tree->column, tree->split_value, larger_inds,
                                   smaller_inds, larger_smaller_n);

                if (tree->larger != nullptr)
                    delete tree->larger;
                if (tree->smaller != nullptr)
                    delete tree->smaller;

                tree->larger = new Tree_node<T>(tree->depth + 1);
                tree->smaller = new Tree_node<T>(tree->depth + 1);

                train(tree->larger, train_x, train_y, ind_delta, prediction, monotonic_constraints, loss_function_type,
                      larger_inds, larger_smaller_n[0]);
                train(tree->smaller, train_x, train_y, ind_delta, prediction, monotonic_constraints, loss_function_type,
                      smaller_inds, larger_smaller_n[1]);
                delete larger_inds, smaller_inds, uniques;
                larger_inds = nullptr, smaller_inds = nullptr, uniques = nullptr;
            }
            else {
                tree->last_node = true;
                delete larger_inds, smaller_inds, uniques;
                larger_inds = nullptr, smaller_inds = nullptr, uniques = nullptr;
            }

        }
        else {

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

            int *larger_inds = new int[n_rows], *smaller_inds = new int[n_rows];
            int larger_smaller_n[2] = {0, 0};
            T larger_beta, smaller_beta, loss = tree->loss;
            T *uniques = new T[n_rows];
            for(int i = 0; i < n_rows; ++i)
                uniques[i] = 0;

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

                    if ( (larger_beta - smaller_beta) * monotonic_constraints[col_inds[i]] < 0 )
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
                delete larger_inds, smaller_inds, uniques;
                larger_inds = nullptr, smaller_inds = nullptr, uniques = nullptr;
            }
            else {
                tree->last_node = true;
                delete larger_inds, smaller_inds, uniques;
                larger_inds = nullptr, smaller_inds = nullptr, uniques = nullptr;
            }

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



