//
// Created by xixuan on 10/10/16.
//

#include "base_learner_trainer.h"

#include <cassert>
#include <limits>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>

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
    class Kmeans2d_trainer<double>;

    template
    class Kmeans2d_trainer<float>;

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
            display_training_progress(params.dbm_display_training_progress),
            loss_function(Loss_function<T>(params)) {};

    template <typename T>
    Mean_trainer<T>::~Mean_trainer() {}

    template <typename T>
    void Mean_trainer<T>::train(Global_mean<T> *mean,
                                const Matrix<T> &train_x,
                                const Matrix<T> &ind_delta,
                                const Matrix<T> &prediction,
                                char loss_function_type,
                                const int *row_inds,
                                int no_rows) {
        if(display_training_progress)
            std::cout << "Training Global Mean at "
                      << mean
                      << " number of samples: "
                      << (no_rows != 0 ? std::to_string(no_rows) : "all")
                      << " ... "
                      << std::endl;

        if(row_inds == nullptr) {
            mean->mean = loss_function.estimate_mean(ind_delta,
                                                     loss_function_type);

        }
        else {
            #ifdef _DEBUG_BASE_LEARNER_TRAINER
                assert(no_rows > 0);
            #endif
            mean->mean = loss_function.estimate_mean(ind_delta,
                                                     loss_function_type,
                                                     row_inds,
                                                     no_rows);
        }

    }

}

namespace dbm {

    template <typename T>
    Neural_network_trainer<T>::Neural_network_trainer(const Params &params) :
            batch_size(params.nn_batch_size),
            nn_max_iteration(params.nn_max_iteration),
            step_size(params.nn_step_size),
            validate_portion(params.nn_validate_portion),
            no_rise_of_loss_on_validate(params.nn_no_rise_of_loss_on_validate),
            loss_function(Loss_function<T>(params)) {}

    template <typename T>
    Neural_network_trainer<T>::~Neural_network_trainer() {}

    template <typename T>
    inline T Neural_network_trainer<T>::activation_derivative(const T &input) {
        return input * (1 - input);
    }

    template <typename T>
    void Neural_network_trainer<T>::backward(Neural_network<T> *neural_network,
                                             const Matrix<T> &input_output,
                                             Matrix<T> &hidden_output,
                                             T &output_output,
                                             Matrix<T> &hidden_delta,
                                             Matrix<T> &input_delta,
                                             T ind_delta,
                                             T weight) {
        T temp = step_size * weight * (output_output - ind_delta);

        for(int i = 0; i < neural_network->no_hidden_neurons + 1; ++i)
            hidden_delta.assign(0, i,
                                hidden_delta.get(0, i) - temp * hidden_output.get(i, 0));

        for(int i = 0; i < neural_network->no_hidden_neurons; ++i)
            for(int j = 0; j < neural_network->no_predictors + 1; ++j)
                input_delta.assign(i, j,
                                   input_delta.get(i, j) -
                                           temp * neural_network->hidden_weight->get(0, i) *
                                                   activation_derivative(hidden_output.get(i, 0))*
                                                   input_output.get(j, 0));
    }

    template <typename T>
    void Neural_network_trainer<T>::train(Neural_network<T> *neural_network,
                                          const Matrix<T> &train_x,
                                          const Matrix<T> &ind_delta,
                                          const int *row_inds,
                                          int no_rows,
                                          const int *col_inds,
                                          int no_cols) {

        if(row_inds == nullptr) {

            #ifdef _DEBUG_BASE_LEARNER_TRAINER
            assert(neural_network->no_predictors == train_x.get_width());
            #endif

            Matrix<T> input_output(neural_network->no_predictors + 1, 1, 0);
            Matrix<T> hidden_output(neural_network->no_hidden_neurons + 1, 1, 0);
            T output_output;

            // no_hidden_neurons * (no_predictors + 1)
            Matrix<T> input_delta(neural_network->no_hidden_neurons,
                                  neural_network->no_predictors + 1, 0);
            // 1 * (no_hidden_neurons + 1)
            Matrix<T> hidden_delta(1, neural_network->no_predictors + 1, 0);

            int data_height = train_x.get_height(),
                    data_width = train_x.get_width();

            for(int i = 0; i < data_width; ++i)
                neural_network->col_inds[i] = i;

            int no_validate = int(data_height * validate_portion),
                    no_train = data_height - no_validate;
            int *validate_row_inds = new int[no_validate],
                    *train_row_inds = new int[no_train];

            unsigned int *seeds = new unsigned int[nn_max_iteration];
            for(int i = 0; i < nn_max_iteration; ++i)
                seeds[i] = (unsigned int)(std::rand() / (RAND_MAX / 1e5));

            T weight_sum_in_batch,
                    validate_weight_sum = 0;
            for(int i = 0; i < no_validate; ++i) {
                validate_row_inds[i] = i;
                validate_weight_sum += ind_delta.get(i, 1);
            }
            for(int i = no_validate; i < data_height; ++i)
                train_row_inds[i] = i;

            int no_batch = no_train / batch_size,
                    row_index,
                    count_to_break = 0;
            T last_mse = std::numeric_limits<T>::max(),
                    mse;

            for(int i = 0; i < nn_max_iteration; ++i) {

                shuffle(train_row_inds, no_train, seeds[i]);

                for(int j = 0; j < no_batch / 3; ++j) {

                    input_delta.clear();
                    hidden_delta.clear();

                    weight_sum_in_batch = 0;
                    for(int k = 0; k < batch_size; ++k) {
                        weight_sum_in_batch += ind_delta.get(train_row_inds[batch_size * j + k], 1);
                    }

                    for(int k = 0; k < batch_size; ++k) {

                        row_index = train_row_inds[batch_size * j + k];

                        for(int l = 0; l < neural_network->no_predictors; ++l)
                            input_output.assign(l, 0,
                                                train_x.get(row_index, neural_network->col_inds[l]));

                        input_output.assign(neural_network->no_predictors,
                                            0, 1);

                        neural_network->forward(input_output, hidden_output, output_output);
                        backward(neural_network,
                                 input_output,
                                 hidden_output,
                                 output_output,
                                 hidden_delta,
                                 input_delta,
                                 ind_delta.get(row_index, 0),
                                 ind_delta.get(row_index, 1) / weight_sum_in_batch);
                    }

                    Matrix<T> temp_input_delta = plus(*neural_network->input_weight,
                                                      input_delta);
                    Matrix<T> temp_hidden_delta = plus(*neural_network->hidden_weight,
                                                       hidden_delta);

                    copy(temp_input_delta,
                         *neural_network->input_weight);
                    copy(temp_hidden_delta,
                         *neural_network->hidden_weight);
                }

                mse = 0;
                for(int j = 0; j < no_validate; ++j) {

                    for(int k = 0; k < neural_network->no_predictors; ++k)
                        input_output.assign(k, 0,
                                            train_x.get(validate_row_inds[j], neural_network->col_inds[k]));

                    input_output.assign(neural_network->no_predictors,
                                        0, 1);
                    neural_network->forward(input_output, hidden_output, output_output);

                    mse += std::pow(ind_delta.get(validate_row_inds[j], 0) - output_output, 2.0) *
                           ind_delta.get(validate_row_inds[j], 1) / validate_weight_sum;
                }

                if(mse > last_mse) {
                    count_to_break += 1;
                    if(count_to_break > no_rise_of_loss_on_validate) {
//                        std::cout << "Training of "
//                                  << neural_network
//                                  << " was early stoped!"
//                                  << std::endl;
                        break;
                    }
                }
            }

            delete[] validate_row_inds;
            delete[] train_row_inds;
            delete[] seeds;
            validate_row_inds = nullptr, train_row_inds = nullptr, seeds = nullptr;

        }
        else {

            #ifdef _DEBUG_BASE_LEARNER_TRAINER
                assert(no_rows > 0 && neural_network->no_predictors == no_cols);
            #endif

            Matrix<T> input_output(neural_network->no_predictors + 1, 1, 0);
            Matrix<T> hidden_output(neural_network->no_hidden_neurons + 1, 1, 0);
            T output_output;

            // no_hidden_neurons * (no_predictors + 1)
            Matrix<T> input_delta(neural_network->no_hidden_neurons,
                                                   neural_network->no_predictors + 1, 0);
            // 1 * (no_hidden_neurons + 1)
            Matrix<T> hidden_delta(1, neural_network->no_hidden_neurons + 1, 0);

            for(int i = 0; i < no_cols; ++i)
                neural_network->col_inds[i] = col_inds[i];

            int no_validate = int(no_rows * validate_portion),
                    no_train = no_rows - no_validate;
            int *validate_row_inds = new int[no_validate],
                    *train_row_inds = new int[no_train];

            unsigned int *seeds = new unsigned int[nn_max_iteration];
            for(int i = 0; i < nn_max_iteration; ++i)
                seeds[i] = (unsigned int)(std::rand() / (RAND_MAX / 1e5));

            T weight_sum_ino_batch = 0,
                    validate_weight_sum = 0;
            for(int i = 0; i < no_validate; ++i) {
                validate_row_inds[i] = row_inds[i];
                validate_weight_sum += ind_delta.get(validate_row_inds[i], 1);
            }

            for(int i = no_validate; i < no_rows; ++i)
                train_row_inds[i - no_validate] = row_inds[i];

            int no_batch = no_train / batch_size, row_index, count_to_break = 0;
            T last_mse = std::numeric_limits<T>::max(), mse;
//            T first_mse;

            for(int i = 0; i < nn_max_iteration; ++i) {
                shuffle(train_row_inds,
                        no_train,
                        seeds[i]);
                for(int j = 0; j < no_batch / 3; ++j) {

                    input_delta.clear();
                    hidden_delta.clear();

                    weight_sum_ino_batch = 0;
                    for(int k = 0; k < batch_size; ++k) {
                        weight_sum_ino_batch += ind_delta.get(train_row_inds[batch_size * j + k], 1);
                    }

                    for(int k = 0; k < batch_size; ++k) {

                        row_index = train_row_inds[batch_size * j + k];

                        for(int l = 0; l < neural_network->no_predictors; ++l)
                            input_output.assign(l, 0,
                                                 train_x.get(row_index, neural_network->col_inds[l]));

                        input_output.assign(neural_network->no_predictors, 0, 1);

                        neural_network->forward(input_output, hidden_output, output_output);
                        backward(neural_network,
                                 input_output,
                                 hidden_output,
                                 output_output,
                                 hidden_delta,
                                 input_delta,
                                 ind_delta.get(row_index, 0),
                                 ind_delta.get(row_index, 1) / weight_sum_ino_batch);
                    }

                    Matrix<T> temp_input_delta = plus(*neural_network->input_weight,
                                                      input_delta);
                    Matrix<T> temp_hidden_delta = plus(*neural_network->hidden_weight,
                                                       hidden_delta);

                    copy(temp_input_delta,
                         *neural_network->input_weight);
                    copy(temp_hidden_delta,
                         *neural_network->hidden_weight);
                }

                mse = 0;
                for(int j = 0; j < no_validate; ++j) {

                    for(int k = 0; k < neural_network->no_predictors; ++k)
                        input_output.assign(k, 0,
                                            train_x.get(validate_row_inds[j], neural_network->col_inds[k]));

                    input_output.assign(neural_network->no_predictors, 0, 1);
                    neural_network->forward(input_output, hidden_output, output_output);

                    mse += std::pow(ind_delta.get(validate_row_inds[j], 0) - output_output, 2.0) *
                           ind_delta.get(validate_row_inds[j], 1) / validate_weight_sum;

                }

//                std::cout << "( " << i << " ) MSE: "<< mse << std::endl;

//                if(i == 0)
//                    first_mse = mse;

                if(mse > last_mse) {
                    count_to_break += 1;
                    if(count_to_break > no_rise_of_loss_on_validate) {
//                        std::cout << "Training of Neural Network at " << neural_network
//                                  << " was early stoped after " << i << "/" << nn_max_iteration
//                                  << " iterations with starting MSE: " << first_mse
//                                  << " and ending MSE: " << mse
//                                  << " !"
//                                  << std::endl;
                        break;
                    }
                }
                last_mse = mse;
            }

            delete[] validate_row_inds;
            delete[] train_row_inds;
            delete[] seeds;
            validate_row_inds = nullptr, train_row_inds = nullptr, seeds = nullptr;

        }

    }

}

namespace dbm {

    template <typename T>
    Splines_trainer<T>::Splines_trainer(const Params &params)
            : regularization(params.splines_regularization) {

        no_pairs = params.dbm_no_candidate_feature * (params.dbm_no_candidate_feature - 1) / 2;

        predictor_pairs_inds = new int *[no_pairs];
        for(int i = 0; i < no_pairs; ++i) {
            predictor_pairs_inds[i] = new int[2];
        }

        for(int i = 0, count = 0; i < params.dbm_no_candidate_feature - 1; ++i)
            for(int j = i + 1; j < params.dbm_no_candidate_feature; ++j) {
                predictor_pairs_inds[count][0] = i;
                predictor_pairs_inds[count][1] = j;
                count += 1;
            }

    }

    template <typename T>
    Splines_trainer<T>::~Splines_trainer() {

        for(int i = 0; i < no_pairs; ++i) {
            delete[] predictor_pairs_inds[i];
        }
        delete[] predictor_pairs_inds;

    }

    template <typename T>
    void Splines_trainer<T>::train(Splines<T> *splines,
                                   const Matrix<T> &train_x,
                                   const Matrix<T> &ind_delta,
                                   const int *row_inds,
                                   int no_rows,
                                   const int *col_inds,
                                   int no_cols) {

        if(row_inds == nullptr) {

            int data_height = train_x.get_height(),
                    data_width = train_x.get_width();

            Matrix<T> w(1, data_width, 0);
            for(int i = 0; i < data_width; ++i)
                w.assign(0, i, ind_delta.get(i, 1));

            T scaling = 0.7;
            T **knots, min, max;
            knots = new T*[data_width];
            for(int i = 0; i < data_width; ++i) {
                knots[i] = new T[splines->no_knots];
                min = train_x.get_col_min(i);
                max = train_x.get_col_max(i);
                range(min,
                      max,
                      splines->no_knots,
                      knots[i],
                      scaling);
            }

            std::random_shuffle(predictor_pairs_inds,
                                predictor_pairs_inds + no_pairs);

            T mse, lowest_mse = std::numeric_limits<T>::max();
            T knot;

            int lowest_mse_pair_ind = 0;

            T x, y, prediction;

            int no_splines = splines->no_knots * 4;

            Matrix<T> lowest_mse_coefs(no_splines, 1, 0);
            Matrix<T> design_matrix(no_rows, no_splines, 0);
            Matrix<T> left_in_left;
            Matrix<T> left;
            int y_ind[] = {0}, no_y_ind = 1;
            Matrix<T> Y;
            Matrix<T> coefs;

            Matrix<T> eye(no_splines, no_splines, 0);

            for(int i = 0; i < no_splines; ++i)
                eye.assign(i, i, regularization);

            for(int i = 0; i < no_pairs / 3; ++i) {

                splines->col_inds[0] = predictor_pairs_inds[i][0];
                splines->col_inds[1] = predictor_pairs_inds[i][0];

                design_matrix.clear();
                for(int j = 0; j < data_height; ++j) {

                    x = train_x.get(j,
                                    splines->col_inds[0]);
                    y = train_x.get(j,
                                    splines->col_inds[1]);

                    for(int k = 0; k < splines->no_knots; ++k) {

                        knot = knots[ predictor_pairs_inds[i][0] ][k];
                        design_matrix.assign(j,
                                             4 * k,
                                             splines->x_left_hinge(x, y, knot) );
                        design_matrix.assign(j,
                                             4 * k + 1,
                                             splines->x_right_hinge(x, y, knot) );

                        knot = knots[ predictor_pairs_inds[i][1] ][k];
                        design_matrix.assign(j,
                                             4 * k + 2,
                                             splines->y_left_hinge(x, y, knot) );
                        design_matrix.assign(j,
                                             4 * k + 3,
                                             splines->y_right_hinge(x, y, knot) );

                    }
                }

                left_in_left = transpose(design_matrix);
                left_in_left.inplace_elewise_prod_mat_with_row_vec(w);
                left = inner_product(left_in_left, design_matrix);
                left = inverse(plus(left, eye));

                Y = ind_delta.submatrix(row_inds,
                                        no_rows,
                                        y_ind,
                                        no_y_ind);

                coefs = inner_product(left, inner_product(left_in_left, Y));


                // check this part, should I use predict_for_row?
                mse = 0;
                for(int j = 0; j < data_height; ++j) {

                    prediction = 0;

                    x = train_x.get(j, splines->col_inds[0]);
                    y = train_x.get(j, splines->col_inds[1]);

                    for(int k = 0; k < splines->no_knots; ++k) {

                        knot = knots[ predictor_pairs_inds[i][0] ][k];
                        prediction += splines->x_left_hinge(x, y, knot) * coefs.get(4 * k, 0);
                        prediction += splines->x_right_hinge(x, y, knot) * coefs.get(4 * k + 1, 0);

                        knot = knots[ predictor_pairs_inds[i][1] ][k];
                        prediction += splines->y_left_hinge(x, y, knot) * coefs.get(4 * k + 2, 0);
                        prediction += splines->y_right_hinge(x, y, knot) * coefs.get(4 * k + 3, 0);

                    }
                    mse += std::pow(ind_delta.get(j, 0) - prediction, 2.0);
                }

                mse /= no_rows;

                if(mse < lowest_mse) {
                    lowest_mse = mse;
                    lowest_mse_pair_ind = i;
                    copy(coefs, lowest_mse_coefs);
                }

            }

            splines->col_inds[0] = predictor_pairs_inds[lowest_mse_pair_ind][0];
            splines->col_inds[1] = predictor_pairs_inds[lowest_mse_pair_ind][1];

            for(int j = 0; j < splines->no_knots; ++j) {

                splines->x_knots[j] = knots[ predictor_pairs_inds[lowest_mse_pair_ind][0] ][j];

                splines->y_knots[j] = knots[ predictor_pairs_inds[lowest_mse_pair_ind][1] ][j];

            }

            for(int j = 0; j < splines->no_knots; ++j) {

                splines->x_left_coefs[j] = lowest_mse_coefs.get(4 * j, 0);
                splines->x_right_coefs[j] = lowest_mse_coefs.get(4 * j + 1, 0);

                splines->y_left_coefs[j] = lowest_mse_coefs.get(4 * j + 2, 0);
                splines->y_right_coefs[j] = lowest_mse_coefs.get(4 * j + 3, 0);

            }


            for(int i = 0; i < no_cols; ++i)
                delete[] knots[i];
            delete[] knots;

        }
        else {
            Matrix<T> w(1, no_rows, 0);
            for(int i = 0; i < no_rows; ++i)
                w.assign(0, i, ind_delta.get(row_inds[i], 1));

            #ifdef _DEBUG_BASE_LEARNER_TRAINER
                assert(no_rows > 0 && no_cols >= splines->no_predictors);
            #endif

            T scaling = 0.7;
            T **knots, min, max;
            knots = new T*[no_cols];
            for(int i = 0; i < no_cols; ++i) {
                knots[i] = new T[splines->no_knots];
                min = train_x.get_col_min(col_inds[i], row_inds, no_rows);
                max = train_x.get_col_max(col_inds[i], row_inds, no_rows);
                range(min, max, splines->no_knots, knots[i], scaling);
            }

            T mse, lowest_mse = std::numeric_limits<T>::max();
            T knot;

            int lowest_mse_pair_ind = 0;

            T x, y, prediction;

            int no_splines = splines->no_knots * 4;

            Matrix<T> lowest_mse_coefs(no_splines, 1, 0);
            Matrix<T> design_matrix(no_rows, no_splines, 0);
            Matrix<T> left_in_left;
            Matrix<T> left;
            int y_ind[] = {0}, no_y_ind = 1;
            Matrix<T> Y;
            Matrix<T> coefs;

            Matrix<T> eye(no_splines, no_splines, 0);

            for(int i = 0; i < no_splines; ++i)
                eye.assign(i, i, regularization);

            for(int i = 0; i < no_pairs / 3; ++i) {

                splines->col_inds[0] = col_inds[ predictor_pairs_inds[i][0] ];
                splines->col_inds[1] = col_inds[ predictor_pairs_inds[i][1] ];

                design_matrix.clear();
                for(int j = 0; j < no_rows; ++j) {

                    x = train_x.get(row_inds[j],
                                    splines->col_inds[0]);
                    y = train_x.get(row_inds[j],
                                    splines->col_inds[1]);

                    for(int k = 0; k < splines->no_knots; ++k) {

                        knot = knots[ predictor_pairs_inds[i][0] ][k];
                        design_matrix.assign(j,
                                             4 * k,
                                             splines->x_left_hinge(x, y, knot) );
                        design_matrix.assign(j,
                                             4 * k + 1,
                                             splines->x_right_hinge(x, y, knot) );

                        knot = knots[ predictor_pairs_inds[i][1] ][k];
                        design_matrix.assign(j,
                                             4 * k + 2,
                                             splines->y_left_hinge(x, y, knot) );
                        design_matrix.assign(j,
                                             4 * k + 3,
                                             splines->y_right_hinge(x, y, knot) );

                    }
                }

                left_in_left = transpose(design_matrix);
                left_in_left.inplace_elewise_prod_mat_with_row_vec(w);
                left = inner_product(left_in_left, design_matrix);
                left = inverse(plus(left, eye));

                Y = ind_delta.submatrix(row_inds,
                                        no_rows,
                                        y_ind,
                                        no_y_ind);

                coefs = inner_product(left, inner_product(left_in_left, Y));


                // check this part, should I use predict_for_row?
                mse = 0;
                for(int j = 0; j < no_rows; ++j) {

                    prediction = 0;

                    x = train_x.get(row_inds[j],
                                    splines->col_inds[0]);
                    y = train_x.get(row_inds[j],
                                    splines->col_inds[1]);

                    for(int k = 0; k < splines->no_knots; ++k) {

                        knot = knots[ predictor_pairs_inds[i][0] ][k];
                        prediction += splines->x_left_hinge(x, y, knot) * coefs.get(4 * k, 0);
                        prediction += splines->x_right_hinge(x, y, knot) * coefs.get(4 * k + 1, 0);

                        knot = knots[ predictor_pairs_inds[i][1] ][k];
                        prediction += splines->y_left_hinge(x, y, knot) * coefs.get(4 * k + 2, 0);
                        prediction += splines->y_right_hinge(x, y, knot) * coefs.get(4 * k + 3, 0);

                    }
                    mse += std::pow(ind_delta.get(row_inds[j], 0) - prediction, 2.0);
                }

                mse /= no_rows;

                if(mse < lowest_mse) {
                    lowest_mse = mse;
                    lowest_mse_pair_ind = i;
                    copy(coefs, lowest_mse_coefs);
                }

            }

            splines->col_inds[0] = col_inds[ predictor_pairs_inds[lowest_mse_pair_ind][0] ];
            splines->col_inds[1] = col_inds[ predictor_pairs_inds[lowest_mse_pair_ind][1] ];

            for(int j = 0; j < splines->no_knots; ++j) {

                splines->x_knots[j] = knots[ predictor_pairs_inds[lowest_mse_pair_ind][0] ][j];

                splines->y_knots[j] = knots[ predictor_pairs_inds[lowest_mse_pair_ind][1] ][j];

            }

            for(int j = 0; j < splines->no_knots; ++j) {

                splines->x_left_coefs[j] = lowest_mse_coefs.get(4 * j, 0);
                splines->x_right_coefs[j] = lowest_mse_coefs.get(4 * j + 1, 0);

                splines->y_left_coefs[j] = lowest_mse_coefs.get(4 * j + 2, 0);
                splines->y_right_coefs[j] = lowest_mse_coefs.get(4 * j + 3, 0);

            }


            for(int i = 0; i < no_cols; ++i)
                delete[] knots[i];
            delete[] knots;
        }
    }

}

namespace dbm {

    template <typename T>
    Kmeans2d_trainer<T>::Kmeans2d_trainer(const Params &params) :
            no_centroids(params.kmeans_no_centroids),
            no_candidate_feature(params.dbm_no_candidate_feature),
            kmeans_max_iteration(params.kmeans_max_iteration),
            kmeans_tolerance(params.kmeans_tolerance),
            loss_function(Loss_function<T>(params)) {

        no_pairs = params.dbm_no_candidate_feature * (params.dbm_no_candidate_feature - 1) / 2;

        predictor_pairs_inds = new int *[no_pairs];
        for(int i = 0; i < no_pairs; ++i) {
            predictor_pairs_inds[i] = new int[2];
        }

        for(int i = 0, count = 0; i < params.dbm_no_candidate_feature - 1; ++i)
            for(int j = i + 1; j < params.dbm_no_candidate_feature; ++j) {
                predictor_pairs_inds[count][0] = i;
                predictor_pairs_inds[count][1] = j;
                count += 1;
            }

    }

    template <typename T>
    Kmeans2d_trainer<T>::~Kmeans2d_trainer() {

        for(int i = 0; i < no_pairs; ++i)
            delete[] predictor_pairs_inds[i];
        delete[] predictor_pairs_inds;

    }

    template <typename T>
    void Kmeans2d_trainer<T>::train(Kmeans2d<T> *kmeans2d,
                                  const Matrix<T> &train_x,
                                  const Matrix<T> &ind_delta,
                                  char loss_function_type,
                                  const int *row_inds,
                                  int no_rows,
                                  const int *col_inds,
                                  int no_cols) {

        if(row_inds == nullptr) {

            int height = train_x.get_height();

            T **start_centroids = new T*[no_centroids],
                    *start_prediction = new T[no_centroids],
                    **next_centroids = new T*[no_centroids],
                    *best_prediction = new T[no_centroids];

            for(int i = 0; i < no_centroids; ++i) {
                start_centroids[i] = new T[kmeans2d->no_predictors];
                next_centroids[i] = new T[kmeans2d->no_predictors];
            }

            int *no_samples_in_each_centroid = new int[no_centroids],
                    predictor_col_inds[kmeans2d->no_predictors];

            int closest_centroid_ind = 0;

            T feature_mins[kmeans2d->no_predictors],
                    feature_maxes[kmeans2d->no_predictors];

            T dist, lowest_dist, standard_dev,
                    largest_standard_dev = std::numeric_limits<T>::min();

            int **sample_inds_for_each_centroid = new int*[no_centroids];
            for(int i = 0; i < no_centroids; ++i)
                sample_inds_for_each_centroid[i] = nullptr;

            Matrix<T> prediction(no_centroids, 1, 0);

            T denominator_in_prediction;

            for(int l = 0; l < no_pairs / 3; ++l) {

                for(int i = 0; i < kmeans2d->no_predictors; ++i) {

                    predictor_col_inds[i] = col_inds[ predictor_pairs_inds[l][i] ];

                    feature_mins[i] = train_x.get_col_min(col_inds[i]);
                    feature_maxes[i] = train_x.get_col_max(col_inds[i]);

                }

                std::srand((unsigned int)
                                   std::chrono::duration_cast< std::chrono::milliseconds >
                                           (std::chrono::system_clock::now().time_since_epoch()).count() );

                for(int i = 0; i < no_centroids; ++i)
                    for(int j = 0; j < kmeans2d->no_predictors; ++j) {

                        start_centroids[i][j] =
                                std::rand() / double(RAND_MAX) *
                                (feature_maxes[j] - feature_mins[j]) + feature_mins[j];

                    }

                for(int iter = 0; iter < kmeans_max_iteration; ++iter) {

                    for(int i = 0; i < no_centroids; ++i) {

                        no_samples_in_each_centroid[i] = 0;

                        for(int j = 0; j < kmeans2d->no_predictors; ++j) {

                            next_centroids[i][j] = 0;

                        }
                    }

                    for(int i = 0; i < height; ++i) {

                        lowest_dist = std::numeric_limits<T>::max();

                        for(int j = 0; j < no_centroids; ++j) {

                            dist = 0;

                            for(int k = 0; k < kmeans2d->no_predictors; ++k) {
                                dist += std::pow(start_centroids[j][k] -
                                                 train_x.get(i, predictor_col_inds[k]), 2.0);
                            }

                            dist = std::sqrt(dist);

                            if(dist < lowest_dist) {
                                lowest_dist = dist;
                                closest_centroid_ind = j;
                            }

                        }

                        no_samples_in_each_centroid[closest_centroid_ind] += 1;

                        for(int j = 0; j < kmeans2d->no_predictors; ++j)

                            next_centroids[closest_centroid_ind][j] +=
                                    train_x.get(i, predictor_col_inds[j]);

                    }

                    dist = 0;
                    for(int i = 0; i < no_centroids; ++i)
                        for(int j = 0; j < kmeans2d->no_predictors; ++j) {

                            next_centroids[i][j] /= no_samples_in_each_centroid[i];

                            dist += std::pow(start_centroids[i][j] - next_centroids[i][j], 2.0);

                        }

                    dist = std::sqrt(dist);

                    if(dist < kmeans_tolerance) {

//                        std::cout << "Training of Kmeans(" << predictor_col_inds[0]
//                                  <<", " << predictor_col_inds[1]
//                                  << ") at " << kmeans2d
//                                  << " was early stoped after " << iter << "/" << kmeans_max_iteration
//                                  << " with the ending distance change: " << dist
//                                  << " !"
//                                  << std::endl;

                        break;

                    }
                    else if(iter < kmeans_max_iteration - 1) {

                        for(int i = 0; i < no_centroids; ++i)
                            for(int j = 0; j < kmeans2d->no_predictors; ++j)
                                start_centroids[i][j] = next_centroids[i][j];

                    }

                }

                for(int i = 0; i < no_centroids; ++i) {

                    sample_inds_for_each_centroid[i] = new int[no_samples_in_each_centroid[i]];

                    no_samples_in_each_centroid[i] = 0;

                }

                for(int i = 0; i < height; ++i) {

                    lowest_dist = std::numeric_limits<T>::max();

                    for(int j = 0; j < no_centroids; ++j) {

                        dist = 0;

                        for(int k = 0; k < kmeans2d->no_predictors; ++k) {
                            dist += std::pow(start_centroids[j][k] - train_x.get(i, predictor_col_inds[k]), 2.0);
                        }

                        dist = std::sqrt(dist);

                        if(dist < lowest_dist) {
                            lowest_dist = dist;
                            closest_centroid_ind = j;
                        }

                    }

                    sample_inds_for_each_centroid[closest_centroid_ind][no_samples_in_each_centroid[closest_centroid_ind]] = i;

                    no_samples_in_each_centroid[closest_centroid_ind] += 1;

                }

                for(int i = 0; i < no_centroids; ++i) {

                    start_prediction[i] = 0, denominator_in_prediction = 0;

                    for(int j = 0; j < no_samples_in_each_centroid[i]; ++j) {
                        start_prediction[i] += ind_delta.get(sample_inds_for_each_centroid[i][j], 0) *
                                               ind_delta.get(sample_inds_for_each_centroid[i][j], 1);
                        denominator_in_prediction += ind_delta.get(sample_inds_for_each_centroid[i][j], 1);
                    }

                    start_prediction[i] /= denominator_in_prediction;

                    delete[] sample_inds_for_each_centroid[i];

                    sample_inds_for_each_centroid[i] = nullptr;

                }

                for(int i = 0; i < height; ++i) {

                    lowest_dist = std::numeric_limits<T>::max();

                    for(int j = 0; j < no_centroids; ++j) {

                        dist = 0;

                        for(int k = 0; k < kmeans2d->no_predictors; ++k) {
                            dist += std::pow(start_centroids[j][k] - train_x.get(i, predictor_col_inds[k]), 2.0);
                        }

                        dist = std::sqrt(dist);

                        if(dist < lowest_dist) {
                            lowest_dist = dist;
                            closest_centroid_ind = j;
                        }

                    }

                    standard_dev += std::pow(ind_delta.get(i, 0) - start_prediction[closest_centroid_ind], 2.0);

                }

                standard_dev = std::sqrt(standard_dev / (height - 1));

                if(standard_dev > largest_standard_dev) {
                    largest_standard_dev = standard_dev;

                    kmeans2d->col_inds[0] = predictor_col_inds[0];
                    kmeans2d->col_inds[1] = predictor_col_inds[1];

                    for(int i = 0; i < no_centroids; ++i) {

                        kmeans2d->predictions[i] = start_prediction[i];

                        for(int j = 0; j < kmeans2d->no_predictors; ++j)
                            kmeans2d->centroids[i][j] = start_centroids[i][j];

                    }

                }

            }

            delete[] start_prediction;
            delete[] best_prediction;
            delete[] sample_inds_for_each_centroid;

            for(int i = 0; i < no_centroids; ++i) {
                delete[] start_centroids[i];
                delete[] next_centroids[i];
            }
            delete[] start_centroids;
            delete[] next_centroids;

            delete[] no_samples_in_each_centroid;

        }
        else {

            #ifdef _DEBUG_BASE_LEARNER_TRAINER
                assert(no_rows > 0 && no_cols == no_candidate_feature);
            #endif

            T **start_centroids = new T*[no_centroids],
                    *start_prediction = new T[no_centroids],
                    **next_centroids = new T*[no_centroids],
                    *best_prediction = new T[no_centroids];

            for(int i = 0; i < no_centroids; ++i) {
                start_centroids[i] = new T[kmeans2d->no_predictors];
                next_centroids[i] = new T[kmeans2d->no_predictors];
            }

            int *no_samples_in_each_centroid = new int[no_centroids],
                    predictor_col_inds[kmeans2d->no_predictors];

            int closest_centroid_ind = 0;

            T feature_mins[kmeans2d->no_predictors],
                    feature_maxes[kmeans2d->no_predictors];

            T dist, lowest_dist, standard_dev,
                    largest_standard_dev = std::numeric_limits<T>::min();

            int **sample_inds_for_each_centroid = new int*[no_centroids];
            for(int i = 0; i < no_centroids; ++i)
                sample_inds_for_each_centroid[i] = nullptr;

//            T mse, lowest_mse = std::numeric_limits<T>::max();

            Matrix<T> prediction(no_centroids, 1, 0);

            T denominator_in_prediction;

            for(int l = 0; l < no_pairs / 3; ++l) {

                for(int i = 0; i < kmeans2d->no_predictors; ++i) {

                    predictor_col_inds[i] = col_inds[ predictor_pairs_inds[l][i] ];

                    feature_mins[i] = train_x.get_col_min(col_inds[i], row_inds, no_rows);
                    feature_maxes[i] = train_x.get_col_max(col_inds[i], row_inds, no_rows);

                }

                std::srand((unsigned int)
                                   std::chrono::duration_cast< std::chrono::milliseconds >
                                           (std::chrono::system_clock::now().time_since_epoch()).count() );

                for(int i = 0; i < no_centroids; ++i)
                    for(int j = 0; j < kmeans2d->no_predictors; ++j) {

                        start_centroids[i][j] =
                                std::rand() / double(RAND_MAX) *
                                                    (feature_maxes[j] - feature_mins[j]) + feature_mins[j];

                    }

                for(int iter = 0; iter < kmeans_max_iteration; ++iter) {

                    for(int i = 0; i < no_centroids; ++i) {

                        no_samples_in_each_centroid[i] = 0;

                        for(int j = 0; j < kmeans2d->no_predictors; ++j) {

                            next_centroids[i][j] = 0;

                        }
                    }

                    for(int i = 0; i < no_rows; ++i) {

                        lowest_dist = std::numeric_limits<T>::max();

                        for(int j = 0; j < no_centroids; ++j) {

                            dist = 0;

                            for(int k = 0; k < kmeans2d->no_predictors; ++k) {
                                dist += std::pow(start_centroids[j][k] -
                                                         train_x.get(row_inds[i], predictor_col_inds[k]), 2.0);
                            }

                            dist = std::sqrt(dist);

                            if(dist < lowest_dist) {
                                lowest_dist = dist;
                                closest_centroid_ind = j;
                            }

                        }

                        no_samples_in_each_centroid[closest_centroid_ind] += 1;

                        for(int j = 0; j < kmeans2d->no_predictors; ++j)

                            next_centroids[closest_centroid_ind][j] +=
                                    train_x.get(row_inds[i], predictor_col_inds[j]);

                    }

                    dist = 0;
                    for(int i = 0; i < no_centroids; ++i)
                        for(int j = 0; j < kmeans2d->no_predictors; ++j) {

                            next_centroids[i][j] /= no_samples_in_each_centroid[i];

                            dist += std::pow(start_centroids[i][j] - next_centroids[i][j], 2.0);

                        }

                    dist = std::sqrt(dist);

                    if(dist < kmeans_tolerance) {

//                        std::cout << "Training of Kmeans(" << predictor_col_inds[0]
//                                  <<", " << predictor_col_inds[1]
//                                  << ") at " << kmeans2d
//                                  << " was early stoped after " << iter << "/" << kmeans_max_iteration
//                                  << " with the ending distance change: " << dist
//                                  << " !"
//                                  << std::endl;

                        break;

                    }
                    else if(iter < kmeans_max_iteration - 1) {

                        for(int i = 0; i < no_centroids; ++i)
                            for(int j = 0; j < kmeans2d->no_predictors; ++j)
                                start_centroids[i][j] = next_centroids[i][j];

                    }

                }

                for(int i = 0; i < no_centroids; ++i) {

                    sample_inds_for_each_centroid[i] = new int[no_samples_in_each_centroid[i]];

                    no_samples_in_each_centroid[i] = 0;

                }

                for(int i = 0; i < no_rows; ++i) {

                    lowest_dist = std::numeric_limits<T>::max();

                    for(int j = 0; j < no_centroids; ++j) {

                        dist = 0;

                        for(int k = 0; k < kmeans2d->no_predictors; ++k) {
                            dist += std::pow(start_centroids[j][k] - train_x.get(row_inds[i], predictor_col_inds[k]), 2.0);
                        }

                        dist = std::sqrt(dist);

                        if(dist < lowest_dist) {
                            lowest_dist = dist;
                            closest_centroid_ind = j;
                        }

                    }

                    sample_inds_for_each_centroid[closest_centroid_ind][no_samples_in_each_centroid[closest_centroid_ind]] = row_inds[i];

                    no_samples_in_each_centroid[closest_centroid_ind] += 1;

                }

                for(int i = 0; i < no_centroids; ++i) {

                    start_prediction[i] = 0, denominator_in_prediction = 0;

                    for(int j = 0; j < no_samples_in_each_centroid[i]; ++j) {
                        start_prediction[i] += ind_delta.get(sample_inds_for_each_centroid[i][j], 0) *
                                ind_delta.get(sample_inds_for_each_centroid[i][j], 1);
                        denominator_in_prediction += ind_delta.get(sample_inds_for_each_centroid[i][j], 1);
                    }

                    start_prediction[i] /= denominator_in_prediction;

//                    prediction.assign(i, 0, start_prediction[i]);

                    delete[] sample_inds_for_each_centroid[i];

                    sample_inds_for_each_centroid[i] = nullptr;

                }

                for(int i = 0; i < no_rows; ++i) {

                    lowest_dist = std::numeric_limits<T>::max();

                    for(int j = 0; j < no_centroids; ++j) {

                        dist = 0;

                        for(int k = 0; k < kmeans2d->no_predictors; ++k) {
                            dist += std::pow(start_centroids[j][k] - train_x.get(row_inds[i], predictor_col_inds[k]), 2.0);
                        }

                        dist = std::sqrt(dist);

                        if(dist < lowest_dist) {
                            lowest_dist = dist;
                            closest_centroid_ind = j;
                        }

                    }

                    standard_dev += std::pow(ind_delta.get(row_inds[i], 0) - start_prediction[closest_centroid_ind], 2.0);

                }

                standard_dev = std::sqrt(standard_dev / (no_rows - 1));

//                standard_dev = prediction.col_std(0);
                if(standard_dev > largest_standard_dev) {
                    largest_standard_dev = standard_dev;

                    kmeans2d->col_inds[0] = predictor_col_inds[0];
                    kmeans2d->col_inds[1] = predictor_col_inds[1];

                    for(int i = 0; i < no_centroids; ++i) {

                        kmeans2d->predictions[i] = start_prediction[i];

                        for(int j = 0; j < kmeans2d->no_predictors; ++j)
                            kmeans2d->centroids[i][j] = start_centroids[i][j];

                    }

                }

            }

            delete[] start_prediction;
            delete[] best_prediction;
            delete[] sample_inds_for_each_centroid;

            for(int i = 0; i < no_centroids; ++i) {
                delete[] start_centroids[i];
                delete[] next_centroids[i];
            }
            delete[] start_centroids;
            delete[] next_centroids;

            delete[] no_samples_in_each_centroid;

        }

    }
}

namespace dbm {

    template <typename T>
    Linear_regression_trainer<T>::Linear_regression_trainer(const Params &params) :
            regularization(params.lr_regularization) {}

    template <typename T>
    Linear_regression_trainer<T>::~Linear_regression_trainer() {}

    template <typename T>
    void Linear_regression_trainer<T>::train(Linear_regression<T> *linear_regression,
                                             const Matrix<T> &train_x,
                                             const Matrix<T> &ind_delta,
                                             const int *row_inds,
                                             int no_rows,
                                             const int *col_inds,
                                             int no_cols) {

        if(row_inds == nullptr) {
            int height = train_x.get_height(),
                    width = train_x.get_width();

            Matrix<T> w(1, height, 0);
            for(int i = 0; i < height; ++i)
                w.assign(0, i, ind_delta.get(i, 1));

            #ifdef _DEBUG_BASE_LEARNER_TRAINER
                assert(train_x.get_width() == linear_regression->no_predictors);
            #endif

            for(int j = 0; j < width; ++j)
                linear_regression->col_inds[j] = j;

            Matrix<T> intercept(height, 1, 1);
            Matrix<T> x = hori_merge(intercept, train_x);

            Matrix<T> left_in_left = transpose(x);
            left_in_left.inplace_elewise_prod_mat_with_row_vec(w);
            Matrix<T> left = inner_product(left_in_left, x);

            Matrix<T> y = ind_delta.col(0);

            Matrix<T> eye(width + 1, width + 1, 0);

            for(int i = 0; i < width + 1; ++i)
                eye.assign(i, i, regularization);

            left = inverse(plus(left, eye));

            Matrix<T> coefs = inner_product(left, inner_product(left_in_left, y));

            #ifdef _DEBUG_BASE_LEARNER_TRAINER
                assert(coefs.get_width() == 1 && coefs.get_height() == width + 1);
            #endif
            linear_regression->intercept = coefs.get(0, 0);
            for(int j = 1; j < width + 1; ++j)
                linear_regression->coefs_no_intercept[j - 1] = coefs.get(j, 0);
        }
        else {
            Matrix<T> w(1, no_rows, 0);
            for(int i = 0; i < no_rows; ++i)
                w.assign(0, i, ind_delta.get(row_inds[i], 1));

            #ifdef _DEBUG_BASE_LEARNER_TRAINER
                assert(no_rows > 0 && no_cols == linear_regression->no_predictors);
            #endif

            for(int j = 0; j < no_cols; ++j)
                linear_regression->col_inds[j] = col_inds[j];

            Matrix<T> intercept(no_rows, 1, 1);
            Matrix<T> x = hori_merge(intercept, train_x.submatrix(row_inds,
                                                                  no_rows,
                                                                  col_inds,
                                                                  no_cols));

            Matrix<T> left_in_left = transpose(x);
            left_in_left.inplace_elewise_prod_mat_with_row_vec(w);
            Matrix<T> left = inner_product(left_in_left, x);

            int y_ind[] = {0}, no_y_ind = 1;
            Matrix<T> y = ind_delta.submatrix(row_inds,
                                              no_rows,
                                              y_ind,
                                              no_y_ind);

            Matrix<T> eye(no_cols + 1, no_cols + 1, 0);

            for(int i = 0; i < no_cols + 1; ++i)
                eye.assign(i, i, regularization);

            left = inverse(plus(left, eye));

            Matrix<T> coefs = inner_product(left, inner_product(left_in_left, y));

            #ifdef _DEBUG_BASE_LEARNER_TRAINER
                assert(coefs.get_width() == 1 && coefs.get_height() == no_cols + 1);
            #endif
            linear_regression->intercept = coefs.get(0, 0);
            for(int j = 1; j < no_cols + 1; ++j)
                linear_regression->coefs_no_intercept[j - 1] = coefs.get(j, 0);
        }
    }

}

namespace dbm {

    template<typename T>
    Tree_trainer<T>::Tree_trainer(const Params &params) :
            max_depth(params.cart_max_depth),
            portion_candidate_split_point(params.cart_portion_candidate_split_point),
            loss_function(Loss_function<T>(params)) {};

    template<typename T>
    Tree_trainer<T>::~Tree_trainer() {};

    template<typename T>
    void Tree_trainer<T>::train(Tree_node<T> *tree,
                                const Matrix<T> &train_x,
                                const Matrix<T> &train_y,
                                const Matrix<T> &ind_delta,
                                const Matrix<T> &prediction,
                                const Matrix<T> &monotonic_constraints,
                                char loss_function_type,
                                const int *row_inds,
                                int no_rows,
                                const int *col_inds,
                                int no_cols) {

        if(row_inds == nullptr) {

            int data_height = train_x.get_height(),
                    data_width = train_x.get_width();

            tree->no_training_samples = data_height;

            tree->prediction = loss_function.estimate_mean(ind_delta, loss_function_type);
            if (tree->depth == max_depth || no_rows < min_samples_in_a_node) {
                tree->last_node = true;
                return;
            }

            tree->loss = std::numeric_limits<T>::max();

            int *larger_inds = new int[data_height],
                    *smaller_inds = new int[data_height];

            int larger_smaller_n[2] = {0, 0};

            T larger_beta,
                    smaller_beta,
                    loss = tree->loss;

            T *uniques = new T[data_height];
            for(int i = 0; i < data_height; ++i)
                uniques[i] = 0;

            for (int i = 0; i < data_width; ++i) {

                int no_uniques = train_x.unique_vals_col(i,
                                                         uniques);
                no_uniques = middles(uniques,
                                     no_uniques);
                shuffle(uniques,
                        no_uniques);

                no_uniques = (int)(no_uniques * portion_candidate_split_point) > threshold_using_all_split_point ?
                             (int)(no_uniques * portion_candidate_split_point) :
                             (no_uniques > threshold_using_all_split_point ?
                              threshold_using_all_split_point : no_uniques);

                for (int j = 0; j < no_uniques; ++j) {
                    train_x.inds_split(i,
                                       uniques[j],
                                       larger_inds,
                                       smaller_inds,
                                       larger_smaller_n);

                    larger_beta = loss_function.estimate_mean(ind_delta,
                                                              loss_function_type,
                                                              larger_inds,
                                                              larger_smaller_n[0]);
                    smaller_beta = loss_function.estimate_mean(ind_delta,
                                                               loss_function_type,
                                                               smaller_inds,
                                                               larger_smaller_n[1]);

                    if ( (larger_beta - smaller_beta) * monotonic_constraints.get(i, 0) < 0 )
                        continue;

                    loss = loss_function.loss(train_y,
                                              prediction,
                                              loss_function_type,
                                              larger_beta,
                                              larger_inds,
                                              larger_smaller_n[0]) +
                           loss_function.loss(train_y, prediction,
                                              loss_function_type,
                                              smaller_beta,
                                              smaller_inds,
                                              larger_smaller_n[1]);

                    if (loss < tree->loss) {
                        tree->loss = loss;
                        tree->column = i;
                        tree->split_value = uniques[j];
                    }

                }

            }

            if(tree->loss < std::numeric_limits<T>::max()) {
                train_x.inds_split(tree->column,
                                   tree->split_value,
                                   larger_inds,
                                   smaller_inds,
                                   larger_smaller_n);

                if (tree->larger != nullptr)
                    delete tree->larger;
                if (tree->smaller != nullptr)
                    delete tree->smaller;

                tree->larger = new Tree_node<T>(tree->depth + 1);
                tree->smaller = new Tree_node<T>(tree->depth + 1);

                train(tree->larger,
                      train_x,
                      train_y,
                      ind_delta,
                      prediction,
                      monotonic_constraints,
                      loss_function_type,
                      larger_inds,
                      larger_smaller_n[0]);

                train(tree->smaller,
                      train_x,
                      train_y,
                      ind_delta,
                      prediction,
                      monotonic_constraints,
                      loss_function_type,
                      smaller_inds,
                      larger_smaller_n[1]);

                delete[] larger_inds;
                delete[] smaller_inds;
                delete[] uniques;
                larger_inds = nullptr;
                smaller_inds = nullptr;
                uniques = nullptr;
            }
            else {
                tree->last_node = true;
                delete[] larger_inds;
                delete[] smaller_inds;
                delete[] uniques;
                larger_inds = nullptr;
                smaller_inds = nullptr;
                uniques = nullptr;
            }

        }
        else {

            tree->no_training_samples = no_rows;

            #ifdef _DEBUG_BASE_LEARNER_TRAINER
            assert(no_rows > 0 && no_cols > 0);
            #endif

            tree->prediction = loss_function.estimate_mean(ind_delta,
                                                           loss_function_type,
                                                           row_inds,
                                                           no_rows);
            if (tree->depth == max_depth || no_rows < min_samples_in_a_node) {
                tree->last_node = true;
                return;
            }

            tree->loss = std::numeric_limits<T>::max();

            int *larger_inds = new int[no_rows],
                    *smaller_inds = new int[no_rows];

            int larger_smaller_n[2] = {0, 0};

            T larger_beta,
                    smaller_beta,
                    loss = tree->loss;

            T *uniques = new T[no_rows];
            for(int i = 0; i < no_rows; ++i)
                uniques[i] = 0;

            for (int i = 0; i < no_cols; ++i) {

                int no_uniques = train_x.unique_vals_col(col_inds[i],
                                                         uniques,
                                                         row_inds,
                                                         no_rows);
                no_uniques = middles(uniques,
                                     no_uniques);
                shuffle(uniques,
                        no_uniques);

                no_uniques = (int)(no_uniques * portion_candidate_split_point) > threshold_using_all_split_point ?
                             (int)(no_uniques * portion_candidate_split_point) :
                             (no_uniques > threshold_using_all_split_point ?
                              threshold_using_all_split_point : no_uniques);

                for (int j = 0; j < no_uniques; ++j) {
                    train_x.inds_split(col_inds[i],
                                       uniques[j],
                                       larger_inds,
                                       smaller_inds,
                                       larger_smaller_n,
                                       row_inds, no_rows);

                    larger_beta = loss_function.estimate_mean(ind_delta,
                                                              loss_function_type,
                                                              larger_inds,
                                                              larger_smaller_n[0]);
                    smaller_beta = loss_function.estimate_mean(ind_delta,
                                                               loss_function_type,
                                                               smaller_inds,
                                                               larger_smaller_n[1]);

                    if ( (larger_beta - smaller_beta) * monotonic_constraints.get(col_inds[i], 0) < 0 )
                        continue;

                    loss = loss_function.loss(train_y,
                                              prediction,
                                              loss_function_type,
                                              larger_beta,
                                              larger_inds,
                                              larger_smaller_n[0]) +
                           loss_function.loss(train_y,
                                              prediction,
                                              loss_function_type,
                                              smaller_beta,
                                              smaller_inds,
                                              larger_smaller_n[1]);

                    if (loss < tree->loss) {
                        tree->loss = loss;
                        tree->column = col_inds[i];
                        tree->split_value = uniques[j];
                    }

                }

            }

            if(tree->loss < std::numeric_limits<T>::max()) {
                train_x.inds_split(tree->column,
                                   tree->split_value,
                                   larger_inds,
                                   smaller_inds,
                                   larger_smaller_n,
                                   row_inds,
                                   no_rows);

                if (tree->larger != nullptr)
                    delete tree->larger;
                if (tree->smaller != nullptr)
                    delete tree->smaller;

                tree->larger = new Tree_node<T>(tree->depth + 1);
                tree->smaller = new Tree_node<T>(tree->depth + 1);

                train(tree->larger,
                      train_x,
                      train_y,
                      ind_delta,
                      prediction,
                      monotonic_constraints,
                      loss_function_type,
                      larger_inds,
                      larger_smaller_n[0],
                      col_inds,
                      no_cols);
                train(tree->smaller,
                      train_x,
                      train_y,
                      ind_delta,
                      prediction,
                      monotonic_constraints,
                      loss_function_type,
                      smaller_inds,
                      larger_smaller_n[1],
                      col_inds,
                      no_cols);
                delete[] larger_inds;
                delete[] smaller_inds;
                delete[] uniques;
                larger_inds = nullptr;
                smaller_inds = nullptr;
                uniques = nullptr;
            }
            else {
                tree->last_node = true;
                delete[] larger_inds;
                delete[] smaller_inds;
                delete[] uniques;
                larger_inds = nullptr;
                smaller_inds = nullptr;
                uniques = nullptr;
            }

        }

    }

    template<typename T>
    void Tree_trainer<T>::prune(Tree_node<T> *tree) {

        if (tree->last_node) return;
        #ifdef _DEBUG_BASE_LEARNER_TRAINER
            assert(tree->larger != NULL && tree->smaller != NULL);
        #endif
        if (tree->larger->loss > tree->smaller->loss) {
            tree->larger->last_node = true;

            delete_tree(tree->larger->larger);
            delete_tree(tree->larger->smaller);

            tree->larger->larger = nullptr,
                    tree->larger->smaller = nullptr;

            prune(tree->smaller);
        } else {
            tree->smaller->last_node = true;

            delete_tree(tree->smaller->larger);
            delete_tree(tree->smaller->smaller);

            tree->smaller->larger = nullptr,
                    tree->smaller->smaller = nullptr;

            prune(tree->larger);
        }
    }

}



