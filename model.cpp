//
// Created by xixuan on 10/10/16.
//

#include "model.h"

#include <cassert>
#include <iostream>

namespace dbm {

    template
    class DBM<double>;

    template
    class DBM<float>;

}

namespace dbm {

    template<typename T>
    DBM<T>::DBM(int no_learners, int no_candidate_feature, int no_train_sample):
            no_learners(no_learners), no_candidate_feature(no_candidate_feature), no_train_sample(no_train_sample),
            loss_function(Loss_function<T>(params)) {

        learners = new Base_learner<T> *[no_learners];

        for (int i = 0; i < no_learners; ++i) {
            learners[i] = nullptr;
        }

        tree_trainer = nullptr;
        mean_trainer = nullptr;
        linear_regression_trainer = nullptr;
    }

    template<typename T>
    DBM<T>::DBM(const std::string &param_string) : loss_function(Loss_function<T>(params)) {

        params = set_params(param_string);

        no_learners = params.no_learners;
        no_candidate_feature = params.no_candidate_feature;
        no_train_sample = params.no_train_sample;

        learners = new Base_learner<T> *[no_learners];

        learners[0] = new Global_mean<T>;

        std::srand((unsigned int)std::time(NULL));

        double type_choose;
        for (int i = 1; i < no_learners; ++i) {
            type_choose = double(std::rand()) / RAND_MAX;
            if(type_choose < params.portion_for_trees)
                learners[i] = new Tree_node<T>(0);
            else if(type_choose < (params.portion_for_trees + params.portion_for_lr))
                learners[i] = new Linear_regression<T>(no_candidate_feature);
        }

        tree_trainer = new Tree_trainer<T>(params);
        mean_trainer = new Mean_trainer<T>(params);
        linear_regression_trainer = new Linear_regression_trainer<T>(params);
    }

    template<typename T>
    DBM<T>::~DBM() {

        for (int i = 0; i < no_learners; ++i) {
            delete learners[i];
        }
        delete[] learners;

        delete tree_trainer;
        delete mean_trainer;
        delete linear_regression_trainer;

        if(prediction_train_data != nullptr)
            delete prediction_train_data;

        if(test_loss_record != nullptr)
            delete test_loss_record;

    }

    template<typename T>
    void DBM<T>::train(const Matrix<T> &train_x, const Matrix<T> &train_y, const int * input_monotonic_constaints) {

        dbm::Time_measurer timer;

        int n_samples = train_y.get_height(), n_features = train_x.get_width();

        #if _DEBUG_MODEL
            assert(no_train_sample < n_samples && no_candidate_feature < n_features);
        #endif

        int * monotonic_constraints = nullptr;
        if (input_monotonic_constaints == nullptr) {
            monotonic_constraints = new int[n_features];
            for (int i = 0; i < n_features; ++i)
                monotonic_constraints[i] = 0;
        }
        else {
            monotonic_constraints = new int[n_features];
            for (int i = 0; i < n_features; ++i) {
                monotonic_constraints[i] = input_monotonic_constaints[i];

                // serves as a check of whether the length of monotonic_constraints is equal to the length of features
                // in some sense
                #if _DEBUG_MODEL
                    assert(monotonic_constraints[i] != 0 ||
                           monotonic_constraints[i] != -1 ||
                           monotonic_constraints[i] != 1);
                #endif
            }
        }

        int row_inds[n_samples], col_inds[n_features];
        for (int i = 0; i < n_features; ++i)
            col_inds[i] = i;
        for (int i = 0; i < n_samples; ++i)
            row_inds[i] = i;

        if (prediction_train_data != nullptr)
            delete prediction_train_data;
        prediction_train_data = new Matrix<T>(n_samples, 1, 0);

        /*
         * ind_delta:
         * col 0: individual delta
         * col 1: numerator of individual delta
         * col 2: denominator of individual delta
         */
        Matrix<T> ind_delta(n_samples, 3, 0);

        char type;

        if (params.display_training_progress) {
            if (params.record_every_tree)
                for (int i = 0; i < no_learners; ++i) {

                    std::cout << "Learner " << "(" << learners[i]->get_type() << ") " << " No. " << i << " -> ";

                    shuffle(row_inds, n_samples);
                    shuffle(col_inds, n_features);

                    loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                      ind_delta, params.loss_function, row_inds, no_train_sample);

                    type = learners[i]->get_type();
                    switch (type) {
                        case 'm': {
                            if (i > 0) {
                                mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[i]),
                                                    train_x, ind_delta, *prediction_train_data,
                                                    params.loss_function, row_inds, no_train_sample);
                                learners[i]->predict(train_x, *prediction_train_data);
                            }
                            else if (i == 0) {
                                loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                                  ind_delta, params.loss_function);
                                mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[i]),
                                                    train_x, ind_delta, *prediction_train_data);
                                learners[i]->predict(train_x, *prediction_train_data);
                            }
                            break;
                        }
                        case 't': {
                            tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[i]),
                                                train_x, train_y, ind_delta, *prediction_train_data,
                                                monotonic_constraints, params.loss_function,
                                                row_inds, no_train_sample,
                                                col_inds, no_candidate_feature);
                            tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[i]));
                            learners[i]->predict(train_x, *prediction_train_data, params.shrinkage);

                            {
                                tree_info = new Tree_info<T>(dynamic_cast<Tree_node<T> *>(learners[i]));
                                tree_info->print_to_file("trees.txt", i);
                            }

                            break;
                        }
                        case 'l': {
                            linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>(learners[i]),
                                                             train_x, ind_delta,
                                                             row_inds, no_train_sample,
                                                             col_inds, no_candidate_feature);
                            learners[i]->predict(train_x, *prediction_train_data, params.shrinkage);
                            break;
                        }
                        default: {
                            std::cout << "Wrong learner type: " << type <<  std::endl;
                            throw std::invalid_argument("Specified learner does not exist.");
                        }
                    }

                }
            else
                for (int i = 0; i < no_learners; ++i) {

                    std::cout << "Learner " << "(" << learners[i]->get_type() << ") " << " No. " << i << " -> ";

                    shuffle(row_inds, n_samples);
                    shuffle(col_inds, n_features);

                    loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                      ind_delta, params.loss_function, row_inds, no_train_sample);

                    type = learners[i]->get_type();
                    switch (type) {
                        case 'm': {
                            if (i > 0) {
                                mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[i]),
                                                    train_x, ind_delta, *prediction_train_data,
                                                    params.loss_function, row_inds, no_train_sample);
                                learners[i]->predict(train_x, *prediction_train_data);
                            }
                            else if (i == 0) {
                                loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                                  ind_delta, params.loss_function);
                                mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[i]),
                                                    train_x, ind_delta, *prediction_train_data);
                                learners[i]->predict(train_x, *prediction_train_data);
                            };
                            break;
                        }
                        case 't': {
                            tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[i]),
                                                train_x, train_y, ind_delta, *prediction_train_data,
                                                monotonic_constraints, params.loss_function,
                                                row_inds, no_train_sample,
                                                col_inds, no_candidate_feature);
                            tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[i]));
                            learners[i]->predict(train_x, *prediction_train_data, params.shrinkage);
                            break;
                        }
                        case 'l': {
                            linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>(learners[i]),
                                                             train_x, ind_delta,
                                                             row_inds, no_train_sample,
                                                             col_inds, no_candidate_feature);
                            learners[i]->predict(train_x, *prediction_train_data, params.shrinkage);
                            break;
                        }
                        default: {
                            std::cout << "Wrong learner type: " << type <<  std::endl;
                            throw std::invalid_argument("Specified learner does not exist.");
                        }
                    }

                }
        } else {
            if (params.record_every_tree)
                for (int i = 0; i < no_learners; ++i) {

                    std::cout << '.';

                    shuffle(row_inds, n_samples);
                    shuffle(col_inds, n_features);

                    loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                      ind_delta, params.loss_function, row_inds, no_train_sample);

                    type = learners[i]->get_type();
                    switch (type) {
                        case 'm': {
                            if (i > 0) {
                                mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[i]),
                                                    train_x, ind_delta, *prediction_train_data,
                                                    params.loss_function, row_inds, no_train_sample);
                                learners[i]->predict(train_x, *prediction_train_data);
                            }
                            else if (i == 0) {
                                loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                                  ind_delta, params.loss_function);
                                mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[i]),
                                                    train_x, ind_delta, *prediction_train_data);
                                learners[i]->predict(train_x, *prediction_train_data);
                            };
                            break;
                        }
                        case 't': {
                            tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[i]),
                                                train_x, train_y, ind_delta, *prediction_train_data,
                                                monotonic_constraints, params.loss_function,
                                                row_inds, no_train_sample,
                                                col_inds, no_candidate_feature);
                            tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[i]));
                            learners[i]->predict(train_x, *prediction_train_data, params.shrinkage);

                            {
                                tree_info = new Tree_info<T>(dynamic_cast<Tree_node<T> *>(learners[i]));
                                tree_info->print_to_file("trees.txt", i);
                            }

                            break;
                        }
                        case 'l': {
                            linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>(learners[i]),
                                                             train_x, ind_delta,
                                                             row_inds, no_train_sample,
                                                             col_inds, no_candidate_feature);
                            learners[i]->predict(train_x, *prediction_train_data, params.shrinkage);
                            break;
                        }
                        default: {
                            std::cout << "Wrong learner type: " << type <<  std::endl;
                            throw std::invalid_argument("Specified learner does not exist.");
                        }
                    }

                }
            else
                for (int i = 0; i < no_learners; ++i) {

                    std::cout << '.';

                    shuffle(row_inds, n_samples);
                    shuffle(col_inds, n_features);

                    loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                      ind_delta, params.loss_function, row_inds, no_train_sample);

                    type = learners[i]->get_type();
                    switch (type) {
                        case 'm': {
                            if (i > 0) {
                                mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[i]),
                                                    train_x, ind_delta, *prediction_train_data,
                                                    params.loss_function, row_inds, no_train_sample);
                                learners[i]->predict(train_x, *prediction_train_data);
                            }
                            else if (i == 0) {
                                loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                                  ind_delta, params.loss_function);
                                mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[i]),
                                                    train_x, ind_delta, *prediction_train_data);
                                learners[i]->predict(train_x, *prediction_train_data);
                            };
                            break;
                        }
                        case 't': {
                            tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[i]),
                                                train_x, train_y, ind_delta, *prediction_train_data,
                                                monotonic_constraints, params.loss_function,
                                                row_inds, no_train_sample,
                                                col_inds, no_candidate_feature);
                            tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[i]));
                            learners[i]->predict(train_x, *prediction_train_data, params.shrinkage);
                            break;
                        }
                        case 'l': {
                            linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>(learners[i]),
                                                             train_x, ind_delta,
                                                             row_inds, no_train_sample,
                                                             col_inds, no_candidate_feature);
                            learners[i]->predict(train_x, *prediction_train_data, params.shrinkage);
                            break;
                        }
                        default: {
                            std::cout << "Wrong learner type: " << type <<  std::endl;
                            throw std::invalid_argument("Specified learner does not exist.");
                        }
                    }

                }
        }

        std::cout << std::endl;

    }

    template<typename T>
    void DBM<T>::train(const Data_set<T> &data_set, const int * input_monotonic_constaints) {

        dbm::Time_measurer timer;

        Matrix<T> const &train_x = data_set.get_train_x();
        Matrix<T> const &train_y = data_set.get_train_y();
        Matrix<T> const &test_x = data_set.get_test_x();
        Matrix<T> const &test_y = data_set.get_test_y();

        int n_samples = train_x.get_height(), n_features = train_x.get_width();
        int test_n_samples = test_x.get_height();

        #if _DEBUG_MODEL
            assert(no_train_sample < n_samples && no_candidate_feature < n_features);
        #endif

        int * monotonic_constraints = nullptr;
        if (input_monotonic_constaints == nullptr) {
            monotonic_constraints = new int[n_features];
            for (int i = 0; i < n_features; ++i)
                monotonic_constraints[i] = 0;
        }
        else {
            monotonic_constraints = new int[n_features];
            for (int i = 0; i < n_features; ++i) {
                monotonic_constraints[i] = input_monotonic_constaints[i];

                // serves as a check of whether the length of monotonic_constraints is equal to the length of features
                // in some sense
                #if _DEBUG_MODEL
                    assert(monotonic_constraints[i] != 0 ||
                           monotonic_constraints[i] != -1 ||
                           monotonic_constraints[i] != 1);
                #endif
            }
        }

        int row_inds[n_samples], col_inds[n_features];

        for (int i = 0; i < n_features; ++i)
            col_inds[i] = i;
        for (int i = 0; i < n_samples; ++i)
            row_inds[i] = i;

        if (prediction_train_data != nullptr)
            delete prediction_train_data;

        prediction_train_data = new Matrix<T>(n_samples, 1, 0);
        test_loss_record = new T[no_learners / params.freq_showing_loss_on_test];

        /*
         * ind_delta:
         * col 0: individual delta
         * col 1: numerator of individual delta
         * col 2: denominator of individual delta
         */
        Matrix<T> ind_delta(n_samples, 3, 0);

        Matrix<T> prediction_test_data(test_n_samples, 1, 0);

        char type;

        if (params.display_training_progress) {
            if (params.record_every_tree)
                for (int i = 0; i < no_learners; ++i) {

                    std::cout << "Learner " << "(" << learners[i]->get_type() << ") " << " No. " << i << " -> ";

                    shuffle(row_inds, n_samples);
                    shuffle(col_inds, n_features);

                    loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                      ind_delta, params.loss_function, row_inds, no_train_sample);

                    type = learners[i]->get_type();
                    switch (type) {
                        case 'm': {
                            if (i > 0) {
                                mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[i]),
                                                    train_x, ind_delta, *prediction_train_data,
                                                    params.loss_function, row_inds, no_train_sample);
                                learners[i]->predict(train_x, *prediction_train_data);
                            }
                            else {
                                loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                                  ind_delta, params.loss_function);
                                mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[i]),
                                                    train_x, ind_delta, *prediction_train_data, params.loss_function);
                                learners[i]->predict(train_x, *prediction_train_data);
                            };
                            break;
                        }
                        case 't': {
                            tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[i]),
                                                train_x, train_y, ind_delta, *prediction_train_data,
                                                monotonic_constraints, params.loss_function,
                                                row_inds, no_train_sample,
                                                col_inds, no_candidate_feature);
                            tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[i]));
                            learners[i]->predict(train_x, *prediction_train_data, params.shrinkage);

                            {
                                tree_info = new Tree_info<T>(dynamic_cast<Tree_node<T> *>(learners[i]));
                                tree_info->print_to_file("trees.txt", i);
                            }
                            break;
                        }
                        case 'l': {
                            linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>(learners[i]),
                                                             train_x, ind_delta,
                                                             row_inds, no_train_sample,
                                                             col_inds, no_candidate_feature);
                            learners[i]->predict(train_x, *prediction_train_data, params.shrinkage);
                            break;
                        }
                        default: {
                            std::cout << "Wrong learner type: " << type <<  std::endl;
                            throw std::invalid_argument("Specified learner does not exist.");
                        }
                    }

                    learners[i]->predict(test_x, prediction_test_data, params.shrinkage);
                    if (!(i % params.freq_showing_loss_on_test)) {
                        test_loss_record[i / params.freq_showing_loss_on_test] = loss_function.loss(
                                test_y, prediction_test_data, params.loss_function);
                        std::cout << std::endl
                                  << '(' << i / 10 << ')'
                                  << " Loss on test set: "
                                  << test_loss_record[i / 10]
                                  << std::endl << std::endl;
                    }

                }
            else
                for (int i = 0; i < no_learners; ++i) {

                    std::cout << "Learner " << "(" << learners[i]->get_type() << ") " << " No. " << i << " -> ";

                    shuffle(row_inds, n_samples);
                    shuffle(col_inds, n_features);

                    loss_function.calculate_ind_delta(train_y, *prediction_train_data, ind_delta, params.loss_function, row_inds, no_train_sample);

                    type = learners[i]->get_type();
                    switch (type) {
                        case 'm': {
                            if (i > 0) {
                                mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[i]),
                                                    train_x, ind_delta, *prediction_train_data,
                                                    params.loss_function, row_inds, no_train_sample);
                                learners[i]->predict(train_x, *prediction_train_data);
                            }
                            else if (i == 0) {
                                loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                                  ind_delta, params.loss_function);
                                mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[i]),
                                                    train_x, ind_delta, *prediction_train_data);
                                learners[i]->predict(train_x, *prediction_train_data);
                            };
                            break;
                        }
                        case 't': {
                            tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[i]),
                                                train_x, train_y, ind_delta, *prediction_train_data,
                                                monotonic_constraints, params.loss_function,
                                                row_inds, no_train_sample,
                                                col_inds, no_candidate_feature);
                            tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[i]));
                            learners[i]->predict(train_x, *prediction_train_data, params.shrinkage);
                            break;
                        }
                        case 'l': {
                            linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>(learners[i]),
                                                             train_x, ind_delta,
                                                             row_inds, no_train_sample,
                                                             col_inds, no_candidate_feature);
                            learners[i]->predict(train_x, *prediction_train_data, params.shrinkage);
                            break;
                        }
                        default: {
                            std::cout << "Wrong learner type: " << type <<  std::endl;
                            throw std::invalid_argument("Specified learner does not exist.");
                        }
                    }

                    learners[i]->predict(test_x, prediction_test_data, params.shrinkage);
                    if (!(i % params.freq_showing_loss_on_test)) {
                        test_loss_record[i / params.freq_showing_loss_on_test] = loss_function.loss(
                                test_y, prediction_test_data, params.loss_function);
                        std::cout << std::endl
                                  << '(' << i / 10 << ')'
                                  << " Loss on test set: "
                                  << test_loss_record[i / 10]
                                  << std::endl << std::endl;
                    }

                }
        } else {
            if (params.record_every_tree)
                for (int i = 0; i < no_learners; ++i) {

                    std::cout << '.';

                    shuffle(row_inds, n_samples);
                    shuffle(col_inds, n_features);

                    loss_function.calculate_ind_delta(train_y, *prediction_train_data, ind_delta, params.loss_function, row_inds, no_train_sample);

                    type = learners[i]->get_type();
                    switch (type) {
                        case 'm': {
                            if (i > 0) {
                                mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[i]),
                                                    train_x, ind_delta, *prediction_train_data,
                                                    params.loss_function, row_inds, no_train_sample);
                                learners[i]->predict(train_x, *prediction_train_data);
                            }
                            else if (i == 0) {
                                loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                                  ind_delta, params.loss_function);
                                mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[i]),
                                                    train_x, ind_delta, *prediction_train_data);
                                learners[i]->predict(train_x, *prediction_train_data);
                            };
                            break;
                        }
                        case 't': {
                            tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[i]),
                                                train_x, train_y, ind_delta, *prediction_train_data,
                                                monotonic_constraints, params.loss_function,
                                                row_inds, no_train_sample,
                                                col_inds, no_candidate_feature);
                            tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[i]));
                            learners[i]->predict(train_x, *prediction_train_data, params.shrinkage);

                            {
                                tree_info = new Tree_info<T>(dynamic_cast<Tree_node<T> *>(learners[i]));
                                tree_info->print_to_file("trees.txt", i);
                            }
                            break;
                        }
                        case 'l': {
                            linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>(learners[i]),
                                                             train_x, ind_delta,
                                                             row_inds, no_train_sample,
                                                             col_inds, no_candidate_feature);
                            learners[i]->predict(train_x, *prediction_train_data, params.shrinkage);
                            break;
                        }
                        default: {
                            std::cout << "Wrong learner type: " << type <<  std::endl;
                            throw std::invalid_argument("Specified learner does not exist.");
                        }
                    }

                    learners[i]->predict(test_x, prediction_test_data, params.shrinkage);
                    if (!(i % params.freq_showing_loss_on_test)) {
                        test_loss_record[i / params.freq_showing_loss_on_test] = loss_function.loss(
                                test_y, prediction_test_data, params.loss_function);
                    }

                    tree_info = new Tree_info<T>(dynamic_cast<Tree_node<T> *>(learners[i]));
                    tree_info->print_to_file("trees.txt", i);
                    delete tree_info;

                }
            else
                for (int i = 0; i < no_learners; ++i) {

                    std::cout << '.';

                    shuffle(row_inds, n_samples);
                    shuffle(col_inds, n_features);

                    loss_function.calculate_ind_delta(train_y, *prediction_train_data, ind_delta, params.loss_function, row_inds, no_train_sample);

                    type = learners[i]->get_type();
                    switch (type) {
                        case 'm': {
                            if (i > 0) {
                                mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[i]),
                                                    train_x, ind_delta, *prediction_train_data,
                                                    params.loss_function, row_inds, no_train_sample);
                                learners[i]->predict(train_x, *prediction_train_data);
                            }
                            else if (i == 0) {
                                loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                                  ind_delta, params.loss_function);
                                mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[i]),
                                                    train_x, ind_delta, *prediction_train_data);
                                learners[i]->predict(train_x, *prediction_train_data);
                            };
                            break;
                        }
                        case 't': {
                            tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[i]),
                                                train_x, train_y, ind_delta, *prediction_train_data,
                                                monotonic_constraints, params.loss_function,
                                                row_inds, no_train_sample,
                                                col_inds, no_candidate_feature);
                            tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[i]));
                            learners[i]->predict(train_x, *prediction_train_data, params.shrinkage);
                            break;
                        }
                        case 'l': {
                            linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>(learners[i]),
                                                             train_x, ind_delta,
                                                             row_inds, no_train_sample,
                                                             col_inds, no_candidate_feature);
                            learners[i]->predict(train_x, *prediction_train_data, params.shrinkage);
                            break;
                        }
                        default: {
                            std::cout << "Wrong learner type: " << type <<  std::endl;
                            throw std::invalid_argument("Specified learner does not exist.");
                        }
                    }

                    learners[i]->predict(test_x, prediction_test_data, params.shrinkage);
                    if (!(i % params.freq_showing_loss_on_test)) {
                        test_loss_record[i / params.freq_showing_loss_on_test] = loss_function.loss(
                                test_y, prediction_test_data, params.loss_function);
                    }

                }
        }

        loss_function.mean_function(*prediction_train_data, params.loss_function);

        std::cout << std::endl << "Losses on Test Set: " << std::endl;
        for (int i = 0; i < no_learners / 10; ++i)
            std::cout << "(" << i << ") " << test_loss_record[i] << ' ';
        std::cout << std::endl;

    }

    template<typename T>
    void DBM<T>::predict(const Matrix<T> &data_x, Matrix<T> &predict_y) {

        int data_height = data_x.get_height();

        #if _DEBUG_MODEL
            assert(data_height == predict_y.get_height() && predict_y.get_width() == 1);
        #endif

        for (int i = 0; i < data_height; ++i) predict_y[i][0] = 0;

        if (learners[0]->get_type() == 'm') {
            learners[0]->predict(data_x, predict_y);
        }
        else {
            learners[0]->predict(data_x, predict_y, params.shrinkage);
        }

        for (int i = 1; i < no_learners; ++i) {

            learners[i]->predict(data_x, predict_y, params.shrinkage);

        }

        loss_function.mean_function(predict_y, params.loss_function);
    }

}

namespace dbm {

    template <typename T>
    Matrix<T> *DBM<T>::get_prediction_on_train_data() const {

        return prediction_train_data;

    }

    template <typename T>
    T *DBM<T>::get_test_loss() const {

        return test_loss_record;

    }

    template <typename T>
    void DBM<T>::set_loss_function_and_shrinkage(const char &type, const T &shrinkage) {

        params.loss_function = type;
        params.shrinkage = shrinkage;

    }

}

namespace dbm {

    template<typename T>
    void save_dbm(const DBM<T> *dbm, std::ofstream &out) {

        out << dbm->no_learners << ' '
            << dbm->no_candidate_feature << ' '
            << dbm->no_train_sample << ' '
            << dbm->params.loss_function << ' '
            << dbm->params.shrinkage << ' '
            << std::endl;
        char type;
        for (int i = 0; i < dbm->no_learners; ++i) {
            type = dbm->learners[i]->get_type();
            switch (type) {

                case 'm': {
                    out << "== Mean " << std::to_string(i) << " ==" << std::endl;
                    dbm::save_global_mean(dynamic_cast<Global_mean<T> *>(dbm->learners[i]), out);
                    out << "== End of Mean " << std::to_string(i) << " ==" << std::endl;
                    break;
                }

                case 't': {
                    out << "== Tree " << std::to_string(i) << " ==" << std::endl;
                    dbm::save_tree_node(dynamic_cast<Tree_node<T> *>(dbm->learners[i]), out);
                    out << "== End of Tree " << std::to_string(i) << " ==" << std::endl;
                    break;
                }

                case 'l': {
                    out << "== LinReg " << std::to_string(i) << " ==" << std::endl;
                    dbm::save_linear_regression(dynamic_cast<Linear_regression<T> *>(dbm->learners[i]), out);
                    out << "== End of LinReg " << std::to_string(i) << " ==" << std::endl;
                    break;
                }

                default: {
                    std::cout << "Wrong learner type: " << type <<  std::endl;
                    throw std::invalid_argument("Specified learner does not exist.");
                }

            }

        }

    }

    template<typename T>
    void load_dbm(std::ifstream &in, DBM<T> *&dbm) {

        std::string line;
        std::getline(in, line);

        std::string words[100];
        int count = split_into_words(line, words);

        dbm = new DBM<T>(std::stoi(words[0]), std::stoi(words[1]), std::stoi(words[2]));
        dbm->set_loss_function_and_shrinkage(words[3].front(), T(std::stod(words[4])));

        Tree_node<T> *temp_tree_ptr;
        Global_mean<T> *temp_mean_ptr;
        Linear_regression<T> *temp_linear_regression_ptr;

        char type;

        for (int i = 0; i < dbm->no_learners; ++i) {

            line.clear();
            std::getline(in, line);

            split_into_words(line, words);

            type = words[1].front();
            switch (type) {

                case 'M': {
                    temp_mean_ptr = nullptr;
                    load_global_mean(in, temp_mean_ptr);
                    dbm->learners[i] = temp_mean_ptr;

                    // skip the end line
                    std::getline(in, line);

                    break;
                }

                case 'T': {
                    temp_tree_ptr = nullptr;
                    load_tree_node(in, temp_tree_ptr);
                    dbm->learners[i] = temp_tree_ptr;

                    // skip the end line
                    std::getline(in, line);

                    break;
                }

                case 'L': {
                    temp_linear_regression_ptr = nullptr;
                    load_linear_regression(in, temp_linear_regression_ptr);
                    dbm->learners[i] = temp_linear_regression_ptr;

                    // skip the end line
                    std::getline(in, line);

                    break;
                }

                default: {
                    std::cout << "Wrong learner type: " << type <<  std::endl;
                    throw std::invalid_argument("Specified learner does not exist.");
                }

            }

        }

    }

}

namespace dbm {

    template void save_dbm<double>(const DBM<double> *dbm, std::ofstream &out);

    template void save_dbm<float>(const DBM<float> *dbm, std::ofstream &out);

    template void load_dbm<double>(std::ifstream &in, DBM<double> *&dbm);

    template void load_dbm<float>(std::ifstream &in, DBM<float> *&dbm);

}




