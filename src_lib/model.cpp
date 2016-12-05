//
// Created by xixuan on 10/10/16.
//

#include "model.h"

#include <cassert>
#include <iostream>
#include <limits>
#include <cmath>

#ifdef _OMP
#include <omp.h>
#endif

namespace dbm {

    template
    class DBM<double>;

    template
    class DBM<float>;

}

namespace dbm {

    template<typename T>
    DBM<T>::DBM(int no_bunches_of_learners,
                int no_cores,
                int no_candidate_feature,
                int no_train_sample,
                int total_no_feature):
            no_bunches_of_learners(no_bunches_of_learners),
            no_cores(no_cores),
            total_no_feature(total_no_feature),
            no_candidate_feature(no_candidate_feature),
            no_train_sample(no_train_sample),
            loss_function(Loss_function<T>(params)) {

        learners = new Base_learner<T> *[(no_bunches_of_learners - 1) * no_cores + 1];

        for (int i = 0; i < (no_bunches_of_learners - 1) * no_cores + 1; ++i) {
            learners[i] = nullptr;
        }

        tree_trainer = nullptr;
        mean_trainer = nullptr;
        linear_regression_trainer = nullptr;
        neural_network_trainer = nullptr;
        splines_trainer = nullptr;
        kmeans2d_trainer = nullptr;
    }

    template<typename T>
    DBM<T>::DBM(const Params &params) :
            params(params),
            loss_function(Loss_function<T>(params)) {

        std::srand((unsigned int)std::time(NULL));

        no_cores = params.no_cores;
        no_bunches_of_learners = params.no_bunches_of_learners;
        no_candidate_feature = params.no_candidate_feature;
        no_train_sample = params.no_train_sample;

        #ifdef _OMP
        if(no_cores == 0 || no_cores > omp_get_max_threads()) {
            std::cout << std::endl
                      << "================================="
                      << std::endl
                      << "no_cores: " << no_cores
                      << " is ignored, using " << omp_get_max_threads() << " !"
                      << std::endl
                      << "================================="
                      << std::endl;
            no_cores = omp_get_max_threads();
        }
        else if(no_cores < 0)
            throw std::invalid_argument("Specified no_cores is negative.");
        #else
        std::cout << std::endl
                  << "================================="
                  << std::endl
                  << "OpenMP is disabled!"
                  << std::endl
                  << "no_cores: " << no_cores
                  << " is ignored, using " << 1 << " !"
                  << std::endl
                  << "================================="
                  << std::endl;
        no_cores = 1;
        #endif

        learners = new Base_learner<T> *[(no_bunches_of_learners - 1) * no_cores + 1];

        learners[0] = new Global_mean<T>;

        double type_choose;
        for (int i = 1; i < no_bunches_of_learners; ++i) {

            type_choose = double(std::rand()) / RAND_MAX;

            if(type_choose < params.portion_for_trees)
                for(int j = 0; j < no_cores; ++j)
                    learners[no_cores * (i - 1) + j + 1] = new Tree_node<T>(0);

            else if(type_choose < (params.portion_for_trees + params.portion_for_lr))
                for(int j = 0; j < no_cores; ++j)
                    learners[no_cores * (i - 1) + j + 1] = new Linear_regression<T>(no_candidate_feature,
                                                                                    params.loss_function);

            else if(type_choose < (params.portion_for_trees + params.portion_for_lr + params.portion_for_s))
                for(int j = 0; j < no_cores; ++j)
                    learners[no_cores * (i - 1) + j + 1] = new Splines<T>(no_candidate_feature,
                                                                          params.loss_function);

            else if(type_choose < (params.portion_for_trees + params.portion_for_lr +
                                    params.portion_for_s + params.portion_for_k))
                for(int j = 0; j < no_cores; ++j)
                    learners[no_cores * (i - 1) + j + 1] = new Kmeans2d<T>(params.no_centroids,
                                                                           params.loss_function);

            else if(type_choose < (params.portion_for_trees + params.portion_for_lr +
                                    params.portion_for_s + params.portion_for_k +
                                    params.portion_for_nn))
                for(int j = 0; j < no_cores; ++j)
                    learners[no_cores * (i - 1) + j + 1] = new Neural_network<T>(no_candidate_feature,
                                                                                 params.no_hidden_neurons,
                                                                                 params.loss_function);
        }

        tree_trainer = new Tree_trainer<T>(params);
        mean_trainer = new Mean_trainer<T>(params);
        linear_regression_trainer = new Linear_regression_trainer<T>(params);
        neural_network_trainer = new Neural_network_trainer<T>(params);
        splines_trainer = new Splines_trainer<T>(params);
        kmeans2d_trainer = new Kmeans2d_trainer<T>(params);
    }

    template<typename T>
    DBM<T>::~DBM() {

        for (int i = 0; i < (no_bunches_of_learners - 1) * no_cores + 1; ++i) {
            delete learners[i];
        }
        delete[] learners;

        delete tree_trainer;
        delete mean_trainer;
        delete linear_regression_trainer;
        delete neural_network_trainer;
        delete splines_trainer;
        delete kmeans2d_trainer;

        if(prediction_train_data != nullptr)
            delete prediction_train_data;

        if(test_loss_record != nullptr)
            delete test_loss_record;

        if(pdp_result != nullptr)
            delete pdp_result;

        if(ss_result != nullptr)
            delete ss_result;

    }

    template<typename T>
    void DBM<T>::train(const Matrix<T> &train_x,
                       const Matrix<T> &train_y,
                       const Matrix<T> &input_monotonic_constraints) {
        /*
         * This function is not tested.
         */

        Time_measurer timer;

        int n_samples = train_y.get_height(), n_features = train_x.get_width();

        total_no_feature = n_features;

        #ifdef _DEBUG_MODEL
            assert(no_train_sample <= n_samples && no_candidate_feature < n_features);
        #endif

        for (int i = 0; i < n_features; ++i) {
            // serves as a check of whether the length of monotonic_constraints
            // is equal to the length of features in some sense
            #ifdef _DEBUG_MODEL
                assert(input_monotonic_constraints.get(i, 0) == 0 ||
                               input_monotonic_constraints.get(i, 0) == -1 ||
                               input_monotonic_constraints.get(i, 0) == 1);
            #endif
        }

        int *row_inds = new int[n_samples], *col_inds = new int[n_features];
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

        #ifdef _OMP
        omp_set_num_threads(no_cores);
        #endif

        char type;

        unsigned int *seeds = new unsigned int[(no_bunches_of_learners - 1) * no_cores];
        for(int i = 0; i < (no_bunches_of_learners - 1) * no_cores; ++i)
            seeds[i] = (unsigned int)(std::rand() / (RAND_MAX / 1e5));

        if (params.display_training_progress) {
            if (params.record_every_tree) {
                std::cout << "Learner " << "(" << learners[0]->get_type() << ") " << " No. " << 0 << " -> ";

                loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                  ind_delta, params.loss_function);
                mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[0]),
                                    train_x, ind_delta, *prediction_train_data, params.loss_function);
                learners[0]->predict(train_x, *prediction_train_data);


                for (int i = 1; i < no_bunches_of_learners; ++i) {

                    loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                      ind_delta, params.loss_function);

                    type = learners[(i - 1) * no_cores + 1]->get_type();
                    switch (type) {
                        case 't': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                            {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Tree at %p "
                                                    "number of samples: %d "
                                                    "max_depth: %d "
                                                    "no_candidate_split_point: %d...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, params.max_depth, params.no_candidate_split_point);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[learner_id]),
                                                    train_x, train_y, ind_delta, *prediction_train_data,
                                                    input_monotonic_constraints, params.loss_function,
                                                    thread_row_inds, no_train_sample,
                                                    thread_col_inds, no_candidate_feature);
                                tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[learner_id]));
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    {
                                        Tree_info<T> tree_info(dynamic_cast<Tree_node<T> *>
                                                               (learners[learner_id]));
                                        tree_info.print_to_file("trees.txt", learner_id);
                                    }
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'l': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                            {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Linear Regression at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>
                                                                 (learners[learner_id]),
                                                                 train_x, ind_delta,
                                                                 thread_row_inds, no_train_sample,
                                                                 thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                #ifdef _OMP
                                    #pragma omp critical
                                #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'k': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Kmeans at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of centroids: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature, params.no_centroids);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                kmeans2d_trainer->train(dynamic_cast<Kmeans2d<T> *> (learners[learner_id]),
                                                      train_x, ind_delta, params.loss_function,
                                                      thread_row_inds, no_train_sample,
                                                      thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 's': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Splines at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of knots: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature, params.no_knot);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                splines_trainer->train(dynamic_cast<Splines<T> *>
                                                       (learners[learner_id]),
                                                       train_x, ind_delta,
                                                       thread_row_inds, no_train_sample,
                                                       thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'n': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Neural Network at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of hidden neurons: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature, params.no_hidden_neurons);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                neural_network_trainer->train(dynamic_cast<Neural_network<T> *>
                                                              (learners[learner_id]),
                                                              train_x, ind_delta,
                                                              thread_row_inds, no_train_sample,
                                                              thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        default: {
                            std::cout << "Wrong learner type: " << type << std::endl;
                            throw std::invalid_argument("Specified learner does not exist.");
                        }
                    }
                }
            }
            else {
                std::cout << "Learner " << "(" << learners[0]->get_type() << ") " << " No. " << 0 << " -> ";

                loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                  ind_delta, params.loss_function);
                mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[0]),
                                    train_x, ind_delta, *prediction_train_data, params.loss_function);
                learners[0]->predict(train_x, *prediction_train_data);


                for (int i = 1; i < no_bunches_of_learners; ++i) {

                    loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                      ind_delta, params.loss_function);

                    type = learners[(i - 1) * no_cores + 1]->get_type();
                    switch (type) {
                        case 't': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                            {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Tree at %p "
                                                    "number of samples: %d "
                                                    "max_depth: %d "
                                                    "no_candidate_split_point: %d...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, params.max_depth, params.no_candidate_split_point);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[learner_id]),
                                                    train_x, train_y, ind_delta, *prediction_train_data,
                                                    input_monotonic_constraints, params.loss_function,
                                                    thread_row_inds, no_train_sample,
                                                    thread_col_inds, no_candidate_feature);
                                tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[learner_id]));
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                #ifdef _OMP
                                    #pragma omp critical
                                #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'l': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                            {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Linear Regression at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>
                                                                 (learners[learner_id]),
                                                                 train_x, ind_delta,
                                                                 thread_row_inds, no_train_sample,
                                                                 thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                #ifdef _OMP
                                    #pragma omp critical
                                #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'k': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Kmeans at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of centroids: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature, params.no_centroids);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                kmeans2d_trainer->train(dynamic_cast<Kmeans2d<T> *> (learners[learner_id]),
                                                      train_x, ind_delta, params.loss_function,
                                                      thread_row_inds, no_train_sample,
                                                      thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 's': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Splines at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of knots: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature, params.no_knot);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                splines_trainer->train(dynamic_cast<Splines<T> *>
                                                       (learners[learner_id]),
                                                       train_x, ind_delta,
                                                       thread_row_inds, no_train_sample,
                                                       thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'n': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Neural Network at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of hidden neurons: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature, params.no_hidden_neurons);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                neural_network_trainer->train(dynamic_cast<Neural_network<T> *>
                                                              (learners[learner_id]),
                                                              train_x, ind_delta,
                                                              thread_row_inds, no_train_sample,
                                                              thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        default: {
                            std::cout << "Wrong learner type: " << type << std::endl;
                            throw std::invalid_argument("Specified learner does not exist.");
                        }
                    }
                }
            }
        }
        else {
            if (params.record_every_tree) {
                std::cout << ".";

                loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                  ind_delta, params.loss_function);
                mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[0]),
                                    train_x, ind_delta, *prediction_train_data, params.loss_function);
                learners[0]->predict(train_x, *prediction_train_data);


                for (int i = 1; i < no_bunches_of_learners; ++i) {

                    loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                      ind_delta, params.loss_function);

                    type = learners[(i - 1) * no_cores + 1]->get_type();
                    switch (type) {
                        case 't': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                            {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[learner_id]),
                                                    train_x, train_y, ind_delta, *prediction_train_data,
                                                    input_monotonic_constraints, params.loss_function,
                                                    thread_row_inds, no_train_sample,
                                                    thread_col_inds, no_candidate_feature);
                                tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[learner_id]));
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                #ifdef _OMP
                                    #pragma omp critical
                                #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                #ifdef _OMP
                                    #pragma omp critical
                                #endif
                                    {
                                        Tree_info<T> tree_info(dynamic_cast<Tree_node<T> *>
                                                               (learners[learner_id]));
                                        tree_info.print_to_file("trees.txt", learner_id);
                                    }
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'l': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                            {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>
                                                                 (learners[learner_id]),
                                                                 train_x, ind_delta,
                                                                 thread_row_inds, no_train_sample,
                                                                 thread_col_inds, no_candidate_feature);
                            #ifdef _OMP
                                #pragma omp barrier
                            #endif
                                {
                            #ifdef _OMP
                                    #pragma omp critical
                            #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;
                            }
                            break;
                        }
                        case 'k': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                kmeans2d_trainer->train(dynamic_cast<Kmeans2d<T> *> (learners[learner_id]),
                                                      train_x, ind_delta, params.loss_function,
                                                      thread_row_inds, no_train_sample,
                                                      thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 's': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                splines_trainer->train(dynamic_cast<Splines<T> *>
                                                       (learners[learner_id]),
                                                       train_x, ind_delta,
                                                       thread_row_inds, no_train_sample,
                                                       thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'n': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                neural_network_trainer->train(dynamic_cast<Neural_network<T> *>
                                                              (learners[learner_id]),
                                                              train_x, ind_delta,
                                                              thread_row_inds, no_train_sample,
                                                              thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        default: {
                            std::cout << "Wrong learner type: " << type << std::endl;
                            throw std::invalid_argument("Specified learner does not exist.");
                        }
                    }
                }
            }
            else {
                std::cout << ".";

                loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                  ind_delta, params.loss_function);
                mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[0]),
                                    train_x, ind_delta, *prediction_train_data, params.loss_function);
                learners[0]->predict(train_x, *prediction_train_data);


                for (int i = 1; i < no_bunches_of_learners; ++i) {

                    loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                      ind_delta, params.loss_function);

                    type = learners[(i - 1) * no_cores + 1]->get_type();
                    switch (type) {
                        case 't': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                            {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[learner_id]),
                                                    train_x, train_y, ind_delta, *prediction_train_data,
                                                    input_monotonic_constraints, params.loss_function,
                                                    thread_row_inds, no_train_sample,
                                                    thread_col_inds, no_candidate_feature);
                                tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[learner_id]));
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                #ifdef _OMP
                                    #pragma omp critical
                                #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'l': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                            {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>
                                                                 (learners[learner_id]),
                                                                 train_x, ind_delta,
                                                                 thread_row_inds, no_train_sample,
                                                                 thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                #ifdef _OMP
                                    #pragma omp critical
                                #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'k': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                kmeans2d_trainer->train(dynamic_cast<Kmeans2d<T> *> (learners[learner_id]),
                                                      train_x, ind_delta, params.loss_function,
                                                      thread_row_inds, no_train_sample,
                                                      thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 's': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                splines_trainer->train(dynamic_cast<Splines<T> *>
                                                       (learners[learner_id]),
                                                       train_x, ind_delta,
                                                       thread_row_inds, no_train_sample,
                                                       thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'n': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                neural_network_trainer->train(dynamic_cast<Neural_network<T> *>
                                                              (learners[learner_id]),
                                                              train_x, ind_delta,
                                                              thread_row_inds, no_train_sample,
                                                              thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        default: {
                            std::cout << "Wrong learner type: " << type << std::endl;
                            throw std::invalid_argument("Specified learner does not exist.");
                        }
                    }
                }
            }
        }

        std::cout << std::endl;
        delete[] row_inds;
        delete[] col_inds;
        delete[] seeds;
        row_inds = nullptr, col_inds = nullptr, seeds = nullptr;

    }

    template<typename T>
    void DBM<T>::train(const Data_set<T> &data_set,
                       const Matrix<T> &input_monotonic_constraints) {

        Time_measurer timer;

        Matrix<T> const &train_x = data_set.get_train_x();
        Matrix<T> const &train_y = data_set.get_train_y();
        Matrix<T> const &test_x = data_set.get_test_x();
        Matrix<T> const &test_y = data_set.get_test_y();

        int n_samples = train_x.get_height(), n_features = train_x.get_width();
        int test_n_samples = test_x.get_height();

        total_no_feature = n_features;

        #ifdef _DEBUG_MODEL
            assert(no_train_sample <= n_samples && no_candidate_feature <= n_features);
        #endif

        for (int i = 0; i < n_features; ++i) {
            // serves as a check of whether the length of monotonic_constraints
            // is equal to the length of features in some sense
            #ifdef _DEBUG_MODEL
                assert(input_monotonic_constraints.get(i, 0) == 0 ||
                               input_monotonic_constraints.get(i, 0) == -1 ||
                               input_monotonic_constraints.get(i, 0) == 1);
            #endif
        }

        int *row_inds = new int[n_samples], *col_inds = new int[n_features];

        for (int i = 0; i < n_features; ++i)
            col_inds[i] = i;
        for (int i = 0; i < n_samples; ++i)
            row_inds[i] = i;

        if (prediction_train_data != nullptr)
            delete prediction_train_data;

        prediction_train_data = new Matrix<T>(n_samples, 1, 0);
        test_loss_record = new T[no_bunches_of_learners / params.freq_showing_loss_on_test];

        /*
         * ind_delta:
         * col 0: individual delta
         * col 1: denominator of individual delta
         */
        Matrix<T> ind_delta(n_samples, 2, 0);

        Matrix<T> prediction_test_data(test_n_samples, 1, 0);

        #ifdef _OMP
        omp_set_num_threads(no_cores);
        #endif

        unsigned int *seeds = new unsigned int[(no_bunches_of_learners - 1) * no_cores];
        for(int i = 0; i < (no_bunches_of_learners - 1) * no_cores; ++i)
            seeds[i] = (unsigned int)(std::rand() / (RAND_MAX / 1e5));

        char type;
        if (params.display_training_progress) {
            if (params.record_every_tree) {

                std::cout << "Learner " << "(" << learners[0]->get_type() << ") " << " No. " << 0 << " -> ";

                loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                  ind_delta, params.loss_function);
                mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[0]),
                                    train_x, ind_delta, *prediction_train_data, params.loss_function);
                learners[0]->predict(train_x, *prediction_train_data);
                learners[0]->predict(test_x, prediction_test_data);

                test_loss_record[0] = loss_function.loss(test_y, prediction_test_data, params.loss_function);
                std::cout << std::endl
                          << '(' << 0 << ')'
                          << " Loss on test set: "
                          << test_loss_record[0]
                          << std::endl << std::endl;

                for (int i = 1; i < no_bunches_of_learners; ++i) {

                    loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                      ind_delta, params.loss_function);

                    type = learners[(i - 1) * no_cores + 1]->get_type();
                    switch (type) {
                        case 't': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                            {
                                int thread_id = 0, learner_id = i;
                            #endif
                                    std::printf("Learner (%c) No. %d -> "
                                                        "Training Tree at %p "
                                                        "number of samples: %d "
                                                        "max_depth: %d "
                                                        "no_candidate_split_point: %d...\n",
                                                type, learner_id, learners[learner_id],
                                                no_train_sample, params.max_depth, params.no_candidate_split_point);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[learner_id]),
                                                    train_x, train_y, ind_delta, *prediction_train_data,
                                                    input_monotonic_constraints, params.loss_function,
                                                    thread_row_inds, no_train_sample,
                                                    thread_col_inds, no_candidate_feature);
                                tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[learner_id]));
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                     params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                     params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    {
                                        Tree_info<T> tree_info(dynamic_cast<Tree_node<T> *>
                                                                     (learners[learner_id]));
                                        tree_info.print_to_file("trees.txt", learner_id);
                                    }
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'l': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #pragma omp critical
                            #else
                            {
                                int thread_id = 0, learner_id = i;
                            #endif
                                    std::printf("Learner (%c) No. %d -> "
                                                        "Training Linear Regression at %p "
                                                        "number of samples: %d "
                                                        "number of predictors: %d ...\n",
                                                type, learner_id, learners[learner_id],
                                                no_train_sample, no_candidate_feature);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>
                                                                 (learners[learner_id]),
                                                                 train_x, ind_delta,
                                                                 thread_row_inds, no_train_sample,
                                                                 thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                     params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                     params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'k': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Kmeans2d at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of centroids: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature, params.no_centroids);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                kmeans2d_trainer->train(dynamic_cast<Kmeans2d<T> *> (learners[learner_id]),
                                                       train_x, ind_delta, params.loss_function,
                                                       thread_row_inds, no_train_sample,
                                                       thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 's': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Splines at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of knots: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature, params.no_knot);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                splines_trainer->train(dynamic_cast<Splines<T> *>
                                                       (learners[learner_id]),
                                                       train_x, ind_delta,
                                                       thread_row_inds, no_train_sample,
                                                       thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'n': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Neural Network at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of hidden neurons: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature, params.no_hidden_neurons);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                neural_network_trainer->train(dynamic_cast<Neural_network<T> *>
                                                                 (learners[learner_id]),
                                                                 train_x, ind_delta,
                                                                 thread_row_inds, no_train_sample,
                                                                 thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        default: {
                            std::cout << "Wrong learner type: " << type << std::endl;
                            throw std::invalid_argument("Specified learner does not exist.");
                        }
                    }

                    if (!(i % params.freq_showing_loss_on_test)) {
                        test_loss_record[i / params.freq_showing_loss_on_test] =
                                loss_function.loss(test_y, prediction_test_data, params.loss_function);
                        std::cout << std::endl
                                  << '(' << i / params.freq_showing_loss_on_test << ')'
                                  << " Loss on test set: "
                                  << test_loss_record[i / params.freq_showing_loss_on_test]
                                  << std::endl << std::endl;
                    }

                }

            }
            else {

                std::cout << "Learner " << "(" << learners[0]->get_type() << ") " << " No. " << 0 << " -> ";

                loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                  ind_delta, params.loss_function);
                mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[0]),
                                    train_x, ind_delta, *prediction_train_data, params.loss_function);
                learners[0]->predict(train_x, *prediction_train_data);
                learners[0]->predict(test_x, prediction_test_data);

                test_loss_record[0] = loss_function.loss(test_y, prediction_test_data, params.loss_function);
                std::cout << std::endl
                          << '(' << 0 << ')'
                          << " Loss on test set: "
                          << test_loss_record[0]
                          << std::endl << std::endl;

                for (int i = 1; i < no_bunches_of_learners; ++i) {

                    loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                      ind_delta, params.loss_function);

                    type = learners[(i - 1) * no_cores + 1]->get_type();
                    switch (type) {
                        case 't': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                            {
                                int thread_id = 0, learner_id = i;
                            #endif
                                    std::printf("Learner (%c) No. %d -> "
                                                        "Training Tree at %p "
                                                        "number of samples: %d "
                                                        "max_depth: %d "
                                                        "no_candidate_split_point: %d...\n",
                                                type, learner_id, learners[learner_id],
                                                no_train_sample, params.max_depth, params.no_candidate_split_point);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[learner_id]),
                                                    train_x, train_y, ind_delta, *prediction_train_data,
                                                    input_monotonic_constraints, params.loss_function,
                                                    thread_row_inds, no_train_sample,
                                                    thread_col_inds, no_candidate_feature);
                                tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[learner_id]));
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'l': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                            {
                                int thread_id = 0, learner_id = i;
                            #endif
                                    std::printf("Learner (%c) No. %d -> "
                                                        "Training Linear Regression at %p "
                                                        "number of samples: %d "
                                                        "number of predictors: %d ...\n",
                                                type, learner_id, learners[learner_id],
                                                no_train_sample, no_candidate_feature);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>
                                                                 (learners[learner_id]),
                                                                 train_x, ind_delta,
                                                                 thread_row_inds, no_train_sample,
                                                                 thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'k': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Kmeans at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of centroids: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature, params.no_centroids);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                kmeans2d_trainer->train(dynamic_cast<Kmeans2d<T> *> (learners[learner_id]),
                                                      train_x, ind_delta, params.loss_function,
                                                      thread_row_inds, no_train_sample,
                                                      thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 's': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Splines at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of knots: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature, params.no_knot);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                splines_trainer->train(dynamic_cast<Splines<T> *>
                                                       (learners[learner_id]),
                                                       train_x, ind_delta,
                                                       thread_row_inds, no_train_sample,
                                                       thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'n': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Neural Network at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of hidden neurons: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature, params.no_hidden_neurons);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                neural_network_trainer->train(dynamic_cast<Neural_network<T> *>
                                                              (learners[learner_id]),
                                                              train_x, ind_delta,
                                                              thread_row_inds, no_train_sample,
                                                              thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        default: {
                            std::cout << "Wrong learner type: " << type << std::endl;
                            throw std::invalid_argument("Specified learner does not exist.");
                        }
                    }

                    if (!(i % params.freq_showing_loss_on_test)) {
                        test_loss_record[i / params.freq_showing_loss_on_test] =
                                loss_function.loss(test_y, prediction_test_data, params.loss_function);
                        std::cout << std::endl
                                  << '(' << i / params.freq_showing_loss_on_test << ')'
                                  << " Loss on test set: "
                                  << test_loss_record[i / params.freq_showing_loss_on_test]
                                  << std::endl << std::endl;
                    }
                }
            }
        } else {
            if (params.record_every_tree) {

                std::printf(".");;

                loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                  ind_delta, params.loss_function);
                mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[0]),
                                    train_x, ind_delta, *prediction_train_data, params.loss_function);
                learners[0]->predict(train_x, *prediction_train_data);
                learners[0]->predict(test_x, prediction_test_data);

                test_loss_record[0] = loss_function.loss(test_y, prediction_test_data, params.loss_function);

                for (int i = 1; i < no_bunches_of_learners; ++i) {

                    loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                      ind_delta, params.loss_function);

                    type = learners[(i - 1) * no_cores + 1]->get_type();
                    switch (type) {
                        case 't': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                            {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[learner_id]),
                                                    train_x, train_y, ind_delta, *prediction_train_data,
                                                    input_monotonic_constraints, params.loss_function,
                                                    thread_row_inds, no_train_sample,
                                                    thread_col_inds, no_candidate_feature);
                                tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[learner_id]));
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    {
                                        Tree_info<T> tree_info(dynamic_cast<Tree_node<T> *>
                                                               (learners[learner_id]));
                                        tree_info.print_to_file("trees.txt", learner_id);
                                    }
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'l': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                            {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>
                                                                 (learners[learner_id]),
                                                                 train_x, ind_delta,
                                                                 thread_row_inds, no_train_sample,
                                                                 thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'k': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                kmeans2d_trainer->train(dynamic_cast<Kmeans2d<T> *> (learners[learner_id]),
                                                      train_x, ind_delta, params.loss_function,
                                                      thread_row_inds, no_train_sample,
                                                      thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 's': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                splines_trainer->train(dynamic_cast<Splines<T> *>
                                                       (learners[learner_id]),
                                                       train_x, ind_delta,
                                                       thread_row_inds, no_train_sample,
                                                       thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'n': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                neural_network_trainer->train(dynamic_cast<Neural_network<T> *>
                                                              (learners[learner_id]),
                                                              train_x, ind_delta,
                                                              thread_row_inds, no_train_sample,
                                                              thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        default: {
                            std::cout << "Wrong learner type: " << type << std::endl;
                            throw std::invalid_argument("Specified learner does not exist.");
                        }
                    }

                    if (!(i % params.freq_showing_loss_on_test)) {
                        test_loss_record[i / params.freq_showing_loss_on_test] =
                                loss_function.loss(test_y, prediction_test_data, params.loss_function);
                    }

                }

            }
            else {

                std::printf(".");;

                loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                  ind_delta, params.loss_function);
                mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[0]),
                                    train_x, ind_delta, *prediction_train_data, params.loss_function);
                learners[0]->predict(train_x, *prediction_train_data);
                learners[0]->predict(test_x, prediction_test_data);

                test_loss_record[0] = loss_function.loss(
                        test_y, prediction_test_data, params.loss_function);

                for (int i = 1; i < no_bunches_of_learners; ++i) {

                    loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                      ind_delta, params.loss_function);

                    type = learners[(i - 1) * no_cores + 1]->get_type();
                    switch (type) {
                        case 't': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                            {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[learner_id]),
                                                    train_x, train_y, ind_delta, *prediction_train_data,
                                                    input_monotonic_constraints, params.loss_function,
                                                    thread_row_inds, no_train_sample,
                                                    thread_col_inds, no_candidate_feature);
                                tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[learner_id]));
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'l': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                            {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>
                                                                 (learners[learner_id]),
                                                                 train_x, ind_delta,
                                                                 thread_row_inds, no_train_sample,
                                                                 thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'k': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                kmeans2d_trainer->train(dynamic_cast<Kmeans2d<T> *> (learners[learner_id]),
                                                      train_x, ind_delta, params.loss_function,
                                                      thread_row_inds, no_train_sample,
                                                      thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 's': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                splines_trainer->train(dynamic_cast<Splines<T> *>
                                                       (learners[learner_id]),
                                                       train_x, ind_delta,
                                                       thread_row_inds, no_train_sample,
                                                       thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'n': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                neural_network_trainer->train(dynamic_cast<Neural_network<T> *>
                                                              (learners[learner_id]),
                                                              train_x, ind_delta,
                                                              thread_row_inds, no_train_sample,
                                                              thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        default: {
                            std::cout << "Wrong learner type: " << type << std::endl;
                            throw std::invalid_argument("Specified learner does not exist.");
                        }
                    }

                    if (!(i % params.freq_showing_loss_on_test)) {
                        test_loss_record[i / params.freq_showing_loss_on_test] =
                                loss_function.loss(test_y, prediction_test_data, params.loss_function);
                        std::cout << std::endl
                                  << '(' << i / params.freq_showing_loss_on_test << ')'
                                  << " Loss on test set: "
                                  << test_loss_record[i / params.freq_showing_loss_on_test]
                                  << std::endl << std::endl;
                    }

                }

            }
        }

        loss_function.mean_function(*prediction_train_data, params.loss_function);

        std::cout << std::endl << "Losses on Test Set: " << std::endl;
        for (int i = 0; i < no_bunches_of_learners / params.freq_showing_loss_on_test; ++i)
            std::cout << "(" << i << ") " << test_loss_record[i] << ' ';
        std::cout << std::endl;

        delete[] row_inds;
        delete[] col_inds;
        delete[] seeds;
        row_inds = nullptr, col_inds = nullptr, seeds = nullptr;

    }

    template<typename T>
    void DBM<T>::train(const Matrix<T> &train_x,
                       const Matrix<T> &train_y) {
        /*
         * This function is not tested.
         */

        Time_measurer timer;

        int n_samples = train_y.get_height(), n_features = train_x.get_width();

        total_no_feature = n_features;

        #ifdef _DEBUG_MODEL
        assert(no_train_sample <= n_samples && no_candidate_feature < n_features);
        #endif

        Matrix<T> input_monotonic_constraints(n_features, 1, 0);

        int *row_inds = new int[n_samples], *col_inds = new int[n_features];
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

        #ifdef _OMP
            omp_set_num_threads(no_cores);
        #endif

        char type;

        unsigned int *seeds = new unsigned int[(no_bunches_of_learners - 1) * no_cores];
        for(int i = 0; i < (no_bunches_of_learners - 1) * no_cores; ++i)
            seeds[i] = (unsigned int)(std::rand() / (RAND_MAX / 1e5));

        if (params.display_training_progress) {
            if (params.record_every_tree) {
                std::cout << "Learner " << "(" << learners[0]->get_type() << ") " << " No. " << 0 << " -> ";

                loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                  ind_delta, params.loss_function);
                mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[0]),
                                    train_x, ind_delta, *prediction_train_data, params.loss_function);
                learners[0]->predict(train_x, *prediction_train_data);


                for (int i = 1; i < no_bunches_of_learners; ++i) {

                    loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                      ind_delta, params.loss_function);

                    type = learners[(i - 1) * no_cores + 1]->get_type();
                    switch (type) {
                        case 't': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Tree at %p "
                                                    "number of samples: %d "
                                                    "max_depth: %d "
                                                    "no_candidate_split_point: %d...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, params.max_depth, params.no_candidate_split_point);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[learner_id]),
                                                    train_x, train_y, ind_delta, *prediction_train_data,
                                                    input_monotonic_constraints, params.loss_function,
                                                    thread_row_inds, no_train_sample,
                                                    thread_col_inds, no_candidate_feature);
                                tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[learner_id]));
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    {
                                        Tree_info<T> tree_info(dynamic_cast<Tree_node<T> *>
                                                               (learners[learner_id]));
                                        tree_info.print_to_file("trees.txt", learner_id);
                                    }
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'l': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Linear Regression at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>
                                                                 (learners[learner_id]),
                                                                 train_x, ind_delta,
                                                                 thread_row_inds, no_train_sample,
                                                                 thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'k': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Kmeans at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of centroids: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature, params.no_centroids);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                kmeans2d_trainer->train(dynamic_cast<Kmeans2d<T> *> (learners[learner_id]),
                                                        train_x, ind_delta, params.loss_function,
                                                        thread_row_inds, no_train_sample,
                                                        thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 's': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Splines at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of knots: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature, params.no_knot);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                splines_trainer->train(dynamic_cast<Splines<T> *>
                                                       (learners[learner_id]),
                                                       train_x, ind_delta,
                                                       thread_row_inds, no_train_sample,
                                                       thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'n': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Neural Network at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of hidden neurons: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature, params.no_hidden_neurons);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                neural_network_trainer->train(dynamic_cast<Neural_network<T> *>
                                                              (learners[learner_id]),
                                                              train_x, ind_delta,
                                                              thread_row_inds, no_train_sample,
                                                              thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        default: {
                            std::cout << "Wrong learner type: " << type << std::endl;
                            throw std::invalid_argument("Specified learner does not exist.");
                        }
                    }
                }
            }
            else {
                std::cout << "Learner " << "(" << learners[0]->get_type() << ") " << " No. " << 0 << " -> ";

                loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                  ind_delta, params.loss_function);
                mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[0]),
                                    train_x, ind_delta, *prediction_train_data, params.loss_function);
                learners[0]->predict(train_x, *prediction_train_data);


                for (int i = 1; i < no_bunches_of_learners; ++i) {

                    loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                      ind_delta, params.loss_function);

                    type = learners[(i - 1) * no_cores + 1]->get_type();
                    switch (type) {
                        case 't': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Tree at %p "
                                                    "number of samples: %d "
                                                    "max_depth: %d "
                                                    "no_candidate_split_point: %d...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, params.max_depth, params.no_candidate_split_point);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[learner_id]),
                                                    train_x, train_y, ind_delta, *prediction_train_data,
                                                    input_monotonic_constraints, params.loss_function,
                                                    thread_row_inds, no_train_sample,
                                                    thread_col_inds, no_candidate_feature);
                                tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[learner_id]));
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'l': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Linear Regression at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>
                                                                 (learners[learner_id]),
                                                                 train_x, ind_delta,
                                                                 thread_row_inds, no_train_sample,
                                                                 thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'k': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Kmeans at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of centroids: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature, params.no_centroids);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                kmeans2d_trainer->train(dynamic_cast<Kmeans2d<T> *> (learners[learner_id]),
                                                        train_x, ind_delta, params.loss_function,
                                                        thread_row_inds, no_train_sample,
                                                        thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 's': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Splines at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of knots: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature, params.no_knot);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                splines_trainer->train(dynamic_cast<Splines<T> *>
                                                       (learners[learner_id]),
                                                       train_x, ind_delta,
                                                       thread_row_inds, no_train_sample,
                                                       thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'n': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Neural Network at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of hidden neurons: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature, params.no_hidden_neurons);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                neural_network_trainer->train(dynamic_cast<Neural_network<T> *>
                                                              (learners[learner_id]),
                                                              train_x, ind_delta,
                                                              thread_row_inds, no_train_sample,
                                                              thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        default: {
                            std::cout << "Wrong learner type: " << type << std::endl;
                            throw std::invalid_argument("Specified learner does not exist.");
                        }
                    }
                }
            }
        }
        else {
            if (params.record_every_tree) {
                std::cout << ".";

                loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                  ind_delta, params.loss_function);
                mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[0]),
                                    train_x, ind_delta, *prediction_train_data, params.loss_function);
                learners[0]->predict(train_x, *prediction_train_data);


                for (int i = 1; i < no_bunches_of_learners; ++i) {

                    loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                      ind_delta, params.loss_function);

                    type = learners[(i - 1) * no_cores + 1]->get_type();
                    switch (type) {
                        case 't': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[learner_id]),
                                                    train_x, train_y, ind_delta, *prediction_train_data,
                                                    input_monotonic_constraints, params.loss_function,
                                                    thread_row_inds, no_train_sample,
                                                    thread_col_inds, no_candidate_feature);
                                tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[learner_id]));
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    {
                                        Tree_info<T> tree_info(dynamic_cast<Tree_node<T> *>
                                                               (learners[learner_id]));
                                        tree_info.print_to_file("trees.txt", learner_id);
                                    }
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'l': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>
                                                                 (learners[learner_id]),
                                                                 train_x, ind_delta,
                                                                 thread_row_inds, no_train_sample,
                                                                 thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;
                            }
                            break;
                        }
                        case 'k': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                kmeans2d_trainer->train(dynamic_cast<Kmeans2d<T> *> (learners[learner_id]),
                                                        train_x, ind_delta, params.loss_function,
                                                        thread_row_inds, no_train_sample,
                                                        thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 's': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                splines_trainer->train(dynamic_cast<Splines<T> *>
                                                       (learners[learner_id]),
                                                       train_x, ind_delta,
                                                       thread_row_inds, no_train_sample,
                                                       thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'n': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                neural_network_trainer->train(dynamic_cast<Neural_network<T> *>
                                                              (learners[learner_id]),
                                                              train_x, ind_delta,
                                                              thread_row_inds, no_train_sample,
                                                              thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        default: {
                            std::cout << "Wrong learner type: " << type << std::endl;
                            throw std::invalid_argument("Specified learner does not exist.");
                        }
                    }
                }
            }
            else {
                std::cout << ".";

                loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                  ind_delta, params.loss_function);
                mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[0]),
                                    train_x, ind_delta, *prediction_train_data, params.loss_function);
                learners[0]->predict(train_x, *prediction_train_data);


                for (int i = 1; i < no_bunches_of_learners; ++i) {

                    loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                      ind_delta, params.loss_function);

                    type = learners[(i - 1) * no_cores + 1]->get_type();
                    switch (type) {
                        case 't': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[learner_id]),
                                                    train_x, train_y, ind_delta, *prediction_train_data,
                                                    input_monotonic_constraints, params.loss_function,
                                                    thread_row_inds, no_train_sample,
                                                    thread_col_inds, no_candidate_feature);
                                tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[learner_id]));
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'l': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>
                                                                 (learners[learner_id]),
                                                                 train_x, ind_delta,
                                                                 thread_row_inds, no_train_sample,
                                                                 thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'k': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                kmeans2d_trainer->train(dynamic_cast<Kmeans2d<T> *> (learners[learner_id]),
                                                        train_x, ind_delta, params.loss_function,
                                                        thread_row_inds, no_train_sample,
                                                        thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 's': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                splines_trainer->train(dynamic_cast<Splines<T> *>
                                                       (learners[learner_id]),
                                                       train_x, ind_delta,
                                                       thread_row_inds, no_train_sample,
                                                       thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'n': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                neural_network_trainer->train(dynamic_cast<Neural_network<T> *>
                                                              (learners[learner_id]),
                                                              train_x, ind_delta,
                                                              thread_row_inds, no_train_sample,
                                                              thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        default: {
                            std::cout << "Wrong learner type: " << type << std::endl;
                            throw std::invalid_argument("Specified learner does not exist.");
                        }
                    }
                }
            }
        }

        std::cout << std::endl;
        delete[] row_inds;
        delete[] col_inds;
        delete[] seeds;
        row_inds = nullptr, col_inds = nullptr, seeds = nullptr;

    }

    template<typename T>
    void DBM<T>::train(const Data_set<T> &data_set) {

        Time_measurer timer;

        Matrix<T> const &train_x = data_set.get_train_x();
        Matrix<T> const &train_y = data_set.get_train_y();
        Matrix<T> const &test_x = data_set.get_test_x();
        Matrix<T> const &test_y = data_set.get_test_y();

        int n_samples = train_x.get_height(), n_features = train_x.get_width();
        int test_n_samples = test_x.get_height();

        total_no_feature = n_features;

        #ifdef _DEBUG_MODEL
            assert(no_train_sample <= n_samples && no_candidate_feature <= n_features);
        #endif

        Matrix<T> input_monotonic_constraints(n_features, 1, 0);

        int *row_inds = new int[n_samples], *col_inds = new int[n_features];

        for (int i = 0; i < n_features; ++i)
            col_inds[i] = i;
        for (int i = 0; i < n_samples; ++i)
            row_inds[i] = i;

        if (prediction_train_data != nullptr)
            delete prediction_train_data;

        prediction_train_data = new Matrix<T>(n_samples, 1, 0);
        test_loss_record = new T[no_bunches_of_learners / params.freq_showing_loss_on_test];

        /*
         * ind_delta:
         * col 0: individual delta
         * col 1: denominator of individual delta
         */
        Matrix<T> ind_delta(n_samples, 2, 0);

        Matrix<T> prediction_test_data(test_n_samples, 1, 0);

        #ifdef _OMP
            omp_set_num_threads(no_cores);
        #endif

        unsigned int *seeds = new unsigned int[(no_bunches_of_learners - 1) * no_cores];
        for(int i = 0; i < (no_bunches_of_learners - 1) * no_cores; ++i)
            seeds[i] = (unsigned int)(std::rand() / (RAND_MAX / 1e5));

        char type;
        if (params.display_training_progress) {
            if (params.record_every_tree) {

                std::cout << "Learner " << "(" << learners[0]->get_type() << ") " << " No. " << 0 << " -> ";

                loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                  ind_delta, params.loss_function);
                mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[0]),
                                    train_x, ind_delta, *prediction_train_data, params.loss_function);
                learners[0]->predict(train_x, *prediction_train_data);
                learners[0]->predict(test_x, prediction_test_data);

                test_loss_record[0] = loss_function.loss(test_y, prediction_test_data, params.loss_function);
                std::cout << std::endl
                          << '(' << 0 << ')'
                          << " Loss on test set: "
                          << test_loss_record[0]
                          << std::endl << std::endl;

                for (int i = 1; i < no_bunches_of_learners; ++i) {

                    loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                      ind_delta, params.loss_function);

                    type = learners[(i - 1) * no_cores + 1]->get_type();
                    switch (type) {
                        case 't': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Tree at %p "
                                                    "number of samples: %d "
                                                    "max_depth: %d "
                                                    "no_candidate_split_point: %d...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, params.max_depth, params.no_candidate_split_point);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[learner_id]),
                                                    train_x, train_y, ind_delta, *prediction_train_data,
                                                    input_monotonic_constraints, params.loss_function,
                                                    thread_row_inds, no_train_sample,
                                                    thread_col_inds, no_candidate_feature);
                                tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[learner_id]));
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    {
                                        Tree_info<T> tree_info(dynamic_cast<Tree_node<T> *>
                                                               (learners[learner_id]));
                                        tree_info.print_to_file("trees.txt", learner_id);
                                    }
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'l': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Linear Regression at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>
                                                                 (learners[learner_id]),
                                                                 train_x, ind_delta,
                                                                 thread_row_inds, no_train_sample,
                                                                 thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'k': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Kmeans2d at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of centroids: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature, params.no_centroids);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                kmeans2d_trainer->train(dynamic_cast<Kmeans2d<T> *> (learners[learner_id]),
                                                        train_x, ind_delta, params.loss_function,
                                                        thread_row_inds, no_train_sample,
                                                        thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 's': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Splines at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of knots: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature, params.no_knot);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                splines_trainer->train(dynamic_cast<Splines<T> *>
                                                       (learners[learner_id]),
                                                       train_x, ind_delta,
                                                       thread_row_inds, no_train_sample,
                                                       thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'n': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Neural Network at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of hidden neurons: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature, params.no_hidden_neurons);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                neural_network_trainer->train(dynamic_cast<Neural_network<T> *>
                                                              (learners[learner_id]),
                                                              train_x, ind_delta,
                                                              thread_row_inds, no_train_sample,
                                                              thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        default: {
                            std::cout << "Wrong learner type: " << type << std::endl;
                            throw std::invalid_argument("Specified learner does not exist.");
                        }
                    }

                    if (!(i % params.freq_showing_loss_on_test)) {
                        test_loss_record[i / params.freq_showing_loss_on_test] =
                                loss_function.loss(test_y, prediction_test_data, params.loss_function);
                        std::cout << std::endl
                                  << '(' << i / params.freq_showing_loss_on_test << ')'
                                  << " Loss on test set: "
                                  << test_loss_record[i / params.freq_showing_loss_on_test]
                                  << std::endl << std::endl;
                    }

                }

            }
            else {

                std::cout << "Learner " << "(" << learners[0]->get_type() << ") " << " No. " << 0 << " -> ";

                loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                  ind_delta, params.loss_function);
                mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[0]),
                                    train_x, ind_delta, *prediction_train_data, params.loss_function);
                learners[0]->predict(train_x, *prediction_train_data);
                learners[0]->predict(test_x, prediction_test_data);

                test_loss_record[0] = loss_function.loss(test_y, prediction_test_data, params.loss_function);
                std::cout << std::endl
                          << '(' << 0 << ')'
                          << " Loss on test set: "
                          << test_loss_record[0]
                          << std::endl << std::endl;

                for (int i = 1; i < no_bunches_of_learners; ++i) {

                    loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                      ind_delta, params.loss_function);

                    type = learners[(i - 1) * no_cores + 1]->get_type();
                    switch (type) {
                        case 't': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Tree at %p "
                                                    "number of samples: %d "
                                                    "max_depth: %d "
                                                    "no_candidate_split_point: %d...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, params.max_depth, params.no_candidate_split_point);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[learner_id]),
                                                    train_x, train_y, ind_delta, *prediction_train_data,
                                                    input_monotonic_constraints, params.loss_function,
                                                    thread_row_inds, no_train_sample,
                                                    thread_col_inds, no_candidate_feature);
                                tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[learner_id]));
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'l': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Linear Regression at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>
                                                                 (learners[learner_id]),
                                                                 train_x, ind_delta,
                                                                 thread_row_inds, no_train_sample,
                                                                 thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'k': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Kmeans at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of centroids: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature, params.no_centroids);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                kmeans2d_trainer->train(dynamic_cast<Kmeans2d<T> *> (learners[learner_id]),
                                                        train_x, ind_delta, params.loss_function,
                                                        thread_row_inds, no_train_sample,
                                                        thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 's': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Splines at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of knots: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature, params.no_knot);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                splines_trainer->train(dynamic_cast<Splines<T> *>
                                                       (learners[learner_id]),
                                                       train_x, ind_delta,
                                                       thread_row_inds, no_train_sample,
                                                       thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'n': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Neural Network at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of hidden neurons: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature, params.no_hidden_neurons);

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                neural_network_trainer->train(dynamic_cast<Neural_network<T> *>
                                                              (learners[learner_id]),
                                                              train_x, ind_delta,
                                                              thread_row_inds, no_train_sample,
                                                              thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        default: {
                            std::cout << "Wrong learner type: " << type << std::endl;
                            throw std::invalid_argument("Specified learner does not exist.");
                        }
                    }

                    if (!(i % params.freq_showing_loss_on_test)) {
                        test_loss_record[i / params.freq_showing_loss_on_test] =
                                loss_function.loss(test_y, prediction_test_data, params.loss_function);
                        std::cout << std::endl
                                  << '(' << i / params.freq_showing_loss_on_test << ')'
                                  << " Loss on test set: "
                                  << test_loss_record[i / params.freq_showing_loss_on_test]
                                  << std::endl << std::endl;
                    }
                }
            }
        } else {
            if (params.record_every_tree) {

                std::printf(".");;

                loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                  ind_delta, params.loss_function);
                mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[0]),
                                    train_x, ind_delta, *prediction_train_data, params.loss_function);
                learners[0]->predict(train_x, *prediction_train_data);
                learners[0]->predict(test_x, prediction_test_data);

                test_loss_record[0] = loss_function.loss(test_y, prediction_test_data, params.loss_function);

                for (int i = 1; i < no_bunches_of_learners; ++i) {

                    loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                      ind_delta, params.loss_function);

                    type = learners[(i - 1) * no_cores + 1]->get_type();
                    switch (type) {
                        case 't': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[learner_id]),
                                                    train_x, train_y, ind_delta, *prediction_train_data,
                                                    input_monotonic_constraints, params.loss_function,
                                                    thread_row_inds, no_train_sample,
                                                    thread_col_inds, no_candidate_feature);
                                tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[learner_id]));
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    {
                                        Tree_info<T> tree_info(dynamic_cast<Tree_node<T> *>
                                                               (learners[learner_id]));
                                        tree_info.print_to_file("trees.txt", learner_id);
                                    }
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'l': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>
                                                                 (learners[learner_id]),
                                                                 train_x, ind_delta,
                                                                 thread_row_inds, no_train_sample,
                                                                 thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'k': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #pragma omp critical
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                kmeans2d_trainer->train(dynamic_cast<Kmeans2d<T> *> (learners[learner_id]),
                                                        train_x, ind_delta, params.loss_function,
                                                        thread_row_inds, no_train_sample,
                                                        thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 's': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                splines_trainer->train(dynamic_cast<Splines<T> *>
                                                       (learners[learner_id]),
                                                       train_x, ind_delta,
                                                       thread_row_inds, no_train_sample,
                                                       thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'n': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                neural_network_trainer->train(dynamic_cast<Neural_network<T> *>
                                                              (learners[learner_id]),
                                                              train_x, ind_delta,
                                                              thread_row_inds, no_train_sample,
                                                              thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        default: {
                            std::cout << "Wrong learner type: " << type << std::endl;
                            throw std::invalid_argument("Specified learner does not exist.");
                        }
                    }

                    if (!(i % params.freq_showing_loss_on_test)) {
                        test_loss_record[i / params.freq_showing_loss_on_test] =
                                loss_function.loss(test_y, prediction_test_data, params.loss_function);
                    }

                }

            }
            else {

                std::printf(".");;

                loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                  ind_delta, params.loss_function);
                mean_trainer->train(dynamic_cast<Global_mean<T> *>(learners[0]),
                                    train_x, ind_delta, *prediction_train_data, params.loss_function);
                learners[0]->predict(train_x, *prediction_train_data);
                learners[0]->predict(test_x, prediction_test_data);

                test_loss_record[0] = loss_function.loss(
                        test_y, prediction_test_data, params.loss_function);

                for (int i = 1; i < no_bunches_of_learners; ++i) {

                    loss_function.calculate_ind_delta(train_y, *prediction_train_data,
                                                      ind_delta, params.loss_function);

                    type = learners[(i - 1) * no_cores + 1]->get_type();
                    switch (type) {
                        case 't': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[learner_id]),
                                                    train_x, train_y, ind_delta, *prediction_train_data,
                                                    input_monotonic_constraints, params.loss_function,
                                                    thread_row_inds, no_train_sample,
                                                    thread_col_inds, no_candidate_feature);
                                tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[learner_id]));
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'l': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>
                                                                 (learners[learner_id]),
                                                                 train_x, ind_delta,
                                                                 thread_row_inds, no_train_sample,
                                                                 thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'k': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                kmeans2d_trainer->train(dynamic_cast<Kmeans2d<T> *> (learners[learner_id]),
                                                        train_x, ind_delta, params.loss_function,
                                                        thread_row_inds, no_train_sample,
                                                        thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 's': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                splines_trainer->train(dynamic_cast<Splines<T> *>
                                                       (learners[learner_id]),
                                                       train_x, ind_delta,
                                                       thread_row_inds, no_train_sample,
                                                       thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        case 'n': {
                            #ifdef _OMP
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                                {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int *thread_row_inds = new int[n_samples], *thread_col_inds = new int[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                neural_network_trainer->train(dynamic_cast<Neural_network<T> *>
                                                              (learners[learner_id]),
                                                              train_x, ind_delta,
                                                              thread_row_inds, no_train_sample,
                                                              thread_col_inds, no_candidate_feature);
                                #ifdef _OMP
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef _OMP
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }

                                delete[] thread_row_inds;
                                delete[] thread_col_inds;

                            }
                            break;
                        }
                        default: {
                            std::cout << "Wrong learner type: " << type << std::endl;
                            throw std::invalid_argument("Specified learner does not exist.");
                        }
                    }

                    if (!(i % params.freq_showing_loss_on_test)) {
                        test_loss_record[i / params.freq_showing_loss_on_test] =
                                loss_function.loss(test_y, prediction_test_data, params.loss_function);
                        std::cout << std::endl
                                  << '(' << i / params.freq_showing_loss_on_test << ')'
                                  << " Loss on test set: "
                                  << test_loss_record[i / params.freq_showing_loss_on_test]
                                  << std::endl << std::endl;
                    }

                }

            }
        }

        loss_function.mean_function(*prediction_train_data, params.loss_function);

        std::cout << std::endl << "Losses on Test Set: " << std::endl;
        for (int i = 0; i < no_bunches_of_learners / params.freq_showing_loss_on_test; ++i)
            std::cout << "(" << i << ") " << test_loss_record[i] << ' ';
        std::cout << std::endl;

        delete[] row_inds;
        delete[] col_inds;
        delete[] seeds;
        row_inds = nullptr, col_inds = nullptr, seeds = nullptr;

    }

    template<typename T>
    void DBM<T>::predict(const Matrix<T> &data_x, Matrix<T> &predict_y) {

        int data_height = data_x.get_height();

        #ifdef _DEBUG_MODEL
            assert(total_no_feature == data_x.get_width() &&
                           data_height == predict_y.get_height() &&
                           predict_y.get_width() == 1);
        #endif

        for (int i = 0; i < data_height; ++i)
            predict_y[i][0] = 0;

        if (learners[0]->get_type() == 'm') {
            learners[0]->predict(data_x, predict_y);
        }
        else {
            learners[0]->predict(data_x, predict_y, params.shrinkage);
        }

        for (int i = 1; i < (no_bunches_of_learners - 1) * no_cores + 1; ++i) {

            learners[i]->predict(data_x, predict_y, params.shrinkage);

        }

        loss_function.mean_function(predict_y, params.loss_function);

    }

    template <typename T>
    Matrix<T> &DBM<T>::predict(const Matrix<T> &data_x) {
        if(prediction != nullptr) {
            delete prediction;
            prediction = nullptr;
        }
        prediction = new Matrix<T>(data_x.get_height(), 1, 0);
        predict(data_x, *prediction);
        return *prediction;
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

    template <typename T>
    Matrix<T> &DBM<T>::partial_dependence_plot(const Matrix<T> &data,
                                              const int &predictor_ind) {
        #ifdef _DEBUG_MODEL
            assert(total_no_feature == data.get_width());
        #endif

        Matrix<T> modified_data = copy(data);

        int data_height = data.get_height(),
                *row_inds = new int[data_height],
                resampling_size = int(data_height * params.resampling_portion);
        for(int i = 0; i < data_height; ++i)
            row_inds[i] = i;

        if(pdp_result != nullptr)
            delete pdp_result;
        pdp_result = new Matrix<T>(params.no_x_ticks, 4, 0);
        T predictor_min, predictor_max, standard_dev;

        int total_no_resamplings = params.no_resamplings * no_cores;
        Matrix<T> bootstraping(params.no_x_ticks, total_no_resamplings, 0);

        #ifdef _OMP

            omp_set_num_threads(no_cores);

            unsigned int *seeds = new unsigned int[total_no_resamplings];
            for(int i = 0; i < total_no_resamplings; ++i)
                seeds[i] = (unsigned int)(std::rand() / (RAND_MAX / 1e5));

        #else

            Matrix<T> resampling_prediction(resampling_size, 1, 0);

        #endif

        Time_measurer timer;
        std::cout << std::endl
                  << "Started bootstraping..."
                  << std::endl;

        predictor_min = data.get_col_min(predictor_ind),
                predictor_max = data.get_col_max(predictor_ind);

        for(int i = 0; i < params.no_x_ticks; ++i) {

            pdp_result->assign(i, 0, predictor_min + i * (predictor_max - predictor_min) / (params.no_x_ticks - 1));

            for(int j = 0; j < data_height; ++j)
                modified_data.assign(j, predictor_ind, pdp_result->get(i, 0));

            for(int j = 0; j < params.no_resamplings; ++j) {

                #ifdef _OMP
                #pragma omp parallel default(shared)
                {

                    int resampling_id = no_cores * j + omp_get_thread_num();

                #else
                {
                #endif

                    #ifdef _OMP

                        int *thread_row_inds = new int[data_height];
                        std::copy(row_inds, row_inds + data_height, thread_row_inds);
                        shuffle(thread_row_inds, data_height, seeds[resampling_id]);

                        Matrix<T> *resampling_prediction = new Matrix<T>(resampling_size, 1, 0);
                        predict(modified_data.rows(thread_row_inds, resampling_size), *resampling_prediction);

                        bootstraping.assign(i, resampling_id, resampling_prediction->col_average(0));

                        delete[] thread_row_inds;
                        delete resampling_prediction;

                    #else

                        shuffle(row_inds, data_height);

                        resampling_prediction.clear();
                        predict(modified_data.rows(row_inds, resampling_size), resampling_prediction);

                        bootstraping.assign(i, j, resampling_prediction.col_average(0));

                    #endif

                }

            }

            pdp_result->assign(i, 1, bootstraping.row_average(i));

            standard_dev = bootstraping.row_std(i);

            pdp_result->assign(i, 2, pdp_result->get(i, 1) - params.ci_bandwidth / 2.0 * standard_dev);
            pdp_result->assign(i, 3, pdp_result->get(i, 1) + params.ci_bandwidth / 2.0 * standard_dev);

        }

        #ifdef _OMP
            delete[] seeds;
            seeds = nullptr;
        #endif

        delete[] row_inds;
        row_inds = nullptr;

        return *pdp_result;

    }

    template <typename T>
    Matrix<T> &DBM<T>::partial_dependence_plot(const Matrix<T> &data,
                                              const int &predictor_ind,
                                              const T &x_tick_min,
                                              const T &x_tick_max) {
        #ifdef _DEBUG_MODEL
            assert(total_no_feature == data.get_width());
        #endif

        Matrix<T> modified_data = copy(data);

        int data_height = data.get_height(),
                *row_inds = new int[data_height],
                resampling_size = int(data_height * params.resampling_portion);
        for(int i = 0; i < data_height; ++i)
            row_inds[i] = i;

        if(pdp_result != nullptr)
            delete pdp_result;
        pdp_result = new Matrix<T>(params.no_x_ticks, 4, 0);
        T predictor_min, predictor_max, standard_dev;

        int total_no_resamplings = params.no_resamplings * no_cores;
        Matrix<T> bootstraping(params.no_x_ticks, total_no_resamplings, 0);

        #ifdef _OMP

            omp_set_num_threads(no_cores);

            unsigned int *seeds = new unsigned int[total_no_resamplings];
            for(int i = 0; i < total_no_resamplings; ++i)
                seeds[i] = (unsigned int)(std::rand() / (RAND_MAX / 1e5));

        #else

            Matrix<T> resampling_prediction(resampling_size, 1, 0);

        #endif

        Time_measurer timer;
        std::cout << std::endl
                  << "Started bootstraping..."
                  << std::endl;

        predictor_min = x_tick_min,
                predictor_max = x_tick_max;

        for(int i = 0; i < params.no_x_ticks; ++i) {

            pdp_result->assign(i, 0, predictor_min + i * (predictor_max - predictor_min) / (params.no_x_ticks - 1));

            for(int j = 0; j < data_height; ++j)
                modified_data.assign(j, predictor_ind, pdp_result->get(i, 0));

            for(int j = 0; j < params.no_resamplings; ++j) {

                #ifdef _OMP
                #pragma omp parallel default(shared)
                {

                    int resampling_id = no_cores * j + omp_get_thread_num();

                #else
                {
                #endif

                    #ifdef _OMP

                        int *thread_row_inds = new int[data_height];
                        std::copy(row_inds, row_inds + data_height, thread_row_inds);
                        shuffle(thread_row_inds, data_height, seeds[resampling_id]);

                        Matrix<T> *resampling_prediction = new Matrix<T>(resampling_size, 1, 0);
                        predict(modified_data.rows(thread_row_inds, resampling_size), *resampling_prediction);

                        bootstraping.assign(i, resampling_id, resampling_prediction->col_average(0));

                        delete[] thread_row_inds;
                        delete resampling_prediction;

                    #else

                        shuffle(row_inds, data_height);

                        resampling_prediction.clear();
                        predict(modified_data.rows(row_inds, resampling_size), resampling_prediction);

                        bootstraping.assign(i, j, resampling_prediction.col_average(0));

                    #endif

                }

            }

            pdp_result->assign(i, 1, bootstraping.row_average(i));

            standard_dev = bootstraping.row_std(i);

            pdp_result->assign(i, 2, pdp_result->get(i, 1) - params.ci_bandwidth / 2.0 * standard_dev);
            pdp_result->assign(i, 3, pdp_result->get(i, 1) + params.ci_bandwidth / 2.0 * standard_dev);

        }

        #ifdef _OMP
            delete[] seeds;
            seeds = nullptr;
        #endif

        delete[] row_inds;
        row_inds = nullptr;

        return *pdp_result;

    }

    template <typename T>
    Matrix<T> &DBM<T>::statistical_significance(const Matrix<T> &data) {

        #ifdef _DEBUG_MODEL
            assert(total_no_feature == data.get_width());
        #endif

        Matrix<T> *modified_data = nullptr;

        int data_height = data.get_height(),
                data_width = data.get_width(),
                *row_inds = new int[data_height],
                resampling_size = int(data_height * params.resampling_portion);
        for(int i = 0; i < data_height; ++i)
            row_inds[i] = i;

        T predictor_min,
                predictor_max,
                x_tick;

        Matrix<T> x_ticks(total_no_feature, params.no_x_ticks, 0);
        Matrix<T> means(total_no_feature, params.no_x_ticks, 0);
        Matrix<T> stds(total_no_feature, params.no_x_ticks, 0);

        int total_no_resamplings = params.no_resamplings * no_cores;
        Matrix<T> bootstraping(params.no_x_ticks, total_no_resamplings, 0);

        ss_result = new Matrix<T>(total_no_feature, 1, 0);

        const int no_probs = 30;
        T z_scores[no_probs] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
                          1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2,
                          2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3};
        T probs[no_probs] = {0.0796, 0.1586, 0.23582, 0.31084, 0.38292, 0.4515, 0.51608, 0.57628, 0.63188, 0.68268,
                       0.72866, 0.76986, 0.8064, 0.83848, 0.86638, 0.8904, 0.91086, 0.92814, 0.94256, 0.9545,
                       0.96428, 0.9722, 0.97856, 0.9836, 0.98758, 0.99068, 0.99306, 0.99488, 0.99626, 0.9973};

        #ifdef _OMP

            omp_set_num_threads(no_cores);

            unsigned int *seeds = new unsigned int[total_no_resamplings];
            for(int i = 0; i < total_no_resamplings; ++i)
                seeds[i] = (unsigned int)(std::rand() / (RAND_MAX / 1e5));

        #else

            Matrix<T> resampling_prediction(resampling_size, 1, 0);

        #endif

        Time_measurer timer;
        std::cout << std::endl
                  << "Started bootstraping..."
                  << std::endl;

        for(int i = 0; i < total_no_feature; ++i) {

            predictor_min = data.get_col_min(i),
                    predictor_max = data.get_col_max(i);

            modified_data = new Matrix<T>(data_height, data_width, 0);
            copy(data, *modified_data);

            for(int j = 0; j < params.no_x_ticks; ++j) {

                x_tick = predictor_min + j * (predictor_max - predictor_min) / (params.no_x_ticks - 1);

                for(int k = 0; k < data_height; ++k)
                    modified_data->assign(k, i, x_tick);

                for(int k = 0; k < params.no_resamplings; ++k) {

                    #ifdef _OMP
                    #pragma omp parallel default(shared)
                    {

                        int resampling_id = no_cores * k + omp_get_thread_num();

                    #else
                    {
                    #endif

                    #ifdef _OMP

                        int *thread_row_inds = new int[data_height];
                        std::copy(row_inds, row_inds + data_height, thread_row_inds);
                        shuffle(thread_row_inds, data_height, seeds[resampling_id]);

                        Matrix<T> *resampling_prediction = new Matrix<T>(resampling_size, 1, 0);
                        predict(modified_data->rows(thread_row_inds, resampling_size), *resampling_prediction);

                        bootstraping.assign(j, resampling_id, resampling_prediction->col_average(0));

                        delete[] thread_row_inds;
                        delete resampling_prediction;

                    #else

                        shuffle(row_inds, data_height);

                        resampling_prediction.clear();
                        predict(modified_data->rows(row_inds, resampling_size), resampling_prediction);

                        bootstraping.assign(j, k, resampling_prediction.col_average(0));

                    #endif

                    }

                }

                x_ticks.assign(i, j, x_tick);
                means.assign(i, j, bootstraping.row_average(j));
                stds.assign(i, j, bootstraping.row_std(j));
            }

            delete modified_data;

            std::cout << "Predictor ( " << i
                      << " ) --> bootstraping completed..."
                      << std::endl;

        }

        if(params.save_files) {
            x_ticks.print_to_file("x_ticks.txt");
            means.print_to_file("means.txt");
            stds.print_to_file("stds.txt");
        }

        T largest_lower_ci, smallest_higher_ci;

        for(int i = 0; i < total_no_feature; ++i) {

            int j;
            for(j = 0; j < no_probs; ++j) {

                largest_lower_ci = std::numeric_limits<T>::lowest(),
                        smallest_higher_ci = std::numeric_limits<T>::max();

                for(int k = 0; k < params.no_x_ticks; ++k) {

                    if(means.get(i, k) - z_scores[j] * stds.get(i, k) > largest_lower_ci) {
                        largest_lower_ci = means.get(i, k) - z_scores[j] * stds.get(i, k);
                    }
                    if(means.get(i, k) + z_scores[j] * stds.get(i, k) < smallest_higher_ci) {
                        smallest_higher_ci = means.get(i, k) + z_scores[j] * stds.get(i, k);
                    }

                }

                if(largest_lower_ci < smallest_higher_ci)
                    break;

            }

            ss_result->assign(i, 0, probs[std::max(j - 1, 0)]);

        }

        #ifdef _OMP
            delete[] seeds;
            seeds = nullptr;
        #endif

        delete[] row_inds;
        row_inds = nullptr;

        return *ss_result;

    }

}

namespace dbm {

    template<typename T>
    void save_dbm(const DBM<T> *dbm, std::ofstream &out) {

        out << dbm->no_bunches_of_learners << ' '
            << dbm->no_cores << ' '
            << dbm->no_candidate_feature << ' '
            << dbm->no_train_sample << ' '
            << dbm->params.loss_function << ' '
            << dbm->params.shrinkage << ' '
            << dbm->total_no_feature << ' '
            << std::endl;
        char type;
        for (int i = 0; i < (dbm->no_bunches_of_learners - 1) * dbm->no_cores + 1; ++i) {
            type = dbm->learners[i]->get_type();
            switch (type) {

                case 'm': {
                    out << "== Mean " << i << " ==" << std::endl;
                    dbm::save_global_mean(dynamic_cast<Global_mean<T> *>(dbm->learners[i]), out);
                    out << "== End of Mean " << i << " ==" << std::endl;
                    break;
                }

                case 't': {
                    out << "== Tree " << i << " ==" << std::endl;
                    dbm::save_tree_node(dynamic_cast<Tree_node<T> *>(dbm->learners[i]), out);
                    out << "== End of Tree " << i << " ==" << std::endl;
                    break;
                }

                case 'l': {
                    out << "== LinReg " << i << " ==" << std::endl;
                    dbm::save_linear_regression(dynamic_cast<Linear_regression<T> *>(dbm->learners[i]), out);
                    out << "== End of LinReg " << i << " ==" << std::endl;
                    break;
                }

                case 'k': {
                    out << "== Kmeans " << i << " ==" << std::endl;
                    dbm::save_kmeans2d(dynamic_cast<Kmeans2d<T> *>(dbm->learners[i]), out);
                    out << "== End of Kmeans " << i << " ==" << std::endl;
                    break;
                }

                case 's': {
                    out << "== Splines " << i << " ==" << std::endl;
                    dbm::save_splines(dynamic_cast<Splines<T> *>(dbm->learners[i]), out);
                    out << "== End of Splines " << i << " ==" << std::endl;
                    break;
                }


                case 'n': {
                    out << "== NN " << i << " ==" << std::endl;
                    dbm::save_neural_network(dynamic_cast<Neural_network<T> *>(dbm->learners[i]), out);
                    out << "== End of NN " << i << " ==" << std::endl;
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

        #ifdef _DEBUG_MODEL
            assert(count == 7);
        #endif

        dbm = new DBM<T>(std::stoi(words[0]), std::stoi(words[1]), std::stoi(words[2]), std::stoi(words[3]), std::stoi(words[6]));
        dbm->set_loss_function_and_shrinkage(words[4].front(), T(std::stod(words[5])));

        Tree_node<T> *temp_tree_ptr;
        Global_mean<T> *temp_mean_ptr;
        Linear_regression<T> *temp_linear_regression_ptr;
        Neural_network<T> *temp_neural_network_ptr;
        Splines<T> *temp_splines_ptr;
        Kmeans2d<T> *temp_kmeans2d_ptr;

        char type;

        for (int i = 0; i < (dbm->no_bunches_of_learners - 1) * dbm->no_cores + 1; ++i) {

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

                case 'K': {
                    temp_kmeans2d_ptr = nullptr;
                    load_kmeans2d(in, temp_kmeans2d_ptr);
                    dbm->learners[i] = temp_kmeans2d_ptr;

                    // skip the end line
                    std::getline(in, line);

                    break;
                }

                case 'S': {
                    temp_splines_ptr = nullptr;
                    load_splines(in, temp_splines_ptr);
                    dbm->learners[i] = temp_splines_ptr;

                    // skip the end line
                    std::getline(in, line);

                    break;
                }

                case 'N': {
                    temp_neural_network_ptr = nullptr;
                    load_neural_network(in, temp_neural_network_ptr);
                    dbm->learners[i] = temp_neural_network_ptr;

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

    template<typename T>
    void DBM<T>::save_dbm_to(const std::string &file_name) {

        std::ofstream out(file_name);
        dbm::save_dbm(this, out);

    }

    template<typename T>
    void DBM<T>::load_dbm_from(const std::string &file_name) {

        for (int i = 0; i < (no_bunches_of_learners - 1) * no_cores + 1; ++i) {
            delete learners[i];
        }
        delete[] learners;

        if(prediction_train_data != nullptr)
            delete prediction_train_data;

        if(test_loss_record != nullptr)
            delete test_loss_record;

        if(pdp_result != nullptr)
            delete pdp_result;

        if(ss_result != nullptr)
            delete ss_result;

        std::ifstream in(file_name);

        std::string line;
        std::getline(in, line);

        std::string words[100];
        int count = split_into_words(line, words);

        #ifdef _DEBUG_MODEL
            assert(count == 7);
        #endif

        no_bunches_of_learners = std::stoi(words[0]);
        no_cores = std::stoi(words[1]);
        total_no_feature = std::stoi(words[6]);
        no_candidate_feature = std::stoi(words[2]);
        no_train_sample = std::stoi(words[3]);

        loss_function = Loss_function<T>(params);
        set_loss_function_and_shrinkage(words[4].front(), T(std::stod(words[5])));

        learners = new Base_learner<T> *[(no_bunches_of_learners - 1) * no_cores + 1];

        for (int i = 0; i < (no_bunches_of_learners - 1) * no_cores + 1; ++i) {
            learners[i] = nullptr;
        }

        Tree_node<T> *temp_tree_ptr;
        Global_mean<T> *temp_mean_ptr;
        Linear_regression<T> *temp_linear_regression_ptr;
        Neural_network<T> *temp_neural_network_ptr;
        Splines<T> *temp_splines_ptr;
        Kmeans2d<T> *temp_kmeans2d_ptr;

        char type;

        for (int i = 0; i < (no_bunches_of_learners - 1) * no_cores + 1; ++i) {

            line.clear();
            std::getline(in, line);

            split_into_words(line, words);

            type = words[1].front();
            switch (type) {

                case 'M': {
                    temp_mean_ptr = nullptr;
                    load_global_mean(in, temp_mean_ptr);
                    learners[i] = temp_mean_ptr;

                    // skip the end line
                    std::getline(in, line);

                    break;
                }

                case 'T': {
                    temp_tree_ptr = nullptr;
                    load_tree_node(in, temp_tree_ptr);
                    learners[i] = temp_tree_ptr;

                    // skip the end line
                    std::getline(in, line);

                    break;
                }

                case 'L': {
                    temp_linear_regression_ptr = nullptr;
                    load_linear_regression(in, temp_linear_regression_ptr);
                    learners[i] = temp_linear_regression_ptr;

                    // skip the end line
                    std::getline(in, line);

                    break;
                }

                case 'K': {
                    temp_kmeans2d_ptr = nullptr;
                    load_kmeans2d(in, temp_kmeans2d_ptr);
                    learners[i] = temp_kmeans2d_ptr;

                    // skip the end line
                    std::getline(in, line);

                    break;
                }

                case 'S': {
                    temp_splines_ptr = nullptr;
                    load_splines(in, temp_splines_ptr);
                    learners[i] = temp_splines_ptr;

                    // skip the end line
                    std::getline(in, line);

                    break;
                }

                case 'N': {
                    temp_neural_network_ptr = nullptr;
                    load_neural_network(in, temp_neural_network_ptr);
                    learners[i] = temp_neural_network_ptr;

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



