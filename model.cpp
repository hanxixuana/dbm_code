//
// Created by xixuan on 10/10/16.
//

#include "model.h"

#include <cassert>
#include <iostream>

#ifdef __OMP__
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
    DBM<T>::DBM(int no_bunches_of_learners, int no_cores, int no_candidate_feature, int no_train_sample):
            no_bunches_of_learners(no_bunches_of_learners), no_cores(no_cores),
            no_candidate_feature(no_candidate_feature), no_train_sample(no_train_sample),
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
        kmeans_trainer = nullptr;
    }

    template<typename T>
    DBM<T>::DBM(const std::string &param_string) : loss_function(Loss_function<T>(params)) {
        std::srand((unsigned int)std::time(NULL));

        params = set_params(param_string);

        no_cores = params.no_cores;
        no_bunches_of_learners = params.no_bunches_of_learners;
        no_candidate_feature = params.no_candidate_feature;
        no_train_sample = params.no_train_sample;

        learners = new Base_learner<T> *[(no_bunches_of_learners - 1) * no_cores + 1];

        learners[0] = new Global_mean<T>;

        #ifdef __OMP__
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
        no_cores = 1;
        #endif

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
                    learners[no_cores * (i - 1) + j + 1] = new Kmeans<T>(no_candidate_feature,
                                                                         params.no_centroids,
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
        kmeans_trainer = new Kmeans_trainer<T>(params);
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
        delete kmeans_trainer;

        if(prediction_train_data != nullptr)
            delete prediction_train_data;

        if(test_loss_record != nullptr)
            delete test_loss_record;

    }

    template<typename T>
    void DBM<T>::train(const Matrix<T> &train_x, const Matrix<T> &train_y, const int * input_monotonic_constaints) {
        /*
         * This function is not tested.
         */

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

        #ifdef __OMP__
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
                            #ifdef __OMP__
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

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[learner_id]),
                                                    train_x, train_y, ind_delta, *prediction_train_data,
                                                    monotonic_constraints, params.loss_function,
                                                    thread_row_inds, no_train_sample,
                                                    thread_col_inds, no_candidate_feature);
                                tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[learner_id]));
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    {
                                        Tree_info<T> tree_info(dynamic_cast<Tree_node<T> *>
                                                               (learners[learner_id]));
                                        tree_info.print_to_file("trees.txt", learner_id);
                                    }
                                }
                            }
                            break;
                        }
                        case 'l': {
                            #ifdef __OMP__
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                            {
                                int thread_id = 0, learner_id = 1;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Linear Regression at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature);

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>
                                                                 (learners[learner_id]),
                                                                 train_x, ind_delta,
                                                                 thread_row_inds, no_train_sample,
                                                                 thread_col_inds, no_candidate_feature);
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                #ifdef __OMP__
                                    #pragma omp critical
                                #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }
                            }
                            break;
                        }
                        case 'k': {
                            #ifdef __OMP__
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
                                                    "Kmeans at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of centroids: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature, params.no_centroids);

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                kmeans_trainer->train(dynamic_cast<Kmeans<T> *> (learners[learner_id]),
                                                      train_x, ind_delta, params.loss_function,
                                                      thread_row_inds, no_train_sample,
                                                      thread_col_inds, no_candidate_feature);
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }
                            }
                            break;
                        }
                        case 's': {
                            #ifdef __OMP__
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

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                splines_trainer->train(dynamic_cast<Splines<T> *>
                                                       (learners[learner_id]),
                                                       train_x, ind_delta,
                                                       thread_row_inds, no_train_sample,
                                                       thread_col_inds, no_candidate_feature);
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }
                            }
                            break;
                        }
                        case 'n': {
                            #ifdef __OMP__
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

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                neural_network_trainer->train(dynamic_cast<Neural_network<T> *>
                                                              (learners[learner_id]),
                                                              train_x, ind_delta,
                                                              thread_row_inds, no_train_sample,
                                                              thread_col_inds, no_candidate_feature);
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }
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
                            #ifdef __OMP__
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                            {
                                int thread_id = 0, learner_id = 1;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Tree at %p "
                                                    "number of samples: %d "
                                                    "max_depth: %d "
                                                    "no_candidate_split_point: %d...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, params.max_depth, params.no_candidate_split_point);

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[learner_id]),
                                                    train_x, train_y, ind_delta, *prediction_train_data,
                                                    monotonic_constraints, params.loss_function,
                                                    thread_row_inds, no_train_sample,
                                                    thread_col_inds, no_candidate_feature);
                                tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[learner_id]));
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                #ifdef __OMP__
                                    #pragma omp critical
                                #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }
                            }
                            break;
                        }
                        case 'l': {
                            #ifdef __OMP__
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                            {
                                int thread_id = 0, learner_id = 1;
                            #endif
                                std::printf("Learner (%c) No. %d -> "
                                                    "Training Linear Regression at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature);

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>
                                                                 (learners[learner_id]),
                                                                 train_x, ind_delta,
                                                                 thread_row_inds, no_train_sample,
                                                                 thread_col_inds, no_candidate_feature);
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                #ifdef __OMP__
                                    #pragma omp critical
                                #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }
                            }
                            break;
                        }
                        case 'k': {
                            #ifdef __OMP__
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
                                                    "Kmeans at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of centroids: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature, params.no_centroids);

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                kmeans_trainer->train(dynamic_cast<Kmeans<T> *> (learners[learner_id]),
                                                      train_x, ind_delta, params.loss_function,
                                                      thread_row_inds, no_train_sample,
                                                      thread_col_inds, no_candidate_feature);
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }
                            }
                            break;
                        }
                        case 's': {
                            #ifdef __OMP__
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

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                splines_trainer->train(dynamic_cast<Splines<T> *>
                                                       (learners[learner_id]),
                                                       train_x, ind_delta,
                                                       thread_row_inds, no_train_sample,
                                                       thread_col_inds, no_candidate_feature);
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }
                            }
                            break;
                        }
                        case 'n': {
                            #ifdef __OMP__
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

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                neural_network_trainer->train(dynamic_cast<Neural_network<T> *>
                                                              (learners[learner_id]),
                                                              train_x, ind_delta,
                                                              thread_row_inds, no_train_sample,
                                                              thread_col_inds, no_candidate_feature);
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }
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
                            #ifdef __OMP__
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                            {
                                int thread_id = 0, learner_id = 1;
                            #endif
                                std::printf(".");

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[learner_id]),
                                                    train_x, train_y, ind_delta, *prediction_train_data,
                                                    monotonic_constraints, params.loss_function,
                                                    thread_row_inds, no_train_sample,
                                                    thread_col_inds, no_candidate_feature);
                                tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[learner_id]));
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                #ifdef __OMP__
                                    #pragma omp critical
                                #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                #ifdef __OMP__
                                    #pragma omp critical
                                #endif
                                    {
                                        Tree_info<T> tree_info(dynamic_cast<Tree_node<T> *>
                                                               (learners[learner_id]));
                                        tree_info.print_to_file("trees.txt", learner_id);
                                    }
                                }
                            }
                            break;
                        }
                        case 'l': {
                            #ifdef __OMP__
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                            {
                                int thread_id = 0, learner_id = 1;
                            #endif
                                std::printf(".");

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>
                                                                 (learners[learner_id]),
                                                                 train_x, ind_delta,
                                                                 thread_row_inds, no_train_sample,
                                                                 thread_col_inds, no_candidate_feature);
                            #ifdef __OMP__
                                #pragma omp barrier
                            #endif
                                {
                            #ifdef __OMP__
                                    #pragma omp critical
                            #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }
                            }
                            break;
                        }
                        case 'k': {
                            #ifdef __OMP__
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

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                kmeans_trainer->train(dynamic_cast<Kmeans<T> *> (learners[learner_id]),
                                                      train_x, ind_delta, params.loss_function,
                                                      thread_row_inds, no_train_sample,
                                                      thread_col_inds, no_candidate_feature);
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }
                            }
                            break;
                        }
                        case 's': {
                            #ifdef __OMP__
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

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                splines_trainer->train(dynamic_cast<Splines<T> *>
                                                       (learners[learner_id]),
                                                       train_x, ind_delta,
                                                       thread_row_inds, no_train_sample,
                                                       thread_col_inds, no_candidate_feature);
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }
                            }
                            break;
                        }
                        case 'n': {
                            #ifdef __OMP__
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

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                neural_network_trainer->train(dynamic_cast<Neural_network<T> *>
                                                              (learners[learner_id]),
                                                              train_x, ind_delta,
                                                              thread_row_inds, no_train_sample,
                                                              thread_col_inds, no_candidate_feature);
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }
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
                            #ifdef __OMP__
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                            {
                                int thread_id = 0, learner_id = 1;
                            #endif
                                std::printf(".");

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[learner_id]),
                                                    train_x, train_y, ind_delta, *prediction_train_data,
                                                    monotonic_constraints, params.loss_function,
                                                    thread_row_inds, no_train_sample,
                                                    thread_col_inds, no_candidate_feature);
                                tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[learner_id]));
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                #ifdef __OMP__
                                    #pragma omp critical
                                #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }
                            }
                            break;
                        }
                        case 'l': {
                            #ifdef __OMP__
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                            {
                                int thread_id = 0, learner_id = 1;
                            #endif
                                std::printf(".");

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>
                                                                 (learners[learner_id]),
                                                                 train_x, ind_delta,
                                                                 thread_row_inds, no_train_sample,
                                                                 thread_col_inds, no_candidate_feature);
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                #ifdef __OMP__
                                    #pragma omp critical
                                #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }
                            }
                            break;
                        }
                        case 'k': {
                            #ifdef __OMP__
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
                                                    "Kmeans at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of centroids: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature, params.no_centroids);

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                kmeans_trainer->train(dynamic_cast<Kmeans<T> *> (learners[learner_id]),
                                                      train_x, ind_delta, params.loss_function,
                                                      thread_row_inds, no_train_sample,
                                                      thread_col_inds, no_candidate_feature);
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }
                            }
                            break;
                        }
                        case 's': {
                            #ifdef __OMP__
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

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                splines_trainer->train(dynamic_cast<Splines<T> *>
                                                       (learners[learner_id]),
                                                       train_x, ind_delta,
                                                       thread_row_inds, no_train_sample,
                                                       thread_col_inds, no_candidate_feature);
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }
                            }
                            break;
                        }
                        case 'n': {
                            #ifdef __OMP__
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

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                neural_network_trainer->train(dynamic_cast<Neural_network<T> *>
                                                              (learners[learner_id]),
                                                              train_x, ind_delta,
                                                              thread_row_inds, no_train_sample,
                                                              thread_col_inds, no_candidate_feature);
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                }
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
        delete row_inds, col_inds, seeds;
        row_inds = nullptr, col_inds = nullptr, seeds = nullptr;

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

        #ifdef __OMP__
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
                            #ifdef __OMP__
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                                #pragma omp critical
                            #else
                            {
                                int thread_id = 0, learner_id = 1;
                            #endif
                                    std::printf("Learner (%c) No. %d -> "
                                                        "Training Tree at %p "
                                                        "number of samples: %d "
                                                        "max_depth: %d "
                                                        "no_candidate_split_point: %d...\n",
                                                type, learner_id, learners[learner_id],
                                                no_train_sample, params.max_depth, params.no_candidate_split_point);

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[learner_id]),
                                                    train_x, train_y, ind_delta, *prediction_train_data,
                                                    monotonic_constraints, params.loss_function,
                                                    thread_row_inds, no_train_sample,
                                                    thread_col_inds, no_candidate_feature);
                                tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[learner_id]));
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                     params.shrinkage);
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                     params.shrinkage);
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    {
                                        Tree_info<T> tree_info(dynamic_cast<Tree_node<T> *>
                                                                     (learners[learner_id]));
                                        tree_info.print_to_file("trees.txt", learner_id);
                                    }
                                }
                            }
                            break;
                        }
                        case 'l': {
                            #ifdef __OMP__
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

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>
                                                                 (learners[learner_id]),
                                                                 train_x, ind_delta,
                                                                 thread_row_inds, no_train_sample,
                                                                 thread_col_inds, no_candidate_feature);
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                     params.shrinkage);
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                     params.shrinkage);
                                }
                            }
                            break;
                        }
                        case 'k': {
                            #ifdef __OMP__
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
                                                    "Kmeans at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of centroids: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature, params.no_centroids);

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                kmeans_trainer->train(dynamic_cast<Kmeans<T> *> (learners[learner_id]),
                                                       train_x, ind_delta, params.loss_function,
                                                       thread_row_inds, no_train_sample,
                                                       thread_col_inds, no_candidate_feature);
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }
                            }
                            break;
                        }
                        case 's': {
                            #ifdef __OMP__
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

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                splines_trainer->train(dynamic_cast<Splines<T> *>
                                                       (learners[learner_id]),
                                                       train_x, ind_delta,
                                                       thread_row_inds, no_train_sample,
                                                       thread_col_inds, no_candidate_feature);
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }
                            }
                            break;
                        }
                        case 'n': {
                            #ifdef __OMP__
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

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                neural_network_trainer->train(dynamic_cast<Neural_network<T> *>
                                                                 (learners[learner_id]),
                                                                 train_x, ind_delta,
                                                                 thread_row_inds, no_train_sample,
                                                                 thread_col_inds, no_candidate_feature);
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }
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
                            #ifdef __OMP__
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

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[learner_id]),
                                                    train_x, train_y, ind_delta, *prediction_train_data,
                                                    monotonic_constraints, params.loss_function,
                                                    thread_row_inds, no_train_sample,
                                                    thread_col_inds, no_candidate_feature);
                                tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[learner_id]));
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }
                            }
                            break;
                        }
                        case 'l': {
                            #ifdef __OMP__
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

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>
                                                                 (learners[learner_id]),
                                                                 train_x, ind_delta,
                                                                 thread_row_inds, no_train_sample,
                                                                 thread_col_inds, no_candidate_feature);
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }
                            }
                            break;
                        }
                        case 'k': {
                            #ifdef __OMP__
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
                                                    "Kmeans at %p "
                                                    "number of samples: %d "
                                                    "number of predictors: %d "
                                                    "number of centroids: %d ...\n",
                                            type, learner_id, learners[learner_id],
                                            no_train_sample, no_candidate_feature, params.no_centroids);

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                kmeans_trainer->train(dynamic_cast<Kmeans<T> *> (learners[learner_id]),
                                                      train_x, ind_delta, params.loss_function,
                                                      thread_row_inds, no_train_sample,
                                                      thread_col_inds, no_candidate_feature);
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }
                            }
                            break;
                        }
                        case 's': {
                            #ifdef __OMP__
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

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                splines_trainer->train(dynamic_cast<Splines<T> *>
                                                       (learners[learner_id]),
                                                       train_x, ind_delta,
                                                       thread_row_inds, no_train_sample,
                                                       thread_col_inds, no_candidate_feature);
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }
                            }
                            break;
                        }
                        case 'n': {
                            #ifdef __OMP__
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

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                neural_network_trainer->train(dynamic_cast<Neural_network<T> *>
                                                              (learners[learner_id]),
                                                              train_x, ind_delta,
                                                              thread_row_inds, no_train_sample,
                                                              thread_col_inds, no_candidate_feature);
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }
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
                            #ifdef __OMP__
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                            {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[learner_id]),
                                                    train_x, train_y, ind_delta, *prediction_train_data,
                                                    monotonic_constraints, params.loss_function,
                                                    thread_row_inds, no_train_sample,
                                                    thread_col_inds, no_candidate_feature);
                                tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[learner_id]));
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    {
                                        Tree_info<T> tree_info(dynamic_cast<Tree_node<T> *>
                                                               (learners[learner_id]));
                                        tree_info.print_to_file("trees.txt", learner_id);
                                    }
                                }
                            }
                            break;
                        }
                        case 'l': {
                            #ifdef __OMP__
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                            {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>
                                                                 (learners[learner_id]),
                                                                 train_x, ind_delta,
                                                                 thread_row_inds, no_train_sample,
                                                                 thread_col_inds, no_candidate_feature);
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }
                            }
                            break;
                        }
                        case 'k': {
                            #ifdef __OMP__
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

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                kmeans_trainer->train(dynamic_cast<Kmeans<T> *> (learners[learner_id]),
                                                      train_x, ind_delta, params.loss_function,
                                                      thread_row_inds, no_train_sample,
                                                      thread_col_inds, no_candidate_feature);
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }
                            }
                            break;
                        }
                        case 's': {
                            #ifdef __OMP__
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

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                splines_trainer->train(dynamic_cast<Splines<T> *>
                                                       (learners[learner_id]),
                                                       train_x, ind_delta,
                                                       thread_row_inds, no_train_sample,
                                                       thread_col_inds, no_candidate_feature);
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }
                            }
                            break;
                        }
                        case 'n': {
                            #ifdef __OMP__
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

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                neural_network_trainer->train(dynamic_cast<Neural_network<T> *>
                                                              (learners[learner_id]),
                                                              train_x, ind_delta,
                                                              thread_row_inds, no_train_sample,
                                                              thread_col_inds, no_candidate_feature);
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }
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
                            #ifdef __OMP__
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                            {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                tree_trainer->train(dynamic_cast<Tree_node<T> *>(learners[learner_id]),
                                                    train_x, train_y, ind_delta, *prediction_train_data,
                                                    monotonic_constraints, params.loss_function,
                                                    thread_row_inds, no_train_sample,
                                                    thread_col_inds, no_candidate_feature);
                                tree_trainer->prune(dynamic_cast<Tree_node<T> *>(learners[learner_id]));
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }
                            }
                            break;
                        }
                        case 'l': {
                            #ifdef __OMP__
                            #pragma omp parallel default(shared)
                            {
                                int thread_id = omp_get_thread_num(),
                                        learner_id = no_cores * (i - 1) + thread_id + 1;
                            #else
                            {
                                int thread_id = 0, learner_id = i;
                            #endif
                                std::printf(".");

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                linear_regression_trainer->train(dynamic_cast<Linear_regression<T> *>
                                                                 (learners[learner_id]),
                                                                 train_x, ind_delta,
                                                                 thread_row_inds, no_train_sample,
                                                                 thread_col_inds, no_candidate_feature);
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }
                            }
                            break;
                        }
                        case 'k': {
                            #ifdef __OMP__
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

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                kmeans_trainer->train(dynamic_cast<Kmeans<T> *> (learners[learner_id]),
                                                      train_x, ind_delta, params.loss_function,
                                                      thread_row_inds, no_train_sample,
                                                      thread_col_inds, no_candidate_feature);
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }
                            }
                            break;
                        }
                        case 's': {
                            #ifdef __OMP__
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

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                splines_trainer->train(dynamic_cast<Splines<T> *>
                                                       (learners[learner_id]),
                                                       train_x, ind_delta,
                                                       thread_row_inds, no_train_sample,
                                                       thread_col_inds, no_candidate_feature);
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }
                            }
                            break;
                        }
                        case 'n': {
                            #ifdef __OMP__
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

                                int thread_row_inds[n_samples], thread_col_inds[n_features];
                                std::copy(row_inds, row_inds + n_samples, thread_row_inds);
                                std::copy(col_inds, col_inds + n_features, thread_col_inds);
                                shuffle(thread_row_inds, n_samples, seeds[learner_id - 1]);
                                shuffle(thread_col_inds, n_features, seeds[learner_id - 1]);

                                neural_network_trainer->train(dynamic_cast<Neural_network<T> *>
                                                              (learners[learner_id]),
                                                              train_x, ind_delta,
                                                              thread_row_inds, no_train_sample,
                                                              thread_col_inds, no_candidate_feature);
                                #ifdef __OMP__
                                #pragma omp barrier
                                #endif
                                {
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(train_x, *prediction_train_data,
                                                                  params.shrinkage);
                                    #ifdef __OMP__
                                    #pragma omp critical
                                    #endif
                                    learners[learner_id]->predict(test_x, prediction_test_data,
                                                                  params.shrinkage);
                                }
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

        delete row_inds, col_inds, seeds;
        row_inds = nullptr, col_inds = nullptr, seeds = nullptr;

    }

    template<typename T>
    void DBM<T>::predict(const Matrix<T> &data_x, Matrix<T> &predict_y) {

        int data_height = data_x.get_height();

        #if _DEBUG_MODEL
            assert(data_height == predict_y.get_height() && predict_y.get_width() == 1);
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

        out << dbm->no_bunches_of_learners << ' '
            << dbm->no_cores << ' '
            << dbm->no_candidate_feature << ' '
            << dbm->no_train_sample << ' '
            << dbm->params.loss_function << ' '
            << dbm->params.shrinkage << ' '
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
                    dbm::save_kmeans(dynamic_cast<Kmeans<T> *>(dbm->learners[i]), out);
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

        #if _DEBUG_MODEL
            assert(count == 6);
        #endif

        dbm = new DBM<T>(std::stoi(words[0]), std::stoi(words[1]), std::stoi(words[2]), std::stoi(words[3]));
        dbm->set_loss_function_and_shrinkage(words[4].front(), T(std::stod(words[5])));

        Tree_node<T> *temp_tree_ptr;
        Global_mean<T> *temp_mean_ptr;
        Linear_regression<T> *temp_linear_regression_ptr;
        Neural_network<T> *temp_neural_network_ptr;
        Splines<T> *temp_splines_ptr;
        Kmeans<T> *temp_kmeans_ptr;

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
                    temp_kmeans_ptr = nullptr;
                    load_kmeans(in, temp_kmeans_ptr);
                    dbm->learners[i] = temp_kmeans_ptr;

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

}

namespace dbm {

    template void save_dbm<double>(const DBM<double> *dbm, std::ofstream &out);

    template void save_dbm<float>(const DBM<float> *dbm, std::ofstream &out);

    template void load_dbm<double>(std::ifstream &in, DBM<double> *&dbm);

    template void load_dbm<float>(std::ifstream &in, DBM<float> *&dbm);

}




