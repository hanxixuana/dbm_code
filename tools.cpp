//
// Created by xixuan on 10/10/16.
//

#include "tools.h"
#include "matrix.h"

#include <iostream>
#include <cassert>
#include <algorithm>

namespace dbm {

    void pause_at_end(std::string words) {
        std::cout << std::endl << std::endl << words << std::endl;
        std::string stop_at_end;
        std::getline(std::cin, stop_at_end);
    }

    Time_measurer::Time_measurer() {
        begin_time = std::clock();
        std::cout << std::endl
                  << "Timer at " << this
                  << " ---> Start recording time..."
                  << std::endl << std::endl;
    }

    Time_measurer::~Time_measurer() {
        end_time = std::clock();
        std::cout << std::endl
                  << "Timer at " << this
                  << " ---> " << "Elapsed Time: "
                  << double(end_time - begin_time) / CLOCKS_PER_SEC
                  << " seconds"
                  << std::endl << std::endl;
    }

    int split_into_words(const std::string &line, std::string *words, const char sep) {
        size_t prev = 0, next = 0;
        int n_words = 0;
        while ((next = line.find_first_of(sep, prev)) != std::string::npos) {
            if (next - prev != 0) {
                words[n_words] = line.substr(prev, next - prev);
                n_words += 1;
            }
            prev = next + 1;
        }
        if (prev < line.size()) {
            words[n_words] = line.substr(prev);
            n_words += 1;
        }
        return n_words;
    }

    template <typename T>
    void range(const T &start, const T &end, const int & number, T *result, const T &scaling) {
        #if _DEBUG_TOOLS
            assert(end > start);
        #endif
        T length = (end - start) / (number - 1);
        result[0] = start * scaling;
        for(int i = 1; i < number - 1; ++i)
            result[i] = (start + i * length) * scaling;
        result[number - 1] = end * scaling;
    }

    template<typename T>
    inline int middles(T *uniques, int no_uniques) {
        for (int i = 0; i < no_uniques - 1; ++i) uniques[i] = (uniques[i] + uniques[i + 1]) / 2.0;
        return no_uniques - 1;
    }

    template<typename T>
    inline void shuffle(T *values, int no_values, unsigned int seed) {
        std::srand(seed);
        std::random_shuffle(values, values + no_values);
    }

    template<typename T>
    void make_data(const std::string &file_name, int n_samples, int n_features, char data_type,
                   const int *sig_lin_inds, const T *coef_sig_lin, int n_sig_lin_feats,
                   const int *sig_quad_inds, const T *coef_sig_quad, int n_sig_quad_feats) {

        if (sig_lin_inds == NULL ||
                coef_sig_lin == NULL ||
                sig_quad_inds == NULL ||
                coef_sig_quad == NULL) {

            n_sig_lin_feats = 8, n_sig_quad_feats = 8;
            int lin_inds[] = {int(n_features * 0.1), int(n_features * 0.2),
                              int(n_features * 0.3), int(n_features * 0.4),
                              int(n_features * 0.5), int(n_features * 0.6),
                              int(n_features * 0.7), int(n_features * 0.8)};
            T coef_lin[] = {-10, 10, 1, 2, 5, -5, 10, -10};
            int quad_inds[] = {int(n_features * 0.15), int(n_features * 0.25),
                               int(n_features * 0.35), int(n_features * 0.45),
                               int(n_features * 0.55), int(n_features * 0.65),
                               int(n_features * 0.75), int(n_features * 0.85)};
            T coef_quad[] = {5, -3, -10, 4, 10, -5, 1, -2};


            dbm::Matrix<T> train_data(n_samples, n_features + 1);

            for (int i = 0; i < n_samples; ++i) {
                for (int j = 0; j < n_sig_lin_feats; ++j)
                    train_data[i][n_features] += coef_lin[j] * train_data[i][lin_inds[j]];
                for (int j = 0; j < n_sig_quad_feats; ++j)
                    train_data[i][n_features] +=
                            coef_quad[j] * std::pow(train_data[i][quad_inds[j]], T(2.0));
                switch (data_type) {

                    case 'n' : {
                        break;
                    }

                    case 'p' : {
                        train_data[i][n_features] =
                                std::round(std::max(train_data[i][n_features], T(0.001)));
                        break;
                    }

                    case 'b' : {
                        train_data[i][n_features] = train_data[i][n_features] < 0 ? 0 : 1;
                        break;
                    }

                    case 't': {
                        train_data[i][n_features] = std::max(train_data[i][n_features], T(0));
                        break;
                    }

                    default: {
                        throw std::invalid_argument("Specified data type does not exist.");
                    }

                }
            }

            train_data.print_to_file(file_name);
        } else {
            dbm::Matrix<T> train_data(n_samples, n_features + 1);

            for (int i = 0; i < n_samples; ++i) {
                for (int j = 0; j < n_sig_lin_feats; ++j)
                    train_data[i][n_features] +=
                            coef_sig_lin[j] * train_data[i][sig_lin_inds[j]];
                for (int j = 0; j < n_sig_quad_feats; ++j)
                    train_data[i][n_features] +=
                            coef_sig_quad[j] * std::pow(train_data[i][sig_quad_inds[j]], T(2.0));
                switch (data_type) {

                    case 'n' : {
                        break;
                    }

                    case 'p' : {
                        train_data[i][n_features] =
                                std::round(std::max(train_data[i][n_features], T(0.001)));
                        break;
                    }

                    case 'b' : {
                        train_data[i][n_features] = train_data[i][n_features] < 0 ? 0 : 1;
                        break;
                    }

                    case 't': {
                        train_data[i][n_features] = std::max(train_data[i][n_features], T(0));
                        break;
                    }

                    default: {
                        throw std::invalid_argument("Specified data type does not exist.");
                    }

                }
            }

            train_data.print_to_file(file_name);
        }

    }

    Params set_params(const std::string &param_string, const char delimiter) {

        std::string words[100];

        size_t prev = 0, next = 0;
        int count = 0;
        while ((next = param_string.find_first_of(delimiter, prev)) != std::string::npos) {
            if (next - prev != 0) {
                words[count] = param_string.substr(prev, next - prev);
                count += 1;
            }
            prev = next + 1;
        }

        if (prev < param_string.size()) {
            words[count] = param_string.substr(prev);
            count += 1;
        }

        #if _DEBUG_TOOLS
            assert(count % 2 == 0);
        #endif

        Params params;

        for (int i = 0; i < count / 2; ++i) {

            // DBM
            if (words[2 * i] == "no_bunches_of_learners")
                params.no_bunches_of_learners = std::stoi(words[2 * i + 1]);
            else if (words[2 * i] == "no_candidate_feature")
                params.no_candidate_feature = std::stoi(words[2 * i + 1]);
            else if (words[2 * i] == "no_train_sample")
                params.no_train_sample = std::stoi(words[2 * i + 1]);
            else if (words[2 * i] == "no_cores")
                params.no_cores = std::stoi(words[2 * i + 1]);

            else if (words[2 * i] == "loss_function")
                params.loss_function = words[2 * i + 1].front();

            else if (words[2 * i] == "display_training_progress")
                params.display_training_progress = std::stoi(words[2 * i + 1]) > 0;
            else if (words[2 * i] == "record_every_tree")
                params.record_every_tree = std::stoi(words[2 * i + 1]) > 0;
            else if (words[2 * i] == "freq_showing_loss_on_test")
                params.freq_showing_loss_on_test = std::stoi(words[2 * i + 1]);

            else if (words[2 * i] == "shrinkage")
                params.shrinkage = std::stod(words[2 * i + 1]);

            else if (words[2 * i] == "portion_for_trees")
                params.portion_for_trees = std::stod(words[2 * i + 1]);
            else if (words[2 * i] == "portion_for_lr")
                params.portion_for_lr = std::stod(words[2 * i + 1]);
            else if (words[2 * i] == "portion_for_nn")
                params.portion_for_nn = std::stod(words[2 * i + 1]);
            else if (words[2 * i] == "portion_for_s")
                params.portion_for_s = std::stod(words[2 * i + 1]);
            else if (words[2 * i] == "portion_for_k")
                params.portion_for_k = std::stod(words[2 * i + 1]);

            // tweedie
            else if (words[2 * i] == "tweedie_p")
                params.tweedie_p = std::stod(words[2 * i + 1]);

            // splines
            else if (words[2 * i] == "no_knot")
                params.no_knot = std::stoi(words[2 * i + 1]);

            // kmeans
            else if (words[2 * i] == "no_centroids")
                params.no_centroids = std::stoi(words[2 * i + 1]);
            else if (words[2 * i] == "kmeans_max_iteration")
                params.kmeans_max_iteration = std::stoi(words[2 * i + 1]);
            else if (words[2 * i] == "kmeans_tolerance")
                params.kmeans_tolerance = std::stod(words[2 * i + 1]);

            // neural networks
            else if (words[2 * i] == "no_hidden_neurons")
                params.no_hidden_neurons = std::stoi(words[2 * i + 1]);
            else if (words[2 * i] == "step_size")
                params.step_size = std::stod(words[2 * i + 1]);
            else if (words[2 * i] == "validate_portion")
                params.validate_portion = std::stod(words[2 * i + 1]);
            else if (words[2 * i] == "batch_size")
                params.batch_size = std::stoi(words[2 * i + 1]);
            else if (words[2 * i] == "nn_max_iteration")
                params.nn_max_iteration = std::stoi(words[2 * i + 1]);

            // CART
            else if (words[2 * i] == "max_depth")
                params.max_depth = std::stoi(words[2 * i + 1]);
            else if (words[2 * i] == "no_candidate_split_point")
                params.no_candidate_split_point = std::stoi(words[2 * i + 1]);

            // partial dependence plot
            else if (words[2 * i] == "no_x_ticks")
                params.no_x_ticks = std::stoi(words[2 * i + 1]);
            else if (words[2 * i] == "no_resamplings")
                params.no_resamplings = std::stoi(words[2 * i + 1]);
            else if (words[2 * i] == "resampling_portion")
                params.resampling_portion = std::stod(words[2 * i + 1]);
            else if (words[2 * i] == "ci_bandwidth")
                params.ci_bandwidth = std::stod(words[2 * i + 1]);

            // throw an exception
            else {
                throw std::invalid_argument("Specified parameter does not exist.");
            }

        }

        return params;

    }

}

// explicit instantiation of templated functions
namespace dbm {

    template void range(const double &start, const double &end, const int & number, double *result, const double &scaling);

    template void range(const float &start, const float &end, const int & number, float *result, const float &scaling);

    template int middles<float>(float *uniqes, int no_uniques);

    template int middles<double>(double *uniqes, int no_uniques);

    template void shuffle<int>(int *values, int no_values, unsigned int seed);

    template void shuffle<float>(float *values, int no_values, unsigned int seed);

    template void shuffle<double>(double *values, int no_values, unsigned int seed);

    template void make_data<double>(const std::string &file_name, int n_samples, int n_features, char data_type,
                                    const int *sig_lin_inds, const double *coef_sig_lin, int n_sig_lin_feats,
                                    const int *sig_quad_inds, const double *coef_sig_quad, int n_sig_quad_feats);

    template void make_data<float>(const std::string &file_name, int n_samples, int n_features, char data_type,
                                   const int *sig_lin_inds, const float *coef_sig_lin, int n_sig_lin_feats,
                                   const int *sig_quad_inds, const float *coef_sig_quad, int n_sig_quad_feats);


}





