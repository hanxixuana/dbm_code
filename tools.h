//
// Created by xixuan on 10/10/16.
//

#ifndef DBM_CODE_TOOLS_H
#define DBM_CODE_TOOLS_H

#ifndef _DEBUG_TOOLS
#define _DEBUG_TOOLS 1
#endif

#include <string>
#include <ctime>

/*
 * tools for parameters
 */

namespace dbm {

    struct Params {

        // DBM
        int no_bunches_of_learners = 100;
        int no_candidate_feature = 5;
        int no_train_sample = 50000;

        int no_cores = 0;

        char loss_function = 'n';

        bool display_training_progress = true;
        bool record_every_tree = true;
        int freq_showing_loss_on_test = 10;

        double shrinkage = 0.25;

        // portions should be summed to 1
        double portion_for_trees = 0.2;
        double portion_for_lr = 0.2;
        double portion_for_s = 0.2;
        double portion_for_k = 0.2;
        double portion_for_nn = 0.2;

        // tweedie: p should in (1, 2)
        double tweedie_p = 1.6;

        // splines
        int no_knot = 5;

        // kmeans
        int no_centroids = 15;
        int kmeans_max_iteration = 50;
        double kmeans_tolerance = 1e-2;

        // neural networks
        int no_hidden_neurons = 5;
        double step_size = 0.01;
        double validate_portion = 0.25;
        int batch_size = 100;
        int nn_max_iteration = 100;
        int no_rise_of_loss_on_validate = 20;

        // CART
        int max_depth = 5;
        int no_candidate_split_point = 5;

        // partial dependence plot
        int no_x_ticks = 10;
        int no_resamplings = 10;
        double resampling_portion = 0.5;
        double ci_bandwidth = 4;

    };

    Params set_params(const std::string &param_string, const char delimiter = ' ');
}

/*
 * other tools
 */

namespace dbm {

    // pause at end
    void pause_at_end(std::string words = "=============== \nEND OF PROGRAM. \n=============== \n ");

    // elapsed time measurer
    class Time_measurer {
    private:
        std::time_t begin_time;
        std::time_t end_time;
    public:
        Time_measurer();

        ~Time_measurer();
    };

    // split a line into words
    int split_into_words(const std::string &line, std::string *words, const char sep = ' ');

}

/*
 * tools for matrices
 */

namespace dbm {

    // return n-dimensional array of the form:
    // [start, start + (end - start) / (n - 1), start + 2 * (end - start) / (n - 1), ..., end]
    template <typename T>
    void range(const T &start, const T &end, const int & number, T *result, const T &scaling = 1);

    // find middle values in sorted uniques
    template<typename T>
    int middles(T *uniques, int no_uniques);

    // shuffle an array
    template<typename T>
    void shuffle(T *values, int no_values, unsigned int seed = (unsigned int)(std::time(NULL)));

    template<typename T>
    void make_data(const std::string &file_name, int n_samples = 100000, int n_features = 30, char data_type = 'n',
                   const int *sig_lin_inds = NULL, const T *coef_sig_lin = NULL, int n_sig_lin_feats = 0,
                   const int *sig_quad_inds = NULL, const T *coef_sig_quad = NULL, int n_sig_quad_feats = 0);

}

#endif //DBM_CODE_TOOLS_H




