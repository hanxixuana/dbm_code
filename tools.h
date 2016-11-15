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
#include "base_learner.h"

/*
 * tools for parameters
 */

namespace dbm {

    struct Params {

        // DBM
        int no_bunches_of_learners = 100;
        int no_candidate_feature = 5;
        int no_train_sample = 5000;

        int no_cores = 0;

        char loss_function = 'n';

        bool display_training_progress = true;
        bool record_every_tree = true;
        int freq_showing_loss_on_test = 10;

        double shrinkage = 0.01;

        // portions should be summed to 1
        double portion_for_trees = 0.5;
        double portion_for_lr = 0.5;

        // tweedie: p should in (1, 2)
        double tweedie_p = 1.6;

        // CART
        int max_depth = 5;
        int no_candidate_split_point = 5;

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
    int split_into_words(const std::string &line, std::string *words);

}

/*
 * tools for matrices
 */

namespace dbm {

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


/*
 * tools for base learners
 */

// for global means
namespace dbm {

    template <typename T>
    void save_global_mean(const Global_mean<T> *mean, std::ofstream &out);

    template <typename T>
    void load_global_mean(std::ifstream &in, Global_mean<T> *&mean);

}

// for linear regression
namespace dbm {

    template <typename T>
    void save_linear_regression(const Linear_regression<T> *linear_regression, std::ofstream &out);

    template <typename T>
    void load_linear_regression(std::ifstream &in, Linear_regression<T> *&linear_regression);

}

// for trees
namespace dbm {

    template <typename T>
    void save_tree_node(const Tree_node<T> *node, std::ofstream &out);

    template <typename T>
    void load_tree_node(std::ifstream &in, Tree_node<T> *&node);

    template <typename T>
    void delete_tree(Tree_node<T> *tree);

    // display tree information
    template<typename T>
    void print_tree_info(const dbm::Tree_node<T> *tree);

    template<typename T>
    class Tree_info {
    private:
        std::string **tree_nodes;
        int depth = 0;
        int height = 0;

        void get_depth(const dbm::Tree_node<T> *tree);

        void fill(const dbm::Tree_node<T> *tree, int h);

    public:
        Tree_info(const dbm::Tree_node<T> *tree);

        ~Tree_info();

        void print() const;

        void print_to_file(const std::string &file_name, const int & number) const;
    };

}




#endif //DBM_CODE_TOOLS_H




