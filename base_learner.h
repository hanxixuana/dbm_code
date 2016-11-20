//
// Created by xixuan on 10/10/16.
//

#ifndef DBM_CODE_BASE_LEARNER_H
#define DBM_CODE_BASE_LEARNER_H

#ifndef _DEBUG_BASE_LEARNER
#define _DEBUG_BASE_LEARNER 1
#endif

#include "matrix.h"

#include <string>

namespace dbm {

    template<typename T>
    class Base_learner {
    protected:
        char learner_type;

        virtual T predict_for_row(const Matrix<T> &data_x, int row_ind) = 0;

    public:
        Base_learner(const char &type) : learner_type(type) {};

        char get_type() const { return learner_type; };

        virtual void
        predict(const Matrix<T> &data_x,
                Matrix<T> &prediction,
                const T shrinkage = 1,
                const int *row_inds = NULL,
                int n_rows = 0) = 0;
    };

}

namespace dbm {

    template <typename T>
    class Global_mean;

    template <typename T>
    class Mean_trainer;

    template <typename T>
    void save_global_mean(const Global_mean<T> *mean, std::ofstream &out);

    template <typename T>
    void load_global_mean(std::ifstream &in, Global_mean<T> *&mean);

    template <typename T>
    class Global_mean : public Base_learner<T> {
    private:
        T mean = 0;
        T predict_for_row(const Matrix<T> &data_x, int row_ind);
    public:
        Global_mean();
        ~Global_mean();

        void predict(const Matrix<T> &data_x,
                     Matrix<T> &prediction,
                     const T shrinkage = 1,
                     const int *row_inds = nullptr,
                     int n_rows = 0);

        friend void save_global_mean<>(const Global_mean<T> *mean, std::ofstream &out);

        friend void load_global_mean<>(std::ifstream &in, Global_mean<T> *&mean);

        friend class Mean_trainer<T>;

    };

}


namespace dbm {

    template <typename T>
    class Neural_network;

    template <typename T>
    class Neural_network_trainer;

    template <typename T>
    void save_neural_network(const Neural_network<T> *neural_network, std::ofstream &out);

    template <typename T>
    void load_neural_network(std::ifstream &in, Neural_network<T> *&neural_network);

    template <typename T>
    class Neural_network : public Base_learner<T> {
    private:
        int n_predictor;
        int n_hidden_neuron;

        char loss_type;

        int *col_inds = nullptr;

        // n_hidden_neuron * (n_predictor + 1)
        Matrix<T> *input_weight;
        // 1 * (n_hidden_neuron + 1)
        Matrix<T> *hidden_weight;

        // (n_predictor + 1) * 1
        Matrix<T> *input_output;
        // (n_hidden_neuron + 1) * 1
        Matrix<T> *hidden_output;
        // 1 * 1
        T output_output;

        T activation(const T &input);
        void forward();

        T predict_for_row(const Matrix<T> &data_x, int row_ind);

    public:
        Neural_network(int n_predictor, int n_hidden_neuron, char loss_type);
        ~Neural_network();

        void predict(const Matrix<T> &data_x,
                     Matrix<T> &prediction,
                     const T shrinkage = 1,
                     const int *row_inds = nullptr,
                     int n_rows = 0);

        friend void save_neural_network<>(const Neural_network<T> *neural_network, std::ofstream &out);

        friend void load_neural_network<>(std::ifstream &in, Neural_network<T> *&neural_network);

        friend class Neural_network_trainer<T>;

    };

}

namespace dbm {

    template <typename T>
    class Linear_regression;

    template <typename T>
    class Linear_regression_trainer;

    template <typename T>
    void save_linear_regression(const Linear_regression<T> *linear_regression, std::ofstream &out);

    template <typename T>
    void load_linear_regression(std::ifstream &in, Linear_regression<T> *&linear_regression);

    template <typename T>
    class Linear_regression : public Base_learner<T> {
    private:
        int n_predictor;
        char loss_type;

        int *col_inds = nullptr;

        T intercept;
        T *coefs_no_intercept = nullptr;

        T predict_for_row(const Matrix<T> &data_x, int row_ind);
    public:
        Linear_regression(int n_predictor, char loss_type);
        ~Linear_regression();

        void predict(const Matrix<T> &data_x,
                     Matrix<T> &prediction,
                     const T shrinkage = 1,
                     const int *row_inds = nullptr,
                     int n_rows = 0);

        friend void save_linear_regression<>(const Linear_regression<T> *linear_regression, std::ofstream &out);

        friend void load_linear_regression<>(std::ifstream &in, Linear_regression<T> *&linear_regression);

        friend class Linear_regression_trainer<T>;

    };

}

namespace dbm {

    template<typename T>
    class Tree_node;
    template<typename T>
    class Tree_trainer;
    template<typename T>
    class Tree_info;

    template<typename T>
    void save_tree_node(const Tree_node<T> *node, std::ofstream &out);
    template<typename T>
    void load_tree_node(std::ifstream &in, Tree_node<T> *&tree);

    template<typename T>
    void delete_tree(Tree_node<T> *tree);
    template<typename T>
    void print_tree_info(const dbm::Tree_node<T> *tree);

    template<typename T>
    class Tree_node : public Base_learner<T> {
    private:

        Tree_node *larger;
        Tree_node *smaller;

        int depth;

        int column;
        T split_value;
        T loss;

        bool last_node;
        T prediction;

        int no_training_samples;

        T predict_for_row(const Matrix<T> &data_x, int row_ind);

    public:

        Tree_node(int depth);

        Tree_node(int depth, int column, bool last_node, T split_value,
                  T loss, T prediction, int no_tr_samples);

        ~Tree_node();

        void predict(const Matrix<T> &data_x, Matrix<T> &prediction, const T shrinkage = 1, const int *row_inds = nullptr, int n_rows = 0);

        friend void save_tree_node<>(const Tree_node<T> *node, std::ofstream &out);

        friend void load_tree_node<>(std::ifstream &in, Tree_node<T> *&tree);

        friend void delete_tree<>(Tree_node<T> *tree);

        friend void print_tree_info<>(const Tree_node<T> *tree);

        friend class Tree_trainer<T>;

        friend class Tree_info<T>;

    };

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

#endif //DBM_CODE_BASE_LEARNER_H



