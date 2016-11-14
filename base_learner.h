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
        predict(const Matrix<T> &data_x, Matrix<T> &prediction, const T shrinkage = 1, const int *row_inds = NULL, int n_rows = 0) = 0;
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

        void predict(const Matrix<T> &data_x, Matrix<T> &prediction, const T shrinkage = 1, const int *row_inds = nullptr, int n_rows = 0);

        friend void save_global_mean<>(const Global_mean<T> *mean, std::ofstream &out);

        friend void load_global_mean<>(std::ifstream &in, Global_mean<T> *&mean);

        friend class Mean_trainer<T>;

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

        void predict(const Matrix<T> &data_x, Matrix<T> &prediction, const T shrinkage = 1, const int *row_inds = nullptr, int n_rows = 0);

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

#endif //DBM_CODE_BASE_LEARNER_H



