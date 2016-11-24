//
// Created by xixuan on 10/10/16.
//

#ifndef DBM_CODE_BASE_LEARNER_TRAINER_H
#define DBM_CODE_BASE_LEARNER_TRAINER_H

#ifndef _DEBUG_BASE_LEARNER_TRAINER
#define _DEBUG_BASE_LEARNER_TRAINER 1
#endif

#include "matrix.h"
#include "base_learner.h"
#include "loss_function.h"
#include "tools.h"

namespace dbm {

    // for means
    template <typename T>
    class Mean_trainer {
    private:
        bool display_training_progress;
        Loss_function<T> loss_function;
    public:
        Mean_trainer(const Params &params);
        ~Mean_trainer();

        void train(Global_mean<T> *mean, const Matrix<T> &train_x,
                   const Matrix<T> &ind_delta, const Matrix<T> &prediction,
                   char loss_function_type = 'n',
                   const int *row_inds = nullptr, int n_rows = 0);

    };

    // for neural networks
    template <typename T>
    class Neural_network_trainer {
    private:
        bool display_training_progress;
        int batch_size;
        int max_iteration;
        T step_size;
        double validate_portion;
        T shrinkage;
        int n_rise_of_loss_on_validate;

        // n_hidden_neuron * (n_predictor + 1)
        Matrix<T> *input_delta;
        // 1 * (n_hidden_neuron + 1)
        Matrix<T> *hidden_delta;

        Loss_function<T> loss_function;

        T activation_derivative(const T &input);
        void backward(Neural_network<T> *neural_network, T ind_delta, T weight);

    public:
        Neural_network_trainer(const Params &params);
        ~Neural_network_trainer();

        void train(Neural_network<T> *neural_network, const Matrix<T> &train_x, const Matrix<T> &ind_delta,
                   const int *row_inds = nullptr, int n_rows = 0,
                   const int *col_inds = nullptr, int n_cols = 0);

    };

    // for splines
    template <typename T>
    class Splines_trainer {
    private:
        bool display_training_progress;

        int n_pairs;

        int **predictor_pairs_inds;

    public:
        Splines_trainer(const Params &params);
        ~Splines_trainer();

        void train(Splines<T> *splines, const Matrix<T> &train_x, const Matrix<T> &ind_delta,
                   const int *row_inds = nullptr, int n_rows = 0,
                   const int *col_inds = nullptr, int n_cols = 0);

    };

    // for linear regression
    template <typename T>
    class Linear_regression_trainer {
    private:
        bool display_training_progress;
    public:
        Linear_regression_trainer(const Params &params);
        ~Linear_regression_trainer();

        void train(Linear_regression<T> *linear_regression, const Matrix<T> &train_x, const Matrix<T> &ind_delta,
                   const int *row_inds = nullptr, int n_rows = 0,
                   const int *col_inds = nullptr, int n_cols = 0);

    };

    // for trees
    template<typename T>
    class Tree_trainer {

    private:
        int max_depth;
        int no_candidate_split_point;
        bool display_training_progress;

        Loss_function<T> loss_function;

    public:
        Tree_trainer(const Params &params);
        ~Tree_trainer();

        void train(Tree_node<T> *tree, const Matrix<T> &train_x, const Matrix<T> &train_y,
                   const Matrix<T> &ind_delta, const Matrix<T> &prediction,
                   const int *monotonic_constraints, char loss_function_type = 'n',
                   const int *row_inds = nullptr, int n_rows = 0,
                   const int *col_inds = nullptr, int n_cols = 0);

        void prune(Tree_node<T> *tree);
    };


}

#endif //DBM_CODE_BASE_LEARNER_TRAINER_H



