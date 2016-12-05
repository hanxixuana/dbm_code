//
// Created by xixuan on 10/10/16.
//

#ifndef DBM_CODE_BASE_LEARNER_TRAINER_H
#define DBM_CODE_BASE_LEARNER_TRAINER_H

//#ifndef _DEBUG_BASE_LEARNER_TRAINER
//#define _DEBUG_BASE_LEARNER_TRAINER
//#endif

#include "matrix.h"
#include "base_learner.h"
#include "loss_function.h"
#include "tools.h"

// for means
namespace dbm {

    template<typename T>
    class Mean_trainer {
    private:

        bool display_training_progress;

        Loss_function<T> loss_function;

    public:
        Mean_trainer(const Params &params);

        ~Mean_trainer();

        void train(Global_mean<T> *mean,
                   const Matrix<T> &train_x,
                   const Matrix<T> &ind_delta,
                   const Matrix<T> &prediction,
                   char loss_function_type = 'n',
                   const int *row_inds = nullptr,
                   int no_rows = 0);

    };

}

// for neural networks
namespace dbm {

    template<typename T>
    class Neural_network_trainer {
    private:

        int batch_size;
        int nn_max_iteration;
        T step_size;
        double validate_portion;
        T shrinkage;
        int no_rise_of_loss_on_validate;

        Loss_function<T> loss_function;

        T activation_derivative(const T &input);

        void backward(Neural_network<T> *neural_network,
                      const Matrix<T> &input_output,
                      Matrix<T> &hidden_output,
                      T &output_output,
                      Matrix<T> &hidden_delta,
                      Matrix<T> &input_delta,
                      T ind_delta,
                      T weight);

    public:
        Neural_network_trainer(const Params &params);

        ~Neural_network_trainer();

        void train(Neural_network<T> *neural_network,
                   const Matrix<T> &train_x,
                   const Matrix<T> &ind_delta,
                   const int *row_inds = nullptr,
                   int no_rows = 0,
                   const int *col_inds = nullptr,
                   int no_cols = 0);

    };

}

// for splines
namespace dbm {

    template<typename T>
    class Splines_trainer {
    private:

        int no_pairs;

        double regularization;

        int **predictor_pairs_inds;

    public:
        Splines_trainer(const Params &params);

        ~Splines_trainer();

        void train(Splines<T> *splines,
                   const Matrix<T> &train_x,
                   const Matrix<T> &ind_delta,
                   const int *row_inds = nullptr,
                   int no_rows = 0,
                   const int *col_inds = nullptr,
                   int no_cols = 0);

    };

}

// for k-means-2d
namespace dbm {

    template<typename T>
    class Kmeans2d_trainer {
    private:

        int no_centroids;
        int no_candidate_feature;
        int no_pairs;

        int kmeans_max_iteration;
        T kmeans_tolerance;

        int **predictor_pairs_inds;

        Loss_function<T> loss_function;

    public:
        Kmeans2d_trainer(const Params &params);

        ~Kmeans2d_trainer();

        void train(Kmeans2d<T> *kmeans2d,
                   const Matrix<T> &train_x,
                   const Matrix<T> &ind_delta,
                   char loss_function_type = 'n',
                   const int *row_inds = nullptr,
                   int no_rows = 0,
                   const int *col_inds = nullptr,
                   int no_cols = 0);

    };

}

// for linear regression
namespace dbm {

    template<typename T>
    class Linear_regression_trainer {
    private:

    public:
        Linear_regression_trainer(const Params &params);

        ~Linear_regression_trainer();

        void train(Linear_regression<T> *linear_regression,
                   const Matrix<T> &train_x,
                   const Matrix<T> &ind_delta,
                   const int *row_inds = nullptr,
                   int no_rows = 0,
                   const int *col_inds = nullptr,
                   int no_cols = 0);

    };

}

// for trees
namespace dbm {

    template<typename T>
    class Tree_trainer {

    private:
        int max_depth;
        int no_candidate_split_point;

        Loss_function<T> loss_function;

    public:
        Tree_trainer(const Params &params);
        ~Tree_trainer();

        void train(Tree_node<T> *tree,
                   const Matrix<T> &train_x,
                   const Matrix<T> &train_y,
                   const Matrix<T> &ind_delta,
                   const Matrix<T> &prediction,
                   const Matrix<T> &monotonic_constraints,
                   char loss_function_type = 'n',
                   const int *row_inds = nullptr,
                   int no_rows = 0,
                   const int *col_inds = nullptr,
                   int no_cols = 0);

        void prune(Tree_node<T> *tree);
    };

}

#endif //DBM_CODE_BASE_LEARNER_TRAINER_H



