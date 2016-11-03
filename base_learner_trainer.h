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

    /* to-do note:
     * loss type has not included in trainer as a parameter from Params
     */

    template <typename T>
    class Mean_trainer {
    private:
        Loss_function<T> loss_function;
    public:
        Mean_trainer(const Params &params);
        ~Mean_trainer();

        void train(Global_mean<T> *mean, const Matrix<T> &train_x,
                   const Matrix<T> &train_y, const Matrix<T> &prediction,
                   const int *row_inds = nullptr, int n_rows = 0,
                   const int *col_inds = nullptr, int n_cols = 0);

    };

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

        void train(Tree_node<T> *tree, const Matrix<T> &train_x,
                   const Matrix<T> &train_y, const Matrix<T> &prediction,
                   const int *monotonic_constraints,
                   const int *row_inds = nullptr, int n_rows = 0,
                   const int *col_inds = nullptr, int n_cols = 0);

        void prune(Tree_node<T> *tree);
    };


}

#endif //DBM_CODE_BASE_LEARNER_TRAINER_H



