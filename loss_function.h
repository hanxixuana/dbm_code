//
// Created by xixuan on 10/10/16.
//

#ifndef DBM_CODE_LOSS_FUNCTION_H
#define DBM_CODE_LOSS_FUNCTION_H

#ifndef _DEBUG_LOSS_FUNCTION
#define _DEBUG_LOSS_FUNCTION 1
#endif

#include "matrix.h"

namespace dbm {

    template<typename T>
    class Loss_function {

    public:

        T loss(const Matrix<T> &train_y, const Matrix<T> &prediction, const char &dist,
               const T beta = 0, const int *row_inds = NULL, int n_rows = 0) const;

        T estimate_mean(const Matrix<T> &train_y, const Matrix<T> &prediction, const char &dist,
                        const int *row_inds = NULL, int n_rows = 0) const;

        void mean_function(Matrix<T> &in_and_out, char &dist);

//        T calculate_ind_delta(const Matrix<T> &train_y, const Matrix<T> &prediction,
//                              Matrix<T> &ind_delta, const char &dist);

    };

}

#endif //DBM_CODE_LOSS_FUNCTION_H





