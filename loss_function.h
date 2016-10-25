//
// Created by xixuan on 10/10/16.
//

#ifndef DBM_CODE_LOSS_FUNCTION_H
#define DBM_CODE_LOSS_FUNCTION_H

#define _DEBUG_LOSS_FUNCTION 1

#include "matrix.h"

namespace dbm {

    template<typename T>
    class Loss_function {

    public:

        T loss(const Matrix<T> &train_y, const Matrix<T> &prediction, const char &dist,
               const T beta = 0, const int *row_inds = NULL, int n_rows = 0) const;

        T estimate_mean(const Matrix<T> &train_y, const Matrix<T> &prediction, const char &dist,
                        const int *row_inds = NULL, int n_rows = 0) const;

    };

}

#endif //DBM_CODE_LOSS_FUNCTION_H
