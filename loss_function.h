//
// Created by xixuan on 10/10/16.
//

#ifndef DBM_CODE_LOSS_FUNCTION_H
#define DBM_CODE_LOSS_FUNCTION_H

#ifndef _DEBUG_LOSS_FUNCTION
#define _DEBUG_LOSS_FUNCTION 1
#endif

#include "matrix.h"
#include "tools.h"

#define MAX_PROB_BERNOULLI 0.99999
#define MIN_PROB_BERNOULLI 0.00001

#define MAX_IND_DELTA 999
#define MIN_IND_DELTA -999

#define MIN_NUMERATOR_TWEEDIE 0.00001

namespace dbm {

    template<typename T>
    class Loss_function {
    private:



    public:
        Params params;

        Loss_function(const Params &params);

        T loss(const Matrix<T> &train_y, const Matrix<T> &prediction, const char &dist,
               const T beta = 0, const int *row_inds = nullptr, int n_rows = 0) const;

        T estimate_mean(const Matrix<T> &ind_delta, const Matrix<T> &prediction, const char &dist,
                        const int *row_inds = nullptr, int n_rows = 0) const;

        void mean_function(Matrix<T> &in_and_out, char &dist);

        void calculate_ind_delta(const Matrix<T> &train_y, const Matrix<T> &prediction,
                                 Matrix<T> &ind_delta, const char &dist, const int *row_inds = nullptr, int n_rows = 0);

    };

}

#endif //DBM_CODE_LOSS_FUNCTION_H





