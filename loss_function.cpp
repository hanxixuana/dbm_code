//
// Created by xixuan on 10/10/16.
//

#include "loss_function.h"

#include <cassert>
#include <math.h>

namespace dbm {

    template
    class Loss_function<double>;

    template
    class Loss_function<float>;

}

namespace dbm {

    template<typename T>
    inline T Loss_function<T>::loss(const Matrix<T> &train_y, const Matrix<T> &prediction, const char &dist,
                                    const T beta, const int *row_inds, int n_rows) const {

        if (row_inds == NULL) {
            switch (dist) {
                case 'n': {
                    int train_y_width = train_y.get_width(), train_y_height = train_y.get_height();
#if _DEBUG_LOSS_FUNCTION
                    assert(train_y_width == 1);
#endif
                    T result = 0;
                    for (int i = 0; i < train_y_height; ++i) {
                        result += pow(train_y.get(i, 0) - prediction.get(i, 0) - beta, 2.0);
                    }
                    return result / T(train_y_height);
                }
            }
        } else {
            switch (dist) {
                case 'n': {
                    int train_y_width = train_y.get_width();
#if _DEBUG_LOSS_FUNCTION
                    assert(train_y_width == 1);
#endif
                    T result = 0;
                    for (int i = 0; i < n_rows; ++i) {
                        result += pow(train_y.get(row_inds[i], 0) -
                                      prediction.get(row_inds[i], 0) - beta, 2.0);
                    }
                    return result / T(n_rows);
                }
            }
        }
    }

    template<typename T>
    inline T Loss_function<T>::estimate_mean(const Matrix<T> &train_y, const Matrix<T> &prediction,
                                             const char &dist, const int *row_inds, int n_rows) const {

        if (row_inds == NULL) {
            switch (dist) {
                case 'n': {
                    int train_y_width = train_y.get_width(), train_y_height = train_y.get_height();
#if _DEBUG_LOSS_FUNCTION
                    assert(train_y_width == 1);
#endif
                    T result = 0;
                    for (int i = 0; i < train_y_height; ++i) {
                        result += train_y.get(i, 0) - prediction.get(i, 0);
                    }
                    return result / T(train_y_height);
                }
            }
        } else {
            switch (dist) {
                case 'n': {
                    int train_y_width = train_y.get_width();
#if _DEBUG_LOSS_FUNCTION
                    assert(train_y_width == 1);
#endif
                    T result = 0;
                    for (int i = 0; i < n_rows; ++i) {
                        result += train_y.get(row_inds[i], 0) - prediction.get(row_inds[i], 0);
                    }
                    return result / T(n_rows);
                }
            }
        }

    }

}