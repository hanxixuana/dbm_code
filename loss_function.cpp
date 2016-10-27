//
// Created by xixuan on 10/10/16.
//

#include "loss_function.h"

#include <cassert>
#include <math.h>
#include <stdexcept>

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
        int train_y_width = train_y.get_width(), train_y_height = train_y.get_height();
        #if _DEBUG_LOSS_FUNCTION
            assert(train_y_width == 1);
        #endif
        if (row_inds == NULL) {
            switch (dist) {
                case 'n': {
                    // mean suared error
                    T result = 0;
                    for (int i = 0; i < train_y_height; ++i) {
                        result += pow(train_y.get(i, 0) - prediction.get(i, 0) - beta, 2.0);
                    }
                    return result / T(train_y_height);
                }
                case 'p': {
                    // negative log likelihood of poission distribution
                    T result = 0, lambda, k;
                    for (int i = 0; i < train_y_height; ++i) {
                        lambda = std::max(prediction.get(i, 0) + beta, T(0.01));
                        k = round(std::max(train_y.get(i, 0), T(0.01)));
                        result += lambda - k * log(lambda) + tgamma(k + 1);
                    }
                    return result;
                }
                case 'b': {
                    // negative log likelihood of bernoulli distribution
                    // negative is considered to be 0
                    // positive is considered to be 1
                    T result = 0, k, p;
                    for (int i = 0; i < train_y_height; ++i) {
                        k = train_y.get(i, 0) < 0? 0 : 1;
                        p = 1 / (1 + exp(prediction.get(i, 0)));
                        result += -k * log(p) - (1 - k) * log(1 - p);
                    }
                    return result;
                }
                default: {
                    throw std::invalid_argument("Specified distribution does not exist.");
                }
            }
        } else {
            switch (dist) {
                case 'n': {
                    T result = 0;
                    for (int i = 0; i < n_rows; ++i) {
                        result += pow(train_y.get(row_inds[i], 0) -
                                      prediction.get(row_inds[i], 0) - beta, 2.0);
                    }
                    return result / T(n_rows);
                }
                case 'p': {
                    T result = 0, lambda, k;
                    for (int i = 0; i < n_rows; ++i) {
                        lambda = std::max(prediction.get(row_inds[i], 0) + beta, T(0.01));
                        k = round(std::max(train_y.get(row_inds[i], 0), T(0.01)));
                        result += lambda - k * log(lambda) + tgamma(k + 1);
                    }
                    return result;
                }
                case 'b': {
                    T result = 0, k, p;
                    for (int i = 0; i < n_rows; ++i) {
                        k = train_y.get(row_inds[i], 0) < 0? 0 : 1;
                        p = 1 / (1 + exp(prediction.get(row_inds[i], 0)));
                        result += -k * log(p) - (1 - k) * log(1 - p);
                    }
                    return result;
                }
                default: {
                    throw std::invalid_argument("Specified distribution does not exist.");
                }
            }
        }
    }

    template<typename T>
    inline T Loss_function<T>::estimate_mean(const Matrix<T> &train_y, const Matrix<T> &prediction,
                                             const char &dist, const int *row_inds, int n_rows) const {
        int train_y_width = train_y.get_width(), train_y_height = train_y.get_height();
        #if _DEBUG_LOSS_FUNCTION
            assert(train_y_width == 1);
        #endif
        if (row_inds == NULL) {
            switch (dist) {
                case 'n': {
                    T result = 0;
                    for (int i = 0; i < train_y_height; ++i) {
                        result += train_y.get(i, 0) - prediction.get(i, 0);
                    }
                    return result / T(train_y_height);
                }
                case 'p': {
                    T y_sum = 0, exp_pred_sum = 0;
                    for (int i = 0; i < train_y_height; ++i) {
                        y_sum += train_y.get(i, 0);
                        exp_pred_sum += exp(prediction.get(i, 0));
                    }
                    return log(y_sum / exp_pred_sum);
                }
                case 'b': {
                    T numerator = 0, denominator = 0, p;
                    for (int i = 0; i < train_y_height; ++i) {
                        p = 1 / (1 + exp(prediction.get(i, 0)));
                        numerator += train_y.get(i, 0) < 0? 0 : 1 - p;
                        denominator += p * (1 - p);
                    }
                    return numerator / denominator;
                }
                default: {
                    throw std::invalid_argument("Specified distribution does not exist.");
                }
            }
        } else {
            switch (dist) {
                case 'n': {
                    T result = 0;
                    for (int i = 0; i < n_rows; ++i) {
                        result += train_y.get(row_inds[i], 0) - prediction.get(row_inds[i], 0);
                    }
                    return result / T(n_rows);
                }
                case 'p': {
                    T y_sum = 0, exp_pred_sum = 0;
                    for (int i = 0; i < n_rows; ++i) {
                        y_sum += train_y.get(row_inds[i], 0);
                        exp_pred_sum += exp(prediction.get(row_inds[i], 0));
                    }
                    return log(y_sum / exp_pred_sum);
                }
                case 'b': {
                    T numerator = 0, denominator = 0, p;
                    for (int i = 0; i < n_rows; ++i) {
                        p = 1 / (1 + exp(prediction.get(row_inds[i], 0)));
                        numerator += train_y.get(row_inds[i], 0) < 0? 0 : 1 - p;
                        denominator += p * (1 - p);
                    }
                    return numerator / denominator;
                }
                default: {
                    throw std::invalid_argument("Specified distribution does not exist.");
                }
            }
        }

    }

}



