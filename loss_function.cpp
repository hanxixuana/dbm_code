//
// Created by xixuan on 10/10/16.
//

#include "loss_function.h"

#include <cassert>
#include <cmath>
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
                        result += std::pow(train_y.get(i, 0) - prediction.get(i, 0) - beta, 2.0);
                    }
                    return result / T(train_y_height);
                }
                case 'p': {
                    // negative log likelihood of poission distribution
                    T result = 0;
                    for (int i = 0; i < train_y_height; ++i) {
                        result += std::exp(prediction.get(i, 0) + beta) -
                                train_y.get(i, 0) * (prediction.get(i, 0) + beta);
                    }
                    return result;
                }
                case 'b': {
                    // negative log likelihood of bernoulli distribution
                    T result = 0, pi_inversed;
                    for (int i = 0; i < train_y_height; ++i) {
                        pi_inversed = 1 + std::exp(-prediction.get(i, 0));
                        result += train_y.get(i, 0) * std::log(pi_inversed) -
                                (1 - train_y.get(i, 0)) * std::log(1 - 1 / pi_inversed);
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
                        result += std::pow(train_y.get(row_inds[i], 0) -
                                                   prediction.get(row_inds[i], 0) - beta, 2.0);
                    }
                    return result / T(n_rows);
                }
                case 'p': {
                    T result = 0;
                    for (int i = 0; i < n_rows; ++i) {
                        result += std::exp(prediction.get(row_inds[i], 0) + beta) -
                                  train_y.get(row_inds[i], 0) * (prediction.get(row_inds[i], 0) + beta);
                    }
                    return result;
                }
                case 'b': {
                    T result = 0, pi_inversed;
                    for (int i = 0; i < n_rows; ++i) {
                        pi_inversed = 1 + std::exp(-prediction.get(row_inds[i], 0));
                        result += train_y.get(row_inds[i], 0) * std::log(pi_inversed) -
                                  (1 - train_y.get(row_inds[i], 0)) * std::log(1 - 1 / pi_inversed);
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
                    return log(y_sum / exp_pred_sum + T(0.000001));
                }
                case 'b': {
                    T numerator = 0, denominator = 0, p;
                    for (int i = 0; i < train_y_height; ++i) {
                        p = 1 / (1 + exp(-prediction.get(i, 0)));
                        numerator += train_y.get(i, 0) - p;
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
                    return log(y_sum / exp_pred_sum + T(0.000001));
                }
                case 'b': {
                    T numerator = 0, denominator = 0, p;
                    for (int i = 0; i < n_rows; ++i) {
                        p = 1 / (1 + exp(-prediction.get(row_inds[i], 0)));
                        numerator += train_y.get(row_inds[i], 0) - p;
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

    template <typename T>
    void Loss_function<T>::mean_function(Matrix<T> &in_and_out, char &dist) {

        int lpp_height = in_and_out.get_height(),
                lpp_width = in_and_out.get_width();
        #if _DEBUG_LOSS_FUNCTION
            assert(lpp_width == 1);
        #endif

        T temp = 0;
        switch (dist) {
            case 'n': {
                //do nothing
                break;
            }
            case 'p': {
                for(int i = 0; i < lpp_height; ++i) {
                    temp = std::exp(in_and_out.get(i, 0));
                    in_and_out.assign(i, 0, temp);
                }
                break;
            }
            case 'b': {
                for(int i = 0; i < lpp_height; ++i) {
                    temp = 1 + std::exp(-in_and_out.get(i, 0));
                    in_and_out.assign(i, 0, 1 / temp);
                }
                break;
            }
            default: {
                throw std::invalid_argument("Specified distribution does not exist.");
            }
        }

    }

}



