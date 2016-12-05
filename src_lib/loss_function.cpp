//
// Created by xixuan on 10/10/16.
//

#include "loss_function.h"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <limits>

namespace dbm {

    template
    class Loss_function<double>;

    template
    class Loss_function<float>;

}

namespace dbm {

    template <typename T>
    Loss_function<T>::Loss_function(const Params &params) :
            params(params) {};

    template<typename T>
    inline T Loss_function<T>::loss(const Matrix<T> &train_y,
                                    const Matrix<T> &prediction,
                                    const char &dist,
                                    const T beta,
                                    const int *row_inds,
                                    int no_rows) const {
        int train_y_width = train_y.get_width(), train_y_height = train_y.get_height();

        #ifdef _DEBUG_LOSS_FUNCTION
            assert(train_y_width == 1);
        #endif


        /*
         *  1. Remember that a link function may be needed when calculating losses
         *  2. Also remember to put beta in it
         */

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
                    auto prob = [](auto &&f, auto &&b) {
                        auto temp = 1 / (1 + std::exp( - f - b));
                        return temp < MAX_PROB_BERNOULLI ?
                               (temp > MIN_PROB_BERNOULLI ? temp : MIN_PROB_BERNOULLI) : MAX_PROB_BERNOULLI;
                    };
                    auto nll = [&prob](auto &&y, auto &&f, auto &&b) {
                        auto p = prob(f, b);
                        return -y * std::log(p) - (1 - y) * std::log(1 - p);
                    };
                    T result = 0;
                    for (int i = 0; i < train_y_height; ++i) {
                        result += nll(train_y.get(i, 0), prediction.get(i, 0), beta);
                    }
                    return result;
                }
                case 't': {
                    T result = 0;
                    for (int i = 0; i < train_y_height; ++i) {
                        result += std::pow(std::exp(prediction.get(i, 0) + beta), 2 - T(params.tweedie_p)) /
                                          (2 - params.tweedie_p) -
                                train_y.get(i, 0) * std::pow(std::exp(prediction.get(i, 0)),
                                                             1 - T(params.tweedie_p)) / (1 - params.tweedie_p);
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
                    for (int i = 0; i < no_rows; ++i) {
                        result += std::pow(train_y.get(row_inds[i], 0) -
                                           prediction.get(row_inds[i], 0) - beta, 2.0);
                    }
                    return result / T(no_rows);
                }
                case 'p': {
                    T result = 0;
                    for (int i = 0; i < no_rows; ++i) {
                        result += std::exp(prediction.get(row_inds[i], 0) + beta) -
                                  train_y.get(row_inds[i], 0) * (prediction.get(row_inds[i], 0) + beta);
                    }
                    return result;
                }
                case 'b': {
                    auto prob = [](auto &&f, auto &&b) {
                        auto temp = 1 / (1 + std::exp( - f - b));
                        return temp < MAX_PROB_BERNOULLI ?
                               (temp > MIN_PROB_BERNOULLI ? temp : MIN_PROB_BERNOULLI) : MAX_PROB_BERNOULLI;
                    };
                    auto nll = [&prob](auto &&y, auto &&f, auto &&b) {
                        auto p = prob(f, b);
                        return - y * std::log(p) - (1 - y) * std::log(1 - p);
                    };
                    T result = 0;
                    for (int i = 0; i < no_rows; ++i) {
                        result += nll(train_y.get(row_inds[i], 0), prediction.get(row_inds[i], 0), beta);
                    }
                    return result;
                }
                case 't': {
                    T result = 0;
                    for (int i = 0; i < no_rows; ++i) {
                        result += std::pow(std::exp(prediction.get(row_inds[i], 0) + beta),
                                           2 - T(params.tweedie_p)) /
                                  (2 - params.tweedie_p) -
                                train_y.get(row_inds[i], 0) *
                                          std::pow(std::exp(prediction.get(row_inds[i], 0)),
                                                   1 - T(params.tweedie_p)) / (1 - params.tweedie_p);
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
    inline T Loss_function<T>::estimate_mean(const Matrix<T> &ind_delta,
                                             const char &dist,
                                             const int *row_inds,
                                             int no_rows) const {

        int ind_delta_width = ind_delta.get_width(), ind_delta_height = ind_delta.get_height();

        #ifdef _DEBUG_LOSS_FUNCTION
            assert(ind_delta_width == 2);
        #endif

        if (row_inds == nullptr) {
            switch (dist) {
                case 'n': {
                    T result = 0;
                    for (int i = 0; i < ind_delta_height; ++i) {
                        result += ind_delta.get(i, 0);
                    }
                    return result / T(ind_delta_height);
                }
                case 'p': {
                    T y_sum = 0, exp_pred_sum = 0;
                    for (int i = 0; i < ind_delta_height; ++i) {
                        y_sum += ind_delta.get(i, 0) * ind_delta.get(i, 1);
                        exp_pred_sum += ind_delta.get(i, 1);
                    }
                    return std::log(y_sum / exp_pred_sum);
                }
                case 'b': {
                    T numerator = 0, denominator = 0;
                    for (int i = 0; i < ind_delta_height; ++i) {
                        numerator += ind_delta.get(i, 0) * ind_delta.get(i, 1);
                        denominator += ind_delta.get(i, 1);
                    }
                    return numerator / denominator;
                }
                case 't': {
                    T numerator = 0, denominator = 0;
                    for (int i = 0; i < ind_delta_height; ++i) {
                        numerator += ind_delta.get(i, 0) * ind_delta.get(i, 1);
                        denominator += ind_delta.get(i, 1);
                    }
                    numerator += MIN_NUMERATOR_TWEEDIE;
                    return std::log(numerator / denominator);
                }
                default: {
                    throw std::invalid_argument("Specified distribution does not exist.");
                }
            }
        } else {
            switch (dist) {
                case 'n': {
                    T result = 0;
                    for (int i = 0; i < no_rows; ++i) {
                        result += ind_delta.get(row_inds[i], 0);
                    }
                    return result / T(no_rows);
                }
                case 'p': {
                    T y_sum = 0, exp_pred_sum = 0;
                    for (int i = 0; i < no_rows; ++i) {
                        y_sum += ind_delta.get(row_inds[i], 0) * ind_delta.get(row_inds[i], 1);
                        exp_pred_sum += ind_delta.get(row_inds[i], 1);
                    }
                    return std::log(y_sum / exp_pred_sum);
                }
                case 'b': {
                    T numerator = 0, denominator = 0;
                    for (int i = 0; i < no_rows; ++i) {
                        numerator += ind_delta.get(row_inds[i], 0) * ind_delta.get(row_inds[i], 1);
                        denominator += ind_delta.get(row_inds[i], 1);
                    }
                    return numerator / denominator;
                }
                case 't': {
                    T numerator = 0, denominator = 0;
                    for (int i = 0; i < no_rows; ++i) {
                        numerator += ind_delta.get(row_inds[i], 0) * ind_delta.get(row_inds[i], 1);
                        denominator += ind_delta.get(row_inds[i], 1);
                    }
                    numerator += MIN_NUMERATOR_TWEEDIE;
                    return std::log(numerator / denominator);
                }
                default: {
                    throw std::invalid_argument("Specified distribution does not exist.");
                }
            }
        }

    }

    template <typename T>
    void Loss_function<T>::mean_function(Matrix<T> &in_and_out,
                                         char &dist) {

        int lpp_height = in_and_out.get_height(),
                lpp_width = in_and_out.get_width();

        #ifdef _DEBUG_LOSS_FUNCTION
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
            case 't': {
                for(int i = 0; i < lpp_height; ++i) {
                    temp = std::exp(in_and_out.get(i, 0));
                    in_and_out.assign(i, 0, temp);
                }
                break;
            }
            default: {
                throw std::invalid_argument("Specified distribution does not exist.");
            }
        }

    }

    template <typename T>
    void Loss_function<T>::calculate_ind_delta(const Matrix<T> &train_y,
                                               const Matrix<T> &prediction,
                                               Matrix<T> &ind_delta,
                                               const char &dist,
                                               const int *row_inds,
                                               int no_rows) {

        if( row_inds == nullptr) {

            int y_height = train_y.get_height();

            #ifdef _DEBUG_LOSS_FUNCTION
                assert(y_height == prediction.get_height() &&
                               prediction.get_height() == ind_delta.get_height() &&
                               ind_delta.get_width() == 2);
            #endif

            switch (dist) {
                case 'n': {
                    for(int i = 0; i < y_height; ++i) {
                        ind_delta.assign(i, 0, train_y.get(i, 0) - prediction.get(i, 0));
                        ind_delta.assign(i, 1, 1);
                    }
                    break;
                }
                case 'p': {
                    auto result = [](auto &&y, auto &&pred) {
                        auto temp = y / std::exp(pred);
                        return temp < MAX_IND_DELTA ? (temp > MIN_IND_DELTA ? temp : MIN_IND_DELTA) : MAX_IND_DELTA;
                    };
                    for(int i = 0; i < y_height; ++i) {
                        ind_delta.assign(i, 0, result(train_y.get(i, 0), prediction.get(i, 0)));
                        ind_delta.assign(i, 1, std::exp(prediction.get(i, 0)));
                    }
                    break;
                }
                case 'b': {
                    auto prob = [](auto &&f) {
                        auto temp = 1 / (1 + std::exp( - f));
                        return temp < MAX_PROB_BERNOULLI ?
                               (temp > MIN_PROB_BERNOULLI ? temp : MIN_PROB_BERNOULLI) : MAX_PROB_BERNOULLI;
                    };
                    auto delta = [&prob](auto &&y, auto &&f) {
                        auto p = prob(f);
                        return (y - p) / p / (1 - p);
                    };
                    auto result = [&delta](auto &&y, auto &&f) {
                        auto little_delta = delta(y, f);
                        return little_delta < MAX_IND_DELTA ?
                               (little_delta > MIN_IND_DELTA ? little_delta : MIN_IND_DELTA) : MAX_IND_DELTA;
                    };
                    auto denominator = [&prob](auto &&f) {
                        auto p = prob(f);
                        return p * (1 - p);
                    };
                    for(int i = 0; i < y_height; ++i) {
                        ind_delta.assign(i, 0, result(train_y.get(i, 0), prediction.get(i, 0)));
                        ind_delta.assign(i, 1, denominator(prediction.get(i, 0)));
                    }
                    break;
                }
                case 't': {
                    auto result = [](auto &&y, auto &&pred) {
                        auto temp = y / std::exp(pred);
                        return temp < MAX_IND_DELTA ? (temp > MIN_IND_DELTA ? temp : MIN_IND_DELTA) : MAX_IND_DELTA;
                    };
                    for(int i = 0; i < y_height; ++i) {

                        ind_delta.assign(i, 0, result(train_y.get(i, 0), prediction.get(i, 0)));
                        ind_delta.assign(i, 1, std::exp(prediction.get(i, 0) * (2 - params.tweedie_p)));
                    }
                    break;
                }
                default: {
                    throw std::invalid_argument("Specified distribution does not exist.");
                }
            }
        }
        else {
            #ifdef _DEBUG_LOSS_FUNCTION
                assert(no_rows > 0);
            #endif
            switch (dist) {
                case 'n': {
                    for(int i = 0; i < no_rows; ++i) {
                        ind_delta.assign(row_inds[i], 0, train_y.get(row_inds[i], 0) - prediction.get(row_inds[i], 0));
                        ind_delta.assign(row_inds[i], 1, 1);
                    }
                    break;
                }
                case 'p': {
                    auto result = [](auto &&y, auto &&pred) {
                        auto temp = y / std::exp(pred);
                        return temp < MAX_IND_DELTA ? (temp > MIN_IND_DELTA ? temp : MIN_IND_DELTA) : MAX_IND_DELTA;
                    };
                    for(int i = 0; i < no_rows; ++i) {
                        ind_delta.assign(row_inds[i], 0, result(train_y.get(row_inds[i], 0), prediction.get(row_inds[i], 0)));
                        ind_delta.assign(row_inds[i], 1, std::exp(prediction.get(row_inds[i], 0)));
                    }
                    break;
                }
                case 'b': {
                    auto prob = [](auto &&f) {
                        auto temp = 1 / (1 + std::exp( - f));
                        return temp < MAX_PROB_BERNOULLI ?
                               (temp > MIN_PROB_BERNOULLI ? temp : MIN_PROB_BERNOULLI) : MAX_PROB_BERNOULLI;
                    };
                    auto delta = [&prob](auto &&y, auto &&f) {
                        auto p = prob(f);
                        return (y - p) / p / (1 - p);
                    };
                    auto result = [&delta](auto &&y, auto &&f) {
                        auto little_delta = delta(y, f);
                        return little_delta < MAX_IND_DELTA ?
                               (little_delta > MIN_IND_DELTA ? little_delta : MIN_IND_DELTA) : MAX_IND_DELTA;
                    };
                    auto denominator = [&prob](auto &&f) {
                        auto p = prob(f);
                        return p * (1 - p);
                    };
                    for(int i = 0; i < no_rows; ++i) {
                        ind_delta.assign(row_inds[i], 0, result(train_y.get(row_inds[i], 0), prediction.get(row_inds[i], 0)));
                        ind_delta.assign(row_inds[i], 1, denominator(prediction.get(row_inds[i], 0)));
                    }
                    break;
                }
                case 't': {
                    auto result = [](auto &&y, auto &&pred) {
                        auto temp = y / std::exp(pred);
                        return temp < MAX_IND_DELTA ? (temp > MIN_IND_DELTA ? temp : MIN_IND_DELTA) : MAX_IND_DELTA;
                    };
                    for(int i = 0; i < no_rows; ++i) {

                        ind_delta.assign(row_inds[i], 0, result(train_y.get(row_inds[i], 0), prediction.get(row_inds[i], 0)));
                        ind_delta.assign(row_inds[i], 1, std::exp(prediction.get(row_inds[i], 0) * (2 - params.tweedie_p)));
                    }
                    break;
                }
                default: {
                    throw std::invalid_argument("Specified distribution does not exist.");
                }
            }
        }

    }

}



