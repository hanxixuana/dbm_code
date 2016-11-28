//
// Created by xixuan on 10/10/16.
//

#ifndef DBM_CODE_MODEL_H
#define DBM_CODE_MODEL_H

#ifndef _DEBUG_MODEL
#define _DEBUG_MODEL 1
#endif

#include "matrix.h"
#include "data_set.h"
#include "base_learner.h"
#include "base_learner_trainer.h"
#include "tools.h"

#include <string>
#include <fstream>

namespace dbm {

    template<typename T>
    class DBM;

    template<typename T>
    void save_dbm(const DBM<T> *dbm,
                  std::ofstream &out);

    template<typename T>
    void load_dbm(std::ifstream &in,
                  DBM<T> *&dbm);

    template<typename T>
    class Regressor {
    public:

        virtual void train(const Matrix<T> &train_x,
                           const Matrix<T> &train_y,
                           const int * input_monotonic_constaints) = 0;

        virtual void train(const Data_set<T> &data_set,
                           const int * input_monotonic_constaints) = 0;

        virtual void predict(const Matrix<T> &data_x,
                             Matrix<T> &predict_y) = 0;

    };

    template<typename T>
    class DBM : public Regressor<T> {
    private:
        int no_bunches_of_learners;
        int no_cores;

        int total_no_feature;
        int no_candidate_feature;
        int no_train_sample;

        Base_learner<T> **learners = nullptr;

        Tree_trainer<T> *tree_trainer = nullptr;
        Mean_trainer<T> *mean_trainer = nullptr;
        Linear_regression_trainer<T> *linear_regression_trainer = nullptr;
        Neural_network_trainer<T> *neural_network_trainer = nullptr;
        Splines_trainer<T> *splines_trainer = nullptr;
        Kmeans2d_trainer<T> *kmeans2d_trainer = nullptr;

        Params params;
        Loss_function<T> loss_function;

        Matrix<T> *prediction_train_data = nullptr;

        T *test_loss_record = nullptr;

    public:

        DBM(int no_bunches_of_learners,
            int no_cores,
            int no_candidate_feature,
            int no_train_sample,
            int total_no_feature);

        DBM(const std::string &param_string);

        ~DBM();

        // comments on monotonic_constraints
        // 1: positive relationship; 0: anything; -1: negative relationship
        // by only allowing 1, 0, -1, we could be able to check if the length is correct in some sense
        void train(const Matrix<T> &train_x,
                   const Matrix<T> &train_y,
                   const int * input_monotonic_constaints = nullptr);

        void train(const Data_set<T> &data_set,
                   const int * input_monotonic_constaints = nullptr);

        void predict(const Matrix<T> &data_x,
                     Matrix<T> &predict_y);


        /*
         *  TOOLS
         */

        Matrix<T> *get_prediction_on_train_data() const;
        T *get_test_loss() const;

        void set_loss_function_and_shrinkage(const char &type, const T &shrinkage);

        Matrix<T> partial_dependence_plot(const Matrix<T> &data,
                                          const int &predictor_ind,
                                          const T *x_tick_min = nullptr,
                                          const T *x_tick_max = nullptr);

        /*
         *  IO
         */
        friend void save_dbm<>(const DBM *dbm,
                               std::ofstream &out);
        friend void load_dbm<>(std::ifstream &in,
                               DBM *&dbm);

    };

}


#endif //DBM_CODE_MODEL_H




