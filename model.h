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

#include <string>
#include <fstream>

namespace dbm {

    template<typename T>
    class DBM;

    template<typename T>
    void save_dbm(const DBM<T> *dbm, std::ofstream &out);

    template<typename T>
    void load_dbm(std::ifstream &in, DBM<T> *&dbm);


    template<typename T>
    class Regressor {
    public:

        virtual void train(const Matrix<T> &train_x, const Matrix<T> &train_y, const int * input_monotonic_constaints) = 0;

        virtual void train(const Data_set<T> &data_set, const int * input_monotonic_constaints) = 0;

        virtual void predict(const Matrix<T> &data_x, Matrix<T> &predict_y) = 0;

    };

    template<typename T>
    class DBM : public Regressor<T> {
    private:
        int no_learners;

        int no_candidate_feature;
        int no_train_sample;

        Base_learner<T> **learners = nullptr;

        Tree_trainer<T> *tree_trainer = nullptr;
        Mean_trainer<T> *mean_trainer = nullptr;

        Params params;
        Loss_function<T> loss_function;

        Tree_info<T> *tree_info = nullptr;

        Matrix<T> *prediction_train_data = nullptr;

        T *test_loss_record = nullptr;

    public:

        DBM(int no_learners, int no_candidate_feature, int no_train_sample);

        DBM(const std::string &param_string);

        ~DBM();

        // comments on monotonic_constraints
        // 1: positive relationship; 0: anything; -1: negative relationship
        // by only allowing 1, 0, -1, we could be able to check if the length is correct in some sense
        void train(const Matrix<T> &train_x, const Matrix<T> &train_y, const int * input_monotonic_constaints = nullptr);
        void train(const Data_set<T> &data_set, const int * input_monotonic_constaints = nullptr);
        void predict(const Matrix<T> &data_x, Matrix<T> &predict_y);

        Matrix<T> *get_prediction_on_train_data() const;
        T *get_test_loss() const;

        void set_loss_function(const char &type);

        friend void save_dbm<>(const DBM *dbm, std::ofstream &out);
        friend void load_dbm<>(std::ifstream &in, DBM *&dbm);

    };

}


#endif //DBM_CODE_MODEL_H




