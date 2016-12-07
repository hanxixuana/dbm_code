#include <iostream>
#include <fstream>

#include "data_set.h"
#include "base_learner.h"
#include "base_learner_trainer.h"
#include "model.h"

using namespace std;

void train_test_save_load_dbm();

void train_test_save_load_nn();

void prepare_data();

int main() {

    prepare_data();
    train_test_save_load_dbm();

    return 0;

}

void prepare_data() {
    string file_name = "train_data.txt";
    dbm::make_data<float>(file_name, 10000, 30, 'p');
}

void train_test_save_load_dbm() {
    int n_samples = 10000, n_features = 30, n_width = 31;

    dbm::Matrix<float> train_data(n_samples, n_width, "train_data.txt");

    int *col_inds = new int[n_features];

    for (int i = 0; i < n_features; ++i)
        col_inds[i] = i;

    dbm::Matrix<float> train_x = train_data.cols(col_inds, n_features);
    dbm::Matrix<float> train_y = train_data.col(n_features);

    // ================

    dbm::Data_set<float> data_set(train_x, train_y, 0.25);
    dbm::Matrix<float> train_prediction(int(0.75 * n_samples), 1, 0);
    dbm::Matrix<float> test_prediction(int(0.25 * n_samples), 1, 0);
    dbm::Matrix<float> re_test_prediction(int(0.25 * n_samples), 1, 0);

    // ================
    string param_string = "dbm_no_bunches_of_learners 31 dbm_no_cores 0 dbm_loss_function p "
            "dbm_portion_train_sample 0.75 dbm_no_candidate_feature 5 dbm_shrinkage 0.25 "
            "dbm_portion_for_trees 0 dbm_portion_for_lr 0 dbm_portion_for_s 0 "
            "dbm_portion_for_k 0 dbm_portion_for_nn 1";
    dbm::Params params = dbm::set_params(param_string);
    dbm::DBM<float> dbm(params);

    dbm.train(data_set);

//    dbm.predict(data_set.get_train_x(), train_prediction);
    dbm.predict(data_set.get_test_x(), test_prediction);

    dbm::Matrix<float> pred = dbm.predict(data_set.get_test_x());
    pred.print_to_file("pred.txt");

//    dbm::Matrix<float> pdp = dbm.partial_dependence_plot(data_set.get_train_x(), 6);
//    pdp.print_to_file("pdp.txt");

//    dbm::Matrix<float> ss = dbm.statistical_significance(data_set.get_train_x());
//    ss.print_to_file("ss.txt");

    dbm.save_dbm_to("dbm.txt");

//    {
//        ofstream out("dbm.txt");
//        dbm::save_dbm(&dbm, out);
//    }

    // ===================

    dbm::DBM<float> re_dbm(params);

    re_dbm.load_dbm_from("dbm.txt");

//    {
//        ifstream in("dbm.txt");
//        dbm::load_dbm(in, re_dbm);
//    }

    re_dbm.save_dbm_to("re_dbm.txt");

//    {
//        ofstream out("re_dbm.txt");
//        dbm::save_dbm(re_dbm, out);
//    }

    re_dbm.predict(data_set.get_test_x(), re_test_prediction);

    dbm::Matrix<float> temp = dbm::hori_merge(*dbm.get_prediction_on_train_data(), train_prediction);
//    dbm::Matrix<float> check = dbm::hori_merge(dbm::hori_merge(data_set.get_train_x(), data_set.get_train_y()), temp);
    dbm::Matrix<float> check = dbm::hori_merge(data_set.get_train_y(), temp);
    check.print_to_file("check_train_and_pred.txt");

    dbm::Matrix<float> combined = dbm::hori_merge(test_prediction, re_test_prediction);
//    dbm::Matrix<float> result = dbm::hori_merge(dbm::hori_merge(data_set.get_test_x(), data_set.get_test_y()), combined);
    dbm::Matrix<float> result = dbm::hori_merge(data_set.get_test_y(), combined);
    result.print_to_file("whole_result.txt");

    delete[] col_inds;
}

void train_test_save_load_nn() {
    int n_samples = 50000, n_features = 30, n_width = 31;

    dbm::Matrix<float> train_data(n_samples, n_width, "train_data.txt");
    dbm::Matrix<float> prediction(n_samples, 1, 0);
    dbm::Matrix<float> ind_delta(n_samples, 2, 0);

    int row_inds[n_samples], col_inds[n_features];

    for (int i = 0; i < n_features; ++i)
        col_inds[i] = i;
    for (int i = 0; i < n_samples; ++i)
        row_inds[i] = i;

    dbm::Matrix<float> train_x = train_data.cols(col_inds, n_features);
    dbm::Matrix<float> train_y = train_data.col(n_features);

    // ========================================================

    dbm::Params params = dbm::set_params("no_candidate_feature 5 no_hidden_neurons 5 loss_function b");

    dbm::Neural_network<float> *nn = new dbm::Neural_network<float>(params.dbm_no_candidate_feature,
                                                                    params.nn_no_hidden_neurons,
                                                                    params.dbm_loss_function);
    dbm::Neural_network_trainer<float> trainer(params);

    dbm::Loss_function<float> loss_function(params);
    loss_function.calculate_ind_delta(train_y, prediction,
                                      ind_delta, params.dbm_loss_function, row_inds, n_samples);

    {
        dbm::Time_measurer time_measurer;
        trainer.train(nn, train_x, ind_delta, row_inds, n_samples, col_inds, 5);
    }

    nn->predict(train_x, prediction);

    loss_function.mean_function(prediction, params.dbm_loss_function);
    dbm::Matrix<float> result = dbm::hori_merge(train_y, prediction);
    result.print_to_file("result.txt");

    {
        ofstream out("save.txt");
        dbm::save_neural_network(nn, out);
    }


    dbm::Neural_network<float> *re_nn;
    {
        ifstream in("save.txt");
        dbm::load_neural_network(in, re_nn);
        ofstream out("re_save.txt");
        dbm::save_neural_network(re_nn, out);
    }

    delete re_nn;
    delete nn;
    re_nn = nullptr, nn = nullptr;
}
