#include <iostream>
#include <fstream>

#include "data_set.h"
#include "base_learner.h"
#include "base_learner_trainer.h"
#include "model.h"

using namespace std;

void train_test_save_load_auto_dbm();

void train_test_save_load_dbm();

void train_test_save_load_nn();

void train_test_save_load_dpcs();

void prepare_data();

int main() {

    train_test_save_load_dbm();
//    train_test_save_load_auto_dbm();

    return 0;

}

void prepare_data() {
    string file_name = "train_data.txt";
    dbm::make_data<float>(file_name, 10000, 30, 'b');
}

void train_test_save_load_auto_dbm() {
    int n_samples = 136573, n_features = 50, n_width = 51;

    dbm::Matrix<float> train_data(n_samples, n_width, "numerai_training_data.csv", ',');

    int *col_inds = new int[n_features];

    for (int i = 0; i < n_features; ++i)
        col_inds[i] = i;

    dbm::Matrix<float> train_x = train_data.cols(col_inds, n_features);
    dbm::Matrix<float> train_y = train_data.col(n_features);

    // ================

    dbm::Data_set<float> data_set(train_x, train_y, 0.2);
    dbm::Matrix<float> train_prediction(data_set.get_train_x().get_height(), 1, 0);
    dbm::Matrix<float> test_prediction(data_set.get_test_x().get_height(), 1, 0);
    dbm::Matrix<float> re_test_prediction(data_set.get_test_x().get_height(), 1, 0);

    // ================
    string param_string = "dbm_no_bunches_of_learners 1001 dbm_no_cores 1 dbm_loss_function b "
            "dbm_portion_train_sample 1 dbm_no_candidate_feature 5 dbm_shrinkage 0.02";
    dbm::Params params = dbm::set_params(param_string);
    dbm::AUTO_DBM<float> auto_dbm(params);

    auto_dbm.train(data_set);

//    auto_dbm.train_two_way_model(data_set.get_train_x());
//    dbm::Matrix<float> twm_pred = auto_dbm.predict_two_way_model(data_set.get_train_x());

    auto_dbm.predict(data_set.get_train_x(), train_prediction);
    auto_dbm.predict(data_set.get_test_x(), test_prediction);

    dbm::Matrix<float> pred = auto_dbm.predict(data_set.get_test_x());
    pred.print_to_file("pred.txt");

//    dbm::Matrix<float> pdp = auto_dbm.partial_dependence_plot(data_set.get_train_x(), 6);
//    pdp.print_to_file("pdp.txt");

//    dbm::Matrix<float> ss = auto_dbm.statistical_significance(data_set.get_train_x());
//    ss.print_to_file("ss.txt");

    auto_dbm.save_auto_dbm_to("dbm.txt");

    // ===================

    dbm::AUTO_DBM<float> re_auto_dbm(params);

    re_auto_dbm.load_auto_dbm_from("dbm.txt");

    re_auto_dbm.save_auto_dbm_to("re_dbm.txt");

    re_auto_dbm.predict(data_set.get_test_x(), re_test_prediction);

//    dbm::Matrix<float> temp = dbm::hori_merge(*auto_dbm.get_prediction_on_train_data(), dbm::hori_merge(train_prediction, twm_pred));
    dbm::Matrix<float> temp = dbm::hori_merge(*auto_dbm.get_prediction_on_train_data(), train_prediction);
//    dbm::Matrix<float> check = dbm::hori_merge(dbm::hori_merge(data_set.get_train_x(), data_set.get_train_y()), temp);
    dbm::Matrix<float> check = dbm::hori_merge(data_set.get_train_y(), temp);
    check.print_to_file("check_train_and_pred.txt");

    dbm::Matrix<float> combined = dbm::hori_merge(test_prediction, re_test_prediction);
//    dbm::Matrix<float> result = dbm::hori_merge(dbm::hori_merge(data_set.get_test_x(), data_set.get_test_y()), combined);
    dbm::Matrix<float> result = dbm::hori_merge(data_set.get_test_y(), combined);
    result.print_to_file("whole_result.txt");

    dbm::Matrix<float> oos_data(13512, 50, "numerai_tournament_data.csv", ',');
    dbm::Matrix<float> oos_prediction = oos_data.col(0);
    oos_prediction.clear();
    auto_dbm.predict(oos_data, oos_prediction);


    oos_prediction.print_to_file("oos_prediction.txt");

    delete[] col_inds;
}

void train_test_save_load_dbm() {
    int n_samples = 136573, n_features = 50, n_width = 51;

    dbm::Matrix<float> train_data(n_samples, n_width, "numerai_training_data.csv", ',');

    int *col_inds = new int[n_features];

    for (int i = 0; i < n_features; ++i)
        col_inds[i] = i;

//    dbm::Matrix<float> train_x = train_data.cols(col_inds, n_features);
//    dbm::Matrix<float> train_y = train_data.col(n_features);

    int no_rows = 136573;
    int *row_inds = new int[no_rows];
    for(int i = 0; i < no_rows; ++i)
        row_inds[i] = i;

    dbm::Matrix<float> train_x = train_data.submatrix(row_inds, no_rows, col_inds, n_features);
    dbm::Matrix<float> train_y = train_data.col(n_features).rows(row_inds, no_rows);

    // ================

    dbm::Data_set<float> data_set(train_x, train_y, 0.1);
    dbm::Matrix<float> train_prediction(data_set.get_train_x().get_height(), 1, 0);
    dbm::Matrix<float> test_prediction(data_set.get_test_x().get_height(), 1, 0);
    dbm::Matrix<float> re_test_prediction(data_set.get_test_x().get_height(), 1, 0);

    // ================
    /*
     * no_rows = 136573
     * best: 1200 1 0.5 50 0.02 1
     */
    string param_string = "dbm_no_bunches_of_learners 1200 dbm_no_cores 1 dbm_loss_function b "
            "dbm_portion_train_sample 0.5 dbm_no_candidate_feature 50 dbm_shrinkage 0.02 "
            "dbm_portion_for_trees 1 dbm_portion_for_lr 0 dbm_portion_for_s 0 "
            "dbm_portion_for_k 0 dbm_portion_for_nn 0 dbm_portion_for_d 0 "
            "cart_prune 1";
    dbm::Params params = dbm::set_params(param_string);
    dbm::DBM<float> dbm(params);

    dbm.train(data_set);

//    dbm.train_two_way_model(data_set.get_train_x());
//    dbm::Matrix<float> twm_pred = dbm.predict_two_way_model(data_set.get_train_x());

    dbm.predict(data_set.get_train_x(), train_prediction);
    dbm.predict(data_set.get_test_x(), test_prediction);

    dbm::Matrix<float> pred = dbm.predict(data_set.get_test_x());
    pred.print_to_file("pred.txt");

//    dbm::Matrix<float> pdp = dbm.partial_dependence_plot(data_set.get_train_x(), 6);
//    pdp.print_to_file("pdp.txt");
//
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

//    dbm::Matrix<float> temp = dbm::hori_merge(*dbm.get_prediction_on_train_data(), dbm::hori_merge(train_prediction, twm_pred));
    dbm::Matrix<float> temp = dbm::hori_merge(*dbm.get_prediction_on_train_data(), train_prediction);
//    dbm::Matrix<float> check = dbm::hori_merge(dbm::hori_merge(data_set.get_train_x(), data_set.get_train_y()), temp);
    dbm::Matrix<float> check = dbm::hori_merge(data_set.get_train_y(), temp);
    check.print_to_file("check_train_and_pred.txt");

    dbm::Matrix<float> combined = dbm::hori_merge(test_prediction, re_test_prediction);
//    dbm::Matrix<float> result = dbm::hori_merge(dbm::hori_merge(data_set.get_test_x(), data_set.get_test_y()), combined);
    dbm::Matrix<float> result = dbm::hori_merge(data_set.get_test_y(), combined);
    result.print_to_file("whole_result.txt");

    dbm::Matrix<float> oos_data(13512, 50, "numerai_tournament_data.csv", ',');
    dbm::Matrix<float> oos_prediction = oos_data.col(0);
    oos_prediction.clear();
    dbm.predict(oos_data, oos_prediction);

    oos_prediction.print_to_file("oos_prediction.txt");

    delete[] col_inds;
    delete[] row_inds;
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

    loss_function.link_function(prediction, params.dbm_loss_function);
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

void train_test_save_load_dpcs() {
    int n_samples = 10000, n_features = 2, n_width = 3;

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

    dbm::Params params = dbm::set_params("dbm_no_candidate_feature 2 dbm_loss_function n");

    dbm::DPC_stairs<float> *dpc_stairs = new dbm::DPC_stairs<float>(params.dbm_no_candidate_feature,
                                                                    params.dbm_loss_function,
                                                                    params.dpcs_no_ticks);
    dbm::DPC_stairs_trainer<float> trainer(params);

    dbm::Loss_function<float> loss_function(params);
    loss_function.calculate_ind_delta(train_y, prediction,
                                      ind_delta, params.dbm_loss_function, row_inds, n_samples);

    {
        dbm::Time_measurer time_measurer;
        trainer.train(dpc_stairs, train_x, ind_delta, row_inds, n_samples, col_inds, 2);
    }

    dpc_stairs->predict(train_x, prediction);

    loss_function.link_function(prediction, params.dbm_loss_function);
    dbm::Matrix<float> result = dbm::hori_merge(train_y, prediction);
    result.print_to_file("result.txt");

    {
        ofstream out("save.txt");
        dbm::save_dpc_stairs(dpc_stairs, out);
    }


    dbm::DPC_stairs<float> *re_dpc_stairs;
    {
        ifstream in("save.txt");
        dbm::load_dpc_stairs(in, re_dpc_stairs);
        ofstream out("re_save.txt");
        dbm::save_dpc_stairs(re_dpc_stairs, out);
    }

    delete re_dpc_stairs;
    delete dpc_stairs;
}
