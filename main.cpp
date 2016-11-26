#include <iostream>
#include <fstream>

#include "data_set.h"
#include "base_learner.h"
#include "base_learner_trainer.h"
#include "model.h"

using namespace std;

void train_test_save_load_dbm();

void train_test_save_load_nn();

void train_test_save_load_s();

void train_test_save_load_k();

void test_save_load_tree();

void test_save_load_lr();

void train_a_dbm();

void prepare_data();

void build_a_tree();

int main() {

    prepare_data();
    train_test_save_load_dbm();

    return 0;

}

void prepare_data() {
    string file_name = "train_data.txt";
    dbm::make_data<float>(file_name, 100000, 30, 'n');
}

void train_test_save_load_dbm() {
    int n_samples = 100000, n_features = 30, n_width = 31;

    dbm::Matrix<float> train_data(n_samples, n_width, "train_data.txt");

    int row_inds[n_samples], col_inds[n_features];

    for (int i = 0; i < n_features; ++i)
        col_inds[i] = i;
    for (int i = 0; i < n_samples; ++i)
        row_inds[i] = i;

    dbm::Matrix<float> train_x = train_data.cols(col_inds, n_features);
    dbm::Matrix<float> train_y = train_data.col(n_features);

    // ================
    int mon_const[n_features] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    dbm::Data_set<float> data_set(train_x, train_y, 0.25);
    dbm::Matrix<float> train_prediction(int(0.75 * n_samples), 1, 0);
    dbm::Matrix<float> test_prediction(int(0.25 * n_samples), 1, 0);
    dbm::Matrix<float> re_test_prediction(int(0.25 * n_samples), 1, 0);

    // ================
    string param_string = "no_bunches_of_learners 51 no_cores 5 loss_function n "
            "no_train_sample 10000 no_candidate_feature 5 no_centroids 10 "
            "portion_for_trees 0 portion_for_lr 0 portion_for_s 0 "
            "portion_for_k 1 portion_for_nn 0";
    dbm::DBM<float> dbm(param_string);

    dbm.train(data_set);

    dbm.predict(data_set.get_train_x(), train_prediction);
    dbm.predict(data_set.get_test_x(), test_prediction);

    {
        ofstream out("dbm.txt");
        dbm::save_dbm(&dbm, out);
    }

    // ===================

    dbm::DBM<float> *re_dbm;

    {
        ifstream in("dbm.txt");
        dbm::load_dbm(in, re_dbm);
    }

    {
        ofstream out("re_dbm.txt");
        dbm::save_dbm(re_dbm, out);
    }

    re_dbm->predict(data_set.get_test_x(), re_test_prediction);

    dbm::Matrix<float> temp = dbm::hori_merge(*dbm.get_prediction_on_train_data(), train_prediction);
    dbm::Matrix<float> check = dbm::hori_merge(data_set.get_train_y(), temp);
    check.print_to_file("check_train_and_pred.txt");

    dbm::Matrix<float> combined = dbm::hori_merge(test_prediction, re_test_prediction);
    dbm::Matrix<float> result = dbm::hori_merge(data_set.get_test_y(), combined);
    result.print_to_file("whole_result.txt");
}

void train_test_save_load_k() {
    int n_samples = 10000, n_features = 30, n_width = 31;

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

    dbm::Params params = dbm::set_params("no_train_sample 10000 no_candidate_feature 2 no_centroids 10 "
                                                 "loss_function n kmeans_tolerance 1e-5");

    dbm::Kmeans<float> *kmeans = new dbm::Kmeans<float>(params.no_candidate_feature,
                                                         params.no_centroids,
                                                         params.loss_function);
    dbm::Kmeans_trainer<float> trainer(params);

    dbm::Loss_function<float> loss_function(params);
    loss_function.calculate_ind_delta(train_y, prediction,
                                      ind_delta, params.loss_function, row_inds, params.no_train_sample);

    {
        dbm::Time_measurer time_measurer;
        dbm::shuffle(col_inds, n_features);
        trainer.train(kmeans, train_x, ind_delta, params.loss_function,
                      row_inds, params.no_train_sample, col_inds, params.no_candidate_feature);
    }

    kmeans->predict(train_x, prediction);

    loss_function.mean_function(prediction, params.loss_function);
    dbm::Matrix<float> result = dbm::hori_merge(train_data, prediction);

    {
        ofstream out("save.txt");
        dbm::save_kmeans(kmeans, out);
    }


    dbm::Kmeans<float> *re_kmeans = nullptr;
    {
        ifstream in("save.txt");
        dbm::load_kmeans(in, re_kmeans);
        ofstream out("re_save.txt");
        dbm::save_kmeans(re_kmeans, out);
    }

    dbm::Matrix<float> re_prediction(n_samples, 1, 0);
    re_kmeans->predict(train_x, re_prediction);

    loss_function.mean_function(re_prediction, params.loss_function);

    dbm::Matrix<float> re_result = dbm::hori_merge(result, re_prediction);
    re_result.print_to_file("result.txt");

    delete re_kmeans, kmeans;
    re_kmeans = nullptr, kmeans = nullptr;
}

void train_test_save_load_s() {
    int n_samples = 100000, n_features = 30, n_width = 31;

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

    dbm::shuffle(col_inds, n_features);

    // ========================================================

    dbm::Params params = dbm::set_params("no_knot 5 loss_function b");

    dbm::Splines<float> *splines = new dbm::Splines<float>(params.no_hidden_neurons, params.loss_function);
    dbm::Splines_trainer<float> trainer(params);

    dbm::Loss_function<float> loss_function(params);
    loss_function.calculate_ind_delta(train_y, prediction,
                                      ind_delta, params.loss_function, row_inds, n_samples);

    {
        dbm::Time_measurer time_measurer;
        trainer.train(splines, train_x, ind_delta, row_inds, n_samples, col_inds, params.no_candidate_feature);
    }

    splines->predict(train_x, prediction);

    loss_function.mean_function(prediction, params.loss_function);
    dbm::Matrix<float> result = dbm::hori_merge(train_y, prediction);

    {
        ofstream out("save.txt");
        dbm::save_splines(splines, out);
    }


    dbm::Splines<float> *re_splines;
    {
        ifstream in("save.txt");
        dbm::load_splines(in, re_splines);
        ofstream out("re_save.txt");
        dbm::save_splines(re_splines, out);
    }

    dbm::Matrix<float> re_prediction(n_samples, 1, 0);
    re_splines->predict(train_x, re_prediction);

    loss_function.mean_function(re_prediction, params.loss_function);

    dbm::Matrix<float> re_result = dbm::hori_merge(result, re_prediction);
    re_result.print_to_file("result.txt");

    delete re_splines, splines;
    re_splines = nullptr, splines = nullptr;
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

    dbm::Neural_network<float> *nn = new dbm::Neural_network<float>(params.no_candidate_feature,
                                                                    params.no_hidden_neurons,
                                                                    params.loss_function);
    dbm::Neural_network_trainer<float> trainer(params);

    dbm::Loss_function<float> loss_function(params);
    loss_function.calculate_ind_delta(train_y, prediction,
                                      ind_delta, params.loss_function, row_inds, n_samples);

    {
        dbm::Time_measurer time_measurer;
        trainer.train(nn, train_x, ind_delta, row_inds, n_samples, col_inds, 5);
    }

    nn->predict(train_x, prediction);

    loss_function.mean_function(prediction, params.loss_function);
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

    delete re_nn, nn;
    re_nn = nullptr, nn = nullptr;
}

void train_a_dbm() {

    int n_samples = 100000, n_features = 30, n_width = 31;

    dbm::Matrix<float> train_data(n_samples, n_width, "train_data.txt");
    dbm::Matrix<float> prediction(n_samples, 1, 0);

    int row_inds[n_samples], col_inds[n_features];

    for (int i = 0; i < n_features; ++i)
        col_inds[i] = i;
    for (int i = 0; i < n_samples; ++i)
        row_inds[i] = i;

    dbm::Matrix<float> train_x = train_data.cols(col_inds, n_features);
    dbm::Matrix<float> train_y = train_data.col(n_features);

    // ================

    dbm::Data_set<float> data_set(train_x, train_y, 0.25);

    // ================
//    int mon_const[n_features] = {0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1,
//                                 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1};
    int mon_const[n_features] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    string param_string = "no_bunches_of_learners 670 no_candidate_feature 5 "
            "no_train_sample 10000 max_depth 5 no_candidate_split_point 5";
    dbm::DBM<float> dbm(param_string);

//    dbm::Time_measurer * timer_0 = new dbm::Time_measurer();
//    dbm.train(train_x, train_y);
//    delete timer_0;

    {
        dbm::Time_measurer timer_1;
        dbm.train(data_set, mon_const);
    }

    dbm.predict(train_x, prediction);

    dbm::Matrix<float> temp = dbm::hori_merge(data_set.get_train_y(), *dbm.get_prediction_on_train_data());
    temp.print_to_file("train_result.txt");

    dbm::Matrix<float> result = dbm::hori_merge(train_y, prediction);
    result.print_to_file("whole_result.txt");
}

void test_save_load_tree() {
    int n_samples = 100000, n_features = 30, n_width = 31;

    dbm::Matrix<float> train_data(n_samples, n_width, "train_data.txt");
    dbm::Matrix<float> prediction(n_samples, 1, 0);
    dbm::Matrix<float> ind_delta(n_samples, 1, 0);

    int row_inds[n_samples], col_inds[n_features];

    for (int i = 0; i < n_features; ++i)
        col_inds[i] = i;
    for (int i = 0; i < n_samples; ++i)
        row_inds[i] = i;

    dbm::Matrix<float> train_x = train_data.cols(col_inds, n_features);
    dbm::Matrix<float> train_y = train_data.col(n_features);

    // ========================================================

    dbm::Params params = dbm::set_params("max_depth 4 no_candidate_split_point 5");

    dbm::Tree_node<float> *tree = new dbm::Tree_node<float>(0);
    dbm::Tree_trainer<float> trainer(params);

    dbm::Loss_function<float> loss_function(params);
    loss_function.calculate_ind_delta(train_y, prediction,
                                      ind_delta, params.loss_function, row_inds, n_samples);

    {
        dbm::Time_measurer time_measurer;
        trainer.train(tree, train_x, train_y, ind_delta, prediction, nullptr, 'n', row_inds, n_samples, col_inds, n_features);
//        trainer.prune(tree);
    }

    {
        dbm::Tree_info<float> info(tree);
        info.print_to_file("tree.txt", 0);
    }

    {
        ofstream out("save.txt");
        dbm::save_tree_node(tree, out);
    }


    dbm::Tree_node<float> *re_tree;
    {
        ifstream in("save.txt");
        dbm::load_tree_node(in, re_tree);
        dbm::Tree_info<float> info(re_tree);
        info.print_to_file("re_tree.txt", 0);
    }
}

void test_save_load_lr() {
    prepare_data();

    int n_samples = 10000, n_features = 30, n_width = 31;

    dbm::Matrix<float> train_data(n_samples, n_width, "train_data.txt");
    dbm::Matrix<float> prediction(n_samples, 1, 0);
    dbm::Matrix<float> ind_delta(n_samples, 3, 0);

    int row_inds[n_samples], col_inds[n_features];

    for (int i = 0; i < n_features; ++i)
        col_inds[i] = i;
    for (int i = 0; i < n_samples; ++i)
        row_inds[i] = i;

    dbm::Matrix<float> train_x = train_data.cols(col_inds, n_features);
    dbm::Matrix<float> train_y = train_data.col(n_features);

    // ========================================================

    dbm::Params params = dbm::set_params("");

    dbm::Linear_regression<float> *lr = new dbm::Linear_regression<float>(n_features, 'n');
    dbm::Linear_regression_trainer<float> lr_trainer(params);

    dbm::Loss_function<float> loss_function(params);
    loss_function.calculate_ind_delta(train_y, prediction, ind_delta, params.loss_function);

    {
        dbm::Time_measurer time_measurer;
        lr_trainer.train(lr, train_x, ind_delta, row_inds, n_samples, col_inds, n_features);
    }

    {
        ofstream out("save.txt");
        dbm::save_linear_regression(lr, out);
    }


    dbm::Linear_regression<float> *re_lr;
    {
        ifstream in("save.txt");
        dbm::load_linear_regression(in, re_lr);
        ofstream out("re_save.txt");
        dbm::save_linear_regression(re_lr, out);
    }
}

void build_a_tree() {

    int n_samples = 100000, n_features = 30, n_width = 31;

    dbm::Matrix<float> train_data(n_samples, n_width, "train_data.txt");
    dbm::Matrix<float> prediction(n_samples, 1, 0);
    dbm::Matrix<float> ind_delta(n_samples, 1, 0);

    int row_inds[n_samples], col_inds[n_features];

    for (int i = 0; i < n_features; ++i)
        col_inds[i] = i;
    for (int i = 0; i < n_samples; ++i)
        row_inds[i] = i;

    dbm::Matrix<float> train_x = train_data.cols(col_inds, n_features);
    dbm::Matrix<float> train_y = train_data.col(n_features);

    // ========================================================

    dbm::Params params = dbm::set_params("max_depth 10 no_candidate_split_point 5");

    dbm::Tree_node<float> *tree = new dbm::Tree_node<float>(0);
    dbm::Tree_trainer<float> trainer(params);

    dbm::Time_measurer time_measurer;

    dbm::Loss_function<float> loss_function(params);
    loss_function.calculate_ind_delta(train_y, prediction,
                                      ind_delta, params.loss_function, row_inds, n_samples);

    trainer.train(tree, train_x, train_y, ind_delta, prediction, nullptr, 'n', row_inds, n_samples, col_inds, n_features);

    time_measurer.~Time_measurer();

    cout << tree->get_type() << endl;

    tree->predict(train_x, prediction);

    dbm::Matrix<float> result = dbm::hori_merge(train_y, prediction);
    result.print_to_file("result.txt");

    // ========================================================

    dbm::Tree_info<float> original_tree_info(tree);
    original_tree_info.print_to_file("original_tree.txt", 0);

    trainer.prune(tree);
    dbm::Tree_info<float> pruned_tree_info(tree);
    pruned_tree_info.print_to_file("pruned_tree.txt", 0);

}




