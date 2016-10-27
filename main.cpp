#include <iostream>
#include <algorithm>
#include <fstream>

#include "tools.h"
#include "data_set.h"
#include "base_learner_trainer.h"
#include "model.h"


using namespace std;

void test_save_load_dbm();

void test_save_load_tree();

void train_a_dbm();

void prepare_data();

void build_a_tree();

int main() {

    prepare_data();
    test_save_load_dbm();

    return 0;
}

void test_save_load_dbm() {
    int n_samples = 100000, n_features = 30, n_width = 31;

    dbm::Matrix<float> train_data(n_samples, n_width, "train_data.txt");
    dbm::Matrix<float> prediction(n_samples, 1, 0);
    dbm::Matrix<float> re_prediction(n_samples, 1, 0);

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
    string param_string = "no_learners 15 no_candidate_feature 5 "
            "no_train_sample 10000 max_depth 5 no_candidate_split_point 5";
    dbm::DBM<float> dbm(param_string);

    {
        dbm::Time_measurer *timer_1 = new dbm::Time_measurer();
        dbm.train(data_set);
    }

    dbm.predict(train_x, prediction);

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

    re_dbm->predict(train_x, re_prediction);

    dbm::Matrix<float> combined = dbm::hori_merge(prediction, re_prediction);
    combined.print_to_file("combined.txt");
}

void test_save_load_tree() {
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

    // ========================================================

    dbm::Params params = dbm::set_params("max_depth 4 no_candidate_split_point 5");

    dbm::Tree_node<float> *tree = new dbm::Tree_node<float>(0);
    dbm::Tree_trainer<float> trainer(params);

    {
        dbm::Time_measurer time_measurer;
        trainer.train(tree, train_x, train_y, prediction, row_inds, n_samples, col_inds, n_features);
//        trainer.prune(tree);
    }

    {
        dbm::Tree_info<float> info(tree);
        info.print_to_file("tree.txt");
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
        info.print_to_file("re_tree.txt");
    }
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
    string param_string = "no_learners 670 no_candidate_feature 5 "
            "no_train_sample 10000 max_depth 5 no_candidate_split_point 5";
    dbm::DBM<float> dbm(param_string);

//    dbm::Time_measurer * timer_0 = new dbm::Time_measurer();
//    dbm.train(train_x, train_y);
//    delete timer_0;

    dbm::Time_measurer *timer_1 = new dbm::Time_measurer();
    dbm.train(data_set);
    delete timer_1;

    dbm.predict(train_x, prediction);

    dbm::Matrix<float> temp = dbm::hori_merge(data_set.get_train_y(), *dbm.prediction_train_data);
    temp.print_to_file("train_result.txt");

    dbm::Matrix<float> result = dbm::hori_merge(train_y, prediction);
    result.print_to_file("whole_result.txt");
}

void prepare_data() {
    string file_name = "train_data.txt";
    dbm::make_data<float>(file_name);
}

void build_a_tree() {

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

    // ========================================================

    dbm::Params params = dbm::set_params("max_depth 10 no_candidate_split_point 5");

    dbm::Tree_node<float> *tree = new dbm::Tree_node<float>(0);
    dbm::Tree_trainer<float> trainer(params);

    dbm::Time_measurer time_measurer;

    trainer.train(tree, train_x, train_y, prediction, row_inds, n_samples, col_inds, n_features);

    time_measurer.~Time_measurer();

    cout << tree->get_type() << endl;

    tree->predict(train_x, prediction);

    dbm::Matrix<float> result = dbm::hori_merge(train_y, prediction);
    result.print_to_file("result.txt");

    // ========================================================

    dbm::Tree_info<float> original_tree_info(tree);
    original_tree_info.print_to_file("original_tree.txt");

    trainer.prune(tree);
    dbm::Tree_info<float> pruned_tree_info(tree);
    pruned_tree_info.print_to_file("pruned_tree.txt");

}