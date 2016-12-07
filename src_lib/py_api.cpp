//
// Created by xixuan on 12/1/16.
//

#include <boost/python.hpp>
//#include <numpy/ndarrayobject.h>

#include <iostream>
#include <fstream>

#include "matrix.h"
#include "data_set.h"
#include "model.h"

using namespace boost::python;

BOOST_PYTHON_MODULE(lib_dbm_code_py)
{

    class_<dbm::Matrix<float>>("Float_Matrix", init<int, int>())
            .def(init<int, int, float>())
            .def(init<int, int, std::string>())
            .def(init<int, int, std::string, char>())

            .def("show", &dbm::Matrix<float>::print)
            .def("__str__", &dbm::Matrix<float>::print)
            .def("print_to_file", &dbm::Matrix<float>::print_to_file)

            .def("get_height", &dbm::Matrix<float>::get_height)
            .def("get_width", &dbm::Matrix<float>::get_width)
            .def("get", &dbm::Matrix<float>::get)

            .def("clear", &dbm::Matrix<float>::clear)

            .def("assign", &dbm::Matrix<float>::assign)
            ;

    class_<dbm::Data_set<float>>("Data_Set",
                                 init<const dbm::Matrix<float> &,
                                         const dbm::Matrix<float> &,
                                         float>())
            .def("get_train_x",
                 &dbm::Data_set<float>::get_train_x,
                 return_value_policy<copy_const_reference>())
            .def("get_train_y",
                 &dbm::Data_set<float>::get_train_y,
                 return_value_policy<copy_const_reference>())
            .def("get_test_x",
                 &dbm::Data_set<float>::get_test_x,
                 return_value_policy<copy_const_reference>())
            .def("get_test_y",
                 &dbm::Data_set<float>::get_test_y,
                 return_value_policy<copy_const_reference>())
            ;

    class_<dbm::Params>("Params")
            .def_readwrite("dbm_no_bunches_of_learners",
                           &dbm::Params::dbm_no_bunches_of_learners)
            .def_readwrite("dbm_no_candidate_feature",
                           &dbm::Params::dbm_no_candidate_feature)
            .def_readwrite("dbm_portion_train_sample",
                           &dbm::Params::dbm_portion_train_sample)
            .def_readwrite("dbm_no_cores",
                           &dbm::Params::dbm_no_cores)
            .def_readwrite("dbm_loss_function",
                           &dbm::Params::dbm_loss_function)
            .def_readwrite("dbm_display_training_progress",
                           &dbm::Params::dbm_display_training_progress)
            .def_readwrite("dbm_record_every_tree",
                           &dbm::Params::dbm_record_every_tree)
            .def_readwrite("dbm_freq_showing_loss_on_test",
                           &dbm::Params::dbm_freq_showing_loss_on_test)
            .def_readwrite("dbm_shrinkage",
                           &dbm::Params::dbm_shrinkage)

            .def_readwrite("dbm_portion_for_trees",
                           &dbm::Params::dbm_portion_for_trees)
            .def_readwrite("dbm_portion_for_lr",
                           &dbm::Params::dbm_portion_for_lr)
            .def_readwrite("dbm_portion_for_s",
                           &dbm::Params::dbm_portion_for_s)
            .def_readwrite("dbm_portion_for_k",
                           &dbm::Params::dbm_portion_for_k)
            .def_readwrite("dbm_portion_for_nn",
                           &dbm::Params::dbm_portion_for_nn)

            .def_readwrite("tweedie_p",
                           &dbm::Params::tweedie_p)

            .def_readwrite("splines_no_knot",
                           &dbm::Params::splines_no_knot)
            .def_readwrite("splines_regularization",
                           &dbm::Params::splines_regularization)

            .def_readwrite("kmeans_no_centroids",
                           &dbm::Params::kmeans_no_centroids)
            .def_readwrite("kmeans_max_iteration",
                           &dbm::Params::kmeans_max_iteration)
            .def_readwrite("kmeans_tolerance",
                           &dbm::Params::kmeans_tolerance)

            .def_readwrite("nn_no_hidden_neurons",
                           &dbm::Params::nn_no_hidden_neurons)
            .def_readwrite("nn_step_size",
                           &dbm::Params::nn_step_size)
            .def_readwrite("nn_validate_portion",
                           &dbm::Params::nn_validate_portion)
            .def_readwrite("nn_batch_size",
                           &dbm::Params::nn_batch_size)
            .def_readwrite("nn_max_iteration",
                           &dbm::Params::nn_max_iteration)
            .def_readwrite("nn_no_rise_of_loss_on_validate",
                           &dbm::Params::nn_no_rise_of_loss_on_validate)

            .def_readwrite("cart_max_depth",
                           &dbm::Params::cart_max_depth)
            .def_readwrite("cart_portion_candidate_split_point",
                           &dbm::Params::cart_portion_candidate_split_point)

            .def_readwrite("pdp_no_x_ticks",
                           &dbm::Params::pdp_no_x_ticks)
            .def_readwrite("pdp_no_resamplings",
                           &dbm::Params::pdp_no_resamplings)
            .def_readwrite("pdp_resampling_portion",
                           &dbm::Params::pdp_resampling_portion)
            .def_readwrite("pdp_ci_bandwidth",
                           &dbm::Params::pdp_ci_bandwidth)
            .def_readwrite("pdp_save_files",
                           &dbm::Params::pdp_save_files)
            ;

    def("set_params", &dbm::set_params);

    void (dbm::DBM<float>::*train_val_no_const)(const dbm::Data_set<float> &) = &dbm::DBM<float>::train;
    void (dbm::DBM<float>::*train_no_val_no_const)(const dbm::Matrix<float> &,
                                                   const dbm::Matrix<float> &) = &dbm::DBM<float>::train;
    void (dbm::DBM<float>::*train_no_val_const)(const dbm::Matrix<float> &,
                                                const dbm::Matrix<float> &,
                                                const dbm::Matrix<float> &) = &dbm::DBM<float>::train;
    void (dbm::DBM<float>::*train_val_const)(const dbm::Data_set<float> &,
                                             const dbm::Matrix<float> &) = &dbm::DBM<float>::train;

    dbm::Matrix<float> &(dbm::DBM<float>::*pdp_auto)(const dbm::Matrix<float> &,
                        const int &) = &dbm::DBM<float>::partial_dependence_plot;
    dbm::Matrix<float> &(dbm::DBM<float>::*pdp_min_max)(const dbm::Matrix<float> &,
                        const int &,
                        const float &,
                        const float &) = &dbm::DBM<float>::partial_dependence_plot;

    dbm::Matrix<float> &(dbm::DBM<float>::*predict_out)(const dbm::Matrix<float> &) = &dbm::DBM<float>::predict;
    void (dbm::DBM<float>::*predict_in_place)(const dbm::Matrix<float> &, dbm::Matrix<float> &) = &dbm::DBM<float>::predict;

    class_<dbm::DBM<float>>("DBM", init<const dbm::Params &>())
            .def("train_val_no_const", train_val_no_const)
            .def("train_no_val_no_const", train_no_val_no_const)
            .def("train_no_val_const", train_no_val_const)
            .def("train_val_const", train_val_const)

            .def("predict_out",
                 predict_out,
                 return_value_policy<copy_non_const_reference>())
            .def("predict_in_place",
                 predict_in_place)

            .def("pdp_auto",
                 pdp_auto,
                 return_value_policy<copy_non_const_reference>())
            .def("pdp_min_max",
                 pdp_min_max,
                 return_value_policy<copy_non_const_reference>())

            .def("statistical_significance",
                 &dbm::DBM<float>::statistical_significance,
                 return_value_policy<copy_non_const_reference>())

            .def("save_dbm", &dbm::DBM<float>::save_dbm_to)
            .def("load_dbm", &dbm::DBM<float>::load_dbm_from)
            ;

}