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
            .def_readwrite("no_bunches_of_learners",
                           &dbm::Params::no_bunches_of_learners)
            .def_readwrite("no_candidate_feature",
                           &dbm::Params::no_candidate_feature)
            .def_readwrite("no_train_sample",
                           &dbm::Params::no_train_sample)
            .def_readwrite("no_cores",
                           &dbm::Params::no_cores)
            .def_readwrite("loss_function",
                           &dbm::Params::loss_function)
            .def_readwrite("display_training_progress",
                           &dbm::Params::display_training_progress)
            .def_readwrite("record_every_tree",
                           &dbm::Params::record_every_tree)
            .def_readwrite("freq_showing_loss_on_test",
                           &dbm::Params::freq_showing_loss_on_test)
            .def_readwrite("shrinkage",
                           &dbm::Params::shrinkage)

            .def_readwrite("portion_for_trees",
                           &dbm::Params::portion_for_trees)
            .def_readwrite("portion_for_lr",
                           &dbm::Params::portion_for_lr)
            .def_readwrite("portion_for_s",
                           &dbm::Params::portion_for_s)
            .def_readwrite("portion_for_k",
                           &dbm::Params::portion_for_k)
            .def_readwrite("portion_for_nn",
                           &dbm::Params::portion_for_nn)

            .def_readwrite("tweedie_p",
                           &dbm::Params::tweedie_p)

            .def_readwrite("no_knot",
                           &dbm::Params::no_knot)
            .def_readwrite("regularization",
                           &dbm::Params::regularization)

            .def_readwrite("no_centroids",
                           &dbm::Params::no_centroids)
            .def_readwrite("kmeans_max_iteration",
                           &dbm::Params::kmeans_max_iteration)
            .def_readwrite("kmeans_tolerance",
                           &dbm::Params::kmeans_tolerance)

            .def_readwrite("no_hidden_neurons",
                           &dbm::Params::no_hidden_neurons)
            .def_readwrite("step_size",
                           &dbm::Params::step_size)
            .def_readwrite("validate_portion",
                           &dbm::Params::validate_portion)
            .def_readwrite("batch_size",
                           &dbm::Params::batch_size)
            .def_readwrite("nn_max_iteration",
                           &dbm::Params::nn_max_iteration)
            .def_readwrite("no_rise_of_loss_on_validate",
                           &dbm::Params::no_rise_of_loss_on_validate)

            .def_readwrite("max_depth",
                           &dbm::Params::max_depth)
            .def_readwrite("no_candidate_split_point",
                           &dbm::Params::no_candidate_split_point)

            .def_readwrite("no_x_ticks",
                           &dbm::Params::no_x_ticks)
            .def_readwrite("no_resamplings",
                           &dbm::Params::no_resamplings)
            .def_readwrite("resampling_portion",
                           &dbm::Params::resampling_portion)
            .def_readwrite("ci_bandwidth",
                           &dbm::Params::ci_bandwidth)
            .def_readwrite("save_files",
                           &dbm::Params::save_files)
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