//
// Created by Xixuan Han on 10/10/2016.
//

#include "data_set.h"
#include "tools.h"

namespace dbm {

    template
    class Data_set<double>;

    template
    class Data_set<float>;

}

namespace dbm {

    template<typename T>
    Data_set<T>::Data_set(const Matrix<T> &data_x, const Matrix<T> &data_y, T test_portion):
            portion_for_test(test_portion) {

        no_samples = data_x.get_height(), no_features = data_x.get_width();

        no_test_samples = int(no_samples * portion_for_test), no_train_samples = no_samples - no_test_samples;

        int row_inds[no_samples];
        for (int i = 0; i < no_samples; ++i) {
            row_inds[i] = i;
        }

        shuffle(row_inds, no_samples);

        Matrix<T> temp_1 = data_x.rows(row_inds, no_train_samples);
        train_x = new Matrix<T>(no_train_samples, no_features, 0);
        copy(temp_1, *train_x);

        Matrix<T> temp_3 = data_y.rows(row_inds, no_train_samples);
        train_y = new Matrix<T>(no_train_samples, 1, 0);
        copy(temp_3, *train_y);

        Matrix<T> temp_2 = data_x.rows(&row_inds[no_train_samples], no_test_samples);
        test_x = new Matrix<T>(no_test_samples, no_features, 0);
        copy(temp_2, *test_x);

        Matrix<T> temp_4 = data_y.rows(&row_inds[no_train_samples], no_test_samples);
        test_y = new Matrix<T>(no_test_samples, 1, 0);
        copy(temp_4, *test_y);

    }

    template<typename T>
    Data_set<T>::~Data_set() {
        delete train_x, train_y, test_x, test_y;
    }

    template<typename T>
    const Matrix<T> &Data_set<T>::get_train_x() const {
        Matrix<T> const &ref_to_train_x = *train_x;
        return ref_to_train_x;
    }

    template<typename T>
    const Matrix<T> &Data_set<T>::get_train_y() const {
        Matrix<T> const &ref_to_train_y = *train_y;
        return ref_to_train_y;
    }

    template<typename T>
    const Matrix<T> &Data_set<T>::get_test_x() const {
        Matrix<T> const &ref_to_test_x = *test_x;
        return ref_to_test_x;
    }

    template<typename T>
    const Matrix<T> &Data_set<T>::get_test_y() const {
        Matrix<T> const &ref_to_test_y = *test_y;
        return ref_to_test_y;
    }

    template<typename T>
    void Data_set<T>::shuffle_all() {

        Matrix<T> data_x = vert_merge(*train_x, *test_x);
        Matrix<T> data_y = vert_merge(*train_y, *test_y);

        delete train_x, train_y, test_x, test_y;

        int row_inds[no_samples];
        for (int i = 0; i < no_samples; ++i) {
            row_inds[i] = i;
        }

        shuffle(row_inds, no_samples);

        Matrix<T> temp_1 = data_x.rows(row_inds, no_train_samples);
        train_x = new Matrix<T>(no_train_samples, no_features, 0);
        copy(temp_1, *train_x);

        Matrix<T> temp_3 = data_y.rows(row_inds, no_train_samples);
        train_y = new Matrix<T>(no_train_samples, 1, 0);
        copy(temp_3, *train_y);

        Matrix<T> temp_2 = data_x.rows(&row_inds[no_train_samples], no_test_samples);
        test_x = new Matrix<T>(no_test_samples, no_features, 0);
        copy(temp_2, *test_x);

        Matrix<T> temp_4 = data_y.rows(&row_inds[no_train_samples], no_test_samples);
        test_y = new Matrix<T>(no_test_samples, 1, 0);
        copy(temp_4, *test_y);

    }

}