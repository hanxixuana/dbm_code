#!/usr/bin/env python

import lib_dbm_code_py as dbm
import numpy as np


class Matrix(object):

    def __init__(self, height = None, width = None, val = None, file_name = None, sep = None, mat = None):

        if(height is not None and width is not None and mat is None):

            if(val is None and file_name is None and sep is None):
                self.mat = dbm.Float_Matrix(height, width)

            elif(val is not None and file_name is None and sep is None):
                self.mat = dbm.Float_Matrix(height, width, val)

            elif(val is None and file_name is not None and sep is None):
                self.mat = dbm.Float_Matrix(height, width, file_name)

            elif(val is None and file_name is not None and sep is not None):
                self.mat = dbm.Float_Matrix(height, width, file_name, sep)

            else:
                raise ValueError('Error!')

        elif(height is None and width is None and mat is not None):
            self.mat = mat

    # shape
    def shape(self):
        return [self.mat.get_height(), self.mat.get_width()]

    # get
    def get(self, i, j):
        return self.mat.get(i, j)

    # print to screen
    def show(self):
        self.mat.show()

    # save to file
    def save(self, file_name, sep = '\t'):
        self.mat.print_to_file(file_name, sep)

    # clear all values to 0
    def clear(self):
        self.mat.clear()

    # elementwise assignment
    def assign(self, i, j, val):
        self.mat.assign(i, j, val)

    # conversion of Numpy 2d array to Matrix
    def from_np2darray(self, source):
        try:
            assert source.shape.__len__() == 2
            assert self.mat.get_height() == source.shape[0]
            assert self.mat.get_width() == source.shape[1]
        except AssertionError as e:
            print(source.shape)
            print((self.mat.get_height(), self.mat.get_width()))
            raise ValueError('The Numpy array may not have the same shape as the target.')
        for i in range(self.mat.get_height()):
            for j in range(self.mat.get_width()):
                self.mat.assign(i, j, source[i][j])

    # conversion of Matrix to Numpy 2d array
    def to_np2darray(self):
        result = np.zeros([self.mat.get_height(), self.mat.get_width()])
        for i in range(self.mat.get_height()):
            for j in range(self.mat.get_width()):
                result[i][j] = self.mat.get(i, j)
        return result



class Data_set(object):

    def __init__(self, data_x, data_y, portion_for_validating):
        self.data_set = dbm.Data_Set(data_x.mat, data_y.mat, portion_for_validating)

    def get_train_x(self):
        return Matrix(mat = self.data_set.get_train_x())

    def get_train_y(self):
        return Matrix(mat = self.data_set.get_train_y())

    def get_test_x(self):
        return Matrix(mat = self.data_set.get_test_x())

    def get_test_y(self):
        return Matrix(mat = self.data_set.get_test_y())

class Params(object):

    def __init__(self, params = None):
        if params is None:
            self.params = dbm.Params()
        else:
            self.params = params

    def set_params(self, string, sep = ' '):
        self.params = dbm.set_params(string, sep)

    def print_all(self):
        attrs = [attr for attr in dir(self.params) if not callable(attr) and not attr.startswith("__")]
        for attr in attrs:
            print("%s = %s" % (attr, getattr(self.params, attr)))

class DBM(object):

    def __init__(self, params):
        self.dbm = dbm.DBM(params.params)

    def train(self, data_set):
        self.dbm.train_val_no_const(data_set.data_set)

    def predict(self, data_x):
        data_y = Matrix(data_x.shape()[0], 1, 0)
        self.dbm.predict_in_place(data_x.mat, data_y.mat)
        return data_y

    def pdp(self, data_x, feature_index):
        return Matrix(mat = self.dbm.pdp_auto(data_x.mat, feature_index))

    def ss(self, data_x):
        return Matrix(mat = self.dbm.statistical_significance(data_x.mat))

    def save(self, file_name):
        self.dbm.save_dbm(file_name)

    def load(self, file_name):
        self.dbm.load_dbm(file_name)

class AUTO_DBM(object):

    def __init__(self, params):
        self.dbm = dbm.AUTO_DBM(params.params)

    def train(self, data_set):
        self.dbm.train_val_no_const(data_set.data_set)

    def predict(self, data_x):
        data_y = Matrix(data_x.shape()[0], 1, 0)
        self.dbm.predict_in_place(data_x.mat, data_y.mat)
        return data_y

    def pdp(self, data_x, feature_index):
        return Matrix(mat = self.dbm.pdp_auto(data_x.mat, feature_index))

    def ss(self, data_x):
        return Matrix(mat = self.dbm.statistical_significance(data_x.mat))

    def save(self, file_name):
        self.dbm.save_dbm(file_name)

    def load(self, file_name):
        self.dbm.load_dbm(file_name)

def np2darray_to_float_matrix(source):
    try:
        assert type(source) is np.ndarray
    except AssertionError as e:
        raise ValueError('The argument may not be a Numpy array.')
    try:
        assert source.shape.__len__() == 2
    except AssertionError as e:
        raise ValueError('The argument may not be a 2d array.')
    target = Matrix(source.shape[0], source.shape[1], 0)
    target.from_np2darray(source)
    return target

def float_matrix_to_np2darray(source):
    try:
        assert type(source) is Matrix
    except AssertionError as e:
        raise ValueError('The argument may not be a Matrix.')
    return source.to_np2darray()

def string_to_params(string, sep = ' '):
    return Params(params = dbm.set_params(string, sep))