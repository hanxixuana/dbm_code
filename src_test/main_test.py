#!/usr/bin/env python

import numpy as np
import os
import pandas as pd
import sys

sys.path.append(os.getcwd() + "/api")

import dbm_py.dbm_py as dbm

x = dbm.Matrix(5000, 10)
x_nd_array = dbm.float_matrix_to_np2darray(x)
y_nd_array = x_nd_array[:, 0] ** 2 / 2  - \
             x_nd_array[:, 3] * 2 * np.exp(x_nd_array[:, 6] / 10) + \
             x_nd_array[:, 8] / 2 + \
             np.random.randn(5000)

y = dbm.np2darray_to_float_matrix(y_nd_array[:, np.newaxis])

c = dbm.Data_set(x, y, 0.2)
train_x = c.get_train_x()

s = 'no_bunches_of_learners 41 no_train_sample 3500 no_cores 0 shrinkage 0.04'

params = dbm.Params()
params.set_params(s)

model = dbm.DBM(params)

model.train(c)

predict = model.predict(c.get_test_x())

result = pd.DataFrame(np.concatenate([predict.to_np2darray(), c.get_test_y().to_np2darray()], 1))

model.save('dbm.txt')

pdp = model.pdp(c.get_train_x(), 3)

ss = model.ss(c.get_train_x())

re_model = dbm.DBM(dbm.Params())
re_model.load('dbm.txt')

re_predict = re_model.predict(c.get_test_x())

re_result = pd.DataFrame(np.concatenate([predict.to_np2darray(),
                                         re_predict.to_np2darray(),
                                         c.get_test_y().to_np2darray()], 1))

re_pdp = re_model.pdp(c.get_train_x(), 3)

re_ss = re_model.ss(c.get_train_x())

re_model.train(c)