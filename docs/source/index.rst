.. DBM documentation master file, created by
   sphinx-quickstart on Thu Mar  2 15:40:49 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DBM's documentation!
===============================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

This is the document of the Python APIs of Delta Boosting Machine. Classes and functions are listed and described.

* :ref:`genindex`

Classes
==================

.. autoclass:: dbm_py.interface.Matrix
   :members: __init__, shape, get, show, save, clear, assign, from_np2darray, to_np2darray

.. autoclass:: dbm_py.interface.Data_set
   :members: __init__, get_train_x, get_train_y, get_validate_x, get_validate_y

.. autoclass:: dbm_py.interface.Params
   :members: __init__, set_params, print_all

.. autoclass:: dbm_py.interface.DBM
   :members: __init__, train, predict, pdp, ss, calibrate_plot, interact, save_performance, save, load

.. autoclass:: dbm_py.interface.AUTO_DBM
   :members: __init__, train, predict, pdp, ss, calibrate_plot, interact, save_performance, save, load

Functions
=================

.. autofunction:: dbm_py.interface.np2darray_to_float_matrix

.. autofunction:: dbm_py.interface.float_matrix_to_np2darray

.. autofunction:: dbm_py.interface.string_to_params


