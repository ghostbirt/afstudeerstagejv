from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import tensorflow as tf

DUMMYTEST = 4
learningrate = 0.1
regularizationstrength = 0.001
hiddenunits = None
modellocation = None
regressor = None
metrics=None

if (DUMMYTEST == 0):
    COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio", "medv"]
    FEATURES = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio"]
    LABEL = ["medv"]
    learningrate = 0.1
    regularizationstrength = 0.001
    hiddenunits = [10, 10]
    modellocation = "/tmp/boston_model"

    training_set = pd.read_csv("data/boston/boston_train.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS)
    test_set = pd.read_csv("data/boston/boston_test.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
    prediction_set = pd.read_csv("data/boston/boston_predict.csv", skipinitialspace=True,
                                 skiprows=1, names=COLUMNS)

if (DUMMYTEST == 1):
    COLUMNS = ["gaspro", "gaspre", "quakes"]
    FEATURES = ["gaspro", "gaspre"]
    LABEL = ["quakes"]

    learningrate = 0.4
    regularizationstrength = 0.001
    hiddenunits = [10, 10]
    modellocation = "/tmp/quake_model"

    training_set = pd.read_csv("data/quake/quake_train.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS)
    test_set = pd.read_csv("data/quake/quake_test.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
    prediction_set = pd.read_csv("quake_predict.csv", skipinitialspace=True,
                                 skiprows=1, names=COLUMNS)

if (DUMMYTEST == 2):
    COLUMNS = ["time", "gaspre"]
    FEATURES = ["time"]
    LABEL = ["gaspre"]

    learningrate = 0.4
    regularizationstrength = 0.001
    hiddenunits = [10, 10,10]
    modellocation = "/tmp/gaspresure"

    training_set = pd.read_csv("gaspresure.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS)
    test_set = pd.read_csv("gaspresure2.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
    prediction_set = pd.read_csv("gaspresure3.csv", skipinitialspace=True,
                                 skiprows=1, names=COLUMNS)




    validation_metrics = {
            "accuracy":
                tf.contrib.learn.MetricSpec(
                    metric_fn=tf.contrib.metrics.streaming_accuracy,
                    prediction_key=tf.contrib.learn.PredictionKey.CLASSES
                    )

    }




if (DUMMYTEST == 3):
    COLUMNS = ["time", "gaspre"]
    FEATURES = ["time"]
    LABEL = ["gaspre"]

    learningrate = 0.1
    regularizationstrength = 0.001
    hiddenunits = [10, 1,10]
    modellocation = "/tmp/gaspresure_model"

    training_set = pd.read_csv("data/gaspresurev2/PressureDatav2_train.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS)
    test_set = pd.read_csv("data/gaspresurev2/PressureDatav2_test.csv", skipinitialspace=True,
                                 skiprows=1, names=COLUMNS)
    prediction_set = pd.read_csv("data/gaspresurev2/PressureDatav2_predict.csv", skipinitialspace=True,
                                 skiprows=1, names=COLUMNS)

if (DUMMYTEST == 4):
    COLUMNS = ["gaspro", "gaspre", "quakes", "quakes15"]
    FEATURES = ["gaspro", "gaspre"]
    LABEL = ["quakes", "quakes15"]

    learningrate = 0.1
    regularizationstrength = 0.001
    hiddenunits = [10, 10]
    modellocation = "/tmp/gastes_tmodel"

    training_set = pd.read_csv("data/gastest/gastest_train.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS)
    test_set = training_set
    prediction_set = pd.read_csv("data/gastest/gastest_predict.csv", skipinitialspace=True,
                                 skiprows=1, names=COLUMNS)



def input_fn(data_set):
    feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
    #print(feature_cols)
    labels = tf.constant(data_set[LABEL].values)
    #print(labels)
    return feature_cols, labels


def regressorinput():
    feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]
    regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                              hidden_units=hiddenunits,
                                              activation_fn=tf.nn.relu,
                                              optimizer=tf.train.ProximalAdagradOptimizer(
                                                  learning_rate=learningrate,
                                                  l1_regularization_strength=regularizationstrength
                                              ),
                                              model_dir=modellocation,
                                              label_dimension=LABEL.__len__()
                                               )


    return regressor
