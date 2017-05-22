from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import tensorflow as tf

DATAINPUT = 'GASQUAKE'
estimatortype='REGRESSOR'
learningrate = 0.1
regularizationstrength = 0.001
numberofgroups=2
hiddenunits = None
modellocation = None
metrics=None


if (DATAINPUT =='TEST'):
    COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio", "medv"]
    FEATURES = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio"]
    LABEL = ["medv"]
    estimatortype = 'REGRESSOR'
    modellocation = "/tmp/boston_model"

    training_set = pd.read_csv("data/boston/boston_train.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS)
    test_set = pd.read_csv("data/boston/boston_test.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
    prediction_set = pd.read_csv("data/boston/boston_predict.csv", skipinitialspace=True,
                                 skiprows=1, names=COLUMNS)
    if estimatortype=='REGRESSOR':
        learningrate = 0.1
        regularizationstrength = 0.001
        hiddenunits = [10, 10, 8]
        metrics={
            "accuracy":
               tf.contrib.learn.MetricSpec(
                   metric_fn=tf.contrib.metrics.streaming_accuracy,
                   prediction_key=tf.contrib.learn.PredictionKey.SCORES
               )
           }


if (DATAINPUT == 'QUAKE'):
    COLUMNS = ["gaspro", "gaspre", "quakes"]
    FEATURES = ["gaspro", "gaspre"]
    LABEL = ["quakes"]



    modellocation = "/tmp/quake_model"
    estimatortype = 'CLASSIFIER'

    training_set = pd.read_csv("data/quake/quake_train.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS)
    test_set = pd.read_csv("data/quake/quake_test.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
    prediction_set = pd.read_csv("data/quake/quake_predict.csv", skipinitialspace=True,
                                 skiprows=1, names=COLUMNS)

    if estimatortype=='REGRESSOR':
        hiddenunits = [10, 3, 10]
        learningrate = 0.1
        regularizationstrength = 0.001

        metrics={
            "accuracy":
               tf.contrib.learn.MetricSpec(
                   metric_fn=tf.contrib.metrics.streaming_accuracy,
                   prediction_key=tf.contrib.learn.PredictionKey.SCORES
               )
           }
    if estimatortype=='CLASSIFIER':
        hiddenunits = [10, 3, 10]
        learningrate = 0.1
        numberofgroups = 25

        metrics = {
            "accuracy":
                tf.contrib.learn.MetricSpec(
                    metric_fn=tf.contrib.metrics.streaming_accuracy,
                    prediction_key=tf.contrib.learn.PredictionKey.CLASSES
                )
        }



if DATAINPUT == 'GASPRESURE':
    COLUMNS = ["time", "gaspre"]
    FEATURES = ["time"]
    LABEL = ["gaspre"]

    estimatortype = 'REGRESSOR'
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

    if estimatortype == 'REGRESSOR':
        hiddenunits = [10, 10, 10]
        learningrate = 0.3
        regularizationstrength = 0.001

        metrics = {
            "accuracy":
                tf.contrib.learn.MetricSpec(
                    metric_fn=tf.contrib.metrics.streaming_accuracy,
                    prediction_key=tf.contrib.learn.PredictionKey.SCORES
                )
        }


if DATAINPUT == 'GASQUAKE':
    COLUMNS = ["gaspro", "gaspre", "quakes", "quakes15"]
    FEATURES = ["gaspro", "gaspre"]
    LABEL = ["quakes", "quakes15"]

    modellocation = "/tmp/gastestmodel"
    estimatortype='REGRESSOR'

    training_set = pd.read_csv("data/gastest/gastest_train.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS)
    test_set = pd.read_csv("data/gastest/gastest_test.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS)
    prediction_set = pd.read_csv("data/gastest/gastest_predict.csv", skipinitialspace=True,
                                 skiprows=1, names=COLUMNS)

    if estimatortype=='REGRESSOR':
        learningrate = 0.1
        regularizationstrength = 0.001
        hiddenunits = [10, 10, 8]
        metrics={
            "accuracy":
               tf.contrib.learn.MetricSpec(
                   metric_fn=tf.contrib.metrics.streaming_accuracy,
                   prediction_key=tf.contrib.learn.PredictionKey.SCORES
               )
           }

    if estimatortype=='CLASSIFIER':
        learningrate = 0.1
        hiddenunits = [10, 10, 8]
        numberofgroups = 20  # Classifier not yet working with multiple labels as regressor
        metrics = {
            "accuracy":
                tf.contrib.learn.MetricSpec(
                    metric_fn=tf.contrib.metrics.streaming_accuracy,
                    prediction_key=tf.contrib.learn.PredictionKey.CLASSES
                )
        }


def input_fn(data_set):
    feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
    #print(feature_cols)
    labels = tf.constant(data_set[LABEL].values)
    #print(labels)
    return feature_cols, labels

#error melding maken als er geen estimator is geselecteerd
def estimatorinput():
    feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]
    estimator = None
    if estimatortype=='REGRESSOR':
        estimator = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                                  hidden_units=hiddenunits,
                                                  activation_fn=tf.nn.relu,
                                                  optimizer=tf.train.ProximalAdagradOptimizer(
                                                      learning_rate=learningrate,
                                                      l1_regularization_strength=regularizationstrength
                                                  ),
                                                  model_dir=modellocation+"/regression",
                                                  label_dimension=LABEL.__len__()
                                                   )

    if estimatortype=='CLASSIFIER':
        estimator = tf.contrib.learn.DNNClassifier(feature_columns=feature_cols,
                                                    hidden_units=hiddenunits,
                                                    n_classes=numberofgroups,
                                                   activation_fn=tf.nn.relu,
                                                   optimizer=tf.train.AdagradOptimizer(
                                                       learning_rate=learningrate,
                                                   ),
                                                   model_dir=modellocation+"/classification",
                                                    )

    return estimator
