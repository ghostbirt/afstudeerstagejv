from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf

DATAINPUT = 'QUAKE'
estimatortype = 'REGRESSOR'
learningrate = 0.1
regularizationstrength = 0.001
numberofgroups = 2              # Defines how many classes are used for the CLASSIFICATION model
hiddenunits = None              # Defines the neural network structure
modellocation = None            # Defines the place the place where the model will be stored
metrics = None                  # Defines the metrics that are being used to monitor the process,
                                # Loss is standard being monitored

# Defines the inputdata for the model and sets what variables to use with the model.
# The input exist of columns with has all the different fields that are used in the dataset.
# Features defines the columns that are the known values
# Label defines the columns that are the unknown values that are being predicted

if (DATAINPUT == 'TEST'):  # Testdataset using the boston dataset
    COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio", "medv"]
    FEATURES = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio"]
    LABEL = ["medv"]
    estimatortype = 'REGRESSOR'
    modellocation = "/tmp/boston_model"

    # Defines the training set, test set and the prediction set to use for the model
    training_set = pd.read_csv("data/boston/boston_train.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS)
    test_set = pd.read_csv("data/boston/boston_test.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
    prediction_set = pd.read_csv("data/boston/boston_predict.csv", skipinitialspace=True,
                                 skiprows=1, names=COLUMNS)
    if estimatortype == 'REGRESSOR':
        learningrate = 0.1
        regularizationstrength = 0.001
        hiddenunits = [10, 10, 8]
        metrics = {
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
    estimatortype = 'REGRESSOR'

    # Defines the training set, test set and the prediction set to use for the model
    training_set = pd.read_csv("data/quake/quake_train.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS, usecols=[0,1,2])
    test_set = pd.read_csv("data/quake/quake_test.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS, usecols=[0,1,2])
    prediction_set = pd.read_csv("data/quake/quake_predict.csv", skipinitialspace=True,
                                 skiprows=1, names=COLUMNS)

    if estimatortype == 'REGRESSOR':
        hiddenunits = [10, 3, 10]
        learningrate = 0.1
        regularizationstrength = 0.001

        metrics = {
            "accuracy":
                tf.contrib.learn.MetricSpec(
                    metric_fn=tf.contrib.metrics.streaming_accuracy,
                    prediction_key=tf.contrib.learn.PredictionKey.SCORES
                )
        }
    if estimatortype == 'CLASSIFIER':
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
    hiddenunits = [10, 1, 10]
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
    estimatortype = 'REGRESSOR'

    training_set = pd.read_csv("data/gastest/gastest_train.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS)
    test_set = pd.read_csv("data/gastest/gastest_test.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
    prediction_set = pd.read_csv("data/gastest/gastest_predict.csv", skipinitialspace=True,
                                 skiprows=1, names=COLUMNS)

    if estimatortype == 'REGRESSOR':
        learningrate = 0.1
        regularizationstrength = 0.001
        hiddenunits = [10, 10, 8]
        metrics = {
            "accuracy":
                tf.contrib.learn.MetricSpec(
                    metric_fn=tf.contrib.metrics.streaming_accuracy,
                    prediction_key=tf.contrib.learn.PredictionKey.SCORES
                )
        }

    if estimatortype == 'CLASSIFIER':
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


# Function that defines the input that the model is using.
# It takes the features and labels from the used datainput and returns those
def input_fn(data_set):
    feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
    # print(feature_cols)
    labels = tf.constant(data_set[LABEL].values)
    # print(labels)
    return feature_cols, labels


# Function that defines the type of model that is being used
# The current modeltypes that are being used are REGRESSOR and CLASSIFIER
def estimatorinput():
    feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]
    estimator = None
    if estimatortype == 'REGRESSOR':
        estimator = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                                  hidden_units=hiddenunits,
                                                  activation_fn=tf.nn.relu,
                                                  optimizer=tf.train.ProximalAdagradOptimizer(
                                                      learning_rate=learningrate,
                                                      l1_regularization_strength=regularizationstrength
                                                  ),
                                                  model_dir=modellocation + "/regression",
                                                  label_dimension=LABEL.__len__()
                                                  )

    if estimatortype == 'CLASSIFIER':
        estimator = tf.contrib.learn.DNNClassifier(feature_columns=feature_cols,
                                                   hidden_units=hiddenunits,
                                                   n_classes=numberofgroups,
                                                   activation_fn=tf.nn.relu,
                                                   optimizer=tf.train.AdagradOptimizer(
                                                       learning_rate=learningrate,
                                                   ),
                                                   model_dir=modellocation + "/classification",
                                                   )
    # error melding maken als er geen estimator is geselecteerd
    return estimator
