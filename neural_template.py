from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import neural_input as ni
import tensorflow as tf
import numpy as np


def fit(numberofsteps):
    #print(ni.training_set)
    estimator=ni.estimatorinput()
    estimator.fit(input_fn=lambda: ni.input_fn(ni.training_set), steps=numberofsteps)


def evaluate():
    estimator = ni.estimatorinput()
    if ni.estimatortype=='REGRESSOR':
        ev = estimator.evaluate(input_fn=lambda: ni.input_fn(ni.test_set), steps=1,metrics=ni.metrics)

        print(ni.metrics)
        loss_score = ev["loss"]
        print("Loss: {0:f}".format(loss_score))
        print(ev)

    if ni.estimatortype == 'CLASSIFIER':
        ev = estimator.evaluate(input_fn=lambda: ni.input_fn(ni.test_set), steps=1, metrics=ni.metrics)

        print(ni.metrics)
        loss_score = ev["loss"]
        print("Loss: {0:f}".format(loss_score))
        print(ev)


def predictfromcsv():
    estimator = ni.estimatorinput()
    if ni.estimatortype=='REGRESSOR':
        y = estimator.predict_scores(input_fn=lambda: ni.input_fn(ni.prediction_set))
        predictions = list(y)
        print(predictions)
        for x in predictions:
            print("Predictions: {}".format(str(x)))

    if ni.estimatortype=='CLASSIFIER':
        y = estimator.predict_classes(input_fn=lambda: ni.input_fn(ni.prediction_set))
        predictions = list(y)
        print(predictions)
        for x in predictions:
            print("Predictions: {}".format(str(x)))



