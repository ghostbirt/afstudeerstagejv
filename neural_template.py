from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import neural_input as ni
import tensorflow as tf
import numpy as np


def fit(numberofsteps):
    #print(ni.training_set)
    regressor=ni.regressorinput()
    regressor.fit(input_fn=lambda: ni.input_fn(ni.training_set), steps=numberofsteps)


def evaluate(numberofsteps):
    regressor = ni.regressorinput()
    ev = regressor.evaluate(input_fn=lambda: ni.input_fn(ni.test_set), steps=1)

    print(ni.test_set)
    loss_score = ev["loss"]
    print("Loss: {0:f}".format(loss_score))
    print(ev)


def predictfromcsv():
    regressor = ni.regressorinput()
    y = regressor.predict_scores(input_fn=lambda: ni.input_fn(ni.prediction_set))

    predictions = list(y)
    print(predictions)
    for x in predictions:
        print("Predictions: {}".format(str(x)))



