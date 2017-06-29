from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import neural_input as ni


# Function that is used to train the model. It takes the training set as input and uses the fit function from
# the used estimator. It will run the fit function the number of times that has been giving with numberofsteps
def fit(numberofsteps):
    # print(ni.training_set)
    estimator = ni.estimatorinput()
    estimator.fit(input_fn=lambda: ni.input_fn(ni.training_set), steps=numberofsteps)


# Funtion that is used to test the model. It takes the test set as input and uses the evaluate function from
# the used estimor. Steps determine the number of times the evaluate function is used. Metrics determine what
# information is shown in the test.
def evaluate():
    estimator = ni.estimatorinput()
    ev = estimator.evaluate(input_fn=lambda: ni.input_fn(ni.test_set), steps=1, metrics=ni.metrics)
    #print(ni.metrics)
    loss_score = ev["loss"]
    print("Loss: {0:f}".format(loss_score))
    print(ev)




# Function that is used to do predictions with the model. It takes the prediction set as input and uses the
# predict function from the chosen estimator.
def predictfromcsv():
    estimator = ni.estimatorinput()
    if ni.ESTIMATORTYPE == 'REGRESSOR':
        y = estimator.predict_scores(input_fn=lambda: ni.input_fn(ni.prediction_set))
        predictions = list(y)
        #print(predictions)
        for x in predictions:
            print("Predictions: {}".format(str(x)))

    if ni.ESTIMATORTYPE == 'CLASSIFIER':
        y = estimator.predict_classes(input_fn=lambda: ni.input_fn(ni.prediction_set))
        predictions = list(y)
        #print(predictions)
        for x in predictions:
            print("Predictions: {}".format(str(x)))
