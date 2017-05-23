from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import neural_template as nt

tf.logging.set_verbosity(tf.logging.ERROR)


# Main class that is used for the training functionality of the model

def train(numberofsteps):
    nt.fit(numberofsteps)


def trainmain(numberofsteps):  # pylint: disable=unused-argument
    train(numberofsteps)


if __name__ == '__trainmain__':
    tf.app.run()
