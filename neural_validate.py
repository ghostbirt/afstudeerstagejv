from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import neural_template as nt

tf.logging.set_verbosity(tf.logging.INFO)


# Main class that is used for the testing functionality of the model

def validate():
    nt.evaluate()


def validatemain():  # pylint: disable=unused-argument
    validate()


if __name__ == '__validatemain__':
    tf.app.run()
