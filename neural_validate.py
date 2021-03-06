from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import neural_template as nt

tf.logging.set_verbosity(tf.logging.INFO)


def validate(numberof):
    nt.evaluate(numberof)


def validatemain(numberofsteps):  # pylint: disable=unused-argument
    validate(numberofsteps)


if __name__ == '__validatemain__':
    tf.app.run()
