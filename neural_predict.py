from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import neural_template as nt

tf.logging.set_verbosity(tf.logging.ERROR)
usecsv=True

def predict():
    if (usecsv==True):
        nt.predictfromcsv()




def predictmain(argv=None):  # pylint: disable=unused-argument
    predict()


if __name__ == '__predictmain__':
    tf.app.run()
