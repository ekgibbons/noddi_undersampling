from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from keras import backend as K
from keras.models import Model
from keras.models import Sequential
from keras.layers.core import Dropout
from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.utils import plot_model


def simple2d(input_size):

    model = Sequential()

    model.add(Conv2D(128,(3,3),input_shape=input_size,activation="relu",
                     padding="same"))
    model.add(Dropout(0.1))

    model.add(Conv2D(256,(3,3),activation="relu",
                     padding="same"))
    model.add(Dropout(0.1))

    model.add(Conv2D(512,(3,3),activation="relu",
                     padding="same"))
    model.add(Dropout(0.1))

    model.add(Conv2D(7,(1,1),activation="relu",
                     padding="same"))

    return model
