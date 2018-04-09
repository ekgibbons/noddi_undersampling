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
from keras.layers import Add
from keras.layers import ZeroPadding2D
from keras.utils import plot_model

def res2d(input_size):

    img_input = Input(shape=input_size)
    
    axis_use = 3 if K.image_data_format() == 'channels_last' else 1

    x = Conv2D(128,(1,1),input_shape=input_size,activation="relu",
               padding="same")(img_input)

    x1 = Conv2D(128,(3,3),activation="relu",padding="same")(x)
    x1 = Conv2D(128,(3,3),activation="relu",padding="same")(x1)

    x = Add()([x,x1])

    x = Conv2D(256,(1,1),activation="relu",padding="same")(x)

    x2 = Conv2D(256,(3,3),activation="relu",padding="same")(x)
    x2 = Conv2D(256,(3,3),activation="relu",padding="same")(x2)

    x = Add()([x,x2])

    out = Conv2D(4,(1,1),activation="relu",padding="same")(x)

    model = Model(inputs=[img_input], outputs=[out])

    return model
    

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

    model.add(Conv2D(4,(1,1),activation="relu",
                     padding="same"))

    return model

def main():

    image_size = (128, 128, 64)

    model = res2d(image_size)

    model.summary()
    
    plot_model(model,
               to_file="1dnet_im.png",
               show_shapes=True)


if __name__ == "__main__":
    main()


