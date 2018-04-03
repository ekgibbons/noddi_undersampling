from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras.models import Model
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
from keras.layers import UpSampling2D
from keras.layers import ZeroPadding2D
from keras.utils import plot_model

def unet2d(input_size):

    img_input = Input(shape=input_size)

    axis_use = 3 if K.image_data_format() == 'channels_last' else 1

    x = Conv2D(128, (1,1), padding="same", activation="relu")(img_input)

    x_down1 = MaxPooling2D(pool_size=(2,2))(x)

    x_down1 = Conv2D(256, (3,3), padding="same", activation="relu")(x_down1)
    x_down1 = Conv2D(256, (3,3), padding="same", activation="relu")(x_down1)

    x_down2 = MaxPooling2D(pool_size=(2,2))(x_down1)

    x_down2 = Conv2D(512, (3,3), padding="same", activation="relu")(x_down2)
    x_down2 = Conv2D(512, (3,3), padding="same", activation="relu")(x_down2)

    x_up1 = UpSampling2D(size=(2,2))(x_down2)
    x_up1 = Concatenate(axis=axis_use)([x_up1, x_down1])

    x_up1 = Conv2D(256, (3,3), padding="same", activation="relu")(x_up1)
    x_up1 = Conv2D(256, (3,3), padding="same", activation="relu")(x_up1)

    x_up2 = UpSampling2D(size=(2,2))(x_up1)
    x_up2 = Concatenate(axis=axis_use)([x_up2, x])

    out = Conv2D(4, (1,1), padding="same", activation="relu")(x_up2)

    model = Model(inputs=[img_input], outputs=[out])
        
    return model

def main():

    image_size = (128, 128, 64)

    model = unet2d(image_size)

    model.summary()
    
    plot_model(model,
               to_file="1dnet_im.png",
               show_shapes=True)


if __name__ == "__main__":
    main()


