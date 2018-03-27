from __future__ import print_function, division

from matplotlib import pyplot as plt
import numpy as np

import tensorflow as tf

from keras import models
from keras import layers
from keras import losses
import keras.utils
from keras.applications import densenet
from keras.layers import Lambda
from keras.layers import BatchNormalization as BN
from keras import backend as K

from utils import display


def unet2d_model(input_size):

    features = np.arange(0,6)
    nfeatures = [input_size[2]*(2**feature_size) for feature_size in features]
    depth = len(nfeatures)

    conv_ptr = []

    # input layer
    inputs = layers.Input(input_size)

    # step down convolutional layers
    pool = inputs
    for depth_cnt in range(depth):
        conv = layers.Conv2D(nfeatures[depth_cnt], (3,3), 
                             padding='same', 
                             activation='relu',
                             kernel_initializer='he_normal')(pool)
        conv = layers.Conv2D(nfeatures[depth_cnt], (3,3), 
                             padding='same', 
                             activation='relu',
                             kernel_initializer='he_normal')(conv)

        conv = BN(axis=-1, momentum=0.95, epsilon=0.001)(conv)
        conv = layers.Dropout(rate=0.20)(conv)

        conv_ptr.append(conv)

        if depth_cnt < depth-1:
            pool = layers.MaxPooling2D(pool_size=(2,2))(conv)
    
    # step up convolutional layers
    for depth_cnt in range(depth-2,-1,-1):

        deconv_shape = conv_ptr[depth_cnt].shape.as_list()
        deconv_shape[0] = None
      
        up = layers.concatenate([layers.Conv2DTranspose(nfeatures[depth_cnt],(3,3),
                                                        padding='same',
                                                        strides=(2,2))(conv),
                                 conv_ptr[depth_cnt]], 
                                axis=3)

        conv = layers.Conv2D(nfeatures[depth_cnt], (3,3), 
                             padding='same', 
                             activation='relu',
                             kernel_initializer='he_normal')(up)
        conv = layers.Conv2D(nfeatures[depth_cnt], (3,3), 
                             padding='same', 
                             activation='relu',
                             kernel_initializer='he_normal')(conv)
        
        conv = BN(axis=1, momentum=0.95, epsilon=0.001)(conv)
        conv = layers.Dropout(rate=0.20)(conv)

    recon = layers.Conv2D(3, (1,1),
                          padding='same',
                          activation='relu')(conv)

    model = models.Model(inputs=[inputs], outputs=[recon])
    keras.utils.plot_model(model, to_file='unet3d.png',show_shapes=True)
    
    return model

    

def main():
    input_size = (128, 128, 64)


    model = unet2d_model(input_size)
    
    model.summary()
    




if __name__ == "__main__":
    main()



