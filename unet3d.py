from __future__ import print_function, division

import numpy as np

from keras import models
from keras import layers
import keras.utils
from keras.layers import Lambda
from keras.layers import BatchNormalization as BN
from keras import backend as K

def unet3d_model(input_size):

    # input size is a tuple of the size of the image
    # assuming channel last
    # input_size = (dim1, dim2, dim3, ch)
    # unet begins

    nfeatures = [16,32,64,128,256,512]
    depth = len(nfeatures)

    conv_ptr = []

    # input layer
    inputs = layers.Input(input_size)

    # step down convolutional layers
    pool = inputs
    for depth_cnt in range(depth):


        conv = layers.Conv3D(nfeatures[depth_cnt], (3,3,3), 
                             padding='same', 
                             activation='relu',
                             kernel_initializer='he_normal')(pool)
        conv = layers.Conv3D(nfeatures[depth_cnt], (3,3,3), 
                             padding='same', 
                             activation='relu',
                             kernel_initializer='he_normal')(conv)

        conv = BN(axis=-1, momentum=0.95, epsilon=0.001)(conv)
        conv = layers.Dropout(rate=0.20)(conv)

        conv_ptr.append(conv)

        if depth_cnt < depth-1:
            pool = layers.MaxPooling3D(pool_size=(2,2,2))(conv)
    
    # step up convolutional layers
    for depth_cnt in range(depth-2,-1,-1):

        deconv_shape = conv_ptr[depth_cnt].shape.as_list()
        deconv_shape[0] = None
      
        up = layers.concatenate([layers.Conv3DTranspose(nfeatures[depth_cnt],(3,3,3),
                                                        padding='same',
                                                        strides=(2,2,2))(conv),
                                 conv_ptr[depth_cnt]], 
                                axis=4)

        conv = layers.Conv3D(nfeatures[depth_cnt], (3,3,3), 
                             padding='same', 
                             activation='relu',
                             kernel_initializer='he_normal')(up)
        conv = layers.Conv3D(nfeatures[depth_cnt], (3,3,3), 
                             padding='same', 
                             activation='relu',
                             kernel_initializer='he_normal')(conv)
        
        conv = BN(axis=1, momentum=0.95, epsilon=0.001)(conv)
        conv = layers.Dropout(rate=0.20)(conv)

    # combine features
    # conv = layers.Conv3D(1, (1,1,1), padding='same', activation='relu')(conv)
    # conv_shape = conv.shape.as_list()

    # step down and combine features
    depth_total = int(np.log2(input_size[2]))
    for depth_cnt in range(depth_total):
        conv = layers.Conv3D(1, (3,3,3),
                             padding='same',
                             activation='relu',
                             kernel_initializer='he_normal')(conv)

        conv = layers.MaxPooling3D(pool_size=(1,1,2))(conv)
        
    # conv = layers.Conv3D(1, (3,3,1),
    #                      padding='same',
    #                      activation='relu',
    #                      kernel_initializer='he_normal')(conv)


    # print(conv)
    recon = layers.Conv3D(1, (1,1,1),
                         padding='same',
                         activation='sigmoid')(conv)

    model = models.Model(inputs=[inputs], outputs=[recon])
    keras.utils.plot_model(model, to_file='unet3d.png',show_shapes=True)
    
    return model


def main():
    input_size = (128, 128, 64, 1)

    model = unet3d_model(input_size)



if __name__ == "__main__":
    main()
