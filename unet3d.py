from __future__ import print_function, division

from keras import models
from keras import layers
import keras.utils


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
    for depth_cnt in xrange(depth):


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
    for depth_cnt in xrange(depth-2,-1,-1):

        deconv_shape = conv_ptr[depth_cnt].shape.as_list()
        deconv_shape[0] = None

        up = layers.concatenate([convolutional.Deconvolution3D(nfeatures[depth_cnt],(3,3,3),
                                                               padding='same',
                                                               strides=(2,2,2),
                                                               output_shape=deconv_shape)(conv),
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
    recon = layers.Conv3D(1, (1,1,1), padding='same', activation='sigmoid')(conv)

    model = models.Model(inputs=[inputs], outputs=[recon])
    keras.utils.plot_model(model, to_file='unet3d.png',show_shapes=True)
    
    return model
