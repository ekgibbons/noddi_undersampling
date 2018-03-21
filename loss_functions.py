from __future__ import print_function, division

from matplotlib import pyplot as plt
import numpy as np

import tensorflow as tf

from keras import models
from keras.applications import vgg16

from keras import backend as K

import noddistudy
from utils import display

def perceptual_loss(y_true, y_predict):

    n_batches, y_dim, x_dim, n_channels = K.get_variable_shape(y_true)

    vgg = vgg16.VGG16(include_top=False,
                      weights='imagenet',
                      input_shape=(y_dim, x_dim, 3))

    loss_model = models.Model(inputs=vgg.input,
                              outputs=vgg.get_layer('block3_conv3').output)
    
    loss_model.trainable = False

    loss = 0
    for ii in range(3):
        y_true_slice = tf.expand_dims(y_true[:,:,:,ii],-1)
        y_true_rgb = tf.image.grayscale_to_rgb(y_true_slice,
                                               name=None)
        
        y_predict_slice = tf.expand_dims(y_predict[:,:,:,ii],-1)
        y_predict_rgb = tf.image.grayscale_to_rgb(y_predict_slice,
                                                  name=None)

        loss += K.mean(K.square(loss_model(y_true_rgb) - loss_model(y_predict_rgb)))

    return loss

def main():
    
    patient_number = "N011618"
    noddi_data = noddistudy.NoddiData(patient_number)
    # data_full = noddi_data.get_full()
    data_odi = noddi_data.get_odi()
    # data_fiso = noddi_data.get_fiso()

    data_odi = data_odi.transpose(2,0,1)[:,:,:,None]
    data_odi = np.concatenate((data_odi,data_odi,data_odi),
                              axis=3)

    
    data_odi = tf.convert_to_tensor(data_odi)
    perceptual_loss(data_odi,data_odi)

if __name__ == "__main__":
    main()
