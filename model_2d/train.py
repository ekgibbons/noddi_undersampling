import os
import sys
import time 

loss_type = "l1"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping 
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.training_utils import multi_gpu_model

sys.path.append("/home/mirl/egibbons/noddi")

import dense2d
import models2d
from noddi_utils import network_utils
import simple2d
import unet2d
from utils import readhd5
from utils import display

def augmentation(x, y):

    datagen_args = dict(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        shear_range=0.2,
        zoom_range=0.2
        )

    image_datagen = ImageDataGenerator(**datagen_args)
    target_datagen = ImageDataGenerator(**datagen_args)

    seed = 1
    image_datagen.fit(x, augment=True, seed=seed)
    target_datagen.fit(y, augment=True, seed=seed)
    
    image_generator = image_datagen.flow(x, shuffle=False,
                                         batch_size=9, seed=seed)
    target_generator = image_datagen.flow(y, shuffle=False,
                                          batch_size=9, seed=seed)
    
    generator = zip(image_generator, target_generator)

    return generator

def train(n_directions):
    
    print("running 2D network with %s loss and %i directions"
          % (loss_type, n_directions))
    

    n_gpu = 1
    n_epochs = 100
    batch_size = 10
    learning_rate = 1e-3
    
    image_size = (128,128,n_directions)
    
    model = simple2d.res2d(image_size)


    optimizer = Adam(lr=learning_rate)
    if loss_type == "l1":
        model.compile(optimizer=optimizer,
                      loss="mean_absolute_error")
    else:
        model.compile(optimizer=optimizer,
                      loss=network_utils.perceptual_loss)

    ### DATA LOADING ###
    x_path = ("/v/raid1b/egibbons/MRIdata/DTI/noddi/x_%i_directions_2d.h5" %
              n_directions)
    y_odi_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_odi_2d.h5"
    y_fiso_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_fiso_2d.h5"
    y_ficvf_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_ficvf_2d.h5"
    y_gfa_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_gfa_2d.h5"

    print("Loading data...")

    start = time.time()
    y_odi = readhd5.ReadHDF5(y_odi_path,"y_odi")
    y_fiso = readhd5.ReadHDF5(y_fiso_path,"y_fiso")
    y_ficvf = readhd5.ReadHDF5(y_ficvf_path,"y_ficvf")
    y_gfa = readhd5.ReadHDF5(y_gfa_path,"y_gfa")

    diffusivity_scaling = 1
    y = np.concatenate((y_odi, y_fiso, y_ficvf,
                        diffusivity_scaling*y_gfa),
                       axis=3)
    
    x = readhd5.ReadHDF5(x_path,"x_%i_directions" % n_directions)
    
    print("Data is loaded...took: %f seconds" % (time.time() - start))

    # ### DATA SPLITTING ###
    # n_samples, _, _, _ = x.shape
    # x_train = x[:int(0.8*n_samples)]
    # y_train = y[:int(0.8*n_samples)]

    # x_val = x[int(0.8*n_samples):]
    # y_val = y[int(0.8*n_samples):]
    
    # ### IMAGE PROCESSING ###
    # print("Preprocessing images...")
    # start = time.time()

    # training_generator = augmentation(x_train, y_train)
    # validation_generator = augmentation(x_val, y_val)

    # print("Images are processed...took: %f seconds" % (time.time() - start))

    ### MODEL FITTING ###
    batch_size_multi_gpu = n_gpu*batch_size

    tensorboard = TensorBoard(log_dir='./logs',
                              histogram_freq=1,
                              batch_size=batch_size_multi_gpu,
                              write_graph=True,
                              write_grads=True,
                              write_images=True,
                              embeddings_freq=0,
                              embeddings_layer_names=None,
                              embeddings_metadata=None)
    
    save_path = ("/v/raid1b/egibbons/models/noddi-%i_2d_%s.h5"
                 % (n_directions, loss_type))
    print("saving to: %s" % save_path)
    checkpointer = ModelCheckpoint(filepath=save_path,
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=True,
                                   period=1)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=5, min_lr=1e-7) 
    
    lrate = LearningRateScheduler(network_utils.step_decay)

    stopping = EarlyStopping(monitor='val_loss', min_delta=0,
                             patience=20, verbose=0, mode='auto')
    
    # model.fit_generator(
    #     training_generator,
    #     steps_per_epoch=x_train.shape[0]//batch_size_multi_gpu,
    #     validation_data=validation_generator,
    #     validation_steps=x_val.shape[0]//batch_size_multi_gpu,
    #     epochs=n_epochs,
    #     verbose=2,
    #     callbacks=[checkpointer, lrate, stopping],
    #     shuffle=True,
    # )

 
    model.fit(x=x,
              y=y,
              batch_size=batch_size_multi_gpu,
              epochs=n_epochs,
              verbose=2,
              callbacks=[checkpointer, lrate, stopping],
              validation_split=0.2,
              shuffle=True,
    )



    
    print("trained %i direction model" % n_directions)

    
def main(argv):

    if (len(argv) == 1) or (argv[1] == "64"):
        n_directions = 64
    else:
        n_directions = int(argv[1])
    
    train(n_directions)
    

if __name__ == "__main__":

    main(sys.argv)
    
