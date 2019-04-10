import os
import sys
import time 

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

sys.path.append("/home/mirl/egibbons/noddi")

from noddi_utils import network_utils
import simple2d
from utils import readhdf5
from utils import display

def train(n_directions):

    loss_type = "l1"
    
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
    y_gfa_path = "/v/raid1b/egibbons/MRIdata/DTI/noddi/y_gfa_2d.h5"

    print("Loading data...")

    start = time.time()
    y_gfa = readhdf5.read_hdf5(y_gfa_path,"y_gfa")

    y = y_gfa
    
    x = readhdf5.read_hdf5(x_path,"x_%i_directions" % n_directions)
    
    print("Data is loaded...took: %f seconds" % (time.time() - start))

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
    
    save_path = ("/v/raid1b/egibbons/models/noddi-%i_2d_gfa_no_scale.h5"
                 % (n_directions))
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
    
